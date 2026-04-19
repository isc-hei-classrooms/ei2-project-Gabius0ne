"""
train_lgbm_v15.py
=================
Entraînement LightGBM v15 avec normalisation per-unit de la cible diurne.

OBJECTIF
--------
Prédire la charge nette normalisée d'Oiken pour les 96 pas de 15 min
du jour J+1, à partir des features v15 (v13 + feature pv_normalizer_90j).

CHANGEMENT v13 → v15
--------------------
Application d'une normalisation per-unit (analogue au pu en
électrotechnique) sur la cible DAY_STEPS uniquement, pour traiter la
non-stationnarité du PV due à la croissance du parc Oiken.

  Principe :
    - Pour chaque jour, on récupère pv_normalizer_90j (feature v15)
      calculée comme mean(solar_remote, 10h-17h UTC, 90j précédents)
    - On convertit en facteur relatif sans dimension :
        normalizer_ref = mean(pv_normalizer_90j sur le train set)
        normalizer_rel = pv_normalizer_90j / normalizer_ref
      Ce qui donne un facteur ~0.6 (début 2022, petit parc) → ~1.4 (2025,
      parc agrandi), centré autour de 1.0 = S_base en analogie pu
    - Pour les DAY_STEPS (indices 40-67, pic PV) : Y_pu = Y / normalizer_rel
    - Pour les NIGHT_STEPS (indices 0-39 et 68-95) : Y_pu = Y (inchangé)
    - Le modèle est entraîné sur Y_pu
    - À l'inférence, on dénormalise : preds = preds_pu × normalizer_rel
    - Le RMSE/MAE est calculé dans l'espace dénormalisé, donc
      directement comparable à la baseline Oiken et aux runs précédents

  Pourquoi seulement les DAY_STEPS :
    - La conso nocturne n'a aucune raison d'être proportionnelle à la
      capacité PV — normaliser la nuit introduirait de la distorsion
    - Les DAY_STEPS (10h-16h45 UTC) couvrent le pic PV où la cible
      varie effectivement avec la capacité installée

  Jours sans normalizer :
    - Les ~90 premiers jours du dataset n'ont pas de fenêtre de 90j
      précédents → pv_normalizer_90j est None
    - Ces jours sont exclus du train, val ET test (skipés globalement)
    - Choix cohérent avec la règle "fenêtre glissante finissant à J-2"

  Anti-leakage :
    - pv_normalizer_90j est calculé par pipeline_features_v15 avec une
      fenêtre finissant strictement à J-2 (aucune donnée ≥ J-1 touchée)
    - Même normalizer utilisé à l'entraînement et à l'inférence pour
      chaque jour, donc aucune fuite temporelle

ARCHITECTURE
------------
- 96 modèles LightGBM indépendants, un par pas de 15 min
- Split jour/nuit avec hyperparamètres Optuna séparés
- Entraînement final en deux étapes (ES propre + réentraînement, v13)

STRATÉGIE D'ÉVALUATION
----------------------
- Split chronologique 47% train / 20% val / 33% test (strict, pas de shuffle)
- Optuna optimise sur val avec ES sur val (test jamais touché)
- Entraînement final en deux étapes :
    1) ES propre sur train seul + val pour déterminer best_iteration
    2) Réentraînement sur train ∪ val avec num_boost_round = best_iteration
- Exclusion des 13-16 septembre 2025 des métriques finales
- Jours sans pv_normalizer_90j exclus de tout (train/val/test)
- Métriques finales TOUJOURS calculées dans l'espace original (dénormalisé)

SORTIES
-------
  DATA/models15v0/lgbm_t{000..095}.pkl     — 96 modèles sérialisés (espace pu pour DAY, brut pour NIGHT)
  DATA/models15v0/metrics.parquet          — RMSE/MAE par pas de temps (espace dénormalisé)
  DATA/models15v0/predictions_test.parquet — prédictions sur le test set (espace dénormalisé)
  DATA/models15v0/best_params_night.json   — hyperparamètres NUIT+HORS-PIC
  DATA/models15v0/best_params_day.json     — hyperparamètres JOUR (pic PV, entraîné en pu)
"""

import polars as pl
import numpy as np
import pickle
import json
from pathlib import Path
from datetime import date
from collections import defaultdict
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
import optuna
from optuna.samplers import TPESampler

# ─────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────

BASE = Path(__file__).resolve().parents[2] / "DATA"

X_PATH = BASE / "processed" / "X_features_v15.parquet"   # features v15
Y_PATH = BASE / "processed" / "Y_target_v15.parquet"     # cible load net normalisé (96 pas)
B_PATH = BASE / "processed" / "B_baseline_v15.parquet"   # baseline Oiken pour comparaison
OUT    = BASE / "models15v0"
OUT.mkdir(parents=True, exist_ok=True)

# Nom de la colonne contenant le normalizer dans X (créée par pipeline_features_v15)
NORMALIZER_COL = "pv_normalizer_90j"

# Split chronologique strict
TRAIN_RATIO = 0.47
VAL_RATIO   = 0.20

# Hyperparamètres d'entraînement
N_OPTUNA_TRIALS  = 40
N_ESTIMATORS_MAX = 1000
EARLY_STOPPING   = 50
RANDOM_SEED      = 42

EXCLUDE_DATES = {date(2025, 9, 13), date(2025, 9, 14), date(2025, 9, 15), date(2025, 9, 16)}

# ─────────────────────────────────────────────────────────────────────
# GROUPES JOUR / NUIT
# ─────────────────────────────────────────────────────────────────────
# step 0 = 00h00, step 40 = 10h00, step 68 = 17h00, step 95 = 23h45 UTC.
# DAY_STEPS = indices normalisés en per-unit par pv_normalizer_90j.
# NIGHT_STEPS = indices laissés bruts (pas de sens physique à normaliser).
NIGHT_STEPS = list(range(0, 40)) + list(range(41, 96))   # 00h-09h45 + 17h-23h45
DAY_STEPS   = list(range(40, 41))                         # 10h-16h45

OPTUNA_STEPS_DAY   = [41]
OPTUNA_STEPS_NIGHT = [0, 12, 28, 48, 52, 54, 56, 58, 72, 84, 92]


# ─────────────────────────────────────────────────────────────────────
# FEATURE IMPORTANCE GROUPÉE PAR THÈME
# ─────────────────────────────────────────────────────────────────────

def classify_feature(name: str) -> str:
    if name.startswith("load_"):
        return "Load historique"
    if name.startswith("solar_") or "pv_yield" in name or "pv_capacity" in name or "remote_max" in name \
       or name == NORMALIZER_COL:
        return "PV mesuré / capacité"
    if name.startswith("pred_pv_adj"):
        return "PV prévu (interaction)"
    if name.startswith("pred_glob_rad"):
        return "Irradiance prévue J+1"
    if name.startswith("predJ_glob_rad"):
        return "Irradiance prévue J (obsolète v13+)"
    if name.startswith("pred_temp") or name.startswith("pred_pressure") or name.startswith("pred_relhum"):
        return "Météo prévue J+1 (temp/pres/hum)"
    if name.startswith("predJ_temp") or name.startswith("predJ_pressure") or name.startswith("predJ_relhum"):
        return "Météo prévue J (obsolète v13+)"
    if name.startswith("pred_wind") or name.startswith("pred_precip") or name.startswith("pred_sunshine"):
        return "Météo prévue J+1 (vent/pluie/soleil)"
    if name.startswith("predJ_wind") or name.startswith("predJ_precip") or name.startswith("predJ_sunshine"):
        return "Météo prévue J (obsolète v13+)"
    if name.startswith("rmet_"):
        return "Météo mesurée"
    if name in ("dayofweek", "sin_dow", "cos_dow", "is_weekend", "is_holiday",
                "is_school_holiday", "is_bridge_day",
                "month", "sin_month", "cos_month", "sin_doy", "cos_doy"):
        return "Calendaire"
    if "ramadan" in name:
        return "Ramadan"
    return "Autre"


def print_grouped_importance(feat_names, models, group_name, top_n=20):
    imp = np.zeros(len(feat_names))
    for m in models:
        imp += m.feature_importance()
    imp /= len(models)

    theme_imp = {}
    for name, val in zip(feat_names, imp):
        theme = classify_feature(name)
        theme_imp[theme] = theme_imp.get(theme, 0) + val
    total = sum(theme_imp.values())

    print(f"\n{'='*60}")
    print(f"  Feature importance {group_name} — par thème")
    print(f"{'='*60}")
    for theme, val in sorted(theme_imp.items(), key=lambda x: -x[1]):
        pct = val / total * 100 if total > 0 else 0
        bar = "█" * int(pct / 2)
        print(f"  {pct:5.1f}%  {bar:25s}  {theme}")

    print(f"\n  Top {top_n} features {group_name} :")
    top = sorted(zip(feat_names, imp), key=lambda x: -x[1])[:top_n]
    for name, val in top:
        theme = classify_feature(name)
        print(f"  {val:8.1f}  [{theme:.<30s}]  {name}")


# ─────────────────────────────────────────────────────────────────────
# 1. CHARGEMENT DES DONNÉES
# ─────────────────────────────────────────────────────────────────────

print("=== Chargement des données v15 ===")
X = pl.read_parquet(X_PATH)
Y = pl.read_parquet(Y_PATH)
B = pl.read_parquet(B_PATH)

# Vérification : la colonne normalizer doit être présente
if NORMALIZER_COL not in X.columns:
    raise ValueError(
        f"Colonne '{NORMALIZER_COL}' absente de X_features_v15.parquet. "
        f"As-tu bien régénéré les features avec pipeline_features_v15.py ?"
    )

# ─────────────────────────────────────────────────────────────────────
# 1bis. EXCLUSION DES JOURS SANS NORMALIZER
# ─────────────────────────────────────────────────────────────────────
# Les ~90 premiers jours du dataset (fenêtre de 90j précédents non disponible)
# ont pv_normalizer_90j = None. On les exclut globalement avant le split.

normalizer_series = X[NORMALIZER_COL]
mask_has_normalizer = normalizer_series.is_not_null().to_numpy()
n_dropped = (~mask_has_normalizer).sum()
print(f"  Jours sans pv_normalizer_90j (exclus) : {n_dropped}")

X = X.filter(pl.col(NORMALIZER_COL).is_not_null())
Y = Y.filter(pl.Series(mask_has_normalizer))
B = B.filter(pl.Series(mask_has_normalizer))

# Extraction des dates et features
dates = X["date"]
feat_names = [c for c in X.columns if c != "date"]

X_arr = X.drop("date").to_numpy().astype(np.float32)
Y_arr = Y.drop("date").to_numpy().astype(np.float32)
B_arr = B.drop("date").to_numpy().astype(np.float32)

n_samples = X_arr.shape[0]
n_steps   = Y_arr.shape[1]

# Index de la colonne normalizer pour extraction rapide
norm_idx = feat_names.index(NORMALIZER_COL)
normalizer_all = X_arr[:, norm_idx].astype(np.float64)   # (n_samples,) float64 pour précision division

print(f"  Samples retenus : {n_samples} jours")
print(f"  Features : {X_arr.shape[1]}")
print(f"  Pas de temps : {n_steps}")
print(f"  Groupe NUIT (hors pic PV) : {len(NIGHT_STEPS)} pas (00h-09h45 + 17h-23h45) — cible brute")
print(f"  Groupe JOUR (pic PV)      : {len(DAY_STEPS)} pas (10h00-16h45) — cible normalisée en pu relatif")
print(f"  Normalizer brut (kWh) : min={normalizer_all.min():.1f}, max={normalizer_all.max():.1f}, "
      f"mean={normalizer_all.mean():.1f}")

# ─────────────────────────────────────────────────────────────────────
# 1ter. NORMALISATION PER-UNIT RELATIVE DE LA CIBLE SUR DAY_STEPS
# ─────────────────────────────────────────────────────────────────────
# Le pv_normalizer_90j brut est en kWh (~760 à ~7400), alors que Y est
# en z-score (~-3 à +3). Diviser Y par ~3400 donnerait des cibles
# microscopiques (~10⁻⁴) inapprenant pour le modèle.
#
# Solution : convertir le normalizer en facteur relatif SANS DIMENSION,
# centré autour de 1.0. On divise chaque normalizer par la moyenne du
# normalizer sur le TRAIN SET SEUL (anti-leakage : pas de stats test/val).
#
# Résultat : normalizer_rel ≈ 0.6 début 2022 (petit parc PV) → 1.4 en
# 2025 (parc agrandi). La cible Y_pu reste dans le même ordre de grandeur
# que Y_arr, mais ajustée pour la croissance du parc.
#
# Analogie pu exacte : normalizer_ref = S_base, normalizer_rel = S / S_base.
# Y_pu = Y / (S / S_base) = Y × (S_base / S).
#
# Y_pu[i, t] = Y[i, t] / normalizer_rel[i] pour t dans DAY_STEPS
# Y_pu[i, t] = Y[i, t]                      pour t dans NIGHT_STEPS

# NOTE : normalizer_ref est calculé APRÈS le split (section 2) pour
# utiliser uniquement les données train. On prépare ici le conteneur,
# la normalisation effective est dans la section 2bis.
normalizer_all_raw = normalizer_all.copy()   # sauvegarde brute pour log

# ─────────────────────────────────────────────────────────────────────
# 2. SPLIT TRAIN / VAL / TEST (chronologique strict)
# ─────────────────────────────────────────────────────────────────────

split_train = int(n_samples * TRAIN_RATIO)
split_val   = int(n_samples * (TRAIN_RATIO + VAL_RATIO))

# X identique train/val/test
X_train, X_val, X_test = X_arr[:split_train], X_arr[split_train:split_val], X_arr[split_val:]

# ─────────────────────────────────────────────────────────────────────
# 2bis. CALCUL DU NORMALIZER RELATIF + NORMALISATION EFFECTIVE
# ─────────────────────────────────────────────────────────────────────
# On calcule la référence (= S_base en pu) sur le TRAIN SEUL pour
# éviter tout leakage. Le normalizer relatif est sans dimension et
# centré autour de 1.0.

normalizer_ref = normalizer_all_raw[:split_train].mean()   # S_base, scalaire
normalizer_rel = normalizer_all_raw / normalizer_ref        # (n_samples,), ~[0.6, 1.4]

print(f"\n  Normalizer relatif (pu) :")
print(f"    Référence (mean train) : {normalizer_ref:.1f} kWh")
print(f"    Facteur relatif : min={normalizer_rel.min():.3f}, max={normalizer_rel.max():.3f}, "
      f"mean={normalizer_rel.mean():.3f}")

# Normalisation effective de Y sur DAY_STEPS
Y_pu = Y_arr.copy()
for t in DAY_STEPS:
    Y_pu[:, t] = Y_arr[:, t] / normalizer_rel

print(f"\n  Cible normalisée :")
print(f"    Y_arr[:, DAY_STEPS]  (brut)  — mean={np.nanmean(Y_arr[:, DAY_STEPS]):+.4f}, "
      f"std={np.nanstd(Y_arr[:, DAY_STEPS]):.4f}")
print(f"    Y_pu[:,  DAY_STEPS]  (pu)    — mean={np.nanmean(Y_pu[:, DAY_STEPS]):+.4f}, "
      f"std={np.nanstd(Y_pu[:, DAY_STEPS]):.4f}")

# Y_pu pour l'entraînement (cible normalisée sur DAY_STEPS)
Y_train_pu = Y_pu[:split_train]
Y_val_pu   = Y_pu[split_train:split_val]
Y_test_pu  = Y_pu[split_val:]

# Y_arr (brut) pour les métriques finales dénormalisées
Y_test = Y_arr[split_val:]

# Normalizer relatif par split (pour dénormaliser les prédictions test)
normalizer_rel_train = normalizer_rel[:split_train]
normalizer_rel_val   = normalizer_rel[split_train:split_val]
normalizer_rel_test  = normalizer_rel[split_val:]

B_test      = B_arr[split_val:]
dates_test  = dates[split_val:]

print(f"\n=== Split chronologique ===")
print(f"  Train : {split_train} jours ({dates[0]} → {dates[split_train-1]})  [{TRAIN_RATIO*100:.0f}%]")
print(f"  Val   : {split_val - split_train} jours ({dates[split_train]} → {dates[split_val-1]})  [{VAL_RATIO*100:.0f}%]")
print(f"  Test  : {n_samples - split_val} jours ({dates[split_val]} → {dates[-1]})  [{(1-TRAIN_RATIO-VAL_RATIO)*100:.0f}%]")
print(f"  Normalizer relatif test : min={normalizer_rel_test.min():.3f}, max={normalizer_rel_test.max():.3f}")

# Masque d'exclusion des 13-16 sept sur le test
dates_test_list = dates_test.to_list()
exclude_mask = np.array([
    d not in EXCLUDE_DATES for d in dates_test_list
], dtype=bool)
n_excluded = (~exclude_mask).sum()
print(f"  Dates exclues des métriques : {n_excluded} ({[str(d) for d in sorted(EXCLUDE_DATES)]})")


# ─────────────────────────────────────────────────────────────────────
# 3. OPTUNA — TUNING SÉPARÉ JOUR / NUIT
# ─────────────────────────────────────────────────────────────────────
# IMPORTANT : Optuna travaille sur la cible Y_pu (normalisée pour DAY,
# brute pour NIGHT). Le RMSE affiché pendant Optuna est dans l'espace pu
# pour le groupe JOUR — il n'est donc pas directement comparable au
# RMSE final (qui sera dans l'espace dénormalisé). Ce n'est pas un
# problème pour la sélection d'hyperparamètres : minimiser le RMSE pu
# sur un pas donné équivaut à minimiser le RMSE dénormalisé (division
# par une constante positive ligne par ligne).

def run_optuna(group_name, optuna_steps, n_trials):
    """
    Tuning Optuna pour un groupe (NUIT ou JOUR).
    Utilise Y_train_pu et Y_val_pu (cibles en espace d'entraînement).
    """

    def objective(trial):
        params = {
            "objective":         "regression",
            "metric":            "rmse",
            "verbosity":         -1,
            "n_jobs":            -1,
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves":        trial.suggest_int("num_leaves", 15, 127),
            "max_depth":         trial.suggest_int("max_depth", 3, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.2, 1.0),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "min_split_gain":    trial.suggest_float("min_split_gain", 0.0, 1.0),
        }

        rmse_list = []
        for t in optuna_steps:
            y_tr = Y_train_pu[:, t]
            y_va = Y_val_pu[:, t]
            mask_tr = ~np.isnan(y_tr)
            mask_va = ~np.isnan(y_va)
            if mask_tr.sum() < 10 or mask_va.sum() < 10:
                continue

            dtrain = lgb.Dataset(X_train[mask_tr], label=y_tr[mask_tr],
                                 feature_name=feat_names, free_raw_data=False)
            dval = lgb.Dataset(X_val[mask_va], label=y_va[mask_va],
                               reference=dtrain, free_raw_data=False)

            model = lgb.train(
                params, dtrain, num_boost_round=N_ESTIMATORS_MAX,
                valid_sets=[dval],
                callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False),
                           lgb.log_evaluation(-1)],
            )
            pred = model.predict(X_val[mask_va])
            rmse_list.append(float(np.sqrt(mean_squared_error(y_va[mask_va], pred))))

        return float(np.mean(rmse_list)) if rmse_list else float("inf")

    print(f"\n{'='*60}")
    print(f"  Optuna {group_name} — {n_trials} trials (pas: {optuna_steps})")
    if group_name == "JOUR":
        print(f"  [NB] Cible en per-unit — RMSE pu, pas directement comparable au RMSE final")
    print(f"{'='*60}")

    sampler = TPESampler(seed=RANDOM_SEED)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_trial.params
    print(f"\n  Meilleur trial #{study.best_trial.number}")
    print(f"  RMSE validation : {study.best_value:.6f}")
    for k, v in best.items():
        print(f"    {k}: {v}")

    return best


best_night = run_optuna("NUIT", OPTUNA_STEPS_NIGHT, N_OPTUNA_TRIALS)
best_day   = run_optuna("JOUR", OPTUNA_STEPS_DAY, N_OPTUNA_TRIALS)

with open(OUT / "best_params_night.json", "w") as f:
    json.dump(best_night, f, indent=2)
with open(OUT / "best_params_day.json", "w") as f:
    json.dump(best_day, f, indent=2)


# ─────────────────────────────────────────────────────────────────────
# 4. ENTRAÎNEMENT FINAL — DEUX ÉTAPES
# ─────────────────────────────────────────────────────────────────────
# Identique à v13 : ES propre sur train seul, puis réentraînement sur
# train+val avec num_boost_round fixé. Cible en Y_pu (normalisée pour DAY).
# La dénormalisation est appliquée APRÈS prédiction, avant calcul des métriques.

ITER_SCALE = 1.10

print(f"\n=== Entraînement final (96 modèles, 2 étapes)  ===")
print(f"  Étape 1 : ES propre sur train → val pour déterminer best_iteration")
print(f"  Étape 2 : réentraînement sur train+val avec num_boost_round fixé (× {ITER_SCALE})")
print(f"  DAY_STEPS : cible en per-unit | NIGHT_STEPS : cible brute")
print(f"  Dénormalisation des prédictions DAY_STEPS appliquée avant calcul des métriques")

# Concaténation train+val pour l'étape 2
X_trainval         = np.concatenate([X_train, X_val], axis=0)
Y_trainval_pu      = np.concatenate([Y_train_pu, Y_val_pu], axis=0)
normalizer_rel_trainval = np.concatenate([normalizer_rel_train, normalizer_rel_val], axis=0)

night_set = set(NIGHT_STEPS)

# Prédictions dans l'espace ORIGINAL (après dénormalisation pour DAY_STEPS)
preds_test = np.zeros_like(Y_test)
metrics    = []

for t in range(n_steps):
    is_night = t in night_set
    best_params = best_night if is_night else best_day
    group_label = "NUIT" if is_night else "JOUR"

    final_params = {
        "objective": "regression",
        "metric":    "rmse",
        "verbosity": -1,
        "n_jobs":    -1,
        **best_params,
    }

    # Target pour ce pas : Y_pu (cible d'entraînement)
    y_tr_pu = Y_train_pu[:, t]
    y_va_pu = Y_val_pu[:, t]
    y_tv_pu = Y_trainval_pu[:, t]
    # Pour les métriques : Y_test (brut, jamais normalisé)
    y_te = Y_test[:, t]

    mask_tr = ~np.isnan(y_tr_pu)
    mask_va = ~np.isnan(y_va_pu)
    mask_tv = ~np.isnan(y_tv_pu)
    mask_te = ~np.isnan(y_te) & exclude_mask

    if mask_tr.sum() < 10 or mask_va.sum() < 10:
        print(f"  t={t:03d} — skip (données insuffisantes)")
        preds_test[:, t] = np.nan
        metrics.append({"step": t, "time_label": f"{(t * 15) // 60:02d}h{(t * 15) % 60:02d}",
                        "group": group_label, "rmse_model": None, "mae_model": None,
                        "rmse_baseline": None, "mae_baseline": None,
                        "n_estimators_es": None, "n_estimators_final": None})
        continue

    # ── ÉTAPE 1 : ES propre sur train seul → val
    dtrain_p1 = lgb.Dataset(X_train[mask_tr], label=y_tr_pu[mask_tr],
                            feature_name=feat_names, free_raw_data=False)
    dval_p1   = lgb.Dataset(X_val[mask_va], label=y_va_pu[mask_va],
                            reference=dtrain_p1, free_raw_data=False)

    model_p1 = lgb.train(
        final_params, dtrain_p1, num_boost_round=N_ESTIMATORS_MAX,
        valid_sets=[dval_p1],
        callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False),
                   lgb.log_evaluation(-1)],
    )
    best_iter_raw = model_p1.best_iteration
    if best_iter_raw is None or best_iter_raw <= 0:
        best_iter_raw = N_ESTIMATORS_MAX // 4
    best_iter_final = max(1, int(round(best_iter_raw * ITER_SCALE)))

    # ── ÉTAPE 2 : réentraînement sur train+val
    dtrain_p2 = lgb.Dataset(X_trainval[mask_tv], label=y_tv_pu[mask_tv],
                            feature_name=feat_names, free_raw_data=False)

    model = lgb.train(
        final_params, dtrain_p2, num_boost_round=best_iter_final,
        callbacks=[lgb.log_evaluation(-1)],
    )

    # ── Prédiction sur le test set (en espace d'entraînement : pu pour DAY, brut pour NIGHT)
    pred_pu = model.predict(X_test)

    # ── DÉNORMALISATION pour DAY_STEPS uniquement
    if is_night:
        pred_denorm = pred_pu                                   # déjà en espace brut
    else:
        pred_denorm = pred_pu * normalizer_rel_test             # retour à l'espace brut

    preds_test[:, t] = pred_denorm

    # Métriques dans l'espace dénormalisé (comparable baseline et runs précédents)
    if mask_te.sum() > 0:
        rmse_m = float(np.sqrt(mean_squared_error(y_te[mask_te], pred_denorm[mask_te])))
        mae_m  = float(mean_absolute_error(y_te[mask_te], pred_denorm[mask_te]))
    else:
        rmse_m, mae_m = None, None

    b_t = B_test[:, t]
    mask_b = ~np.isnan(y_te) & ~np.isnan(b_t) & exclude_mask
    rmse_b = float(np.sqrt(mean_squared_error(y_te[mask_b], b_t[mask_b]))) if mask_b.sum() > 0 else None
    mae_b  = float(mean_absolute_error(y_te[mask_b], b_t[mask_b]))          if mask_b.sum() > 0 else None

    metrics.append({
        "step":               t,
        "time_label":         f"{(t * 15) // 60:02d}h{(t * 15) % 60:02d}",
        "group":              group_label,
        "rmse_model":         rmse_m,
        "mae_model":          mae_m,
        "rmse_baseline":      rmse_b,
        "mae_baseline":       mae_b,
        "n_estimators_es":    best_iter_raw,
        "n_estimators_final": best_iter_final,
    })

    with open(OUT / f"lgbm_t{t:03d}.pkl", "wb") as f:
        pickle.dump(model, f)

    if t % 12 == 0:
        base_str = f"{rmse_b:.4f}" if rmse_b is not None else "N/A"
        model_str = f"{rmse_m:.4f}" if rmse_m is not None else "N/A"
        print(f"  t={t:03d} ({metrics[-1]['time_label']}) [{group_label}] — "
              f"RMSE model={model_str} | baseline={base_str} | "
              f"iter_es={best_iter_raw} → final={best_iter_final}")


# ─────────────────────────────────────────────────────────────────────
# 5. MÉTRIQUES GLOBALES ET PAR TRANCHE HORAIRE
# ─────────────────────────────────────────────────────────────────────
# Toutes les métriques sont calculées dans l'espace ORIGINAL (Y_test brut
# et preds_test déjà dénormalisé dans la boucle).

metrics_df = pl.DataFrame(metrics)

mask_all = ~np.isnan(Y_test) & ~np.isnan(preds_test) & exclude_mask[:, None]
rmse_global = float(np.sqrt(mean_squared_error(Y_test[mask_all], preds_test[mask_all])))
mae_global  = float(mean_absolute_error(Y_test[mask_all], preds_test[mask_all]))

mask_b_all = ~np.isnan(Y_test) & ~np.isnan(B_test) & exclude_mask[:, None]
rmse_base = float(np.sqrt(mean_squared_error(Y_test[mask_b_all], B_test[mask_b_all])))
mae_base  = float(mean_absolute_error(Y_test[mask_b_all], B_test[mask_b_all]))

print(f"\n=== Résultats globaux v15 (excl. 13-16 sept) ===")
print(f"  Test set : {exclude_mask.sum()} jours (exclu {n_excluded})")
print(f"  Modèle   — RMSE : {rmse_global:.4f} | MAE : {mae_global:.4f}")
print(f"  Baseline — RMSE : {rmse_base:.4f} | MAE : {mae_base:.4f}")
print(f"  Amélioration RMSE : {(1 - rmse_global / rmse_base) * 100:+.1f}%")
print(f"  Amélioration MAE  : {(1 - mae_global / mae_base) * 100:+.1f}%")

for group, steps in [("NUIT", NIGHT_STEPS), ("JOUR", DAY_STEPS)]:
    y_g = Y_test[:, steps]
    p_g = preds_test[:, steps]
    b_g = B_test[:, steps]
    mask_m = ~np.isnan(y_g) & ~np.isnan(p_g) & exclude_mask[:, None]
    mask_b = ~np.isnan(y_g) & ~np.isnan(b_g) & exclude_mask[:, None]
    rmse_m = float(np.sqrt(mean_squared_error(y_g[mask_m], p_g[mask_m])))
    rmse_b = float(np.sqrt(mean_squared_error(y_g[mask_b], b_g[mask_b])))
    mae_m  = float(mean_absolute_error(y_g[mask_m], p_g[mask_m]))
    mae_b  = float(mean_absolute_error(y_g[mask_b], b_g[mask_b]))
    imp = (1 - rmse_m / rmse_b) * 100
    print(f"  {group:5s} — RMSE modèle={rmse_m:.4f} | baseline={rmse_b:.4f} | {imp:+.1f}% | "
          f"MAE modèle={mae_m:.4f} | baseline={mae_b:.4f}")

print(f"\n=== RMSE par tranche horaire ===")
for h_start in range(0, 24, 3):
    t_start = h_start * 4
    t_end   = min(t_start + 12, n_steps)
    steps   = list(range(t_start, t_end))
    y_s = Y_test[:, steps]
    p_s = preds_test[:, steps]
    b_s = B_test[:, steps]
    mask_m = ~np.isnan(y_s) & ~np.isnan(p_s) & exclude_mask[:, None]
    mask_b = ~np.isnan(y_s) & ~np.isnan(b_s) & exclude_mask[:, None]
    rmse_m = float(np.sqrt(mean_squared_error(y_s[mask_m], p_s[mask_m])))
    rmse_b = float(np.sqrt(mean_squared_error(y_s[mask_b], b_s[mask_b])))
    delta = (1 - rmse_m / rmse_b) * 100
    group = "NUIT" if t_start in night_set else "JOUR"
    print(f"  {h_start:02d}h–{h_start+3:02d}h [{group}] : modèle={rmse_m:.4f} | baseline={rmse_b:.4f} | {delta:+.1f}%")


# ─────────────────────────────────────────────────────────────────────
# 6. FEATURE IMPORTANCE GROUPÉE
# ─────────────────────────────────────────────────────────────────────

for group, steps in [("NUIT", NIGHT_STEPS), ("JOUR", DAY_STEPS)]:
    models = [pickle.load(open(OUT / f"lgbm_t{t:03d}.pkl", "rb")) for t in steps]
    print_grouped_importance(feat_names, models, group)


# ─────────────────────────────────────────────────────────────────────
# 7. DIAGNOSTIC : BIAIS DIURNE PAR MOIS (espace dénormalisé)
# ─────────────────────────────────────────────────────────────────────

print(f"\n=== Diagnostic biais diurne (10h-17h) ===")
day_steps = list(range(40, 68))

monthly_bias = defaultdict(list)
for i, d in enumerate(dates_test_list):
    if not exclude_mask[i]:
        continue
    y_day = Y_test[i, day_steps]
    p_day = preds_test[i, day_steps]
    mask = ~np.isnan(y_day) & ~np.isnan(p_day)
    if mask.sum() == 0:
        continue
    bias = float(np.mean(y_day[mask] - p_day[mask]))
    monthly_bias[f"{d.year}-{d.month:02d}"].append(bias)

print(f"  {'Mois':10s} | {'Nb jours':>8s} | {'Biais moyen':>11s} | {'Interprétation'}")
print(f"  {'-'*10}-+-{'-'*8}-+-{'-'*11}-+-{'-'*30}")
for month_key in sorted(monthly_bias.keys()):
    vals = monthly_bias[month_key]
    mean_bias = np.mean(vals)
    n = len(vals)
    if mean_bias > 0.05:
        interp = "→ surestime PV (prédit trop bas)"
    elif mean_bias < -0.05:
        interp = "→ sous-estime PV (prédit trop haut)"
    else:
        interp = "→ ~neutre"
    print(f"  {month_key:10s} | {n:8d} | {mean_bias:+11.4f} | {interp}")


# ─────────────────────────────────────────────────────────────────────
# 8. SAUVEGARDE FINALE
# ─────────────────────────────────────────────────────────────────────

metrics_df.write_parquet(OUT / "metrics.parquet")

pred_cols = {f"pred_t{t:03d}": preds_test[:, t].tolist() for t in range(n_steps)}
pred_cols["date"] = dates_test.to_list()
pl.DataFrame(pred_cols).select(
    ["date"] + [f"pred_t{t:03d}" for t in range(n_steps)]
).write_parquet(OUT / "predictions_test.parquet")

print(f"\n✓ Modèles : {OUT}/lgbm_t000.pkl … lgbm_t095.pkl")
print(f"✓ Métriques : {OUT}/metrics.parquet")
print(f"✓ Prédictions : {OUT}/predictions_test.parquet (espace dénormalisé)")
print(f"✓ Params nuit : {OUT}/best_params_night.json")
print(f"✓ Params jour : {OUT}/best_params_day.json")