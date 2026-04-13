"""
train_lgbm_v12.py
=================
Entraînement LightGBM v12 avec features d'interaction PV.

OBJECTIF
--------
Prédire la charge nette normalisée d'Oiken pour les 96 pas de 15 min
du jour J+1, à partir des features v12 (météo prévue, load historique,
production PV mesurée, interactions irradiance × capacité PV).

ARCHITECTURE
------------
- 96 modèles LightGBM indépendants, un par pas de 15 min.
- Split jour/nuit avec hyperparamètres Optuna séparés :
  le signal diurne (dominé par le PV) et nocturne (dominé par la conso)
  ont des structures très différentes et bénéficient d'hyperparamètres
  distincts.

STRATÉGIE D'ÉVALUATION
----------------------
- Split chronologique 60% train / 20% val / 20% test (strict, pas de shuffle).
- Optuna optimise sur val, early stopping sur val, test jamais touché
  pendant l'entraînement.
- Exclusion des 13-15 septembre 2025 des métriques finales : la baseline
  Oiken est anormalement dégradée sur ces 3 jours (probablement un bug
  de leur système de prévision) et gonflerait artificiellement
  l'amélioration du modèle si on les gardait.

SORTIES
-------
  DATA/models12/lgbm_t{000..095}.pkl     — 96 modèles sérialisés
  DATA/models12/metrics.parquet          — RMSE/MAE par pas de temps
  DATA/models12/predictions_test.parquet — prédictions sur le test set
  DATA/models12/best_params_night.json   — hyperparamètres NUIT
  DATA/models12/best_params_day.json     — hyperparamètres JOUR
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

# Chemins : le script est censé tourner depuis src/ML/, remonte 2 niveaux pour DATA/
BASE = Path(__file__).resolve().parents[2] / "DATA"

X_PATH = BASE / "processed" / "X_features_v12.parquet"   # features v12
Y_PATH = BASE / "processed" / "Y_target_v12.parquet"     # cible : load net normalisé (96 pas)
B_PATH = BASE / "processed" / "B_baseline_v12.parquet"   # baseline Oiken pour comparaison
OUT    = BASE / "models12"
OUT.mkdir(parents=True, exist_ok=True)

# Split chronologique strict (pas de shuffle car données temporelles)
TRAIN_RATIO = 0.60
VAL_RATIO   = 0.20
# Le test prend le reste (20%) = la période la plus récente

# Hyperparamètres d'entraînement
N_OPTUNA_TRIALS  = 50     # nombre de combinaisons testées par Optuna par groupe
N_ESTIMATORS_MAX = 1000   # plafond de boosting rounds (early stopping coupe avant)
EARLY_STOPPING   = 50     # arrêt si pas d'amélioration val sur 50 rounds consécutifs
RANDOM_SEED      = 42     # reproductibilité du sampler Optuna

# Jours où la baseline Oiken a des valeurs aberrantes (trou de données dans leur système)
# Exclus UNIQUEMENT des métriques finales, pas de l'entraînement
EXCLUDE_DATES = {date(2025, 9, 13), date(2025, 9, 14), date(2025, 9, 15)}

# ─────────────────────────────────────────────────────────────────────
# GROUPES JOUR / NUIT
# ─────────────────────────────────────────────────────────────────────
# Les 96 pas de 15 min représentent 00h00 à 23h45 UTC.
# step 0 = 00h00, step 48 = 12h00, step 72 = 18h00, step 95 = 23h45.
#
# On sépare en 2 groupes parce que le signal est structurellement différent :
#   - NUIT : pas de PV → conso brute → très prédictible, signal simple
#   - JOUR : conso brute - PV → signal complexe, bruit PV dominant
#
# ATTENTION : le docstring de la config dit "JOUR = 09h-14h45" mais
# les indices ci-dessous définissent JOUR = 12h-17h45. Ce sont les
# indices qui font foi. À harmoniser dans une prochaine version ou
# à vérifier lequel des deux est le choix voulu — c'est un point à clarifier
# car selon le choix, le biais diurne observé peut se déplacer de tranche.
NIGHT_STEPS = list(range(0, 48)) + list(range(72, 96))   # 00h-11h45 + 18h-23h45
DAY_STEPS   = list(range(48, 72))                         # 12h-17h45

# Optuna ne teste pas tous les pas (trop coûteux) : on sélectionne quelques
# pas représentatifs pour évaluer la qualité d'un jeu d'hyperparamètres.
# Le RMSE moyen sur ces pas sert de score à minimiser.
# Le choix des pas doit couvrir les heures caractéristiques du groupe.
OPTUNA_STEPS_DAY   = [48, 52, 56, 60, 68]   # 12h, 13h, 14h, 15h, 17h — cœur du jour
OPTUNA_STEPS_NIGHT = [0, 12, 28, 72, 84, 92]  # 00h, 03h, 07h, 18h, 21h, 23h


# ─────────────────────────────────────────────────────────────────────
# FEATURE IMPORTANCE GROUPÉE PAR THÈME
# ─────────────────────────────────────────────────────────────────────
# Avec ~3584 features, lire le top 15 brut n'est pas informatif.
# On agrège les importances par catégorie thématique pour voir d'un
# coup d'œil si le modèle utilise les "bonnes" familles de features
# (ex. beaucoup de poids sur "PV prévu" en JOUR est un bon signe).

def classify_feature(name: str) -> str:
    """Attribue un thème à une feature selon son préfixe/contenu."""
    # Load historique (le plus gros groupe : 196 features, J-1 à J-7)
    if name.startswith("load_"):
        return "Load historique"
    # Production PV mesurée + capacité + ratios de yield
    if name.startswith("solar_") or "pv_yield" in name or "pv_capacity" in name or "remote_max" in name:
        return "PV mesuré / capacité"
    # Features d'interaction irradiance × pv_yield (nouveautés v12)
    if name.startswith("pred_pv_adj"):
        return "PV prévu (interaction)"
    # Irradiance prévue brute
    if name.startswith("pred_glob_rad"):
        return "Irradiance prévue J+1"
    if name.startswith("predJ_glob_rad"):
        return "Irradiance prévue J"
    # Variables météo prévues J+1 (hors vent/pluie/soleil qui sont groupés à part)
    if name.startswith("pred_temp") or name.startswith("pred_pressure") or name.startswith("pred_relhum"):
        return "Météo prévue J+1 (temp/pres/hum)"
    if name.startswith("predJ_temp") or name.startswith("predJ_pressure") or name.startswith("predJ_relhum"):
        return "Météo prévue J (temp/pres/hum)"
    # Vent + précipitations + ensoleillement (variables plus bruitées ou proxy indirect)
    if name.startswith("pred_wind") or name.startswith("pred_precip") or name.startswith("pred_sunshine"):
        return "Météo prévue J+1 (vent/pluie/soleil)"
    if name.startswith("predJ_wind") or name.startswith("predJ_precip") or name.startswith("predJ_sunshine"):
        return "Météo prévue J (vent/pluie/soleil)"
    # Mesures météo réelles (J-1 + J matin jusqu'à 10h)
    if name.startswith("rmet_"):
        return "Météo mesurée"
    # Calendaire
    if name in ("dayofweek", "sin_dow", "cos_dow", "is_weekend", "is_holiday",
                "month", "sin_month", "cos_month", "sin_doy", "cos_doy"):
        return "Calendaire"
    if "ramadan" in name:
        return "Ramadan"
    return "Autre"


def print_grouped_importance(feat_names, models, group_name, top_n=20):
    """
    Agrège les feature importances sur l'ensemble des modèles d'un groupe,
    affiche la répartition par thème + le top N des features individuelles.
    """
    # Moyenne des importances sur tous les modèles du groupe
    imp = np.zeros(len(feat_names))
    for m in models:
        imp += m.feature_importance()
    imp /= len(models)

    # Somme par thème
    theme_imp = {}
    for name, val in zip(feat_names, imp):
        theme = classify_feature(name)
        theme_imp[theme] = theme_imp.get(theme, 0) + val
    total = sum(theme_imp.values())

    # Affichage par thème avec bar chart ASCII (1 bloc = 2%)
    print(f"\n{'='*60}")
    print(f"  Feature importance {group_name} — par thème")
    print(f"{'='*60}")
    for theme, val in sorted(theme_imp.items(), key=lambda x: -x[1]):
        pct = val / total * 100 if total > 0 else 0
        bar = "█" * int(pct / 2)
        print(f"  {pct:5.1f}%  {bar:25s}  {theme}")

    # Top N features individuelles avec leur thème entre crochets
    print(f"\n  Top {top_n} features {group_name} :")
    top = sorted(zip(feat_names, imp), key=lambda x: -x[1])[:top_n]
    for name, val in top:
        theme = classify_feature(name)
        print(f"  {val:8.1f}  [{theme:.<30s}]  {name}")


# ─────────────────────────────────────────────────────────────────────
# 1. CHARGEMENT DES DONNÉES
# ─────────────────────────────────────────────────────────────────────

print("=== Chargement des données ===")
X = pl.read_parquet(X_PATH)
Y = pl.read_parquet(Y_PATH)
B = pl.read_parquet(B_PATH)

# Extraction des dates (index chronologique) et des noms de features
dates = X["date"]
feat_names = [c for c in X.columns if c != "date"]

# Conversion en numpy float32 pour LightGBM (plus rapide, moins de mémoire)
# LightGBM gère nativement les NaN, pas besoin de les imputer
X_arr = X.drop("date").to_numpy().astype(np.float32)
Y_arr = Y.drop("date").to_numpy().astype(np.float32)
B_arr = B.drop("date").to_numpy().astype(np.float32)

n_samples = X_arr.shape[0]   # nombre de jours
n_steps   = Y_arr.shape[1]   # 96 pas de 15 min

print(f"  Samples : {n_samples} jours")
print(f"  Features : {X_arr.shape[1]}")
print(f"  Pas de temps : {n_steps}")
# REMARQUE : les libellés ci-dessous reflètent le choix NIGHT/DAY réel (indices 48-72)
print(f"  Groupe NUIT : {len(NIGHT_STEPS)} pas (00h-11h45 + 18h-23h45)")
print(f"  Groupe JOUR : {len(DAY_STEPS)} pas (12h-17h45)")

# ─────────────────────────────────────────────────────────────────────
# 2. SPLIT TRAIN / VAL / TEST (chronologique strict)
# ─────────────────────────────────────────────────────────────────────
# Pas de shuffle : on respecte l'ordre temporel pour éviter toute fuite
# de futur vers passé et simuler les conditions de production.

split_train = int(n_samples * TRAIN_RATIO)                  # fin du train
split_val   = int(n_samples * (TRAIN_RATIO + VAL_RATIO))    # fin du val

# Découpage par slicing numpy (contigu en mémoire)
X_train, X_val, X_test = X_arr[:split_train], X_arr[split_train:split_val], X_arr[split_val:]
Y_train, Y_val, Y_test = Y_arr[:split_train], Y_arr[split_train:split_val], Y_arr[split_val:]
B_test                  = B_arr[split_val:]    # baseline seulement sur le test
dates_test              = dates[split_val:]

print(f"\n=== Split chronologique ===")
print(f"  Train : {split_train} jours ({dates[0]} → {dates[split_train-1]})")
print(f"  Val   : {split_val - split_train} jours ({dates[split_train]} → {dates[split_val-1]})")
print(f"  Test  : {n_samples - split_val} jours ({dates[split_val]} → {dates[-1]})")

# Masque booléen pour exclure les 13-15 septembre des métriques finales.
# Ces 3 jours sont gardés dans le test set mais ne comptent pas dans
# le calcul des RMSE/MAE : la baseline Oiken y est aberrante.
dates_test_list = dates_test.to_list()
exclude_mask = np.array([
    d not in EXCLUDE_DATES for d in dates_test_list
], dtype=bool)
n_excluded = (~exclude_mask).sum()
print(f"  Dates exclues des métriques : {n_excluded} ({[str(d) for d in sorted(EXCLUDE_DATES)]})")


# ─────────────────────────────────────────────────────────────────────
# 3. OPTUNA — TUNING SÉPARÉ JOUR / NUIT
# ─────────────────────────────────────────────────────────────────────
# Justification du split : un seul tuning Optuna sur tous les pas
# donnait des hyperparamètres compromis (favorables à la nuit qui
# domine en nombre de pas). Séparer les deux permet à chaque groupe
# d'avoir des réglages adaptés à son type de signal.

def run_optuna(group_name, optuna_steps, n_trials):
    """
    Lance un tuning Optuna pour un groupe (NUIT ou JOUR).
    Pour chaque trial, entraîne un modèle par pas dans `optuna_steps`
    et retourne la moyenne des RMSE sur validation comme score.
    """

    def objective(trial):
        # Espace de recherche LightGBM — conservateur, couvre les cas usuels
        params = {
            "objective":       "regression",
            "metric":          "rmse",
            "verbosity":       -1,         # silence LightGBM
            "n_jobs":          -1,         # tous les cœurs dispo
            "learning_rate":   trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves":      trial.suggest_int("num_leaves", 15, 127),
            "max_depth":       trial.suggest_int("max_depth", 3, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "subsample":       trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
            "reg_alpha":       trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda":      trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "min_split_gain":  trial.suggest_float("min_split_gain", 0.0, 1.0),
        }

        rmse_list = []
        for t in optuna_steps:
            y_tr = Y_train[:, t]
            y_va = Y_val[:, t]
            # Masquer les NaN (jours sans cible valide pour ce pas)
            mask_tr = ~np.isnan(y_tr)
            mask_va = ~np.isnan(y_va)
            # Sécurité : au moins 10 samples sinon on skip ce pas
            if mask_tr.sum() < 10 or mask_va.sum() < 10:
                continue

            # Datasets LightGBM natifs pour ce pas de temps
            dtrain = lgb.Dataset(X_train[mask_tr], label=y_tr[mask_tr],
                                 feature_name=feat_names, free_raw_data=False)
            dval = lgb.Dataset(X_val[mask_va], label=y_va[mask_va],
                               reference=dtrain, free_raw_data=False)

            # Entraînement avec early stopping sur val
            model = lgb.train(
                params, dtrain, num_boost_round=N_ESTIMATORS_MAX,
                valid_sets=[dval],
                callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False),
                           lgb.log_evaluation(-1)],
            )
            pred = model.predict(X_val[mask_va])
            rmse_list.append(float(np.sqrt(mean_squared_error(y_va[mask_va], pred))))

        # Score du trial = moyenne RMSE sur les pas testés
        return float(np.mean(rmse_list)) if rmse_list else float("inf")

    print(f"\n{'='*60}")
    print(f"  Optuna {group_name} — {n_trials} trials (pas: {optuna_steps})")
    print(f"{'='*60}")

    # TPE = Tree-structured Parzen Estimator, l'algo par défaut d'Optuna,
    # bon compromis entre exploration et exploitation
    sampler = TPESampler(seed=RANDOM_SEED)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_trial.params
    print(f"\n  Meilleur trial #{study.best_trial.number}")
    print(f"  RMSE validation : {study.best_value:.6f}")
    for k, v in best.items():
        print(f"    {k}: {v}")

    return best


# Lancement des deux tunings successifs
best_night = run_optuna("NUIT", OPTUNA_STEPS_NIGHT, N_OPTUNA_TRIALS)
best_day   = run_optuna("JOUR", OPTUNA_STEPS_DAY, N_OPTUNA_TRIALS)

# Sauvegarde des hyperparamètres pour reproductibilité
with open(OUT / "best_params_night.json", "w") as f:
    json.dump(best_night, f, indent=2)
with open(OUT / "best_params_day.json", "w") as f:
    json.dump(best_day, f, indent=2)


# ─────────────────────────────────────────────────────────────────────
# 4. ENTRAÎNEMENT FINAL
# ─────────────────────────────────────────────────────────────────────
# On entraîne maintenant sur train+val concaténés (80% du dataset),
# avec les hyperparamètres Optuna fixés. Un modèle par pas de temps.
# Early stopping toujours sur val pour arrêter à la bonne itération
# sans toucher le test set.

print(f"\n=== Entraînement final (96 modèles, hyperparamètres séparés) ===")

# Concaténation train+val pour maximiser les données d'entraînement final
X_trainval = np.concatenate([X_train, X_val], axis=0)
Y_trainval = np.concatenate([Y_train, Y_val], axis=0)

# Set pour lookup O(1) lors de la boucle
night_set = set(NIGHT_STEPS)

# Conteneurs pour les prédictions et les métriques
preds_test = np.zeros_like(Y_test)
metrics    = []

for t in range(n_steps):
    # Choix du jeu d'hyperparamètres selon le groupe
    is_night = t in night_set
    best_params = best_night if is_night else best_day
    group_label = "NUIT" if is_night else "JOUR"

    # Params finaux = params Optuna + boilerplate
    final_params = {
        "objective": "regression",
        "metric":    "rmse",
        "verbosity": -1,
        "n_jobs":    -1,
        **best_params,
    }

    # Target pour ce pas de temps
    y_tv = Y_trainval[:, t]
    y_te = Y_test[:, t]
    mask_tv = ~np.isnan(y_tv)
    # Masque test = NaN + exclusion des 3 jours aberrants
    mask_te = ~np.isnan(y_te) & exclude_mask

    # Dataset d'entraînement (train + val concaténés)
    dtrain = lgb.Dataset(X_trainval[mask_tv], label=y_tv[mask_tv],
                         feature_name=feat_names, free_raw_data=False)

    # Dataset d'early stopping : on utilise SEULEMENT le val set
    # (pas le test set !) pour décider quand arrêter le boosting.
    # C'est critique : utiliser le test set ici ferait fuiter de l'info
    # et biaiserait les métriques finales.
    mask_val_t = ~np.isnan(Y_val[:, t])
    dval_es = lgb.Dataset(X_val[mask_val_t], label=Y_val[:, t][mask_val_t],
                          reference=dtrain, free_raw_data=False)

    # REMARQUE : il y a une petite incohérence ici — on entraîne sur
    # X_trainval (qui inclut déjà X_val) ET on utilise X_val pour
    # l'early stopping. LightGBM va donc valider sur des données
    # qu'il a aussi vues en training, ce qui fait perdre le signal
    # de régularisation de l'early stopping (loss val → 0 rapidement).
    # Version plus propre : entraîner sur X_train seul avec early stopping
    # sur X_val, récupérer le best_iteration, puis réentraîner sur
    # X_trainval avec num_boost_round=best_iteration fixé.
    # À considérer pour une version propre du script.

    model = lgb.train(
        final_params, dtrain, num_boost_round=N_ESTIMATORS_MAX,
        valid_sets=[dval_es],
        callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False),
                   lgb.log_evaluation(-1)],
    )

    # Prédiction sur le test set
    pred_t = model.predict(X_test)
    preds_test[:, t] = pred_t

    # Métriques modèle sur test (avec exclusion des dates aberrantes)
    if mask_te.sum() > 0:
        rmse_m = float(np.sqrt(mean_squared_error(y_te[mask_te], pred_t[mask_te])))
        mae_m  = float(mean_absolute_error(y_te[mask_te], pred_t[mask_te]))
    else:
        rmse_m, mae_m = None, None

    # Métriques baseline Oiken sur les mêmes jours (masque sync avec NaN baseline)
    b_t = B_test[:, t]
    mask_b = ~np.isnan(y_te) & ~np.isnan(b_t) & exclude_mask
    rmse_b = float(np.sqrt(mean_squared_error(y_te[mask_b], b_t[mask_b]))) if mask_b.sum() > 0 else None
    mae_b  = float(mean_absolute_error(y_te[mask_b], b_t[mask_b]))          if mask_b.sum() > 0 else None

    # Enregistrement pour le parquet de métriques
    metrics.append({
        "step":           t,
        "time_label":     f"{(t * 15) // 60:02d}h{(t * 15) % 60:02d}",
        "group":          group_label,
        "rmse_model":     rmse_m,
        "mae_model":      mae_m,
        "rmse_baseline":  rmse_b,
        "mae_baseline":   mae_b,
        "n_estimators":   model.best_iteration,
    })

    # Sauvegarde du modèle (pickle → chargement rapide pour inférence)
    with open(OUT / f"lgbm_t{t:03d}.pkl", "wb") as f:
        pickle.dump(model, f)

    # Log tous les 12 pas (toutes les 3h) pour suivre la progression
    if t % 12 == 0:
        base_str = f"{rmse_b:.4f}" if rmse_b is not None else "N/A"
        model_str = f"{rmse_m:.4f}" if rmse_m is not None else "N/A"
        print(f"  t={t:03d} ({metrics[-1]['time_label']}) [{group_label}] — RMSE model={model_str} | baseline={base_str}")


# ─────────────────────────────────────────────────────────────────────
# 5. MÉTRIQUES GLOBALES ET PAR TRANCHE HORAIRE
# ─────────────────────────────────────────────────────────────────────

metrics_df = pl.DataFrame(metrics)

# ── Métriques globales (toutes tranches confondues) ───────────────
# Le masque 2D exclude_mask[:, None] est broadcasté sur les 96 colonnes :
# chaque ligne (jour) est soit entièrement exclue soit entièrement incluse.
mask_all = ~np.isnan(Y_test) & ~np.isnan(preds_test) & exclude_mask[:, None]
rmse_global = float(np.sqrt(mean_squared_error(Y_test[mask_all], preds_test[mask_all])))
mae_global  = float(mean_absolute_error(Y_test[mask_all], preds_test[mask_all]))

mask_b_all = ~np.isnan(Y_test) & ~np.isnan(B_test) & exclude_mask[:, None]
rmse_base = float(np.sqrt(mean_squared_error(Y_test[mask_b_all], B_test[mask_b_all])))
mae_base  = float(mean_absolute_error(Y_test[mask_b_all], B_test[mask_b_all]))

print(f"\n=== Résultats globaux (excl. 13-15 sept) ===")
print(f"  Test set : {exclude_mask.sum()} jours (exclu {n_excluded})")
print(f"  Modèle   — RMSE : {rmse_global:.4f} | MAE : {mae_global:.4f}")
print(f"  Baseline — RMSE : {rmse_base:.4f} | MAE : {mae_base:.4f}")
print(f"  Amélioration RMSE : {(1 - rmse_global / rmse_base) * 100:+.1f}%")
print(f"  Amélioration MAE  : {(1 - mae_global / mae_base) * 100:+.1f}%")

# ── Métriques par groupe (NUIT vs JOUR) ───────────────────────────
# Permet de voir où le modèle gagne ou perd vs la baseline.
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
    print(f"  {group:5s} — RMSE modèle={rmse_m:.4f} | baseline={rmse_b:.4f} | {imp:+.1f}% | MAE modèle={mae_m:.4f} | baseline={mae_b:.4f}")

# ── Métriques par tranche de 3h ───────────────────────────────────
# Granularité fine pour localiser précisément où sont les gains/pertes.
# Le RMSE nocturne étant beaucoup plus petit que diurne, cette vue
# est essentielle : un RMSE global masque souvent de grosses disparités.
print(f"\n=== RMSE par tranche horaire ===")
for h_start in range(0, 24, 3):
    t_start = h_start * 4                           # 4 pas par heure
    t_end   = min(t_start + 12, n_steps)            # +3h = 12 pas
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
# Recharge tous les modèles de chaque groupe et affiche les thèmes
# dominants + le top 20. Sert à diagnostiquer ce que le modèle "regarde"
# et à orienter les itérations de feature engineering.

for group, steps in [("NUIT", NIGHT_STEPS), ("JOUR", DAY_STEPS)]:
    models = [pickle.load(open(OUT / f"lgbm_t{t:03d}.pkl", "rb")) for t in steps]
    print_grouped_importance(feat_names, models, group)


# ─────────────────────────────────────────────────────────────────────
# 7. DIAGNOSTIC : BIAIS DIURNE PAR MOIS
# ─────────────────────────────────────────────────────────────────────
# Sur la tranche diurne 09h-15h UTC (indices 36-59, couvrant le pic PV),
# on calcule pour chaque jour le biais moyen = moyenne(réel - prédit).
# Puis on agrège par mois pour voir si le biais a une saisonnalité,
# ce qui signalerait un problème structurel de modélisation PV.
#
# REMARQUE : la tranche diagnostic (36-59 = 09h-14h45) est FIXE ici,
# indépendamment de la définition DAY_STEPS du modèle. C'est voulu :
# on veut toujours diagnostiquer les heures à fort PV, pas les heures
# que le modèle considère comme "JOUR".
#
# Interprétation du signe :
#   biais > 0 : réel > prédit → prédit trop bas → modèle surestime le PV
#               (soustrait trop de PV → load net sous-évalué)
#   biais < 0 : réel < prédit → prédit trop haut → modèle sous-estime le PV
#               (soustrait pas assez de PV → load net sur-évalué)

print(f"\n=== Diagnostic biais diurne (09h-15h) ===")
day_steps = list(range(36, 60))   # 09h00-14h45 UTC, tranche pic PV

# Biais journalier agrégé par mois
monthly_bias = defaultdict(list)
for i, d in enumerate(dates_test_list):
    if not exclude_mask[i]:   # on saute les jours exclus
        continue
    y_day = Y_test[i, day_steps]
    p_day = preds_test[i, day_steps]
    mask = ~np.isnan(y_day)
    if mask.sum() == 0:
        continue
    # Biais = moyenne des (réel - prédit) sur la tranche diurne
    bias = float(np.mean(y_day[mask] - p_day[mask]))
    monthly_bias[f"{d.year}-{d.month:02d}"].append(bias)

print(f"  {'Mois':10s} | {'Nb jours':>8s} | {'Biais moyen':>11s} | {'Interprétation'}")
print(f"  {'-'*10}-+-{'-'*8}-+-{'-'*11}-+-{'-'*30}")
for month_key in sorted(monthly_bias.keys()):
    vals = monthly_bias[month_key]
    mean_bias = np.mean(vals)
    n = len(vals)
    # Seuil ±0.05 en unités normalisées (σ ≈ 34.46 MW → ~1.7 MW)
    if mean_bias > 0.05:
        interp = "→ surestime PV (prédit trop bas)"
    elif mean_bias < -0.05:
        interp = "→ sous-estime PV- (prédit trop haut)"
    else:
        interp = "→ ~neutre"
    print(f"  {month_key:10s} | {n:8d} | {mean_bias:+11.4f} | {interp}")


# ─────────────────────────────────────────────────────────────────────
# 8. SAUVEGARDE FINALE
# ─────────────────────────────────────────────────────────────────────

# Métriques au format parquet pour analyses ultérieures (Polars friendly)
metrics_df.write_parquet(OUT / "metrics.parquet")

# Prédictions test set : une colonne par pas de temps + colonne date
pred_cols = {f"pred_t{t:03d}": preds_test[:, t].tolist() for t in range(n_steps)}
pred_cols["date"] = dates_test.to_list()
pl.DataFrame(pred_cols).select(
    ["date"] + [f"pred_t{t:03d}" for t in range(n_steps)]
).write_parquet(OUT / "predictions_test.parquet")

print(f"\n✓ Modèles : {OUT}/lgbm_t000.pkl … lgbm_t095.pkl")
print(f"✓ Métriques : {OUT}/metrics.parquet")
print(f"✓ Prédictions : {OUT}/predictions_test.parquet")
print(f"✓ Params nuit : {OUT}/best_params_night.json")
print(f"✓ Params jour : {OUT}/best_params_day.json")