"""
train_lgbm_v12_residual.py
==========================
Entraînement LightGBM sur le RÉSIDU par rapport à J-7.

Target : y_residual[i, t] = load_net[i, t] - load_net[i-7, t]

Au lieu de prédire la valeur absolue du load net, le modèle prédit
le delta par rapport au même jour de la semaine précédente.

Avantage : la composante consommation brute (stable d'une semaine à
l'autre) s'annule dans le delta. Ce qui reste est dominé par les
différences de production PV entre les deux semaines, ce qui force
le modèle à utiliser l'irradiance comme signal principal.

Reconstruction : prediction_absolue[i, t] = load_net[i-7, t] + résidu_prédit[i, t]

Gestion des cas problématiques :
  - Si load J-7 manque : fallback J-14
  - J-7 férié/atypique : le modèle reçoit load J-7 en feature implicite
    (déjà dans load_jm6 = J-1 de J-7... non, load_jm6 = target-7 jours)
    En fait load_jm7 dans les features = target - 8 jours, pas -7.
    On ajoute donc le load J-7 brut par pas comme feature supplémentaire.

Features : v12 + load_j7_tXX (96 features = load net de J-7 par pas de 15min)

Split / Optuna / groupes : identiques au script normal v12.

Sorties :
  DATA/models12_residual/
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

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
BASE = Path(__file__).resolve().parents[2] / "DATA"

X_PATH = BASE / "processed" / "X_features_v12.parquet"
Y_PATH = BASE / "processed" / "Y_target_v12.parquet"
B_PATH = BASE / "processed" / "B_baseline_v12.parquet"
OUT    = BASE / "models12_residual"
OUT.mkdir(parents=True, exist_ok=True)

TRAIN_RATIO = 0.60
VAL_RATIO   = 0.20
J7_OFFSET   = 7  # décalage en jours pour le résidu

N_OPTUNA_TRIALS  = 50
N_ESTIMATORS_MAX = 1000
EARLY_STOPPING   = 50
RANDOM_SEED      = 42

EXCLUDE_DATES = {date(2025, 9, 13), date(2025, 9, 14), date(2025, 9, 15)}

NIGHT_STEPS = list(range(0, 48)) + list(range(72, 96))
DAY_STEPS   = list(range(48, 72))
OPTUNA_STEPS_NIGHT = [0, 12, 28, 72, 84, 92]
OPTUNA_STEPS_DAY   = [48, 52, 56, 60, 68]


# ─────────────────────────────────────────────
# FEATURE IMPORTANCE GROUPÉE
# ─────────────────────────────────────────────

def classify_feature(name: str) -> str:
    if name.startswith("load_j7_"):
        return "Load J-7 (ancrage résidu)"
    if name.startswith("load_"):
        return "Load historique"
    if name.startswith("solar_") or "pv_yield" in name or "pv_capacity" in name or "remote_max" in name:
        return "PV mesuré / capacité"
    if name.startswith("pred_pv_adj"):
        return "PV prévu (interaction)"
    if name.startswith("pred_glob_rad"):
        return "Irradiance prévue J+1"
    if name.startswith("predJ_glob_rad"):
        return "Irradiance prévue J"
    if name.startswith("pred_temp") or name.startswith("pred_pressure") or name.startswith("pred_relhum"):
        return "Météo prévue J+1 (temp/pres/hum)"
    if name.startswith("predJ_temp") or name.startswith("predJ_pressure") or name.startswith("predJ_relhum"):
        return "Météo prévue J (temp/pres/hum)"
    if name.startswith("pred_wind") or name.startswith("pred_precip") or name.startswith("pred_sunshine"):
        return "Météo prévue J+1 (vent/pluie/soleil)"
    if name.startswith("predJ_wind") or name.startswith("predJ_precip") or name.startswith("predJ_sunshine"):
        return "Météo prévue J (vent/pluie/soleil)"
    if name.startswith("rmet_"):
        return "Météo mesurée"
    if name in ("dayofweek", "sin_dow", "cos_dow", "is_weekend", "is_holiday",
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


# ─────────────────────────────────────────────
# 1. CHARGEMENT
# ─────────────────────────────────────────────

print("=== Chargement des données ===")
X_df = pl.read_parquet(X_PATH)
Y_df = pl.read_parquet(Y_PATH)
B_df = pl.read_parquet(B_PATH)

dates = X_df["date"]
X_base = X_df.drop("date").to_numpy().astype(np.float32)
Y_arr  = Y_df.drop("date").to_numpy().astype(np.float32)
B_arr  = B_df.drop("date").to_numpy().astype(np.float32)

n_samples = X_base.shape[0]
n_steps   = Y_arr.shape[1]

print(f"  Samples : {n_samples} jours")
print(f"  Features base : {X_base.shape[1]}")
print(f"  Pas de temps : {n_steps}")


# ─────────────────────────────────────────────
# 2. CONSTRUCTION TARGET RÉSIDUELLE + FEATURES J-7
# ─────────────────────────────────────────────

print(f"\n=== Construction target résiduelle (delta J-7) ===")

# Y_j7[i] = Y[i - 7] = load net du même jour de la semaine précédente
Y_j7 = np.full_like(Y_arr, np.nan)
Y_j7[J7_OFFSET:] = Y_arr[:-J7_OFFSET]

# Target résiduelle = load_net - load_net_j7
Y_residual = Y_arr - Y_j7  # NaN pour les 7 premiers jours

# Fallback J-14 si J-7 manque (rare mais possible)
for i in range(J7_OFFSET, n_samples):
    for t in range(n_steps):
        if np.isnan(Y_j7[i, t]) and i >= 2 * J7_OFFSET:
            Y_j7[i, t] = Y_arr[i - 2 * J7_OFFSET, t]
            Y_residual[i, t] = Y_arr[i, t] - Y_j7[i, t]

# Stats sur le résidu
valid_res = Y_residual[~np.isnan(Y_residual)]
print(f"  Résidu : mean={valid_res.mean():.4f}, std={valid_res.std():.4f}")
print(f"  Résidu : min={valid_res.min():.4f}, max={valid_res.max():.4f}")
print(f"  NaN : {np.isnan(Y_residual).sum()} / {Y_residual.size}")

# Ajouter le load J-7 par pas comme features supplémentaires
# Le modèle en a besoin pour savoir si J-7 était atypique
# Anti-leakage : load J-7 = target_date - 7 jours = J-1 - 6 jours → disponible
feat_names_base = [c for c in X_df.columns if c != "date"]
feat_names_j7 = [f"load_j7_t{t:03d}" for t in range(n_steps)]

X_arr = np.concatenate([X_base, Y_j7], axis=1).astype(np.float32)
feat_names = feat_names_base + feat_names_j7

print(f"  Features totales : {X_arr.shape[1]} ({X_base.shape[1]} base + {n_steps} load J-7)")


# ─────────────────────────────────────────────
# 3. SPLIT
# ─────────────────────────────────────────────

split_train = int(n_samples * TRAIN_RATIO)
split_val   = int(n_samples * (TRAIN_RATIO + VAL_RATIO))

X_train = X_arr[:split_train]
X_val   = X_arr[split_train:split_val]
X_test  = X_arr[split_val:]

# Target résiduelle pour training
YR_train = Y_residual[:split_train]
YR_val   = Y_residual[split_train:split_val]
YR_test  = Y_residual[split_val:]

# Target absolue et J-7 pour reconstruction
Y_test   = Y_arr[split_val:]
Y_j7_test = Y_j7[split_val:]
B_test   = B_arr[split_val:]
dates_test = dates[split_val:]

print(f"\n=== Split chronologique ===")
print(f"  Train : {split_train} jours ({dates[0]} → {dates[split_train-1]})")
print(f"  Val   : {split_val - split_train} jours ({dates[split_train]} → {dates[split_val-1]})")
print(f"  Test  : {n_samples - split_val} jours ({dates[split_val]} → {dates[-1]})")

# Vérifier combien de jours ont un résidu valide
n_valid_train = np.sum(~np.isnan(YR_train[:, 0]))
n_valid_test  = np.sum(~np.isnan(YR_test[:, 0]))
print(f"  Train avec résidu valide : {n_valid_train}/{split_train}")
print(f"  Test  avec résidu valide : {n_valid_test}/{n_samples - split_val}")

# Masque exclusion dates + résidu disponible
dates_test_list = dates_test.to_list()
exclude_mask = np.array([d not in EXCLUDE_DATES for d in dates_test_list], dtype=bool)
j7_available = ~np.isnan(Y_j7_test[:, 0])  # J-7 dispo pour reconstruction
eval_mask = exclude_mask & j7_available
n_excluded = (~exclude_mask).sum()
n_no_j7 = (~j7_available).sum()
print(f"  Dates exclues : {n_excluded} (baseline foireuse)")
print(f"  Dates sans J-7 : {n_no_j7}")
print(f"  Dates évaluables : {eval_mask.sum()}")


# ─────────────────────────────────────────────
# 4. OPTUNA
# ─────────────────────────────────────────────

def run_optuna(group_name, optuna_steps, n_trials):
    def objective(trial):
        params = {
            "objective":       "regression",
            "metric":          "rmse",
            "verbosity":       -1,
            "n_jobs":          -1,
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
            y_tr = YR_train[:, t]
            y_va = YR_val[:, t]
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
    print(f"  Optuna {group_name} — {n_trials} trials")
    print(f"{'='*60}")

    sampler = TPESampler(seed=RANDOM_SEED)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_trial.params
    print(f"\n  Meilleur trial #{study.best_trial.number}")
    print(f"  RMSE validation (résidu) : {study.best_value:.6f}")
    for k, v in best.items():
        print(f"    {k}: {v}")

    return best


best_night = run_optuna("NUIT", OPTUNA_STEPS_NIGHT, N_OPTUNA_TRIALS)
best_day   = run_optuna("JOUR", OPTUNA_STEPS_DAY, N_OPTUNA_TRIALS)

with open(OUT / "best_params_night.json", "w") as f:
    json.dump(best_night, f, indent=2)
with open(OUT / "best_params_day.json", "w") as f:
    json.dump(best_day, f, indent=2)


# ─────────────────────────────────────────────
# 5. ENTRAÎNEMENT FINAL
# ─────────────────────────────────────────────

print(f"\n=== Entraînement final (96 modèles, target=résidu J-7) ===")

X_trainval = np.concatenate([X_train, X_val], axis=0)
YR_trainval = np.concatenate([YR_train, YR_val], axis=0)

night_set = set(NIGHT_STEPS)

# Prédictions du résidu
preds_residual = np.zeros_like(Y_test)
# Prédictions absolues reconstruites
preds_absolute = np.full_like(Y_test, np.nan)
metrics = []

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

    y_tv = YR_trainval[:, t]
    mask_tv = ~np.isnan(y_tv)

    dtrain = lgb.Dataset(X_trainval[mask_tv], label=y_tv[mask_tv],
                         feature_name=feat_names, free_raw_data=False)

    mask_val_t = ~np.isnan(YR_val[:, t])
    dval_es = lgb.Dataset(X_val[mask_val_t], label=YR_val[:, t][mask_val_t],
                          reference=dtrain, free_raw_data=False)

    model = lgb.train(
        final_params, dtrain, num_boost_round=N_ESTIMATORS_MAX,
        valid_sets=[dval_es],
        callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False),
                   lgb.log_evaluation(-1)],
    )

    # Prédire le résidu
    pred_res_t = model.predict(X_test)
    preds_residual[:, t] = pred_res_t

    # Reconstruire la prédiction absolue : load_j7 + résidu prédit
    preds_absolute[:, t] = Y_j7_test[:, t] + pred_res_t
    # Là où J-7 manque, la prédiction reste NaN

    # Métriques sur la prédiction absolue reconstruite
    y_te = Y_test[:, t]
    pred_abs_t = preds_absolute[:, t]
    mask_te = ~np.isnan(y_te) & ~np.isnan(pred_abs_t) & eval_mask

    if mask_te.sum() > 0:
        rmse_m = float(np.sqrt(mean_squared_error(y_te[mask_te], pred_abs_t[mask_te])))
        mae_m  = float(mean_absolute_error(y_te[mask_te], pred_abs_t[mask_te]))
    else:
        rmse_m, mae_m = None, None

    b_t = B_test[:, t]
    mask_b = ~np.isnan(y_te) & ~np.isnan(b_t) & eval_mask
    rmse_b = float(np.sqrt(mean_squared_error(y_te[mask_b], b_t[mask_b]))) if mask_b.sum() > 0 else None
    mae_b  = float(mean_absolute_error(y_te[mask_b], b_t[mask_b]))          if mask_b.sum() > 0 else None

    metrics.append({
        "step": t,
        "time_label": f"{(t * 15) // 60:02d}h{(t * 15) % 60:02d}",
        "group": group_label,
        "rmse_model": rmse_m,
        "mae_model": mae_m,
        "rmse_baseline": rmse_b,
        "mae_baseline": mae_b,
        "n_estimators": model.best_iteration,
    })

    with open(OUT / f"lgbm_t{t:03d}.pkl", "wb") as f:
        pickle.dump(model, f)

    if t % 12 == 0:
        base_str = f"{rmse_b:.4f}" if rmse_b is not None else "N/A"
        model_str = f"{rmse_m:.4f}" if rmse_m is not None else "N/A"
        print(f"  t={t:03d} ({metrics[-1]['time_label']}) [{group_label}] — RMSE model={model_str} | baseline={base_str}")


# ─────────────────────────────────────────────
# 6. MÉTRIQUES
# ─────────────────────────────────────────────

metrics_df = pl.DataFrame(metrics)

# Global
mask_all = ~np.isnan(Y_test) & ~np.isnan(preds_absolute) & eval_mask[:, None]
rmse_global = float(np.sqrt(mean_squared_error(Y_test[mask_all], preds_absolute[mask_all])))
mae_global  = float(mean_absolute_error(Y_test[mask_all], preds_absolute[mask_all]))

mask_b_all = ~np.isnan(Y_test) & ~np.isnan(B_test) & eval_mask[:, None]
rmse_base = float(np.sqrt(mean_squared_error(Y_test[mask_b_all], B_test[mask_b_all])))
mae_base  = float(mean_absolute_error(Y_test[mask_b_all], B_test[mask_b_all]))

# Aussi comparer avec la "baseline J-7 naïve" (prédire load_j7 sans correction)
mask_j7 = ~np.isnan(Y_test) & ~np.isnan(Y_j7_test) & eval_mask[:, None]
rmse_j7_naive = float(np.sqrt(mean_squared_error(Y_test[mask_j7], Y_j7_test[mask_j7])))
mae_j7_naive  = float(mean_absolute_error(Y_test[mask_j7], Y_j7_test[mask_j7]))

print(f"\n=== Résultats globaux (résidu J-7, excl. 13-15 sept) ===")
print(f"  Jours évaluables : {eval_mask.sum()}")
print(f"  Modèle résidu  — RMSE : {rmse_global:.4f} | MAE : {mae_global:.4f}")
print(f"  Baseline Oiken — RMSE : {rmse_base:.4f} | MAE : {mae_base:.4f}")
print(f"  Baseline J-7   — RMSE : {rmse_j7_naive:.4f} | MAE : {mae_j7_naive:.4f}")
print(f"  vs Oiken  RMSE : {(1 - rmse_global / rmse_base) * 100:+.1f}%")
print(f"  vs J-7    RMSE : {(1 - rmse_global / rmse_j7_naive) * 100:+.1f}%")

for group, steps in [("NUIT", NIGHT_STEPS), ("JOUR", DAY_STEPS)]:
    y_g = Y_test[:, steps]
    p_g = preds_absolute[:, steps]
    b_g = B_test[:, steps]
    j7_g = Y_j7_test[:, steps]
    mask_m = ~np.isnan(y_g) & ~np.isnan(p_g) & eval_mask[:, None]
    mask_b = ~np.isnan(y_g) & ~np.isnan(b_g) & eval_mask[:, None]
    mask_j = ~np.isnan(y_g) & ~np.isnan(j7_g) & eval_mask[:, None]
    rmse_m = float(np.sqrt(mean_squared_error(y_g[mask_m], p_g[mask_m])))
    rmse_b = float(np.sqrt(mean_squared_error(y_g[mask_b], b_g[mask_b])))
    rmse_j = float(np.sqrt(mean_squared_error(y_g[mask_j], j7_g[mask_j])))
    imp_b = (1 - rmse_m / rmse_b) * 100
    imp_j = (1 - rmse_m / rmse_j) * 100
    print(f"  {group:5s} — modèle={rmse_m:.4f} | Oiken={rmse_b:.4f} ({imp_b:+.1f}%) | J-7={rmse_j:.4f} ({imp_j:+.1f}%)")

print(f"\n=== RMSE par tranche horaire ===")
for h_start in range(0, 24, 3):
    t_start = h_start * 4
    t_end   = min(t_start + 12, n_steps)
    steps   = list(range(t_start, t_end))
    y_s = Y_test[:, steps]
    p_s = preds_absolute[:, steps]
    b_s = B_test[:, steps]
    mask_m = ~np.isnan(y_s) & ~np.isnan(p_s) & eval_mask[:, None]
    mask_b = ~np.isnan(y_s) & ~np.isnan(b_s) & eval_mask[:, None]
    rmse_m = float(np.sqrt(mean_squared_error(y_s[mask_m], p_s[mask_m])))
    rmse_b = float(np.sqrt(mean_squared_error(y_s[mask_b], b_s[mask_b])))
    delta = (1 - rmse_m / rmse_b) * 100
    group = "NUIT" if t_start in night_set else "JOUR"
    print(f"  {h_start:02d}h–{h_start+3:02d}h [{group}] : modèle={rmse_m:.4f} | baseline={rmse_b:.4f} | {delta:+.1f}%")


# ─────────────────────────────────────────────
# 7. FEATURE IMPORTANCE
# ─────────────────────────────────────────────

for group, steps in [("NUIT", NIGHT_STEPS), ("JOUR", DAY_STEPS)]:
    models = [pickle.load(open(OUT / f"lgbm_t{t:03d}.pkl", "rb")) for t in steps]
    print_grouped_importance(feat_names, models, group)


# ─────────────────────────────────────────────
# 8. DIAGNOSTIC BIAIS
# ─────────────────────────────────────────────

print(f"\n=== Diagnostic biais diurne (09h-15h) ===")
day_steps_diag = list(range(36, 60))

monthly_bias = defaultdict(list)
for i, d in enumerate(dates_test_list):
    if not eval_mask[i]:
        continue
    y_day = Y_test[i, day_steps_diag]
    p_day = preds_absolute[i, day_steps_diag]
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


# ─────────────────────────────────────────────
# 9. SAUVEGARDE
# ─────────────────────────────────────────────

metrics_df.write_parquet(OUT / "metrics.parquet")

pred_cols = {f"pred_t{t:03d}": preds_absolute[:, t].tolist() for t in range(n_steps)}
pred_cols["date"] = dates_test.to_list()
pl.DataFrame(pred_cols).select(
    ["date"] + [f"pred_t{t:03d}" for t in range(n_steps)]
).write_parquet(OUT / "predictions_test.parquet")

# Aussi sauver les résidus prédits pour analyse
res_cols = {f"res_t{t:03d}": preds_residual[:, t].tolist() for t in range(n_steps)}
res_cols["date"] = dates_test.to_list()
pl.DataFrame(res_cols).select(
    ["date"] + [f"res_t{t:03d}" for t in range(n_steps)]
).write_parquet(OUT / "residuals_test.parquet")

print(f"\n✓ Sauvegardé dans : {OUT}")