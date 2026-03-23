"""
train_lgbm_final.py
===================
Entraînement final LightGBM avec hyperparamètres fixés (meilleur run v9).
Pas d'Optuna — résultat reproductible.

Split : 80% train / 20% test (pas besoin de val set sans Optuna)
Post-processing : correction de biais glissante sur les 30 derniers jours.

Sorties :
  DATA/models_final/lgbm_t{000..095}.pkl
  DATA/models_final/metrics.parquet
  DATA/models_final/predictions_test.parquet
  DATA/models_final/predictions_test_corrected.parquet
"""

import polars as pl
import numpy as np
import pickle
import json
from pathlib import Path
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
BASE = Path(r"C:\Users\gab1a\OneDrive\Documents\energyinfo2\DATA")

X_PATH = BASE / "processed" / "X_features_v11.parquet"
Y_PATH = BASE / "processed" / "Y_target_v11.parquet"
B_PATH = BASE / "processed" / "B_baseline_v11.parquet"
OUT    = BASE / "models_final"
OUT.mkdir(exist_ok=True)

TEST_RATIO = 0.20
N_ESTIMATORS_MAX = 1000
EARLY_STOPPING   = 50

# Groupes jour/nuit
NIGHT_STEPS = list(range(0, 36)) + list(range(60, 96))  # 00h-08h45 + 15h-23h45
DAY_STEPS   = list(range(36, 60))                         # 09h-14h45

# Hyperparamètres fixés (meilleur run v9)
PARAMS_NIGHT = {
    "objective": "regression", "metric": "rmse", "verbosity": -1, "n_jobs": -1,
    "learning_rate": 0.01255988014851149,
    "num_leaves": 60,
    "max_depth": 12,
    "min_child_samples": 39,
    "subsample": 0.5479414561012861,
    "colsample_bytree": 0.39430307346641236,
    "reg_alpha": 2.9644843466351587e-07,
    "reg_lambda": 0.0011747388320672474,
    "min_split_gain": 0.016105352900692253,
}

PARAMS_DAY = {
    "objective": "regression", "metric": "rmse", "verbosity": -1, "n_jobs": -1,
    "learning_rate": 0.018373652925318132,
    "num_leaves": 96,
    "max_depth": 11,
    "min_child_samples": 51,
    "subsample": 0.5137214945056575,
    "colsample_bytree": 0.3843138591455166,
    "reg_alpha": 1.0323277691235342e-08,
    "reg_lambda": 3.3347014873972953e-06,
    "min_split_gain": 0.06616689386745139,
}

# Fenêtre glissante pour correction de biais (jours)
BIAS_WINDOW = 30


# ─────────────────────────────────────────────
# 1. CHARGEMENT
# ─────────────────────────────────────────────

print("=== Chargement des données ===")
X = pl.read_parquet(X_PATH)
Y = pl.read_parquet(Y_PATH)
B = pl.read_parquet(B_PATH)

dates = X["date"]
feat_names = [c for c in X.columns if c != "date"]
X_arr = X.drop("date").to_numpy().astype(np.float32)
Y_arr = Y.drop("date").to_numpy().astype(np.float32)
B_arr = B.drop("date").to_numpy().astype(np.float32)

n_samples = X_arr.shape[0]
n_steps   = Y_arr.shape[1]

print(f"  Samples : {n_samples} jours")
print(f"  Features : {X_arr.shape[1]}")
print(f"  Pas de temps : {n_steps}")

# ─────────────────────────────────────────────
# 2. SPLIT TRAIN / TEST (80/20 chronologique)
# ─────────────────────────────────────────────

split_idx = int(n_samples * (1 - TEST_RATIO))

X_train, X_test = X_arr[:split_idx], X_arr[split_idx:]
Y_train, Y_test = Y_arr[:split_idx], Y_arr[split_idx:]
B_test          = B_arr[split_idx:]
dates_train     = dates[:split_idx]
dates_test      = dates[split_idx:]

print(f"\n=== Split train/test ===")
print(f"  Train : {split_idx} jours ({dates[0]} → {dates[split_idx-1]})")
print(f"  Test  : {n_samples - split_idx} jours ({dates[split_idx]} → {dates[-1]})")

# ─────────────────────────────────────────────
# 3. ENTRAÎNEMENT — 96 MODÈLES
# ─────────────────────────────────────────────

print(f"\n=== Entraînement (hyperparamètres fixés, 96 modèles) ===")

night_set = set(NIGHT_STEPS)
preds_test = np.zeros_like(Y_test)
metrics = []

for t in range(n_steps):
    is_night = t in night_set
    params = PARAMS_NIGHT if is_night else PARAMS_DAY
    group_label = "NUIT" if is_night else "JOUR"

    y_train_t = Y_train[:, t]
    y_test_t  = Y_test[:, t]
    mask_train = ~np.isnan(y_train_t)
    mask_test  = ~np.isnan(y_test_t)

    dtrain = lgb.Dataset(
        X_train[mask_train], label=y_train_t[mask_train],
        feature_name=feat_names, free_raw_data=False,
    )
    dtest = lgb.Dataset(
        X_test[mask_test], label=y_test_t[mask_test],
        reference=dtrain, free_raw_data=False,
    )

    model = lgb.train(
        params, dtrain, num_boost_round=N_ESTIMATORS_MAX,
        valid_sets=[dtest],
        callbacks=[
            lgb.early_stopping(EARLY_STOPPING, verbose=False),
            lgb.log_evaluation(-1),
        ],
    )

    pred_t = model.predict(X_test)
    preds_test[:, t] = pred_t

    rmse_m = float(np.sqrt(mean_squared_error(y_test_t[mask_test], pred_t[mask_test])))
    mae_m  = float(mean_absolute_error(y_test_t[mask_test], pred_t[mask_test]))

    b_t = B_test[:, t]
    mask_b = ~np.isnan(y_test_t) & ~np.isnan(b_t)
    rmse_b = float(np.sqrt(mean_squared_error(y_test_t[mask_b], b_t[mask_b]))) if mask_b.sum() > 0 else None
    mae_b  = float(mean_absolute_error(y_test_t[mask_b], b_t[mask_b]))          if mask_b.sum() > 0 else None

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
        print(f"  t={t:03d} ({metrics[-1]['time_label']}) [{group_label}] — RMSE={rmse_m:.4f} | base={base_str}")


# ─────────────────────────────────────────────
# 4. CORRECTION DE BIAIS (post-processing)
# ─────────────────────────────────────────────

print(f"\n=== Correction de biais glissante ({BIAS_WINDOW} jours) ===")

# Pour chaque jour du test set, calculer le biais sur les BIAS_WINDOW jours
# précédents CONNUS (train set ou test set déjà passé).
# En production: on connaît le réel de J-2 (dispo à 2h), donc on peut
# recalculer le biais sur les 30 derniers jours complets.

# Ici on simule: pour le jour i du test set, le biais est calculé sur
# les jours i-BIAS_WINDOW à i-1 dans l'ensemble complet (train+test).
# Les jours de test "passés" ont un réel connu (simulation de production).

Y_all = Y_arr  # tout le dataset
P_all = np.full_like(Y_all, np.nan)
P_all[split_idx:] = preds_test  # seul le test set a des prédictions

# Pour le train set, on fait des prédictions in-sample (pour le calcul du biais initial)
print("  Calcul prédictions in-sample (train) pour biais initial...")
preds_train = np.zeros_like(Y_train)
for t in range(n_steps):
    model = pickle.load(open(OUT / f"lgbm_t{t:03d}.pkl", "rb"))
    preds_train[:, t] = model.predict(X_train)
P_all[:split_idx] = preds_train

# Correction glissante
preds_corrected = np.copy(preds_test)

for i in range(len(preds_test)):
    global_idx = split_idx + i
    # Fenêtre: les BIAS_WINDOW jours précédents
    window_start = max(0, global_idx - BIAS_WINDOW)
    window_end = global_idx  # exclu (pas le jour actuel)

    if window_end <= window_start:
        continue

    y_window = Y_all[window_start:window_end]
    p_window = P_all[window_start:window_end]

    # Biais par pas
    bias = np.nanmean(p_window - y_window, axis=0)
    preds_corrected[i] = preds_test[i] - bias


# ─────────────────────────────────────────────
# 5. MÉTRIQUES
# ─────────────────────────────────────────────

metrics_df = pl.DataFrame(metrics)

# --- Sans correction ---
mask_all = ~np.isnan(Y_test) & ~np.isnan(preds_test)
rmse_orig = float(np.sqrt(mean_squared_error(Y_test[mask_all], preds_test[mask_all])))
mae_orig  = float(mean_absolute_error(Y_test[mask_all], preds_test[mask_all]))

# --- Avec correction ---
mask_corr = ~np.isnan(Y_test) & ~np.isnan(preds_corrected)
rmse_corr = float(np.sqrt(mean_squared_error(Y_test[mask_corr], preds_corrected[mask_corr])))
mae_corr  = float(mean_absolute_error(Y_test[mask_corr], preds_corrected[mask_corr]))

# --- Baseline ---
mask_b = ~np.isnan(Y_test) & ~np.isnan(B_test)
rmse_base = float(np.sqrt(mean_squared_error(Y_test[mask_b], B_test[mask_b])))
mae_base  = float(mean_absolute_error(Y_test[mask_b], B_test[mask_b]))

print(f"\n=== Résultats globaux ===")
print(f"  ML brut       — RMSE : {rmse_orig:.4f} | MAE : {mae_orig:.4f} | vs baseline: {(1-rmse_orig/rmse_base)*100:+.1f}%")
print(f"  ML + biais    — RMSE : {rmse_corr:.4f} | MAE : {mae_corr:.4f} | vs baseline: {(1-rmse_corr/rmse_base)*100:+.1f}%")
print(f"  Baseline      — RMSE : {rmse_base:.4f} | MAE : {mae_base:.4f}")

# Par groupe
for label, steps in [("NUIT", NIGHT_STEPS), ("JOUR", DAY_STEPS)]:
    y_g = Y_test[:, steps].flatten()
    p_g = preds_test[:, steps].flatten()
    c_g = preds_corrected[:, steps].flatten()
    b_g = B_test[:, steps].flatten()

    mask_p = ~np.isnan(y_g) & ~np.isnan(p_g)
    mask_c = ~np.isnan(y_g) & ~np.isnan(c_g)
    mask_b = ~np.isnan(y_g) & ~np.isnan(b_g)

    rmse_p = float(np.sqrt(mean_squared_error(y_g[mask_p], p_g[mask_p])))
    rmse_c = float(np.sqrt(mean_squared_error(y_g[mask_c], c_g[mask_c])))
    rmse_b = float(np.sqrt(mean_squared_error(y_g[mask_b], b_g[mask_b])))

    print(f"  {label:5s} brut: {rmse_p:.4f} ({(1-rmse_p/rmse_b)*100:+.1f}%) | corrigé: {rmse_c:.4f} ({(1-rmse_c/rmse_b)*100:+.1f}%) | base: {rmse_b:.4f}")

# Par tranche horaire
print(f"\n=== RMSE par tranche horaire ===")
for h_start in range(0, 24, 3):
    t_start = h_start * 4
    t_end = min(t_start + 12, n_steps)
    steps = list(range(t_start, t_end))

    y_s = Y_test[:, steps].flatten()
    p_s = preds_test[:, steps].flatten()
    c_s = preds_corrected[:, steps].flatten()
    b_s = B_test[:, steps].flatten()

    mask_p = ~np.isnan(y_s) & ~np.isnan(p_s)
    mask_c = ~np.isnan(y_s) & ~np.isnan(c_s)
    mask_b = ~np.isnan(y_s) & ~np.isnan(b_s)

    rmse_p = float(np.sqrt(mean_squared_error(y_s[mask_p], p_s[mask_p])))
    rmse_c = float(np.sqrt(mean_squared_error(y_s[mask_c], c_s[mask_c])))
    rmse_b = float(np.sqrt(mean_squared_error(y_s[mask_b], b_s[mask_b])))

    group = "NUIT" if t_start in night_set else "JOUR"
    print(f"  {h_start:02d}h–{h_start+3:02d}h [{group}] : brut={rmse_p:.4f} ({(1-rmse_p/rmse_b)*100:+.1f}%) | corrigé={rmse_c:.4f} ({(1-rmse_c/rmse_b)*100:+.1f}%)")


# ─────────────────────────────────────────────
# 6. FEATURE IMPORTANCE
# ─────────────────────────────────────────────

for group, steps in [("NUIT", NIGHT_STEPS), ("JOUR", DAY_STEPS)]:
    print(f"\n=== Top 15 features {group} ===")
    models = [pickle.load(open(OUT / f"lgbm_t{t:03d}.pkl", "rb")) for t in steps]
    imp = np.zeros(len(feat_names))
    for m in models:
        imp += m.feature_importance()
    imp /= len(steps)
    top15 = sorted(zip(feat_names, imp), key=lambda x: -x[1])[:15]
    for name, val in top15:
        print(f"  {val:8.1f}  {name}")

# Importance groupée
print(f"\n=== Importance GROUPÉE ===")
all_models = [pickle.load(open(OUT / f"lgbm_t{t:03d}.pkl", "rb")) for t in range(n_steps)]
feat_imp = np.zeros(len(feat_names))
for m in all_models:
    feat_imp += m.feature_importance()
feat_imp /= n_steps

group_imp = {}
for name, val in zip(feat_names, feat_imp):
    if "glob_rad" in name and name.startswith("pred_") and not name.startswith("predJ_"):
        g = "PV-related J+1"
    elif "pv_MW" in name and name.startswith("pred_"):
        g = "PV-related J+1"
    elif "pv_total" in name or "pv_day" in name or "pv_capacity" in name:
        g = "PV-related J+1"
    elif "wind_dir" in name:
        g = "Wind direction"
    elif name.startswith("load_"):
        g = "Load historique"
    elif name.startswith("solar_"):
        g = "Solar prod mesurée"
    elif name.startswith("rmet_"):
        g = "Météo réelle"
    elif name.startswith("pred_") and not name.startswith("predJ_"):
        g = "Autres prévisions J+1"
    elif name.startswith("predJ_"):
        g = "Prévisions J"
    else:
        g = "Calendaire"
    group_imp[g] = group_imp.get(g, 0) + val

for g, v in sorted(group_imp.items(), key=lambda x: -x[1]):
    print(f"  {v:8.1f}  {g}")


# ─────────────────────────────────────────────
# 7. SAUVEGARDE
# ─────────────────────────────────────────────

metrics_df.write_parquet(OUT / "metrics.parquet")

# Prédictions brutes
pred_cols = {f"pred_t{t:03d}": preds_test[:, t].tolist() for t in range(n_steps)}
pred_cols["date"] = dates_test.to_list()
pl.DataFrame(pred_cols).select(
    ["date"] + [f"pred_t{t:03d}" for t in range(n_steps)]
).write_parquet(OUT / "predictions_test.parquet")

# Prédictions corrigées
corr_cols = {f"pred_t{t:03d}": preds_corrected[:, t].tolist() for t in range(n_steps)}
corr_cols["date"] = dates_test.to_list()
pl.DataFrame(corr_cols).select(
    ["date"] + [f"pred_t{t:03d}" for t in range(n_steps)]
).write_parquet(OUT / "predictions_test_corrected.parquet")

print(f"\n✓ Modèles            : {OUT}/lgbm_t000.pkl … lgbm_t095.pkl")
print(f"✓ Métriques          : {OUT}/metrics.parquet")
print(f"✓ Prédictions brutes : {OUT}/predictions_test.parquet")
print(f"✓ Prédictions corr.  : {OUT}/predictions_test_corrected.parquet")