"""
train_xgboost_multioutput.py
============================
Entraînement d'UN SEUL modèle XGBoost multi-output (96 sorties simultanées).
Chaque arbre produit un vecteur de 96 valeurs → les splits sont optimisés
globalement sur la journée entière, forçant une cohérence temporelle implicite
entre pas voisins (contrairement à 96 modèles LightGBM indépendants).

Features : v5 (identiques au LightGBM pour comparaison directe)
Split train/test : derniers 20% chronologiquement (identique au LightGBM)

Sorties :
  DATA/models_xgb/xgb_multi96.json       — modèle sérialisé
  DATA/models_xgb/metrics.parquet         — RMSE / MAE par pas + global vs baseline
  DATA/models_xgb/predictions_test.parquet — prédictions sur le jeu de test
"""

import polars as pl
import numpy as np
from pathlib import Path
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
BASE = Path(r"C:\Users\gab1a\OneDrive\Documents\energyinfo2\DATA")

X_PATH = BASE / "processed" / "X_features_v6.parquet"
Y_PATH = BASE / "processed" / "Y_target_v6.parquet"
B_PATH = BASE / "processed" / "B_baseline_v6.parquet"
OUT    = BASE / "models_xgb6"
OUT.mkdir(exist_ok=True)

TEST_RATIO = 0.20   # derniers 20% en test

XGB_PARAMS = {
    "n_estimators":      1000,
    "tree_method":       "hist",
    "multi_strategy":    "multi_output_tree",  # UN arbre → 96 sorties
    "early_stopping_rounds": 50,
    "eval_metric":       "rmse",
    "learning_rate":     0.1,
    "max_depth":         6,
    "verbosity":         0,
    "n_jobs":            -1,
    "random_state":      42,
}


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
n_steps   = Y_arr.shape[1]   # 96

print(f"  Samples : {n_samples} jours")
print(f"  Features : {X_arr.shape[1]}")
print(f"  Pas de temps : {n_steps}")

# ─────────────────────────────────────────────
# 2. NETTOYAGE NaN DANS Y
# ─────────────────────────────────────────────
# XGBoost multi-output ne supporte pas de NaN partiels dans la cible.
# On supprime les jours ayant au moins un NaN dans Y OU dans B.

mask_y_valid = ~np.isnan(Y_arr).any(axis=1)
mask_b_valid = ~np.isnan(B_arr).any(axis=1)
mask_valid   = mask_y_valid  # On garde tous les jours Y valides pour le train

n_dropped_y = n_samples - mask_y_valid.sum()
n_dropped_b = n_samples - (mask_y_valid & mask_b_valid).sum()
print(f"\n  Jours avec NaN dans Y : {n_dropped_y} (supprimés)")
print(f"  Jours avec NaN dans B : {n_dropped_b - n_dropped_y} (baseline indisponible)")

X_arr_clean = X_arr[mask_valid]
Y_arr_clean = Y_arr[mask_valid]
B_arr_clean = B_arr[mask_valid]  # B peut encore avoir des NaN → géré à l'évaluation
dates_clean = dates.filter(pl.Series(mask_valid))

n_clean = X_arr_clean.shape[0]
print(f"  Jours retenus : {n_clean}")

# ─────────────────────────────────────────────
# 3. SPLIT TRAIN / TEST (chronologique)
# ─────────────────────────────────────────────

split_idx = int(n_clean * (1 - TEST_RATIO))

X_train, X_test = X_arr_clean[:split_idx], X_arr_clean[split_idx:]
Y_train, Y_test = Y_arr_clean[:split_idx], Y_arr_clean[split_idx:]
B_test          = B_arr_clean[split_idx:]
dates_test      = dates_clean[split_idx:]

print(f"\n=== Split train/test ===")
print(f"  Train : {split_idx} jours ({dates_clean[0]} → {dates_clean[split_idx-1]})")
print(f"  Test  : {n_clean - split_idx} jours ({dates_clean[split_idx]} → {dates_clean[-1]})")

# ─────────────────────────────────────────────
# 4. ENTRAÎNEMENT — 1 MODÈLE MULTI-OUTPUT
# ─────────────────────────────────────────────

print(f"\n=== Entraînement XGBoost multi-output ({n_steps} sorties) ===")

model = xgb.XGBRegressor(**XGB_PARAMS)

model.fit(
    X_train, Y_train,
    eval_set=[(X_test, Y_test)],
    verbose=50,  # log toutes les 50 rounds
)

print(f"  Best iteration : {model.best_iteration}")
print(f"  Best score     : {model.best_score:.6f}")

# ─────────────────────────────────────────────
# 5. PRÉDICTIONS ET MÉTRIQUES
# ─────────────────────────────────────────────

preds_test = model.predict(X_test)
print(f"\n  Prédictions shape : {preds_test.shape}")

# ── Métriques par pas de 15min
metrics = []
for t in range(n_steps):
    y_t = Y_test[:, t]
    p_t = preds_test[:, t]
    b_t = B_test[:, t]

    rmse_model = float(np.sqrt(mean_squared_error(y_t, p_t)))
    mae_model  = float(mean_absolute_error(y_t, p_t))

    mask_b = ~np.isnan(b_t)
    rmse_base = float(np.sqrt(mean_squared_error(y_t[mask_b], b_t[mask_b]))) if mask_b.sum() > 0 else None
    mae_base  = float(mean_absolute_error(y_t[mask_b], b_t[mask_b]))          if mask_b.sum() > 0 else None

    metrics.append({
        "step":          t,
        "time_label":    f"{(t * 15) // 60:02d}h{(t * 15) % 60:02d}",
        "rmse_model":    rmse_model,
        "mae_model":     mae_model,
        "rmse_baseline": rmse_base,
        "mae_baseline":  mae_base,
    })

metrics_df = pl.DataFrame(metrics)

# ── Métriques globales
mask_all = ~np.isnan(Y_test) & ~np.isnan(preds_test)
rmse_global_model = float(np.sqrt(mean_squared_error(Y_test[mask_all], preds_test[mask_all])))
mae_global_model  = float(mean_absolute_error(Y_test[mask_all], preds_test[mask_all]))

mask_b_all = ~np.isnan(Y_test) & ~np.isnan(B_test)
rmse_global_base = float(np.sqrt(mean_squared_error(Y_test[mask_b_all], B_test[mask_b_all])))
mae_global_base  = float(mean_absolute_error(Y_test[mask_b_all], B_test[mask_b_all]))

print(f"\n=== Résultats globaux (jeu de test) ===")
print(f"  Modèle   — RMSE : {rmse_global_model:.4f} | MAE : {mae_global_model:.4f}")
print(f"  Baseline — RMSE : {rmse_global_base:.4f} | MAE : {mae_global_base:.4f}")
improvement = (1 - rmse_global_model / rmse_global_base) * 100
print(f"  Amélioration RMSE : {improvement:+.1f}%")

# ── Comparaison par tranche horaire
print(f"\n=== RMSE par tranche horaire ===")
for h_start in range(0, 24, 3):
    t_start = h_start * 4
    t_end   = min(t_start + 12, n_steps)
    steps   = list(range(t_start, t_end))

    y_slice = Y_test[:, steps].flatten()
    p_slice = preds_test[:, steps].flatten()
    b_slice = B_test[:, steps].flatten()

    mask_p = ~np.isnan(y_slice) & ~np.isnan(p_slice)
    mask_b = ~np.isnan(y_slice) & ~np.isnan(b_slice)

    rmse_m = float(np.sqrt(mean_squared_error(y_slice[mask_p], p_slice[mask_p]))) if mask_p.sum() > 0 else None
    rmse_b = float(np.sqrt(mean_squared_error(y_slice[mask_b], b_slice[mask_b]))) if mask_b.sum() > 0 else None

    label = f"{h_start:02d}h–{h_start+3:02d}h"
    if rmse_m and rmse_b:
        delta = (1 - rmse_m / rmse_b) * 100
        print(f"  {label} : modèle={rmse_m:.4f} | baseline={rmse_b:.4f} | {delta:+.1f}%")
    else:
        print(f"  {label} : données insuffisantes")

# ─────────────────────────────────────────────
# 6. FEATURE IMPORTANCE
# ─────────────────────────────────────────────

print(f"\n=== Top 20 features (importance globale) ===")
importances = model.feature_importances_
top20_idx = np.argsort(importances)[::-1][:20]
for idx in top20_idx:
    print(f"  {importances[idx]:8.1f}  {feat_names[idx]}")

# ─────────────────────────────────────────────
# 7. SAUVEGARDE
# ─────────────────────────────────────────────

# Modèle (format JSON natif XGBoost — plus portable que pickle)
model.save_model(str(OUT / "xgb_multi96.json"))

# Métriques
metrics_df.write_parquet(OUT / "metrics.parquet")

# Prédictions test (même format que LightGBM pour comparaison)
pred_cols = {f"pred_t{t:03d}": preds_test[:, t].tolist() for t in range(n_steps)}
pred_cols["date"] = dates_test.to_list()
predictions_df = pl.DataFrame(pred_cols).select(
    ["date"] + [f"pred_t{t:03d}" for t in range(n_steps)]
)
predictions_df.write_parquet(OUT / "predictions_test.parquet")

print(f"\n✓ Modèle sauvegardé     : {OUT}/xgb_multi96.json")
print(f"✓ Métriques             : {OUT}/metrics.parquet")
print(f"✓ Prédictions test      : {OUT}/predictions_test.parquet")
print(f"✓ Best iteration        : {model.best_iteration}")