"""
train_lgbm.py
=============
Entraînement de 96 modèles LightGBM indépendants (un par pas de 15min).
Split train/test : derniers 20% chronologiquement.
Hyperparamètres : valeurs par défaut LightGBM.

Sorties :
  DATA/models/lgbm_t{000..095}.pkl   — 96 modèles sérialisés
  DATA/models/metrics.parquet        — RMSE / MAE par pas + global vs baseline
  DATA/models/predictions_test.parquet — prédictions sur le jeu de test
"""

import polars as pl
import numpy as np
import pickle
from pathlib import Path
from datetime import date
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
BASE    = Path(r"C:\Users\gab1a\OneDrive\Documents\energyinfo2\DATA")
OUT     = BASE / "models"
OUT.mkdir(exist_ok=True)

X_PATH = BASE / "processed" / "X_features_v4.parquet"
Y_PATH = BASE / "processed" / "Y_target_v4.parquet"
B_PATH = BASE / "processed" / "B_baseline_v4.parquet"
OUT    = BASE / "models3"

TEST_RATIO = 0.20   # derniers 20% en test

LGBM_PARAMS = {
    "objective":    "regression",
    "metric":       "rmse",
    "verbosity":    -1,
    "n_jobs":       -1,
}
N_ESTIMATORS = 500
EARLY_STOPPING = 50


# ─────────────────────────────────────────────
# 1. CHARGEMENT
# ─────────────────────────────────────────────

print("=== Chargement des données ===")
X = pl.read_parquet(X_PATH)
Y = pl.read_parquet(Y_PATH)
B = pl.read_parquet(B_PATH)

dates = X["date"]
X_arr = X.drop("date").to_numpy().astype(np.float32)
Y_arr = Y.drop("date").to_numpy().astype(np.float32)
B_arr = B.drop("date").to_numpy().astype(np.float32)

n_samples  = X_arr.shape[0]
n_steps    = Y_arr.shape[1]   # 96
feat_names = [c for c in X.columns if c != "date"]

print(f"  Samples : {n_samples} jours")
print(f"  Features : {X_arr.shape[1]}")
print(f"  Pas de temps : {n_steps}")

# ─────────────────────────────────────────────
# 2. SPLIT TRAIN / TEST
# ─────────────────────────────────────────────

split_idx = int(n_samples * (1 - TEST_RATIO))
X_train, X_test = X_arr[:split_idx], X_arr[split_idx:]
Y_train, Y_test = Y_arr[:split_idx], Y_arr[split_idx:]
B_test          = B_arr[split_idx:]
dates_test      = dates[split_idx:]

print(f"\n=== Split train/test ===")
print(f"  Train : {split_idx} jours ({dates[0]} → {dates[split_idx-1]})")
print(f"  Test  : {n_samples - split_idx} jours ({dates[split_idx]} → {dates[-1]})")

# ─────────────────────────────────────────────
# 3. ENTRAÎNEMENT — 96 MODÈLES
# ─────────────────────────────────────────────

print(f"\n=== Entraînement ({n_steps} modèles) ===")

preds_test  = np.zeros_like(Y_test)
metrics     = []

for t in range(n_steps):
    y_train_t = Y_train[:, t]
    y_test_t  = Y_test[:, t]

    # Supprimer les NaN dans le train
    mask_train = ~np.isnan(y_train_t)
    mask_test  = ~np.isnan(y_test_t)

    dtrain = lgb.Dataset(
        X_train[mask_train],
        label=y_train_t[mask_train],
        feature_name=feat_names,
        free_raw_data=False,
    )
    dval = lgb.Dataset(
        X_test[mask_test],
        label=y_test_t[mask_test],
        reference=dtrain,
        free_raw_data=False,
    )

    callbacks = [
        lgb.early_stopping(EARLY_STOPPING, verbose=False),
        lgb.log_evaluation(period=-1),
    ]

    model = lgb.train(
        LGBM_PARAMS,
        dtrain,
        num_boost_round=N_ESTIMATORS,
        valid_sets=[dval],
        callbacks=callbacks,
    )

    pred_t = model.predict(X_test)
    preds_test[:, t] = pred_t

    # Métriques modèle
    rmse_model = float(np.sqrt(mean_squared_error(y_test_t[mask_test], pred_t[mask_test])))
    mae_model  = float(mean_absolute_error(y_test_t[mask_test], pred_t[mask_test]))

    # Métriques baseline Oiken
    b_t = B_test[:, t]
    mask_b = ~np.isnan(y_test_t) & ~np.isnan(b_t)
    rmse_base = float(np.sqrt(mean_squared_error(y_test_t[mask_b], b_t[mask_b]))) if mask_b.sum() > 0 else None
    mae_base  = float(mean_absolute_error(y_test_t[mask_b], b_t[mask_b]))          if mask_b.sum() > 0 else None

    metrics.append({
        "step":           t,
        "time_label":     f"{(t * 15) // 60:02d}h{(t * 15) % 60:02d}",
        "rmse_model":     rmse_model,
        "mae_model":      mae_model,
        "rmse_baseline":  rmse_base,
        "mae_baseline":   mae_base,
        "n_estimators":   model.best_iteration,
    })

    # Sauvegarde modèle
    with open(OUT / f"lgbm_t{t:03d}.pkl", "wb") as f:
        pickle.dump(model, f)

    if t % 12 == 0:
        print(f"  t={t:03d} ({metrics[-1]['time_label']}) — RMSE model={rmse_model:.4f} | baseline={rmse_base:.4f}")

# ─────────────────────────────────────────────
# 4. MÉTRIQUES GLOBALES
# ─────────────────────────────────────────────

metrics_df = pl.DataFrame(metrics)

# RMSE global (sur tous les pas, toutes les journées)
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

# ─────────────────────────────────────────────
# 5. SAUVEGARDE
# ─────────────────────────────────────────────

metrics_df.write_parquet(OUT / "metrics.parquet")

# Prédictions test
pred_cols = {f"pred_t{t:03d}": preds_test[:, t].tolist() for t in range(n_steps)}
pred_cols["date"] = dates_test.to_list()
predictions_df = pl.DataFrame(pred_cols).select(
    ["date"] + [f"pred_t{t:03d}" for t in range(n_steps)]
)
predictions_df.write_parquet(OUT / "predictions_test.parquet")

# ── Top 20 features importantes (moyenne sur tous les modèles)
print("\n=== Top 20 features (importance moyenne sur 96 modèles) ===")
all_models = [pickle.load(open(OUT / f"lgbm_t{t:03d}.pkl", "rb")) for t in range(n_steps)]
feat_imp = np.zeros(len(feat_names))
for m in all_models:
    feat_imp += m.feature_importance()
feat_imp /= n_steps
top20 = sorted(zip(feat_names, feat_imp), key=lambda x: -x[1])[:20]
for name, val in top20:
    print(f"  {val:8.1f}  {name}")

print(f"\n✓ Modèles sauvegardés : {OUT}/lgbm_t000.pkl … lgbm_t095.pkl")
print(f"✓ Métriques           : {OUT}/metrics.parquet")
print(f"✓ Prédictions test    : {OUT}/predictions_test.parquet")