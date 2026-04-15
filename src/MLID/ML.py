"""
train_lgbm_intraday_v1.py
=========================
Entraînement intraday : 8 LightGBM indépendants (un par horizon k=1..8).

Modèle k=4 (H+1h) sera utilisé comme prédiction "principale" pour
le visioneur HTML (option A : un horizon fixe).

Split chronologique 60/20/20 sur les jours uniques.

Sorties :
  DATA/models15_intraday/lgbm_h{1..8}.pkl
  DATA/models15_intraday/best_params_h{1..8}.json
  DATA/models15_intraday/metrics.parquet
  DATA/models15_intraday/predictions_test_long.parquet
  DATA/models15_intraday/predictions_test.parquet      (format viewer)
  DATA/models15_intraday/Y_target_v12.parquet          (copié pour viewer)
  DATA/models15_intraday/B_baseline_v12.parquet        (copié pour viewer)
"""

import polars as pl
import numpy as np
import pickle
import json
import shutil
from pathlib import Path
from datetime import date, timedelta
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
import optuna
from optuna.samplers import TPESampler

BASE = Path(__file__).resolve().parents[2] / "DATA"

X_PATH = BASE / "processed" / "X_intraday_v1.parquet"
Y_PATH = BASE / "processed" / "Y_intraday_v1.parquet"
B_PATH = BASE / "processed" / "B_intraday_v1.parquet"

Y_DAYAHEAD_PATH = BASE / "processed" / "Y_target_v12.parquet"
B_DAYAHEAD_PATH = BASE / "processed" / "B_baseline_v12.parquet"

OUT = BASE / "models15_intraday"
OUT.mkdir(parents=True, exist_ok=True)

TRAIN_RATIO = 0.60
VAL_RATIO   = 0.20

N_OPTUNA_TRIALS  = 30
N_ESTIMATORS_MAX = 1000
EARLY_STOPPING   = 50
RANDOM_SEED      = 42

HORIZONS = list(range(1, 9))
EXPORT_HORIZON = 4   # k=4 → 1h avant la livraison


# ─────────────────────────────────────────────
# 1. CHARGEMENT
# ─────────────────────────────────────────────

print("=== Chargement ===")
X = pl.read_parquet(X_PATH)
Y = pl.read_parquet(Y_PATH)
B = pl.read_parquet(B_PATH)

ID_COLS = ["date", "launch_hour"]
feat_names = [c for c in X.columns if c not in ID_COLS]

print(f"  Samples : {X.shape[0]}")
print(f"  Features : {len(feat_names)}")
print(f"  Horizons : {len(HORIZONS)} (H+15min à H+2h)")

# Split sur jours uniques
unique_dates = sorted(X["date"].unique().to_list())
n_days = len(unique_dates)
split_train_day = int(n_days * TRAIN_RATIO)
split_val_day   = int(n_days * (TRAIN_RATIO + VAL_RATIO))

train_dates = unique_dates[:split_train_day]
val_dates   = unique_dates[split_train_day:split_val_day]
test_dates  = unique_dates[split_val_day:]

print(f"\n=== Split chronologique (par jour) ===")
print(f"  Train : {len(train_dates)} jours ({train_dates[0]} → {train_dates[-1]})")
print(f"  Val   : {len(val_dates)} jours ({val_dates[0]} → {val_dates[-1]})")
print(f"  Test  : {len(test_dates)} jours ({test_dates[0]} → {test_dates[-1]})")

mask_train = X["date"].is_in(train_dates).to_numpy()
mask_val   = X["date"].is_in(val_dates).to_numpy()
mask_test  = X["date"].is_in(test_dates).to_numpy()

feat_arr = X.select(feat_names).to_numpy().astype(np.float32)

X_train = feat_arr[mask_train]
X_val   = feat_arr[mask_val]
X_test  = feat_arr[mask_test]

ids_test = X.filter(pl.col("date").is_in(test_dates)).select(ID_COLS)
ids_test_dates = ids_test["date"].to_list()
ids_test_hours = ids_test["launch_hour"].to_list()

Y_arr = Y.select([f"y_h{k}" for k in HORIZONS]).to_numpy().astype(np.float32)
B_arr = B.select([f"b_h{k}" for k in HORIZONS]).to_numpy().astype(np.float32)

Y_train = Y_arr[mask_train]
Y_val   = Y_arr[mask_val]
Y_test  = Y_arr[mask_test]
B_test  = B_arr[mask_test]

print(f"  Train samples : {X_train.shape[0]}")
print(f"  Val samples   : {X_val.shape[0]}")
print(f"  Test samples  : {X_test.shape[0]}")


# ─────────────────────────────────────────────
# 2. OPTUNA PAR HORIZON
# ─────────────────────────────────────────────

def run_optuna_for_horizon(k_idx: int, n_trials: int):
    k = k_idx + 1
    y_tr = Y_train[:, k_idx]
    y_va = Y_val[:, k_idx]
    mask_tr = ~np.isnan(y_tr)
    mask_va = ~np.isnan(y_va)

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
        return float(np.sqrt(mean_squared_error(y_va[mask_va], pred)))

    print(f"\n{'='*60}")
    print(f"  Optuna HORIZON h={k} (H+{k*15}min) — {n_trials} trials")
    print(f"{'='*60}")
    sampler = TPESampler(seed=RANDOM_SEED)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_trial.params
    print(f"  Best trial #{study.best_trial.number} | RMSE val = {study.best_value:.4f}")
    return best


best_params_per_h = {}
for k in HORIZONS:
    best_params_per_h[k] = run_optuna_for_horizon(k - 1, N_OPTUNA_TRIALS)
    with open(OUT / f"best_params_h{k}.json", "w") as f:
        json.dump(best_params_per_h[k], f, indent=2)


# ─────────────────────────────────────────────
# 3. ENTRAÎNEMENT FINAL PAR HORIZON
# ─────────────────────────────────────────────

print(f"\n=== Entraînement final ({len(HORIZONS)} modèles) ===")

X_trainval = np.concatenate([X_train, X_val], axis=0)
Y_trainval = np.concatenate([Y_train, Y_val], axis=0)

preds_test = np.zeros_like(Y_test)
metrics = []

for k_idx, k in enumerate(HORIZONS):
    final_params = {
        "objective": "regression",
        "metric":    "rmse",
        "verbosity": -1,
        "n_jobs":    -1,
        **best_params_per_h[k],
    }

    y_tv = Y_trainval[:, k_idx]
    y_te = Y_test[:, k_idx]
    mask_tv = ~np.isnan(y_tv)
    mask_te = ~np.isnan(y_te)

    dtrain = lgb.Dataset(X_trainval[mask_tv], label=y_tv[mask_tv],
                         feature_name=feat_names, free_raw_data=False)
    mask_val_t = ~np.isnan(Y_val[:, k_idx])
    dval_es = lgb.Dataset(X_val[mask_val_t], label=Y_val[mask_val_t, k_idx],
                          reference=dtrain, free_raw_data=False)

    model = lgb.train(
        final_params, dtrain, num_boost_round=N_ESTIMATORS_MAX,
        valid_sets=[dval_es],
        callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False),
                   lgb.log_evaluation(-1)],
    )

    pred_t = model.predict(X_test)
    preds_test[:, k_idx] = pred_t

    rmse_m = float(np.sqrt(mean_squared_error(y_te[mask_te], pred_t[mask_te])))
    mae_m  = float(mean_absolute_error(y_te[mask_te], pred_t[mask_te]))
    b_t = B_test[:, k_idx]
    mask_b = ~np.isnan(y_te) & ~np.isnan(b_t)
    rmse_b = float(np.sqrt(mean_squared_error(y_te[mask_b], b_t[mask_b]))) if mask_b.sum() > 0 else None
    mae_b  = float(mean_absolute_error(y_te[mask_b], b_t[mask_b]))         if mask_b.sum() > 0 else None

    metrics.append({
        "horizon_k":     k,
        "delay_min":     k * 15,
        "rmse_model":    rmse_m,
        "mae_model":     mae_m,
        "rmse_baseline": rmse_b,
        "mae_baseline":  mae_b,
        "n_estimators":  model.best_iteration,
    })

    with open(OUT / f"lgbm_h{k}.pkl", "wb") as f:
        pickle.dump(model, f)

    base_str = f"{rmse_b:.4f}" if rmse_b is not None else "N/A"
    print(f"  h={k} (H+{k*15:>3}min) — RMSE model={rmse_m:.4f} | baseline={base_str}")


# ─────────────────────────────────────────────
# 4. MÉTRIQUES GLOBALES
# ─────────────────────────────────────────────

metrics_df = pl.DataFrame(metrics)
metrics_df.write_parquet(OUT / "metrics.parquet")

mask_all = ~np.isnan(Y_test) & ~np.isnan(preds_test)
rmse_global = float(np.sqrt(mean_squared_error(Y_test[mask_all], preds_test[mask_all])))
mae_global  = float(mean_absolute_error(Y_test[mask_all], preds_test[mask_all]))
mask_b_all = ~np.isnan(Y_test) & ~np.isnan(B_test)
rmse_base = float(np.sqrt(mean_squared_error(Y_test[mask_b_all], B_test[mask_b_all])))
mae_base  = float(mean_absolute_error(Y_test[mask_b_all], B_test[mask_b_all]))

print(f"\n=== Résultats globaux (tous horizons) ===")
print(f"  Modèle   — RMSE : {rmse_global:.4f} | MAE : {mae_global:.4f}")
print(f"  Baseline — RMSE : {rmse_base:.4f}  | MAE : {mae_base:.4f}")
print(f"  Amélioration RMSE : {(1 - rmse_global / rmse_base) * 100:+.1f}%")
print(f"  Amélioration MAE  : {(1 - mae_global / mae_base) * 100:+.1f}%")


# ─────────────────────────────────────────────
# 5. EXPORT FORMAT VIEWER (k=EXPORT_HORIZON)
# ─────────────────────────────────────────────

print(f"\n=== Export viewer (horizon k={EXPORT_HORIZON} = H+{EXPORT_HORIZON*15}min) ===")

# Long format — toutes les prédictions par sample
preds_long = pl.DataFrame({
    "date":        ids_test_dates,
    "launch_hour": ids_test_hours,
    **{f"pred_h{k}": preds_test[:, k - 1].tolist() for k in HORIZONS},
})
preds_long.write_parquet(OUT / "predictions_test_long.parquet")
print(f"  ✓ predictions_test_long.parquet ({preds_long.shape[0]} samples)")

# Format viewer : pour chaque (date, pas t) → prédiction du lancement à H = pas_time - k*15min
EXPORT_K = EXPORT_HORIZON
pred_dict = {}
for i in range(len(ids_test_dates)):
    key = (ids_test_dates[i], ids_test_hours[i])
    pred_dict[key] = preds_test[i, EXPORT_K - 1]

test_dates_sorted = sorted(set(ids_test_dates))
viewer_rows = []
for d in test_dates_sorted:
    row = {"date": d}
    for t in range(96):
        # Pas t = (t * 15) min UTC. Lancement à minute_pas - k*15
        minute_lancement = t * 15 - EXPORT_K * 15
        if minute_lancement < 0:
            launch_date = d - timedelta(days=1)
            minute_lancement += 24 * 60
        else:
            launch_date = d
        launch_hour = minute_lancement // 60
        if minute_lancement % 60 != 0:
            row[f"pred_t{t:03d}"] = None
            continue
        key = (launch_date, int(launch_hour))
        row[f"pred_t{t:03d}"] = pred_dict.get(key, None)
    viewer_rows.append(row)

viewer_df = pl.DataFrame(viewer_rows).with_columns(pl.col("date").cast(pl.Date))
viewer_df.write_parquet(OUT / "predictions_test.parquet")

n_filled = sum(1 for row in viewer_rows for k in row if k.startswith("pred_t") and row[k] is not None)
n_total = len(viewer_rows) * 96
print(f"  ✓ predictions_test.parquet ({viewer_df.shape[0]} jours × 96 pas)")
print(f"    Pas remplis : {n_filled}/{n_total} ({100*n_filled/n_total:.1f}%)")

# Copie Y_target et B_baseline du day-ahead pour le viewer
if Y_DAYAHEAD_PATH.exists():
    shutil.copy(Y_DAYAHEAD_PATH, OUT / "Y_target_v12.parquet")
    print(f"  ✓ Y_target_v12.parquet copié")
else:
    print(f"  ⚠ {Y_DAYAHEAD_PATH} introuvable")

if B_DAYAHEAD_PATH.exists():
    shutil.copy(B_DAYAHEAD_PATH, OUT / "B_baseline_v12.parquet")
    print(f"  ✓ B_baseline_v12.parquet copié")
else:
    print(f"  ⚠ {B_DAYAHEAD_PATH} introuvable")

print(f"\n✓ Sauvegardé dans : {OUT}")
print(f"  Pour générer le CSV viewer, copie ton script export_csv.py dans {OUT}")