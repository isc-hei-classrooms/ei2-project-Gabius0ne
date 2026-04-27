"""
train_lgbm_intraday_v3.py
=========================
Entraînement intraday v3 : 96 LightGBM (un par pas de 15 min).
Architecture IDENTIQUE au day-ahead v13 mais avec features fraîches.

Chaque modèle t est entraîné sur les samples où target_step == t.
Nombre de samples par modèle = nombre de jours (~1080).

Split jour/nuit avec Optuna séparé. ES en deux étapes (comme v13).
Exclusion 13-16 sept 2025. Format de sortie identique au day-ahead.

Sorties dans DATA/models15_intraday/
"""

import polars as pl
import numpy as np
import pickle
import json
import shutil
from pathlib import Path
from datetime import date
from collections import defaultdict
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
import optuna
from optuna.samplers import TPESampler

BASE = Path(__file__).resolve().parents[2] / "DATA"

X_PATH = BASE / "processed" / "X_intraday_v3.parquet"
Y_PATH = BASE / "processed" / "Y_intraday_v3.parquet"
B_PATH = BASE / "processed" / "B_intraday_v3.parquet"

Y_DAYAHEAD_PATH = BASE / "processed" / "Y_target_v13.parquet"
B_DAYAHEAD_PATH = BASE / "processed" / "B_baseline_v13.parquet"

OUT = BASE / "models15_intraday"
OUT.mkdir(parents=True, exist_ok=True)

TRAIN_RATIO = 0.47
VAL_RATIO   = 0.20

N_OPTUNA_TRIALS  = 50
N_ESTIMATORS_MAX = 1000
EARLY_STOPPING   = 50
ITER_SCALE       = 1.10
RANDOM_SEED      = 42

EXCLUDE_DATES = {date(2025, 9, 13), date(2025, 9, 14), date(2025, 9, 15), date(2025, 9, 16)}

NIGHT_STEPS = list(range(0, 40)) + list(range(68, 96))
DAY_STEPS   = list(range(40, 68))
OPTUNA_STEPS_DAY   = [48, 52, 54, 56, 58]
OPTUNA_STEPS_NIGHT = [0, 12, 28, 72, 84, 92]

N_STEPS = 96


# ─────────────────────────────────────────────
# 1. CHARGEMENT + RESTRUCTURATION
# ─────────────────────────────────────────────

print("=== Chargement ===")
X_all = pl.read_parquet(X_PATH)
Y_all = pl.read_parquet(Y_PATH)
B_all = pl.read_parquet(B_PATH)

ID_COLS = ["date", "target_step"]
feat_names = [c for c in X_all.columns if c not in ID_COLS]
n_features = len(feat_names)

# Extraire dates et steps
dates_col = X_all["date"]
steps_col = X_all["target_step"]

# Jours uniques pour le split
unique_dates = sorted(X_all["date"].unique().to_list())
n_days = len(unique_dates)
split_train_day = int(n_days * TRAIN_RATIO)
split_val_day   = int(n_days * (TRAIN_RATIO + VAL_RATIO))

train_dates_set = set(unique_dates[:split_train_day])
val_dates_set   = set(unique_dates[split_train_day:split_val_day])
test_dates_set  = set(unique_dates[split_val_day:])

print(f"  Total samples : {X_all.shape[0]:,}")
print(f"  Features : {n_features}")
print(f"  Jours : {n_days} (train {len(train_dates_set)} | val {len(val_dates_set)} | test {len(test_dates_set)})")
print(f"  Train : {sorted(train_dates_set)[0]} → {sorted(train_dates_set)[-1]}")
print(f"  Val   : {sorted(val_dates_set)[0]} → {sorted(val_dates_set)[-1]}")
print(f"  Test  : {sorted(test_dates_set)[0]} → {sorted(test_dates_set)[-1]}")

# Restructurer en numpy : pour chaque pas t, extraire les samples correspondants
# et les splitter en train/val/test
print("\n=== Restructuration par pas de temps ===")

# Pré-extraire arrays
feat_arr = X_all.select(feat_names).to_numpy().astype(np.float32)
y_arr = Y_all["y"].to_numpy().astype(np.float32)
b_arr = B_all["b"].to_numpy().astype(np.float32)
dates_arr = X_all["date"].to_list()
steps_arr = X_all["target_step"].to_numpy()

# Pour chaque pas t, on construit des indices dans le grand array
data_by_step = {}
for t in range(N_STEPS):
    mask_step = steps_arr == t
    idx = np.where(mask_step)[0]
    step_dates = [dates_arr[i] for i in idx]

    mask_tr = np.array([d in train_dates_set for d in step_dates])
    mask_va = np.array([d in val_dates_set for d in step_dates])
    mask_te = np.array([d in test_dates_set for d in step_dates])

    data_by_step[t] = {
        "X_train": feat_arr[idx[mask_tr]],
        "X_val":   feat_arr[idx[mask_va]],
        "X_test":  feat_arr[idx[mask_te]],
        "Y_train": y_arr[idx[mask_tr]],
        "Y_val":   y_arr[idx[mask_va]],
        "Y_test":  y_arr[idx[mask_te]],
        "B_test":  b_arr[idx[mask_te]],
        "dates_test": [step_dates[i] for i in np.where(mask_te)[0]],
    }

# Vérification
t0 = data_by_step[0]
print(f"  Step 0 : train={t0['X_train'].shape[0]} | val={t0['X_val'].shape[0]} | test={t0['X_test'].shape[0]}")

# Masque d'exclusion global sur les dates test (pour les métriques agrégées)
test_dates_list = data_by_step[0]["dates_test"]
exclude_mask = np.array([d not in EXCLUDE_DATES for d in test_dates_list], dtype=bool)
n_excluded = (~exclude_mask).sum()
print(f"  Dates exclues : {n_excluded}")


# ─────────────────────────────────────────────
# 2. OPTUNA — SÉPARÉ JOUR/NUIT
# ─────────────────────────────────────────────

def run_optuna(group_name, optuna_steps, n_trials):
    def objective(trial):
        params = {
            "objective":"regression", "metric":"rmse", "verbosity":-1, "n_jobs":-1,
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
            d = data_by_step[t]
            y_tr = d["Y_train"]; y_va = d["Y_val"]
            m_tr = ~np.isnan(y_tr); m_va = ~np.isnan(y_va)
            if m_tr.sum() < 10 or m_va.sum() < 10: continue
            dtrain = lgb.Dataset(d["X_train"][m_tr], label=y_tr[m_tr],
                                 feature_name=feat_names, free_raw_data=False)
            dval = lgb.Dataset(d["X_val"][m_va], label=y_va[m_va],
                               reference=dtrain, free_raw_data=False)
            model = lgb.train(params, dtrain, num_boost_round=N_ESTIMATORS_MAX,
                valid_sets=[dval],
                callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False),
                           lgb.log_evaluation(-1)])
            pred = model.predict(d["X_val"][m_va])
            rmse_list.append(float(np.sqrt(mean_squared_error(y_va[m_va], pred))))
        return float(np.mean(rmse_list)) if rmse_list else float("inf")

    print(f"\n{'='*60}")
    print(f"  Optuna {group_name} — {n_trials} trials (pas: {optuna_steps})")
    print(f"{'='*60}")
    sampler = TPESampler(seed=RANDOM_SEED)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    best = study.best_trial.params
    print(f"  Best #{study.best_trial.number} | RMSE val = {study.best_value:.6f}")
    return best

best_night = run_optuna("NUIT", OPTUNA_STEPS_NIGHT, N_OPTUNA_TRIALS)
best_day   = run_optuna("JOUR", OPTUNA_STEPS_DAY, N_OPTUNA_TRIALS)

with open(OUT / "best_params_night.json", "w") as f: json.dump(best_night, f, indent=2)
with open(OUT / "best_params_day.json", "w") as f: json.dump(best_day, f, indent=2)


# ─────────────────────────────────────────────
# 3. ENTRAÎNEMENT FINAL — 96 MODÈLES, 2 ÉTAPES
# ─────────────────────────────────────────────

print(f"\n=== Entraînement final (96 modèles, ES 2 étapes) ===")

night_set = set(NIGHT_STEPS)
preds_test = np.zeros((len(test_dates_list), N_STEPS), dtype=np.float32)
Y_test_all = np.zeros_like(preds_test)
B_test_all = np.zeros_like(preds_test)
metrics = []

for t in range(N_STEPS):
    d = data_by_step[t]
    is_night = t in night_set
    best_params = best_night if is_night else best_day
    group_label = "NUIT" if is_night else "JOUR"

    final_params = {"objective":"regression","metric":"rmse","verbosity":-1,"n_jobs":-1,
                    **best_params}

    y_tr = d["Y_train"]; y_va = d["Y_val"]; y_te = d["Y_test"]
    m_tr = ~np.isnan(y_tr); m_va = ~np.isnan(y_va)
    m_te = ~np.isnan(y_te) & exclude_mask

    if m_tr.sum() < 10 or m_va.sum() < 10:
        preds_test[:, t] = np.nan
        Y_test_all[:, t] = y_te
        B_test_all[:, t] = d["B_test"]
        metrics.append({"step":t, "time_label":f"{(t*15)//60:02d}h{(t*15)%60:02d}",
            "group":group_label, "rmse_model":None, "mae_model":None,
            "rmse_baseline":None, "mae_baseline":None})
        continue

    # Étape 1 : ES propre train → val
    dtrain_p1 = lgb.Dataset(d["X_train"][m_tr], label=y_tr[m_tr],
                            feature_name=feat_names, free_raw_data=False)
    dval_p1 = lgb.Dataset(d["X_val"][m_va], label=y_va[m_va],
                          reference=dtrain_p1, free_raw_data=False)
    model_p1 = lgb.train(final_params, dtrain_p1, num_boost_round=N_ESTIMATORS_MAX,
        valid_sets=[dval_p1],
        callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False), lgb.log_evaluation(-1)])
    best_iter = model_p1.best_iteration
    if best_iter is None or best_iter <= 0: best_iter = N_ESTIMATORS_MAX // 4
    best_iter_final = max(1, int(round(best_iter * ITER_SCALE)))

    # Étape 2 : réentraînement train+val
    X_tv = np.concatenate([d["X_train"], d["X_val"]], axis=0)
    y_tv = np.concatenate([y_tr, y_va], axis=0)
    m_tv = ~np.isnan(y_tv)
    dtrain_p2 = lgb.Dataset(X_tv[m_tv], label=y_tv[m_tv],
                            feature_name=feat_names, free_raw_data=False)
    model = lgb.train(final_params, dtrain_p2, num_boost_round=best_iter_final,
                      callbacks=[lgb.log_evaluation(-1)])

    pred_t = model.predict(d["X_test"])
    preds_test[:, t] = pred_t
    Y_test_all[:, t] = y_te
    B_test_all[:, t] = d["B_test"]

    rmse_m = float(np.sqrt(mean_squared_error(y_te[m_te], pred_t[m_te]))) if m_te.sum() > 0 else None
    mae_m = float(mean_absolute_error(y_te[m_te], pred_t[m_te])) if m_te.sum() > 0 else None
    b_t = d["B_test"]
    m_b = ~np.isnan(y_te) & ~np.isnan(b_t) & exclude_mask
    rmse_b = float(np.sqrt(mean_squared_error(y_te[m_b], b_t[m_b]))) if m_b.sum() > 0 else None
    mae_b = float(mean_absolute_error(y_te[m_b], b_t[m_b])) if m_b.sum() > 0 else None

    metrics.append({"step":t, "time_label":f"{(t*15)//60:02d}h{(t*15)%60:02d}",
        "group":group_label, "rmse_model":rmse_m, "mae_model":mae_m,
        "rmse_baseline":rmse_b, "mae_baseline":mae_b,
        "n_estimators_es":best_iter, "n_estimators_final":best_iter_final})

    with open(OUT / f"lgbm_t{t:03d}.pkl", "wb") as f:
        pickle.dump(model, f)

    if t % 12 == 0:
        bs = f"{rmse_b:.4f}" if rmse_b else "N/A"
        ms = f"{rmse_m:.4f}" if rmse_m else "N/A"
        print(f"  t={t:03d} ({(t*15)//60:02d}h{(t*15)%60:02d}) [{group_label}] — RMSE={ms} | base={bs} | iter={best_iter}→{best_iter_final}")


# ─────────────────────────────────────────────
# 4. MÉTRIQUES
# ─────────────────────────────────────────────

metrics_df = pl.DataFrame(metrics)
metrics_df.write_parquet(OUT / "metrics.parquet")

mask_all = ~np.isnan(Y_test_all) & ~np.isnan(preds_test) & exclude_mask[:, None]
rmse_g = float(np.sqrt(mean_squared_error(Y_test_all[mask_all], preds_test[mask_all])))
mae_g = float(mean_absolute_error(Y_test_all[mask_all], preds_test[mask_all]))
mask_ba = ~np.isnan(Y_test_all) & ~np.isnan(B_test_all) & exclude_mask[:, None]
rmse_bg = float(np.sqrt(mean_squared_error(Y_test_all[mask_ba], B_test_all[mask_ba])))
mae_bg = float(mean_absolute_error(Y_test_all[mask_ba], B_test_all[mask_ba]))

print(f"\n=== Résultats globaux (excl. 13-16 sept) ===")
print(f"  Test set : {exclude_mask.sum()} jours (exclu {n_excluded})")
print(f"  Modèle   — RMSE : {rmse_g:.4f} | MAE : {mae_g:.4f}")
print(f"  Baseline — RMSE : {rmse_bg:.4f} | MAE : {mae_bg:.4f}")
print(f"  Amélioration RMSE : {(1-rmse_g/rmse_bg)*100:+.1f}%")
print(f"  Amélioration MAE  : {(1-mae_g/mae_bg)*100:+.1f}%")

for group, steps in [("NUIT", NIGHT_STEPS), ("JOUR", DAY_STEPS)]:
    y_g = Y_test_all[:, steps]; p_g = preds_test[:, steps]; b_g = B_test_all[:, steps]
    mm = ~np.isnan(y_g) & ~np.isnan(p_g) & exclude_mask[:, None]
    mb = ~np.isnan(y_g) & ~np.isnan(b_g) & exclude_mask[:, None]
    rm = float(np.sqrt(mean_squared_error(y_g[mm], p_g[mm])))
    rb = float(np.sqrt(mean_squared_error(y_g[mb], b_g[mb])))
    imp = (1-rm/rb)*100
    print(f"  {group:5s} — RMSE modèle={rm:.4f} | baseline={rb:.4f} | {imp:+.1f}%")

print(f"\n=== RMSE par tranche horaire ===")
for h_start in range(0, 24, 3):
    t_start = h_start * 4; t_end = min(t_start + 12, N_STEPS)
    steps = list(range(t_start, t_end))
    y_s = Y_test_all[:, steps]; p_s = preds_test[:, steps]; b_s = B_test_all[:, steps]
    mm = ~np.isnan(y_s) & ~np.isnan(p_s) & exclude_mask[:, None]
    mb = ~np.isnan(y_s) & ~np.isnan(b_s) & exclude_mask[:, None]
    rm = float(np.sqrt(mean_squared_error(y_s[mm], p_s[mm])))
    rb = float(np.sqrt(mean_squared_error(y_s[mb], b_s[mb])))
    d = (1-rm/rb)*100
    g = "NUIT" if t_start in night_set else "JOUR"
    print(f"  {h_start:02d}h–{h_start+3:02d}h [{g}] : modèle={rm:.4f} | baseline={rb:.4f} | {d:+.1f}%")


# ─────────────────────────────────────────────
# 5. DIAGNOSTIC BIAIS DIURNE
# ─────────────────────────────────────────────

print(f"\n=== Diagnostic biais diurne (10h-17h) ===")
day_diag = list(range(40, 68))
monthly_bias = defaultdict(list)
for i, d in enumerate(test_dates_list):
    if not exclude_mask[i]: continue
    y_day = Y_test_all[i, day_diag]; p_day = preds_test[i, day_diag]
    mask = ~np.isnan(y_day) & ~np.isnan(p_day)
    if mask.sum() == 0: continue
    monthly_bias[f"{d.year}-{d.month:02d}"].append(float(np.mean(y_day[mask] - p_day[mask])))

print(f"  {'Mois':10s} | {'Nb jours':>8s} | {'Biais moyen':>11s} | Interprétation")
print(f"  {'-'*10}-+-{'-'*8}-+-{'-'*11}-+-{'-'*30}")
for mk in sorted(monthly_bias):
    vals = monthly_bias[mk]; mb = np.mean(vals)
    interp = "→ surestime PV" if mb > 0.05 else ("→ sous-estime PV" if mb < -0.05 else "→ ~neutre")
    print(f"  {mk:10s} | {len(vals):8d} | {mb:+11.4f} | {interp}")


# ─────────────────────────────────────────────
# 6. EXPORT FORMAT VIEWER (identique day-ahead)
# ─────────────────────────────────────────────

print(f"\n=== Export viewer ===")

pred_cols = {f"pred_t{t:03d}": preds_test[:, t].tolist() for t in range(N_STEPS)}
pred_cols["date"] = test_dates_list
viewer_df = pl.DataFrame(pred_cols).select(
    ["date"] + [f"pred_t{t:03d}" for t in range(N_STEPS)]
)
viewer_df.write_parquet(OUT / "predictions_test.parquet")
print(f"  ✓ predictions_test.parquet ({viewer_df.shape[0]} jours × {viewer_df.shape[1]} cols)")

for src, dst in [(Y_DAYAHEAD_PATH, "Y_target_v13.parquet"),
                 (B_DAYAHEAD_PATH, "B_baseline_v13.parquet")]:
    if src.exists():
        shutil.copy(src, OUT / dst)
        print(f"  ✓ {dst} copié")

print(f"\n✓ Sauvegardé dans : {OUT}")