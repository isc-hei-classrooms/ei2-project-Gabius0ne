"""
train_lgbm_v9_split_day_night.py
================================
Entraînement LightGBM avec hyperparamètres séparés jour/nuit.

Constat v8 : le modèle sous-estime systématiquement les creux PV en journée.
Les hyperparamètres optimisés globalement par Optuna sont un compromis
qui favorise la nuit (60% des pas, signal simple) au détriment du jour
(40% des pas, signal PV complexe).

Solution v9 : deux tunings Optuna indépendants.
  - Groupe NUIT (21h00–05h45 UTC) : 36 modèles, pas 0–23 + 84–95
  - Groupe JOUR (06h00–20h45 UTC) : 60 modèles, pas 24–83
  
Chaque groupe développe ses propres hyperparamètres optimaux.
Les prédictions sont fusionnées pour l'évaluation globale.

Features : v8 (identiques, pas de changement)
Split : 60% train / 20% val (Optuna) / 20% test

Sorties :
  DATA/models9/lgbm_t{000..095}.pkl
  DATA/models9/metrics.parquet
  DATA/models9/predictions_test.parquet
  DATA/models9/best_params_night.json
  DATA/models9/best_params_day.json
"""

import polars as pl
import numpy as np
import pickle
import json
from pathlib import Path
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
import optuna
from optuna.samplers import TPESampler

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
BASE = Path(r"C:\Users\gab1a\OneDrive\Documents\energyinfo2\DATA")

X_PATH = BASE / "processed" / "X_features_v8.parquet"
Y_PATH = BASE / "processed" / "Y_target_v8.parquet"
B_PATH = BASE / "processed" / "B_baseline_v8.parquet"
OUT    = BASE / "models9"
OUT.mkdir(exist_ok=True)

TRAIN_RATIO = 0.60
VAL_RATIO   = 0.20

N_OPTUNA_TRIALS  = 50
N_ESTIMATORS_MAX = 1000
EARLY_STOPPING   = 50
RANDOM_SEED      = 42

# Groupes de pas de temps (indices 0–95 = 00h00–23h45 UTC)
# NUIT : 00h00–05h45 (pas 0–23) + 21h00–23h45 (pas 84–95) = 36 pas
# JOUR : 06h00–20h45 (pas 24–83) = 60 pas
NIGHT_STEPS = list(range(0, 24)) + list(range(84, 96))   # 36 pas
DAY_STEPS   = list(range(24, 84))                         # 60 pas

# Pas représentatifs pour Optuna (accélération)
# Nuit : minuit, 3h, 22h, 23h
OPTUNA_STEPS_NIGHT = [0, 12, 84, 92]
# Jour : 7h, 10h, 12h, 15h, 18h (couvre rampe matin, pic PV, après-midi, soirée)
OPTUNA_STEPS_DAY   = [28, 40, 48, 60, 72]


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
print(f"  Groupe NUIT : {len(NIGHT_STEPS)} pas (00h–05h45 + 21h–23h45)")
print(f"  Groupe JOUR : {len(DAY_STEPS)} pas (06h–20h45)")

# ─────────────────────────────────────────────
# 2. SPLIT TRAIN / VAL / TEST
# ─────────────────────────────────────────────

split_train = int(n_samples * TRAIN_RATIO)
split_val   = int(n_samples * (TRAIN_RATIO + VAL_RATIO))

X_train, X_val, X_test = X_arr[:split_train], X_arr[split_train:split_val], X_arr[split_val:]
Y_train, Y_val, Y_test = Y_arr[:split_train], Y_arr[split_train:split_val], Y_arr[split_val:]
B_test                  = B_arr[split_val:]
dates_test              = dates[split_val:]

print(f"\n=== Split ===")
print(f"  Train : {split_train} jours ({dates[0]} → {dates[split_train-1]})")
print(f"  Val   : {split_val - split_train} jours ({dates[split_train]} → {dates[split_val-1]})")
print(f"  Test  : {n_samples - split_val} jours ({dates[split_val]} → {dates[-1]})")


# ─────────────────────────────────────────────
# 3. OPTUNA — TUNING SÉPARÉ JOUR/NUIT
# ─────────────────────────────────────────────

def run_optuna(group_name, optuna_steps, n_trials):
    """Optimise les hyperparamètres pour un groupe de pas."""

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
            y_tr = Y_train[:, t]
            y_va = Y_val[:, t]
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
    print(f"  Optuna {group_name} — {n_trials} trials (pas représentatifs: {optuna_steps})")
    print(f"{'='*60}")

    sampler = TPESampler(seed=RANDOM_SEED)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_trial.params
    print(f"\n  Meilleur trial #{study.best_trial.number}")
    print(f"  RMSE validation : {study.best_value:.6f}")
    print(f"  Hyperparamètres :")
    for k, v in best.items():
        print(f"    {k}: {v}")

    return best


# Tuning NUIT
best_night = run_optuna("NUIT", OPTUNA_STEPS_NIGHT, N_OPTUNA_TRIALS)

# Tuning JOUR
best_day = run_optuna("JOUR", OPTUNA_STEPS_DAY, N_OPTUNA_TRIALS)

# Sauvegarder
with open(OUT / "best_params_night.json", "w") as f:
    json.dump(best_night, f, indent=2)
with open(OUT / "best_params_day.json", "w") as f:
    json.dump(best_day, f, indent=2)


# ─────────────────────────────────────────────
# 4. ENTRAÎNEMENT FINAL — train+val → test
# ─────────────────────────────────────────────

print(f"\n=== Entraînement final (96 modèles, hyperparamètres séparés) ===")

X_trainval = np.concatenate([X_train, X_val], axis=0)
Y_trainval = np.concatenate([Y_train, Y_val], axis=0)

night_set = set(NIGHT_STEPS)

preds_test = np.zeros_like(Y_test)
metrics    = []

for t in range(n_steps):
    # Choisir les hyperparamètres selon le groupe
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

    y_tv = Y_trainval[:, t]
    y_te = Y_test[:, t]
    mask_tv = ~np.isnan(y_tv)
    mask_te = ~np.isnan(y_te)

    dtrain = lgb.Dataset(X_trainval[mask_tv], label=y_tv[mask_tv],
                         feature_name=feat_names, free_raw_data=False)
    dtest = lgb.Dataset(X_test[mask_te], label=y_te[mask_te],
                        reference=dtrain, free_raw_data=False)

    model = lgb.train(
        final_params, dtrain, num_boost_round=N_ESTIMATORS_MAX,
        valid_sets=[dtest],
        callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False),
                   lgb.log_evaluation(-1)],
    )

    pred_t = model.predict(X_test)
    preds_test[:, t] = pred_t

    rmse_m = float(np.sqrt(mean_squared_error(y_te[mask_te], pred_t[mask_te])))
    mae_m  = float(mean_absolute_error(y_te[mask_te], pred_t[mask_te]))

    b_t = B_test[:, t]
    mask_b = ~np.isnan(y_te) & ~np.isnan(b_t)
    rmse_b = float(np.sqrt(mean_squared_error(y_te[mask_b], b_t[mask_b]))) if mask_b.sum() > 0 else None
    mae_b  = float(mean_absolute_error(y_te[mask_b], b_t[mask_b]))          if mask_b.sum() > 0 else None

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

    with open(OUT / f"lgbm_t{t:03d}.pkl", "wb") as f:
        pickle.dump(model, f)

    if t % 12 == 0:
        print(f"  t={t:03d} ({metrics[-1]['time_label']}) [{group_label}] — RMSE model={rmse_m:.4f} | baseline={rmse_b:.4f if rmse_b else 'N/A'}")


# ─────────────────────────────────────────────
# 5. MÉTRIQUES
# ─────────────────────────────────────────────

metrics_df = pl.DataFrame(metrics)

# Global
mask_all = ~np.isnan(Y_test) & ~np.isnan(preds_test)
rmse_global = float(np.sqrt(mean_squared_error(Y_test[mask_all], preds_test[mask_all])))
mae_global  = float(mean_absolute_error(Y_test[mask_all], preds_test[mask_all]))

mask_b_all = ~np.isnan(Y_test) & ~np.isnan(B_test)
rmse_base = float(np.sqrt(mean_squared_error(Y_test[mask_b_all], B_test[mask_b_all])))
mae_base  = float(mean_absolute_error(Y_test[mask_b_all], B_test[mask_b_all]))

print(f"\n=== Résultats globaux ===")
print(f"  Modèle   — RMSE : {rmse_global:.4f} | MAE : {mae_global:.4f}")
print(f"  Baseline — RMSE : {rmse_base:.4f} | MAE : {mae_base:.4f}")
print(f"  Amélioration RMSE : {(1 - rmse_global / rmse_base) * 100:+.1f}%")

# Par groupe
for group, steps in [("NUIT", NIGHT_STEPS), ("JOUR", DAY_STEPS)]:
    y_g = Y_test[:, steps].flatten()
    p_g = preds_test[:, steps].flatten()
    b_g = B_test[:, steps].flatten()
    mask_m = ~np.isnan(y_g) & ~np.isnan(p_g)
    mask_b = ~np.isnan(y_g) & ~np.isnan(b_g)
    rmse_m = float(np.sqrt(mean_squared_error(y_g[mask_m], p_g[mask_m])))
    rmse_b = float(np.sqrt(mean_squared_error(y_g[mask_b], b_g[mask_b])))
    imp = (1 - rmse_m / rmse_b) * 100
    print(f"  {group:5s} — RMSE modèle={rmse_m:.4f} | baseline={rmse_b:.4f} | {imp:+.1f}%")

# Par tranche horaire
print(f"\n=== RMSE par tranche horaire ===")
for h_start in range(0, 24, 3):
    t_start = h_start * 4
    t_end   = min(t_start + 12, n_steps)
    steps   = list(range(t_start, t_end))
    y_s = Y_test[:, steps].flatten()
    p_s = preds_test[:, steps].flatten()
    b_s = B_test[:, steps].flatten()
    mask_m = ~np.isnan(y_s) & ~np.isnan(p_s)
    mask_b = ~np.isnan(y_s) & ~np.isnan(b_s)
    rmse_m = float(np.sqrt(mean_squared_error(y_s[mask_m], p_s[mask_m])))
    rmse_b = float(np.sqrt(mean_squared_error(y_s[mask_b], b_s[mask_b])))
    delta = (1 - rmse_m / rmse_b) * 100
    group = "NUIT" if t_start in night_set else "JOUR"
    print(f"  {h_start:02d}h–{h_start+3:02d}h [{group}] : modèle={rmse_m:.4f} | baseline={rmse_b:.4f} | {delta:+.1f}%")


# ─────────────────────────────────────────────
# 6. FEATURE IMPORTANCE PAR GROUPE
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


# ─────────────────────────────────────────────
# 7. SAUVEGARDE
# ─────────────────────────────────────────────

metrics_df.write_parquet(OUT / "metrics.parquet")

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