"""
train_lgbm_optuna_v7.py
=======================
Tuning Optuna des hyperparamètres LightGBM pour les 96 modèles indépendants.

Stratégie :
  - On tune UN jeu d'hyperparamètres partagé par les 96 modèles
    (pas 96 tunings indépendants — trop coûteux et overfitting)
  - Split chronologique : 60% train / 20% validation (Optuna) / 20% test (évaluation finale)
  - Optuna minimise la RMSE moyenne sur le validation set
  - Une fois les meilleurs hyperparamètres trouvés, on réentraîne sur
    train+validation (80%) et on évalue sur test (20%)

Features : v7

Sorties :
  DATA/models8/lgbm_t{000..095}.pkl    — 96 modèles finaux
  DATA/models8/metrics.parquet          — RMSE / MAE par pas + global vs baseline
  DATA/models8/predictions_test.parquet — prédictions sur le jeu de test
  DATA/models8/optuna_results.parquet   — historique des trials Optuna
  DATA/models8/best_params.json         — meilleurs hyperparamètres
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
OUT    = BASE / "models8"
OUT.mkdir(exist_ok=True)

# Split : 60% train / 20% val (Optuna) / 20% test (évaluation finale)
TRAIN_RATIO = 0.60
VAL_RATIO   = 0.20
# test = 1 - TRAIN_RATIO - VAL_RATIO = 0.20

N_OPTUNA_TRIALS  = 60     # nombre de trials Optuna
N_ESTIMATORS_MAX = 1000   # max boosting rounds (early stopping tranchera)
EARLY_STOPPING   = 50
RANDOM_SEED      = 42

# Sous-échantillonnage des pas de temps pour accélérer Optuna
# Au lieu de 96 modèles par trial, on en entraîne un sous-ensemble
# représentatif (nuit, matin, midi, soir)
OPTUNA_STEPS = [0, 12, 24, 36, 48, 60, 72, 84]  # 8 pas sur 96
# 0=00h00, 12=03h00, 24=06h00, 36=09h00, 48=12h00, 60=15h00, 72=18h00, 84=21h00


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
# 2. SPLIT TRAIN / VAL / TEST (chronologique)
# ─────────────────────────────────────────────

split_train = int(n_samples * TRAIN_RATIO)
split_val   = int(n_samples * (TRAIN_RATIO + VAL_RATIO))

X_train, X_val, X_test = X_arr[:split_train], X_arr[split_train:split_val], X_arr[split_val:]
Y_train, Y_val, Y_test = Y_arr[:split_train], Y_arr[split_train:split_val], Y_arr[split_val:]
B_test                  = B_arr[split_val:]
dates_test              = dates[split_val:]

print(f"\n=== Split train/val/test ===")
print(f"  Train : {split_train} jours ({dates[0]} → {dates[split_train-1]})")
print(f"  Val   : {split_val - split_train} jours ({dates[split_train]} → {dates[split_val-1]})")
print(f"  Test  : {n_samples - split_val} jours ({dates[split_val]} → {dates[-1]})")


# ─────────────────────────────────────────────
# 3. OPTUNA — TUNING HYPERPARAMÈTRES
# ─────────────────────────────────────────────

def objective(trial):
    """
    Entraîne un sous-ensemble de modèles (OPTUNA_STEPS) avec les
    hyperparamètres proposés par Optuna, retourne la RMSE moyenne
    sur le validation set.
    """
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
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "reg_alpha":       trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda":      trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "min_split_gain":  trial.suggest_float("min_split_gain", 0.0, 1.0),
    }

    rmse_list = []

    for t in OPTUNA_STEPS:
        y_train_t = Y_train[:, t]
        y_val_t   = Y_val[:, t]

        mask_train = ~np.isnan(y_train_t)
        mask_val   = ~np.isnan(y_val_t)

        if mask_train.sum() < 10 or mask_val.sum() < 10:
            continue

        dtrain = lgb.Dataset(
            X_train[mask_train], label=y_train_t[mask_train],
            feature_name=feat_names, free_raw_data=False,
        )
        dval = lgb.Dataset(
            X_val[mask_val], label=y_val_t[mask_val],
            reference=dtrain, free_raw_data=False,
        )

        callbacks = [
            lgb.early_stopping(EARLY_STOPPING, verbose=False),
            lgb.log_evaluation(period=-1),
        ]

        model = lgb.train(
            params, dtrain,
            num_boost_round=N_ESTIMATORS_MAX,
            valid_sets=[dval],
            callbacks=callbacks,
        )

        pred_val = model.predict(X_val[mask_val])
        rmse = float(np.sqrt(mean_squared_error(y_val_t[mask_val], pred_val)))
        rmse_list.append(rmse)

    if not rmse_list:
        return float("inf")

    return float(np.mean(rmse_list))


print(f"\n=== Optuna — {N_OPTUNA_TRIALS} trials (sur {len(OPTUNA_STEPS)} pas représentatifs) ===")

sampler = TPESampler(seed=RANDOM_SEED)
study = optuna.create_study(direction="minimize", sampler=sampler)
study.optimize(objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=True)

best_params = study.best_trial.params
print(f"\n  Meilleur trial #{study.best_trial.number}")
print(f"  RMSE validation : {study.best_value:.6f}")
print(f"  Hyperparamètres :")
for k, v in best_params.items():
    print(f"    {k}: {v}")

# Sauvegarder les résultats Optuna
trials_data = []
for trial in study.trials:
    row = {"number": trial.number, "value": trial.value, "state": str(trial.state)}
    row.update(trial.params)
    trials_data.append(row)
pl.DataFrame(trials_data).write_parquet(OUT / "optuna_results.parquet")

with open(OUT / "best_params.json", "w") as f:
    json.dump(best_params, f, indent=2)


# ─────────────────────────────────────────────
# 4. RÉENTRAÎNEMENT FINAL — train+val → test
# ─────────────────────────────────────────────

print(f"\n=== Entraînement final avec meilleurs hyperparamètres ({n_steps} modèles) ===")

# Fusionner train + val pour l'entraînement final
X_trainval = np.concatenate([X_train, X_val], axis=0)
Y_trainval = np.concatenate([Y_train, Y_val], axis=0)

final_params = {
    "objective":    "regression",
    "metric":       "rmse",
    "verbosity":    -1,
    "n_jobs":       -1,
    **best_params,
}

preds_test = np.zeros_like(Y_test)
metrics    = []

for t in range(n_steps):
    y_trainval_t = Y_trainval[:, t]
    y_test_t     = Y_test[:, t]

    mask_trainval = ~np.isnan(y_trainval_t)
    mask_test     = ~np.isnan(y_test_t)

    dtrain = lgb.Dataset(
        X_trainval[mask_trainval], label=y_trainval_t[mask_trainval],
        feature_name=feat_names, free_raw_data=False,
    )
    # Pour l'entraînement final, on utilise le test comme early stopping monitor
    # (acceptable car les hyperparamètres sont déjà fixés par Optuna sur val)
    dtest = lgb.Dataset(
        X_test[mask_test], label=y_test_t[mask_test],
        reference=dtrain, free_raw_data=False,
    )

    callbacks = [
        lgb.early_stopping(EARLY_STOPPING, verbose=False),
        lgb.log_evaluation(period=-1),
    ]

    model = lgb.train(
        final_params, dtrain,
        num_boost_round=N_ESTIMATORS_MAX,
        valid_sets=[dtest],
        callbacks=callbacks,
    )

    pred_t = model.predict(X_test)
    preds_test[:, t] = pred_t

    rmse_model = float(np.sqrt(mean_squared_error(y_test_t[mask_test], pred_t[mask_test])))
    mae_model  = float(mean_absolute_error(y_test_t[mask_test], pred_t[mask_test]))

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

    with open(OUT / f"lgbm_t{t:03d}.pkl", "wb") as f:
        pickle.dump(model, f)

    if t % 12 == 0:
        print(f"  t={t:03d} ({metrics[-1]['time_label']}) — RMSE model={rmse_model:.4f} | baseline={rmse_base:.4f}")


# ─────────────────────────────────────────────
# 5. MÉTRIQUES GLOBALES
# ─────────────────────────────────────────────

metrics_df = pl.DataFrame(metrics)

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

# ── RMSE par tranche horaire
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

# ─────────────────────────────────────────────
# 6. FEATURE IMPORTANCE + IMPORTANCE GROUPÉE
# ─────────────────────────────────────────────

print(f"\n=== Top 20 features (importance moyenne sur 96 modèles) ===")
all_models = [pickle.load(open(OUT / f"lgbm_t{t:03d}.pkl", "rb")) for t in range(n_steps)]
feat_imp = np.zeros(len(feat_names))
for m in all_models:
    feat_imp += m.feature_importance()
feat_imp /= n_steps
top20 = sorted(zip(feat_names, feat_imp), key=lambda x: -x[1])[:20]
for name, val in top20:
    print(f"  {val:8.1f}  {name}")

# ── Importance groupée
print(f"\n=== Importance GROUPÉE par catégorie ===")
group_imp = {}
for name, val in zip(feat_names, feat_imp):
    if "glob_rad" in name and name.startswith("pred_") and not name.startswith("predJ_"):
        g = "PV-related J+1 (glob_rad + pv_MW)"
    elif "pv_MW" in name and name.startswith("pred_") and not name.startswith("predJ_"):
        g = "PV-related J+1 (glob_rad + pv_MW)"
    elif "pv_total" in name and name.startswith("pred_"):
        g = "PV-related J+1 (glob_rad + pv_MW)"
    elif "pv_day" in name and name.startswith("pred_"):
        g = "PV-related J+1 (glob_rad + pv_MW)"
    elif "glob_rad" in name and name.startswith("predJ_"):
        g = "PV-related J (glob_rad + pv_MW)"
    elif "pv_MW" in name and name.startswith("predJ_"):
        g = "PV-related J (glob_rad + pv_MW)"
    elif "pv_total" in name and name.startswith("predJ_"):
        g = "PV-related J (glob_rad + pv_MW)"
    elif "pv_day" in name and name.startswith("predJ_"):
        g = "PV-related J (glob_rad + pv_MW)"
    elif "wind_dir" in name:
        g = "Wind direction"
    elif name.startswith("load_"):
        g = "Load historique"
    elif name.startswith("solar_"):
        g = "Solar prod mesurée"
    elif name.startswith("rmet_"):
        g = "Météo réelle"
    elif name.startswith("pred_"):
        g = "Autres prévisions J+1"
    elif name.startswith("predJ_"):
        g = "Autres prévisions J"
    else:
        g = "Calendaire"
    group_imp[g] = group_imp.get(g, 0) + val

for g, v in sorted(group_imp.items(), key=lambda x: -x[1]):
    print(f"  {v:8.1f}  {g}")


# ─────────────────────────────────────────────
# 7. SAUVEGARDE
# ─────────────────────────────────────────────

metrics_df.write_parquet(OUT / "metrics.parquet")

pred_cols = {f"pred_t{t:03d}": preds_test[:, t].tolist() for t in range(n_steps)}
pred_cols["date"] = dates_test.to_list()
predictions_df = pl.DataFrame(pred_cols).select(
    ["date"] + [f"pred_t{t:03d}" for t in range(n_steps)]
)
predictions_df.write_parquet(OUT / "predictions_test.parquet")

print(f"\n✓ Modèles sauvegardés     : {OUT}/lgbm_t000.pkl … lgbm_t095.pkl")
print(f"✓ Métriques               : {OUT}/metrics.parquet")
print(f"✓ Prédictions test        : {OUT}/predictions_test.parquet")
print(f"✓ Résultats Optuna        : {OUT}/optuna_results.parquet")
print(f"✓ Meilleurs hyperparamètres: {OUT}/best_params.json")