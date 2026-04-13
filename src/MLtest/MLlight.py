"""
train_lgbm_v12_nowind.py
========================
Entraînement LightGBM v12 SANS features de vent et SANS variances d'irradiance.

Hypothèse : le modèle utilise le vent (35% du poids JOUR) comme proxy
indirect de la météo au lieu de l'irradiance directe. Les variances
d'irradiance (stde, q10, q90, std_stations, spread) permettent au modèle
de "savoir" que la prévision est incertaine et de l'ignorer.

En supprimant ces 1321 features :
  - Le modèle passe de ~3584 à ~2263 features
  - Le ratio features/samples passe de ~5.5 à ~3.5
  - L'irradiance brute a plus de chances d'être sélectionnée par arbre
  - Le modèle ne peut plus esquiver l'irradiance via le vent ou la variance

Features supprimées :
  - pred_wind_* et predJ_wind_* (dir sin/cos + speed) : 864
  - pred_glob_rad_stde_*, _q10_*, _q90_* : 432
  - pred_glob_rad_std_stations_* : 24
  - pred_glob_rad_spread_day : 1
  Total : ~1321

Sorties :
  DATA/models12_nowind/
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
OUT    = BASE / "models12_nowind"
OUT.mkdir(parents=True, exist_ok=True)

TRAIN_RATIO = 0.60
VAL_RATIO   = 0.20

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
# FILTRE FEATURES
# ─────────────────────────────────────────────

def should_keep(col: str) -> bool:
    """Retourne False pour les features à supprimer."""
    # Vent (direction + vitesse), J+1 et J
    if "wind" in col:
        return False
    # Variances d'irradiance
    if "glob_rad_stde" in col:
        return False
    if "glob_rad_q10" in col:
        return False
    if "glob_rad_q90" in col:
        return False
    if "glob_rad_std_stations" in col:
        return False
    if col == "pred_glob_rad_spread_day":
        return False
    return True


# ─────────────────────────────────────────────
# FEATURE IMPORTANCE GROUPÉE
# ─────────────────────────────────────────────

def classify_feature(name: str) -> str:
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
    if name.startswith("pred_precip") or name.startswith("pred_sunshine"):
        return "Météo prévue J+1 (pluie/soleil)"
    if name.startswith("predJ_precip") or name.startswith("predJ_sunshine"):
        return "Météo prévue J (pluie/soleil)"
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
# 1. CHARGEMENT + FILTRE
# ─────────────────────────────────────────────

print("=== Chargement des données ===")
X = pl.read_parquet(X_PATH)
Y = pl.read_parquet(Y_PATH)
B = pl.read_parquet(B_PATH)

# Appliquer le filtre
all_feat_cols = [c for c in X.columns if c != "date"]
keep_cols = [c for c in all_feat_cols if should_keep(c)]
drop_cols = [c for c in all_feat_cols if not should_keep(c)]

print(f"  Features avant filtre : {len(all_feat_cols)}")
print(f"  Features supprimées : {len(drop_cols)}")
print(f"  Features gardées : {len(keep_cols)}")

# Détail des suppressions
wind_dropped = [c for c in drop_cols if "wind" in c]
irr_var_dropped = [c for c in drop_cols if "wind" not in c]
print(f"    - Vent : {len(wind_dropped)}")
print(f"    - Variance irradiance : {len(irr_var_dropped)}")

dates = X["date"]
feat_names = keep_cols
X_arr = X.select(keep_cols).to_numpy().astype(np.float32)
Y_arr = Y.drop("date").to_numpy().astype(np.float32)
B_arr = B.drop("date").to_numpy().astype(np.float32)

n_samples = X_arr.shape[0]
n_steps   = Y_arr.shape[1]

print(f"\n  Samples : {n_samples} jours")
print(f"  Features : {X_arr.shape[1]}")
print(f"  Ratio features/samples : {X_arr.shape[1] / n_samples:.1f}")

# ─────────────────────────────────────────────
# 2. SPLIT
# ─────────────────────────────────────────────

split_train = int(n_samples * TRAIN_RATIO)
split_val   = int(n_samples * (TRAIN_RATIO + VAL_RATIO))

X_train, X_val, X_test = X_arr[:split_train], X_arr[split_train:split_val], X_arr[split_val:]
Y_train, Y_val, Y_test = Y_arr[:split_train], Y_arr[split_train:split_val], Y_arr[split_val:]
B_test                  = B_arr[split_val:]
dates_test              = dates[split_val:]

print(f"\n=== Split chronologique ===")
print(f"  Train : {split_train} jours ({dates[0]} → {dates[split_train-1]})")
print(f"  Val   : {split_val - split_train} jours ({dates[split_train]} → {dates[split_val-1]})")
print(f"  Test  : {n_samples - split_val} jours ({dates[split_val]} → {dates[-1]})")

dates_test_list = dates_test.to_list()
exclude_mask = np.array([d not in EXCLUDE_DATES for d in dates_test_list], dtype=bool)
n_excluded = (~exclude_mask).sum()
print(f"  Dates exclues : {n_excluded}")


# ─────────────────────────────────────────────
# 3. OPTUNA
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
    print(f"  Optuna {group_name} — {n_trials} trials")
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


# ─────────────────────────────────────────────
# 4. ENTRAÎNEMENT FINAL
# ─────────────────────────────────────────────

print(f"\n=== Entraînement final (96 modèles, sans vent ni variance irradiance) ===")

X_trainval = np.concatenate([X_train, X_val], axis=0)
Y_trainval = np.concatenate([Y_train, Y_val], axis=0)

night_set = set(NIGHT_STEPS)

preds_test = np.zeros_like(Y_test)
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

    y_tv = Y_trainval[:, t]
    y_te = Y_test[:, t]
    mask_tv = ~np.isnan(y_tv)
    mask_te = ~np.isnan(y_te) & exclude_mask

    dtrain = lgb.Dataset(X_trainval[mask_tv], label=y_tv[mask_tv],
                         feature_name=feat_names, free_raw_data=False)

    mask_val_t = ~np.isnan(Y_val[:, t])
    dval_es = lgb.Dataset(X_val[mask_val_t], label=Y_val[:, t][mask_val_t],
                          reference=dtrain, free_raw_data=False)

    model = lgb.train(
        final_params, dtrain, num_boost_round=N_ESTIMATORS_MAX,
        valid_sets=[dval_es],
        callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False),
                   lgb.log_evaluation(-1)],
    )

    pred_t = model.predict(X_test)
    preds_test[:, t] = pred_t

    if mask_te.sum() > 0:
        rmse_m = float(np.sqrt(mean_squared_error(y_te[mask_te], pred_t[mask_te])))
        mae_m  = float(mean_absolute_error(y_te[mask_te], pred_t[mask_te]))
    else:
        rmse_m, mae_m = None, None

    b_t = B_test[:, t]
    mask_b = ~np.isnan(y_te) & ~np.isnan(b_t) & exclude_mask
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
# 5. MÉTRIQUES
# ─────────────────────────────────────────────

metrics_df = pl.DataFrame(metrics)

mask_all = ~np.isnan(Y_test) & ~np.isnan(preds_test) & exclude_mask[:, None]
rmse_global = float(np.sqrt(mean_squared_error(Y_test[mask_all], preds_test[mask_all])))
mae_global  = float(mean_absolute_error(Y_test[mask_all], preds_test[mask_all]))

mask_b_all = ~np.isnan(Y_test) & ~np.isnan(B_test) & exclude_mask[:, None]
rmse_base = float(np.sqrt(mean_squared_error(Y_test[mask_b_all], B_test[mask_b_all])))
mae_base  = float(mean_absolute_error(Y_test[mask_b_all], B_test[mask_b_all]))

print(f"\n=== Résultats globaux (sans vent ni var. irradiance, excl. 13-15 sept) ===")
print(f"  Test set : {exclude_mask.sum()} jours (exclu {n_excluded})")
print(f"  Features : {len(feat_names)} (vs 3584 complet)")
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
    print(f"  {group:5s} — RMSE modèle={rmse_m:.4f} | baseline={rmse_b:.4f} | {imp:+.1f}% | MAE modèle={mae_m:.4f} | baseline={mae_b:.4f}")

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


# ─────────────────────────────────────────────
# 6. FEATURE IMPORTANCE
# ─────────────────────────────────────────────

for group, steps in [("NUIT", NIGHT_STEPS), ("JOUR", DAY_STEPS)]:
    models = [pickle.load(open(OUT / f"lgbm_t{t:03d}.pkl", "rb")) for t in steps]
    print_grouped_importance(feat_names, models, group)


# ─────────────────────────────────────────────
# 7. DIAGNOSTIC BIAIS
# ─────────────────────────────────────────────

print(f"\n=== Diagnostic biais diurne (09h-15h) ===")
day_steps_diag = list(range(36, 60))

monthly_bias = defaultdict(list)
for i, d in enumerate(dates_test_list):
    if not exclude_mask[i]:
        continue
    y_day = Y_test[i, day_steps_diag]
    p_day = preds_test[i, day_steps_diag]
    mask = ~np.isnan(y_day)
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
# 8. SAUVEGARDE
# ─────────────────────────────────────────────

metrics_df.write_parquet(OUT / "metrics.parquet")

pred_cols = {f"pred_t{t:03d}": preds_test[:, t].tolist() for t in range(n_steps)}
pred_cols["date"] = dates_test.to_list()
pl.DataFrame(pred_cols).select(
    ["date"] + [f"pred_t{t:03d}" for t in range(n_steps)]
).write_parquet(OUT / "predictions_test.parquet")

print(f"\n✓ Sauvegardé dans : {OUT}")