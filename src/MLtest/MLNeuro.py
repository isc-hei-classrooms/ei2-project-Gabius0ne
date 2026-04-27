"""
train_mlp_v13.py
================
Réseau de neurones (MLP) multi-output pour la prévision day-ahead Oiken.
Basé sur les features v13 (~1308 colonnes).

MOTIVATION
----------
LightGBM (tree-based) ne peut pas extrapoler au-delà des valeurs vues
en training : ses prédictions plafonnent au min/max des feuilles terminales.
Quand la capacité PV d'Oiken croît (55 → 108 MWp), le load net descend
à des niveaux jamais vus en training, et LightGBM ne peut pas suivre.

Un MLP utilise des fonctions d'activation continues et des multiplications
de poids → il PEUT extrapoler linéairement au-delà du training. Si le
réseau apprend que load_net ≈ conso - w × irradiance × pv_yield, le
coefficient w s'applique même quand pv_yield dépasse la plage du training.

ARCHITECTURE
------------
- Un seul modèle avec 96 sorties (une par pas de 15 min)
- Les couches cachées sont partagées → le réseau apprend des
  représentations communes (profil journalier, saisonnalité)
- Couches : Input → 512 → 256 → 128 → 96
- BatchNorm + Dropout entre chaque couche
- Activation : ReLU (permet l'extrapolation linéaire dans le domaine positif)

PRÉPARATION DES DONNÉES
-----------------------
Contrairement à LightGBM, un MLP nécessite :
  - Standardisation des features (StandardScaler fitté sur train seul)
  - Imputation des NaN (médiane du train, car le MLP ne gère pas les NaN)
  - Standardisation de la cible Y (optionnel mais aide la convergence)

ENTRAÎNEMENT
------------
- Optimiseur : AdamW (Adam + weight decay = régularisation L2 intégrée)
- Loss : MSE (équivalent à minimiser le RMSE)
- Learning rate scheduler : ReduceLROnPlateau (réduit le LR quand val stagne)
- Early stopping : patience 30 époques sur la val loss
- Batch size : 64
- Époques max : 500

Split identique au v13 LightGBM : 47% train / 20% val / 33% test.
Exclusion 13-16 sept 2025 des métriques.

SORTIES
-------
  DATA/models_mlp_v13/mlp_model.pt          — modèle PyTorch
  DATA/models_mlp_v13/scaler_X.pkl          — StandardScaler features
  DATA/models_mlp_v13/scaler_Y.pkl          — StandardScaler cible
  DATA/models_mlp_v13/imputer_medians.npy   — médianes pour imputation NaN
  DATA/models_mlp_v13/metrics.parquet
  DATA/models_mlp_v13/predictions_test.parquet
"""

import polars as pl
import numpy as np
import pickle
import json
from pathlib import Path
from datetime import date
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
BASE = Path(__file__).resolve().parents[2] / "DATA"

X_PATH = BASE / "processed" / "X_features_v13.parquet"
Y_PATH = BASE / "processed" / "Y_target_v13.parquet"
B_PATH = BASE / "processed" / "B_baseline_v13.parquet"
OUT    = BASE / "models_mlp_v13"
OUT.mkdir(parents=True, exist_ok=True)

TRAIN_RATIO = 0.47
VAL_RATIO   = 0.20

# MLP hyperparamètres
HIDDEN_SIZES   = [512, 256, 128]
DROPOUT        = 0.2
BATCH_SIZE     = 64
LR_INITIAL     = 1e-3
WEIGHT_DECAY   = 1e-4
EPOCHS_MAX     = 500
PATIENCE       = 30        # early stopping patience
LR_PATIENCE    = 10        # reduce LR patience
LR_FACTOR      = 0.5       # reduce LR factor
RANDOM_SEED    = 42

EXCLUDE_DATES = {date(2025, 9, 13), date(2025, 9, 14), date(2025, 9, 15), date(2025, 9, 16)}

# Groupes pour métriques (identique v13)
NIGHT_STEPS = list(range(0, 40)) + list(range(68, 96))
DAY_STEPS   = list(range(40, 68))

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Device : {DEVICE}")
if DEVICE.type == "cuda":
    print(f"  GPU : {torch.cuda.get_device_name(0)}")


# ─────────────────────────────────────────────
# MODÈLE MLP
# ─────────────────────────────────────────────

class OikenMLP(nn.Module):
    """
    MLP multi-output : n_features → 96 sorties (un par pas de 15 min).
    Architecture : couches cachées avec BatchNorm + ReLU + Dropout.
    """
    def __init__(self, n_features: int, n_outputs: int = 96,
                 hidden_sizes: list[int] = None, dropout: float = 0.2):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [512, 256, 128]

        layers = []
        in_size = n_features
        for h_size in hidden_sizes:
            layers.append(nn.Linear(in_size, h_size))
            layers.append(nn.BatchNorm1d(h_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_size = h_size
        layers.append(nn.Linear(in_size, n_outputs))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# ─────────────────────────────────────────────
# 1. CHARGEMENT
# ─────────────────────────────────────────────

print("=== Chargement ===")
X = pl.read_parquet(X_PATH)
Y = pl.read_parquet(Y_PATH)
B = pl.read_parquet(B_PATH)

dates = X["date"]
feat_names = [c for c in X.columns if c != "date"]

X_arr = X.drop("date").to_numpy().astype(np.float32)
Y_arr = Y.drop("date").to_numpy().astype(np.float32)
B_arr = B.drop("date").to_numpy().astype(np.float32)

n_samples = X_arr.shape[0]
n_features = X_arr.shape[1]
n_steps = Y_arr.shape[1]

print(f"  Samples : {n_samples}")
print(f"  Features : {n_features}")
print(f"  Steps : {n_steps}")


# ─────────────────────────────────────────────
# 2. SPLIT CHRONOLOGIQUE
# ─────────────────────────────────────────────

split_train = int(n_samples * TRAIN_RATIO)
split_val   = int(n_samples * (TRAIN_RATIO + VAL_RATIO))

X_train_raw = X_arr[:split_train]
X_val_raw   = X_arr[split_train:split_val]
X_test_raw  = X_arr[split_val:]

Y_train_raw = Y_arr[:split_train]
Y_val_raw   = Y_arr[split_train:split_val]
Y_test      = Y_arr[split_val:]
B_test      = B_arr[split_val:]
dates_test  = dates[split_val:]

print(f"\n=== Split ===")
print(f"  Train : {split_train} ({dates[0]} → {dates[split_train-1]})")
print(f"  Val   : {split_val - split_train} ({dates[split_train]} → {dates[split_val-1]})")
print(f"  Test  : {n_samples - split_val} ({dates[split_val]} → {dates[-1]})")

dates_test_list = dates_test.to_list()
exclude_mask = np.array([d not in EXCLUDE_DATES for d in dates_test_list], dtype=bool)
n_excluded = (~exclude_mask).sum()
print(f"  Dates exclues : {n_excluded}")


# ─────────────────────────────────────────────
# 3. IMPUTATION NaN + STANDARDISATION
# ─────────────────────────────────────────────
# Le MLP ne gère pas les NaN → on impute par la médiane du train.
# Puis on standardise X et Y séparément (fitté sur train uniquement).

print("\n=== Préparation données ===")

# Imputation NaN par médiane du train (par colonne)
medians = np.nanmedian(X_train_raw, axis=0)
# Cas edge : si une colonne est 100% NaN dans le train, on met 0
medians = np.where(np.isnan(medians), 0.0, medians)

def impute(arr, medians):
    arr = arr.copy()
    for j in range(arr.shape[1]):
        mask = np.isnan(arr[:, j])
        arr[mask, j] = medians[j]
    return arr

X_train_imp = impute(X_train_raw, medians)
X_val_imp   = impute(X_val_raw, medians)
X_test_imp  = impute(X_test_raw, medians)

# Standardisation X (mean=0, std=1 par feature)
scaler_X = StandardScaler()
X_train_sc = scaler_X.fit_transform(X_train_imp).astype(np.float32)
X_val_sc   = scaler_X.transform(X_val_imp).astype(np.float32)
X_test_sc  = scaler_X.transform(X_test_imp).astype(np.float32)

# Imputation Y (remplacer NaN par 0 — rare, juste pour ne pas crasher)
Y_train_imp = np.where(np.isnan(Y_train_raw), 0.0, Y_train_raw).astype(np.float32)
Y_val_imp   = np.where(np.isnan(Y_val_raw), 0.0, Y_val_raw).astype(np.float32)

# Standardisation Y (aide la convergence car les 96 sorties ont des
# échelles légèrement différentes jour vs nuit)
scaler_Y = StandardScaler()
Y_train_sc = scaler_Y.fit_transform(Y_train_imp).astype(np.float32)
Y_val_sc   = scaler_Y.transform(Y_val_imp).astype(np.float32)

print(f"  X après scaling : mean≈{X_train_sc.mean():.4f}, std≈{X_train_sc.std():.4f}")
print(f"  Y après scaling : mean≈{Y_train_sc.mean():.4f}, std≈{Y_train_sc.std():.4f}")
print(f"  NaN imputés (X train) : {np.isnan(X_train_raw).sum():,}")
print(f"  NaN imputés (Y train) : {np.isnan(Y_train_raw).sum():,}")

# Sauvegarde des transformateurs pour inférence
np.save(OUT / "imputer_medians.npy", medians)
with open(OUT / "scaler_X.pkl", "wb") as f:
    pickle.dump(scaler_X, f)
with open(OUT / "scaler_Y.pkl", "wb") as f:
    pickle.dump(scaler_Y, f)


# ─────────────────────────────────────────────
# 4. DATALOADERS PYTORCH
# ─────────────────────────────────────────────

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

train_ds = TensorDataset(torch.from_numpy(X_train_sc), torch.from_numpy(Y_train_sc))
val_ds   = TensorDataset(torch.from_numpy(X_val_sc),   torch.from_numpy(Y_val_sc))

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)


# ─────────────────────────────────────────────
# 5. ENTRAÎNEMENT
# ─────────────────────────────────────────────

model = OikenMLP(
    n_features=n_features,
    n_outputs=n_steps,
    hidden_sizes=HIDDEN_SIZES,
    dropout=DROPOUT,
).to(DEVICE)

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n=== Modèle MLP ===")
print(f"  Architecture : {n_features} → {' → '.join(map(str, HIDDEN_SIZES))} → {n_steps}")
print(f"  Paramètres : {n_params:,}")
print(f"  Dropout : {DROPOUT}")
print(f"  Batch size : {BATCH_SIZE}")
print(f"  LR : {LR_INITIAL} | Weight decay : {WEIGHT_DECAY}")

criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR_INITIAL, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=LR_FACTOR, patience=LR_PATIENCE
)

best_val_loss = float('inf')
best_epoch = 0
epochs_no_improve = 0

print(f"\n=== Entraînement (max {EPOCHS_MAX} époques, patience {PATIENCE}) ===")

for epoch in range(1, EPOCHS_MAX + 1):
    # ── Train ──
    model.train()
    train_loss_sum = 0.0
    train_n = 0
    for X_batch, Y_batch in train_loader:
        X_batch = X_batch.to(DEVICE)
        Y_batch = Y_batch.to(DEVICE)

        optimizer.zero_grad()
        pred = model(X_batch)
        loss = criterion(pred, Y_batch)
        loss.backward()
        optimizer.step()

        train_loss_sum += loss.item() * X_batch.shape[0]
        train_n += X_batch.shape[0]

    train_loss = train_loss_sum / train_n

    # ── Validation ──
    model.eval()
    val_loss_sum = 0.0
    val_n = 0
    with torch.no_grad():
        for X_batch, Y_batch in val_loader:
            X_batch = X_batch.to(DEVICE)
            Y_batch = Y_batch.to(DEVICE)
            pred = model(X_batch)
            loss = criterion(pred, Y_batch)
            val_loss_sum += loss.item() * X_batch.shape[0]
            val_n += X_batch.shape[0]

    val_loss = val_loss_sum / val_n
    scheduler.step(val_loss)

    current_lr = optimizer.param_groups[0]['lr']

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch
        epochs_no_improve = 0
        # Sauvegarder le meilleur modèle
        torch.save(model.state_dict(), OUT / "mlp_model_best.pt")
    else:
        epochs_no_improve += 1

    if epoch % 10 == 0 or epoch <= 5 or epochs_no_improve == 0:
        print(f"  Epoch {epoch:4d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | "
              f"lr={current_lr:.2e} | best={best_epoch} (no_improve={epochs_no_improve})")

    if epochs_no_improve >= PATIENCE:
        print(f"\n  Early stopping à epoch {epoch} (best={best_epoch}, val_loss={best_val_loss:.6f})")
        break

# Charger le meilleur modèle
model.load_state_dict(torch.load(OUT / "mlp_model_best.pt", weights_only=True))
torch.save(model.state_dict(), OUT / "mlp_model.pt")
print(f"  Meilleur modèle : epoch {best_epoch}, val_loss={best_val_loss:.6f}")


# ─────────────────────────────────────────────
# 6. PRÉDICTION TEST
# ─────────────────────────────────────────────

model.eval()
with torch.no_grad():
    X_test_tensor = torch.from_numpy(X_test_sc).to(DEVICE)
    preds_sc = model(X_test_tensor).cpu().numpy()

# Dénormalisation Y (inverse du StandardScaler)
preds_test = scaler_Y.inverse_transform(preds_sc).astype(np.float32)

print(f"\n  Prédictions test : shape={preds_test.shape}")
print(f"  Range preds : [{preds_test.min():.3f}, {preds_test.max():.3f}]")
print(f"  Range Y_test : [{np.nanmin(Y_test):.3f}, {np.nanmax(Y_test):.3f}]")


# ─────────────────────────────────────────────
# 7. MÉTRIQUES
# ─────────────────────────────────────────────

mask_all = ~np.isnan(Y_test) & ~np.isnan(preds_test) & exclude_mask[:, None]
rmse_global = float(np.sqrt(mean_squared_error(Y_test[mask_all], preds_test[mask_all])))
mae_global  = float(mean_absolute_error(Y_test[mask_all], preds_test[mask_all]))

mask_b_all = ~np.isnan(Y_test) & ~np.isnan(B_test) & exclude_mask[:, None]
rmse_base = float(np.sqrt(mean_squared_error(Y_test[mask_b_all], B_test[mask_b_all])))
mae_base  = float(mean_absolute_error(Y_test[mask_b_all], B_test[mask_b_all]))

print(f"\n=== Résultats globaux MLP (excl. 13-16 sept) ===")
print(f"  Test set : {exclude_mask.sum()} jours (exclu {n_excluded})")
print(f"  MLP      — RMSE : {rmse_global:.4f} | MAE : {mae_global:.4f}")
print(f"  Baseline — RMSE : {rmse_base:.4f} | MAE : {mae_base:.4f}")
print(f"  Amélioration RMSE : {(1 - rmse_global / rmse_base) * 100:+.1f}%")
print(f"  Amélioration MAE  : {(1 - mae_global / mae_base) * 100:+.1f}%")

night_set = set(NIGHT_STEPS)
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
    print(f"  {group:5s} — RMSE modèle={rmse_m:.4f} | baseline={rmse_b:.4f} | {imp:+.1f}% | "
          f"MAE modèle={mae_m:.4f} | baseline={mae_b:.4f}")

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
# 8. DIAGNOSTIC BIAIS DIURNE
# ─────────────────────────────────────────────

print(f"\n=== Diagnostic biais diurne (10h-17h) ===")
day_steps = list(range(40, 68))

monthly_bias = defaultdict(list)
for i, d in enumerate(dates_test_list):
    if not exclude_mask[i]:
        continue
    y_day = Y_test[i, day_steps]
    p_day = preds_test[i, day_steps]
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

# Métriques par pas
metrics = []
for t in range(n_steps):
    y_te = Y_test[:, t]
    p_te = preds_test[:, t]
    b_te = B_test[:, t]
    mask_te = ~np.isnan(y_te) & exclude_mask
    mask_b = ~np.isnan(y_te) & ~np.isnan(b_te) & exclude_mask

    rmse_m = float(np.sqrt(mean_squared_error(y_te[mask_te], p_te[mask_te]))) if mask_te.sum() > 0 else None
    mae_m  = float(mean_absolute_error(y_te[mask_te], p_te[mask_te])) if mask_te.sum() > 0 else None
    rmse_b = float(np.sqrt(mean_squared_error(y_te[mask_b], b_te[mask_b]))) if mask_b.sum() > 0 else None
    mae_b  = float(mean_absolute_error(y_te[mask_b], b_te[mask_b])) if mask_b.sum() > 0 else None

    metrics.append({
        "step": t,
        "time_label": f"{(t*15)//60:02d}h{(t*15)%60:02d}",
        "group": "NUIT" if t in night_set else "JOUR",
        "rmse_model": rmse_m,
        "mae_model": mae_m,
        "rmse_baseline": rmse_b,
        "mae_baseline": mae_b,
    })

pl.DataFrame(metrics).write_parquet(OUT / "metrics.parquet")

pred_cols = {f"pred_t{t:03d}": preds_test[:, t].tolist() for t in range(n_steps)}
pred_cols["date"] = dates_test.to_list()
pl.DataFrame(pred_cols).select(
    ["date"] + [f"pred_t{t:03d}" for t in range(n_steps)]
).write_parquet(OUT / "predictions_test.parquet")

# Sauvegarder la config pour reproductibilité
config = {
    "hidden_sizes": HIDDEN_SIZES,
    "dropout": DROPOUT,
    "batch_size": BATCH_SIZE,
    "lr_initial": LR_INITIAL,
    "weight_decay": WEIGHT_DECAY,
    "best_epoch": best_epoch,
    "best_val_loss": best_val_loss,
    "n_features": n_features,
    "n_steps": n_steps,
    "n_params": n_params,
    "train_ratio": TRAIN_RATIO,
    "val_ratio": VAL_RATIO,
}
with open(OUT / "config.json", "w") as f:
    json.dump(config, f, indent=2)

print(f"\n✓ Modèle : {OUT}/mlp_model.pt ({n_params:,} paramètres)")
print(f"✓ Scalers : {OUT}/scaler_X.pkl, scaler_Y.pkl")
print(f"✓ Métriques : {OUT}/metrics.parquet")
print(f"✓ Prédictions : {OUT}/predictions_test.parquet")
print(f"✓ Config : {OUT}/config.json")