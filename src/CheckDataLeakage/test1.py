"""
plot_semaine.py
===============
Plot une semaine de prévisions LightGBM vs réel vs baseline Oiken.
Corrigé : indexation par date pour aligner correctement les trois séries.
"""

import polars as pl
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from datetime import timedelta

BASE = Path(r"C:\Users\gab1a\OneDrive\Documents\energyinfo2\DATA")

# ── Semaine à afficher — doit être dans le jeu de test (après 2025-02-24)
START = "2025-05-07"
END   = "2025-05-13"

# ── Chargement
df_oiken = pd.read_csv(BASE / "oiken-data.csv",
    parse_dates=["timestamp"], dayfirst=True, na_values=["#N/A"])
df_oiken = df_oiken.rename(columns={
    "standardised load [-]":         "load",
    "standardised forecast load [-]": "forecast"
}).set_index("timestamp").sort_index()

preds = pl.read_parquet(BASE / "models" / "predictions_test.parquet").to_pandas()
Y     = pl.read_parquet(BASE / "processed" / "Y_target.parquet").to_pandas()

pred_cols = [f"pred_t{t:03d}" for t in range(96)]
Y_cols    = [f"load_t{t:03d}" for t in range(96)]

# ── Indexer par date pour aligner correctement
preds["date"] = pd.to_datetime(preds["date"]).dt.date
Y["date"]     = pd.to_datetime(Y["date"]).dt.date
preds = preds.set_index("date")
Y     = Y.set_index("date")

# ── Reconstruire les séries temporelles
rows = []
for d in pd.date_range(START, END):
    d_date = d.date()
    if d_date not in preds.index or d_date not in Y.index:
        continue
    for t in range(96):
        ts = pd.Timestamp(d_date) + timedelta(minutes=15 * t)
        rows.append({
            "timestamp": ts,
            "pred":      preds.loc[d_date, pred_cols[t]],
            "real":      Y.loc[d_date, Y_cols[t]],
        })

df_week  = pd.DataFrame(rows).set_index("timestamp").sort_index()
baseline = df_oiken.loc[START:END, "forecast"]

print(f"Jours dans le plot : {len(rows) // 96}")
print(f"Première valeur réelle  : {df_week['real'].iloc[0]:.4f}")
print(f"CSV load au {START} 00:15 : {df_oiken.loc[START + ' 00:15', 'load']:.4f}")

# ── Plot
fig, ax = plt.subplots(figsize=(16, 6))
fig.patch.set_facecolor("#1a1a2e")
ax.set_facecolor("#16213e")
ax.tick_params(colors="#e0e0e0")
ax.spines[:].set_color("#444")

ax.plot(df_week.index,  df_week["real"],    color="#e0e0e0", lw=1.2, label="Réel")
ax.plot(df_week.index,  df_week["pred"],    color="#00b4d8", lw=1.0, alpha=0.9, label="Modèle LightGBM")
ax.plot(baseline.index, baseline.values,    color="#ff6b6b", lw=1.0, alpha=0.7,
        linestyle="--", label="Baseline Oiken")

ax.set_ylabel("Load normalisé [-]", color="#e0e0e0")
ax.set_title(f"Prévision de charge — {START} → {END} (normalisé)", color="white", fontsize=13)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
ax.xaxis.set_major_locator(mdates.DayLocator())
ax.legend(facecolor="#1a1a2e", labelcolor="white")
plt.xticks(rotation=30, color="#e0e0e0")
plt.tight_layout()

out = BASE / "processed" / "prevision_semaine_norm.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
print(f"✓ Sauvegardé : {out}")