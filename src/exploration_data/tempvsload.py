"""
plot_correlation.py
===================
Visualisation de la corrélation entre la consommation Oiken (load normalisé)
et la température mesurée à Sion.

Sorties :
  - correlation_load_temp.png
"""

import polars as pl
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from pathlib import Path

# ─────────────────────────────────────────────
# CONFIGURATION — adapter les chemins
# ─────────────────────────────────────────────
BASE   = Path(r"C:\Users\gab1a\OneDrive\Documents\energyinfo2\DATA")
CSV    = BASE / "oiken-data.csv"
METEO  = BASE / "meteo_multistation_v3.parquet"
OUTPUT = BASE / "processed" / "correlation_load_temp.png"

STATION = "Sion"   # station de référence pour la température

# Périodes à afficher dans les séries temporelles
PERIOD_HIVER = ("2023-01-01", "2023-03-31")
PERIOD_ETE   = ("2023-06-01", "2023-08-31")


# ─────────────────────────────────────────────
# CHARGEMENT
# ─────────────────────────────────────────────

df_oiken = pd.read_csv(
    CSV,
    parse_dates=["timestamp"],
    dayfirst=True,
    na_values=["#N/A", "N/A", "NA", ""],
)
df_oiken = df_oiken.rename(columns={"standardised load [-]": "load"})
df_oiken = df_oiken.set_index("timestamp").sort_index()
load_h = df_oiken["load"].resample("1h").mean()

df_meteo = pl.read_parquet(METEO)
temp_col = f"temp_2m_{STATION}"
df_temp = df_meteo.select(["timestamp", temp_col]).to_pandas()
df_temp["timestamp"] = (
    pd.to_datetime(df_temp["timestamp"], utc=True)
      .dt.tz_convert("Europe/Zurich")
      .dt.tz_localize(None)
)
df_temp = df_temp.set_index("timestamp").sort_index()
temp_h = df_temp[temp_col].resample("1h").mean()

# Alignement
common = load_h.index.intersection(temp_h.index)
load_aligned = load_h.loc[common].dropna()
temp_aligned = temp_h.loc[common].dropna()
common = load_aligned.index.intersection(temp_aligned.index)
load_aligned = load_aligned.loc[common]
temp_aligned = temp_aligned.loc[common]

# Moyennes journalières pour le scatter
load_d = load_aligned.resample("1D").mean()
temp_d = temp_aligned.resample("1D").mean()
common_d = load_d.index.intersection(temp_d.index)
load_d = load_d.loc[common_d]
temp_d = temp_d.loc[common_d]
corr = temp_d.corr(load_d)


# ─────────────────────────────────────────────
# FIGURE
# ─────────────────────────────────────────────

fig, axes = plt.subplots(3, 1, figsize=(16, 12),
    gridspec_kw={"height_ratios": [2, 2, 1.8]})
fig.patch.set_facecolor("#1a1a2e")
for ax in axes:
    ax.set_facecolor("#16213e")
    ax.tick_params(colors="#e0e0e0")
    ax.spines[:].set_color("#444")

def plot_period(ax, start, end, title):
    mask = (common >= start) & (common <= end)
    ax_twin = ax.twinx()
    ax.plot(common[mask], load_aligned[mask],
            color="#00b4d8", lw=0.8, alpha=0.9, label="Load normalisé")
    ax_twin.plot(common[mask], temp_aligned[mask],
                 color="#ff6b6b", lw=0.8, alpha=0.9, label=f"Température {STATION} (°C)")
    ax.set_ylabel("Load normalisé", color="#00b4d8", fontsize=10)
    ax_twin.set_ylabel("Température (°C)", color="#ff6b6b", fontsize=10)
    ax.tick_params(axis="y", colors="#00b4d8")
    ax_twin.tick_params(axis="y", colors="#ff6b6b")
    ax_twin.tick_params(colors="#e0e0e0")
    ax_twin.spines[:].set_color("#444")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.set_title(title, color="white", fontsize=12, pad=8)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax_twin.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2,
              loc="upper right", facecolor="#1a1a2e", labelcolor="white", fontsize=9)

plot_period(axes[0], *PERIOD_HIVER, f"Consommation vs Température — {PERIOD_HIVER[0]} → {PERIOD_HIVER[1]}")
plot_period(axes[1], *PERIOD_ETE,   f"Consommation vs Température — {PERIOD_ETE[0]} → {PERIOD_ETE[1]}")

# Scatter journalier
ax3 = axes[2]
sc = ax3.scatter(temp_d, load_d,
    c=common_d.month, cmap="plasma", alpha=0.5, s=10)
z = np.polyfit(temp_d, load_d, 2)
p = np.poly1d(z)
x_line = np.linspace(temp_d.min(), temp_d.max(), 200)
ax3.plot(x_line, p(x_line), color="white", lw=1.5, alpha=0.8, label="Tendance quadratique")
ax3.set_xlabel(f"Température moyenne journalière {STATION} (°C)", color="#e0e0e0", fontsize=10)
ax3.set_ylabel("Load moyen journalier", color="#e0e0e0", fontsize=10)
ax3.set_title(f"Scatter load vs température (r = {corr:.3f})", color="white", fontsize=12, pad=8)
ax3.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=9)
ax3.tick_params(colors="#e0e0e0")
cbar = plt.colorbar(sc, ax=ax3)
cbar.set_label("Mois", color="white")
cbar.ax.yaxis.set_tick_params(color="white")
plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

plt.suptitle(f"Corrélation Consommation Oiken — Température {STATION}",
             color="white", fontsize=14, y=1.01)
plt.tight_layout(pad=2.0)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUTPUT, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
print(f"✓ Sauvegardé : {OUTPUT}")