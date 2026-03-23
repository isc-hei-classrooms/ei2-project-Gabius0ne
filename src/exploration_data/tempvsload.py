"""
plot_load_vs_temp.py
Superposition de la consommation Oiken et de la température mesurée à Sion.
"""

import polars as pl
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

BASE   = Path(r"C:\Users\gab1a\OneDrive\Documents\energyinfo2\DATA")
OUTPUT = BASE / "processed" / "load_vs_temp.png"

STATION = "Sion"
PERIOD_HIVER = ("2023-01-01", "2023-03-31")
PERIOD_ETE   = ("2023-06-01", "2023-08-31")

# ── CHARGEMENT ──
df_oiken = (
    pd.read_csv(BASE / "oiken-data.csv",
        parse_dates=["timestamp"], dayfirst=True, na_values=["#N/A","N/A","NA",""])
    .rename(columns={"standardised load [-]": "load"})
    .set_index("timestamp")
    .sort_index()
)
load_h = df_oiken["load"].resample("1h").mean()

# meteo : temperature mesurée à Sion, conversion UTC -> heure locale
temp_col = f"temp_2m_{STATION}"
df_temp = pl.read_parquet(BASE / "meteo_multistation_v3.parquet").select(["timestamp", temp_col]).to_pandas()
df_temp["timestamp"] = pd.to_datetime(df_temp["timestamp"], utc=True).dt.tz_convert("Europe/Zurich").dt.tz_localize(None)
df_temp = df_temp.set_index("timestamp").sort_index()
temp_h = df_temp[temp_col].resample("1h").mean()

# alignement sur les timestamps communs
common = load_h.dropna().index.intersection(temp_h.dropna().index)
load_aligned = load_h.loc[common]
temp_aligned = temp_h.loc[common]

# ── FIGURE ──
fig, axes = plt.subplots(2, 1, figsize=(16, 9))
fig.patch.set_facecolor("#1a1a2e")

def plot_period(ax, start, end, title):
    ax.set_facecolor("#16213e")
    ax.tick_params(colors="#e0e0e0")
    ax.spines[:].set_color("#444")

    mask = (common >= start) & (common <= end)
    ax_twin = ax.twinx()

    ax.plot(common[mask], load_aligned[mask], color="#00b4d8", lw=0.8, alpha=0.9, label="Load normalisé")
    ax_twin.plot(common[mask], temp_aligned[mask], color="#ff6b6b", lw=0.8, alpha=0.9, label=f"Température {STATION} (°C)")

    ax.set_ylabel("Load normalisé", color="#00b4d8", fontsize=10)
    ax_twin.set_ylabel("Température (°C)", color="#ff6b6b", fontsize=10)
    ax.tick_params(axis="y", colors="#00b4d8")
    ax_twin.tick_params(axis="y", colors="#ff6b6b")
    ax_twin.spines[:].set_color("#444")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.set_title(title, color="white", fontsize=12, pad=8)

    # legende combinée des deux axes
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax_twin.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2,
              loc="upper right", facecolor="#1a1a2e", labelcolor="white", fontsize=9)

plot_period(axes[0], *PERIOD_HIVER, f"Load vs Température — Hiver 2023")
plot_period(axes[1], *PERIOD_ETE,   f"Load vs Température — Été 2023")

plt.suptitle(f"Consommation Oiken — Température {STATION}",
             color="white", fontsize=14, y=1.01)
plt.tight_layout(pad=2.0)
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUTPUT, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")