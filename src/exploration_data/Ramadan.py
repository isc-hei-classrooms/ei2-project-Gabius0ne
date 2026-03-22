import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from datetime import date

BASE = Path(r"C:\Users\gab1a\OneDrive\Documents\energyinfo2\DATA")

RAMADAN_START = date(2025, 3, 1)
RAMADAN_END   = date(2025, 3, 30)

df = pd.read_csv(BASE / "oiken-data.csv",
    parse_dates=["timestamp"], dayfirst=True, na_values=["#N/A"])
df = df.rename(columns={
    "standardised load [-]":         "load",
    "standardised forecast load [-]": "forecast"
}).set_index("timestamp").sort_index()

ramadan_2025   = df.loc[str(RAMADAN_START):str(RAMADAN_END)]
reference_2024 = df.loc["2024-03-01":"2024-03-30"]

night_mask_r = (ramadan_2025.index.hour >= 21)   | (ramadan_2025.index.hour <= 5)
night_mask_f = (reference_2024.index.hour >= 21) | (reference_2024.index.hour <= 5)

ramadan_night  = ramadan_2025[night_mask_r].groupby(ramadan_2025[night_mask_r].index.hour)["load"].mean()
ref_night      = reference_2024[night_mask_f].groupby(reference_2024[night_mask_f].index.hour)["load"].mean()
forecast_night = ramadan_2025[night_mask_r].groupby(ramadan_2025[night_mask_r].index.hour)["forecast"].mean()

error_ramadan = (ramadan_2025[night_mask_r]["load"] - ramadan_2025[night_mask_r]["forecast"]).dropna()
error_ref     = (reference_2024[night_mask_f]["load"] - reference_2024[night_mask_f]["forecast"]).dropna()

error_by_hour_ramadan = error_ramadan.groupby(error_ramadan.index.hour).mean()
error_by_hour_ref     = error_ref.groupby(error_ref.index.hour).mean()

fig, axes = plt.subplots(2, 1, figsize=(14, 10))
fig.patch.set_facecolor("#1a1a2e")
for ax in axes:
    ax.set_facecolor("#16213e")
    ax.tick_params(colors="#e0e0e0")
    ax.spines[:].set_color("#444")

ax1 = axes[0]
ax1.plot(ramadan_night.index,  ramadan_night.values,  color="#ff6b6b", lw=2, label="Load réel Ramadan 2025")
ax1.plot(ref_night.index,      ref_night.values,      color="#00b4d8", lw=2, label="Load réel Mars 2024 (référence)")
ax1.plot(forecast_night.index, forecast_night.values, color="#ffaa00", lw=1.5, linestyle="--", label="Forecast Oiken Ramadan 2025")
ax1.set_title("Profil horaire nocturne — Ramadan 2025 vs Mars 2024", color="white", fontsize=12)
ax1.set_xlabel("Heure", color="#e0e0e0")
ax1.set_ylabel("Load normalisé [-]", color="#e0e0e0")
ax1.legend(facecolor="#1a1a2e", labelcolor="white")
ax1.set_xticks([0, 1, 2, 3, 4, 5, 21, 22, 23])

ax2 = axes[1]
ax2.bar(np.array(list(error_by_hour_ramadan.index)) - 0.2,
        error_by_hour_ramadan.values, width=0.4,
        color="#ff6b6b", alpha=0.8, label="Erreur Ramadan 2025")
ax2.bar(np.array(list(error_by_hour_ref.index)) + 0.2,
        error_by_hour_ref.values, width=0.4,
        color="#00b4d8", alpha=0.8, label="Erreur Mars 2024 (référence)")
ax2.axhline(0, color="white", lw=0.8, alpha=0.5)
ax2.set_title("Erreur de prévision Oiken (load réel - forecast) par heure nocturne", color="white", fontsize=12)
ax2.set_xlabel("Heure", color="#e0e0e0")
ax2.set_ylabel("Erreur normalisée [-]", color="#e0e0e0")
ax2.legend(facecolor="#1a1a2e", labelcolor="white")
ax2.set_xticks([0, 1, 2, 3, 4, 5, 21, 22, 23])

plt.tight_layout(pad=2.0)
plt.savefig(BASE / "processed" / "ramadan_analysis.png", dpi=150,
    bbox_inches="tight", facecolor="#1a1a2e")
print("✓ Sauvegardé")