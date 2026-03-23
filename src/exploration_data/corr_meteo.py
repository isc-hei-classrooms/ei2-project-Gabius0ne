"""
check_correlations.py
Corrélation entre l'erreur de prévision Oiken et les variables météo.
"""

import polars as pl
import matplotlib.pyplot as plt
from pathlib import Path

DATASET = Path(r"C:\Users\gab1a\OneDrive\Documents\energyinfo2\DATA\processed\dataset_15min.parquet")

METEO = [
    "pred_glob_rad_ctrl", "pred_sunshine_ctrl", "pred_temp_ctrl",
    "pred_pressure_ctrl", "pred_relhum_ctrl", "pred_precip_ctrl",
    "wind_speed",
]

df = pl.read_parquet(DATASET).select(["forecast_error"] + METEO).drop_nulls()
#prend les 8 colonnes (7 météo et forecast err) et enlève les lignes "nulls"

corrs = {col: df.select(pl.corr("forecast_error", col)).item() for col in METEO}
#calcul pour chaque pair un peitt Datafram contenant le coef de Pearson
corrs = dict(sorted(corrs.items(), key=lambda x: abs(x[1]), reverse=True))
#trie par ordre décroissant

plt.figure(figsize=(8, 5))
plt.barh(list(corrs.keys()), list(corrs.values()), color="steelblue")
#barplot
plt.axvline(0, color="black", linewidth=0.8)
#millieu ( corr 0)
plt.title("Corrélation avec l'erreur de prévision Oiken")
plt.tight_layout()
plt.savefig(r"C:\Users\gab1a\OneDrive\Documents\energyinfo2\check_correlations.png", dpi=150)
