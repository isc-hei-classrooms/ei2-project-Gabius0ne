"""
corr_pv.py
Corrélation entre la production PV et l'erreur de prévision Oiken.
"""

import polars as pl
import matplotlib.pyplot as plt
from pathlib import Path

DATASET = Path(r"C:\Users\gab1a\OneDrive\Documents\energyinfo2\DATA\processed\dataset_15min.parquet")

PV_COLS = ["solar_central_valais", "solar_sion", "solar_sierre", "solar_remote"]

df = pl.read_parquet(DATASET).select(["forecast_error"] + PV_COLS).drop_nulls()
#drop les lignes avec un null

corrs = {col: df.select(pl.corr("forecast_error", col)).item() for col in PV_COLS}
#pour chaque pair calcul le coef de pearson retourne un dict
corrs = dict(sorted(corrs.items(), key=lambda x: abs(x[1]), reverse=True))
#trie du plus petit au plus élevé

plt.figure(figsize=(7, 4))
plt.barh(list(corrs.keys()), list(corrs.values()), color="steelblue")
plt.axvline(0, color="black", linewidth=0.8)
plt.title("Corrélation production PV vs erreur de prévision Oiken")
plt.tight_layout()
plt.savefig(r"C:\Users\gab1a\OneDrive\Documents\energyinfo2\corr_pv.png", dpi=150)
print("Sauvegardé : corr_pv.png")