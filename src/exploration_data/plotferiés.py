"""
check_holidays.py
Vérifie visuellement les jours fériés/vacances dans le dataset.
"""

import polars as pl
import matplotlib.pyplot as plt
from pathlib import Path

DATASET = Path(r"C:\Users\gab1a\OneDrive\Documents\energyinfo2\DATA\processed\dataset_15min.parquet")

df = pl.read_parquet(DATASET).select(["timestamp", "is_holiday"])

# Un graphique par année
years = [2022, 2023, 2024, 2025]
fig, axes = plt.subplots(len(years), 1, figsize=(18, 10))

for ax, year in zip(axes, years):
    sub = df.filter(pl.col("timestamp").dt.year() == year)
    ax.fill_between(sub["timestamp"].to_list(), sub["is_holiday"].to_list(), alpha=0.6, color="steelblue")
    ax.set_ylabel(str(year))
    ax.set_ylim(0, 1.2)
    ax.set_yticks([])

fig.suptitle("Jours fériés et vacances scolaires (is_holiday)", fontsize=13)
plt.tight_layout()
plt.savefig(r"C:\Users\gab1a\OneDrive\Documents\energyinfo2\check_holidays.png", dpi=150)