"""
Corrélation PV totale vs prévisions météo (Sion, h+24), par mois.
"""

import polars as pl
import numpy as np
import matplotlib
matplotlib.use("Agg")  # backend sans GUI, pour sauvegarder directement en PNG
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

# ── CONFIG ──
BASE = Path(r"C:\Users\gab1a\OneDrive\Documents\energyinfo2\DATA")
OUT  = BASE / "processed"
OUT.mkdir(exist_ok=True)

H_START, H_END = 10, 16  # fenetre solaire, on garde que les heures ou le PV produit vraiement

PROD_COLS = [
    "central valais solar production [kWh]",
    "sion area solar production [kWh]",
    "sierre area production [kWh]",
    "remote solar production [kWh]",
]


METEO_VARS = {
    "pred_glob_rad_h24_Sion":  "Irradiance globale prévue [W/m²]",
    "pred_sunshine_h24_Sion":  "Ensoleillement prévu [min]",
    "pred_relhum_h24_Sion":    "Humidité relative prévue [%]",
    "pred_temp_h24_Sion":      "Température prévue [°C]",
}

MOIS = ["Jan","Fév","Mar","Avr","Mai","Jun","Jul","Aoû","Sep","Oct","Nov","Déc"]

# ── CHARGEMENT OIKEN ──
df_oiken = (
    pl.read_csv(BASE / "oiken-data.csv", try_parse_dates=True, null_values=["#N/A","N/A","NA",""])
    .with_columns(pl.sum_horizontal(PROD_COLS).alias("pv_total"))  # somme des 4 zones PV
    .select(["timestamp", "pv_total"])
    .with_columns(pl.col("timestamp").dt.replace_time_zone(None))  # enleve le timezone pour pouvoir joindre apres
    .filter(pl.col("timestamp").dt.hour().is_between(H_START, H_END))  # garde que 10h-16h
    .group_by(pl.col("timestamp").dt.date().alias("date"))
    .agg(pl.col("pv_total").sum())  # agregation journaliere : somme du PV sur la fenetre
    .sort("date")
)

# ── CHARGEMENT MÉTÉO ──
df_meteo = pl.read_parquet(BASE / "meteo_multistation_v3.parquet")

# si timestamps en UTC on converti en heure locale suisse 
if df_meteo["timestamp"].dtype.time_zone == "UTC":
    df_meteo = df_meteo.with_columns(
        pl.col("timestamp").dt.convert_time_zone("Europe/Zurich")
    )

# garde que les colonnes meteo qui existent dans le parquet
meteo_cols = [c for c in METEO_VARS if c in df_meteo.columns]

df_meteo = (
    df_meteo
    .select(["timestamp"] + meteo_cols)
    .with_columns(pl.col("timestamp").dt.replace_time_zone(None))
    .filter(pl.col("timestamp").dt.hour().is_between(H_START, H_END))
    .group_by(pl.col("timestamp").dt.date().alias("date"))
    .agg([pl.col(c).mean() for c in meteo_cols])  # moyenne journaliere (pas somme, c'est des grandeurs intensives)
    .sort("date")
)

# ── FUSION ──
# inner join sur la date : garde que les jours ou on a PV ET meteo
df = (
    df_oiken.join(df_meteo, on="date", how="inner")
    .with_columns([
        pl.col("date").dt.month().alias("month"),
        pl.col("date").dt.year().alias("year"),
    ])
    .drop_nulls()
    .to_pandas()  # passage pandas pour matplotlib / scipy
)

# ── FIGURES ──
# une figure par variable meteo, avec 12 subplots (1 par mois)
for meteo_col, meteo_label in METEO_VARS.items():
    if meteo_col not in df.columns:
        continue

    fig, axes = plt.subplots(3, 4, figsize=(20, 14))
    fig.patch.set_facecolor("#1a1a2e")
    fig.suptitle(
        f"Corrélation PV vs {meteo_label}\n(Sion, h+24, {H_START}h–{H_END}h)",
        color="white", fontsize=14, y=1.01,
    )

    for m in range(1, 13):
        ax = axes[(m - 1) // 4][(m - 1) % 4]  # position dans la grille 3x4
        ax.set_facecolor("#16213e")
        ax.tick_params(colors="#e0e0e0", labelsize=7)
        ax.spines[:].set_color("#444")

        subset = df[df["month"] == m]
        ax.set_title(MOIS[m - 1], color="white", fontsize=9, pad=4)

        if len(subset) < 5:  # pas assez de points pour une regression fiable
            ax.text(0.5, 0.5, "Données\ninsuffisantes",
                    ha="center", va="center", color="#888", transform=ax.transAxes)
            continue

        x, y = subset[meteo_col].values, subset["pv_total"].values
        ax.scatter(x, y, c=subset["year"].values, cmap="plasma", alpha=0.6, s=12)

        # regression lineaire + coeff de pearson
        slope, intercept, r, p, _ = stats.linregress(x, y)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_line, slope * x_line + intercept, color="white", lw=1.2, alpha=0.8)

        # couleur du r selon la force de la correlation
        color_r = "#00ff88" if abs(r) > 0.5 else "#ffaa00" if abs(r) > 0.3 else "#ff6b6b"
        ax.text(0.05, 0.92, f"r = {r:.2f}", transform=ax.transAxes,
                color=color_r, fontsize=8, fontweight="bold")
        ax.text(0.05, 0.82, f"n = {len(subset)}", transform=ax.transAxes,
                color="#aaaaaa", fontsize=7)

        ax.set_xlabel(meteo_label.split("[")[0].strip(), color="#aaaaaa", fontsize=7)
        ax.set_ylabel("PV total [kWh]", color="#aaaaaa", fontsize=7)

    plt.tight_layout(pad=2.0)
    safe_name = meteo_col.replace("/", "-").replace(" ", "_")
    plt.savefig(OUT / f"corr_pv_{safe_name}.png", dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close()
