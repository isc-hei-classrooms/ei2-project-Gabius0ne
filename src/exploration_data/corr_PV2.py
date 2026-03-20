"""
corr_pv_meteo.py
================
Corrélation entre production PV totale et prévisions météo (station Sion, horizon 24h).

Paramètres :
  - Fenêtre temporelle : 10h–16h (pic solaire)
  - Station : Sion
  - Horizon : 24h
  - Variables météo prévues (pas de mesures réelles) :
      * pred_glob_rad_h24_Sion  — irradiance globale prévue
      * pred_sunshine_h24_Sion  — ensoleillement prévu
      * pred_relhum_h24_Sion    — humidité relative prévue
      * pred_temp_h24_Sion      — température prévue
  - Production PV : somme de solar_central_valais + solar_sion + solar_sierre + solar_remote
    agrégée sur 10h–16h (total de la fenêtre)

Structure graphique :
  - 4 figures (une par variable météo)
  - Chaque figure : 12 scatter en grille 3×4 (un par mois Jan–Déc)
  - Axe X : valeur météo moyenne 10h–16h
  - Axe Y : production PV totale 10h–16h
  - Coefficient r affiché sur chaque scatter
  - Droite de tendance linéaire
"""

import polars as pl
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
BASE   = Path(r"C:\Users\gab1a\OneDrive\Documents\energyinfo2\DATA")
CSV    = BASE / "oiken-data.csv"
METEO  = BASE / "meteo_multistation_v3.parquet"
OUT    = BASE / "processed"
OUT.mkdir(exist_ok=True)

STATION  = "Sion"
HORIZON  = "24"
H_START  = 10   # heure début fenêtre solaire
H_END    = 16   # heure fin fenêtre solaire

PROD_COLS = [
    "central valais solar production [kWh]",
    "sion area solar production [kWh]",
    "sierre area production [kWh]",
    "remote solar production [kWh]",
]

# Variables météo prévues à corréler avec le PV
METEO_VARS = {
    f"pred_glob_rad_h{HORIZON}_{STATION}":  "Irradiance globale prévue [W/m²]",
    f"pred_sunshine_h{HORIZON}_{STATION}":  "Ensoleillement prévu [min]",
    f"pred_relhum_h{HORIZON}_{STATION}":    "Humidité relative prévue [%]",
    f"pred_temp_h{HORIZON}_{STATION}":      "Température prévue [°C]",
}

MOIS = ["Jan", "Fév", "Mar", "Avr", "Mai", "Jun",
        "Jul", "Aoû", "Sep", "Oct", "Nov", "Déc"]

# ─────────────────────────────────────────────
# 1. CHARGEMENT
# ─────────────────────────────────────────────

print("=== Chargement CSV Oiken ===")
df_oiken = pd.read_csv(CSV,
    parse_dates=["timestamp"], dayfirst=True,
    na_values=["#N/A", "N/A", "NA", ""],
    dtype={col: float for col in PROD_COLS}
)
df_oiken["pv_total"] = df_oiken[PROD_COLS].sum(axis=1)
df_oiken = df_oiken[["timestamp", "pv_total"]].set_index("timestamp").sort_index()

# Filtrer fenêtre solaire 10h–16h et agréger par jour
df_oiken_solar = df_oiken.between_time(f"{H_START:02d}:00", f"{H_END:02d}:00")
pv_daily = df_oiken_solar["pv_total"].resample("1D").sum().rename("pv_total")
print(f"  PV journalier : {len(pv_daily)} jours")

print("=== Chargement météo ===")
df_meteo = pl.read_parquet(METEO)
if df_meteo["timestamp"].dtype == pl.Datetime("us", "UTC"):
    df_meteo = df_meteo.with_columns(
        pl.col("timestamp").dt.convert_time_zone("Europe/Zurich")
    )

# Vérifier colonnes disponibles
available = df_meteo.columns
missing = [c for c in METEO_VARS if c not in available]
if missing:
    print(f"  ⚠ Colonnes manquantes : {missing}")
    print(f"  Colonnes disponibles avec 'Sion' : {[c for c in available if 'Sion' in c and 'pred' in c][:10]}")

# Filtrer fenêtre solaire et agréger par jour
df_meteo_pd = df_meteo.select(
    ["timestamp"] + [c for c in METEO_VARS if c in available]
).to_pandas()
df_meteo_pd["timestamp"] = pd.to_datetime(df_meteo_pd["timestamp"]).dt.tz_localize(None)
df_meteo_pd = df_meteo_pd.set_index("timestamp").sort_index()

df_meteo_solar = df_meteo_pd.between_time(f"{H_START:02d}:00", f"{H_END:02d}:00")
meteo_daily = df_meteo_solar.resample("1D").mean()
print(f"  Météo journalière : {len(meteo_daily)} jours")

# ─────────────────────────────────────────────
# 2. FUSION
# ─────────────────────────────────────────────

df = pd.concat([pv_daily, meteo_daily], axis=1).dropna()
df["month"] = df.index.month
df["year"]  = df.index.year
print(f"  Dataset fusionné : {len(df)} jours valides")

# ─────────────────────────────────────────────
# 3. FIGURES
# ─────────────────────────────────────────────

for meteo_col, meteo_label in METEO_VARS.items():
    if meteo_col not in df.columns:
        print(f"  Skipping {meteo_col} — colonne absente")
        continue

    fig, axes = plt.subplots(3, 4, figsize=(20, 14))
    fig.patch.set_facecolor("#1a1a2e")
    fig.suptitle(
        f"Corrélation Production PV vs {meteo_label}\n"
        f"(Station {STATION}, horizon {HORIZON}h, fenêtre {H_START}h–{H_END}h)",
        color="white", fontsize=14, y=1.01
    )

    for m in range(1, 13):
        ax = axes[(m - 1) // 4][(m - 1) % 4]
        ax.set_facecolor("#16213e")
        ax.tick_params(colors="#e0e0e0", labelsize=7)
        ax.spines[:].set_color("#444")

        subset = df[df["month"] == m]

        if len(subset) < 5:
            ax.text(0.5, 0.5, "Données\ninsuffisantes",
                    ha="center", va="center", color="#888", transform=ax.transAxes)
            ax.set_title(MOIS[m - 1], color="white", fontsize=9)
            continue

        x = subset[meteo_col].values
        y = subset["pv_total"].values

        # Scatter coloré par année
        years = subset["year"].values
        sc = ax.scatter(x, y, c=years, cmap="plasma", alpha=0.6, s=12)

        # Droite de tendance
        slope, intercept, r, p, _ = stats.linregress(x, y)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_line, slope * x_line + intercept,
                color="white", lw=1.2, alpha=0.8)

        # Coefficient r
        color_r = "#00ff88" if abs(r) > 0.5 else "#ffaa00" if abs(r) > 0.3 else "#ff6b6b"
        ax.text(0.05, 0.92, f"r = {r:.2f}", transform=ax.transAxes,
                color=color_r, fontsize=8, fontweight="bold")
        ax.text(0.05, 0.82, f"n = {len(subset)}", transform=ax.transAxes,
                color="#aaaaaa", fontsize=7)

        ax.set_title(MOIS[m - 1], color="white", fontsize=9, pad=4)
        ax.set_xlabel(meteo_label.split("[")[0].strip(), color="#aaaaaa", fontsize=7)
        ax.set_ylabel("PV total [kWh]", color="#aaaaaa", fontsize=7)

    plt.tight_layout(pad=2.0)

    # Nom de fichier sécurisé
    safe_name = meteo_col.replace("/", "-").replace(" ", "_")
    out_path = OUT / f"corr_pv_{safe_name}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close()
    print(f"  ✓ {out_path.name}")

print("\n✓ Terminé — figures sauvegardées dans DATA/processed/")