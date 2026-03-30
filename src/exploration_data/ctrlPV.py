import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent.parent / "DATA"

# ── Chargement forecast ──
fc = pd.read_csv(HERE / "forecast_data.csv", parse_dates=["date"])
fc = fc.set_index("date")

# ── Erreur diurne (t036–t059, 09h–14h45) ──
real_cols = [f"l{i:03d}" for i in range(36, 60)]
pred_cols = [f"p{i:03d}" for i in range(36, 60)]

fc["err_diurne"] = (fc[real_cols].values - fc[pred_cols].values).mean(axis=1)
fc["month"] = fc.index.to_period("M")

# ── Chargement météo ──
meteo = pd.read_parquet(BASE / "meteo_multistation_v5.parquet")

# Convertir timestamp en date locale Zurich
if meteo.index.tz is not None:
    meteo.index = meteo.index.tz_convert("Europe/Zurich")
else:
    meteo.index = meteo.index.tz_localize("UTC").tz_convert("Europe/Zurich")

# Garder uniquement la mesure réelle glob_rad Sion, tranche 09h–15h
col_rad = "glob_rad_Sion"
if col_rad not in meteo.columns:
    # chercher le nom exact
    candidates = [c for c in meteo.columns if "glob_rad" in c.lower() and "sion" in c.lower() and "pred" not in c.lower()]
    print("Colonnes glob_rad Sion disponibles:", candidates)
    col_rad = candidates[0]

meteo_day = meteo.between_time("09:00", "14:45")[[col_rad]]
meteo_day = meteo_day.resample("D").mean()
meteo_day.index = meteo_day.index.tz_localize(None).normalize()
meteo_day.columns = ["glob_rad_mean"]

# ── Jointure ──
df = fc[["err_diurne", "month"]].join(meteo_day, how="left")
df = df.dropna(subset=["glob_rad_mean"])

print(f"Jours avec irradiance : {len(df)}  ({df.index[0].date()} → {df.index[-1].date()})")
print(f"Corrélation erreur ~ irradiance : {df['err_diurne'].corr(df['glob_rad_mean']):.3f}")

# ── Figure ──
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Diagnostic biais diurne (09h–15h) — v9", fontsize=13)

# Plot 1 : biais mensuel
monthly = df.groupby("month")["err_diurne"].mean()
ax1 = axes[0]
ax1.bar(range(len(monthly)), monthly.values,
        color=["#d9534f" if v < 0 else "#5cb85c" for v in monthly.values])
ax1.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax1.set_xticks(range(len(monthly)))
ax1.set_xticklabels([str(p) for p in monthly.index], rotation=45, ha="right", fontsize=9)
ax1.set_ylabel("Erreur moyenne normalisée (réel − prédit)")
ax1.set_title("Biais mensuel diurne")

# Plot 2 : scatter erreur vs irradiance
ax2 = axes[1]
sc = ax2.scatter(df["glob_rad_mean"], df["err_diurne"],
                 alpha=0.5, s=20, c=df.index.month, cmap="RdYlGn_r")
z = np.polyfit(df["glob_rad_mean"], df["err_diurne"], 1)
x_line = np.linspace(df["glob_rad_mean"].min(), df["glob_rad_mean"].max(), 100)
ax2.plot(x_line, np.poly1d(z)(x_line), "r--", linewidth=1.5,
         label=f"Régression (r={df['err_diurne'].corr(df['glob_rad_mean']):.2f})")
ax2.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax2.set_xlabel("Irradiance moyenne Sion 09h–15h (W/m²)")
ax2.set_ylabel("Erreur moyenne normalisée")
ax2.set_title("Erreur vs irradiance journalière")
ax2.legend(fontsize=9)
plt.colorbar(sc, ax=ax2, label="Mois")

plt.tight_layout()
plt.savefig(HERE / "diagnostic_biais_diurne.png", dpi=150, bbox_inches="tight")
plt.show()
print("Sauvegardé : diagnostic_biais_diurne.png")