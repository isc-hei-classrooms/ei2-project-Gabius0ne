import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "DATA"

# ── Chargement forecast ──
fc = pd.read_csv(HERE / "forecast_data.csv", parse_dates=["date"])
fc = fc.set_index("date")  # index = date du jour prédit (J+1)

# ── Erreur diurne (t036–t059, 09h–14h45) ──
# t000 = 00h00, t036 = 09h00, t059 = 14h45 (pas de 15min)
# On calcule (réel - prédit) pour chaque pas, puis on moyenne sur la tranche diurne
# Erreur positive = modèle surestime le load (sous-estime le PV)
# Erreur négative = modèle sous-estime le load (surestime le PV) ← notre problème
real_cols = [f"l{i:03d}" for i in range(36, 60)]
pred_cols = [f"p{i:03d}" for i in range(36, 60)]
base_cols  = [f"b{i:03d}" for i in range(36, 60)]

fc["err_diurne"]      = (fc[real_cols].values - fc[pred_cols].values).mean(axis=1)
fc["err_diurne_base"] = (fc[real_cols].values - fc[base_cols].values).mean(axis=1)
fc["month"] = fc.index.to_period("M")

# ── RMSE journalier global (96 pas) ──
# Pour chaque jour, on calcule le RMSE sur les 96 pas de 15min
# puis on compare modèle vs baseline jour par jour
all_real = [f"l{i:03d}" for i in range(96)]
all_pred = [f"p{i:03d}" for i in range(96)]
all_base = [f"b{i:03d}" for i in range(96)]

fc["rmse_model"] = np.sqrt(((fc[all_real].values - fc[all_pred].values) ** 2).mean(axis=1))
fc["rmse_base"]  = np.sqrt(((fc[all_real].values - fc[all_base].values) ** 2).mean(axis=1))

# Delta RMSE : positif = modèle PIRE que baseline, négatif = modèle MEILLEUR
fc["delta_rmse"] = fc["rmse_model"] - fc["rmse_base"]

n_better = (fc["delta_rmse"] < 0).sum()
n_worse  = (fc["delta_rmse"] > 0).sum()
print(f"Jours modèle meilleur : {n_better} / {len(fc)} ({100*n_better/len(fc):.1f}%)")
print(f"Jours modèle pire     : {n_worse}  / {len(fc)} ({100*n_worse/len(fc):.1f}%)")
print(f"Delta RMSE médian     : {fc['delta_rmse'].median():.4f}")
print(f"Delta RMSE p95 (pire) : {fc['delta_rmse'].quantile(0.95):.4f}")

# ── Chargement météo ──
meteo = pd.read_parquet(BASE / "meteo_multistation_v5.parquet")
meteo = meteo.set_index("timestamp")
meteo.index = meteo.index.tz_convert("Europe/Zurich")  # timestamp UTC → heure locale

# On garde uniquement l'irradiance mesurée à Sion entre 09h et 14h45
# puis on moyenne par jour → un scalaire par jour représentant l'ensoleillement diurne
col_rad = "glob_rad_Sion"
if col_rad not in meteo.columns:
    candidates = [c for c in meteo.columns if "glob_rad" in c.lower() and "sion" in c.lower() and "pred" not in c.lower()]
    print("Colonnes glob_rad Sion disponibles:", candidates)
    col_rad = candidates[0]

meteo_day = meteo.between_time("09:00", "14:45")[[col_rad]]
meteo_day = meteo_day.resample("D").mean()
meteo_day.index = meteo_day.index.tz_localize(None).normalize()  # supprime timezone pour jointure
meteo_day.columns = ["glob_rad_mean"]

# ── Jointure forecast + météo sur la date ──
df = fc[["err_diurne", "err_diurne_base", "month", "rmse_model", "rmse_base", "delta_rmse"]].join(meteo_day, how="left")
df = df.dropna(subset=["glob_rad_mean"])
# Robustesse : RMSE global sans les jours extrêmes
for seuil in [0.3, 0.5, 1.0]:
    mask = df["delta_rmse"].abs() < seuil
    rmse_m = np.sqrt((df[mask]["rmse_model"]**2).mean())
    rmse_b = np.sqrt((df[mask]["rmse_base"]**2).mean())
    amelio = (rmse_b - rmse_m) / rmse_b * 100
    print(f"Seuil |delta|<{seuil} → {mask.sum()} jours | modèle {rmse_m:.4f} | baseline {rmse_b:.4f} | {amelio:+.1f}%")
print(f"\nJours avec irradiance : {len(df)}  ({df.index[0].date()} → {df.index[-1].date()})")
print(f"Corrélation erreur diurne ~ irradiance : {df['err_diurne'].corr(df['glob_rad_mean']):.3f}")

# ── Figure 1 : diagnostic biais diurne ──
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Diagnostic biais diurne (09h–15h) — v11", fontsize=13)

# Plot 1 : biais moyen par mois
# Rouge = sous-estimation load (PV sous-estimé), Vert = surestimation load
monthly = df.groupby("month")["err_diurne"].mean()
ax1 = axes[0]
ax1.bar(range(len(monthly)), monthly.values,
        color=["#d9534f" if v < 0 else "#5cb85c" for v in monthly.values])
ax1.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax1.set_xticks(range(len(monthly)))
ax1.set_xticklabels([str(p) for p in monthly.index], rotation=45, ha="right", fontsize=9)
ax1.set_ylabel("Erreur moyenne normalisée (réel − prédit)")
ax1.set_title("Biais mensuel diurne")

# Plot 2 : scatter erreur vs irradiance journalière
# Chaque point = un jour. Couleur = mois (vert=hiver, rouge=été)
# La régression linéaire montre la tendance générale
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

# ── Figure 2 : RMSE journalier modèle vs baseline ──
fig2, axes2 = plt.subplots(2, 1, figsize=(14, 8))
fig2.suptitle("RMSE journalier — modèle vs baseline", fontsize=13)

# Plot 1 : RMSE modèle et baseline sur la période test
# Permet de voir visuellement quels jours le modèle est pire
ax3 = axes2[0]
ax3.plot(df.index, df["rmse_model"], color="#378ADD", linewidth=0.8, label="Modèle", alpha=0.8)
ax3.plot(df.index, df["rmse_base"],  color="#E24B4A", linewidth=0.8, label="Baseline", alpha=0.8)
ax3.set_ylabel("RMSE journalier")
ax3.set_title("RMSE par jour")
ax3.legend(fontsize=9)

# Plot 2 : histogramme du delta RMSE (modèle - baseline)
# Barres rouges = jours où le modèle est pire que la baseline
# Barres vertes = jours où le modèle est meilleur
ax4 = axes2[1]
ax4.bar(df.index, df["delta_rmse"],
        color=["#d9534f" if v > 0 else "#5cb85c" for v in df["delta_rmse"]],
        width=1.0)
ax4.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax4.set_ylabel("Delta RMSE (modèle − baseline)\nnégatif = modèle meilleur")
ax4.set_title("Gain/perte journalier vs baseline")

plt.tight_layout()
plt.savefig(HERE / "diagnostic_rmse_journalier.png", dpi=150, bbox_inches="tight")
plt.show()
print("Sauvegardé : diagnostic_biais_diurne.png + diagnostic_rmse_journalier.png")