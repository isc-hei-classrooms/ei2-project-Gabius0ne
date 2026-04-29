"""
Génère le CSV pour le viewer en concaténant :
  - load réel (Y_target)            → colonnes l000..l095 (normalisé)
  - prédictions ML (predictions)    → colonnes p000..p095 (normalisé)
  - baseline Oiken (B_baseline)     → colonnes b000..b095 (normalisé)
  - irradiance moyenne 3 stations   → colonnes i000..i095 (W/m²)

Stations utilisées pour l'irradiance : Sion + Visp + Montana (zone Oiken).
Source : meteo_multistationGOLDEN.parquet, colonne glob_rad_<station>, pas 10 min UTC,
rééchantillonné en pas 15 min par moyenne.
"""
import pandas as pd
from pathlib import Path

HERE      = Path(__file__).resolve().parent
PRED_FILE = HERE / "predictions_test_intraGOLDEN.parquet"
BASE_FILE = HERE / "B_baseline_GOLDEN.parquet"
REAL_FILE = HERE / "Y_target_GOLDEN.parquet"
METEO_FILE = HERE / "meteo_multistationGOLDEN.parquet"
OUTPUT    = HERE / "IntradayGOLDEN.csv"

STATIONS = ['Sion', 'Visp', 'Montana']

# ── Chargement load/preds/baseline ──
preds = pd.read_parquet(PRED_FILE).set_index('date')
base  = pd.read_parquet(BASE_FILE).set_index('date')
real  = pd.read_parquet(REAL_FILE).set_index('date')

# ── Chargement irradiance ──
print("Chargement météo...")
rad_cols = [f'glob_rad_{s}' for s in STATIONS]
meteo = pd.read_parquet(METEO_FILE, columns=['timestamp'] + rad_cols)
meteo = meteo.set_index('timestamp')

# Moyenne sur les 3 stations (skipna pour robustesse aux trous d'une station)
meteo['glob_rad_avg'] = meteo[rad_cols].mean(axis=1, skipna=True)

# Rééchantillonnage 10 min → 15 min (moyenne sur fenêtre)
# label='left' : la valeur 00:15 contient la moyenne de [00:15, 00:30[
rad_15 = meteo['glob_rad_avg'].resample('15min', label='left').mean()

# Conversion UTC → Europe/Zurich pour aligner avec les dates Oiken (locales)
rad_15.index = rad_15.index.tz_convert('Europe/Zurich').tz_localize(None)

# Indexer par (date, slot 0..95)
rad_15_df = rad_15.to_frame(name='glob_rad')
rad_15_df['date'] = rad_15_df.index.date
rad_15_df['slot'] = (rad_15_df.index.hour * 4 + rad_15_df.index.minute // 15)
rad_pivot = rad_15_df.pivot_table(index='date', columns='slot', values='glob_rad', aggfunc='mean')
print(f"Irradiance pivotée : {rad_pivot.shape}, slots {rad_pivot.columns.min()}..{rad_pivot.columns.max()}")

# ── Dates communes ──
common = sorted(set(preds.index) & set(base.index) & set(real.index))
print(f"Dates communes load/preds/baseline : {len(common)}  ({common[0]} → {common[-1]})")

# ── Construction du CSV ──
rows = []
missing_meteo = 0
for d in common:
    row = {'date': str(d)}
    # Charges et prédictions (normalisées)
    for i in range(96):
        t = f'{i:03d}'
        row[f'l{t}'] = round(float(real.loc[d, f'load_t{t}']), 4)
        row[f'p{t}'] = round(float(preds.loc[d, f'pred_t{t}']), 4)
        row[f'b{t}'] = round(float(base.loc[d, f'baseline_t{t}']), 4)

    # Irradiance moyenne 3 stations (W/m²)
    d_date = pd.to_datetime(d).date() if not isinstance(d, pd.Timestamp) else d.date()
    if d_date in rad_pivot.index:
        rad_row = rad_pivot.loc[d_date]
        for i in range(96):
            v = rad_row.get(i, None)
            row[f'i{i:03d}'] = round(float(v), 1) if pd.notna(v) else ''
    else:
        missing_meteo += 1
        for i in range(96):
            row[f'i{i:03d}'] = ''
    rows.append(row)

if missing_meteo:
    print(f"Attention : {missing_meteo} jours sans données météo (irradiance vide)")

df = pd.DataFrame(rows)
df.to_csv(OUTPUT, index=False)
size_kb = Path(OUTPUT).stat().st_size / 1024
print(f"Export : {OUTPUT}  ({len(df)} jours, {df.shape[1]} colonnes, {size_kb:.0f} KB)")