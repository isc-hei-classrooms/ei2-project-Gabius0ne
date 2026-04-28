
import pandas as pd
from pathlib import Path

HERE      = Path(__file__).resolve().parent
PRED_FILE = HERE / "predictions_test.parquet"
BASE_FILE = HERE / "B_baseline_GOLDEN.parquet"
REAL_FILE = HERE / "Y_target_GOLDEN.parquet"
OUTPUT    = HERE / "forecast_dataGOLDEN.INTRA.csv"

# ── Chargement ──
preds = pd.read_parquet(PRED_FILE).set_index('date')
base  = pd.read_parquet(BASE_FILE).set_index('date')
real  = pd.read_parquet(REAL_FILE).set_index('date')

# ── Dates communes ──
common = sorted(set(preds.index) & set(base.index) & set(real.index))
print(f"Dates communes: {len(common)}  ({common[0]} → {common[-1]})")

# ── Construction du CSV ──
rows = []
for d in common:
    row = {'date': str(d)}
    for i in range(96):
        t = f'{i:03d}'
        row[f'l{t}'] = round(float(real.loc[d, f'load_t{t}']), 4)
        row[f'p{t}'] = round(float(preds.loc[d, f'pred_t{t}']), 4)
        row[f'b{t}'] = round(float(base.loc[d, f'baseline_t{t}']), 4)
    rows.append(row)

df = pd.DataFrame(rows)
df.to_csv(OUTPUT, index=False)
print(f"Export: {OUTPUT}  ({len(df)} jours, {df.shape[1]} colonnes, {Path(OUTPUT).stat().st_size/1024:.0f} KB)")