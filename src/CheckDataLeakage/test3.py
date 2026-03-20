import polars as pl
from pathlib import Path

BASE = Path(r"C:\Users\gab1a\OneDrive\Documents\energyinfo2\DATA\processed")
X = pl.read_parquet(BASE / "X_features_v5.parquet")

print(f"Shape : {X.shape}")
print(f"\n{'='*60}")

# Groupes de features à inspecter
groups = {
    "Load historique J-1"    : [c for c in X.columns if c.startswith("load_jm1")][:5],
    "Load historique J-7"    : [c for c in X.columns if c.startswith("load_jm7")][:5],
    "Production solaire"     : [c for c in X.columns if c.startswith("solar")][:5],
    "Météo réelle J-1"       : [c for c in X.columns if c.startswith("rmet_jm1")][:5],
    "Météo réelle J matin"   : [c for c in X.columns if c.startswith("rmet_jmorn")][:5],
    "Prévisions météo 24h"   : [c for c in X.columns if c.startswith("pred_temp_Sion")][:5],
    "PV prévu J+1"           : [c for c in X.columns if c.startswith("pred_pv_MW")][:5],
    "Ramadan hour"           : [c for c in X.columns if c.startswith("is_ramadan")][:5],
    "Calendaire"             : ["dayofweek", "month", "is_weekend", "is_holiday", "is_ramadan"],
}

for group_name, cols in groups.items():
    available = [c for c in cols if c in X.columns]
    if not available:
        print(f"\n⚠ {group_name} — AUCUNE COLONNE TROUVÉE")
        continue
    print(f"\n{group_name} ({len(available)} cols affichées) :")
    subset = X.select(available).head(20)
    null_pct = [f"{X[c].null_count()/len(X)*100:.1f}%" for c in available]
    print(f"  Taux nulls : {dict(zip(available, null_pct))}")
    print(subset)