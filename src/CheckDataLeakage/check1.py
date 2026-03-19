import polars as pl

X = pl.read_parquet(r"C:\Users\gab1a\OneDrive\Documents\energyinfo2\DATA\processed\X_features.parquet")
Y = pl.read_parquet(r"C:\Users\gab1a\OneDrive\Documents\energyinfo2\DATA\processed\Y_target.parquet")

# Prendre un jour exemple — disons le 3ème jour du dataset
i = 2
target_date = Y["date"][i]
print(f"target_date (J+1 à prédire) : {target_date}")
print(f"load_jm1_mean (doit être dimanche) : {X['load_jm1_mean'][i]:.4f}")
print(f"load_jm6_mean (doit être même jour semaine passée) : {X['load_jm6_mean'][i]:.4f}")

# Vérifier contre le CSV directement
import pandas as pd
df = pd.read_csv(
    r"C:\Users\gab1a\OneDrive\Documents\energyinfo2\DATA\oiken-data.csv",
    parse_dates=["timestamp"], dayfirst=True, na_values=["#N/A"]
)
df = df.rename(columns={"standardised load [-]": "load"}).set_index("timestamp").sort_index()

# Dimanche = target_date - 2 jours
import datetime
t = datetime.date.fromisoformat(str(target_date))
dimanche = t - datetime.timedelta(days=2)
semaine_passee = t - datetime.timedelta(days=7)

print(f"\nDimanche attendu : {dimanche}")
print(f"Load moyen dimanche (CSV) : {df.loc[str(dimanche), 'load'].mean():.4f}")
print(f"\nSemaine passée attendue : {semaine_passee}")
print(f"Load moyen semaine passée (CSV) : {df.loc[str(semaine_passee), 'load'].mean():.4f}")