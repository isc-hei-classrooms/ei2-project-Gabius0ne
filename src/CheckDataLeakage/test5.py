import polars as pl

path = r"C:\Users\gab1a\OneDrive\Documents\energyinfo2\DATA\processed"

X = pl.read_parquet(f"{path}\\X_features_v10.parquet")
Y = pl.read_parquet(f"{path}\\Y_target_v10.parquet")

print("=== X (features) — 3 premières lignes, 10 premières colonnes ===")
print(X.select(X.columns[-10:]).head(3))

print(f"\n=== X shape : {X.shape} ===")

print("\n=== Y (cible) — 3 premières lignes ===")
print(Y.head(3))