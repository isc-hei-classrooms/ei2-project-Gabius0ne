import polars as pl
X = pl.read_parquet("DATA/processed/X_features_v14.parquet")
Y = pl.read_parquet("DATA/processed/Y_target_v14.parquet")

# 1. Ordre de grandeur de Y (devrait être autour de -3 à +3 si z-score)
y_arr = Y.drop("date").to_numpy()
print(f"Y : mean={y_arr.mean():.3f}, std={y_arr.std():.3f}, min={y_arr.min():.3f}, max={y_arr.max():.3f}")

# 2. Ordre de grandeur de pv_normalizer_90j
print(f"normalizer : mean={X['pv_normalizer_90j'].mean():.3f}, min={X['pv_normalizer_90j'].min():.3f}, max={X['pv_normalizer_90j'].max():.3f}")