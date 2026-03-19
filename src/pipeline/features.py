"""
pipeline_features.py
Fusionne Oiken + MeteoSuisse et construit les features.
Résultat : DATA/processed/dataset_15min.parquet
"""

import math
import polars as pl
from pathlib import Path
from datetime import date, timedelta

CSV_PATH    = Path(r"C:\Users\gab1a\OneDrive\Documents\energyinfo2\DATA\oiken-data.csv")
METEO_PATH  = Path(r"C:\Users\gab1a\OneDrive\Documents\energyinfo2\DATA\raw\meteo_raw.parquet")
OUTPUT_PATH = Path(r"C:\Users\gab1a\OneDrive\Documents\energyinfo2\DATA\processed\dataset_15min.parquet")

METEO_COLS = [
    "timestamp",
    "temp_2m", "pressure", "glob_rad", "sunshine", "relhum_2m", "precip", "wind_speed",
    "pred_temp_ctrl", "pred_pressure_ctrl", "pred_glob_rad_ctrl",
    "pred_sunshine_ctrl", "pred_relhum_ctrl", "pred_precip_ctrl",
]

# Jours fériés valaisans
FERIES = {
    date(2022, 11,  1), date(2022, 12,  8), date(2022, 12, 25), date(2022, 12, 26),
    date(2023,  1,  1), date(2023,  1,  2), date(2023,  3, 19), date(2023,  4,  7),
    date(2023,  4, 10), date(2023,  5, 18), date(2023,  5, 29), date(2023,  6,  8),
    date(2023,  8,  1), date(2023, 11,  1), date(2023, 12,  8), date(2023, 12, 25), date(2023, 12, 26),
    date(2024,  1,  1), date(2024,  1,  2), date(2024,  3, 19), date(2024,  3, 29),
    date(2024,  4,  1), date(2024,  5,  9), date(2024,  5, 20), date(2024,  5, 30),
    date(2024,  8,  1), date(2024, 11,  1), date(2024, 12,  8), date(2024, 12, 25), date(2024, 12, 26),
    date(2025,  1,  1), date(2025,  1,  2), date(2025,  3, 19), date(2025,  4, 18),
    date(2025,  4, 21), date(2025,  5, 29), date(2025,  6,  9), date(2025,  6, 19),
    date(2025,  8,  1), date(2025, 11,  1), date(2025, 12,  8), date(2025, 12, 25), date(2025, 12, 26),
}

# Vacances scolaires valaisannes
def date_range(start, end):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)

VACANCES = set()
for debut, fin in [
    (date(2022, 10,  8), date(2022, 10, 23)),
    (date(2022, 12, 24), date(2023,  1,  8)),
    (date(2023,  2, 18), date(2023,  2, 26)),
    (date(2023,  4,  8), date(2023,  4, 23)),
    (date(2023,  7,  1), date(2023,  8, 13)),
    (date(2023, 10,  7), date(2023, 10, 22)),
    (date(2023, 12, 23), date(2024,  1,  7)),
    (date(2024,  2, 10), date(2024,  2, 18)),
    (date(2024,  3, 30), date(2024,  4, 14)),
    (date(2024,  6, 29), date(2024,  8, 11)),
    (date(2024, 10,  5), date(2024, 10, 20)),
    (date(2024, 12, 21), date(2025,  1,  5)),
    (date(2025,  3,  1), date(2025,  3,  9)),
    (date(2025,  4, 12), date(2025,  4, 27)),
    (date(2025,  6, 28), date(2025,  8, 10)),
]:
    VACANCES.update(date_range(debut, fin))

SPECIAL_DAYS = {str(d) for d in (FERIES | VACANCES)}


if __name__ == "__main__":
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Chargement Oiken
    oiken = pl.read_csv(
        CSV_PATH,
        null_values=["#N/A", "N/A", ""],
    ).with_columns(
        pl.col("timestamp")
          .str.strptime(pl.Datetime("us"), "%d/%m/%Y %H:%M", strict=False)
          .fill_null(pl.col("timestamp").str.strptime(pl.Datetime("us"), "%d.%m.%Y %H:%M", strict=False))
          .dt.replace_time_zone("UTC")
    ).rename({
        "standardised load [-]":                 "load",
        "standardised forecast load [-]":        "forecast_load",
        "central valais solar production [kWh]": "solar_central_valais",
        "sion area solar production [kWh]":      "solar_sion",
        "sierre area production [kWh]":          "solar_sierre",
        "remote solar production [kWh]":         "solar_remote",
    })

    # Chargement et resample météo → 15 min
    meteo = pl.read_parquet(METEO_PATH).select(METEO_COLS)

    real_cols = ["temp_2m", "pressure", "glob_rad", "sunshine", "relhum_2m", "precip", "wind_speed"]
    pred_cols = ["pred_temp_ctrl", "pred_pressure_ctrl", "pred_glob_rad_ctrl",
                 "pred_sunshine_ctrl", "pred_relhum_ctrl", "pred_precip_ctrl"]

    # Mesures réelles : moyenne sur 15 min
    real_15 = (
        meteo.select(["timestamp"] + real_cols).sort("timestamp")
        .group_by_dynamic("timestamp", every="15m", closed="left")
        .agg([pl.col(c).mean() for c in real_cols])
    )

    # Prévisions : interpolation linéaire sur grille 15 min
    grid = pl.DataFrame({"timestamp": pl.datetime_range(
        start=meteo["timestamp"].min(), end=meteo["timestamp"].max(),
        interval="15m", time_unit="us", time_zone="UTC", eager=True
    )})
    pred_15 = (
        grid.join(meteo.select(["timestamp"] + pred_cols), on="timestamp", how="left")
        .with_columns([pl.col(c).interpolate(method="linear") for c in pred_cols])
    )

    meteo_15 = real_15.join(pred_15, on="timestamp", how="full", coalesce=True).sort("timestamp")

    # Fusion Oiken + météo
    df = oiken.join(meteo_15, on="timestamp", how="left")

    # Features temporelles cycliques
    df = df.with_columns([
        pl.col("timestamp").dt.weekday().alias("weekday"),
        pl.col("timestamp").dt.month().alias("month"),
        pl.col("timestamp").dt.ordinal_day().alias("day_of_year"),
        pl.col("timestamp").dt.week().alias("week_of_year"),
        pl.col("timestamp").dt.date().cast(pl.Utf8).alias("date_str"),
        pl.col("timestamp").dt.hour().alias("hour"),
        pl.col("timestamp").dt.minute().alias("minute"),
    ]).with_columns([
        ((2 * math.pi * (pl.col("hour") * 4 + pl.col("minute") / 15) / 96).sin().alias("sin_time")),
        ((2 * math.pi * (pl.col("hour") * 4 + pl.col("minute") / 15) / 96).cos().alias("cos_time")),
        ((2 * math.pi * pl.col("weekday") / 7).sin().alias("sin_weekday")),
        ((2 * math.pi * pl.col("weekday") / 7).cos().alias("cos_weekday")),
        ((2 * math.pi * pl.col("day_of_year") / 365).sin().alias("sin_doy")),
        ((2 * math.pi * pl.col("day_of_year") / 365).cos().alias("cos_doy")),
        ((2 * math.pi * pl.col("week_of_year") / 52).sin().alias("sin_week")),
        ((2 * math.pi * pl.col("week_of_year") / 52).cos().alias("cos_week")),
        (pl.col("weekday") >= 5).cast(pl.Int8).alias("is_weekend"),
        pl.col("date_str").is_in(list(SPECIAL_DAYS)).cast(pl.Int8).alias("is_holiday"),
        (pl.col("load") - pl.col("forecast_load")).alias("forecast_error"),
    ]).drop(["hour", "minute", "date_str"])

    df.write_parquet(OUTPUT_PATH)
    print(f"OK — {len(df):,} lignes x {len(df.columns)} colonnes -> {OUTPUT_PATH}")