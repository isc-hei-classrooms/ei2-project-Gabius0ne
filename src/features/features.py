"""
pipeline_features.py
====================
Fusion du CSV Oiken + parquet météo multi-stations.
Construction des features ML par jour, en respectant strictement
la contrainte temporelle (prévision émise à 11h le jour J pour J+1).

Nomenclature (J = jour où on émet la prévision, J+1 = jour à prédire) :
  - Load J-1 = dimanche (dernier jour complet disponible à 2h lundi matin)
  - Load J-6 = mardi semaine passée (même jour de la semaine que J+1)
  - Production solaire J-1 complet + J jusqu'à 10h
  - Prévisions météo horizons 13h–36h pour 6 stations (toutes variables)
  - Features calendaires (jour semaine, mois, weekend, férié, encodages cycliques)

Cible (y) : load J+1 (96 pas × 15min)
Baseline  : forecast_load Oiken (pour évaluation uniquement)

Sorties :
  DATA/processed/X_features.parquet
  DATA/processed/Y_target.parquet
  DATA/processed/B_baseline.parquet
"""

import math
import polars as pl
import numpy as np
from pathlib import Path
from datetime import timedelta, date

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
BASE    = Path(r"C:\Users\gab1a\OneDrive\Documents\energyinfo2\DATA")
CSV     = BASE / "oiken-data.csv"
METEO   = BASE / "meteo_multistation_v3.parquet"
OUT     = BASE / "processed"
OUT.mkdir(exist_ok=True)

PROD_COLS = [
    "solar_central_valais",
    "solar_sion",
    "solar_sierre",
    "solar_remote",
]

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

# ─────────────────────────────────────────────
# 1. CHARGEMENT
# ─────────────────────────────────────────────

def load_oiken(path: Path) -> pl.DataFrame:
    df = pl.read_csv(
        path,
        try_parse_dates=False,
        null_values=["#N/A", "N/A", "NA", ""],
        schema_overrides={
            "central valais solar production [kWh]": pl.Float64,
            "sion area solar production [kWh]":      pl.Float64,
            "sierre area production [kWh]":          pl.Float64,
            "remote solar production [kWh]":         pl.Float64,
        }
    )
    df = df.with_columns(
        pl.col("timestamp")
          .str.strptime(pl.Datetime("us"), "%d.%m.%Y %H:%M")
          .dt.replace_time_zone("Europe/Zurich", ambiguous="earliest", non_existent="null")
          .alias("timestamp")
    ).rename({
        "standardised load [-]":                "load",
        "standardised forecast load [-]":        "load_forecast_oiken",
        "central valais solar production [kWh]": "solar_central_valais",
        "sion area solar production [kWh]":      "solar_sion",
        "sierre area production [kWh]":          "solar_sierre",
        "remote solar production [kWh]":         "solar_remote",
    }).sort("timestamp")
    print(f"  Oiken CSV : {len(df):,} lignes | {df['timestamp'].drop_nulls()[0]} → {df['timestamp'][-1]}")
    return df


def load_meteo(path: Path) -> pl.DataFrame:
    df = pl.read_parquet(path).sort("timestamp")
    if df["timestamp"].dtype == pl.Datetime("us", "UTC"):
        df = df.with_columns(
            pl.col("timestamp").dt.convert_time_zone("Europe/Zurich")
        )
    meteo_cols = [c for c in df.columns if c != "timestamp"]
    print(f"  Météo parquet : {len(df):,} lignes | {len(meteo_cols)} variables météo")
    return df


# ─────────────────────────────────────────────
# 2. HELPERS
# ─────────────────────────────────────────────

def get_day_slice(df: pl.DataFrame, day: date) -> pl.DataFrame:
    """Retourne les lignes d'un jour calendaire complet."""
    start = pl.datetime(day.year, day.month, day.day, 0, 0, 0, time_zone="Europe/Zurich")
    end   = pl.datetime(day.year, day.month, day.day, 23, 59, 59, time_zone="Europe/Zurich")
    return df.filter((pl.col("timestamp") >= start) & (pl.col("timestamp") <= end))


def get_morning_slice(df: pl.DataFrame, day: date, until_hour: int = 10) -> pl.DataFrame:
    """Retourne les lignes d'un jour jusqu'à until_hour inclus."""
    start = pl.datetime(day.year, day.month, day.day, 0, 0, 0, time_zone="Europe/Zurich")
    end   = pl.datetime(day.year, day.month, day.day, until_hour, 0, 0, time_zone="Europe/Zurich")
    return df.filter((pl.col("timestamp") >= start) & (pl.col("timestamp") <= end))


def series_stats(series: pl.Series, prefix: str) -> dict:
    """Stats de base d'une série : mean, max, min, std."""
    vals = series.drop_nulls()
    if len(vals) == 0:
        return {f"{prefix}_mean": None, f"{prefix}_max": None,
                f"{prefix}_min": None, f"{prefix}_std": None}
    return {
        f"{prefix}_mean": float(vals.mean()),
        f"{prefix}_max":  float(vals.max()),
        f"{prefix}_min":  float(vals.min()),
        f"{prefix}_std":  float(vals.std()),
    }


def hourly_profile(df_day: pl.DataFrame, col: str, prefix: str) -> dict:
    """Profil horaire moyen sur 24h (24 valeurs)."""
    result = {}
    for h in range(24):
        hour_vals = df_day.filter(pl.col("timestamp").dt.hour() == h)[col].drop_nulls()
        result[f"{prefix}_h{h:02d}"] = float(hour_vals.mean()) if len(hour_vals) > 0 else None
    return result


# ─────────────────────────────────────────────
# 3. CONSTRUCTION FEATURES PAR JOUR
# ─────────────────────────────────────────────

def build_features(
    target_date: date,
    oiken: pl.DataFrame,
    meteo: pl.DataFrame,
    meteo_cols: list[str],
) -> dict | None:
    """
    Construit le vecteur de features pour prédire J+1 = target_date.
    Simule les données disponibles à 11h le jour J = target_date - 1.

    Ex: target_date = mardi → J = lundi → J-1 = dimanche, J-6 = mardi passé
    """
    day_j   = target_date - timedelta(days=1)  # J  : lundi (jour de la prévision)
    day_jm1 = target_date - timedelta(days=2)  # J-1 : dimanche (dernier load complet)
    day_jm6 = target_date - timedelta(days=7)  # J-6 : mardi semaine passée (même jour que J+1)

    # ── Load J-1 (dimanche complet)
    oiken_jm1 = get_day_slice(oiken, day_jm1)
    if len(oiken_jm1) < 90:
        return None

    # ── Load J-6 (mardi semaine passée)
    oiken_jm6 = get_day_slice(oiken, day_jm6)

    # ── Production J-1 complet + J matin jusqu'à 10h
    oiken_j_morning = get_morning_slice(oiken, day_j, until_hour=10)

    # ── Météo pour J+1 (jour cible)
    meteo_target = get_day_slice(meteo, target_date)
    if len(meteo_target) == 0:
        return None

    # ── Cible : load J+1
    oiken_target = get_day_slice(oiken, target_date)
    if len(oiken_target) < 90:
        return None

    features = {}

    # ── Load J-1 : stats + profil horaire
    features.update(series_stats(oiken_jm1["load"], "load_jm1"))
    features.update(hourly_profile(oiken_jm1, "load", "load_jm1"))

    # ── Load J-6 : stats + profil horaire
    if len(oiken_jm6) >= 90:
        features.update(series_stats(oiken_jm6["load"], "load_jm6"))
        features.update(hourly_profile(oiken_jm6, "load", "load_jm6"))
    else:
        for k in ["mean", "max", "min", "std"]:
            features[f"load_jm6_{k}"] = None
        for h in range(24):
            features[f"load_jm6_h{h:02d}"] = None

    # ── Production solaire : J-1 total + J matin total
    for col in PROD_COLS:
        if col in oiken_jm1.columns:
            features[f"{col}_jm1_total"] = float(oiken_jm1[col].sum())
        if col in oiken_j_morning.columns:
            features[f"{col}_j_morning_total"] = float(oiken_j_morning[col].sum())

    # ── Météo J+1 : stats par colonne (toutes stations × variables × horizons)
    for col in meteo_cols:
        if col in meteo_target.columns:
            vals = meteo_target[col].drop_nulls()
            features[f"meteo_{col}_mean"] = float(vals.mean()) if len(vals) > 0 else None
            features[f"meteo_{col}_max"]  = float(vals.max())  if len(vals) > 0 else None
            features[f"meteo_{col}_min"]  = float(vals.min())  if len(vals) > 0 else None

    # ── Calendaire (valeurs brutes)
    doy = target_date.timetuple().tm_yday
    features["dayofweek"]  = target_date.weekday()        # 0=lundi … 6=dimanche
    features["month"]      = target_date.month            # 1–12
    features["is_weekend"] = int(target_date.weekday() >= 5)
    features["is_holiday"] = int(target_date in FERIES)

    # ── Calendaire (encodages cycliques)
    features["sin_dow"]   = math.sin(2 * math.pi * target_date.weekday() / 7)
    features["cos_dow"]   = math.cos(2 * math.pi * target_date.weekday() / 7)
    features["sin_month"] = math.sin(2 * math.pi * (target_date.month - 1) / 12)
    features["cos_month"] = math.cos(2 * math.pi * (target_date.month - 1) / 12)
    features["sin_doy"]   = math.sin(2 * math.pi * doy / 365)
    features["cos_doy"]   = math.cos(2 * math.pi * doy / 365)

    # ── Cible et baseline (retournés séparément)
    target_vals   = oiken_target["load"].to_list()
    baseline_vals = oiken_target["load_forecast_oiken"].to_list()

    return {
        "features": features,
        "target":   target_vals,
        "baseline": baseline_vals,
        "date":     target_date,
    }


# ─────────────────────────────────────────────
# 4. PIPELINE PRINCIPAL
# ─────────────────────────────────────────────

def main():
    print("=== Chargement des données ===")
    oiken = load_oiken(CSV)
    meteo = load_meteo(METEO)

    meteo_cols = [c for c in meteo.columns if c != "timestamp" and c.startswith("pred_")]

    # On a besoin de J-6 donc on commence 8 jours après le début du dataset
    first_ts  = oiken["timestamp"].drop_nulls()[0]
    first_day = first_ts.date() + timedelta(days=8)
    last_day  = oiken["timestamp"][-1].date() - timedelta(days=1)

    all_dates = [first_day + timedelta(days=i)
                 for i in range((last_day - first_day).days + 1)]

    print(f"\n=== Construction features : {first_day} → {last_day} ({len(all_dates)} jours) ===")

    rows_X, rows_Y, rows_B, dates_ok = [], [], [], []

    for i, target_date in enumerate(all_dates):
        if i % 100 == 0:
            print(f"  {i}/{len(all_dates)} — {target_date}")

        result = build_features(target_date, oiken, meteo, meteo_cols)
        if result is None:
            continue

        rows_X.append(result["features"])
        rows_Y.append(result["target"])
        rows_B.append(result["baseline"])
        dates_ok.append(str(result["date"]))

    print(f"\n  {len(dates_ok)} jours valides sur {len(all_dates)}")

    # ── Sérialisation
    X = pl.DataFrame(rows_X).with_columns(
        pl.Series("date", dates_ok).str.strptime(pl.Date, "%Y-%m-%d")
    )
    X = X.select(["date"] + [c for c in X.columns if c != "date"])

    n_steps = len(rows_Y[0])
    Y = pl.DataFrame(
        {f"load_t{i:03d}": [row[i] if i < len(row) else None for row in rows_Y]
         for i in range(n_steps)}
    ).with_columns(pl.Series("date", dates_ok).str.strptime(pl.Date, "%Y-%m-%d"))
    Y = Y.select(["date"] + [f"load_t{i:03d}" for i in range(n_steps)])

    B = pl.DataFrame(
        {f"baseline_t{i:03d}": [row[i] if i < len(row) else None for row in rows_B]
         for i in range(n_steps)}
    ).with_columns(pl.Series("date", dates_ok).str.strptime(pl.Date, "%Y-%m-%d"))
    B = B.select(["date"] + [f"baseline_t{i:03d}" for i in range(n_steps)])

    X.write_parquet(OUT / "X_features.parquet")
    Y.write_parquet(OUT / "Y_target.parquet")
    B.write_parquet(OUT / "B_baseline.parquet")

    print(f"\n✓ X_features : {X.shape[0]} jours × {X.shape[1]} colonnes")
    print(f"✓ Y_target   : {Y.shape[0]} jours × {Y.shape[1]} colonnes")
    print(f"✓ B_baseline : {B.shape[0]} jours × {B.shape[1]} colonnes")
    print(f"✓ Sauvegardé dans : {OUT}")


if __name__ == "__main__":
    main()