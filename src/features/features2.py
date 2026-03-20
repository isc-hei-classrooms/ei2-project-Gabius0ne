"""
pipeline_features_v3.py
=======================
Améliorations par rapport à v2 :
  1. Load J-2 à J-7 complets (6 jours supplémentaires vs J-1 et J-6 seulement)
  2. Mesures réelles météo J-1 complet + J matin jusqu'à 10h
     (température, irradiance, pression, humidité — 6 stations)
     IMPORTANT anti-leakage :
       - Mesures réelles uniquement pour jours PASSÉS (J-1 et J matin ≤10h)
       - Prévisions uniquement pour J+1 (jour cible)
       - Jamais de mesure réelle du jour J+1

Nomenclature (J = jour où on émet la prévision à 11h, J+1 = jour à prédire) :
  Ex: target_date = mardi (J+1)
      day_j   = lundi  (J)
      day_jm1 = dimanche (J-1) — dernier load + météo réelle complète
      day_jm2 = samedi   (J-2)
      ...
      day_jm7 = mardi semaine passée (J-6 dans le sens "même jour semaine")

Cible (y) : load J+1 (96 pas × 15min)
Baseline  : forecast_load Oiken (évaluation uniquement)

Sorties :
  DATA/processed/X_features_v3.parquet
  DATA/processed/Y_target_v3.parquet
  DATA/processed/B_baseline_v3.parquet
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

# Jours de load historique à inclure (par rapport à target_date)
# J-2 = dimanche (J-1), J-3 = samedi, ..., J-8 = mardi semaine passée
LOAD_HISTORY_DAYS = list(range(2, 9))  # [2, 3, 4, 5, 6, 7, 8] → J-1 à J-7

# Variables météo réelles disponibles dans le parquet (mesures, pas prévisions)
REAL_METEO_VARS = [
    "temp_2m",
    "glob_rad",
    "pressure",
    "relhum_2m",
]

STATIONS = [
    "Pully",
    "Sion",
    "Visp",
    "Montana",
    "Col_du_Grand_St-Bernard",
    "Les_Attelas",
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


def load_meteo(path: Path) -> tuple[pl.DataFrame, list[str], list[str]]:
    """
    Charge le parquet météo et retourne :
      - le DataFrame complet
      - pred_cols  : colonnes de prévisions (pour J+1)
      - real_cols  : colonnes de mesures réelles (pour J-1 et J matin)
    """
    df = pl.read_parquet(path).sort("timestamp")
    if df["timestamp"].dtype == pl.Datetime("us", "UTC"):
        df = df.with_columns(
            pl.col("timestamp").dt.convert_time_zone("Europe/Zurich")
        )

    all_cols = [c for c in df.columns if c != "timestamp"]

    # Prévisions : colonnes pred_* — utilisées pour J+1
    pred_cols = [c for c in all_cols if c.startswith("pred_")]

    # Mesures réelles : colonnes sans préfixe pred_ — utilisées pour J-1 et J matin
    # On garde uniquement les variables d'intérêt pour limiter le bruit
    real_cols = []
    for var in REAL_METEO_VARS:
        for station in STATIONS:
            col = f"{var}_{station}"
            if col in all_cols:
                real_cols.append(col)

    print(f"  Météo : {len(df):,} lignes | {len(pred_cols)} colonnes prévisions | {len(real_cols)} colonnes mesures réelles")
    return df, pred_cols, real_cols


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
    """Stats de base : mean, max, min, std."""
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


def real_meteo_stats(df_slice: pl.DataFrame, real_cols: list[str], prefix: str) -> dict:
    """
    Stats (mean, max, min) des mesures réelles météo sur une tranche temporelle.
    prefix : ex 'rmet_jm1' ou 'rmet_jmorn'
    ANTI-LEAKAGE : uniquement appelé sur des tranches passées (J-1 ou J matin ≤10h)
    """
    features = {}
    for col in real_cols:
        if col in df_slice.columns:
            vals = df_slice[col].drop_nulls()
            features[f"{prefix}_{col}_mean"] = float(vals.mean()) if len(vals) > 0 else None
            features[f"{prefix}_{col}_max"]  = float(vals.max())  if len(vals) > 0 else None
            features[f"{prefix}_{col}_min"]  = float(vals.min())  if len(vals) > 0 else None
    return features


# ─────────────────────────────────────────────
# 3. CONSTRUCTION FEATURES PAR JOUR
# ─────────────────────────────────────────────

def build_features(
    target_date: date,
    oiken: pl.DataFrame,
    meteo: pl.DataFrame,
    pred_cols: list[str],
    real_cols: list[str],
) -> dict | None:
    """
    Construit le vecteur de features pour prédire J+1 = target_date.
    Simule les données disponibles à 11h le jour J = target_date - 1.

    ANTI-LEAKAGE strict :
      - load historique : J-1 à J-7 (target_date - 2 à target_date - 8)
      - météo réelle    : J-1 complet + J matin jusqu'à 10h SEULEMENT
      - météo prévisions: J+1 uniquement (pred_cols)
      - jamais de données de J+1 en feature
    """
    day_j = target_date - timedelta(days=1)  # lundi = jour de la prévision

    # ── Cible : load J+1
    oiken_target = get_day_slice(oiken, target_date)
    if len(oiken_target) < 90:
        return None

    # ── Météo prévisions pour J+1 (pas de leakage — ce sont des prévisions)
    meteo_target = get_day_slice(meteo, target_date)
    if len(meteo_target) == 0:
        return None

    features = {}

    # ── Load historique J-1 à J-7 (target_date - 2 à target_date - 8)
    for delta in LOAD_HISTORY_DAYS:
        day_past = target_date - timedelta(days=delta)
        label    = f"jm{delta - 1}"   # jm1=dimanche(J-1), jm2=samedi(J-2), ..., jm7=mardi passé
        oiken_past = get_day_slice(oiken, day_past)

        if len(oiken_past) >= 90:
            features.update(series_stats(oiken_past["load"], f"load_{label}"))
            features.update(hourly_profile(oiken_past, "load", f"load_{label}"))
        else:
            for k in ["mean", "max", "min", "std"]:
                features[f"load_{label}_{k}"] = None
            for h in range(24):
                features[f"load_{label}_h{h:02d}"] = None

    # ── Production solaire : J-1 total + J matin total
    day_jm1 = target_date - timedelta(days=2)
    oiken_jm1      = get_day_slice(oiken, day_jm1)
    oiken_j_morning = get_morning_slice(oiken, day_j, until_hour=10)

    for col in PROD_COLS:
        if col in oiken_jm1.columns:
            features[f"{col}_jm1_total"] = float(oiken_jm1[col].sum())
        if col in oiken_j_morning.columns:
            features[f"{col}_j_morning_total"] = float(oiken_j_morning[col].sum())

    # ── Météo réelle J-1 complet
    # ANTI-LEAKAGE : J-1 = dimanche, connu à 11h le lundi
    meteo_jm1 = get_day_slice(meteo, day_jm1)
    features.update(real_meteo_stats(meteo_jm1, real_cols, "rmet_jm1"))

    # ── Météo réelle J matin jusqu'à 10h
    # ANTI-LEAKAGE : lundi matin jusqu'à 10h, disponible avant la soumission à 11h
    meteo_j_morning = get_morning_slice(meteo, day_j, until_hour=10)
    features.update(real_meteo_stats(meteo_j_morning, real_cols, "rmet_jmorn"))

    # ── Prévisions météo J+1 : stats par colonne (toutes stations × variables × horizons)
    # ANTI-LEAKAGE : ce sont des prévisions, pas des mesures réelles du futur
    for col in pred_cols:
        if col in meteo_target.columns:
            vals = meteo_target[col].drop_nulls()
            features[f"meteo_{col}_mean"] = float(vals.mean()) if len(vals) > 0 else None
            features[f"meteo_{col}_max"]  = float(vals.max())  if len(vals) > 0 else None
            features[f"meteo_{col}_min"]  = float(vals.min())  if len(vals) > 0 else None

    # ── Calendaire (valeurs brutes)
    doy = target_date.timetuple().tm_yday
    features["dayofweek"]  = target_date.weekday()
    features["month"]      = target_date.month
    features["is_weekend"] = int(target_date.weekday() >= 5)
    features["is_holiday"] = int(target_date in FERIES)

    # ── Calendaire (encodages cycliques)
    features["sin_dow"]   = math.sin(2 * math.pi * target_date.weekday() / 7)
    features["cos_dow"]   = math.cos(2 * math.pi * target_date.weekday() / 7)
    features["sin_month"] = math.sin(2 * math.pi * (target_date.month - 1) / 12)
    features["cos_month"] = math.cos(2 * math.pi * (target_date.month - 1) / 12)
    features["sin_doy"]   = math.sin(2 * math.pi * doy / 365)
    features["cos_doy"]   = math.cos(2 * math.pi * doy / 365)

    return {
        "features": features,
        "target":   oiken_target["load"].to_list(),
        "baseline": oiken_target["load_forecast_oiken"].to_list(),
        "date":     target_date,
    }


# ─────────────────────────────────────────────
# 4. PIPELINE PRINCIPAL
# ─────────────────────────────────────────────

def main():
    print("=== Chargement des données ===")
    oiken               = load_oiken(CSV)
    meteo, pred_cols, real_cols = load_meteo(METEO)

    # On a besoin de J-7 donc on commence 9 jours après le début du dataset
    first_ts  = oiken["timestamp"].drop_nulls()[0]
    first_day = first_ts.date() + timedelta(days=9)
    last_day  = oiken["timestamp"][-1].date() - timedelta(days=1)

    all_dates = [first_day + timedelta(days=i)
                 for i in range((last_day - first_day).days + 1)]

    print(f"\n=== Construction features : {first_day} → {last_day} ({len(all_dates)} jours) ===")

    rows_X, rows_Y, rows_B, dates_ok = [], [], [], []

    for i, target_date in enumerate(all_dates):
        if i % 100 == 0:
            print(f"  {i}/{len(all_dates)} — {target_date}")

        result = build_features(target_date, oiken, meteo, pred_cols, real_cols)
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

    X.write_parquet(OUT / "X_features_v3.parquet")
    Y.write_parquet(OUT / "Y_target_v3.parquet")
    B.write_parquet(OUT / "B_baseline_v3.parquet")

    print(f"\n✓ X_features_v3 : {X.shape[0]} jours × {X.shape[1]} colonnes")
    print(f"✓ Y_target_v3   : {Y.shape[0]} jours × {Y.shape[1]} colonnes")
    print(f"✓ B_baseline_v3 : {B.shape[0]} jours × {B.shape[1]} colonnes")
    print(f"✓ Sauvegardé dans : {OUT}")


if __name__ == "__main__":
    main()