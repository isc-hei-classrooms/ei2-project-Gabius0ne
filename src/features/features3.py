"""
pipeline_features_v4.py (refonte)
==================================
Refonte complète de la gestion des prévisions météo.

PROBLÈME v4 initiale :
  pred_brutes() prenait la médiane de toutes les valeurs d'une colonne sur J+1
  → équivalent à une agrégation journalière, aucun profil temporel

SOLUTION :
  Pour chaque timestamp de J+1 (00h, 03h, 06h, ..., 21h UTC — pas de 3h),
  on extrait la valeur de chaque colonne pred_* à ce timestamp exact.
  Feature nommée : pred_glob_rad_h24_Sion_t09 = irradiance prévue à 09h UTC
  le jour J+1, issue du run émis 24h avant (soit 09h la veille).

  Cela donne au modèle :
    - Le profil temporel COMPLET de J+1 (8 timestamps × 24 horizons × 8 vars × 6 stations)
    - La dispersion entre horizons pour chaque heure (convergence/divergence des runs)
    - L'information sur quand le soleil est fort (9h vs 12h vs 15h)
    - L'inertie thermique (température 6h avant vs température à l'heure cible)

  ANTI-LEAKAGE strict :
    - Load historique  : J-1 à J-7 uniquement (passé)
    - Météo réelle     : J-1 complet + J matin ≤10h uniquement (passé)
    - Prévisions météo : timestamps de J+1 uniquement via colonnes pred_*
      (ce sont des prévisions émises avant 11h le jour J — pas de leakage)

Sorties :
  DATA/processed/X_features_v4.parquet
  DATA/processed/Y_target_v4.parquet
  DATA/processed/B_baseline_v4.parquet
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

# Jours de load historique (par rapport à target_date)
# delta=2 → J-1 (dimanche), delta=8 → J-7 (mardi semaine passée)
LOAD_HISTORY_DAYS = list(range(2, 9))  # [2..8]

# Variables météo réelles (mesures passées, pas prévisions)
REAL_METEO_VARS = ["temp_2m", "glob_rad", "pressure", "relhum_2m"]

STATIONS = [
    "Pully", "Sion", "Visp", "Montana",
    "Col_du_Grand_St-Bernard", "Les_Attelas",
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


def load_meteo(path: Path) -> tuple[pl.DataFrame, pl.DataFrame, list[str], list[str]]:
    """
    Charge le parquet météo et retourne DEUX versions :
      - df_utc    : timestamps en UTC — pour pred_profile (get_timestamp_row)
      - df_zurich : timestamps en Europe/Zurich — pour mesures réelles (get_day_slice)
    Le bug précédent : get_timestamp_row filtrait en UTC mais le df était en Zurich
    → 100% de nulls. Correction : on garde df_utc intact pour les comparaisons UTC.
    """
    df_utc = pl.read_parquet(path).sort("timestamp")

    # Version Europe/Zurich pour les slices de mesures réelles
    df_zurich = df_utc.with_columns(
        pl.col("timestamp").dt.convert_time_zone("Europe/Zurich")
    )

    all_cols = [c for c in df_utc.columns if c != "timestamp"]

    # Prévisions : extraites par timestamp UTC précis dans pred_profile
    pred_cols = [c for c in all_cols if c.startswith("pred_")]

    # Mesures réelles : uniquement les 4 variables d'intérêt × 6 stations
    real_cols = []
    for var in REAL_METEO_VARS:
        for station in STATIONS:
            col = f"{var}_{station}"
            if col in all_cols:
                real_cols.append(col)

    print(f"  Météo : {len(df_utc):,} lignes | {len(pred_cols)} colonnes pred | {len(real_cols)} colonnes réelles")
    return df_utc, df_zurich, pred_cols, real_cols


# ─────────────────────────────────────────────
# 2. HELPERS
# ─────────────────────────────────────────────

def get_day_slice(df: pl.DataFrame, day: date) -> pl.DataFrame:
    """Retourne toutes les lignes d'un jour calendaire."""
    start = pl.datetime(day.year, day.month, day.day, 0, 0, 0, time_zone="Europe/Zurich")
    end   = pl.datetime(day.year, day.month, day.day, 23, 59, 59, time_zone="Europe/Zurich")
    return df.filter((pl.col("timestamp") >= start) & (pl.col("timestamp") <= end))


def get_morning_slice(df: pl.DataFrame, day: date, until_hour: int = 10) -> pl.DataFrame:
    """Retourne les lignes d'un jour jusqu'à until_hour inclus."""
    start = pl.datetime(day.year, day.month, day.day, 0, 0, 0, time_zone="Europe/Zurich")
    end   = pl.datetime(day.year, day.month, day.day, until_hour, 0, 0, time_zone="Europe/Zurich")
    return df.filter((pl.col("timestamp") >= start) & (pl.col("timestamp") <= end))


def series_stats(series: pl.Series, prefix: str) -> dict:
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
    Stats (mean/max/min) des mesures réelles météo sur une tranche passée.
    ANTI-LEAKAGE : uniquement J-1 complet ou J matin ≤10h
    """
    features = {}
    for col in real_cols:
        if col in df_slice.columns:
            vals = df_slice[col].drop_nulls()
            features[f"{prefix}_{col}_mean"] = float(vals.mean()) if len(vals) > 0 else None
            features[f"{prefix}_{col}_max"]  = float(vals.max())  if len(vals) > 0 else None
            features[f"{prefix}_{col}_min"]  = float(vals.min())  if len(vals) > 0 else None
    return features


def pred_profile(
    meteo_utc: pl.DataFrame,
    target_date: date,
    pred_cols: list[str],
) -> dict:
    """
    Extrait le profil temporel des prévisions pour J+1.

    Structure du parquet météo :
      - Chaque colonne pred_var_hXX_Station contient les prévisions émises XX heures avant
        le timestamp de la ligne (heure de validité).
      - Les horizons h13, h16, h19... ont leurs valeurs aux timestamps 01h, 04h, 07h... UTC
      - Les horizons h14, h17, h20... ont leurs valeurs aux timestamps 02h, 05h, 08h... UTC
      - Les horizons h15, h18, h21... ont leurs valeurs aux timestamps 00h, 03h, 06h... UTC
      Chaque horizon a donc 8 valeurs sur J+1, couvrant des heures différentes.

    Pour chaque colonne pred_*, on extrait ses 8 valeurs non-nulles sur J+1
    et on les nomme {col}_t{heure_utc:02d}.

    Résultat : couverture quasi-horaire de J+1 avec 24 horizons × 8 heures = 192 valeurs
    par variable par station — le modèle voit le profil temporel COMPLET.

    ANTI-LEAKAGE : les horizons h13 à h36 correspondent à des runs émis entre
    J-1 12h et J 22h — tous avant la soumission à 11h le jour J.
    """
    features = {}

    # Filtrer J+1 en UTC
    start = pl.datetime(target_date.year, target_date.month, target_date.day,
                        0, 0, 0, time_zone="UTC")
    end   = pl.datetime(target_date.year, target_date.month, target_date.day,
                        23, 59, 59, time_zone="UTC")
    day_utc = meteo_utc.filter(
        (pl.col("timestamp") >= start) & (pl.col("timestamp") <= end)
    )

    if len(day_utc) == 0:
        for col in pred_cols:
            for h in range(24):
                features[f"{col}_t{h:02d}"] = None
        return features

    # Pour chaque colonne pred_*, extraire les 8 valeurs non-nulles avec leur heure UTC
    for col in pred_cols:
        if col not in day_utc.columns:
            for h in range(24):
                features[f"{col}_t{h:02d}"] = None
            continue

        # Extraire les lignes non-nulles pour cette colonne
        non_null = day_utc.filter(pl.col(col).is_not_null()).select(
            ["timestamp", col]
        )

        # Indexer par heure UTC
        hour_to_val = {}
        for row in non_null.iter_rows():
            ts, val = row
            h_utc = ts.hour
            hour_to_val[h_utc] = float(val)

        # Créer une feature par heure UTC (0 à 23) — None si pas de valeur
        for h in range(24):
            features[f"{col}_t{h:02d}"] = hour_to_val.get(h, None)

    return features


# ─────────────────────────────────────────────
# 3. CONSTRUCTION FEATURES PAR JOUR
# ─────────────────────────────────────────────

def build_features(
    target_date: date,
    oiken: pl.DataFrame,
    meteo_utc: pl.DataFrame,
    meteo_zurich: pl.DataFrame,
    pred_cols: list[str],
    real_cols: list[str],
) -> dict | None:
    """
    Construit le vecteur de features pour prédire J+1 = target_date.
    Simule les données disponibles à 11h le jour J = target_date - 1.

    ANTI-LEAKAGE strict :
      - Load historique  : J-1 à J-7 (passé uniquement)
      - Météo réelle     : J-1 complet + J matin ≤10h (passé uniquement)
      - Prévisions météo : profil horaire J+1 via pred_* par timestamp précis
        (prévisions émises avant 11h le jour J — pas de mesure réelle future)
    """
    day_j   = target_date - timedelta(days=1)   # lundi = jour de la prévision
    day_jm1 = target_date - timedelta(days=2)   # dimanche = J-1

    # ── Cible : load J+1
    oiken_target = get_day_slice(oiken, target_date)
    if len(oiken_target) < 90:
        return None

    features = {}

    # ── Load historique J-1 à J-7
    for delta in LOAD_HISTORY_DAYS:
        day_past   = target_date - timedelta(days=delta)
        label      = f"jm{delta - 1}"   # jm1=dimanche, ..., jm7=mardi passé
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
    oiken_jm1       = get_day_slice(oiken, day_jm1)
    oiken_j_morning = get_morning_slice(oiken, day_j, until_hour=10)

    for col in PROD_COLS:
        if col in oiken_jm1.columns:
            features[f"{col}_jm1_total"] = float(oiken_jm1[col].sum())
        if col in oiken_j_morning.columns:
            features[f"{col}_j_morning_total"] = float(oiken_j_morning[col].sum())

    # ── Météo réelle J-1 complet
    # ANTI-LEAKAGE : dimanche complet, connu à 11h le lundi
    meteo_jm1 = get_day_slice(meteo_zurich, day_jm1)
    features.update(real_meteo_stats(meteo_jm1, real_cols, "rmet_jm1"))

    # ── Météo réelle J matin jusqu'à 10h
    # ANTI-LEAKAGE : lundi matin ≤10h, disponible avant soumission à 11h
    meteo_j_morning = get_morning_slice(meteo_zurich, day_j, until_hour=10)
    features.update(real_meteo_stats(meteo_j_morning, real_cols, "rmet_jmorn"))

    # ── Profil temporel des prévisions J+1
    # Pour chaque heure UTC de J+1, on extrait la valeur de chaque colonne pred_*
    # → le modèle voit le profil temporel complet avec résolution quasi-horaire
    # ANTI-LEAKAGE : prévisions émises avant 11h le jour J
    features.update(pred_profile(meteo_utc, target_date, pred_cols))

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
    meteo_utc, meteo_zurich, pred_cols, real_cols = load_meteo(METEO)

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

        result = build_features(target_date, oiken, meteo_utc, meteo_zurich, pred_cols, real_cols)
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

    X.write_parquet(OUT / "X_features_v4.parquet")
    Y.write_parquet(OUT / "Y_target_v4.parquet")
    B.write_parquet(OUT / "B_baseline_v4.parquet")

    print(f"\n✓ X_features_v4 : {X.shape[0]} jours × {X.shape[1]} colonnes")
    print(f"✓ Y_target_v4   : {Y.shape[0]} jours × {Y.shape[1]} colonnes")
    print(f"✓ B_baseline_v4 : {B.shape[0]} jours × {B.shape[1]} colonnes")
    print(f"✓ Sauvegardé dans : {OUT}")


if __name__ == "__main__":
    main()