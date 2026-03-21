"""
pipeline_features_v6.py
=======================
Changements par rapport à v5 :

1. CORRECTION LEAKAGE PRÉVISIONS MÉTÉO
   v5 utilisait h13/h14/h15 pour TOUTES les heures de J+1.
   Or h13 pour J+1 01h UTC est émis à J 12h UTC = APRÈS la soumission à 11h.
   
   v6 corrige: pour J+1 heure H UTC, on utilise l'horizon minimum tel que
   l'émission soit ≤ J 10h UTC (= 11h Zurich hiver, pire cas):
     horizon_min = H + 14
   On prend le plus petit horizon disponible ≥ horizon_min qui matche le
   pattern de l'heure (H%3). Résultat: h15 à h36 selon l'heure.
   
   Pour J+1 22h et 23h UTC: h36/h37 nécessaires mais h37 n'existe pas.
   Fallback au max disponible (h34/h35, émis J 12h UTC — 2h de retard
   sur des heures nocturnes, impact négligeable).

2. AJOUT PRÉVISIONS JOUR J (inertie thermique)
   Prévisions météo pour le jour de soumission J, 24 valeurs par variable
   par station. Utilise les horizons les plus courts disponibles (h1–h14)
   tels que l'émission soit ≤ J 10h UTC.
   
   Gain vs v6 initiale (h13/h14/h15 pour tout): jusqu'à 12h de fraîcheur
   en plus pour les heures matinales de J. Ex: J 01h UTC utilise h1
   (émis J 00h) au lieu de h13 (émis J-1 12h).
   
   Motivation physique: l'inertie thermique des bâtiments fait que la
   météo d'aujourd'hui impacte la charge de demain (chauffage/clim).

3. ENCODAGE CYCLIQUE WIND_DIR (sin/cos)
   wind_dir en degrés bruts (0-360) est cyclique: 359° ≈ 1°, mais
   numériquement très éloignés. LightGBM/XGBoost gaspillent des splits
   pour capturer cette circularité, gonflant artificiellement l'importance.
   
   v6 remplace chaque feature wind_dir par deux composantes:
     wind_dir_sin = sin(dir × π/180)
     wind_dir_cos = cos(dir × π/180)
   Applicable aux prévisions J+1 ET J.

4. SUPPRESSION solar_remote_j_morning_total
   Les données solar_remote arrivent à 2h du matin (J-1 complet),
   pas en temps réel. À 11h le jour J, la production remote du matin
   de J n'est pas disponible → leakage. Seules les 3 autres sources
   (central_valais, sion, sierre) sont en live et conservées.

ANTI-LEAKAGE v6 (renforcé) :
  - Load historique  : J-1 à J-7 (delta 2 à 8 vs target_date)
  - Production PV    : J-1 complet (4 sources) + J matin ≤10h (3 sources live, SANS remote)
  - Météo réelle     : J-1 complet + J matin ≤10h (MeteoSuisse temps réel)
  - Prévisions J+1   : horizons h15–h36 (émis ≤ J 10h UTC)
  - Prévisions J     : h1–h14 optimaux (émis ≤ J 10h UTC)
  - Calendaire       : info connue à l'avance (is_ramadan_hour, jours fériés)

Sorties :
  DATA/processed/X_features_v6.parquet
  DATA/processed/Y_target_v6.parquet
  DATA/processed/B_baseline_v6.parquet
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
METEO   = BASE / "meteo_multistation_v4.parquet"
OUT     = BASE / "processed"
OUT.mkdir(exist_ok=True)

PROD_COLS = [
    "solar_central_valais",
    "solar_sion",
    "solar_sierre",
    "solar_remote",
]

# Sources PV disponibles en live (pour J matin)
# solar_remote exclue: données reçues à 2h du matin seulement
PROD_COLS_LIVE = [
    "solar_central_valais",
    "solar_sion",
    "solar_sierre",
]

LOAD_HISTORY_DAYS = list(range(2, 9))  # J-1 à J-7 (delta=2 à 8 par rapport à target_date)

REAL_METEO_VARS = ["temp_2m", "glob_rad", "pressure", "relhum_2m"]

STATIONS = [
    "Pully", "Sion", "Visp", "Montana",
    "Col_du_Grand_St-Bernard", "Les_Attelas",
]

PRED_VARS = ["temp", "glob_rad", "pressure", "relhum", "precip", "sunshine", "wind_speed", "wind_dir"]

# Variables avec encodage cyclique sin/cos (au lieu de valeur brute)
CYCLIC_PRED_VARS = {"wind_dir"}

# Tous les horizons disponibles dans le parquet (v4: h1–h36), par pattern H%3
# H%3==0 → h3, h6, h9, h12, h15, h18, h21, h24, h27, h30, h33, h36
# H%3==1 → h1, h4, h7, h10, h13, h16, h19, h22, h25, h28, h31, h34
# H%3==2 → h2, h5, h8, h11, h14, h17, h20, h23, h26, h29, h32, h35
HORIZONS_BY_MOD = {
    0: list(range(3, 37, 3)),   # h3, h6, ..., h36
    1: list(range(1, 35, 3)),   # h1, h4, ..., h34
    2: list(range(2, 36, 3)),   # h2, h5, ..., h35
}

# Facteur de conversion irradiance → production PV territoire Oiken
PV_SURFACE_M2  = 540_000   # m²
PV_PERF_RATIO  = 0.80      # performance ratio

# Dates du Ramadan par année
RAMADAN_DATES = {
    2022: (date(2022, 4,  2), date(2022, 5,  1)),
    2023: (date(2023, 3, 23), date(2023, 4, 20)),
    2024: (date(2024, 3, 11), date(2024, 4,  9)),
    2025: (date(2025, 3,  1), date(2025, 3, 29)),
    2026: (date(2026, 2, 18), date(2026, 3, 19)),
}

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
# HELPERS CALENDAIRES
# ─────────────────────────────────────────────

def is_ramadan(d: date) -> bool:
    for start, end in RAMADAN_DATES.values():
        if start <= d <= end:
            return True
    return False


def ramadan_night_hours(d: date) -> list[int]:
    if not is_ramadan(d):
        return []
    return list(range(0, 6)) + list(range(20, 24))


# ─────────────────────────────────────────────
# HORIZON MAPPING
# ─────────────────────────────────────────────

def get_correct_horizon_jp1(h_utc: int) -> int:
    """
    Pour J+1 heure h_utc, retourne l'horizon correct (le plus petit
    disponible tel que l'émission soit ≤ J 10h UTC).
    
    Règle: horizon_min = h_utc + 14
    On prend le plus petit horizon du pattern H%3 qui est >= horizon_min.
    Si aucun n'est disponible, on prend le max du pattern (fallback).
    """
    mod = h_utc % 3
    available = HORIZONS_BY_MOD[mod]
    min_needed = h_utc + 14
    
    valid = [h for h in available if h >= min_needed]
    if valid:
        return min(valid)
    else:
        # Fallback: prendre le plus grand horizon disponible
        # (légèrement après 10h UTC mais meilleur que rien — heures nocturnes)
        return max(available)


def get_correct_horizon_j(h_utc: int) -> int:
    """
    Pour le jour J heure h_utc, retourne l'horizon le plus court (= prévision
    la plus récente) tel que l'émission soit ≤ J 10h UTC.
    
    Règle: émission = h_utc - horizon ≤ 10  →  horizon ≥ max(h_utc - 10, 1)
    
    Avec h1–h36 disponibles, on gagne jusqu'à 12h de fraîcheur par rapport
    à l'ancienne version qui utilisait h13/h14/h15 pour tout.
    
    Exemples:
      J 01h UTC → h1 (émis J 00h)    au lieu de h13 (émis J-1 12h)
      J 10h UTC → h1 (émis J 09h)    au lieu de h13 (émis J-1 21h)
      J 16h UTC → h7 (émis J 09h)    au lieu de h13 (émis J 03h)
      J 22h UTC → h13 (émis J 09h)   inchangé
    """
    mod = h_utc % 3
    available = HORIZONS_BY_MOD[mod]
    min_needed = max(h_utc - 10, 1)
    
    valid = [h for h in available if h >= min_needed]
    if valid:
        return min(valid)
    else:
        return max(available)


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
    print(f"  Oiken : {len(df):,} lignes | {df['timestamp'].drop_nulls()[0]} → {df['timestamp'][-1]}")
    return df


def load_meteo(path: Path) -> tuple[pl.DataFrame, pl.DataFrame, list[str]]:
    df_utc = pl.read_parquet(path).sort("timestamp")
    df_zurich = df_utc.with_columns(
        pl.col("timestamp").dt.convert_time_zone("Europe/Zurich")
    )
    all_cols  = [c for c in df_utc.columns if c != "timestamp"]
    real_cols = [
        f"{var}_{station}"
        for var in REAL_METEO_VARS
        for station in STATIONS
        if f"{var}_{station}" in all_cols
    ]
    print(f"  Météo : {len(df_utc):,} lignes | {len(real_cols)} colonnes réelles")
    return df_utc, df_zurich, real_cols


# ─────────────────────────────────────────────
# 2. HELPERS
# ─────────────────────────────────────────────

def get_day_slice(df: pl.DataFrame, day: date) -> pl.DataFrame:
    start = pl.datetime(day.year, day.month, day.day, 0, 0, 0, time_zone="Europe/Zurich")
    end   = pl.datetime(day.year, day.month, day.day, 23, 59, 59, time_zone="Europe/Zurich")
    return df.filter((pl.col("timestamp") >= start) & (pl.col("timestamp") <= end))


def get_morning_slice(df: pl.DataFrame, day: date, until_hour: int = 10) -> pl.DataFrame:
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
    result = {}
    for h in range(24):
        hour_vals = df_day.filter(pl.col("timestamp").dt.hour() == h)[col].drop_nulls()
        result[f"{prefix}_h{h:02d}"] = float(hour_vals.mean()) if len(hour_vals) > 0 else None
    return result


def real_meteo_stats(df_slice: pl.DataFrame, real_cols: list[str], prefix: str) -> dict:
    features = {}
    for col in real_cols:
        if col in df_slice.columns:
            vals = df_slice[col].drop_nulls()
            if len(vals) > 0:
                features[f"{prefix}_{col}_mean"] = float(vals.mean())
                features[f"{prefix}_{col}_max"]  = float(vals.max())
                features[f"{prefix}_{col}_min"]  = float(vals.min())
            else:
                features[f"{prefix}_{col}_mean"] = None
                features[f"{prefix}_{col}_max"]  = None
                features[f"{prefix}_{col}_min"]  = None
    return features


def extract_pred_vector(
    meteo_utc: pl.DataFrame,
    target_day: date,
    horizon_func,
    prefix: str,
) -> dict:
    """
    Construit le vecteur de prévisions météo heure par heure pour un jour donné.
    
    Paramètres:
      target_day   : le jour pour lequel on extrait les prévisions
      horizon_func : fonction h_utc → horizon correct (différent pour J vs J+1)
      prefix       : "pred" pour J+1, "predJ" pour J
    
    Pour wind_dir: encode en sin/cos au lieu de valeur brute.
    
    Retourne un dict de features.
    """
    features = {}

    # Filtrer le jour cible en UTC
    start = pl.datetime(target_day.year, target_day.month, target_day.day,
                        0, 0, 0, time_zone="UTC")
    end   = pl.datetime(target_day.year, target_day.month, target_day.day,
                        23, 59, 59, time_zone="UTC")
    day_utc = meteo_utc.filter(
        (pl.col("timestamp") >= start) & (pl.col("timestamp") <= end)
    )

    # Préparer les noms de features null en cas de données manquantes
    def null_features_for_var(var, station):
        nulls = {}
        if var in CYCLIC_PRED_VARS:
            for h in range(24):
                nulls[f"{prefix}_{var}_sin_{station}_t{h:02d}"] = None
                nulls[f"{prefix}_{var}_cos_{station}_t{h:02d}"] = None
        else:
            for h in range(24):
                nulls[f"{prefix}_{var}_{station}_t{h:02d}"] = None
        return nulls

    if len(day_utc) == 0:
        for var in PRED_VARS:
            for station in STATIONS:
                features.update(null_features_for_var(var, station))
        # PV prévu aussi à null
        for h in range(24):
            features[f"{prefix}_pv_MW_t{h:02d}"] = None
        return features

    # Index heure → position dans day_utc
    hours_utc = day_utc["timestamp"].dt.hour().to_list()
    hour_to_idx = {}
    for idx, h in enumerate(hours_utc):
        if h not in hour_to_idx:
            hour_to_idx[h] = idx

    for var in PRED_VARS:
        is_cyclic = var in CYCLIC_PRED_VARS

        for station in STATIONS:
            # Collecter tous les horizons nécessaires pour ce var/station
            needed_horizons = set()
            for h_utc in range(24):
                needed_horizons.add(horizon_func(h_utc))

            # Charger les colonnes pour ces horizons
            col_values = {}
            for horizon in needed_horizons:
                col_name = f"pred_{var}_h{horizon}_{station}"
                if col_name in day_utc.columns:
                    col_values[horizon] = day_utc[col_name].to_list()

            # Extraire valeur par heure
            for h_utc in range(24):
                horizon = horizon_func(h_utc)
                idx = hour_to_idx.get(h_utc)

                raw_val = None
                if horizon in col_values and idx is not None:
                    v = col_values[horizon][idx]
                    raw_val = float(v) if v is not None else None

                if is_cyclic:
                    # Encodage sin/cos pour wind_dir
                    if raw_val is not None:
                        rad = raw_val * math.pi / 180.0
                        features[f"{prefix}_{var}_sin_{station}_t{h_utc:02d}"] = math.sin(rad)
                        features[f"{prefix}_{var}_cos_{station}_t{h_utc:02d}"] = math.cos(rad)
                    else:
                        features[f"{prefix}_{var}_sin_{station}_t{h_utc:02d}"] = None
                        features[f"{prefix}_{var}_cos_{station}_t{h_utc:02d}"] = None
                else:
                    features[f"{prefix}_{var}_{station}_t{h_utc:02d}"] = raw_val

    # ── Feature PV prévu : moyenne irradiance × facteur conversion
    for h_utc in range(24):
        irr_vals = []
        for station in STATIONS:
            key = f"{prefix}_glob_rad_{station}_t{h_utc:02d}"
            v = features.get(key)
            if v is not None:
                irr_vals.append(v)

        if irr_vals:
            irr_mean = sum(irr_vals) / len(irr_vals)
            features[f"{prefix}_pv_MW_t{h_utc:02d}"] = irr_mean * PV_SURFACE_M2 * PV_PERF_RATIO / 1_000_000
        else:
            features[f"{prefix}_pv_MW_t{h_utc:02d}"] = None

    return features


# ─────────────────────────────────────────────
# 3. CONSTRUCTION FEATURES PAR JOUR
# ─────────────────────────────────────────────

def build_features(
    target_date: date,
    oiken: pl.DataFrame,
    meteo_utc: pl.DataFrame,
    meteo_zurich: pl.DataFrame,
    real_cols: list[str],
) -> dict | None:
    """
    Construit le vecteur de features pour prédire J+1 = target_date.
    Simule les données disponibles à 11h le jour J = target_date - 1.
    """
    day_j   = target_date - timedelta(days=1)   # jour de soumission
    day_jm1 = target_date - timedelta(days=2)   # J-1 (avant-veille)

    # ── Cible : load J+1
    oiken_target = get_day_slice(oiken, target_date)
    if len(oiken_target) < 90:
        return None

    features = {}

    # ── Load historique J-1 à J-7
    for delta in LOAD_HISTORY_DAYS:
        day_past   = target_date - timedelta(days=delta)
        label      = f"jm{delta - 1}"
        oiken_past = get_day_slice(oiken, day_past)

        if len(oiken_past) >= 90:
            features.update(series_stats(oiken_past["load"], f"load_{label}"))
            features.update(hourly_profile(oiken_past, "load", f"load_{label}"))
        else:
            for k in ["mean", "max", "min", "std"]:
                features[f"load_{label}_{k}"] = None
            for h in range(24):
                features[f"load_{label}_h{h:02d}"] = None

    # ── Production solaire : J-1 total (4 sources) + J matin (3 sources live, SANS remote)
    oiken_jm1       = get_day_slice(oiken, day_jm1)
    oiken_j_morning = get_morning_slice(oiken, day_j, until_hour=10)

    # J-1 complet: toutes les 4 sources (disponibles à 2h du matin le jour J)
    for col in PROD_COLS:
        if col in oiken_jm1.columns:
            features[f"{col}_jm1_total"] = float(oiken_jm1[col].sum())

    # J matin: uniquement les 3 sources live (SANS solar_remote)
    for col in PROD_COLS_LIVE:
        if col in oiken_j_morning.columns:
            features[f"{col}_j_morning_total"] = float(oiken_j_morning[col].sum())

    # ── Météo réelle J-1 complet (Zurich)
    meteo_jm1 = get_day_slice(meteo_zurich, day_jm1)
    features.update(real_meteo_stats(meteo_jm1, real_cols, "rmet_jm1"))

    # ── Météo réelle J matin jusqu'à 10h (Zurich)
    meteo_j_morning = get_morning_slice(meteo_zurich, day_j, until_hour=10)
    features.update(real_meteo_stats(meteo_j_morning, real_cols, "rmet_jmorn"))

    # ── Prévisions météo J+1 (horizons corrigés h15–h36)
    features.update(extract_pred_vector(
        meteo_utc, target_date,
        horizon_func=get_correct_horizon_jp1,
        prefix="pred",
    ))

    # ── Prévisions météo J (h13/h14/h15 — inertie thermique)
    features.update(extract_pred_vector(
        meteo_utc, day_j,
        horizon_func=get_correct_horizon_j,
        prefix="predJ",
    ))

    # ── Calendaire (valeurs brutes)
    doy = target_date.timetuple().tm_yday
    features["dayofweek"]  = target_date.weekday()
    features["month"]      = target_date.month
    features["is_weekend"] = int(target_date.weekday() >= 5)
    features["is_holiday"] = int(target_date in FERIES)
    features["is_ramadan"] = int(is_ramadan(target_date))

    # ── is_ramadan_hour
    ramadan_hours = set(ramadan_night_hours(target_date))
    for h in range(24):
        features[f"is_ramadan_h{h:02d}"] = int(h in ramadan_hours)

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
    oiken                        = load_oiken(CSV)
    meteo_utc, meteo_zurich, real_cols = load_meteo(METEO)

    first_ts  = oiken["timestamp"].drop_nulls()[0]
    first_day = first_ts.date() + timedelta(days=9)
    last_day  = oiken["timestamp"][-1].date() - timedelta(days=1)

    all_dates = [first_day + timedelta(days=i)
                 for i in range((last_day - first_day).days + 1)]

    print(f"\n=== Construction features v6 : {first_day} → {last_day} ({len(all_dates)} jours) ===")

    rows_X, rows_Y, rows_B, dates_ok = [], [], [], []

    for i, target_date in enumerate(all_dates):
        if i % 100 == 0:
            print(f"  {i}/{len(all_dates)} — {target_date}")

        result = build_features(target_date, oiken, meteo_utc, meteo_zurich, real_cols)
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

    X.write_parquet(OUT / "X_features_v6.parquet")
    Y.write_parquet(OUT / "Y_target_v6.parquet")
    B.write_parquet(OUT / "B_baseline_v6.parquet")

    print(f"\n✓ X_features_v6 : {X.shape[0]} jours × {X.shape[1]} colonnes")
    print(f"✓ Y_target_v6   : {Y.shape[0]} jours × {Y.shape[1]} colonnes")
    print(f"✓ B_baseline_v6 : {B.shape[0]} jours × {B.shape[1]} colonnes")

    # ── Résumé des features
    cols = [c for c in X.columns if c != "date"]
    pred_jp1 = [c for c in cols if c.startswith("pred_") and not c.startswith("predJ_")]
    pred_j   = [c for c in cols if c.startswith("predJ_")]
    load_c   = [c for c in cols if c.startswith("load_")]
    solar_c  = [c for c in cols if c.startswith("solar_")]
    rmet_c   = [c for c in cols if c.startswith("rmet_")]
    cal_c    = [c for c in cols if c not in pred_jp1 + pred_j + load_c + solar_c + rmet_c]

    print(f"\n  Décomposition features:")
    print(f"    Load historique     : {len(load_c)}")
    print(f"    Production solaire  : {len(solar_c)}")
    print(f"    Météo réelle        : {len(rmet_c)}")
    print(f"    Prévisions J+1      : {len(pred_jp1)}")
    print(f"    Prévisions J        : {len(pred_j)}")
    print(f"    Calendaire + autre  : {len(cal_c)}")

    # Vérification anti-leakage: compter wind_dir sin/cos
    wd_sin = [c for c in cols if "wind_dir_sin" in c]
    wd_cos = [c for c in cols if "wind_dir_cos" in c]
    wd_raw = [c for c in cols if "wind_dir" in c and "sin" not in c and "cos" not in c]
    print(f"\n  Wind dir encodage:")
    print(f"    sin: {len(wd_sin)}, cos: {len(wd_cos)}, brut: {len(wd_raw)}")
    if wd_raw:
        print(f"    ⚠️  Colonnes wind_dir brutes restantes: {wd_raw[:5]}...")
    else:
        print(f"    ✓ Aucune colonne wind_dir brute")

    print(f"\n✓ Sauvegardé dans : {OUT}")


if __name__ == "__main__":
    main()