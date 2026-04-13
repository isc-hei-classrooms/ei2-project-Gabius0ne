"""
pipeline_features_v13.py

Changements par rapport à v12 — réduction drastique du bruit :

  IRRADIANCE J+1 : agrégée en moyenne Sion/Visp/Pully (stations de plaine),
    heures diurnes 06h-20h UTC uniquement. Supprime Montana, Col, Les_Attelas
    et les heures nocturnes. De 144 features → 14 + 1 moyenne jour.
    
  VENT : uniquement wind_speed sur Sion + Pully, heures 06h-20h UTC.
    Supprime wind_dir (sin/cos) et les 4 autres stations.
    De 864 features → 28.
    
  VARIANCES IRRADIANCE : supprimées (glob_rad_q10, q90, stde, std_stations,
    spread). Le modèle utilisait ces features pour ignorer l'irradiance
    quand elle était incertaine. De 457 features → 0.
    
  MÉTÉO J (jour de soumission) : supprimée entièrement.
    Importance toujours <1% dans tous les tests. De ~1320 features → 0.
    
  INTERACTIONS PV : recalculées avec la nouvelle irradiance agrégée
    (moyenne 3 stations plaine), heures 06h-20h.
    
  VACANCES SCOLAIRES : ajout is_school_holiday + is_bridge_day.
    Dates Valais romand (Sion) 2022-2025.
    
  INCHANGÉ : load historique, PV mesuré, calendaire, météo mesurée,
    temp/pres/hum/precip/sunshine J+1.

Estimation : ~800-1000 features (vs 3584 en v12).

Sorties :
  DATA/processed/X_features_v13.parquet
  DATA/processed/Y_target_v13.parquet
  DATA/processed/B_baseline_v13.parquet
"""

import math
import polars as pl
import numpy as np
from pathlib import Path
from datetime import timedelta, date

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
BASE    = Path(__file__).resolve().parents[2] / "DATA"
CSV     = BASE / "oiken-data.csv"
METEO   = BASE / "meteo_multistation_v5.parquet"
OUT     = BASE / "processed"
OUT.mkdir(parents=True, exist_ok=True)

PROD_COLS = [
    "solar_central_valais",
    "solar_sion",
    "solar_sierre",
    "solar_remote",
]

PROD_COLS_LIVE = [
    "solar_central_valais",
    "solar_sion",
    "solar_sierre",
]

LOAD_HISTORY_DAYS = list(range(2, 9))

REAL_METEO_VARS = ["temp_2m", "glob_rad", "pressure", "relhum_2m"]

# Toutes les stations pour les mesures réelles (inchangé)
STATIONS_ALL = [
    "Pully", "Sion", "Visp", "Montana",
    "Col_du_Grand_St-Bernard", "Les_Attelas",
]

# v13 : stations de plaine pour les prévisions d'irradiance et vent
STATIONS_PLAINE = ["Sion", "Visp", "Pully"]

# v13 : stations pour le vent (vitesse seulement)
STATIONS_WIND = ["Sion", "Pully"]

# v13 : heures diurnes UTC pour irradiance et vent
HOURS_DIURNAL = list(range(6, 20))  # 06h-19h UTC

# v13 : variables prévues J+1 (SANS vent et SANS irradiance — traités séparément)
PRED_VARS_JP1 = ["temp", "glob_rad", "pressure", "relhum", "precip", "sunshine"]

# Pas de variables cycliques dans PRED_VARS_JP1 (wind_dir supprimé)
CYCLIC_PRED_VARS = set()  # vide en v13

HORIZONS_BY_MOD = {
    0: list(range(3, 37, 3)),
    1: list(range(1, 35, 3)),
    2: list(range(2, 36, 3)),
}

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

# v13 : Vacances scolaires Valais romand (Sion)
# Tuple (début inclus, fin incluse) — jours de vacances effectifs
VACANCES_SCOLAIRES = [
    # 2022-2023
    (date(2022, 10, 12), date(2022, 10, 23)),  # Automne
    (date(2022, 12, 23), date(2023,  1,  8)),  # Noël
    (date(2023,  2, 17), date(2023,  2, 26)),  # Carnaval
    (date(2023,  4,  6), date(2023,  4, 16)),  # Pâques
    (date(2023,  6, 24), date(2023,  8, 16)),  # Été
    # 2023-2024
    (date(2023, 10, 11), date(2023, 10, 22)),  # Automne
    (date(2023, 12, 22), date(2024,  1,  7)),  # Noël
    (date(2024,  2,  9), date(2024,  2, 18)),  # Carnaval
    (date(2024,  3, 28), date(2024,  4,  7)),  # Pâques
    (date(2024,  6, 22), date(2024,  8, 14)),  # Été
    # 2024-2025
    (date(2024, 10, 16), date(2024, 10, 27)),  # Automne
    (date(2024, 12, 20), date(2025,  1,  6)),  # Noël
    (date(2025,  2, 28), date(2025,  3,  9)),  # Carnaval
    (date(2025,  4, 17), date(2025,  4, 27)),  # Pâques
    (date(2025,  6, 28), date(2025,  8, 17)),  # Été
    # 2025-2026
    (date(2025, 10, 18), date(2025, 10, 26)),  # Automne
    (date(2025, 12, 20), date(2026,  1,  4)),  # Noël
]


# ─────────────────────────────────────────────
# HELPERS
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


def is_school_holiday(d: date) -> bool:
    for start, end in VACANCES_SCOLAIRES:
        if start <= d <= end:
            return True
    return False


def is_bridge_day(d: date) -> bool:
    """Jour ouvré coincé entre un férié/weekend et un weekend/férié."""
    if d.weekday() >= 5:  # déjà weekend
        return False
    if d in FERIES:  # déjà férié
        return False
    if is_school_holiday(d):  # déjà en vacances
        return False
    
    yesterday = d - timedelta(days=1)
    tomorrow = d + timedelta(days=1)
    
    def is_off(day):
        return day.weekday() >= 5 or day in FERIES
    
    # Vendredi après un jeudi férié, ou lundi avant un mardi férié
    if d.weekday() == 4 and is_off(yesterday):  # vendredi, jeudi off
        return True
    if d.weekday() == 0 and is_off(tomorrow):  # lundi, mardi off
        return True
    # Coincé entre deux jours off
    if is_off(yesterday) and is_off(tomorrow):
        return True
    return False


def _get_pv_capacity_proxy(d: date) -> float:
    from datetime import date as _date
    anchors = [
        (_date(2022, 10, 1), 4.65),
        (_date(2022, 12, 31), 4.65),
        (_date(2023, 12, 31), 6.20),
        (_date(2024, 12, 31), 8.00),
        (_date(2025, 12, 31), 9.51),
    ]
    if d <= anchors[0][0]:
        national_gw = anchors[0][1]
    elif d >= anchors[-1][0]:
        national_gw = anchors[-1][1]
    else:
        for i in range(len(anchors) - 1):
            d1, v1 = anchors[i]
            d2, v2 = anchors[i + 1]
            if d1 <= d <= d2:
                frac = (d - d1).days / max((d2 - d1).days, 1)
                national_gw = v1 + frac * (v2 - v1)
                break
    OIKEN_BASE_MWP = 55.0
    NATIONAL_BASE_GW = 4.65
    return OIKEN_BASE_MWP * national_gw / NATIONAL_BASE_GW


def get_correct_horizon_jp1(h_utc: int) -> int | None:
    mod = h_utc % 3
    available = HORIZONS_BY_MOD[mod]
    min_needed = h_utc + 14
    valid = [h for h in available if h >= min_needed]
    if valid:
        return min(valid)
    else:
        return None


# ─────────────────────────────────────────────
# CHARGEMENT
# ─────────────────────────────────────────────

def load_oiken(path: Path) -> pl.DataFrame:
    df = pl.read_csv(
        path, try_parse_dates=False,
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
        for station in STATIONS_ALL
        if f"{var}_{station}" in all_cols
    ]
    print(f"  Météo : {len(df_utc):,} lignes | {len(real_cols)} colonnes réelles")
    return df_utc, df_zurich, real_cols


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


# ─────────────────────────────────────────────
# PRÉVISIONS J+1 (v13 : restructuré)
# ─────────────────────────────────────────────

def extract_pred_jp1(meteo_utc: pl.DataFrame, target_day: date) -> dict:
    """
    v13 : prévisions J+1 restructurées.
    
    - temp/pressure/relhum/precip/sunshine : 6 stations × 24h (inchangé)
    - glob_rad : 3 stations plaine × 14h diurnes + 1 moyenne jour
    - wind_speed : 2 stations × 14h diurnes
    - wind_dir : SUPPRIMÉ
    - glob_rad variance (q10/q90/stde/std_stations/spread) : SUPPRIMÉ
    """
    features = {}
    start = pl.datetime(target_day.year, target_day.month, target_day.day, 0, 0, 0, time_zone="UTC")
    end   = pl.datetime(target_day.year, target_day.month, target_day.day, 23, 59, 59, time_zone="UTC")
    day_utc = meteo_utc.filter((pl.col("timestamp") >= start) & (pl.col("timestamp") <= end))

    if len(day_utc) == 0:
        # Null features pour tout
        for var in PRED_VARS_JP1:
            for station in STATIONS_ALL:
                for h in range(24):
                    features[f"pred_{var}_{station}_t{h:02d}"] = None
        for station in STATIONS_PLAINE:
            for h in HOURS_DIURNAL:
                features[f"pred_glob_rad_{station}_t{h:02d}"] = None
        for station in STATIONS_WIND:
            for h in HOURS_DIURNAL:
                features[f"pred_wind_speed_{station}_t{h:02d}"] = None
        features["pred_glob_rad_mean_day"] = None
        return features

    hours_utc = day_utc["timestamp"].dt.hour().to_list()
    hour_to_idx = {}
    for idx, h in enumerate(hours_utc):
        if h not in hour_to_idx:
            hour_to_idx[h] = idx

    def get_pred_value(var, station, h_utc):
        horizon = get_correct_horizon_jp1(h_utc)
        if horizon is None:
            return None
        col_name = f"pred_{var}_h{horizon}_{station}"
        if col_name not in day_utc.columns:
            return None
        idx = hour_to_idx.get(h_utc)
        if idx is None:
            return None
        v = day_utc[col_name][idx]
        return float(v) if v is not None else None

    # ── temp/pressure/relhum/precip/sunshine : toutes stations, 24h
    for var in PRED_VARS_JP1:
        if var == "glob_rad":
            continue  # traité séparément
        for station in STATIONS_ALL:
            for h_utc in range(24):
                features[f"pred_{var}_{station}_t{h_utc:02d}"] = get_pred_value(var, station, h_utc)

    # ── glob_rad : 3 stations plaine, heures diurnes seulement
    irr_day_vals = []
    for station in STATIONS_PLAINE:
        for h_utc in HOURS_DIURNAL:
            val = get_pred_value("glob_rad", station, h_utc)
            features[f"pred_glob_rad_{station}_t{h_utc:02d}"] = val
            if val is not None:
                irr_day_vals.append(val)

    features["pred_glob_rad_mean_day"] = (
        sum(irr_day_vals) / len(irr_day_vals) if irr_day_vals else None
    )

    # ── wind_speed : 2 stations, heures diurnes seulement
    for station in STATIONS_WIND:
        for h_utc in HOURS_DIURNAL:
            features[f"pred_wind_speed_{station}_t{h_utc:02d}"] = get_pred_value("wind_speed", station, h_utc)

    return features


# ─────────────────────────────────────────────
# PV YIELD RATIO (v11, conservé)
# ─────────────────────────────────────────────

def compute_pv_yield_ratios(
    oiken: pl.DataFrame,
    meteo_zurich: pl.DataFrame,
    target_date: date,
) -> dict:
    features = {}
    day_jm1 = target_date - timedelta(days=2)
    
    for window_days, label in [(30, "30j"), (90, "90j")]:
        window_start = day_jm1 - timedelta(days=window_days - 1)
        
        start_dt = pl.datetime(window_start.year, window_start.month, window_start.day,
                               0, 0, 0, time_zone="Europe/Zurich")
        end_dt = pl.datetime(day_jm1.year, day_jm1.month, day_jm1.day,
                             23, 59, 59, time_zone="Europe/Zurich")
        
        oiken_window = oiken.filter(
            (pl.col("timestamp") >= start_dt) & (pl.col("timestamp") <= end_dt)
        )
        meteo_window = meteo_zurich.filter(
            (pl.col("timestamp") >= start_dt) & (pl.col("timestamp") <= end_dt)
        )
        
        if "solar_remote" in oiken_window.columns and len(oiken_window) > 0:
            remote_max = oiken_window["solar_remote"].drop_nulls().max()
            remote_max = float(remote_max) if remote_max is not None else None
        else:
            remote_max = None
        
        if "glob_rad_Sion" in meteo_window.columns and len(meteo_window) > 0:
            glob_max = meteo_window["glob_rad_Sion"].drop_nulls().max()
            glob_max = float(glob_max) if glob_max is not None else None
        else:
            glob_max = None
        
        if remote_max is not None and glob_max is not None and glob_max > 10:
            features[f"pv_yield_{label}"] = remote_max / glob_max
        else:
            features[f"pv_yield_{label}"] = None
        
        features[f"solar_remote_max_{label}"] = remote_max
    
    return features


# ─────────────────────────────────────────────
# v13 : INTERACTIONS PV (avec irradiance agrégée 3 stations)
# ─────────────────────────────────────────────

def compute_pv_interaction_features(features: dict) -> dict:
    """
    v13 : interactions irradiance × pv_yield, basées sur la moyenne
    des 3 stations de plaine (Sion/Visp/Pully), heures diurnes.
    """
    interaction_feats = {}
    
    pv_yield_30j = features.get("pv_yield_30j")
    pv_yield_90j = features.get("pv_yield_90j")
    
    for h_utc in HOURS_DIURNAL:
        # Moyenne irradiance sur les 3 stations plaine
        irr_vals = []
        for station in STATIONS_PLAINE:
            v = features.get(f"pred_glob_rad_{station}_t{h_utc:02d}")
            if v is not None:
                irr_vals.append(v)
        
        irr_mean = sum(irr_vals) / len(irr_vals) if irr_vals else None
        
        if irr_mean is not None and pv_yield_30j is not None:
            interaction_feats[f"pred_pv_adj_30j_t{h_utc:02d}"] = irr_mean * pv_yield_30j
        else:
            interaction_feats[f"pred_pv_adj_30j_t{h_utc:02d}"] = None
        
        if irr_mean is not None and pv_yield_90j is not None:
            interaction_feats[f"pred_pv_adj_90j_t{h_utc:02d}"] = irr_mean * pv_yield_90j
        else:
            interaction_feats[f"pred_pv_adj_90j_t{h_utc:02d}"] = None
    
    # Agrégés diurnes
    for label in ["30j", "90j"]:
        day_vals = [interaction_feats.get(f"pred_pv_adj_{label}_t{h:02d}")
                    for h in HOURS_DIURNAL]
        day_clean = [v for v in day_vals if v is not None]
        interaction_feats[f"pred_pv_adj_{label}_day"] = sum(day_clean) if day_clean else None
    
    return interaction_feats


# ─────────────────────────────────────────────
# CONSTRUCTION FEATURES
# ─────────────────────────────────────────────

def build_features(
    target_date: date,
    oiken: pl.DataFrame,
    meteo_utc: pl.DataFrame,
    meteo_zurich: pl.DataFrame,
    real_cols: list[str],
) -> dict | None:
    day_j   = target_date - timedelta(days=1)
    day_jm1 = target_date - timedelta(days=2)

    oiken_target = get_day_slice(oiken, target_date)
    if len(oiken_target) < 90:
        return None

    features = {}

    # ── Load historique J-1 à J-7 (inchangé)
    for delta in LOAD_HISTORY_DAYS:
        day_past = target_date - timedelta(days=delta)
        label = f"jm{delta - 1}"
        oiken_past = get_day_slice(oiken, day_past)
        if len(oiken_past) >= 90:
            features.update(series_stats(oiken_past["load"], f"load_{label}"))
            features.update(hourly_profile(oiken_past, "load", f"load_{label}"))
        else:
            for k in ["mean", "max", "min", "std"]:
                features[f"load_{label}_{k}"] = None
            for h in range(24):
                features[f"load_{label}_h{h:02d}"] = None

    # ── Production solaire (inchangé)
    oiken_jm1 = get_day_slice(oiken, day_jm1)
    oiken_j_morning = get_morning_slice(oiken, day_j, until_hour=10)

    for col in PROD_COLS:
        if col in oiken_jm1.columns:
            features[f"{col}_jm1_total"] = float(oiken_jm1[col].sum())
            features.update(hourly_profile(oiken_jm1, col, f"{col}_jm1"))

    for col in PROD_COLS_LIVE:
        if col in oiken_j_morning.columns:
            features[f"{col}_j_morning_total"] = float(oiken_j_morning[col].sum())

    # ── Météo réelle (inchangé — toutes stations)
    meteo_jm1 = get_day_slice(meteo_zurich, day_jm1)
    features.update(real_meteo_stats(meteo_jm1, real_cols, "rmet_jm1"))
    meteo_j_morning = get_morning_slice(meteo_zurich, day_j, until_hour=10)
    features.update(real_meteo_stats(meteo_j_morning, real_cols, "rmet_jmorn"))

    # ── Prévisions J+1 (v13 : restructuré, sans vent dir, sans variance irr, sans météo J)
    features.update(extract_pred_jp1(meteo_utc, target_date))

    # ── Proxy puissance PV installée
    features["pv_capacity_MWp"] = _get_pv_capacity_proxy(target_date)

    # ── PV yield ratio glissant
    features.update(compute_pv_yield_ratios(oiken, meteo_zurich, target_date))

    # ── Interactions PV (v13 : 3 stations plaine, heures diurnes)
    features.update(compute_pv_interaction_features(features))

    # ── Calendaire (v13 : ajout vacances scolaires + pont)
    doy = target_date.timetuple().tm_yday
    features["dayofweek"]        = target_date.weekday()
    features["month"]            = target_date.month
    features["is_weekend"]       = int(target_date.weekday() >= 5)
    features["is_holiday"]       = int(target_date in FERIES)
    features["is_school_holiday"] = int(is_school_holiday(target_date))
    features["is_bridge_day"]    = int(is_bridge_day(target_date))
    features["is_ramadan"]       = int(is_ramadan(target_date))

    ramadan_hours = set(ramadan_night_hours(target_date))
    for h in range(24):
        features[f"is_ramadan_h{h:02d}"] = int(h in ramadan_hours)

    # ── Encodages cycliques
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
# PIPELINE PRINCIPAL
# ─────────────────────────────────────────────

def main():
    print("=== Chargement des données ===")
    oiken = load_oiken(CSV)
    meteo_utc, meteo_zurich, real_cols = load_meteo(METEO)

    first_ts  = oiken["timestamp"].drop_nulls()[0]
    first_day = first_ts.date() + timedelta(days=9)
    last_day  = oiken["timestamp"][-1].date() - timedelta(days=1)

    all_dates = [first_day + timedelta(days=i)
                 for i in range((last_day - first_day).days + 1)]

    print(f"\n=== Construction features v13 : {first_day} → {last_day} ({len(all_dates)} jours) ===")

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

    X.write_parquet(OUT / "X_features_v13.parquet")
    Y.write_parquet(OUT / "Y_target_v13.parquet")
    B.write_parquet(OUT / "B_baseline_v13.parquet")

    print(f"\n✓ X_features_v13 : {X.shape[0]} jours × {X.shape[1]} colonnes")
    print(f"✓ Y_target_v13   : {Y.shape[0]} jours × {Y.shape[1]} colonnes")
    print(f"✓ B_baseline_v13 : {B.shape[0]} jours × {B.shape[1]} colonnes")

    # Vérifications
    check_cols = [
        "pv_yield_30j", "pv_yield_90j", "pv_capacity_MWp",
        "pred_pv_adj_30j_t12", "pred_pv_adj_30j_day",
        "pred_glob_rad_Sion_t12", "pred_glob_rad_mean_day",
        "is_school_holiday", "is_bridge_day",
    ]
    for col in check_cols:
        if col in X.columns:
            s = X[col].drop_nulls()
            if s.dtype in [pl.Float64, pl.Float32]:
                print(f"  {col}: min={s.min():.2f}, max={s.max():.2f}, mean={s.mean():.2f}, nulls={X[col].null_count()}")
            else:
                print(f"  {col}: sum={s.sum()}, nulls={X[col].null_count()}")

    # Vérifier l'absence de features supprimées
    bad_cols = [c for c in X.columns if any(k in c for k in [
        "predJ_", "wind_dir", "glob_rad_q10", "glob_rad_q90",
        "glob_rad_stde", "glob_rad_std_stations", "glob_rad_spread",
    ])]
    if bad_cols:
        print(f"\n⚠ Features qui devraient être supprimées : {bad_cols[:10]}...")
    else:
        print(f"\n✓ Météo J, wind_dir, variances irradiance : supprimées")

    # Compter par catégorie
    n_wind = len([c for c in X.columns if "wind" in c])
    n_glob = len([c for c in X.columns if "glob_rad" in c and "rmet" not in c])
    n_load = len([c for c in X.columns if c.startswith("load_")])
    n_pv   = len([c for c in X.columns if "pv_adj" in c or "pv_yield" in c or "pv_capacity" in c])
    print(f"  Wind speed : {n_wind} features")
    print(f"  Glob rad (prévu) : {n_glob} features")
    print(f"  Load historique : {n_load} features")
    print(f"  PV (yield/capacity/interaction) : {n_pv} features")

    print(f"\n✓ Sauvegardé dans : {OUT}")


if __name__ == "__main__":
    main()