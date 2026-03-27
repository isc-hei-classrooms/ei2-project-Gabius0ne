"""
pipeline_features_v11.py

Changement par rapport à v10 :

  AJOUT pv_yield_30j : ratio glissant max(solar_remote) / max(glob_rad_Sion)
  sur les 30 derniers jours (J-30 à J-1).
  
  Ce ratio capture la croissance de la puissance PV installée sur le
  territoire Oiken, indépendamment de la saisonnalité :
    - Le numérateur (max production PV) augmente avec la puissance installée
    - Le dénominateur (max irradiance) dépend uniquement de la météo/saison
    - Le ratio isole donc la croissance PV
  
  Anti-leakage : basé sur J-1 et avant (solar_remote dispo à 2h, glob_rad temps réel)
  
  Ajout aussi pv_yield_90j pour plus de stabilité.

Sorties :
  DATA/processed/X_features_v11.parquet
  DATA/processed/Y_target_v11.parquet
  DATA/processed/B_baseline_v11.parquet
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
METEO   = BASE / "meteo_multistation_v5.parquet"
OUT     = BASE / "processed"
OUT.mkdir(exist_ok=True)

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

STATIONS = [
    "Pully", "Sion", "Visp", "Montana",
    "Col_du_Grand_St-Bernard", "Les_Attelas",
]

PRED_VARS = ["temp", "glob_rad", "pressure", "relhum", "precip", "sunshine", "wind_speed", "wind_dir"]

PRED_VARS_UNCERTAINTY = ["glob_rad_q10", "glob_rad_q90", "glob_rad_stde"]

CYCLIC_PRED_VARS = {"wind_dir"}

HORIZONS_BY_MOD = {
    0: list(range(3, 37, 3)),
    1: list(range(1, 35, 3)),
    2: list(range(2, 36, 3)),
}

PV_SURFACE_M2  = 540_000
PV_PERF_RATIO  = 0.75

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


def get_correct_horizon_j(h_utc: int) -> int:
    mod = h_utc % 3
    available = HORIZONS_BY_MOD[mod]
    min_needed = max(h_utc - 10, 1)
    valid = [h for h in available if h >= min_needed]
    if valid:
        return min(valid)
    else:
        return max(available)


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
        for station in STATIONS
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


def extract_pred_vector(
    meteo_utc: pl.DataFrame, target_day: date,
    horizon_func, prefix: str, extra_vars: list[str] | None = None,
) -> dict:
    all_vars = list(PRED_VARS) + (extra_vars or [])
    features = {}
    start = pl.datetime(target_day.year, target_day.month, target_day.day, 0, 0, 0, time_zone="UTC")
    end   = pl.datetime(target_day.year, target_day.month, target_day.day, 23, 59, 59, time_zone="UTC")
    day_utc = meteo_utc.filter((pl.col("timestamp") >= start) & (pl.col("timestamp") <= end))

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
        for var in all_vars:
            for station in STATIONS:
                features.update(null_features_for_var(var, station))
        for h in range(24):
            features[f"{prefix}_pv_MW_t{h:02d}"] = None
        return features

    hours_utc = day_utc["timestamp"].dt.hour().to_list()
    hour_to_idx = {}
    for idx, h in enumerate(hours_utc):
        if h not in hour_to_idx:
            hour_to_idx[h] = idx

    for var in all_vars:
        is_cyclic = var in CYCLIC_PRED_VARS
        for station in STATIONS:
            needed_horizons = set()
            for h_utc in range(24):
                horizon = horizon_func(h_utc)
                if horizon is not None:
                    needed_horizons.add(horizon)
            col_values = {}
            for horizon in needed_horizons:
                col_name = f"pred_{var}_h{horizon}_{station}"
                if col_name in day_utc.columns:
                    col_values[horizon] = day_utc[col_name].to_list()
            for h_utc in range(24):
                horizon = horizon_func(h_utc)
                if horizon is None:
                    if is_cyclic:
                        features[f"{prefix}_{var}_sin_{station}_t{h_utc:02d}"] = None
                        features[f"{prefix}_{var}_cos_{station}_t{h_utc:02d}"] = None
                    else:
                        features[f"{prefix}_{var}_{station}_t{h_utc:02d}"] = None
                    continue
                idx = hour_to_idx.get(h_utc)
                raw_val = None
                if horizon in col_values and idx is not None:
                    v = col_values[horizon][idx]
                    raw_val = float(v) if v is not None else None
                if is_cyclic:
                    if raw_val is not None:
                        rad = raw_val * math.pi / 180.0
                        features[f"{prefix}_{var}_sin_{station}_t{h_utc:02d}"] = math.sin(rad)
                        features[f"{prefix}_{var}_cos_{station}_t{h_utc:02d}"] = math.cos(rad)
                    else:
                        features[f"{prefix}_{var}_sin_{station}_t{h_utc:02d}"] = None
                        features[f"{prefix}_{var}_cos_{station}_t{h_utc:02d}"] = None
                else:
                    features[f"{prefix}_{var}_{station}_t{h_utc:02d}"] = raw_val

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
# PV YIELD RATIO (NOUVEAU v11)
# ─────────────────────────────────────────────

def compute_pv_yield_ratios(
    oiken: pl.DataFrame,
    meteo_zurich: pl.DataFrame,
    target_date: date,
) -> dict:
    """
    Calcule le ratio max(solar_remote) / max(glob_rad_Sion) sur des fenêtres
    glissantes de 30 et 90 jours.
    
    Ce ratio capture la puissance PV installée indépendamment de la saison :
      - Numérateur : max production PV (augmente avec puissance installée)
      - Dénominateur : max irradiance (dépend uniquement de la météo)
      - Ratio : proxy direct de la puissance installée
    
    Anti-leakage : J-1 et avant (solar_remote dispo à 2h, glob_rad temps réel)
    """
    features = {}
    
    day_jm1 = target_date - timedelta(days=2)  # J-1 = target_date - 2
    
    for window_days, label in [(30, "30j"), (90, "90j")]:
        window_start = day_jm1 - timedelta(days=window_days - 1)
        
        # Filtrer la fenêtre dans Oiken (Europe/Zurich)
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
        
        # Max solar_remote sur la fenêtre
        if "solar_remote" in oiken_window.columns and len(oiken_window) > 0:
            remote_max = oiken_window["solar_remote"].drop_nulls().max()
            remote_max = float(remote_max) if remote_max is not None else None
        else:
            remote_max = None
        
        # Max glob_rad_Sion sur la fenêtre
        if "glob_rad_Sion" in meteo_window.columns and len(meteo_window) > 0:
            glob_max = meteo_window["glob_rad_Sion"].drop_nulls().max()
            glob_max = float(glob_max) if glob_max is not None else None
        else:
            glob_max = None
        
        # Ratio (protéger division par zéro)
        if remote_max is not None and glob_max is not None and glob_max > 10:
            features[f"pv_yield_{label}"] = remote_max / glob_max
        else:
            features[f"pv_yield_{label}"] = None
        
        # Aussi stocker le max remote brut (proxy direct de capacité)
        features[f"solar_remote_max_{label}"] = remote_max
    
    return features


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

    # ── Load historique J-1 à J-7
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

    # ── Production solaire
    oiken_jm1 = get_day_slice(oiken, day_jm1)
    oiken_j_morning = get_morning_slice(oiken, day_j, until_hour=10)

    for col in PROD_COLS:
        if col in oiken_jm1.columns:
            features[f"{col}_jm1_total"] = float(oiken_jm1[col].sum())
            features.update(hourly_profile(oiken_jm1, col, f"{col}_jm1"))

    for col in PROD_COLS_LIVE:
        if col in oiken_j_morning.columns:
            features[f"{col}_j_morning_total"] = float(oiken_j_morning[col].sum())

    # ── Météo réelle
    meteo_jm1 = get_day_slice(meteo_zurich, day_jm1)
    features.update(real_meteo_stats(meteo_jm1, real_cols, "rmet_jm1"))
    meteo_j_morning = get_morning_slice(meteo_zurich, day_j, until_hour=10)
    features.update(real_meteo_stats(meteo_j_morning, real_cols, "rmet_jmorn"))

    # ── Prévisions J+1
    features.update(extract_pred_vector(
        meteo_utc, target_date,
        horizon_func=get_correct_horizon_jp1,
        prefix="pred",
        extra_vars=PRED_VARS_UNCERTAINTY,
    ))

    # ── Prévisions J
    features.update(extract_pred_vector(
        meteo_utc, day_j,
        horizon_func=get_correct_horizon_j,
        prefix="predJ",
    ))

    # ── Features PV agrégées J+1
    prefix_pv = "pred"
    pv_24h = [features.get(f"{prefix_pv}_pv_MW_t{h:02d}") for h in range(24)]
    pv_clean = [v for v in pv_24h if v is not None]
    features[f"{prefix_pv}_pv_total"] = sum(pv_clean) if pv_clean else None

    pv_day = [features.get(f"{prefix_pv}_pv_MW_t{h:02d}") for h in range(6, 20)]
    pv_day_clean = [v for v in pv_day if v is not None]
    features[f"{prefix_pv}_pv_day"] = sum(pv_day_clean) if pv_day_clean else None

    irr_day_vals = []
    for h in range(6, 20):
        for station in STATIONS:
            v = features.get(f"{prefix_pv}_glob_rad_{station}_t{h:02d}")
            if v is not None:
                irr_day_vals.append(v)
    features[f"{prefix_pv}_glob_rad_mean_day"] = (
        sum(irr_day_vals) / len(irr_day_vals) if irr_day_vals else None
    )

    for h in range(24):
        station_vals = []
        for station in STATIONS:
            v = features.get(f"{prefix_pv}_glob_rad_{station}_t{h:02d}")
            if v is not None:
                station_vals.append(v)
        if len(station_vals) >= 2:
            mean_v = sum(station_vals) / len(station_vals)
            var_v = sum((x - mean_v) ** 2 for x in station_vals) / len(station_vals)
            features[f"{prefix_pv}_glob_rad_std_stations_t{h:02d}"] = var_v ** 0.5
        else:
            features[f"{prefix_pv}_glob_rad_std_stations_t{h:02d}"] = None

    spread_vals = []
    for h in range(6, 20):
        for station in STATIONS:
            q90 = features.get(f"{prefix_pv}_glob_rad_q90_{station}_t{h:02d}")
            q10 = features.get(f"{prefix_pv}_glob_rad_q10_{station}_t{h:02d}")
            if q90 is not None and q10 is not None:
                spread_vals.append(q90 - q10)
    features["pred_glob_rad_spread_day"] = (
        sum(spread_vals) / len(spread_vals) if spread_vals else None
    )

    # ── Calendaire
    doy = target_date.timetuple().tm_yday
    features["dayofweek"]  = target_date.weekday()
    features["month"]      = target_date.month
    features["is_weekend"] = int(target_date.weekday() >= 5)
    features["is_holiday"] = int(target_date in FERIES)
    features["is_ramadan"] = int(is_ramadan(target_date))

    # ── Proxy puissance PV installée
    features["pv_capacity_MWp"] = _get_pv_capacity_proxy(target_date)

    # ── NOUVEAU v11: PV yield ratio glissant (proxy croissance PV via données Oiken)
    features.update(compute_pv_yield_ratios(oiken, meteo_zurich, target_date))

    # ── is_ramadan_hour
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

    print(f"\n=== Construction features v11 : {first_day} → {last_day} ({len(all_dates)} jours) ===")

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

    X.write_parquet(OUT / "X_features_v11.parquet")
    Y.write_parquet(OUT / "Y_target_v11.parquet")
    B.write_parquet(OUT / "B_baseline_v11.parquet")

    print(f"\n✓ X_features_v11 : {X.shape[0]} jours × {X.shape[1]} colonnes")
    print(f"✓ Y_target_v11   : {Y.shape[0]} jours × {Y.shape[1]} colonnes")
    print(f"✓ B_baseline_v11 : {B.shape[0]} jours × {B.shape[1]} colonnes")

    # Vérifier les nouvelles features
    for col in ["pv_yield_30j", "pv_yield_90j", "solar_remote_max_30j", "solar_remote_max_90j", "pv_capacity_MWp"]:
        if col in X.columns:
            s = X[col].drop_nulls()
            print(f"  {col}: min={s.min():.2f}, max={s.max():.2f}, mean={s.mean():.2f}, nulls={X[col].null_count()}")

    print(f"\n✓ Sauvegardé dans : {OUT}")


if __name__ == "__main__":
    main()