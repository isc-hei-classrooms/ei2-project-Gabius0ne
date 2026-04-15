"""
pipeline_features_intraday_v1.py
================================
Pipeline features pour le modèle INTRADAY.

Différence vs day-ahead :
  - Day-ahead : 1 sample par jour (lancé à 11h, prédit J+1)
  - Intraday  : 24 samples par jour (1 par heure de lancement),
                chacun avec 8 horizons cibles (H+15min à H+2h)

Données disponibles à l'instant H :
  - Load Oiken : J-1 complet à J-7 (livré à 2h le matin J)
  - PV mesuré (sans remote) : jusqu'à H-15min, par pas de 15 min
  - Irradiance + Température mesurées : jusqu'à H-20min
  - Prévisions météo rafraîchies : horizons h1, h2, h3 émises à H-1h

Sorties :
  DATA/processed/X_intraday_v1.parquet   (date + launch_hour + features)
  DATA/processed/Y_intraday_v1.parquet   (date + launch_hour + 8 cibles)
  DATA/processed/B_intraday_v1.parquet   (date + launch_hour + 8 baselines Oiken)
"""

import math
import polars as pl
import numpy as np
from pathlib import Path
from datetime import timedelta, date, datetime

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
BASE  = Path(__file__).resolve().parents[2] / "DATA"
CSV   = BASE / "oiken-data.csv"
METEO = BASE / "meteo_multistation_v5.parquet"
OUT   = BASE / "processed"
OUT.mkdir(parents=True, exist_ok=True)

# Horizons en pas de 15 min : H+15min à H+2h
HORIZONS = list(range(1, 9))   # k = 1..8

# Heures de lancement (UTC, toutes les heures rondes)
LAUNCH_HOURS_UTC = list(range(24))

# Sources PV live (sans remote qui arrive à 2h le lendemain)
PV_LIVE_SOURCES = ["solar_central_valais", "solar_sion", "solar_sierre"]
PV_ALL_SOURCES  = PV_LIVE_SOURCES + ["solar_remote"]

STATIONS_ALL = [
    "Pully", "Sion", "Visp", "Montana",
    "Col_du_Grand_St-Bernard", "Les_Attelas",
]
STATIONS_LIVE = ["Sion", "Visp", "Pully"]

REAL_METEO_VARS = ["temp_2m", "glob_rad", "pressure", "relhum_2m"]

PRED_VARS_INTRADAY = ["temp", "glob_rad"]

LOAD_HISTORY_DAYS = list(range(1, 8))   # J-1 à J-7

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


def is_ramadan(d: date) -> bool:
    for start, end in RAMADAN_DATES.values():
        if start <= d <= end:
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
    return 55.0 * national_gw / 4.65


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
    df = df.with_columns(pl.col("timestamp").dt.convert_time_zone("UTC").alias("ts_utc"))
    print(f"  Oiken : {len(df):,} lignes | {df['timestamp'][0]} → {df['timestamp'][-1]}")
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


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def get_day_slice_zurich(df: pl.DataFrame, day: date) -> pl.DataFrame:
    start = pl.datetime(day.year, day.month, day.day, 0, 0, 0, time_zone="Europe/Zurich")
    end   = pl.datetime(day.year, day.month, day.day, 23, 59, 59, time_zone="Europe/Zurich")
    return df.filter((pl.col("timestamp") >= start) & (pl.col("timestamp") <= end))


def get_window_utc(df: pl.DataFrame, start_utc: datetime, end_utc: datetime,
                   ts_col: str = "ts_utc") -> pl.DataFrame:
    start_dt = pl.datetime(start_utc.year, start_utc.month, start_utc.day,
                           start_utc.hour, start_utc.minute, start_utc.second,
                           time_zone="UTC")
    end_dt = pl.datetime(end_utc.year, end_utc.month, end_utc.day,
                         end_utc.hour, end_utc.minute, end_utc.second,
                         time_zone="UTC")
    return df.filter((pl.col(ts_col) >= start_dt) & (pl.col(ts_col) <= end_dt))


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


def get_value_at_utc(df: pl.DataFrame, t_utc: datetime, col: str,
                     ts_col: str = "ts_utc", tolerance_minutes: int = 8) -> float | None:
    """Récupère la valeur de col au timestamp UTC le plus proche de t_utc."""
    if col not in df.columns:
        return None
    tol = timedelta(minutes=tolerance_minutes)
    win_start = pl.datetime(
        (t_utc - tol).year, (t_utc - tol).month, (t_utc - tol).day,
        (t_utc - tol).hour, (t_utc - tol).minute, 0, time_zone="UTC"
    )
    win_end = pl.datetime(
        (t_utc + tol).year, (t_utc + tol).month, (t_utc + tol).day,
        (t_utc + tol).hour, (t_utc + tol).minute, 0, time_zone="UTC"
    )
    sub = df.filter((pl.col(ts_col) >= win_start) & (pl.col(ts_col) <= win_end))
    if len(sub) == 0:
        return None
    vals = sub[col].drop_nulls()
    if len(vals) == 0:
        return None
    return float(vals[0])


# ─────────────────────────────────────────────
# CONSTRUCTION FEATURES POUR UN SAMPLE
# ─────────────────────────────────────────────

def build_intraday_features(
    target_date: date,
    launch_hour_utc: int,
    oiken: pl.DataFrame,
    meteo_utc: pl.DataFrame,
    meteo_zurich: pl.DataFrame,
    real_cols: list[str],
) -> dict | None:
    H_utc = datetime(target_date.year, target_date.month, target_date.day,
                     launch_hour_utc, 0, 0)
    day_jm1 = target_date - timedelta(days=1)
    features = {}

    # ── Load historique J-1 à J-7
    for delta in LOAD_HISTORY_DAYS:
        day_past = target_date - timedelta(days=delta)
        label = f"jm{delta}"
        oiken_past = get_day_slice_zurich(oiken, day_past)
        if len(oiken_past) >= 90:
            features.update(series_stats(oiken_past["load"], f"load_{label}"))
            features.update(hourly_profile(oiken_past, "load", f"load_{label}"))
        else:
            for k in ["mean", "max", "min", "std"]:
                features[f"load_{label}_{k}"] = None
            for h in range(24):
                features[f"load_{label}_h{h:02d}"] = None

    # ── PV mesuré J-1 (toutes sources)
    oiken_jm1 = get_day_slice_zurich(oiken, day_jm1)
    for col in PV_ALL_SOURCES:
        if col in oiken_jm1.columns:
            features[f"{col}_jm1_total"] = float(oiken_jm1[col].sum())
            features.update(hourly_profile(oiken_jm1, col, f"{col}_jm1"))
        else:
            features[f"{col}_jm1_total"] = None
            for h in range(24):
                features[f"{col}_jm1_h{h:02d}"] = None

    # ── Météo réelle J-1
    meteo_jm1 = get_day_slice_zurich(meteo_zurich, day_jm1)
    for col in real_cols:
        if col in meteo_jm1.columns:
            vals = meteo_jm1[col].drop_nulls()
            if len(vals) > 0:
                features[f"rmet_jm1_{col}_mean"] = float(vals.mean())
                features[f"rmet_jm1_{col}_max"]  = float(vals.max())
                features[f"rmet_jm1_{col}_min"]  = float(vals.min())
            else:
                features[f"rmet_jm1_{col}_mean"] = None
                features[f"rmet_jm1_{col}_max"]  = None
                features[f"rmet_jm1_{col}_min"]  = None

    # ── PV yield / capacity
    features["pv_capacity_MWp"] = _get_pv_capacity_proxy(target_date)

    for window_days, label in [(30, "30j"), (90, "90j")]:
        window_start = day_jm1 - timedelta(days=window_days - 1)
        start_dt = pl.datetime(window_start.year, window_start.month, window_start.day,
                               0, 0, 0, time_zone="Europe/Zurich")
        end_dt = pl.datetime(day_jm1.year, day_jm1.month, day_jm1.day,
                             23, 59, 59, time_zone="Europe/Zurich")
        oiken_w = oiken.filter(
            (pl.col("timestamp") >= start_dt) & (pl.col("timestamp") <= end_dt)
        )
        meteo_w = meteo_zurich.filter(
            (pl.col("timestamp") >= start_dt) & (pl.col("timestamp") <= end_dt)
        )
        remote_max = oiken_w["solar_remote"].drop_nulls().max() if "solar_remote" in oiken_w.columns and len(oiken_w) > 0 else None
        glob_max = meteo_w["glob_rad_Sion"].drop_nulls().max() if "glob_rad_Sion" in meteo_w.columns and len(meteo_w) > 0 else None
        remote_max = float(remote_max) if remote_max is not None else None
        glob_max = float(glob_max) if glob_max is not None else None
        if remote_max is not None and glob_max is not None and glob_max > 10:
            features[f"pv_yield_{label}"] = remote_max / glob_max
        else:
            features[f"pv_yield_{label}"] = None
        features[f"solar_remote_max_{label}"] = remote_max

    # ── PV mesuré récent (lags H-15/-30/-45/-60 min)
    pv_lag_offsets_min = [15, 30, 45, 60]
    for src in PV_LIVE_SOURCES:
        for off in pv_lag_offsets_min:
            t = H_utc - timedelta(minutes=off)
            features[f"{src}_lag{off}min"] = get_value_at_utc(oiken, t, src)

        # Moyenne [H-1h, H-15min]
        win_start = H_utc - timedelta(minutes=60)
        win_end   = H_utc - timedelta(minutes=15)
        sub = get_window_utc(oiken, win_start, win_end)
        if src in sub.columns and len(sub) > 0:
            v = sub[src].drop_nulls()
            features[f"{src}_recent_mean"] = float(v.mean()) if len(v) > 0 else None
        else:
            features[f"{src}_recent_mean"] = None

    # Total des 3 sources à H-15min
    total_15 = 0.0
    n_valid = 0
    for src in PV_LIVE_SOURCES:
        v = features.get(f"{src}_lag15min")
        if v is not None:
            total_15 += v
            n_valid += 1
    features["pv_live_total_lag15min"] = total_15 if n_valid == 3 else None

    # ── Irradiance mesurée récente
    irr_lag_offsets_min = [20, 50, 80]
    for st in STATIONS_LIVE:
        col = f"glob_rad_{st}"
        for off in irr_lag_offsets_min:
            t = H_utc - timedelta(minutes=off)
            features[f"{col}_lag{off}min"] = get_value_at_utc(meteo_utc, t, col, "timestamp")

        win_start = H_utc - timedelta(minutes=80)
        win_end   = H_utc - timedelta(minutes=20)
        sub = get_window_utc(meteo_utc, win_start, win_end, "timestamp")
        if col in sub.columns and len(sub) > 0:
            v = sub[col].drop_nulls()
            features[f"{col}_recent_mean"] = float(v.mean()) if len(v) > 0 else None
        else:
            features[f"{col}_recent_mean"] = None

    # ── Température mesurée
    for st in STATIONS_LIVE:
        col = f"temp_2m_{st}"
        t = H_utc - timedelta(minutes=20)
        features[f"{col}_lag20min"] = get_value_at_utc(meteo_utc, t, col, "timestamp")
        win_start = H_utc - timedelta(minutes=120)
        win_end   = H_utc - timedelta(minutes=20)
        sub = get_window_utc(meteo_utc, win_start, win_end, "timestamp")
        if col in sub.columns and len(sub) > 0:
            v = sub[col].drop_nulls()
            features[f"{col}_recent_mean"] = float(v.mean()) if len(v) > 0 else None
        else:
            features[f"{col}_recent_mean"] = None

    # ── Prévisions météo rafraîchies (h1, h2, h3 émises à H-1h)
    # pred_{var}_h{h}_{station} pour timestamp T = prévision émise à T-h
    # Donc à l'instant H, prévisions émises à H-1h dispo : cibles H, H+1h, H+2h
    for var in PRED_VARS_INTRADAY:
        for st in STATIONS_LIVE:
            for h in [1, 2, 3]:
                target_t = H_utc + timedelta(hours=h - 1)
                col_pred = f"pred_{var}_h{h}_{st}"
                features[f"pred_{var}_{st}_h{h}_fresh"] = get_value_at_utc(
                    meteo_utc, target_t, col_pred, "timestamp", tolerance_minutes=5
                )

    # ── Calendaire
    doy = target_date.timetuple().tm_yday
    features["dayofweek"]  = target_date.weekday()
    features["month"]      = target_date.month
    features["is_weekend"] = int(target_date.weekday() >= 5)
    features["is_holiday"] = int(target_date in FERIES)
    features["is_ramadan"] = int(is_ramadan(target_date))

    features["sin_dow"]   = math.sin(2 * math.pi * target_date.weekday() / 7)
    features["cos_dow"]   = math.cos(2 * math.pi * target_date.weekday() / 7)
    features["sin_month"] = math.sin(2 * math.pi * (target_date.month - 1) / 12)
    features["cos_month"] = math.cos(2 * math.pi * (target_date.month - 1) / 12)
    features["sin_doy"]   = math.sin(2 * math.pi * doy / 365)
    features["cos_doy"]   = math.cos(2 * math.pi * doy / 365)

    # ── Contexte du lancement
    features["launch_hour_utc"]    = launch_hour_utc
    features["launch_hour_sin"]    = math.sin(2 * math.pi * launch_hour_utc / 24)
    features["launch_hour_cos"]    = math.cos(2 * math.pi * launch_hour_utc / 24)
    features["target_step_in_day"] = launch_hour_utc * 4

    return features


# ─────────────────────────────────────────────
# CIBLES + BASELINE
# ─────────────────────────────────────────────

def get_targets_and_baseline(
    target_date: date,
    launch_hour_utc: int,
    oiken: pl.DataFrame,
) -> tuple[list, list]:
    """Retourne (targets[8], baselines[8]) pour les horizons k=1..8."""
    H_utc = datetime(target_date.year, target_date.month, target_date.day,
                     launch_hour_utc, 0, 0)
    targets = []
    baselines = []
    for k in HORIZONS:
        t = H_utc + timedelta(minutes=15 * k)
        v_target = get_value_at_utc(oiken, t, "load", "ts_utc", tolerance_minutes=8)
        v_base   = get_value_at_utc(oiken, t, "load_forecast_oiken", "ts_utc", tolerance_minutes=8)
        targets.append(v_target)
        baselines.append(v_base)
    return targets, baselines


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

    print(f"\n=== Construction features intraday v1 ===")
    print(f"  Période : {first_day} → {last_day} ({len(all_dates)} jours)")
    print(f"  Lancements/jour : {len(LAUNCH_HOURS_UTC)}")
    print(f"  Horizons par lancement : {len(HORIZONS)}")
    print(f"  Samples attendus : ~{len(all_dates) * len(LAUNCH_HOURS_UTC):,}")

    rows_X, rows_Y, rows_B = [], [], []
    skipped = 0
    skipped_no_target = 0

    for i, target_date in enumerate(all_dates):
        if i % 50 == 0:
            print(f"  {i}/{len(all_dates)} — {target_date} (samples: {len(rows_X)}, skipped: {skipped})")

        for launch_hour in LAUNCH_HOURS_UTC:
            feats = build_intraday_features(
                target_date, launch_hour,
                oiken, meteo_utc, meteo_zurich, real_cols
            )
            if feats is None:
                skipped += 1
                continue

            targets, baselines = get_targets_and_baseline(target_date, launch_hour, oiken)
            if all(t is None for t in targets):
                skipped_no_target += 1
                continue

            feats["date"] = str(target_date)
            feats["launch_hour"] = launch_hour
            rows_X.append(feats)

            row_y = {"date": str(target_date), "launch_hour": launch_hour}
            row_b = {"date": str(target_date), "launch_hour": launch_hour}
            for k, (t, b) in enumerate(zip(targets, baselines), start=1):
                row_y[f"y_h{k}"] = t
                row_b[f"b_h{k}"] = b
            rows_Y.append(row_y)
            rows_B.append(row_b)

    print(f"\n  {len(rows_X)} samples valides | {skipped} skipped (features) | {skipped_no_target} skipped (no target)")

    X = pl.DataFrame(rows_X)
    id_cols = ["date", "launch_hour"]
    feat_cols = [c for c in X.columns if c not in id_cols]
    X = X.select(id_cols + feat_cols).with_columns(
        pl.col("date").str.strptime(pl.Date, "%Y-%m-%d")
    )

    Y = pl.DataFrame(rows_Y).with_columns(
        pl.col("date").str.strptime(pl.Date, "%Y-%m-%d")
    )
    B = pl.DataFrame(rows_B).with_columns(
        pl.col("date").str.strptime(pl.Date, "%Y-%m-%d")
    )

    X.write_parquet(OUT / "X_intraday_v1.parquet")
    Y.write_parquet(OUT / "Y_intraday_v1.parquet")
    B.write_parquet(OUT / "B_intraday_v1.parquet")

    print(f"\n✓ X_intraday_v1 : {X.shape[0]} samples × {X.shape[1]} colonnes")
    print(f"✓ Y_intraday_v1 : {Y.shape[0]} samples × {Y.shape[1]} colonnes")
    print(f"✓ B_intraday_v1 : {B.shape[0]} samples × {B.shape[1]} colonnes")

    check_cols = [
        "solar_central_valais_lag15min", "solar_sion_lag15min",
        "glob_rad_Sion_lag20min", "temp_2m_Sion_lag20min",
        "pred_glob_rad_Sion_h1_fresh", "pred_temp_Sion_h2_fresh",
        "launch_hour_utc",
    ]
    for col in check_cols:
        if col in X.columns:
            s = X[col].drop_nulls()
            if len(s) > 0 and s.dtype in [pl.Float64, pl.Float32]:
                print(f"  {col}: min={s.min():.2f}, max={s.max():.2f}, mean={s.mean():.2f}, nulls={X[col].null_count()}")

    print(f"\n✓ Sauvegardé dans : {OUT}")


if __name__ == "__main__":
    main()