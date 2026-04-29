"""
pipeline_features_intraday_v3.py
================================
Pipeline features INTRADAY v3 — architecture 96 modèles.

CHANGEMENT v2 → v3
-------------------
v2 : 1 sample par (date, launch_hour) × 8 horizons → 8 modèles
v3 : 1 sample par (date, target_step) → 96 modèles (comme day-ahead)

Pour chaque pas cible t (0-95), on détermine l'heure de lancement
la plus récente H = floor(t * 15 / 60) et on construit :
  - Les features v13 day-ahead complètes (identiques pour tous les pas d'un jour)
  - Les features fraîches basées sur le lancement H (varient par pas)
  - Les prévisions météo aux horizons les plus courts disponibles à H

Résultat : chaque pas t a strictement plus d'information que le day-ahead.
Le modèle pour t=56 (14h00) a accès au PV mesuré jusqu'à 13h45, à
l'irradiance jusqu'à 13h40, aux prévisions h1-h3 émises à 13h00.

Format de sortie IDENTIQUE au day-ahead :
  X : (n_jours, n_features) — 1 ligne par jour, mais features dépendantes du pas t
      → en pratique on génère 96 parquets X_intraday_v3_t{000..095}.parquet
      OU un seul parquet avec toutes les features (historiques communes + fraîches par pas)

Approche choisie : un seul parquet X avec les features historiques (communes à tous les pas)
+ 96 colonnes de "features fraîches par pas" sérialisées.
MAIS c'est trop complexe. On fait plus simple :

  X_intraday_v3.parquet : (n_jours × 96, n_features)
    Chaque ligne = (date, target_step, features_historiques + features_fraîches_pour_ce_pas)
    Le training script filtre par target_step pour entraîner chaque modèle.

Sorties :
  DATA/processed/X_intraday_v3.parquet   (date + target_step + features)
  DATA/processed/Y_intraday_v3.parquet   (date + target_step + y_target)
  DATA/processed/B_intraday_v3.parquet   (date + target_step + b_baseline)
"""

import math
import polars as pl
import numpy as np
import zoneinfo
from pathlib import Path
from datetime import timedelta, date, datetime

BASE  = Path(__file__).resolve().parents[2] / "DATA"
CSV   = BASE / "oiken-golden-dataset.csv"
METEO = BASE / "meteo_multistationGOLDEN.parquet"
OUT   = BASE / "processed"
OUT.mkdir(parents=True, exist_ok=True)

N_STEPS = 96

PV_LIVE_SOURCES = ["solar_central_valais", "solar_sion", "solar_sierre"]
PV_ALL_SOURCES  = PV_LIVE_SOURCES + ["solar_remote"]

STATIONS_ALL = ["Pully", "Sion", "Visp", "Montana",
                "Col_du_Grand_St-Bernard", "Les_Attelas"]
STATIONS_PLAINE = ["Sion", "Visp", "Pully"]
STATIONS_WIND = ["Sion", "Pully"]
STATIONS_FRESH = ["Sion", "Visp", "Montana"]

REAL_METEO_VARS = ["temp_2m", "glob_rad", "pressure", "relhum_2m"]
FRESH_METEO_VARS = ["glob_rad", "temp_2m", "pressure", "relhum_2m",
                    "wind_speed", "precip", "sunshine"]

PRED_VARS_JP1 = ["temp", "glob_rad", "pressure", "relhum", "precip", "sunshine"]
HOURS_DIURNAL = list(range(6, 20))
LOAD_HISTORY_DAYS = list(range(2, 9))
PROD_COLS_LIVE = ["solar_central_valais", "solar_sion", "solar_sierre"]

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

VACANCES_SCOLAIRES = [
    (date(2022, 10, 12), date(2022, 10, 23)), (date(2022, 12, 23), date(2023,  1,  8)),
    (date(2023,  2, 17), date(2023,  2, 26)), (date(2023,  4,  6), date(2023,  4, 16)),
    (date(2023,  6, 24), date(2023,  8, 16)), (date(2023, 10, 11), date(2023, 10, 22)),
    (date(2023, 12, 22), date(2024,  1,  7)), (date(2024,  2,  9), date(2024,  2, 18)),
    (date(2024,  3, 28), date(2024,  4,  7)), (date(2024,  6, 22), date(2024,  8, 14)),
    (date(2024, 10, 16), date(2024, 10, 27)), (date(2024, 12, 20), date(2025,  1,  6)),
    (date(2025,  2, 28), date(2025,  3,  9)), (date(2025,  4, 17), date(2025,  4, 27)),
    (date(2025,  6, 28), date(2025,  8, 17)), (date(2025, 10, 18), date(2025, 10, 26)),
    (date(2025, 12, 20), date(2026,  1,  4)),
]


def is_ramadan(d): return any(s <= d <= e for s,e in RAMADAN_DATES.values())
def is_school_holiday(d): return any(s <= d <= e for s,e in VACANCES_SCOLAIRES)

def is_bridge_day(d):
    if d.weekday() >= 5 or d in FERIES or is_school_holiday(d): return False
    y, t = d - timedelta(days=1), d + timedelta(days=1)
    off = lambda x: x.weekday() >= 5 or x in FERIES
    if d.weekday() == 4 and off(y): return True
    if d.weekday() == 0 and off(t): return True
    if off(y) and off(t): return True
    return False

def _get_pv_capacity_proxy(d):
    from datetime import date as _d
    anchors = [(_d(2022,10,1),4.65),(_d(2022,12,31),4.65),(_d(2023,12,31),6.20),
               (_d(2024,12,31),8.00),(_d(2025,12,31),9.51)]
    if d <= anchors[0][0]: ng = anchors[0][1]
    elif d >= anchors[-1][0]: ng = anchors[-1][1]
    else:
        for i in range(len(anchors)-1):
            d1,v1 = anchors[i]; d2,v2 = anchors[i+1]
            if d1 <= d <= d2: ng = v1 + (d-d1).days/max((d2-d1).days,1)*(v2-v1); break
    return 55.0 * ng / 4.65

def get_best_horizon(h_target_utc, launch_hour_utc):
    """Plus petit horizon h tel que la prévision ait été émise au moins 1h
    avant le lancement (marge pour la latence de publication MeteoSuisse).
    Émission = h_target - h. Condition : (h_target - h) <= launch_hour - 1."""
    mod = h_target_utc % 3
    valid = [h for h in HORIZONS_BY_MOD[mod] if (h_target_utc - h) <= (launch_hour_utc - 1) and h > 0]
    return min(valid) if valid else None

def get_dayahead_horizon(h_utc):
    mod = h_utc % 3
    valid = [h for h in HORIZONS_BY_MOD[mod] if h >= h_utc + 14]
    return min(valid) if valid else None


# ─── Chargement + indexation (identique v2) ───

def load_oiken(path):
    df = pl.read_csv(path, try_parse_dates=False, null_values=["#N/A","N/A","NA",""],
        schema_overrides={"central valais solar production [kWh]":pl.Float64,
            "sion area solar production [kWh]":pl.Float64,
            "sierre area production [kWh]":pl.Float64,
            "remote solar production [kWh]":pl.Float64})
    df = df.with_columns(
        pl.col("timestamp").str.strptime(pl.Datetime("us"),"%d.%m.%Y %H:%M")
          .dt.replace_time_zone("Europe/Zurich",ambiguous="earliest",non_existent="null")
    ).rename({"standardised load [-]":"load","standardised forecast load [-]":"load_forecast_oiken",
        "central valais solar production [kWh]":"solar_central_valais",
        "sion area solar production [kWh]":"solar_sion",
        "sierre area production [kWh]":"solar_sierre",
        "remote solar production [kWh]":"solar_remote"}).sort("timestamp")
    df = df.with_columns(pl.col("timestamp").dt.convert_time_zone("UTC").alias("ts_utc"))
    print(f"  Oiken : {len(df):,} lignes")
    return df

def load_meteo(path):
    df = pl.read_parquet(path).sort("timestamp")
    df_z = df.with_columns(pl.col("timestamp").dt.convert_time_zone("Europe/Zurich"))
    all_c = [c for c in df.columns if c != "timestamp"]
    real_cols = [f"{v}_{s}" for v in REAL_METEO_VARS for s in STATIONS_ALL if f"{v}_{s}" in all_c]
    print(f"  Météo : {len(df):,} lignes | {len(real_cols)} mesures réelles")
    return df, df_z, real_cols

def build_index_oiken_utc(oiken, cols):
    print("  Indexation Oiken UTC...")
    idx = {}
    ts = oiken["ts_utc"].to_list()
    cd = {c: oiken[c].to_list() for c in cols if c in oiken.columns}
    for i, t in enumerate(ts):
        if t is None: continue
        idx[t.replace(tzinfo=None)] = {c: (float(vals[i]) if vals[i] is not None else None) for c, vals in cd.items()}
    print(f"    {len(idx):,} entrées")
    return idx

def build_index_oiken_zurich(oiken):
    print("  Indexation Oiken Zurich...")
    idx = {}; counts = {}
    ts = oiken["timestamp"].to_list(); load_l = oiken["load"].to_list()
    pv = {c: oiken[c].to_list() for c in PV_ALL_SOURCES if c in oiken.columns}
    for i, t in enumerate(ts):
        if t is None: continue
        d, h = t.date(), t.hour
        if d not in idx: idx[d] = {}
        if h not in idx[d]: idx[d][h] = {"load":[], **{c:[] for c in pv}}
        if load_l[i] is not None: idx[d][h]["load"].append(float(load_l[i]))
        for c, vals in pv.items():
            if vals[i] is not None: idx[d][h][c].append(float(vals[i]))
    for d in idx: counts[d] = sum(len(idx[d].get(h,{}).get("load",[])) for h in idx[d])
    print(f"    {len(idx):,} jours")
    return idx, counts

def build_index_meteo_utc(meteo, cols):
    print("  Indexation Météo UTC...")
    idx = {}
    ts = meteo["timestamp"].to_list()
    cd = {c: meteo[c].to_list() for c in cols if c in meteo.columns}
    for i, t in enumerate(ts):
        if t is None: continue
        idx[t.replace(tzinfo=None)] = {c: (float(vals[i]) if vals[i] is not None else None) for c, vals in cd.items()}
    print(f"    {len(idx):,} entrées")
    return idx

def build_index_meteo_zurich(meteo_z, real_cols):
    print("  Indexation Météo Zurich...")
    idx = {}
    ts = meteo_z["timestamp"].to_list()
    cd = {c: meteo_z[c].to_list() for c in real_cols if c in meteo_z.columns}
    for i, t in enumerate(ts):
        if t is None: continue
        d = t.date()
        if d not in idx: idx[d] = {c:[] for c in cd}
        for c, vals in cd.items():
            if vals[i] is not None: idx[d][c].append(float(vals[i]))
    print(f"    {len(idx):,} jours")
    return idx

def fl(idx, t, col):
    r = idx.get(t)
    return r.get(col) if r else None

def fast_day_profile(oi, day, col, prefix):
    r = {}; dd = oi.get(day)
    for h in range(24):
        if dd and h in dd and col in dd[h] and dd[h][col]:
            r[f"{prefix}_h{h:02d}"] = sum(dd[h][col])/len(dd[h][col])
        else: r[f"{prefix}_h{h:02d}"] = None
    return r

def fast_day_stats(oi, day, col, prefix):
    dd = oi.get(day)
    if not dd: return {f"{prefix}_{k}":None for k in ["mean","max","min","std"]}
    vals = []
    for h in range(24):
        if h in dd and col in dd[h]: vals.extend(dd[h][col])
    if not vals: return {f"{prefix}_{k}":None for k in ["mean","max","min","std"]}
    a = np.array(vals)
    return {f"{prefix}_mean":float(a.mean()),f"{prefix}_max":float(a.max()),
            f"{prefix}_min":float(a.min()),f"{prefix}_std":float(a.std()) if len(a)>1 else 0.0}

def fast_day_sum(oi, day, col):
    dd = oi.get(day)
    if not dd: return None
    t = 0.0; n = 0
    for h in range(24):
        if h in dd and col in dd[h]: t += sum(dd[h][col]); n += len(dd[h][col])
    return t if n > 0 else None

def fast_meteo_stats(mi, day, real_cols, prefix):
    f = {}; dd = mi.get(day)
    for c in real_cols:
        if not dd or c not in dd or not dd[c]:
            f[f"{prefix}_{c}_mean"]=None; f[f"{prefix}_{c}_max"]=None; f[f"{prefix}_{c}_min"]=None
        else:
            a = np.array(dd[c])
            f[f"{prefix}_{c}_mean"]=float(a.mean()); f[f"{prefix}_{c}_max"]=float(a.max()); f[f"{prefix}_{c}_min"]=float(a.min())
    return f


# ─────────────────────────────────────────────
# FEATURES HISTORIQUES (communes à tous les pas d'un jour)
# ─────────────────────────────────────────────

def build_historical_features(
    target_date, oiken_day_idx, oiken_day_counts, meteo_day_idx, real_cols
):
    """Features identiques au day-ahead v13. Calculées une seule fois par jour."""
    day_jm2 = target_date - timedelta(days=2)
    day_j = target_date
    features = {}

    # Load J-2 à J-8
    for delta in LOAD_HISTORY_DAYS:
        day_past = target_date - timedelta(days=delta)
        label = f"jm{delta-1}"
        if oiken_day_counts.get(day_past, 0) >= 90:
            features.update(fast_day_stats(oiken_day_idx, day_past, "load", f"load_{label}"))
            features.update(fast_day_profile(oiken_day_idx, day_past, "load", f"load_{label}"))
        else:
            for k in ["mean","max","min","std"]: features[f"load_{label}_{k}"] = None
            for h in range(24): features[f"load_{label}_h{h:02d}"] = None

    # PV mesuré J-2 (profil + total)
    for col in PV_ALL_SOURCES:
        features[f"{col}_jm1_total"] = fast_day_sum(oiken_day_idx, day_jm2, col)
        features.update(fast_day_profile(oiken_day_idx, day_jm2, col, f"{col}_jm1"))
    # NOTE : j_morning_total supprimé (leaking pour les lancements avant 10h Zurich).
    # L'info PV matin est couverte par les lags fraîches (H-15 à H-4h).

    # Météo réelle J-2 complet
    features.update(fast_meteo_stats(meteo_day_idx, day_jm2, real_cols, "rmet_jm1"))
    # NOTE : rmet_jmorn supprimé (leaking pour les lancements avant 10h Zurich).
    # L'info météo matin est couverte par les lags fraîches (H-20 à H-4h).

    # PV capacity + yield
    features["pv_capacity_MWp"] = _get_pv_capacity_proxy(target_date)
    for wdays, label in [(30,"30j"),(90,"90j")]:
        ws = day_jm2 - timedelta(days=wdays-1)
        rmax = None; gmax = None
        d = ws
        while d <= day_jm2:
            dd_o = oiken_day_idx.get(d)
            if dd_o:
                for h in range(24):
                    hd = dd_o.get(h)
                    if hd and "solar_remote" in hd:
                        for v in hd["solar_remote"]:
                            if rmax is None or v > rmax: rmax = v
            dd_m = meteo_day_idx.get(d)
            if dd_m and "glob_rad_Sion" in dd_m:
                for v in dd_m["glob_rad_Sion"]:
                    if gmax is None or v > gmax: gmax = v
            d += timedelta(days=1)
        if rmax is not None and gmax is not None and gmax > 10:
            features[f"pv_yield_{label}"] = rmax / gmax
        else:
            features[f"pv_yield_{label}"] = None
        features[f"solar_remote_max_{label}"] = rmax

    # Calendaire
    doy = target_date.timetuple().tm_yday
    features["dayofweek"] = target_date.weekday()
    features["month"] = target_date.month
    features["is_weekend"] = int(target_date.weekday() >= 5)
    features["is_holiday"] = int(target_date in FERIES)
    features["is_school_holiday"] = int(is_school_holiday(target_date))
    features["is_bridge_day"] = int(is_bridge_day(target_date))
    features["is_ramadan"] = int(is_ramadan(target_date))
    rh = set(list(range(0,6))+list(range(20,24))) if is_ramadan(target_date) else set()
    for h in range(24): features[f"is_ramadan_h{h:02d}"] = int(h in rh)
    features["sin_dow"] = math.sin(2*math.pi*target_date.weekday()/7)
    features["cos_dow"] = math.cos(2*math.pi*target_date.weekday()/7)
    features["sin_month"] = math.sin(2*math.pi*(target_date.month-1)/12)
    features["cos_month"] = math.cos(2*math.pi*(target_date.month-1)/12)
    features["sin_doy"] = math.sin(2*math.pi*doy/365)
    features["cos_doy"] = math.cos(2*math.pi*doy/365)

    return features


# ─────────────────────────────────────────────
# FEATURES DÉPENDANTES DU PAS (fraîches + prévisions courtes)
# ─────────────────────────────────────────────

def build_step_features(
    target_date, target_step, launch_utc_dt,
    oiken_utc_idx, meteo_utc_idx, hist_features,
):
    """
    Features spécifiques au pas cible t, basées sur le lancement H.
    launch_utc_dt : datetime naïf en UTC du lancement.
    """
    H_utc = launch_utc_dt
    launch_hour_utc = H_utc.hour
    features = {}

    # ── Prévisions météo avec horizons courts (pas day-ahead)
    # Les prévisions sont indexées en UTC dans meteo_utc_idx.
    # On utilise la date UTC de H_utc (peut différer de target_date pour steps proches de minuit)
    utc_date = H_utc.date()
    for var in PRED_VARS_JP1:
        if var == "glob_rad": continue
        for station in STATIONS_ALL:
            for h_utc in range(24):
                horizon = get_best_horizon(h_utc, launch_hour_utc)
                if horizon is None: horizon = get_dayahead_horizon(h_utc)
                if horizon is None:
                    features[f"pred_{var}_{station}_t{h_utc:02d}"] = None; continue
                t_target = datetime(utc_date.year, utc_date.month, utc_date.day, h_utc, 0, 0)
                features[f"pred_{var}_{station}_t{h_utc:02d}"] = fl(meteo_utc_idx, t_target, f"pred_{var}_h{horizon}_{station}")

    # glob_rad : stations plaine × heures diurnes
    irr_day_vals = []
    for station in STATIONS_PLAINE:
        for h_utc in HOURS_DIURNAL:
            horizon = get_best_horizon(h_utc, launch_hour_utc)
            if horizon is None: horizon = get_dayahead_horizon(h_utc)
            if horizon is None:
                features[f"pred_glob_rad_{station}_t{h_utc:02d}"] = None; continue
            t_target = datetime(utc_date.year, utc_date.month, utc_date.day, h_utc, 0, 0)
            val = fl(meteo_utc_idx, t_target, f"pred_glob_rad_h{horizon}_{station}")
            features[f"pred_glob_rad_{station}_t{h_utc:02d}"] = val
            if val is not None: irr_day_vals.append(val)
    features["pred_glob_rad_mean_day"] = sum(irr_day_vals)/len(irr_day_vals) if irr_day_vals else None

    # wind_speed
    for station in STATIONS_WIND:
        for h_utc in HOURS_DIURNAL:
            horizon = get_best_horizon(h_utc, launch_hour_utc)
            if horizon is None: horizon = get_dayahead_horizon(h_utc)
            if horizon is None:
                features[f"pred_wind_speed_{station}_t{h_utc:02d}"] = None; continue
            t_target = datetime(utc_date.year, utc_date.month, utc_date.day, h_utc, 0, 0)
            features[f"pred_wind_speed_{station}_t{h_utc:02d}"] = fl(meteo_utc_idx, t_target, f"pred_wind_speed_h{horizon}_{station}")

    # ── Interactions PV (recalculées avec prévisions fraîches)
    pvy30 = hist_features.get("pv_yield_30j")
    pvy90 = hist_features.get("pv_yield_90j")
    for h_utc in HOURS_DIURNAL:
        irr = [features.get(f"pred_glob_rad_{s}_t{h_utc:02d}") for s in STATIONS_PLAINE]
        ic = [v for v in irr if v is not None]
        im = sum(ic)/len(ic) if ic else None
        features[f"pred_pv_adj_30j_t{h_utc:02d}"] = (im*pvy30) if (im and pvy30) else None
        features[f"pred_pv_adj_90j_t{h_utc:02d}"] = (im*pvy90) if (im and pvy90) else None
    for label in ["30j","90j"]:
        dv = [features.get(f"pred_pv_adj_{label}_t{h:02d}") for h in HOURS_DIURNAL]
        dc = [v for v in dv if v is not None]
        features[f"pred_pv_adj_{label}_day"] = sum(dc) if dc else None

    # ── PV mesuré temps réel : 16 pas (H-15min à H-4h)
    for src in PV_LIVE_SOURCES:
        pv_vals = []
        for i in range(1, 17):
            off = i * 15
            v = fl(oiken_utc_idx, H_utc - timedelta(minutes=off), src)
            features[f"{src}_lag{off}min"] = v
            if v is not None: pv_vals.append(v)
        if pv_vals:
            a = np.array(pv_vals)
            features[f"{src}_4h_mean"] = float(a.mean())
            features[f"{src}_4h_max"] = float(a.max())
        else:
            features[f"{src}_4h_mean"] = None
            features[f"{src}_4h_max"] = None
        v15 = features.get(f"{src}_lag15min")
        v30 = features.get(f"{src}_lag30min")
        v60 = features.get(f"{src}_lag60min")
        features[f"{src}_delta_15_30"] = (v15-v30) if (v15 is not None and v30 is not None) else None
        features[f"{src}_delta_15_60"] = (v15-v60) if (v15 is not None and v60 is not None) else None

    tot = 0.0; nv = 0
    for src in PV_LIVE_SOURCES:
        v = features.get(f"{src}_lag15min")
        if v is not None: tot += v; nv += 1
    features["pv_live_total_lag15min"] = tot if nv == 3 else None

    # ── Mesures météo fraîches 4h
    for var in FRESH_METEO_VARS:
        for st in STATIONS_FRESH:
            col = f"{var}_{st}"
            for off in [20, 40, 60, 120, 180, 240]:
                features[f"{col}_lag{off}min"] = fl(meteo_utc_idx, H_utc - timedelta(minutes=off), col)
            wv = []
            for off in range(20, 241, 10):
                v = fl(meteo_utc_idx, H_utc - timedelta(minutes=off), col)
                if v is not None: wv.append(v)
            if wv:
                a = np.array(wv)
                features[f"{col}_4h_mean"] = float(a.mean())
                features[f"{col}_4h_max"] = float(a.max())
                features[f"{col}_4h_min"] = float(a.min())
            else:
                features[f"{col}_4h_mean"] = None
                features[f"{col}_4h_max"] = None
                features[f"{col}_4h_min"] = None
            v20 = features.get(f"{col}_lag20min")
            v60 = features.get(f"{col}_lag60min")
            features[f"{col}_delta_20_60"] = (v20-v60) if (v20 is not None and v60 is not None) else None

    return features


# ─────────────────────────────────────────────
# PIPELINE PRINCIPAL
# ─────────────────────────────────────────────

def main():
    print("=== Chargement ===")
    oiken = load_oiken(CSV)
    meteo_utc, meteo_z, real_cols = load_meteo(METEO)

    print("\n=== Pré-indexation ===")
    oi_utc = build_index_oiken_utc(oiken, ["load","load_forecast_oiken"] + PV_LIVE_SOURCES)
    oi_day, oi_cnt = build_index_oiken_zurich(oiken)

    mc = list(real_cols)
    for var in FRESH_METEO_VARS:
        for st in STATIONS_FRESH:
            c = f"{var}_{st}"
            if c not in mc: mc.append(c)
    for var in PRED_VARS_JP1 + ["wind_speed"]:
        for st in STATIONS_ALL:
            for h in range(1, 37):
                mc.append(f"pred_{var}_h{h}_{st}")
    me_utc = build_index_meteo_utc(meteo_utc, mc)
    me_day = build_index_meteo_zurich(meteo_z, real_cols)

    first_day = oiken["timestamp"].drop_nulls()[0].date() + timedelta(days=9)
    last_day = oiken["timestamp"][-1].date() - timedelta(days=1)
    all_dates = [first_day + timedelta(days=i) for i in range((last_day-first_day).days+1)]

    print(f"\n=== Construction features intraday v3 (96 modèles) ===")
    print(f"  {len(all_dates)} jours × 96 pas = {len(all_dates)*96:,} samples")

    rows_X, rows_Y, rows_B = [], [], []

    # Timezone objects pour conversion Zurich → UTC
    tz_zurich = zoneinfo.ZoneInfo("Europe/Zurich")
    tz_utc = zoneinfo.ZoneInfo("UTC")

    # Clés des features fraîches (pour le fallback si heure invalide)
    # On les récupère du premier appel valide
    step_feat_keys = None

    for i, td in enumerate(all_dates):
        if i % 100 == 0:
            print(f"  {i}/{len(all_dates)} — {td} ({len(rows_X)} samples)")

        # Vérifier que le jour a assez de données
        if oi_cnt.get(td, 0) < 90:
            continue

        # Features historiques (calculées UNE SEULE FOIS par jour)
        hist = build_historical_features(td, oi_day, oi_cnt, me_day, real_cols)

        # Pour chaque pas cible (même référentiel que Y_target_v13 : heure Zurich)
        for t in range(N_STEPS):
            # Step t = (t * 15) minutes après 00:00 Europe/Zurich
            # Il faut convertir en UTC pour les lookups dans les index UTC
            zurich_minute = t * 15
            zurich_hour = zurich_minute // 60
            zurich_min = zurich_minute % 60

            # Construire le timestamp Zurich aware, puis convertir en UTC
            try:
                ts_zurich = datetime(td.year, td.month, td.day,
                                     zurich_hour, zurich_min, 0,
                                     tzinfo=tz_zurich)
                ts_utc = ts_zurich.astimezone(tz_utc)
            except Exception:
                # Heure invalide (passage heure d'été) → skip avec None
                if step_feat_keys:
                    combined = {**hist, **{k: None for k in step_feat_keys}}
                else:
                    combined = {**hist}
                combined["date"] = str(td)
                combined["target_step"] = t
                rows_X.append(combined)
                rows_Y.append({"date": str(td), "target_step": t, "y": None})
                rows_B.append({"date": str(td), "target_step": t, "b": None})
                continue

            # Timestamp UTC naïf pour les lookups
            target_utc_naive = ts_utc.replace(tzinfo=None)

            # Heure de lancement UTC = heure UTC entière du pas cible
            # (arrondi à l'heure inférieure)
            launch_utc_dt = target_utc_naive.replace(minute=0, second=0)

            # Features fraîches + prévisions courtes (en UTC)
            step_feats = build_step_features(td, t, launch_utc_dt, oi_utc, me_utc, hist)

            # Capturer les clés des features fraîches (une seule fois)
            if step_feat_keys is None:
                step_feat_keys = list(step_feats.keys())

            # Combiner historique + fraîches
            combined = {**hist, **step_feats}
            combined["date"] = str(td)
            combined["target_step"] = t
            rows_X.append(combined)

            # Cible : load réel au MÊME instant que Y_target_v13[step t]
            # = load à (t * 15) minutes Zurich = target_utc_naive en UTC
            y_val = fl(oi_utc, target_utc_naive, "load")
            b_val = fl(oi_utc, target_utc_naive, "load_forecast_oiken")
            rows_Y.append({"date": str(td), "target_step": t, "y": y_val})
            rows_B.append({"date": str(td), "target_step": t, "b": b_val})

    print(f"\n  {len(rows_X)} samples générés")

    X = pl.DataFrame(rows_X)
    idc = ["date", "target_step"]
    fc = [c for c in X.columns if c not in idc]
    X = X.select(idc + fc).with_columns(pl.col("date").str.strptime(pl.Date, "%Y-%m-%d"))

    Y = pl.DataFrame(rows_Y).with_columns(pl.col("date").str.strptime(pl.Date, "%Y-%m-%d"))
    B = pl.DataFrame(rows_B).with_columns(pl.col("date").str.strptime(pl.Date, "%Y-%m-%d"))

    X.write_parquet(OUT / "X_intraday_GOLDEN.parquet")
    Y.write_parquet(OUT / "Y_intraday_GOLDEN.parquet")
    B.write_parquet(OUT / "B_intraday_GOLDEN.parquet")

    print(f"\n✓ X : {X.shape[0]:,} × {X.shape[1]}  |  Y : {Y.shape}  |  B : {B.shape}")
    print(f"  Features par sample : {len(fc)}")
    print(f"✓ Sauvegardé dans : {OUT}")


if __name__ == "__main__":
    main()