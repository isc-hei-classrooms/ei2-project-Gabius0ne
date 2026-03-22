"""
pipeline_recup_v5.py
Récupération des données MeteoSuisse depuis InfluxDB.
- Mesures réelles    : toutes variables, 6 stations
- Prévisions         : toutes variables, 6 stations, horizons 13h à 36h
- Colonnes nommées   : {variable}_{station} et pred_{variable}_h{horizon}_{station}
Résultat : meteo_multistation_v3.parquet
"""

import polars as pl
from pathlib import Path
from influxdb_client import InfluxDBClient


# CONFIGURATION

URL    = "https://timeseries.hevs.ch"
TOKEN  = "ixOI8jiwG1nn6a2MaE1pGa8XCiIJ2rqEX6ZCnluhwAyeZcrT6FHoDgnQhNy5k0YmVrk7hZGPpvb_5aaA-ZxhIw=="
ORG    = "HESSOVS"
BUCKET = "MeteoSuisse"
START  = "2022-10-01T00:00:00Z"
STOP   = "2025-10-01T00:00:00Z"

OUTPUT = Path(".")

STATIONS = [
    "Pully",
    "Sion",
    "Visp",
    "Montana",
    "Col du Grand St-Bernard",
    "Les Attelas",
]

# Horizons de prévision retenus : 13h à 36h (tous à 2 chiffres, pas de zéro devant)
HORIZONS = [f"{h:02d}" if h < 10 else str(h) for h in range(1, 37)]

# Mesures réelles (pas de tag Prediction)
REAL_MEASUREMENTS = {
    "Air temperature 2m above ground (current value)":        "temp_2m",
    "Atmospheric pressure at barometric altitude":            "pressure",
    "Global radiation (ten minutes mean)":                    "glob_rad",
    "Precipitation (ten minutes total)":                      "precip",
    "Relative air humidity 2m above ground (current value)":  "relhum_2m",
    "Sunshine duration (ten minutes total)":                  "sunshine",
    "Wind speed scalar (ten minutes mean)":                   "wind_speed",
}

# Prévisions (filtrées par tag Prediction)
PRED_MEASUREMENTS = {
    "PRED_T_2M_ctrl":      "pred_temp",
    "PRED_PS_ctrl":        "pred_pressure",
    "PRED_GLOB_ctrl":      "pred_glob_rad",
    "PRED_RELHUM_2M_ctrl": "pred_relhum",
    "PRED_TOT_PREC_ctrl":  "pred_precip",
    "PRED_DURSUN_ctrl":    "pred_sunshine",
    "PRED_FF_10M_ctrl":    "pred_wind_speed",
    "PRED_DD_10M_ctrl":    "pred_wind_dir",
    "PRED_GLOB_q10":       "pred_glob_rad_q10",
    "PRED_GLOB_q90":       "pred_glob_rad_q90",
    "PRED_GLOB_stde":      "pred_glob_rad_stde",
}


# FONCTIONS


def fetch_real(api, measurement, alias, station):
    """Mesure réelle pour une station — colonne : {alias}_{station}"""
    col = f"{alias}_{station.replace(' ', '_').replace('/', '-')}"
    query = f'''
    from(bucket: "{BUCKET}")
      |> range(start: {START}, stop: {STOP})
      |> filter(fn: (r) => r._measurement == "{measurement}")
      |> filter(fn: (r) => r.Site == "{station}")
      |> keep(columns: ["_time", "_value"])
    '''
    result = api.query(query)
    records = [{"timestamp": r.values["_time"], col: r.values["_value"]}
               for table in result for r in table.records]
    print(f"    {col} : {len(records):,} pts")
    if not records:
        return None
    return pl.DataFrame(records).with_columns(
        pl.col("timestamp").cast(pl.Datetime("us", "UTC"))
    )


def fetch_pred(api, measurement, alias, station, horizon):
    """Prévision pour une station et un horizon — colonne : {alias}_h{horizon}_{station}"""
    col = f"{alias}_h{horizon}_{station.replace(' ', '_').replace('/', '-')}"
    query = f'''
    from(bucket: "{BUCKET}")
      |> range(start: {START}, stop: {STOP})
      |> filter(fn: (r) => r._measurement == "{measurement}")
      |> filter(fn: (r) => r.Site == "{station}")
      |> filter(fn: (r) => r.Prediction == "{horizon}")
      |> keep(columns: ["_time", "_value"])
    '''
    result = api.query(query)
    records = [{"timestamp": r.values["_time"], col: r.values["_value"]}
               for table in result for r in table.records]
    print(f"    {col} : {len(records):,} pts")
    if not records:
        return None
    return pl.DataFrame(records).with_columns(
        pl.col("timestamp").cast(pl.Datetime("us", "UTC"))
    )


# MAIN


if __name__ == "__main__":
    OUTPUT.mkdir(parents=True, exist_ok=True)
    client = InfluxDBClient(url=URL, token=TOKEN, org=ORG)
    api = client.query_api()

    all_frames = []

    # ── Mesures réelles
    for station in STATIONS:
        print(f"\n=== Mesures réelles — {station} ===")
        for measurement, alias in REAL_MEASUREMENTS.items():
            df = fetch_real(api, measurement, alias, station)
            if df is not None:
                all_frames.append(df)

    # ── Prévisions horizons 13–36h
    for station in STATIONS:
        print(f"\n=== Prévisions — {station} ===")
        for measurement, alias in PRED_MEASUREMENTS.items():
            for horizon in HORIZONS:
                df = fetch_pred(api, measurement, alias, station, horizon)
                if df is not None:
                    all_frames.append(df)

    client.close()

    print(f"\n=== Fusion de {len(all_frames)} séries ===")
    merged = all_frames[0]
    for df in all_frames[1:]:
        merged = merged.join(df, on="timestamp", how="full", coalesce=True)

    merged = merged.sort("timestamp")
    out = OUTPUT / "meteo_multistation_v4.parquet"
    merged.write_parquet(out)

    print(f"\n✓ Sauvegardé : {out}")
    print(f"  {len(merged):,} lignes × {len(merged.columns)} colonnes")
    print(f"  Période : {merged['timestamp'][0]} → {merged['timestamp'][-1]}")
    print(f"  Colonnes ({len(merged.columns)}) :")
    for col in merged.columns:
        print(f"    {col}")