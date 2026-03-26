"""
pipeline_recup_v5.py
Récupération des données MeteoSuisse depuis InfluxDB.
- Mesures réelles    : toutes variables, 6 stations
- Prévisions         : toutes variables, 6 stations, horizons 0h à 36h
- Colonnes nommées   : {variable}_{station} et pred_{variable}_h{horizon}_{station}
Résultat : meteo_multistation_v5.parquet
"""


import polars as pl
from pathlib import Path
from influxdb_client import InfluxDBClient

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION — paramètres de connexion InfluxDB et périmètre de requête
# ═══════════════════════════════════════════════════════════════════════════════

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


HORIZONS = [f"{h:02d}" if h < 10 else str(h) for h in range(1, 37)]
#tient compte du souci en desous de 2 chiffres : 01 03 04 ...

REAL_MEASUREMENTS = {
    "Air temperature 2m above ground (current value)":        "temp_2m",
    "Atmospheric pressure at barometric altitude":            "pressure",
    "Global radiation (ten minutes mean)":                    "glob_rad",
    "Precipitation (ten minutes total)":                      "precip",
    "Relative air humidity 2m above ground (current value)":  "relhum_2m",
    "Sunshine duration (ten minutes total)":                  "sunshine",
    "Wind speed scalar (ten minutes mean)":                   "wind_speed",
}

PRED_MEASUREMENTS = {
    "PRED_T_2M_ctrl":      "pred_temp",           # Température à 2m prévue
    "PRED_PS_ctrl":        "pred_pressure",        # Pression atmosphérique prévue
    "PRED_GLOB_ctrl":      "pred_glob_rad",        # Rayonnement global prévu (valeur centrale)
    "PRED_RELHUM_2M_ctrl": "pred_relhum",          # Humidité relative prévue
    "PRED_TOT_PREC_ctrl":  "pred_precip",          # Précipitations prévues
    "PRED_DURSUN_ctrl":    "pred_sunshine",         # Durée d'ensoleillement prévue
    "PRED_FF_10M_ctrl":    "pred_wind_speed",       # Vitesse du vent prévue
    "PRED_DD_10M_ctrl":    "pred_wind_dir",         # Direction du vent prévue
    "PRED_GLOB_q10":       "pred_glob_rad_q10",     # Quantile 10% du rayonnement (borne basse)
    "PRED_GLOB_q90":       "pred_glob_rad_q90",     # Quantile 90% du rayonnement (borne haute)
    "PRED_GLOB_stde":      "pred_glob_rad_stde",    # Écart-type du rayonnement (dispersion)
}

# ═══════════════════════════════════════════════════════════════════════════════
# FONCTIONS — requêtes Flux vers InfluxDB et construction des DataFrames
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_real(api, measurement, alias, station):
    """
    Récupère une série de mesures réelles pour UNE variable et UNE station.
    Retourne un DataFrame Polars à 2 colonnes : timestamp + valeur.
    La colonne valeur est nommée {alias}_{station} (ex: temp_2m_Sion).
    """
    # Construction du nom de colonne : espaces → underscores, slashs → tirets
    col = f"{alias}_{station.replace(' ', '_').replace('/', '-')}"

    # Requête Flux : filtre sur le measurement et le site, ne garde que le temps et la valeur
    query = f'''
    from(bucket: "{BUCKET}")
      |> range(start: {START}, stop: {STOP})
      |> filter(fn: (r) => r._measurement == "{measurement}")
      |> filter(fn: (r) => r.Site == "{station}")
      |> keep(columns: ["_time", "_value"]) 
    '''
    #filtre les mesures
    # Exécution de la requête et extraction des enregistrements
    result = api.query(query)
    #récupère des tables en retour
    records = [{"timestamp": r.values["_time"], col: r.values["_value"]}
               for table in result for r in table.records]
    #crée un dictionnaire avec le timestamp et la mesure ou pred

    # Affichage du nombre de points récupérés (suivi de progression)
    print(f"    {col} : {len(records):,} pts")

    # Si aucun enregistrement, on retourne None (sera ignoré lors de la fusion)
    if not records:
        return None

    # Conversion en DataFrame Polars avec cast du timestamp en Datetime UTC
    return pl.DataFrame(records).with_columns(
        pl.col("timestamp").cast(pl.Datetime("us", "UTC"))
    )

def fetch_pred(api, measurement, alias, station, horizon):
    """
    Récupère une série de prévisions pour UNE variable, UNE station et UN horizon.
    Retourne un DataFrame Polars à 2 colonnes : timestamp + valeur.
    La colonne valeur est nommée {alias}_h{horizon}_{station} (ex: pred_temp_h12_Sion).
    """
    # Nom de colonne incluant l'horizon de prévision
    col = f"{alias}_h{horizon}_{station.replace(' ', '_').replace('/', '-')}"

    # Requête Flux identique aux mesures réelles, avec filtre additionnel sur le tag Prediction
    # Le tag Prediction correspond à l'horizon temporel (ex: "12" = prévision à 12h)
    query = f'''
    from(bucket: "{BUCKET}")
      |> range(start: {START}, stop: {STOP})
      |> filter(fn: (r) => r._measurement == "{measurement}")
      |> filter(fn: (r) => r.Site == "{station}")
      |> filter(fn: (r) => r.Prediction == "{horizon}")
      |> keep(columns: ["_time", "_value"])
    '''

    # Exécution et extraction (même pattern que fetch_real)
    result = api.query(query)
    records = [{"timestamp": r.values["_time"], col: r.values["_value"]}
               for table in result for r in table.records]

    print(f"    {col} : {len(records):,} pts")

    if not records:
        return None

    # Cast identique en Datetime UTC pour assurer la compatibilité lors du join
    return pl.DataFrame(records).with_columns(
        pl.col("timestamp").cast(pl.Datetime("us", "UTC"))
    )

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN — orchestration : récupération, fusion et export
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Création du répertoire de sortie si inexistant
    OUTPUT.mkdir(parents=True, exist_ok=True)

    # Connexion au serveur InfluxDB avec les credentials définis plus haut
    client = InfluxDBClient(url=URL, token=TOKEN, org=ORG)
    # Instanciation de l'API de requête (interface pour envoyer du Flux)
    api = client.query_api()

    # Liste qui accumulera tous les petits DataFrames (1 par série temporelle)
    all_frames = []

    # ── Boucle 1 : Mesures réelles ──
    # Pour chaque station, on récupère toutes les variables observées
    for station in STATIONS:
        print(f"\n=== Mesures réelles — {station} ===")
        for measurement, alias in REAL_MEASUREMENTS.items():
            df = fetch_real(api, measurement, alias, station)
            # On n'ajoute que les séries non vides
            if df is not None:
                all_frames.append(df)

    # ── Boucle 2 : Prévisions ──
    # Pour chaque station × variable × horizon, on récupère la prévision
    # Cela génère potentiellement 6 stations × 11 variables × 36 horizons = 2376 séries
    for station in STATIONS:
        print(f"\n=== Prévisions — {station} ===")
        for measurement, alias in PRED_MEASUREMENTS.items():
            for horizon in HORIZONS:
                df = fetch_pred(api, measurement, alias, station, horizon)
                if df is not None:
                    all_frames.append(df)

    client.close()

    # ── Fusion de toutes les séries sur la colonne timestamp ──
    # Join "full" (outer) pour conserver toutes les dates, même si certaines séries ont des trous
    # coalesce=True fusionne les colonnes timestamp dupliquées après le join
    print(f"\n=== Fusion de {len(all_frames)} séries ===")
    merged = all_frames[0]
    for df in all_frames[1:]:
        merged = merged.join(df, on="timestamp", how="full", coalesce=True)

    # Tri chronologique pour un parquet bien ordonné
    merged = merged.sort("timestamp")

    # Écriture du fichier parquet final
    out = OUTPUT / "meteo_multistation_v4.parquet"
    merged.write_parquet(out)

    # ── Résumé de l'export ──
    print(f"\n✓ Sauvegardé : {out}")
    print(f"  {len(merged):,} lignes × {len(merged.columns)} colonnes")
    print(f"  Période : {merged['timestamp'][0]} → {merged['timestamp'][-1]}")
    # Liste exhaustive des colonnes pour vérification
    print(f"  Colonnes ({len(merged.columns)}) :")
    for col in merged.columns:
        print(f"    {col}")