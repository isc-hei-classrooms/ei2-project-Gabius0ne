"""
pipeline_features_v14_utc.py
============================

Pipeline de construction des features journalières pour le forecast
day-ahead Oiken (96 pas de 15 min).

VERSION v14 : TOUTES LES RÉFÉRENCES TEMPORELLES EN UTC
------------------------------------------------------
Cette version dérive de v13. Elle corrige le mélange de timezones :
  - v13 : load et météo mesurée en Europe/Zurich, prévisions et horizons
          en UTC. Les features `_hXX` étaient locales, `_tXX` étaient UTC.
  - v14 : tout est en UTC. Un jour cible = jour calendaire UTC. Toutes
          les heures (`_hXX`, `_tXX`) sont en UTC.

Conséquences à connaître :
  - Sémantique des features horaires modifiée : `load_jm1_h12` désigne
    désormais la consommation entre 12h et 13h UTC (= 13h-14h locale
    hiver, 14h-15h locale été), pas 12h-13h locale.
  - Les jours DST (passage hiver/été) ne sont plus pathologiques :
    chaque jour UTC fait exactement 24h, contrairement aux jours locaux
    qui ont 23h ou 25h les dimanches de transition.
  - L'output v14 est non comparable directement à v13 sans réentraîner.

PRINCIPE GÉNÉRAL
----------------
Pour chaque jour cible J+1 à prédire, on construit un vecteur de features
qui regroupe toutes les informations disponibles à l'instant de soumission
(11h LOCALE Europe/Zurich le jour J, soit 10h UTC en hiver et 09h UTC
en été selon DST) :
  - Consommation (load) observée les 7 jours précédents → profil récent
  - Production PV observée à J-2 et début J → état du parc solaire
  - Mesures météo réelles à J-2 et début J → conditions physiques vécues
  - Prévisions météo J+1 extraites du run MeteoSuisse de 11h → anticipation
  - Indicateurs calendaires → saisonnalité, weekend, vacances, jours fériés

CHANGEMENTS v13 → v14
---------------------
  CONVERSION TIMEZONE : `load_oiken` convertit explicitement de
    Europe/Zurich vers UTC après attachement du fuseau source. Le DataFrame
    Oiken vit en UTC pour le reste du pipeline.

  SUPPRESSION meteo_zurich : `load_meteo` ne retourne plus que le
    DataFrame UTC. Tout le pipeline utilise meteo_utc.

  SLICES UTC : `get_day_slice` et `get_morning_slice` construisent leurs
    bornes en UTC. `compute_pv_yield_ratios` aussi.

  RAMADAN : `ramadan_night_hours` retourne désormais des heures UTC.
    Le créneau est `[19..23] ∪ [0..4]` (équivalent UTC d'hiver de
    l'ancien `[0..5] ∪ [20..23]` local). Décalage de 1h en été DST,
    accepté comme approximation.

  MORNING CUTOFF : `get_morning_slice(day_j, until_hour_local=10)` collecte
    désormais les données jusqu'à 10h LOCALE Europe/Zurich du jour J,
    converti en UTC selon le DST :
      - Hiver (UTC+1) : data jusqu'à 09h UTC
      - Été   (UTC+2) : data jusqu'à 08h UTC
    Cohérent avec une deadline de soumission à 11h locale (1h de marge
    de sécurité).

  HORIZON RUN MeteoSuisse : `get_correct_horizon_jp1` reçoit désormais
    la date de soumission. La formule devient
    `min_horizon = h_utc + 14 + dst_offset` :
      - Hiver : h_utc + 15 (run 09h UTC dispo à 10h UTC = 11h locale)
      - Été   : h_utc + 16 (run 08h UTC dispo à 09h UTC = 11h locale)
    Hypothèse : les colonnes `pred_*_h<horizon>_*` du parquet sont
    indexées sur le dernier run MeteoSuisse disponible à submission,
    avec ~1h de délai de publication.

  NOMS DE FEATURES : aucun renommage. `_hXX` et `_tXX` désignent tous
    deux des heures UTC dans v14. Pas de changement pour minimiser
    l'impact sur les scripts d'entraînement et d'analyse aval.

CONTRAINTES v13 (préservées en v14)
-----------------------------------
  IRRADIANCE J+1 : agrégée sur 3 stations de plaine (Sion/Visp/Pully)
    aux heures diurnes 06h-19h UTC.
  VENT : uniquement wind_speed sur Sion + Pully, heures 06h-19h UTC.
  VARIANCES IRRADIANCE : supprimées.
  MÉTÉO J : supprimée.
  INTERACTIONS PV : irradiance × pv_yield, 14h diurnes UTC.
  VACANCES SCOLAIRES : `is_school_holiday` + `is_bridge_day`.
  Estimation finale : ~1308 features, ratio features/samples ~1.5.

SORTIES
-------
  DATA/processed/X_features_v14.parquet   — features (1 ligne/jour UTC)
  DATA/processed/Y_target_v14.parquet     — cibles load 96 pas (1 ligne/jour UTC)
  DATA/processed/B_baseline_v14.parquet   — prévisions Oiken 96 pas
"""

import math
import polars as pl
import numpy as np
from pathlib import Path
from datetime import timedelta, date, datetime
from zoneinfo import ZoneInfo

# Fuseau de référence pour la deadline de soumission (11h locale Europe/Zurich).
ZURICH = ZoneInfo("Europe/Zurich")

# ─────────────────────────────────────────────────────────────────────
# CONFIGURATION — chemins, listes de colonnes, paramètres globaux
# ─────────────────────────────────────────────────────────────────────
BASE    = Path(__file__).resolve().parents[2] / "DATA"
CSV     = BASE / "oiken-data.csv"
METEO   = BASE / "meteo_multistation_v5.parquet"
OUT     = BASE / "processed"
OUT.mkdir(parents=True, exist_ok=True)

# Production solaire dans le CSV Oiken.
# /!\ solar_remote présente un offset nocturne suspect (probable
# décalage timezone à la source). Conservé pour pv_yield mais à traiter
# en amont si possible.
PROD_COLS = [
    "solar_central_valais",
    "solar_sion",
    "solar_sierre",
    "solar_remote",
]

# Production "live" disponible au moment de soumettre J+1 (matin J).
# solar_remote exclu car potentiellement différé à la source.
PROD_COLS_LIVE = [
    "solar_central_valais",
    "solar_sion",
    "solar_sierre",
]

# Load historique : J-2 à J-8 par rapport à la cible J+1.
# J-1 incomplet à 11h UTC le jour J (il manque la fin de soirée),
# donc on part de J-2.
LOAD_HISTORY_DAYS = list(range(2, 9))

# Variables météo mesurées (jauges) sur toutes les stations.
REAL_METEO_VARS = ["temp_2m", "glob_rad", "pressure", "relhum_2m"]

STATIONS_ALL = [
    "Pully", "Sion", "Visp", "Montana",
    "Col_du_Grand_St-Bernard", "Les_Attelas",
]

# Stations de plaine pour irradiance prévue (parc PV en plaine).
STATIONS_PLAINE = ["Sion", "Visp", "Pully"]

# Stations pour le vent prévu (régime altitude différent ignoré).
STATIONS_WIND = ["Sion", "Pully"]

# Heures diurnes UTC où l'irradiance et le vent sont extraits.
# 06h-19h UTC = 07h-20h locale hiver, 08h-21h locale été.
HOURS_DIURNAL = list(range(6, 20))

PRED_VARS_JP1 = ["temp", "glob_rad", "pressure", "relhum", "precip", "sunshine"]

# Vide en v13/v14 : wind_dir (seule variable angulaire) supprimée.
CYCLIC_PRED_VARS = set()

# Horizons MeteoSuisse disponibles depuis le run de 11h UTC, par modulo 3.
HORIZONS_BY_MOD = {
    0: list(range(3, 37, 3)),
    1: list(range(1, 35, 3)),
    2: list(range(2, 36, 3)),
}

# Dates de Ramadan (en dates calendaires, donc indépendantes du fuseau).
RAMADAN_DATES = {
    2022: (date(2022, 4,  2), date(2022, 5,  1)),
    2023: (date(2023, 3, 23), date(2023, 4, 20)),
    2024: (date(2024, 3, 11), date(2024, 4,  9)),
    2025: (date(2025, 3,  1), date(2025, 3, 29)),
    2026: (date(2026, 2, 18), date(2026, 3, 19)),
}

# Jours fériés Valais romand + nationaux suisses (dates calendaires).
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

# Vacances scolaires Valais romand (Sion).
VACANCES_SCOLAIRES = [
    (date(2022, 10, 12), date(2022, 10, 23)),
    (date(2022, 12, 23), date(2023,  1,  8)),
    (date(2023,  2, 17), date(2023,  2, 26)),
    (date(2023,  4,  6), date(2023,  4, 16)),
    (date(2023,  6, 24), date(2023,  8, 16)),
    (date(2023, 10, 11), date(2023, 10, 22)),
    (date(2023, 12, 22), date(2024,  1,  7)),
    (date(2024,  2,  9), date(2024,  2, 18)),
    (date(2024,  3, 28), date(2024,  4,  7)),
    (date(2024,  6, 22), date(2024,  8, 14)),
    (date(2024, 10, 16), date(2024, 10, 27)),
    (date(2024, 12, 20), date(2025,  1,  6)),
    (date(2025,  2, 28), date(2025,  3,  9)),
    (date(2025,  4, 17), date(2025,  4, 27)),
    (date(2025,  6, 28), date(2025,  8, 17)),
    (date(2025, 10, 18), date(2025, 10, 26)),
    (date(2025, 12, 20), date(2026,  1,  4)),
]


# ─────────────────────────────────────────────────────────────────────
# HELPERS — indicateurs calendaires et calcul d'horizon
# ─────────────────────────────────────────────────────────────────────

def is_ramadan(d: date) -> bool:
    """True si la date tombe dans une période de Ramadan connue."""
    for start, end in RAMADAN_DATES.values():
        if start <= d <= end:
            return True
    return False


def ramadan_night_hours(d: date) -> list[int]:
    """
    Heures UTC "nocturnes" pendant le Ramadan.

    v13 utilisait [0..5] + [20..23] en heure locale Europe/Zurich pour
    capturer le pic de consommation suhoor/iftar. v14 retourne l'équivalent
    UTC d'hiver (UTC+1) : [19..23] + [0..4]. En été (UTC+2), le créneau
    réel décale de 1h, accepté comme approximation : le Ramadan tombe
    actuellement en cooler months (fin février à début avril), donc
    majoritairement en heure d'hiver.

    Si la précision été est critique, il faudra étendre à [18..23] + [0..4]
    ou détecter le DST par date.
    """
    if not is_ramadan(d):
        return []
    return list(range(19, 24)) + list(range(0, 5))


def is_school_holiday(d: date) -> bool:
    """True si la date tombe dans une période de vacances scolaires Valais."""
    for start, end in VACANCES_SCOLAIRES:
        if start <= d <= end:
            return True
    return False


def is_bridge_day(d: date) -> bool:
    """
    Détecte un jour pont : jour ouvré coincé entre des jours off
    (weekend, férié). Hypothèse métier : profil de consommation proche
    d'un weekend.

    Exclut les jours déjà weekend / fériés / vacances scolaires pour
    éviter la double catégorisation.
    """
    if d.weekday() >= 5:
        return False
    if d in FERIES:
        return False
    if is_school_holiday(d):
        return False

    yesterday = d - timedelta(days=1)
    tomorrow  = d + timedelta(days=1)

    def is_off(day: date) -> bool:
        return day.weekday() >= 5 or day in FERIES

    if d.weekday() == 4 and is_off(yesterday):
        return True
    if d.weekday() == 0 and is_off(tomorrow):
        return True
    if is_off(yesterday) and is_off(tomorrow):
        return True
    return False


def _get_pv_capacity_proxy(d: date) -> float:
    """
    Estime la capacité PV installée Oiken (MWp) par interpolation linéaire
    entre ancrages annuels du parc national suisse, calibrée sur fin 2022.
    """
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

    OIKEN_BASE_MWP   = 55.0
    NATIONAL_BASE_GW = 4.65
    return OIKEN_BASE_MWP * national_gw / NATIONAL_BASE_GW


def get_dst_offset_hours(d: date) -> int:
    """
    Écart UTC de Europe/Zurich pour la date d, en heures.
      - 1 en hiver (UTC+1, heure normale)
      - 2 en été  (UTC+2, DST actif)

    Le passage en DST tombe le dernier dimanche de mars (CH) et le retour
    le dernier dimanche d'octobre. zoneinfo gère ces transitions
    correctement.
    """
    return int(datetime(d.year, d.month, d.day, 12, 0,
                        tzinfo=ZURICH).utcoffset().total_seconds() // 3600)


def get_correct_horizon_jp1(h_utc: int, submission_day: date) -> int | None:
    """
    Pour une heure UTC du jour cible J+1, retourne l'horizon de prévision
    MeteoSuisse à utiliser depuis le run disponible à 11h LOCALE Europe/Zurich
    le jour J = submission_day.

    Calcul de la marge anti-leakage :
      11h locale = (11 - dst_offset) UTC
        - Hiver : 10h UTC
        - Été   : 09h UTC

      En supposant 1h de délai de publication MeteoSuisse, le dernier run
      disponible à submission est celui de (10 - dst_offset) UTC :
        - Hiver : run 09h UTC (= 10h UTC - 1h)
        - Été   : run 08h UTC (= 09h UTC - 1h)

      min_horizon nécessaire = (24 - run_h) + h_utc = h_utc + 14 + dst_offset
        - Hiver : h_utc + 15
        - Été   : h_utc + 16

    /!\\ Hypothèse implicite : les colonnes pred_*_h<horizon>_* du parquet
    représentent l'horizon depuis le dernier run disponible à submission.
    Si le parquet a été extrait avec un run hour fixe différent (ex : toujours
    le 11h UTC), ajuster la constante 14 → 13 et le DST offset.
    """
    offset = get_dst_offset_hours(submission_day)
    mod = h_utc % 3
    available = HORIZONS_BY_MOD[mod]
    min_needed = h_utc + 14 + offset
    valid = [h for h in available if h >= min_needed]
    return min(valid) if valid else None


# ─────────────────────────────────────────────────────────────────────
# CHARGEMENT — lecture du CSV Oiken et du parquet météo
# ─────────────────────────────────────────────────────────────────────

def load_oiken(path: Path) -> pl.DataFrame:
    """
    Charge le CSV Oiken et convertit en UTC.

    Pipeline timezone (v14) :
      1. Parse le timestamp naïf (format 'DD.MM.YYYY HH:MM')
      2. Attache Europe/Zurich (le fichier est en heure locale Suisse)
         avec gestion DST : ambiguous="earliest", non_existent="null"
      3. Convertit en UTC pour la suite du pipeline

    Le DataFrame retourné a des timestamps timezone-aware en UTC.
    """
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
          .dt.convert_time_zone("UTC")
          .alias("timestamp")
    ).rename({
        "standardised load [-]":                 "load",
        "standardised forecast load [-]":        "load_forecast_oiken",
        "central valais solar production [kWh]": "solar_central_valais",
        "sion area solar production [kWh]":      "solar_sion",
        "sierre area production [kWh]":          "solar_sierre",
        "remote solar production [kWh]":         "solar_remote",
    }).sort("timestamp")
    print(f"  Oiken (UTC) : {len(df):,} lignes | "
          f"{df['timestamp'].drop_nulls()[0]} → {df['timestamp'][-1]}")
    return df


def load_meteo(path: Path) -> tuple[pl.DataFrame, list[str]]:
    """
    Charge le parquet météo en UTC.

    Le parquet source a déjà des timestamps UTC timezone-aware (le code
    v13 utilisait `convert_time_zone` et non `replace_time_zone`,
    confirmant la présence d'un label de fuseau à la source).

    real_cols = colonnes de mesures réelles existantes (variables × stations
    filtrées par présence dans le parquet).
    """
    df_utc = pl.read_parquet(path).sort("timestamp")
    all_cols  = [c for c in df_utc.columns if c != "timestamp"]
    real_cols = [
        f"{var}_{station}"
        for var in REAL_METEO_VARS
        for station in STATIONS_ALL
        if f"{var}_{station}" in all_cols
    ]
    print(f"  Météo (UTC) : {len(df_utc):,} lignes | {len(real_cols)} colonnes réelles")
    return df_utc, real_cols


def get_day_slice(df: pl.DataFrame, day: date) -> pl.DataFrame:
    """
    Extrait les 24h d'un jour calendaire UTC (00h00 à 23h59:59 UTC).

    /!\\ Un jour UTC ne correspond pas à un jour locale Suisse :
    décalage de 1-2h selon DST. Utiliser cohérent avec le reste du pipeline.
    """
    start = pl.datetime(day.year, day.month, day.day, 0, 0, 0, time_zone="UTC")
    end   = pl.datetime(day.year, day.month, day.day, 23, 59, 59, time_zone="UTC")
    return df.filter((pl.col("timestamp") >= start) & (pl.col("timestamp") <= end))


def get_morning_slice(df: pl.DataFrame, day: date, until_hour_local: int = 10) -> pl.DataFrame:
    """
    Extrait le début d'un jour UTC, depuis 00h UTC jusqu'à `until_hour_local`
    heure LOCALE Europe/Zurich (convertie en UTC selon le DST du jour).

    until_hour_local=10 par défaut : couvre 00h UTC jusqu'à 10h locale,
    soit 09h UTC en hiver et 08h UTC en été. Ce cutoff laisse 1h de marge
    avant la deadline de soumission de 11h locale.

    Note : la borne basse (00h UTC) ne correspond pas au début d'une
    journée locale (qui commence à 23h UTC J-1 en hiver, 22h UTC J-1 en été).
    En pratique cette tranche couvre donc seulement les premières heures
    locales du matin du jour J, ce qui correspond au comportement v13 où
    le slice était purement local. La différence (~1-2h en moins de matin
    précoce) est négligeable pour le signal de consommation matinal qui
    démarre vers 5-6h locale.
    """
    offset = get_dst_offset_hours(day)
    until_hour_utc = until_hour_local - offset
    if until_hour_utc <= 0:
        # cas pathologique (until_hour_local trop tôt) : retourne tranche vide
        until_hour_utc = 0

    start = pl.datetime(day.year, day.month, day.day, 0, 0, 0, time_zone="UTC")
    end   = pl.datetime(day.year, day.month, day.day, until_hour_utc, 0, 0, time_zone="UTC")
    return df.filter((pl.col("timestamp") >= start) & (pl.col("timestamp") <= end))


def series_stats(series: pl.Series, prefix: str) -> dict:
    """Stats descriptives (mean, max, min, std), None si série vide."""
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
    """
    Profil horaire d'une colonne sur 24h : moyenne par heure UTC,
    produit 24 features {prefix_h00, ..., prefix_h23}.

    /!\\ v14 : les heures sont en UTC (alignées avec le timestamp converti
    dans load_oiken). load_jm1_h12 = moyenne entre 12h et 13h UTC à J-2,
    soit 13h-14h locale hiver / 14h-15h locale été.
    """
    result = {}
    for h in range(24):
        hour_vals = df_day.filter(pl.col("timestamp").dt.hour() == h)[col].drop_nulls()
        result[f"{prefix}_h{h:02d}"] = float(hour_vals.mean()) if len(hour_vals) > 0 else None
    return result


def real_meteo_stats(df_slice: pl.DataFrame, real_cols: list[str], prefix: str) -> dict:
    """
    Stats (mean/max/min) par colonne météo mesurée sur la tranche fournie.

    Utilisé deux fois par jour cible :
      - df_slice = J-2 complet UTC → prefix "rmet_jm1"
      - df_slice = J matin jusqu'à 10h UTC → prefix "rmet_jmorn"
    """
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


# ─────────────────────────────────────────────────────────────────────
# PRÉVISIONS J+1 (inchangé v13 → v14, déjà en UTC)
# ─────────────────────────────────────────────────────────────────────

def extract_pred_jp1(meteo_utc: pl.DataFrame, target_day: date) -> dict:
    """
    Prévisions météo pour le jour cible J+1, depuis le run disponible à
    11h LOCALE Europe/Zurich le jour J = target_day - 1 (cf
    get_correct_horizon_jp1 pour la logique anti-leakage DST-aware).

    Structure :
      - temp/pressure/relhum/precip/sunshine : 5 vars × 6 stations × 24h
      - glob_rad : 3 stations plaine × 14h diurnes UTC + 1 moyenne_day
      - wind_speed : 2 stations × 14h diurnes UTC

    Convention : pred_<var>_<station>_t<heure_UTC>
    """
    features = {}
    submission_day = target_day - timedelta(days=1)

    start = pl.datetime(target_day.year, target_day.month, target_day.day,
                        0, 0, 0, time_zone="UTC")
    end   = pl.datetime(target_day.year, target_day.month, target_day.day,
                        23, 59, 59, time_zone="UTC")
    day_utc = meteo_utc.filter((pl.col("timestamp") >= start) & (pl.col("timestamp") <= end))

    if len(day_utc) == 0:
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
        horizon = get_correct_horizon_jp1(h_utc, submission_day)
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

    # temp/pressure/relhum/precip/sunshine : toutes stations, 24h
    for var in PRED_VARS_JP1:
        if var == "glob_rad":
            continue
        for station in STATIONS_ALL:
            for h_utc in range(24):
                features[f"pred_{var}_{station}_t{h_utc:02d}"] = get_pred_value(var, station, h_utc)

    # glob_rad : 3 stations plaine × 14h diurnes
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

    # wind_speed : 2 stations × 14h diurnes
    for station in STATIONS_WIND:
        for h_utc in HOURS_DIURNAL:
            features[f"pred_wind_speed_{station}_t{h_utc:02d}"] = get_pred_value("wind_speed", station, h_utc)

    return features


# ─────────────────────────────────────────────────────────────────────
# PV YIELD RATIO (UTC en v14)
# ─────────────────────────────────────────────────────────────────────

def compute_pv_yield_ratios(
    oiken: pl.DataFrame,
    meteo: pl.DataFrame,
    target_date: date,
) -> dict:
    """
    Ratios pv_yield_{30j,90j} = max(solar_remote) / max(glob_rad_Sion)
    sur fenêtres glissantes finissant à day_jm1 = target_date - 2j.

    v14 : fenêtres en UTC (cohérent avec oiken et meteo désormais UTC).
    Le max sur une fenêtre est invariant au choix de timezone tant que
    les deux tableaux sont alignés.
    """
    features = {}
    day_jm1 = target_date - timedelta(days=2)

    for window_days, label in [(30, "30j"), (90, "90j")]:
        window_start = day_jm1 - timedelta(days=window_days - 1)

        start_dt = pl.datetime(window_start.year, window_start.month, window_start.day,
                               0, 0, 0, time_zone="UTC")
        end_dt   = pl.datetime(day_jm1.year, day_jm1.month, day_jm1.day,
                               23, 59, 59, time_zone="UTC")

        oiken_window = oiken.filter(
            (pl.col("timestamp") >= start_dt) & (pl.col("timestamp") <= end_dt)
        )
        meteo_window = meteo.filter(
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


# ─────────────────────────────────────────────────────────────────────
# v13 : INTERACTIONS PV — irradiance × pv_yield (inchangé v14)
# ─────────────────────────────────────────────────────────────────────

def compute_pv_interaction_features(features: dict) -> dict:
    """
    Features d'interaction explicite irradiance prévue × rendement PV récent.

    pred_pv_adj_{30j|90j}_t{XX} = mean(irr Sion/Visp/Pully à h=XX) × pv_yield_{30j|90j}
    pred_pv_adj_{30j|90j}_day  = SOMME des 14 valeurs horaires (proxy énergie)

    14 heures diurnes UTC × 2 fenêtres = 28 features horaires + 2 agrégés.
    """
    interaction_feats = {}

    pv_yield_30j = features.get("pv_yield_30j")
    pv_yield_90j = features.get("pv_yield_90j")

    for h_utc in HOURS_DIURNAL:
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

    for label in ["30j", "90j"]:
        day_vals = [interaction_feats.get(f"pred_pv_adj_{label}_t{h:02d}")
                    for h in HOURS_DIURNAL]
        day_clean = [v for v in day_vals if v is not None]
        interaction_feats[f"pred_pv_adj_{label}_day"] = sum(day_clean) if day_clean else None

    return interaction_feats


# ─────────────────────────────────────────────────────────────────────
# CONSTRUCTION FEATURES — orchestrateur par jour cible
# ─────────────────────────────────────────────────────────────────────

def build_features(
    target_date: date,
    oiken: pl.DataFrame,
    meteo: pl.DataFrame,
    real_cols: list[str],
) -> dict | None:
    """
    Construit le vecteur de features pour un jour cible J+1 (jour calendaire UTC).

    target_date = jour à prédire (J+1)
    day_j   = target_date - 1 = jour de soumission (J)
    day_jm1 = target_date - 2 = dernière journée complète disponible

    Note héritée v13 : le suffixe `jm1` est relatif au jour de soumission J,
    donc solar_remote_jm1_hXX = valeur d'il y a 2 jours par rapport à la cible.

    v14 : tous les slices et toutes les heures sont en UTC.

    Retourne None si target_date a < 90 pas de load valides (UTC).
    """
    day_j   = target_date - timedelta(days=1)
    day_jm1 = target_date - timedelta(days=2)

    oiken_target = get_day_slice(oiken, target_date)
    if len(oiken_target) < 90:
        return None

    features = {}

    # ─── Load historique J-2 à J-8 ────────────────────────────────────
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

    # ─── Production solaire mesurée ───────────────────────────────────
    oiken_jm1       = get_day_slice(oiken, day_jm1)
    oiken_j_morning = get_morning_slice(oiken, day_j, until_hour_local=10)

    for col in PROD_COLS:
        if col in oiken_jm1.columns:
            features[f"{col}_jm1_total"] = float(oiken_jm1[col].sum())
            features.update(hourly_profile(oiken_jm1, col, f"{col}_jm1"))

    for col in PROD_COLS_LIVE:
        if col in oiken_j_morning.columns:
            features[f"{col}_j_morning_total"] = float(oiken_j_morning[col].sum())

    # ─── Météo mesurée ────────────────────────────────────────────────
    meteo_jm1 = get_day_slice(meteo, day_jm1)
    features.update(real_meteo_stats(meteo_jm1, real_cols, "rmet_jm1"))
    meteo_j_morning = get_morning_slice(meteo, day_j, until_hour_local=10)
    features.update(real_meteo_stats(meteo_j_morning, real_cols, "rmet_jmorn"))

    # ─── Prévisions J+1 ───────────────────────────────────────────────
    features.update(extract_pred_jp1(meteo, target_date))

    # ─── Capacité PV installée ────────────────────────────────────────
    features["pv_capacity_MWp"] = _get_pv_capacity_proxy(target_date)

    # ─── pv_yield 30j / 90j ───────────────────────────────────────────
    features.update(compute_pv_yield_ratios(oiken, meteo, target_date))

    # ─── Interactions PV (lit pred_glob_rad_* et pv_yield_* déjà posés) ──
    features.update(compute_pv_interaction_features(features))

    # ─── Calendaire ───────────────────────────────────────────────────
    doy = target_date.timetuple().tm_yday
    features["dayofweek"]         = target_date.weekday()
    features["month"]             = target_date.month
    features["is_weekend"]        = int(target_date.weekday() >= 5)
    features["is_holiday"]        = int(target_date in FERIES)
    features["is_school_holiday"] = int(is_school_holiday(target_date))
    features["is_bridge_day"]     = int(is_bridge_day(target_date))
    features["is_ramadan"]        = int(is_ramadan(target_date))

    # Indicateurs horaires Ramadan en UTC (v14).
    ramadan_hours = set(ramadan_night_hours(target_date))
    for h in range(24):
        features[f"is_ramadan_h{h:02d}"] = int(h in ramadan_hours)

    # Encodages cycliques
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


# ─────────────────────────────────────────────────────────────────────
# PIPELINE PRINCIPAL
# ─────────────────────────────────────────────────────────────────────

def main():
    """
    Orchestration :
      1. Charge Oiken (UTC) + météo (UTC)
      2. Détermine la plage de dates cible (UTC)
      3. Itère et appelle build_features
      4. Concatène X, Y, B en parquet v14
      5. Rapport de vérification
    """
    print("=== Chargement des données (timezone UTC) ===")
    oiken = load_oiken(CSV)
    meteo, real_cols = load_meteo(METEO)

    # Plage cible : marge de 9j pour LOAD_HISTORY_DAYS (J-8 + 1j sécurité).
    # `.date()` sur un timestamp UTC retourne la date UTC, cohérent avec
    # le slicing UTC du reste du pipeline.
    first_ts  = oiken["timestamp"].drop_nulls()[0]
    first_day = first_ts.date() + timedelta(days=9)

    last_day = oiken["timestamp"][-1].date() - timedelta(days=1)

    all_dates = [first_day + timedelta(days=i)
                 for i in range((last_day - first_day).days + 1)]

    print(f"\n=== Construction features v14 (UTC) : {first_day} → {last_day} "
          f"({len(all_dates)} jours) ===")

    rows_X, rows_Y, rows_B, dates_ok = [], [], [], []

    for i, target_date in enumerate(all_dates):
        if i % 100 == 0:
            print(f"  {i}/{len(all_dates)} — {target_date}")

        result = build_features(target_date, oiken, meteo, real_cols)
        if result is None:
            continue

        rows_X.append(result["features"])
        rows_Y.append(result["target"])
        rows_B.append(result["baseline"])
        dates_ok.append(str(result["date"]))

    print(f"\n  {len(dates_ok)} jours valides sur {len(all_dates)}")

    # X (features)
    X = pl.DataFrame(rows_X).with_columns(
        pl.Series("date", dates_ok).str.strptime(pl.Date, "%Y-%m-%d")
    )
    X = X.select(["date"] + [c for c in X.columns if c != "date"])

    # Y (cibles)
    n_steps = len(rows_Y[0])
    Y = pl.DataFrame(
        {f"load_t{i:03d}": [row[i] if i < len(row) else None for row in rows_Y]
         for i in range(n_steps)}
    ).with_columns(pl.Series("date", dates_ok).str.strptime(pl.Date, "%Y-%m-%d"))
    Y = Y.select(["date"] + [f"load_t{i:03d}" for i in range(n_steps)])

    # B (baseline)
    B = pl.DataFrame(
        {f"baseline_t{i:03d}": [row[i] if i < len(row) else None for row in rows_B]
         for i in range(n_steps)}
    ).with_columns(pl.Series("date", dates_ok).str.strptime(pl.Date, "%Y-%m-%d"))
    B = B.select(["date"] + [f"baseline_t{i:03d}" for i in range(n_steps)])

    X.write_parquet(OUT / "X_features_v13-1.parquet")
    Y.write_parquet(OUT / "Y_target_v13-1.parquet")
    B.write_parquet(OUT / "B_baseline_v13-1.parquet")

    print(f"\n✓ X_features_v14 : {X.shape[0]} jours × {X.shape[1]} colonnes")
    print(f"✓ Y_target_v14   : {Y.shape[0]} jours × {Y.shape[1]} colonnes")
    print(f"✓ B_baseline_v14 : {B.shape[0]} jours × {B.shape[1]} colonnes")

    # Rapport de vérification
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
                print(f"  {col}: min={s.min():.2f}, max={s.max():.2f}, "
                      f"mean={s.mean():.2f}, nulls={X[col].null_count()}")
            else:
                print(f"  {col}: sum={s.sum()}, nulls={X[col].null_count()}")

    bad_cols = [c for c in X.columns if any(k in c for k in [
        "predJ_", "wind_dir", "glob_rad_q10", "glob_rad_q90",
        "glob_rad_stde", "glob_rad_std_stations", "glob_rad_spread",
    ])]
    if bad_cols:
        print(f"\n⚠ Features qui devraient être supprimées : {bad_cols[:10]}...")
    else:
        print(f"\n✓ Météo J, wind_dir, variances irradiance : supprimées")

    n_wind = len([c for c in X.columns if "wind" in c])
    n_glob = len([c for c in X.columns if "glob_rad" in c and "rmet" not in c])
    n_load = len([c for c in X.columns if c.startswith("load_")])
    n_pv   = len([c for c in X.columns if "pv_adj" in c or "pv_yield" in c or "pv_capacity" in c])
    print(f"  Wind speed : {n_wind} features")
    print(f"  Glob rad (prévu) : {n_glob} features")
    print(f"  Load historique : {n_load} features")
    print(f"  PV (yield/capacity/interaction) : {n_pv} features")

    print(f"\n✓ Sauvegardé dans : {OUT}")
    print(f"\nNote v14 : toutes les heures (_hXX, _tXX) sont en UTC.")
    print(f"           Comparaison v13 ↔ v14 non directe sans réentraînement.")


if __name__ == "__main__":
    main()