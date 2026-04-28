"""
pipeline_features_v13.py
========================

Pipeline de construction des features journalières pour le forecast
day-ahead Oiken (96 pas de 15 min).

PRINCIPE GÉNÉRAL
----------------
Pour chaque jour cible J+1 à prédire, on construit un vecteur de features
qui regroupe toutes les informations disponibles à l'instant de soumission
(11h UTC+1 le jour J) :
  - Consommation (load) observée les 7 jours précédents → profil récent
  - Production PV observée à J-2 et début J → état du parc solaire
  - Mesures météo réelles à J-2 et début J → conditions physiques vécues
  - Prévisions météo J+1 extraites du run MeteoSuisse de 11h → anticipation
  - Indicateurs calendaires → saisonnalité, weekend, vacances, jours fériés

Chaque jour produit une ligne dans X (features), une ligne dans Y
(96 valeurs de load cible) et une ligne dans B (baseline Oiken pour
benchmark).

CHANGEMENTS v12 → v13
---------------------
Le modèle v12 utilisait ~3584 features et avait un ratio features/samples
de 5.5, conduisant à du bruit et à des proxies non physiques (ex : le
modèle utilisait wind_dir comme proxy PV indirect au lieu de l'irradiance
directement). v13 réduit drastiquement les features :

  IRRADIANCE J+1 : agrégée sur 3 stations de plaine (Sion/Visp/Pully)
    aux heures diurnes 06h-19h UTC uniquement. Supprime Montana,
    Col_du_Grand_St-Bernard, Les_Attelas (altitude, peu pertinentes pour
    le parc de plaine) et les heures nocturnes (irradiance = 0 par
    définition). De 144 features → 42 (14h × 3 stations) + 1 moyenne jour.

  VENT : uniquement wind_speed sur Sion + Pully, heures 06h-19h UTC.
    Supprime wind_dir (sin/cos) qui servait de proxy PV détourné, et les
    4 autres stations. De 864 features → 28.

  VARIANCES IRRADIANCE : supprimées (glob_rad_q10, q90, stde,
    std_stations, spread). Le modèle les utilisait pour "ignorer"
    l'irradiance quand elle était incertaine, ce qui était contre-productif.
    De 457 features → 0.

  MÉTÉO J (jour de soumission, prévue le matin même) : supprimée
    entièrement. Importance toujours <1% dans tous les tests, pollue
    l'espace de features. De ~1320 features → 0.

  INTERACTIONS PV : recalculées avec la nouvelle irradiance agrégée
    (moyenne 3 stations plaine), heures 06h-19h.

  VACANCES SCOLAIRES : ajout is_school_holiday + is_bridge_day.
    Dates Valais romand (Sion) 2022-2025. Devrait améliorer la
    distinction vacances/période scolaire, absente en v12.

  INCHANGÉ : load historique (J-2 à J-8), PV mesuré (J-2 + début J),
    calendaire (hors nouveautés ci-dessus), météo mesurée,
    temp/pres/hum/precip/sunshine prévus J+1.

Estimation finale : ~1308 features (vs 3584 en v12), ratio
features/samples ~1.5 pour ~870 jours d'entraînement.

SORTIES
-------
  DATA/processed/X_features_GOLDEN.parquet   — features (1 ligne/jour)
  DATA/processed/Y_target_GOLDEN.parquet     — cibles load 96 pas (1 ligne/jour)
  DATA/processed/B_baseline_GOLDEN.parquet   — prévisions Oiken 96 pas (pour benchmark)
"""

import math
import polars as pl
import numpy as np
from pathlib import Path
from datetime import timedelta, date

# ─────────────────────────────────────────────────────────────────────
# CONFIGURATION — chemins, listes de colonnes, paramètres globaux
# ─────────────────────────────────────────────────────────────────────
# Le script est supposé tourner depuis src/features/, remonte 2 niveaux
# pour atteindre le dossier DATA/.
BASE    = Path(__file__).resolve().parents[2] / "DATA"
CSV     = BASE / "oiken-golden-dataset.csv"                       # fichier brut Oiken
METEO   = BASE / "meteo_multistationGOLDEN.parquet"        # parquet météo (InfluxDB exporté)
OUT     = BASE / "processed"
OUT.mkdir(parents=True, exist_ok=True)

# Les 4 colonnes de production solaire présentes dans le CSV Oiken.
# - solar_central_valais, solar_sion, solar_sierre : productions locales agrégées
# - solar_remote : production d'une centrale "distante" utilisée comme
#   référence physique (utilisée dans pv_yield pour normaliser l'irradiance).
#   /!\\ Cette colonne présente un offset nocturne suspect (probable décalage
#   de timezone à la source) — à traiter en amont si possible.
PROD_COLS = [
    "solar_central_valais",
    "solar_sion",
    "solar_sierre",
    "solar_remote",
]

# Colonnes de production "live" utilisées le matin même (J jusqu'à 10h).
# On exclut solar_remote ici car la donnée est potentiellement différée
# à la source et n'est pas garantie d'être disponible à 11h.
PROD_COLS_LIVE = [
    "solar_central_valais",
    "solar_sion",
    "solar_sierre",
]

# Pour le load historique : on prend les jours J-2 à J-8 (inclus).
# Pourquoi range(2, 9) : J-1 n'est pas complet à 11h le jour J
# (il manque le soir), donc on part de J-2 comme plus ancien jour "complet".
LOAD_HISTORY_DAYS = list(range(2, 9))   # = [2, 3, 4, 5, 6, 7, 8]

# Variables météo mesurées (jauges, pas prévisions) disponibles sur
# toutes les stations. Utilisées pour caractériser les conditions
# physiques vécues à J-2 et le début de J.
REAL_METEO_VARS = ["temp_2m", "glob_rad", "pressure", "relhum_2m"]

# Toutes les stations (plaine + altitude) disponibles pour les mesures.
# Pour les prévisions J+1, on utilise un sous-ensemble selon la variable
# (voir STATIONS_PLAINE et STATIONS_WIND plus bas).
STATIONS_ALL = [
    "Pully", "Sion", "Visp", "Montana",
    "Col_du_Grand_St-Bernard", "Les_Attelas",
]

# v13 : stations de plaine uniquement pour les prévisions d'irradiance.
# Les parcs PV Oiken étant en plaine, les stations d'altitude (Montana,
# Col_du_Grand_St-Bernard, Les_Attelas) ne sont pas représentatives.
STATIONS_PLAINE = ["Sion", "Visp", "Pully"]

# v13 : stations utilisées pour le vent (vitesse uniquement). Les stations
# d'altitude ont un régime de vent très différent de la plaine, ce qui
# bruitait le signal.
STATIONS_WIND = ["Sion", "Pully"]

# v13 : heures diurnes UTC où l'irradiance et le vent sont extraits.
# range(6, 20) = heures 06h, 07h, ..., 19h UTC (14 heures).
# En heure locale Suisse : 07h-21h (été) ou 06h-20h (hiver).
# On exclut la nuit car l'irradiance y est par définition nulle.
HOURS_DIURNAL = list(range(6, 20))

# v13 : variables météo prévues J+1. On traite glob_rad et wind_speed
# séparément (stations et heures restreintes) pour éviter la dilution.
# Les autres variables (temp, pressure, etc.) restent sur toutes les
# stations et toutes les heures car elles ont une valeur physique
# 24h/24 et partout.
PRED_VARS_JP1 = ["temp", "glob_rad", "pressure", "relhum", "precip", "sunshine"]

# Variables cycliques à encoder en sin/cos : vide en v13 car wind_dir
# (la seule variable angulaire) a été supprimée.
CYCLIC_PRED_VARS = set()

# ─── Gestion des horizons de prévision MeteoSuisse ───────────────────
# MeteoSuisse publie des runs de prévision toutes les heures, avec un
# pas de 3h pour les horizons. Selon l'heure UTC à prédire, l'horizon
# disponible depuis le run de 11h UTC-1 dépend de (heure % 3).
# HORIZONS_BY_MOD donne les horizons disponibles pour chaque résidu.
# Par ex. pour h_utc = 13 (13%3 = 1), les horizons dispos sont 1, 4, 7,
# ..., 34.
HORIZONS_BY_MOD = {
    0: list(range(3, 37, 3)),
    1: list(range(1, 35, 3)),
    2: list(range(2, 36, 3)),
}

# Dates de Ramadan (fin ouverte côté 2026) pour isoler le signal
# nocturne spécifique à cette période (forte consommation entre 20h
# et 6h dans certaines populations).
RAMADAN_DATES = {
    2022: (date(2022, 4,  2), date(2022, 5,  1)),
    2023: (date(2023, 3, 23), date(2023, 4, 20)),
    2024: (date(2024, 3, 11), date(2024, 4,  9)),
    2025: (date(2025, 3,  1), date(2025, 3, 29)),
    2026: (date(2026, 2, 18), date(2026, 3, 19)),
}

# Jours fériés Valais romand + fériés nationaux suisses applicables.
# Utilisé pour :
#   - is_holiday : indicateur binaire dans les features
#   - is_bridge_day : détection des jours "pont" (ouvré entre 2 jours off)
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

# v13 : Vacances scolaires Valais romand (Sion), format (début, fin) inclus.
# Hypothèse métier : pendant les vacances, la consommation "écoles +
# activités parascolaires" chute, mais la consommation "résidentiel diurne"
# augmente. Le modèle doit pouvoir distinguer ces jours.
# NB : les vacances d'été chevauchent plusieurs mois, c'est un "mode"
# long (8 semaines) à différencier clairement de l'année scolaire.
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


# ─────────────────────────────────────────────────────────────────────
# HELPERS — fonctions utilitaires pour les indicateurs calendaires et
#           le calcul d'horizon de prévision.
# ─────────────────────────────────────────────────────────────────────

def is_ramadan(d: date) -> bool:
    """Retourne True si la date tombe dans une période de Ramadan connue."""
    for start, end in RAMADAN_DATES.values():
        if start <= d <= end:
            return True
    return False


def ramadan_night_hours(d: date) -> list[int]:
    """
    Retourne les heures "nocturnes" pendant le Ramadan (00h-05h + 20h-23h).
    Ces heures voient un pic de consommation spécifique (suhoor/iftar).
    Si pas en Ramadan, retourne une liste vide.
    """
    if not is_ramadan(d):
        return []
    return list(range(0, 6)) + list(range(20, 24))


def is_school_holiday(d: date) -> bool:
    """Retourne True si la date tombe dans une période de vacances scolaires Valais."""
    for start, end in VACANCES_SCOLAIRES:
        if start <= d <= end:
            return True
    return False


def is_bridge_day(d: date) -> bool:
    """
    Détecte un jour "pont" : jour ouvré coincé entre des jours off.
    Exemples :
      - Vendredi ouvré après un jeudi férié
      - Lundi ouvré avant un mardi férié
      - Mardi ouvré entre un lundi férié et un jeudi férié (cas rare)

    Hypothèse métier : beaucoup de gens prennent ce jour en congé,
    la consommation ressemble plus à un weekend qu'à un jour ouvré.

    Exclut les jours qui sont déjà weekend, fériés, ou en vacances scolaires
    (sinon la catégorisation se superpose avec celle existante).
    """
    if d.weekday() >= 5:
        return False                    # déjà samedi/dimanche
    if d in FERIES:
        return False                    # déjà férié
    if is_school_holiday(d):
        return False                    # déjà en vacances scolaires

    yesterday = d - timedelta(days=1)
    tomorrow  = d + timedelta(days=1)

    def is_off(day: date) -> bool:
        """Un jour est 'off' s'il est weekend ou férié."""
        return day.weekday() >= 5 or day in FERIES

    # Cas 1 : vendredi après un jeudi off
    if d.weekday() == 4 and is_off(yesterday):
        return True
    # Cas 2 : lundi avant un mardi off
    if d.weekday() == 0 and is_off(tomorrow):
        return True
    # Cas 3 : coincé entre deux jours off (ex : mardi entre lundi férié et mercredi férié)
    if is_off(yesterday) and is_off(tomorrow):
        return True
    return False


def _get_pv_capacity_proxy(d: date) -> float:
    """
    Estime la capacité PV installée Oiken (en MWp) à une date donnée,
    par interpolation linéaire entre des ancrages annuels.

    Méthode : on suppose que la capacité Oiken croît proportionnellement
    au parc PV national suisse (valeurs d'ancrage en GW au 31 déc de
    chaque année). On applique un ratio de base calibré sur fin 2022.

    OIKEN_BASE_MWP = 55 MWp → valeur de référence au 1er oct 2022.
    NATIONAL_BASE_GW = 4.65 GW → parc national à la même date.

    Résultat : capacité estimée en MWp, utilisée comme feature
    `pv_capacity_MWp` pour que le modèle sache que le parc grandit.
    """
    from datetime import date as _date
    anchors = [
        (_date(2022, 10, 1), 4.65),
        (_date(2022, 12, 31), 4.65),
        (_date(2023, 12, 31), 6.20),
        (_date(2024, 12, 31), 8.00),
        (_date(2025, 12, 31), 9.51),
    ]
    # Avant le premier ancrage : on prend la valeur du premier ancrage
    if d <= anchors[0][0]:
        national_gw = anchors[0][1]
    # Après le dernier ancrage : on extrapole en plat (pas d'extrapolation linéaire)
    elif d >= anchors[-1][0]:
        national_gw = anchors[-1][1]
    # Entre deux ancrages : interpolation linéaire sur la fraction écoulée
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


def get_correct_horizon_jp1(h_utc: int) -> int | None:
    """
    Pour une heure UTC du jour cible J+1, retourne l'horizon de prévision
    MeteoSuisse à utiliser (issu du run disponible à 11h UTC le jour J).

    Règle anti-leakage : on ne peut utiliser que des horizons ≥ h_utc + 14.
    Le 14 vient du fait que le run est publié à 11h UTC le jour J, donc
    pour prédire l'heure h du jour J+1, il faut un horizon couvrant
    (24 - 11) + h = 13 + h heures. On prend 14 par sécurité (marge de
    disponibilité du run).

    Retourne :
      - L'horizon minimal valide compatible avec le modulo-3 de l'heure
      - None si aucun horizon disponible ne couvre ce besoin
    """
    mod = h_utc % 3
    available = HORIZONS_BY_MOD[mod]
    min_needed = h_utc + 14
    valid = [h for h in available if h >= min_needed]
    if valid:
        return min(valid)                # on prend le plus court (donc plus précis)
    else:
        return None


# ─────────────────────────────────────────────────────────────────────
# CHARGEMENT — lecture du CSV Oiken et du parquet météo
# ─────────────────────────────────────────────────────────────────────

def load_oiken(path: Path) -> pl.DataFrame:
    """
    Charge le CSV Oiken :
      - Parse le timestamp (format 'DD.MM.YYYY HH:MM')
      - L'attache au fuseau Europe/Zurich (le fichier est en heure locale)
      - Renomme les colonnes vers des noms plus courts
      - Trie par timestamp

    Gère les ambiguïtés de fuseau (passage à l'heure d'été/hiver) :
      - ambiguous="earliest" : en cas de répétition (retour à l'heure d'hiver),
        on prend la première occurrence
      - non_existent="null" : en cas d'heure sautée (passage à l'heure d'été),
        on met NaT

    Schema overrides : force Float64 sur les colonnes de production
    (par défaut Polars pourrait inférer Int64 sur des colonnes à 0 initiaux).
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
          .alias("timestamp")
    ).rename({
        "standardised load [-]":                 "load",
        "standardised forecast load [-]":        "load_forecast_oiken",
        "central valais solar production [kWh]": "solar_central_valais",
        "sion area solar production [kWh]":      "solar_sion",
        "sierre area production [kWh]":          "solar_sierre",
        "remote solar production [kWh]":         "solar_remote",
    }).sort("timestamp")
    print(f"  Oiken : {len(df):,} lignes | {df['timestamp'].drop_nulls()[0]} → {df['timestamp'][-1]}")
    return df


def load_meteo(path: Path) -> tuple[pl.DataFrame, pl.DataFrame, list[str]]:
    """
    Charge le parquet météo et retourne deux versions du même DataFrame :
      - df_utc    : timestamps en UTC (utilisé pour aligner avec les horizons
                    de prévision MeteoSuisse qui sont en UTC)
      - df_zurich : timestamps convertis en Europe/Zurich (utilisé pour
                    aligner avec les données Oiken en heure locale)

    real_cols = liste des colonnes de mesures réelles existantes (produit
    cartésien des variables × stations, filtré par présence dans le parquet).
    """
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
    """
    Extrait les 24 heures d'un jour donné (de 00h00 à 23h59:59 heure locale
    Europe/Zurich). Utilisé pour récupérer toutes les mesures d'une journée.
    """
    start = pl.datetime(day.year, day.month, day.day, 0, 0, 0, time_zone="Europe/Zurich")
    end   = pl.datetime(day.year, day.month, day.day, 23, 59, 59, time_zone="Europe/Zurich")
    return df.filter((pl.col("timestamp") >= start) & (pl.col("timestamp") <= end))


def get_morning_slice(df: pl.DataFrame, day: date, until_hour: int = 10) -> pl.DataFrame:
    """
    Extrait le début d'un jour donné (de 00h00 à until_hour:00 heure locale).
    Utilisé pour récupérer le matin du jour J jusqu'à 10h, données
    disponibles au moment de soumettre la prévision J+1 à 11h.
    """
    start = pl.datetime(day.year, day.month, day.day, 0, 0, 0, time_zone="Europe/Zurich")
    end   = pl.datetime(day.year, day.month, day.day, until_hour, 0, 0, time_zone="Europe/Zurich")
    return df.filter((pl.col("timestamp") >= start) & (pl.col("timestamp") <= end))


def series_stats(series: pl.Series, prefix: str) -> dict:
    """
    Calcule 4 statistiques descriptives (mean, max, min, std) sur une série,
    ignorant les NaN. Retourne un dict {prefix_mean: ..., prefix_max: ..., etc.}.

    Si toutes les valeurs sont nulles → retourne None partout (le modèle
    gère les None nativement comme NaN).
    """
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
    Construit le profil horaire d'une colonne sur 24h :
    moyenne par heure locale, produisant 24 features {prefix_h00, ..., prefix_h23}.

    Par exemple : hourly_profile(oiken_jm1, "load", "load_jm1") produit
      load_jm1_h00 = load moyen entre 00h00 et 00h59 locale à J-2
      load_jm1_h01 = load moyen entre 01h00 et 01h59 locale à J-2
      ... etc.

    /!\\ Les heures ici sont en timezone Europe/Zurich (données Oiken),
    alors que les features pred_*_tXX sont en UTC. Ne pas confondre.
    """
    result = {}
    for h in range(24):
        hour_vals = df_day.filter(pl.col("timestamp").dt.hour() == h)[col].drop_nulls()
        result[f"{prefix}_h{h:02d}"] = float(hour_vals.mean()) if len(hour_vals) > 0 else None
    return result


def real_meteo_stats(df_slice: pl.DataFrame, real_cols: list[str], prefix: str) -> dict:
    """
    Pour chaque colonne de mesure météo réelle, calcule (mean, max, min)
    sur la tranche temporelle fournie. Retourne un dict à plat.

    Utilisé deux fois par jour cible :
      - df_slice = J-2 complet → prefix "rmet_jm1"
      - df_slice = J matin jusqu'à 10h → prefix "rmet_jmorn"
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
# PRÉVISIONS J+1 (v13 restructuré)
# ─────────────────────────────────────────────────────────────────────

def extract_pred_jp1(meteo_utc: pl.DataFrame, target_day: date) -> dict:
    """
    Extrait les prévisions météo pour le jour cible J+1.

    Structure v13 :
      - temp/pressure/relhum/precip/sunshine : 5 vars × 6 stations × 24h = 720 feats
      - glob_rad : 3 stations plaine × 14h diurnes = 42 feats + 1 moyenne_day
      - wind_speed : 2 stations × 14h diurnes = 28 feats
      - wind_dir : SUPPRIMÉ (proxy PV indirect, remplacé par irradiance directe)
      - glob_rad variance (q10/q90/stde/...) : SUPPRIMÉ (anti-incentive pour
        le modèle à faire confiance à l'irradiance)

    Anti-leakage : chaque valeur vient du run MeteoSuisse disponible à 11h
    UTC le jour J, via get_correct_horizon_jp1(h_utc) qui applique la
    contrainte min_horizon = h_utc + 14.

    Convention de nommage : pred_<var>_<station>_t<heure_UTC>
    Ex : pred_temp_Sion_t13 = température prévue à 13h UTC le jour J+1
    à Sion, extraite du run du 11h UTC le jour J.
    """
    features = {}

    # Slice UTC du jour cible (24h)
    start = pl.datetime(target_day.year, target_day.month, target_day.day,
                        0, 0, 0, time_zone="UTC")
    end   = pl.datetime(target_day.year, target_day.month, target_day.day,
                        23, 59, 59, time_zone="UTC")
    day_utc = meteo_utc.filter((pl.col("timestamp") >= start) & (pl.col("timestamp") <= end))

    # Cas de bordure : aucune donnée pour ce jour → features remplies de None
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

    # Précompute : mapping heure UTC → index dans le slice (première
    # occurrence, utile si plusieurs timestamps par heure — cas edge).
    hours_utc = day_utc["timestamp"].dt.hour().to_list()
    hour_to_idx = {}
    for idx, h in enumerate(hours_utc):
        if h not in hour_to_idx:
            hour_to_idx[h] = idx

    def get_pred_value(var, station, h_utc):
        """
        Récupère la valeur prévue de 'var' à la station 'station' pour
        l'heure h_utc du jour cible, en utilisant l'horizon correct depuis
        le run de 11h.

        Nom de colonne cherché : pred_<var>_h<horizon>_<station>
        (format du parquet météo, où h<horizon> encode l'horizon depuis le run).
        """
        horizon = get_correct_horizon_jp1(h_utc)
        if horizon is None:
            return None                                 # pas d'horizon dispo (bord)
        col_name = f"pred_{var}_h{horizon}_{station}"
        if col_name not in day_utc.columns:
            return None                                 # colonne absente du parquet
        idx = hour_to_idx.get(h_utc)
        if idx is None:
            return None                                 # heure manquante dans le slice
        v = day_utc[col_name][idx]
        return float(v) if v is not None else None

    # ── temp/pressure/relhum/precip/sunshine : 6 stations × 24h
    # Pour ces variables, on garde toutes les stations et toutes les heures
    # car elles ont un sens physique 24h/24 et varient selon l'altitude.
    for var in PRED_VARS_JP1:
        if var == "glob_rad":
            continue                                    # traité séparément plus bas
        for station in STATIONS_ALL:
            for h_utc in range(24):
                features[f"pred_{var}_{station}_t{h_utc:02d}"] = get_pred_value(var, station, h_utc)

    # ── glob_rad : restreint à 3 stations plaine × 14h diurnes
    # Collecte simultanée des valeurs pour calculer la moyenne journalière
    irr_day_vals = []
    for station in STATIONS_PLAINE:
        for h_utc in HOURS_DIURNAL:
            val = get_pred_value("glob_rad", station, h_utc)
            features[f"pred_glob_rad_{station}_t{h_utc:02d}"] = val
            if val is not None:
                irr_day_vals.append(val)

    # Feature agrégée : moyenne d'irradiance diurne prévue, toutes stations
    # plaine confondues. Plus "concentrée" qu'une heure individuelle, souvent
    # très utilisée par le modèle.
    features["pred_glob_rad_mean_day"] = (
        sum(irr_day_vals) / len(irr_day_vals) if irr_day_vals else None
    )

    # ── wind_speed : 2 stations × 14h diurnes (wind_dir supprimé)
    for station in STATIONS_WIND:
        for h_utc in HOURS_DIURNAL:
            features[f"pred_wind_speed_{station}_t{h_utc:02d}"] = get_pred_value("wind_speed", station, h_utc)

    return features


# ─────────────────────────────────────────────────────────────────────
# PV YIELD RATIO (introduit en v11, conservé en v13)
# ─────────────────────────────────────────────────────────────────────

def compute_pv_yield_ratios(
    oiken: pl.DataFrame,
    meteo_zurich: pl.DataFrame,
    target_date: date,
) -> dict:
    """
    Calcule deux ratios de rendement PV sur des fenêtres glissantes
    30j et 90j :

        pv_yield_{N}j = max(solar_remote sur N jours) / max(glob_rad_Sion sur N jours)

    Interprétation : combien de kWh de production PV on observe par unité
    d'irradiance maximale. Proxy du rendement effectif récent du parc PV.

    Pourquoi utile : le parc PV Oiken grandit au fil du temps (+MWp installés),
    et ce ratio évolue aussi. En fournissant pv_yield glissant, on donne au
    modèle un "calibrage" à jour sans avoir à le coder explicitement.

    Utilisation aval : ces ratios sont multipliés par l'irradiance prévue
    pour produire les features d'interaction pred_pv_adj_{30j|90j}_tXX
    (voir compute_pv_interaction_features).

    Fenêtre : finit à day_jm1 = target_date - 2j (dernière donnée complète
    disponible à 11h le jour J), commence N-1 jours avant day_jm1.

    Retourne aussi solar_remote_max_{N}j (le numérateur brut) comme feature
    à part entière, car le modèle peut s'en servir indépendamment.

    Garde-fous :
      - Si glob_max < 10, on considère que l'échelle est dégénérée (pas
        d'irradiance mesurée) et on retourne None pour pv_yield.
      - Si remote_max ou glob_max est None, idem.
    """
    features = {}
    day_jm1 = target_date - timedelta(days=2)

    for window_days, label in [(30, "30j"), (90, "90j")]:
        window_start = day_jm1 - timedelta(days=window_days - 1)

        start_dt = pl.datetime(window_start.year, window_start.month, window_start.day,
                               0, 0, 0, time_zone="Europe/Zurich")
        end_dt   = pl.datetime(day_jm1.year, day_jm1.month, day_jm1.day,
                               23, 59, 59, time_zone="Europe/Zurich")

        oiken_window = oiken.filter(
            (pl.col("timestamp") >= start_dt) & (pl.col("timestamp") <= end_dt)
        )
        meteo_window = meteo_zurich.filter(
            (pl.col("timestamp") >= start_dt) & (pl.col("timestamp") <= end_dt)
        )

        # Numérateur : max de solar_remote sur la fenêtre (proxy production PV pic)
        if "solar_remote" in oiken_window.columns and len(oiken_window) > 0:
            remote_max = oiken_window["solar_remote"].drop_nulls().max()
            remote_max = float(remote_max) if remote_max is not None else None
        else:
            remote_max = None

        # Dénominateur : max d'irradiance mesurée à Sion (référence plaine)
        if "glob_rad_Sion" in meteo_window.columns and len(meteo_window) > 0:
            glob_max = meteo_window["glob_rad_Sion"].drop_nulls().max()
            glob_max = float(glob_max) if glob_max is not None else None
        else:
            glob_max = None

        # Ratio : production pic / irradiance pic → rendement effectif
        if remote_max is not None and glob_max is not None and glob_max > 10:
            features[f"pv_yield_{label}"] = remote_max / glob_max
        else:
            features[f"pv_yield_{label}"] = None

        # Feature brute additionnelle (le numérateur tout seul)
        features[f"solar_remote_max_{label}"] = remote_max

    return features


# ─────────────────────────────────────────────────────────────────────
# v13 : INTERACTIONS PV — irradiance × pv_yield
# ─────────────────────────────────────────────────────────────────────

def compute_pv_interaction_features(features: dict) -> dict:
    """
    Crée des features d'interaction explicite entre irradiance prévue et
    rendement PV récent :

        pred_pv_adj_{30j|90j}_t{XX} = mean(irr Sion/Visp/Pully à h=XX) × pv_yield_{30j|90j}

    Justification : LightGBM peut capturer des interactions multiplicatives
    via des splits d'arbres profonds, mais cette représentation est
    escaliée et coûteuse. Pré-calculer le produit donne une feature
    continue qui encode directement la physique "production ≈ irradiance ×
    rendement". Le modèle la voit en une seule variable, plus facile à
    exploiter qu'une combinaison à reconstruire.

    Deux versions 30j et 90j pour capter respectivement :
      - le rendement "récent" (mois écoulé, sensible aux extensions
        du parc et aux modifications)
      - le rendement "structurel" (3 mois, plus stable, moins bruité)

    Features produites :
      - 14 heures diurnes × 2 fenêtres = 28 features horaires
      - 2 agrégés journaliers (somme sur les 14h diurnes) = proxy d'énergie
        produite sur la journée
      Total : 30 features.

    Attention : pred_pv_adj_{label}_day est une SOMME (pas une moyenne) des
    14 valeurs horaires. Intention : avoir un proxy d'intégrale temporelle
    de la production PV sur la journée, analogue à kWh.
    """
    interaction_feats = {}

    pv_yield_30j = features.get("pv_yield_30j")
    pv_yield_90j = features.get("pv_yield_90j")

    for h_utc in HOURS_DIURNAL:
        # Moyenne d'irradiance prévue sur les 3 stations plaine à cette heure
        irr_vals = []
        for station in STATIONS_PLAINE:
            v = features.get(f"pred_glob_rad_{station}_t{h_utc:02d}")
            if v is not None:
                irr_vals.append(v)

        irr_mean = sum(irr_vals) / len(irr_vals) if irr_vals else None

        # Produit irradiance × pv_yield pour chaque fenêtre
        if irr_mean is not None and pv_yield_30j is not None:
            interaction_feats[f"pred_pv_adj_30j_t{h_utc:02d}"] = irr_mean * pv_yield_30j
        else:
            interaction_feats[f"pred_pv_adj_30j_t{h_utc:02d}"] = None

        if irr_mean is not None and pv_yield_90j is not None:
            interaction_feats[f"pred_pv_adj_90j_t{h_utc:02d}"] = irr_mean * pv_yield_90j
        else:
            interaction_feats[f"pred_pv_adj_90j_t{h_utc:02d}"] = None

    # Agrégés journaliers : SOMME des 14 valeurs horaires (proxy d'énergie)
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
    meteo_utc: pl.DataFrame,
    meteo_zurich: pl.DataFrame,
    real_cols: list[str],
) -> dict | None:
    """
    Construit le vecteur de features pour un jour cible J+1.

    Convention de référence temporelle :
      - target_date = jour à prédire (J+1 dans la convention métier)
      - day_j   = target_date - 1 jour = jour de soumission (J)
      - day_jm1 = target_date - 2 jours = dernière journée complète (J-2
        dans l'absolu, souvent appelée "J-1" relativement à day_j)

    /!\\ Le suffixe 'jm1' dans les noms de features est relatif au jour de
    soumission J, pas au jour cible. Donc solar_remote_jm1_hXX = valeur
    d'il y a 2 jours par rapport à la cible. Historique : conservé pour
    cohérence avec v11/v12.

    Retourne None si le jour cible a moins de 90 pas de load valides
    (journée trop incomplète pour produire une cible fiable).

    Structure du dict retourné :
      {
        "features": { ... ~1308 features ... },
        "target":   [ 96 valeurs de load ],
        "baseline": [ 96 valeurs de prévision Oiken (pour benchmark) ],
        "date":     target_date,
      }
    """
    day_j   = target_date - timedelta(days=1)
    day_jm1 = target_date - timedelta(days=2)

    # Vérifie que la cible est suffisamment complète (≥ 90 pas sur 96).
    # Si une journée a trop de trous, on la skip pour ne pas polluer Y.
    oiken_target = get_day_slice(oiken, target_date)
    if len(oiken_target) < 90:
        return None

    features = {}

    # ─── Load historique J-2 à J-8 ────────────────────────────────────
    # Pour chaque jour passé, on extrait :
    #   - 4 stats descriptives (mean, max, min, std)
    #   - 24 valeurs du profil horaire (moyenne par heure locale)
    # Labels : jm1, jm2, ..., jm7 (correspondant à delta = 2, 3, ..., 8)
    #
    # Objectif : donner au modèle le "pattern" de consommation récent,
    # notamment la même journée de semaine que la cible (jm7 ≈ même jour
    # semaine précédente).
    for delta in LOAD_HISTORY_DAYS:
        day_past = target_date - timedelta(days=delta)
        label = f"jm{delta - 1}"                        # jm1, jm2, ...
        oiken_past = get_day_slice(oiken, day_past)
        if len(oiken_past) >= 90:
            features.update(series_stats(oiken_past["load"], f"load_{label}"))
            features.update(hourly_profile(oiken_past, "load", f"load_{label}"))
        else:
            # Jour passé incomplet → on remplit de None pour garder
            # la cohérence du schéma (toutes les lignes ont les mêmes colonnes)
            for k in ["mean", "max", "min", "std"]:
                features[f"load_{label}_{k}"] = None
            for h in range(24):
                features[f"load_{label}_h{h:02d}"] = None

    # ─── Production solaire mesurée ───────────────────────────────────
    # J-2 complet : pour chaque colonne PV, total de la journée + profil horaire
    # J matin (jusqu'à 10h) : seulement les colonnes "live" (sans solar_remote)
    oiken_jm1       = get_day_slice(oiken, day_jm1)
    oiken_j_morning = get_morning_slice(oiken, day_j, until_hour=10)

    for col in PROD_COLS:
        if col in oiken_jm1.columns:
            features[f"{col}_jm1_total"] = float(oiken_jm1[col].sum())
            features.update(hourly_profile(oiken_jm1, col, f"{col}_jm1"))

    for col in PROD_COLS_LIVE:
        if col in oiken_j_morning.columns:
            features[f"{col}_j_morning_total"] = float(oiken_j_morning[col].sum())

    # ─── Météo mesurée (températures, irradiance, pression, humidité) ──
    # J-2 complet : stats (mean/max/min) sur la journée entière
    # J matin : idem mais seulement jusqu'à 10h
    #
    # Noter la présence en deux prefixes pour distinguer :
    #   rmet_jm1_*   = mesure à J-2 (jour complet)
    #   rmet_jmorn_* = mesure début J (matin)
    meteo_jm1 = get_day_slice(meteo_zurich, day_jm1)
    features.update(real_meteo_stats(meteo_jm1, real_cols, "rmet_jm1"))
    meteo_j_morning = get_morning_slice(meteo_zurich, day_j, until_hour=10)
    features.update(real_meteo_stats(meteo_j_morning, real_cols, "rmet_jmorn"))

    # ─── Prévisions J+1 (cœur du modèle) ──────────────────────────────
    # Cf extract_pred_jp1 : ~800 features avec convention pred_*_tXX en UTC
    features.update(extract_pred_jp1(meteo_utc, target_date))

    # ─── Proxy capacité PV installée (croît avec le temps) ────────────
    features["pv_capacity_MWp"] = _get_pv_capacity_proxy(target_date)

    # ─── PV yield ratio glissant (30j et 90j) ─────────────────────────
    # Doit être appelé APRÈS extract_pred_jp1 dans le cas où des features
    # dépendantes (pv_yield dépend des mesures solar_remote) seraient
    # utilisées en aval. Ici c'est autonome, mais l'ordre reste cohérent.
    features.update(compute_pv_yield_ratios(oiken, meteo_zurich, target_date))

    # ─── Interactions PV = irradiance prévue × pv_yield ───────────────
    # IMPORTANT : ce bloc DOIT être appelé APRÈS extract_pred_jp1 ET
    # compute_pv_yield_ratios, car il lit des features déjà présentes
    # dans le dict (pred_glob_rad_* et pv_yield_*).
    features.update(compute_pv_interaction_features(features))

    # ─── Calendaire (v13 : ajout vacances scolaires + jours pont) ─────
    doy = target_date.timetuple().tm_yday                  # jour de l'année 1..366
    features["dayofweek"]         = target_date.weekday()   # 0 = lundi, 6 = dimanche
    features["month"]             = target_date.month
    features["is_weekend"]        = int(target_date.weekday() >= 5)
    features["is_holiday"]        = int(target_date in FERIES)
    features["is_school_holiday"] = int(is_school_holiday(target_date))   # v13 nouveau
    features["is_bridge_day"]     = int(is_bridge_day(target_date))        # v13 nouveau
    features["is_ramadan"]        = int(is_ramadan(target_date))

    # Indicateurs horaires Ramadan : flag par heure (00..23) signalant
    # si cette heure tombe dans le créneau "nocturne Ramadan".
    # 24 features binaires, redondantes avec is_ramadan mais plus fines.
    ramadan_hours = set(ramadan_night_hours(target_date))
    for h in range(24):
        features[f"is_ramadan_h{h:02d}"] = int(h in ramadan_hours)

    # ─── Encodages cycliques (sin/cos) ────────────────────────────────
    # Pour que le modèle traite correctement la cyclicité :
    #   - jour de la semaine (période 7)
    #   - mois (période 12)
    #   - jour de l'année (période 365, ignore bissextile pour simplicité)
    # Sans ça, LightGBM verrait dimanche (6) comme "loin" de lundi (0) alors
    # qu'ils sont adjacents. Les encodages sin/cos rendent la distance correcte.
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
# PIPELINE PRINCIPAL — orchestration globale et sauvegarde parquet
# ─────────────────────────────────────────────────────────────────────

def main():
    """
    Orchestre la génération du dataset complet :
      1. Charge Oiken + météo
      2. Détermine la plage de dates cible :
           first_day = 9 jours après le début du CSV (pour avoir J-8 dispo)
           last_day  = avant-dernier jour du CSV (car target_date doit être
                       ≤ dernier_jour_CSV - 1 pour avoir le jour cible complet)
      3. Itère sur chaque jour cible, appelle build_features
      4. Concatène en 3 DataFrames (X, Y, B) et sauve en parquet
      5. Imprime un rapport de vérification (stats + features interdites + comptes)
    """
    print("=== Chargement des données ===")
    oiken = load_oiken(CSV)
    meteo_utc, meteo_zurich, real_cols = load_meteo(METEO)

    # Plage de dates cible : borne inférieure à 9 jours après le début du CSV
    # (on a besoin de J-8 pour LOAD_HISTORY_DAYS + marge de 1 jour pour J-1)
    first_ts  = oiken["timestamp"].drop_nulls()[0]
    first_day = first_ts.date() + timedelta(days=9)

    # Borne supérieure : on ne peut pas prédire le dernier jour du CSV
    # car il n'est pas complet pour les stats cible.
    last_day = oiken["timestamp"][-1].date() - timedelta(days=1)

    all_dates = [first_day + timedelta(days=i)
                 for i in range((last_day - first_day).days + 1)]

    print(f"\n=== Construction features v13 : {first_day} → {last_day} ({len(all_dates)} jours) ===")

    # Accumulateurs : une ligne par jour valide
    rows_X, rows_Y, rows_B, dates_ok = [], [], [], []

    for i, target_date in enumerate(all_dates):
        # Progress every 100 days
        if i % 100 == 0:
            print(f"  {i}/{len(all_dates)} — {target_date}")

        result = build_features(target_date, oiken, meteo_utc, meteo_zurich, real_cols)
        if result is None:
            continue                                    # jour cible trop incomplet → skip

        rows_X.append(result["features"])
        rows_Y.append(result["target"])
        rows_B.append(result["baseline"])
        dates_ok.append(str(result["date"]))

    print(f"\n  {len(dates_ok)} jours valides sur {len(all_dates)}")

    # ─── Construction de X (features) ─────────────────────────────────
    # Polars infère automatiquement le schéma à partir des dicts.
    # Une colonne 'date' est ajoutée comme index temporel (typé pl.Date).
    X = pl.DataFrame(rows_X).with_columns(
        pl.Series("date", dates_ok).str.strptime(pl.Date, "%Y-%m-%d")
    )
    # Place la colonne 'date' en première position, reste dans l'ordre original
    X = X.select(["date"] + [c for c in X.columns if c != "date"])

    # ─── Construction de Y (cibles) ───────────────────────────────────
    # n_steps = 96 en principe (pas de 15 min × 24h). Chaque ligne de Y
    # est un vecteur de 96 valeurs de load normalisé.
    n_steps = len(rows_Y[0])
    Y = pl.DataFrame(
        {f"load_t{i:03d}": [row[i] if i < len(row) else None for row in rows_Y]
         for i in range(n_steps)}
    ).with_columns(pl.Series("date", dates_ok).str.strptime(pl.Date, "%Y-%m-%d"))
    Y = Y.select(["date"] + [f"load_t{i:03d}" for i in range(n_steps)])

    # ─── Construction de B (baseline Oiken) ───────────────────────────
    # Prévision Oiken native pour le même jour, utilisée comme benchmark
    # dans le script d'entraînement.
    B = pl.DataFrame(
        {f"baseline_t{i:03d}": [row[i] if i < len(row) else None for row in rows_B]
         for i in range(n_steps)}
    ).with_columns(pl.Series("date", dates_ok).str.strptime(pl.Date, "%Y-%m-%d"))
    B = B.select(["date"] + [f"baseline_t{i:03d}" for i in range(n_steps)])

    # Sauvegarde parquet (compression efficace, lecture rapide, typage conservé)
    X.write_parquet(OUT / "X_features_GOLDEN.parquet")
    Y.write_parquet(OUT / "Y_target_GOLDEN.parquet")
    B.write_parquet(OUT / "B_baseline_GOLDEN.parquet")

    print(f"\n✓ X_features_GOLDEN : {X.shape[0]} jours × {X.shape[1]} colonnes")
    print(f"✓ Y_target_GOLDEN   : {Y.shape[0]} jours × {Y.shape[1]} colonnes")
    print(f"✓ B_baseline_GOLDEN : {B.shape[0]} jours × {B.shape[1]} colonnes")

    # ─── Rapport de vérification ──────────────────────────────────────
    # Contrôle que quelques features clés ont des valeurs plausibles
    # (pas toutes nulles, échelle raisonnable).
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

    # Contrôle d'intégrité : on vérifie qu'aucune des features explicitement
    # supprimées en v13 n'est revenue par inadvertance (ex : mauvais import,
    # copier-coller depuis v12). Si le listing n'est pas vide, c'est un bug.
    bad_cols = [c for c in X.columns if any(k in c for k in [
        "predJ_", "wind_dir", "glob_rad_q10", "glob_rad_q90",
        "glob_rad_stde", "glob_rad_std_stations", "glob_rad_spread",
    ])]
    if bad_cols:
        print(f"\n⚠ Features qui devraient être supprimées : {bad_cols[:10]}...")
    else:
        print(f"\n✓ Météo J, wind_dir, variances irradiance : supprimées")

    # Compte par catégorie pour traçabilité et sanity-check
    # (évolution v12 → v13 : wind passe de ~864 à 28, etc.)
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