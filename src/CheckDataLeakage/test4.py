"""
pipeline_features_v10.py
=======================
Pipeline de construction de features pour la prédiction de charge électrique J+1
sur le territoire de distribution Oiken (Valais, Suisse).

CONTEXTE GÉNÉRAL :
  Ce script transforme des données brutes (consommation Oiken, production PV,
  prévisions météo multi-stations) en un dataset structuré (X, Y, B) prêt
  pour l'entraînement d'un modèle LightGBM.

  Le problème : chaque jour J à 11h, on soumet une prévision de la courbe
  de charge normalisée pour les 24h de J+1 (96 pas quart-horaires).
  Le modèle doit battre le baseline d'Oiken (leur propre forecast).

ARCHITECTURE DU PIPELINE :
  1. Chargement des données Oiken (CSV) et météo (parquet multi-stations)
  2. Pour chaque jour cible (= J+1), construction d'un vecteur de features
     à partir des données STRICTEMENT disponibles à 11h le jour J :
       - Historique de charge (J-1 à J-7)
       - Production solaire observée (J-1 complet + J matin)
       - Mesures météo réelles (J-1 complet + J matin)
       - Prévisions météo (J et J+1, avec horizons anti-leakage)
       - Features calendaires (jour, mois, fériés, Ramadan)
       - Proxy de capacité PV installée (croissance du parc)
  3. Sérialisation en 3 parquets : X (features), Y (cibles), B (baseline)

ANTI-LEAKAGE :
  Le point critique de ce pipeline est d'éviter le "data leakage" temporel.
  Chaque feature doit être réalistement disponible AVANT la deadline de 11h
  le jour J. Les prévisions météo sont particulièrement sensibles : elles
  sont émises à différents horizons (h1 à h36), et seules celles émises
  avant 10h UTC le jour J sont utilisables. Les fonctions get_correct_horizon_*
  implémentent cette logique.

ÉVOLUTION DES VERSIONS :
  v6  : Correction leakage prévisions météo, ajout horizons courts pour J
  v7  : Profil horaire PV J-1 (forme de la courbe solaire de la veille)
  v10 : Features PV agrégées (total, diurne), écart-type inter-stations,
        incertitude irradiance (quantiles q10/q90), suppression PV J (bruit)

SORTIES :
  DATA/processed/X_features_v11.parquet  — matrice de features (1 ligne/jour)
  DATA/processed/Y_target_v11.parquet    — vecteur cible (96 pas quart-horaires)
  DATA/processed/B_baseline_v11.parquet  — forecast Oiken (pour comparaison)
"""

import math
import polars as pl
import numpy as np
from pathlib import Path
from datetime import timedelta, date

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
# Chemins vers les données sources et le répertoire de sortie.
# BASE pointe vers le dossier racine du projet ; les sous-dossiers
# contiennent le CSV Oiken, le parquet météo, et le dossier "processed"
# où seront écrits les parquets de sortie.

BASE    = Path(r"C:\Users\gab1a\OneDrive\Documents\energyinfo2\DATA")
CSV     = BASE / "oiken-data.csv"          # Données Oiken : charge + PV + baseline
METEO   = BASE / "meteo_multistation_v5.parquet"  # Prévisions + mesures météo 6 stations
OUT     = BASE / "processed"
OUT.mkdir(exist_ok=True)

# ── Colonnes de production solaire ──
# 4 zones géographiques du territoire Oiken avec production PV mesurée.
# "central_valais" couvre la zone principale, "sion" et "sierre" les zones
# urbaines, "remote" les installations éloignées (données reçues en différé).

PROD_COLS = [
    "solar_central_valais",
    "solar_sion",
    "solar_sierre",
    "solar_remote",
]

# Sources PV disponibles en temps réel le matin du jour J.
# solar_remote est EXCLUE car ses données ne sont reçues qu'à 2h du matin,
# donc pas encore disponibles pour le matin de J en temps réel.
# Cette distinction évite un leakage subtil : utiliser des données
# "remote" qui en production ne seraient pas encore arrivées.

PROD_COLS_LIVE = [
    "solar_central_valais",
    "solar_sion",
    "solar_sierre",
]

# ── Historique de charge ──
# On utilise la charge de J-1 à J-7 comme features.
# Le delta est par rapport à target_date (= J+1), donc :
#   delta=2 → J+1 - 2 = J-1 (veille de la soumission)
#   delta=8 → J+1 - 8 = J-7 (même jour de la semaine précédente)
# J-7 est particulièrement utile car la charge a une forte saisonnalité
# hebdomadaire (lundi ressemble au lundi précédent).

LOAD_HISTORY_DAYS = list(range(2, 9))  # delta 2 à 8

# ── Variables météo réelles (mesures) ──
# Ces 4 variables sont les mesures au sol disponibles dans le parquet météo.
# Elles servent de features "météo observée" pour J-1 et J matin.
# temp_2m    : température à 2m (°C)
# glob_rad   : irradiance globale (W/m²) — corrélée à la production PV
# pressure   : pression atmosphérique (hPa)
# relhum_2m  : humidité relative à 2m (%)

REAL_METEO_VARS = ["temp_2m", "glob_rad", "pressure", "relhum_2m"]

# ── Stations météo MeteoSuisse ──
# 6 stations couvrant le territoire Oiken et ses environs en Valais.
# La diversité géographique (plaine, montagne, col) permet au modèle
# de capter les gradients spatiaux de météo.
#   Pully      : Plateau, référence Léman (~460m)
#   Sion       : Plaine du Rhône, centre du Valais (~480m)
#   Visp       : Haut-Valais (~640m)
#   Montana    : Altitude moyenne (~1500m)
#   Col du GSB : Haute altitude (~2470m), conditions extrêmes
#   Les Attelas: Haute altitude (~2730m), station de montagne

STATIONS = [
    "Pully", "Sion", "Visp", "Montana",
    "Col_du_Grand_St-Bernard", "Les_Attelas",
]

# ── Variables de prévision météo ──
# Issues du modèle numérique (probablement COSMO/ICON de MeteoSuisse).
# Chaque variable est disponible pour chaque station et chaque horizon
# de prévision (h1 à h36).
# wind_dir est traitée spécialement : encodage sin/cos au lieu de la
# valeur brute (0-360°) pour éviter la discontinuité à 360°→0°.

PRED_VARS = ["temp", "glob_rad", "pressure", "relhum", "precip", "sunshine", "wind_speed", "wind_dir"]

# ── Variables d'incertitude irradiance (nouveauté v10) ──
# Quantiles et écart-type de l'irradiance prédite.
# Permettent au modèle de savoir si la prévision PV est fiable :
#   glob_rad_q10  : quantile 10% (scénario pessimiste)
#   glob_rad_q90  : quantile 90% (scénario optimiste)
#   glob_rad_stde : écart-type de l'ensemble
# Uniquement pour J+1, car l'analyse d'importance a montré que les
# features PV de J sont du bruit pur (importance ~0.6).

PRED_VARS_UNCERTAINTY = ["glob_rad_q10", "glob_rad_q90", "glob_rad_stde"]

# ── Variables à encodage cyclique ──
# wind_dir (direction du vent en degrés) nécessite un encodage sin/cos
# car 359° et 1° sont proches mais numériquement éloignés.
# L'encodage produit 2 features (sin, cos) au lieu de 1, mais le modèle
# peut correctement interpoler les directions proches de 0°/360°.

CYCLIC_PRED_VARS = {"wind_dir"}

# ── Mapping des horizons de prévision par modulo 3 ──
# Les prévisions météo sont émises toutes les 3 heures (runs du modèle
# numérique). Pour une heure UTC donnée, seuls certains horizons sont
# disponibles selon h_utc % 3 :
#   h_utc % 3 == 0 → horizons h3, h6, h9, ..., h36
#   h_utc % 3 == 1 → horizons h1, h4, h7, ..., h34
#   h_utc % 3 == 2 → horizons h2, h5, h8, ..., h35
# Cette structure vient du format de sortie du modèle météo : chaque run
# produit des prévisions aux heures alignées sur son pattern modulo 3.

HORIZONS_BY_MOD = {
    0: list(range(3, 37, 3)),   # h3, h6, ..., h36
    1: list(range(1, 35, 3)),   # h1, h4, ..., h34
    2: list(range(2, 36, 3)),   # h2, h5, ..., h35
}

# ── Conversion irradiance → production PV ──
# Formule simplifiée : P_pv = irradiance × surface × ratio_performance
# PV_SURFACE_M2  = surface PV totale estimée sur le territoire Oiken
# PV_PERF_RATIO  = ratio de performance global (pertes onduleur, câblage,
#                  température, salissure, etc.)
# Le résultat est en MW (division par 1e6 dans le code).

PV_SURFACE_M2  = 540_000   # m² de panneaux PV
PV_PERF_RATIO  = 0.75      # rendement système global

# ── Dates du Ramadan ──
# Le Ramadan modifie les patterns de consommation électrique :
# - Hausse de consommation la nuit (repas, activités sociales)
# - Baisse possible en journée
# L'effet est pertinent à Sion/Sierre qui ont une population musulmane
# non négligeable. Les dates varient chaque année (calendrier lunaire).

RAMADAN_DATES = {
    2022: (date(2022, 4,  2), date(2022, 5,  1)),
    2023: (date(2023, 3, 23), date(2023, 4, 20)),
    2024: (date(2024, 3, 11), date(2024, 4,  9)),
    2025: (date(2025, 3,  1), date(2025, 3, 29)),
    2026: (date(2026, 2, 18), date(2026, 3, 19)),
}

# ── Jours fériés valaisans ──
# Les fériés provoquent une chute de charge similaire au dimanche.
# Cette liste couvre les fériés fédéraux + cantonaux du Valais :
#   1er janvier, 2 janvier, 19 mars (St-Joseph), Vendredi saint,
#   Lundi de Pâques, Ascension, Lundi de Pentecôte, Fête-Dieu,
#   1er août (fête nationale), Toussaint, Immaculée Conception, Noël, St-Étienne.

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
    """Vérifie si une date tombe pendant le Ramadan.
    
    Parcourt le dictionnaire RAMADAN_DATES et vérifie l'inclusion.
    Retourne True si la date est dans l'une des plages définies.
    """
    for start, end in RAMADAN_DATES.values():
        if start <= d <= end:
            return True
    return False


def ramadan_night_hours(d: date) -> list[int]:
    """Retourne les heures de nuit pendant le Ramadan (0h-5h + 20h-23h).
    
    Pendant le Ramadan, les nuits sont plus actives (iftar, activités
    sociales post-rupture du jeûne). Ces heures sont encodées comme
    features binaires par heure (is_ramadan_h00 à is_ramadan_h23),
    permettant au modèle de moduler sa prédiction heure par heure
    pendant cette période.
    
    Retourne une liste vide si la date n'est pas en période de Ramadan.
    """
    if not is_ramadan(d):
        return []
    return list(range(0, 6)) + list(range(20, 24))


def _get_pv_capacity_proxy(d: date) -> float:
    """
    Estime la puissance PV cumulée installée sur le territoire Oiken (en MWp).
    
    POURQUOI CETTE FEATURE :
    La capacité PV installée croît chaque année (~+25-30% en Suisse).
    Cela signifie que pour une même irradiance, la production PV (et donc
    l'autoconsommation qui réduit la charge nette) augmente dans le temps.
    Sans cette feature, le modèle verrait une dérive temporelle inexpliquée.
    
    MÉTHODE :
    On part des données nationales suisses (Swissolar/IEA-PVPS) :
      Fin 2022 : 4.65 GW  (+1.08 GW installés en 2022)
      Fin 2023 : 6.20 GW  (+1.55 GW)
      Fin 2024 : 8.00 GW  (+1.78 GW)
      Fin 2025 : 9.51 GW  (prévision Swissolar : +1.51 GW)
    
    On interpole linéairement entre ces points d'ancrage, puis on applique
    un ratio proportionnel pour ramener à l'échelle du territoire Oiken
    (base : 55 MWp en octobre 2022 quand la Suisse avait 4.65 GW).
    
    LIMITE : c'est un proxy — la croissance locale peut différer de la
    moyenne nationale. En production, remplacer par la vraie valeur.
    """
    from datetime import date as _date
    
    # Points d'ancrage : (date, capacité nationale en GW)
    anchors = [
        (_date(2022, 10, 1), 4.65),
        (_date(2022, 12, 31), 4.65),
        (_date(2023, 12, 31), 6.20),
        (_date(2024, 12, 31), 8.00),
        (_date(2025, 12, 31), 9.51),
    ]
    
    # Interpolation linéaire entre les ancres
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
    
    # Mise à l'échelle Oiken : règle de trois par rapport à la base
    OIKEN_BASE_MWP = 55.0       # MWp Oiken en octobre 2022
    NATIONAL_BASE_GW = 4.65      # GW national en octobre 2022
    
    return OIKEN_BASE_MWP * national_gw / NATIONAL_BASE_GW


# ─────────────────────────────────────────────
# HORIZON MAPPING
# ─────────────────────────────────────────────
# Ces deux fonctions sont le CŒUR de la logique anti-leakage.
# Elles déterminent quel horizon de prévision utiliser pour chaque heure,
# en garantissant que la prévision a été émise AVANT la deadline de 10h UTC
# le jour J (= 11h heure locale Zurich, heure de soumission).
#
# Rappel du fonctionnement des prévisions météo :
# - Le modèle numérique tourne plusieurs fois par jour (runs)
# - Chaque run produit des prévisions pour les heures futures
# - L'horizon h = nombre d'heures entre l'émission et la cible
# - Plus l'horizon est court, plus la prévision est précise
#
# Contrainte : heure_émission = heure_cible - horizon ≤ 10h UTC
# Donc : horizon ≥ heure_cible - 10

def get_correct_horizon_jp1(h_utc: int) -> int | None:
    """
    Sélectionne l'horizon de prévision correct pour J+1 à l'heure h_utc.
    
    LOGIQUE :
    Pour une heure h_utc de J+1, l'émission doit être ≤ J 10h UTC.
    La distance minimale en heures entre J 10h UTC et J+1 h_utc est :
      horizon_min = h_utc + 14
      (car de J 10h à J+1 0h = 14h, plus h_utc heures supplémentaires)
    
    On prend le plus petit horizon disponible dans le pattern H%3
    qui est ≥ horizon_min. Un horizon plus petit = prévision plus fraîche
    = plus précise, donc on veut le minimum valide.
    
    EXEMPLES :
      J+1 00h UTC → horizon_min = 14 → prend h15 (pattern mod 0)
      J+1 06h UTC → horizon_min = 20 → prend h22 (pattern mod 0)
      J+1 12h UTC → horizon_min = 26 → prend h27 (pattern mod 0)
      J+1 22h UTC → horizon_min = 36 → prend h36 si dispo, sinon None
      J+1 23h UTC → horizon_min = 37 → AUCUN horizon valide → None
    
    Retourne None si aucun horizon ne satisfait la contrainte, ce qui
    signifie qu'on ne peut pas prédire cette heure sans leakage.
    Les features correspondantes seront à None dans le dataset.
    (Concerne les heures 22h-23h UTC = 00h-01h Zurich, peu critiques.)
    """
    mod = h_utc % 3
    available = HORIZONS_BY_MOD[mod]
    min_needed = h_utc + 14
    
    valid = [h for h in available if h >= min_needed]
    if valid:
        return min(valid)
    else:
        return None


def get_correct_horizon_j(h_utc: int) -> int:
    """
    Sélectionne l'horizon de prévision correct pour le jour J à l'heure h_utc.
    
    LOGIQUE :
    Pour le jour J (jour de soumission), on dispose de prévisions plus
    récentes que pour J+1, car la cible est plus proche.
    Contrainte : heure_émission = h_utc - horizon ≤ 10h UTC
    Donc : horizon ≥ max(h_utc - 10, 1)
    
    Par rapport à J+1, les horizons sont beaucoup plus courts, ce qui
    donne des prévisions plus précises pour le jour même.
    
    EXEMPLES ET GAIN vs ANCIENNE VERSION :
      J 01h UTC → horizon h1 (émis J 00h)    vs ancien h13 (émis J-1 12h) → +12h fraîcheur
      J 10h UTC → horizon h1 (émis J 09h)    vs ancien h13 (émis J-1 21h) → +12h fraîcheur
      J 16h UTC → horizon h7 (émis J 09h)    vs ancien h13 (émis J 03h)   → +6h fraîcheur
      J 22h UTC → horizon h13 (émis J 09h)   → inchangé (déjà optimal)
    
    UTILITÉ :
    Les prévisions de J servent à capturer l'inertie thermique et les
    conditions en cours, pas à prédire J+1 directement. L'idée est que
    si le jour J est anormalement chaud/froid, J+1 le sera probablement
    aussi en partie.
    
    Retourne toujours un horizon valide (fallback sur le max du pattern).
    """
    mod = h_utc % 3
    available = HORIZONS_BY_MOD[mod]
    min_needed = max(h_utc - 10, 1)
    
    valid = [h for h in available if h >= min_needed]
    if valid:
        return min(valid)
    else:
        # Fallback : prend l'horizon le plus long disponible.
        # Ne devrait pas arriver en pratique (h1 couvre presque tout).
        return max(available)


# ─────────────────────────────────────────────
# 1. CHARGEMENT
# ─────────────────────────────────────────────

def load_oiken(path: Path) -> pl.DataFrame:
    """
    Charge le CSV Oiken et le transforme en DataFrame Polars propre.
    
    Le CSV contient les colonnes :
      - timestamp          : format "DD.MM.YYYY HH:MM" en heure Zurich
      - standardised load  : charge normalisée (cible à prédire)
      - standardised forecast load : baseline Oiken (à battre)
      - 4 colonnes de production solaire par zone
    
    Transformations appliquées :
      1. Parsing du timestamp avec gestion de l'heure d'été/hiver
         (ambiguous="earliest" pour les heures dupliquées en octobre,
          non_existent="null" pour l'heure manquante en mars)
      2. Renommage des colonnes en noms courts utilisables comme identifiants
      3. Tri chronologique (essentiel pour les slices temporels)
    """
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
    """
    Charge le parquet météo multi-stations et prépare deux vues temporelles.
    
    Le parquet contient ~1200 colonnes :
      - Mesures réelles : {var}_{station} (ex: temp_2m_Sion)
      - Prévisions : pred_{var}_h{horizon}_{station} (ex: pred_temp_h15_Sion)
    
    Retourne :
      meteo_utc    : timestamps en UTC (pour les prévisions, car les horizons
                     sont définis en heures UTC)
      meteo_zurich : timestamps convertis en Europe/Zurich (pour les mesures
                     réelles, car les slices J-1/J matin sont en heure locale)
      real_cols    : liste des colonnes de mesures réelles présentes dans le
                     parquet (intersection de REAL_METEO_VARS × STATIONS)
    """
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
# 2. HELPERS D'EXTRACTION
# ─────────────────────────────────────────────

def get_day_slice(df: pl.DataFrame, day: date) -> pl.DataFrame:
    """
    Extrait toutes les lignes d'un jour donné (00:00:00 → 23:59:59 Zurich).
    
    Utilisé pour récupérer la charge/PV d'une journée complète.
    Le filtrage est en heure Zurich car les jours de consommation sont
    définis en heure locale (minuit à minuit).
    """
    start = pl.datetime(day.year, day.month, day.day, 0, 0, 0, time_zone="Europe/Zurich")
    end   = pl.datetime(day.year, day.month, day.day, 23, 59, 59, time_zone="Europe/Zurich")
    return df.filter((pl.col("timestamp") >= start) & (pl.col("timestamp") <= end))


def get_morning_slice(df: pl.DataFrame, day: date, until_hour: int = 10) -> pl.DataFrame:
    """
    Extrait les lignes d'un jour entre 00:00 et until_hour (défaut 10h Zurich).
    
    Utilisé pour les données du matin du jour J, disponibles avant la
    deadline de soumission à 11h. On prend jusqu'à 10h pour avoir une
    marge de sécurité (les données de 10h-11h pourraient ne pas être
    encore consolidées au moment de la soumission).
    """
    start = pl.datetime(day.year, day.month, day.day, 0, 0, 0, time_zone="Europe/Zurich")
    end   = pl.datetime(day.year, day.month, day.day, until_hour, 0, 0, time_zone="Europe/Zurich")
    return df.filter((pl.col("timestamp") >= start) & (pl.col("timestamp") <= end))


def series_stats(series: pl.Series, prefix: str) -> dict:
    """
    Calcule les statistiques descriptives (mean, max, min, std) d'une série.
    
    Utilisé pour résumer la charge d'un jour en 4 features compactes.
    Le mean capture le niveau moyen, max/min l'amplitude, std la variabilité.
    Retourne des None si la série est vide (jour sans données).
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
    Extrait le profil horaire (24 valeurs) d'une colonne pour un jour.
    
    Pour chaque heure 0-23, calcule la moyenne des valeurs de cette heure.
    (Un jour a ~4 valeurs par heure en données quart-horaires.)
    
    UTILITÉ :
    Le profil horaire capture la FORME de la courbe, pas seulement le niveau.
    Pour la charge : distingue les jours avec/sans pic du matin ou du soir.
    Pour le PV J-1 : une cloche propre indique ciel dégagé (régime qui
    persiste souvent 2-3 jours), une courbe irrégulière indique des nuages.
    
    Nommage : {prefix}_h00, {prefix}_h01, ..., {prefix}_h23
    """
    result = {}
    for h in range(24):
        hour_vals = df_day.filter(pl.col("timestamp").dt.hour() == h)[col].drop_nulls()
        result[f"{prefix}_h{h:02d}"] = float(hour_vals.mean()) if len(hour_vals) > 0 else None
    return result


def real_meteo_stats(df_slice: pl.DataFrame, real_cols: list[str], prefix: str) -> dict:
    """
    Calcule mean/max/min pour chaque variable météo réelle sur un slice temporel.
    
    Produit 3 features par colonne météo réelle (6 stations × 4 variables = 24
    colonnes → 72 features par slice).
    
    Utilisé pour :
      - rmet_jm1_*  : météo réelle de J-1 (journée complète)
      - rmet_jmorn_* : météo réelle du matin de J (jusqu'à 10h)
    
    Ces features complètent les prévisions en apportant les CONDITIONS RÉELLES
    observées, qui peuvent différer des prévisions. L'écart prévision/réalité
    est informatif : si hier la prévision était mauvaise, celle d'aujourd'hui
    est peut-être aussi biaisée.
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


def extract_pred_vector(
    meteo_utc: pl.DataFrame,
    target_day: date,
    horizon_func,
    prefix: str,
    extra_vars: list[str] | None = None,
) -> dict:
    """
    Construit le vecteur complet de prévisions météo heure par heure pour un jour.
    
    C'est la fonction la plus complexe du pipeline. Elle produit, pour chaque
    combinaison (variable × station × heure), la valeur de prévision correspondant
    à l'horizon correct selon la contrainte anti-leakage.
    
    PARAMÈTRES :
      meteo_utc    : DataFrame météo en UTC
      target_day   : jour pour lequel extraire les prévisions
      horizon_func : fonction h_utc → horizon correct
                     (get_correct_horizon_jp1 pour J+1, get_correct_horizon_j pour J)
                     Peut retourner None (= pas d'horizon valide sans leakage)
      prefix       : "pred" pour J+1, "predJ" pour J
                     Sert à nommer les features et éviter les collisions
      extra_vars   : variables supplémentaires (quantiles d'incertitude)
    
    FONCTIONNEMENT :
      1. Filtre le jour cible en UTC dans le DataFrame météo
      2. Pour chaque variable et station :
         a. Identifie les horizons nécessaires pour les 24 heures
         b. Charge les colonnes correspondantes (pred_{var}_h{horizon}_{station})
         c. Pour chaque heure h_utc :
            - Détermine l'horizon via horizon_func
            - Extrait la valeur à la bonne position temporelle
            - Applique l'encodage sin/cos si c'est wind_dir
      3. Calcule les features PV dérivées (irradiance moyenne × facteur conversion)
    
    FEATURES PRODUITES :
      Pour chaque variable standard : {prefix}_{var}_{station}_t{hh}
      Pour wind_dir : {prefix}_wind_dir_sin_{station}_t{hh} + _cos_
      Pour le PV : {prefix}_pv_MW_t{hh} (moyenne inter-stations × facteur)
    
    GESTION DES DONNÉES MANQUANTES :
      Si le jour cible n'a aucune donnée météo (len == 0), toutes les features
      sont mises à None. Idem si un horizon spécifique retourne None (anti-leakage).
      Le modèle LightGBM gère nativement les None.
    """
    all_vars = list(PRED_VARS) + (extra_vars or [])
    
    features = {}

    # ── Étape 1 : Filtrer le jour cible en UTC
    start = pl.datetime(target_day.year, target_day.month, target_day.day,
                        0, 0, 0, time_zone="UTC")
    end   = pl.datetime(target_day.year, target_day.month, target_day.day,
                        23, 59, 59, time_zone="UTC")
    day_utc = meteo_utc.filter(
        (pl.col("timestamp") >= start) & (pl.col("timestamp") <= end)
    )

    # Helper interne : génère les clés de features nulles pour une var/station.
    # Utilisé quand aucune donnée n'est disponible, pour que le DataFrame
    # final ait toujours le même nombre de colonnes (schema constant).
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

    # Si aucune donnée pour ce jour → tout à None
    if len(day_utc) == 0:
        for var in all_vars:
            for station in STATIONS:
                features.update(null_features_for_var(var, station))
        for h in range(24):
            features[f"{prefix}_pv_MW_t{h:02d}"] = None
        return features

    # ── Étape 2 : Index heure → position dans le DataFrame filtré
    # Permet de retrouver la ligne correspondant à une heure UTC donnée
    # sans refiltrer le DataFrame à chaque fois (optimisation).
    hours_utc = day_utc["timestamp"].dt.hour().to_list()
    hour_to_idx = {}
    for idx, h in enumerate(hours_utc):
        if h not in hour_to_idx:
            hour_to_idx[h] = idx

    # ── Étape 3 : Extraction variable par variable, station par station
    for var in all_vars:
        is_cyclic = var in CYCLIC_PRED_VARS

        for station in STATIONS:
            # Pré-collecte : quels horizons sont nécessaires pour cette var/station ?
            # On les charge tous d'un coup pour éviter des accès répétés au DataFrame.
            needed_horizons = set()
            for h_utc in range(24):
                horizon = horizon_func(h_utc)
                if horizon is not None:
                    needed_horizons.add(horizon)

            # Chargement des colonnes pour ces horizons
            # col_name = "pred_{var}_h{horizon}_{station}" dans le parquet
            col_values = {}
            for horizon in needed_horizons:
                col_name = f"pred_{var}_h{horizon}_{station}"
                if col_name in day_utc.columns:
                    col_values[horizon] = day_utc[col_name].to_list()

            # Extraction valeur par heure UTC
            for h_utc in range(24):
                horizon = horizon_func(h_utc)

                # Horizon None → aucun horizon valide sans leakage → feature None
                # (concerne h_utc 22-23 pour J+1 où horizon_min > 36)
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
                    # Encodage cyclique sin/cos pour wind_dir :
                    # sin(θ) et cos(θ) forment une représentation continue
                    # où 359° et 1° sont proches (contrairement à la valeur brute).
                    if raw_val is not None:
                        rad = raw_val * math.pi / 180.0
                        features[f"{prefix}_{var}_sin_{station}_t{h_utc:02d}"] = math.sin(rad)
                        features[f"{prefix}_{var}_cos_{station}_t{h_utc:02d}"] = math.cos(rad)
                    else:
                        features[f"{prefix}_{var}_sin_{station}_t{h_utc:02d}"] = None
                        features[f"{prefix}_{var}_cos_{station}_t{h_utc:02d}"] = None
                else:
                    features[f"{prefix}_{var}_{station}_t{h_utc:02d}"] = raw_val

    # ── Étape 4 : Feature PV estimé par heure
    # Convertit l'irradiance moyenne inter-stations en production PV estimée.
    # Formule : PV(MW) = irradiance(W/m²) × surface(m²) × ratio / 1e6
    # C'est une feature dérivée, pas une mesure : elle synthétise l'info
    # de 6 stations en une seule valeur physiquement interprétable.
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
    Construit le vecteur COMPLET de features pour prédire la charge de J+1 = target_date.
    
    SIMULATION DE LA CONTRAINTE TEMPORELLE :
    On simule exactement ce qui serait disponible à 11h (Zurich) le jour J :
      - J = target_date - 1 (jour de soumission)
      - J-1 = target_date - 2 (avant-veille, données complètes)
    
    STRUCTURE DES FEATURES (par catégorie) :
    
    1. CHARGE HISTORIQUE (J-1 à J-7) :
       - 4 stats (mean/max/min/std) + 24 valeurs horaires par jour
       - = 7 jours × 28 features = 196 features
       - Capte le niveau récent et la saisonnalité hebdomadaire
    
    2. PRODUCTION SOLAIRE OBSERVÉE :
       - J-1 complet : total + profil horaire × 4 sources = 4 + 96 = 100 features
       - J matin : total × 3 sources live = 3 features
       - La forme du profil PV J-1 indique le régime météo récent
    
    3. MÉTÉO RÉELLE :
       - J-1 complet : mean/max/min × ~24 colonnes = ~72 features
       - J matin : mean/max/min × ~24 colonnes = ~72 features
       - Conditions observées vs prévisions → biais du modèle météo
    
    4. PRÉVISIONS MÉTÉO J+1 :
       - 8 vars × 6 stations × 24h + wind_dir sin/cos = ~1200 features
       - + 3 vars incertitude × 6 stations × 24h = ~432 features
       - + PV estimé 24h + agrégats = ~50 features
       - Corps principal des features, avec horizons anti-leakage
    
    5. PRÉVISIONS MÉTÉO J :
       - Même structure que J+1 mais sans incertitude = ~1200 features
       - Capte l'inertie thermique du jour de soumission
    
    6. CALENDAIRE :
       - Jour de semaine, mois, weekend, férié, Ramadan
       - Encodages sin/cos pour la cyclicité
       - Proxy capacité PV installée
       - ~40 features
    
    TOTAL : ~3000+ features par jour (beaucoup seront éliminées par LightGBM)
    
    Retourne None si le jour cible n'a pas assez de données de charge
    (< 90 pas quart-horaires sur 96 attendus = seuil de complétude).
    """
    day_j   = target_date - timedelta(days=1)   # jour de soumission (J)
    day_jm1 = target_date - timedelta(days=2)   # avant-veille (J-1)

    # ── Vérification : le jour cible a-t-il assez de données de charge ?
    # On exige au moins 90 pas sur 96 (quart-horaire, 24h).
    # Un jour incomplet fausserait la cible d'entraînement.
    oiken_target = get_day_slice(oiken, target_date)
    if len(oiken_target) < 90:
        return None

    features = {}

    # ══════════════════════════════════════════════
    # BLOC 1 : CHARGE HISTORIQUE (J-1 à J-7)
    # ══════════════════════════════════════════════
    # Pour chaque jour passé, on extrait :
    #   - 4 stats agrégées (mean, max, min, std) → niveau et variabilité
    #   - 24 valeurs horaires → forme de la courbe de charge
    # Le label "jm{delta-1}" correspond au nombre de jours avant J+1 :
    #   delta=2 → jm1 = J-1, delta=3 → jm2 = J-2, etc.
    
    for delta in LOAD_HISTORY_DAYS:
        day_past   = target_date - timedelta(days=delta)
        label      = f"jm{delta - 1}"
        oiken_past = get_day_slice(oiken, day_past)

        if len(oiken_past) >= 90:
            features.update(series_stats(oiken_past["load"], f"load_{label}"))
            features.update(hourly_profile(oiken_past, "load", f"load_{label}"))
        else:
            # Jour incomplet → features à None (schema constant garanti)
            for k in ["mean", "max", "min", "std"]:
                features[f"load_{label}_{k}"] = None
            for h in range(24):
                features[f"load_{label}_h{h:02d}"] = None

    # ══════════════════════════════════════════════
    # BLOC 2 : PRODUCTION SOLAIRE OBSERVÉE
    # ══════════════════════════════════════════════
    oiken_jm1       = get_day_slice(oiken, day_jm1)
    oiken_j_morning = get_morning_slice(oiken, day_j, until_hour=10)

    # J-1 complet (4 sources, disponibles à 2h du matin le jour J) :
    # - Total journalier : quantité totale de PV injectée
    # - Profil horaire (v7) : FORME de la courbe → indicateur de nébulosité
    #   Cloche régulière = ciel dégagé, irrégulier = passages nuageux
    for col in PROD_COLS:
        if col in oiken_jm1.columns:
            features[f"{col}_jm1_total"] = float(oiken_jm1[col].sum())
            features.update(hourly_profile(oiken_jm1, col, f"{col}_jm1"))

    # J matin (3 sources live SEULEMENT, sans solar_remote) :
    # - Total du matin : production PV accumulée depuis le lever du soleil
    # - Indicateur de conditions actuelles (nuageux ce matin → peut-être demain)
    for col in PROD_COLS_LIVE:
        if col in oiken_j_morning.columns:
            features[f"{col}_j_morning_total"] = float(oiken_j_morning[col].sum())

    # ══════════════════════════════════════════════
    # BLOC 3 : MÉTÉO RÉELLE (MESURES AU SOL)
    # ══════════════════════════════════════════════
    # J-1 complet : conditions météo de la veille (toute la journée)
    meteo_jm1 = get_day_slice(meteo_zurich, day_jm1)
    features.update(real_meteo_stats(meteo_jm1, real_cols, "rmet_jm1"))

    # J matin : conditions météo du matin du jour de soumission (jusqu'à 10h)
    meteo_j_morning = get_morning_slice(meteo_zurich, day_j, until_hour=10)
    features.update(real_meteo_stats(meteo_j_morning, real_cols, "rmet_jmorn"))

    # ══════════════════════════════════════════════
    # BLOC 4 : PRÉVISIONS MÉTÉO J+1
    # ══════════════════════════════════════════════
    # Prévisions pour le jour cible avec horizons h15-h36 (anti-leakage).
    # Inclut les variables d'incertitude irradiance (v10) : q10, q90, stde.
    # Ces quantiles permettent au modèle de savoir si la prévision PV est
    # fiable ou incertaine (grande spread q90-q10 = forte incertitude).
    features.update(extract_pred_vector(
        meteo_utc, target_date,
        horizon_func=get_correct_horizon_jp1,
        prefix="pred",
        extra_vars=PRED_VARS_UNCERTAINTY,
    ))

    # ══════════════════════════════════════════════
    # BLOC 5 : PRÉVISIONS MÉTÉO J (jour de soumission)
    # ══════════════════════════════════════════════
    # Prévisions pour le jour même avec horizons courts h1-h14.
    # SANS variables d'incertitude PV : l'analyse d'importance (v10) a montré
    # que les features PV de J ont une importance groupée de ~0.6, ce qui
    # correspond à du bruit. Le PV du jour J n'aide pas à prédire la charge
    # de J+1 (la corrélation PV→charge passe par la météo, déjà capturée).
    features.update(extract_pred_vector(
        meteo_utc, day_j,
        horizon_func=get_correct_horizon_j,
        prefix="predJ",
    ))

    # ══════════════════════════════════════════════
    # BLOC 6 : FEATURES PV AGRÉGÉES J+1
    # ══════════════════════════════════════════════
    # Condensation du signal PV en quelques features "fortes" au lieu de
    # 144 features diluées (6 stations × 24h). L'idée : le modèle n'a pas
    # besoin de réassembler lui-même "beaucoup de PV demain" à partir de
    # 144 valeurs d'irradiance — on lui donne directement le résumé.
    
    prefix_pv = "pred"

    # Total PV prédit sur 24h (énergie totale estimée pour J+1)
    pv_24h = [features.get(f"{prefix_pv}_pv_MW_t{h:02d}") for h in range(24)]
    pv_clean = [v for v in pv_24h if v is not None]
    features[f"{prefix_pv}_pv_total"] = sum(pv_clean) if pv_clean else None

    # Total PV prédit heures diurnes (06h-20h UTC)
    # Exclut la nuit où le PV est forcément 0, pour un signal plus propre.
    pv_day = [features.get(f"{prefix_pv}_pv_MW_t{h:02d}") for h in range(6, 20)]
    pv_day_clean = [v for v in pv_day if v is not None]
    features[f"{prefix_pv}_pv_day"] = sum(pv_day_clean) if pv_day_clean else None

    # Irradiance moyenne diurne toutes stations (06h-20h UTC)
    # Moyenne spatiale et temporelle : 1 seul nombre résumant "combien de soleil
    # est attendu demain sur tout le territoire".
    irr_day_vals = []
    for h in range(6, 20):
        for station in STATIONS:
            v = features.get(f"{prefix_pv}_glob_rad_{station}_t{h:02d}")
            if v is not None:
                irr_day_vals.append(v)
    features[f"{prefix_pv}_glob_rad_mean_day"] = (
        sum(irr_day_vals) / len(irr_day_vals) if irr_day_vals else None
    )

    # Écart-type inter-stations par heure (proxy de nébulosité partielle)
    # Si toutes les stations ont la même irradiance → ciel uniformément
    # dégagé ou couvert. Si l'écart-type est élevé → nébulosité partielle
    # (soleil sur certaines zones, nuages sur d'autres). Ce pattern
    # affecte la production PV agrégée différemment d'un ciel uniforme.
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

    # Incertitude PV agrégée (v10) : spread moyen q90-q10 sur heures diurnes
    # Un spread élevé signifie que le modèle météo hésite entre scénarios
    # très différents (ex: passage d'un front possible mais incertain).
    # Le modèle ML peut utiliser cette info pour ajuster sa confiance.
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

    # ══════════════════════════════════════════════
    # BLOC 7 : FEATURES CALENDAIRES
    # ══════════════════════════════════════════════
    # La charge électrique a de fortes composantes calendaires :
    # - Hebdomadaire : lundi-vendredi vs weekend (charge ~15-20% plus basse)
    # - Annuelle : chauffage hiver, climatisation été, éclairage
    # - Fériés : chute de charge comme un dimanche
    # - Ramadan : modification des patterns nocturnes
    
    doy = target_date.timetuple().tm_yday  # jour de l'année (1-365)
    
    # Valeurs brutes (utilisables directement par LightGBM via splits)
    features["dayofweek"]  = target_date.weekday()      # 0=lundi, 6=dimanche
    features["month"]      = target_date.month           # 1-12
    features["is_weekend"] = int(target_date.weekday() >= 5)  # binaire
    features["is_holiday"] = int(target_date in FERIES)        # binaire
    features["is_ramadan"] = int(is_ramadan(target_date))      # binaire

    # Proxy de capacité PV installée (cf. _get_pv_capacity_proxy pour détails)
    features["pv_capacity_MWp"] = _get_pv_capacity_proxy(target_date)

    # Features Ramadan par heure : permet au modèle de moduler la prédiction
    # uniquement sur les heures nocturnes affectées par le Ramadan.
    ramadan_hours = set(ramadan_night_hours(target_date))
    for h in range(24):
        features[f"is_ramadan_h{h:02d}"] = int(h in ramadan_hours)

    # Encodages cycliques sin/cos pour les variables calendaires.
    # Nécessaires pour les modèles qui ne gèrent pas bien les discontinuités :
    #   - dayofweek : 6 (dimanche) est proche de 0 (lundi) → sin/cos
    #   - month : décembre (12) est proche de janvier (1) → sin/cos
    #   - doy : jour 365 est proche de jour 1 → sin/cos
    # LightGBM n'en a pas strictement besoin (il fait des splits), mais
    # ces features peuvent aider en combinaison avec d'autres dans les
    # splits multidimensionnels.
    features["sin_dow"]   = math.sin(2 * math.pi * target_date.weekday() / 7)
    features["cos_dow"]   = math.cos(2 * math.pi * target_date.weekday() / 7)
    features["sin_month"] = math.sin(2 * math.pi * (target_date.month - 1) / 12)
    features["cos_month"] = math.cos(2 * math.pi * (target_date.month - 1) / 12)
    features["sin_doy"]   = math.sin(2 * math.pi * doy / 365)
    features["cos_doy"]   = math.cos(2 * math.pi * doy / 365)

    # ══════════════════════════════════════════════
    # ASSEMBLAGE FINAL
    # ══════════════════════════════════════════════
    # Retourne un dict avec les features, la cible (charge 96 pas),
    # le baseline Oiken (pour comparaison), et la date.
    return {
        "features": features,
        "target":   oiken_target["load"].to_list(),       # 96 valeurs quart-horaires
        "baseline": oiken_target["load_forecast_oiken"].to_list(),  # forecast Oiken
        "date":     target_date,
    }


# ─────────────────────────────────────────────
# 4. PIPELINE PRINCIPAL
# ─────────────────────────────────────────────

def main():
    """
    Orchestre le pipeline complet : chargement → construction → sérialisation.
    
    DÉROULEMENT :
    1. Charge les données Oiken et météo
    2. Détermine la plage de dates valides :
       - Début : first_day + 9 jours (pour avoir J-7 disponible)
       - Fin : last_day - 1 (pour que J+1 ait des données complètes)
    3. Itère sur chaque jour de la plage et appelle build_features()
    4. Assemble les résultats en 3 DataFrames Polars et les sérialise
    
    SORTIE :
    3 fichiers parquet alignés par date :
      - X_features_v11.parquet : matrice de features (1 ligne = 1 jour, ~3000 colonnes)
      - Y_target_v11.parquet   : cible de charge (1 ligne = 96 pas quart-horaires)
      - B_baseline_v11.parquet : baseline Oiken (même structure que Y)
    
    Note : les fichiers sont nommés v11 malgré le script v10 — probablement
    une incrémentation de version du fichier de sortie.
    """
    print("=== Chargement des données ===")
    oiken                        = load_oiken(CSV)
    meteo_utc, meteo_zurich, real_cols = load_meteo(METEO)

    # Plage de dates : on démarre 9 jours après le début des données Oiken
    # pour garantir que l'historique J-7 est disponible (delta max = 8).
    first_ts  = oiken["timestamp"].drop_nulls()[0]
    first_day = first_ts.date() + timedelta(days=9)
    last_day  = oiken["timestamp"][-1].date() - timedelta(days=1)

    all_dates = [first_day + timedelta(days=i)
                 for i in range((last_day - first_day).days + 1)]

    print(f"\n=== Construction features v10 : {first_day} → {last_day} ({len(all_dates)} jours) ===")

    rows_X, rows_Y, rows_B, dates_ok = [], [], [], []

    for i, target_date in enumerate(all_dates):
        if i % 100 == 0:
            print(f"  {i}/{len(all_dates)} — {target_date}")

        result = build_features(target_date, oiken, meteo_utc, meteo_zurich, real_cols)
        if result is None:
            # Jour sauté (données de charge insuffisantes)
            continue

        rows_X.append(result["features"])
        rows_Y.append(result["target"])
        rows_B.append(result["baseline"])
        dates_ok.append(str(result["date"]))

    print(f"\n  {len(dates_ok)} jours valides sur {len(all_dates)}")

    # ── Sérialisation en parquets ──
    # X : DataFrame de features avec colonne "date" en première position.
    # Polars infère les types à partir des dicts — Float64 pour les numériques,
    # Null pour les features manquantes.
    X = pl.DataFrame(rows_X).with_columns(
        pl.Series("date", dates_ok).str.strptime(pl.Date, "%Y-%m-%d")
    )
    X = X.select(["date"] + [c for c in X.columns if c != "date"])

    # Y : Cible de charge. Chaque ligne a n_steps colonnes (96 pas quart-horaires).
    # Nommage : load_t000, load_t001, ..., load_t095
    n_steps = len(rows_Y[0])
    Y = pl.DataFrame(
        {f"load_t{i:03d}": [row[i] if i < len(row) else None for row in rows_Y]
         for i in range(n_steps)}
    ).with_columns(pl.Series("date", dates_ok).str.strptime(pl.Date, "%Y-%m-%d"))
    Y = Y.select(["date"] + [f"load_t{i:03d}" for i in range(n_steps)])

    # B : Baseline Oiken. Même structure que Y pour comparaison directe.
    B = pl.DataFrame(
        {f"baseline_t{i:03d}": [row[i] if i < len(row) else None for row in rows_B]
         for i in range(n_steps)}
    ).with_columns(pl.Series("date", dates_ok).str.strptime(pl.Date, "%Y-%m-%d"))
    B = B.select(["date"] + [f"baseline_t{i:03d}" for i in range(n_steps)])

    # Écriture des parquets
    X.write_parquet(OUT / "X_features_v11.parquet")
    Y.write_parquet(OUT / "Y_target_v11.parquet")
    B.write_parquet(OUT / "B_baseline_v11.parquet")

    # ── Résumé diagnostique des features ──
    # Comptage par catégorie pour vérifier la cohérence du pipeline.
    # Ce bloc peut être supprimé une fois la validation faite.
    cols = [c for c in X.columns if c != "date"]
    pred_jp1 = [c for c in cols if c.startswith("pred_") and not c.startswith("predJ_")]
    pred_j   = [c for c in cols if c.startswith("predJ_")]
    load_c   = [c for c in cols if c.startswith("load_")]
    solar_c  = [c for c in cols if c.startswith("solar_")]
    rmet_c   = [c for c in cols if c.startswith("rmet_")]
    cal_c    = [c for c in cols if c not in pred_jp1 + pred_j + load_c + solar_c + rmet_c]

    # Vérification anti-leakage : s'assurer que wind_dir est bien encodé
    # en sin/cos et qu'aucune colonne brute ne subsiste.
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