"""
preprocessing.py

Este archivo prepara los datos del proyecto antes de usarlos en el recomendador.

Responsabilidades:

1. Cargar archivos CSV crudos desde data/raw/.
2. Limpiar nombres de columnas.
3. Construir datos procesados de jugadores desde games_details.csv.
4. Construir datos procesados de equipos desde games.csv y games_details.csv.
5. Construir datos procesados de enfrentamientos desde games.csv.
6. Crear archivos procesados en data/processed/.

Archivos esperados de entrada:

- data/raw/players.csv
- data/raw/teams.csv
- data/raw/games.csv
- data/raw/games_details.csv

Archivos generados:

- data/processed/processed_players.csv
- data/processed/processed_teams.csv
- data/processed/processed_matchups.csv
- data/processed/latest_roster.csv
- data/processed/player_profiles.csv
"""

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:
    from .roster import build_and_save_roster_outputs
except ImportError:
    from roster import build_and_save_roster_outputs


# ---------------------------------------------------------------------
# Rutas del proyecto
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

RAW_PLAYERS_PATH = RAW_DATA_DIR / "players.csv"
RAW_TEAMS_PATH = RAW_DATA_DIR / "teams.csv"
RAW_GAMES_PATH = RAW_DATA_DIR / "games.csv"
RAW_GAME_DETAILS_PATH = RAW_DATA_DIR / "games_details.csv"

PROCESSED_PLAYERS_PATH = PROCESSED_DATA_DIR / "processed_players.csv"
PROCESSED_TEAMS_PATH = PROCESSED_DATA_DIR / "processed_teams.csv"
PROCESSED_MATCHUPS_PATH = PROCESSED_DATA_DIR / "processed_matchups.csv"
LATEST_ROSTER_PATH = PROCESSED_DATA_DIR / "latest_roster.csv"
PLAYER_PROFILES_PATH = PROCESSED_DATA_DIR / "player_profiles.csv"


# ---------------------------------------------------------------------
# Columnas finales esperadas por el recomendador
# ---------------------------------------------------------------------

PLAYER_REQUIRED_COLUMNS = [
    "player_id",
    "player_name",
    "team_id",
    "team",
    "position",
    "games_played",
    "minutes",
    "points",
    "assists",
    "rebounds",
    "steals",
    "blocks",
    "turnovers",
    "fg_pct",
    "three_pct",
    "three_attempts",
    "usage_rate",
    "offensive_rating",
    "defensive_rating",
    "pace",
    "plus_minus",
    "scoring",
    "playmaking",
    "defense",
    "rebounding",
    "spacing",
    "versatility",
]

TEAM_REQUIRED_COLUMNS = [
    "team",
    "offensive_rating",
    "defensive_rating",
    "pace",
]

MATCHUP_REQUIRED_COLUMNS = [
    "game_id",
    "team",
    "opponent_team",
    "team_score",
    "opponent_score",
    "home",
    "win",
]


PLAYER_NUMERIC_COLUMNS = [
    "player_id",
    "team_id",
    "games_played",
    "minutes",
    "points",
    "assists",
    "rebounds",
    "steals",
    "blocks",
    "turnovers",
    "fg_pct",
    "three_pct",
    "three_attempts",
    "usage_rate",
    "offensive_rating",
    "defensive_rating",
    "pace",
    "plus_minus",
    "scoring",
    "playmaking",
    "defense",
    "rebounding",
    "spacing",
    "versatility",
]

TEAM_NUMERIC_COLUMNS = [
    "offensive_rating",
    "defensive_rating",
    "pace",
]

MATCHUP_NUMERIC_COLUMNS = [
    "team_score",
    "opponent_score",
    "home",
    "win",
]


# ---------------------------------------------------------------------
# Funciones generales
# ---------------------------------------------------------------------


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza nombres de columnas.

    Ejemplos:
    PLAYER_NAME -> player_name
    TEAM_ABBREVIATION -> team
    FG3_PCT -> three_pct
    """
    df = df.copy()

    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
        .str.replace("%", "_pct", regex=False)
    )

    rename_map = {
        # Identidad
        "player": "player_name",
        "name": "player_name",
        "player_full_name": "player_name",
        "team_abbreviation": "team",
        "abbreviation": "team",
        "tm": "team",
        "team_name": "team",
        # Posición
        "pos": "position",
        "start_position": "position",
        # Box score
        "min": "minutes",
        "mp": "minutes",
        "pts": "points",
        "ast": "assists",
        "trb": "rebounds",
        "reb": "rebounds",
        "stl": "steals",
        "blk": "blocks",
        "to": "turnovers",
        "tov": "turnovers",
        # Porcentajes
        "fg": "fg_pct",
        "fg_pct_pct": "fg_pct",
        "3p_pct": "three_pct",
        "fg3_pct": "three_pct",
        "three_point_pct": "three_pct",
        "three_pct_pct": "three_pct",
        # Tiros de tres
        "3pa": "three_attempts",
        "fg3a": "three_attempts",
        "three_point_attempts": "three_attempts",
        # Uso / ratings si ya existen
        "usg_pct": "usage_rate",
        "usage_pct": "usage_rate",
        "usage_rate_pct": "usage_rate",
        "ortg": "offensive_rating",
        "off_rtg": "offensive_rating",
        "drtg": "defensive_rating",
        "def_rtg": "defensive_rating",
        # Rival
        "opp": "opponent_team",
        "opponent": "opponent_team",
    }

    df = df.rename(columns=rename_map)

    return df


def clean_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia espacios en columnas de texto sin convertir NaN en texto 'nan'.
    """
    df = df.copy()

    text_columns = df.select_dtypes(include=["object"]).columns

    for col in text_columns:
        df[col] = df[col].apply(
            lambda value: value.strip() if isinstance(value, str) else value
        )

    return df


def validate_required_columns(
    df: pd.DataFrame,
    required_columns: List[str],
    df_name: str,
) -> None:
    """
    Verifica que un DataFrame tenga las columnas necesarias.
    """
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(
            f"El archivo '{df_name}' no contiene las columnas requeridas: "
            f"{missing_columns}"
        )


def convert_numeric_columns(
    df: pd.DataFrame,
    numeric_columns: List[str],
) -> pd.DataFrame:
    """
    Convierte columnas numéricas a float si existen.
    """
    df = df.copy()

    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def fix_percentage_scale(
    df: pd.DataFrame,
    percentage_columns: List[str],
) -> pd.DataFrame:
    """
    Convierte porcentajes de escala 0-100 a escala 0-1.

    Ejemplo:
    45.3 -> 0.453
    """
    df = df.copy()

    for col in percentage_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            median_value = df[col].median(skipna=True)

            if pd.notna(median_value) and median_value > 1:
                df[col] = df[col] / 100

    return df


def fill_missing_numeric_values(
    df: pd.DataFrame,
    numeric_columns: List[str],
) -> pd.DataFrame:
    """
    Rellena valores faltantes en columnas numéricas usando la mediana.
    Si toda una columna está vacía, usa 0.
    """
    df = df.copy()

    for col in numeric_columns:
        if col in df.columns:
            median_value = df[col].median(skipna=True)

            if pd.isna(median_value):
                df[col] = df[col].fillna(0)
            else:
                df[col] = df[col].fillna(median_value)

    return df


def remove_duplicates(df: pd.DataFrame, subset: List[str]) -> pd.DataFrame:
    """
    Elimina duplicados usando columnas existentes.
    """
    existing_subset = [col for col in subset if col in df.columns]

    if not existing_subset:
        return df.reset_index(drop=True)

    return df.drop_duplicates(subset=existing_subset).reset_index(drop=True)


def parse_minutes_value(value) -> float:
    """
    Convierte minutos tipo '12:34' a minutos decimales.

    Ejemplo:
    '12:30' -> 12.5
    """
    if pd.isna(value):
        return 0.0

    if isinstance(value, (int, float)):
        return float(value)

    value = str(value).strip()

    if value == "":
        return 0.0

    if ":" in value:
        try:
            minutes, seconds = value.split(":")[:2]
            return float(minutes) + float(seconds) / 60
        except ValueError:
            return 0.0

    try:
        return float(value)
    except ValueError:
        return 0.0


def parse_minutes_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte la columna minutes a formato numérico.
    """
    df = df.copy()

    if "minutes" in df.columns:
        df["minutes"] = df["minutes"].apply(parse_minutes_value)

    return df


def get_mode_text(series: pd.Series, default_value: str = "UNK") -> str:
    """
    Devuelve el valor de texto más frecuente en una serie.
    """
    cleaned = series.dropna().astype(str).str.strip()

    cleaned = cleaned[~cleaned.isin(["", "nan", "None", "NaN", "NA"])]

    if cleaned.empty:
        return default_value

    return cleaned.mode().iloc[0]


def get_mode_numeric(series: pd.Series, default_value: int = 0) -> int:
    """
    Devuelve el valor numérico más frecuente en una serie.
    """
    cleaned = pd.to_numeric(series, errors="coerce").dropna()

    if cleaned.empty:
        return default_value

    return int(cleaned.mode().iloc[0])


# ---------------------------------------------------------------------
# Inferencia simple de posiciones
# ---------------------------------------------------------------------

UNKNOWN_POSITION_VALUES = {
    "",
    "UNK",
    "UNKNOWN",
    "NAN",
    "NONE",
    "NA",
    "N/A",
    "0",
}


def normalize_position_label(value) -> str:
    """
    Normaliza posiciones a grupos simples usados por el recomendador:
    - G = guard
    - F = forward / wing
    - C = center / big

    Mantiene UNK si no hay información confiable.
    """
    if pd.isna(value):
        return "UNK"

    position = str(value).strip().upper()

    if position in UNKNOWN_POSITION_VALUES:
        return "UNK"

    position = position.replace(" ", "").replace("_", "-")

    if position in ["PG", "SG", "G", "GUARD"]:
        return "G"

    if position in ["SF", "PF", "F", "FORWARD"]:
        return "F"

    if position in ["C", "CENTER"]:
        return "C"

    # Posiciones mixtas.
    # Para clase, las tratamos como wing salvo que sean claramente center.
    if position in ["G-F", "F-G"]:
        return "F"

    if position in ["F-C", "C-F"]:
        return "F"

    if "C" in position and "G" not in position:
        return "C"

    if "G" in position and "C" not in position:
        return "G"

    if "F" in position:
        return "F"

    return "UNK"


def is_unknown_position(value) -> bool:
    """
    Indica si una posición es desconocida.
    """
    return normalize_position_label(value) == "UNK"


def build_player_position_lookup(players_df: pd.DataFrame) -> Dict[str, str]:
    """
    Construye un diccionario player_name -> position usando players.csv si trae posición.

    Esto permite rescatar posiciones cuando games_details.csv trae START_POSITION vacío
    porque muchos jugadores no fueron titulares.
    """
    players_df = clean_column_names(players_df)
    players_df = clean_text_columns(players_df)

    if "player_name" not in players_df.columns or "position" not in players_df.columns:
        return {}

    temp_df = players_df[["player_name", "position"]].copy()
    temp_df["player_name"] = temp_df["player_name"].astype(str).str.strip()
    temp_df["position"] = temp_df["position"].apply(normalize_position_label)

    temp_df = temp_df[temp_df["player_name"] != ""]
    temp_df = temp_df[temp_df["position"] != "UNK"]

    if temp_df.empty:
        return {}

    lookup = (
        temp_df.groupby("player_name")["position"]
        .agg(lambda values: get_mode_text(values, default_value="UNK"))
        .to_dict()
    )

    return lookup


def fill_positions_from_lookup(
    df: pd.DataFrame,
    position_lookup: Dict[str, str],
) -> pd.DataFrame:
    """
    Rellena posiciones desconocidas usando un diccionario player_name -> position.
    """
    df = df.copy()

    if (
        not position_lookup
        or "player_name" not in df.columns
        or "position" not in df.columns
    ):
        return df

    unknown_mask = df["position"].apply(is_unknown_position)

    mapped_positions = df.loc[unknown_mask, "player_name"].map(position_lookup)

    df.loc[unknown_mask, "position"] = mapped_positions.fillna(
        df.loc[unknown_mask, "position"]
    )

    df["position"] = df["position"].apply(normalize_position_label)

    return df


def infer_position_from_stats(row: pd.Series) -> str:
    """
    Infiere posición con una heurística estadística simple y defendible.

    Intuición:
    - Muchos rebotes/bloqueos por 36 minutos -> C.
    - Muchas asistencias por 36 con pocos rebotes/bloqueos -> G.
    - Perfil intermedio o balanceado -> F.

    Esta regla no intenta ser perfecta; solo evita que UNK destruya position_fit.
    """
    current_position = normalize_position_label(row.get("position", "UNK"))

    if current_position != "UNK":
        return current_position

    minutes = pd.to_numeric(row.get("minutes", 0), errors="coerce")

    if pd.isna(minutes) or minutes <= 0:
        return "F"

    factor = 36.0 / minutes

    assists_per36 = pd.to_numeric(row.get("assists", 0), errors="coerce")
    rebounds_per36 = pd.to_numeric(row.get("rebounds", 0), errors="coerce")
    blocks_per36 = pd.to_numeric(row.get("blocks", 0), errors="coerce")
    three_attempts_per36 = pd.to_numeric(row.get("three_attempts", 0), errors="coerce")

    assists_per36 = 0.0 if pd.isna(assists_per36) else assists_per36 * factor
    rebounds_per36 = 0.0 if pd.isna(rebounds_per36) else rebounds_per36 * factor
    blocks_per36 = 0.0 if pd.isna(blocks_per36) else blocks_per36 * factor
    three_attempts_per36 = (
        0.0 if pd.isna(three_attempts_per36) else three_attempts_per36 * factor
    )

    # Big / center: rebote y protección del aro.
    if rebounds_per36 >= 9.0 or blocks_per36 >= 1.4:
        return "C"

    # Guard: creación con poco perfil interior.
    if assists_per36 >= 5.0 and rebounds_per36 <= 6.5 and blocks_per36 <= 0.8:
        return "G"

    # Guard tirador secundario.
    if (
        assists_per36 >= 3.5
        and three_attempts_per36 >= 4.0
        and rebounds_per36 <= 5.5
        and blocks_per36 <= 0.7
    ):
        return "G"

    # Wing / forward como clase intermedia.
    return "F"


def infer_missing_player_positions(players_df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia e infiere posiciones faltantes en el DataFrame agregado de jugadores.
    """
    players_df = players_df.copy()

    if "position" not in players_df.columns:
        players_df["position"] = "UNK"

    players_df["position"] = players_df["position"].apply(normalize_position_label)

    unknown_before = int((players_df["position"] == "UNK").sum())

    if unknown_before > 0:
        players_df["position"] = players_df.apply(infer_position_from_stats, axis=1)

    players_df["position"] = players_df["position"].apply(normalize_position_label)

    return players_df


def first_existing_column(
    df: pd.DataFrame,
    candidates: List[str],
) -> Optional[str]:
    """
    Devuelve la primera columna existente dentro de una lista de candidatas.
    """
    for col in candidates:
        if col in df.columns:
            return col

    return None


def minmax_scale(series: pd.Series) -> pd.Series:
    """
    Escala una columna numérica entre 0 y 1.
    Si todos los valores son iguales, devuelve 0.5.
    """
    series = pd.to_numeric(series, errors="coerce").fillna(0)

    min_value = series.min()
    max_value = series.max()

    if max_value == min_value:
        return pd.Series(0.5, index=series.index)

    return (series - min_value) / (max_value - min_value)


def build_team_id_to_label_map(teams_df: pd.DataFrame) -> Dict[int, str]:
    """
    Construye un diccionario TEAM_ID -> abreviatura/nombre de equipo.

    Ejemplo:
    1610612747 -> LAL
    """
    teams_df = clean_column_names(teams_df)
    teams_df = clean_text_columns(teams_df)

    if "team_id" not in teams_df.columns:
        return {}

    label_col = first_existing_column(
        teams_df,
        ["team", "nickname", "city"],
    )

    if label_col is None:
        return {}

    result = {}

    for _, row in teams_df.iterrows():
        team_id = pd.to_numeric(row["team_id"], errors="coerce")

        if pd.isna(team_id):
            continue

        label = row[label_col]

        if pd.isna(label):
            continue

        result[int(team_id)] = str(label).strip()

    return result


def map_team_ids_to_labels(
    team_id_series: pd.Series,
    team_id_to_label: Dict[int, str],
) -> pd.Series:
    """
    Convierte una serie de TEAM_ID en abreviaturas o nombres de equipo.
    Si no encuentra el ID, usa el ID como texto.
    """
    numeric_ids = pd.to_numeric(team_id_series, errors="coerce")

    labels = numeric_ids.map(
        lambda value: (
            team_id_to_label.get(int(value), np.nan) if pd.notna(value) else np.nan
        )
    )

    fallback = team_id_series.astype(str)

    return labels.fillna(fallback)


def ensure_team_column(
    df: pd.DataFrame,
    teams_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Garantiza que un DataFrame tenga una columna 'team'.
    """
    df = df.copy()

    if "team" in df.columns:
        return df

    team_id_to_label = build_team_id_to_label_map(teams_df)

    if "team_id" in df.columns:
        df["team"] = map_team_ids_to_labels(df["team_id"], team_id_to_label)
        return df

    df["team"] = "UNK"

    return df


# ---------------------------------------------------------------------
# Features de rol para jugadores
# ---------------------------------------------------------------------


def add_player_role_features(players_df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea features de rol para el recomendador.

    Estas columnas resumen el perfil del jugador:
    - scoring
    - playmaking
    - defense
    - rebounding
    - spacing
    - versatility
    """
    players_df = players_df.copy()

    points_scaled = minmax_scale(players_df["points"])
    assists_scaled = minmax_scale(players_df["assists"])
    rebounds_scaled = minmax_scale(players_df["rebounds"])
    steals_scaled = minmax_scale(players_df["steals"])
    blocks_scaled = minmax_scale(players_df["blocks"])
    turnovers_scaled = minmax_scale(players_df["turnovers"])
    fg_scaled = minmax_scale(players_df["fg_pct"])
    three_pct_scaled = minmax_scale(players_df["three_pct"])
    three_attempts_scaled = minmax_scale(players_df["three_attempts"])
    usage_scaled = minmax_scale(players_df["usage_rate"])
    plus_minus_scaled = minmax_scale(players_df["plus_minus"])

    # Menos pérdidas es mejor para creación de juego.
    low_turnovers_scaled = 1 - turnovers_scaled

    # En defensive_rating, normalmente menor es mejor.
    defensive_rating_scaled = 1 - minmax_scale(players_df["defensive_rating"])

    players_df["scoring"] = (
        0.45 * points_scaled
        + 0.25 * usage_scaled
        + 0.15 * fg_scaled
        + 0.15 * three_pct_scaled
    )

    players_df["playmaking"] = 0.70 * assists_scaled + 0.30 * low_turnovers_scaled

    players_df["defense"] = (
        0.30 * steals_scaled
        + 0.30 * blocks_scaled
        + 0.25 * defensive_rating_scaled
        + 0.15 * plus_minus_scaled
    )

    players_df["rebounding"] = rebounds_scaled

    players_df["spacing"] = 0.60 * three_pct_scaled + 0.40 * three_attempts_scaled

    players_df["versatility"] = players_df[
        [
            "scoring",
            "playmaking",
            "defense",
            "rebounding",
            "spacing",
        ]
    ].mean(axis=1)

    role_columns = [
        "scoring",
        "playmaking",
        "defense",
        "rebounding",
        "spacing",
        "versatility",
    ]

    for col in role_columns:
        players_df[col] = players_df[col].clip(lower=0, upper=1)

    return players_df


# ---------------------------------------------------------------------
# Estadísticas agregadas por equipo y partido
# ---------------------------------------------------------------------


def build_team_game_stats_from_details(
    games_details_df: pd.DataFrame,
    teams_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Construye estadísticas agregadas por partido y equipo usando games_details.csv.

    Se usa para estimar posesiones, pace y ratings.
    """
    details_df = clean_column_names(games_details_df)
    details_df = clean_text_columns(details_df)
    details_df = ensure_team_column(details_df, teams_df)
    details_df = parse_minutes_column(details_df)

    required_basic_columns = ["game_id", "team"]

    validate_required_columns(
        details_df,
        required_basic_columns,
        "games_details.csv",
    )

    numeric_cols = [
        "points",
        "fga",
        "fta",
        "oreb",
        "turnovers",
        "rebounds",
        "assists",
        "minutes",
    ]

    for col in numeric_cols:
        if col not in details_df.columns:
            details_df[col] = 0

    details_df = convert_numeric_columns(details_df, numeric_cols)
    details_df = fill_missing_numeric_values(details_df, numeric_cols)

    grouped_df = details_df.groupby(["game_id", "team"], as_index=False).agg(
        team_points=("points", "sum"),
        team_fga=("fga", "sum"),
        team_fta=("fta", "sum"),
        team_oreb=("oreb", "sum"),
        team_turnovers=("turnovers", "sum"),
        team_rebounds=("rebounds", "sum"),
        team_assists=("assists", "sum"),
        team_minutes=("minutes", "sum"),
    )

    grouped_df["possessions"] = (
        grouped_df["team_fga"]
        + 0.44 * grouped_df["team_fta"]
        - grouped_df["team_oreb"]
        + grouped_df["team_turnovers"]
    )

    grouped_df.loc[grouped_df["possessions"] <= 0, "possessions"] = np.nan

    return grouped_df


# ---------------------------------------------------------------------
# Preprocesamiento de matchups desde games.csv
# ---------------------------------------------------------------------


def build_processed_matchups_from_games(
    games_df: pd.DataFrame,
    teams_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Construye processed_matchups.csv desde games.csv.

    Genera dos filas por partido:
    - perspectiva del equipo local
    - perspectiva del equipo visitante
    """
    games_df = clean_column_names(games_df)
    games_df = clean_text_columns(games_df)

    team_id_to_label = build_team_id_to_label_map(teams_df)

    required_columns = [
        "game_id",
        "home_team_id",
        "visitor_team_id",
    ]

    validate_required_columns(games_df, required_columns, "games.csv")

    home_score_col = first_existing_column(
        games_df,
        ["pts_home", "home_pts", "points_home", "home_score"],
    )

    away_score_col = first_existing_column(
        games_df,
        ["pts_away", "away_pts", "points_away", "visitor_score"],
    )

    if home_score_col is None or away_score_col is None:
        raise ValueError(
            "games.csv necesita columnas de puntos. "
            "Ejemplo esperado: PTS_home y PTS_away."
        )

    games_df["home_team"] = map_team_ids_to_labels(
        games_df["home_team_id"],
        team_id_to_label,
    )

    games_df["away_team"] = map_team_ids_to_labels(
        games_df["visitor_team_id"],
        team_id_to_label,
    )

    games_df[home_score_col] = pd.to_numeric(
        games_df[home_score_col],
        errors="coerce",
    )

    games_df[away_score_col] = pd.to_numeric(
        games_df[away_score_col],
        errors="coerce",
    )

    if "home_team_wins" in games_df.columns:
        games_df["home_win"] = (
            pd.to_numeric(
                games_df["home_team_wins"],
                errors="coerce",
            )
            .fillna(0)
            .astype(int)
        )
    else:
        games_df["home_win"] = np.where(
            games_df[home_score_col] > games_df[away_score_col],
            1,
            0,
        )

    game_date_col = first_existing_column(
        games_df,
        ["game_date_est", "game_date", "date"],
    )

    home_rows = pd.DataFrame(
        {
            "game_id": games_df["game_id"],
            "team": games_df["home_team"],
            "opponent_team": games_df["away_team"],
            "team_score": games_df[home_score_col],
            "opponent_score": games_df[away_score_col],
            "home": 1,
            "win": games_df["home_win"],
        }
    )

    away_rows = pd.DataFrame(
        {
            "game_id": games_df["game_id"],
            "team": games_df["away_team"],
            "opponent_team": games_df["home_team"],
            "team_score": games_df[away_score_col],
            "opponent_score": games_df[home_score_col],
            "home": 0,
            "win": 1 - games_df["home_win"],
        }
    )

    if game_date_col is not None:
        home_rows["game_date"] = games_df[game_date_col]
        away_rows["game_date"] = games_df[game_date_col]

    matchups_df = pd.concat([home_rows, away_rows], ignore_index=True)

    matchups_df = clean_text_columns(matchups_df)
    matchups_df = convert_numeric_columns(matchups_df, MATCHUP_NUMERIC_COLUMNS)
    matchups_df = fill_missing_numeric_values(matchups_df, MATCHUP_NUMERIC_COLUMNS)

    matchups_df = matchups_df.dropna(subset=["team", "opponent_team"])

    matchups_df = remove_duplicates(
        matchups_df,
        subset=["game_id", "team", "opponent_team"],
    )

    ordered_columns = MATCHUP_REQUIRED_COLUMNS.copy()

    if "game_date" in matchups_df.columns:
        ordered_columns.append("game_date")

    matchups_df = matchups_df[ordered_columns]

    return matchups_df.reset_index(drop=True)


# ---------------------------------------------------------------------
# Preprocesamiento de equipos
# ---------------------------------------------------------------------


def preprocess_teams(
    teams_df: pd.DataFrame,
    games_df: pd.DataFrame,
    team_game_stats_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Construye processed_teams.csv.

    Si teams.csv ya viene con ratings, los usa.
    Si no, estima ratings desde games.csv y games_details.csv.
    """
    direct_teams_df = clean_column_names(teams_df)
    direct_teams_df = clean_text_columns(direct_teams_df)

    if all(col in direct_teams_df.columns for col in TEAM_REQUIRED_COLUMNS):
        direct_teams_df = convert_numeric_columns(
            direct_teams_df,
            TEAM_NUMERIC_COLUMNS,
        )

        direct_teams_df = fill_missing_numeric_values(
            direct_teams_df,
            TEAM_NUMERIC_COLUMNS,
        )

        direct_teams_df = remove_duplicates(direct_teams_df, subset=["team"])

        return direct_teams_df[TEAM_REQUIRED_COLUMNS].reset_index(drop=True)

    matchups_df = build_processed_matchups_from_games(games_df, teams_df)

    if team_game_stats_df is not None and not team_game_stats_df.empty:
        own_stats = team_game_stats_df[["game_id", "team", "possessions"]].copy()

        own_stats = own_stats.rename(columns={"possessions": "team_possessions"})

        opponent_stats = team_game_stats_df[["game_id", "team", "possessions"]].copy()

        opponent_stats = opponent_stats.rename(
            columns={
                "team": "opponent_team",
                "possessions": "opponent_possessions",
            }
        )

        matchups_df = matchups_df.merge(
            own_stats,
            on=["game_id", "team"],
            how="left",
        )

        matchups_df = matchups_df.merge(
            opponent_stats,
            on=["game_id", "opponent_team"],
            how="left",
        )
    else:
        matchups_df["team_possessions"] = np.nan
        matchups_df["opponent_possessions"] = np.nan

    matchups_df["team_score"] = pd.to_numeric(
        matchups_df["team_score"],
        errors="coerce",
    )

    matchups_df["opponent_score"] = pd.to_numeric(
        matchups_df["opponent_score"],
        errors="coerce",
    )

    matchups_df["offensive_rating"] = np.where(
        matchups_df["team_possessions"].notna() & (matchups_df["team_possessions"] > 0),
        100 * matchups_df["team_score"] / matchups_df["team_possessions"],
        matchups_df["team_score"],
    )

    matchups_df["defensive_rating"] = np.where(
        matchups_df["opponent_possessions"].notna()
        & (matchups_df["opponent_possessions"] > 0),
        100 * matchups_df["opponent_score"] / matchups_df["opponent_possessions"],
        matchups_df["opponent_score"],
    )

    matchups_df["pace"] = np.where(
        matchups_df["team_possessions"].notna() & (matchups_df["team_possessions"] > 0),
        matchups_df["team_possessions"],
        (matchups_df["team_score"] + matchups_df["opponent_score"]) / 2,
    )

    processed_teams_df = matchups_df.groupby("team", as_index=False).agg(
        offensive_rating=("offensive_rating", "mean"),
        defensive_rating=("defensive_rating", "mean"),
        pace=("pace", "mean"),
    )

    processed_teams_df = convert_numeric_columns(
        processed_teams_df,
        TEAM_NUMERIC_COLUMNS,
    )

    processed_teams_df = fill_missing_numeric_values(
        processed_teams_df,
        TEAM_NUMERIC_COLUMNS,
    )

    processed_teams_df = remove_duplicates(processed_teams_df, subset=["team"])

    return processed_teams_df[TEAM_REQUIRED_COLUMNS].reset_index(drop=True)


# ---------------------------------------------------------------------
# Preprocesamiento de jugadores
# ---------------------------------------------------------------------


def preprocess_players(
    players_df: pd.DataFrame,
    games_details_df: pd.DataFrame,
    teams_df: pd.DataFrame,
    team_game_stats_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Construye processed_players.csv.

    Si players.csv ya viene con las columnas finales, las usa.
    Si no, construye estadísticas de jugadores desde games_details.csv.
    """
    direct_players_df = clean_column_names(players_df)
    direct_players_df = clean_text_columns(direct_players_df)

    if all(col in direct_players_df.columns for col in PLAYER_REQUIRED_COLUMNS):
        direct_players_df = convert_numeric_columns(
            direct_players_df,
            PLAYER_NUMERIC_COLUMNS,
        )

        direct_players_df = fix_percentage_scale(
            direct_players_df,
            percentage_columns=[
                "fg_pct",
                "three_pct",
                "usage_rate",
            ],
        )

        direct_players_df = fill_missing_numeric_values(
            direct_players_df,
            PLAYER_NUMERIC_COLUMNS,
        )

        direct_players_df = infer_missing_player_positions(direct_players_df)

        direct_players_df = remove_duplicates(
            direct_players_df,
            subset=["player_name", "team"],
        )

        return direct_players_df[PLAYER_REQUIRED_COLUMNS].reset_index(drop=True)

    details_df = clean_column_names(games_details_df)
    details_df = clean_text_columns(details_df)
    details_df = ensure_team_column(details_df, teams_df)
    details_df = parse_minutes_column(details_df)

    required_columns = [
        "game_id",
        "player_name",
        "team",
    ]

    validate_required_columns(details_df, required_columns, "games_details.csv")

    if "position" not in details_df.columns:
        details_df["position"] = "UNK"

    details_df["position"] = details_df["position"].apply(normalize_position_label)

    position_lookup = build_player_position_lookup(direct_players_df)

    details_df = fill_positions_from_lookup(
        df=details_df,
        position_lookup=position_lookup,
    )

    if "player_id" not in details_df.columns:
        details_df["player_id"] = (
            pd.factorize(
                details_df["player_name"].astype(str)
                + "_"
                + details_df["team"].astype(str)
            )[0]
            + 1
        )

    if "team_id" not in details_df.columns:
        details_df["team_id"] = pd.factorize(details_df["team"].astype(str))[0] + 1

    numeric_columns_needed = [
        "player_id",
        "team_id",
        "minutes",
        "points",
        "assists",
        "rebounds",
        "steals",
        "blocks",
        "turnovers",
        "fg_pct",
        "three_pct",
        "three_attempts",
        "fga",
        "fta",
        "oreb",
        "fgm",
        "fg3m",
        "plus_minus",
    ]

    for col in numeric_columns_needed:
        if col not in details_df.columns:
            details_df[col] = 0

    details_df = convert_numeric_columns(details_df, numeric_columns_needed)

    details_df["fg_pct"] = np.where(
        (details_df["fg_pct"].isna() | (details_df["fg_pct"] <= 0))
        & (details_df["fga"] > 0),
        details_df["fgm"] / details_df["fga"],
        details_df["fg_pct"],
    )

    details_df["three_pct"] = np.where(
        (details_df["three_pct"].isna() | (details_df["three_pct"] <= 0))
        & (details_df["three_attempts"] > 0),
        details_df["fg3m"] / details_df["three_attempts"],
        details_df["three_pct"],
    )

    details_df = fix_percentage_scale(
        details_df,
        percentage_columns=[
            "fg_pct",
            "three_pct",
        ],
    )

    details_df = fill_missing_numeric_values(details_df, numeric_columns_needed)

    # Quitamos DNPs o filas sin minutos reales.
    details_df = details_df[details_df["minutes"] > 0].copy()

    if details_df.empty:
        raise ValueError(
            "No se encontraron jugadores con minutos mayores a 0 en games_details.csv."
        )

    if team_game_stats_df is not None and not team_game_stats_df.empty:
        pace_df = team_game_stats_df[["game_id", "team", "possessions"]].copy()

        pace_df = pace_df.rename(columns={"possessions": "team_possessions"})

        details_df = details_df.merge(
            pace_df,
            on=["game_id", "team"],
            how="left",
        )
    else:
        details_df["team_possessions"] = np.nan

    median_possessions = details_df["team_possessions"].median(skipna=True)

    if pd.isna(median_possessions):
        median_possessions = 100

    details_df["team_possessions"] = details_df["team_possessions"].fillna(
        median_possessions
    )

    # Proxy simple de usage rate.
    # Intuición: proporción de posesiones que el jugador termina con tiro,
    # tiro libre o pérdida.
    usage_numerator = (
        details_df["fga"] + 0.44 * details_df["fta"] + details_df["turnovers"]
    )

    details_df["usage_rate"] = np.where(
        details_df["team_possessions"] > 0,
        usage_numerator / details_df["team_possessions"],
        0,
    )

    details_df["usage_rate"] = details_df["usage_rate"].clip(lower=0, upper=1)

    # Proxies simples de ratings por jugador.
    # No son ratings oficiales, pero sirven como variables consistentes
    # para el recomendador inicial.
    details_df["offensive_rating"] = (
        100
        + 1.2 * details_df["points"]
        + 0.8 * details_df["assists"]
        - 1.5 * details_df["turnovers"]
        + 8.0 * details_df["fg_pct"]
        + 6.0 * details_df["three_pct"]
    )

    details_df["defensive_rating"] = (
        115
        - 1.5 * details_df["steals"]
        - 1.3 * details_df["blocks"]
        - 0.4 * details_df["rebounds"]
        - 0.15 * details_df["plus_minus"]
    )

    details_df["pace"] = details_df["team_possessions"]

    aggregation_dict = {
        "player_id": lambda values: get_mode_numeric(values, default_value=0),
        "team_id": lambda values: get_mode_numeric(values, default_value=0),
        "game_id": "nunique",
        "position": lambda values: get_mode_text(values, default_value="UNK"),
        "minutes": "mean",
        "points": "mean",
        "assists": "mean",
        "rebounds": "mean",
        "steals": "mean",
        "blocks": "mean",
        "turnovers": "mean",
        "fg_pct": "mean",
        "three_pct": "mean",
        "three_attempts": "mean",
        "usage_rate": "mean",
        "offensive_rating": "mean",
        "defensive_rating": "mean",
        "pace": "mean",
        "plus_minus": "mean",
    }

    processed_players_df = details_df.groupby(
        ["player_name", "team"], as_index=False
    ).agg(aggregation_dict)

    processed_players_df = processed_players_df.rename(
        columns={"game_id": "games_played"}
    )

    processed_players_df = infer_missing_player_positions(processed_players_df)

    processed_players_df = add_player_role_features(processed_players_df)

    processed_players_df = convert_numeric_columns(
        processed_players_df,
        PLAYER_NUMERIC_COLUMNS,
    )

    processed_players_df = fix_percentage_scale(
        processed_players_df,
        percentage_columns=[
            "fg_pct",
            "three_pct",
            "usage_rate",
        ],
    )

    processed_players_df = fill_missing_numeric_values(
        processed_players_df,
        PLAYER_NUMERIC_COLUMNS,
    )

    processed_players_df = remove_duplicates(
        processed_players_df,
        subset=["player_name", "team"],
    )

    return processed_players_df[PLAYER_REQUIRED_COLUMNS].reset_index(drop=True)


# ---------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------


def run_preprocessing() -> None:
    """
    Ejecuta el pipeline completo de preprocesamiento.
    """
    print("Iniciando preprocesamiento de datos...")

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    required_files = [
        RAW_PLAYERS_PATH,
        RAW_TEAMS_PATH,
        RAW_GAMES_PATH,
        RAW_GAME_DETAILS_PATH,
    ]

    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(f"No se encontró el archivo: {path}")

    players_df = pd.read_csv(RAW_PLAYERS_PATH, low_memory=False)
    teams_df = pd.read_csv(RAW_TEAMS_PATH, low_memory=False)
    games_df = pd.read_csv(RAW_GAMES_PATH, low_memory=False)
    games_details_df = pd.read_csv(RAW_GAME_DETAILS_PATH, low_memory=False)

    team_game_stats_df = build_team_game_stats_from_details(
        games_details_df=games_details_df,
        teams_df=teams_df,
    )

    processed_players_df = preprocess_players(
        players_df=players_df,
        games_details_df=games_details_df,
        teams_df=teams_df,
        team_game_stats_df=team_game_stats_df,
    )

    processed_teams_df = preprocess_teams(
        teams_df=teams_df,
        games_df=games_df,
        team_game_stats_df=team_game_stats_df,
    )

    processed_matchups_df = build_processed_matchups_from_games(
        games_df=games_df,
        teams_df=teams_df,
    )

    latest_roster_df, player_profiles_df = build_and_save_roster_outputs(
        games_details_df=games_details_df,
        games_df=games_df,
        teams_df=teams_df,
        latest_roster_path=LATEST_ROSTER_PATH,
        player_profiles_path=PLAYER_PROFILES_PATH,
    )

    processed_players_df.to_csv(PROCESSED_PLAYERS_PATH, index=False)
    processed_teams_df.to_csv(PROCESSED_TEAMS_PATH, index=False)
    processed_matchups_df.to_csv(PROCESSED_MATCHUPS_PATH, index=False)

    print("Preprocesamiento completado.")
    print(f"Jugadores procesados: {len(processed_players_df)}")
    print(f"Equipos procesados: {len(processed_teams_df)}")
    print(f"Matchups procesados: {len(processed_matchups_df)}")
    print(f"Jugadores en roster más reciente disponible: {len(latest_roster_df)}")
    print(f"Perfiles históricos de jugador: {len(player_profiles_df)}")

    print("\nArchivos generados:")
    print(f"- {PROCESSED_PLAYERS_PATH}")
    print(f"- {PROCESSED_TEAMS_PATH}")
    print(f"- {PROCESSED_MATCHUPS_PATH}")
    print(f"- {LATEST_ROSTER_PATH}")
    print(f"- {PLAYER_PROFILES_PATH}")


# ---------------------------------------------------------------------
# Ejecución directa
# ---------------------------------------------------------------------

if __name__ == "__main__":
    run_preprocessing()
