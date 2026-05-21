"""
roster.py

Construye dos tablas nuevas para el proyecto NBA Next-Man-Up:

1. latest_roster.csv
   Roster más reciente disponible en la base de datos.
   No representa necesariamente el roster actual real de la NBA.

2. player_profiles.csv
   Perfil histórico agregado por jugador usando todas sus filas disponibles,
   sin importar equipo ni temporada.

Regla conceptual:
- latest_team sirve para decidir en qué roster aparece el jugador.
- player profile sirve para calcular similitud, impacto y recomendación.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

try:
    from .role_features import prepare_players_data
except ImportError:
    from role_features import prepare_players_data


# ---------------------------------------------------------------------
# Columnas y constantes
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

PLAYER_PROFILE_BASE_COLUMNS = [
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
    "rim_pressure",
    "versatility",
]

LATEST_ROSTER_COLUMNS = [
    "player_id",
    "player_name",
    "latest_team_id",
    "latest_team",
    "latest_season",
    "latest_game_date",
    "latest_game_id",
    "latest_position",
    "latest_has_minutes",
]


# ---------------------------------------------------------------------
# Helpers generales
# ---------------------------------------------------------------------


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza nombres de columnas a snake_case y aplica alias usados por el dataset NBA.
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
        "game_date_est": "game_date",
        "player": "player_name",
        "player_full_name": "player_name",
        "team_abbreviation": "team",
        "abbreviation": "team",
        "team_name": "team",
        "start_position": "position",
        "min": "minutes",
        "pts": "points",
        "ast": "assists",
        "reb": "rebounds",
        "stl": "steals",
        "blk": "blocks",
        "to": "turnovers",
        "tov": "turnovers",
        "fg3a": "three_attempts",
        "fg3m": "three_made",
        "fg3_pct": "three_pct",
        "3p_pct": "three_pct",
        "fg_pct_pct": "fg_pct",
        "three_pct_pct": "three_pct",
    }

    return df.rename(columns=rename_map)



def clean_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Limpia espacios en columnas de texto sin convertir NaN en texto."""
    df = df.copy()

    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].apply(
            lambda value: value.strip() if isinstance(value, str) else value
        )

    return df



def parse_minutes_value(value) -> float:
    """
    Convierte minutos tipo '12:34' a minutos decimales.
    Valores vacíos, DNP o inválidos se vuelven 0.0.
    """
    if pd.isna(value):
        return 0.0

    if isinstance(value, (int, float)):
        if np.isfinite(value):
            return float(value)
        return 0.0

    value = str(value).strip()

    if value == "":
        return 0.0

    if ":" in value:
        try:
            minutes, seconds = value.split(":")[:2]
            return float(minutes) + float(seconds) / 60.0
        except ValueError:
            return 0.0

    try:
        number = float(value)
    except ValueError:
        return 0.0

    return number if np.isfinite(number) else 0.0



def normalize_position_label(value) -> str:
    """
    Normaliza posiciones a grupos simples: G, F, C o UNK.
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



def get_mode_text(series: pd.Series, default_value: str = "UNK") -> str:
    """Devuelve el texto más frecuente ignorando valores vacíos."""
    cleaned = series.dropna().astype(str).str.strip()
    cleaned = cleaned[~cleaned.str.upper().isin(UNKNOWN_POSITION_VALUES)]

    if cleaned.empty:
        return default_value

    return str(cleaned.mode().iloc[0])



def get_latest_text(series: pd.Series, default_value: str = "") -> str:
    """Devuelve el último texto no vacío de una serie ordenada temporalmente."""
    cleaned = series.dropna().astype(str).str.strip()
    cleaned = cleaned[cleaned != ""]

    if cleaned.empty:
        return default_value

    return str(cleaned.iloc[-1])



def safe_divide(numerator, denominator, default: float = 0.0) -> float:
    """Divide evitando NaN, infinitos y división entre cero."""
    try:
        numerator = float(numerator)
        denominator = float(denominator)
    except (TypeError, ValueError):
        return default

    if denominator == 0 or not np.isfinite(numerator) or not np.isfinite(denominator):
        return default

    result = numerator / denominator

    return float(result) if np.isfinite(result) else default



def first_existing_column(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    """Devuelve la primera columna existente dentro de una lista."""
    for col in candidates:
        if col in df.columns:
            return col

    return None



def build_team_id_to_abbreviation_map(teams_df: Optional[pd.DataFrame]) -> dict[int, str]:
    """Construye un diccionario team_id -> abreviatura/nombre si teams_df está disponible."""
    if teams_df is None or teams_df.empty:
        return {}

    teams = clean_column_names(teams_df)
    teams = clean_text_columns(teams)

    if "team_id" not in teams.columns:
        return {}

    label_col = first_existing_column(teams, ["team", "nickname", "city"])

    if label_col is None:
        return {}

    mapping: dict[int, str] = {}

    for _, row in teams.iterrows():
        team_id = pd.to_numeric(row.get("team_id"), errors="coerce")

        if pd.isna(team_id):
            continue

        label = row.get(label_col)

        if pd.isna(label):
            continue

        mapping[int(team_id)] = str(label).strip()

    return mapping


# ---------------------------------------------------------------------
# Preparación base desde games_details + games
# ---------------------------------------------------------------------


def prepare_player_game_rows(
    games_details_df: pd.DataFrame,
    games_df: pd.DataFrame,
    teams_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Une games_details.csv con games.csv por game_id y deja una fila por jugador-partido.

    Esta función es la base tanto para latest_roster como para player_profiles.
    """
    details = clean_column_names(games_details_df)
    games = clean_column_names(games_df)
    details = clean_text_columns(details)
    games = clean_text_columns(games)

    required_detail_columns = ["game_id", "player_id", "player_name", "team_id"]
    required_game_columns = ["game_id", "game_date"]

    missing_detail = [col for col in required_detail_columns if col not in details.columns]
    missing_game = [col for col in required_game_columns if col not in games.columns]

    if missing_detail:
        raise ValueError(f"games_details.csv no tiene columnas requeridas: {missing_detail}")

    if missing_game:
        raise ValueError(f"games.csv no tiene columnas requeridas: {missing_game}")

    if "team" not in details.columns:
        team_map = build_team_id_to_abbreviation_map(teams_df)
        details["team"] = (
            pd.to_numeric(details["team_id"], errors="coerce")
            .map(lambda value: team_map.get(int(value), "") if pd.notna(value) else "")
        )

    if "position" not in details.columns:
        details["position"] = "UNK"

    if "minutes" not in details.columns:
        details["minutes"] = 0.0

    details["minutes"] = details["minutes"].apply(parse_minutes_value)
    details["has_minutes"] = details["minutes"] > 0

    details["position"] = details["position"].apply(normalize_position_label)

    games_columns = ["game_id", "game_date"]

    if "season" in games.columns:
        games_columns.append("season")

    games_lookup = games[games_columns].drop_duplicates(subset=["game_id"])

    merged = details.merge(games_lookup, on="game_id", how="left")

    merged["game_date"] = pd.to_datetime(merged["game_date"], errors="coerce")

    merged["game_id_numeric"] = pd.to_numeric(merged["game_id"], errors="coerce")
    merged["player_id"] = pd.to_numeric(merged["player_id"], errors="coerce")
    merged["team_id"] = pd.to_numeric(merged["team_id"], errors="coerce")

    if "season" not in merged.columns:
        merged["season"] = merged["game_date"].dt.year

    merged["season"] = pd.to_numeric(merged["season"], errors="coerce")

    merged = merged.dropna(subset=["player_id", "game_id", "team_id"])
    merged = merged[merged["team"].notna()].copy()
    merged["team"] = merged["team"].astype(str).str.strip()
    merged = merged[merged["team"] != ""].copy()

    merged["player_id"] = merged["player_id"].astype(int)
    merged["team_id"] = merged["team_id"].astype(int)

    return merged.reset_index(drop=True)


# ---------------------------------------------------------------------
# latest_roster.csv
# ---------------------------------------------------------------------


def build_latest_roster(
    games_details_df: pd.DataFrame,
    games_df: pd.DataFrame,
    teams_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Construye el roster más reciente disponible en la base.

    Reglas:
    - Fuente: games_details.csv unido con games.csv por game_id.
    - Criterio principal: game_date.
    - Empate: game_id y prioridad a filas con minutos reales.
    - Salida: una fila por player_id.
    """
    rows = prepare_player_game_rows(
        games_details_df=games_details_df,
        games_df=games_df,
        teams_df=teams_df,
    )

    if rows.empty:
        raise ValueError("No hay filas válidas para construir latest_roster.csv.")

    position_lookup = (
        rows[rows["position"] != "UNK"]
        .groupby("player_id")["position"]
        .agg(lambda values: get_mode_text(values, default_value="UNK"))
        .to_dict()
    )

    sort_columns = ["player_id", "game_date", "game_id_numeric", "has_minutes"]
    rows = rows.sort_values(
        sort_columns,
        ascending=[True, True, True, True],
        na_position="first",
    )

    latest = rows.drop_duplicates(subset=["player_id"], keep="last").copy()

    latest["latest_position"] = latest.apply(
        lambda row: row["position"]
        if normalize_position_label(row.get("position", "UNK")) != "UNK"
        else position_lookup.get(int(row["player_id"]), "UNK"),
        axis=1,
    )

    latest["latest_game_date"] = latest["game_date"].dt.strftime("%Y-%m-%d")
    latest["latest_season"] = pd.to_numeric(latest["season"], errors="coerce")
    latest["latest_game_id"] = pd.to_numeric(latest["game_id"], errors="coerce")

    latest_roster = pd.DataFrame(
        {
            "player_id": latest["player_id"].astype(int),
            "player_name": latest["player_name"].astype(str).str.strip(),
            "latest_team_id": latest["team_id"].astype(int),
            "latest_team": latest["team"].astype(str).str.strip(),
            "latest_season": latest["latest_season"].astype("Int64"),
            "latest_game_date": latest["latest_game_date"],
            "latest_game_id": latest["latest_game_id"].astype("Int64"),
            "latest_position": latest["latest_position"].apply(normalize_position_label),
            "latest_has_minutes": latest["has_minutes"].astype(bool),
        }
    )

    latest_roster = latest_roster.dropna(subset=["latest_team", "latest_game_date"])
    latest_roster = latest_roster[latest_roster["latest_team"].astype(str).str.strip() != ""]
    latest_roster = latest_roster.sort_values("player_name").reset_index(drop=True)

    return latest_roster[LATEST_ROSTER_COLUMNS]


# ---------------------------------------------------------------------
# player_profiles.csv
# ---------------------------------------------------------------------


def _ensure_numeric_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Crea columnas faltantes y convierte a numérico."""
    df = df.copy()

    for col in columns:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    return df



def _infer_position_from_profile(row: pd.Series) -> str:
    """Infiere posición si no hay una posición confiable en START_POSITION."""
    current = normalize_position_label(row.get("position", "UNK"))

    if current != "UNK":
        return current

    minutes = float(row.get("minutes", 0.0) or 0.0)

    if minutes <= 0:
        return "F"

    factor = 36.0 / minutes

    assists_per36 = float(row.get("assists", 0.0) or 0.0) * factor
    rebounds_per36 = float(row.get("rebounds", 0.0) or 0.0) * factor
    blocks_per36 = float(row.get("blocks", 0.0) or 0.0) * factor
    three_attempts_per36 = float(row.get("three_attempts", 0.0) or 0.0) * factor

    if rebounds_per36 >= 9.0 or blocks_per36 >= 1.4:
        return "C"

    if assists_per36 >= 5.0 and rebounds_per36 <= 6.5 and blocks_per36 <= 0.8:
        return "G"

    if (
        assists_per36 >= 3.5
        and three_attempts_per36 >= 4.0
        and rebounds_per36 <= 5.5
        and blocks_per36 <= 0.7
    ):
        return "G"

    return "F"



def _build_team_possessions(player_game_rows: pd.DataFrame) -> pd.DataFrame:
    """Calcula posesiones aproximadas por equipo-partido desde box score."""
    needed = ["fga", "fta", "oreb", "turnovers"]
    rows = _ensure_numeric_columns(player_game_rows, needed)

    team_game = rows.groupby(["game_id", "team_id"], as_index=False).agg(
        team_fga=("fga", "sum"),
        team_fta=("fta", "sum"),
        team_oreb=("oreb", "sum"),
        team_turnovers=("turnovers", "sum"),
    )

    team_game["team_possessions"] = (
        team_game["team_fga"]
        + 0.44 * team_game["team_fta"]
        - team_game["team_oreb"]
        + team_game["team_turnovers"]
    )

    team_game.loc[team_game["team_possessions"] <= 0, "team_possessions"] = np.nan

    median_possessions = team_game["team_possessions"].median(skipna=True)

    if pd.isna(median_possessions):
        median_possessions = 100.0

    team_game["team_possessions"] = team_game["team_possessions"].fillna(
        median_possessions
    )

    return team_game[["game_id", "team_id", "team_possessions"]]



def build_player_profiles(
    games_details_df: pd.DataFrame,
    games_df: pd.DataFrame,
    teams_df: Optional[pd.DataFrame] = None,
    latest_roster_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Construye player_profiles.csv con una fila por player_id usando todo el historial.

    Las métricas principales son consistentes con el recomendador actual:
    - games_played = partidos con minutos reales.
    - minutes, points, assists, etc. = promedio por partido jugado.
    - fg_pct = FGM total / FGA total.
    - three_pct = FG3M total / FG3A total.
    - usage_rate = uso agregado sobre posesiones del equipo en partidos jugados.
    """
    rows = prepare_player_game_rows(
        games_details_df=games_details_df,
        games_df=games_df,
        teams_df=teams_df,
    )

    if rows.empty:
        raise ValueError("No hay filas válidas para construir player_profiles.csv.")

    numeric_columns = [
        "minutes",
        "points",
        "assists",
        "rebounds",
        "steals",
        "blocks",
        "turnovers",
        "fgm",
        "fga",
        "three_made",
        "three_attempts",
        "ftm",
        "fta",
        "oreb",
        "dreb",
        "plus_minus",
    ]

    rows = _ensure_numeric_columns(rows, numeric_columns)

    team_possessions = _build_team_possessions(rows)
    rows = rows.merge(team_possessions, on=["game_id", "team_id"], how="left")

    median_possessions = rows["team_possessions"].median(skipna=True)
    if pd.isna(median_possessions):
        median_possessions = 100.0

    rows["team_possessions"] = rows["team_possessions"].fillna(median_possessions)

    # Solo las apariciones con minutos reales alimentan el perfil estadístico.
    played_rows = rows[rows["minutes"] > 0].copy()

    if played_rows.empty:
        raise ValueError(
            "No se encontraron filas con minutos reales para construir player_profiles.csv."
        )

    played_rows["fg_pct_row"] = np.where(
        played_rows["fga"] > 0,
        played_rows["fgm"] / played_rows["fga"],
        0.0,
    )

    played_rows["three_pct_row"] = np.where(
        played_rows["three_attempts"] > 0,
        played_rows["three_made"] / played_rows["three_attempts"],
        0.0,
    )

    usage_numerator = (
        played_rows["fga"] + 0.44 * played_rows["fta"] + played_rows["turnovers"]
    )

    played_rows["usage_numerator"] = usage_numerator

    played_rows["offensive_rating_row"] = (
        100
        + 1.2 * played_rows["points"]
        + 0.8 * played_rows["assists"]
        - 1.5 * played_rows["turnovers"]
        + 8.0 * played_rows["fg_pct_row"]
        + 6.0 * played_rows["three_pct_row"]
    )

    played_rows["defensive_rating_row"] = (
        115
        - 1.5 * played_rows["steals"]
        - 1.3 * played_rows["blocks"]
        - 0.4 * played_rows["rebounds"]
        - 0.15 * played_rows["plus_minus"]
    )

    played_rows = played_rows.sort_values(
        by=["player_id", "game_date", "game_id_numeric"],
        ascending=[True, True, True],
        na_position="first",
    )

    position_lookup = (
        played_rows[played_rows["position"] != "UNK"]
        .groupby("player_id")["position"]
        .agg(lambda values: get_mode_text(values, default_value="UNK"))
        .to_dict()
    )

    grouped = played_rows.groupby("player_id", as_index=False).agg(
        player_name=("player_name", lambda values: get_latest_text(values, "")),
        games_played=("game_id", "nunique"),
        minutes=("minutes", "mean"),
        points=("points", "mean"),
        assists=("assists", "mean"),
        rebounds=("rebounds", "mean"),
        steals=("steals", "mean"),
        blocks=("blocks", "mean"),
        turnovers=("turnovers", "mean"),
        fgm_total=("fgm", "sum"),
        fga_total=("fga", "sum"),
        three_made_total=("three_made", "sum"),
        three_attempts_total=("three_attempts", "sum"),
        three_attempts=("three_attempts", "mean"),
        usage_numerator_total=("usage_numerator", "sum"),
        team_possessions_total=("team_possessions", "sum"),
        offensive_rating=("offensive_rating_row", "mean"),
        defensive_rating=("defensive_rating_row", "mean"),
        pace=("team_possessions", "mean"),
        plus_minus=("plus_minus", "mean"),
    )

    grouped["position"] = grouped["player_id"].map(position_lookup).fillna("UNK")

    grouped["fg_pct"] = grouped.apply(
        lambda row: safe_divide(row["fgm_total"], row["fga_total"], default=0.0),
        axis=1,
    )

    grouped["three_pct"] = grouped.apply(
        lambda row: safe_divide(
            row["three_made_total"], row["three_attempts_total"], default=0.0
        ),
        axis=1,
    )

    grouped["usage_rate"] = grouped.apply(
        lambda row: safe_divide(
            row["usage_numerator_total"],
            row["team_possessions_total"],
            default=0.0,
        ),
        axis=1,
    )

    grouped["usage_rate"] = grouped["usage_rate"].clip(lower=0, upper=1)

    if latest_roster_df is None:
        latest_roster_df = build_latest_roster(
            games_details_df=games_details_df,
            games_df=games_df,
            teams_df=teams_df,
        )

    latest = latest_roster_df.copy()

    if not latest.empty:
        latest["player_id"] = pd.to_numeric(latest["player_id"], errors="coerce")
        latest = latest.dropna(subset=["player_id"]).copy()
        latest["player_id"] = latest["player_id"].astype(int)

        grouped = grouped.merge(
            latest[
                [
                    "player_id",
                    "latest_team_id",
                    "latest_team",
                    "latest_season",
                    "latest_game_date",
                    "latest_game_id",
                    "latest_position",
                ]
            ],
            on="player_id",
            how="left",
        )
    else:
        grouped["latest_team_id"] = np.nan
        grouped["latest_team"] = ""
        grouped["latest_season"] = np.nan
        grouped["latest_game_date"] = ""
        grouped["latest_game_id"] = np.nan
        grouped["latest_position"] = "UNK"

    # Compatibilidad con el recomendador actual:
    # team/team_id representan el equipo más reciente disponible en la base.
    grouped["team_id"] = pd.to_numeric(
        grouped["latest_team_id"], errors="coerce"
    ).fillna(0).astype(int)
    grouped["team"] = grouped["latest_team"].fillna("").astype(str)

    grouped["position"] = grouped.apply(_infer_position_from_profile, axis=1)

    grouped = prepare_players_data(grouped, overwrite_roles=True)

    for col in PLAYER_PROFILE_BASE_COLUMNS:
        if col not in grouped.columns:
            grouped[col] = 0.0 if col not in ["player_name", "team", "position"] else ""

    metadata_columns = [
        "latest_team_id",
        "latest_team",
        "latest_season",
        "latest_game_date",
        "latest_game_id",
        "latest_position",
    ]

    ordered_columns = PLAYER_PROFILE_BASE_COLUMNS + metadata_columns

    grouped = grouped[ordered_columns].copy()

    numeric_columns_to_clean = [
        col
        for col in grouped.columns
        if col not in ["player_name", "team", "position", "latest_team", "latest_game_date", "latest_position"]
    ]

    for col in numeric_columns_to_clean:
        grouped[col] = pd.to_numeric(grouped[col], errors="coerce")

    grouped["player_id"] = grouped["player_id"].astype(int)
    grouped["games_played"] = grouped["games_played"].astype(int)

    grouped = grouped.sort_values("player_name").reset_index(drop=True)

    return grouped


# ---------------------------------------------------------------------
# Función conveniente para guardar ambos CSV
# ---------------------------------------------------------------------


def build_and_save_roster_outputs(
    games_details_df: pd.DataFrame,
    games_df: pd.DataFrame,
    teams_df: Optional[pd.DataFrame],
    latest_roster_path,
    player_profiles_path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Construye y guarda latest_roster.csv y player_profiles.csv.
    """
    latest_roster_df = build_latest_roster(
        games_details_df=games_details_df,
        games_df=games_df,
        teams_df=teams_df,
    )

    player_profiles_df = build_player_profiles(
        games_details_df=games_details_df,
        games_df=games_df,
        teams_df=teams_df,
        latest_roster_df=latest_roster_df,
    )

    latest_roster_df.to_csv(latest_roster_path, index=False)
    player_profiles_df.to_csv(player_profiles_path, index=False)

    return latest_roster_df, player_profiles_df
