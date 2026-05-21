# src/recommender.py

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

try:
    from src import similarity
    from src import impact
    from src import monte_carlo
    from src import explanations
    from src.role_features import prepare_players_data
except ImportError:
    import similarity
    import impact
    import monte_carlo
    import explanations
    from role_features import prepare_players_data


# ============================================================
# 1. CONTRATO DE DATOS PROCESADOS PARA RECOMENDACIÓN
# ============================================================

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

PLAYER_PROFILE_REQUIRED_COLUMNS = PLAYER_REQUIRED_COLUMNS.copy()

APP_ROSTER_REQUIRED_COLUMNS = [
    "player_id",
    "player_name",
    "latest_team_id",
    "latest_team",
    "latest_season",
    "latest_game_date",
    "position",
    "games_played",
    "minutes",
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
    "rim_pressure",
    "versatility",
]

APP_ROSTER_NUMERIC_COLUMNS = [
    "player_id",
    "latest_team_id",
    "latest_season",
    "latest_game_id",
    "games_played",
    "minutes",
    "points",
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


# ============================================================
# 2. ALIASES SIMPLES DE EQUIPOS NBA
# ============================================================

NBA_TEAM_ALIASES = {
    "ATL": ["atl", "atlanta", "hawks", "atlanta hawks"],
    "BOS": ["bos", "boston", "celtics", "boston celtics"],
    "BKN": ["bkn", "brooklyn", "nets", "brooklyn nets"],
    "CHA": ["cha", "charlotte", "hornets", "charlotte hornets"],
    "CHI": ["chi", "chicago", "bulls", "chicago bulls"],
    "CLE": ["cle", "cleveland", "cavaliers", "cleveland cavaliers"],
    "DAL": ["dal", "dallas", "mavericks", "dallas mavericks"],
    "DEN": ["den", "denver", "nuggets", "denver nuggets"],
    "DET": ["det", "detroit", "pistons", "detroit pistons"],
    "GSW": ["gsw", "golden state", "warriors", "golden state warriors"],
    "HOU": ["hou", "houston", "rockets", "houston rockets"],
    "IND": ["ind", "indiana", "pacers", "indiana pacers"],
    "LAC": ["lac", "la clippers", "clippers", "los angeles clippers"],
    "LAL": ["lal", "la lakers", "lakers", "los angeles lakers"],
    "MEM": ["mem", "memphis", "grizzlies", "memphis grizzlies"],
    "MIA": ["mia", "miami", "heat", "miami heat"],
    "MIL": ["mil", "milwaukee", "bucks", "milwaukee bucks"],
    "MIN": ["min", "minnesota", "timberwolves", "minnesota timberwolves"],
    "NOP": ["nop", "new orleans", "pelicans", "new orleans pelicans"],
    "NYK": ["nyk", "new york", "knicks", "new york knicks"],
    "OKC": ["okc", "oklahoma city", "thunder", "oklahoma city thunder"],
    "ORL": ["orl", "orlando", "magic", "orlando magic"],
    "PHI": ["phi", "philadelphia", "76ers", "sixers", "philadelphia 76ers"],
    "PHX": ["phx", "phoenix", "suns", "phoenix suns"],
    "POR": ["por", "portland", "trail blazers", "portland trail blazers"],
    "SAC": ["sac", "sacramento", "kings", "sacramento kings"],
    "SAS": ["sas", "san antonio", "spurs", "san antonio spurs"],
    "TOR": ["tor", "toronto", "raptors", "toronto raptors"],
    "UTA": ["uta", "utah", "jazz", "utah jazz"],
    "WAS": ["was", "washington", "wizards", "washington wizards"],
}


# ============================================================
# 3. HELPERS GENERALES
# ============================================================


def _normalize_text(value: Any) -> str:
    """Normaliza texto para comparar equipos y jugadores."""
    if pd.isna(value):
        return ""

    return str(value).strip().lower()


def _team_aliases(value: Any) -> set[str]:
    """Devuelve textos equivalentes para buscar un equipo."""
    value_norm = _normalize_text(value)
    aliases = {value_norm}

    for abbreviation, known_aliases in NBA_TEAM_ALIASES.items():
        normalized_aliases = {_normalize_text(alias) for alias in known_aliases}
        normalized_aliases.add(_normalize_text(abbreviation))

        if value_norm in normalized_aliases:
            aliases.update(normalized_aliases)
            aliases.add(_normalize_text(abbreviation))

    return aliases


def _to_dict(row: pd.Series | dict) -> dict:
    """Convierte una fila de pandas a diccionario plano."""
    if isinstance(row, pd.Series):
        return row.to_dict()

    return dict(row)


def _to_list(value: Any) -> list:
    """Convierte arrays o series a listas serializables."""
    if value is None:
        return []

    if isinstance(value, list):
        return value

    if isinstance(value, (np.ndarray, pd.Series)):
        return value.tolist()

    try:
        return list(value)
    except TypeError:
        return [value]


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Convierte valores a float evitando NaN e infinitos."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default

    if not np.isfinite(number):
        return default

    return number


def _safe_int(value: Any, default: int = 0) -> int:
    """Convierte valores a int de forma segura."""
    return int(round(_safe_float(value, default=default)))


def _safe_id_set(values: list[int] | tuple[int] | pd.Series | np.ndarray | None) -> set[int]:
    """Convierte una colección de ids a set[int]."""
    if values is None:
        return set()

    result: set[int] = set()

    for value in values:
        numeric = pd.to_numeric(value, errors="coerce")
        if pd.notna(numeric):
            result.add(int(numeric))

    return result


def _clean_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Limpia espacios en columnas de texto sin convertir NaN en 'nan'."""
    df = df.copy()

    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].apply(
            lambda value: value.strip() if isinstance(value, str) else value
        )

    return df


def _validate_columns(
    df: pd.DataFrame,
    required_columns: list[str],
    table_name: str,
) -> None:
    """Verifica que una tabla tenga las columnas mínimas esperadas."""
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(
            f"{table_name} no tiene las columnas necesarias: {missing_columns}"
        )


def _coerce_numeric_columns(
    df: pd.DataFrame,
    numeric_columns: list[str],
) -> pd.DataFrame:
    """Convierte columnas numéricas existentes y rellena NaN de forma simple."""
    df = df.copy()

    for col in numeric_columns:
        if col not in df.columns:
            continue

        df[col] = pd.to_numeric(df[col], errors="coerce")
        median_value = df[col].median(skipna=True)

        if pd.isna(median_value):
            df[col] = df[col].fillna(0)
        else:
            df[col] = df[col].fillna(median_value)

    return df


def _same_team_mask(
    df: pd.DataFrame,
    team_id: int | None,
    team_name: str | None,
    id_column: str = "team_id",
    name_column: str = "team",
) -> pd.Series:
    """Máscara flexible para filtrar por equipo usando id y/o texto."""
    mask = pd.Series(False, index=df.index)

    if team_id is not None and id_column in df.columns:
        mask = mask | (pd.to_numeric(df[id_column], errors="coerce") == team_id)

    if team_name is not None and name_column in df.columns:
        aliases = _team_aliases(team_name)
        team_norm = df[name_column].astype(str).map(_normalize_text)
        mask = mask | team_norm.isin(aliases)

    return mask


# ============================================================
# 4. CARGA Y PREPARACIÓN RUNTIME
# ============================================================


def _ensure_team_columns_from_latest(players: pd.DataFrame) -> pd.DataFrame:
    """
    Asegura columnas team/team_id en perfiles.

    En Fase 3, player_profiles.csv es perfil histórico, pero trae latest_team y
    latest_team_id para compatibilidad. Usamos esos valores como team/team_id
    solo para contexto y salida; la pertenencia real al equipo se decide con
    app_roster.csv.
    """
    players = players.copy()

    if "team" not in players.columns and "latest_team" in players.columns:
        players["team"] = players["latest_team"]

    if "team_id" not in players.columns and "latest_team_id" in players.columns:
        players["team_id"] = players["latest_team_id"]

    if "team" in players.columns:
        players["team"] = players["team"].fillna("").astype(str).str.strip()

    if "team_id" in players.columns:
        players["team_id"] = pd.to_numeric(players["team_id"], errors="coerce").fillna(0)
        players["team_id"] = players["team_id"].astype(int)

    return players


def prepare_runtime_data(
    player_profiles: pd.DataFrame,
    teams: pd.DataFrame,
    matchups: pd.DataFrame,
    app_roster: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    """
    Prepara perfiles, equipos, matchups y app_roster para el recomendador.

    No recalcula el preprocesamiento completo. Solo crea aliases y columnas
    auxiliares necesarias para similarity.py, impact.py y monte_carlo.py.
    """
    players = _clean_text_columns(player_profiles)
    teams = _clean_text_columns(teams)
    matchups = _clean_text_columns(matchups)
    roster = _clean_text_columns(app_roster) if app_roster is not None else None

    players = _ensure_team_columns_from_latest(players)

    players = _coerce_numeric_columns(players, PLAYER_NUMERIC_COLUMNS)
    teams = _coerce_numeric_columns(teams, TEAM_NUMERIC_COLUMNS)
    matchups = _coerce_numeric_columns(matchups, MATCHUP_NUMERIC_COLUMNS)

    if roster is not None:
        roster = _coerce_numeric_columns(roster, APP_ROSTER_NUMERIC_COLUMNS)
        roster["player_id"] = pd.to_numeric(roster["player_id"], errors="coerce").fillna(0).astype(int)
        roster["latest_team"] = roster["latest_team"].fillna("").astype(str).str.strip()
        roster["player_name"] = roster["player_name"].fillna("").astype(str).str.strip()

    players["player_id"] = pd.to_numeric(players["player_id"], errors="coerce").fillna(0).astype(int)
    players["player_name"] = players["player_name"].fillna("").astype(str).str.strip()
    players["position"] = (
        players["position"]
        .fillna("UNK")
        .replace(["", "nan", "None", "NaN", "NA"], "UNK")
        .astype(str)
        .str.strip()
    )

    teams["team"] = teams["team"].fillna("").astype(str).str.strip()
    matchups["team"] = matchups["team"].fillna("").astype(str).str.strip()
    matchups["opponent_team"] = matchups["opponent_team"].fillna("").astype(str).str.strip()

    # En player_profiles.csv, minutes representa minutos promedio por partido.
    # Por eso expected_minutes debe ser igual a minutes.
    players["expected_minutes"] = pd.to_numeric(
        players.get("minutes", 24.0),
        errors="coerce",
    ).fillna(24.0)

    # role_features.py crea aliases y la feature rim_pressure si hiciera falta.
    players = prepare_players_data(players, overwrite_roles=False)

    for col in ["player_id", "team_id", "games_played"]:
        if col in players.columns:
            players[col] = pd.to_numeric(players[col], errors="coerce").fillna(0)
            players[col] = players[col].astype(int)

    if "team_id" not in teams.columns:
        if "latest_team_id" in roster.columns if roster is not None else False:
            team_id_map = (
                roster.dropna(subset=["latest_team", "latest_team_id"])
                .groupby("latest_team")["latest_team_id"]
                .agg(lambda values: values.mode().iloc[0])
                .to_dict()
            )
        else:
            team_id_map = (
                players.dropna(subset=["team", "team_id"])
                .groupby("team")["team_id"]
                .agg(lambda values: values.mode().iloc[0])
                .to_dict()
            )

        teams["team_id"] = teams["team"].map(team_id_map)

    teams["team_id"] = pd.to_numeric(teams["team_id"], errors="coerce")
    missing_team_id = teams["team_id"].isna()

    if missing_team_id.any():
        max_existing_id = pd.to_numeric(players.get("team_id", pd.Series([0])), errors="coerce").max()

        if pd.isna(max_existing_id):
            max_existing_id = 0

        fallback_ids = (
            pd.factorize(teams.loc[missing_team_id, "team"].astype(str))[0]
            + int(max_existing_id)
            + 1
        )

        teams.loc[missing_team_id, "team_id"] = fallback_ids

    teams["team_id"] = teams["team_id"].astype(int)
    teams["net_rating"] = pd.to_numeric(teams["offensive_rating"], errors="coerce").fillna(0) - pd.to_numeric(teams["defensive_rating"], errors="coerce").fillna(0)
    teams["ORtg"] = teams["offensive_rating"]
    teams["DRtg"] = teams["defensive_rating"]

    team_to_id = teams.set_index("team")["team_id"].to_dict()

    if "team_id" not in matchups.columns:
        matchups["team_id"] = matchups["team"].map(team_to_id)

    if "opponent_team_id" not in matchups.columns:
        matchups["opponent_team_id"] = matchups["opponent_team"].map(team_to_id)

    matchups["team_id"] = pd.to_numeric(matchups["team_id"], errors="coerce")
    matchups["opponent_team_id"] = pd.to_numeric(matchups["opponent_team_id"], errors="coerce")

    return players, teams, matchups, roster


def load_recommendation_data(
    processed_dir: str | Path = "data/processed",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Carga los archivos correctos para Fase 3.

    Fuentes principales:
    - player_profiles.csv: perfil histórico para similitud, impacto y ranking.
    - app_roster.csv: pertenencia al equipo seleccionable y banca.
    - processed_teams.csv: contexto de equipo.
    - processed_matchups.csv: contexto de matchups.
    """
    processed_dir = Path(processed_dir)

    profiles_path = processed_dir / "player_profiles.csv"
    app_roster_path = processed_dir / "app_roster.csv"
    teams_path = processed_dir / "processed_teams.csv"
    matchups_path = processed_dir / "processed_matchups.csv"

    missing = [
        str(path)
        for path in [profiles_path, app_roster_path, teams_path, matchups_path]
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Faltan archivos procesados para Fase 3. "
            f"Archivos faltantes: {missing}. "
            "Ejecuta primero: python src/preprocessing.py"
        )

    player_profiles = pd.read_csv(profiles_path)
    app_roster = pd.read_csv(app_roster_path)
    teams = pd.read_csv(teams_path)
    matchups = pd.read_csv(matchups_path)

    _validate_columns(player_profiles, PLAYER_PROFILE_REQUIRED_COLUMNS, "player_profiles.csv")
    _validate_columns(app_roster, APP_ROSTER_REQUIRED_COLUMNS, "app_roster.csv")
    _validate_columns(teams, TEAM_REQUIRED_COLUMNS, "processed_teams.csv")
    _validate_columns(matchups, MATCHUP_REQUIRED_COLUMNS, "processed_matchups.csv")

    player_profiles, teams, matchups, app_roster = prepare_runtime_data(
        player_profiles=player_profiles,
        teams=teams,
        matchups=matchups,
        app_roster=app_roster,
    )

    if app_roster is None:
        raise ValueError("app_roster.csv no pudo cargarse correctamente.")

    return player_profiles, teams, matchups, app_roster


def load_processed_data(
    processed_dir: str | Path = "data/processed",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Compatibilidad temporal con app.py actual.

    Devuelve player_profiles como primer DataFrame para que la app antigua pueda
    cargar datos sin romperse. Para recomendaciones nuevas, usar
    load_recommendation_data().
    """
    player_profiles, teams, matchups, _ = load_recommendation_data(processed_dir)
    return player_profiles, teams, matchups


# ============================================================
# 5. RESOLVER EQUIPO, JUGADOR Y MATCHUP
# ============================================================


def resolve_team_row(
    teams: pd.DataFrame,
    team_value: str | int,
) -> pd.Series:
    """Encuentra un equipo por team_id, abreviación, nombre exacto o parcial."""
    value_text = _normalize_text(team_value)

    if str(team_value).strip().isdigit() and "team_id" in teams.columns:
        team_id = int(team_value)
        match = teams[pd.to_numeric(teams["team_id"], errors="coerce") == team_id]

        if not match.empty:
            return match.iloc[0]

    aliases = _team_aliases(team_value)
    team_text = teams["team"].astype(str).map(_normalize_text)

    exact_or_alias = teams[team_text.isin(aliases)]

    if not exact_or_alias.empty:
        return exact_or_alias.iloc[0]

    partial = teams[team_text.str.contains(value_text, na=False, regex=False)]

    if not partial.empty:
        return partial.iloc[0]

    available_teams = teams["team"].dropna().drop_duplicates().head(30).to_list()

    raise ValueError(
        f"No se encontró el equipo: {team_value}. "
        f"Equipos disponibles, muestra: {available_teams}"
    )


def get_app_roster_for_team(
    app_roster: pd.DataFrame,
    selected_team_id: Optional[int],
    selected_team_name: str,
) -> pd.DataFrame:
    """Devuelve roster usable para la app de un equipo seleccionado."""
    mask = _same_team_mask(
        app_roster,
        team_id=selected_team_id,
        team_name=selected_team_name,
        id_column="latest_team_id",
        name_column="latest_team",
    )

    roster = app_roster[mask].copy()

    if roster.empty:
        available = (
            app_roster["latest_team"].dropna().astype(str).drop_duplicates().sort_values().head(30).to_list()
        )
        raise ValueError(
            f"No hay jugadores en app_roster.csv para el equipo seleccionado: "
            f"{selected_team_name}. Equipos disponibles, muestra: {available}"
        )

    return roster.reset_index(drop=True)


def get_profiles_for_player_ids(
    player_profiles: pd.DataFrame,
    player_ids: set[int] | list[int],
) -> pd.DataFrame:
    """Devuelve perfiles históricos para una lista/set de player_id."""
    ids = _safe_id_set(list(player_ids))
    return player_profiles[player_profiles["player_id"].isin(ids)].copy()


def resolve_player_profile_by_id(
    player_profiles: pd.DataFrame,
    player_id: int,
) -> pd.Series:
    """Encuentra perfil histórico de un jugador por player_id."""
    player_id = _safe_int(player_id, default=-1)
    match = player_profiles[player_profiles["player_id"] == player_id]

    if match.empty:
        raise ValueError(
            f"No se encontró perfil histórico para player_id={player_id} "
            "en player_profiles.csv."
        )

    return match.iloc[0]


def resolve_player_id_from_value(
    player_profiles: pd.DataFrame,
    current_roster: pd.DataFrame,
    player_value: str | int,
) -> int:
    """
    Compatibilidad temporal: resuelve un jugador por id o nombre dentro del roster.

    La Fase 3 debe usar replaced_player_id, pero esta función permite migración.
    """
    if str(player_value).strip().isdigit():
        candidate_id = int(player_value)
        if candidate_id in set(current_roster["player_id"].astype(int)):
            return candidate_id

    value_text = _normalize_text(player_value)

    roster_text = current_roster["player_name"].astype(str).map(_normalize_text)
    exact_roster = current_roster[roster_text == value_text]

    if not exact_roster.empty:
        return int(exact_roster.iloc[0]["player_id"])

    partial_roster = current_roster[roster_text.str.contains(value_text, na=False, regex=False)]

    if not partial_roster.empty:
        return int(partial_roster.iloc[0]["player_id"])

    profile_text = player_profiles["player_name"].astype(str).map(_normalize_text)
    exact_profile = player_profiles[profile_text == value_text]

    if not exact_profile.empty:
        return int(exact_profile.iloc[0]["player_id"])

    raise ValueError(
        f"No se pudo resolver el jugador a reemplazar: {player_value}. "
        "En Fase 3 se recomienda pasar replaced_player_id explícitamente."
    )


def resolve_matchup_row(
    matchups: pd.DataFrame,
    opponent_team_id: Optional[int],
    opponent_team_name: str,
) -> pd.Series:
    """Busca una fila de processed_matchups.csv asociada al rival."""
    candidates = matchups.copy()

    if opponent_team_id is not None and "team_id" in candidates.columns:
        by_id = candidates[pd.to_numeric(candidates["team_id"], errors="coerce") == opponent_team_id]

        if not by_id.empty:
            return by_id.iloc[0]

    aliases = _team_aliases(opponent_team_name)
    team_text = candidates["team"].astype(str).map(_normalize_text)

    by_team = candidates[team_text.isin(aliases)]

    if not by_team.empty:
        return by_team.iloc[0]

    available_matchup_teams = candidates["team"].dropna().drop_duplicates().head(30).to_list()

    raise ValueError(
        f"No se encontró matchup para el rival: {opponent_team_name}. "
        f"Equipos disponibles en matchups, muestra: {available_matchup_teams}"
    )


# ============================================================
# 6. VALIDACIÓN DE LINEUP Y CANDIDATOS DE BANCA
# ============================================================


def validate_lineup_inputs(
    lineup_player_ids: list[int] | tuple[int] | pd.Series | np.ndarray | None,
    replaced_player_id: int,
    current_roster: pd.DataFrame,
    selected_team_name: str,
) -> set[int]:
    """Valida quinteta, reemplazado y pertenencia al equipo en app_roster.csv."""
    lineup_ids = _safe_id_set(lineup_player_ids)

    if len(lineup_ids) != 5:
        raise ValueError(
            "lineup_player_ids debe contener exactamente 5 jugadores únicos. "
            f"Recibidos: {len(lineup_ids)}."
        )

    replaced_player_id = _safe_int(replaced_player_id, default=-1)

    if replaced_player_id not in lineup_ids:
        raise ValueError(
            f"El replaced_player_id={replaced_player_id} debe estar dentro de "
            "lineup_player_ids."
        )

    roster_ids = set(current_roster["player_id"].astype(int).tolist())
    missing_from_roster = sorted(lineup_ids - roster_ids)

    if missing_from_roster:
        roster_preview = (
            current_roster[["player_id", "player_name", "latest_team"]]
            .head(15)
            .to_dict(orient="records")
        )
        raise ValueError(
            f"Estos player_id de la quinteta no pertenecen a {selected_team_name} "
            f"según app_roster.csv: {missing_from_roster}. "
            f"Muestra del roster válido: {roster_preview}"
        )

    return lineup_ids


def build_bench_candidates(
    player_profiles: pd.DataFrame,
    app_roster: pd.DataFrame,
    selected_team_id: Optional[int],
    selected_team_name: str,
    lineup_player_ids: set[int],
    replaced_player_id: int,
    min_minutes: Optional[float] = 10.0,
    min_games: Optional[int] = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Construye candidatos desde la banca del mismo equipo ANTES de rankear.

    Reglas:
    - mismo equipo seleccionado en app_roster.csv
    - fuera de lineup_player_ids
    - distinto al jugador reemplazado
    - existente en player_profiles.csv
    - cumple min_games y min_minutes
    """
    current_roster = get_app_roster_for_team(
        app_roster=app_roster,
        selected_team_id=selected_team_id,
        selected_team_name=selected_team_name,
    )

    roster_ids = set(current_roster["player_id"].astype(int).tolist())
    bench_ids = roster_ids - set(lineup_player_ids)
    bench_ids.discard(_safe_int(replaced_player_id, default=-1))

    if not bench_ids:
        raise ValueError(
            f"No hay jugadores de banca disponibles para {selected_team_name} "
            "después de excluir la quinteta."
        )

    candidates = player_profiles[player_profiles["player_id"].isin(bench_ids)].copy()

    if candidates.empty:
        raise ValueError(
            f"La banca de {selected_team_name} no tiene perfiles históricos "
            "disponibles en player_profiles.csv."
        )

    # Adjuntamos metadata de app_roster para salida y validación.
    roster_metadata_cols = [
        "player_id",
        "latest_team_id",
        "latest_team",
        "latest_season",
        "latest_game_date",
    ]
    roster_metadata_cols = [col for col in roster_metadata_cols if col in current_roster.columns]

    candidates = candidates.merge(
        current_roster[roster_metadata_cols].drop_duplicates(subset=["player_id"]),
        on="player_id",
        how="left",
        suffixes=("", "_app"),
    )

    candidates = candidates[candidates["player_id"] != _safe_int(replaced_player_id, default=-1)]

    if min_minutes is not None:
        candidates = candidates[pd.to_numeric(candidates["minutes"], errors="coerce") >= float(min_minutes)]

    if min_games is not None:
        candidates = candidates[pd.to_numeric(candidates["games_played"], errors="coerce") >= int(min_games)]

    if candidates.empty:
        raise ValueError(
            "No hay candidatos válidos en la banca después de aplicar filtros. "
            f"Equipo={selected_team_name}, min_games={min_games}, "
            f"min_minutes={min_minutes}. Prueba bajar los filtros."
        )

    return candidates.reset_index(drop=True), current_roster.reset_index(drop=True)


# ============================================================
# 7. SCORING E IMPACTO
# ============================================================


def score_candidate(
    candidate: pd.Series,
    replaced_player: pd.Series,
    selected_team: pd.Series,
    opponent_team: pd.Series,
    opponent_matchup: pd.Series,
    team_need: Optional[dict[str, float]] = None,
    opponent_context: Optional[dict[str, float]] = None,
) -> dict:
    """
    Calcula métricas de un candidato usando similarity.py e impact.py.
    """
    _ = opponent_team
    _ = opponent_matchup

    replaced_dict = _to_dict(replaced_player)
    candidate_dict = _to_dict(candidate)
    selected_team_dict = _to_dict(selected_team)

    score_data = similarity.calculate_replacement_score(
        replaced_player=replaced_dict,
        candidate=candidate_dict,
        team_need=team_need,
        opponent_context=opponent_context,
    )

    impact_data = impact.estimate_player_impact(
        candidate=candidate_dict,
        replaced_player=replaced_dict,
        score_data=score_data,
        team_context=selected_team_dict,
    )

    latest_team_value = candidate.get("latest_team", candidate.get("team", ""))
    latest_team_id_value = candidate.get("latest_team_id", candidate.get("team_id", 0))

    return {
        "player_id": _safe_int(candidate.get("player_id")),
        "player_name": str(candidate.get("player_name", "")),
        "team_id": _safe_int(candidate.get("team_id")),
        "team": str(candidate.get("team", "")),
        "latest_team_id": _safe_int(latest_team_id_value),
        "latest_team": str(latest_team_value),
        "position": str(candidate.get("position", "")),
        "position_group": str(candidate.get("position_group", "")),
        "games_played": _safe_int(candidate.get("games_played")),
        "minutes": _safe_float(candidate.get("minutes")),
        "points": _safe_float(candidate.get("points")),
        "role_similarity": _safe_float(score_data.get("role_similarity")),
        "position_fit": _safe_float(score_data.get("position_fit")),
        "team_fit": _safe_float(score_data.get("team_fit")),
        "opponent_fit": _safe_float(score_data.get("opponent_fit")),
        "replacement_score": _safe_float(score_data.get("final_score")),
        "recommendation_score": _safe_float(score_data.get("final_score")),
        "offensive_impact": _safe_float(impact_data.get("offensive_impact")),
        "defensive_impact": _safe_float(impact_data.get("defensive_impact")),
        "pace_impact": _safe_float(impact_data.get("pace_impact")),
        "estimated_net_impact": _safe_float(
            impact_data.get("estimated_net_impact", impact_data.get("net_impact"))
        ),
        "candidate_value": _safe_float(impact_data.get("candidate_value")),
        "replaced_value": _safe_float(impact_data.get("replaced_value")),
        "raw_net_delta": _safe_float(impact_data.get("raw_net_delta")),
        "impact_mode": str(impact_data.get("impact_mode", "relative")),
        "fit_multiplier": _safe_float(impact_data.get("fit_multiplier")),
        "comparison_minutes": _safe_float(impact_data.get("comparison_minutes")),
        "cap_applied": bool(impact_data.get("cap_applied", False)),
    }


# ============================================================
# 8. MONTE CARLO
# ============================================================


def simulate_replacement(
    selected_team: pd.Series,
    opponent_team: pd.Series,
    player_impact: dict,
    num_simulations: int,
    use_baseline: bool = False,
) -> dict:
    """Ejecuta Monte Carlo con monte_carlo.monte_carlo_replacement_analysis()."""
    raw_result = monte_carlo.monte_carlo_replacement_analysis(
        team_context=_to_dict(selected_team),
        opponent_context=_to_dict(opponent_team),
        player_impact=player_impact,
        n_simulations=num_simulations,
    )

    scenario_key = "without_replacement" if use_baseline else "with_replacement"
    probability_key = "win_probability_without" if use_baseline else "win_probability_with"

    scenario = raw_result.get(scenario_key, {})

    return {
        "win_probability": _safe_float(raw_result.get(probability_key)),
        "expected_margin": _safe_float(scenario.get("expected_margin")),
        "expected_team_points": _safe_float(scenario.get("expected_team_points")),
        "expected_opponent_points": _safe_float(scenario.get("expected_opponent_points")),
        "margins": _to_list(scenario.get("margins")),
        "team_scores": _to_list(scenario.get("team_scores")),
        "opponent_scores": _to_list(scenario.get("opponent_scores")),
        "raw": raw_result,
    }


# ============================================================
# 9. EXPLICACIONES
# ============================================================


def build_candidate_explanation(
    candidate_result: dict,
    candidate_simulation: dict,
) -> str:
    """Construye explicación individual para un candidato."""
    score_data = {
        "final_score": candidate_result.get("replacement_score", 0.0),
        "role_similarity": candidate_result.get("role_similarity", 0.0),
        "position_fit": candidate_result.get("position_fit", 0.0),
        "team_fit": candidate_result.get("team_fit", 0.0),
        "opponent_fit": candidate_result.get("opponent_fit", 0.0),
    }

    impact_data = {
        "offensive_impact": candidate_result.get("offensive_impact", 0.0),
        "defensive_impact": candidate_result.get("defensive_impact", 0.0),
        "pace_impact": candidate_result.get("pace_impact", 0.0),
        "estimated_net_impact": candidate_result.get("estimated_net_impact", 0.0),
    }

    explanation = explanations.explain_replacement_choice(
        candidate_name=candidate_result.get("player_name", "Candidato"),
        score_data=score_data,
        impact_data=impact_data,
        monte_carlo_data=candidate_simulation.get("raw"),
    )

    explanation += (
        " Nota técnica: los impactos reportados son diferenciales respecto "
        "al jugador reemplazado, no aportes absolutos del candidato."
    )

    return explanation


def build_text_explanation(
    replaced_player: pd.Series,
    selected_team: pd.Series,
    opponent_team: pd.Series,
    top_replacements: list[dict],
    baseline_win_probability: float,
) -> str:
    """Construye explicación global del ranking."""
    summary_ready = []

    for result in top_replacements:
        copied = dict(result)
        copied["final_score"] = copied.get("replacement_score", 0.0)
        summary_ready.append(copied)

    baseline_text = (
        f"Escenario: reemplazar a {replaced_player.get('player_name', 'N/A')} "
        f"en {selected_team.get('team', 'N/A')} contra "
        f"{opponent_team.get('team', 'N/A')}. "
        f"La probabilidad simulada de ganar sin reemplazo es "
        f"{baseline_win_probability * 100:.1f}%."
    )

    candidates_text = explanations.summarize_top_candidates(
        summary_ready,
        top_n=len(summary_ready),
    )

    return f"{baseline_text}\n\n{candidates_text}"


# ============================================================
# 10. FLUJO PRINCIPAL DEL RECOMENDADOR
# ============================================================


def recommend_replacements(
    selected_team_value: str | int,
    opponent_team_value: str | int,
    lineup_player_ids: list[int] | tuple[int] | pd.Series | np.ndarray | None = None,
    replaced_player_id: int | None = None,
    player_to_replace_value: str | int | None = None,
    num_simulations: int = 1000,
    monte_carlo_simulations: int | None = None,
    processed_dir: str | Path = "data/processed",
    top_n: int = 3,
    min_minutes: Optional[float] = 10.0,
    min_games: Optional[int] = 10,
    exclude_opponent_players: bool = True,
    random_state: Optional[int] = 42,
) -> dict:
    """
    Flujo completo del recomendador para Fase 3.

    app_roster.csv decide pertenencia al roster usable y banca.
    player_profiles.csv decide el perfil estadístico usado para ranking.
    """
    _ = exclude_opponent_players  # Se conserva por compatibilidad, pero ya no define la banca.
    _ = random_state  # Monte Carlo actual usa semillas internas fijas.

    if monte_carlo_simulations is not None:
        num_simulations = int(monte_carlo_simulations)

    player_profiles, teams, matchups, app_roster = load_recommendation_data(processed_dir)

    selected_team = resolve_team_row(
        teams=teams,
        team_value=selected_team_value,
    )

    opponent_team = resolve_team_row(
        teams=teams,
        team_value=opponent_team_value,
    )

    selected_team_id = _safe_int(selected_team.get("team_id"))
    opponent_team_id = _safe_int(opponent_team.get("team_id"))

    selected_team_name = str(selected_team.get("team", ""))
    opponent_team_name = str(opponent_team.get("team", ""))

    current_roster = get_app_roster_for_team(
        app_roster=app_roster,
        selected_team_id=selected_team_id,
        selected_team_name=selected_team_name,
    )

    if replaced_player_id is None:
        if player_to_replace_value is None:
            raise ValueError(
                "Debes pasar replaced_player_id. "
                "Compatibilidad temporal: también puedes pasar player_to_replace_value."
            )
        replaced_player_id = resolve_player_id_from_value(
            player_profiles=player_profiles,
            current_roster=current_roster,
            player_value=player_to_replace_value,
        )

    replaced_player_id = _safe_int(replaced_player_id, default=-1)

    lineup_ids = validate_lineup_inputs(
        lineup_player_ids=lineup_player_ids,
        replaced_player_id=replaced_player_id,
        current_roster=current_roster,
        selected_team_name=selected_team_name,
    )

    replaced_player = resolve_player_profile_by_id(
        player_profiles=player_profiles,
        player_id=replaced_player_id,
    )

    opponent_matchup = resolve_matchup_row(
        matchups=matchups,
        opponent_team_id=opponent_team_id,
        opponent_team_name=opponent_team_name,
    )

    candidates, current_roster = build_bench_candidates(
        player_profiles=player_profiles,
        app_roster=app_roster,
        selected_team_id=selected_team_id,
        selected_team_name=selected_team_name,
        lineup_player_ids=lineup_ids,
        replaced_player_id=replaced_player_id,
        min_minutes=min_minutes,
        min_games=min_games,
    )

    selected_roster_profile_ids = set(current_roster["player_id"].astype(int).tolist())
    selected_team_players = get_profiles_for_player_ids(
        player_profiles=player_profiles,
        player_ids=selected_roster_profile_ids,
    )

    opponent_roster = get_app_roster_for_team(
        app_roster=app_roster,
        selected_team_id=opponent_team_id,
        selected_team_name=opponent_team_name,
    )

    opponent_players = get_profiles_for_player_ids(
        player_profiles=player_profiles,
        player_ids=set(opponent_roster["player_id"].astype(int).tolist()),
    )

    if not selected_team_players.empty:
        team_need = similarity.build_team_need(selected_team_players)
    else:
        team_need = None

    if not opponent_players.empty:
        opponent_context = similarity.build_opponent_context_from_players(opponent_players)
    else:
        opponent_context = similarity.build_opponent_context_from_matchup(_to_dict(opponent_matchup))

    zero_impact = {
        "offensive_impact": 0.0,
        "defensive_impact": 0.0,
        "pace_impact": 0.0,
        "estimated_net_impact": 0.0,
        "net_impact": 0.0,
    }

    baseline_simulation = simulate_replacement(
        selected_team=selected_team,
        opponent_team=opponent_team,
        player_impact=zero_impact,
        num_simulations=num_simulations,
        use_baseline=True,
    )

    baseline_win_probability = baseline_simulation["win_probability"]

    scored_candidates = []

    for _, candidate in candidates.iterrows():
        candidate_result = score_candidate(
            candidate=candidate,
            replaced_player=replaced_player,
            selected_team=selected_team,
            opponent_team=opponent_team,
            opponent_matchup=opponent_matchup,
            team_need=team_need,
            opponent_context=opponent_context,
        )

        player_impact = {
            "offensive_impact": candidate_result["offensive_impact"],
            "defensive_impact": candidate_result["defensive_impact"],
            "pace_impact": candidate_result["pace_impact"],
            "estimated_net_impact": candidate_result["estimated_net_impact"],
            "net_impact": candidate_result["estimated_net_impact"],
        }

        candidate_simulation = simulate_replacement(
            selected_team=selected_team,
            opponent_team=opponent_team,
            player_impact=player_impact,
            num_simulations=num_simulations,
            use_baseline=False,
        )

        candidate_result["win_probability_with_replacement"] = candidate_simulation["win_probability"]
        candidate_result["win_probability_delta"] = (
            candidate_result["win_probability_with_replacement"] - baseline_win_probability
        )
        candidate_result["expected_margin_with_replacement"] = candidate_simulation["expected_margin"]
        candidate_result["simulation_distribution"] = {
            "margins": candidate_simulation["margins"],
            "team_scores": candidate_simulation["team_scores"],
            "opponent_scores": candidate_simulation["opponent_scores"],
        }
        candidate_result["explanation"] = build_candidate_explanation(
            candidate_result=candidate_result,
            candidate_simulation=candidate_simulation,
        )

        scored_candidates.append(candidate_result)

    results_df = pd.DataFrame(scored_candidates)

    if results_df.empty:
        raise ValueError("No se pudo calcular score para ningún candidato de banca.")

    results_df = results_df.sort_values(
        by=[
            "win_probability_with_replacement",
            "replacement_score",
            "estimated_net_impact",
        ],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    top_replacements = results_df.head(top_n).to_dict(orient="records")

    explanation = build_text_explanation(
        replaced_player=replaced_player,
        selected_team=selected_team,
        opponent_team=opponent_team,
        top_replacements=top_replacements,
        baseline_win_probability=baseline_win_probability,
    )

    current_roster_ids = set(current_roster["player_id"].astype(int).tolist())
    bench_ids_before_filters = sorted(current_roster_ids - lineup_ids - {replaced_player_id})

    return {
        "selected_team": _to_dict(selected_team),
        "opponent_team": _to_dict(opponent_team),
        "replaced_player": _to_dict(replaced_player),
        "replaced_player_id": replaced_player_id,
        "lineup_player_ids": sorted(lineup_ids),
        "num_simulations": num_simulations,
        "filters": {
            "min_minutes": min_minutes,
            "min_games": min_games,
            "source_roster": "app_roster.csv",
            "source_profiles": "player_profiles.csv",
            "candidate_rule": "same_team_bench_before_ranking",
        },
        "roster_debug": {
            "current_roster_size": len(current_roster),
            "bench_size_before_filters": len(bench_ids_before_filters),
            "candidate_size_after_filters": len(candidates),
        },
        "contexts": {
            "team_need": team_need,
            "opponent_context": opponent_context,
        },
        "baseline": {
            "win_probability_without_replacement": baseline_win_probability,
            "expected_margin_without_replacement": baseline_simulation["expected_margin"],
            "simulation_distribution": {
                "margins": baseline_simulation["margins"],
                "team_scores": baseline_simulation["team_scores"],
                "opponent_scores": baseline_simulation["opponent_scores"],
            },
        },
        "top_replacements": top_replacements,
        "all_candidates": results_df,
        "explanation": explanation,
    }


# ============================================================
# 11. EJEMPLO DE USO DIRECTO
# ============================================================

if __name__ == "__main__":
    # Ejemplo mínimo. Para pruebas completas, usa checks/check_recommender_backend.py.
    processed_dir = Path("data/processed")
    player_profiles, _, _, app_roster = load_recommendation_data(processed_dir)

    lal = app_roster[app_roster["latest_team"].astype(str).str.upper() == "LAL"].copy()
    names = [
        "LeBron James",
        "Anthony Davis",
        "Austin Reaves",
        "Dennis Schroder",
        "Lonnie Walker IV",
    ]

    lineup = []
    for name in names:
        row = lal[lal["player_name"].astype(str).str.lower() == name.lower()]
        if not row.empty:
            lineup.append(int(row.iloc[0]["player_id"]))

    if len(lineup) == 5:
        result = recommend_replacements(
            selected_team_value="LAL",
            opponent_team_value="BOS",
            lineup_player_ids=lineup,
            replaced_player_id=lineup[0],
            num_simulations=1000,
            processed_dir=processed_dir,
            top_n=3,
            min_minutes=10,
            min_games=10,
        )

        print("Top reemplazos:")
        for i, player in enumerate(result["top_replacements"], start=1):
            print(
                f"{i}. {player['player_name']} - {player['latest_team']} "
                f"score={player['replacement_score']:.3f}"
            )
    else:
        print("No se pudo construir la quinteta ejemplo de LAL.")
