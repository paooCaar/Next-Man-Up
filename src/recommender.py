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
# 1. CONTRATO REAL DE DATOS PROCESADOS
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
    """
    Normaliza texto para comparar equipos y jugadores.
    """
    if pd.isna(value):
        return ""

    return str(value).strip().lower()


def _team_aliases(value: Any) -> set[str]:
    """
    Devuelve posibles textos equivalentes para buscar un equipo.
    """
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
    """
    Convierte una fila de pandas a diccionario plano.
    """
    if isinstance(row, pd.Series):
        return row.to_dict()

    return dict(row)


def _to_list(value: Any) -> list:
    """
    Convierte arrays o series a listas serializables.
    """
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
    """
    Convierte valores a float evitando NaN e infinitos.
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default

    if not np.isfinite(number):
        return default

    return number


def _safe_int(value: Any, default: int = 0) -> int:
    """
    Convierte valores a int de forma segura.
    """
    return int(round(_safe_float(value, default=default)))


def _clean_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia espacios en columnas de texto sin convertir NaN en 'nan'.
    """
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
    """
    Verifica que una tabla tenga las columnas mínimas esperadas.
    """
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(
            f"{table_name} no tiene las columnas necesarias: {missing_columns}"
        )


def _coerce_numeric_columns(
    df: pd.DataFrame,
    numeric_columns: list[str],
) -> pd.DataFrame:
    """
    Convierte columnas numéricas existentes y rellena NaN de forma simple.
    """
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
) -> pd.Series:
    """
    Máscara flexible para filtrar por equipo.
    Usa team_id si existe y también compara texto.
    """
    mask = pd.Series(False, index=df.index)

    if team_id is not None and "team_id" in df.columns:
        mask = mask | (pd.to_numeric(df["team_id"], errors="coerce") == team_id)

    if team_name is not None and "team" in df.columns:
        aliases = _team_aliases(team_name)
        team_norm = df["team"].astype(str).map(_normalize_text)
        mask = mask | team_norm.isin(aliases)

    return mask


# ============================================================
# 4. PREPARACIÓN RUNTIME
# ============================================================


def prepare_runtime_data(
    players: pd.DataFrame,
    teams: pd.DataFrame,
    matchups: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Ajusta columnas derivadas para conectar los CSV procesados actuales
    con similarity.py, impact.py y monte_carlo.py.

    No reemplaza el preprocesamiento.
    Solo crea columnas auxiliares necesarias en memoria:
    - expected_minutes
    - position_group
    - rim_pressure
    - columnas per36
    - aliases ORtg, DRtg, USG
    - team_id en teams y matchups si falta
    - net_rating en teams
    """
    players = _clean_text_columns(players)
    teams = _clean_text_columns(teams)
    matchups = _clean_text_columns(matchups)

    players = _coerce_numeric_columns(players, PLAYER_NUMERIC_COLUMNS)
    teams = _coerce_numeric_columns(teams, TEAM_NUMERIC_COLUMNS)
    matchups = _coerce_numeric_columns(matchups, MATCHUP_NUMERIC_COLUMNS)

    players["player_name"] = players["player_name"].fillna("").astype(str).str.strip()
    players["team"] = players["team"].fillna("").astype(str).str.strip()
    players["position"] = (
        players["position"]
        .fillna("UNK")
        .replace(["", "nan", "None", "NaN", "NA"], "UNK")
        .astype(str)
        .str.strip()
    )

    teams["team"] = teams["team"].fillna("").astype(str).str.strip()

    matchups["team"] = matchups["team"].fillna("").astype(str).str.strip()
    matchups["opponent_team"] = (
        matchups["opponent_team"].fillna("").astype(str).str.strip()
    )

    # En processed_players.csv, minutes ya representa minutos promedio por partido.
    # Por eso expected_minutes debe ser igual a minutes.
    if "expected_minutes" not in players.columns:
        players["expected_minutes"] = pd.to_numeric(
            players["minutes"],
            errors="coerce",
        ).fillna(24.0)

    # role_features.py crea aliases y la feature rim_pressure que similarity.py espera.
    players = prepare_players_data(players, overwrite_roles=False)

    for col in ["player_id", "team_id", "games_played"]:
        if col in players.columns:
            players[col] = pd.to_numeric(players[col], errors="coerce").fillna(0)
            players[col] = players[col].astype(int)

    if "team_id" not in teams.columns:
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
        max_existing_id = pd.to_numeric(players["team_id"], errors="coerce").max()

        if pd.isna(max_existing_id):
            max_existing_id = 0

        fallback_ids = (
            pd.factorize(teams.loc[missing_team_id, "team"].astype(str))[0]
            + int(max_existing_id)
            + 1
        )

        teams.loc[missing_team_id, "team_id"] = fallback_ids

    teams["team_id"] = teams["team_id"].astype(int)

    teams["net_rating"] = pd.to_numeric(
        teams["offensive_rating"], errors="coerce"
    ).fillna(0) - pd.to_numeric(teams["defensive_rating"], errors="coerce").fillna(0)

    teams["ORtg"] = teams["offensive_rating"]
    teams["DRtg"] = teams["defensive_rating"]

    team_to_id = teams.set_index("team")["team_id"].to_dict()

    if "team_id" not in matchups.columns:
        matchups["team_id"] = matchups["team"].map(team_to_id)

    if "opponent_team_id" not in matchups.columns:
        matchups["opponent_team_id"] = matchups["opponent_team"].map(team_to_id)

    matchups["team_id"] = pd.to_numeric(matchups["team_id"], errors="coerce")
    matchups["opponent_team_id"] = pd.to_numeric(
        matchups["opponent_team_id"],
        errors="coerce",
    )

    return players, teams, matchups


# ============================================================
# 5. CARGA DE DATOS
# ============================================================


def load_processed_data(
    processed_dir: str | Path = "data/processed",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Carga y valida los tres archivos procesados.

    Archivos esperados:
    - processed_players.csv
    - processed_teams.csv
    - processed_matchups.csv
    """
    processed_dir = Path(processed_dir)

    players_path = processed_dir / "processed_players.csv"
    teams_path = processed_dir / "processed_teams.csv"
    matchups_path = processed_dir / "processed_matchups.csv"

    if not players_path.exists():
        raise FileNotFoundError(f"No se encontró: {players_path}")

    if not teams_path.exists():
        raise FileNotFoundError(f"No se encontró: {teams_path}")

    if not matchups_path.exists():
        raise FileNotFoundError(f"No se encontró: {matchups_path}")

    players = pd.read_csv(players_path)
    teams = pd.read_csv(teams_path)
    matchups = pd.read_csv(matchups_path)

    _validate_columns(players, PLAYER_REQUIRED_COLUMNS, "processed_players.csv")
    _validate_columns(teams, TEAM_REQUIRED_COLUMNS, "processed_teams.csv")
    _validate_columns(matchups, MATCHUP_REQUIRED_COLUMNS, "processed_matchups.csv")

    players, teams, matchups = prepare_runtime_data(
        players=players,
        teams=teams,
        matchups=matchups,
    )

    return players, teams, matchups


# ============================================================
# 6. RESOLVER EQUIPO, JUGADOR Y MATCHUP
# ============================================================


def resolve_team_row(
    teams: pd.DataFrame,
    team_value: str | int,
) -> pd.Series:
    """
    Encuentra un equipo por:
    - team_id
    - abreviación: LAL, BOS, etc.
    - nombre exacto o parcial en la columna team
    """
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


def resolve_player_row(
    players: pd.DataFrame,
    player_value: str | int,
    selected_team_id: Optional[int] = None,
    selected_team_name: Optional[str] = None,
) -> pd.Series:
    """
    Encuentra al jugador a reemplazar por:
    - player_id
    - player_name exacto
    - player_name parcial

    Primero busca dentro del equipo seleccionado.
    """
    search_space = players.copy()

    team_mask = _same_team_mask(
        search_space,
        team_id=selected_team_id,
        team_name=selected_team_name,
    )

    if team_mask.any():
        search_space = search_space[team_mask].copy()

    if str(player_value).strip().isdigit():
        player_id = int(player_value)
        match = search_space[
            pd.to_numeric(search_space["player_id"], errors="coerce") == player_id
        ]

        if not match.empty:
            return match.iloc[0]

    value_text = _normalize_text(player_value)
    player_text = search_space["player_name"].astype(str).map(_normalize_text)

    exact = search_space[player_text == value_text]

    if not exact.empty:
        return exact.iloc[0]

    partial = search_space[player_text.str.contains(value_text, na=False, regex=False)]

    if not partial.empty:
        return partial.iloc[0]

    available_players = (
        search_space[["player_name", "team"]]
        .drop_duplicates()
        .head(30)
        .to_dict(orient="records")
    )

    raise ValueError(
        f"No se encontró el jugador: {player_value}. "
        f"Jugadores disponibles en el espacio de búsqueda, muestra: "
        f"{available_players}"
    )


def resolve_matchup_row(
    matchups: pd.DataFrame,
    opponent_team_id: Optional[int],
    opponent_team_name: str,
) -> pd.Series:
    """
    Busca una fila de processed_matchups.csv asociada al rival.

    El matchup actual no trae perfil defensivo agregado; por ahora usamos
    cualquier fila histórica del rival como ancla para construir contexto
    neutral si no hay jugadores del rival disponibles.
    """
    candidates = matchups.copy()

    if opponent_team_id is not None and "team_id" in candidates.columns:
        by_id = candidates[
            pd.to_numeric(candidates["team_id"], errors="coerce") == opponent_team_id
        ]

        if not by_id.empty:
            return by_id.iloc[0]

    aliases = _team_aliases(opponent_team_name)
    team_text = candidates["team"].astype(str).map(_normalize_text)

    by_team = candidates[team_text.isin(aliases)]

    if not by_team.empty:
        return by_team.iloc[0]

    available_matchup_teams = (
        candidates["team"].dropna().drop_duplicates().head(30).to_list()
    )

    raise ValueError(
        f"No se encontró matchup para el rival: {opponent_team_name}. "
        f"Equipos disponibles en matchups, muestra: {available_matchup_teams}"
    )


def get_team_players(
    players: pd.DataFrame,
    team_id: Optional[int],
    team_name: str,
) -> pd.DataFrame:
    """
    Devuelve jugadores de un equipo usando team_id y/o texto.
    """
    mask = _same_team_mask(players, team_id=team_id, team_name=team_name)

    return players[mask].copy()


# ============================================================
# 7. FILTRO Y DEDUPLICACIÓN DE CANDIDATOS
# ============================================================


def build_player_deduplication_key(row: pd.Series) -> str:
    """
    Construye una llave estable para identificar jugadores únicos.

    Prioridad:
    1. player_id válido.
    2. player_name normalizado como respaldo.
    """
    player_id = pd.to_numeric(row.get("player_id"), errors="coerce")

    if pd.notna(player_id) and int(player_id) > 0:
        return f"id:{int(player_id)}"

    player_name = _normalize_text(row.get("player_name", ""))

    return f"name:{player_name}"


def deduplicate_candidate_players(candidates: pd.DataFrame) -> pd.DataFrame:
    """
    Elimina duplicados de jugador en candidatos.

    Decisión de diseño:
    - Si player_id existe, deduplicamos por player_id.
    - Si player_id falta o no es válido, deduplicamos por player_name.
    - Nos quedamos con la fila más representativa:
        1. mayor games_played
        2. mayor minutes
        3. mayor points
    """
    if candidates.empty:
        return candidates.reset_index(drop=True)

    candidates = candidates.copy()

    candidates["_dedupe_key"] = candidates.apply(
        build_player_deduplication_key,
        axis=1,
    )

    candidates["_sort_games_played"] = pd.to_numeric(
        candidates.get("games_played", 0),
        errors="coerce",
    ).fillna(0)

    candidates["_sort_minutes"] = pd.to_numeric(
        candidates.get("minutes", 0),
        errors="coerce",
    ).fillna(0)

    candidates["_sort_points"] = pd.to_numeric(
        candidates.get("points", 0),
        errors="coerce",
    ).fillna(0)

    candidates = candidates.sort_values(
        by=[
            "_dedupe_key",
            "_sort_games_played",
            "_sort_minutes",
            "_sort_points",
        ],
        ascending=[True, False, False, False],
    )

    candidates = candidates.drop_duplicates(
        subset=["_dedupe_key"],
        keep="first",
    )

    candidates = candidates.drop(
        columns=[
            "_dedupe_key",
            "_sort_games_played",
            "_sort_minutes",
            "_sort_points",
        ],
        errors="ignore",
    )

    return candidates.reset_index(drop=True)


def filter_candidates(
    players: pd.DataFrame,
    replaced_player: pd.Series,
    opponent_team_id: Optional[int],
    opponent_team_name: Optional[str],
    min_minutes: Optional[float] = 10.0,
    min_games: Optional[int] = 10,
    exclude_opponent_players: bool = True,
    deduplicate_players: bool = True,
) -> pd.DataFrame:
    """
    Filtra candidatos válidos.

    Reglas:
    - excluye al jugador reemplazado
    - opcionalmente excluye jugadores del rival
    - opcionalmente filtra por minutos mínimos
    - opcionalmente filtra por partidos mínimos
    - opcionalmente elimina duplicados de jugador
    """
    candidates = players.copy()

    replaced_player_id = _safe_int(replaced_player.get("player_id", -1), default=-1)

    candidates = candidates[
        pd.to_numeric(candidates["player_id"], errors="coerce") != replaced_player_id
    ]

    if exclude_opponent_players:
        opponent_mask = _same_team_mask(
            candidates,
            team_id=opponent_team_id,
            team_name=opponent_team_name,
        )

        candidates = candidates[~opponent_mask]

    if min_minutes is not None:
        candidates = candidates[
            pd.to_numeric(candidates["minutes"], errors="coerce") >= float(min_minutes)
        ]

    if min_games is not None:
        candidates = candidates[
            pd.to_numeric(candidates["games_played"], errors="coerce") >= int(min_games)
        ]

    if deduplicate_players:
        candidates = deduplicate_candidate_players(candidates)

    return candidates.reset_index(drop=True)


# ============================================================
# 8. SCORING E IMPACTO
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
    Calcula las métricas de un candidato usando las funciones reales:

    - similarity.calculate_replacement_score()
    - impact.estimate_player_impact()
    """
    replaced_dict = _to_dict(replaced_player)
    candidate_dict = _to_dict(candidate)
    selected_team_dict = _to_dict(selected_team)

    score_data = similarity.calculate_replacement_score(
        replaced_player=replaced_dict,
        candidate=candidate_dict,
        team_need=team_need,
        opponent_context=opponent_context,
    )


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
    Calcula las métricas de un candidato usando las funciones reales:

    - similarity.calculate_replacement_score()
    - impact.estimate_player_impact()

    Punto 2:
    El impacto ahora es relativo:
        impact(candidate) - impact(replaced_player)
    """
    replaced_dict = _to_dict(replaced_player)
    candidate_dict = _to_dict(candidate)
    selected_team_dict = _to_dict(selected_team)

    score_data = similarity.calculate_replacement_score(
        replaced_player=replaced_dict,
        candidate=candidate_dict,
        team_need=team_need,
        opponent_context=opponent_context,
    )


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
    Calcula las métricas de un candidato usando las funciones reales:

    - similarity.calculate_replacement_score()
    - impact.estimate_player_impact()

    Punto 2:
    El impacto ahora es relativo:
        impact(candidate) - impact(replaced_player)
    """
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

    return {
        "player_id": _safe_int(candidate.get("player_id")),
        "player_name": str(candidate.get("player_name", "")),
        "team_id": _safe_int(candidate.get("team_id")),
        "team": str(candidate.get("team", "")),
        "position": str(candidate.get("position", "")),
        "position_group": str(candidate.get("position_group", "")),
        "minutes": _safe_float(candidate.get("minutes")),
        "games_played": _safe_int(candidate.get("games_played")),
        "points": _safe_float(candidate.get("points")),
        "role_similarity": _safe_float(score_data.get("role_similarity")),
        "position_fit": _safe_float(score_data.get("position_fit")),
        "team_fit": _safe_float(score_data.get("team_fit")),
        "opponent_fit": _safe_float(score_data.get("opponent_fit")),
        "replacement_score": _safe_float(score_data.get("final_score")),
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

    return {
        "player_id": _safe_int(candidate.get("player_id")),
        "player_name": str(candidate.get("player_name", "")),
        "team_id": _safe_int(candidate.get("team_id")),
        "team": str(candidate.get("team", "")),
        "position": str(candidate.get("position", "")),
        "position_group": str(candidate.get("position_group", "")),
        "minutes": _safe_float(candidate.get("minutes")),
        "games_played": _safe_int(candidate.get("games_played")),
        "points": _safe_float(candidate.get("points")),
        "role_similarity": _safe_float(score_data.get("role_similarity")),
        "position_fit": _safe_float(score_data.get("position_fit")),
        "team_fit": _safe_float(score_data.get("team_fit")),
        "opponent_fit": _safe_float(score_data.get("opponent_fit")),
        "replacement_score": _safe_float(score_data.get("final_score")),
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

    return {
        "player_id": _safe_int(candidate.get("player_id")),
        "player_name": str(candidate.get("player_name", "")),
        "team_id": _safe_int(candidate.get("team_id")),
        "team": str(candidate.get("team", "")),
        "position": str(candidate.get("position", "")),
        "position_group": str(candidate.get("position_group", "")),
        "minutes": _safe_float(candidate.get("minutes")),
        "games_played": _safe_int(candidate.get("games_played")),
        "points": _safe_float(candidate.get("points")),
        "role_similarity": _safe_float(score_data.get("role_similarity")),
        "position_fit": _safe_float(score_data.get("position_fit")),
        "team_fit": _safe_float(score_data.get("team_fit")),
        "opponent_fit": _safe_float(score_data.get("opponent_fit")),
        "replacement_score": _safe_float(score_data.get("final_score")),
        "offensive_impact": _safe_float(impact_data.get("offensive_impact")),
        "defensive_impact": _safe_float(impact_data.get("defensive_impact")),
        "pace_impact": _safe_float(impact_data.get("pace_impact")),
        "estimated_net_impact": _safe_float(
            impact_data.get("estimated_net_impact", impact_data.get("net_impact"))
        ),
    }


# ============================================================
# 9. MONTE CARLO
# ============================================================


def simulate_replacement(
    selected_team: pd.Series,
    opponent_team: pd.Series,
    player_impact: dict,
    num_simulations: int,
    use_baseline: bool = False,
) -> dict:
    """
    Ejecuta Monte Carlo con monte_carlo.monte_carlo_replacement_analysis().

    Si use_baseline=True, devuelve el escenario sin reemplazo.
    Si use_baseline=False, devuelve el escenario con reemplazo.
    """
    raw_result = monte_carlo.monte_carlo_replacement_analysis(
        team_context=_to_dict(selected_team),
        opponent_context=_to_dict(opponent_team),
        player_impact=player_impact,
        n_simulations=num_simulations,
    )

    scenario_key = "without_replacement" if use_baseline else "with_replacement"
    probability_key = (
        "win_probability_without" if use_baseline else "win_probability_with"
    )

    scenario = raw_result.get(scenario_key, {})

    return {
        "win_probability": _safe_float(raw_result.get(probability_key)),
        "expected_margin": _safe_float(scenario.get("expected_margin")),
        "expected_team_points": _safe_float(scenario.get("expected_team_points")),
        "expected_opponent_points": _safe_float(
            scenario.get("expected_opponent_points")
        ),
        "margins": _to_list(scenario.get("margins")),
        "team_scores": _to_list(scenario.get("team_scores")),
        "opponent_scores": _to_list(scenario.get("opponent_scores")),
        "raw": raw_result,
    }


# ============================================================
# 10. EXPLICACIONES
# ============================================================


def build_candidate_explanation(
    candidate_result: dict,
    candidate_simulation: dict,
) -> str:
    """
    Construye explicación individual para un candidato.

    Nota:
    Los impactos ya son diferenciales respecto al jugador reemplazado.
    """
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
    """
    Construye explicación global del ranking.
    """
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
# 11. FLUJO PRINCIPAL DEL RECOMENDADOR
# ============================================================


def recommend_replacements(
    selected_team_value: str | int,
    player_to_replace_value: str | int,
    opponent_team_value: str | int,
    num_simulations: int = 1000,
    processed_dir: str | Path = "data/processed",
    top_n: int = 3,
    min_minutes: Optional[float] = 10.0,
    min_games: Optional[int] = 10,
    exclude_opponent_players: bool = True,
    random_state: Optional[int] = 42,
) -> dict:
    """
    Flujo completo del recomendador.
    """
    _ = random_state  # Monte Carlo actual usa semillas internas fijas.

    players, teams, matchups = load_processed_data(processed_dir)

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

    replaced_player = resolve_player_row(
        players=players,
        player_value=player_to_replace_value,
        selected_team_id=selected_team_id,
        selected_team_name=selected_team_name,
    )

    opponent_matchup = resolve_matchup_row(
        matchups=matchups,
        opponent_team_id=opponent_team_id,
        opponent_team_name=opponent_team_name,
    )

    selected_team_players = get_team_players(
        players=players,
        team_id=selected_team_id,
        team_name=selected_team_name,
    )

    opponent_players = get_team_players(
        players=players,
        team_id=opponent_team_id,
        team_name=opponent_team_name,
    )

    if not selected_team_players.empty:
        team_need = similarity.build_team_need(selected_team_players)
    else:
        team_need = None

    if not opponent_players.empty:
        opponent_context = similarity.build_opponent_context_from_players(
            opponent_players
        )
    else:
        opponent_context = similarity.build_opponent_context_from_matchup(
            _to_dict(opponent_matchup)
        )

    candidates = filter_candidates(
        players=players,
        replaced_player=replaced_player,
        opponent_team_id=opponent_team_id,
        opponent_team_name=opponent_team_name,
        min_minutes=min_minutes,
        min_games=min_games,
        exclude_opponent_players=exclude_opponent_players,
        deduplicate_players=True,
    )

    if candidates.empty:
        raise ValueError(
            "No hay candidatos válidos después de aplicar los filtros. "
            "Prueba bajar min_minutes o min_games."
        )

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

        candidate_result["win_probability_with_replacement"] = candidate_simulation[
            "win_probability"
        ]

        candidate_result["win_probability_delta"] = (
            candidate_result["win_probability_with_replacement"]
            - baseline_win_probability
        )

        candidate_result["expected_margin_with_replacement"] = candidate_simulation[
            "expected_margin"
        ]

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
        raise ValueError("No se pudo calcular score para ningún candidato.")

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

    return {
        "selected_team": _to_dict(selected_team),
        "opponent_team": _to_dict(opponent_team),
        "replaced_player": _to_dict(replaced_player),
        "num_simulations": num_simulations,
        "filters": {
            "min_minutes": min_minutes,
            "min_games": min_games,
            "exclude_opponent_players": exclude_opponent_players,
            "deduplicate_players": True,
        },
        "contexts": {
            "team_need": team_need,
            "opponent_context": opponent_context,
        },
        "baseline": {
            "win_probability_without_replacement": baseline_win_probability,
            "expected_margin_without_replacement": baseline_simulation[
                "expected_margin"
            ],
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
# 12. EJEMPLO DE USO DIRECTO
# ============================================================

if __name__ == "__main__":
    result = recommend_replacements(
        selected_team_value="LAL",
        player_to_replace_value="LeBron James",
        opponent_team_value="BOS",
        num_simulations=1000,
        processed_dir="data/processed",
        top_n=3,
        min_minutes=10,
        min_games=10,
        exclude_opponent_players=True,
        random_state=42,
    )

    print("\nProbabilidad sin reemplazo:")
    print(result["baseline"]["win_probability_without_replacement"])

    print("\nTop reemplazos:")

    for i, player in enumerate(result["top_replacements"], start=1):
        print(f"\n{i}. {player['player_name']} - {player['team']}")
        print(f"   role_similarity: {player['role_similarity']:.3f}")
        print(f"   position_fit: {player['position_fit']:.3f}")
        print(f"   team_fit: {player['team_fit']:.3f}")
        print(f"   opponent_fit: {player['opponent_fit']:.3f}")
        print(f"   replacement_score: {player['replacement_score']:.3f}")
        print(f"   estimated_net_impact: {player['estimated_net_impact']:.3f}")
        print(
            "   win_probability_with_replacement: "
            f"{player['win_probability_with_replacement']:.3f}"
        )

    print("\nExplicación:")
    print(result["explanation"])
