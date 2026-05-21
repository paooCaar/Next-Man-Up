# src/recommender.py

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

try:
    from src import similarity, impact, monte_carlo, explanations
    from src.role_features import prepare_players_data
except ImportError:
    import similarity, impact, monte_carlo, explanations
    from role_features import prepare_players_data


PLAYER_REQUIRED_COLUMNS = [
    "player_id", "player_name", "team_id", "team", "position", "games_played", "minutes",
    "points", "assists", "rebounds", "steals", "blocks", "turnovers", "fg_pct", "three_pct",
    "three_attempts", "usage_rate", "offensive_rating", "defensive_rating", "pace", "plus_minus",
    "scoring", "playmaking", "defense", "rebounding", "spacing", "versatility",
]

TEAM_REQUIRED_COLUMNS = ["team", "offensive_rating", "defensive_rating", "pace"]
MATCHUP_REQUIRED_COLUMNS = ["game_id", "team", "opponent_team", "team_score", "opponent_score", "home", "win"]
APP_ROSTER_REQUIRED_COLUMNS = ["player_id", "player_name", "latest_team", "latest_season"]

PLAYER_NUMERIC_COLUMNS = [
    "player_id", "team_id", "games_played", "minutes", "points", "assists", "rebounds", "steals",
    "blocks", "turnovers", "fg_pct", "three_pct", "three_attempts", "usage_rate", "offensive_rating",
    "defensive_rating", "pace", "plus_minus", "scoring", "playmaking", "defense", "rebounding",
    "spacing", "rim_pressure", "versatility", "latest_team_id", "latest_season", "latest_game_id",
    "recent_season", "recent_minutes", "previous_season", "previous_minutes", "activity_score",
    "trend_score", "recent_impact_per36", "previous_impact_per36", "recency_weighted_minutes_total",
]
TEAM_NUMERIC_COLUMNS = ["offensive_rating", "defensive_rating", "pace"]
MATCHUP_NUMERIC_COLUMNS = ["team_score", "opponent_score", "home", "win"]

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


def _normalize_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


def _team_aliases(value: Any) -> set[str]:
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
    if isinstance(row, pd.Series):
        return row.to_dict()
    return dict(row)


def _to_list(value: Any) -> list:
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
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(number):
        return default
    return number


def _safe_int(value: Any, default: int = 0) -> int:
    return int(round(_safe_float(value, default=default)))


def _clean_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].apply(lambda value: value.strip() if isinstance(value, str) else value)
    return df


def _validate_columns(df: pd.DataFrame, required_columns: list[str], table_name: str) -> None:
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"{table_name} no tiene las columnas necesarias: {missing}")


def _coerce_numeric_columns(df: pd.DataFrame, numeric_columns: list[str]) -> pd.DataFrame:
    df = df.copy()
    for col in numeric_columns:
        if col not in df.columns:
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if col in ["activity_score"]:
            df[col] = df[col].fillna(1.0).clip(0, 1)
        elif col in ["trend_score"]:
            df[col] = df[col].fillna(0.0).clip(-0.25, 0.25)
        else:
            median = df[col].median(skipna=True)
            df[col] = df[col].fillna(0 if pd.isna(median) else median)
    return df


def _same_team_mask(df: pd.DataFrame, team_id: int | None = None, team_name: str | None = None, team_col: str = "team") -> pd.Series:
    mask = pd.Series(False, index=df.index)
    if team_id is not None:
        for id_col in ["team_id", "latest_team_id"]:
            if id_col in df.columns:
                mask = mask | (pd.to_numeric(df[id_col], errors="coerce") == team_id)
    if team_name is not None and team_col in df.columns:
        aliases = _team_aliases(team_name)
        team_norm = df[team_col].astype(str).map(_normalize_text)
        mask = mask | team_norm.isin(aliases)
    return mask


# ============================================================
# Preparación y carga
# ============================================================


def prepare_runtime_data(players: pd.DataFrame, teams: pd.DataFrame, matchups: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    players = _clean_text_columns(players)
    teams = _clean_text_columns(teams)
    matchups = _clean_text_columns(matchups)
    players = _coerce_numeric_columns(players, PLAYER_NUMERIC_COLUMNS)
    teams = _coerce_numeric_columns(teams, TEAM_NUMERIC_COLUMNS)
    matchups = _coerce_numeric_columns(matchups, MATCHUP_NUMERIC_COLUMNS)

    players["player_name"] = players["player_name"].fillna("").astype(str).str.strip()
    players["team"] = players["team"].fillna("").astype(str).str.strip()
    players["position"] = players["position"].fillna("UNK").replace(["", "nan", "None", "NaN", "NA"], "UNK").astype(str).str.strip()
    teams["team"] = teams["team"].fillna("").astype(str).str.strip()
    matchups["team"] = matchups["team"].fillna("").astype(str).str.strip()
    matchups["opponent_team"] = matchups["opponent_team"].fillna("").astype(str).str.strip()

    if "expected_minutes" not in players.columns:
        players["expected_minutes"] = pd.to_numeric(players["minutes"], errors="coerce").fillna(24.0)

    players = prepare_players_data(players, overwrite_roles=False)

    for col in ["player_id", "team_id", "games_played"]:
        if col in players.columns:
            players[col] = pd.to_numeric(players[col], errors="coerce").fillna(0).astype(int)

    if "team_id" not in teams.columns:
        team_id_map = (
            players.dropna(subset=["team", "team_id"])
            .groupby("team")["team_id"]
            .agg(lambda values: values.mode().iloc[0])
            .to_dict()
        )
        teams["team_id"] = teams["team"].map(team_id_map)

    teams["team_id"] = pd.to_numeric(teams["team_id"], errors="coerce")
    missing = teams["team_id"].isna()
    if missing.any():
        max_id = pd.to_numeric(players.get("team_id", pd.Series([0])), errors="coerce").max()
        if pd.isna(max_id):
            max_id = 0
        teams.loc[missing, "team_id"] = pd.factorize(teams.loc[missing, "team"].astype(str))[0] + int(max_id) + 1
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
    return players, teams, matchups


def prepare_app_roster_data(app_roster: pd.DataFrame) -> pd.DataFrame:
    app_roster = _clean_text_columns(app_roster)
    for col in ["player_id", "latest_team_id", "latest_season", "games_played", "minutes", "points", "activity_score", "recent_minutes", "trend_score"]:
        if col in app_roster.columns:
            app_roster[col] = pd.to_numeric(app_roster[col], errors="coerce")
    if "low_activity_flag" in app_roster.columns:
        app_roster["low_activity_flag"] = app_roster["low_activity_flag"].astype(bool)
    app_roster["player_id"] = app_roster["player_id"].astype(int)
    app_roster["latest_team"] = app_roster["latest_team"].fillna("").astype(str).str.strip()
    return app_roster


def load_processed_data(processed_dir: str | Path = "data/processed") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compatibilidad: carga processed_players.csv si un script viejo lo necesita."""
    processed_dir = Path(processed_dir)
    players_path = processed_dir / "processed_players.csv"
    teams_path = processed_dir / "processed_teams.csv"
    matchups_path = processed_dir / "processed_matchups.csv"
    for path in [players_path, teams_path, matchups_path]:
        if not path.exists():
            raise FileNotFoundError(f"No se encontró: {path}")
    players = pd.read_csv(players_path)
    teams = pd.read_csv(teams_path)
    matchups = pd.read_csv(matchups_path)
    _validate_columns(players, PLAYER_REQUIRED_COLUMNS, "processed_players.csv")
    _validate_columns(teams, TEAM_REQUIRED_COLUMNS, "processed_teams.csv")
    _validate_columns(matchups, MATCHUP_REQUIRED_COLUMNS, "processed_matchups.csv")
    return prepare_runtime_data(players, teams, matchups)


def load_recommendation_data(processed_dir: str | Path = "data/processed") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    processed_dir = Path(processed_dir)
    profiles_path = processed_dir / "player_profiles.csv"
    app_roster_path = processed_dir / "app_roster.csv"
    teams_path = processed_dir / "processed_teams.csv"
    matchups_path = processed_dir / "processed_matchups.csv"
    for path in [profiles_path, app_roster_path, teams_path, matchups_path]:
        if not path.exists():
            raise FileNotFoundError(f"No se encontró: {path}. Ejecuta python src/preprocessing.py")

    players = pd.read_csv(profiles_path)
    app_roster = pd.read_csv(app_roster_path)
    teams = pd.read_csv(teams_path)
    matchups = pd.read_csv(matchups_path)

    _validate_columns(players, PLAYER_REQUIRED_COLUMNS, "player_profiles.csv")
    _validate_columns(app_roster, APP_ROSTER_REQUIRED_COLUMNS, "app_roster.csv")
    _validate_columns(teams, TEAM_REQUIRED_COLUMNS, "processed_teams.csv")
    _validate_columns(matchups, MATCHUP_REQUIRED_COLUMNS, "processed_matchups.csv")

    players, teams, matchups = prepare_runtime_data(players, teams, matchups)
    app_roster = prepare_app_roster_data(app_roster)
    return players, app_roster, teams, matchups


# ============================================================
# Resolución
# ============================================================


def resolve_team_row(teams: pd.DataFrame, team_value: str | int) -> pd.Series:
    value_text = _normalize_text(team_value)
    if str(team_value).strip().isdigit() and "team_id" in teams.columns:
        match = teams[pd.to_numeric(teams["team_id"], errors="coerce") == int(team_value)]
        if not match.empty:
            return match.iloc[0]
    aliases = _team_aliases(team_value)
    team_text = teams["team"].astype(str).map(_normalize_text)
    exact = teams[team_text.isin(aliases)]
    if not exact.empty:
        return exact.iloc[0]
    partial = teams[team_text.str.contains(value_text, na=False, regex=False)]
    if not partial.empty:
        return partial.iloc[0]
    raise ValueError(f"No se encontró el equipo: {team_value}. Equipos disponibles: {teams['team'].dropna().drop_duplicates().head(30).to_list()}")


def resolve_player_by_id(players: pd.DataFrame, player_id: int) -> pd.Series:
    match = players[pd.to_numeric(players["player_id"], errors="coerce") == int(player_id)]
    if match.empty:
        raise ValueError(f"No se encontró player_id={player_id} en player_profiles.csv.")
    return match.iloc[0]


def resolve_player_row(players: pd.DataFrame, player_value: str | int, selected_team_id: Optional[int] = None, selected_team_name: Optional[str] = None) -> pd.Series:
    search = players.copy()
    team_mask = _same_team_mask(search, selected_team_id, selected_team_name)
    if team_mask.any():
        search = search[team_mask].copy()
    if str(player_value).strip().isdigit():
        match = search[pd.to_numeric(search["player_id"], errors="coerce") == int(player_value)]
        if not match.empty:
            return match.iloc[0]
    value_text = _normalize_text(player_value)
    player_text = search["player_name"].astype(str).map(_normalize_text)
    exact = search[player_text == value_text]
    if not exact.empty:
        return exact.iloc[0]
    partial = search[player_text.str.contains(value_text, na=False, regex=False)]
    if not partial.empty:
        return partial.iloc[0]
    raise ValueError(f"No se encontró el jugador: {player_value}.")


def resolve_matchup_row(matchups: pd.DataFrame, opponent_team_id: Optional[int], opponent_team_name: str) -> pd.Series:
    candidates = matchups.copy()
    if opponent_team_id is not None and "team_id" in candidates.columns:
        by_id = candidates[pd.to_numeric(candidates["team_id"], errors="coerce") == opponent_team_id]
        if not by_id.empty:
            return by_id.iloc[0]
    aliases = _team_aliases(opponent_team_name)
    by_team = candidates[candidates["team"].astype(str).map(_normalize_text).isin(aliases)]
    if not by_team.empty:
        return by_team.iloc[0]
    raise ValueError(f"No se encontró matchup para el rival: {opponent_team_name}.")


def get_team_players(players: pd.DataFrame, team_id: Optional[int], team_name: str) -> pd.DataFrame:
    return players[_same_team_mask(players, team_id=team_id, team_name=team_name)].copy()


def get_app_roster_for_team(app_roster: pd.DataFrame, selected_team_name: str) -> pd.DataFrame:
    aliases = _team_aliases(selected_team_name)
    team_rows = app_roster[app_roster["latest_team"].astype(str).map(_normalize_text).isin(aliases)].copy()
    return team_rows.reset_index(drop=True)


def validate_lineup(current_roster: pd.DataFrame, lineup_player_ids: list[int], replaced_player_id: int) -> None:
    if len(lineup_player_ids) != 5 or len(set(lineup_player_ids)) != 5:
        raise ValueError("lineup_player_ids debe contener exactamente 5 jugadores únicos.")
    if int(replaced_player_id) not in {int(x) for x in lineup_player_ids}:
        raise ValueError("replaced_player_id debe estar dentro de lineup_player_ids.")
    roster_ids = set(pd.to_numeric(current_roster["player_id"], errors="coerce").dropna().astype(int).tolist())
    invalid = [int(pid) for pid in lineup_player_ids if int(pid) not in roster_ids]
    if invalid:
        raise ValueError(f"Estos player_id de la quinteta no pertenecen al equipo seleccionado según app_roster.csv: {invalid}")


def build_bench_candidates(
    player_profiles: pd.DataFrame,
    app_roster: pd.DataFrame,
    selected_team_name: str,
    lineup_player_ids: Optional[list[int]],
    replaced_player_id: int,
    min_games: Optional[int],
    min_minutes: Optional[float],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    current_roster = get_app_roster_for_team(app_roster, selected_team_name)
    if current_roster.empty:
        raise ValueError(f"No hay jugadores para {selected_team_name} en app_roster.csv.")

    if lineup_player_ids is not None:
        validate_lineup(current_roster, lineup_player_ids, int(replaced_player_id))
        exclude_ids = {int(x) for x in lineup_player_ids}
    else:
        # Compatibilidad temporal: si no hay quinteta, usamos todo el roster menos el reemplazado.
        exclude_ids = {int(replaced_player_id)}

    bench_roster = current_roster[~current_roster["player_id"].astype(int).isin(exclude_ids)].copy()
    bench_roster = bench_roster[bench_roster["player_id"].astype(int) != int(replaced_player_id)].copy()

    bench_ids = set(bench_roster["player_id"].astype(int).tolist())
    candidates = player_profiles[player_profiles["player_id"].astype(int).isin(bench_ids)].copy()
    candidates = candidates[candidates["player_id"].astype(int) != int(replaced_player_id)].copy()

    if min_games is not None:
        candidates = candidates[pd.to_numeric(candidates["games_played"], errors="coerce") >= int(min_games)]
    if min_minutes is not None:
        candidates = candidates[pd.to_numeric(candidates["minutes"], errors="coerce") >= float(min_minutes)]

    if candidates.empty:
        raise ValueError(
            f"No hay candidatos válidos en la banca después de aplicar filtros. "
            f"Equipo={selected_team_name}, min_games={min_games}, min_minutes={min_minutes}. Prueba bajar los filtros."
        )
    return current_roster.reset_index(drop=True), bench_roster.reset_index(drop=True), candidates.reset_index(drop=True)


# ============================================================
# Scoring
# ============================================================


def calculate_recency_adjusted_score(replacement_score: float, estimated_net_impact: float, activity_score: float, trend_score: float) -> float:
    """
    Score final conservador.

    Mantiene el replacement_score del modelo anterior, pero penaliza baja actividad reciente
    y suma una señal pequeña de tendencia.
    """
    replacement_score = np.clip(float(replacement_score), 0.0, 1.0)
    activity_score = np.clip(float(activity_score), 0.0, 1.0)
    trend_score = np.clip(float(trend_score), -0.25, 0.25)
    # estimated_net_impact se convierte a una señal suave 0-1 para no dominar.
    impact_component = np.clip((float(estimated_net_impact) + 8.0) / 16.0, 0.0, 1.0)
    base_score = 0.80 * replacement_score + 0.10 * impact_component + 0.05 * activity_score + 0.05 * (0.5 + trend_score)
    final_score = base_score * (0.60 + 0.40 * activity_score)
    return float(np.clip(final_score, 0.0, 1.0))


def score_candidate(candidate: pd.Series, replaced_player: pd.Series, selected_team: pd.Series, opponent_team: pd.Series, opponent_matchup: pd.Series, team_need: Optional[dict[str, float]] = None, opponent_context: Optional[dict[str, float]] = None) -> dict:
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

    activity_score = _safe_float(candidate.get("activity_score", 1.0), 1.0)
    trend_score = _safe_float(candidate.get("trend_score", 0.0), 0.0)
    estimated_net_impact = _safe_float(impact_data.get("estimated_net_impact", impact_data.get("net_impact")))
    replacement_score = _safe_float(score_data.get("final_score"))
    recommendation_score = calculate_recency_adjusted_score(
        replacement_score=replacement_score,
        estimated_net_impact=estimated_net_impact,
        activity_score=activity_score,
        trend_score=trend_score,
    )

    return {
        "player_id": _safe_int(candidate.get("player_id")),
        "player_name": str(candidate.get("player_name", "")),
        "team_id": _safe_int(candidate.get("team_id")),
        "team": str(candidate.get("team", "")),
        "latest_team": str(candidate.get("latest_team", candidate.get("team", ""))),
        "position": str(candidate.get("position", "")),
        "position_group": str(candidate.get("position_group", "")),
        "games_played": _safe_int(candidate.get("games_played")),
        "minutes": _safe_float(candidate.get("minutes")),
        "points": _safe_float(candidate.get("points")),
        "recent_season": _safe_int(candidate.get("recent_season")),
        "recent_minutes": _safe_float(candidate.get("recent_minutes")),
        "activity_score": activity_score,
        "trend_score": trend_score,
        "low_activity_flag": bool(candidate.get("low_activity_flag", False)),
        "role_similarity": _safe_float(score_data.get("role_similarity")),
        "position_fit": _safe_float(score_data.get("position_fit")),
        "team_fit": _safe_float(score_data.get("team_fit")),
        "opponent_fit": _safe_float(score_data.get("opponent_fit")),
        "replacement_score": replacement_score,
        "recommendation_score": recommendation_score,
        "offensive_impact": _safe_float(impact_data.get("offensive_impact")),
        "defensive_impact": _safe_float(impact_data.get("defensive_impact")),
        "pace_impact": _safe_float(impact_data.get("pace_impact")),
        "estimated_net_impact": estimated_net_impact,
        "candidate_value": _safe_float(impact_data.get("candidate_value")),
        "replaced_value": _safe_float(impact_data.get("replaced_value")),
        "raw_net_delta": _safe_float(impact_data.get("raw_net_delta")),
        "impact_mode": str(impact_data.get("impact_mode", "relative")),
        "fit_multiplier": _safe_float(impact_data.get("fit_multiplier")),
        "comparison_minutes": _safe_float(impact_data.get("comparison_minutes")),
        "cap_applied": bool(impact_data.get("cap_applied", False)),
    }


def simulate_replacement(selected_team: pd.Series, opponent_team: pd.Series, player_impact: dict, num_simulations: int, use_baseline: bool = False) -> dict:
    raw = monte_carlo.monte_carlo_replacement_analysis(
        team_context=_to_dict(selected_team),
        opponent_context=_to_dict(opponent_team),
        player_impact=player_impact,
        n_simulations=num_simulations,
    )
    scenario_key = "without_replacement" if use_baseline else "with_replacement"
    probability_key = "win_probability_without" if use_baseline else "win_probability_with"
    scenario = raw.get(scenario_key, {})
    return {
        "win_probability": _safe_float(raw.get(probability_key)),
        "expected_margin": _safe_float(scenario.get("expected_margin")),
        "expected_team_points": _safe_float(scenario.get("expected_team_points")),
        "expected_opponent_points": _safe_float(scenario.get("expected_opponent_points")),
        "margins": _to_list(scenario.get("margins")),
        "team_scores": _to_list(scenario.get("team_scores")),
        "opponent_scores": _to_list(scenario.get("opponent_scores")),
        "raw": raw,
    }


def build_candidate_explanation(candidate_result: dict, candidate_simulation: dict) -> str:
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
    if candidate_result.get("low_activity_flag"):
        explanation += " Advertencia: este jugador tiene poca actividad reciente; la recomendación es menos confiable."
    return explanation


def build_text_explanation(replaced_player: pd.Series, selected_team: pd.Series, opponent_team: pd.Series, top_replacements: list[dict], baseline_win_probability: float) -> str:
    summary_ready = []
    for result in top_replacements:
        copied = dict(result)
        copied["final_score"] = copied.get("recommendation_score", copied.get("replacement_score", 0.0))
        summary_ready.append(copied)
    baseline = (
        f"Escenario: reemplazar a {replaced_player.get('player_name', 'N/A')} "
        f"en {selected_team.get('team', 'N/A')} contra {opponent_team.get('team', 'N/A')}. "
        f"La probabilidad simulada de ganar sin reemplazo es {baseline_win_probability * 100:.1f}%."
    )
    return f"{baseline}\n\n{explanations.summarize_top_candidates(summary_ready, top_n=len(summary_ready))}"


# ============================================================
# Flujo principal
# ============================================================


def recommend_replacements(
    selected_team_value: str | int,
    opponent_team_value: str | int,
    lineup_player_ids: Optional[list[int]] = None,
    replaced_player_id: Optional[int] = None,
    player_to_replace_value: Optional[str | int] = None,
    num_simulations: int = 1000,
    monte_carlo_simulations: Optional[int] = None,
    processed_dir: str | Path = "data/processed",
    top_n: int = 3,
    min_minutes: Optional[float] = 10.0,
    min_games: Optional[int] = 10,
    exclude_opponent_players: bool = True,
    random_state: Optional[int] = 42,
) -> dict:
    _ = random_state
    if monte_carlo_simulations is not None:
        num_simulations = int(monte_carlo_simulations)

    players, app_roster, teams, matchups = load_recommendation_data(processed_dir)

    selected_team = resolve_team_row(teams, selected_team_value)
    opponent_team = resolve_team_row(teams, opponent_team_value)
    selected_team_id = _safe_int(selected_team.get("team_id"))
    opponent_team_id = _safe_int(opponent_team.get("team_id"))
    selected_team_name = str(selected_team.get("team", selected_team_value))
    opponent_team_name = str(opponent_team.get("team", opponent_team_value))

    if replaced_player_id is None:
        if player_to_replace_value is None:
            raise ValueError("Debes enviar replaced_player_id o player_to_replace_value.")
        replaced_player = resolve_player_row(players, player_to_replace_value, selected_team_id, selected_team_name)
        replaced_player_id = _safe_int(replaced_player.get("player_id"))
    else:
        replaced_player = resolve_player_by_id(players, int(replaced_player_id))

    current_roster, bench_roster, candidates = build_bench_candidates(
        player_profiles=players,
        app_roster=app_roster,
        selected_team_name=selected_team_name,
        lineup_player_ids=lineup_player_ids,
        replaced_player_id=int(replaced_player_id),
        min_games=min_games,
        min_minutes=min_minutes,
    )

    opponent_matchup = resolve_matchup_row(matchups, opponent_team_id, opponent_team_name)

    selected_team_profile_ids = set(current_roster["player_id"].astype(int).tolist())
    selected_team_players = players[players["player_id"].astype(int).isin(selected_team_profile_ids)].copy()
    opponent_roster = get_app_roster_for_team(app_roster, opponent_team_name)
    opponent_profile_ids = set(opponent_roster["player_id"].astype(int).tolist())
    opponent_players = players[players["player_id"].astype(int).isin(opponent_profile_ids)].copy()

    team_need = similarity.build_team_need(selected_team_players) if not selected_team_players.empty else None
    if not opponent_players.empty:
        opponent_context = similarity.build_opponent_context_from_players(opponent_players)
    else:
        opponent_context = similarity.build_opponent_context_from_matchup(_to_dict(opponent_matchup))

    zero_impact = {"offensive_impact": 0.0, "defensive_impact": 0.0, "pace_impact": 0.0, "estimated_net_impact": 0.0, "net_impact": 0.0}
    baseline_simulation = simulate_replacement(selected_team, opponent_team, zero_impact, num_simulations, use_baseline=True)
    baseline_win_probability = baseline_simulation["win_probability"]

    scored_candidates: list[dict] = []
    for _, candidate in candidates.iterrows():
        candidate_result = score_candidate(candidate, replaced_player, selected_team, opponent_team, opponent_matchup, team_need, opponent_context)
        player_impact = {
            "offensive_impact": candidate_result["offensive_impact"],
            "defensive_impact": candidate_result["defensive_impact"],
            "pace_impact": candidate_result["pace_impact"],
            "estimated_net_impact": candidate_result["estimated_net_impact"],
            "net_impact": candidate_result["estimated_net_impact"],
        }
        candidate_simulation = simulate_replacement(selected_team, opponent_team, player_impact, num_simulations, use_baseline=False)
        candidate_result["win_probability_with_replacement"] = candidate_simulation["win_probability"]
        candidate_result["win_probability_delta"] = candidate_result["win_probability_with_replacement"] - baseline_win_probability
        candidate_result["expected_margin_with_replacement"] = candidate_simulation["expected_margin"]
        candidate_result["simulation_distribution"] = {
            "margins": candidate_simulation["margins"],
            "team_scores": candidate_simulation["team_scores"],
            "opponent_scores": candidate_simulation["opponent_scores"],
        }
        candidate_result["explanation"] = build_candidate_explanation(candidate_result, candidate_simulation)
        scored_candidates.append(candidate_result)

    results_df = pd.DataFrame(scored_candidates)
    if results_df.empty:
        raise ValueError("No se pudo calcular score para ningún candidato.")

    results_df = results_df.sort_values(
        by=["recommendation_score", "win_probability_with_replacement", "replacement_score", "estimated_net_impact"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    top_replacements = results_df.head(int(top_n)).to_dict(orient="records")

    explanation = build_text_explanation(replaced_player, selected_team, opponent_team, top_replacements, baseline_win_probability)

    return {
        "selected_team": _to_dict(selected_team),
        "opponent_team": _to_dict(opponent_team),
        "replaced_player": _to_dict(replaced_player),
        "lineup_player_ids": [int(x) for x in lineup_player_ids] if lineup_player_ids is not None else None,
        "num_simulations": num_simulations,
        "filters": {"min_minutes": min_minutes, "min_games": min_games, "deduplicate_players": True, "recency_adjustment": True},
        "roster_debug": {
            "current_roster_size": int(len(current_roster)),
            "bench_size_before_filters": int(len(bench_roster)),
            "candidate_size_after_filters": int(len(candidates)),
        },
        "contexts": {"team_need": team_need, "opponent_context": opponent_context},
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


if __name__ == "__main__":
    result = recommend_replacements(
        selected_team_value="LAL",
        opponent_team_value="BOS",
        lineup_player_ids=[2544, 203076, 1630559, 203471, 1629022],
        replaced_player_id=2544,
        num_simulations=1000,
        processed_dir="data/processed",
        top_n=3,
        min_minutes=10,
        min_games=10,
    )
    print("Probabilidad sin reemplazo:", result["baseline"]["win_probability_without_replacement"])
    print("Debug roster:", result["roster_debug"])
    for i, player in enumerate(result["top_replacements"], start=1):
        print(f"{i}. {player['player_name']} - {player['latest_team']} | rec_score={player['recommendation_score']:.3f} | activity={player['activity_score']:.2f} | trend={player['trend_score']:.2f}")
