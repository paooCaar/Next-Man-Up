"""
src/roster.py

Construye las tablas de roster y perfiles para NBA Next-Man-Up.

Tablas generadas:
- latest_roster.csv: último equipo conocido por jugador en toda la base histórica.
- player_season_profiles.csv: una fila por jugador-temporada.
- player_profiles.csv: una fila por jugador con estadísticas ponderadas por recencia.
- app_roster.csv: roster usable por la app, solo la temporada más reciente global.

Idea central:
- app_roster decide quién puede aparecer en la interfaz y en la banca.
- player_profiles calcula similitud, impacto y ranking usando perfil histórico ponderado.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    from .role_features import prepare_players_data
except ImportError:
    from role_features import prepare_players_data


UNKNOWN_POSITION_VALUES = {"", "UNK", "UNKNOWN", "NAN", "NONE", "NA", "N/A", "0"}

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

RECENCY_PROFILE_COLUMNS = [
    "recent_season",
    "recent_minutes",
    "previous_season",
    "previous_minutes",
    "activity_score",
    "low_activity_flag",
    "trend_score",
    "recent_impact_per36",
    "previous_impact_per36",
    "recency_weighted_minutes_total",
]

METRIC_COLUMNS = [
    "minutes",
    "points",
    "assists",
    "rebounds",
    "steals",
    "blocks",
    "turnovers",
    "three_attempts",
    "usage_rate",
    "offensive_rating",
    "defensive_rating",
    "pace",
    "plus_minus",
]


# ---------------------------------------------------------------------
# Limpieza y helpers
# ---------------------------------------------------------------------


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
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
        "trb": "rebounds",
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
    df = df.copy()
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].apply(lambda value: value.strip() if isinstance(value, str) else value)
    return df


def parse_minutes_value(value) -> float:
    if pd.isna(value):
        return 0.0
    if isinstance(value, (int, float)):
        return float(value) if np.isfinite(value) else 0.0
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
    if pd.isna(value):
        return "UNK"
    pos = str(value).strip().upper().replace(" ", "").replace("_", "-")
    if pos in UNKNOWN_POSITION_VALUES:
        return "UNK"
    if pos in ["PG", "SG", "G", "GUARD"]:
        return "G"
    if pos in ["SF", "PF", "F", "FORWARD"]:
        return "F"
    if pos in ["C", "CENTER"]:
        return "C"
    if pos in ["G-F", "F-G"]:
        return "F"
    if pos in ["F-C", "C-F"]:
        return "F"
    if "C" in pos and "G" not in pos:
        return "C"
    if "G" in pos and "C" not in pos:
        return "G"
    if "F" in pos:
        return "F"
    return "UNK"


def get_mode_text(series: pd.Series, default_value: str = "UNK") -> str:
    cleaned = series.dropna().astype(str).str.strip()
    cleaned = cleaned[~cleaned.str.upper().isin(UNKNOWN_POSITION_VALUES)]
    if cleaned.empty:
        return default_value
    return str(cleaned.mode().iloc[0])


def get_latest_text(series: pd.Series, default_value: str = "") -> str:
    cleaned = series.dropna().astype(str).str.strip()
    cleaned = cleaned[cleaned != ""]
    if cleaned.empty:
        return default_value
    return str(cleaned.iloc[-1])


def safe_divide(numerator, denominator, default: float = 0.0) -> float:
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
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _ensure_numeric_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    df = df.copy()
    for col in columns:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return df


def _weighted_mean(values: pd.Series, weights: pd.Series, default: float = 0.0) -> float:
    values = pd.to_numeric(values, errors="coerce")
    weights = pd.to_numeric(weights, errors="coerce")
    mask = values.notna() & weights.notna() & (weights > 0)
    if not mask.any():
        return default
    return float(np.average(values[mask], weights=weights[mask]))


def _weighted_ratio(numerators: pd.Series, denominators: pd.Series, weights: pd.Series, default: float = 0.0) -> float:
    numerators = pd.to_numeric(numerators, errors="coerce").fillna(0.0)
    denominators = pd.to_numeric(denominators, errors="coerce").fillna(0.0)
    weights = pd.to_numeric(weights, errors="coerce").fillna(0.0)
    den = float((denominators * weights).sum())
    if den <= 0:
        return default
    return float((numerators * weights).sum() / den)


def build_team_id_to_abbreviation_map(teams_df: Optional[pd.DataFrame]) -> dict[int, str]:
    if teams_df is None or teams_df.empty:
        return {}
    teams = clean_text_columns(clean_column_names(teams_df))
    if "team_id" not in teams.columns:
        return {}
    label_col = first_existing_column(teams, ["team", "nickname", "city"])
    if label_col is None:
        return {}
    mapping: dict[int, str] = {}
    for _, row in teams.iterrows():
        team_id = pd.to_numeric(row.get("team_id"), errors="coerce")
        label = row.get(label_col)
        if pd.notna(team_id) and pd.notna(label):
            mapping[int(team_id)] = str(label).strip()
    return mapping


# ---------------------------------------------------------------------
# Base por jugador-partido
# ---------------------------------------------------------------------


def prepare_player_game_rows(
    games_details_df: pd.DataFrame,
    games_df: pd.DataFrame,
    teams_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    details = clean_text_columns(clean_column_names(games_details_df))
    games = clean_text_columns(clean_column_names(games_df))

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

    merged = merged.dropna(subset=["player_id", "game_id", "team_id", "season"])
    merged = merged[merged["team"].notna()].copy()
    merged["team"] = merged["team"].astype(str).str.strip()
    merged = merged[merged["team"] != ""].copy()
    merged["player_id"] = merged["player_id"].astype(int)
    merged["team_id"] = merged["team_id"].astype(int)
    merged["season"] = merged["season"].astype(int)
    return merged.reset_index(drop=True)


# ---------------------------------------------------------------------
# latest_roster.csv
# ---------------------------------------------------------------------


def build_latest_roster(
    games_details_df: pd.DataFrame,
    games_df: pd.DataFrame,
    teams_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    rows = prepare_player_game_rows(games_details_df, games_df, teams_df)
    if rows.empty:
        raise ValueError("No hay filas válidas para construir latest_roster.csv.")

    position_lookup = (
        rows[rows["position"] != "UNK"]
        .groupby("player_id")["position"]
        .agg(lambda values: get_mode_text(values, default_value="UNK"))
        .to_dict()
    )

    rows = rows.sort_values(
        ["player_id", "game_date", "game_id_numeric", "has_minutes"],
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

    latest_roster = pd.DataFrame(
        {
            "player_id": latest["player_id"].astype(int),
            "player_name": latest["player_name"].astype(str).str.strip(),
            "latest_team_id": latest["team_id"].astype(int),
            "latest_team": latest["team"].astype(str).str.strip(),
            "latest_season": pd.to_numeric(latest["season"], errors="coerce").astype("Int64"),
            "latest_game_date": latest["game_date"].dt.strftime("%Y-%m-%d"),
            "latest_game_id": pd.to_numeric(latest["game_id"], errors="coerce").astype("Int64"),
            "latest_position": latest["latest_position"].apply(normalize_position_label),
            "latest_has_minutes": latest["has_minutes"].astype(bool),
        }
    )
    latest_roster = latest_roster.dropna(subset=["latest_team", "latest_game_date"])
    latest_roster = latest_roster[latest_roster["latest_team"].astype(str).str.strip() != ""]
    latest_roster = latest_roster.sort_values("player_name").reset_index(drop=True)
    return latest_roster[LATEST_ROSTER_COLUMNS]


# ---------------------------------------------------------------------
# player_season_profiles.csv
# ---------------------------------------------------------------------


def _build_team_possessions(player_game_rows: pd.DataFrame) -> pd.DataFrame:
    rows = _ensure_numeric_columns(player_game_rows, ["fga", "fta", "oreb", "turnovers"])
    team_game = rows.groupby(["game_id", "team_id"], as_index=False).agg(
        team_fga=("fga", "sum"),
        team_fta=("fta", "sum"),
        team_oreb=("oreb", "sum"),
        team_turnovers=("turnovers", "sum"),
    )
    team_game["team_possessions"] = (
        team_game["team_fga"] + 0.44 * team_game["team_fta"] - team_game["team_oreb"] + team_game["team_turnovers"]
    )
    team_game.loc[team_game["team_possessions"] <= 0, "team_possessions"] = np.nan
    median_possessions = team_game["team_possessions"].median(skipna=True)
    if pd.isna(median_possessions):
        median_possessions = 100.0
    team_game["team_possessions"] = team_game["team_possessions"].fillna(median_possessions)
    return team_game[["game_id", "team_id", "team_possessions"]]


def build_player_season_profiles(
    games_details_df: pd.DataFrame,
    games_df: pd.DataFrame,
    teams_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    rows = prepare_player_game_rows(games_details_df, games_df, teams_df)
    if rows.empty:
        raise ValueError("No hay filas válidas para construir player_season_profiles.csv.")

    numeric_columns = [
        "minutes", "points", "assists", "rebounds", "steals", "blocks", "turnovers",
        "fgm", "fga", "three_made", "three_attempts", "ftm", "fta", "oreb", "dreb", "plus_minus",
    ]
    rows = _ensure_numeric_columns(rows, numeric_columns)
    team_possessions = _build_team_possessions(rows)
    rows = rows.merge(team_possessions, on=["game_id", "team_id"], how="left")

    played = rows[rows["minutes"] > 0].copy()
    if played.empty:
        raise ValueError("No hay filas con minutos reales para construir perfiles por temporada.")

    played["usage_numerator"] = played["fga"] + 0.44 * played["fta"] + played["turnovers"]
    played["fg_pct_row"] = np.where(played["fga"] > 0, played["fgm"] / played["fga"], 0.0)
    played["three_pct_row"] = np.where(
        played["three_attempts"] > 0,
        played["three_made"] / played["three_attempts"],
        0.0,
    )
    played["offensive_rating_row"] = (
        100
        + 1.2 * played["points"]
        + 0.8 * played["assists"]
        - 1.5 * played["turnovers"]
        + 8.0 * played["fg_pct_row"]
        + 6.0 * played["three_pct_row"]
    )
    played["defensive_rating_row"] = (
        115
        - 1.5 * played["steals"]
        - 1.3 * played["blocks"]
        - 0.4 * played["rebounds"]
        - 0.15 * played["plus_minus"]
    )

    played = played.sort_values(["player_id", "season", "game_date", "game_id_numeric"])

    grouped = played.groupby(["player_id", "season"], as_index=False).agg(
        player_name=("player_name", lambda values: get_latest_text(values, "")),
        team_id=("team_id", lambda values: int(pd.Series(values).mode().iloc[0])),
        team=("team", lambda values: get_latest_text(values, "")),
        position=("position", lambda values: get_mode_text(values, "UNK")),
        games_played=("game_id", "nunique"),
        season_minutes_total=("minutes", "sum"),
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

    grouped["fg_pct"] = grouped.apply(lambda r: safe_divide(r["fgm_total"], r["fga_total"]), axis=1)
    grouped["three_pct"] = grouped.apply(lambda r: safe_divide(r["three_made_total"], r["three_attempts_total"]), axis=1)
    grouped["usage_rate"] = grouped.apply(lambda r: safe_divide(r["usage_numerator_total"], r["team_possessions_total"]), axis=1)
    grouped["usage_rate"] = grouped["usage_rate"].clip(0, 1)

    simple_impact = (
        grouped["points"]
        + 0.7 * grouped["rebounds"]
        + 0.7 * grouped["assists"]
        + 1.5 * grouped["steals"]
        + 1.5 * grouped["blocks"]
        - grouped["turnovers"]
    )
    grouped["impact_per36"] = np.where(
        grouped["minutes"] > 0,
        simple_impact / grouped["minutes"] * 36.0,
        0.0,
    )

    grouped["player_id"] = grouped["player_id"].astype(int)
    grouped["season"] = grouped["season"].astype(int)
    grouped["position"] = grouped["position"].apply(normalize_position_label)

    return grouped.sort_values(["player_name", "season"]).reset_index(drop=True)


# ---------------------------------------------------------------------
# player_profiles.csv ponderado por recencia
# ---------------------------------------------------------------------


def _infer_position_from_profile(row: pd.Series) -> str:
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
    if assists_per36 >= 3.5 and three_attempts_per36 >= 4.0 and rebounds_per36 <= 5.5 and blocks_per36 <= 0.7:
        return "G"
    return "F"


def build_player_profiles(
    games_details_df: pd.DataFrame,
    games_df: pd.DataFrame,
    teams_df: Optional[pd.DataFrame] = None,
    latest_roster_df: Optional[pd.DataFrame] = None,
    player_season_profiles_df: Optional[pd.DataFrame] = None,
    lambda_decay: float = 0.65,
    activity_reference_minutes: float = 800.0,
    min_recent_minutes_warning: float = 200.0,
) -> pd.DataFrame:
    if player_season_profiles_df is None:
        season_profiles = build_player_season_profiles(games_details_df, games_df, teams_df)
    else:
        season_profiles = player_season_profiles_df.copy()

    if latest_roster_df is None:
        latest_roster_df = build_latest_roster(games_details_df, games_df, teams_df)

    if season_profiles.empty:
        raise ValueError("No hay perfiles por temporada para construir player_profiles.csv.")

    global_latest_season = int(pd.to_numeric(season_profiles["season"], errors="coerce").max())
    season_profiles["season_gap"] = global_latest_season - pd.to_numeric(season_profiles["season"], errors="coerce")
    season_profiles["season_weight"] = np.exp(-float(lambda_decay) * season_profiles["season_gap"].clip(lower=0))
    season_profiles["recency_weight"] = season_profiles["season_weight"] * np.sqrt(
        pd.to_numeric(season_profiles["season_minutes_total"], errors="coerce").fillna(0).clip(lower=0)
    )
    # Si un jugador tiene muy pocos minutos en todas sus temporadas, evitamos peso cero.
    season_profiles.loc[season_profiles["recency_weight"] <= 0, "recency_weight"] = 1e-6

    output_rows: list[dict] = []

    for player_id, group in season_profiles.groupby("player_id"):
        group = group.sort_values("season")
        latest_row = group.iloc[-1]
        previous_row = group.iloc[-2] if len(group) >= 2 else None
        weights = group["recency_weight"]

        profile: dict = {
            "player_id": int(player_id),
            "player_name": get_latest_text(group["player_name"], ""),
            "games_played": int(group["games_played"].sum()),
            "position": get_mode_text(group["position"], "UNK"),
            "recent_season": int(latest_row["season"]),
            "recent_minutes": float(latest_row["season_minutes_total"]),
            "previous_season": int(previous_row["season"]) if previous_row is not None else np.nan,
            "previous_minutes": float(previous_row["season_minutes_total"]) if previous_row is not None else 0.0,
            "recent_impact_per36": float(latest_row.get("impact_per36", 0.0)),
            "previous_impact_per36": float(previous_row.get("impact_per36", 0.0)) if previous_row is not None else 0.0,
        }

        for metric in METRIC_COLUMNS:
            profile[metric] = _weighted_mean(group[metric], weights, default=0.0)

        profile["fg_pct"] = _weighted_ratio(group["fgm_total"], group["fga_total"], weights, default=0.0)
        profile["three_pct"] = _weighted_ratio(
            group["three_made_total"], group["three_attempts_total"], weights, default=0.0
        )

        profile["activity_score"] = float(np.clip(profile["recent_minutes"] / activity_reference_minutes, 0.0, 1.0))
        profile["low_activity_flag"] = bool(profile["recent_minutes"] < min_recent_minutes_warning)

        if previous_row is None or float(previous_row.get("season_minutes_total", 0.0)) <= 0:
            trend_score = 0.0
        else:
            previous_impact = float(previous_row.get("impact_per36", 0.0))
            recent_impact = float(latest_row.get("impact_per36", 0.0))
            trend = (recent_impact - previous_impact) / (abs(previous_impact) + 1e-6)
            trend_score = float(np.clip(trend, -0.25, 0.25))
        profile["trend_score"] = trend_score
        profile["recency_weighted_minutes_total"] = float((group["season_minutes_total"] * weights).sum())

        output_rows.append(profile)

    profiles = pd.DataFrame(output_rows)

    latest = latest_roster_df.copy()
    latest["player_id"] = pd.to_numeric(latest["player_id"], errors="coerce")
    latest = latest.dropna(subset=["player_id"]).copy()
    latest["player_id"] = latest["player_id"].astype(int)

    profiles = profiles.merge(
        latest[
            [
                "player_id", "latest_team_id", "latest_team", "latest_season",
                "latest_game_date", "latest_game_id", "latest_position",
            ]
        ],
        on="player_id",
        how="left",
    )

    profiles["team_id"] = pd.to_numeric(profiles["latest_team_id"], errors="coerce").fillna(0).astype(int)
    profiles["team"] = profiles["latest_team"].fillna("").astype(str)
    profiles["position"] = profiles.apply(_infer_position_from_profile, axis=1)

    profiles = prepare_players_data(profiles, overwrite_roles=True)

    for col in PLAYER_PROFILE_BASE_COLUMNS + RECENCY_PROFILE_COLUMNS:
        if col not in profiles.columns:
            if col in ["player_name", "team", "position"]:
                profiles[col] = ""
            elif col == "low_activity_flag":
                profiles[col] = False
            else:
                profiles[col] = 0.0

    metadata_columns = [
        "latest_team_id", "latest_team", "latest_season", "latest_game_date", "latest_game_id", "latest_position",
    ]
    ordered_columns = PLAYER_PROFILE_BASE_COLUMNS + metadata_columns + RECENCY_PROFILE_COLUMNS
    profiles = profiles[ordered_columns].copy()

    numeric_exclusions = {"player_name", "team", "position", "latest_team", "latest_game_date", "latest_position", "low_activity_flag"}
    for col in profiles.columns:
        if col not in numeric_exclusions:
            profiles[col] = pd.to_numeric(profiles[col], errors="coerce")

    profiles["player_id"] = profiles["player_id"].astype(int)
    profiles["games_played"] = profiles["games_played"].fillna(0).astype(int)
    profiles["low_activity_flag"] = profiles["low_activity_flag"].astype(bool)

    return profiles.sort_values("player_name").reset_index(drop=True)


# ---------------------------------------------------------------------
# app_roster.csv
# ---------------------------------------------------------------------


def build_app_roster(
    latest_roster_df: pd.DataFrame,
    player_profiles_df: pd.DataFrame,
    latest_season: Optional[int] = None,
) -> pd.DataFrame:
    latest = latest_roster_df.copy()
    profiles = player_profiles_df.copy()

    latest["latest_season"] = pd.to_numeric(latest["latest_season"], errors="coerce")
    if latest_season is None:
        latest_season = int(latest["latest_season"].max())

    latest = latest[latest["latest_season"] == int(latest_season)].copy()
    latest["player_id"] = pd.to_numeric(latest["player_id"], errors="coerce")
    profiles["player_id"] = pd.to_numeric(profiles["player_id"], errors="coerce")
    latest = latest.dropna(subset=["player_id"]).copy()
    profiles = profiles.dropna(subset=["player_id"]).copy()
    latest["player_id"] = latest["player_id"].astype(int)
    profiles["player_id"] = profiles["player_id"].astype(int)

    profile_cols = [
        "player_id", "position", "games_played", "minutes", "points",
        "activity_score", "recent_minutes", "trend_score", "low_activity_flag",
    ]
    available_profile_cols = [col for col in profile_cols if col in profiles.columns]

    app_roster = latest.merge(profiles[available_profile_cols], on="player_id", how="inner")
    app_roster["position"] = app_roster["position"].apply(normalize_position_label)

    ordered = [
        "player_id", "player_name", "latest_team_id", "latest_team", "latest_season",
        "latest_game_date", "latest_game_id", "position", "latest_position", "latest_has_minutes",
        "games_played", "minutes", "points", "activity_score", "recent_minutes", "trend_score", "low_activity_flag",
    ]
    for col in ordered:
        if col not in app_roster.columns:
            app_roster[col] = np.nan

    app_roster = app_roster.sort_values(
        ["latest_team", "games_played", "minutes", "points"],
        ascending=[True, False, False, False],
    )
    return app_roster[ordered].drop_duplicates(subset=["player_id"]).reset_index(drop=True)


# ---------------------------------------------------------------------
# Guardado conveniente
# ---------------------------------------------------------------------


def build_and_save_roster_outputs(
    games_details_df: pd.DataFrame,
    games_df: pd.DataFrame,
    teams_df: Optional[pd.DataFrame],
    latest_roster_path,
    player_profiles_path,
    app_roster_path=None,
    player_season_profiles_path=None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    latest_roster_df = build_latest_roster(games_details_df, games_df, teams_df)
    player_season_profiles_df = build_player_season_profiles(games_details_df, games_df, teams_df)
    player_profiles_df = build_player_profiles(
        games_details_df=games_details_df,
        games_df=games_df,
        teams_df=teams_df,
        latest_roster_df=latest_roster_df,
        player_season_profiles_df=player_season_profiles_df,
    )
    app_roster_df = build_app_roster(latest_roster_df, player_profiles_df)

    latest_roster_path = Path(latest_roster_path)
    player_profiles_path = Path(player_profiles_path)
    latest_roster_path.parent.mkdir(parents=True, exist_ok=True)
    player_profiles_path.parent.mkdir(parents=True, exist_ok=True)

    latest_roster_df.to_csv(latest_roster_path, index=False)
    player_profiles_df.to_csv(player_profiles_path, index=False)

    if app_roster_path is not None:
        app_roster_path = Path(app_roster_path)
        app_roster_path.parent.mkdir(parents=True, exist_ok=True)
        app_roster_df.to_csv(app_roster_path, index=False)

    if player_season_profiles_path is not None:
        player_season_profiles_path = Path(player_season_profiles_path)
        player_season_profiles_path.parent.mkdir(parents=True, exist_ok=True)
        player_season_profiles_df.to_csv(player_season_profiles_path, index=False)

    return latest_roster_df, player_profiles_df, app_roster_df
