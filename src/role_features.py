"""
role_features.py

Este módulo prepara los datos de jugadores y construye los vectores de rol.

Objetivo:
- Aceptar CSV simulados o reales.
- Estandarizar nombres de columnas.
- Calcular features de rol si no vienen ya calculadas.
- Devolver un DataFrame listo para similarity.py, impact.py y monte_carlo.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


ROLE_FEATURES = [
    "scoring",
    "playmaking",
    "defense",
    "rebounding",
    "spacing",
    "rim_pressure",
    "versatility",
]


# Alias para poder aceptar diferentes fuentes de datos.
# Ejemplo: un CSV puede usar "points", otro "PTS".
COLUMN_ALIASES = {
    "player_name": ["player_name", "name", "PLAYER_NAME"],
    "team": ["team", "team_name", "TEAM_NAME"],
    "position": ["position", "pos", "POSITION"],
    "position_group": ["position_group", "position", "pos", "POSITION"],
    "minutes": ["minutes", "MIN", "min"],
    "games_played": ["games_played", "GP", "games"],
    "PTS": ["PTS", "points", "pts"],
    "AST": ["AST", "assists", "ast"],
    "REB": ["REB", "rebounds", "reb"],
    "OREB": ["OREB", "offensive_rebounds", "oreb"],
    "DREB": ["DREB", "defensive_rebounds", "dreb"],
    "STL": ["STL", "steals", "stl"],
    "BLK": ["BLK", "blocks", "blk"],
    "TOV": ["TOV", "turnovers", "tov"],
    "PF": ["PF", "fouls", "personal_fouls"],
    "FGA": ["FGA", "field_goal_attempts", "fga"],
    "FTA": ["FTA", "free_throw_attempts", "fta"],
    "FG3A": ["FG3A", "three_attempts", "three_point_attempts", "3PA"],
    "FG_PCT": ["FG_PCT", "fg_pct", "field_goal_pct"],
    "FG3_PCT": ["FG3_PCT", "three_pct", "three_point_pct", "3P%"],
    "FT_PCT": ["FT_PCT", "ft_pct", "free_throw_pct"],
    "TS": ["TS", "ts", "true_shooting", "true_shooting_pct"],
    "USG": ["USG", "usage_rate", "usage", "USG_PCT"],
    "ORtg": ["ORtg", "offensive_rating", "off_rating"],
    "DRtg": ["DRtg", "defensive_rating", "def_rating"],
    "pace_on_court": ["pace_on_court", "pace"],
    "PTS_per36": ["PTS_per36", "points_per36"],
    "AST_per36": ["AST_per36", "assists_per36"],
    "REB_per36": ["REB_per36", "rebounds_per36"],
    "OREB_per36": ["OREB_per36", "offensive_rebounds_per36"],
    "DREB_per36": ["DREB_per36", "defensive_rebounds_per36"],
    "STL_per36": ["STL_per36", "steals_per36"],
    "BLK_per36": ["BLK_per36", "blocks_per36"],
    "TOV_per36": ["TOV_per36", "turnovers_per36"],
    "PF_per36": ["PF_per36", "fouls_per36"],
    "FGA_per36": ["FGA_per36", "field_goal_attempts_per36"],
    "FTA_per36": ["FTA_per36", "free_throw_attempts_per36"],
    "FG3A_per36": ["FG3A_per36", "three_attempts_per36"],
    "AST_PCT": ["AST_PCT", "assist_pct"],
    "REB_PCT": ["REB_PCT", "rebound_pct"],
    "OREB_PCT": ["OREB_PCT", "offensive_rebound_pct"],
    "DREB_PCT": ["DREB_PCT", "defensive_rebound_pct"],
    "STL_PCT": ["STL_PCT", "steal_pct"],
    "BLK_PCT": ["BLK_PCT", "block_pct"],
    "AST_TO": ["AST_TO", "assist_turnover_ratio"],
    "expected_minutes": ["expected_minutes", "projected_minutes"],
}


def clamp(value: float, min_value: float = 0.0, max_value: float = 1.0) -> float:
    """Limita un valor a un rango."""
    return max(min_value, min(max_value, value))


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


def normalize_position_to_group(position: str) -> str:
    """
    Convierte posiciones detalladas a grupos simples:
    G = guard, F = forward, C = center.

    Si la posición es desconocida, devuelve UNK.
    El fit neutral para UNK se decide en similarity.py.
    """
    if pd.isna(position):
        return "UNK"

    pos = str(position).upper().strip().replace(" ", "").replace("_", "-")

    if pos in UNKNOWN_POSITION_VALUES:
        return "UNK"

    if pos in ["PG", "SG", "G", "GUARD"]:
        return "G"

    if pos in ["SF", "PF", "F", "FORWARD"]:
        return "F"

    if pos in ["C", "CENTER"]:
        return "C"

    # Casos mixtos: los tratamos como wing salvo que sean claramente center.
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


def _copy_alias_column(df: pd.DataFrame, canonical: str) -> pd.DataFrame:
    """
    Si la columna canónica no existe, busca un alias y la copia.
    """
    if canonical in df.columns:
        return df

    for alias in COLUMN_ALIASES.get(canonical, []):
        if alias in df.columns:
            df[canonical] = df[alias]
            return df

    return df


def standardize_player_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estandariza columnas para que el resto del modelo use nombres consistentes.

    No elimina columnas originales. Solo agrega columnas canónicas si faltan.
    """
    df = df.copy()

    canonical_columns = list(COLUMN_ALIASES.keys())

    for col in canonical_columns:
        df = _copy_alias_column(df, col)

    # Mantener compatibilidad con CSV anteriores que usan "name".
    if "player_name" not in df.columns and "name" in df.columns:
        df["player_name"] = df["name"]

    if "team" not in df.columns and "team_name" in df.columns:
        df["team"] = df["team_name"]

    if "position_group" not in df.columns:
        if "position" in df.columns:
            df["position_group"] = df["position"].apply(normalize_position_to_group)
        else:
            df["position_group"] = "UNK"
    else:
        df["position_group"] = df["position_group"].apply(normalize_position_to_group)

    if "position" not in df.columns:
        df["position"] = df["position_group"]

    # Si no existe TS, usamos FG% como aproximación simple.
    if "TS" not in df.columns:
        if "FG_PCT" in df.columns:
            df["TS"] = df["FG_PCT"]
        else:
            df["TS"] = 0.56

    # Si no existe USG, usamos un valor neutral.
    if "USG" not in df.columns:
        df["USG"] = 0.20

    # Si no existe ORtg/DRtg, usamos promedios de liga aproximados.
    if "ORtg" not in df.columns:
        df["ORtg"] = 114.0

    if "DRtg" not in df.columns:
        df["DRtg"] = 114.0

    if "pace_on_court" not in df.columns:
        df["pace_on_court"] = 99.0

    if "games_played" not in df.columns:
        df["games_played"] = 1

    if "minutes" not in df.columns:
        df["minutes"] = 0.0

    if "expected_minutes" not in df.columns:
        # Si tenemos minutos totales y partidos jugados, aproximamos minutos esperados.
        safe_gp = df["games_played"].replace(0, np.nan)
        df["expected_minutes"] = (df["minutes"] / safe_gp).fillna(24.0)

    return df


def _create_per36_column(
    df: pd.DataFrame,
    total_col: str,
    per36_col: str,
    default_value: float = 0.0,
) -> pd.DataFrame:
    """
    Crea una columna por 36 minutos si no existe.

    Sirve tanto para datos totales como para promedios por partido, siempre que
    'minutes' esté en la misma escala.
    """
    if per36_col in df.columns:
        return df

    if total_col in df.columns and "minutes" in df.columns:
        safe_minutes = df["minutes"].replace(0, np.nan)
        df[per36_col] = 36.0 * df[total_col] / safe_minutes
        df[per36_col] = df[per36_col].replace([np.inf, -np.inf], np.nan)
        df[per36_col] = df[per36_col].fillna(default_value)
    else:
        df[per36_col] = default_value

    return df


def add_basic_per36_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega estadísticas por 36 minutos si faltan."""
    df = df.copy()

    per36_specs = [
        ("PTS", "PTS_per36", 12.0),
        ("AST", "AST_per36", 3.0),
        ("REB", "REB_per36", 5.0),
        ("OREB", "OREB_per36", 1.0),
        ("DREB", "DREB_per36", 4.0),
        ("STL", "STL_per36", 1.0),
        ("BLK", "BLK_per36", 0.5),
        ("TOV", "TOV_per36", 2.0),
        ("PF", "PF_per36", 2.5),
        ("FGA", "FGA_per36", 10.0),
        ("FTA", "FTA_per36", 3.0),
        ("FG3A", "FG3A_per36", 3.0),
    ]

    for total_col, per36_col, default_value in per36_specs:
        df = _create_per36_column(df, total_col, per36_col, default_value)

    # Si FGA_per36 falta o es muy bajo, aproximamos desde puntos.
    df["FGA_per36"] = df["FGA_per36"].fillna(df["PTS_per36"] / 1.2)
    df["FGA_per36"] = df["FGA_per36"].clip(lower=1.0)

    return df


def percentile_normalize(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    """
    Convierte una columna a percentiles entre 0 y 1.

    Ventaja: es simple, interpretable y no depende de unidades.
    """
    numeric = pd.to_numeric(series, errors="coerce")

    if numeric.notna().sum() <= 1:
        return pd.Series(0.5, index=series.index)

    ranks = numeric.rank(pct=True)

    if not higher_is_better:
        ranks = 1 - ranks

    return ranks.fillna(0.5).clip(0, 1)


def ensure_advanced_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea columnas avanzadas opcionales si no existen.
    """
    df = df.copy()

    if "AST_TO" not in df.columns:
        df["AST_TO"] = df["AST_per36"] / (df["TOV_per36"] + 0.1)

    if "FG3_PCT" not in df.columns:
        df["FG3_PCT"] = 0.34

    if "FT_PCT" not in df.columns:
        df["FT_PCT"] = 0.76

    if "AST_PCT" not in df.columns:
        df["AST_PCT"] = df["AST_per36"]

    if "REB_PCT" not in df.columns:
        df["REB_PCT"] = df["REB_per36"]

    if "OREB_PCT" not in df.columns:
        df["OREB_PCT"] = df["OREB_per36"]

    if "DREB_PCT" not in df.columns:
        df["DREB_PCT"] = df["DREB_per36"]

    if "STL_PCT" not in df.columns:
        df["STL_PCT"] = df["STL_per36"]

    if "BLK_PCT" not in df.columns:
        df["BLK_PCT"] = df["BLK_per36"]

    return df


def compute_role_features(df: pd.DataFrame, overwrite: bool = False) -> pd.DataFrame:
    """
    Calcula las 7 features de rol.

    Si overwrite=False, solo calcula una feature si no existe.
    """
    df = standardize_player_columns(df)
    df = add_basic_per36_stats(df)
    df = ensure_advanced_columns(df)

    # Normalizaciones principales.
    pts_norm = percentile_normalize(df["PTS_per36"])
    ts_norm = percentile_normalize(df["TS"])
    usg_norm = percentile_normalize(df["USG"])
    fta_norm = percentile_normalize(df["FTA_per36"])

    ast_norm = percentile_normalize(df["AST_per36"])
    ast_pct_norm = percentile_normalize(df["AST_PCT"])
    ast_to_norm = percentile_normalize(df["AST_TO"])

    stl_norm = percentile_normalize(df["STL_PCT"])
    blk_norm = percentile_normalize(df["BLK_PCT"])
    dreb_norm = percentile_normalize(df["DREB_PCT"])
    drtg_norm = percentile_normalize(df["DRtg"], higher_is_better=False)
    pf_norm = percentile_normalize(df["PF_per36"], higher_is_better=False)

    reb_norm = percentile_normalize(df["REB_PCT"])
    oreb_norm = percentile_normalize(df["OREB_PCT"])

    three_attempt_norm = percentile_normalize(df["FG3A_per36"])
    three_pct_norm = percentile_normalize(df["FG3_PCT"])
    ft_pct_norm = percentile_normalize(df["FT_PCT"])

    two_pa_per36 = (df["FGA_per36"] - df["FG3A_per36"]).clip(lower=0)
    two_pa_norm = percentile_normalize(two_pa_per36)

    role_values = {
        "scoring": (
            0.40 * pts_norm + 0.25 * ts_norm + 0.20 * usg_norm + 0.15 * fta_norm
        ),
        "playmaking": (
            0.45 * ast_norm + 0.30 * ast_pct_norm + 0.15 * ast_to_norm + 0.10 * usg_norm
        ),
        "defense": (
            0.22 * stl_norm
            + 0.22 * blk_norm
            + 0.18 * dreb_norm
            + 0.28 * drtg_norm
            + 0.10 * pf_norm
        ),
        "rebounding": (0.45 * reb_norm + 0.30 * dreb_norm + 0.25 * oreb_norm),
        "spacing": (
            0.45 * three_attempt_norm + 0.35 * three_pct_norm + 0.20 * ft_pct_norm
        ),
        "rim_pressure": (0.50 * fta_norm + 0.30 * two_pa_norm + 0.20 * usg_norm),
    }

    for feature, values in role_values.items():
        if overwrite or feature not in df.columns:
            df[feature] = values.clip(0, 1)

    # Versatility: promedio de las mejores 4 features base.
    if overwrite or "versatility" not in df.columns:
        base = [
            "scoring",
            "playmaking",
            "defense",
            "rebounding",
            "spacing",
            "rim_pressure",
        ]

        def top4_average(row: pd.Series) -> float:
            values = sorted([row[f] for f in base], reverse=True)
            return float(np.mean(values[:4]))

        df["versatility"] = df.apply(top4_average, axis=1).clip(0, 1)

    return df


def prepare_players_data(
    df: pd.DataFrame, overwrite_roles: bool = False
) -> pd.DataFrame:
    """
    Función principal para preparar jugadores.

    Entrada:
    - CSV simulado o real.

    Salida:
    - DataFrame con columnas estandarizadas y features de rol listas.
    """
    df = standardize_player_columns(df)
    df = add_basic_per36_stats(df)

    missing_roles = [feature for feature in ROLE_FEATURES if feature not in df.columns]

    if missing_roles or overwrite_roles:
        df = compute_role_features(df, overwrite=overwrite_roles)

    # Asegurar que las features estén en rango 0-1.
    for feature in ROLE_FEATURES:
        df[feature] = pd.to_numeric(df[feature], errors="coerce").fillna(0.5).clip(0, 1)

    return df
