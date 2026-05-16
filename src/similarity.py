"""
similarity.py

Este módulo calcula:
- Similitud de rol.
- Fit por posición.
- Fit con el equipo.
- Fit contra el rival.
- Score final de reemplazo.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from .role_features import ROLE_FEATURES, clamp, normalize_position_to_group
except ImportError:
    from role_features import ROLE_FEATURES, clamp, normalize_position_to_group


BASE_WEIGHTS = {
    "scoring": 0.20,
    "playmaking": 0.18,
    "defense": 0.18,
    "rebounding": 0.10,
    "spacing": 0.16,
    "rim_pressure": 0.12,
    "versatility": 0.06,
}

POSITION_FIT_TABLE = {
    ("G", "G"): 1.00,
    ("G", "F"): 0.85,
    ("G", "C"): 0.55,
    ("F", "G"): 0.85,
    ("F", "F"): 1.00,
    ("F", "C"): 0.80,
    ("C", "G"): 0.55,
    ("C", "F"): 0.80,
    ("C", "C"): 1.00,
}

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

UNKNOWN_POSITION_FIT = 0.75


def is_unknown_position(position) -> bool:
    """
    Detecta posiciones desconocidas.

    Importante:
    Si queda algún UNK residual, no lo forzamos a F.
    Le damos un fit neutral para evitar castigar o premiar demasiado.
    """
    if pd.isna(position):
        return True

    position_text = str(position).strip().upper()

    return position_text in UNKNOWN_POSITION_VALUES


def row_value(row, key: str, default=None):
    """Obtiene un valor desde dict o pandas Series."""
    if isinstance(row, pd.Series):
        return row.get(key, default)
    return row.get(key, default)


def calculate_adaptive_weights(
    replaced_player,
    features: list[str] = ROLE_FEATURES,
    base_weights: dict[str, float] = BASE_WEIGHTS,
    alpha: float = 0.60,
) -> dict[str, float]:
    """
    Combina pesos generales con el perfil del jugador reemplazado.

    alpha = 0.60 significa:
    - 60% pesos generales.
    - 40% importancia derivada del propio jugador.
    """
    total_feature_value = sum(
        float(row_value(replaced_player, f, 0.0)) for f in features
    )

    if total_feature_value <= 0:
        return base_weights.copy()

    player_weights = {
        f: float(row_value(replaced_player, f, 0.0)) / total_feature_value
        for f in features
    }

    final_weights = {
        f: alpha * base_weights[f] + (1 - alpha) * player_weights[f] for f in features
    }

    total_weight = sum(final_weights.values())

    return {f: final_weights[f] / total_weight for f in features}


def calculate_role_similarity(
    replaced_player,
    candidate,
    features: list[str] = ROLE_FEATURES,
    weights: dict[str, float] | None = None,
) -> float:
    """
    Calcula similitud usando distancia euclidiana ponderada.

    role_similarity = 1 - weighted_distance
    """
    if weights is None:
        weights = BASE_WEIGHTS

    distance = 0.0

    for feature in features:
        a = float(row_value(replaced_player, feature, 0.5))
        b = float(row_value(candidate, feature, 0.5))
        diff = a - b
        distance += weights[feature] * diff**2

    distance = np.sqrt(distance)
    similarity = 1 - distance

    return clamp(float(similarity), 0.0, 1.0)


def calculate_position_fit(replaced_position: str, candidate_position: str) -> float:
    """
    Calcula compatibilidad posicional usando grupos G/F/C.

    Si alguna posición sigue siendo desconocida, usamos un fit neutral.
    Esto evita que UNK se convierta artificialmente en F y distorsione el ranking.
    """
    if is_unknown_position(replaced_position) or is_unknown_position(candidate_position):
        return UNKNOWN_POSITION_FIT

    pos_a = normalize_position_to_group(replaced_position)
    pos_b = normalize_position_to_group(candidate_position)

    if is_unknown_position(pos_a) or is_unknown_position(pos_b):
        return UNKNOWN_POSITION_FIT

    return POSITION_FIT_TABLE.get((pos_a, pos_b), UNKNOWN_POSITION_FIT)


def calculate_context_fit(
    candidate,
    context_vector: dict[str, float] | None,
    features: list[str] = ROLE_FEATURES,
) -> float:
    """
    Calcula fit del candidato con un vector de contexto.

    Sirve para:
    - team_fit
    - opponent_fit
    """
    if context_vector is None:
        return 0.50

    numerator = 0.0
    denominator = 0.0

    for feature in features:
        importance = float(context_vector.get(feature, 0.0))
        value = float(row_value(candidate, feature, 0.5))

        numerator += importance * value
        denominator += importance

    if denominator <= 0:
        return 0.50

    return clamp(numerator / denominator, 0.0, 1.0)


def build_team_need(team_players_df: pd.DataFrame) -> dict[str, float]:
    """
    Construye un vector simple de necesidades del equipo.

    Intuición:
    - Si el equipo es bajo en una feature, esa feature se vuelve necesidad.
    """
    team_means = team_players_df[ROLE_FEATURES].mean()

    team_need = {}

    for feature in ROLE_FEATURES:
        need = 1 - float(team_means[feature])
        team_need[feature] = clamp(need, 0.20, 0.90)

    return team_need


def build_opponent_context_from_matchup(matchup_row) -> dict[str, float]:
    """
    Construye opponent_context desde una fila de matchups.csv.

    Espera columnas como:
    - scoring_importance
    - defense_importance
    - spacing_importance
    """
    return {
        "scoring": float(row_value(matchup_row, "scoring_importance", 0.50)),
        "playmaking": float(row_value(matchup_row, "playmaking_importance", 0.50)),
        "defense": float(row_value(matchup_row, "defense_importance", 0.50)),
        "rebounding": float(row_value(matchup_row, "rebounding_importance", 0.50)),
        "spacing": float(row_value(matchup_row, "spacing_importance", 0.50)),
        "rim_pressure": float(row_value(matchup_row, "rim_pressure_importance", 0.50)),
        "versatility": float(row_value(matchup_row, "versatility_importance", 0.50)),
    }


def build_opponent_context_from_players(
    opponent_players_df: pd.DataFrame,
) -> dict[str, float]:
    """
    Construye un contexto de rival automáticamente usando sus jugadores.

    Es una aproximación:
    - Si el rival tiene mucho scoring/playmaking/rim pressure, valoramos defensa.
    - Si el rival es fuerte en rebote, valoramos rebounding.
    """
    opponent_mean = opponent_players_df[ROLE_FEATURES].mean()

    offensive_pressure = float(
        np.mean(
            [
                opponent_mean["scoring"],
                opponent_mean["playmaking"],
                opponent_mean["rim_pressure"],
            ]
        )
    )

    return {
        "scoring": 0.50,
        "playmaking": 0.50,
        "defense": clamp(offensive_pressure, 0.20, 0.90),
        "rebounding": clamp(float(opponent_mean["rebounding"]), 0.20, 0.90),
        "spacing": clamp(1 - float(opponent_mean["defense"]), 0.20, 0.90),
        "rim_pressure": clamp(1 - float(opponent_mean["defense"]), 0.20, 0.90),
        "versatility": 0.60,
    }


def calculate_replacement_score(
    replaced_player,
    candidate,
    team_need: dict[str, float] | None = None,
    opponent_context: dict[str, float] | None = None,
    alpha: float = 0.60,
) -> dict:
    """
    Calcula el score final de reemplazo.

    Fórmula:
    replacement_score =
        0.70 * role_similarity
      + 0.15 * position_fit
      + 0.10 * team_fit
      + 0.05 * opponent_fit
    """
    weights = calculate_adaptive_weights(
        replaced_player=replaced_player,
        features=ROLE_FEATURES,
        base_weights=BASE_WEIGHTS,
        alpha=alpha,
    )

    role_similarity = calculate_role_similarity(
        replaced_player=replaced_player,
        candidate=candidate,
        features=ROLE_FEATURES,
        weights=weights,
    )

    replaced_position = row_value(
        replaced_player, "position_group", row_value(replaced_player, "position", "F")
    )
    candidate_position = row_value(
        candidate, "position_group", row_value(candidate, "position", "F")
    )

    position_fit = calculate_position_fit(replaced_position, candidate_position)

    team_fit = calculate_context_fit(candidate, team_need, ROLE_FEATURES)
    opponent_fit = calculate_context_fit(candidate, opponent_context, ROLE_FEATURES)

    final_score = (
        0.70 * role_similarity
        + 0.15 * position_fit
        + 0.10 * team_fit
        + 0.05 * opponent_fit
    )

    return {
        "final_score": clamp(final_score, 0.0, 1.0),
        "role_similarity": role_similarity,
        "position_fit": position_fit,
        "team_fit": team_fit,
        "opponent_fit": opponent_fit,
        "weights_used": weights,
    }


def rank_replacements(
    players_df: pd.DataFrame,
    replaced_player,
    team_need: dict[str, float] | None = None,
    opponent_context: dict[str, float] | None = None,
    top_n: int | None = None,
) -> list[dict]:
    """
    Calcula ranking de candidatos para reemplazar a un jugador.
    """
    replaced_name = row_value(
        replaced_player, "player_name", row_value(replaced_player, "name", None)
    )

    results = []

    for _, candidate_row in players_df.iterrows():
        candidate_name = row_value(
            candidate_row, "player_name", row_value(candidate_row, "name", None)
        )

        if candidate_name == replaced_name:
            continue

        score_data = calculate_replacement_score(
            replaced_player=replaced_player,
            candidate=candidate_row,
            team_need=team_need,
            opponent_context=opponent_context,
        )

        results.append(
            {
                "player_name": candidate_name,
                "team": row_value(candidate_row, "team", ""),
                "position_group": row_value(candidate_row, "position_group", ""),
                **score_data,
                "candidate_row": candidate_row.to_dict(),
            }
        )

    results = sorted(results, key=lambda x: x["final_score"], reverse=True)

    if top_n is not None:
        return results[:top_n]

    return results
