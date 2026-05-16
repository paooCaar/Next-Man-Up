"""
monte_carlo.py

Este módulo usa el impacto estimado del jugador para simular partidos.

Objetivo:
- Calcular probabilidad de ganar sin reemplazo.
- Calcular probabilidad de ganar con reemplazo.
- Comparar distribuciones de margen.
"""

from __future__ import annotations

import numpy as np


def get_team_value(team_context: dict, key: str, default: float) -> float:
    """
    Obtiene una métrica de equipo aceptando nombres alternativos.
    """
    aliases = {
        "ORtg": ["ORtg", "offensive_rating", "OFF_RATING"],
        "DRtg": ["DRtg", "defensive_rating", "DEF_RATING"],
        "pace": ["pace", "PACE"],
        "net_rating": ["net_rating", "NET_RATING"],
    }

    for alias in aliases.get(key, [key]):
        if alias in team_context:
            return float(team_context[alias])

    return default


def estimate_expected_possessions(team_pace: float, opponent_pace: float) -> float:
    """
    Aproxima posesiones esperadas del partido.
    """
    return (team_pace + opponent_pace) / 2.0


def estimate_expected_points(
    team_ortg: float,
    opponent_drtg: float,
    expected_possessions: float,
) -> float:
    """
    Estima puntos esperados usando ataque propio y defensa rival.

    ORtg y DRtg están en puntos por 100 posesiones.
    """
    expected_rating = 0.50 * team_ortg + 0.50 * opponent_drtg

    return expected_rating * expected_possessions / 100.0


def adjust_expected_points_with_impact(
    team_expected_points: float,
    opponent_expected_points: float,
    offensive_impact: float,
    defensive_impact: float,
    pace_impact: float = 0.0,
) -> tuple[float, float]:
    """
    Ajusta las medias esperadas usando impacto ofensivo, defensivo y de ritmo.
    """
    adjusted_team_points = team_expected_points + offensive_impact + pace_impact / 2.0

    adjusted_opponent_points = (
        opponent_expected_points - defensive_impact - pace_impact / 2.0
    )

    return adjusted_team_points, adjusted_opponent_points


def simulate_games(
    team_mean: float,
    opponent_mean: float,
    team_std: float = 12.0,
    opponent_std: float = 12.0,
    n_simulations: int = 1000,
    random_seed: int | None = 42,
) -> dict:
    """
    Simula partidos usando distribución normal para los puntos.

    team_points ~ Normal(team_mean, team_std)
    opponent_points ~ Normal(opponent_mean, opponent_std)
    """
    rng = np.random.default_rng(random_seed)

    team_scores = rng.normal(
        loc=team_mean,
        scale=team_std,
        size=n_simulations,
    )

    opponent_scores = rng.normal(
        loc=opponent_mean,
        scale=opponent_std,
        size=n_simulations,
    )

    # Puntos no negativos y redondeados.
    team_scores = np.maximum(0, np.round(team_scores))
    opponent_scores = np.maximum(0, np.round(opponent_scores))

    margins = team_scores - opponent_scores
    wins = margins > 0

    return {
        "win_probability": float(wins.mean()),
        "expected_team_points": float(team_scores.mean()),
        "expected_opponent_points": float(opponent_scores.mean()),
        "expected_margin": float(margins.mean()),
        "margins": margins,
        "team_scores": team_scores,
        "opponent_scores": opponent_scores,
    }


def monte_carlo_replacement_analysis(
    team_context: dict,
    opponent_context: dict,
    player_impact: dict,
    team_std: float = 12.0,
    opponent_std: float = 12.0,
    n_simulations: int = 1000,
) -> dict:
    """
    Compara dos escenarios:
    - Sin reemplazo.
    - Con reemplazo.
    """
    team_ortg = get_team_value(team_context, "ORtg", 114.0)
    team_drtg = get_team_value(team_context, "DRtg", 114.0)
    team_pace = get_team_value(team_context, "pace", 99.0)

    opponent_ortg = get_team_value(opponent_context, "ORtg", 114.0)
    opponent_drtg = get_team_value(opponent_context, "DRtg", 114.0)
    opponent_pace = get_team_value(opponent_context, "pace", 99.0)

    expected_possessions = estimate_expected_possessions(
        team_pace=team_pace,
        opponent_pace=opponent_pace,
    )

    team_expected_points = estimate_expected_points(
        team_ortg=team_ortg,
        opponent_drtg=opponent_drtg,
        expected_possessions=expected_possessions,
    )

    opponent_expected_points = estimate_expected_points(
        team_ortg=opponent_ortg,
        opponent_drtg=team_drtg,
        expected_possessions=expected_possessions,
    )

    without_replacement = simulate_games(
        team_mean=team_expected_points,
        opponent_mean=opponent_expected_points,
        team_std=team_std,
        opponent_std=opponent_std,
        n_simulations=n_simulations,
        random_seed=42,
    )

    adjusted_team_points, adjusted_opponent_points = adjust_expected_points_with_impact(
        team_expected_points=team_expected_points,
        opponent_expected_points=opponent_expected_points,
        offensive_impact=float(player_impact.get("offensive_impact", 0.0)),
        defensive_impact=float(player_impact.get("defensive_impact", 0.0)),
        pace_impact=float(player_impact.get("pace_impact", 0.0)),
    )

    with_replacement = simulate_games(
        team_mean=adjusted_team_points,
        opponent_mean=adjusted_opponent_points,
        team_std=team_std,
        opponent_std=opponent_std,
        n_simulations=n_simulations,
        random_seed=43,
    )

    return {
        "without_replacement": without_replacement,
        "with_replacement": with_replacement,
        "win_probability_without": without_replacement["win_probability"],
        "win_probability_with": with_replacement["win_probability"],
        "win_probability_delta": (
            with_replacement["win_probability"] - without_replacement["win_probability"]
        ),
        "expected_margin_without": without_replacement["expected_margin"],
        "expected_margin_with": with_replacement["expected_margin"],
        "expected_margin_delta": (
            with_replacement["expected_margin"] - without_replacement["expected_margin"]
        ),
        "team_expected_points_base": team_expected_points,
        "opponent_expected_points_base": opponent_expected_points,
        "team_expected_points_with_replacement": adjusted_team_points,
        "opponent_expected_points_with_replacement": adjusted_opponent_points,
    }
