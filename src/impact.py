"""
impact.py

Este módulo estima el impacto de un candidato sobre el equipo.

Versión actual:
- Calcula impacto relativo, no absoluto.
- El impacto responde:

    ¿Cuánto mejora o empeora el equipo si entra el candidato
    en lugar del jugador reemplazado?

Fórmula conceptual:

    estimated_net_impact =
        replacement_score * (candidate_value - replaced_value)

Además:
- Se separa impacto ofensivo, defensivo y de ritmo.
- Se limita el impacto máximo para evitar saltos exagerados
  en la simulación Monte Carlo.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from .role_features import clamp
except ImportError:
    from role_features import clamp


# ============================================================
# 1. CONSTANTES DEL MODELO
# ============================================================

DEFAULT_LEAGUE_AVG = {
    "PTS_per36": 13.0,
    "TS": 0.58,
    "ORtg": 114.0,
    "DRtg": 114.0,
    "USG": 0.20,
}

# Caps razonables para impacto en puntos de margen.
# Estos valores son deliberadamente conservadores para evitar que
# Monte Carlo convierta cualquier estrella en +20 puntos automáticos.
DEFAULT_IMPACT_CAPS = {
    "offensive_impact": 5.0,
    "defensive_impact": 4.0,
    "pace_impact": 1.5,
    "net_impact": 8.0,
}


# ============================================================
# 2. HELPERS GENERALES
# ============================================================


def row_value(row, key: str, default=None):
    """Obtiene un valor desde dict o pandas Series."""
    if isinstance(row, pd.Series):
        return row.get(key, default)

    if isinstance(row, dict):
        return row.get(key, default)

    return default


def safe_float(value, default: float = 0.0) -> float:
    """Convierte un valor a float evitando NaN e infinitos."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default

    if not np.isfinite(number):
        return default

    return number


def cap_value(value: float, max_abs_value: float) -> float:
    """
    Limita un valor al rango [-max_abs_value, +max_abs_value].
    """
    return float(np.clip(value, -max_abs_value, max_abs_value))


def context_multiplier(replacement_score: float) -> float:
    """
    Convierte replacement_score en un multiplicador suave.

    Se mantiene por compatibilidad con funciones anteriores.
    Para el cálculo relativo principal usamos fit_multiplier().
    """
    replacement_score = clamp(float(replacement_score), 0.0, 1.0)

    return 0.85 + 0.30 * replacement_score


def fit_multiplier(replacement_score: float) -> float:
    """
    Suaviza el delta relativo usando el score de reemplazo.

    Intuición:
    - Si el candidato encaja muy bien, dejamos pasar casi todo el delta.
    - Si encaja mal, reducimos mucho el impacto estimado.

    Fórmula:
        adjusted_delta = replacement_score * raw_delta
    """
    return clamp(float(replacement_score), 0.0, 1.0)


def get_expected_minutes(player, fallback: float = 24.0) -> float:
    """
    Obtiene minutos esperados del jugador.
    """
    minutes = safe_float(row_value(player, "expected_minutes", fallback), fallback)

    return clamp(minutes, 0.0, 48.0)


def get_comparison_minutes(candidate, replaced_player) -> float:
    """
    Define minutos comparables para evaluar candidato y reemplazado.

    Decisión de diseño:
    - El candidato no debería recibir crédito por más minutos que el jugador
      que reemplaza.
    - Si el candidato históricamente juega menos, usamos sus minutos como
      una forma simple de no inflar su impacto.

    Fórmula:
        comparison_minutes = min(candidate_expected_minutes, replaced_expected_minutes)

    Si alguno viene vacío, usamos fallback razonable.
    """
    candidate_minutes = get_expected_minutes(candidate, fallback=24.0)
    replaced_minutes = get_expected_minutes(replaced_player, fallback=candidate_minutes)

    comparison_minutes = min(candidate_minutes, replaced_minutes)

    if comparison_minutes <= 0:
        comparison_minutes = min(
            max(candidate_minutes, 1.0),
            max(replaced_minutes, 1.0),
        )

    return clamp(comparison_minutes, 1.0, 42.0)


# ============================================================
# 3. VALOR ABSOLUTO INTERNO POR COMPONENTE
# ============================================================


def estimate_offensive_value(
    player,
    league_avg: dict | None = None,
    team_pace: float = 99.0,
    minutes_override: float | None = None,
) -> float:
    """
    Estima valor ofensivo interno de un jugador.

    Importante:
    - Esto NO es todavía el impacto final.
    - Sirve para comparar candidate_value vs replaced_value.
    """
    if league_avg is None:
        league_avg = DEFAULT_LEAGUE_AVG

    expected_minutes = (
        float(minutes_override)
        if minutes_override is not None
        else get_expected_minutes(player)
    )

    expected_minutes = clamp(expected_minutes, 0.0, 48.0)

    minutes_factor = expected_minutes / 36.0
    possessions_on_court = team_pace * expected_minutes / 48.0

    pts_per36 = safe_float(
        row_value(player, "PTS_per36", league_avg["PTS_per36"]),
        league_avg["PTS_per36"],
    )

    fga_per36 = safe_float(row_value(player, "FGA_per36", 10.0), 10.0)
    fta_per36 = safe_float(row_value(player, "FTA_per36", 3.0), 3.0)

    ts = safe_float(
        row_value(player, "TS", league_avg["TS"]),
        league_avg["TS"],
    )

    ortg = safe_float(
        row_value(player, "ORtg", league_avg["ORtg"]),
        league_avg["ORtg"],
    )

    usg = safe_float(
        row_value(player, "USG", league_avg["USG"]),
        league_avg["USG"],
    )

    usage_factor = usg / league_avg["USG"]
    usage_factor = clamp(usage_factor, 0.75, 1.25)

    scoring_volume_value = (pts_per36 - league_avg["PTS_per36"]) * minutes_factor

    shooting_possessions = (fga_per36 + 0.44 * fta_per36) * minutes_factor

    efficiency_value = 2.0 * (ts - league_avg["TS"]) * shooting_possessions

    offensive_rating_value = (
        ((ortg - league_avg["ORtg"]) / 100.0) * possessions_on_court * usage_factor
    )

    offensive_value = (
        0.40 * scoring_volume_value
        + 0.30 * efficiency_value
        + 0.30 * offensive_rating_value
    )

    return float(offensive_value)


def estimate_defensive_value(
    player,
    league_avg: dict | None = None,
    team_pace: float = 99.0,
    minutes_override: float | None = None,
) -> float:
    """
    Estima valor defensivo interno.

    Como DRtg menor es mejor:

        defensive_value = league_avg_DRtg - player_DRtg
    """
    if league_avg is None:
        league_avg = DEFAULT_LEAGUE_AVG

    expected_minutes = (
        float(minutes_override)
        if minutes_override is not None
        else get_expected_minutes(player)
    )

    expected_minutes = clamp(expected_minutes, 0.0, 48.0)

    possessions_on_court = team_pace * expected_minutes / 48.0

    drtg = safe_float(
        row_value(player, "DRtg", league_avg["DRtg"]),
        league_avg["DRtg"],
    )

    defensive_value = ((league_avg["DRtg"] - drtg) / 100.0) * possessions_on_court

    return float(defensive_value)


def estimate_pace_value(
    player,
    team_pace: float = 99.0,
    team_net_rating: float = 0.0,
    minutes_override: float | None = None,
) -> float:
    """
    Estima valor de ritmo.

    Si un jugador acelera el partido, ayuda más a equipos con net_rating positivo.
    Si el equipo tiene net_rating negativo, acelerar puede ser perjudicial.
    """
    expected_minutes = (
        float(minutes_override)
        if minutes_override is not None
        else get_expected_minutes(player)
    )

    expected_minutes = clamp(expected_minutes, 0.0, 48.0)

    pace_on_court = safe_float(
        row_value(player, "pace_on_court", team_pace),
        team_pace,
    )

    pace_delta = (pace_on_court - team_pace) * expected_minutes / 48.0

    pace_value = pace_delta * (team_net_rating / 100.0)

    return float(pace_value)


def estimate_raw_player_value(
    player,
    league_avg: dict | None = None,
    team_pace: float = 99.0,
    team_net_rating: float = 0.0,
    minutes_override: float | None = None,
) -> dict:
    """
    Estima valor absoluto interno de un jugador.

    Devuelve componentes separados para luego poder calcular:

        candidate_value - replaced_value
    """
    if league_avg is None:
        league_avg = DEFAULT_LEAGUE_AVG

    offensive_value = estimate_offensive_value(
        player=player,
        league_avg=league_avg,
        team_pace=team_pace,
        minutes_override=minutes_override,
    )

    defensive_value = estimate_defensive_value(
        player=player,
        league_avg=league_avg,
        team_pace=team_pace,
        minutes_override=minutes_override,
    )

    pace_value = estimate_pace_value(
        player=player,
        team_pace=team_pace,
        team_net_rating=team_net_rating,
        minutes_override=minutes_override,
    )

    total_value = offensive_value + defensive_value + pace_value

    return {
        "offensive_value": offensive_value,
        "defensive_value": defensive_value,
        "pace_value": pace_value,
        "total_value": total_value,
    }


# ============================================================
# 4. CAPS
# ============================================================


def cap_component_impacts(
    offensive_impact: float,
    defensive_impact: float,
    pace_impact: float,
    caps: dict | None = None,
) -> dict:
    """
    Aplica caps por componente.
    """
    if caps is None:
        caps = DEFAULT_IMPACT_CAPS

    return {
        "offensive_impact": cap_value(
            offensive_impact,
            caps.get("offensive_impact", 5.0),
        ),
        "defensive_impact": cap_value(
            defensive_impact,
            caps.get("defensive_impact", 4.0),
        ),
        "pace_impact": cap_value(
            pace_impact,
            caps.get("pace_impact", 1.5),
        ),
    }


def cap_net_impact(
    offensive_impact: float,
    defensive_impact: float,
    pace_impact: float,
    caps: dict | None = None,
) -> dict:
    """
    Aplica cap al impacto neto manteniendo la proporción entre componentes.
    """
    if caps is None:
        caps = DEFAULT_IMPACT_CAPS

    net_cap = caps.get("net_impact", 8.0)

    net_impact = offensive_impact + defensive_impact + pace_impact

    if abs(net_impact) <= net_cap or net_impact == 0:
        return {
            "offensive_impact": offensive_impact,
            "defensive_impact": defensive_impact,
            "pace_impact": pace_impact,
            "estimated_net_impact": net_impact,
            "cap_applied": False,
        }

    scale = net_cap / abs(net_impact)

    offensive_impact = offensive_impact * scale
    defensive_impact = defensive_impact * scale
    pace_impact = pace_impact * scale

    net_impact = offensive_impact + defensive_impact + pace_impact

    return {
        "offensive_impact": offensive_impact,
        "defensive_impact": defensive_impact,
        "pace_impact": pace_impact,
        "estimated_net_impact": net_impact,
        "cap_applied": True,
    }


# ============================================================
# 5. FUNCIONES LEGACY COMPATIBLES
# ============================================================


def estimate_offensive_impact(
    candidate,
    replacement_score: float,
    league_avg: dict | None = None,
    team_pace: float = 99.0,
) -> float:
    """
    Función mantenida por compatibilidad.

    Devuelve impacto ofensivo absoluto suavizado.
    El flujo principal ahora usa estimate_player_impact() con replaced_player.
    """
    value = estimate_offensive_value(
        player=candidate,
        league_avg=league_avg,
        team_pace=team_pace,
    )

    return value * context_multiplier(replacement_score)


def estimate_defensive_impact(
    candidate,
    replacement_score: float,
    league_avg: dict | None = None,
    team_pace: float = 99.0,
) -> float:
    """
    Función mantenida por compatibilidad.
    """
    value = estimate_defensive_value(
        player=candidate,
        league_avg=league_avg,
        team_pace=team_pace,
    )

    return value * context_multiplier(replacement_score)


def estimate_pace_impact(
    candidate,
    team_pace: float = 99.0,
    team_net_rating: float = 0.0,
) -> float:
    """
    Función mantenida por compatibilidad.
    """
    return estimate_pace_value(
        player=candidate,
        team_pace=team_pace,
        team_net_rating=team_net_rating,
    )


# ============================================================
# 6. FUNCIÓN PRINCIPAL: IMPACTO RELATIVO
# ============================================================


def estimate_player_impact(
    candidate,
    score_data: dict,
    team_context: dict,
    replaced_player=None,
    league_avg: dict | None = None,
    impact_caps: dict | None = None,
) -> dict:
    """
    Estima impacto total del candidato.

    Si replaced_player existe:
        impacto = replacement_score * (candidate_value - replaced_value)

    Si replaced_player no existe:
        usa modo legacy absoluto, pero con caps.
    """
    if league_avg is None:
        league_avg = DEFAULT_LEAGUE_AVG

    if impact_caps is None:
        impact_caps = DEFAULT_IMPACT_CAPS

    replacement_score = safe_float(score_data.get("final_score", 0.0), 0.0)
    replacement_score = clamp(replacement_score, 0.0, 1.0)

    team_pace = safe_float(
        team_context.get("pace", team_context.get("PACE", 99.0)),
        99.0,
    )

    team_net_rating = safe_float(
        team_context.get("net_rating", team_context.get("NET_RATING", 0.0)),
        0.0,
    )

    # --------------------------------------------------------
    # Modo legacy: impacto absoluto del candidato.
    # Se conserva para no romper usos externos.
    # --------------------------------------------------------
    if replaced_player is None:
        candidate_value = estimate_raw_player_value(
            player=candidate,
            league_avg=league_avg,
            team_pace=team_pace,
            team_net_rating=team_net_rating,
        )

        multiplier = fit_multiplier(replacement_score)

        offensive_impact = candidate_value["offensive_value"] * multiplier
        defensive_impact = candidate_value["defensive_value"] * multiplier
        pace_impact = candidate_value["pace_value"] * multiplier

        capped_components = cap_component_impacts(
            offensive_impact=offensive_impact,
            defensive_impact=defensive_impact,
            pace_impact=pace_impact,
            caps=impact_caps,
        )

        capped_net = cap_net_impact(
            offensive_impact=capped_components["offensive_impact"],
            defensive_impact=capped_components["defensive_impact"],
            pace_impact=capped_components["pace_impact"],
            caps=impact_caps,
        )

        return {
            **capped_net,
            "net_impact": capped_net["estimated_net_impact"],
            "candidate_value": candidate_value["total_value"],
            "replaced_value": 0.0,
            "raw_net_delta": candidate_value["total_value"],
            "impact_mode": "absolute_legacy_capped",
            "fit_multiplier": multiplier,
            "comparison_minutes": get_expected_minutes(candidate),
        }

    # --------------------------------------------------------
    # Modo principal: impacto relativo.
    # --------------------------------------------------------
    comparison_minutes = get_comparison_minutes(
        candidate=candidate,
        replaced_player=replaced_player,
    )

    candidate_value = estimate_raw_player_value(
        player=candidate,
        league_avg=league_avg,
        team_pace=team_pace,
        team_net_rating=team_net_rating,
        minutes_override=comparison_minutes,
    )

    replaced_value = estimate_raw_player_value(
        player=replaced_player,
        league_avg=league_avg,
        team_pace=team_pace,
        team_net_rating=team_net_rating,
        minutes_override=comparison_minutes,
    )

    raw_offensive_delta = (
        candidate_value["offensive_value"] - replaced_value["offensive_value"]
    )

    raw_defensive_delta = (
        candidate_value["defensive_value"] - replaced_value["defensive_value"]
    )

    raw_pace_delta = candidate_value["pace_value"] - replaced_value["pace_value"]

    raw_net_delta = candidate_value["total_value"] - replaced_value["total_value"]

    multiplier = fit_multiplier(replacement_score)

    offensive_impact = raw_offensive_delta * multiplier
    defensive_impact = raw_defensive_delta * multiplier
    pace_impact = raw_pace_delta * multiplier

    capped_components = cap_component_impacts(
        offensive_impact=offensive_impact,
        defensive_impact=defensive_impact,
        pace_impact=pace_impact,
        caps=impact_caps,
    )

    capped_net = cap_net_impact(
        offensive_impact=capped_components["offensive_impact"],
        defensive_impact=capped_components["defensive_impact"],
        pace_impact=capped_components["pace_impact"],
        caps=impact_caps,
    )

    return {
        "offensive_impact": capped_net["offensive_impact"],
        "defensive_impact": capped_net["defensive_impact"],
        "pace_impact": capped_net["pace_impact"],
        "estimated_net_impact": capped_net["estimated_net_impact"],
        "net_impact": capped_net["estimated_net_impact"],
        "candidate_value": candidate_value["total_value"],
        "replaced_value": replaced_value["total_value"],
        "candidate_offensive_value": candidate_value["offensive_value"],
        "replaced_offensive_value": replaced_value["offensive_value"],
        "candidate_defensive_value": candidate_value["defensive_value"],
        "replaced_defensive_value": replaced_value["defensive_value"],
        "candidate_pace_value": candidate_value["pace_value"],
        "replaced_pace_value": replaced_value["pace_value"],
        "raw_offensive_delta": raw_offensive_delta,
        "raw_defensive_delta": raw_defensive_delta,
        "raw_pace_delta": raw_pace_delta,
        "raw_net_delta": raw_net_delta,
        "impact_mode": "relative",
        "fit_multiplier": multiplier,
        "comparison_minutes": comparison_minutes,
        "cap_applied": capped_net["cap_applied"],
    }
