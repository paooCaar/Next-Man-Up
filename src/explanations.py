"""
explanations.py

Este módulo genera explicaciones textuales para que el modelo sea interpretable.

La idea es que el dashboard no solo muestre números, sino también:
- Por qué se eligió un candidato.
- Qué componente tuvo más peso.
- Cómo cambia la probabilidad de ganar.
"""

from __future__ import annotations


def format_percent(value: float) -> str:
    """Convierte 0.612 en 61.2%."""
    return f"{value * 100:.1f}%"


def format_points(value: float) -> str:
    """Formatea puntos con signo."""
    return f"{value:+.2f}"


def identify_main_strength(score_data: dict) -> str:
    """
    Identifica el componente más fuerte del candidato.
    """
    components = {
        "similitud de rol": score_data.get("role_similarity", 0.0),
        "fit posicional": score_data.get("position_fit", 0.0),
        "fit con el equipo": score_data.get("team_fit", 0.0),
        "fit contra el rival": score_data.get("opponent_fit", 0.0),
    }

    return max(components, key=components.get)


def identify_main_weakness(score_data: dict) -> str:
    """
    Identifica el componente más débil del candidato.
    """
    components = {
        "similitud de rol": score_data.get("role_similarity", 0.0),
        "fit posicional": score_data.get("position_fit", 0.0),
        "fit con el equipo": score_data.get("team_fit", 0.0),
        "fit contra el rival": score_data.get("opponent_fit", 0.0),
    }

    return min(components, key=components.get)


def explain_replacement_choice(
    candidate_name: str,
    score_data: dict,
    impact_data: dict | None = None,
    monte_carlo_data: dict | None = None,
) -> str:
    """
    Genera una explicación breve de por qué se eligió un reemplazo.
    """
    final_score = score_data.get("final_score", 0.0)
    role_similarity = score_data.get("role_similarity", 0.0)
    position_fit = score_data.get("position_fit", 0.0)
    team_fit = score_data.get("team_fit", 0.0)
    opponent_fit = score_data.get("opponent_fit", 0.0)

    main_strength = identify_main_strength(score_data)
    main_weakness = identify_main_weakness(score_data)

    explanation = (
        f"{candidate_name} aparece como candidato porque obtiene un score final "
        f"de {final_score:.2f}. Su similitud de rol es {role_similarity:.2f}, "
        f"su fit posicional es {position_fit:.2f}, su fit con el equipo es "
        f"{team_fit:.2f} y su fit contra el rival es {opponent_fit:.2f}. "
        f"Su principal fortaleza dentro del modelo es {main_strength}. "
    )

    if main_weakness != main_strength:
        explanation += (
            f"El punto más débil relativo es {main_weakness}, lo cual debe "
            f"considerarse al interpretar la recomendación. "
        )

    if impact_data is not None:
        offensive_impact = impact_data.get("offensive_impact", 0.0)
        defensive_impact = impact_data.get("defensive_impact", 0.0)
        pace_impact = impact_data.get("pace_impact", 0.0)
        net_impact = impact_data.get(
            "estimated_net_impact", impact_data.get("net_impact", 0.0)
        )

        explanation += (
            f"En términos de impacto, se estima un aporte ofensivo de "
            f"{format_points(offensive_impact)} puntos, un aporte defensivo de "
            f"{format_points(defensive_impact)} puntos y un ajuste de ritmo de "
            f"{format_points(pace_impact)} puntos. El impacto neto estimado es "
            f"{format_points(net_impact)} puntos de margen. "
        )

    if monte_carlo_data is not None:
        win_without = monte_carlo_data.get("win_probability_without", 0.0)
        win_with = monte_carlo_data.get("win_probability_with", 0.0)
        win_delta = monte_carlo_data.get("win_probability_delta", 0.0)

        direction = "aumenta" if win_delta >= 0 else "reduce"

        explanation += (
            f"En la simulación Monte Carlo, la probabilidad de ganar pasa de "
            f"{format_percent(win_without)} a {format_percent(win_with)}, "
            f"lo que {direction} la probabilidad en "
            f"{win_delta * 100:+.1f} puntos porcentuales."
        )

    return explanation


def explain_score_formula() -> str:
    """
    Explicación corta de la fórmula de replacement_score.
    """
    return (
        "El score final de reemplazo combina cuatro componentes: "
        "70% similitud de rol, 15% compatibilidad posicional, "
        "10% fit con las necesidades del equipo y 5% fit contra el rival. "
        "Esto hace que el modelo priorice reemplazos similares, pero sin ignorar "
        "el contexto del equipo y del partido."
    )


def explain_monte_carlo_result(monte_carlo_data: dict) -> str:
    """
    Explica el resultado general de la simulación Monte Carlo.
    """
    win_without = monte_carlo_data.get("win_probability_without", 0.0)
    win_with = monte_carlo_data.get("win_probability_with", 0.0)
    margin_without = monte_carlo_data.get("expected_margin_without", 0.0)
    margin_with = monte_carlo_data.get("expected_margin_with", 0.0)
    margin_delta = monte_carlo_data.get("expected_margin_delta", 0.0)

    return (
        f"Sin reemplazo, la probabilidad simulada de ganar es "
        f"{format_percent(win_without)} con un margen esperado de "
        f"{format_points(margin_without)} puntos. Con el reemplazo, la probabilidad "
        f"simulada es {format_percent(win_with)} y el margen esperado cambia a "
        f"{format_points(margin_with)} puntos. La diferencia esperada en margen es "
        f"{format_points(margin_delta)} puntos."
    )


def summarize_top_candidates(results: list[dict], top_n: int = 3) -> str:
    """
    Crea un resumen textual del top de candidatos.
    """
    selected = results[:top_n]

    if not selected:
        return "No se encontraron candidatos disponibles."

    lines = ["Resumen de candidatos recomendados:"]

    for i, result in enumerate(selected, start=1):
        name = result.get("player_name", result.get("name", f"Candidato {i}"))
        final_score = result.get("final_score", 0.0)
        role_similarity = result.get("role_similarity", 0.0)

        lines.append(
            f"{i}. {name}: score final {final_score:.2f}, "
            f"similitud de rol {role_similarity:.2f}."
        )

    return "\n".join(lines)
