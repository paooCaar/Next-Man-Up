# app.py

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


# ============================================================
# 1. CONFIGURACIÓN E IMPORTS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

try:
    from src.recommender import load_processed_data, recommend_replacements
except Exception as error:
    st.set_page_config(page_title="Next Man Up", page_icon="🏀", layout="wide")
    st.error("No se pudo importar `src.recommender`.")
    st.exception(error)
    st.stop()


# ============================================================
# 2. CONFIGURACIÓN DE STREAMLIT
# ============================================================

st.set_page_config(
    page_title="Next Man Up - NBA Recommender",
    page_icon="🏀",
    layout="wide",
)


# ============================================================
# 3. HELPERS
# ============================================================


def format_probability(value: Any) -> str:
    try:
        return f"{float(value) * 100:.2f}%"
    except Exception:
        return "N/A"


def format_delta_pp(value: Any) -> str:
    try:
        return f"{float(value) * 100:+.2f} pp"
    except Exception:
        return "N/A"


def format_number(value: Any, decimals: int = 3) -> str:
    try:
        return f"{float(value):.{decimals}f}"
    except Exception:
        return "N/A"


def format_player_label(row: pd.Series) -> str:
    player_name = row.get("player_name", "N/A")
    position = row.get("position", "N/A")
    games_played = row.get("games_played", 0)
    minutes = row.get("minutes", 0)

    try:
        games_played = int(games_played)
    except Exception:
        games_played = 0

    try:
        minutes = float(minutes)
    except Exception:
        minutes = 0.0

    return f"{player_name} | {position} | {games_played} GP | {minutes:.1f} MIN"


def get_distribution_margins(distribution: dict) -> list[float]:
    if not isinstance(distribution, dict):
        return []

    margins = distribution.get("margins", [])

    if margins is None:
        return []

    try:
        return list(margins)
    except Exception:
        return []


def validate_processed_files(processed_dir: Path) -> list[str]:
    required_files = [
        processed_dir / "processed_players.csv",
        processed_dir / "processed_teams.csv",
        processed_dir / "processed_matchups.csv",
    ]

    return [str(path) for path in required_files if not path.exists()]


@st.cache_data(show_spinner=False)
def cached_load_processed_data(processed_dir: str):
    """
    Carga datos usando el backend real.

    No reimplementa preprocessing, similarity, impact ni Monte Carlo.
    """
    return load_processed_data(processed_dir)


def build_summary_dataframe(
    top_replacements: list[dict],
    baseline_probability: float | None,
) -> pd.DataFrame:
    rows = []

    for replacement in top_replacements:
        win_probability = replacement.get("win_probability_with_replacement")

        win_delta = None
        if baseline_probability is not None and win_probability is not None:
            win_delta = float(win_probability) - float(baseline_probability)

        rows.append(
            {
                "Jugador": replacement.get("player_name"),
                "Equipo": replacement.get("team"),
                "Posición": replacement.get("position"),
                "GP": replacement.get("games_played"),
                "MIN": replacement.get("minutes"),
                "PTS": replacement.get("points"),
                "replacement_score": replacement.get("replacement_score"),
                "role_similarity": replacement.get("role_similarity"),
                "position_fit": replacement.get("position_fit"),
                "team_fit": replacement.get("team_fit"),
                "opponent_fit": replacement.get("opponent_fit"),
                "estimated_net_impact": replacement.get("estimated_net_impact"),
                "offensive_impact": replacement.get("offensive_impact"),
                "defensive_impact": replacement.get("defensive_impact"),
                "pace_impact": replacement.get("pace_impact"),
                "win_probability": win_probability,
                "win_delta_pp": win_delta * 100 if win_delta is not None else None,
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# 4. HEADER
# ============================================================

st.title("🏀 Next Man Up: NBA Player Replacement Recommender")

st.markdown(
    """
Esta app recomienda reemplazos para un jugador NBA combinando:

- similitud de rol,
- compatibilidad posicional,
- ajuste al equipo,
- ajuste contra el rival,
- impacto relativo respecto al jugador reemplazado,
- simulación Monte Carlo para estimar probabilidad de victoria.

La app usa `src/recommender.py` como motor principal.  
No recalcula similarity, impact ni Monte Carlo dentro de Streamlit.
"""
)


# ============================================================
# 5. SIDEBAR
# ============================================================

st.sidebar.header("Configuración")

processed_dir_input = st.sidebar.text_input(
    "Carpeta de datos procesados",
    value="data/processed",
)

processed_dir = Path(processed_dir_input)

show_debug = st.sidebar.checkbox(
    "Mostrar debug crudo",
    value=False,
)

missing_files = validate_processed_files(processed_dir)

if missing_files:
    st.error("Faltan archivos procesados.")
    st.write("Archivos faltantes:")
    st.write(missing_files)
    st.info("Ejecuta primero: `python src/preprocessing.py`")
    st.stop()

try:
    players_df, teams_df, matchups_df = cached_load_processed_data(str(processed_dir))
except Exception as error:
    st.error("Ocurrió un error al cargar los datos procesados.")
    st.exception(error)
    st.stop()


# ============================================================
# 6. RESUMEN DE DATOS
# ============================================================

with st.expander("Resumen de datos cargados", expanded=False):
    col1, col2, col3 = st.columns(3)

    col1.metric("Jugadores", f"{len(players_df):,}")
    col2.metric("Equipos", f"{len(teams_df):,}")
    col3.metric("Matchups", f"{len(matchups_df):,}")

    if "position" in players_df.columns:
        position_counts = (
            players_df["position"].fillna("UNK").astype(str).str.upper().value_counts()
        )

        unk_count = int(position_counts.get("UNK", 0))
        unk_pct = unk_count / len(players_df) * 100 if len(players_df) else 0

        st.write("Distribución de posiciones:")
        st.dataframe(
            position_counts.rename("count"),
            use_container_width=True,
        )

        if unk_count == 0:
            st.success("No hay jugadores con position = UNK.")
        else:
            st.warning(
                f"Hay {unk_count} jugadores con position = UNK " f"({unk_pct:.2f}%)."
            )


# ============================================================
# 7. SELECTORES DEL ESCENARIO
# ============================================================

st.header("1. Define el escenario")

if "team" not in teams_df.columns:
    st.error("`processed_teams.csv` no tiene columna `team`.")
    st.stop()

teams_available = teams_df["team"].dropna().astype(str).sort_values().unique().tolist()

if not teams_available:
    st.error("No hay equipos disponibles.")
    st.stop()

default_team_index = teams_available.index("LAL") if "LAL" in teams_available else 0

col_team, col_player, col_opponent = st.columns(3)

with col_team:
    selected_team = st.selectbox(
        "Equipo propio",
        options=teams_available,
        index=default_team_index,
    )

team_players_df = players_df[
    players_df["team"].astype(str).str.upper() == selected_team.upper()
].copy()

if team_players_df.empty:
    st.warning(f"No se encontraron jugadores para {selected_team}.")
    st.stop()

team_players_df = team_players_df.sort_values(
    by=["games_played", "minutes", "points"],
    ascending=[False, False, False],
)

player_label_to_name = {}
player_labels = []

for _, row in team_players_df.iterrows():
    label = format_player_label(row)
    player_labels.append(label)
    player_label_to_name[label] = row.get("player_name")

default_player_index = 0

for idx, label in enumerate(player_labels):
    if "LeBron James" in label:
        default_player_index = idx
        break

with col_player:
    selected_player_label = st.selectbox(
        "Jugador a reemplazar",
        options=player_labels,
        index=default_player_index,
    )

selected_player_name = player_label_to_name[selected_player_label]

opponent_options = [team for team in teams_available if team != selected_team]

default_opponent_index = (
    opponent_options.index("BOS") if "BOS" in opponent_options else 0
)

with col_opponent:
    opponent_team = st.selectbox(
        "Equipo rival",
        options=opponent_options,
        index=default_opponent_index,
    )


# ============================================================
# 8. PARÁMETROS DEL MODELO
# ============================================================

st.header("2. Parámetros de recomendación")

col_sim, col_games, col_minutes, col_top = st.columns(4)

with col_sim:
    num_simulations = st.number_input(
        "Simulaciones Monte Carlo",
        min_value=1000,
        max_value=50000,
        value=10000,
        step=1000,
    )

with col_games:
    min_games = st.number_input(
        "Mínimo de partidos",
        min_value=1,
        max_value=500,
        value=50,
        step=1,
    )

with col_minutes:
    min_minutes = st.number_input(
        "Mínimo de minutos",
        min_value=1.0,
        max_value=48.0,
        value=15.0,
        step=1.0,
    )

with col_top:
    top_n = st.number_input(
        "Número de reemplazos",
        min_value=1,
        max_value=10,
        value=3,
        step=1,
    )

st.caption(
    "Defaults recomendados para la versión estable: "
    "10,000 simulaciones, min_games=50, min_minutes=15, top_n=3."
)


# ============================================================
# 9. EJECUCIÓN
# ============================================================

st.header("3. Ejecutar recomendación")

params = {
    "equipo_propio": selected_team,
    "jugador_reemplazado": selected_player_name,
    "equipo_rival": opponent_team,
    "simulaciones": int(num_simulations),
    "min_games": int(min_games),
    "min_minutes": float(min_minutes),
    "top_n": int(top_n),
}

with st.expander("Parámetros seleccionados", expanded=False):
    st.write(params)

run_button = st.button(
    "Ejecutar recomendación",
    type="primary",
    use_container_width=True,
)

if run_button:
    try:
        with st.spinner("Ejecutando modelo..."):
            result = recommend_replacements(
                selected_team_value=selected_team,
                player_to_replace_value=selected_player_name,
                opponent_team_value=opponent_team,
                num_simulations=int(num_simulations),
                processed_dir=str(processed_dir),
                top_n=int(top_n),
                min_minutes=float(min_minutes),
                min_games=int(min_games),
                exclude_opponent_players=True,
                random_state=42,
            )

        st.session_state["last_result"] = result
        st.session_state["last_params"] = params

    except Exception as error:
        st.error("No se pudo ejecutar la recomendación.")
        st.exception(error)
        st.warning(
            "Posibles causas: filtros demasiado estrictos, datos faltantes, "
            "jugador no encontrado o CSV procesados desactualizados."
        )
        st.stop()

if "last_result" not in st.session_state:
    st.info("Presiona el botón para ejecutar la recomendación.")
    st.stop()

result = st.session_state["last_result"]

if show_debug:
    st.header("Debug crudo")
    st.write(result)

if result is None:
    st.warning("El recomendador devolvió None.")
    st.stop()

if not isinstance(result, dict):
    st.warning("El recomendador no devolvió un diccionario.")
    st.write(result)
    st.stop()

top_replacements = result.get("top_replacements", [])

if not top_replacements:
    st.warning("No se encontraron reemplazos. " "Prueba bajar min_games o min_minutes.")
    st.stop()


# ============================================================
# 10. RESULTADO GENERAL
# ============================================================

st.header("4. Resultado del escenario")

selected_team_result = result.get("selected_team", {})
opponent_team_result = result.get("opponent_team", {})
replaced_player_result = result.get("replaced_player", {})
baseline_result = result.get("baseline", {})

baseline_probability = baseline_result.get(
    "win_probability_without_replacement",
    None,
)

baseline_margin = baseline_result.get(
    "expected_margin_without_replacement",
    None,
)

summary_df = build_summary_dataframe(
    top_replacements=top_replacements,
    baseline_probability=baseline_probability,
)

best_replacement = top_replacements[0]
best_win_probability = best_replacement.get("win_probability_with_replacement")

best_delta = None
if baseline_probability is not None and best_win_probability is not None:
    best_delta = float(best_win_probability) - float(baseline_probability)

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Equipo",
    selected_team_result.get("team", selected_team),
)

col2.metric(
    "Jugador reemplazado",
    replaced_player_result.get("player_name", selected_player_name),
)

col3.metric(
    "Rival",
    opponent_team_result.get("team", opponent_team),
)

col4.metric(
    "Probabilidad sin reemplazo",
    format_probability(baseline_probability),
)

col5, col6, col7 = st.columns(3)

col5.metric(
    "Mejor reemplazo",
    best_replacement.get("player_name", "N/A"),
)

col6.metric(
    "Probabilidad con mejor reemplazo",
    format_probability(best_win_probability),
    delta=format_delta_pp(best_delta) if best_delta is not None else None,
)

col7.metric(
    "Impacto neto estimado",
    format_number(best_replacement.get("estimated_net_impact"), 3),
)

if baseline_margin is not None:
    st.caption(f"Margen esperado sin reemplazo: {format_number(baseline_margin, 2)}")


# ============================================================
# 11. TABLA DEL TOP
# ============================================================

st.header("5. Top reemplazos")

display_df = summary_df.copy()

numeric_columns = [
    "MIN",
    "PTS",
    "replacement_score",
    "role_similarity",
    "position_fit",
    "team_fit",
    "opponent_fit",
    "estimated_net_impact",
    "offensive_impact",
    "defensive_impact",
    "pace_impact",
    "win_probability",
    "win_delta_pp",
]

for col in numeric_columns:
    if col in display_df.columns:
        display_df[col] = pd.to_numeric(display_df[col], errors="coerce").round(3)

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True,
)


# ============================================================
# 12. DETALLE POR CANDIDATO
# ============================================================

st.header("6. Explicaciones individuales")

for idx, replacement in enumerate(top_replacements, start=1):
    player_name = replacement.get("player_name", f"Candidato {idx}")
    player_team = replacement.get("team", "N/A")
    player_position = replacement.get("position", "N/A")

    with st.expander(
        f"#{idx} - {player_name} ({player_team}, {player_position})",
        expanded=(idx == 1),
    ):
        metric_cols = st.columns(5)

        metric_cols[0].metric(
            "Replacement score",
            format_number(replacement.get("replacement_score")),
        )

        metric_cols[1].metric(
            "Role similarity",
            format_number(replacement.get("role_similarity")),
        )

        metric_cols[2].metric(
            "Position fit",
            format_number(replacement.get("position_fit")),
        )

        metric_cols[3].metric(
            "Impacto neto",
            format_number(replacement.get("estimated_net_impact")),
        )

        metric_cols[4].metric(
            "Win probability",
            format_probability(replacement.get("win_probability_with_replacement")),
        )

        impact_cols = st.columns(3)

        impact_cols[0].metric(
            "Impacto ofensivo",
            format_number(replacement.get("offensive_impact")),
        )

        impact_cols[1].metric(
            "Impacto defensivo",
            format_number(replacement.get("defensive_impact")),
        )

        impact_cols[2].metric(
            "Impacto de ritmo",
            format_number(replacement.get("pace_impact")),
        )

        fit_cols = st.columns(2)

        fit_cols[0].metric(
            "Team fit",
            format_number(replacement.get("team_fit")),
        )

        fit_cols[1].metric(
            "Opponent fit",
            format_number(replacement.get("opponent_fit")),
        )

        st.markdown("**Explicación**")
        st.write(replacement.get("explanation", "Sin explicación disponible."))


# ============================================================
# 13. VISUALIZACIONES
# ============================================================

st.header("7. Visualizaciones")

# ------------------------------------------------------------
# 13.1 Probabilidad baseline vs reemplazos
# ------------------------------------------------------------

probability_rows = [
    {
        "Escenario": "Sin reemplazo",
        "Probabilidad de ganar (%)": float(baseline_probability) * 100,
    }
]

for replacement in top_replacements:
    probability_rows.append(
        {
            "Escenario": replacement.get("player_name", "Candidato"),
            "Probabilidad de ganar (%)": float(
                replacement.get("win_probability_with_replacement", 0)
            )
            * 100,
        }
    )

probability_df = pd.DataFrame(probability_rows)

st.subheader("Probabilidad de ganar: baseline vs reemplazos")

st.bar_chart(
    probability_df.set_index("Escenario")["Probabilidad de ganar (%)"],
    use_container_width=True,
)


# ------------------------------------------------------------
# 13.2 Scores e impacto
# ------------------------------------------------------------

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.subheader("Replacement score")

    score_chart_df = summary_df[["Jugador", "replacement_score"]].copy()
    score_chart_df = score_chart_df.set_index("Jugador")

    st.bar_chart(score_chart_df, use_container_width=True)

with chart_col2:
    st.subheader("Impacto neto estimado")

    impact_chart_df = summary_df[["Jugador", "estimated_net_impact"]].copy()
    impact_chart_df = impact_chart_df.set_index("Jugador")

    st.bar_chart(impact_chart_df, use_container_width=True)


# ------------------------------------------------------------
# 13.3 Distribución de márgenes simulados
# ------------------------------------------------------------

st.subheader("Distribución de márgenes simulados")

baseline_distribution = baseline_result.get("simulation_distribution", {})
baseline_margins = get_distribution_margins(baseline_distribution)

replacement_names = [
    replacement.get("player_name", f"Candidato {idx}")
    for idx, replacement in enumerate(top_replacements, start=1)
]

selected_distribution_player = st.selectbox(
    "Selecciona un reemplazo para comparar distribución",
    options=replacement_names,
)

selected_replacement = next(
    (
        replacement
        for replacement in top_replacements
        if replacement.get("player_name") == selected_distribution_player
    ),
    None,
)

if selected_replacement is not None:
    replacement_distribution = selected_replacement.get(
        "simulation_distribution",
        {},
    )

    replacement_margins = get_distribution_margins(replacement_distribution)

    if baseline_margins and replacement_margins:
        fig, ax = plt.subplots(figsize=(10, 5))

        ax.hist(
            baseline_margins,
            bins=40,
            alpha=0.55,
            label="Sin reemplazo",
        )

        ax.hist(
            replacement_margins,
            bins=40,
            alpha=0.55,
            label=f"Con {selected_distribution_player}",
        )

        ax.axvline(0, linestyle="--", linewidth=1)
        ax.set_xlabel("Margen simulado")
        ax.set_ylabel("Frecuencia")
        ax.set_title("Distribución de márgenes simulados")
        ax.legend()

        st.pyplot(fig)

        baseline_series = pd.Series(baseline_margins)
        replacement_series = pd.Series(replacement_margins)

        dist_col1, dist_col2 = st.columns(2)

        dist_col1.metric(
            "Margen promedio baseline",
            format_number(baseline_series.mean(), 2),
        )

        dist_col2.metric(
            f"Margen promedio con {selected_distribution_player}",
            format_number(replacement_series.mean(), 2),
        )

    else:
        st.info("No hay márgenes simulados suficientes para graficar.")


# ============================================================
# 14. EXPLICACIÓN GLOBAL
# ============================================================

st.header("8. Explicación global")

st.write(result.get("explanation", "Sin explicación global disponible."))


# ============================================================
# 15. VALIDACIONES RÁPIDAS
# ============================================================

with st.expander("Validaciones rápidas del modelo", expanded=False):
    duplicate_names = summary_df["Jugador"].duplicated().any()

    max_abs_impact = (
        pd.to_numeric(
            summary_df["estimated_net_impact"],
            errors="coerce",
        )
        .abs()
        .max()
    )

    score_valid = (
        pd.to_numeric(
            summary_df["replacement_score"],
            errors="coerce",
        )
        .between(0, 1)
        .all()
    )

    win_prob_valid = (
        pd.to_numeric(
            summary_df["win_probability"],
            errors="coerce",
        )
        .between(0, 1)
        .all()
    )

    no_unk = not summary_df["Posición"].astype(str).str.upper().eq("UNK").any()

    validations = pd.DataFrame(
        [
            {
                "Validación": "Sin jugadores duplicados en el top",
                "Resultado": "OK" if not duplicate_names else "Revisar",
            },
            {
                "Validación": "replacement_score entre 0 y 1",
                "Resultado": "OK" if score_valid else "Revisar",
            },
            {
                "Validación": "win_probability entre 0 y 1",
                "Resultado": "OK" if win_prob_valid else "Revisar",
            },
            {
                "Validación": "estimated_net_impact dentro de [-8, +8]",
                "Resultado": "OK" if max_abs_impact <= 8 else "Revisar",
            },
            {
                "Validación": "Sin UNK en top",
                "Resultado": "OK" if no_unk else "Revisar",
            },
        ]
    )

    st.dataframe(
        validations,
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "Rangos esperados: replacement_score 0-1, win_probability 0-1, "
        "estimated_net_impact aproximadamente entre -8 y +8."
    )
