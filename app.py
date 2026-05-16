# app.py

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import pandas as pd
import streamlit as st


# ============================================================
# 1. CONFIGURACIÓN BÁSICA
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

st.set_page_config(
    page_title="Next Man Up - Debug",
    page_icon="🏀",
    layout="wide",
)


# ============================================================
# 2. IMPORTAR BACKEND
# ============================================================

try:
    from src.recommender import load_processed_data, recommend_replacements
except Exception as error:
    st.error("Error importando src.recommender.")
    st.exception(error)
    st.stop()


# ============================================================
# 3. HELPERS
# ============================================================


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


def validate_processed_files(processed_dir: Path) -> list[str]:
    required_files = [
        processed_dir / "processed_players.csv",
        processed_dir / "processed_teams.csv",
        processed_dir / "processed_matchups.csv",
    ]

    return [str(path) for path in required_files if not path.exists()]


@st.cache_data(show_spinner=False)
def cached_load_processed_data(processed_dir: str):
    return load_processed_data(processed_dir)


# ============================================================
# 4. HEADER
# ============================================================

st.title("🏀 Next Man Up - Debug de Streamlit")

st.write(
    "Esta versión de la app sirve para depurar la conexión entre Streamlit "
    "y `src/recommender.py`. No reimplementa similarity, impact ni Monte Carlo."
)


# ============================================================
# 5. CONFIGURACIÓN DE DATOS
# ============================================================

st.sidebar.header("Configuración")

processed_dir_input = st.sidebar.text_input(
    "Carpeta de datos procesados",
    value="data/processed",
)

processed_dir = Path(processed_dir_input)

missing_files = validate_processed_files(processed_dir)

if missing_files:
    st.error("Faltan archivos procesados.")
    st.write(missing_files)
    st.info("Ejecuta primero: python src/preprocessing.py")
    st.stop()


try:
    players_df, teams_df, matchups_df = cached_load_processed_data(str(processed_dir))
except Exception as error:
    st.error("Error cargando datos procesados.")
    st.exception(error)
    st.stop()


st.success("Datos procesados cargados correctamente.")

with st.expander("Debug: resumen de datos cargados", expanded=True):
    col1, col2, col3 = st.columns(3)

    col1.metric("Jugadores", f"{len(players_df):,}")
    col2.metric("Equipos", f"{len(teams_df):,}")
    col3.metric("Matchups", f"{len(matchups_df):,}")

    st.write("Columnas players_df:")
    st.write(players_df.columns.tolist())

    st.write("Columnas teams_df:")
    st.write(teams_df.columns.tolist())

    st.write("Columnas matchups_df:")
    st.write(matchups_df.columns.tolist())

    if "position" in players_df.columns:
        st.write("Distribución de posiciones:")
        st.write(
            players_df["position"].fillna("UNK").astype(str).str.upper().value_counts()
        )


# ============================================================
# 6. SELECTORES
# ============================================================

st.header("1. Selección del escenario")

if "team" not in teams_df.columns:
    st.error("teams_df no tiene columna 'team'.")
    st.stop()

teams_available = teams_df["team"].dropna().astype(str).sort_values().unique().tolist()

if not teams_available:
    st.error("No hay equipos disponibles.")
    st.stop()

default_team_index = teams_available.index("LAL") if "LAL" in teams_available else 0

selected_team = st.selectbox(
    "Equipo propio",
    options=teams_available,
    index=default_team_index,
)

team_players_df = players_df[
    players_df["team"].astype(str).str.upper() == selected_team.upper()
].copy()

if team_players_df.empty:
    st.warning(f"No hay jugadores disponibles para {selected_team}.")
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

opponent_team = st.selectbox(
    "Equipo rival",
    options=opponent_options,
    index=default_opponent_index,
)


# ============================================================
# 7. PARÁMETROS
# ============================================================

st.header("2. Parámetros del modelo")

col1, col2, col3, col4 = st.columns(4)

with col1:
    num_simulations = st.number_input(
        "Simulaciones",
        min_value=1000,
        max_value=50000,
        value=10000,
        step=1000,
    )

with col2:
    min_games = st.number_input(
        "min_games",
        min_value=1,
        max_value=500,
        value=50,
        step=1,
    )

with col3:
    min_minutes = st.number_input(
        "min_minutes",
        min_value=1.0,
        max_value=48.0,
        value=15.0,
        step=1.0,
    )

with col4:
    top_n = st.number_input(
        "top_n",
        min_value=1,
        max_value=10,
        value=3,
        step=1,
    )


# ============================================================
# 8. MOSTRAR PARÁMETROS ANTES DE EJECUTAR
# ============================================================

st.header("3. Parámetros seleccionados")

debug_params = {
    "equipo_propio": selected_team,
    "jugador_reemplazado": selected_player_name,
    "equipo_rival": opponent_team,
    "simulaciones": int(num_simulations),
    "min_games": int(min_games),
    "min_minutes": float(min_minutes),
    "top_n": int(top_n),
    "processed_dir": str(processed_dir),
}

st.write(debug_params)


# ============================================================
# 9. BOTÓN Y EJECUCIÓN
# ============================================================

st.header("4. Ejecutar")

run_button = st.button(
    "Ejecutar recomendación",
    type="primary",
    use_container_width=True,
)

if run_button:
    st.success("Botón presionado")

    st.write("Debug: parámetros enviados a recommend_replacements()")
    st.write(debug_params)

    try:
        with st.spinner("Corriendo modelo con recommend_replacements()..."):
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

        st.success("recommend_replacements() terminó correctamente.")

        st.header("5. Resultado crudo devuelto por recommender.py")
        st.write(result)

        if result is None:
            st.warning("El resultado vino como None.")
            st.stop()

        if not isinstance(result, dict):
            st.warning(
                "El resultado no es un diccionario. "
                "Revisa la salida de recommend_replacements()."
            )
            st.stop()

        top_replacements = result.get("top_replacements", [])

        if not top_replacements:
            st.warning(
                "El resultado no contiene candidatos en 'top_replacements'. "
                "Puede que los filtros hayan dejado cero candidatos."
            )
            st.stop()

        st.header("6. Validación rápida del resultado")

        baseline = result.get("baseline", {})
        baseline_probability = baseline.get("win_probability_without_replacement")

        st.write("Baseline:")
        st.write(baseline)

        st.write("Top replacements:")
        st.write(top_replacements)

        st.subheader("Resumen legible")

        rows = []

        for replacement in top_replacements:
            rows.append(
                {
                    "player_name": replacement.get("player_name"),
                    "team": replacement.get("team"),
                    "position": replacement.get("position"),
                    "replacement_score": replacement.get("replacement_score"),
                    "role_similarity": replacement.get("role_similarity"),
                    "position_fit": replacement.get("position_fit"),
                    "team_fit": replacement.get("team_fit"),
                    "opponent_fit": replacement.get("opponent_fit"),
                    "estimated_net_impact": replacement.get("estimated_net_impact"),
                    "offensive_impact": replacement.get("offensive_impact"),
                    "defensive_impact": replacement.get("defensive_impact"),
                    "pace_impact": replacement.get("pace_impact"),
                    "win_probability_with_replacement": replacement.get(
                        "win_probability_with_replacement"
                    ),
                }
            )

        summary_df = pd.DataFrame(rows)

        st.dataframe(summary_df, use_container_width=True, hide_index=True)

        st.write("Probabilidad baseline:")
        st.write(baseline_probability)

        st.subheader("Explicaciones individuales")

        for idx, replacement in enumerate(top_replacements, start=1):
            st.markdown(f"### #{idx} - {replacement.get('player_name')}")
            st.write(replacement.get("explanation", "Sin explicación."))

        st.subheader("Explicación global")
        st.write(result.get("explanation", "Sin explicación global."))

    except Exception as error:
        st.error("Ocurrió un error durante recommend_replacements().")
        st.exception(error)

else:
    st.info("Presiona el botón para ejecutar la recomendación.")
