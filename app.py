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
    from src.recommender import load_recommendation_data, recommend_replacements
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
# 3. HELPERS DE FORMATO
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



def safe_int(value: Any, default: int | None = None) -> int | None:
    try:
        if pd.isna(value):
            return default
        return int(value)
    except Exception:
        return default



def normalize_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().lower()



def format_player_label(row: pd.Series, include_id_if_needed: bool = False) -> str:
    player_name = row.get("player_name", "N/A")
    position = row.get("position", "N/A")
    games_played = safe_int(row.get("games_played", 0), default=0) or 0

    try:
        minutes = float(row.get("minutes", 0))
    except Exception:
        minutes = 0.0

    label = f"{player_name} | {position} | {games_played} GP | {minutes:.1f} MIN"

    if include_id_if_needed:
        label += f" | ID {safe_int(row.get('player_id'), default=0)}"

    return label



def build_player_label_maps(players_df: pd.DataFrame) -> tuple[list[str], dict[str, int], dict[int, str]]:
    """
    Construye etiquetas visibles y mapas internos.

    La UI muestra nombres y datos simples, pero internamente siempre usamos player_id.
    Si dos etiquetas fueran iguales, agregamos el ID al label para evitar ambigüedad.
    """
    base_labels = [format_player_label(row) for _, row in players_df.iterrows()]
    duplicated_labels = pd.Series(base_labels).duplicated(keep=False).tolist()

    labels: list[str] = []
    label_to_id: dict[str, int] = {}
    id_to_label: dict[int, str] = {}

    for duplicated, (_, row) in zip(duplicated_labels, players_df.iterrows()):
        player_id = safe_int(row.get("player_id"))
        if player_id is None:
            continue

        label = format_player_label(row, include_id_if_needed=duplicated)
        labels.append(label)
        label_to_id[label] = player_id
        id_to_label[player_id] = label

    return labels, label_to_id, id_to_label



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
        processed_dir / "player_profiles.csv",
        processed_dir / "app_roster.csv",
        processed_dir / "processed_teams.csv",
        processed_dir / "processed_matchups.csv",
    ]

    return [str(path) for path in required_files if not path.exists()]


@st.cache_data(show_spinner=False)
def cached_load_recommendation_data(processed_dir: str):
    """
    Carga los datos usando el backend real.

    app.py no recalcula similarity, impact ni Monte Carlo.
    Solo usa app_roster.csv para construir selectores y luego llama al backend.
    """
    return load_recommendation_data(processed_dir)



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
                "player_id": replacement.get("player_id"),
                "Jugador": replacement.get("player_name"),
                "Equipo": replacement.get("latest_team", replacement.get("team")),
                "Posición": replacement.get("position"),
                "GP": replacement.get("games_played"),
                "MIN": replacement.get("minutes"),
                "PTS": replacement.get("points"),
                "recommendation_score": replacement.get(
                    "recommendation_score", replacement.get("replacement_score")
                ),
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



def get_default_lineup_labels(
    selected_team: str,
    team_roster_df: pd.DataFrame,
    id_to_label: dict[int, str],
) -> list[str]:
    """
    Define defaults útiles para probar la app.

    Para LAL usamos la quinteta validada en consola. Para otros equipos usamos los
    primeros 5 jugadores ordenados por muestra/minutos/puntos.
    """
    if selected_team.upper() == "LAL":
        desired_names = [
            "LeBron James",
            "Anthony Davis",
            "Austin Reaves",
            "Dennis Schroder",
            "Lonnie Walker IV",
        ]

        selected_ids: list[int] = []
        for name in desired_names:
            match = team_roster_df[
                team_roster_df["player_name"].astype(str).map(normalize_text)
                == normalize_text(name)
            ]
            if not match.empty:
                player_id = safe_int(match.iloc[0].get("player_id"))
                if player_id is not None:
                    selected_ids.append(player_id)

        if len(selected_ids) == 5:
            return [id_to_label[player_id] for player_id in selected_ids if player_id in id_to_label]

    default_rows = team_roster_df.head(5)
    default_ids = [
        safe_int(row.get("player_id"))
        for _, row in default_rows.iterrows()
    ]

    return [id_to_label[player_id] for player_id in default_ids if player_id in id_to_label]


# ============================================================
# 4. HEADER E INSTRUCCIONES
# ============================================================

st.title("🏀 Next Man Up: NBA Player Replacement Recommender")

st.markdown(
    """
Selecciona tu equipo, define los **5 jugadores que están en cancha**, elige a quién
quieres reemplazar y la app recomendará candidatos disponibles desde la banca del
mismo equipo.
"""
)

st.subheader("¿Cómo usar Next Man Up?")

st.markdown(
    """
1. **Selecciona tu equipo** y el **equipo rival**.
2. **Elige exactamente 5 jugadores en cancha** desde el roster más reciente disponible en la base.
3. **Selecciona cuál de esos 5 jugadores quieres reemplazar**.
4. Ajusta simulaciones, mínimo de partidos, mínimo de minutos y cantidad de recomendaciones.
5. Presiona **Ejecutar recomendación** para obtener reemplazos desde la banca del mismo equipo.
"""
)

with st.expander("¿Cómo funciona el modelo por detrás?", expanded=False):
    st.markdown(
        """
La interfaz solo recolecta el escenario y llama al backend. El backend usa:

- `app_roster.csv` para saber quién pertenece al roster seleccionable del equipo.
- `player_profiles.csv` para calcular similitud, impacto y ranking con el historial completo del jugador.
- La quinteta seleccionada para excluir a los jugadores que ya están en cancha.
- Simulación Monte Carlo para estimar cambios en probabilidad de victoria.
"""
    )


# ============================================================
# 5. SIDEBAR: OPCIONES AVANZADAS
# ============================================================

with st.sidebar:
    st.title("🏀 Next Man Up")
    st.caption("Define la quinteta en cancha y ejecuta el modelo de reemplazo.")

    with st.expander("Configuración avanzada / modo desarrollador", expanded=False):
        developer_mode = st.checkbox(
            "Activar modo desarrollador",
            value=False,
            help="Muestra diagnósticos útiles para validar datos y resultados internos.",
        )

        processed_dir_input = st.text_input(
            "Carpeta de datos procesados",
            value="data/processed",
            help=(
                "Ruta donde se encuentran player_profiles.csv, app_roster.csv, "
                "processed_teams.csv y processed_matchups.csv."
            ),
        )

        show_debug = st.checkbox(
            "Mostrar debug crudo",
            value=False,
            help="Muestra el diccionario completo devuelto por src/recommender.py.",
        )

processed_dir = Path(processed_dir_input)
diagnostic_enabled = developer_mode or show_debug


# ============================================================
# 6. CARGA DE DATOS
# ============================================================

missing_files = validate_processed_files(processed_dir)

if missing_files:
    st.error("Faltan archivos procesados para la lógica de quinteta.")
    st.write("Archivos faltantes:")
    st.write(missing_files)
    st.info("Ejecuta primero: `python src/preprocessing.py`")
    st.stop()

try:
    player_profiles_df, teams_df, matchups_df, app_roster_df = cached_load_recommendation_data(
        str(processed_dir)
    )
except Exception as error:
    st.error("Ocurrió un error al cargar los datos procesados.")
    st.exception(error)
    st.stop()


# ============================================================
# 7. DIAGNÓSTICO DE DATOS
# ============================================================

if diagnostic_enabled:
    with st.expander("Resumen de datos cargados", expanded=False):
        st.caption("Diagnóstico de los archivos usados por la lógica nueva.")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Perfiles históricos", f"{len(player_profiles_df):,}")
        col2.metric("Roster app", f"{len(app_roster_df):,}")
        col3.metric("Equipos", f"{len(teams_df):,}")
        col4.metric("Matchups", f"{len(matchups_df):,}")

        if "latest_team" in app_roster_df.columns:
            st.write("Jugadores por equipo en app_roster.csv:")
            st.dataframe(
                app_roster_df["latest_team"]
                .fillna("UNK")
                .astype(str)
                .value_counts()
                .rename("count"),
                use_container_width=True,
            )

        if "position" in app_roster_df.columns:
            st.write("Distribución de posiciones en app_roster.csv:")
            st.dataframe(
                app_roster_df["position"]
                .fillna("UNK")
                .astype(str)
                .str.upper()
                .value_counts()
                .rename("count"),
                use_container_width=True,
            )


# ============================================================
# 8. SELECTORES DEL ESCENARIO
# ============================================================

st.header("1. Define el escenario")

required_app_roster_cols = {"player_id", "player_name", "latest_team", "position"}
missing_app_cols = required_app_roster_cols - set(app_roster_df.columns)

if missing_app_cols:
    st.error(f"`app_roster.csv` no tiene columnas necesarias: {sorted(missing_app_cols)}")
    st.stop()

teams_available = (
    app_roster_df["latest_team"]
    .dropna()
    .astype(str)
    .sort_values()
    .unique()
    .tolist()
)

if not teams_available:
    st.error("No hay equipos disponibles en app_roster.csv.")
    st.stop()

default_team_index = teams_available.index("LAL") if "LAL" in teams_available else 0

col_team, col_opponent = st.columns(2)

with col_team:
    selected_team = st.selectbox(
        "Equipo propio",
        options=teams_available,
        index=default_team_index,
        help="Equipo que necesita encontrar un reemplazo desde su banca.",
    )

team_roster_df = app_roster_df[
    app_roster_df["latest_team"].astype(str).str.upper() == selected_team.upper()
].copy()

if team_roster_df.empty:
    st.warning(f"No se encontraron jugadores para {selected_team} en app_roster.csv.")
    st.stop()

sort_columns = [col for col in ["games_played", "minutes", "points"] if col in team_roster_df.columns]

if sort_columns:
    team_roster_df = team_roster_df.sort_values(
        by=sort_columns,
        ascending=[False] * len(sort_columns),
    )

teams_for_opponent = teams_df["team"].dropna().astype(str).sort_values().unique().tolist()
opponent_options = [team for team in teams_for_opponent if team != selected_team]

if not opponent_options:
    st.error("No hay equipos rivales disponibles.")
    st.stop()

default_opponent_index = opponent_options.index("BOS") if "BOS" in opponent_options else 0

with col_opponent:
    opponent_team = st.selectbox(
        "Equipo rival",
        options=opponent_options,
        index=default_opponent_index,
        help="Rival contra el que se evaluará el reemplazo.",
    )

player_labels, player_label_to_id, player_id_to_label = build_player_label_maps(team_roster_df)

default_lineup_labels = get_default_lineup_labels(
    selected_team=selected_team,
    team_roster_df=team_roster_df,
    id_to_label=player_id_to_label,
)

st.subheader("Quinteta en cancha")

selected_lineup_labels = st.multiselect(
    "Selecciona exactamente 5 jugadores en cancha",
    options=player_labels,
    default=default_lineup_labels,
    help="Estos jugadores serán excluidos de las recomendaciones de banca.",
)

lineup_player_ids = [player_label_to_id[label] for label in selected_lineup_labels]
lineup_size = len(lineup_player_ids)

if lineup_size != 5:
    st.warning("Selecciona exactamente 5 jugadores en cancha.")
else:
    st.success("Quinteta válida: 5 jugadores seleccionados.")

replacement_label = None
replaced_player_id = None

if selected_lineup_labels:
    replacement_label = st.selectbox(
        "Jugador a reemplazar",
        options=selected_lineup_labels,
        index=0,
        help="Solo puedes reemplazar a uno de los 5 jugadores en cancha.",
        disabled=lineup_size != 5,
    )

    if replacement_label is not None:
        replaced_player_id = player_label_to_id.get(replacement_label)
else:
    st.info("Selecciona una quinteta para elegir el jugador a reemplazar.")

with st.expander("Ver quinteta seleccionada", expanded=False):
    if selected_lineup_labels:
        lineup_display_df = team_roster_df[
            team_roster_df["player_id"].astype(int).isin(lineup_player_ids)
        ][
            [
                "player_id",
                "player_name",
                "latest_team",
                "position",
                "games_played",
                "minutes",
                "points",
            ]
        ].copy()

        st.dataframe(lineup_display_df, use_container_width=True, hide_index=True)
    else:
        st.write("No hay jugadores seleccionados.")


# ============================================================
# 9. PARÁMETROS DEL MODELO
# ============================================================

st.header("2. Ajusta los parámetros")

st.markdown(
    """
Puedes dejar estos valores por defecto para una recomendación estable. Si quieres ampliar la banca disponible,
baja los mínimos de partidos o minutos; si quieres candidatos más consolidados, súbelos.
"""
)

col_sim, col_games, col_minutes, col_top = st.columns(4)

with col_sim:
    num_simulations = st.number_input(
        "Simulaciones Monte Carlo",
        min_value=1000,
        max_value=50000,
        value=10000,
        step=1000,
        help="Más simulaciones dan una estimación más estable, pero tardan más.",
    )

with col_games:
    min_games = st.number_input(
        "Mínimo de partidos",
        min_value=1,
        max_value=500,
        value=10,
        step=1,
        help="Filtra candidatos con poca muestra de partidos.",
    )

with col_minutes:
    min_minutes = st.number_input(
        "Mínimo de minutos",
        min_value=1.0,
        max_value=48.0,
        value=10.0,
        step=1.0,
        help="Filtra candidatos con pocos minutos promedio.",
    )

with col_top:
    top_n = st.number_input(
        "Número de reemplazos",
        min_value=1,
        max_value=10,
        value=3,
        step=1,
        help="Cantidad de candidatos a mostrar en el ranking final.",
    )

st.caption(
    "Valores recomendados para pruebas: 10,000 simulaciones, mínimo 10 partidos, mínimo 10 minutos y top 3 reemplazos."
)


# ============================================================
# 10. EJECUCIÓN
# ============================================================

st.header("3. Ejecuta la recomendación")

selected_replacement_name = "N/A"
if replaced_player_id is not None:
    replacement_row = team_roster_df[
        team_roster_df["player_id"].astype(int) == int(replaced_player_id)
    ]
    if not replacement_row.empty:
        selected_replacement_name = str(replacement_row.iloc[0].get("player_name", "N/A"))

params = {
    "equipo_propio": selected_team,
    "equipo_rival": opponent_team,
    "lineup_player_ids": lineup_player_ids,
    "replaced_player_id": replaced_player_id,
    "jugador_reemplazado": selected_replacement_name,
    "simulaciones": int(num_simulations),
    "min_games": int(min_games),
    "min_minutes": float(min_minutes),
    "top_n": int(top_n),
}

st.markdown(
    f"""
**Escenario actual:** `{selected_team}` enfrenta a `{opponent_team}`.
Jugador a reemplazar: **{selected_replacement_name}**.
"""
)

if diagnostic_enabled:
    with st.expander("Parámetros internos seleccionados", expanded=False):
        st.write(params)

can_run = lineup_size == 5 and replaced_player_id is not None

run_button = st.button(
    "Ejecutar recomendación",
    type="primary",
    use_container_width=True,
    disabled=not can_run,
)

if not can_run:
    st.info("Completa una quinteta de exactamente 5 jugadores y selecciona a quién reemplazar para ejecutar el modelo.")

if run_button:
    try:
        with st.spinner("Ejecutando modelo..."):
            result = recommend_replacements(
                selected_team_value=selected_team,
                opponent_team_value=opponent_team,
                lineup_player_ids=lineup_player_ids,
                replaced_player_id=int(replaced_player_id),
                monte_carlo_simulations=int(num_simulations),
                processed_dir=str(processed_dir),
                top_n=int(top_n),
                min_minutes=float(min_minutes),
                min_games=int(min_games),
                random_state=42,
            )

        st.session_state["last_result"] = result
        st.session_state["last_params"] = params

    except ValueError as error:
        st.error("No se pudo generar una recomendación con este escenario.")
        st.warning(str(error))
        st.info("Prueba bajar min_games o min_minutes, o revisa la quinteta seleccionada.")
        st.stop()

    except Exception as error:
        st.error("No se pudo ejecutar la recomendación.")
        st.exception(error)
        st.warning(
            "Posibles causas: filtros demasiado estrictos, datos faltantes, "
            "quinteta inválida o CSV procesados desactualizados."
        )
        st.stop()

if "last_result" not in st.session_state:
    st.info("Presiona el botón para ejecutar la recomendación.")
    st.stop()

result = st.session_state["last_result"]

if show_debug:
    with st.expander("Debug crudo: resultado de recommender.py", expanded=False):
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
    st.warning("No se encontraron reemplazos. Prueba bajar min_games o min_minutes.")
    st.stop()


# ============================================================
# 11. RESULTADO GENERAL
# ============================================================

st.header("4. Resultado del escenario")

selected_team_result = result.get("selected_team", {})
opponent_team_result = result.get("opponent_team", {})
replaced_player_result = result.get("replaced_player", {})
baseline_result = result.get("baseline", {})
roster_debug = result.get("roster_debug", {})

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
    replaced_player_result.get("player_name", selected_replacement_name),
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

if diagnostic_enabled and roster_debug:
    with st.expander("Debug de roster y banca", expanded=False):
        st.write(roster_debug)


# ============================================================
# 12. TABLA DEL TOP
# ============================================================

st.header("5. Top reemplazos desde la banca")

# Validación visible: todos deberían ser del equipo seleccionado y fuera de la quinteta.
recommended_ids = set(pd.to_numeric(summary_df["player_id"], errors="coerce").dropna().astype(int))
lineup_ids_set = set(int(player_id) for player_id in lineup_player_ids)
recommended_teams = set(summary_df["Equipo"].dropna().astype(str).str.upper().tolist())

same_team_ok = recommended_teams == {selected_team.upper()}
bench_ok = recommended_ids.isdisjoint(lineup_ids_set)
replaced_absent_ok = replaced_player_id not in recommended_ids

validation_col1, validation_col2, validation_col3 = st.columns(3)
validation_col1.metric("Mismo equipo", "OK" if same_team_ok else "Revisar")
validation_col2.metric("Fuera de quinteta", "OK" if bench_ok else "Revisar")
validation_col3.metric("Reemplazado excluido", "OK" if replaced_absent_ok else "Revisar")

display_df = summary_df.copy()

numeric_columns = [
    "MIN",
    "PTS",
    "recommendation_score",
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
# 13. DETALLE POR CANDIDATO
# ============================================================

st.header("6. Explicaciones individuales")

for idx, replacement in enumerate(top_replacements, start=1):
    player_name = replacement.get("player_name", f"Candidato {idx}")
    player_team = replacement.get("latest_team", replacement.get("team", "N/A"))
    player_position = replacement.get("position", "N/A")

    with st.expander(
        f"#{idx} - {player_name} ({player_team}, {player_position})",
        expanded=(idx == 1),
    ):
        metric_cols = st.columns(5)

        metric_cols[0].metric(
            "Recommendation score",
            format_number(
                replacement.get("recommendation_score", replacement.get("replacement_score"))
            ),
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
# 14. VISUALIZACIONES
# ============================================================

st.header("7. Visualizaciones")

# ------------------------------------------------------------
# 14.1 Probabilidad baseline vs reemplazos
# ------------------------------------------------------------

if baseline_probability is not None:
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
else:
    st.info("No hay probabilidad baseline suficiente para graficar escenarios.")


# ------------------------------------------------------------
# 14.2 Scores e impacto
# ------------------------------------------------------------

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.subheader("Recommendation score")

    score_chart_df = summary_df[["Jugador", "recommendation_score"]].copy()
    score_chart_df = score_chart_df.set_index("Jugador")

    st.bar_chart(score_chart_df, use_container_width=True)

with chart_col2:
    st.subheader("Impacto neto estimado")

    impact_chart_df = summary_df[["Jugador", "estimated_net_impact"]].copy()
    impact_chart_df = impact_chart_df.set_index("Jugador")

    st.bar_chart(impact_chart_df, use_container_width=True)


# ------------------------------------------------------------
# 14.3 Distribución de márgenes simulados
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
# 15. EXPLICACIÓN GLOBAL
# ============================================================

st.header("8. Explicación global")

st.write(result.get("explanation", "Sin explicación global disponible."))


# ============================================================
# 16. VALIDACIONES RÁPIDAS
# ============================================================

if diagnostic_enabled:
    with st.expander("Diagnóstico del resultado", expanded=False):
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
                summary_df["recommendation_score"],
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
                    "Validación": "Todos los recomendados pertenecen al equipo seleccionado",
                    "Resultado": "OK" if same_team_ok else "Revisar",
                },
                {
                    "Validación": "Ningún recomendado está en la quinteta",
                    "Resultado": "OK" if bench_ok else "Revisar",
                },
                {
                    "Validación": "El jugador reemplazado no aparece",
                    "Resultado": "OK" if replaced_absent_ok else "Revisar",
                },
                {
                    "Validación": "Sin jugadores duplicados en el top",
                    "Resultado": "OK" if not duplicate_names else "Revisar",
                },
                {
                    "Validación": "recommendation_score entre 0 y 1",
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
