# run_recommender.py

from __future__ import annotations

import argparse
from pathlib import Path
import traceback

import pandas as pd

from src.recommender import recommend_replacements


# ============================================================
# 1. CONFIGURACIÓN DE EJEMPLO
# ============================================================

DEFAULT_PROCESSED_DIR = "data/processed"

# Ajusta estos valores según jugadores/equipos que existan en tus CSV.
DEFAULT_SELECTED_TEAM = "LAL"
DEFAULT_PLAYER_TO_REPLACE = "LeBron James"
DEFAULT_OPPONENT_TEAM = "BOS"
DEFAULT_NUM_SIMULATIONS = 1000


# ============================================================
# 2. COLUMNAS ESPERADAS
# ============================================================

# Estas columnas coinciden con el processed_players.csv que genera
# src/preprocessing.py en la versión actual del proyecto.
REQUIRED_PLAYER_COLUMNS = [
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
    "versatility",
]

# Estas columnas coinciden con el processed_teams.csv actual.
# No exigimos todavía métricas avanzadas como win_pct, pace_proxy,
# offensive_strength o defensive_strength porque este pipeline inicial
# usa offensive_rating, defensive_rating y pace.
REQUIRED_TEAM_COLUMNS = [
    "team",
    "offensive_rating",
    "defensive_rating",
    "pace",
]

# Estas columnas coinciden con el processed_matchups.csv que estamos
# generando desde games.csv.
REQUIRED_MATCHUP_COLUMNS = [
    "game_id",
    "team",
    "opponent_team",
    "team_score",
    "opponent_score",
    "home",
    "win",
]


# ============================================================
# 3. FUNCIONES DE VALIDACIÓN Y UTILIDAD
# ============================================================


def print_section(title: str) -> None:
    """
    Imprime una sección visual en consola.
    """
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def check_file_exists(path: Path) -> None:
    """
    Verifica que exista un archivo.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo requerido:\n{path}\n\n"
            "Primero ejecuta:\n"
            "python src/preprocessing.py"
        )


def validate_processed_files(processed_dir: str | Path) -> None:
    """
    Verifica que existan los tres CSV procesados.
    """
    processed_dir = Path(processed_dir)

    players_path = processed_dir / "processed_players.csv"
    teams_path = processed_dir / "processed_teams.csv"
    matchups_path = processed_dir / "processed_matchups.csv"

    check_file_exists(players_path)
    check_file_exists(teams_path)
    check_file_exists(matchups_path)


def validate_columns(
    df: pd.DataFrame,
    required_columns: list[str],
    table_name: str,
) -> None:
    """
    Verifica columnas necesarias.
    """
    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        raise ValueError(f"{table_name} tiene columnas faltantes:\n{missing}")


def load_processed_data(
    processed_dir: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Carga los tres CSV procesados.
    """
    processed_dir = Path(processed_dir)

    players = pd.read_csv(processed_dir / "processed_players.csv")
    teams = pd.read_csv(processed_dir / "processed_teams.csv")
    matchups = pd.read_csv(processed_dir / "processed_matchups.csv")

    return players, teams, matchups


def validate_processed_data(
    processed_dir: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Lee los CSV procesados y valida columnas mínimas antes de correr
    el recomendador.
    """
    players, teams, matchups = load_processed_data(processed_dir)

    validate_columns(
        players,
        REQUIRED_PLAYER_COLUMNS,
        "processed_players.csv",
    )

    validate_columns(
        teams,
        REQUIRED_TEAM_COLUMNS,
        "processed_teams.csv",
    )

    validate_columns(
        matchups,
        REQUIRED_MATCHUP_COLUMNS,
        "processed_matchups.csv",
    )

    print_section("DATOS PROCESADOS CARGADOS CORRECTAMENTE")

    print(f"Jugadores disponibles: {len(players)}")
    print(f"Equipos disponibles: {len(teams)}")
    print(f"Matchups disponibles: {len(matchups)}")

    print("\nEjemplos de equipos disponibles:")
    team_cols = [
        col
        for col in ["team", "offensive_rating", "defensive_rating", "pace"]
        if col in teams.columns
    ]
    print(teams[team_cols].head(10).to_string(index=False))

    print("\nEjemplos de jugadores disponibles:")
    player_cols = [
        col
        for col in [
            "player_id",
            "player_name",
            "team",
            "position",
            "minutes",
            "games_played",
        ]
        if col in players.columns
    ]
    print(players[player_cols].head(10).to_string(index=False))

    return players, teams, matchups


def print_available_data(processed_dir: str | Path, max_rows: int = 30) -> None:
    """
    Muestra equipos y jugadores disponibles para elegir argumentos válidos.
    """
    players, teams, matchups = load_processed_data(processed_dir)

    print_section("EQUIPOS DISPONIBLES")

    team_cols = [
        col
        for col in ["team", "offensive_rating", "defensive_rating", "pace"]
        if col in teams.columns
    ]
    print(teams[team_cols].drop_duplicates().head(max_rows).to_string(index=False))

    print_section("JUGADORES DISPONIBLES")

    player_cols = [
        col
        for col in [
            "player_name",
            "team",
            "position",
            "games_played",
            "minutes",
            "points",
        ]
        if col in players.columns
    ]

    print(players[player_cols].head(max_rows).to_string(index=False))

    print_section("MATCHUPS DISPONIBLES")

    matchup_cols = [
        col
        for col in [
            "team",
            "opponent_team",
            "team_score",
            "opponent_score",
            "home",
            "win",
        ]
        if col in matchups.columns
    ]

    print(matchups[matchup_cols].head(max_rows).to_string(index=False))


def suggest_valid_inputs(
    processed_dir: str | Path,
    team: str,
    player: str,
    opponent: str,
) -> None:
    """
    Ayuda a depurar nombres de equipo o jugador que no coincidan exactamente.
    """
    try:
        players, teams, _ = load_processed_data(processed_dir)
    except Exception:
        return

    print_section("AYUDA PARA ELEGIR VALORES VÁLIDOS")

    if "team" in teams.columns:
        print("\nEquipos disponibles:")
        print(teams["team"].dropna().drop_duplicates().head(30).to_list())

    if "team" in players.columns and "player_name" in players.columns:
        same_team_players = players[
            players["team"].astype(str).str.lower() == str(team).lower()
        ]

        if not same_team_players.empty:
            print(f"\nJugadores disponibles en {team}:")
            print(
                same_team_players[
                    ["player_name", "team", "position", "minutes", "games_played"]
                ]
                .head(30)
                .to_string(index=False)
            )
        else:
            print(f"\nNo encontré jugadores para el equipo escrito como: {team}")

    print("\nValores usados en esta corrida:")
    print(f"- team: {team}")
    print(f"- player: {player}")
    print(f"- opponent: {opponent}")


# ============================================================
# 4. FUNCIONES PARA FORMATO E IMPRESIÓN
# ============================================================


def format_probability(value) -> str:
    """
    Convierte una probabilidad 0-1 a porcentaje legible.
    """
    if value is None:
        return "N/A"

    try:
        return f"{float(value) * 100:.2f}%"
    except Exception:
        return "N/A"


def format_number(value, decimals: int = 3) -> str:
    """
    Formatea números de manera segura.
    """
    if value is None:
        return "N/A"

    try:
        return f"{float(value):.{decimals}f}"
    except Exception:
        return "N/A"


def get_nested_value(data: dict, possible_keys: list[str], default=None):
    """
    Devuelve el primer valor disponible entre varias llaves posibles.
    Sirve para tolerar pequeños cambios en recommender.py.
    """
    for key in possible_keys:
        if key in data and data[key] is not None:
            return data[key]

    return default


def get_recommendation_explanation(
    replacement: dict,
    global_explanation: str | None = None,
) -> str:
    """
    Obtiene una explicación textual para cada reemplazo.

    Si recommender.py ya devuelve una explicación individual dentro de cada
    reemplazo, la usa. Si no, construye una explicación breve.
    """
    for key in ["explanation", "text_explanation", "recommendation_explanation"]:
        if key in replacement and replacement[key]:
            return str(replacement[key])

    player_name = replacement.get("player_name", "Este jugador")

    role_similarity = replacement.get("role_similarity")
    position_fit = replacement.get("position_fit")
    team_fit = replacement.get("team_fit")
    opponent_fit = replacement.get("opponent_fit")
    estimated_net_impact = replacement.get("estimated_net_impact")

    explanation = (
        f"{player_name} aparece como candidato porque combina "
        f"similitud de rol={format_number(role_similarity)}, "
        f"ajuste posicional={format_number(position_fit)}, "
        f"ajuste al equipo={format_number(team_fit)} y "
        f"ajuste contra el rival={format_number(opponent_fit)}. "
        f"Su impacto neto estimado es {format_number(estimated_net_impact)} puntos."
    )

    if global_explanation:
        explanation += (
            "\n\nNota: el recomendador también generó una explicación global "
            "del ranking completo más abajo."
        )

    return explanation


def print_recommender_result(result: dict) -> None:
    """
    Imprime en consola los resultados principales del recomendador.
    """
    selected_team = result.get("selected_team", {})
    opponent_team = result.get("opponent_team", {})
    replaced_player = result.get("replaced_player", {})
    baseline = result.get("baseline", {})
    top_replacements = result.get("top_replacements", [])
    global_explanation = result.get("explanation", "")

    print_section("RESULTADO DEL ESCENARIO")

    if isinstance(selected_team, dict):
        selected_team_name = selected_team.get("team", "N/A")
    else:
        selected_team_name = str(selected_team)

    if isinstance(opponent_team, dict):
        opponent_team_name = opponent_team.get("team", "N/A")
    else:
        opponent_team_name = str(opponent_team)

    if isinstance(replaced_player, dict):
        replaced_player_name = replaced_player.get("player_name", "N/A")
    else:
        replaced_player_name = str(replaced_player)

    print(f"Equipo seleccionado: {selected_team_name}")
    print(f"Jugador reemplazado: {replaced_player_name}")
    print(f"Equipo rival: {opponent_team_name}")
    print(f"Número de simulaciones: {result.get('num_simulations', 'N/A')}")

    baseline_probability = get_nested_value(
        baseline,
        [
            "win_probability_without_replacement",
            "baseline_win_probability",
            "win_probability",
        ],
    )

    print("\nProbabilidad de ganar sin reemplazo:")
    print(f"  {format_probability(baseline_probability)}")

    print_section("TOP 3 REEMPLAZOS")

    if not top_replacements:
        print("No se encontraron reemplazos.")
        return

    for idx, replacement in enumerate(top_replacements, start=1):
        print(f"\n#{idx} - {replacement.get('player_name', 'N/A')}")
        print("-" * 90)

        print(f"Equipo actual del candidato: {replacement.get('team', 'N/A')}")
        print(f"Posición: {replacement.get('position', 'N/A')}")
        print(f"Partidos jugados: {format_number(replacement.get('games_played'), 0)}")
        print(f"Minutos por partido: {format_number(replacement.get('minutes'), 2)}")
        print(f"Puntos por partido: {format_number(replacement.get('points'), 2)}")

        print("\nMétricas del modelo:")
        print(
            f"  replacement_score:       {format_number(replacement.get('replacement_score'))}"
        )
        print(
            f"  role_similarity:         {format_number(replacement.get('role_similarity'))}"
        )
        print(
            f"  position_fit:            {format_number(replacement.get('position_fit'))}"
        )
        print(
            f"  team_fit:                {format_number(replacement.get('team_fit'))}"
        )
        print(
            f"  opponent_fit:            {format_number(replacement.get('opponent_fit'))}"
        )
        print(
            f"  estimated_net_impact:    {format_number(replacement.get('estimated_net_impact'))}"
        )

        win_probability_with_replacement = get_nested_value(
            replacement,
            [
                "win_probability_with_replacement",
                "replacement_win_probability",
                "win_probability",
            ],
        )

        print("\nProbabilidad de ganar con este reemplazo:")
        print(f"  {format_probability(win_probability_with_replacement)}")

        print("\nExplicación textual:")
        print(
            get_recommendation_explanation(
                replacement,
                global_explanation=global_explanation,
            )
        )

    print_section("EXPLICACIÓN GLOBAL DEL RECOMENDADOR")

    if global_explanation:
        print(global_explanation)
    else:
        print(
            "No se recibió una explicación global desde explanations.py. "
            "Esto no necesariamente es un error, pero conviene revisar "
            "build_recommendation_explanation() en src/explanations.py."
        )


def get_margins_from_distribution(distribution) -> list:
    """
    Extrae márgenes simulados de manera flexible.

    Espera algo como:
    {"margins": [...]}

    Si no existe, devuelve lista vacía.
    """
    if not isinstance(distribution, dict):
        return []

    margins = distribution.get("margins", [])

    if margins is None:
        return []

    try:
        return list(margins)
    except Exception:
        return []


def print_simulation_summary(result: dict) -> None:
    """
    Imprime un resumen simple de las distribuciones simuladas.

    Esto ayuda a validar que Monte Carlo sí está generando resultados.
    """
    print_section("RESUMEN DE DISTRIBUCIONES SIMULADAS")

    baseline_distribution = result.get("baseline", {}).get(
        "simulation_distribution", {}
    )

    baseline_margins = get_margins_from_distribution(baseline_distribution)

    if len(baseline_margins) > 0:
        margins = pd.Series(baseline_margins)

        print("\nDistribución sin reemplazo:")
        print(f"  Margen promedio: {margins.mean():.2f}")
        print(f"  Margen mínimo:   {margins.min():.2f}")
        print(f"  Margen máximo:   {margins.max():.2f}")
        print(f"  Desv. estándar:  {margins.std():.2f}")
    else:
        print("\nNo se recibieron márgenes simulados para el baseline.")

    for replacement in result.get("top_replacements", []):
        distribution = replacement.get("simulation_distribution", {})
        margins = get_margins_from_distribution(distribution)

        if len(margins) > 0:
            margins_series = pd.Series(margins)

            print(f"\nDistribución con {replacement.get('player_name', 'N/A')}:")
            print(f"  Margen promedio: {margins_series.mean():.2f}")
            print(f"  Margen mínimo:   {margins_series.min():.2f}")
            print(f"  Margen máximo:   {margins_series.max():.2f}")
            print(f"  Desv. estándar:  {margins_series.std():.2f}")
        else:
            print(
                f"\nNo se recibieron márgenes simulados para "
                f"{replacement.get('player_name', 'N/A')}."
            )


# ============================================================
# 5. MANEJO DE ERRORES
# ============================================================


def handle_error(
    error: Exception,
    processed_dir: str | Path,
    team: str,
    player: str,
    opponent: str,
    show_traceback: bool = False,
) -> None:
    """
    Muestra errores de manera entendible para validar el motor.
    """
    print_section("ERROR DURANTE LA EJECUCIÓN")

    error_message = str(error)

    print(error_message)

    print("\nPosibles causas:")

    if isinstance(error, FileNotFoundError):
        print("- No existen los CSV procesados en data/processed/.")
        print("- Ejecuta primero: python src/preprocessing.py")
        print("- Revisa que las rutas y nombres de archivo sean correctos.")

    elif isinstance(error, ValueError):
        if "No se encontró el jugador" in error_message:
            print("- El nombre del jugador no existe en processed_players.csv.")
            print("- Puede estar escrito diferente.")
            print("- Puede no pertenecer al equipo seleccionado.")
            print("- Revisa los nombres disponibles abajo.")

        elif "No se encontró el equipo" in error_message:
            print("- El equipo seleccionado o rival no existe en processed_teams.csv.")
            print("- Usa el nombre exacto que aparece en processed_teams.csv.")
            print("- Revisa los equipos disponibles abajo.")

        elif "No se encontró matchup" in error_message:
            print(
                "- El rival existe como equipo, pero no tiene fila en processed_matchups.csv."
            )
            print("- Revisa que processed_matchups.csv se haya generado correctamente.")
            print("- Ejecuta de nuevo: python src/preprocessing.py")

        elif (
            "columnas" in error_message.lower() or "faltantes" in error_message.lower()
        ):
            print("- Algún CSV procesado no tiene las columnas que espera el modelo.")
            print("- Revisa src/preprocessing.py.")
            print("- Ejecuta de nuevo: python src/preprocessing.py")

        elif "No hay candidatos válidos" in error_message:
            print("- Los filtros dejaron cero candidatos.")
            print("- Prueba bajar min_minutes o min_games.")
            print(
                "- También puedes permitir jugadores del rival con --allow_opponent_players."
            )

        else:
            print(
                "- Revisa que los datos procesados tengan equipos, jugadores y métricas válidas."
            )
            print(
                "- Revisa que los módulos similarity.py, impact.py, monte_carlo.py y explanations.py tengan las funciones esperadas."
            )

        suggest_valid_inputs(
            processed_dir=processed_dir,
            team=team,
            player=player,
            opponent=opponent,
        )

    else:
        print(
            "- Puede haber un error en la integración entre recommender.py y los módulos del modelo."
        )
        print(
            "- Revisa nombres de funciones y argumentos en similarity.py, impact.py, monte_carlo.py y explanations.py."
        )

    if show_traceback:
        print("\nTraceback completo:")
        traceback.print_exc()


# ============================================================
# 6. SCRIPT PRINCIPAL
# ============================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run full recommender engine from processed NBA data."
    )

    parser.add_argument(
        "--processed_dir",
        default=DEFAULT_PROCESSED_DIR,
        help="Carpeta donde están los CSV procesados.",
    )

    parser.add_argument(
        "--team",
        default=DEFAULT_SELECTED_TEAM,
        help="Equipo seleccionado. Ejemplo: LAL",
    )

    parser.add_argument(
        "--player",
        default=DEFAULT_PLAYER_TO_REPLACE,
        help="Jugador a reemplazar. Ejemplo: LeBron James",
    )

    parser.add_argument(
        "--opponent",
        default=DEFAULT_OPPONENT_TEAM,
        help="Equipo rival. Ejemplo: BOS",
    )

    parser.add_argument(
        "--simulations",
        type=int,
        default=DEFAULT_NUM_SIMULATIONS,
        help="Número de simulaciones Monte Carlo.",
    )

    parser.add_argument(
        "--min_minutes",
        type=float,
        default=10.0,
        help="Minutos mínimos por partido para candidatos.",
    )

    parser.add_argument(
        "--min_games",
        type=int,
        default=10,
        help="Partidos mínimos jugados para candidatos.",
    )

    parser.add_argument(
        "--allow_opponent_players",
        action="store_true",
        help="Permite que jugadores del equipo rival aparezcan como candidatos.",
    )

    parser.add_argument(
        "--list_data",
        action="store_true",
        help="Muestra equipos, jugadores y matchups disponibles sin correr el recomendador.",
    )

    parser.add_argument(
        "--traceback",
        action="store_true",
        help="Muestra traceback completo si ocurre un error.",
    )

    args = parser.parse_args()

    try:
        print_section("VALIDANDO ARCHIVOS PROCESADOS")

        validate_processed_files(args.processed_dir)

        if args.list_data:
            validate_processed_data(args.processed_dir)
            print_available_data(args.processed_dir)
            return

        validate_processed_data(args.processed_dir)

        print_section("EJECUTANDO RECOMMENDER")

        print(f"Equipo seleccionado: {args.team}")
        print(f"Jugador a reemplazar: {args.player}")
        print(f"Equipo rival: {args.opponent}")
        print(f"Simulaciones: {args.simulations}")
        print(f"Filtro minutos mínimos: {args.min_minutes}")
        print(f"Filtro partidos mínimos: {args.min_games}")
        print(f"Excluir jugadores del rival: {not args.allow_opponent_players}")

        result = recommend_replacements(
            selected_team_value=args.team,
            player_to_replace_value=args.player,
            opponent_team_value=args.opponent,
            num_simulations=args.simulations,
            processed_dir=args.processed_dir,
            top_n=3,
            min_minutes=args.min_minutes,
            min_games=args.min_games,
            exclude_opponent_players=not args.allow_opponent_players,
            random_state=42,
        )

        print_recommender_result(result)
        print_simulation_summary(result)

        print_section("VALIDACIÓN COMPLETA")

        print("El motor completo corrió correctamente.")
        print(
            "Ya puedes avanzar a construir la app de Streamlit usando src/recommender.py."
        )

    except Exception as error:
        handle_error(
            error=error,
            processed_dir=args.processed_dir,
            team=args.team,
            player=args.player,
            opponent=args.opponent,
            show_traceback=args.traceback,
        )


if __name__ == "__main__":
    main()
