"""
Checks de backend para Fase 3.

Ejecutar desde la raíz del proyecto:

    python checks/check_recommender_backend.py

Valida que src/recommender.py use:
- app_roster.csv para pertenencia al equipo y banca.
- player_profiles.csv para perfil histórico y ranking.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.recommender import recommend_replacements
except ImportError:
    # Fallback útil si el script se ejecuta en una carpeta de trabajo temporal
    # donde los módulos están en la raíz.
    if str(PROJECT_ROOT / "src") not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from recommender import recommend_replacements


PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
APP_ROSTER_PATH = PROCESSED_DIR / "app_roster.csv"
PLAYER_PROFILES_PATH = PROCESSED_DIR / "player_profiles.csv"


def require_files() -> None:
    required = [
        APP_ROSTER_PATH,
        PLAYER_PROFILES_PATH,
        PROCESSED_DIR / "processed_teams.csv",
        PROCESSED_DIR / "processed_matchups.csv",
    ]
    missing = [str(path) for path in required if not path.exists()]

    if missing:
        raise FileNotFoundError(
            "Faltan archivos procesados para probar el backend: "
            f"{missing}. Ejecuta primero python src/preprocessing.py"
        )


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    app_roster = pd.read_csv(APP_ROSTER_PATH)
    player_profiles = pd.read_csv(PLAYER_PROFILES_PATH)

    app_roster["player_id"] = pd.to_numeric(app_roster["player_id"], errors="coerce").astype("Int64")
    player_profiles["player_id"] = pd.to_numeric(player_profiles["player_id"], errors="coerce").astype("Int64")

    return app_roster, player_profiles


def get_player_id(app_roster: pd.DataFrame, team: str, player_name: str) -> int:
    team_mask = app_roster["latest_team"].astype(str).str.upper().eq(team.upper())
    name_mask = app_roster["player_name"].astype(str).str.lower().eq(player_name.lower())
    match = app_roster[team_mask & name_mask]

    if match.empty:
        raise ValueError(f"No encontré a {player_name} en app_roster para {team}.")

    return int(match.iloc[0]["player_id"])


def top_5_ids_from_team(app_roster: pd.DataFrame, team: str) -> list[int]:
    team_df = app_roster[app_roster["latest_team"].astype(str).str.upper().eq(team.upper())].copy()

    if len(team_df) < 5:
        raise ValueError(f"{team} no tiene al menos 5 jugadores en app_roster.csv.")

    team_df["games_played"] = pd.to_numeric(team_df["games_played"], errors="coerce").fillna(0)
    team_df["minutes"] = pd.to_numeric(team_df["minutes"], errors="coerce").fillna(0)
    team_df["points"] = pd.to_numeric(team_df["points"], errors="coerce").fillna(0)

    selected = team_df.sort_values(
        by=["games_played", "minutes", "points"],
        ascending=[False, False, False],
    ).head(5)

    return selected["player_id"].astype(int).tolist()


def validate_result(
    result: dict,
    team: str,
    lineup_ids: list[int],
    replaced_player_id: int,
    min_games: int,
    min_minutes: float,
    top_n: int,
) -> list[str]:
    errors: list[str] = []
    top = result.get("top_replacements", [])

    if len(top) > top_n:
        errors.append(f"Devuelve más de top_n candidatos: {len(top)} > {top_n}")

    if not top:
        errors.append("No devolvió candidatos en top_replacements.")
        return errors

    lineup_set = set(int(x) for x in lineup_ids)

    for rec in top:
        player_id = int(rec.get("player_id"))
        latest_team = str(rec.get("latest_team"))
        games_played = float(rec.get("games_played", 0))
        minutes = float(rec.get("minutes", 0))

        if latest_team.upper() != team.upper():
            errors.append(
                f"{rec.get('player_name')} tiene latest_team={latest_team}, esperado {team}."
            )

        if player_id in lineup_set:
            errors.append(f"{rec.get('player_name')} está en lineup_player_ids.")

        if player_id == int(replaced_player_id):
            errors.append(f"{rec.get('player_name')} es el jugador reemplazado.")

        if games_played < min_games:
            errors.append(
                f"{rec.get('player_name')} no cumple min_games: {games_played} < {min_games}."
            )

        if minutes < min_minutes:
            errors.append(
                f"{rec.get('player_name')} no cumple min_minutes: {minutes} < {min_minutes}."
            )

    return errors


def print_recommendations(result: dict) -> None:
    rows = []
    for rec in result.get("top_replacements", []):
        rows.append(
            {
                "player_id": rec.get("player_id"),
                "player_name": rec.get("player_name"),
                "latest_team": rec.get("latest_team"),
                "position": rec.get("position"),
                "games_played": rec.get("games_played"),
                "minutes": round(float(rec.get("minutes", 0)), 2),
                "points": round(float(rec.get("points", 0)), 2),
                "replacement_score": round(float(rec.get("replacement_score", 0)), 3),
                "estimated_net_impact": round(float(rec.get("estimated_net_impact", 0)), 3),
                "win_probability": round(float(rec.get("win_probability_with_replacement", 0)), 3),
            }
        )

    if rows:
        print(pd.DataFrame(rows).to_string(index=False))
    else:
        print("Sin recomendaciones.")


def run_case(
    label: str,
    team: str,
    opponent: str,
    lineup_ids: list[int],
    replaced_player_id: int,
    min_games: int = 10,
    min_minutes: float = 10.0,
    top_n: int = 3,
) -> dict | None:
    print("\n" + "=" * 80)
    print(label)
    print(f"Equipo={team} | Rival={opponent}")
    print(f"lineup_player_ids={lineup_ids}")
    print(f"replaced_player_id={replaced_player_id}")

    try:
        result = recommend_replacements(
            selected_team_value=team,
            opponent_team_value=opponent,
            lineup_player_ids=lineup_ids,
            replaced_player_id=replaced_player_id,
            processed_dir=PROCESSED_DIR,
            top_n=top_n,
            min_games=min_games,
            min_minutes=min_minutes,
            num_simulations=1000,
            random_state=42,
        )
    except Exception as exc:
        print(f"ERROR CONTROLADO: {exc}")
        return None

    errors = validate_result(
        result=result,
        team=team,
        lineup_ids=lineup_ids,
        replaced_player_id=replaced_player_id,
        min_games=min_games,
        min_minutes=min_minutes,
        top_n=top_n,
    )

    if errors:
        print("VALIDACIÓN: REVISAR")
        for error in errors:
            print(f"- {error}")
    else:
        print("VALIDACIÓN: OK")

    print("Debug roster:", result.get("roster_debug"))
    print_recommendations(result)

    return result


def run_negative_case(app_roster: pd.DataFrame) -> None:
    print("\n" + "=" * 80)
    print("Caso negativo: filtros demasiado estrictos")

    lineup = top_5_ids_from_team(app_roster, "LAL")
    replaced = lineup[0]

    try:
        _ = recommend_replacements(
            selected_team_value="LAL",
            opponent_team_value="BOS",
            lineup_player_ids=lineup,
            replaced_player_id=replaced,
            processed_dir=PROCESSED_DIR,
            top_n=3,
            min_games=99999,
            min_minutes=48.0,
            num_simulations=1000,
        )
        print("REVISAR: se esperaba un error controlado por filtros imposibles.")
    except Exception as exc:
        print("ERROR CONTROLADO OK:", exc)


def main() -> None:
    require_files()
    app_roster, _ = load_inputs()

    # Caso 1: LAL fijo por nombres solicitados.
    lal_names = [
        "LeBron James",
        "Anthony Davis",
        "Austin Reaves",
        "Dennis Schroder",
        "Lonnie Walker IV",
    ]
    lal_lineup = [get_player_id(app_roster, "LAL", name) for name in lal_names]
    lebron_id = get_player_id(app_roster, "LAL", "LeBron James")

    lal_result = run_case(
        label="Caso 1: LAL vs BOS, reemplazar a LeBron James",
        team="LAL",
        opponent="BOS",
        lineup_ids=lal_lineup,
        replaced_player_id=lebron_id,
        min_games=10,
        min_minutes=10.0,
        top_n=3,
    )

    # Caso 2: GSW, quinteta tomada desde app_roster.csv.
    gsw_lineup = top_5_ids_from_team(app_roster, "GSW")
    run_case(
        label="Caso 2: GSW vs LAL, quinteta automática desde app_roster",
        team="GSW",
        opponent="LAL",
        lineup_ids=gsw_lineup,
        replaced_player_id=gsw_lineup[0],
        min_games=10,
        min_minutes=10.0,
        top_n=3,
    )

    # Caso 3: BOS, quinteta tomada desde app_roster.csv.
    bos_lineup = top_5_ids_from_team(app_roster, "BOS")
    run_case(
        label="Caso 3: BOS vs MIL, quinteta automática desde app_roster",
        team="BOS",
        opponent="MIL",
        lineup_ids=bos_lineup,
        replaced_player_id=bos_lineup[0],
        min_games=10,
        min_minutes=10.0,
        top_n=3,
    )

    run_negative_case(app_roster)

    if lal_result is not None:
        print("\n" + "=" * 80)
        print("Tabla de ejemplo LAL para reporte")
        print_recommendations(lal_result)


if __name__ == "__main__":
    main()
