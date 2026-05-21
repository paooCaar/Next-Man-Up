"""
check_app_roster_outputs.py

Valida los archivos generados para la Fase 1.5 del proyecto NBA Next-Man-Up.

Uso:
    python check_app_roster_outputs.py
"""

from pathlib import Path
import pandas as pd

PROCESSED_DIR = Path("data/processed")
LATEST_ROSTER_PATH = PROCESSED_DIR / "latest_roster.csv"
PLAYER_PROFILES_PATH = PROCESSED_DIR / "player_profiles.csv"
APP_ROSTER_PATH = PROCESSED_DIR / "app_roster.csv"

KEY_TEAMS = ["LAL", "BOS", "GSW", "MIL", "PHX", "DAL", "DEN", "MIA", "NYK", "BKN"]
LAL_CHECK_NAMES = ["LeBron James", "Anthony Davis", "Austin Reaves", "Russell Westbrook"]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    latest = pd.read_csv(LATEST_ROSTER_PATH)
    profiles = pd.read_csv(PLAYER_PROFILES_PATH)
    app_roster = pd.read_csv(APP_ROSTER_PATH)

    latest["latest_season"] = pd.to_numeric(latest["latest_season"], errors="coerce")
    app_roster["latest_season"] = pd.to_numeric(app_roster["latest_season"], errors="coerce")

    max_latest_season = int(latest["latest_season"].max())

    print("Max latest_season:", max_latest_season)
    print("\nColumnas app_roster.csv:")
    print(app_roster.columns.tolist())

    require(
        latest["player_id"].nunique() == len(latest),
        "latest_roster.csv debe tener una sola fila por player_id.",
    )
    require(
        profiles["player_id"].nunique() == len(profiles),
        "player_profiles.csv debe tener una sola fila por player_id.",
    )
    require(
        app_roster["player_id"].nunique() == len(app_roster),
        "app_roster.csv debe tener una sola fila por player_id.",
    )
    require(
        app_roster["latest_team"].notna().all()
        and app_roster["latest_team"].astype(str).str.strip().ne("").all(),
        "app_roster.csv no debe tener latest_team vacío.",
    )
    require(
        app_roster["latest_game_date"].notna().all()
        and app_roster["latest_game_date"].astype(str).str.strip().ne("").all(),
        "app_roster.csv no debe tener latest_game_date vacío.",
    )
    require(
        app_roster["latest_season"].eq(max_latest_season).all(),
        "app_roster.csv solo debe incluir latest_season == max(latest_season).",
    )
    require(
        set(app_roster["player_id"]).issubset(set(profiles["player_id"])),
        "Todos los jugadores de app_roster.csv deben existir en player_profiles.csv.",
    )

    print("\nConteos generales:")
    print("latest_roster jugadores únicos:", latest["player_id"].nunique())
    print("player_profiles jugadores únicos:", profiles["player_id"].nunique())
    print("app_roster jugadores únicos:", app_roster["player_id"].nunique())

    print("\nConteo por equipo en app_roster.csv:")
    team_counts = app_roster["latest_team"].value_counts().sort_index()
    print(team_counts.to_string())

    print("\nConteo de equipos clave:")
    print(team_counts.reindex(KEY_TEAMS).dropna().astype(int).to_string())

    print("\nEjemplo app_roster para LAL:")
    lal_cols = [
        "player_id",
        "player_name",
        "latest_team",
        "latest_season",
        "latest_game_date",
        "position",
        "latest_position",
        "latest_has_minutes",
        "games_played",
        "minutes",
    ]
    lal = app_roster[app_roster["latest_team"] == "LAL"].copy()
    lal = lal.sort_values(
        by=["latest_game_date", "games_played", "minutes"],
        ascending=[False, False, False],
    )
    print(lal[lal_cols].to_string(index=False))

    print("\nConfirmación de jugadores LAL esperados:")
    for name in LAL_CHECK_NAMES:
        row = app_roster[app_roster["player_name"].eq(name)]
        print(name, "OK" if not row.empty else "NO ENCONTRADO")

    old_lal = latest[
        (latest["latest_team"] == "LAL")
        & (latest["latest_season"] < max_latest_season)
    ].copy()
    old_lal_in_app = old_lal[old_lal["player_id"].isin(app_roster["player_id"])]

    require(
        old_lal_in_app.empty,
        "Jugadores históricos LAL con latest_season vieja no deben aparecer en app_roster.csv.",
    )

    print("\nJugadores históricos LAL excluidos del selector principal:", len(old_lal))
    print(
        old_lal[
            ["player_id", "player_name", "latest_team", "latest_season", "latest_game_date"]
        ]
        .head(15)
        .to_string(index=False)
    )

    print("\nTodas las validaciones de Fase 1.5 pasaron correctamente.")


if __name__ == "__main__":
    main()
