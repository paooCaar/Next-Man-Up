"""
Checks de Fase 2 para latest_roster.csv y player_profiles.csv.

Ejecutar desde la raíz del proyecto:
    python check_roster_outputs.py
"""

from pathlib import Path

import pandas as pd

PROCESSED_DIR = Path("data/processed")
LATEST_ROSTER_PATH = PROCESSED_DIR / "latest_roster.csv"
PLAYER_PROFILES_PATH = PROCESSED_DIR / "player_profiles.csv"

RECOMMENDER_REQUIRED_COLUMNS = [
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


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    require(LATEST_ROSTER_PATH.exists(), f"No existe {LATEST_ROSTER_PATH}")
    require(PLAYER_PROFILES_PATH.exists(), f"No existe {PLAYER_PROFILES_PATH}")

    latest = pd.read_csv(LATEST_ROSTER_PATH)
    profiles = pd.read_csv(PLAYER_PROFILES_PATH)

    require(
        len(latest) == latest["player_id"].nunique(),
        "latest_roster.csv debe tener una sola fila por player_id.",
    )
    require(
        len(profiles) == profiles["player_id"].nunique(),
        "player_profiles.csv debe tener una sola fila por player_id.",
    )
    require(
        latest["latest_team"].notna().all()
        and (latest["latest_team"].astype(str).str.strip() != "").all(),
        "latest_team no debe estar vacío.",
    )
    require(
        latest["latest_game_date"].notna().all()
        and (latest["latest_game_date"].astype(str).str.strip() != "").all(),
        "latest_game_date no debe estar vacío.",
    )
    require(
        not latest["latest_team"].astype(str).str.upper().eq("TOT").any(),
        "latest_team no debe contener TOT.",
    )

    missing = [col for col in RECOMMENDER_REQUIRED_COLUMNS if col not in profiles.columns]
    require(
        not missing,
        f"player_profiles.csv no conserva columnas requeridas por el recomendador: {missing}",
    )

    print("Checks OK")
    print(f"latest_roster.csv: {len(latest):,} jugadores únicos")
    print(f"player_profiles.csv: {len(profiles):,} jugadores únicos")

    print("\nJugadores por equipo, muestra:")
    for team in ["LAL", "BOS", "GSW", "MIL", "PHX", "DAL", "DEN", "MIA"]:
        latest_count = int((latest["latest_team"] == team).sum())
        profile_count = int((profiles["latest_team"] == team).sum())
        print(f"{team}: latest_roster={latest_count}, player_profiles={profile_count}")

    print("\nRoster más reciente disponible para LAL, muestra:")
    lal = latest[latest["latest_team"] == "LAL"].sort_values(
        ["latest_game_date", "player_name"], ascending=[False, True]
    )
    print(lal.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
