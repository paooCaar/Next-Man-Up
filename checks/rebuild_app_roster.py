"""
Rebuild data/processed/app_roster.csv from latest_roster.csv + player_profiles.csv.

Use this when Streamlit says app_roster.csv is missing columns such as:
latest_team, player_id, player_name, position.

Run from the project root:
    python checks/rebuild_app_roster.py
"""
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
LATEST_ROSTER_PATH = PROCESSED_DIR / "latest_roster.csv"
PLAYER_PROFILES_PATH = PROCESSED_DIR / "player_profiles.csv"
APP_ROSTER_PATH = PROCESSED_DIR / "app_roster.csv"

REQUIRED_LATEST_COLUMNS = [
    "player_id",
    "player_name",
    "latest_team_id",
    "latest_team",
    "latest_season",
    "latest_game_date",
    "latest_game_id",
]

REQUIRED_PROFILE_COLUMNS = [
    "player_id",
    "position",
    "games_played",
    "minutes",
    "points",
]

OPTIONAL_PROFILE_COLUMNS = [
    "activity_score",
    "recent_minutes",
    "trend_score",
    "low_activity_flag",
]

OUTPUT_COLUMNS = [
    "player_id",
    "player_name",
    "latest_team_id",
    "latest_team",
    "latest_season",
    "latest_game_date",
    "latest_game_id",
    "position",
    "latest_position",
    "latest_has_minutes",
    "games_played",
    "minutes",
    "points",
    "activity_score",
    "recent_minutes",
    "trend_score",
    "low_activity_flag",
]


def main() -> None:
    if not LATEST_ROSTER_PATH.exists():
        raise FileNotFoundError(f"No existe {LATEST_ROSTER_PATH}. Corre python src/preprocessing.py primero.")
    if not PLAYER_PROFILES_PATH.exists():
        raise FileNotFoundError(f"No existe {PLAYER_PROFILES_PATH}. Corre python src/preprocessing.py primero.")

    latest = pd.read_csv(LATEST_ROSTER_PATH)
    profiles = pd.read_csv(PLAYER_PROFILES_PATH)

    missing_latest = [c for c in REQUIRED_LATEST_COLUMNS if c not in latest.columns]
    missing_profiles = [c for c in REQUIRED_PROFILE_COLUMNS if c not in profiles.columns]

    if missing_latest:
        raise ValueError(
            f"latest_roster.csv no tiene columnas necesarias: {missing_latest}. "
            "Esto indica que latest_roster.csv está viejo o dañado. Vuelve a correr python src/preprocessing.py."
        )

    if missing_profiles:
        raise ValueError(
            f"player_profiles.csv no tiene columnas necesarias: {missing_profiles}. "
            "Esto indica que player_profiles.csv está viejo o dañado. Vuelve a correr python src/preprocessing.py."
        )

    latest = latest.copy()
    profiles = profiles.copy()

    latest["latest_season"] = pd.to_numeric(latest["latest_season"], errors="coerce")
    latest = latest.dropna(subset=["player_id", "latest_season", "latest_team"])

    latest_season = int(latest["latest_season"].max())
    latest = latest[latest["latest_season"] == latest_season].copy()

    latest["player_id"] = pd.to_numeric(latest["player_id"], errors="coerce")
    profiles["player_id"] = pd.to_numeric(profiles["player_id"], errors="coerce")
    latest = latest.dropna(subset=["player_id"])
    profiles = profiles.dropna(subset=["player_id"])
    latest["player_id"] = latest["player_id"].astype(int)
    profiles["player_id"] = profiles["player_id"].astype(int)

    profile_cols = ["player_id"] + REQUIRED_PROFILE_COLUMNS[1:] + [c for c in OPTIONAL_PROFILE_COLUMNS if c in profiles.columns]
    profile_cols = list(dict.fromkeys(profile_cols))

    app_roster = latest.merge(profiles[profile_cols], on="player_id", how="inner")

    for col in OUTPUT_COLUMNS:
        if col not in app_roster.columns:
            if col == "latest_position":
                app_roster[col] = "UNK"
            elif col == "latest_has_minutes":
                app_roster[col] = False
            elif col == "low_activity_flag":
                app_roster[col] = False
            else:
                app_roster[col] = pd.NA

    app_roster = app_roster[OUTPUT_COLUMNS].copy()
    app_roster = app_roster.drop_duplicates(subset=["player_id"]).reset_index(drop=True)
    app_roster = app_roster.sort_values(
        ["latest_team", "games_played", "minutes", "points"],
        ascending=[True, False, False, False],
    ).reset_index(drop=True)

    APP_ROSTER_PATH.parent.mkdir(parents=True, exist_ok=True)
    app_roster.to_csv(APP_ROSTER_PATH, index=False)

    required_app = ["latest_team", "player_id", "player_name", "position"]
    missing_app = [c for c in required_app if c not in app_roster.columns]
    if missing_app:
        raise ValueError(f"No se pudo reconstruir app_roster.csv. Faltan: {missing_app}")

    print("OK: app_roster.csv reconstruido.")
    print(f"Temporada más reciente: {latest_season}")
    print(f"Filas: {len(app_roster)}")
    print("Columnas:")
    print(app_roster.columns.tolist())
    print("\nConteo por equipo, muestra:")
    print(app_roster["latest_team"].value_counts().sort_index().head(30))
    if "LAL" in set(app_roster["latest_team"].astype(str)):
        print("\nRoster LAL, muestra:")
        print(app_roster.loc[app_roster["latest_team"] == "LAL", ["player_id", "player_name", "latest_team", "position", "games_played", "minutes", "points"]].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
