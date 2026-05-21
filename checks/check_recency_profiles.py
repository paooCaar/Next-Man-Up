"""
checks/check_recency_profiles.py

Valida que player_profiles.csv tenga columnas de recencia, actividad y tendencia.
Ejecutar desde la raíz del proyecto:

    python checks/check_recency_profiles.py
"""

from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

profiles_path = PROCESSED_DIR / "player_profiles.csv"
app_roster_path = PROCESSED_DIR / "app_roster.csv"
season_profiles_path = PROCESSED_DIR / "player_season_profiles.csv"

required_profile_columns = [
    "player_id",
    "player_name",
    "recent_season",
    "recent_minutes",
    "activity_score",
    "trend_score",
    "low_activity_flag",
]


def main() -> None:
    print("Validando perfiles con recencia...\n")

    if not profiles_path.exists():
        raise FileNotFoundError(f"No se encontró {profiles_path}")
    if not app_roster_path.exists():
        raise FileNotFoundError(f"No se encontró {app_roster_path}")

    profiles = pd.read_csv(profiles_path)
    app_roster = pd.read_csv(app_roster_path)

    missing = [col for col in required_profile_columns if col not in profiles.columns]
    if missing:
        raise ValueError(f"Faltan columnas en player_profiles.csv: {missing}")

    duplicate_profiles = profiles["player_id"].duplicated().sum()
    duplicate_app_roster = app_roster["player_id"].duplicated().sum()

    print(f"player_profiles.csv filas: {len(profiles):,}")
    print(f"app_roster.csv filas: {len(app_roster):,}")
    print(f"Duplicados en player_profiles por player_id: {duplicate_profiles}")
    print(f"Duplicados en app_roster por player_id: {duplicate_app_roster}")

    if season_profiles_path.exists():
        season_profiles = pd.read_csv(season_profiles_path)
        print(f"player_season_profiles.csv filas: {len(season_profiles):,}")
    else:
        print("Aviso: player_season_profiles.csv no existe. Ejecuta python src/preprocessing.py con los archivos actualizados.")

    activity = pd.to_numeric(profiles["activity_score"], errors="coerce")
    trend = pd.to_numeric(profiles["trend_score"], errors="coerce")
    recent_minutes = pd.to_numeric(profiles["recent_minutes"], errors="coerce")

    print("\nRangos esperados:")
    print(f"activity_score min/max: {activity.min():.3f} / {activity.max():.3f}")
    print(f"trend_score min/max: {trend.min():.3f} / {trend.max():.3f}")
    print(f"Jugadores con baja actividad reciente: {(profiles['low_activity_flag'].astype(bool)).sum():,}")
    print(f"Jugadores con recent_minutes < 200: {(recent_minutes < 200).sum():,}")

    if not activity.between(0, 1).all():
        raise ValueError("activity_score debe estar entre 0 y 1")
    if not trend.between(-0.25, 0.25).all():
        raise ValueError("trend_score debe estar entre -0.25 y 0.25")
    if duplicate_profiles != 0:
        raise ValueError("player_profiles.csv tiene duplicados por player_id")
    if duplicate_app_roster != 0:
        raise ValueError("app_roster.csv tiene duplicados por player_id")

    print("\nEjemplos LAL en app_roster con señales de recencia:")
    merged = app_roster.merge(
        profiles[["player_id", "activity_score", "recent_minutes", "trend_score", "low_activity_flag"]],
        on="player_id",
        how="left",
        suffixes=("", "_profile"),
    )
    lal = merged[merged["latest_team"].astype(str).str.upper() == "LAL"].copy()
    cols = ["player_id", "player_name", "latest_team", "games_played", "minutes", "recent_minutes", "activity_score", "trend_score", "low_activity_flag"]
    print(lal[cols].head(15).to_string(index=False))

    print("\nOK: perfiles de recencia validados.")


if __name__ == "__main__":
    main()
