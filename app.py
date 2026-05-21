# app.py

from __future__ import annotations

from pathlib import Path
from html import escape
import sys
from typing import Any

import pandas as pd
import streamlit as st

try:
    import altair as alt
except Exception:  # pragma: no cover
    alt = None

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

try:
    from src.recommender import recommend_replacements
except Exception as error:
    st.set_page_config(page_title="NBA Next-Man-Up", page_icon="🏀", layout="wide")
    st.error("No se pudo importar `src.recommender`.")
    st.exception(error)
    st.stop()

st.set_page_config(
    page_title="NBA Next-Man-Up",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# Tema visual + CSS NBA2K
# ============================================================

THEME_KEY = "nba2k_theme_mode"
if THEME_KEY not in st.session_state:
    st.session_state[THEME_KEY] = "dark"


def toggle_theme() -> None:
    current = st.session_state.get(THEME_KEY, "dark")
    st.session_state[THEME_KEY] = "light" if current == "dark" else "dark"
    try:
        st.rerun()
    except AttributeError:  # compatibilidad con versiones antiguas de Streamlit
        st.experimental_rerun()


def inject_nba2k_css(theme_mode: str) -> None:
    """Inyecta un tema tipo NBA2K sin tocar la lógica del modelo."""
    if theme_mode == "light":
        tokens = {
            "app_bg": "#f4f6fb",
            "app_text": "#0f172a",
            "muted_text": "#475569",
            "card_bg": "#ffffff",
            "card_bg_2": "#f8fafc",
            "input_bg": "#ffffff",
            "border": "rgba(15, 23, 42, 0.12)",
            "sidebar_bg": "#101827",
            "shadow": "rgba(15, 23, 42, 0.10)",
            "hero_start": "#101827",
            "hero_mid": "#1d428a",
            "hero_end": "#c8102e",
            "nav_bg": "rgba(255, 255, 255, 0.86)",
            "chart_label": "#0f172a",
        }
    else:
        tokens = {
            "app_bg": "#05070d",
            "app_text": "#f8fafc",
            "muted_text": "#cbd5e1",
            "card_bg": "#0f172a",
            "card_bg_2": "#111c31",
            "input_bg": "#111827",
            "border": "rgba(255, 255, 255, 0.13)",
            "sidebar_bg": "#05070d",
            "shadow": "rgba(0, 0, 0, 0.38)",
            "hero_start": "#05070d",
            "hero_mid": "#102a63",
            "hero_end": "#9f1239",
            "nav_bg": "rgba(5, 7, 13, 0.82)",
            "chart_label": "#f8fafc",
        }

    css = """
    <style>
    :root {
        --nba-bg: __APP_BG__;
        --nba-text: __APP_TEXT__;
        --nba-muted: __MUTED_TEXT__;
        --nba-card: __CARD_BG__;
        --nba-card-2: __CARD_BG_2__;
        --nba-input: __INPUT_BG__;
        --nba-border: __BORDER__;
        --nba-sidebar: __SIDEBAR_BG__;
        --nba-shadow: __SHADOW__;
        --nba-blue: #1d428a;
        --nba-red: #c8102e;
        --nba-orange: #f58420;
        --nba-gold: #facc15;
        --nba-green: #10b981;
        --chart-label: __CHART_LABEL__;
    }
    .stApp {
        color: var(--nba-text);
        background:
            radial-gradient(circle at 8% 0%, rgba(245, 132, 32, 0.22), transparent 26%),
            radial-gradient(circle at 88% 7%, rgba(29, 66, 138, 0.26), transparent 32%),
            radial-gradient(circle at 50% 100%, rgba(200, 16, 46, 0.16), transparent 34%),
            var(--nba-bg);
    }
    [data-testid="stHeader"] { background: transparent !important; }
    .block-container { padding-top: 1.2rem; max-width: 1320px; }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, var(--nba-sidebar), #111827) !important;
        border-right: 1px solid var(--nba-border);
    }
    [data-testid="stSidebar"] * { color: #f9fafb !important; }
    h1, h2, h3, h4, h5, h6, p, span, label, div { color: inherit; }
    .top-nav-shell {
        position: sticky;
        top: 0;
        z-index: 999;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 14px;
        padding: 10px 12px;
        margin-bottom: 14px;
        border: 1px solid var(--nba-border);
        border-radius: 20px;
        background: __NAV_BG__;
        backdrop-filter: blur(14px);
        box-shadow: 0 18px 44px var(--nba-shadow);
    }
    .brand-lockup {
        display: flex;
        align-items: center;
        gap: 10px;
        font-weight: 1000;
        letter-spacing: .08em;
        text-transform: uppercase;
        white-space: nowrap;
    }
    .brand-mark {
        width: 35px;
        height: 35px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        border-radius: 12px;
        color: white;
        background: linear-gradient(135deg, var(--nba-blue), var(--nba-red));
        box-shadow: 0 0 24px rgba(245, 132, 32, 0.35);
    }
    .top-nav {
        display: flex;
        justify-content: flex-end;
        flex-wrap: wrap;
        gap: 8px;
    }
    .top-nav a {
        text-decoration: none;
        color: var(--nba-text) !important;
        border: 1px solid var(--nba-border);
        border-radius: 999px;
        padding: 8px 13px;
        font-size: .82rem;
        font-weight: 900;
        letter-spacing: .02em;
        background: linear-gradient(180deg, rgba(255,255,255,.08), rgba(255,255,255,.02));
    }
    .top-nav a:hover {
        color: white !important;
        border-color: rgba(245, 132, 32, .72);
        background: linear-gradient(135deg, var(--nba-orange), var(--nba-red));
    }
    .hero {
        position: relative;
        overflow: hidden;
        padding: 38px 40px;
        border-radius: 30px;
        color: white;
        background:
            linear-gradient(135deg, __HERO_START__ 0%, __HERO_MID__ 52%, __HERO_END__ 100%);
        box-shadow: 0 24px 56px var(--nba-shadow);
        margin-bottom: 24px;
        border: 1px solid rgba(255,255,255,0.16);
    }
    .hero:after {
        content: "";
        position: absolute;
        inset: 0;
        background:
            linear-gradient(90deg, transparent 0 45%, rgba(255,255,255,.08) 45% 46%, transparent 46% 100%),
            repeating-linear-gradient(135deg, rgba(255,255,255,.05) 0 2px, transparent 2px 16px);
        pointer-events: none;
        opacity: .65;
    }
    .hero > * { position: relative; z-index: 1; }
    .hero-badge {
        display: inline-block;
        padding: 7px 13px;
        border-radius: 999px;
        background: rgba(255,255,255,0.13);
        border: 1px solid rgba(255,255,255,0.24);
        margin-bottom: 12px;
        font-weight: 900;
        letter-spacing: .08em;
        text-transform: uppercase;
        font-size: .78rem;
    }
    .hero h1 { font-size: clamp(2.4rem, 5vw, 4.2rem); line-height: .94; margin: 0 0 12px 0; letter-spacing: -.04em; }
    .hero p { font-size: 1.08rem; max-width: 880px; opacity: .96; margin-bottom: 0; color: rgba(255,255,255,.92); }
    .section-card, .mini-card, .chart-card, .validation-card {
        background: linear-gradient(180deg, var(--nba-card), var(--nba-card-2));
        border: 1px solid var(--nba-border);
        color: var(--nba-text);
        box-shadow: 0 16px 36px var(--nba-shadow);
    }
    .section-card { border-radius: 24px; padding: 22px 24px; margin: 14px 0 20px 0; }
    .mini-card { border-radius: 20px; padding: 18px; min-height: 128px; }
    .mini-card p, .chart-caption, .player-subtitle { color: var(--nba-muted) !important; }
    .metric-card {
        background: linear-gradient(135deg, #0b1020, #172554);
        color: white;
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 22px;
        padding: 18px;
        box-shadow: 0 16px 32px var(--nba-shadow);
        min-height: 112px;
    }
    .metric-card .label { color: #cbd5e1; font-size: .85rem; font-weight: 800; text-transform: uppercase; letter-spacing: .06em; }
    .metric-card .value { color: white; font-size: 1.45rem; font-weight: 1000; margin-top: 4px; }
    .player-card {
        position: relative;
        overflow: hidden;
        background: linear-gradient(180deg, var(--nba-card), var(--nba-card-2));
        color: var(--nba-text);
        border: 1px solid var(--nba-border);
        border-left: 8px solid var(--nba-orange);
        border-radius: 26px;
        padding: 23px 25px;
        box-shadow: 0 18px 38px var(--nba-shadow);
        margin-bottom: 18px;
    }
    .player-card:before {
        content: "";
        position: absolute;
        inset: 0 auto 0 0;
        width: 5px;
        background: linear-gradient(180deg, var(--nba-orange), var(--nba-red), var(--nba-blue));
    }
    .player-card h3 { margin: 0; font-size: 1.55rem; font-weight: 1000; color: var(--nba-text); }
    .pill {
        display:inline-block;
        padding: 6px 10px;
        margin: 3px 5px 3px 0;
        border-radius: 999px;
        background: rgba(29, 66, 138, 0.14);
        color: var(--nba-text);
        border: 1px solid var(--nba-border);
        font-weight: 900;
        font-size: .82rem;
    }
    .rank-pill { background: linear-gradient(135deg, var(--nba-orange), var(--nba-red)); color: white; border: none; }
    .warning-pill { background: rgba(245, 132, 32, .16); color: #fb923c; border-color: rgba(251,146,60,.35); }
    .success-box, .info-box, .error-box, .lineup-balance-box, .context-alert {
        padding: 12px 14px;
        border-radius: 16px;
        font-weight: 800;
        margin: 10px 0;
    }
    .success-box { background: rgba(16, 185, 129, .14); color: #34d399; border: 1px solid rgba(16,185,129,.35); }
    .info-box { background: rgba(59, 130, 246, .14); color: #60a5fa; border: 1px solid rgba(96,165,250,.35); }
    .error-box { background: rgba(239, 68, 68, .14); color: #f87171; border: 1px solid rgba(248,113,113,.40); }
    .lineup-balance-box { background: rgba(15, 23, 42, .16); border: 1px solid var(--nba-border); color: var(--nba-text); }
    .lineup-balance-box.danger { background: rgba(239,68,68,.14); border-color: rgba(248,113,113,.45); color: #f87171; }
    .lineup-balance-box.warning { background: rgba(245,158,11,.13); border-color: rgba(251,191,36,.45); color: #fbbf24; }
    .context-alert { font-size: .92rem; line-height: 1.45; }
    .context-alert.danger { background: rgba(239, 68, 68, .16); color: #f87171; border: 1px solid rgba(248,113,113,.48); }
    .context-alert.warning { background: rgba(245, 158, 11, .15); color: #fbbf24; border: 1px solid rgba(251,191,36,.45); }
    .context-alert.ok { background: rgba(16, 185, 129, .14); color: #34d399; border: 1px solid rgba(52,211,153,.40); }
    .context-alert.neutral { background: rgba(59, 130, 246, .12); color: #60a5fa; border: 1px solid rgba(96,165,250,.35); }
    .validation-card { border-radius: 20px; padding: 16px 18px; min-height: 105px; }
    .validation-card.good { border-left: 6px solid #10b981; }
    .validation-card.bad { border-left: 6px solid #ef4444; }
    .validation-title { font-size: .86rem; color: var(--nba-muted); font-weight: 900; }
    .validation-value { font-size: 1.35rem; margin-top: 6px; font-weight: 1000; }
    .validation-card.good .validation-value { color: #10b981; }
    .validation-card.bad .validation-value { color: #ef4444; }
    .chart-card { border-radius: 24px; padding: 20px 22px; margin: 16px 0 24px 0; }
    .chart-title { font-size: 1.25rem; font-weight: 1000; color: var(--nba-text); margin-bottom: 4px; }
    .chart-caption { line-height: 1.45; margin-bottom: 12px; }
    div.stButton > button:first-child {
        width: 100%;
        border-radius: 16px;
        padding: 0.82rem 1.05rem;
        background: linear-gradient(135deg, var(--nba-orange), var(--nba-red));
        color: white;
        border: 1px solid rgba(255,255,255,.16);
        font-weight: 1000;
        font-size: .98rem;
        box-shadow: 0 12px 28px rgba(200, 16, 46, .28);
    }
    div.stButton > button:hover {
        transform: translateY(-1px);
        border-color: rgba(250, 204, 21, .55);
        box-shadow: 0 16px 34px rgba(245, 132, 32, .30);
    }
    div.stButton > button:disabled {
        background: #334155; color: #94a3b8; box-shadow: none; transform: none;
    }
    [data-baseweb="select"] > div, [data-baseweb="input"] > div, textarea {
        background: var(--nba-input) !important;
        color: var(--nba-text) !important;
        border-color: var(--nba-border) !important;
    }
    .stDataFrame, .stTable { color: var(--nba-text); }
    </style>
    """
    replacements = {
        "__APP_BG__": tokens["app_bg"],
        "__APP_TEXT__": tokens["app_text"],
        "__MUTED_TEXT__": tokens["muted_text"],
        "__CARD_BG__": tokens["card_bg"],
        "__CARD_BG_2__": tokens["card_bg_2"],
        "__INPUT_BG__": tokens["input_bg"],
        "__BORDER__": tokens["border"],
        "__SIDEBAR_BG__": tokens["sidebar_bg"],
        "__SHADOW__": tokens["shadow"],
        "__HERO_START__": tokens["hero_start"],
        "__HERO_MID__": tokens["hero_mid"],
        "__HERO_END__": tokens["hero_end"],
        "__NAV_BG__": tokens["nav_bg"],
        "__CHART_LABEL__": tokens["chart_label"],
    }
    for key, value in replacements.items():
        css = css.replace(key, value)
    st.markdown(css, unsafe_allow_html=True)


inject_nba2k_css(st.session_state.get(THEME_KEY, "dark"))

# ============================================================
# Helpers
# ============================================================

POSITION_LABELS = {
    "G": "G — Guardia",
    "F": "F — Alero",
    "C": "C — Centro",
    "UNK": "Posición no disponible",
}

POSITION_EXPLANATIONS = {
    "G — Guardia": "organiza jugadas, maneja el balón y tira desde fuera.",
    "F — Alero": "jugador versátil que puede anotar, defender y rebotear.",
    "C — Centro": "jugador interior que protege el aro, bloquea tiros y toma rebotes.",
}

REQUIRED_APP_ROSTER_COLUMNS = ["latest_team", "player_id", "player_name", "position"]


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except Exception:
        return default
    if pd.isna(number):
        return default
    return number


def format_probability(value: Any) -> str:
    try:
        return f"{float(value) * 100:.1f}%"
    except Exception:
        return "N/A"


def format_number(value: Any, decimals: int = 3) -> str:
    try:
        return f"{float(value):.{decimals}f}"
    except Exception:
        return "N/A"


def score_to_10(value: Any) -> float | None:
    """Convierte scores internos 0-1 a una calificación visible 0-10."""
    try:
        number = float(value)
    except Exception:
        return None
    if pd.isna(number):
        return None
    # Si algún score ya viene en escala 0-10, lo respetamos.
    if number > 1.5:
        return max(0.0, min(10.0, number))
    return max(0.0, min(10.0, number * 10.0))


def format_score_10(value: Any, decimals: int = 1) -> str:
    score = score_to_10(value)
    if score is None:
        return "N/A"
    return f"{score:.{decimals}f}/10"


def clean_display_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Prepara la tabla visible sin usar pd.to_numeric(errors="ignore")."""
    result = df.copy()
    text_columns = {"Jugador recomendado", "Equipo", "Posición", "Baja actividad"}
    for col in result.columns:
        if col in text_columns:
            result[col] = result[col].astype(str)
            continue
        converted = pd.to_numeric(result[col], errors="coerce")
        if converted.notna().any():
            result[col] = converted.round(3)
    return result


def show_safe_dataframe(df: pd.DataFrame) -> None:
    """Muestra una tabla sin romper la app si Streamlit/Pandas falla."""
    try:
        st.dataframe(df, use_container_width=True, hide_index=True)
    except Exception as error:
        st.warning("No pude renderizar la tabla interactiva. Muestro una versión simple para evitar que la app se detenga.")
        if st.session_state.get("show_table_error", False):
            st.exception(error)
        st.table(df.astype(str))


def validation_card(title: str, ok: bool, detail: str) -> None:
    status = "OK" if ok else "Revisar"
    css = "good" if ok else "bad"
    st.markdown(
        f"""
        <div class="validation-card {css}">
            <div class="validation-title">{title}</div>
            <div class="validation-value">{status}</div>
            <div style="color:#64748b; font-size:.88rem; margin-top:4px;">{detail}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def numeric_chart_data(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def position_label(value: Any) -> str:
    key = str(value).strip().upper() if value is not None else "UNK"
    return POSITION_LABELS.get(key, f"{key} — Posición")


def html_escape(value: Any) -> str:
    return escape(str(value)) if value is not None else ""


def safe_bool(value: Any) -> bool:
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except Exception:
        pass
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "sí", "si"}
    return bool(value)


def simple_position(value: Any) -> str:
    pos = str(value).strip().upper() if value is not None else "UNK"
    if pos.startswith("C"):
        return "C"
    if pos.startswith("F"):
        return "F"
    if pos.startswith("G"):
        return "G"
    return "UNK"


def position_short_label(pos: str) -> str:
    return {"G": "Guardia", "F": "Alero/Poste", "C": "Centro", "UNK": "Sin posición"}.get(pos, pos)


def build_lineup_context(lineup_player_ids: list[int], replaced_player_id: int | None, id_to_row: dict[int, pd.Series]) -> dict:
    """Resume qué queda en cancha después de sacar al jugador reemplazado."""
    if replaced_player_id is None:
        return {}

    replaced_id = int(replaced_player_id)
    remaining_ids = [int(pid) for pid in lineup_player_ids if int(pid) != replaced_id and int(pid) in id_to_row]
    replaced_row = id_to_row.get(replaced_id)
    replaced_pos = simple_position(replaced_row.get("position")) if replaced_row is not None else "UNK"
    replaced_name = replaced_row.get("player_name", "Jugador") if replaced_row is not None else "Jugador"

    remaining_players = []
    remaining_positions = []
    for pid in remaining_ids:
        row = id_to_row[pid]
        pos = simple_position(row.get("position"))
        remaining_positions.append(pos)
        remaining_players.append({
            "player_id": pid,
            "player_name": row.get("player_name", "Jugador"),
            "position": pos,
        })

    counts = {pos: remaining_positions.count(pos) for pos in ["G", "F", "C", "UNK"]}
    return {
        "replaced_id": replaced_id,
        "replaced_name": replaced_name,
        "replaced_position": replaced_pos,
        "remaining_players": remaining_players,
        "remaining_positions": remaining_positions,
        "remaining_counts": counts,
        "remaining_center_count": counts.get("C", 0),
        "remaining_frontcourt_count": counts.get("F", 0) + counts.get("C", 0),
    }


def render_lineup_balance_box(lineup_context: dict, compact: bool = False) -> str:
    if not lineup_context:
        return ""

    replaced_name = html_escape(lineup_context.get("replaced_name", "Jugador"))
    replaced_pos = lineup_context.get("replaced_position", "UNK")
    counts = lineup_context.get("remaining_counts", {})
    remaining_center_count = lineup_context.get("remaining_center_count", 0)
    remaining_frontcourt_count = lineup_context.get("remaining_frontcourt_count", 0)
    remaining_text = " · ".join(
        f"{html_escape(p.get('player_name'))} ({position_short_label(p.get('position', 'UNK'))})"
        for p in lineup_context.get("remaining_players", [])
    )

    css = "lineup-balance-box"
    title = "Balance de los 4 jugadores que quedan"
    detail = (
        f"Después de sacar a <b>{replaced_name}</b>, quedan "
        f"G:{counts.get('G', 0)} · F/Poste:{counts.get('F', 0)} · C:{counts.get('C', 0)}."
    )

    if replaced_pos == "C" and remaining_center_count == 0:
        css += " danger"
        title = "Alerta roja: los 4 que quedan no tienen centro natural"
        detail += " Si el reemplazo no es centro, la quinteta queda sin protector de aro natural."
    elif remaining_frontcourt_count < 2:
        css += " warning"
        title = "Alerta de tamaño: pocos jugadores interiores"
        detail += " La quinteta podría quedar muy cargada a guardias y perder rebote interior."

    if compact:
        return f"<div class='{css}'><b>{title}</b><br>{detail}</div>"
    return f"<div class='{css}'><b>{title}</b><br>{detail}<br><span style='font-weight:600;'>{remaining_text}</span></div>"


def replacement_context_alert(replacement: dict, lineup_context: dict) -> dict:
    """Evalúa si cada candidato arregla o empeora el balance posicional de la quinteta."""
    if not lineup_context:
        return {"level": "neutral", "title": "Contexto no disponible", "text": "No se pudo evaluar la composición de la quinteta."}

    candidate_pos = simple_position(replacement.get("position", "UNK"))
    candidate_name = replacement.get("player_name", "Este jugador")
    final_positions = lineup_context.get("remaining_positions", []) + [candidate_pos]
    final_center_count = final_positions.count("C")
    final_frontcourt_count = final_positions.count("F") + final_positions.count("C")
    replaced_pos = lineup_context.get("replaced_position", "UNK")
    replaced_name = lineup_context.get("replaced_name", "el jugador reemplazado")

    if replaced_pos == "C" and final_center_count == 0:
        return {
            "level": "danger",
            "title": "Alerta roja: quinteta sin centro",
            "text": (
                f"Al salir {replaced_name}, los cuatro titulares restantes no tienen centro natural. "
                f"Si entra {candidate_name}, que es {position_short_label(candidate_pos)}, el equipo queda sin poste/centro para proteger el aro."
            ),
        }

    if replaced_pos == "C" and candidate_pos == "C":
        return {
            "level": "ok",
            "title": "Balance interior corregido",
            "text": (
                f"Como sale un centro, {candidate_name} mantiene un centro natural en cancha. "
                "Esto reduce el riesgo de perder rebote, tamaño y protección del aro."
            ),
        }

    if final_frontcourt_count < 2:
        return {
            "level": "warning",
            "title": "Advertencia de tamaño",
            "text": (
                f"Con {candidate_name}, la quinteta final tendría pocos jugadores interiores. "
                "Puede funcionar si buscas velocidad, pero aumenta el riesgo en rebote y defensa cerca del aro."
            ),
        }

    if candidate_pos == lineup_context.get("replaced_position", "UNK"):
        return {
            "level": "ok",
            "title": "Sustitución posicional limpia",
            "text": (
                f"{candidate_name} entra en la misma familia posicional que {replaced_name}, "
                "por lo que la estructura de la quinteta se mantiene más estable."
            ),
        }

    return {
        "level": "neutral",
        "title": "Cambio de perfil",
        "text": (
            f"{candidate_name} cambia parcialmente el perfil de la quinteta. "
            "La recomendación puede tener sentido por similitud e impacto, pero conviene revisar el balance posicional."
        ),
    }


def render_context_alert(alert: dict) -> str:
    level = html_escape(alert.get("level", "neutral"))
    title = html_escape(alert.get("title", "Contexto de quinteta"))
    text = html_escape(alert.get("text", ""))
    return f"<div class='context-alert {level}'><b>{title}</b><br>{text}</div>"


def player_label(row: pd.Series) -> str:
    name = row.get("player_name", "N/A")
    pos = position_label(row.get("position", "UNK"))
    gp = int(safe_float(row.get("games_played", 0)))
    minutes = safe_float(row.get("minutes", 0))
    recent = safe_float(row.get("recent_minutes", 0))
    return f"{name} | {pos} | {gp} PJ | {minutes:.1f} MIN | reciente: {recent:.0f} min"


def pick_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def standardize_app_roster_columns(app_roster: pd.DataFrame) -> pd.DataFrame:
    """Normaliza app_roster si viene con sufijos por merges anteriores."""
    df = app_roster.copy()

    rename_candidates = {
        "player_id": ["player_id", "PLAYER_ID"],
        "player_name": ["player_name", "player_name_x", "player_name_y", "PLAYER_NAME"],
        "latest_team": ["latest_team", "latest_team_x", "latest_team_y", "team", "TEAM_ABBREVIATION"],
        "latest_team_id": ["latest_team_id", "latest_team_id_x", "latest_team_id_y", "team_id", "TEAM_ID"],
        "latest_season": ["latest_season", "latest_season_x", "latest_season_y", "season", "SEASON"],
        "latest_game_date": ["latest_game_date", "latest_game_date_x", "latest_game_date_y", "game_date", "GAME_DATE_EST"],
        "latest_game_id": ["latest_game_id", "latest_game_id_x", "latest_game_id_y", "game_id", "GAME_ID"],
        "position": ["position", "position_y", "position_x", "latest_position"],
        "games_played": ["games_played", "games_played_y", "games_played_x"],
        "minutes": ["minutes", "minutes_y", "minutes_x"],
        "points": ["points", "points_y", "points_x"],
        "recent_minutes": ["recent_minutes", "recent_minutes_y", "recent_minutes_x"],
        "activity_score": ["activity_score", "activity_score_y", "activity_score_x"],
        "trend_score": ["trend_score", "trend_score_y", "trend_score_x"],
        "low_activity_flag": ["low_activity_flag", "low_activity_flag_y", "low_activity_flag_x"],
    }

    out = pd.DataFrame(index=df.index)
    for canonical, candidates in rename_candidates.items():
        source = pick_column(df, candidates)
        if source is not None:
            out[canonical] = df[source]

    for col in ["player_id", "latest_team_id", "latest_season", "latest_game_id", "games_played"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    for col in ["minutes", "points", "recent_minutes", "activity_score", "trend_score"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    if "position" in out.columns:
        out["position"] = out["position"].fillna("UNK").astype(str).str.strip().str.upper()
    if "latest_team" in out.columns:
        out["latest_team"] = out["latest_team"].fillna("").astype(str).str.strip()
    if "player_name" in out.columns:
        out["player_name"] = out["player_name"].fillna("").astype(str).str.strip()

    return out.reset_index(drop=True)


def rebuild_app_roster_from_sources(processed_dir: Path) -> pd.DataFrame:
    latest_path = processed_dir / "latest_roster.csv"
    profiles_path = processed_dir / "player_profiles.csv"
    app_roster_path = processed_dir / "app_roster.csv"

    if not latest_path.exists() or not profiles_path.exists():
        raise FileNotFoundError(
            "No pude reconstruir app_roster.csv porque falta latest_roster.csv o player_profiles.csv. "
            "Ejecuta: python src/preprocessing.py"
        )

    latest = pd.read_csv(latest_path)
    profiles = pd.read_csv(profiles_path)
    latest = standardize_app_roster_columns(latest)
    profiles_std = standardize_app_roster_columns(profiles)

    if "latest_season" not in latest.columns:
        raise ValueError("latest_roster.csv no tiene latest_season. Ejecuta de nuevo python src/preprocessing.py")

    max_season = pd.to_numeric(latest["latest_season"], errors="coerce").max()
    latest_season_roster = latest[pd.to_numeric(latest["latest_season"], errors="coerce") == max_season].copy()

    profile_cols = [
        c for c in [
            "player_id", "player_name", "position", "games_played", "minutes", "points",
            "recent_minutes", "activity_score", "trend_score", "low_activity_flag",
        ] if c in profiles_std.columns
    ]
    merged = latest_season_roster.merge(
        profiles_std[profile_cols],
        on="player_id",
        how="inner",
        suffixes=("_latest", "_profile"),
    )

    app = pd.DataFrame()
    app["player_id"] = pd.to_numeric(merged["player_id"], errors="coerce").astype("Int64")
    app["player_name"] = merged.get("player_name_profile", merged.get("player_name_latest", "")).fillna("").astype(str)
    app["latest_team_id"] = merged.get("latest_team_id", pd.NA)
    app["latest_team"] = merged.get("latest_team", "").fillna("").astype(str).str.strip()
    app["latest_season"] = merged.get("latest_season", pd.NA)
    app["latest_game_date"] = merged.get("latest_game_date", pd.NA)
    app["latest_game_id"] = merged.get("latest_game_id", pd.NA)
    app["position"] = merged.get("position_profile", merged.get("position_latest", "UNK")).fillna("UNK").astype(str).str.upper()

    for col in ["games_played", "minutes", "points", "recent_minutes", "activity_score", "trend_score", "low_activity_flag"]:
        if col in merged.columns:
            app[col] = merged[col]

    app = app.dropna(subset=["player_id"])
    app["player_id"] = app["player_id"].astype(int)
    app = app.drop_duplicates(subset=["player_id"]).sort_values(["latest_team", "player_name"]).reset_index(drop=True)
    app.to_csv(app_roster_path, index=False)
    return app


def ensure_valid_app_roster(processed_dir: Path) -> tuple[pd.DataFrame, bool, str]:
    app_roster_path = processed_dir / "app_roster.csv"
    rebuilt = False
    message = "app_roster.csv cargado correctamente."

    if not app_roster_path.exists():
        app = rebuild_app_roster_from_sources(processed_dir)
        return app, True, "app_roster.csv no existía; fue reconstruido desde latest_roster.csv + player_profiles.csv."

    raw = pd.read_csv(app_roster_path)
    app = standardize_app_roster_columns(raw)
    missing = [c for c in REQUIRED_APP_ROSTER_COLUMNS if c not in app.columns]

    if missing:
        app = rebuild_app_roster_from_sources(processed_dir)
        rebuilt = True
        message = f"app_roster.csv estaba incompleto ({missing}); fue reconstruido automáticamente."
    else:
        # Sobrescribe una versión normalizada para evitar que el backend falle por columnas con sufijos.
        app.to_csv(app_roster_path, index=False)

    missing_after = [c for c in REQUIRED_APP_ROSTER_COLUMNS if c not in app.columns]
    if missing_after:
        raise ValueError(f"app_roster.csv sigue sin columnas necesarias después de reconstruir: {missing_after}")

    app["player_id"] = pd.to_numeric(app["player_id"], errors="coerce")
    app = app.dropna(subset=["player_id", "latest_team", "player_name"]).copy()
    app["player_id"] = app["player_id"].astype(int)
    app["position"] = app["position"].fillna("UNK").astype(str).str.upper()

    return app.reset_index(drop=True), rebuilt, message


@st.cache_data(show_spinner=False)
def cached_load_ui_data(processed_dir_str: str):
    processed_dir = Path(processed_dir_str)
    required = [
        processed_dir / "processed_teams.csv",
        processed_dir / "processed_matchups.csv",
        processed_dir / "player_profiles.csv",
        processed_dir / "latest_roster.csv",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Faltan archivos procesados: {missing}. Ejecuta python src/preprocessing.py")

    app_roster, rebuilt, message = ensure_valid_app_roster(processed_dir)
    teams = pd.read_csv(processed_dir / "processed_teams.csv")
    profiles = pd.read_csv(processed_dir / "player_profiles.csv")
    return app_roster, teams, profiles, rebuilt, message


def build_summary_dataframe(top_replacements: list[dict], baseline_probability: float | None) -> pd.DataFrame:
    rows = []
    for r in top_replacements:
        win_probability = r.get("win_probability_with_replacement")
        win_delta = None
        if baseline_probability is not None and win_probability is not None:
            win_delta = float(win_probability) - float(baseline_probability)
        rows.append({
            "Jugador recomendado": r.get("player_name"),
            "Equipo": r.get("latest_team", r.get("team")),
            "Posición": position_label(r.get("position")),
            "Partidos jugados": r.get("games_played"),
            "Minutos promedio": r.get("minutes"),
            "Puntos promedio": r.get("points"),
            "Calificación total": score_to_10(r.get("recommendation_score", r.get("replacement_score"))),
            "Score base": score_to_10(r.get("replacement_score")),
            "Similitud": r.get("role_similarity"),
            "Actividad reciente": r.get("activity_score"),
            "Minutos recientes": r.get("recent_minutes"),
            "Tendencia": r.get("trend_score"),
            "Baja actividad": r.get("low_activity_flag"),
            "Impacto estimado": r.get("estimated_net_impact"),
            "Probabilidad de victoria": win_probability,
            "Cambio vs baseline (pp)": win_delta * 100 if win_delta is not None else None,
        })
    return pd.DataFrame(rows)


def human_explanation(
    replacement: dict,
    replaced_name: str,
    lineup_context: dict | None = None,
    context_alert: dict | None = None,
) -> str:
    name = replacement.get("player_name", "Este jugador")
    pos = position_label(replacement.get("position", "UNK"))
    score = safe_float(replacement.get("recommendation_score", replacement.get("replacement_score", 0)))
    activity = safe_float(replacement.get("activity_score", 1.0), 1.0)
    trend = safe_float(replacement.get("trend_score", 0.0), 0.0)
    impact = safe_float(replacement.get("estimated_net_impact", 0.0), 0.0)
    low_activity = safe_bool(replacement.get("low_activity_flag", False))

    if score >= 0.75:
        why = f"{name} aparece como una opción fuerte porque combina buen ajuste con el rol que deja {replaced_name} y suficiente valor dentro del modelo."
    elif score >= 0.55:
        why = f"{name} aparece como una opción razonable porque puede cubrir parte del rol de {replaced_name}, aunque no sea un reemplazo perfecto."
    else:
        why = f"{name} aparece como alternativa disponible desde la banca, pero el modelo lo considera una opción menos fuerte."

    if "Guardia" in pos:
        aporte = "Aporta manejo de balón, creación de juego y capacidad para iniciar ofensiva."
    elif "Alero" in pos:
        aporte = "Aporta versatilidad: puede anotar, defender varias posiciones y apoyar en rebote."
    elif "Centro" in pos:
        aporte = "Aporta presencia interior, rebote y protección del aro."
    else:
        aporte = "Aporta profundidad de rotación, aunque su posición no está completamente clara en los datos."

    riesgo = ""
    if low_activity or activity < 0.35:
        riesgo = "Su principal riesgo es que tiene poca actividad reciente, así que la recomendación debe tomarse con más cautela."
    elif trend < -0.10:
        riesgo = "El modelo detecta una tendencia reciente a la baja, así que podría estar perdiendo impacto respecto a temporadas anteriores."
    else:
        riesgo = "El riesgo principal es que no replica exactamente el perfil estadístico del jugador reemplazado."

    if impact >= 0:
        impact_text = "Su impacto estimado sugiere que podría ayudar a mantener o mejorar el margen esperado del equipo."
    else:
        impact_text = "Su impacto estimado sugiere que el equipo podría perder algo de eficiencia, aunque sigue siendo de las mejores opciones disponibles según los filtros."

    context_text = ""
    if context_alert:
        level = context_alert.get("level", "neutral")
        if level == "danger":
            context_text = "En el contexto de la quinteta, esta opción es riesgosa porque no resuelve la falta de centro/poste después del cambio."
        elif level == "ok":
            context_text = "En el contexto de la quinteta, esta opción ayuda a mantener una estructura posicional balanceada."
        elif level == "warning":
            context_text = "En el contexto de la quinteta, esta opción puede funcionar, pero deja dudas de tamaño, rebote o defensa interior."
        else:
            context_text = "En el contexto de la quinteta, esta opción cambia parcialmente el perfil de los cinco jugadores en cancha."

    return " ".join(part for part in [why, aporte, riesgo, impact_text, context_text] if part)


def altair_bar(df: pd.DataFrame, x: str, y: str, title: str, y_title: str, color: str = "#1d428a"):
    """Barra horizontal limpia con etiquetas. Fallback seguro si Altair no está disponible."""
    plot_df = numeric_chart_data(df[[x, y]].copy(), [y]).dropna(subset=[x, y])
    if plot_df.empty:
        st.info("No hay datos suficientes para esta gráfica.")
        return
    plot_df = plot_df.sort_values(y, ascending=True)

    if alt is None:
        st.bar_chart(plot_df.set_index(x)[y], use_container_width=True)
        return

    height = max(250, min(420, 58 * len(plot_df)))
    bars = (
        alt.Chart(plot_df)
        .mark_bar(cornerRadiusEnd=10, color=color)
        .encode(
            y=alt.Y(f"{x}:N", sort=plot_df[x].tolist(), title=None),
            x=alt.X(f"{y}:Q", title=y_title),
            tooltip=[
                alt.Tooltip(f"{x}:N", title="Jugador"),
                alt.Tooltip(f"{y}:Q", title=y_title, format=".2f"),
            ],
        )
    )
    labels = (
        alt.Chart(plot_df)
        .mark_text(align="left", baseline="middle", dx=6, color="#0f172a", fontWeight="bold")
        .encode(
            y=alt.Y(f"{x}:N", sort=plot_df[x].tolist(), title=None),
            x=alt.X(f"{y}:Q"),
            text=alt.Text(f"{y}:Q", format=".2f"),
        )
    )
    st.altair_chart((bars + labels).properties(title=title, height=height), use_container_width=True)


def altair_scatter_points_minutes(df: pd.DataFrame) -> None:
    needed = ["Jugador recomendado", "Minutos promedio", "Puntos promedio", "Calificación total", "Probabilidad de victoria"]
    plot_df = numeric_chart_data(df[[c for c in needed if c in df.columns]].copy(), ["Minutos promedio", "Puntos promedio", "Calificación total", "Probabilidad de victoria"]).dropna(subset=["Minutos promedio", "Puntos promedio"])
    if plot_df.empty:
        st.info("No hay datos suficientes para esta gráfica.")
        return

    if alt is None:
        st.scatter_chart(plot_df, x="Minutos promedio", y="Puntos promedio", use_container_width=True)
        return

    points = (
        alt.Chart(plot_df)
        .mark_circle(size=260, opacity=0.88, stroke="white", strokeWidth=2)
        .encode(
            x=alt.X("Minutos promedio:Q", title="Minutos promedio por partido"),
            y=alt.Y("Puntos promedio:Q", title="Puntos promedio por partido"),
            color=alt.Color("Calificación total:Q", title="Calificación", scale=alt.Scale(range=["#1d428a", "#f58420", "#c8102e"])),
            tooltip=[
                alt.Tooltip("Jugador recomendado:N", title="Jugador"),
                alt.Tooltip("Puntos promedio:Q", title="Puntos promedio", format=".1f"),
                alt.Tooltip("Minutos promedio:Q", title="Minutos promedio", format=".1f"),
                alt.Tooltip("Calificación total:Q", title="Calificación", format=".1f"),
            ],
        )
    )
    labels = (
        alt.Chart(plot_df)
        .mark_text(dx=10, dy=-8, color="#334155", fontWeight="bold")
        .encode(
            x="Minutos promedio:Q",
            y="Puntos promedio:Q",
            text="Jugador recomendado:N",
        )
    )
    st.altair_chart((points + labels).properties(height=360), use_container_width=True)


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:
    st.title("🏀 Next Man Up")
    st.caption("Define tu quinteta y encuentra una opción desde la banca.")
    with st.expander("Modo desarrollador", expanded=False):
        processed_dir_input = st.text_input("Carpeta de datos procesados", value="data/processed")
        developer_mode = st.checkbox("Mostrar diagnóstico", value=False)
        show_debug = st.checkbox("Mostrar output crudo", value=False)
        if st.button("Limpiar caché de datos"):
            st.cache_data.clear()
            st.success("Caché limpiado. Recarga la app.")

processed_dir = Path(processed_dir_input)

# ============================================================
# Header / menú superior
# ============================================================

current_theme = st.session_state.get(THEME_KEY, "dark")
theme_button_label = "Modo claro" if current_theme == "dark" else "Modo oscuro"

nav_col, theme_col = st.columns([5.8, 1.15])
with nav_col:
    st.markdown(
        """
        <div class="top-nav-shell">
          <div class="brand-lockup"><span class="brand-mark">2K</span><span>Next-Man-Up</span></div>
          <div class="top-nav">
            <a href="#inicio">Inicio</a>
            <a href="#partido">Partido</a>
            <a href="#quinteta">Quinteta</a>
            <a href="#reemplazo">Reemplazo</a>
            <a href="#resultados">Resultados</a>
            <a href="#graficas">Gráficas</a>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with theme_col:
    if st.button(theme_button_label, key="nba2k_theme_toggle_button", help="Cambiar entre tema oscuro y claro"):
        toggle_theme()

st.markdown('<div id="inicio"></div>', unsafe_allow_html=True)
st.markdown(
    """
    <div class="hero">
      <div class="hero-badge">Rotación NBA · Quinteta · Banca</div>
      <h1>NBA Next-Man-Up</h1>
      <p>Analiza una sustitución como si estuvieras ajustando la rotación en NBA2K: selecciona equipo, rival, quinteta en cancha y jugador que sale. La app recomienda reemplazos de banca y avisa si el cambio rompe el balance posicional.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

intro_cols = st.columns(3)
with intro_cols[0]:
    st.markdown("<div class='mini-card'><h3>1. Define el partido</h3><p>Elige tu equipo y el rival para contextualizar el análisis.</p></div>", unsafe_allow_html=True)
with intro_cols[1]:
    st.markdown("<div class='mini-card'><h3>2. Arma la quinteta</h3><p>Selecciona exactamente 5 jugadores que están en cancha.</p></div>", unsafe_allow_html=True)
with intro_cols[2]:
    st.markdown("<div class='mini-card'><h3>3. Revisa el balance</h3><p>Además del score, la app alerta si la sustitución deja huecos de posición.</p></div>", unsafe_allow_html=True)

with st.expander("¿Qué significan las posiciones?", expanded=False):
    for label, text in POSITION_EXPLANATIONS.items():
        st.markdown(f"**{label}:** {text}")

# ============================================================
# Carga de datos UI
# ============================================================

try:
    app_roster_df, teams_df, profiles_df, rebuilt_roster, roster_message = cached_load_ui_data(str(processed_dir))
except Exception as error:
    st.error("No se pudieron cargar los datos para la app.")
    st.exception(error)
    st.info("Corre `python src/preprocessing.py` y luego `streamlit cache clear`.")
    st.stop()

if rebuilt_roster:
    st.warning(roster_message)
else:
    if developer_mode:
        st.success(roster_message)

if developer_mode:
    with st.expander("Diagnóstico de datos cargados", expanded=False):
        st.write("Columnas app_roster.csv:", app_roster_df.columns.tolist())
        st.write("Filas app_roster:", len(app_roster_df))
        st.write("Columnas player_profiles.csv:", profiles_df.columns.tolist())
        st.dataframe(app_roster_df.head(20), use_container_width=True)

# ============================================================
# Flujo principal
# ============================================================

st.markdown('<div id="partido"></div>', unsafe_allow_html=True)
st.markdown("<div class='section-card'>", unsafe_allow_html=True)
st.header("1. Define el partido")
team_options = sorted(app_roster_df["latest_team"].dropna().astype(str).unique().tolist())
if not team_options:
    st.error("No hay equipos en app_roster.csv.")
    st.stop()

col_team, col_opp = st.columns(2)
with col_team:
    default_team_index = team_options.index("LAL") if "LAL" in team_options else 0
    selected_team = st.selectbox("Equipo propio", team_options, index=default_team_index)
with col_opp:
    opponent_options = [t for t in team_options if t != selected_team]
    default_opp_index = opponent_options.index("BOS") if "BOS" in opponent_options else 0
    opponent_team = st.selectbox("Equipo rival", opponent_options, index=default_opp_index)
st.markdown("</div>", unsafe_allow_html=True)

team_roster = app_roster_df[app_roster_df["latest_team"].astype(str).str.upper() == selected_team.upper()].copy()
team_roster = team_roster.sort_values(["games_played", "minutes", "points"], ascending=[False, False, False])

id_to_row = {int(row["player_id"]): row for _, row in team_roster.iterrows()}
player_ids = list(id_to_row.keys())

def format_player_id(pid: int) -> str:
    return player_label(id_to_row[int(pid)])

st.markdown('<div id="quinteta"></div>', unsafe_allow_html=True)
st.markdown("<div class='section-card'>", unsafe_allow_html=True)
st.header("2. Selecciona tu quinteta")
st.caption("Elige exactamente 5 jugadores en cancha. Los recomendados saldrán de la banca de este mismo equipo.")

# Preselección útil para LAL si existen.
def find_ids_by_names(names: list[str]) -> list[int]:
    found = []
    roster_names = team_roster.set_index("player_name")["player_id"].to_dict() if "player_name" in team_roster.columns else {}
    for name in names:
        if name in roster_names:
            found.append(int(roster_names[name]))
    return found

if selected_team == "LAL":
    default_lineup = find_ids_by_names(["LeBron James", "Anthony Davis", "Austin Reaves", "Dennis Schroder", "Lonnie Walker IV"])
else:
    default_lineup = player_ids[:5]

lineup_player_ids = st.multiselect(
    "Jugadores en cancha",
    options=player_ids,
    default=default_lineup[:5],
    format_func=format_player_id,
    max_selections=5,
)

lineup_count = len(lineup_player_ids)
if lineup_count == 5:
    st.markdown(f"<div class='success-box'>Quinteta lista: {lineup_count}/5 jugadores seleccionados.</div>", unsafe_allow_html=True)
else:
    st.markdown(f"<div class='error-box'>Selecciona exactamente 5 jugadores en cancha. Seleccionados: {lineup_count}/5.</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div id="reemplazo"></div>', unsafe_allow_html=True)
st.markdown("<div class='section-card'>", unsafe_allow_html=True)
st.header("3. Elige al jugador a reemplazar")
if lineup_count == 5:
    replaced_player_id = st.selectbox(
        "Jugador a reemplazar",
        options=lineup_player_ids,
        format_func=format_player_id,
        help="La app buscará un reemplazo disponible desde la banca del mismo equipo.",
    )
    replaced_name = id_to_row[int(replaced_player_id)].get("player_name", "Jugador")
    lineup_context_preview = build_lineup_context([int(x) for x in lineup_player_ids], int(replaced_player_id), id_to_row)
    st.markdown(f"<div class='info-box'>Buscaremos un reemplazo desde la banca para <b>{html_escape(replaced_name)}</b>.</div>", unsafe_allow_html=True)
    st.markdown(render_lineup_balance_box(lineup_context_preview), unsafe_allow_html=True)
else:
    replaced_player_id = None
    lineup_context_preview = {}
    st.info("Primero completa la quinteta para elegir a quién reemplazar.")
st.markdown("</div>", unsafe_allow_html=True)

with st.expander("4. Ajustes avanzados", expanded=False):
    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        min_games = st.number_input("Mínimo de partidos", min_value=1, max_value=5000, value=10, step=1)
    with col_b:
        min_minutes = st.number_input("Mínimo de minutos promedio", min_value=0.0, max_value=48.0, value=10.0, step=1.0)
    with col_c:
        top_n = st.number_input("Número de recomendaciones", min_value=1, max_value=10, value=3, step=1)
    with col_d:
        num_simulations = st.number_input("Simulaciones", min_value=500, max_value=50000, value=3000, step=500)

can_run = lineup_count == 5 and replaced_player_id is not None
run_button = st.button("Buscar reemplazo 🏀", disabled=not can_run)

if not can_run:
    st.stop()

if run_button:
    try:
        with st.spinner("Analizando banca, perfil histórico y escenario de partido..."):
            result = recommend_replacements(
                selected_team_value=selected_team,
                opponent_team_value=opponent_team,
                lineup_player_ids=[int(x) for x in lineup_player_ids],
                replaced_player_id=int(replaced_player_id),
                monte_carlo_simulations=int(num_simulations),
                processed_dir=str(processed_dir),
                top_n=int(top_n),
                min_minutes=float(min_minutes),
                min_games=int(min_games),
                random_state=42,
            )
        st.session_state["last_result"] = result
        st.session_state["last_inputs"] = {
            "selected_team": selected_team,
            "opponent_team": opponent_team,
            "lineup_player_ids": [int(x) for x in lineup_player_ids],
            "replaced_player_id": int(replaced_player_id),
            "replaced_name": replaced_name,
            "min_games": int(min_games),
            "min_minutes": float(min_minutes),
            "top_n": int(top_n),
        }
    except Exception as error:
        st.error("No se pudo ejecutar la recomendación.")
        st.exception(error)
        st.warning("Revisa que app_roster.csv tenga jugadores para ese equipo y que los filtros no sean demasiado estrictos.")
        st.stop()

if "last_result" not in st.session_state:
    st.info("Presiona **Buscar reemplazo 🏀** para ver recomendaciones.")
    st.stop()

result = st.session_state["last_result"]
last_inputs = st.session_state.get("last_inputs", {})
top_replacements = result.get("top_replacements", [])

if show_debug:
    with st.expander("Debug crudo del backend", expanded=False):
        st.write(result)

if not top_replacements:
    st.warning("No se encontraron reemplazos. Prueba bajar mínimo de partidos o minutos.")
    st.stop()

baseline = result.get("baseline", {})
baseline_probability = baseline.get("win_probability_without_replacement")
summary_df = build_summary_dataframe(top_replacements, baseline_probability)

# ============================================================
# Resultados
# ============================================================

st.markdown('<div id="resultados"></div>', unsafe_allow_html=True)
st.markdown("---")
st.header("Resultado del escenario")

best = top_replacements[0]
metric_cols = st.columns(4)
with metric_cols[0]:
    st.markdown(f"<div class='metric-card'><div class='label'>Equipo</div><div class='value'>{selected_team}</div></div>", unsafe_allow_html=True)
with metric_cols[1]:
    st.markdown(f"<div class='metric-card'><div class='label'>Reemplazado</div><div class='value'>{last_inputs.get('replaced_name', 'N/A')}</div></div>", unsafe_allow_html=True)
with metric_cols[2]:
    st.markdown(f"<div class='metric-card'><div class='label'>Mejor opción</div><div class='value'>{best.get('player_name', 'N/A')}</div></div>", unsafe_allow_html=True)
with metric_cols[3]:
    st.markdown(f"<div class='metric-card'><div class='label'>Victoria estimada</div><div class='value'>{format_probability(best.get('win_probability_with_replacement'))}</div></div>", unsafe_allow_html=True)

st.subheader("Player cards: recomendaciones desde la banca")
lineup_set = set(int(x) for x in last_inputs.get("lineup_player_ids", []))
result_lineup_context = build_lineup_context(
    [int(x) for x in last_inputs.get("lineup_player_ids", [])],
    int(last_inputs.get("replaced_player_id")),
    id_to_row,
)
st.markdown(render_lineup_balance_box(result_lineup_context, compact=True), unsafe_allow_html=True)

for idx, r in enumerate(top_replacements, start=1):
    low_activity = safe_bool(r.get("low_activity_flag", False))
    warning = "<span class='pill warning-pill'>Poca actividad reciente</span>" if low_activity else ""
    context_alert = replacement_context_alert(r, result_lineup_context)
    context_alert_html = render_context_alert(context_alert)
    explanation = human_explanation(
        r,
        last_inputs.get("replaced_name", "el jugador reemplazado"),
        result_lineup_context,
        context_alert,
    )
    st.markdown(
        f"""
        <div class="player-card">
          <div class="pill rank-pill">#{idx}</div> {warning}
          <h3>{html_escape(r.get('player_name', 'Jugador'))}</h3>
          <div class="player-subtitle">{html_escape(position_label(r.get('position')))} | {html_escape(r.get('latest_team', r.get('team', '')))}</div>
          <span class="pill">Calificación: {format_score_10(r.get("recommendation_score", r.get("replacement_score")))} </span>
          <span class="pill">Victoria: {format_probability(r.get('win_probability_with_replacement'))}</span>
          <span class="pill">PTS: {format_number(r.get('points'), 1)}</span>
          <span class="pill">MIN: {format_number(r.get('minutes'), 1)}</span>
          <span class="pill">PJ: {int(safe_float(r.get('games_played'), 0))}</span>
          <span class="pill">Actividad: {format_number(r.get('activity_score'), 2)}</span>
          {context_alert_html}
          <p style="margin-top:14px; color:var(--nba-muted); line-height:1.58;">{html_escape(explanation)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Validaciones visibles
valid_same_team = all(str(r.get("latest_team", r.get("team", ""))).upper() == selected_team.upper() for r in top_replacements)
valid_not_lineup = all(int(r.get("player_id")) not in lineup_set for r in top_replacements)
valid_not_replaced = all(int(r.get("player_id")) != int(last_inputs.get("replaced_player_id")) for r in top_replacements)
st.markdown("<div class='section-card'>", unsafe_allow_html=True)
st.subheader("Validación rápida")
st.caption("Estos checks confirman que la recomendación sí viene desde la banca del equipo seleccionado.")
vc1, vc2, vc3 = st.columns(3)
with vc1:
    validation_card("Mismo equipo", valid_same_team, "Todos los candidatos pertenecen al equipo elegido.")
with vc2:
    validation_card("Fuera de la quinteta", valid_not_lineup, "Ningún recomendado está entre los 5 en cancha.")
with vc3:
    validation_card("Reemplazado excluido", valid_not_replaced, "El jugador a reemplazar no aparece como candidato.")
st.markdown("</div>", unsafe_allow_html=True)

with st.expander("Ver tabla detallada", expanded=False):
    display_df = clean_display_dataframe(summary_df)
    show_safe_dataframe(display_df)

with st.expander("¿Cómo leer los resultados?", expanded=False):
    st.markdown(
        """
        - **Calificación total:** se muestra en escala de 0 a 10. Resume qué tan buena es la opción considerando similitud, ajuste al equipo, impacto esperado, actividad reciente y tendencia.
        - **Similitud:** mide qué tanto se parece el candidato al jugador que quieres reemplazar.
        - **Actividad reciente:** indica si el jugador tuvo minutos recientes suficientes para confiar más en la recomendación.
        - **Impacto estimado:** indica si el equipo podría mejorar o empeorar con ese reemplazo.
        - **Probabilidad estimada de victoria:** es una simulación de muchos escenarios posibles. No es una predicción perfecta, pero ayuda a comparar opciones.
        """
    )

# ============================================================
# Visualizaciones
# ============================================================

st.markdown('<div id="graficas"></div>', unsafe_allow_html=True)
st.header("Visualizaciones")
chart_df = numeric_chart_data(
    summary_df.copy(),
    [
        "Calificación total",
        "Puntos promedio",
        "Minutos promedio",
        "Probabilidad de victoria",
        "Impacto estimado",
    ],
)

st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
st.markdown("<div class='chart-title'>A. Comparación de calificación total</div>", unsafe_allow_html=True)
st.markdown("<div class='chart-caption'>Compara qué tan fuerte es cada candidato según el modelo. La escala va de 0 a 10: más alto significa mejor opción.</div>", unsafe_allow_html=True)
altair_bar(chart_df, "Jugador recomendado", "Calificación total", "Calificación total de recomendación", "Calificación total (0 a 10)", color="#1d428a")
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
st.markdown("<div class='chart-title'>B. Puntos vs minutos</div>", unsafe_allow_html=True)
st.markdown("<div class='chart-caption'>Ayuda a ver qué jugadores producen más y suelen tener más responsabilidad en cancha.</div>", unsafe_allow_html=True)
altair_scatter_points_minutes(chart_df)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
st.markdown("<div class='chart-title'>C. Probabilidad estimada de victoria</div>", unsafe_allow_html=True)
st.markdown("<div class='chart-caption'>Muestra qué reemplazo genera el mejor escenario simulado. No es una predicción exacta, sino una comparación basada en datos.</div>", unsafe_allow_html=True)
prob_df = chart_df[["Jugador recomendado", "Probabilidad de victoria"]].copy()
prob_df["Probabilidad de victoria (%)"] = pd.to_numeric(prob_df["Probabilidad de victoria"], errors="coerce") * 100
altair_bar(prob_df, "Jugador recomendado", "Probabilidad de victoria (%)", "Probabilidad estimada de victoria", "Probabilidad estimada de victoria (%)", color="#f58420")
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
st.markdown("<div class='chart-title'>D. Impacto estimado</div>", unsafe_allow_html=True)
st.markdown("<div class='chart-caption'>Valores más altos indican un mejor impacto esperado para el equipo. Valores negativos sugieren pérdida de eficiencia frente al jugador reemplazado.</div>", unsafe_allow_html=True)
altair_bar(chart_df, "Jugador recomendado", "Impacto estimado", "Impacto estimado en el equipo", "Impacto estimado en puntos de margen", color="#c8102e")
st.markdown("</div>", unsafe_allow_html=True)

if show_debug or developer_mode:
    with st.expander("Diagnóstico final", expanded=False):
        st.write("Roster debug:", result.get("roster_debug"))
        st.write("Inputs:", last_inputs)
        st.write("Summary:")
        st.dataframe(summary_df, use_container_width=True)
