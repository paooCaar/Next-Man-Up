# NBA Player Replacement Recommender

Proyecto de recomendación de reemplazos de jugadores NBA usando similitud de rol, ajuste al equipo, ajuste contra el rival, impacto estimado y simulación Monte Carlo.

El objetivo es responder una pregunta práctica:

> Si un equipo pierde o reemplaza a un jugador, ¿qué otros jugadores podrían cumplir un rol similar y mejorar la probabilidad de ganar contra un rival específico?

---

## 1. ¿Qué hace el proyecto?

El sistema recomienda reemplazos para un jugador seleccionado considerando:

- similitud de rol entre jugadores
- compatibilidad de posición
- ajuste al estilo del equipo
- ajuste contra el equipo rival
- impacto ofensivo y defensivo estimado
- simulación Monte Carlo para estimar probabilidad de victoria

El proyecto está diseñado para comenzar con CSV simples y luego poder conectarse a fuentes reales de datos NBA sin cambiar la lógica principal del modelo.

---

## 2. Estructura del proyecto

```text
nba-player-replacement-recommender/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── run_recommender.py
├── streamlit_app.py
│
├── data/
│   ├── raw/
│   │   └── .gitkeep
│   └── processed/
│       └── .gitkeep
│
├── src/
│   ├── __init__.py
│   ├── role_features.py
│   ├── similarity.py
│   ├── impact.py
│   ├── monte_carlo.py
│   ├── explanations.py
│   ├── recommender.py
│   └── preprocessing.py
│
└── notebooks/
    └── .gitkeep