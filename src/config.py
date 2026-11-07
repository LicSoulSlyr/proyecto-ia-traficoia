"""Rutas y constantes compartidas por scripts y cuadernos de TráficoIA."""
from pathlib import Path

# Raíz del repositorio (carpeta que contiene src/, data/, notebooks/...)
RAIZ = Path(__file__).resolve().parents[1]
DATA_RAW = RAIZ / "data" / "raw"
DATA_PROCESSED = RAIZ / "data" / "processed"
FIGURAS = RAIZ / "reports" / "figures"
MODELOS = RAIZ / "models"

# --- Dataset tabular: Metro Interstate Traffic Volume (UCI #492) ---
DIR_TRAFICO = DATA_RAW / "metro_traffic"
CSV_TRAFICO = DIR_TRAFICO / "Metro_Interstate_Traffic_Volume.csv"
URL_TRAFICO_ZIP = (
    "https://archive.ics.uci.edu/static/public/492/"
    "metro+interstate+traffic+volume.zip"
)
CSV_TRAFICO_LIMPIO = DATA_PROCESSED / "traffic_clean.csv"

# --- Dataset de imágenes: GTSRB ---
DIR_GTSRB = DATA_RAW / "gtsrb"

# Clases del prototipo (propuesta inicial; se confirma en la propuesta del equipo).
# 5 límites de velocidad (para analizar confusiones) + 7 señales distintas = 12 clases.
SUBSET_CLASES = [1, 2, 3, 4, 5, 13, 14, 17, 18, 25, 28, 38]

NOMBRES_CLASES = {
    0: "Límite 20 km/h", 1: "Límite 30 km/h", 2: "Límite 50 km/h",
    3: "Límite 60 km/h", 4: "Límite 70 km/h", 5: "Límite 80 km/h",
    6: "Fin límite 80 km/h", 7: "Límite 100 km/h", 8: "Límite 120 km/h",
    9: "Prohibido adelantar", 10: "Prohibido adelantar (camiones)",
    11: "Prioridad en próxima intersección", 12: "Calzada con prioridad",
    13: "Ceda el paso", 14: "Alto", 15: "Circulación prohibida",
    16: "Prohibido camiones", 17: "Entrada prohibida", 18: "Precaución general",
    19: "Curva peligrosa a la izquierda", 20: "Curva peligrosa a la derecha",
    21: "Curva doble", 22: "Camino irregular", 23: "Calzada deslizante",
    24: "Estrechamiento por la derecha", 25: "Obras en la vía", 26: "Semáforo",
    27: "Peatones", 28: "Niños", 29: "Bicicletas", 30: "Hielo o nieve",
    31: "Animales salvajes", 32: "Fin de todas las restricciones",
    33: "Giro obligatorio a la derecha", 34: "Giro obligatorio a la izquierda",
    35: "Siga recto", 36: "Recto o derecha", 37: "Recto o izquierda",
    38: "Mantenga la derecha", 39: "Mantenga la izquierda",
    40: "Rotonda obligatoria", 41: "Fin prohibición de adelantar",
    42: "Fin prohibición de adelantar (camiones)",
}

# Niveles de tráfico (variable objetivo del módulo supervisado)
NIVELES = ["bajo", "medio", "alto"]

SEMILLA = 42
