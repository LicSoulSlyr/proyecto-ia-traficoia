"""Carga y limpieza de datos de TráficoIA (semana 1)."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import config


# ----------------------------------------------------------------------------
# Tabular: Metro Interstate Traffic Volume
# ----------------------------------------------------------------------------
def cargar_trafico_crudo(ruta: Path | None = None) -> pd.DataFrame:
    """Lee el CSV original sin modificarlo."""
    ruta = Path(ruta or config.CSV_TRAFICO)
    if not ruta.exists():
        raise FileNotFoundError(
            f"No se encontró {ruta}. Ejecuta: python src/descargar_datos.py --tabular"
        )
    return pd.read_csv(ruta)


def limpiar_trafico(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Limpieza básica documentada. Devuelve (df_limpio, resumen_de_cambios).

    Decisiones (se justifican en 01_eda.ipynb):
    1. date_time a datetime y orden cronológico.
    2. La etiqueta de festivo suele aparecer solo en una fila del día; se propaga
       a todas las horas de ese día (``festivo_dia``).
    3. Hay varias filas por la misma hora (distintas descripciones del clima con el
       mismo volumen); se conserva una fila por hora.
    4. temp en Kelvin -> °C; temp <= 0 K es un error de sensor -> NaN.
    5. rain_1h > 500 mm/h es físicamente implausible -> NaN (queda marcado).
    """
    resumen: dict = {"filas_originales": len(df)}
    d = df.copy()
    d["date_time"] = pd.to_datetime(d["date_time"])
    d = d.sort_values("date_time").reset_index(drop=True)

    # 2. festivos
    festivo_fila = d["holiday"].fillna("None").astype(str).str.strip().ne("None")
    d["festivo_dia"] = festivo_fila.groupby(d["date_time"].dt.date).transform("max")
    resumen["filas_con_etiqueta_festivo"] = int(festivo_fila.sum())
    resumen["filas_en_dias_festivos"] = int(d["festivo_dia"].sum())

    # 3. duplicados por hora
    resumen["horas_duplicadas"] = int(d["date_time"].duplicated().sum())
    d = d.drop_duplicates(subset="date_time", keep="first").reset_index(drop=True)
    resumen["filas_tras_quitar_duplicados"] = len(d)

    # 4. temperatura
    d["temp_c"] = d["temp"] - 273.15
    malas_temp = d["temp"] <= 0
    resumen["temp_invalidas"] = int(malas_temp.sum())
    d.loc[malas_temp, "temp_c"] = np.nan

    # 5. lluvia
    d["rain_atipica"] = d["rain_1h"] > 500
    resumen["lluvia_atipica"] = int(d["rain_atipica"].sum())
    d.loc[d["rain_atipica"], "rain_1h"] = np.nan

    # Variables de calendario
    d["hora"] = d["date_time"].dt.hour
    d["dia_semana"] = d["date_time"].dt.dayofweek  # 0 = lunes
    d["mes"] = d["date_time"].dt.month
    d["anio"] = d["date_time"].dt.year
    d["fin_de_semana"] = d["dia_semana"] >= 5
    d["fecha"] = d["date_time"].dt.normalize()

    return d, resumen


def cortes_nivel(volumen: pd.Series) -> tuple[float, float]:
    """Cortes por terciles. En la semana 2 deben calcularse SOLO con el conjunto
    de entrenamiento para evitar fuga de información."""
    q1, q2 = volumen.quantile([1 / 3, 2 / 3])
    return float(q1), float(q2)


def asignar_nivel(volumen: pd.Series, cortes: tuple[float, float]) -> pd.Categorical:
    """bajo / medio / alto según los cortes dados."""
    q1, q2 = cortes
    etiquetas = pd.cut(
        volumen, bins=[-np.inf, q1, q2, np.inf], labels=config.NIVELES, right=True
    )
    return pd.Categorical(etiquetas, categories=config.NIVELES, ordered=True)


def perfil_horario_diario(d: pd.DataFrame, max_faltantes: int = 0) -> pd.DataFrame:
    """Matriz días x 24 horas con el volumen. Base del módulo no supervisado.

    ``max_faltantes``: horas faltantes toleradas por día (se interpolan).
    """
    tabla = d.pivot_table(index="fecha", columns="hora", values="traffic_volume", aggfunc="mean")
    tabla = tabla.reindex(columns=range(24))
    faltantes = tabla.isna().sum(axis=1)
    tabla = tabla[faltantes <= max_faltantes]
    if max_faltantes > 0:
        tabla = tabla.interpolate(axis=1, limit_direction="both")
    return tabla


# ----------------------------------------------------------------------------
# Imágenes: GTSRB
# ----------------------------------------------------------------------------
def carpeta_entrenamiento_gtsrb(raiz: Path | None = None) -> Path | None:
    """Localiza la carpeta .../GTSRB/Training (la ruta cambia según la descarga)."""
    raiz = Path(raiz or config.DIR_GTSRB)
    if not raiz.exists():
        return None
    for ruta in raiz.rglob("Training"):
        if ruta.is_dir() and any(re.fullmatch(r"\d{5}", p.name) for p in ruta.iterdir()):
            return ruta
    return None


def conteo_por_clase(entrenamiento: Path) -> pd.Series:
    """Número de imágenes por clase (índice 0..42)."""
    conteos = {}
    for carpeta in sorted(entrenamiento.iterdir()):
        if carpeta.is_dir() and re.fullmatch(r"\d{5}", carpeta.name):
            conteos[int(carpeta.name)] = sum(1 for _ in carpeta.glob("*.ppm"))
    serie = pd.Series(conteos, name="imagenes").sort_index()
    serie.index.name = "clase"
    return serie


def imagenes_de_clase(entrenamiento: Path, clase: int) -> list[Path]:
    return sorted((entrenamiento / f"{clase:05d}").glob("*.ppm"))
