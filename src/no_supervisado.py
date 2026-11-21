"""Módulo no supervisado de TráficoIA (semana 2): tipos de día por su perfil horario.

Cada día es un punto de 24 valores (volumen promedio por hora). Se escala con
StandardScaler y se agrupa con K-Means y DBSCAN; PCA solo sirve para visualizar.
Depende de src/config.py y src/datos.py (semana 1).
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from . import config, datos

RUTA_KMEANS = config.MODELOS / "kmeans_dias.joblib"
RUTA_DBSCAN = config.MODELOS / "dbscan_dias.joblib"
DIAS = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]

# Supuesto de partida para el semáforo (se valida con el simulador de la semana 3)
CICLO_POR_NIVEL = {"bajo": 45, "medio": 70, "alto": 100}  # segundos del ciclo completo


# --------------------------------------------------------------------------- datos
def cargar_datos_limpios() -> pd.DataFrame:
    """Datos limpios (reconstruidos desde el CSV original con las funciones de la semana 1)."""
    if config.CSV_TRAFICO.exists():
        df, _ = datos.limpiar_trafico(datos.cargar_trafico_crudo())
        return df
    if config.CSV_TRAFICO_LIMPIO.exists():
        df = pd.read_csv(config.CSV_TRAFICO_LIMPIO, parse_dates=["date_time", "fecha"])
        df["festivo_dia"] = df["festivo_dia"].astype(bool)
        return df
    raise FileNotFoundError("No hay datos. Ejecuta: python src/descargar_datos.py --tabular")


def matriz_perfiles(df: pd.DataFrame, max_faltantes: int = 3) -> pd.DataFrame:
    """Días x 24 horas. Tolera hasta ``max_faltantes`` horas sin dato (se interpolan)."""
    return datos.perfil_horario_diario(df, max_faltantes=max_faltantes)


def metadatos_dias(df: pd.DataFrame) -> pd.DataFrame:
    """Una fila por día con su día de la semana y si fue festivo."""
    return df.groupby("fecha").agg(dia_semana=("dia_semana", "first"),
                                   festivo=("festivo_dia", "max"),
                                   fin_de_semana=("fin_de_semana", "max"))


def escalar(perfiles: pd.DataFrame):
    """StandardScaler por hora (cada una de las 24 columnas)."""
    esc = StandardScaler()
    return esc, esc.fit_transform(perfiles.values)


# --------------------------------------------------------------------------- K-Means
def evaluar_k(X: np.ndarray, ks=range(2, 11), semilla: int = config.SEMILLA) -> pd.DataFrame:
    """Inercia (método del codo) y coeficiente de silueta para cada k."""
    filas = []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=semilla).fit(X)
        filas.append({"k": k, "inercia": km.inertia_, "silueta": silhouette_score(X, km.labels_)})
    return pd.DataFrame(filas)


def k_por_silueta(tabla: pd.DataFrame) -> int:
    return int(tabla.loc[tabla["silueta"].idxmax(), "k"])


def kmeans_final(X: np.ndarray, k: int, semilla: int = config.SEMILLA) -> KMeans:
    return KMeans(n_clusters=k, n_init=20, random_state=semilla).fit(X)


# --------------------------------------------------------------------------- DBSCAN
def distancias_kvecinos(X: np.ndarray, k: int) -> np.ndarray:
    """Distancia ordenada al k-ésimo vecino: su 'codo' sugiere un eps razonable."""
    d, _ = NearestNeighbors(n_neighbors=k).fit(X).kneighbors(X)
    return np.sort(d[:, -1])


def evaluar_dbscan(X: np.ndarray, eps_lista, min_samples: int) -> pd.DataFrame:
    filas = []
    for eps in eps_lista:
        etiquetas = DBSCAN(eps=float(eps), min_samples=min_samples).fit_predict(X)
        ruido = etiquetas == -1
        n_cl = len(set(etiquetas[~ruido]))
        sil = np.nan
        if n_cl >= 2 and (~ruido).sum() > n_cl:
            sil = silhouette_score(X[~ruido], etiquetas[~ruido])
        filas.append({"eps": round(float(eps), 3), "min_samples": min_samples, "n_grupos": n_cl,
                      "proporcion_ruido": ruido.mean(), "silueta_sin_ruido": sil})
    return pd.DataFrame(filas)


def elegir_eps(tabla: pd.DataFrame, max_ruido: float = 0.15) -> pd.Series:
    """Mejor silueta entre las configuraciones con >= 2 grupos y poco ruido (regla simple y explicable)."""
    ok = tabla[(tabla["n_grupos"] >= 2) & (tabla["proporcion_ruido"] <= max_ruido)]
    if len(ok):
        return ok.loc[ok["silueta_sin_ruido"].idxmax()]
    con_grupos = tabla[tabla["n_grupos"] >= 1]
    if len(con_grupos):
        return con_grupos.loc[con_grupos["proporcion_ruido"].idxmin()]
    return tabla.iloc[-1]


# --------------------------------------------------------------------------- interpretación
def _picos(perfil: pd.Series) -> tuple[int, int]:
    manana = perfil.loc[5:11].idxmax()
    tarde = perfil.loc[14:19].idxmax()
    return int(manana), int(tarde)


def describir_clusters(perfiles: pd.DataFrame, etiquetas, meta: pd.DataFrame) -> pd.DataFrame:
    """Una fila por grupo: tamaño, composición calendario, volumen y horas pico."""
    et = pd.Series(etiquetas, index=perfiles.index, name="grupo")
    m = meta.reindex(perfiles.index)
    medios = perfiles.groupby(et).mean()
    filas = []
    for g in sorted(et.unique()):
        dias = et[et == g].index
        pico_m, pico_t = _picos(medios.loc[g])
        filas.append({"grupo": g, "n_dias": len(dias),
                      "pct_fin_de_semana": m.loc[dias, "fin_de_semana"].mean(),
                      "pct_festivo": m.loc[dias, "festivo"].mean(),
                      "volumen_medio": medios.loc[g].mean(),
                      "hora_pico_manana": pico_m, "hora_pico_tarde": pico_t,
                      "volumen_pico": medios.loc[g].max()})
    return pd.DataFrame(filas).set_index("grupo")


def nombrar_clusters(resumen: pd.DataFrame) -> dict[int, str]:
    """Nombres automáticos de partida. Revísalos y corrígelos a mano en el cuaderno."""
    base = {g: ("Fin de semana o festivo" if r["pct_fin_de_semana"] >= 0.6 else "Laboral")
            for g, r in resumen.iterrows()}
    nombres: dict[int, str] = {}
    for nombre in sorted(set(base.values())):
        miembros = sorted([g for g in resumen.index if base[g] == nombre],
                          key=lambda g: -resumen.loc[g, "volumen_medio"])
        for i, g in enumerate(miembros):
            nombres[g] = nombre if len(miembros) == 1 else f"{nombre} {chr(65 + i)}"
    return nombres


def cortes_globales(perfiles: pd.DataFrame) -> tuple[float, float]:
    """Terciles de todos los valores horarios: separan hora de volumen bajo, medio y alto."""
    q1, q2 = perfiles.stack().quantile([1 / 3, 2 / 3])
    return float(q1), float(q2)


def _nivel(v: float, cortes) -> str:
    return "bajo" if v <= cortes[0] else ("medio" if v <= cortes[1] else "alto")


def propuesta_semaforo(perfil_medio: pd.Series, cortes) -> pd.DataFrame:
    """Franjas horarias con el ciclo de semáforo sugerido (hipótesis a validar en la semana 3)."""
    niveles = [_nivel(v, cortes) for v in perfil_medio.values]
    horas = list(perfil_medio.index)
    filas, ini = [], 0
    for i in range(1, len(horas) + 1):
        if i == len(horas) or niveles[i] != niveles[ini]:
            fin = i - 1
            tramo = perfil_medio.iloc[ini:i]
            filas.append({"franja": f"{horas[ini]:02d}:00 a {horas[fin]:02d}:59",
                          "nivel": niveles[ini],
                          "volumen_medio": round(float(tramo.mean())),
                          "ciclo_sugerido_s": CICLO_POR_NIVEL[niveles[ini]]})
            ini = i
    return pd.DataFrame(filas)


# --------------------------------------------------------------------------- guardar / usar
def guardar(paquete: dict, ruta) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(paquete, ruta)


def cargar_kmeans(ruta=RUTA_KMEANS) -> dict:
    if not ruta.exists():
        raise FileNotFoundError(f"No existe {ruta}. Ejecuta el cuaderno 03_no_supervisado.ipynb.")
    return joblib.load(ruta)


def clasificar_dia(paquete: dict, perfil24) -> dict:
    """Asigna un día nuevo (24 valores de volumen) a su tipo de día. Lo usará la app (semana 4)."""
    x = np.asarray(perfil24, dtype=float).reshape(1, -1)
    if x.shape[1] != 24:
        raise ValueError("El perfil debe tener 24 valores (uno por hora).")
    g = int(paquete["kmeans"].predict(paquete["scaler"].transform(x))[0])
    return {"grupo": g, "tipo_de_dia": paquete["nombres"][g]}
