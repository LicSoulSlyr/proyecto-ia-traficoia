"""Módulo supervisado de TráficoIA (semana 2): predecir el nivel de tráfico.

Todo el código reutilizable vive aquí; el cuaderno 02_supervisado.ipynb solo lo
llama, grafica e interpreta. Depende de src/config.py y src/datos.py (semana 1).

Decisiones clave:
* Los cortes de nivel (bajo / medio / alto) se calculan SOLO con el conjunto de
  entrenamiento y se aplican después a prueba (evita fuga de información).
* La elección del mejor modelo se hace con validación cruzada sobre
  entrenamiento, nunca mirando el conjunto de prueba.
"""
from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from . import config, datos

# Variables de entrada pedidas en la especificación
FEATURES = ["hora", "dia_semana", "temp_c", "rain_1h", "festivo_dia"]
# Variante para el experimento: hora y día como círculo (23 h queda cerca de 0 h)
FEATURES_CICLICAS = ["hora_sin", "hora_cos", "dia_sin", "dia_cos", "temp_c", "rain_1h", "festivo_dia"]

RUTA_MODELO = config.MODELOS / "mejor_modelo_supervisado.joblib"
BASE = "Base (clase más frecuente)"


# --------------------------------------------------------------------------- datos
def cargar_datos_modelado() -> pd.DataFrame:
    """Datos limpios listos para modelar (reconstruidos desde el CSV original)."""
    if config.CSV_TRAFICO.exists():
        df, _ = datos.limpiar_trafico(datos.cargar_trafico_crudo())
        return df
    if config.CSV_TRAFICO_LIMPIO.exists():
        df = pd.read_csv(config.CSV_TRAFICO_LIMPIO, parse_dates=["date_time", "fecha"])
        df["festivo_dia"] = df["festivo_dia"].astype(bool)
        return df
    raise FileNotFoundError("No hay datos. Ejecuta: python src/descargar_datos.py --tabular")


def preparar_features(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Arma la matriz X con las columnas pedidas (calcula las cíclicas si hacen falta)."""
    X = pd.DataFrame(index=df.index)
    for c in features:
        if c in df.columns:
            X[c] = df[c].astype(float)
        elif c == "hora_sin":
            X[c] = np.sin(2 * np.pi * df["hora"] / 24)
        elif c == "hora_cos":
            X[c] = np.cos(2 * np.pi * df["hora"] / 24)
        elif c == "dia_sin":
            X[c] = np.sin(2 * np.pi * df["dia_semana"] / 7)
        elif c == "dia_cos":
            X[c] = np.cos(2 * np.pi * df["dia_semana"] / 7)
        else:
            raise KeyError(f"No sé construir la variable '{c}'")
    return X


@dataclass
class Particion:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    cortes: tuple[float, float]
    features: list[str]
    tipo: str


def dividir(df: pd.DataFrame, features: list[str] | None = None, test_size: float = 0.2,
            semilla: int = config.SEMILLA, temporal: bool = False) -> Particion:
    """80 % entrenamiento / 20 % prueba. Los cortes de nivel salen SOLO de entrenamiento.

    ``temporal=True`` entrena con el 80 % más antiguo y prueba con el 20 % más reciente
    (una prueba más exigente y más parecida a predecir el futuro).
    """
    features = features or FEATURES
    if temporal:
        orden = df.sort_values("date_time").index
        n_train = int(len(orden) * (1 - test_size))
        idx_train, idx_test = orden[:n_train], orden[n_train:]
    else:
        idx_train, idx_test = train_test_split(df.index, test_size=test_size, random_state=semilla)

    cortes = datos.cortes_nivel(df.loc[idx_train, "traffic_volume"])

    def etiquetas(idx):
        v = df.loc[idx, "traffic_volume"]
        return pd.Series(datos.asignar_nivel(v, cortes), index=idx).astype(str)

    X = preparar_features(df, features)
    return Particion(X.loc[idx_train], X.loc[idx_test], etiquetas(idx_train), etiquetas(idx_test),
                     cortes, list(features), "temporal" if temporal else "aleatoria")


# --------------------------------------------------------------------------- modelos
def construir_modelos(semilla: int = config.SEMILLA) -> dict:
    """Modelo base + 4 algoritmos, cada uno con su rejilla pequeña de hiperparámetros."""
    def imp():
        return SimpleImputer(strategy="median")  # hay pocos NaN en temp_c y rain_1h

    return {
        BASE: (Pipeline([("imp", imp()), ("m", DummyClassifier(strategy="most_frequent"))]), {}),
        "Árbol de decisión": (
            Pipeline([("imp", imp()), ("m", DecisionTreeClassifier(random_state=semilla))]),
            {"m__max_depth": [3, 5, 8, 12, 16], "m__min_samples_leaf": [1, 10, 30]}),
        "Random Forest": (
            Pipeline([("imp", imp()), ("m", RandomForestClassifier(n_estimators=150, n_jobs=-1, random_state=semilla))]),
            {"m__max_depth": [8, 14, None], "m__min_samples_leaf": [1, 5]}),
        "KNN": (
            Pipeline([("imp", imp()), ("esc", StandardScaler()), ("m", KNeighborsClassifier())]),
            {"m__n_neighbors": [5, 15, 31]}),
        "Regresión logística": (
            Pipeline([("imp", imp()), ("esc", StandardScaler()), ("m", LogisticRegression(max_iter=1000))]),
            {"m__C": [0.1, 1, 10]}),
    }


def buscar_mejores(part: Particion, modelos: dict | None = None, cv: int = 5,
                   semilla: int = config.SEMILLA) -> dict[str, GridSearchCV]:
    """Validación cruzada de 5 particiones (estratificada) dentro del conjunto de entrenamiento."""
    modelos = modelos or construir_modelos(semilla)
    kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=semilla)
    salida = {}
    for nombre, (pipe, rejilla) in modelos.items():
        gs = GridSearchCV(pipe, rejilla, cv=kf, n_jobs=-1, refit="f1_macro", return_train_score=True,
                          scoring={"f1_macro": "f1_macro", "accuracy": "accuracy"})
        gs.fit(part.X_train, part.y_train)
        salida[nombre] = gs
    return salida


def tabla_cv(resultados: dict[str, GridSearchCV]) -> pd.DataFrame:
    """Resumen de la validación cruzada (mejor combinación de cada modelo)."""
    filas = []
    for nombre, gs in resultados.items():
        r, i = gs.cv_results_, gs.best_index_
        filas.append({
            "modelo": nombre,
            "mejores_parametros": {k.replace("m__", ""): v for k, v in gs.best_params_.items()},
            "exactitud_cv": r["mean_test_accuracy"][i],
            "f1_macro_cv": r["mean_test_f1_macro"][i],
            "f1_macro_cv_desv": r["std_test_f1_macro"][i],
            "f1_macro_entrenamiento": r["mean_train_f1_macro"][i],
        })
    t = pd.DataFrame(filas)
    t["brecha_sobreajuste"] = t["f1_macro_entrenamiento"] - t["f1_macro_cv"]
    return t


def evaluar_en_prueba(resultados: dict[str, GridSearchCV], part: Particion):
    """Evalúa una sola vez en prueba. Devuelve (tabla, predicciones por modelo)."""
    filas, preds = [], {}
    for nombre, gs in resultados.items():
        p = gs.best_estimator_.predict(part.X_test)
        preds[nombre] = pd.Series(p, index=part.X_test.index)
        filas.append({"modelo": nombre,
                      "exactitud_prueba": accuracy_score(part.y_test, p),
                      "f1_macro_prueba": f1_score(part.y_test, p, average="macro")})
    return pd.DataFrame(filas), preds


def reentrenar_en(resultados: dict[str, GridSearchCV], nombre: str, part: Particion) -> dict:
    """Reentrena el modelo (con sus mejores hiperparámetros) en otra partición o con otras variables."""
    est = clone(resultados[nombre].best_estimator_).fit(part.X_train, part.y_train)
    p = est.predict(part.X_test)
    return {"exactitud": accuracy_score(part.y_test, p),
            "f1_macro": f1_score(part.y_test, p, average="macro")}


# --------------------------------------------------------------------------- guardar / usar
def crear_paquete(resultados: dict[str, GridSearchCV], nombre: str, part: Particion, metricas: dict) -> dict:
    """Todo lo que la app necesita: modelo, variables, cortes de nivel y métricas."""
    return {
        "version": "semana2",
        "nombre": nombre,
        "modelo": resultados[nombre].best_estimator_,
        "features": part.features,
        "cortes": part.cortes,
        "niveles": config.NIVELES,
        "metricas": metricas,
    }


def guardar_modelo(paquete: dict, ruta=RUTA_MODELO):
    ruta.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(paquete, ruta)
    return ruta


def cargar_modelo(ruta=RUTA_MODELO) -> dict:
    if not ruta.exists():
        raise FileNotFoundError(f"No existe {ruta}. Ejecuta el cuaderno 02_supervisado.ipynb.")
    return joblib.load(ruta)


def predecir_nivel(paquete: dict, hora: int, dia_semana: int, temp_c: float = 10.0,
                   rain_1h: float = 0.0, festivo: bool = False) -> dict:
    """Predice el nivel de tráfico de una hora. Lo usará la app Streamlit (semana 4).

    dia_semana: 0 = lunes ... 6 = domingo.
    """
    fila = pd.DataFrame([{"hora": hora, "dia_semana": dia_semana, "temp_c": temp_c,
                          "rain_1h": rain_1h, "festivo_dia": float(festivo)}])
    X = preparar_features(fila, paquete["features"])
    modelo = paquete["modelo"]
    nivel = modelo.predict(X)[0]
    probs = dict(zip(modelo.classes_, modelo.predict_proba(X)[0]))
    return {"nivel": nivel, "probabilidades": {n: float(probs.get(n, 0.0)) for n in paquete["niveles"]}}
