"""Entrenamiento y evaluación emparejada de las políticas del cruce."""
import numpy as np
from src.cruce import Cruce


def entrenar(alfa, gamma, decaimiento, episodios=700, semilla=42):
    entorno = Cruce()
    rng = np.random.default_rng(semilla)
    q = np.zeros((5, 5, 2, 2))
    retornos = []
    for e in range(episodios):
        estado, _ = entorno.reset(seed=semilla + e)
        epsilon = max(0.03, decaimiento ** e)
        retorno = 0
        while True:
            valores = q[tuple(estado)]
            accion = rng.integers(2) if rng.random() < epsilon else valores.argmax()
            siguiente, premio, _, fin, _ = entorno.step(accion)
            # El horizonte corta la observación de un cruce que sigue operando.
            objetivo = premio + gamma * q[tuple(siguiente)].max()
            valores[accion] += alfa * (objetivo - valores[accion])
            estado = siguiente
            retorno += premio
            if fin:
                break
        retornos.append(retorno)
    return q, retornos


def evaluar(q, semillas):
    """q=None alterna la vía verde después de cada cinco pasos."""
    entorno = Cruce()
    espera = llegadas = 0
    retornos = []
    for semilla in semillas:
        estado, _ = entorno.reset(seed=semilla)
        retorno = 0
        while True:
            if q is None:
                accion = int(entorno.paso > 0 and entorno.paso % 5 == 0)
            else:
                accion = q[tuple(estado)].argmax()
            estado, premio, _, fin, info = entorno.step(accion)
            espera += info["auto_pasos"]
            llegadas += info["llegadas"]
            retorno += premio
            if fin:
                break
        retornos.append(retorno)
    return float(np.mean(retornos)), espera / llegadas
