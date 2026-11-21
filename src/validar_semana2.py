"""Comprobación rápida de la semana 2: ¿existen los modelos y funcionan?

Uso (con el entorno activado, desde la raíz del repositorio):
    python src/validar_semana2.py

Termina con código 0 si todo está bien y 1 si algo falta.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config, no_supervisado, supervisado  # noqa: E402

ARCHIVOS = [
    config.MODELOS / "mejor_modelo_supervisado.joblib",
    config.MODELOS / "kmeans_dias.joblib",
    config.MODELOS / "dbscan_dias.joblib",
    config.RAIZ / "reports" / "tabla_comparativa_supervisado.csv",
    config.RAIZ / "reports" / "tabla_clusters.csv",
    config.RAIZ / "reports" / "propuesta_semaforo.csv",
    config.RAIZ / "notebooks" / "02_supervisado.ipynb",
    config.RAIZ / "notebooks" / "03_no_supervisado.ipynb",
]


def main() -> int:
    errores = 0
    print("=" * 60)
    print(" TráficoIA · Validación de la semana 2")
    print("=" * 60)
    for ruta in ARCHIVOS:
        ok = ruta.exists()
        errores += not ok
        print(f" [{'OK' if ok else 'FALTA'}] {ruta.relative_to(config.RAIZ)}")

    try:
        p = supervisado.cargar_modelo()
        r = supervisado.predecir_nivel(p, hora=8, dia_semana=1)
        print(f"\n Supervisado: {p['nombre']} | F1 macro prueba = {p['metricas'].get('f1_macro_prueba', float('nan')):.3f}")
        print(f"   Martes 08:00 -> {r['nivel']}  {r['probabilidades']}")
    except Exception as exc:  # noqa: BLE001
        errores += 1
        print(f"\n [ERROR] modelo supervisado: {exc}")

    try:
        k = no_supervisado.cargar_kmeans()
        medio = [0] * 24
        r = no_supervisado.clasificar_dia(k, medio)
        print(f"\n K-Means: {k['k']} tipos de día -> {list(k['nombres'].values())}")
        print(f"   Perfil de ceros se asigna a: {r['tipo_de_dia']}")
    except Exception as exc:  # noqa: BLE001
        errores += 1
        print(f"\n [ERROR] modelo no supervisado: {exc}")

    print("\n" + ("Todo en orden." if not errores else f"Hay {errores} problema(s) por resolver."))
    return 1 if errores else 0


if __name__ == "__main__":
    raise SystemExit(main())
