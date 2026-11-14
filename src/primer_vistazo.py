"""Primer vistazo a los datos (entregable del sábado): dimensiones, tipos,
faltantes y ejemplos de imágenes.

Uso:  python src/primer_vistazo.py
Guarda el resumen en reports/primer_vistazo.txt y una figura en reports/figures/.
"""
from __future__ import annotations

import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

from src import config, datos  # noqa: E402


def vistazo_tabular() -> None:
    print("=" * 60)
    print("DATASET TABULAR · Metro Interstate Traffic Volume")
    print("=" * 60)
    try:
        df = datos.cargar_trafico_crudo()
    except FileNotFoundError as exc:
        print(exc)
        return
    print(f"Dimensiones: {df.shape[0]:,} filas x {df.shape[1]} columnas\n")
    print("Tipos de datos:")
    print(df.dtypes.to_string(), "\n")
    print("Valores faltantes por columna:")
    print(df.isna().sum().to_string(), "\n")
    print("Primeras filas:")
    print(df.head().to_string(), "\n")
    print("Resumen numérico:")
    print(df.describe().T.to_string(), "\n")
    rango = (df["date_time"].min(), df["date_time"].max())
    print(f"Rango de fechas: {rango[0]}  ->  {rango[1]}")
    print(f"Filas con fecha-hora repetida: {df['date_time'].duplicated().sum():,}")


def vistazo_imagenes() -> None:
    print("\n" + "=" * 60)
    print("DATASET DE IMÁGENES · GTSRB")
    print("=" * 60)
    entren = datos.carpeta_entrenamiento_gtsrb()
    if entren is None:
        print("No se encontró GTSRB. Ejecuta: python src/descargar_datos.py --imagenes")
        return
    conteo = datos.conteo_por_clase(entren)
    print(f"Carpeta de entrenamiento: {entren}")
    print(f"Clases: {len(conteo)} | Imágenes totales: {conteo.sum():,}")
    print(f"Mínimo por clase: {conteo.min()} (clase {conteo.idxmin()}) | "
          f"máximo: {conteo.max()} (clase {conteo.idxmax()})")

    clases = config.SUBSET_CLASES
    fig, ejes = plt.subplots(2, 6, figsize=(12, 4.6))
    for ax, c in zip(ejes.ravel(), clases):
        rutas = datos.imagenes_de_clase(entren, c)
        if rutas:
            ax.imshow(Image.open(rutas[len(rutas) // 2]))
        ax.set_title(f"{c}: {config.NOMBRES_CLASES[c][:18]}", fontsize=8)
        ax.axis("off")
    fig.suptitle("GTSRB · una muestra por cada clase propuesta", fontsize=11)
    fig.tight_layout()
    config.FIGURAS.mkdir(parents=True, exist_ok=True)
    salida = config.FIGURAS / "gtsrb_muestras.png"
    fig.savefig(salida, dpi=130)
    print(f"Figura guardada en {salida}")


def main() -> None:
    buffer = io.StringIO()

    class Tee:
        def write(self, s):
            sys.__stdout__.write(s)
            buffer.write(s)

        def flush(self):
            sys.__stdout__.flush()

    with redirect_stdout(Tee()):
        vistazo_tabular()
        vistazo_imagenes()
    ruta = config.RAIZ / "reports" / "primer_vistazo.txt"
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(buffer.getvalue(), encoding="utf-8")
    print(f"\nResumen guardado en {ruta}")


if __name__ == "__main__":
    main()
