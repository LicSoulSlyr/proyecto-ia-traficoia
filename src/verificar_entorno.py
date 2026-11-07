"""Verifica que el entorno `proyecto_ia` tenga todo lo necesario.

Uso (con el entorno activado):
    python src/verificar_entorno.py

Toma una captura de pantalla de la salida para el entregable del sábado.
"""
import importlib
import platform
import sys
from datetime import datetime

PAQUETES = [
    ("numpy", "numpy"), ("pandas", "pandas"), ("scipy", "scipy"),
    ("scikit-learn", "sklearn"), ("matplotlib", "matplotlib"),
    ("seaborn", "seaborn"), ("joblib", "joblib"), ("pillow", "PIL"),
    ("jupyterlab", "jupyterlab"), ("torch", "torch"),
    ("torchvision", "torchvision"), ("gymnasium", "gymnasium"),
    ("stable-baselines3", "stable_baselines3"), ("streamlit", "streamlit"),
    ("gradio", "gradio"),
]


def main() -> int:
    print("=" * 60)
    print(" TráficoIA · Verificación del entorno")
    print("=" * 60)
    print(f" Fecha        : {datetime.now():%Y-%m-%d %H:%M}")
    print(f" Sistema      : {platform.system()} {platform.release()}")
    print(f" Python       : {sys.version.split()[0]}")
    print(f" Ejecutable   : {sys.executable}")
    ok_python = sys.version_info[:2] == (3, 11)
    print(f" Python 3.11  : {'OK' if ok_python else 'AVISO: se pidió 3.11'}")
    print("-" * 60)

    faltan = []
    for nombre, modulo in PAQUETES:
        try:
            mod = importlib.import_module(modulo)
            version = getattr(mod, "__version__", "instalado")
            print(f" [OK]   {nombre:<20} {version}")
        except Exception as exc:  # noqa: BLE001
            faltan.append(nombre)
            print(f" [FALTA] {nombre:<19} {type(exc).__name__}")

    print("-" * 60)
    try:
        import torch
        cuda = torch.cuda.is_available()
        mps = getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
        dispositivo = "GPU CUDA" if cuda else ("GPU Apple (MPS)" if mps else "CPU")
        print(f" PyTorch usará: {dispositivo}")
        if dispositivo == "CPU":
            print(" Nota: sin GPU, la CNN puede entrenarse en Google Colab.")
    except Exception:  # noqa: BLE001
        pass

    print("=" * 60)
    if faltan:
        print(" Faltan paquetes:", ", ".join(faltan))
        print(" Revisa la sección de instalación del README.")
        return 1
    print(" Entorno listo")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
