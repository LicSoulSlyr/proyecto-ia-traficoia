"""Descarga los datasets de TráficoIA y registra su origen en un manifiesto.

Uso (con el entorno activado, desde la raíz del repositorio):
    python src/descargar_datos.py              # todo
    python src/descargar_datos.py --tabular    # solo Metro Interstate Traffic Volume
    python src/descargar_datos.py --imagenes   # solo GTSRB

Genera data/raw/manifest.json con fuente, fecha, tamaño y huella SHA-256.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import shutil
import sys
import urllib.request
import zipfile
from datetime import datetime
from pathlib import Path

# Permite ejecutar el script directamente: python src/descargar_datos.py
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402

MANIFIESTO = config.DATA_RAW / "manifest.json"


def _sha256(ruta: Path) -> str:
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def _registrar(clave: str, info: dict) -> None:
    config.DATA_RAW.mkdir(parents=True, exist_ok=True)
    datos = json.loads(MANIFIESTO.read_text(encoding="utf-8")) if MANIFIESTO.exists() else {}
    datos[clave] = info
    MANIFIESTO.write_text(json.dumps(datos, indent=2, ensure_ascii=False), encoding="utf-8")


def descargar_tabular() -> None:
    """Metro Interstate Traffic Volume (UCI #492), licencia CC BY 4.0."""
    destino = config.CSV_TRAFICO
    if destino.exists():
        print(f"[tabular] Ya existe: {destino}")
    else:
        config.DIR_TRAFICO.mkdir(parents=True, exist_ok=True)
        print(f"[tabular] Descargando {config.URL_TRAFICO_ZIP} ...")
        try:
            req = urllib.request.Request(
                config.URL_TRAFICO_ZIP, headers={"User-Agent": "Mozilla/5.0 (TraficoIA)"}
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                contenido = resp.read()
            with zipfile.ZipFile(io.BytesIO(contenido)) as zf:
                nombres = zf.namelist()
                print(f"[tabular] Archivos en el zip: {nombres}")
                for n in nombres:
                    if n.endswith(".csv.gz"):
                        with gzip.open(zf.open(n)) as gz, open(destino, "wb") as out:
                            shutil.copyfileobj(gz, out)
                        break
                    if n.endswith(".csv"):
                        with zf.open(n) as src, open(destino, "wb") as out:
                            shutil.copyfileobj(src, out)
                        break
                else:
                    raise RuntimeError("El zip no contiene un CSV reconocible.")
        except Exception as exc:  # noqa: BLE001
            print(f"[tabular] Falló la descarga directa ({exc}).")
            print("[tabular] Alternativa: descarga manual desde")
            print("          https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume")
            print(f"          y guarda el CSV en: {destino}")
            raise SystemExit(1)

    _registrar("metro_interstate_traffic_volume", {
        "fuente": "UCI Machine Learning Repository, dataset 492",
        "url": "https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume",
        "licencia": "CC BY 4.0 (verificar en la página del dataset)",
        "archivo": str(destino.relative_to(config.RAIZ)),
        "bytes": destino.stat().st_size,
        "sha256": _sha256(destino),
        "descargado": datetime.now().isoformat(timespec="seconds"),
    })
    print(f"[tabular] Listo: {destino} ({destino.stat().st_size / 1e6:.1f} MB)")


def descargar_imagenes() -> None:
    """GTSRB mediante torchvision (entrenamiento y prueba)."""
    try:
        from torchvision.datasets import GTSRB
    except ImportError:
        print("[imágenes] torchvision no está instalado. Revisa el entorno.")
        raise SystemExit(1)

    config.DIR_GTSRB.mkdir(parents=True, exist_ok=True)
    print("[imágenes] Descargando GTSRB (≈ 300 MB, puede tardar)...")
    try:
        GTSRB(root=str(config.DIR_GTSRB), split="train", download=True)
        GTSRB(root=str(config.DIR_GTSRB), split="test", download=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[imágenes] Falló la descarga automática ({exc}).")
        print("[imágenes] Alternativa manual: https://benchmark.ini.rub.de/gtsrb_news.html")
        print(f"           Descomprime en {config.DIR_GTSRB} (debe quedar GTSRB/Training/00000/...).")
        raise SystemExit(1)

    total = sum(1 for _ in config.DIR_GTSRB.rglob("*.ppm"))
    _registrar("gtsrb", {
        "fuente": "German Traffic Sign Recognition Benchmark (INI, Ruhr-Universität Bochum)",
        "url": "https://benchmark.ini.rub.de/gtsrb_news.html",
        "descarga": "torchvision.datasets.GTSRB",
        "licencia": "Verificar condiciones de uso en la página oficial y citar a Stallkamp et al. (2012)",
        "carpeta": str(config.DIR_GTSRB.relative_to(config.RAIZ)),
        "imagenes_ppm_encontradas": total,
        "descargado": datetime.now().isoformat(timespec="seconds"),
    })
    print(f"[imágenes] Listo: {total} imágenes .ppm en {config.DIR_GTSRB}")


def main() -> None:
    p = argparse.ArgumentParser(description="Descarga de datos de TráficoIA")
    p.add_argument("--tabular", action="store_true", help="solo dataset tabular")
    p.add_argument("--imagenes", action="store_true", help="solo imágenes GTSRB")
    a = p.parse_args()
    todo = not (a.tabular or a.imagenes)
    if a.tabular or todo:
        descargar_tabular()
    if a.imagenes or todo:
        descargar_imagenes()
    print("\nSiguiente paso: python src/primer_vistazo.py")


if __name__ == "__main__":
    main()
