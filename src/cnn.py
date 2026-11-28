"""Clasificación de señales GTSRB. Los índices de salida siguen SUBSET_CLASES."""
import csv

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset

from src import config
from src.datos import carpeta_entrenamiento_gtsrb, imagenes_de_clase


def preparar_imagen(imagen):
    rgb = imagen.convert("RGB").resize((32, 32), Image.Resampling.BILINEAR)
    return torch.from_numpy(np.array(rgb)).permute(2, 0, 1).contiguous()


def particiones():
    carpeta = carpeta_entrenamiento_gtsrb()
    if carpeta is None:
        raise FileNotFoundError("Ejecuta python src/descargar_datos.py --imagenes desde la raíz.")
    rng = np.random.default_rng(config.SEMILLA)
    train, val, test = [], [], []
    for indice, clase in enumerate(config.SUBSET_CLASES):
        rutas = imagenes_de_clase(carpeta, clase)
        pistas = sorted({p.stem.split("_")[0] for p in rutas})
        if len(pistas) < 2:
            raise ValueError(f"Faltan secuencias para la clase {clase}")
        reservadas = set(rng.permutation(pistas)[:max(1, round(len(pistas) * 0.2))])
        for ruta in rutas:
            destino = val if ruta.stem.split("_")[0] in reservadas else train
            destino.append((ruta, indice))
    etiquetas = next(config.DIR_GTSRB.rglob("GT-final_test.csv"), None)
    imagenes = next(config.DIR_GTSRB.rglob("Final_Test/Images"), None)
    if etiquetas is None or imagenes is None:
        raise FileNotFoundError("Faltan las imágenes de prueba de GTSRB o GT-final_test.csv.")
    with etiquetas.open(encoding="utf-8-sig", newline="") as archivo:
        for fila in csv.DictReader(archivo, delimiter=";"):
            clase = int(fila["ClassId"])
            if clase in config.SUBSET_CLASES:
                test.append((imagenes / fila["Filename"], config.SUBSET_CLASES.index(clase)))
    return train, val, test


def dataset(filas):
    imagenes = []
    for ruta, _ in filas:
        with Image.open(ruta) as imagen:
            imagenes.append(preparar_imagen(imagen))
    return TensorDataset(torch.stack(imagenes), torch.tensor([y for _, y in filas]))


def normalizar(x, aumentar=False):
    x = x.float() / 255
    if aumentar:
        # Pequeños giros y desplazamientos independientes, sin invertir las señales.
        n = len(x)
        angulo = (torch.rand(n, device=x.device) - 0.5) * np.deg2rad(16)
        matriz = torch.zeros(n, 2, 3, device=x.device)
        matriz[:, 0, 0] = matriz[:, 1, 1] = angulo.cos()
        matriz[:, 0, 1] = -angulo.sin()
        matriz[:, 1, 0] = angulo.sin()
        matriz[:, :, 2].uniform_(-0.08, 0.08)
        rejilla = F.affine_grid(matriz, x.shape, align_corners=False)
        x = F.grid_sample(x, rejilla, padding_mode="border", align_corners=False)
        brillo = torch.empty(n, 1, 1, 1, device=x.device).uniform_(0.85, 1.15)
        x = (x * brillo).clamp(0, 1)
    return (x - 0.5) / 0.5


class Clasificador(nn.Sequential):
    def __init__(self, clases):
        capas = []
        entrada = 3
        for salida in (16, 32, 48):
            capas.extend([nn.Conv2d(entrada, salida, 3, padding=1),
                          nn.ReLU(), nn.MaxPool2d(2)])
            entrada = salida
        capas.extend([nn.Flatten(), nn.Linear(48 * 4 * 4, 96), nn.ReLU(),
                      nn.Dropout(0.25), nn.Linear(96, clases)])
        super().__init__(*capas)


def epoca(modelo, lotes, dispositivo, optimizador=None):
    entrenar = optimizador is not None
    modelo.train(entrenar)
    perdida, aciertos = 0.0, 0
    with torch.set_grad_enabled(entrenar):
        for x, y in lotes:
            y = y.to(dispositivo)
            logits = modelo(normalizar(x.to(dispositivo), aumentar=entrenar))
            coste = F.cross_entropy(logits, y)
            if entrenar:
                optimizador.zero_grad()
                coste.backward()
                optimizador.step()
            perdida += coste.item() * len(y)
            aciertos += (logits.argmax(1) == y).sum().item()
    return perdida / len(lotes.dataset), aciertos / len(lotes.dataset)
