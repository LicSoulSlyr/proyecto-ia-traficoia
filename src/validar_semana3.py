"""Ejecutar desde la raíz: python -m src.validar_semana3 [cnn|refuerzo|todo]."""
import argparse
from src import config


def verificar_cnn():
    import pandas as pd
    import torch
    from src.cnn import Clasificador
    paquete = torch.load(config.MODELOS / "cnn_senales.pt", map_location="cpu", weights_only=True)
    assert paquete["clases"] == config.SUBSET_CLASES
    modelo = Clasificador(len(paquete["clases"]))
    modelo.load_state_dict(paquete["pesos"])
    modelo.eval()
    with torch.inference_mode():
        logits = modelo(torch.zeros(1, 3, 32, 32))
    assert logits.shape == (1, 12) and torch.isfinite(logits).all()
    datos = pd.read_csv(config.RAIZ / "reports/cnn_particiones.csv")
    assert datos["ID GTSRB"].tolist() == config.SUBSET_CLASES
    assert datos["Secuencias compartidas"].eq(0).all()
    assert datos["Prueba"].sum() == 6270
    for nombre in ["curvas", "confusion", "errores"]:
        assert (config.FIGURAS / f"cnn_{nombre}.png").is_file()
    print("CNN verificada: pesos, 12 salidas, particiones y tres figuras.")


def verificar_refuerzo():
    import numpy as np
    import pandas as pd
    from gymnasium.utils.env_checker import check_env
    from src.cruce import Cruce
    from src.refuerzo import evaluar
    check_env(Cruce(), skip_render_check=True)
    q = np.load(config.MODELOS / "q_semaforo.npy")
    assert q.shape == (5, 5, 2, 2) and np.isfinite(q).all()
    guardado = pd.read_csv(config.RAIZ / "reports/rl_comparacion.csv")
    for i, politica in enumerate([None, q]):
        observado = evaluar(politica, range(5000, 5100))
        np.testing.assert_allclose(observado, guardado.iloc[i, 1:].to_numpy(dtype=float))
    assert (config.FIGURAS / "rl_aprendizaje.png").is_file()
    print("Refuerzo verificado: entorno, tabla Q y métricas de ambas políticas.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("modulo", choices=["cnn", "refuerzo", "todo"], nargs="?", default="todo")
    modulo = parser.parse_args().modulo
    if modulo in ("cnn", "todo"):
        verificar_cnn()
    if modulo in ("refuerzo", "todo"):
        verificar_refuerzo()
