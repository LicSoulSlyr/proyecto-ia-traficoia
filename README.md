# TráficoIA

Predicción de tráfico, señales de tránsito y control de semáforos.
Proyecto Final · Versión 3 de 8 · Python 3.11 + Anaconda.

Integra aprendizaje supervisado, no supervisado, por refuerzo y una CNN.

> **Estado:** Semana 1 (entorno, datos, EDA y propuesta).

## 1. Instalar el entorno

Instala [Anaconda Distribution](https://www.anaconda.com/download), abre *Anaconda Prompt* (Windows) o la terminal (macOS/Linux) y ejecuta desde la carpeta del repositorio:

```bash
conda env create -f environment.yml
conda activate proyecto_ia
python src/verificar_entorno.py
```

Debe terminar con `Entorno listo`. Si el entorno ya existe y cambió `environment.yml`:
`conda env update -f environment.yml --prune`.

Para registrar el kernel en Jupyter (si no aparece):
`python -m ipykernel install --user --name proyecto_ia --display-name "Python 3 (proyecto_ia)"`

> Sin tarjeta gráfica, la CNN se puede entrenar en Google Colab y descargar el modelo `.pt`. El resto del proyecto corre en CPU.

## 2. Descargar los datos

```bash
python src/descargar_datos.py          # tabular + imágenes (≈ 300 MB)
python src/descargar_datos.py --tabular
python src/descargar_datos.py --imagenes
```

Los datos **no** se suben a GitHub. Origen, licencia y estructura: [`data/README.md`](data/README.md). Cada descarga queda registrada en `data/raw/manifest.json`.

## 3. Primer vistazo y EDA

```bash
python src/primer_vistazo.py           # dimensiones, tipos, faltantes, ejemplos de imágenes
jupyter lab                            # abrir notebooks/01_eda.ipynb
```

El cuaderno `01_eda.ipynb` guarda las figuras en `reports/figures/` y los datos limpios en `data/processed/traffic_clean.csv`.

## 4. Estructura del repositorio

```
proyecto_ia/
  data/            raw/ y processed/ (no se suben a GitHub)
  notebooks/       01_eda.ipynb (semana 1); 02 a 05 en semanas siguientes
  src/             config.py, datos.py, descargar_datos.py, primer_vistazo.py, verificar_entorno.py
  models/          modelos guardados (.joblib, .pt, .npy)
  app/             app.py (Streamlit, semana 4)
  reports/         propuesta, figuras y, más adelante, informe y presentación
  docs/            guía de GitHub, reporte semanal y generador de la propuesta
  environment.yml  requirements.txt  README.md
```

## 5. Equipo

| Integrante | Rol | Tema que lidera |
|---|---|---|
| Cielo Ruiz | Coordinación general | Supervisado |
| Salvador Moreno | Datos, limpieza y EDA | No supervisado |
| Alexander Villalva | Simulación y experimentos | Refuerzo |
| Javier Dávalos | Interfaz, repositorio y video | CNN |

## 6. Datos y referencias

- Hogue, J. (2019). *Metro Interstate Traffic Volume*. UCI Machine Learning Repository.
- Stallkamp, J., Schlipsing, M., Salmen, J., y Igel, C. (2012). Man vs. computer: Benchmarking machine learning algorithms for traffic sign recognition. *Neural Networks, 32*, 323-332.
