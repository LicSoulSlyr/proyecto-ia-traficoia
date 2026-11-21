# TráficoIA

Predicción de tráfico, señales de tránsito y control de semáforos.
Proyecto Final · Versión 3 de 8 · Python 3.11 + Anaconda.

Integra aprendizaje supervisado, no supervisado, por refuerzo y una CNN.

> **Estado:** Semana 1 (entorno, datos, EDA y propuesta).
> **Estado:** Semana 2 (supervisado y no supervisado). 

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

## Semana 2: supervisado y no supervisado

Requiere haber descargado los datos (`python src/descargar_datos.py --tabular`) y tener el entorno `proyecto_ia` activo. Los comandos se ejecutan desde la raíz del repositorio.

### Aprendizaje supervisado: nivel de tráfico

Predice si el tráfico de una hora será **bajo, medio o alto** a partir de hora, día de la semana, temperatura, lluvia y festivo. Compara árbol de decisión, Random Forest, KNN y regresión logística contra un modelo base.

- Cuaderno: `notebooks/02_supervisado.ipynb` (abrir con `jupyter lab`)
- Código reutilizable: `src/supervisado.py`
- Modelo guardado: `models/mejor_modelo_supervisado.joblib`
- Tabla de resultados: `reports/tabla_comparativa_supervisado.csv`

Uso del modelo guardado:

```python
from src import supervisado
paquete = supervisado.cargar_modelo()
supervisado.predecir_nivel(paquete, hora=8, dia_semana=1)   # martes 08:00
```

### Aprendizaje no supervisado: tipos de día

Agrupa los días por su perfil horario (24 valores) con K-Means y DBSCAN, visualiza con PCA y propone un horario de semáforo para cada tipo de día.

- Cuaderno: `notebooks/03_no_supervisado.ipynb`
- Código reutilizable: `src/no_supervisado.py`
- Modelos guardados: `models/kmeans_dias.joblib` y `models/dbscan_dias.joblib`
- Tablas: `reports/tabla_clusters.csv` y `reports/propuesta_semaforo.csv`

Uso del modelo guardado:

```python
from src import no_supervisado
paquete = no_supervisado.cargar_kmeans()
no_supervisado.clasificar_dia(paquete, perfil_de_24_valores)
```

### Comprobar que todo está en orden

```bash
python src/validar_semana2.py
```

Termina con `Todo en orden.` cuando existen los modelos y tablas y funcionan. La guía para sumar la semana 2 con ramas y *pull requests* está en `docs/GUIA_SEMANA2_GITHUB.md`.


## 5. Equipo

| Integrante | Rol | Tema que lidera |
|---|---|---|
| Cielo Ruiz | Coordinación general | Supervisado |
| Salvador Moreno | Datos, limpieza y EDA | No supervisado |
| Alexander Villalva Villa| Simulación y experimentos | Refuerzo |
| Javier Dávalos | Interfaz, repositorio y video | CNN |

## 6. Datos y referencias

- Hogue, J. (2019). *Metro Interstate Traffic Volume*. UCI Machine Learning Repository.
- Stallkamp, J., Schlipsing, M., Salmen, J., y Igel, C. (2012). Man vs. computer: Benchmarking machine learning algorithms for traffic sign recognition. *Neural Networks, 32*, 323-332.
