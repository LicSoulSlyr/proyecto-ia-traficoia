# Datos de TráficoIA

Los archivos pesados **no** están en el repositorio. Se descargan con:

```bash
python src/descargar_datos.py
```

La descarga escribe `data/raw/manifest.json` (fuente, fecha, tamaño, huella SHA-256). Ese archivo sí se sube a GitHub.

## 1. Metro Interstate Traffic Volume (tabular)

| Campo | Detalle |
|---|---|
| Uso en el proyecto | Aprendizaje supervisado (nivel de tráfico) y no supervisado (perfiles diarios) |
| Fuente | UCI Machine Learning Repository, dataset 492 |
| Enlace | https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume |
| Licencia | CC BY 4.0 (**Confirmado por Salvador y Javier** fecha de revisión: 14/11/2026) |
| Cita | Hogue, J. (2019). *Metro Interstate Traffic Volume* [Conjunto de datos]. UCI Machine Learning Repository. |
| Archivo | `data/raw/metro_traffic/Metro_Interstate_Traffic_Volume.csv` |
| Contenido | Volumen horario de tráfico de la autopista I-94 (sentido oeste, Minneapolis–St. Paul), con clima y festivos |

Columnas: `holiday`, `temp` (Kelvin), `rain_1h`, `snow_1h`, `clouds_all`, `weather_main`, `weather_description`, `date_time`, `traffic_volume`.

Problemas conocidos (tratados en `src/datos.py` y justificados en `01_eda.ipynb`): varias filas por la misma hora, etiqueta de festivo solo en una fila del día, temperaturas de 0 K, lluvia atípica y huecos en el tiempo.

Datos limpios: `data/processed/traffic_clean.csv` (los genera `01_eda.ipynb`).

## 2. GTSRB (imágenes)

| Campo | Detalle |
|---|---|
| Uso en el proyecto | CNN de reconocimiento de señales (12 de las 43 clases) |
| Fuente | German Traffic Sign Recognition Benchmark, Institut für Neuroinformatik, Ruhr-Universität Bochum |
| Enlace | https://benchmark.ini.rub.de/gtsrb_news.html |
| Licencia / condiciones | **Confirmar en la página oficial** y anotar la fecha de revisión: ____ |
| Cita | Stallkamp, J., Schlipsing, M., Salmen, J., y Igel, C. (2012). Man vs. computer: Benchmarking machine learning algorithms for traffic sign recognition. *Neural Networks, 32*, 323-332. |
| Descarga | `torchvision.datasets.GTSRB` (usa los archivos oficiales) |
| Carpeta | `data/raw/gtsrb/` (buscar `GTSRB/Training/00000/...`) |

Imágenes `.ppm` a color de tamaño variable. Cada carpeta numerada (`00000` a `00042`) es una clase. Las imágenes vienen en secuencias de ~30 fotos de la misma señal física.

Clases propuestas: 1, 2, 3, 4, 5 (límites 30, 50, 60, 70, 80 km/h), 13 (ceda el paso), 14 (alto), 17 (entrada prohibida), 18 (precaución general), 25 (obras), 28 (niños), 38 (mantenga la derecha). Se definen en `src/config.py` (`SUBSET_CLASES`).

## 3. Cruce simulado (refuerzo)

No requiere dataset. Se programa como clase de Gymnasium en `src/` (semana 3).

## Si un enlace falla

1. Descarga manual desde los enlaces de arriba y copia los archivos en las rutas indicadas.
2. Si el dataset desapareció, proponer uno equivalente al docente y registrar el cambio aquí.
