# Lista de verificación · Semana 2

## Sábado 21 de noviembre (10 %), antes de las 23:59 h · Supervisado
- [ ] `notebooks/02_supervisado.ipynb` ejecutado de principio a fin, con salidas visibles
- [ ] Se comparan al menos tres algoritmos (árbol, Random Forest y KNN o regresión logística) con validación cruzada de 5 particiones y conjunto de prueba
- [ ] Tabla comparativa con exactitud y F1 macro, y matriz de confusión
- [ ] Análisis escrito de **a qué horas y condiciones se confunden los niveles**
- [ ] Sobreajuste e importancia de variables interpretados
- [ ] `models/mejor_modelo_supervisado.joblib` guardado y subido
- [ ] Cada gráfica con título y ejes; cada resultado con interpretación escrita (sin texto de plantilla)
- [ ] PR 1 revisado y fusionado; commits de los 4 integrantes en la semana

## Lunes 23 de noviembre (10 %), antes de las 23:59 h · No supervisado
- [ ] `notebooks/03_no_supervisado.ipynb` ejecutado de principio a fin, con salidas visibles
- [ ] K-Means y DBSCAN aplicados, con variables escaladas (`StandardScaler`) y PCA en 2D
- [ ] Número de grupos **justificado** con codo y silueta; proporción de ruido de DBSCAN reportada
- [ ] Cada grupo interpretado en términos del problema (laboral, fin de semana, festivo)
- [ ] Horario de semáforo propuesto para cada tipo de día
- [ ] `models/kmeans_dias.joblib` y `models/dbscan_dias.joblib` guardados y subidos
- [ ] Resumen comparativo de una página (`docs/RESUMEN_COMPARATIVO_S2.md`, entregado en Word o PDF)
- [ ] PR 2, 3 y 4 revisados y fusionados

## Cierre
- [ ] `python src/validar_semana2.py` dice «Todo en orden» en `main`
- [ ] La prueba de reproducibilidad (clon nuevo + `conda env create`) funcionó
- [ ] Reporte semanal (`docs/REPORTE_SEMANAL_S2.md`) completo
- [ ] Etiqueta `semana-2` creada
- [ ] Fuentes y datasets citados
