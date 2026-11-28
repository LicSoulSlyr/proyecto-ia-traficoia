# Revisión de datos para la CNN

El archivo reports/cnn_particiones.csv contiene los conteos de las 12 clases seleccionadas en src/config.py. Las imágenes provienen de GTSRB. La partición utiliza 10,290 imágenes para entrenamiento, 2,550 para validación y 6,270 del conjunto oficial de prueba. El identificador de secuencia se interpreta dentro de cada clase. Ninguna secuencia de entrenamiento aparece en validación.

Las clases tienen distintos tamaños. Por ejemplo, la prueba incluye 750 imágenes con límite de 50 km/h y solo 150 con la señal de niños. Cada error de esta última clase representa una fracción mayor de su muestra. Para analizar los errores por clase se puede dividir cada fila de la matriz de confusión por la suma de esa fila.

Para revisar los valores, ejecutar la sección de preparación del cuaderno 04_cnn.ipynb y contrastar sus conteos con el CSV. Revisar que cada clase aparezca en los tres conjuntos y que la columna de secuencias compartidas valga cero. Documentar cualquier diferencia si se modifica la selección de clases o la semilla.
