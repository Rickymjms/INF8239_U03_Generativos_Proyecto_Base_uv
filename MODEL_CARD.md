# Model Card · Autoencoder y VAE

## Modelos y versiones
* **Tipo de modelos:** Autoencoder Determinista (AE) y Autoencoder Variacional (VAE).
* **Framework:** TensorFlow / Keras (Versión del entorno local: 2.21.0 / 3.15.1).
* **Versión:** 1.0 (Entrenamiento de desarrollo local - LAB10).
* **Autor:** Manuel De Jesus Mañana Santana (INF-8239, Unidad 03).

## Uso previsto y usuarios
* **Propósito principal:** Demostración académica y exploración de conceptos de modelos generativos, compresión de dimensionalidad, y comparación directa entre regularización probabilística (VAE) y reconstrucción determinista (AE).
* **Usuarios previstos:** Estudiantes, docentes e investigadores en Ciencia de Datos evaluando la topología de espacios latentes y las compensaciones computacionales (costo vs. capacidad generativa).

## Usos fuera de alcance
* **Producción comercial:** Este modelo es una prueba de concepto académica y no está diseñado para sistemas en producción, ni para compresión de imágenes de alta resolución.
* **Clasificación:** No es un modelo discriminativo; no etiqueta ni clasifica prendas de vestir.
* **Anonimización garantizada:** No debe asumirse que la generación de imágenes sintéticas mediante este VAE protege de forma absoluta la privacidad de los datos originales sin un estudio de memorización previo.

## Dataset, licencia y particiones
* **Dataset:** Fashion-MNIST (creado por Zalando).
* **Descripción:** Imágenes en escala de grises de $28 \times 28$ píxeles representando 10 categorías de artículos de moda y calzado.
* **Particiones:** $60,000$ imágenes para entrenamiento y $10,000$ reservadas para evaluación y pruebas visuales. Preprocesado mediante normalización al rango $[0, 1]$.
* **Licencia:** MIT License.

## Arquitectura y espacio latente
* **Autoencoder (AE):** Arquitectura determinista. Comprime la imagen a un **cuello de botella de 16 dimensiones** que prioriza la retención de características de reconstrucción.
* **Autoencoder Variacional (VAE):** Arquitectura probabilística. El codificador parametriza la media y la varianza para forzar un **espacio latente bidimensional (2D)** continuo. El decodificador reconstruye utilizando una capa densa final de 784 neuronas con activación sigmoide, procesando los vectores muestreados mediante el *reparameterization trick*.

## Métricas de reconstrucción y generación
* **Métricas del AE (6 Épocas):** Pérdida de validación final de `0.3071`. Retiene de manera efectiva las formas y siluetas (e.g., zapatillas y pantalones) pero no posee capacidad para generación desde muestreo aleatorio.
* **Métricas del VAE (8 Épocas):** Pérdida de reconstrucción de `267.32` y divergencia KL de `6.25`. La penalización KL fuerza la continuidad del espacio, pero produce salidas visualmente más borrosas y difuminadas en comparación con el AE.

## Diversidad y vecinos cercanos
* **Interpolación (VAE):** El espacio latente 2D demuestra transiciones matemáticamente continuas (e.g., de una bota a un pantalón). Sin embargo, regiones intermedias pueden generar formas visualmente ininteligibles o ambiguas.
* **Diversidad de Muestras:** El muestreo aleatorio produce familias de siluetas distintas (prendas superiores, calzado, pantalones).
* **Vecindad y Colapso:** Las categorías visualmente afines (como botas y sandalias) quedan mapeadas en regiones sumamente cercanas debido a la restricción bidimensional, provocando colapso de características (e.g., la sandalia se decodifica como un botín). Se pierden los detalles de alta frecuencia (como estampados o logotipos).

## Hardware, tiempo y tamaño
* **Hardware de entrenamiento:** Equipo local Windows ejecutando un procesador **Intel(R) Core(TM) i7-10750H CPU @ 2.60GHz** con **8.00 GB de RAM**. 
* **Aceleración:** Aunque el sistema cuenta con una GPU NVIDIA GeForce GTX 1660 Ti (6 GB), el entrenamiento se ejecutó de manera aislada en CPU aprovechando las optimizaciones matemáticas `oneDNN` nativas, respetando las restricciones de compatibilidad de TensorFlow >= 2.11 en entornos Windows.
* **Eficiencia (Criterios Green AI):** 
  * **AE:** Extremadamente ligero, con $25,888$ parámetros y un tiempo de convergencia de $\approx 6.86$ segundos.
  * **VAE:** Mayor huella computacional, con $101,520$ parámetros (solo en el decodificador) y un tiempo de convergencia de $\approx 10.58$ segundos, reflejando el costo de parametrizar distribuciones latentes.

## Limitaciones, riesgos y supervisión
* **Limitación de comparabilidad (AE vs VAE):** Siguiendo fielmente la arquitectura del proyecto base, **se evaluó un AE con un espacio latente de 16 dimensiones frente a un VAE de solo 2 dimensiones**. Se optó por dejar los modelos como están y declarar esta diferencia dimensional como una limitación metodológica estricta. La pérdida de nitidez y el "impuesto de borrosidad" del VAE no solo se deben a la regularización probabilística, sino que están fuertemente condicionados por este severo cuello de botella de 2D.
* **Riesgos:** La principal limitación es el riesgo de mala interpretación. Debe evitarse afirmar que el VAE mejora la nitidez o garantiza diversidad perfecta; empíricamente, favorece representaciones promedio para evadir penalizaciones probabilísticas.
* **Supervisión humana:** Las interpolaciones y muestras sintéticas requieren escrutinio humano para confirmar su plausibilidad visual frente al ruido semántico.