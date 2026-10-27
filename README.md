# Reporte de Laboratorio: LAB10 (Autoencoder y VAE con Fashion-MNIST)

**Asignatura:** INF-8239 Ciencia de Datos II (Unidad 03)  
**Estudiante:** Manuel De Jesus Mañana Santana  
**Universidad:** Universidad Autónoma de Santo Domingo (UASD) 
**Gestor de entorno:** `uv` (Python 3.12) 

---

## 1. Fundamento: Reconstruir no es lo mismo que generar
En este laboratorio se evalúan dos arquitecturas generativas sobre el dataset **Fashion-MNIST**:
* **Autoencoder Determinista (AE):** Comprime las imágenes en un cuello de botella de menor dimensión y aprende a reconstruirlas. Minimizar el error de reconstrucción no garantiza que el modelo pueda generar datos nuevos y plausibles desde puntos aleatorios.
* **Autoencoder Variacional (VAE):** Aprende parámetros de distribuciones latentes y añade la divergencia KL como regularización para forzar un espacio muestreable y continuo, permitiendo la generación de nuevas imágenes.

---

## 2. Preparación del Entorno (CPU)
Dado el costo computacional de TensorFlow, se ejecutó el entorno aislado configurado para CPU.
* **Comando de verificación:** `uv run python scripts/check_runtime.py`
* **Validación:** El sistema confirmó el uso de CPU, mostrando las advertencias esperadas de optimización (`oneDNN custom operations`) y confirmando la ausencia de aceleración nativa por GPU, lo cual es el comportamiento correcto y esperado en Windows.

---

## 3. Entrenamiento de los Modelos (Fashion-MNIST)
Se procedió a entrenar ambos modelos especificando las épocas para comparar su costo computacional y métricas de pérdida.
* **Comando:** `uv run python scripts/train_ae_vae.py --epochs-ae 6 --epochs-vae 8`

**Resultados y Métricas de Rendimiento (Green AI):**
* **Autoencoder (AE - 6 épocas):** Tiempo de entrenamiento de $\approx 6.86$ segundos con $25,888$ parámetros. Alcanzó un *Loss* final de validación de `0.3071`.
* **Autoencoder Variacional (VAE - 8 épocas):** Tiempo de entrenamiento de $\approx 10.58$ segundos con un decodificador sustancialmente más pesado ($101,520$ parámetros). Finalizó con una divergencia KL de `6.2564` y una pérdida de reconstrucción de `267.3215`.

---

## 4, 5 y 6. Interpretación Técnica de Pérdidas
* **Pérdida del AE:** El error bajo demuestra que el cuello de botella retiene información suficiente para la reconstrucción, pero esto no lo convierte en un modelo generativo, ya que su espacio latente no está probabilísticamente estructurado.
* **Pérdida del VAE:** El costo de obligar al espacio latente a organizarse (Divergencia KL) compite contra la fidelidad de la imagen (Pérdida de Reconstrucción). Esta tensión explica por qué los VAE suelen producir imágenes ligeramente más difusas.

---

## 7. Evaluación Visual con Paneles (Comparación AE vs. VAE)

A partir de los paneles gráficos generados (`ae_vae_panel.png` e `interpolation.png`), se realizó una lectura empírica prudente sobre la nitidez, diversidad y continuidad espacial.

### Panel de Reconstrucción y Generación (`ae_vae_panel.png`)
1. **Autoencoder (Fila 2 - AE):** Conserva la forma global y la clase en las 12 columnas; los pantalones y las zapatillas quedan casi nítidos. Sin embargo, pierde textura y detalles finos: el texto “Lee” del suéter y la cuadrícula de la camisa (columna 8) desaparecen por completo. **Conclusión:** El cuello de botella de 16 dimensiones guarda la forma global, pero descarta el detalle fino.
2. **Autoencoder Variacional (Fila 3 - VAE):** El resultado es claramente más borroso que en el AE. Todas las prendas de manga larga colapsan hacia una misma plantilla gris, volviendo irreconocible la camisa a rayas (columna 7). 
   * *Explicación analítica:* Esta salida borrosa evidencia que la función de pérdida favorece promedios, o bien que el espacio latente de solo 2 dimensiones es limitante.
   * *Confusión de clases:* La sandalia (columna 12) se reconstruye con forma de bota o zapatilla en ambos modelos; en el VAE, se parece a la bota de la columna 1. Esto demuestra que, al contar con solo 2 dimensiones, las categorías parecidas se ven forzadas a ocupar regiones espacialmente cercanas.
3. **Muestras Nuevas (Fila 4 - VAE):** Las imágenes generadas desde el ruido son borrosas. A simple vista aparecen siluetas de calzado (cols. 1, 5, 8, 9, 11), prendas superiores (2, 6, 7) y pantalones (10, 12). Mientras algunas son formas coherentes (columna 3 muestra una bota bien formada), otras son manchas mixtas (columna 4). Hay variedad entre familias de prendas, pero repetición dentro de una misma silueta.
4. **Advertencia Teórica:** Los resultados empíricos de este experimento contradicen categóricamente la afirmación de que el VAE "maximiza la nitidez". Queda demostrado que el AE determinista reconstruye con muchísimo mayor detalle visual.

### Evaluación del Espacio Latente (`interpolation.png`)
* **Continuidad:** La secuencia de interpolación transita desde una bota (izquierda) hacia un pantalón (derecha), pasando por una forma intermedia de prenda superior en el centro.
* **Transición y Plausibilidad:** La transición matemática es gradual, sin saltos bruscos. Sin embargo, los puntos intermedios (columnas 4 y 5) se manifiestan como manchas ambiguas con varios lóbulos que no corresponden a ninguna prenda real. 
* **Conclusión:** La interpolación es matemáticamente suave, pero solo *parcialmente plausible* visualmente. Este es el patrón típico y esperado de un espacio latente de 2 dimensiones: es continuo, pero las vastas regiones de transición entre categorías carecen de significado visual claro.

## 8. Exploración Interactiva (Aplicación Local Streamlit)

Se ejecutó la interfaz gráfica local mediante Streamlit (`app/streamlit_app.py`) para interactuar con el decodificador del VAE cargado en memoria (`models/decoder.keras`). 

### Observaciones del Espacio Latente (Interactividad)
A través de los deslizadores paramétricos bidimensionales (Latente X y Latente Y en rango de -3.0 a 3.0), se comprobó la organización topológica de las prendas:
* **Agrupación Semántica:** El modelo posicionó el calzado (botas y zapatillas) en los valores negativos del eje Y (`-3.0`), mientras que los pantalones se ubicaron hacia el extremo positivo del eje X (`1.5` a `3.0` con Y neutral). Las prendas superiores (suéteres y camisetas) dominaron el eje Y positivo (`1.5` a `1.8`).
* **Interpolación en vivo:** Al arrastrar los deslizadores, la transición entre estas familias de prendas resultó fluida, evidenciando un espacio continuo y sin cortes abruptos, aunque con pérdida de resolución en las fronteras entre clases.
* **Aviso de Responsabilidad:** La aplicación cumple con las directrices de explicabilidad mostrando la advertencia de uso responsable: *"Una muestra sintética no garantiza originalidad ni privacidad"*, previniendo la falsa creencia de que un dato sintético es automáticamente un dato anónimo o seguro.