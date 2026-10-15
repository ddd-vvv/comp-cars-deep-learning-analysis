# CompCars Deep Learning Analysis

[English](README.md)

**Autor:** Diego Andrés Vega Acuña  
**Profesor guía:** Carlos Navarro Clavería  
**Enlaces:** [Memoria (PDF)](https://drive.google.com/file/d/1vWpu36wMhaFOe7DIaxsMcTYh8jedxFZU/view) | [Resultados completos (Google Drive)](https://drive.google.com/drive/folders/1V83e9XZ7O_Hcy8RAhOQ6QsEIfco-g_Xv)

Repositorio de investigación de la memoria para optar al título de Ingeniero Civil Eléctrico (Universidad de Chile, 2026), enfocado en la identificación no supervisada de modelos de vehículos mediante representaciones profundas (*embeddings*) y técnicas de agrupamiento (*clustering*).

**Título de la memoria:**  
*Detección de modelos y tipos de vehículos fraudulentos en autopistas mediante aprendizaje profundo no supervisado.*

---

## Contexto de investigación

Este proyecto estudia si la estructura semántica de los *embeddings* profundos, extraídos de modelos de visión pura y multimodales ajustados finamente (*fine-tuned*), es lo suficientemente robusta y discriminativa para identificar modelos de vehículos (vistos o no vistos previamente) mediante técnicas de agrupamiento, sin utilizar métodos de clasificación tradicionales.

**Planteamiento del problema:**
- En escenarios reales, como la vigilancia de autopistas, es común que aparezcan modelos de vehículos no vistos previamente.
- La disponibilidad de datos etiquetados es limitada y presenta una distribución de cola larga (*long-tail*).
- Los métodos de clasificación tradicionales dependen de grandes conjuntos de datos etiquetados y de que las clases a identificar estén presentes en el conjunto de entrenamiento (*closed-set*).

**Hipótesis de trabajo:**  
La combinación de representaciones profundas multimodales (visión y texto, basadas en *Transformers*) optimizadas mediante *Metric Learning* y técnicas de *clustering* no supervisado (con auto-exploración de hiperparámetros) resulta en *embeddings* con mayor calidad discriminativa y robustez de generalización (*Zero-Shot* / *Few-Shot*) que las representaciones generadas por modelos unimodales (visión pura), lo que permite alcanzar una mayor pureza de agrupamiento de modelos de vehículos (vistos y no vistos).

**Objetivo general:**  
Desarrollar, implementar y evaluar un sistema basado en representaciones profundas y técnicas de agrupamiento no supervisado para la identificación efectiva de modelos de vehículos en escenarios con datos limitados y modelos no vistos previamente.

**Objetivos específicos:**
- Investigar, seleccionar y comparar diversas arquitecturas de redes neuronales (CNN, *Vision Transformers* y modelos de visión-lenguaje) para la generación de representaciones profundas de vehículos.
- Implementar un proceso de ajuste fino supervisado con datos especializados para optimizar la calidad de los *embeddings*.
- Comparar diversas técnicas de agrupamiento no supervisado para agrupar e identificar modelos de vehículos a partir de dichas representaciones.
- Proponer e implementar un sistema de clasificación no supervisada de modelos de vehículos.
- Analizar la efectividad del sistema en términos de pureza de los *clusters* y de su capacidad de generalización ante modelos no vistos previamente.

---

## Infraestructura Cloud (GCP) y Stack Tecnológico

Los experimentos computacionalmente costosos se ejecutaron en **Google Cloud Platform (GCP)**, en dos instancias de máquinas virtuales: una con GPU para el *fine-tuning* de los modelos profundos y otra con una CPU de alto rendimiento para el análisis de los *embeddings* (reducción de dimensionalidad y *clustering*).

| Instancia en GCP | CPU | Memoria RAM | GPU | Almacenamiento | Rol en el pipeline |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **GPU-VM** | `n1-standard-4` (4 vCPUs) | 15 GB | NVIDIA Tesla T4 (16 GB VRAM) | 100 GB SSD | *Fine-tuning* supervisado y multimodal (AMP, muestreo P × K). |
| **CPU-VM** | `c4-standard-16` (16 vCPUs) | 60 GB | N/A | 100 GB SSD | Reducción de dimensionalidad (PCA incremental, t-SNE, UMAP) y optimización de *clustering* con Optuna. |

**Herramientas principales:**
- **Deep Learning y ML:** PyTorch (con *Automatic Mixed Precision*, `AMP` y `GradScaler`), `pytorch-metric-learning`, Scikit-learn, UMAP-learn.
- **Optimización de hiperparámetros:** Optuna (muestreador bayesiano *Tree-structured Parzen Estimator*, `TPE`, y *Median Pruning*).
- **Gestión de entorno y datos:** Python 3.10, `uv` (gestor de paquetes escrito en Rust), Pandas, JSON y CSV.
- **Control de versiones:** Git y GitHub.

---

## Resumen metodológico

El flujo de trabajo, modular y orientado a objetos, se organiza en cuatro etapas secuenciales:
1. **Etapa 0:** Estrategia de datos, manejo de la distribución *long-tail* y preprocesamiento con aumento de datos estocástico.
2. **Etapa 1:** *Fine-tuning* supervisado (Clasificación vs. *Metric Learning* en visión pura; ajuste por capas y fases en CLIP).
3. **Etapa 2:** Optimización de la reducción de dimensionalidad del espacio latente.
4. **Etapa 3:** Agrupamiento no supervisado (*clustering*) sin conocer el número de clases a priori, y evaluación.

### Estrategia de datos (Dataset CompCars)

El módulo `DataFrameMaker` transforma el dataset *Comprehensive Cars* (CompCars) en un archivo CSV maestro que filtra exclusivamente las vistas **frontal (`Front`)** y **trasera (`Rear`)**:
- **Definición de clases:** combinación de `marca + modelo + año de lanzamiento` (ej. `Audi_A4_2010` vs. `Audi_A4_2015`).
- **Manejo del desbalance *long-tail*:** un umbral `min_images` divide los datos en **Clases Regulares** (≥ `min_images`, usadas para entrenamiento, validación y prueba) y **Clases *One-Shot*** (< `min_images`, excluidas del entrenamiento y reservadas para evaluar la generalización ante modelos no vistos).
- **Intersección multi-vista (`Both`):** el universo de clases con ambas vistas suma 3,598 clases y 31,073 imágenes; en esta configuración, el umbral `min_images` debe cumplirse simultáneamente en ambas perspectivas. Con `min_images = 8` se obtienen 359 clases regulares y 3,239 clases *One-Shot*.
- **Aumento de datos:** rotaciones, traslaciones, *Random Horizontal Flip*, *Color Jitter*, *Gaussian Blur*, conversión a escala de grises (replicada en 3 canales para mantener compatibilidad con los pesos preentrenados) y *Random Erasing*.

### Aprendizaje de representaciones

Se evalúan tres familias de arquitecturas partiendo de pesos preentrenados:
- **CNN:** ResNet50 y ResNet101 (D = 2048).
- **Vision Transformers:** ViT-B/32 y ViT-B/16 (D = 768).
- **Multimodal (visión-lenguaje):** CLIP ViT-B/32 (D = 512).

En los modelos de visión pura se contrasta la pérdida **Cross-Entropy** frente a cinco funciones de **Metric Learning** (con muestreo P × K mediante un *wrapper* de `MPerClassSampler`): *Contrastive Loss*, *Triplet Loss*, *NT-Xent*, *ArcFace* y *Multi-Similarity Loss*.

Para **CLIP** se diseñó una estrategia de entrenamiento **asimétrica y secuencial por fases**:
- Durante el **entrenamiento** se usan descripciones textuales completas (*"The front view image of a Skoda Haorui vehicle from 2012 type sedan"*).
- Durante la **validación y prueba** se ocultan el modelo y el año (*"The front view image of a Skoda vehicle type sedan"*), simulando el escenario real en que el sistema desconoce el modelo exacto.
- Se aplica descongelamiento progresivo por capas y ajuste en dos fases (**Fase 1: Refinamiento Visual → Fase 2: Adaptación Textual**).

El *pipeline* de entrenamiento incluye *Weight Decay* diferenciado (excluyendo sesgos y capas de normalización), precisión mixta (AMP), *Gradient Clipping*, *WarmUp* y *Early Stopping*.

### Reducción de dimensionalidad y clustering

Las proyecciones con **PCA Incremental**, **t-SNE** y **UMAP** se optimizan con **Optuna** (50 *trials* por algoritmo). Para t-SNE y UMAP se maximiza una función compuesta por *Silhouette Score* normalizado (70%) y *Trustworthiness* (30%); para PCA se maximiza solo el *Silhouette Score*.

El agrupamiento descarta algoritmos que exigen fijar el número de clases k a priori (como K-Means) y evalúa, con **250 *trials* de Optuna por algoritmo**:
- **Basados en densidad:** DBSCAN, HDBSCAN y OPTICS.
- **Jerárquico:** Agglomerative Clustering.

La función objetivo del *clustering* combina métricas internas y externas: 0.35 · *Silhouette* + 0.25 · *Calinski-Harabasz* (ambas normalizadas) + 0.25 · ARI + 0.15 · NMI. Es decir, durante la optimización se usan las etiquetas disponibles (40% del puntaje) para guiar la búsqueda, y se descartan configuraciones degeneradas (por ejemplo, ruido > 40% o menos de 2 *clusters*).

---

## Estudios experimentales

| Estudio | Pregunta de investigación | Variables principales |
| :--- | :--- | :--- |
| **Experimento 1: Línea base unimodal** | ¿Qué arquitectura de visión pura (CNN vs. Transformer) y qué paradigma de aprendizaje generan el mejor espacio latente? | Arquitectura (ResNet50 vs. ViT-B/32), objetivo (Clasificación vs. 5 pérdidas de *Metric Learning*), vistas (`Front` vs. `Front + Rear`), con `min_images = 8`. |
| **Experimento 2: Adaptación de CLIP** | ¿Cuánta plasticidad requiere cada codificador y cuál es el orden de fases más eficaz para adaptar CLIP al dominio vehicular? | Descongelamiento progresivo (1 a 12 capas en `Image Encoder` y `Text Encoder`), orden de fases (*Visión → Texto* vs. *Texto → Visión*). |
| **Experimento 3: Sensibilidad al volumen de datos** | ¿Qué tan robusto es CLIP ante la escasez de ejemplos por clase? | Umbral `min_images` ∈ {4, 6, 8, 10, 12}. |
| **Experimento 4: Escalabilidad y generalización** | ¿Mejora el agrupamiento al escalar la arquitectura y cómo responde el sistema ante clases nunca vistas (*Open-Set*)? | Escalado (ResNet101, ViT-B/16); inyección progresiva (0%, 25%, 50%, 75%, 100%) de las 3,239 clases *One-Shot*. |

---

## Resultados principales

1. **Metric Learning y fusión multi-vista (Exp. 1):** en promedio, *Metric Learning* superó a la clasificación con *Cross-Entropy* en ARI (0.22 vs. 0.17) y en porcentaje de clústeres puros (48.1% vs. 42.0%), generando del orden de 125–130 clústeres puros frente a 10–15 en clasificación. **Multi-Similarity** y **NT-Xent** lideraron en Recall@k y ARI; *ArcFace* obtuvo el mayor porcentaje de clústeres puros pero un ARI muy bajo, por sobre-fragmentación del espacio latente; *Triplet Loss* tuvo el rendimiento más pobre. Combinar las vistas frontal y trasera (`Front + Rear`) elevó el ARI promedio de 0.07 a 0.38. **ResNet50** superó a **ViT-B/32** (ARI promedio 0.2605 vs. 0.1718) con menor tiempo de entrenamiento promedio (88.8 vs. 97.0 min). La mejor línea base unimodal fue ResNet50 + Multi-Similarity con vistas `Front + Rear`.
2. **Adaptación de CLIP (Exp. 2):** el ajuste del codificador visual fue el principal motor del rendimiento, mientras que el codificador textual cumplió un rol complementario en la estructuración global del espacio latente. Según la métrica de éxito definida, se fijó el descongelamiento de **12 capas del `Image Encoder` y 9 capas del `Text Encoder`**. La secuencia **Visión → Texto** superó al orden inverso en Recall, ARI, pureza y número de clases solapadas, con tiempos de cómputo similares.
3. **Sensibilidad al volumen de datos (Exp. 3):** incluso con `min_images = 4`, el ajuste fino de CLIP superó claramente al modelo *baseline* sin ajustar. El rendimiento creció con el volumen de datos, alcanzando un **ARI de 0.86** con 12 imágenes por clase. Se identificó `min_images = 8` como punto de equilibrio entre el esfuerzo de recolección de datos y la calidad del agrupamiento; para el Exp. 4 se usaron 10 imágenes por clase para maximizar el rendimiento.
4. **Escalabilidad y generalización (Exp. 4):**
   - **Escalado de arquitectura:** pasar de ResNet50 a ResNet101 no mejoró el rendimiento, lo que sugiere saturación de capacidad en la CNN. En cambio, pasar de **ViT-B/32 a ViT-B/16** (parches de 16 × 16 píxeles) elevó el ARI de ~0.66 a ~0.86 y redujo los clústeres mixtos, presumiblemente por capturar detalles finos como faros o logotipos. ViT-B/16 lideró en ARI entre las arquitecturas evaluadas; CLIP-ViT-B/32 se mantuvo sin escalar.
   - **Generalización *Open-Set*:** se evaluaron los tres mejores modelos (CLIP-ViT-B/32, ViT-B/16 con *Contrastive Loss* y ResNet50 con *Multi-Similarity*). Sobre las 359 clases regulares (`Ratio 0.0`), **CLIP-ViT-B/32** alcanzó un **89.0% de pureza de clústeres**, frente a 79.4% de ViT-B/16 y 69.9% de ResNet50. Al inyectar el 100% de las 3,239 clases *One-Shot* (3,598 clases en total, `Ratio 1.0`), **CLIP** mantuvo la mayor pureza (**38.8%**, con **770 clústeres puros**), seguido por ViT-B/16 (35.8%, 681 clústeres puros) y ResNet50 (23.7%, 401 clústeres puros).

### Relevancia e impacto práctico

Los resultados indican que es posible identificar modelos de vehículos de forma no supervisada a partir de representaciones profundas, incluso ante modelos no incluidos en el ajuste fino. Esto tiene implicancias directas para la detección de anomalías y fraude en autopistas, al permitir detectar modelos desconocidos o poco representados sin necesidad de reentrenar la red ante cada nuevo modelo.

---

## Estructura del repositorio

- [`src/`](src): Módulos centrales orientados a objetos (`config`, `data`, `models`, `pipeline`, `utils`).
- [`scripts/`](scripts): Scripts ejecutables para los *pipelines* de *fine-tuning* y *embeddings*, extracción (`JSON → CSV`), análisis estadístico y visualización.
- [`configs/`](configs): Archivos de configuración experimental en YAML (`resnet50_*.yaml`, `vitb32_*.yaml`, `CLIP.yaml`).
- [`tests/`](tests): Pruebas unitarias y de integración.
- [`pyproject.toml`](pyproject.toml) y [`uv.lock`](uv.lock): Definición del proyecto y bloqueo de dependencias con `uv`.

Las carpetas `results/`, `dataset/image/` y `.venv/` no se versionan; los resultados completos están disponibles en [Google Drive](https://drive.google.com/drive/folders/1V83e9XZ7O_Hcy8RAhOQ6QsEIfco-g_Xv).

---

## Configuración reproducible

**Requisitos:**
- Python 3.10
- [`uv`](https://github.com/astral-sh/uv)

**Instalación del entorno:**

```bash
uv sync
```

**Flujo de ejecución principal:**

```bash
# 1) Ajuste fino supervisado / multimodal (GPU-VM)
uv run scripts/pipeline/Finetuning.py

# 2) Embeddings, reducción de dimensionalidad y clustering con Optuna (CPU-VM)
uv run scripts/pipeline/Embeddings.py

# 3) Extracción tabular de resultados (JSON a CSV)
uv run scripts/extraction/extract_vision_results.py
uv run scripts/extraction/extract_clip_layers_results.py

# 4) Análisis estadístico y generación de gráficos
uv run scripts/analysis/analyze_vision_models.py
uv run scripts/analysis/analyze_clip_layers.py
uv run scripts/visualization/visualize_vision_models.py
uv run scripts/visualization/visualize_clip_layers.py
```

**Ejecución de pruebas:**

```bash
uv run pytest tests/
```

---

## Limitaciones y trabajo futuro

**Limitaciones:**
- Los experimentos se basan en CompCars, cuyas imágenes provienen de Internet y pueden diferir de las capturadas por cámaras fijas en pórticos de autopistas.
- La optimización de hiperparámetros del *clustering* utiliza las etiquetas reales (ARI y NMI) como parte de su función objetivo, y las etapas de reducción y *clustering* requieren búsquedas con Optuna para cada conjunto de *embeddings*.

**Líneas de trabajo futuro:**
- **Aprendizaje auto-supervisado (*Self-Supervised Learning*):** aprovechar grandes volúmenes de imágenes de vehículos sin etiquetar, disponibles en la web o en cámaras de seguridad.
- **Sistemas de recuperación (*Content-Based Image Retrieval*):** evaluar los *embeddings* en búsqueda de vehículos similares, con aplicaciones en forense digital.
- **Re-identificación a nivel de instancia:** distinguir vehículos específicos de un mismo modelo por detalles como adhesivos, daños o accesorios, o bien forzar la invarianza a ellos.
- **Ingeniería de etiquetas textuales (*Prompt Engineering*):** estudiar el impacto de descripciones más ricas (año, tipo de carrocería, país de origen) en el ajuste de modelos multimodales.
- **Nuevos algoritmos de reducción y *clustering*:** evaluar métodos que preserven mejor la topología local y global sin requerir búsquedas de hiperparámetros tan exhaustivas.
- **Arquitecturas híbridas CNN-Transformer:** combinar la extracción de patrones locales de las convoluciones con la comprensión global de los Transformers.

---

## Citación

Si utilizas este código o los hallazgos en tu investigación, por favor cita la memoria de título y el dataset CompCars:

```text
Diego Andrés Vega Acuña.
Detección de modelos y tipos de vehículos fraudulentos en autopistas mediante aprendizaje profundo no supervisado.
Memoria para optar al título de Ingeniero Civil Eléctrico, Departamento de Ingeniería Eléctrica, Facultad de Ciencias Físicas y Matemáticas, Universidad de Chile, 2026.
```

```text
Linjie Yang, Ping Luo, Chen Change Loy, Xiaoou Tang.
A Large-Scale Car Dataset for Fine-Grained Categorization and Verification.
Proceedings of IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2015.
```

---

## Declaración de uso del dataset CompCars

Este repositorio utiliza el dataset CompCars bajo sus restricciones publicadas para investigación no comercial:

```text
The CompCars database is available for non-commercial research purposes only. All images of the CompCars database are obtained from the Internet which are not property of MMLAB, The Chinese University of Hong Kong. The MMLAB is not responsible for the content nor the meaning of these images. You agree not to reproduce, duplicate, copy, sell, trade, resell or exploit for any commercial purposes, any portion of the images and any portion of derived data. You agree not to further copy, publish or distribute any portion of the CompCars database. Except, for internal use at a single site within the same organization it is allowed to make copies of the database.
```

---

## Licencia

Este repositorio está licenciado bajo Creative Commons Attribution-NonCommercial-NoDerivs 3.0 Chile. Ver [LICENSE](LICENSE).
