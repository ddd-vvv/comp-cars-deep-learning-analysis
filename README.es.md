# CompCars Deep Learning Analysis

[English](README.md)

**Autor:** Diego Vega  
**Enlaces:** [Memoria (PDF)](https://drive.google.com/file/d/1vWpu36wMhaFOe7DIaxsMcTYh8jedxFZU/view) | [Resultados completos (Google Drive)](https://drive.google.com/file/d/1vWpu36wMhaFOe7DIaxsMcTYh8jedxFZU/view)

Repositorio de investigación de la memoria de título de Ingeniería Eléctrica, enfocado en identificación no supervisada de modelos de vehículos mediante representaciones profundas y clustering.

Título de la memoria:
Detección de modelos y tipos de vehículos fraudulentos en autopistas mediante aprendizaje profundo no supervisado.

## Contexto de investigación

Este proyecto estudia si los embeddings profundos, extraídos de modelos de visión y multimodales ajustados finamente, pueden separar modelos de vehículos en el espacio latente sin clasificación cerrada tradicional en inferencia.

Planteamiento del problema:
- En escenarios reales de autopistas pueden aparecer modelos no vistos.
- La disponibilidad de etiquetas es limitada y existe fuerte desbalance de clases long-tail.
- Los enfoques supervisados clásicos son frágiles en condiciones open-set.

Hipótesis de trabajo:
- Las representaciones profundas multimodales (visión y texto), optimizadas con metric learning y preprocesamiento orientado a clustering, entregan una estructura discriminativa más robusta que los baselines unimodales de visión pura para descubrir modelos de vehículos en escenarios open-set.

Objetivos de investigación:
- Evaluar si los embeddings profundos permiten identificar modelos de vehículos sin clasificación cerrada durante la inferencia.
- Comparar backbones CNN, Transformer y visión-lenguaje bajo un protocolo experimental unificado.
- Medir el efecto de metric learning, la fusión multi-vista, la reducción de dimensionalidad y el clustering no supervisado sobre la separabilidad del espacio latente.
- Evaluar generalización frente a clases con pocos datos y modelos no vistos previamente.

## Resumen metodológico

El flujo de trabajo se organiza en cuatro etapas:
1. Curación y preprocesamiento de datos
2. Fine-tuning supervisado para mejorar la calidad de embeddings
3. Optimización de reducción de dimensionalidad
4. Clustering no supervisado y análisis estadístico

### Estrategia de datos

El repositorio operacionaliza CompCars en un dataset tabular preparado para entrenamiento, centrado en vistas frontales y traseras, que son las perspectivas más relevantes para monitoreo vehicular en autopistas. Una etapa dedicada de curación filtra viewpoints no relevantes, consolida metadatos y genera un CSV maestro que actúa como fuente de verdad para los pipelines.

El diseño experimental aborda explícitamente el desbalance long-tail:
- Las clases se definen a nivel modelo-año.
- Un umbral mínimo de imágenes separa clases regulares de clases one-shot.
- Las clases regulares se usan para el fine-tuning supervisado.
- Las clases con pocas muestras se reservan para evaluar generalización open-set y few-shot.
- En la configuración front+rear, el umbral debe cumplirse simultáneamente en ambas vistas para garantizar pares válidos.

### Etapa de aprendizaje de representaciones

Se estudian tres familias de modelos bajo un marco común de transferencia de aprendizaje:
- ResNet50 como baseline CNN.
- ViT-B/32 como baseline de visión basado en Transformers.
- CLIP como baseline multimodal visión-lenguaje.

El fine-tuning se evalúa tanto con objetivos de clasificación como de metric learning. La rama de metric learning estudia pérdidas orientadas a mejorar la compacidad intra-clase y la separación inter-clase, incluyendo contrastive, triplet, ArcFace, NT-Xent y Multi-Similarity.

Para los experimentos multimodales, el entrenamiento sigue una lógica asimétrica:
- En entrenamiento se utilizan descripciones textuales más ricas para alinear información visual y semántica.
- En validación y prueba se emplean descripciones parciales para simular incertidumbre real respecto del modelo exacto del vehículo.
- Se estudian estrategias de congelamiento por capas y fine-tuning por fases para adaptar CLIP sin degradar excesivamente su estructura semántica preentrenada.

### Reducción de dimensionalidad y clustering

Los embeddings de alta dimensión se reducen antes del clustering para mejorar tanto la eficiencia computacional como la organización del espacio latente. El proyecto compara PCA, t-SNE y UMAP, y optimiza automáticamente sus hiperparámetros con Optuna.

La etapa de clustering se centra en algoritmos compatibles con escenarios open-set, es decir, métodos que no requieren conocer a priori el número de clases:
- DBSCAN
- HDBSCAN
- OPTICS
- Agglomerative clustering

La selección de modelos y configuraciones se guía mediante métricas internas y externas, incluyendo silhouette score, trustworthiness, criterios orientados a pureza, Adjusted Rand Index (ARI) y otras medidas de calidad de cluster.

Familias principales de modelos:
- ResNet50
- ViT-B/32
- CLIP (visión-texto)

Métodos principales de reducción y clustering:
- Clustering: DBSCAN, HDBSCAN, OPTICS, Agglomerative
- Reducción: PCA, t-SNE, UMAP

## Estudios experimentales

El repositorio agrupa los experimentos de la memoria en cuatro etapas complementarias:

| Estudio | Pregunta de investigación | Variables principales |
| --- | --- | --- |
| **Experimento 1: Establecimiento de Línea Base Unimodal** | ¿Qué arquitectura pura de visión (CNN vs Transformer) y método de aprendizaje son más efectivos? | Arquitectura (ResNet vs ViT), Paradigma (Clasificación vs Metric Learning), Vistas (Frontal vs Ambas) |
| **Experimento 2: Dinámica de Adaptación en Modelos Multimodales (CLIP)** | ¿Cuál es la estrategia óptima para adaptar un modelo visión-lenguaje al dominio vehicular sin perder semántica? | Componente ajustado (Visión y/o Texto), Profundidad entrenable (capas), Orden secuencial de entrenamiento |
| **Experimento 3: Sensibilidad al Volumen de Datos** | ¿Qué tan robusto es el sistema ante una extrema falta de ejemplos por clase (*Few-Shot*)? | Umbral de imágenes mínimas por clase (4 a 12), Generalización en escasez |
| **Experimento 4: Escalabilidad y Generalización (Testing Final)** | ¿Puede escalar esto con arquitecturas más grandes y seguir operando ante miles de clases vehículos completamente desconocidos (*Zero-Shot/Open-Set*)? | Escalamiento de arquitectura (ResNet101, ViT-B/16), Progresión de inyección de clases *One-Shot* |

En conjunto, estos estudios validan rigurosamente cómo las representaciones profundas optimizadas pueden generalizar en un ecosistema abierto con alta variabilidad.

## Resultados principales

La memoria concluye que la optimización explícita del espacio latente es la variable central detrás de una identificación vehicular exitosa. **El paradigma de *Metric Learning* superó de forma indiscutible a la clasificación tradicional**, ya que funciones como *MultiSimilarity* y *NTXent* incrementan la compacidad intra-clase y la separación inter-clase de forma más apta para el agrupamiento. Asimismo, la fusión de información multi-vista (*Front + Rear*) resultó en mejoras dramáticas al resolver ambigüedades de diseño frontal comunes.

Principales hallazgos por etapa experimental:

- **Experimento 1 (Línea Base Unimodal):** Bajo los métodos de *Metric Learning*, **ResNet50** superó consistentemente a **ViT-B/32**. La inicialización de la CNN maneja de mucha mejor manera los sesgos de invarianza y captura detalles sin la necesidad masiva de datos propia de los Transformers. ArcFace demostró alta pureza en clústeres artificiales y pequeños pero a costa de de una fuerte sobre-fragmentación.
- **Experimento 2 (Adaptación CLIP):** Se comprobó que la optimización más crítica ocurre en las capas visuales de CLIP, mientras que el encoder textual complementa para estructurar semánticamente el espacio latente. La estrategia secuencial óptima es **Vision $\to$ Text** con ajuste profundo.
- **Experimento 3 (Volumen de datos):** La arquitectura CLIP probó ser excepcionalmente robusta frente a condiciones de *Few-Shot*. Incluso limitando a 4 o 6 imágenes por clase, el ajuste multimodal generó subespacios agrupables superiores a un *baseline* aleatorio o unimodal.
- **Experimento 4 (Testing Final y Generalización):** En la validación Open-Set extrema —inyectando progresivamente más de 3000 clases minoritarias (*One-Shot*) no vistas durante el entrenamiento— se comprobó la principal limitante de las CNN frente a los modelos multimodales: mientras que ResNet50 decayó abruptamente disolviendo las clases desconocidas en ruido, **CLIP-ViT-B/32 logró estabilizarse, aislando con éxito alrededor del 40% de los clusters (con más de 770 clústeres completamente puros de clases desconocidas).** A su vez, el modelo unimodal de ViT-B/16 también presentó una robusta mejoría escalando de forma considerable al reducir el parche visual, acortando distancia con CLIP.

### Interpretación y Relevancia (Acreditada por el Testing Final)

Desde el punto de vista de investigación, la principal conclusión y aporte de esta memoria es validar que el reconocimiento de modelos vehiculares granulares es óptimamente manejable como un problema de **aprendizaje de representaciones (Metric Learning) y clustering**, en vez de ser estipulado como clasificación cerrada.

Precisamente, el **Experimento 4** respalda directamente esta interpretación: una vez que la representación latente (en particular multimodal con CLIP) adquiere suficiente capacidad discriminativa asimétrica de grano fino, los algoritmos de clustering no supervisado logran detectar y recuperar la identidad de vehículos ante modelos ausentes o desconocidos en el entrenamiento. Esto provee una fundación sólida para sistemas de detección de anomalías y fraude vehicular en autopistas, que pueden asimilar automáticamente nuevos patrones vehiculares sin costosos y constantes reentrenamientos corporativos.

## Estructura del repositorio

- [src](src): módulos centrales reutilizables (data, models, pipelines, utils)
- [scripts](scripts): scripts ejecutables de experimentos y análisis
- [configs](configs): configuraciones YAML de experimentos
- [tests](tests): suite de pruebas
- [pyproject.toml](pyproject.toml): metadata del proyecto y dependencias
- [uv.lock](uv.lock): lockfile reproducible para uv

## Configuración reproducible

Requisitos:
- Python 3.10
- uv

Instalación y sincronización:

```bash
uv sync
```

Flujo típico de ejecución:

```bash
# 1) Fine-tuning
uv run scripts/pipeline/Finetuning.py

# 2) Embeddings y clustering
uv run scripts/pipeline/Embeddings.py

# 3) Extracción de resultados
uv run scripts/extraction/extract_vision_results.py
uv run scripts/extraction/extract_clip_layers_results.py

# 4) Análisis estadístico y visualizaciones
uv run scripts/analysis/analyze_vision_models.py
uv run scripts/analysis/analyze_clip_layers.py
uv run scripts/visualization/visualize_vision_models.py
uv run scripts/visualization/visualize_clip_layers.py
```

Ejecución de pruebas:

```bash
uv run pytest tests/
```

## Contribuciones de la investigación

- Un pipeline modular de extremo a extremo para transformar CompCars en un benchmark de identificación vehicular open-set orientado a investigación.
- Una comparación controlada entre representaciones CNN, ViT y CLIP bajo objetivos de clasificación y metric learning.
- Una evaluación multi-etapa que conecta fine-tuning, reducción de dimensionalidad, clustering y análisis estadístico.
- Evidencia empírica de que embeddings de alta calidad permiten descubrimiento no supervisado de modelos vehiculares con alta pureza de cluster.

## Limitaciones

- Los experimentos se centran en CompCars, por lo que sesgos del dataset y artefactos propios de imágenes obtenidas desde la web pueden afectar la generalización a otros dominios vehiculares.
- La memoria prioriza vistas frontales y traseras, que son operacionalmente relevantes, pero no agotan toda la variabilidad de sistemas reales de vigilancia.
- La calidad del clustering sigue dependiendo de decisiones de reducción de dimensionalidad y búsqueda de hiperparámetros, incluso cuando el espacio de embeddings es fuerte.
- Algunos de los mejores resultados provienen de configuraciones de investigación cuidadosamente ajustadas y no deben interpretarse directamente como rendimiento listo para despliegue.

## Trabajo futuro

- **Exploración con Imágenes No Etiquetadas:** Uso de técnicas de aprendizaje auto-supervisado (*Self-Supervised Learning*) en grandes volúmenes de datos sin curación manual para entrenamiento con datos de cámaras de la vida real.
- **Sistemas de Recuperación (*Retrieval*):** Evaluar los *embeddings* generados en tareas de recuperación de imágenes basada en contenido (búsqueda de vehículos similares), con aplicabilidad a investigaciones forenses.
- **Análisis de Identidad y Re-identificación:** Evaluar si las representaciones permiten discriminar por instancia única (daños, calcomanías, variaciones específicas del vehículo) en lugar de solo por modelo global.
- **Ingeniería de Etiquetas Textuales (*Prompt Engineering*):** Analizar el impacto de usar descripciones más ricas y detalladas al estructurar el espacio visual durante el ajuste fino en modelos multimodales (ej. CLIP).
- **Nuevos Algoritmos de Reducción y Clustering:** Evaluar técnicas nuevas o más centradas a la alta dimensión para preservar de mejor manera topologías de manera local y global sin necesidad de un ajuste profundo de hiperparámetros.
- **Arquitecturas Híbridas:** Combinar fortalezas de redes convolucionales convencionales (para texturas y patrones locales) y *Transformers* (para campo receptivo global).

## Citación

Si utilizas este repositorio en trabajo académico, cita tanto la memoria como el dataset CompCars.

Cita sugerida para la memoria:

```text
Diego Andrés Vega Acuña.
Detección de modelos y tipos de vehículos fraudulentos en autopistas mediante aprendizaje profundo no supervisado.
Memoria para optar al título de Ingeniero Civil Eléctrico, Universidad de Chile, 2026.
```

Referencia del repositorio:

```text
Diego Vega.
CompCars Deep Learning Analysis.
Repositorio GitHub.
```

## Declaración de uso del dataset CompCars

Este repositorio utiliza el dataset CompCars bajo sus restricciones publicadas.

Note:
The CompCars database is available for non-commercial research purposes only.
All images of the CompCars database are obtained from the Internet which are not property of MMLAB, The Chinese University of Hong Kong. The MMLAB is not responsible for the content nor the meaning of these images.
You agree not to reproduce, duplicate, copy, sell, trade, resell or exploit for any commercial purposes, any portion of the images and any portion of derived data.
You agree not to further copy, publish or distribute any portion of the CompCars database. Except, for internal use at a single site within the same organization it is allowed to make copies of the database.
The MMLAB reserves the right to terminate your access to the database at any time.
All submitted papers or any publicly available text using the CompCars database must cite the following paper:
Linjie Yang, Ping Luo, Chen Change Loy, Xiaoou Tang. A Large-Scale Car Dataset for Fine-Grained Categorization and Verification, In Computer Vision and Pattern Recognition (CVPR), 2015.

Cita de CompCars:

```text
Linjie Yang, Ping Luo, Chen Change Loy, Xiaoou Tang.
A Large-Scale Car Dataset for Fine-Grained Categorization and Verification.
Proceedings of IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2015.
```

## Licencia

Este repositorio está licenciado bajo Creative Commons Attribution-NonCommercial-NoDerivs 3.0 Chile.
Ver [LICENSE](LICENSE).
