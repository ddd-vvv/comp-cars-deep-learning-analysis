# CompCars Deep Learning Analysis

[Español](README.es.md)

**Author:** Diego Andrés Vega Acuña  
**Advisor:** Carlos Navarro Clavería  
**Links:** [Thesis (PDF)](https://drive.google.com/file/d/1vWpu36wMhaFOe7DIaxsMcTYh8jedxFZU/view) | [Full results (Google Drive)](https://drive.google.com/drive/folders/1V83e9XZ7O_Hcy8RAhOQ6QsEIfco-g_Xv)

Research repository for the undergraduate thesis submitted for the degree of Electrical Engineer (Ingeniero Civil Eléctrico, Universidad de Chile, 2026), focused on the unsupervised identification of vehicle models using deep representations (*embeddings*) and clustering techniques.

**Thesis title (original in Spanish):**  
*Detección de modelos y tipos de vehículos fraudulentos en autopistas mediante aprendizaje profundo no supervisado.*  
(*Detection of fraudulent vehicle models and types on highways using unsupervised deep learning.*)

---

## Research context

This project studies whether the semantic structure of deep *embeddings*, extracted from fine-tuned pure-vision and multimodal models, is robust and discriminative enough to identify vehicle models (seen or previously unseen) through clustering techniques, without relying on traditional classification methods.

**Problem statement:**
- In real-world scenarios, such as highway surveillance, previously unseen vehicle models frequently appear.
- Labeled data is limited and follows a long-tail distribution.
- Traditional classification methods depend on large labeled datasets and require the target classes to be present in the training set (*closed-set*).

**Working hypothesis:**  
Combining multimodal deep representations (vision and text, based on *Transformers*) optimized through *Metric Learning* with unsupervised clustering techniques (with automatic hyperparameter exploration) yields *embeddings* with higher discriminative quality and generalization robustness (*Zero-Shot* / *Few-Shot*) than representations produced by unimodal (pure-vision) models, enabling higher clustering purity for vehicle models (both seen and unseen).

**General objective:**  
Develop, implement and evaluate a system based on deep representations and unsupervised clustering techniques for the effective identification of vehicle models in scenarios with limited data and previously unseen models.

**Specific objectives:**
- Investigate, select and compare neural network architectures (CNNs, *Vision Transformers* and vision-language models) for generating deep representations of vehicles.
- Implement a supervised fine-tuning process on specialized data to optimize the quality of the *embeddings*.
- Compare unsupervised clustering techniques for grouping and identifying vehicle models from those representations.
- Propose and implement an unsupervised vehicle model classification system.
- Analyze the system's effectiveness in terms of cluster purity and its ability to generalize to previously unseen models.

---

## Cloud Infrastructure (GCP) and Tech Stack

The computationally expensive experiments ran on **Google Cloud Platform (GCP)** using two virtual machine instances: one with a GPU for fine-tuning the deep models, and one with a high-performance CPU for analyzing the *embeddings* (dimensionality reduction and clustering).

| GCP instance | CPU | RAM | GPU | Storage | Role in the pipeline |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **GPU-VM** | `n1-standard-4` (4 vCPUs) | 15 GB | NVIDIA Tesla T4 (16 GB VRAM) | 100 GB SSD | Supervised and multimodal fine-tuning (AMP, P × K sampling). |
| **CPU-VM** | `c4-standard-16` (16 vCPUs) | 60 GB | N/A | 100 GB SSD | Dimensionality reduction (incremental PCA, t-SNE, UMAP) and clustering optimization with Optuna. |

**Main tools:**
- **Deep Learning and ML:** PyTorch (with *Automatic Mixed Precision*, `AMP` and `GradScaler`), `pytorch-metric-learning`, Scikit-learn, UMAP-learn.
- **Hyperparameter optimization:** Optuna (Bayesian *Tree-structured Parzen Estimator* sampler, `TPE`, and *Median Pruning*).
- **Environment and data management:** Python 3.10, `uv` (package manager written in Rust), Pandas, JSON and CSV.
- **Version control:** Git and GitHub.

---

## Methodology overview

The modular, object-oriented workflow is organized into four sequential stages:
1. **Stage 0:** Data strategy, long-tail handling and preprocessing with stochastic data augmentation.
2. **Stage 1:** Supervised fine-tuning (Classification vs. *Metric Learning* for pure-vision models; layer-wise and phased tuning for CLIP).
3. **Stage 2:** Optimization of the latent-space dimensionality reduction.
4. **Stage 3:** Unsupervised clustering without knowing the number of classes a priori, and evaluation.

### Data strategy (CompCars dataset)

The `DataFrameMaker` module turns the *Comprehensive Cars* (CompCars) dataset into a master CSV file that keeps only the **front (`Front`)** and **rear (`Rear`)** viewpoints:
- **Class definition:** combination of `make + model + release year` (e.g. `Audi_A4_2010` vs. `Audi_A4_2015`).
- **Long-tail imbalance handling:** a `min_images` threshold splits the data into **Regular Classes** (≥ `min_images`, used for training, validation and testing) and **One-Shot Classes** (< `min_images`, excluded from training and reserved for evaluating generalization to unseen models).
- **Multi-view intersection (`Both`):** the set of classes with both viewpoints totals 3,598 classes and 31,073 images; in this configuration, the `min_images` threshold must be met in both viewpoints simultaneously. With `min_images = 8`, this yields 359 regular classes and 3,239 One-Shot classes.
- **Data augmentation:** rotations, translations, *Random Horizontal Flip*, *Color Jitter*, *Gaussian Blur*, grayscale conversion (replicated across 3 channels to stay compatible with pretrained weights) and *Random Erasing*.

### Representation learning

Three architecture families are evaluated, starting from pretrained weights:
- **CNN:** ResNet50 and ResNet101 (D = 2048).
- **Vision Transformers:** ViT-B/32 and ViT-B/16 (D = 768).
- **Multimodal (vision-language):** CLIP ViT-B/32 (D = 512).

For pure-vision models, the **Cross-Entropy** loss is compared against five **Metric Learning** losses (with P × K sampling through a wrapper around `MPerClassSampler`): *Contrastive Loss*, *Triplet Loss*, *NT-Xent*, *ArcFace* and *Multi-Similarity Loss*.

For **CLIP**, an **asymmetric, sequential phased** training strategy was designed:
- During **training**, full text descriptions are used (*"The front view image of a Skoda Haorui vehicle from 2012 type sedan"*).
- During **validation and testing**, the model and year are hidden (*"The front view image of a Skoda vehicle type sedan"*), simulating the real-world scenario in which the system does not know the exact model.
- Progressive layer-wise unfreezing and two-phase tuning are applied (**Phase 1: Visual Refinement → Phase 2: Textual Adaptation**).

The training pipeline includes differentiated *Weight Decay* (excluding biases and normalization layers), mixed precision (AMP), *Gradient Clipping*, *WarmUp* and *Early Stopping*.

### Dimensionality reduction and clustering

Projections with **Incremental PCA**, **t-SNE** and **UMAP** are optimized with **Optuna** (50 trials per algorithm). For t-SNE and UMAP, the objective is a composite of normalized *Silhouette Score* (70%) and *Trustworthiness* (30%); for PCA, only the *Silhouette Score* is maximized.

Clustering excludes algorithms that require fixing the number of classes k a priori (such as K-Means) and evaluates the following, with **250 Optuna trials per algorithm**:
- **Density-based:** DBSCAN, HDBSCAN and OPTICS.
- **Hierarchical:** Agglomerative Clustering.

The clustering objective function combines internal and external metrics: 0.35 · *Silhouette* + 0.25 · *Calinski-Harabasz* (both normalized) + 0.25 · ARI + 0.15 · NMI. In other words, the available labels (40% of the score) are used to guide the search during optimization, and degenerate configurations are discarded (e.g. noise > 40% or fewer than 2 clusters).

---

## Experimental studies

| Study | Research question | Main variables |
| :--- | :--- | :--- |
| **Experiment 1: Unimodal baseline** | Which pure-vision architecture (CNN vs. Transformer) and which learning paradigm produce the best latent space? | Architecture (ResNet50 vs. ViT-B/32), objective (Classification vs. 5 *Metric Learning* losses), viewpoints (`Front` vs. `Front + Rear`), with `min_images = 8`. |
| **Experiment 2: CLIP adaptation** | How much plasticity does each encoder need, and what phase order is most effective for adapting CLIP to the vehicle domain? | Progressive unfreezing (1 to 12 layers in the `Image Encoder` and `Text Encoder`), phase order (*Vision → Text* vs. *Text → Vision*). |
| **Experiment 3: Sensitivity to data volume** | How robust is CLIP to a scarcity of examples per class? | `min_images` threshold ∈ {4, 6, 8, 10, 12}. |
| **Experiment 4: Scalability and generalization** | Does clustering improve when scaling the architecture, and how does the system respond to never-seen classes (*Open-Set*)? | Scaling (ResNet101, ViT-B/16); progressive injection (0%, 25%, 50%, 75%, 100%) of the 3,239 One-Shot classes. |

---

## Main results

1. **Metric Learning and multi-view fusion (Exp. 1):** on average, *Metric Learning* outperformed *Cross-Entropy* classification in ARI (0.22 vs. 0.17) and in the percentage of pure clusters (48.1% vs. 42.0%), producing around 125–130 pure clusters versus 10–15 for classification. **Multi-Similarity** and **NT-Xent** led in Recall@k and ARI; *ArcFace* achieved the highest percentage of pure clusters but a very low ARI, due to over-fragmentation of the latent space; *Triplet Loss* performed worst. Combining front and rear views (`Front + Rear`) raised the average ARI from 0.07 to 0.38. **ResNet50** outperformed **ViT-B/32** (average ARI 0.2605 vs. 0.1718) with a shorter average training time (88.8 vs. 97.0 min). The best unimodal baseline was ResNet50 + Multi-Similarity with `Front + Rear` views.
2. **CLIP adaptation (Exp. 2):** tuning the visual encoder was the main driver of performance, while the text encoder played a complementary role in the global structuring of the latent space. Based on the defined success metric, unfreezing was fixed at **12 layers of the `Image Encoder` and 9 layers of the `Text Encoder`**. The **Vision → Text** sequence outperformed the reverse order in Recall, ARI, purity and number of overlapping classes, with similar compute times.
3. **Sensitivity to data volume (Exp. 3):** even with `min_images = 4`, fine-tuned CLIP clearly outperformed the non-tuned baseline. Performance grew with data volume, reaching an **ARI of 0.86** with 12 images per class. `min_images = 8` was identified as the balance point between data collection effort and clustering quality; Exp. 4 used 10 images per class to maximize performance.
4. **Scalability and generalization (Exp. 4):**
   - **Architecture scaling:** moving from ResNet50 to ResNet101 did not improve performance, suggesting capacity saturation in the CNN. In contrast, moving from **ViT-B/32 to ViT-B/16** (16 × 16 pixel patches) raised ARI from ~0.66 to ~0.86 and reduced mixed clusters, presumably by capturing fine details such as headlights or logos. ViT-B/16 led in ARI among the evaluated architectures; CLIP-ViT-B/32 was kept unscaled.
   - **Open-Set generalization:** the three best models were evaluated (CLIP-ViT-B/32, ViT-B/16 with *Contrastive Loss* and ResNet50 with *Multi-Similarity*). On the 359 regular classes (`Ratio 0.0`), **CLIP-ViT-B/32** reached **89.0% cluster purity**, versus 79.4% for ViT-B/16 and 69.9% for ResNet50. When injecting 100% of the 3,239 One-Shot classes (3,598 classes in total, `Ratio 1.0`), **CLIP** kept the highest purity (**38.8%**, with **770 pure clusters**), followed by ViT-B/16 (35.8%, 681 pure clusters) and ResNet50 (23.7%, 401 pure clusters).

### Relevance and practical impact

The results indicate that vehicle models can be identified in an unsupervised way from deep representations, even for models not included in fine-tuning. This has direct implications for anomaly and fraud detection on highways, since it allows detecting unknown or underrepresented models without retraining the network for each new model.

---

## Repository structure

- [`src/`](src): Core object-oriented modules (`config`, `data`, `models`, `pipeline`, `utils`).
- [`scripts/`](scripts): Executable scripts for the fine-tuning and embeddings pipelines, extraction (`JSON → CSV`), statistical analysis and visualization.
- [`configs/`](configs): Experimental configuration files in YAML (`resnet50_*.yaml`, `vitb32_*.yaml`, `CLIP.yaml`).
- [`tests/`](tests): Unit and integration tests.
- [`pyproject.toml`](pyproject.toml) and [`uv.lock`](uv.lock): Project definition and dependency lock with `uv`.

The `results/`, `dataset/image/` and `.venv/` folders are not version-controlled; full results are available on [Google Drive](https://drive.google.com/drive/folders/1V83e9XZ7O_Hcy8RAhOQ6QsEIfco-g_Xv).

---

## Reproducible setup

**Requirements:**
- Python 3.10
- [`uv`](https://github.com/astral-sh/uv)

**Environment installation:**

```bash
uv sync
```

**Main execution flow:**

```bash
# 1) Supervised / multimodal fine-tuning (GPU-VM)
uv run scripts/pipeline/Finetuning.py

# 2) Embeddings, dimensionality reduction and clustering with Optuna (CPU-VM)
uv run scripts/pipeline/Embeddings.py

# 3) Tabular extraction of results (JSON to CSV)
uv run scripts/extraction/extract_vision_results.py
uv run scripts/extraction/extract_clip_layers_results.py

# 4) Statistical analysis and plot generation
uv run scripts/analysis/analyze_vision_models.py
uv run scripts/analysis/analyze_clip_layers.py
uv run scripts/visualization/visualize_vision_models.py
uv run scripts/visualization/visualize_clip_layers.py
```

**Running tests:**

```bash
uv run pytest tests/
```

---

## Limitations and future work

**Limitations:**
- The experiments are based on CompCars, whose images come from the Internet and may differ from those captured by fixed cameras on highway gantries.
- Clustering hyperparameter optimization uses the ground-truth labels (ARI and NMI) as part of its objective function, and the reduction and clustering stages require Optuna searches for each set of *embeddings*.

**Future work:**
- **Self-Supervised Learning:** leverage large volumes of unlabeled vehicle images available on the web or from security cameras.
- **Content-Based Image Retrieval:** evaluate the *embeddings* on similar-vehicle search, with applications in digital forensics.
- **Instance-level re-identification:** distinguish specific vehicles of the same model by details such as stickers, damage or accessories, or alternatively enforce invariance to them.
- **Text label engineering (*Prompt Engineering*):** study the impact of richer descriptions (year, body type, country of origin) on fine-tuning multimodal models.
- **New reduction and clustering algorithms:** evaluate methods that better preserve local and global topology without requiring such exhaustive hyperparameter searches.
- **Hybrid CNN-Transformer architectures:** combine the local pattern extraction of convolutions with the global understanding of Transformers.

---

## Citation

If you use this code or its findings in your research, please cite the thesis and the CompCars dataset:

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

## CompCars dataset usage statement

This repository uses the CompCars dataset under its published restrictions for non-commercial research:

```text
The CompCars database is available for non-commercial research purposes only. All images of the CompCars database are obtained from the Internet which are not property of MMLAB, The Chinese University of Hong Kong. The MMLAB is not responsible for the content nor the meaning of these images. You agree not to reproduce, duplicate, copy, sell, trade, resell or exploit for any commercial purposes, any portion of the images and any portion of derived data. You agree not to further copy, publish or distribute any portion of the CompCars database. Except, for internal use at a single site within the same organization it is allowed to make copies of the database.
```

---

## License

This repository is licensed under Creative Commons Attribution-NonCommercial-NoDerivs 3.0 Chile. See [LICENSE](LICENSE).
