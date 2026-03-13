# CompCars Deep Learning Analysis

[Español](README.es.md)

**Author:** Diego Vega  
**Links:** [Thesis (PDF) - Placeholder](#) | [Full Results (Google Drive) - Placeholder](#)

Research repository for the Electrical Engineering thesis focused on non-supervised vehicle model identification using deep representations and clustering.

Thesis title:
Detection of fraudulent vehicle models and types on highways through unsupervised deep learning.

## Research context

This project studies whether deep embeddings extracted from fine-tuned vision and multimodal models can separate vehicle models in latent space without traditional closed-set classification at inference time.

Problem setting:
- Real-world highway scenarios may contain unseen vehicle models.
- Label availability is limited and long-tail class imbalance is severe.
- Classical supervised pipelines are brittle under open-set conditions.

Working hypothesis:
- Multimodal deep representations (vision plus text), optimized with metric learning and clustering-oriented preprocessing, provide stronger discriminative structure than unimodal vision-only baselines for open-set vehicle model discovery.

Research objectives:
- Evaluate whether deep embeddings can support vehicle model identification without closed-set classification at inference time.
- Compare CNN, Transformer, and vision-language backbones under a unified experimental protocol.
- Measure the effect of metric learning, multi-view fusion, dimensionality reduction, and unsupervised clustering on latent-space separability.
- Assess generalization to low-data and previously unseen vehicle classes.

## Methodology overview

The workflow is organized in four stages:
1. Data curation and preprocessing
2. Supervised fine-tuning for embedding quality improvement
3. Dimensionality reduction optimization
4. Fully unsupervised clustering and statistical analysis

### Data strategy

The repository operationalizes CompCars into a training-ready tabular dataset centered on front and rear views, which are the most relevant perspectives for highway monitoring. A dedicated curation stage filters irrelevant viewpoints, consolidates metadata, and generates a single CSV used as the source of truth for downstream pipelines.

The experimental design explicitly addresses long-tail imbalance:
- Classes are defined at the vehicle model-year level.
- A minimum-image threshold separates regular classes from one-shot classes.
- Regular classes are used for supervised fine-tuning.
- Low-sample classes are held out to probe open-set and few-shot generalization.
- In the front+rear setting, the threshold must be satisfied in both views to guarantee valid paired samples.

### Representation learning stage

Three model families are studied under a common transfer-learning framework:
- ResNet50 as a CNN baseline.
- ViT-B/32 as a pure Transformer vision baseline.
- CLIP as a multimodal vision-language baseline.

Fine-tuning is evaluated with both classification and metric-learning objectives. The metric-learning branch studies losses designed to improve intra-class compactness and inter-class separation, including contrastive, triplet, ArcFace, NT-Xent, and Multi-Similarity objectives.

For multimodal experiments, the training procedure is intentionally asymmetric:
- Training uses richer textual descriptions to align visual and semantic information.
- Validation and test use partial descriptions to simulate realistic uncertainty about the exact vehicle model.
- Layer freezing and staged fine-tuning are used to study how much CLIP should adapt without eroding pretrained semantic structure.

### Dimensionality reduction and clustering

High-dimensional embeddings are reduced before clustering to improve both computational efficiency and latent-space organization. The project compares PCA, t-SNE, and UMAP, and optimizes their hyperparameters automatically with Optuna.

The clustering stage focuses on open-set-compatible algorithms that do not require the number of classes in advance:
- DBSCAN
- HDBSCAN
- OPTICS
- Agglomerative clustering

Model selection is guided by internal and external metrics, including silhouette score, trustworthiness, purity-oriented criteria, Adjusted Rand Index (ARI), and other cluster-quality measures.

Main model families:
- ResNet50
- ViT-B/32
- CLIP (vision-text)

Main clustering and reduction methods:
- Clustering: DBSCAN, HDBSCAN, OPTICS, Agglomerative
- Reduction: PCA, t-SNE, UMAP

## Experimental studies

The repository groups the thesis experiments into four complementary stages:

| Study | Research question | Main variables |
| --- | --- | --- |
| **Experiment 1: Unimodal Baseline Establishment** | Which pure vision architecture (CNN vs Transformer) and learning paradigm are most effective? | Architecture (ResNet vs ViT), Paradigm (Classification vs Metric Learning), Views (Frontal vs Both) |
| **Experiment 2: Multimodal Model Adaptation Dynamics (CLIP)** | What is the optimal strategy to adapt a vision-language model to the proper vehicle domain without losing semantics? | Tuned component (Vision and/or Text), Trainable depth (layers), Sequential training order |
| **Experiment 3: Data Volume Sensitivity** | How resilient is the system against extreme sample scarcity (*Few-Shot*)? | Minimum images per class threshold (4 to 12), Generalization in scarcity |
| **Experiment 4: Scalability and Generalization (Final Testing)** | Does this scale with larger architectures and generalize well over thousands of completely unknown vehicles (*Zero-Shot/Open-Set*)? | Architecture scaling (ResNet101, ViT-B/16), Progressive injection of *One-Shot* classes |

Together, these studies rigorously validate how optimized deep representations can generalize in an open ecosystem with high variability.

## Key results

The thesis concludes that explicitly optimizing the latent space is the central variable behind successful vehicle identification. **The *Metric Learning* paradigm indisputably outperformed traditional classification**, as functions like *MultiSimilarity* and *NTXent* increase intra-class compactness and inter-class separation, making the space much more suitable for clustering. Likewise, multi-view fusion (*Front + Rear*) resulted in dramatic improvements by resolving common frontal design ambiguities.

Main findings by experimental stage:

- **Experiment 1 (Unimodal Baseline):** Under *Metric Learning* methods, **ResNet50** consistently outperformed **ViT-B/32**. The CNN's initialization handles invariance biases much better and captures fine details without the massive data appetite of Transformers. ArcFace demonstrated high purity in small artificial clusters but suffered from severe over-fragmentation.
- **Experiment 2 (CLIP Adaptation):** It was proven that the most critical optimization occurs in CLIP's visual layers, while the textual encoder complements to semantically structure the latent space. The optimal sequential strategy is **Vision $\to$ Text** with deep layer unfreezing.
- **Experiment 3 (Data Volume):** The CLIP architecture proved exceptionally robust against *Few-Shot* conditions. Even when constrained to 4 or 6 images per class, the multimodal fine-tuning generated clusterable subspaces superior to a random or unimodal baseline.
- **Experiment 4 (Final Testing & Generalization):** In the extreme Open-Set validation —progressively injecting more than 3000 unseen minority classes (*One-Shot*)— the main limitation of CNNs versus multimodal models was exposed: while ResNet50 decayed abruptly, dissolving unknown classes into noise, **CLIP-ViT-B/32 managed to stabilize, successfully isolating around 40% of the clusters (with over 770 completely pure clusters of unknown classes).** In turn, the unimodal ViT-B/16 model also presented a robust improvement scaling considerably when reducing the visual patch size, bridging the gap with CLIP.

### Interpretation and Relevance (Backed up by Final Testing)

From a research perspective, the main conclusion and contribution of this thesis is validating that granular vehicle model recognition is optimally handled as a **representation learning (Metric Learning) and clustering problem**, rather than being constrained as closed-set classification.

Precisely, **Experiment 4** directly supports this interpretation: once the latent representation (particularly multimodal with CLIP) acquires sufficient fine-grained discriminative asymmetric capacity, unsupervised clustering algorithms manage to detect and retrieve the identity of vehicles, even for models absent or unknown during training. This provides a solid foundation for vehicular anomaly and fraud detection systems on highways, capable of automatically assimilating new vehicle patterns without costly, continuous corporate retraining routines.

## Repository structure

- [src](src): reusable core modules (data, models, pipelines, utils)
- [scripts](scripts): runnable experiment and analysis scripts
- [configs](configs): YAML experiment configurations
- [tests](tests): test suite
- [pyproject.toml](pyproject.toml): project metadata and dependencies
- [uv.lock](uv.lock): reproducible lockfile for uv

## Reproducible setup

Requirements:
- Python 3.10
- uv

Install and sync:

```bash
uv sync
```

Typical execution flow:

```bash
# 1) Fine-tuning
uv run scripts/pipeline/Finetuning.py

# 2) Embeddings and clustering
uv run scripts/pipeline/Embeddings.py

# 3) Result extraction
uv run scripts/extraction/extract_vision_results.py
uv run scripts/extraction/extract_clip_layers_results.py

# 4) Statistical analysis and plots
uv run scripts/analysis/analyze_vision_models.py
uv run scripts/analysis/analyze_clip_layers.py
uv run scripts/visualization/visualize_vision_models.py
uv run scripts/visualization/visualize_clip_layers.py
```

Run tests:

```bash
uv run pytest tests/
```

## Research contributions

- A modular end-to-end pipeline for transforming CompCars into a research-ready open-set vehicle identification benchmark.
- A controlled comparison of CNN, ViT, and CLIP-based representations under classification and metric-learning objectives.
- A multi-stage evaluation pipeline connecting fine-tuning, dimensionality reduction, clustering, and statistical analysis.
- Empirical evidence that high-quality embeddings enable non-supervised vehicle model discovery with strong cluster purity.

## Limitations

- The experiments are centered on CompCars, so dataset bias and web-sourced image artifacts may affect generalization to other traffic domains.
- The thesis prioritizes front and rear views, which are operationally relevant but do not exhaust the full variability of real surveillance footage.
- Clustering quality still depends on reduction and hyperparameter-search choices, even when the embedding space is strong.
- Some of the strongest results come from carefully tuned research settings and should not be interpreted as plug-and-play deployment performance.

## Future work

- **Exploration with Unlabeled Images:** Use of *Self-Supervised Learning* techniques on large volumes of uncurated web/camera data to evaluate model efficiency in the wild without labels.
- **Content-Based Image Retrieval:** Evaluate the generated *embeddings* directly in image retrieval tasks (searching for "similar vehicles") with applications in security and forensics.
- **Identity Analysis and Re-identification:** Explore whether representations can discriminate at a single-instance level (based on damage, stickers, specific variations) rather than just broad vehicle models.
- **Prompt Engineering:** Study the impact of using richer, more detailed textual descriptions on visual space structuring during multimodal fine-tuning (e.g. CLIIP).
- **New Reduction and Clustering Algorithms:** Evaluate new dimensionality reduction and clustering methods that better preserve local and global topologies without requiring an exhaustive hyperparameter search.
- **Hybrid Architectures:** Investigate architectures combining the strengths of convolutional networks (for local patterns and textures extraction) and Transformers (for global semantics).

## Citation

If you use this repository in academic work, cite both the thesis and the CompCars dataset.

Suggested thesis citation:

```text
Diego Andrés Vega Acuña.
Detection of fraudulent vehicle models and types on highways through unsupervised deep learning.
Electrical Engineering Thesis, Universidad de Chile, 2026.
```

Repository reference:

```text
Diego Vega.
CompCars Deep Learning Analysis.
GitHub repository.
```

## CompCars dataset usage declaration

This repository uses the CompCars dataset under its published restrictions.

Note:
The CompCars database is available for non-commercial research purposes only.
All images of the CompCars database are obtained from the Internet which are not property of MMLAB, The Chinese University of Hong Kong. The MMLAB is not responsible for the content nor the meaning of these images.
You agree not to reproduce, duplicate, copy, sell, trade, resell or exploit for any commercial purposes, any portion of the images and any portion of derived data.
You agree not to further copy, publish or distribute any portion of the CompCars database. Except, for internal use at a single site within the same organization it is allowed to make copies of the database.
The MMLAB reserves the right to terminate your access to the database at any time.
All submitted papers or any publicly available text using the CompCars database must cite the following paper:
Linjie Yang, Ping Luo, Chen Change Loy, Xiaoou Tang. A Large-Scale Car Dataset for Fine-Grained Categorization and Verification, In Computer Vision and Pattern Recognition (CVPR), 2015.

CompCars citation:

```text
Linjie Yang, Ping Luo, Chen Change Loy, Xiaoou Tang.
A Large-Scale Car Dataset for Fine-Grained Categorization and Verification.
Proceedings of IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2015.
```

## License

This repository is licensed under Creative Commons Attribution-NonCommercial-NoDerivs 3.0 Chile.
See [LICENSE](LICENSE).