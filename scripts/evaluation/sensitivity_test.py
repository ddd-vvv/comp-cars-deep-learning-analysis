import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score, silhouette_score

# --- PROJECT IMPORTS ---
from src.models.MyVisionModel import create_vision_model
from src.models.MyCLIPModel import create_clip_model
from src.data.MyDataset import create_car_dataset
from src.config.TransformConfig import create_standard_transform

import src.models.MyVisionModel
src.models.MyVisionModel.DEFAULT_WEIGHTS_FILENAME = "vision_model.pth"
# ------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve_repo_path(path_str: str) -> Path:
    """Resolve a path relative to repository root unless absolute."""
    path = Path(path_str)
    return path if path.is_absolute() else REPO_ROOT / path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# 1. HARDCODED CONFIGURATION
# ============================================================================

ONESHOT_RATIOS = [0.0, 0.25, 0.5, 0.75, 1.0]

DATASET_CONFIG = {
    "dataset_path": "dataset.csv",
    "views": ["front", "rear"],
    "class_granularity": "model+year",
    "min_images": 8,
    "image_size": [224, 224],
    "batch_size": 32,
    "train_ratio": 0.7,
    "val_ratio": 0.15,
    "test_ratio": 0.15,
    "grayscale": True,
    "use_bbox": True
}

# --- EXPERIMENTS ---
EXP_RESNET = {
    "id": "ResNet50_MultiSim",
    "type": "vision",
    "model_name": "resnet50",
    "weights_type": "IMAGENET1K_V1",
    "objective": "metric_learning",
    "model_dir": "results/models/resnet50/metric learning/multisimilarity/finetuning/front + rear",
    "tsne_params": {
        'n_components': 2, 'perplexity': 148, 'learning_rate': 42.10, 'max_iter': 1657, 
        'early_exaggeration': 34.08, 'metric': 'euclidean', 'init': 'pca'
    },
    "dbscan_params": {
        'eps': 0.3999, 'min_samples': 2, 'metric': 'euclidean'
    }
}

EXP_VIT = {
    "id": "ViT-B16_Contrastive",
    "type": "vision",
    "model_name": "vit_b_16",
    "weights_type": "IMAGENET1K_V1",
    "objective": "metric_learning",
    "model_dir": "results/models/vitb16/metric learning/contrastive/finetuning/front + rear",
    "tsne_params": {
        'n_components': 2, 'perplexity': 135, 'learning_rate': 45.67, 'max_iter': 2148, 
        'early_exaggeration': 13.26, 'metric': 'euclidean', 'init': 'pca'
    },
    "dbscan_params": {
        'eps': 0.4825, 'min_samples': 2, 'metric': 'euclidean'
    }
}

EXP_CLIP = {
    "id": "CLIP_10Images",
    "type": "clip",
    "model_name": "clip-vit-base-patch32",
    "objective": "CLIP",
    "model_dir": "results/models/clip/min_images_study/finetuning/10 images/front + rear",
    "tsne_params": {
        'n_components': 2, 'perplexity': 116, 'learning_rate': 37.06, 'max_iter': 2023, 
        'early_exaggeration': 5.75, 'metric': 'euclidean', 'init': 'pca'
    },
    "dbscan_params": {
        'eps': 0.2098, 'min_samples': 2, 'metric': 'euclidean'
    }
}

EXPERIMENTS = [EXP_RESNET, EXP_VIT, EXP_CLIP]

# ============================================================================
# 2. FUNCTIONS
# ============================================================================

def extract_features_and_labels(model, dataset_dict, model_type):
    """Extract embeddings and labels from the test loader.

    Args:
        model: Loaded vision or CLIP model.
        dataset_dict: Dataset dictionary returned by `create_car_dataset`.
        model_type: Either ``'vision'`` or ``'clip'``.

    Returns:
        Tuple with:
        - Embeddings matrix as numpy array.
        - Ground-truth label indices as numpy array.
    """
    device = model.device
    test_loader = dataset_dict['test_loader']
    all_embeddings = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Extracting Embeddings", leave=False):
            images = batch['images'].to(device)
            labels = batch['labels']

            if model_type == 'vision':
                embeddings = model(images)
            else:
                texts = batch['text_description']
                img_emb, txt_emb = model.extract_joint_embeddings(images, texts)
                embeddings = torch.cat([img_emb, txt_emb], dim=1)

            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.append(labels.numpy())

    return np.concatenate(all_embeddings, axis=0), np.concatenate(all_labels, axis=0)


def create_dataset_for_ratio(df_full, ratio, experiment_type):
    """Build dataset loaders for a given one-shot ratio.

    Args:
        df_full: Full dataset dataframe.
        ratio: One-shot ratio in test split.
        experiment_type: Either ``'clip'`` or ``'vision'``.

    Returns:
        Dataset dictionary compatible with training/evaluation utilities.
    """
    val_transform = create_standard_transform(
        size=tuple(DATASET_CONFIG["image_size"]),
        augment=False,
        grayscale=DATASET_CONFIG["grayscale"],
        use_bbox=DATASET_CONFIG["use_bbox"],
    )

    return create_car_dataset(
        df=df_full,
        views=DATASET_CONFIG["views"],
        min_images=DATASET_CONFIG["min_images"],
        train_ratio=DATASET_CONFIG["train_ratio"],
        val_ratio=DATASET_CONFIG["val_ratio"],
        test_ratio=DATASET_CONFIG["test_ratio"],
        val_transform=val_transform,
        batch_size=DATASET_CONFIG["batch_size"],
        include_oneshot_in_test=True,
        oneshot_ratio=ratio,
        model_type='multimodal' if experiment_type == 'clip' else 'vision',
        description_include='full',
    )


def load_experiment_model(exp, dataset_dict, device):
    """Load a vision or CLIP model with pretrained weights for an experiment.

    Args:
        exp: Experiment configuration dictionary.
        dataset_dict: Dataset dictionary.
        device: Torch device.

    Returns:
        Loaded model ready for inference.

    Raises:
        FileNotFoundError: If required model files are missing.
    """
    dir_path = resolve_repo_path(exp["model_dir"])

    if exp['type'] == 'vision':
        model = create_vision_model(
            model_name=exp['model_name'],
            weights=exp['weights_type'],
            device=device,
            objective=exp['objective'],
            dataset_dict=dataset_dict,
            batch_size=DATASET_CONFIG["batch_size"],
        )

        # VisionModel expects `vision_model.pth` inside the `model.pth` folder.
        vision_weights_folder = dir_path / "model.pth"
        if not vision_weights_folder.exists():
            raise FileNotFoundError(f"Folder not found: {vision_weights_folder}")
        model.load_model(vision_weights_folder)
        return model

    model = create_clip_model(
        model_name=exp['model_name'],
        device=device,
        dataset_dict=dataset_dict,
        batch_size=DATASET_CONFIG["batch_size"],
    )

    # CLIP expects `model.pth` directly in the experiment directory.
    clip_file_path = dir_path / "model.pth"
    if not clip_file_path.exists():
        raise FileNotFoundError(f"model.pth does not exist in {dir_path}")
    model.load_model(clip_file_path)
    return model


def build_clusters_output(y_pred, y_true_names):
    """Build JSON-serializable cluster-to-class mapping."""
    clusters_dict = {}
    for cluster_id in np.unique(y_pred):
        idx = np.where(y_pred == cluster_id)[0]
        names = y_true_names[idx].tolist()
        clusters_dict[str(int(cluster_id))] = names
    return clusters_dict

# ============================================================================
# 3. EXECUTION
# ============================================================================

def run_study():
    """Run one-shot sensitivity analysis for all configured experiments."""
    results_summary = []

    # Validate dataset
    dataset_path = resolve_repo_path(DATASET_CONFIG["dataset_path"])
    if not dataset_path.exists():
        logger.error(f"DATASET NOT FOUND: {DATASET_CONFIG['dataset_path']}")
        return

    df_full = pd.read_csv(dataset_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_out_dir = REPO_ROOT / "results/sensitivity_test"
    base_out_dir.mkdir(parents=True, exist_ok=True)

    for exp in EXPERIMENTS:
        logger.info(f"\n{'='*80}\nPROCESSING: {exp['id']}\n{'='*80}")
        
        dir_path = resolve_repo_path(exp["model_dir"])
        if not dir_path.exists():
            logger.error(f"DIRECTORY NOT FOUND: {dir_path}")
            continue

        for ratio in ONESHOT_RATIOS:
            logger.info(f"--> One-shot ratio: {ratio}")

            try:
                # 1. Dataset
                dataset_dict = create_dataset_for_ratio(df_full, ratio, exp['type'])
                label_encoder = dataset_dict['dataset'].get_label_encoder_for_evaluation()

                # 2. Load model
                model = load_experiment_model(exp, dataset_dict, device)

                # 3. Embeddings
                X, y_true_indices = extract_features_and_labels(model, dataset_dict, exp['type'])

                if X.shape[0] == 0:
                    logger.warning("Dataset is empty. Skipping.")
                    continue

                # 4. t-SNE
                tsne_kwargs = exp['tsne_params'].copy()
                if 'n_jobs' in tsne_kwargs:
                    del tsne_kwargs['n_jobs']

                tsne = TSNE(random_state=42, **tsne_kwargs)
                X_reduced = tsne.fit_transform(X)

                # 5. DBSCAN
                clusterer = DBSCAN(**exp['dbscan_params'])
                y_pred = clusterer.fit_predict(X_reduced)

                # 6. Metrics
                unique_clusters = set(y_pred)
                n_clusters = len(unique_clusters) - (1 if -1 in y_pred else 0)
                n_noise = list(y_pred).count(-1)

                ari = adjusted_rand_score(y_true_indices, y_pred) if len(unique_clusters) > 1 else 0.0
                sil = silhouette_score(X_reduced, y_pred) if len(unique_clusters) > 1 else -1.0

                logger.info(f"    ARI={ari:.4f}, Sil={sil:.4f}, Clusters={n_clusters}, Noise={n_noise}")

                # 7. JSON Output
                y_true_names = label_encoder.inverse_transform(y_true_indices)
                clusters_dict = build_clusters_output(y_pred, y_true_names)

                json_output = {
                    "experiment_id": exp['id'],
                    "ratio": ratio,
                    "metrics": {
                        "ari": float(ari),
                        "silhouette": float(sil),
                        "n_clusters": int(n_clusters),
                        "n_noise": int(n_noise),
                    },
                    "clusters": clusters_dict,
                }

                json_filename = f"clusters_{exp['id']}_ratio_{ratio}.json"
                with open(base_out_dir / json_filename, 'w', encoding='utf-8') as f:
                    json.dump(json_output, f, indent=4, ensure_ascii=False)

                results_summary.append({
                    "Model": exp['id'],
                    "OneShot_Ratio": ratio,
                    "ARI": ari,
                    "Silhouette": sil,
                    "Clusters": n_clusters,
                    "Noise_Samples": n_noise,
                    "Total_Samples": len(X),
                    "JSON_File": json_filename,
                })

                del model
                del dataset_dict
                torch.cuda.empty_cache()

            except Exception as e:
                logger.error(f"CRITICAL ERROR: {e}")
                import traceback
                traceback.print_exc()

    # Save CSV
    if results_summary:
        df_results = pd.DataFrame(results_summary)
        out_csv = base_out_dir / "oneshot_sensitivity_results.csv"
        df_results.to_csv(out_csv, index=False)
        logger.info(f"\nCompleted. CSV saved to: {out_csv}")
        print(df_results)

if __name__ == "__main__":
    run_study()

