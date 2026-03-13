"""Script to extract results for architecture scaling experiments.

Compares multiple architectures (ResNet50, ResNet101, ViT-B/32, ViT-B/16,
and CLIP variants) using consolidated finetuning and embeddings outputs.
"""

import hashlib
import json
import logging
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / 'results/models'
OUTPUT_DIR = REPO_ROOT / 'results/analysis/data/architecture_scaling'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def analyze_cluster_quality(cluster_analysis):
    """Analyze quality statistics for generated clusters.

    Args:
        cluster_analysis: Cluster analysis dictionary.

    Returns:
        Dictionary with summary cluster quality metrics.
    """
    if not cluster_analysis:
        return {}

    total_clusters = len(cluster_analysis)
    pure_clusters = []
    mixed_clusters = []
    cluster_sizes = []
    models_per_mixed_cluster = []

    for cluster_id, cluster_info in cluster_analysis.items():
        cluster_sizes.append(cluster_info['size'])

        if cluster_info.get('is_pure', False):
            pure_clusters.append(cluster_id)

        if cluster_info.get('is_mixed', False):
            mixed_clusters.append(cluster_id)
            models_per_mixed_cluster.append(cluster_info['n_unique_models'])

    n_pure = len(pure_clusters)
    n_mixed = len(mixed_clusters)
    pure_percentage = (n_pure / total_clusters * 100) if total_clusters > 0 else 0

    return {
        'n_clusters': total_clusters,
        'n_pure_clusters': n_pure,
        'n_mixed_clusters': n_mixed,
        'pure_percentage': pure_percentage,
    }


def extract_class_overlap_info(class_cluster_overlap):
    """Extract overlap metrics for classes appearing in multiple clusters.

    Args:
        class_cluster_overlap: List with class-cluster overlap entries.

    Returns:
        Dictionary with overlap statistics.
    """
    if not class_cluster_overlap:
        return {'n_overlapping_classes': 0}

    return {
        'n_overlapping_classes': len(class_cluster_overlap),
    }


CONFIGURATIONS = [
    {
        'model': 'resnet50',
        'architecture': 'ResNet50',
        'loss': 'multisimilarity',
        'views': 'front + rear',
        'type': 'metric_learning',
    },
    {
        'model': 'resnet101',
        'architecture': 'ResNet101',
        'loss': 'multisimilarity',
        'views': 'front + rear',
        'type': 'metric_learning',
    },
    {
        'model': 'vitb32',
        'architecture': 'ViT-B/32',
        'loss': 'contrastive',
        'views': 'front + rear',
        'type': 'metric_learning',
    },
    {
        'model': 'vitb16',
        'architecture': 'ViT-B/16',
        'loss': 'contrastive',
        'views': 'front + rear',
        'type': 'metric_learning',
    },
    {
        'model': 'clip',
        'architecture': 'CLIP-ViT-B/32',
        'loss': None,
        'views': 'front + rear',
        'type': 'clip_min_images',
        'min_images': 10,
    },
    {
        'model': 'clip',
        'architecture': 'CLIP-ViT-B/16',
        'loss': None,
        'views': 'front + rear',
        'type': 'clip_vitb16',
    },
]


def create_config_key(config_dict):
    """Create a unique key based on experiment configuration.

    Args:
        config_dict: Experiment configuration dictionary.

    Returns:
        Unique MD5 hash key.
    """
    key_params = ['views', 'model_name', 'objective', 'finetune_criterion']
    key_values = []

    for param in key_params:
        value = config_dict.get(param)
        if value is not None:
            if isinstance(value, list):
                value = '_'.join(map(str, value))
            key_values.append(f"{param}:{value}")

    config_str = '|'.join(sorted(key_values))
    return hashlib.md5(config_str.encode()).hexdigest()


def extract_finetuning_data(config):
    """Extract finetuning metrics for a single architecture configuration.

    Args:
        config: Configuration dictionary.

    Returns:
        Finetuning metrics dictionary, or ``None`` if results are missing.
    """
    model = config['model']
    loss = config['loss']
    views_str = config['views']
    config_type = config.get('type', 'metric_learning')

    if config_type == 'clip_min_images':
        min_images = config['min_images']
        base_path = RESULTS_DIR / 'clip' / 'min_images_study' / 'finetuning' / f'{min_images} images' / views_str
    elif config_type == 'clip_vitb16':
        base_path = RESULTS_DIR / 'clip_vitb16' / 'finetuning' / views_str
    else:
        base_path = RESULTS_DIR / model / 'metric learning' / loss / 'finetuning' / views_str

    results_file = base_path / 'results.json'

    if not results_file.exists():
        logger.debug(f"results.json not found: {results_file}")
        return None

    with open(results_file, 'r', encoding='utf-8') as f:
        results = json.load(f)

    json_config = results.get('config', {})
    config_key = create_config_key(json_config)

    if config_type in ['clip_min_images', 'clip_vitb16']:
        finetuned_eval = results.get('finetuned', {}).get('evaluation', {})
        best_val_recall1 = finetuned_eval.get('recall@1')
        best_val_recall3 = finetuned_eval.get('recall@3')
        best_val_recall5 = finetuned_eval.get('recall@5')

        finetuning_section = results.get('finetuning', {})
        training_history = finetuning_section.get('training_history', {})

        phase_keys = list(training_history.keys())
        total_epochs = 0
        for phase_key in phase_keys:
            phase_hist = training_history.get(phase_key, {})
            phase_epochs = len(phase_hist.get('train_loss', []))
            total_epochs += phase_epochs

        best_epoch = None
    else:
        finetuning_section = results.get('finetuning', {})
        training_history = finetuning_section.get('training_history', {})

        val_recall1_list = training_history.get('val_recall@1', [])
        val_recall3_list = training_history.get('val_recall@3', [])
        val_recall5_list = training_history.get('val_recall@5', [])
        total_epochs = len(val_recall1_list)

        if val_recall1_list:
            best_epoch = int(np.argmax(val_recall1_list))
            raw_recall1 = float(np.max(val_recall1_list))
            best_val_recall1 = raw_recall1 / 100.0 if raw_recall1 > 1 else raw_recall1
            raw_recall3 = val_recall3_list[best_epoch] if val_recall3_list else None
            best_val_recall3 = raw_recall3 / 100.0 if raw_recall3 is not None and raw_recall3 > 1 else raw_recall3
            raw_recall5 = val_recall5_list[best_epoch] if val_recall5_list else None
            best_val_recall5 = raw_recall5 / 100.0 if raw_recall5 is not None and raw_recall5 > 1 else raw_recall5
        else:
            best_epoch = None
            best_val_recall1 = None
            best_val_recall3 = None
            best_val_recall5 = None

    dataset_stats = results.get('dataset_stats', {})

    if config_type not in ['clip_min_images', 'clip_vitb16']:
        total_epochs = len(training_history.get('val_recall@1', []))

    start_time = results.get('start_time')
    end_time = results.get('end_time')
    duration_minutes = None
    if start_time and end_time:
        try:
            start_dt = datetime.fromisoformat(start_time)
            end_dt = datetime.fromisoformat(end_time)
            duration_minutes = (end_dt - start_dt).total_seconds() / 60.0
        except Exception:
            pass

    data = {
        'config_key': config_key,
        'model_name': config['model'],
        'architecture': config['architecture'],
        'criterion': config['loss'],
        'views': views_str,
        'best_epoch': best_epoch,
        'best_val_recall@1': best_val_recall1,
        'best_val_recall@3': best_val_recall3,
        'best_val_recall@5': best_val_recall5,
        'num_training_classes': dataset_stats.get('num_training_classes'),
        'train_samples': dataset_stats.get('splits', {}).get('train', {}).get('samples'),
        'val_samples': dataset_stats.get('splits', {}).get('val', {}).get('samples'),
        'duration_minutes': duration_minutes,
    }

    return data


def extract_embeddings_data(config):
    """Extract clustering/embeddings evaluation metrics for one configuration.

    Args:
        config: Configuration dictionary.

    Returns:
        Embeddings metrics dictionary, or ``None`` if results are missing.
    """
    model = config['model']
    loss = config['loss']
    views_str = config['views']
    config_type = config.get('type', 'metric_learning')

    if config_type == 'clip_min_images':
        min_images = config['min_images']
        base_path = RESULTS_DIR / 'clip' / 'min_images_study' / 'embeddings' / f'{min_images} images' / views_str
    elif config_type == 'clip_vitb16':
        base_path = RESULTS_DIR / 'clip_vitb16' / 'embeddings' / views_str
    else:
        base_path = RESULTS_DIR / model / 'metric learning' / loss / 'embeddings' / views_str

    results_file = base_path / 'results.json'

    if not results_file.exists():
        logger.debug(f"results.json not found: {results_file}")
        return None

    with open(results_file, 'r', encoding='utf-8') as f:
        results = json.load(f)

    json_config = results.get('config', {})
    config_key = create_config_key(json_config)

    finetuned_analysis = results.get('finetuned_analysis', {})
    finetuned_clustering = finetuned_analysis.get('clustering', {})
    finetuned_best_comparison = finetuned_clustering.get('comparison_df', [{}])

    if finetuned_best_comparison:
        finetuned_best = finetuned_best_comparison[0]
        finetuned_ari = finetuned_best.get('adjusted_rand_score')
        finetuned_nmi = finetuned_best.get('normalized_mutual_info')
        finetuned_purity = finetuned_best.get('purity')
    else:
        finetuned_ari = None
        finetuned_nmi = None
        finetuned_purity = None

    finetuned_visualization = finetuned_analysis.get('visualization', {})
    finetuned_cluster_analysis = finetuned_visualization.get('cluster_analysis', {})
    finetuned_cluster_quality = analyze_cluster_quality(finetuned_cluster_analysis)

    finetuned_overlap = finetuned_visualization.get('class_cluster_overlap', [])
    finetuned_overlap_info = extract_class_overlap_info(finetuned_overlap)

    data = {
        'config_key': config_key,
        'finetuned_ari': finetuned_ari,
        'finetuned_nmi': finetuned_nmi,
        'finetuned_purity': finetuned_purity,
        'finetuned_n_clusters': finetuned_cluster_quality.get('n_clusters'),
        'finetuned_n_pure_clusters': finetuned_cluster_quality.get('n_pure_clusters'),
        'finetuned_n_mixed_clusters': finetuned_cluster_quality.get('n_mixed_clusters'),
        'finetuned_pure_percentage': finetuned_cluster_quality.get('pure_percentage'),
        'finetuned_n_overlapping_classes': finetuned_overlap_info.get('n_overlapping_classes'),
    }

    return data


def main():
    """Extract architecture scaling results and write a consolidated CSV."""
    logger.info("=" * 80)
    logger.info("Starting extraction of results - Architecture Scaling Study")
    logger.info("=" * 80)

    logger.info("Extracting finetuning data")
    finetuning_data = []
    for config in CONFIGURATIONS:
        logger.debug(f"Processing {config['architecture']} ({config['loss']})")
        data = extract_finetuning_data(config)
        if data:
            finetuning_data.append(data)
        else:
            logger.debug(f"No data found for {config['architecture']}")

    df_finetuning = pd.DataFrame(finetuning_data)
    logger.info(f"Finetuning: {len(df_finetuning)} extracted configurations")

    logger.info("Extracting embeddings data")
    embeddings_data = []
    for config in CONFIGURATIONS:
        logger.debug(f"Processing embeddings: {config['architecture']} ({config['loss']})")
        data = extract_embeddings_data(config)
        if data:
            embeddings_data.append(data)
        else:
            logger.debug(f"No embeddings data found for {config['architecture']}")

    df_embeddings = pd.DataFrame(embeddings_data)
    logger.info(f"Embeddings: {len(df_embeddings)} extracted configurations")

    logger.info("Merging finetuning and embeddings datasets")
    df = pd.merge(df_finetuning, df_embeddings, on='config_key', how='outer', suffixes=('', '_emb'))

    logger.info(f"Merged dataset: {len(df)} configurations")

    output_file = OUTPUT_DIR / 'architecture_scaling_results.csv'
    df.to_csv(output_file, index=False)
    logger.info(f"Results saved to: {output_file}")

    logger.info("=" * 80)
    logger.info("Extraction summary")
    logger.info("=" * 80)

    logger.info(f"Total configurations: {len(df)}")
    logger.info(f"Total extracted columns: {len(df.columns)}")

    logger.info("Processed architectures:")
    for arch in df['architecture'].unique():
        count = len(df[df['architecture'] == arch])
        logger.info(f"  {arch}: {count} configurations")

    logger.info("=" * 80)
    logger.info("Extraction completed successfully")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
