"""Script to extract results from CLIP layers study experiments.

Processes finetuning and embeddings data, generating consolidated CSVs.
Compares the effect of unfreezing different numbers of layers in the vision
and text components of CLIP.
"""

import json
import hashlib
import logging
import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
CLIP_LAYERS_PATH = REPO_ROOT / 'results/models/clip/layers_study'
OUTPUT_DIR = REPO_ROOT / 'results/analysis/data/clip_layers'


def extract_clip_layer_info(config: Dict) -> Dict[str, Any]:
    """Extract CLIP layer-specific information from an experiment configuration.

    Args:
        config: Experiment configuration dictionary.

    Returns:
        Dict with keys ``component_type``, ``num_layers``, ``phase_name``,
        ``lr``, ``epochs_configured``, ``warmup_steps``, and ``patience``.
    """
    phases = config.get('clip_finetuning_phases', {})

    if not phases:
        return {
            'component_type': None,
            'num_layers': None,
            'phase_name': None,
            'lr': None,
            'epochs_configured': None,
            'warmup_steps': None,
            'patience': None
        }

    phase_name = list(phases.keys())[0]
    phase_config = phases[phase_name]

    component_type = phase_config.get('type')

    if component_type == 'vision':
        num_layers = phase_config.get('num_vision_layers')
    elif component_type == 'text':
        num_layers = phase_config.get('num_text_layers')
    else:
        num_layers = None

    return {
        'component_type': component_type,
        'num_layers': num_layers,
        'phase_name': phase_name,
        'lr': phase_config.get('lr'),
        'epochs_configured': phase_config.get('epochs'),
        'warmup_steps': phase_config.get('warmup_steps'),
        'patience': phase_config.get('early_stopping', {}).get('patience')
    }


def create_clip_config_key(config: Dict) -> str:
    """Create a unique MD5 hash key for a CLIP experiment configuration.

    Includes component type and number of layers to distinguish experiments.

    Args:
        config: Experiment configuration dictionary.

    Returns:
        MD5 hex-digest string.
    """
    layer_info = extract_clip_layer_info(config)
    key_params = [
        'views',
        'class_granularity',
        'min_images',
        'train_ratio',
        'val_ratio',
        'test_ratio',
        'seed',
        'image_size',
        'grayscale',
        'use_bbox',
        'augment',
        'model_name',
        'objective',
        'batch_size',
        'finetune_optimizer_type',
        'finetune_optimizer_weight_decay',
        'use_amp',
    ]

    key_values = []

    for param in key_params:
        value = config.get(param)
        if value is not None:
            if isinstance(value, list):
                value = '_'.join(map(str, value))
            key_values.append(f"{param}:{value}")

    key_values.append(f"component:{layer_info['component_type']}")
    key_values.append(f"layers:{layer_info['num_layers']}")

    config_str = '|'.join(sorted(key_values))
    return hashlib.md5(config_str.encode()).hexdigest()


def calculate_duration_minutes(start_time_str: str, end_time_str: str) -> Optional[float]:
    """Calculate the duration in minutes between two ISO timestamps.

    Args:
        start_time_str: Start timestamp in ISO format.
        end_time_str: End timestamp in ISO format.

    Returns:
        Duration in minutes, or ``None`` if timestamps are missing or invalid.
    """
    if not start_time_str or not end_time_str:
        return None

    try:
        start_dt = datetime.fromisoformat(start_time_str)
        end_dt = datetime.fromisoformat(end_time_str)
        return (end_dt - start_dt).total_seconds() / 60.0
    except Exception:
        return None


def load_json_safe(filepath: Path) -> Optional[Dict]:
    """Load a JSON file safely, returning ``None`` on any error.

    Args:
        filepath: Path to the JSON file.

    Returns:
        Parsed dictionary, or ``None`` if the file is missing or unreadable.
    """
    try:
        if not filepath.exists():
            return None

        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)

    except Exception as e:
        logger.error(f"Error loading {filepath}: {e}")
        return None


def extract_clip_finetuning_data(layers_study_path: Path) -> List[Dict]:
    """Extract finetuning data from a CLIP layers study.

    Args:
        layers_study_path: Path to ``results/models/clip/layers_study``.

    Returns:
        List of dicts with finetuning records.
    """
    finetune_data = []

    finetuning_path = layers_study_path / 'finetuning'

    if not finetuning_path.exists():
        logger.info(f"Directory not found: {finetuning_path}")
        return finetune_data

    for component_type in ['vision', 'text']:
        component_path = finetuning_path / component_type

        if not component_path.exists():
            logger.info(f"Directory not found: {component_path}")
            continue

        logger.info(f"Processing CLIP {component_type}")

        layer_folders = sorted([f for f in component_path.iterdir() if f.is_dir()])

        for layer_folder in layer_folders:
            results_file = layer_folder / 'results.json'

            if not results_file.exists():
                logger.info(f"results.json not found in {layer_folder}")
                continue

            logger.debug(f"Reading: {component_type}/{layer_folder.name}")

            results = load_json_safe(results_file)
            if results is None:
                continue

            config = results.get('config', {})
            config_key = create_clip_config_key(config)
            layer_info = extract_clip_layer_info(config)

            experiment_name = results.get('experiment_name', layer_folder.name)
            start_time = results.get('start_time')
            end_time = results.get('end_time')
            duration_minutes = calculate_duration_minutes(start_time, end_time)
            
            dataset_stats = results.get('dataset_stats', {})

            baseline = results.get('baseline', {}).get('evaluation', {})
            baseline_recall1 = baseline.get('recall@1', None)
            baseline_recall3 = baseline.get('recall@3', None)
            baseline_recall5 = baseline.get('recall@5', None)

            finetuning = results.get('finetuning', {})
            training_history = finetuning.get('training_history', {})

            phase_name = layer_info['phase_name']
            if phase_name and phase_name in training_history:
                history = training_history[phase_name]
            else:
                phase_keys = [k for k in training_history.keys() if k.startswith('phase_')]
                if phase_keys:
                    history = training_history[phase_keys[0]]
                else:
                    history = training_history

            train_loss_list = history.get('train_loss', [])
            val_loss_list = history.get('val_loss', [])
            val_recall1_list = history.get('val_recall@1', [])
            val_recall3_list = history.get('val_recall@3', [])
            val_recall5_list = history.get('val_recall@5', [])

            total_epochs = len(train_loss_list)

            if val_recall1_list:
                best_epoch = int(np.argmax(val_recall1_list))
                best_val_recall1 = float(np.max(val_recall1_list))
                best_val_recall3 = val_recall3_list[best_epoch] if val_recall3_list and best_epoch < len(val_recall3_list) else None
                best_val_recall5 = val_recall5_list[best_epoch] if val_recall5_list and best_epoch < len(val_recall5_list) else None
                best_val_loss = val_loss_list[best_epoch] if val_loss_list and best_epoch < len(val_loss_list) else None
            else:
                best_epoch = None
                best_val_recall1 = None
                best_val_recall3 = None
                best_val_recall5 = None
                best_val_loss = None

            final_train_loss = train_loss_list[-1] if train_loss_list else None
            final_val_loss = val_loss_list[-1] if val_loss_list else None

            finetuned = results.get('finetuned', {}).get('evaluation', {})
            finetuned_recall1 = finetuned.get('recall@1', None)
            finetuned_recall3 = finetuned.get('recall@3', None)
            finetuned_recall5 = finetuned.get('recall@5', None)

            summary = results.get('summary', {})
            recall1_improvement = summary.get('recall@1_improvement', None)
            recall3_improvement = summary.get('recall@3_improvement', None)
            recall5_improvement = summary.get('recall@5_improvement', None)

            if recall1_improvement is None and baseline_recall1 is not None and finetuned_recall1 is not None:
                recall1_improvement = finetuned_recall1 - baseline_recall1

            if recall3_improvement is None and baseline_recall3 is not None and finetuned_recall3 is not None:
                recall3_improvement = finetuned_recall3 - baseline_recall3

            if recall5_improvement is None and baseline_recall5 is not None and finetuned_recall5 is not None:
                recall5_improvement = finetuned_recall5 - baseline_recall5
            
            record = {
                'config_key': config_key,
                'experiment_name': experiment_name,
                'timestamp': start_time,

                # CLIP-specific configuration
                'model_name': config.get('model_name', 'clip-vit-base-patch32'),
                'objective': 'CLIP',
                'component_type': layer_info['component_type'],
                'num_layers': layer_info['num_layers'],
                'phase_name': layer_info['phase_name'],
                'views': config.get('views', []),
                'views_str': ' + '.join(config.get('views', [])),
                'num_views': len(config.get('views', [])),
                
                # Hyperparameters
                'batch_size': config.get('batch_size'),
                'lr': layer_info['lr'],
                'weight_decay': config.get('finetune_optimizer_weight_decay'),
                'optimizer': config.get('finetune_optimizer_type'),
                'use_amp': config.get('use_amp'),
                'warmup_steps': layer_info['warmup_steps'],
                'patience': layer_info['patience'],
                'epochs_configured': layer_info['epochs_configured'],
                'seed': config.get('seed'),
                
                # Dataset
                'num_training_classes': dataset_stats.get('num_training_classes'),
                'num_total_classes': dataset_stats.get('num_total_classes'),
                'num_oneshot_classes': dataset_stats.get('num_oneshot_classes'),
                'train_samples': dataset_stats.get('splits', {}).get('train', {}).get('samples'),
                'val_samples': dataset_stats.get('splits', {}).get('val', {}).get('samples'),
                'test_samples': dataset_stats.get('splits', {}).get('test', {}).get('samples'),
                
                # Timing
                'start_time': start_time,
                'end_time': end_time,
                'duration_minutes': duration_minutes,
                
                # Training
                'total_epochs': total_epochs,
                'best_epoch': best_epoch,
                
                # Baseline (recall only, CLIP does not use accuracy)
                'baseline_recall@1': baseline_recall1,
                'baseline_recall@3': baseline_recall3,
                'baseline_recall@5': baseline_recall5,
                
                # Best metrics
                'best_val_loss': best_val_loss,
                'best_val_recall@1': best_val_recall1,
                'best_val_recall@3': best_val_recall3,
                'best_val_recall@5': best_val_recall5,
                
                # Final metrics
                'final_train_loss': final_train_loss,
                'final_val_loss': final_val_loss,
                
                # Finetuned
                'finetuned_recall@1': finetuned_recall1,
                'finetuned_recall@3': finetuned_recall3,
                'finetuned_recall@5': finetuned_recall5,
                
                # Improvements
                'recall@1_improvement': recall1_improvement,
                'recall@3_improvement': recall3_improvement,
                'recall@5_improvement': recall5_improvement,
            }
            
            finetune_data.append(record)

    return finetune_data


def analyze_cluster_quality(cluster_analysis: Dict) -> Dict[str, Any]:
    """Analyse the quality of generated clusters.

    Args:
        cluster_analysis: Dict mapping cluster IDs to their analysis info.

    Returns:
        Dict with cluster quality metrics.
    """
    if not cluster_analysis:
        return {}

    total_clusters = len(cluster_analysis)
    pure_clusters = []
    mixed_clusters = []
    dominant_clusters = []
    cluster_sizes = []
    models_per_mixed_cluster = []

    for cluster_id, cluster_info in cluster_analysis.items():
        cluster_sizes.append(cluster_info['size'])

        if cluster_info.get('is_pure', False):
            pure_clusters.append(cluster_id)

        if cluster_info.get('is_mixed', False):
            mixed_clusters.append(cluster_id)
            models_per_mixed_cluster.append(cluster_info['n_unique_models'])

        if cluster_info.get('is_dominant', False):
            dominant_clusters.append(cluster_id)

    n_pure = len(pure_clusters)
    n_mixed = len(mixed_clusters)
    n_dominant = len(dominant_clusters)

    pure_percentage = (n_pure / total_clusters * 100) if total_clusters > 0 else 0
    mixed_percentage = (n_mixed / total_clusters * 100) if total_clusters > 0 else 0
    dominant_percentage = (n_dominant / total_clusters * 100) if total_clusters > 0 else 0

    avg_models_per_mixed = np.mean(models_per_mixed_cluster) if models_per_mixed_cluster else 0
    avg_cluster_size = np.mean(cluster_sizes) if cluster_sizes else 0
    median_cluster_size = np.median(cluster_sizes) if cluster_sizes else 0
    
    return {
        'n_clusters': total_clusters,
        'n_pure_clusters': n_pure,
        'n_mixed_clusters': n_mixed,
        'n_dominant_clusters': n_dominant,
        'pure_percentage': pure_percentage,
        'mixed_percentage': mixed_percentage,
        'dominant_percentage': dominant_percentage,
        'avg_models_per_mixed_cluster': avg_models_per_mixed,
        'avg_cluster_size': avg_cluster_size,
        'median_cluster_size': median_cluster_size,
        'min_cluster_size': int(np.min(cluster_sizes)) if cluster_sizes else 0,
        'max_cluster_size': int(np.max(cluster_sizes)) if cluster_sizes else 0,
    }


def extract_class_overlap_info(class_cluster_overlap: List[Dict]) -> Dict[str, Any]:
    """Extract metrics about classes that span multiple clusters.

    Args:
        class_cluster_overlap: List of dicts with per-class overlap information.

    Returns:
        Dict with overlap metrics.
    """
    if not class_cluster_overlap:
        return {
            'n_overlapping_classes': 0,
            'avg_clusters_per_class': 0,
            'max_clusters_per_class': 0,
            'overlapping_class_names': []
        }

    n_overlapping = len(class_cluster_overlap)
    clusters_per_class = [item['n_clusters'] for item in class_cluster_overlap]

    avg_clusters = np.mean(clusters_per_class) if clusters_per_class else 0
    max_clusters = np.max(clusters_per_class) if clusters_per_class else 0

    sorted_classes = sorted(class_cluster_overlap, key=lambda x: x['n_clusters'], reverse=True)
    top_classes = [
        f"{item['class_name']}({item['n_clusters']})"
        for item in sorted_classes[:10]
    ]

    return {
        'n_overlapping_classes': n_overlapping,
        'avg_clusters_per_overlapping_class': avg_clusters,
        'max_clusters_per_overlapping_class': int(max_clusters),
        'top_overlapping_classes': '; '.join(top_classes)
    }


def extract_clip_embeddings_data(layers_study_path: Path) -> List[Dict]:
    """Extract embeddings data from a CLIP layers study.

    Args:
        layers_study_path: Path to ``results/models/clip/layers_study``.

    Returns:
        List of dicts with embeddings records.
    """
    embeddings_data = []

    embeddings_path = layers_study_path / 'embeddings'
    if not embeddings_path.exists():
        logger.info(f"Directory not found: {embeddings_path}")
        return embeddings_data

    for component_type in ['vision', 'text']:
        component_path = embeddings_path / component_type

        if not component_path.exists():
            logger.info(f"Directory not found: {component_path}")
            continue

        logger.info(f"Processing CLIP {component_type} embeddings")

        layer_folders = sorted([f for f in component_path.iterdir() if f.is_dir()])

        for layer_folder in layer_folders:
            results_file = layer_folder / 'results.json'

            if not results_file.exists():
                logger.info(f"results.json not found in {layer_folder}")
                continue

            logger.debug(f"Reading: {component_type}/{layer_folder.name}/embeddings")

            results = load_json_safe(results_file)
            if results is None:
                continue

            config = results.get('config', {})
            config_key = create_clip_config_key(config)
            layer_info = extract_clip_layer_info(config)

            experiment_name = results.get('experiment_name', layer_folder.name)
            start_time = results.get('start_time')
            end_time = results.get('end_time')
            duration_minutes = calculate_duration_minutes(start_time, end_time)

            finetuned_analysis = results.get('finetuned_analysis', {})
            finetuned_reduction = finetuned_analysis.get('reduction', {})
            finetuned_best_reducer = finetuned_reduction.get('best_method')
            finetuned_best_shape = finetuned_reduction.get('best_embeddings_shape')

            finetuned_clustering = finetuned_analysis.get('clustering', {})

            finetuned_best_comparison = finetuned_clustering.get('comparison_df', [{}])
            if finetuned_best_comparison:
                finetuned_best = finetuned_best_comparison[0]
                finetuned_clusterer = finetuned_best.get('method')
                finetuned_ari = finetuned_best.get('adjusted_rand_score')
                finetuned_nmi = finetuned_best.get('normalized_mutual_info')
                finetuned_purity = finetuned_best.get('purity')
                finetuned_silhouette = finetuned_best.get('silhouette_score')
                finetuned_n_clusters = finetuned_best.get('n_clusters')
            else:
                finetuned_clusterer = None
                finetuned_ari = None
                finetuned_nmi = None
                finetuned_purity = None
                finetuned_silhouette = None
                finetuned_n_clusters = None

            finetuned_visualization = finetuned_analysis.get('visualization', {})
            finetuned_cluster_analysis = finetuned_visualization.get('cluster_analysis', {})
            finetuned_cluster_quality = analyze_cluster_quality(finetuned_cluster_analysis)

            finetuned_overlap = finetuned_visualization.get('class_cluster_overlap', [])
            finetuned_overlap_info = extract_class_overlap_info(finetuned_overlap)
            
            record = {
                'config_key': config_key,
                'experiment_name': experiment_name,
                'timestamp': start_time,

                # Configuration
                'model_name': 'clip-vit-base-patch32',
                'objective': 'CLIP',
                'component_type': layer_info['component_type'],
                'num_layers': layer_info['num_layers'],
                'views_str': ' + '.join(config.get('views', [])),
                
                # Embeddings shapes
                'finetuned_embedding_shape': finetuned_best_shape,
                
                # Reduction methods
                'finetuned_reducer': finetuned_best_reducer,
                
                # Clustering methods
                'finetuned_clusterer': finetuned_clusterer,
                
                # Finetuned clustering metrics
                'finetuned_ari': finetuned_ari,
                'finetuned_nmi': finetuned_nmi,
                'finetuned_purity': finetuned_purity,
                'finetuned_silhouette': finetuned_silhouette,
                'finetuned_n_clusters': finetuned_n_clusters,
                
                # Finetuned cluster quality
                'finetuned_n_pure_clusters': finetuned_cluster_quality.get('n_pure_clusters'),
                'finetuned_pure_percentage': finetuned_cluster_quality.get('pure_percentage'),
                'finetuned_n_mixed_clusters': finetuned_cluster_quality.get('n_mixed_clusters'),
                'finetuned_avg_models_per_mixed': finetuned_cluster_quality.get('avg_models_per_mixed_cluster'),
                'finetuned_avg_cluster_size': finetuned_cluster_quality.get('avg_cluster_size'),
                'finetuned_median_cluster_size': finetuned_cluster_quality.get('median_cluster_size'),
                'finetuned_min_cluster_size': finetuned_cluster_quality.get('min_cluster_size'),
                'finetuned_max_cluster_size': finetuned_cluster_quality.get('max_cluster_size'),
                
                # Class overlap (finetuned)
                'finetuned_n_overlapping_classes': finetuned_overlap_info.get('n_overlapping_classes'),
                'finetuned_avg_clusters_per_class': finetuned_overlap_info.get('avg_clusters_per_overlapping_class'),
                'finetuned_max_clusters_per_class': finetuned_overlap_info.get('max_clusters_per_overlapping_class'),
                'finetuned_top_overlapping': finetuned_overlap_info.get('top_overlapping_classes'),
            }
            
            embeddings_data.append(record)

    return embeddings_data


def main():
    """Extract all CLIP layers study results and save consolidated CSV files.

    Processes experiments where the number of unfrozen layers varies across the
    vision and text components of CLIP.  Extracts finetuning and embeddings data,
    merges results by config key, and writes CSV files to
    ``results/analysis/data/clip_layers/``.
    """
    if not CLIP_LAYERS_PATH.exists():
        logger.error(f"Directory not found: {CLIP_LAYERS_PATH}")
        return

    logger.info("=" * 80)
    logger.info("Starting extraction of results - CLIP Layers Study")
    logger.info("=" * 80)

    logger.info("Extracting finetuning data")
    finetuning_data = extract_clip_finetuning_data(CLIP_LAYERS_PATH)

    logger.info("Extracting embeddings data")
    embeddings_data = extract_clip_embeddings_data(CLIP_LAYERS_PATH)

    logger.info("Building DataFrames")
    df_finetuning = pd.DataFrame(finetuning_data)
    df_embeddings = pd.DataFrame(embeddings_data)

    logger.info(f"Finetuning: {len(df_finetuning)} records")
    logger.info(f"Embeddings: {len(df_embeddings)} records")

    if df_finetuning.empty and df_embeddings.empty:
        logger.error("No data found to process")
        return

    logger.info("Merging finetuning and embeddings results")
    if not df_finetuning.empty and not df_embeddings.empty:
        df_combined = pd.merge(
            df_finetuning,
            df_embeddings,
            on='config_key',
            how='outer',
            suffixes=('_ft', '_emb')
        )

        for col in ['model_name', 'objective', 'component_type', 'num_layers', 'views_str']:
            col_ft = f'{col}_ft'
            col_emb = f'{col}_emb'
            if col_ft in df_combined.columns and col_emb in df_combined.columns:
                df_combined[col] = df_combined[col_ft].combine_first(df_combined[col_emb])
                df_combined = df_combined.drop(columns=[col_ft, col_emb])

        logger.info(f"Combined: {len(df_combined)} unique configurations")

    elif not df_finetuning.empty:
        df_combined = df_finetuning
        logger.info("Using finetuning data only")
    else:
        df_combined = df_embeddings
        logger.info("Using embeddings data only")

    if 'component_type' in df_combined.columns and 'num_layers' in df_combined.columns:
        df_combined = df_combined.sort_values(['component_type', 'num_layers'])
        logger.info("Sorted by component and number of layers")

    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

    combined_csv = OUTPUT_DIR / 'clip_layers_results.csv'
    df_combined.to_csv(combined_csv, index=False)
    logger.info(f"Results saved to: {combined_csv}")
    logger.info(f"Total configurations: {len(df_combined)}")

    if not df_finetuning.empty:
        ft_csv = OUTPUT_DIR / 'clip_layers_finetuning_results.csv'
        df_finetuning.to_csv(ft_csv, index=False)
        logger.info(f"Finetuning saved to: {ft_csv}")

    if not df_embeddings.empty:
        emb_csv = OUTPUT_DIR / 'clip_layers_embeddings_results.csv'
        df_embeddings.to_csv(emb_csv, index=False)
        logger.info(f"Embeddings saved to: {emb_csv}")

    logger.info("=" * 80)
    logger.info("Extraction summary")
    logger.info("=" * 80)

    if not df_combined.empty:
        if 'component_type' in df_combined.columns:
            logger.info("Components processed:")
            for comp in df_combined['component_type'].unique():
                if pd.notna(comp):
                    count = len(df_combined[df_combined['component_type'] == comp])
                    logger.info(f"  {comp}: {count} configurations")

        if 'num_layers' in df_combined.columns:
            layers = df_combined['num_layers'].dropna().unique()
            if len(layers) > 0:
                logger.info(f"Layer range: {int(min(layers))} to {int(max(layers))}")
                logger.info(f"Total distinct configurations: {len(layers)}")

    logger.info("=" * 80)
    logger.info("Extraction completed successfully")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
