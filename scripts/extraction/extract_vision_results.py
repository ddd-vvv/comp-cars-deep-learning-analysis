"""
Script to extract results from vision model experiments (ResNet50 and ViT-B/32).
Processes fine-tuning and embeddings directories, generating consolidated CSVs
with detailed per-experiment information.
"""

import json
import hashlib
import logging
import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]


def create_config_key(config: Dict) -> str:
    """Create a unique MD5 hash key from an experiment configuration.

    Used to pair fine-tuning results with their corresponding embeddings.

    Args:
        config: Experiment configuration dictionary.

    Returns:
        Unique MD5 hex digest string.
    """
    # Parameters included in the config hash
    keys = [
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
        'model_type',
        'model_name',
        'weights',
        'objective',
        'sampling_strategy',
        'batch_size',
        'P',
        'K',
        'finetune_criterion',
        'finetune_optimizer_type',
        'finetune_backbone_lr',
        'finetune_head_lr',
        'finetune_optimizer_weight_decay',
        'use_scheduler',
        'scheduler_type',
        'warmup_epochs',
        'use_early_stopping',
        'patience',
        'use_amp',
    ]

    key_values = []

    for param in keys:
        value = config.get(param)
        if value is not None:
            if isinstance(value, list):
                value = '_'.join(map(str, value))
            key_values.append(f"{param}:{value}")

    config_str = '|'.join(sorted(key_values))
    return hashlib.md5(config_str.encode()).hexdigest()


def calculate_duration_minutes(start_time_str: str, end_time_str: str) -> Optional[float]:
    """Calculate the duration in minutes between two ISO timestamps.

    Args:
        start_time_str: Start timestamp in ISO format.
        end_time_str: End timestamp in ISO format.

    Returns:
        Duration in minutes, or None if the timestamps are missing or invalid.
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
    """Load a JSON file, returning None on any error.

    Args:
        filepath: Path to the JSON file.

    Returns:
        Parsed dictionary, or None if the file does not exist or cannot be parsed.
    """
    try:
        if not filepath.exists():
            return None
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading {filepath}: {e}")
        return None


def extract_finetuning_data(model_base_path: Path, model_name: str, objective: str, 
                           loss_function: Optional[str] = None) -> List[Dict]:
    """Extract fine-tuning experiment data for a specific model.

    Args:
        model_base_path: Base path for the model (e.g. results/models/resnet50).
        model_name: Model name ('resnet50', 'vit_b_32').
        objective: Training objective ('classification' or 'metric_learning').
        loss_function: Loss function name (required for metric_learning).

    Returns:
        List of record dictionaries with fine-tuning data.
    """
    finetune_data = []

    if objective == 'classification':
        finetuning_path = model_base_path / 'classification' / 'finetuning'
    else:
        if not loss_function:
            return finetune_data
        finetuning_path = model_base_path / 'metric learning' / loss_function / 'finetuning'

    if not finetuning_path.exists():
        logger.info(f"Directory not found: {finetuning_path}")
        return []

    view_folders = [f for f in finetuning_path.iterdir() if f.is_dir()]

    for view_folder in view_folders:
        views_str = view_folder.name
        results_file = view_folder / 'results.json'

        if not results_file.exists():
            logger.info(f"results.json not found in {view_folder}")
            continue

        logger.debug(f"Reading: {model_name}/{objective}/{loss_function or 'classification'}/{views_str}")

        results = load_json_safe(results_file)
        if results is None:
            continue

        # Generate unique configuration hash
        config = results.get('config', {})
        config_key = create_config_key(config)

        experiment_name = results.get('experiment_name', view_folder.name)
        start_time = results.get('start_time')
        end_time = results.get('end_time')
        duration_minutes = calculate_duration_minutes(start_time, end_time)

        dataset_stats = results.get('dataset_stats', {})

        baseline = results.get('baseline', {}).get('evaluation', {})
        baseline_accuracy = baseline.get('accuracy', None)
        baseline_recall1 = baseline.get('recall@1', None)
        baseline_recall3 = baseline.get('recall@3', None)
        baseline_recall5 = baseline.get('recall@5', None)

        finetuning = results.get('finetuning', {})
        training_history = finetuning.get('training_history', {})

        train_loss_list = training_history.get('train_loss', [])
        val_loss_list = training_history.get('val_loss', [])
        val_accuracy_list = training_history.get('val_accuracy', [])
        val_recall1_list = training_history.get('val_recall@1', [])
        val_recall3_list = training_history.get('val_recall@3', [])
        val_recall5_list = training_history.get('val_recall@5', [])

        total_epochs = finetuning.get('total_epochs', len(train_loss_list))
        
        if objective == 'classification':
            if val_accuracy_list:
                best_epoch = int(np.argmax(val_accuracy_list))
                best_val_accuracy = float(np.max(val_accuracy_list))
                best_val_loss = val_loss_list[best_epoch] if val_loss_list else None
                best_val_recall1 = val_recall1_list[best_epoch] if val_recall1_list else None
                best_val_recall3 = val_recall3_list[best_epoch] if val_recall3_list else None
                best_val_recall5 = val_recall5_list[best_epoch] if val_recall5_list else None
            else:
                best_epoch = None
                best_val_accuracy = None
                best_val_loss = None
                best_val_recall1 = None
                best_val_recall3 = None
                best_val_recall5 = None
        else:
            if val_recall1_list:
                best_epoch = int(np.argmax(val_recall1_list))
                # Normalise from percentage (0-100) to decimal (0-1) if needed
                raw_recall1 = float(np.max(val_recall1_list))
                best_val_recall1 = raw_recall1 / 100.0 if raw_recall1 > 1 else raw_recall1
                raw_recall3 = val_recall3_list[best_epoch] if val_recall3_list else None
                best_val_recall3 = raw_recall3 / 100.0 if raw_recall3 is not None and raw_recall3 > 1 else raw_recall3
                raw_recall5 = val_recall5_list[best_epoch] if val_recall5_list else None
                best_val_recall5 = raw_recall5 / 100.0 if raw_recall5 is not None and raw_recall5 > 1 else raw_recall5
                best_val_loss = val_loss_list[best_epoch] if val_loss_list else None
                best_val_accuracy = val_accuracy_list[best_epoch] if val_accuracy_list else None
            else:
                best_epoch = None
                best_val_recall1 = None
                best_val_recall3 = None
                best_val_recall5 = None
                best_val_loss = None
                best_val_accuracy = None
        
        if best_epoch is not None:
            finetuned_accuracy = val_accuracy_list[best_epoch] if val_accuracy_list and len(val_accuracy_list) > best_epoch else None
            raw_recall1 = val_recall1_list[best_epoch] if val_recall1_list and len(val_recall1_list) > best_epoch else None
            finetuned_recall1 = raw_recall1 / 100.0 if raw_recall1 is not None and raw_recall1 > 1 else raw_recall1
            raw_recall3 = val_recall3_list[best_epoch] if val_recall3_list and len(val_recall3_list) > best_epoch else None
            finetuned_recall3 = raw_recall3 / 100.0 if raw_recall3 is not None and raw_recall3 > 1 else raw_recall3
            raw_recall5 = val_recall5_list[best_epoch] if val_recall5_list and len(val_recall5_list) > best_epoch else None
            finetuned_recall5 = raw_recall5 / 100.0 if raw_recall5 is not None and raw_recall5 > 1 else raw_recall5
            final_train_loss = train_loss_list[best_epoch] if train_loss_list and len(train_loss_list) > best_epoch else None
            final_val_loss = val_loss_list[best_epoch] if val_loss_list and len(val_loss_list) > best_epoch else None
        else:
            finetuned_accuracy = None
            finetuned_recall1 = None
            finetuned_recall3 = None
            finetuned_recall5 = None
            final_train_loss = None
            final_val_loss = None

        # Compute improvements: improvement = finetuned - baseline
        accuracy_improvement = None
        if baseline_accuracy is not None and finetuned_accuracy is not None:
            accuracy_improvement = finetuned_accuracy - baseline_accuracy
        
        recall1_improvement = None
        if baseline_recall1 is not None and finetuned_recall1 is not None:
            recall1_improvement = finetuned_recall1 - baseline_recall1

        recall3_improvement = None
        if baseline_recall3 is not None and finetuned_recall3 is not None:
            recall3_improvement = finetuned_recall3 - baseline_recall3

        recall5_improvement = None
        if baseline_recall5 is not None and finetuned_recall5 is not None:
            recall5_improvement = finetuned_recall5 - baseline_recall5

        record = {
            # Metadata
            'config_key': config_key,
            'experiment_name': experiment_name,
            'timestamp': start_time,

            # Configuration
            'model_name': model_name,
            'objective': objective,
            'criterion': loss_function if loss_function else 'CrossEntropyLoss',
            'views': views_str,
            'views_list': config.get('views', []),
            'num_views': len(config.get('views', [])),
            'batch_size': config.get('batch_size'),
            'backbone_lr': config.get('finetune_backbone_lr'),
            'head_lr': config.get('finetune_head_lr'),
            'weight_decay': config.get('finetune_optimizer_weight_decay'),
            'optimizer': config.get('finetune_optimizer_type'),
            'scheduler': config.get('scheduler_type'),
            'use_amp': config.get('use_amp'),
            'patience': config.get('patience'),
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

            # Baseline
            'baseline_accuracy': baseline_accuracy,
            'baseline_recall@1': baseline_recall1,
            'baseline_recall@3': baseline_recall3,
            'baseline_recall@5': baseline_recall5,

            # Best metrics
            'best_val_loss': best_val_loss,
            'best_val_accuracy': best_val_accuracy,
            'best_val_recall@1': best_val_recall1,
            'best_val_recall@3': best_val_recall3,
            'best_val_recall@5': best_val_recall5,

            # Final metrics at best epoch
            'final_train_loss': final_train_loss,
            'final_val_loss': final_val_loss,

            # Fine-tuned values
            'finetuned_accuracy': finetuned_accuracy,
            'finetuned_recall@1': finetuned_recall1,
            'finetuned_recall@3': finetuned_recall3,
            'finetuned_recall@5': finetuned_recall5,

            # Improvements (finetuned - baseline)
            'accuracy_improvement': accuracy_improvement,
            'recall@1_improvement': recall1_improvement,
            'recall@3_improvement': recall3_improvement,
            'recall@5_improvement': recall5_improvement,
        }

        finetune_data.append(record)

    return finetune_data


def analyze_cluster_quality(cluster_analysis: Dict) -> Dict[str, Any]:
    """Compute summary quality metrics from a cluster analysis dictionary.

    Args:
        cluster_analysis: Per-cluster analysis mapping from the results JSON.

    Returns:
        Dictionary with aggregate cluster quality statistics.
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
    """Extract metrics for classes appearing in multiple clusters.

    Args:
        class_cluster_overlap: List of dictionaries with class-cluster overlap info.

    Returns:
        Dictionary with class overlap statistics.
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


def extract_embeddings_data(model_base_path: Path, model_name: str, objective: str,
                           loss_function: Optional[str] = None) -> List[Dict]:
    """Extract embeddings experiment data for a specific model.

    Args:
        model_base_path: Base path of the model.
        model_name: Model name.
        objective: ``'classification'`` or ``'metric_learning'``.
        loss_function: Loss function name (metric learning only).

    Returns:
        List of dicts with embeddings data.
    """
    embeddings_data = []
    
    if objective == 'classification':
        embeddings_path = model_base_path / 'classification' / 'embeddings'
    else:
        if not loss_function:
            return []
        embeddings_path = model_base_path / 'metric learning' / loss_function / 'embeddings'

    if not embeddings_path.exists():
        logger.info(f"Directory not found: {embeddings_path}")
        return []
    
    view_folders = [f for f in embeddings_path.iterdir() if f.is_dir()]

    for view_folder in view_folders:
        views_str = view_folder.name
        results_file = view_folder / 'results.json'

        if not results_file.exists():
            logger.info(f"results.json not found in {view_folder}")
            continue

        logger.debug(f"Reading: {model_name}/{objective}/{loss_function or 'classification'}/{views_str}/embeddings")

        results = load_json_safe(results_file)
        if results is None:
            continue

        config = results.get('config', {})
        config_key = create_config_key(config)

        experiment_name = results.get('experiment_name', view_folder.name)
        start_time = results.get('start_time')

        embeddings_info = results.get('embeddings_info', {})
        baseline_shape = embeddings_info.get('baseline', {}).get('shape', [])
        finetuned_shape = embeddings_info.get('finetuned', {}).get('shape', [])
        
        baseline_analysis = results.get('baseline_analysis', {})
        baseline_best_reducer = baseline_analysis.get('reduction', {}).get('best_method')
        baseline_best_shape = baseline_analysis.get('reduction', {}).get('best_embeddings_shape')

        baseline_clustering = baseline_analysis.get('clustering', {})

        baseline_best_comparison = baseline_clustering.get('comparison_df', [{}])
        if baseline_best_comparison:
            baseline_best = baseline_best_comparison[0]
            baseline_clusterer = baseline_best.get('method')
            baseline_ari = baseline_best.get('adjusted_rand_score')
            baseline_nmi = baseline_best.get('normalized_mutual_info')
            baseline_purity = baseline_best.get('purity')
            baseline_silhouette = baseline_best.get('silhouette_score')
            baseline_n_clusters = baseline_best.get('n_clusters')
        else:
            baseline_clusterer = None
            baseline_ari = None
            baseline_nmi = None
            baseline_purity = None
            baseline_silhouette = None
            baseline_n_clusters = None

        baseline_visualization = baseline_analysis.get('visualization', {})
        baseline_cluster_analysis = baseline_visualization.get('cluster_analysis', {})
        baseline_cluster_quality = analyze_cluster_quality(baseline_cluster_analysis)

        finetuned_analysis = results.get('finetuned_analysis', {})
        finetuned_best_reducer = finetuned_analysis.get('reduction', {}).get('best_method')
        finetuned_best_shape = finetuned_analysis.get('reduction', {}).get('best_embeddings_shape')

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
        
        baseline_overlap = baseline_visualization.get('class_cluster_overlap', [])
        baseline_overlap_info = extract_class_overlap_info(baseline_overlap)
        finetuned_overlap = finetuned_visualization.get('class_cluster_overlap', [])
        finetuned_overlap_info = extract_class_overlap_info(finetuned_overlap)

        comparison = results.get('comparison', {})
        performance = comparison.get('performance_improvement', {})

        ari_improvement = performance.get('ari_improvement')

        if ari_improvement is None and baseline_ari is not None and finetuned_ari is not None:
            ari_improvement = finetuned_ari - baseline_ari

        record = {
            'config_key': config_key,
            'experiment_name': experiment_name,
            'timestamp': start_time,
            
            # Configuration (minimal — merged with finetuning data later)
            'model_name': model_name,
            'objective': objective,
            'criterion': loss_function if loss_function else 'CrossEntropyLoss',
            'views': views_str,
            
            # Timing
            'embeddings_start_time': start_time,
            
            # Embeddings shapes
            'baseline_embedding_shape': baseline_shape,
            'finetuned_embedding_shape': finetuned_shape,
            
            # Reduction methods
            'baseline_reducer': baseline_best_reducer,
            'baseline_reduced_shape': baseline_best_shape,
            'finetuned_reducer': finetuned_best_reducer,
            'finetuned_reduced_shape': finetuned_best_shape,
            
            # Clustering methods
            'baseline_clusterer': baseline_clusterer,
            'finetuned_clusterer': finetuned_clusterer,
            
            # Baseline clustering metrics
            'baseline_ari': baseline_ari,
            'baseline_nmi': baseline_nmi,
            'baseline_purity': baseline_purity,
            'baseline_silhouette': baseline_silhouette,
            'baseline_n_clusters': baseline_n_clusters,
            
            # Finetuned clustering metrics
            'finetuned_ari': finetuned_ari,
            'finetuned_nmi': finetuned_nmi,
            'finetuned_purity': finetuned_purity,
            'finetuned_silhouette': finetuned_silhouette,
            'finetuned_n_clusters': finetuned_n_clusters,
            
            # Improvements
            'ari_improvement': ari_improvement,
            'nmi_improvement': (finetuned_nmi - baseline_nmi) if (finetuned_nmi and baseline_nmi) else None,
            'purity_improvement': (finetuned_purity - baseline_purity) if (finetuned_purity and baseline_purity) else None,
            
            # Baseline cluster quality
            'baseline_n_pure_clusters': baseline_cluster_quality.get('n_pure_clusters'),
            'baseline_pure_percentage': baseline_cluster_quality.get('pure_percentage'),
            'baseline_n_mixed_clusters': baseline_cluster_quality.get('n_mixed_clusters'),
            'baseline_avg_models_per_mixed': baseline_cluster_quality.get('avg_models_per_mixed_cluster'),
            'baseline_avg_cluster_size': baseline_cluster_quality.get('avg_cluster_size'),
            
            # Finetuned cluster quality
            'finetuned_n_pure_clusters': finetuned_cluster_quality.get('n_pure_clusters'),
            'finetuned_pure_percentage': finetuned_cluster_quality.get('pure_percentage'),
            'finetuned_n_mixed_clusters': finetuned_cluster_quality.get('n_mixed_clusters'),
            'finetuned_avg_models_per_mixed': finetuned_cluster_quality.get('avg_models_per_mixed_cluster'),
            'finetuned_avg_cluster_size': finetuned_cluster_quality.get('avg_cluster_size'),
            'finetuned_median_cluster_size': finetuned_cluster_quality.get('median_cluster_size'),
            'finetuned_min_cluster_size': finetuned_cluster_quality.get('min_cluster_size'),
            'finetuned_max_cluster_size': finetuned_cluster_quality.get('max_cluster_size'),
            
            # Class overlap (baseline)
            'baseline_n_overlapping_classes': baseline_overlap_info.get('n_overlapping_classes'),
            'baseline_avg_clusters_per_class': baseline_overlap_info.get('avg_clusters_per_overlapping_class'),
            'baseline_max_clusters_per_class': baseline_overlap_info.get('max_clusters_per_overlapping_class'),
            'baseline_top_overlapping': baseline_overlap_info.get('top_overlapping_classes'),
            
            # Class overlap (finetuned)
            'finetuned_n_overlapping_classes': finetuned_overlap_info.get('n_overlapping_classes'),
            'finetuned_avg_clusters_per_class': finetuned_overlap_info.get('avg_clusters_per_overlapping_class'),
            'finetuned_max_clusters_per_class': finetuned_overlap_info.get('max_clusters_per_overlapping_class'),
            'finetuned_top_overlapping': finetuned_overlap_info.get('top_overlapping_classes'),
        }
        
        embeddings_data.append(record)
    
    return embeddings_data


def propagate_embeddings_baselines(embeddings_data: List[Dict]) -> List[Dict]:
    """Propagate classification baselines to metric-learning records.

    The embeddings baseline only exists under ``'classification'`` and is shared
    across all objectives for the same model/view combination.

    Args:
        embeddings_data: List of embeddings records.

    Returns:
        Updated list with baselines filled in for metric-learning records.
    """
    baselines = {}

    for record in embeddings_data:
        if record['objective'] == 'classification':
            key = (record['model_name'], record['views'].strip())
            baselines[key] = {
                'baseline_ari': record.get('baseline_ari'),
                'baseline_nmi': record.get('baseline_nmi'),
                'baseline_purity': record.get('baseline_purity'),
                'baseline_silhouette': record.get('baseline_silhouette'),
                'baseline_n_clusters': record.get('baseline_n_clusters'),
                'baseline_clusterer': record.get('baseline_clusterer'),
                'baseline_reducer': record.get('baseline_reducer'),
                'baseline_reduced_shape': record.get('baseline_reduced_shape'),
                'baseline_n_pure_clusters': record.get('baseline_n_pure_clusters'),
                'baseline_pure_percentage': record.get('baseline_pure_percentage'),
                'baseline_n_mixed_clusters': record.get('baseline_n_mixed_clusters'),
                'baseline_avg_models_per_mixed': record.get('baseline_avg_models_per_mixed'),
                'baseline_avg_cluster_size': record.get('baseline_avg_cluster_size'),
                'baseline_n_overlapping_classes': record.get('baseline_n_overlapping_classes'),
                'baseline_avg_clusters_per_class': record.get('baseline_avg_clusters_per_class'),
                'baseline_max_clusters_per_class': record.get('baseline_max_clusters_per_class'),
                'baseline_top_overlapping': record.get('baseline_top_overlapping'),
            }
    
    for record in embeddings_data:
        if record['objective'] == 'metric_learning':
            key = (record['model_name'], record['views'].strip())
            if key in baselines:
                baseline = baselines[key]
                for k, v in baseline.items():
                    if record.get(k) is None:
                        record[k] = v
                
                # Recalcular improvements con baselines propagados
                finetuned_ari = record.get('finetuned_ari')
                baseline_ari = record.get('baseline_ari')
                if finetuned_ari is not None and baseline_ari is not None:
                    record['ari_improvement'] = finetuned_ari - baseline_ari
                
                finetuned_nmi = record.get('finetuned_nmi')
                baseline_nmi = record.get('baseline_nmi')
                if finetuned_nmi is not None and baseline_nmi is not None:
                    record['nmi_improvement'] = finetuned_nmi - baseline_nmi
                
                finetuned_purity = record.get('finetuned_purity')
                baseline_purity = record.get('baseline_purity')
                if finetuned_purity is not None and baseline_purity is not None:
                    record['purity_improvement'] = finetuned_purity - baseline_purity
    
    return embeddings_data


def process_model(model_path: Path, model_name: str) -> Tuple[List[Dict], List[Dict]]:
    """Process all experiments for a model (classification and metric learning).

    Args:
        model_path: Base path of the model.
        model_name: Model name.

    Returns:
        Tuple of (finetuning records, embeddings records).
    """
    all_finetuning = []
    all_embeddings = []

    logger.info(f"Processing model: {model_name.upper()}")

    logger.debug("Reading classification configurations")
    ft_class = extract_finetuning_data(model_path, model_name, 'classification')
    emb_class = extract_embeddings_data(model_path, model_name, 'classification')
    all_finetuning.extend(ft_class)
    all_embeddings.extend(emb_class)

    logger.debug("Reading metric learning configurations")
    metric_learning_path = model_path / 'metric learning'

    if metric_learning_path.exists():
        loss_functions = ['arcface', 'contrastive', 'multisimilarity', 'ntxent', 'triplet']

        for loss_fn in loss_functions:
            logger.debug(f"Reading: {loss_fn}")
            ft_ml = extract_finetuning_data(model_path, model_name, 'metric_learning', loss_fn)
            emb_ml = extract_embeddings_data(model_path, model_name, 'metric_learning', loss_fn)
            all_finetuning.extend(ft_ml)
            all_embeddings.extend(emb_ml)
    
    return all_finetuning, all_embeddings


def main():
    """Extract all vision model results and save consolidated CSV files.

    Processes ResNet50 and ViT-B/32 experiments in both classification and metric
    learning modes.  Extracts finetuning and embeddings data, merges results by
    config key, and writes CSV files to ``results/analysis/data/vision_models/``.
    """
    results_dir = REPO_ROOT / 'results/models'

    if not results_dir.exists():
        logger.error(f"Directory not found: {results_dir}")
        return

    logger.info("=" * 80)
    logger.info("Starting extraction of results - Vision Models")
    logger.info("=" * 80)

    all_finetuning_data = []
    all_embeddings_data = []

    logger.info("Processing ResNet50")
    resnet_path = results_dir / 'resnet50'
    if resnet_path.exists():
        ft, emb = process_model(resnet_path, 'resnet50')
        all_finetuning_data.extend(ft)
        all_embeddings_data.extend(emb)

    logger.info("Processing ViT-B/32")
    vit_path = results_dir / 'vitb32'
    if vit_path.exists():
        ft, emb = process_model(vit_path, 'vit_b_32')
        all_finetuning_data.extend(ft)
        all_embeddings_data.extend(emb)

    logger.info("Building DataFrames")
    df_finetuning = pd.DataFrame(all_finetuning_data)
    df_embeddings = pd.DataFrame(all_embeddings_data)

    logger.info(f"Finetuning: {len(df_finetuning)} records")
    logger.info(f"Embeddings: {len(df_embeddings)} records")

    if all_embeddings_data:
        logger.info("Propagating embeddings baselines")
        all_embeddings_data = propagate_embeddings_baselines(all_embeddings_data)
        df_embeddings = pd.DataFrame(all_embeddings_data)
        logger.info("Baselines propagated successfully")

    logger.info("Merging finetuning and embeddings results")

    if df_finetuning.empty and df_embeddings.empty:
        logger.error("No data found to process")
        return

    if not df_finetuning.empty and not df_embeddings.empty:
        df_combined = pd.merge(
            df_finetuning,
            df_embeddings,
            on='config_key',
            how='outer',
            suffixes=('_ft', '_emb')
        )

        for col in ['model_name', 'objective', 'criterion', 'views']:
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

    if 'timestamp' in df_combined.columns:
        df_combined['timestamp'] = pd.to_datetime(df_combined['timestamp'])
        df_combined = df_combined.sort_values('timestamp')

    output_dir = REPO_ROOT / 'results/analysis/data/vision_models'
    output_dir.mkdir(exist_ok=True, parents=True)

    logger.info("Saving results")

    combined_csv = output_dir / 'vision_models_results.csv'
    df_combined.to_csv(combined_csv, index=False)
    logger.info(f"Results saved to: {combined_csv}")
    logger.info(f"Total configurations: {len(df_combined)}")

    if not df_finetuning.empty:
        ft_csv = output_dir / 'vision_finetuning_results.csv'
        df_finetuning.to_csv(ft_csv, index=False)
        logger.info(f"Finetuning saved to: {ft_csv}")

    if not df_embeddings.empty:
        emb_csv = output_dir / 'vision_embeddings_results.csv'
        df_embeddings.to_csv(emb_csv, index=False)
        logger.info(f"Embeddings saved to: {emb_csv}")

    logger.info("=" * 80)
    logger.info("Extraction summary")
    logger.info("=" * 80)

    if not df_combined.empty:
        logger.info("Models processed:")
        if 'model_name' in df_combined.columns:
            for model in df_combined['model_name'].unique():
                count = len(df_combined[df_combined['model_name'] == model])
                logger.info(f"  {model}: {count} configurations")

        logger.info("Objectives:")
        if 'objective' in df_combined.columns:
            for obj in df_combined['objective'].unique():
                count = len(df_combined[df_combined['objective'] == obj])
                logger.info(f"  {obj}: {count} configurations")

        logger.info("Loss functions:")
        if 'criterion' in df_combined.columns:
            for crit in df_combined['criterion'].unique():
                count = len(df_combined[df_combined['criterion'] == crit])
                logger.info(f"  {crit}: {count} configurations")

        logger.info("Views:")
        if 'views' in df_combined.columns:
            for view in df_combined['views'].unique():
                count = len(df_combined[df_combined['views'] == view])
                logger.info(f"  {view}: {count} configurations")

    logger.info("=" * 80)
    logger.info("Extraction completed successfully")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()