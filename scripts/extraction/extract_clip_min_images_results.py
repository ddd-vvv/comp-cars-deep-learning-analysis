"""Script to extract results from CLIP Min Images Study experiments.

Focused on robustness under data scarcity (4, 6, 8, 10, 12 images per class).
Analyses how the model adapts when training data is reduced.
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
MIN_IMAGES_STUDY_PATH = REPO_ROOT / 'results/models/clip/min_images_study'
OUTPUT_DIR = REPO_ROOT / 'results/analysis/data/clip_min_images'


def extract_min_images_info(config: Dict) -> Dict[str, Any]:
    """Extract min-images specific metadata from a configuration.

    Args:
        config: Experiment configuration.

    Returns:
        Dict with min-images and CLIP phase information.
    """
    min_images = config.get('min_images', 8)

    phases = config.get('clip_finetuning_phases', {})

    phase_keys = list(phases.keys())

    if not phases:
        return {
            'min_images': min_images,
            'phase_order': None,
            'phase_1_type': None,
            'phase_2_type': None,
        }

    if 'phase_1_vision' in phase_keys or any('vision' in k and 'phase_1' in k for k in phase_keys):
        phase_order = 'vision_text'
        phase_1_key = 'phase_1_vision'
        phase_2_key = 'pahse_2_text'
    else:
        phase_order = 'text_vision'
        phase_1_key = 'phase_1_text'
        phase_2_key = 'pahse_2_vision'

    phase_1_config = phases.get(phase_1_key, {})
    phase_2_config = phases.get(phase_2_key, {})

    return {
        'min_images': min_images,
        'phase_order': phase_order,
        'phase_1_type': phase_1_config.get('type'),
        'phase_2_type': phase_2_config.get('type'),
        'phase_1_lr': phase_1_config.get('lr'),
        'phase_2_lr': phase_2_config.get('lr'),
        'phase_1_epochs': phase_1_config.get('epochs'),
        'phase_2_epochs': phase_2_config.get('epochs'),
    }


def create_min_images_config_key(config: Dict, min_images: int) -> str:
    """Create a unique configuration key for Min Images Study runs.

    Args:
        config: Experiment configuration.
        min_images: Min-images value for the scenario.

    Returns:
        Unique MD5 hash key.
    """
    key_params = ['views', 'class_granularity', 'seed', 'model_name', 'objective', 'batch_size']

    key_values = []

    for param in key_params:
        value = config.get(param)
        if value is not None:
            if isinstance(value, list):
                value = '_'.join(map(str, value))

            key_values.append(f"{param}:{value}")

    key_values.append(f"min_images:{min_images}")
    config_str = '|'.join(sorted(key_values))
    return hashlib.md5(config_str.encode()).hexdigest()


def calculate_duration_minutes(start_time_str: str, end_time_str: str) -> Optional[float]:
    """Calculate the duration in minutes between two ISO timestamps.

    Args:
        start_time_str: Start timestamp in ISO format.
        end_time_str: End timestamp in ISO format.

    Returns:
        Duration in minutes, or ``None`` if it cannot be computed.
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
    """Load a JSON file safely.

    Args:
        filepath: JSON file path.

    Returns:
        Parsed dictionary, or ``None`` on error.
    """
    try:
        if not filepath.exists():
            return None

        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)

    except Exception as e:
        logger.error(f"Error loading {filepath}: {e}")
        return None

def analyze_cluster_quality(cluster_analysis: Dict) -> Dict[str, Any]:
    """Analyse the quality of generated clusters.

    Args:
        cluster_analysis: Cluster analysis dictionary.

    Returns:
        Cluster quality metrics.
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
    """Extract metrics about classes that overlap across clusters.

    Args:
        class_cluster_overlap: List of class-overlap dictionaries.

    Returns:
        Overlap metrics.
    """
    if not class_cluster_overlap:
        return {
            'n_overlapping_classes': 0,
            'avg_clusters_per_class': 0,
            'max_clusters_per_class': 0,
        }

    n_overlapping = len(class_cluster_overlap)
    clusters_per_class = [item['n_clusters'] for item in class_cluster_overlap]

    return {
        'n_overlapping_classes': n_overlapping,
        'avg_clusters_per_class': np.mean(clusters_per_class) if clusters_per_class else 0,
        'max_clusters_per_class': max(clusters_per_class) if clusters_per_class else 0,
    }


def extract_min_images_finetuning_data(min_images_study_path: Path) -> List[Dict]:
    """Extract finetuning data from CLIP Min Images Study experiments.

    Args:
        min_images_study_path: Study root directory.

    Returns:
        List of extracted finetuning records.
    """
    finetune_data = []

    finetuning_path = min_images_study_path / 'finetuning'

    if not finetuning_path.exists():
        logger.info(f"Directory not found: {finetuning_path}")
        return finetune_data

    for min_images_val in [4, 6, 8, 10, 12]:
        scenario_path = finetuning_path / f'{min_images_val} images'

        if not scenario_path.exists():
            logger.info(f"Directory not found: {scenario_path}")
            continue

        logger.info(f"Processing scenario: {min_images_val} images")

        for views_folder in ['front', 'front + rear']:
            views_path = scenario_path / views_folder

            if not views_path.exists():
                logger.info(f"Directory not found: {views_path}")
                continue

            results_file = views_path / 'results.json'
            if not results_file.exists():
                logger.info(f"results.json not found in {views_path}")
                continue

            logger.debug(f"Reading: {min_images_val} images / {views_folder}")

            results = load_json_safe(results_file)
            if results is None:
                continue

            config = results.get('config', {})
            min_images_info = extract_min_images_info(config)
            config_key = create_min_images_config_key(config, min_images_val)
            experiment_name = results.get('experiment_name', f'{min_images_val}_{views_folder}')
            start_time = results.get('start_time')
            end_time = results.get('end_time')
            duration_minutes = calculate_duration_minutes(start_time, end_time)
            
            dataset_stats = results.get('dataset_stats', {})
            
            # Baseline evaluation
            baseline = results.get('baseline', {}).get('evaluation', {})
            baseline_recall1 = baseline.get('recall@1')
            baseline_recall3 = baseline.get('recall@3')
            baseline_recall5 = baseline.get('recall@5')
            baseline_loss = baseline.get('loss')
            
            # Training history in phased format.
            finetuning = results.get('finetuning', {})
            training_history = finetuning.get('training_history', {})
            
            phase_order = min_images_info['phase_order']
            
            if phase_order == 'vision_text':
                phase_1_key = 'phase_1_vision'
                phase_2_key = 'pahse_2_text'
            else:
                phase_1_key = 'phase_1_text'
                phase_2_key = 'pahse_2_vision'
            
            phase_1_history = training_history.get(phase_1_key, {})
            phase_1_train_loss = phase_1_history.get('train_loss', [])
            phase_1_val_recall1 = phase_1_history.get('val_recall@1', [])
            phase_1_val_recall5 = phase_1_history.get('val_recall@5', [])
            
            phase_1_epochs = len(phase_1_train_loss)
            phase_1_best_recall1 = max(phase_1_val_recall1) if phase_1_val_recall1 else None
            
            phase_2_history = training_history.get(phase_2_key, {})
            phase_2_train_loss = phase_2_history.get('train_loss', [])
            phase_2_val_recall1 = phase_2_history.get('val_recall@1', [])
            phase_2_val_recall5 = phase_2_history.get('val_recall@5', [])
            
            phase_2_epochs = len(phase_2_train_loss)
            phase_2_best_recall1 = max(phase_2_val_recall1) if phase_2_val_recall1 else None
            
            total_epochs = phase_1_epochs + phase_2_epochs
            
            # Combine both phases to compute global metrics.
            all_train_loss = phase_1_train_loss + phase_2_train_loss
            all_val_recall1 = phase_1_val_recall1 + phase_2_val_recall1
            all_val_recall5 = phase_1_val_recall5 + phase_2_val_recall5
            
            if all_val_recall1:
                best_epoch = int(np.argmax(all_val_recall1))
                best_val_recall1 = float(np.max(all_val_recall1))
                best_val_recall5 = all_val_recall5[best_epoch] if all_val_recall5 and best_epoch < len(all_val_recall5) else None
            else:
                best_epoch = None
                best_val_recall1 = None
                best_val_recall5 = None
            
            final_train_loss = all_train_loss[-1] if all_train_loss else None
            
            finetuned = results.get('finetuned', {}).get('evaluation', {})
            finetuned_recall1 = finetuned.get('recall@1')
            finetuned_recall3 = finetuned.get('recall@3')
            finetuned_recall5 = finetuned.get('recall@5')
            
            summary = results.get('summary', {})
            recall1_improvement = summary.get('recall@1_improvement')
            recall3_improvement = summary.get('recall@3_improvement')
            recall5_improvement = summary.get('recall@5_improvement')
            
            if recall1_improvement is None and baseline_recall1 and finetuned_recall1:
                recall1_improvement = finetuned_recall1 - baseline_recall1
            if recall3_improvement is None and baseline_recall3 and finetuned_recall3:
                recall3_improvement = finetuned_recall3 - baseline_recall3
            if recall5_improvement is None and baseline_recall5 and finetuned_recall5:
                recall5_improvement = finetuned_recall5 - baseline_recall5
            
            record = {
                'config_key': config_key,
                'experiment_name': experiment_name,
                'timestamp': start_time,

                # Configuration-specific fields
                'model_name': config.get('model_name', 'clip-vit-base-patch32'),
                'objective': 'CLIP',
                'min_images': min_images_val,
                'phase_order': phase_order,
                'views': config.get('views', []),
                'views_str': ' + '.join(config.get('views', [])),
                'num_views': len(config.get('views', [])),
                
                # Hyperparameters
                'batch_size': config.get('batch_size'),
                'phase_1_lr': min_images_info['phase_1_lr'],
                'phase_2_lr': min_images_info['phase_2_lr'],
                'weight_decay': config.get('finetune_optimizer_weight_decay'),
                'optimizer': config.get('finetune_optimizer_type'),
                'use_amp': config.get('use_amp'),
                'phase_1_epochs_configured': min_images_info['phase_1_epochs'],
                'phase_2_epochs_configured': min_images_info['phase_2_epochs'],
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
                'phase_1_epochs_trained': phase_1_epochs,
                'phase_2_epochs_trained': phase_2_epochs,
                'best_epoch': best_epoch,
                
                # Baseline
                'baseline_recall@1': baseline_recall1,
                'baseline_recall@3': baseline_recall3,
                'baseline_recall@5': baseline_recall5,
                'baseline_loss': baseline_loss,
                
                # Best metrics during training
                'best_val_recall@1': best_val_recall1,
                'best_val_recall@5': best_val_recall5,
                'phase_1_best_recall@1': phase_1_best_recall1,
                'phase_2_best_recall@1': phase_2_best_recall1,
                
                # Final metrics
                'final_train_loss': final_train_loss,
                
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


def extract_min_images_embeddings_data(min_images_study_path: Path) -> List[Dict]:
    """Extract embeddings and clustering metrics from Min Images Study.

    Args:
        min_images_study_path: Study root directory.

    Returns:
        List of clustering metric records.
    """
    embeddings_data = []
    
    embeddings_path = min_images_study_path / 'embeddings'
    if not embeddings_path.exists():
        logger.info(f"Directory not found: {embeddings_path}")
        return []
    
    for min_images_val in [4, 6, 8, 10, 12]:
        scenario_path = embeddings_path / f'{min_images_val} images'
        
        if not scenario_path.exists():
            continue
        
        logger.debug(f"Processing clustering: {min_images_val} images")
        
        for views_folder in ['front', 'front + rear']:
            views_path = scenario_path / views_folder
            
            if not views_path.exists():
                continue
            
            logger.debug(f"Reading view: {views_folder}")
            
            results_file = views_path / 'results.json'
            if not results_file.exists():
                continue
            
            results = load_json_safe(results_file)
            if results is None:
                continue
            
            config = results.get('config', {})
            min_images_info = extract_min_images_info(config)
            config_key = create_min_images_config_key(config, min_images_val)
            
            # Baseline clustering metrics
            baseline_analysis = results.get('baseline_analysis', {})
            baseline_clustering = baseline_analysis.get('clustering', {})
            baseline_comparison = baseline_clustering.get('comparison_df', [{}])
            baseline_best = baseline_comparison[0] if baseline_comparison else {}
            
            baseline_visualization = baseline_analysis.get('visualization', {})
            baseline_cluster_analysis = baseline_visualization.get('cluster_analysis', {})
            baseline_overlap = baseline_visualization.get('class_cluster_overlap', [])
            
            baseline_cluster_quality = analyze_cluster_quality(baseline_cluster_analysis)
            baseline_overlap_info = extract_class_overlap_info(baseline_overlap)
            
            # Finetuned clustering metrics
            finetuned_analysis = results.get('finetuned_analysis', {})
            finetuned_clustering = finetuned_analysis.get('clustering', {})
            finetuned_comparison = finetuned_clustering.get('comparison_df', [{}])
            finetuned_best = finetuned_comparison[0] if finetuned_comparison else {}
            
            finetuned_visualization = finetuned_analysis.get('visualization', {})
            finetuned_cluster_analysis = finetuned_visualization.get('cluster_analysis', {})
            finetuned_overlap = finetuned_visualization.get('class_cluster_overlap', [])
            
            finetuned_cluster_quality = analyze_cluster_quality(finetuned_cluster_analysis)
            finetuned_overlap_info = extract_class_overlap_info(finetuned_overlap)
            
            record = {
                'config_key': config_key,
                'min_images': min_images_val,
                
                # Baseline clustering
                'baseline_ari': baseline_best.get('adjusted_rand_score'),
                'baseline_nmi': baseline_best.get('normalized_mutual_info'),
                'baseline_purity': baseline_best.get('purity'),
                'baseline_n_clusters': baseline_cluster_quality.get('n_clusters'),
                'baseline_n_pure_clusters': baseline_cluster_quality.get('n_pure_clusters'),
                'baseline_n_mixed_clusters': baseline_cluster_quality.get('n_mixed_clusters'),
                'baseline_pure_percentage': baseline_cluster_quality.get('pure_percentage'),
                'baseline_n_overlapping_classes': baseline_overlap_info.get('n_overlapping_classes'),
                
                # Finetuned clustering
                'finetuned_ari': finetuned_best.get('adjusted_rand_score'),
                'finetuned_nmi': finetuned_best.get('normalized_mutual_info'),
                'finetuned_purity': finetuned_best.get('purity'),
                'finetuned_n_clusters': finetuned_cluster_quality.get('n_clusters'),
                'finetuned_n_pure_clusters': finetuned_cluster_quality.get('n_pure_clusters'),
                'finetuned_n_mixed_clusters': finetuned_cluster_quality.get('n_mixed_clusters'),
                'finetuned_pure_percentage': finetuned_cluster_quality.get('pure_percentage'),
                'finetuned_n_overlapping_classes': finetuned_overlap_info.get('n_overlapping_classes'),
            }
            
            embeddings_data.append(record)
    
    return embeddings_data


def main():
    """Extract all CLIP Min Images Study results and save consolidated CSVs.

    Processes experiments varying the minimum number of images per class
    (4, 6, 8, 10, 12) to evaluate robustness under data scarcity.
    """
    logger.info("=" * 80)
    logger.info("Starting extraction of results - CLIP Min Images Study")
    logger.info("=" * 80)

    if not MIN_IMAGES_STUDY_PATH.exists():
        logger.error(f"Directory not found: {MIN_IMAGES_STUDY_PATH}")
        return

    logger.info("Extracting finetuning data")
    finetune_data = extract_min_images_finetuning_data(MIN_IMAGES_STUDY_PATH)

    logger.info("Extracting embeddings and clustering data")
    embeddings_data = extract_min_images_embeddings_data(MIN_IMAGES_STUDY_PATH)

    logger.info("Building DataFrames")
    df_finetune = pd.DataFrame(finetune_data) if finetune_data else pd.DataFrame()
    df_embeddings = pd.DataFrame(embeddings_data) if embeddings_data else pd.DataFrame()

    logger.info(f"Finetuning: {len(df_finetune)} records")
    logger.info(f"Embeddings: {len(df_embeddings)} records")

    if df_finetune.empty and df_embeddings.empty:
        logger.error("No data found to process")
        return

    logger.info("Merging finetuning and embeddings results")
    if finetune_data and embeddings_data:
        df_finetune = pd.DataFrame(finetune_data)
        df_embeddings = pd.DataFrame(embeddings_data)
        
        df_merged = pd.merge(
            df_finetune,
            df_embeddings,
            on='config_key',
            how='outer',
            suffixes=('', '_emb')
        )
        
        if 'min_images_emb' in df_merged.columns:
            df_merged = df_merged.drop(columns=['min_images_emb'])

        logger.info(f"Combined: {len(df_merged)} unique configurations")
        logger.info(f"Available columns: {len(list(df_merged.columns))}")

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        output_file = OUTPUT_DIR / 'clip_min_images_results.csv'
        df_merged.to_csv(output_file, index=False)
        logger.info(f"Results saved to: {output_file}")

        logger.info("=" * 80)
        logger.info("Results summary by min_images scenario")
        logger.info("=" * 80)

        if 'min_images' in df_merged.columns:
            min_img_scenarios = sorted(df_merged['min_images'].unique())
            logger.info(f"Processed scenarios: {min_img_scenarios}")

            for min_img in min_img_scenarios:
                subset = df_merged[df_merged['min_images'] == min_img]
                logger.info(f"min_images={min_img}: {len(subset)} configurations")

                if 'finetuned_recall@1' in subset.columns:
                    recall_mean = subset['finetuned_recall@1'].mean()
                    logger.info(f"  Average Recall@1: {recall_mean:.2f}%")

                if 'finetuned_ari' in subset.columns:
                    ari_mean = subset['finetuned_ari'].mean()
                    logger.info(f"  Average ARI: {ari_mean:.4f}")

        if 'phase_order' in df_merged.columns:
            phase_orders = df_merged['phase_order'].unique()
            logger.info(f"Processed phase orders: {list(phase_orders)}")

        logger.info("=" * 80)
        logger.info("Extraction completed successfully")
        logger.info("=" * 80)

    else:
        logger.warning("Could not merge finetuning and embeddings data")

    logger.info("=" * 80)
    logger.info("Extraction completed")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
