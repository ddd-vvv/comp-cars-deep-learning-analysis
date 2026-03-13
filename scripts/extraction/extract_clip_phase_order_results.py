"""Script to extract results from CLIP Phase Order Study experiments.

This module processes experiments where the fine-tuning phase order is varied
(vision-first then text, or vice versa) in CLIP. It extracts finetuning and
embeddings data and generates consolidated CSV files.
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
PHASE_ORDER_STUDY_PATH = REPO_ROOT / 'results/models/clip/phase_order_study'
OUTPUT_DIR = REPO_ROOT / 'results/analysis/data/clip_phase_order'


def extract_phase_order_info(config: Dict) -> Dict[str, Any]:
    """Extract phase-order metadata from an experiment configuration.

    Args:
        config: Experiment configuration.

    Returns:
        Phase-order information dictionary.
    """
    phases = config.get('clip_finetuning_phases', {})

    if not phases:
        return {
            'phase_order': None,
            'phase_1_type': None,
            'phase_2_type': None,
            'phase_1_lr': None,
            'phase_2_lr': None,
            'phase_1_epochs': None,
            'phase_2_epochs': None,
        }
    
    phase_keys = list(phases.keys())

    if 'phase_1_vision' in phase_keys:
        phase_order = 'vision_text'
        phase_1_key = 'phase_1_vision'
        phase_2_key = 'phase_2_text'
    else:
        phase_order = 'text_vision'
        phase_1_key = 'phase_1_text'
        phase_2_key = 'phase_2_vision'

    phase_1_config = phases.get(phase_1_key, {})
    phase_2_config = phases.get(phase_2_key, {})

    return {
        'phase_order': phase_order,
        'phase_1_type': phase_1_config.get('type'),
        'phase_2_type': phase_2_config.get('type'),
        'phase_1_lr': phase_1_config.get('lr'),
        'phase_2_lr': phase_2_config.get('lr'),
        'phase_1_epochs': phase_1_config.get('epochs'),
        'phase_2_epochs': phase_2_config.get('epochs'),
        'phase_1_patience': phase_1_config.get('early_stopping', {}).get('patience'),
        'phase_2_patience': phase_2_config.get('early_stopping', {}).get('patience'),
    }


def create_phase_order_config_key(config: Dict, phase_order: str) -> str:
    """Create a unique configuration key for phase-order experiments.

    Args:
        config: Experiment configuration.
        phase_order: Phase order (``'vision_text'`` or ``'text_vision'``).

    Returns:
        Unique MD5 hash key.
    """
    key_params = ['views', 'class_granularity', 'min_images', 'seed', 'model_name', 'objective', 'batch_size']

    key_values = []

    for param in key_params:
        value = config.get(param)
        if value is not None:
            if isinstance(value, list):
                value = '_'.join(map(str, value))
            key_values.append(f"{param}:{value}")

    key_values.append(f"phase_order:{phase_order}")
    config_str = '|'.join(sorted(key_values))
    return hashlib.md5(config_str.encode()).hexdigest()


def calculate_duration_minutes(start_time_str: str, end_time_str: str) -> Optional[float]:
    """Calculate duration in minutes between two ISO timestamps.

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


def extract_phase_order_finetuning_data(phase_order_path: Path) -> List[Dict]:
    """Extract finetuning data from CLIP phase-order study experiments.

    Args:
        phase_order_path: Path to phase-order results.

    Returns:
        List of extracted records.
    """
    finetune_data = []

    finetuning_path = phase_order_path / 'finetuning'

    if not finetuning_path.exists():
        logger.info(f"Directory not found: {finetuning_path}")
        return finetune_data

    for phase_order in ['vision_text', 'text_vision']:
        order_path = finetuning_path / phase_order

        if not order_path.exists():
            logger.info(f"Directory not found: {order_path}")
            continue

        logger.info(f"Processing finetuning: {phase_order}")

        results_file = order_path / 'results.json'

        if not results_file.exists():
            logger.info(f"results.json not found in {order_path}")
            continue

        logger.debug(f"Reading configuration: {phase_order}")

        results = load_json_safe(results_file)

        if results is None: 
            continue

        config = results.get('config', {})
        phase_info = extract_phase_order_info(config)
        config_key = create_phase_order_config_key(config, phase_order)

        experiment_name = results.get('experiment_name', phase_order)
        start_time = results.get('start_time')
        end_time = results.get('end_time')
        duration_minutes = calculate_duration_minutes(start_time, end_time)

        dataset_stats = results.get('dataset_stats', {})

        baseline = results.get('baseline', {}).get('evaluation', {})
        baseline_recall1 = baseline.get('recall@1')
        baseline_recall3 = baseline.get('recall@3')
        baseline_recall5 = baseline.get('recall@5')

        finetuning = results.get('finetuning', {})
        training_history = finetuning.get('training_history', {})

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
        phase_1_best_recall5 = max(phase_1_val_recall5) if phase_1_val_recall5 else None

        phase_2_history = training_history.get(phase_2_key, {})
        phase_2_train_loss = phase_2_history.get('train_loss', [])
        phase_2_val_recall1 = phase_2_history.get('val_recall@1', [])
        phase_2_val_recall5 = phase_2_history.get('val_recall@5', [])
        phase_2_epochs = len(phase_2_train_loss)
        phase_2_best_recall1 = max(phase_2_val_recall1) if phase_2_val_recall1 else None
        phase_2_best_recall5 = max(phase_2_val_recall5) if phase_2_val_recall5 else None

        total_epochs = phase_1_epochs + phase_2_epochs
        all_val_recall1 = phase_1_val_recall1 + phase_2_val_recall1
        best_val_recall1 = max(all_val_recall1) if all_val_recall1 else None

        finetuned = results.get('finetuned', {}).get('evaluation', {})
        finetuned_recall1 = finetuned.get('recall@1')
        finetuned_recall3 = finetuned.get('recall@3')
        finetuned_recall5 = finetuned.get('recall@5')

        summary = results.get('summary', {})
        recall1_improvement = summary.get('recall@1_improvement')
        recall5_improvement = summary.get('recall@5_improvement')

        if recall1_improvement is None and baseline_recall1 and finetuned_recall1:
            recall1_improvement = finetuned_recall1 - baseline_recall1
        if recall5_improvement is None and baseline_recall5 and finetuned_recall5:
            recall5_improvement = finetuned_recall5 - baseline_recall5

        recall3_improvement = None
        if baseline_recall3 and finetuned_recall3:
            recall3_improvement = finetuned_recall3 - baseline_recall3

        record = {
            'config_key': config_key,
            'experiment_name': experiment_name,
            'timestamp': start_time,
            'model_name': config.get('model_name', 'clip-vit-base-patch32'),
            'objective': 'CLIP',
            'phase_order': phase_order,
            'phase_1_type': phase_info['phase_1_type'],
            'phase_2_type': phase_info['phase_2_type'],
            'views_str': ' + '.join(config.get('views', [])),
            'batch_size': config.get('batch_size'),
            'seed': config.get('seed'),
            'num_training_classes': dataset_stats.get('num_training_classes'),
            'num_total_classes': dataset_stats.get('num_total_classes'),
            'start_time': start_time,
            'end_time': end_time,
            'duration_minutes': duration_minutes,
            'total_epochs': total_epochs,
            'phase_1_epochs_trained': phase_1_epochs,
            'phase_2_epochs_trained': phase_2_epochs,
            'baseline_recall@1': baseline_recall1,
            'baseline_recall@3': baseline_recall3,
            'baseline_recall@5': baseline_recall5,
            'phase_1_best_recall@1': phase_1_best_recall1,
            'phase_1_best_recall@5': phase_1_best_recall5,
            'phase_2_best_recall@1': phase_2_best_recall1,
            'phase_2_best_recall@5': phase_2_best_recall5,
            'best_val_recall@1': best_val_recall1,
            'finetuned_recall@1': finetuned_recall1,
            'finetuned_recall@3': finetuned_recall3,
            'finetuned_recall@5': finetuned_recall5,
            'recall@1_improvement': recall1_improvement,
            'recall@3_improvement': recall3_improvement,
            'recall@5_improvement': recall5_improvement,
        }
        
        finetune_data.append(record)

    return finetune_data


def analyze_cluster_quality(cluster_analysis: Dict) -> Dict[str, Any]:
    """Analyse the quality of generated clusters.

    Args:
        cluster_analysis: Experiment cluster-analysis dictionary.

    Returns:
        Cluster quality metrics.
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
            models_per_mixed_cluster.append(cluster_info.get('n_unique_models', 0))

    n_pure = len(pure_clusters)
    n_mixed = len(mixed_clusters)
    pure_percentage = (n_pure / total_clusters * 100) if total_clusters > 0 else 0

    return {
        'n_clusters': total_clusters,
        'n_pure_clusters': n_pure,
        'n_mixed_clusters': n_mixed,
        'pure_percentage': pure_percentage,
        'avg_cluster_size': np.mean(cluster_sizes) if cluster_sizes else 0,
    }


def extract_class_overlap_info(class_cluster_overlap: List[Dict]) -> Dict[str, Any]:
    """Extract overlap metrics for classes spanning multiple clusters.

    Args:
        class_cluster_overlap: List of class-cluster overlap dictionaries.

    Returns:
        Class-overlap metrics.
    """
    if not class_cluster_overlap:
        return {'n_overlapping_classes': 0, 'avg_clusters_per_class': 0}

    n_overlapping = len(class_cluster_overlap)
    clusters_per_class = [item['n_clusters'] for item in class_cluster_overlap]
    return {
        'n_overlapping_classes': n_overlapping,
        'avg_clusters_per_class': np.mean(clusters_per_class) if clusters_per_class else 0,
    }


def extract_phase_order_embeddings_data(phase_order_path: Path) -> List[Dict]:
    """Extract embeddings data from CLIP phase-order study experiments.

    Args:
        phase_order_path: Path to phase-order results.

    Returns:
        List of extracted records.
    """
    embeddings_data = []

    embeddings_path = phase_order_path / 'embeddings'

    if not embeddings_path.exists():
        logger.info(f"Directory not found: {embeddings_path}")
        return embeddings_data

    for phase_order in ['vision_text', 'text_vision']:
        order_path = embeddings_path / phase_order
        if not order_path.exists():
            continue

        logger.info(f"Processing embeddings: {phase_order}")

        results_file = order_path / 'results.json'
        if not results_file.exists() :
            continue

        logger.debug(f"Reading: {phase_order}/embeddings")

        results = load_json_safe(results_file)
        if results is None:
            continue

        config = results.get('config', {})
        config_key = create_phase_order_config_key(config, phase_order)
        experiment_name = results.get('experiment_name', phase_order)
        start_time = results.get('start_time')

        finetuned_analysis = results.get('finetuned_analysis', {})
        finetuned_reduction = finetuned_analysis.get('reduction', {})
        finetuned_best_shape = finetuned_reduction.get('best_embeddings_shape')
        finetuned_clustering = finetuned_analysis.get('clustering', {})
        finetuned_best_comparison = finetuned_clustering.get('comparison_df', [{}])

        if finetuned_best_comparison:
            finetuned_best = finetuned_best_comparison[0]
            finetuned_ari = finetuned_best.get('adjusted_rand_score')
            finetuned_nmi = finetuned_best.get('normalized_mutual_info')
            finetuned_purity = finetuned_best.get('purity')
            finetuned_n_clusters = finetuned_best.get('n_clusters')
        else:
            finetuned_ari = finetuned_nmi = finetuned_purity = finetuned_n_clusters = None

        finetuned_visualization = finetuned_analysis.get('visualization', {})
        finetuned_cluster_analysis = finetuned_visualization.get('cluster_analysis', {})
        finetuned_cluster_quality = analyze_cluster_quality(finetuned_cluster_analysis)
        finetuned_overlap = finetuned_visualization.get('class_cluster_overlap', [])
        finetuned_overlap_info = extract_class_overlap_info(finetuned_overlap)

        baseline_analysis = results.get('baseline_analysis', {})
        baseline_clustering = baseline_analysis.get('clustering', {})
        baseline_best_comparison = baseline_clustering.get('comparison_df', [{}])
        baseline_ari = baseline_best_comparison[0].get('adjusted_rand_score') if baseline_best_comparison else None
        baseline_purity = baseline_best_comparison[0].get('purity') if baseline_best_comparison else None

        record = {
            'config_key': config_key,
            'experiment_name': experiment_name,
            'timestamp': start_time,
            'model_name': 'clip-vit-base-patch32',
            'objective': 'CLIP',
            'phase_order': phase_order,
            'views_str': ' + '.join(config.get('views', [])),
            'finetuned_embedding_shape': finetuned_best_shape,
            'baseline_ari': baseline_ari,
            'baseline_purity': baseline_purity,
            'finetuned_ari': finetuned_ari,
            'finetuned_nmi': finetuned_nmi,
            'finetuned_purity': finetuned_purity,
            'finetuned_n_clusters': finetuned_n_clusters,
            'finetuned_n_pure_clusters': finetuned_cluster_quality.get('n_pure_clusters'),
            'finetuned_pure_percentage': finetuned_cluster_quality.get('pure_percentage'),
            'finetuned_n_mixed_clusters': finetuned_cluster_quality.get('n_mixed_clusters'),
            'finetuned_avg_cluster_size': finetuned_cluster_quality.get('avg_cluster_size'),
            'finetuned_n_overlapping_classes': finetuned_overlap_info.get('n_overlapping_classes'),
        }
        
        embeddings_data.append(record)

    return embeddings_data


def main():
    """Extract all CLIP phase-order study results and save consolidated CSV files."""

    if not PHASE_ORDER_STUDY_PATH.exists():
        logger.error(f"Directory not found: {PHASE_ORDER_STUDY_PATH}")
        return

    logger.info("=" * 80)
    logger.info("Starting extraction of results - CLIP Phase Order Study")
    logger.info("=" * 80)

    logger.info("Extracting finetuning data")
    finetuning_data = extract_phase_order_finetuning_data(PHASE_ORDER_STUDY_PATH)

    logger.info("Extracting embeddings data")
    embeddings_data = extract_phase_order_embeddings_data(PHASE_ORDER_STUDY_PATH)

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
        
        for col in ['model_name', 'objective', 'phase_order', 'views_str']:
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

    if 'phase_order' in df_combined.columns:
        df_combined = df_combined.sort_values('phase_order')
        logger.info("Sorted by phase_order")

    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

    combined_csv = OUTPUT_DIR / 'clip_phase_order_results.csv'
    df_combined.to_csv(combined_csv, index=False)
    logger.info(f"Results saved to: {combined_csv}")
    logger.info(f"Total configurations: {len(df_combined)}")

    if not df_combined.empty and 'finetuned_recall@1' in df_combined.columns:
        logger.info("Recall@1 summary by phase order:")
        for _, row in df_combined.iterrows():
            order = row.get('phase_order', 'N/A')
            recall = row.get('finetuned_recall@1', 0)
            baseline = row.get('baseline_recall@1', 0)
            improvement = row.get('recall@1_improvement', 0)
            logger.info(
                f"  {order}: Recall@1={recall:.2f}% "
                f"(baseline={baseline:.2f}%, improvement=+{improvement:.2f}%)"
            )

    logger.info("=" * 80)
    logger.info("Extraction completed successfully")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
