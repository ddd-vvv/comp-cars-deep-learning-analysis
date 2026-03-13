"""
Script to visualize training history.
Generates loss and metric plots per epoch for each experiment.
FIXED: Automatically detects whether to use accuracy or recall based on objective.
"""

import json
import torch
import matplotlib.pyplot as plt
import warnings
from pathlib import Path
from typing import Dict, Optional

warnings.filterwarnings('ignore')

REPO_ROOT = Path(__file__).resolve().parents[2]

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'figure.figsize': (12, 5),
    'font.size': 10,
    'font.family': 'serif',
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'axes.titleweight': 'bold',
    'axes.labelweight': 'bold',
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.titlesize': 13,
    'figure.titleweight': 'bold',
})

BLUES = plt.cm.Blues
COLOR_TRAIN = '#2171b5'      # Dark blue (train)
COLOR_VAL = '#6baed6'        # Medium blue (validation)
COLOR_RECALL1 = '#08519c'    # Very dark blue
COLOR_RECALL3 = '#3182bd'    # Medium blue
COLOR_RECALL5 = '#9ecae1'    # Light blue
COLOR_ACCURACY = '#2171b5'   # Dark blue


def load_training_history(experiment_dir: Path) -> Optional[Dict]:
    """Load training history from training_history.pt or results.json."""
    # Try loading from training_history.pt
    history_file = experiment_dir / 'training_history.pt'
    if history_file.exists():
        try:
            history = torch.load(history_file, map_location='cpu')
            return history
        except Exception as e:
            print(f"    [WARNING] Error loading training_history.pt: {e}")
    
    # Try loading from results.json
    results_file = experiment_dir / 'results.json'
    if results_file.exists() and results_file.stat().st_size < 10 * 1024 * 1024:  # < 10MB
        try:
            with open(results_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
                finetuning = results.get('finetuning', {})
                history = finetuning.get('training_history', {})
                if history:
                    return history
        except Exception as e:
            print(f"    [WARNING] Error loading results.json: {e}")
    
    return None


def flatten_history_if_needed(history: Dict) -> Dict:
    """Flatten history if it contains phases (CLIP)."""
    if any(k.startswith('phase_') for k in history.keys()):
        combined_history = {}
        phases = sorted([k for k in history.keys() if k.startswith('phase_')])
        
        for phase in phases:
            phase_history = history[phase]
            for key, value in phase_history.items():
                if key not in combined_history:
                    combined_history[key] = []
                if isinstance(value, list):
                    combined_history[key].extend(value)
                else:
                    combined_history[key].append(value)
        return combined_history
    return history


def load_config(experiment_dir: Path) -> Optional[Dict]:
    """Load experiment configuration."""
    config_file = experiment_dir / 'config.json'
    if config_file.exists():
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"    [WARNING] Error loading config.json: {e}")
    return None


def plot_training_history(experiment_dir: Path, output_dir: Path):
    """Generate training history plots for one experiment."""
    experiment_name = experiment_dir.name
    
    # Load history and configuration
    history = load_training_history(experiment_dir)
    config = load_config(experiment_dir)
    
    if history is None:
        print(f"   Training history not found for {experiment_name}")
        return False
    
    # Flatten history if needed (by phases)
    history = flatten_history_if_needed(history)
    
    # Extract configuration information
    model_name = config.get('model_name', 'unknown') if config else 'unknown'
    objective = config.get('objective', 'unknown') if config else 'unknown'
    criterion = config.get('finetune_criterion', 'unknown') if config else 'unknown'
    
    # Detect available metrics
    has_loss = 'train_loss' in history and len(history['train_loss']) > 0
    has_val_loss = 'val_loss' in history and len(history['val_loss']) > 0
    has_accuracy = 'val_accuracy' in history and len(history['val_accuracy']) > 0
    has_recall1 = 'val_recall@1' in history and len(history['val_recall@1']) > 0
    has_recall3 = 'val_recall@3' in history and len(history['val_recall@3']) > 0
    has_recall5 = 'val_recall@5' in history and len(history['val_recall@5']) > 0
    
    # Determine which plots to create based on objective
    # MAIN RULE: use objective to decide which metric to plot
    # - classification -> accuracy
    # - metric_learning -> recall
    # - CLIP -> recall
    metrics_to_plot = []
    
    if has_loss:
        metrics_to_plot.append('loss')
    
    # Decide based on objective
    if objective == 'classification':
        # Classification always uses accuracy
        if has_accuracy:
            metrics_to_plot.append('accuracy')
    else:
        # metric_learning and CLIP use recall
        if has_recall1 or has_recall3 or has_recall5:
            metrics_to_plot.append('recall')
    
    if not metrics_to_plot:
        print(f"   No metrics available to plot in {experiment_name}")
        return False
    
    # Determine layout
    n_plots = len(metrics_to_plot)
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
    
    if n_plots == 1:
        axes = [axes]
    
    plot_idx = 0
    
    # 1. Loss plot
    if 'loss' in metrics_to_plot:
        ax = axes[plot_idx]
        plot_idx += 1
        
        epochs = list(range(1, len(history['train_loss']) + 1))
        
        ax.plot(epochs, history['train_loss'], label='Train Loss', marker='o', markersize=3, linewidth=2, color=COLOR_TRAIN)
        if has_val_loss:
            ax.plot(epochs, history['val_loss'], label='Val Loss', marker='s', markersize=3, linewidth=2, color=COLOR_VAL)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training and Validation Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 2. Accuracy plot (classification only)
    if 'accuracy' in metrics_to_plot:
        ax = axes[plot_idx]
        plot_idx += 1
        
        epochs = list(range(1, len(history['val_accuracy']) + 1))
        ax.plot(epochs, history['val_accuracy'], label='Val Accuracy', marker='s', markersize=3, linewidth=2, color=COLOR_ACCURACY)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Validation Accuracy')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 3. Recall plot (for metric_learning and CLIP)
    if 'recall' in metrics_to_plot:
        ax = axes[plot_idx]
        plot_idx += 1
        
        # Use the longest available metric for epochs
        recall_lengths = []
        if has_recall1:
            recall_lengths.append(len(history['val_recall@1']))
        if has_recall3:
            recall_lengths.append(len(history['val_recall@3']))
        if has_recall5:
            recall_lengths.append(len(history['val_recall@5']))
        
        max_length = max(recall_lengths) if recall_lengths else 0
        epochs = list(range(1, max_length + 1))
        
        if has_recall1:
            ax.plot(epochs[:len(history['val_recall@1'])], history['val_recall@1'], 
                   label='Recall@1', marker='o', markersize=3, linewidth=2, color=COLOR_RECALL1)
        if has_recall3:
            ax.plot(epochs[:len(history['val_recall@3'])], history['val_recall@3'], 
                   label='Recall@3', marker='s', markersize=3, linewidth=2, color=COLOR_RECALL3)
        if has_recall5:
            ax.plot(epochs[:len(history['val_recall@5'])], history['val_recall@5'], 
                   label='Recall@5', marker='^', markersize=3, linewidth=2, color=COLOR_RECALL5)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Recall')
        ax.set_title('Validation Recall Metrics')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # General title with useful information
    title_parts = [model_name]
    if objective and objective != 'unknown':
        title_parts.append(objective)
    if criterion and criterion != 'unknown':
        title_parts.append(criterion)
    
    title = ' - '.join(title_parts)
    fig.suptitle(f'{title}', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    path_parts = experiment_dir.parts
    
    # Find key components in the path
    views_str = 'unknown'
    loss_str = ''
    component_str = ''
    layers_str = ''
    
    # Determine whether this is a CLIP layers study
    if 'layers_study' in path_parts:
        # CLIP layers: extract component (vision/text) and layers
        for i, part in enumerate(path_parts):
            if part in ['vision', 'text']:
                component_str = part
            if 'layer' in part.lower():
                layers_str = part.replace(' ', '_')
        views_str = experiment_dir.name
        if component_str and layers_str:
            filename = f"{layers_str}_{component_str}_{model_name}_CLIP_training.png"
        else:
            filename = f"{experiment_dir.name}_{model_name}_CLIP_training.png"
    else:
        # Vision models: extract loss function if present
        if 'metric learning' in path_parts:
            for i, part in enumerate(path_parts):
                if part in ['arcface', 'contrastive', 'multisimilarity', 'ntxent', 'triplet']:
                    loss_str = part
        
        # Extract views from directory name
        views_str = experiment_dir.name.replace(' ', '_').replace('+', 'plus')
        
        # Build filename
        name_parts = [views_str, model_name]
        if objective and objective != 'unknown':
            name_parts.append(objective)
        if loss_str:
            name_parts.append(loss_str)
        elif criterion and criterion != 'unknown' and criterion != 'CrossEntropyLoss':
            name_parts.append(criterion.lower())
        
        filename = '_'.join(name_parts) + '_training.png'
    
    # Clean special characters
    filename = filename.replace('', '').replace(':', '-').replace('/', '_')
    
    output_file = output_dir / filename
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   {filename}")
    return True


def main():
    """Main function."""
    results_dir = REPO_ROOT / 'results'
    
    if not results_dir.exists():
        print(f"Error: Directory does not exist {results_dir}")
        return
    
    print("=" * 80)
    print("TRAINING HISTORY VISUALIZATION")
    print("=" * 80)
    
    # Create output directory
    output_dir = REPO_ROOT / 'results/training_history_plots'
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"\nGenerating plots in: {output_dir}")
    
    # Search finetuning experiments in the full structure
    print("\nSearching for finetuning experiments...")
    finetune_folders = []
    
    # Search in vision models (resnet50, vitb32)
    models_dir = results_dir / 'models'
    if models_dir.exists():
        for model_dir in models_dir.iterdir():
            if model_dir.is_dir() and model_dir.name in ['resnet50', 'vitb32']:
                # Classification
                class_ft = model_dir / 'classification' / 'finetuning'
                if class_ft.exists():
                    for view_dir in class_ft.iterdir():
                        if view_dir.is_dir():
                            finetune_folders.append(view_dir)
                
                # Metric learning
                ml_dir = model_dir / 'metric learning'
                if ml_dir.exists():
                    for loss_dir in ml_dir.iterdir():
                        if loss_dir.is_dir():
                            loss_ft = loss_dir / 'finetuning'
                            if loss_ft.exists():
                                for view_dir in loss_ft.iterdir():
                                    if view_dir.is_dir():
                                        finetune_folders.append(view_dir)
            
            # CLIP layers study
            elif model_dir.name == 'clip':
                layers_study = model_dir / 'layers_study' / 'finetuning'
                if layers_study.exists():
                    for component_dir in layers_study.iterdir():
                        if component_dir.is_dir() and component_dir.name in ['vision', 'text']:
                            for layer_dir in component_dir.iterdir():
                                if layer_dir.is_dir():
                                    finetune_folders.append(layer_dir)
    
    print(f"\nFound {len(finetune_folders)} finetuning experiments")
    print("-" * 80)
    
    # Generate individual plots
    success_count = 0
    for folder in sorted(finetune_folders):
        result = plot_training_history(folder, output_dir)
        if result:
            success_count += 1
    
    print("-" * 80)
    print(f"\nSuccessfully generated plots: {success_count}/{len(finetune_folders)}")
    
    print("\n" + "=" * 80)
    print("VISUALIZATION COMPLETED")
    print("=" * 80)
    print(f"\nFiles saved in: {output_dir.absolute()}")


if __name__ == '__main__':
    main()






