#!/usr/bin/env python3
"""
Visualization script for One-Shot Sensitivity Analysis.
Analyzes model generalization capacity under one-shot test classes.
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from pathlib import Path
from collections import Counter

warnings.filterwarnings('ignore')

REPO_ROOT = Path(__file__).resolve().parents[2]

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'figure.figsize': (14, 10),
    'font.size': 10,
    'font.family': 'serif',
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'axes.titleweight': 'bold',
    'axes.labelweight': 'bold',
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.titlesize': 14,
    'figure.titleweight': 'bold',
})

COLOR_RESNET = '#08519c'
COLOR_VIT = '#3182bd'
COLOR_CLIP = '#08306b'
COLOR_PURE = '#2171b5'
COLOR_MIXED = '#6baed6'
COLOR_NOISE = '#9ecae1'

COLORS_MODEL = {
    'ResNet50_MultiSim': COLOR_RESNET,
    'ViT-B16_Contrastive': COLOR_VIT,
    'CLIP_10Images': COLOR_CLIP,
}

# Paths
SENSITIVITY_DIR = REPO_ROOT / 'results/sensitivity_test'
PLOTS_DIR = REPO_ROOT / 'results/analysis/plots/oneshot_sensitivity'
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Dataset information
TOTAL_REGULAR_CLASSES = 359
TOTAL_ONESHOT_CLASSES = 3239
TOTAL_CLASSES = 3598

ONESHOT_RATIOS = [0.0, 0.25, 0.5, 0.75, 1.0]

# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def load_json_data(json_path):
    """Load a clusters JSON file."""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def analyze_cluster_purity(cluster_classes):
    """
    Determine whether a cluster is pure or mixed.
    
    Args:
        cluster_classes: List of class names in the cluster
        
    Returns:
        tuple: (is_pure, dominant_class, class_counts)
    """
    if len(cluster_classes) == 0:
        return False, None, {}
    
    class_counts = Counter(cluster_classes)
    dominant_class = class_counts.most_common(1)[0][0]
    is_pure = len(class_counts) == 1
    
    return is_pure, dominant_class, dict(class_counts)

def get_oneshot_classes_used(ratio):
    """Calculate how many one-shot classes are used for a given ratio."""
    return int(TOTAL_ONESHOT_CLASSES * ratio)

def analyze_json_clusters(json_data):
    """
    Analyze a clusters JSON file and extract verifiable metrics.
    
    Args:
        json_data: Dictionary with JSON data
        
    Returns:
        dict: Dictionary with all analysis metrics
    """
    clusters = json_data['clusters']
    metrics = json_data['metrics']
    
    n_pure_clusters = 0
    n_mixed_clusters = 0
    total_samples = 0
    noise_samples = metrics['n_noise']
    
    for cluster_id, class_list in clusters.items():
        if cluster_id != '-1':  # Exclude noise
            # Analyze purity
            is_pure, dominant_class, class_counts = analyze_cluster_purity(class_list)
            
            if is_pure:
                n_pure_clusters += 1
            else:
                n_mixed_clusters += 1
            
            total_samples += len(class_list)
    
    total_clusters = n_pure_clusters + n_mixed_clusters
    pure_percentage = (n_pure_clusters / total_clusters * 100) if total_clusters > 0 else 0
    mixed_percentage = (n_mixed_clusters / total_clusters * 100) if total_clusters > 0 else 0
    
    # Percentage of samples in noise
    total_with_noise = total_samples + noise_samples
    noise_percentage = (noise_samples / total_with_noise * 100) if total_with_noise > 0 else 0
    
    return {
        'ari': metrics['ari'],
        'silhouette': metrics['silhouette'],
        'n_clusters': metrics['n_clusters'],
        'n_noise_samples': noise_samples,
        'n_pure_clusters': n_pure_clusters,
        'n_mixed_clusters': n_mixed_clusters,
        'pure_percentage': pure_percentage,
        'mixed_percentage': mixed_percentage,
        'noise_percentage': noise_percentage,
        'total_samples': total_with_noise,
    }

# ============================================================================
# DATA LOADING AND PROCESSING
# ============================================================================

print("="*80)
print("GENERATING VISUALIZATIONS - ONE-SHOT SENSITIVITY ANALYSIS")
print("="*80)
print()
print(f"Dataset: {TOTAL_CLASSES} total classes")
print(f"  - Regular (8 imgs): {TOTAL_REGULAR_CLASSES}")
print(f"  - One-shot (<8 imgs): {TOTAL_ONESHOT_CLASSES}")
print()

# Load CSV with general metrics
csv_path = SENSITIVITY_DIR / 'oneshot_sensitivity_results.csv'
df_summary = pd.read_csv(csv_path)

# Process JSON files to extract detailed metrics
detailed_results = []

for _, row in df_summary.iterrows():
    model = row['Model']
    ratio = row['OneShot_Ratio']
    json_filename = row['JSON_File']
    json_path = SENSITIVITY_DIR / json_filename
    
    print(f"Processing: {model} - Ratio {ratio}")
    
    # Load JSON
    json_data = load_json_data(json_path)
    
    # Compute available one-shot classes
    n_oneshot_available = get_oneshot_classes_used(ratio)
    
    # Analyze clusters (without trying to identify specific one-shot classes)
    analysis = analyze_json_clusters(json_data)
    
    detailed_results.append({
        'Model': model,
        'Ratio': ratio,
        'N_OneShot_Available': n_oneshot_available,
        **analysis
    })

df_detailed = pd.DataFrame(detailed_results)

# Save detailed results
df_detailed.to_csv(PLOTS_DIR / 'oneshot_detailed_metrics.csv', index=False)
print(f"\n Detailed metrics saved in: {PLOTS_DIR / 'oneshot_detailed_metrics.csv'}")
print()

# ============================================================================
# VISUALIZATIONS BY MODEL
# ============================================================================

models = df_detailed['Model'].unique()

for model in models:
    print(f"Generating visualizations for: {model}")
    
    df_model = df_detailed[df_detailed['Model'] == model].sort_values('Ratio')
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f'One-Shot Sensitivity Analysis: {model}', fontsize=18, fontweight='bold', y=0.98)
    
    # 3x2 grid (6 subplots)
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    color = COLORS_MODEL.get(model, '#969696')
    
    # --- SUBPLOT 1: ARI vs Ratio ---
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(df_model['Ratio'], df_model['ari'], marker='o', linewidth=2.5, 
             markersize=10, color=color, label='ARI')
    ax1.set_title('ARI vs One-Shot Ratio', fontweight='bold', fontsize=13)
    ax1.set_xlabel('One-Shot Ratio', fontsize=12)
    ax1.set_ylabel('ARI', fontsize=12)
    ax1.grid(alpha=0.3)
    ax1.set_xticks(ONESHOT_RATIOS)
    for x, y in zip(df_model['Ratio'], df_model['ari']):
        ax1.text(x, y+0.0005, f'{y:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # --- SUBPLOT 2: Silhouette Score ---
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(df_model['Ratio'], df_model['silhouette'], marker='s', linewidth=2.5,
             markersize=10, color=color, label='Silhouette')
    ax2.set_title('Silhouette Score vs Ratio', fontweight='bold', fontsize=13)
    ax2.set_xlabel('One-Shot Ratio', fontsize=12)
    ax2.set_ylabel('Silhouette Score', fontsize=12)
    ax2.grid(alpha=0.3)
    ax2.set_xticks(ONESHOT_RATIOS)
    for x, y in zip(df_model['Ratio'], df_model['silhouette']):
        ax2.text(x, y+0.01, f'{y:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # --- SUBPLOT 3: Number of Clusters ---
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(df_model['Ratio'], df_model['n_clusters'], marker='D', linewidth=2.5,
             markersize=10, color=color)
    ax3.set_title('Number of Clusters vs Ratio', fontweight='bold', fontsize=13)
    ax3.set_xlabel('One-Shot Ratio', fontsize=12)
    ax3.set_ylabel('Number of Clusters', fontsize=12)
    ax3.grid(alpha=0.3)
    ax3.set_xticks(ONESHOT_RATIOS)
    for x, y in zip(df_model['Ratio'], df_model['n_clusters']):
        ax3.text(x, y+20, f'{int(y)}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # --- SUBPLOT 4: % Pure Clusters vs Mixed ---
    ax4 = fig.add_subplot(gs[1, 1])
    x_pos = np.arange(len(df_model))
    width = 0.35
    ax4.bar(x_pos - width/2, df_model['pure_percentage'], width, label='Pure',
            color=COLOR_PURE, edgecolor='black', linewidth=1.5)
    ax4.bar(x_pos + width/2, df_model['mixed_percentage'], width, label='Mixed',
            color=COLOR_MIXED, edgecolor='black', linewidth=1.5)
    ax4.set_title('Distribution Pure vs Mixed', fontweight='bold', fontsize=13)
    ax4.set_xlabel('One-Shot Ratio', fontsize=12)
    ax4.set_ylabel('Percentage (%)', fontsize=12)
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels([f'{r:.2f}' for r in df_model['Ratio']])
    ax4.legend(fontsize=10)
    ax4.grid(axis='y', alpha=0.3)
    
    # --- SUBPLOT 5: Number of Pure and Mixed Clusters ---
    ax5 = fig.add_subplot(gs[2, 0])
    x_pos = np.arange(len(df_model))
    width = 0.35
    ax5.bar(x_pos - width/2, df_model['n_pure_clusters'], width, label='Pure',
            color=COLOR_PURE, edgecolor='black', linewidth=1.5)
    ax5.bar(x_pos + width/2, df_model['n_mixed_clusters'], width, label='Mixed',
            color=COLOR_MIXED, edgecolor='black', linewidth=1.5)
    ax5.set_title('Cluster Count', fontweight='bold', fontsize=13)
    ax5.set_xlabel('One-Shot Ratio', fontsize=12)
    ax5.set_ylabel('Number of Clusters', fontsize=12)
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels([f'{r:.2f}' for r in df_model['Ratio']])
    ax5.legend(fontsize=10)
    ax5.grid(axis='y', alpha=0.3)
    
    # --- SUBPLOT 6: Samples in Noise ---
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.bar(df_model['Ratio'], df_model['noise_percentage'], color=COLOR_NOISE,
            edgecolor='black', alpha=0.8, linewidth=1.5)
    ax6.set_title('% Samples Classified as Noise', fontweight='bold', fontsize=13)
    ax6.set_xlabel('One-Shot Ratio', fontsize=12)
    ax6.set_ylabel('Percentage (%)', fontsize=12)
    ax6.grid(axis='y', alpha=0.3)
    ax6.set_xticks(ONESHOT_RATIOS)
    for x, y in zip(df_model['Ratio'], df_model['noise_percentage']):
        ax6.text(x, y+0.5, f'{y:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Save figure
    safe_model_name = model.replace('/', '-').replace(' ', '_')
    plt.savefig(PLOTS_DIR / f'oneshot_{safe_model_name}.png', dpi=300, bbox_inches='tight')
    plt.close()

# ============================================================================
# COMPARATIVE VISUALIZATION: ALL MODELS
# ============================================================================

print("\nGenerating comparative visualization...")

fig = plt.figure(figsize=(18, 10))
fig.suptitle('One-Shot Sensitivity: Comparison Across Models', 
             fontsize=20, fontweight='bold', y=0.96)

gs = fig.add_gridspec(2, 3, hspace=0.25, wspace=0.3)

# --- SUBPLOT 1: Comparative ARI ---
ax1 = fig.add_subplot(gs[0, 0])
for model in models:
    df_model = df_detailed[df_detailed['Model'] == model].sort_values('Ratio')
    color = COLORS_MODEL.get(model, '#969696')
    ax1.plot(df_model['Ratio'], df_model['ari'], marker='o', linewidth=2.5,
             markersize=10, color=color, label=model, alpha=0.85)
ax1.set_title('ARI vs One-Shot Ratio', fontweight='bold', fontsize=14)
ax1.set_xlabel('One-Shot Ratio', fontsize=12)
ax1.set_ylabel('ARI', fontsize=12)
ax1.legend(fontsize=11, loc='best')
ax1.grid(alpha=0.3)
ax1.set_xticks(ONESHOT_RATIOS)

# --- SUBPLOT 2: Comparative Number of Clusters ---
ax2 = fig.add_subplot(gs[0, 1])
for model in models:
    df_model = df_detailed[df_detailed['Model'] == model].sort_values('Ratio')
    color = COLORS_MODEL.get(model, '#969696')
    ax2.plot(df_model['Ratio'], df_model['n_clusters'], marker='s', linewidth=2.5,
             markersize=10, color=color, label=model, alpha=0.85)
ax2.set_title('Number of Clusters vs Ratio', fontweight='bold', fontsize=14)
ax2.set_xlabel('One-Shot Ratio', fontsize=12)
ax2.set_ylabel('# Clusters', fontsize=12)
ax2.legend(fontsize=11, loc='best')
ax2.grid(alpha=0.3)
ax2.set_xticks(ONESHOT_RATIOS)

# --- SUBPLOT 3: Comparative Silhouette ---
ax3 = fig.add_subplot(gs[0, 2])
for model in models:
    df_model = df_detailed[df_detailed['Model'] == model].sort_values('Ratio')
    color = COLORS_MODEL.get(model, '#969696')
    ax3.plot(df_model['Ratio'], df_model['silhouette'], marker='^', linewidth=2.5,
             markersize=10, color=color, label=model, alpha=0.85)
ax3.set_title('Silhouette Score vs Ratio', fontweight='bold', fontsize=14)
ax3.set_xlabel('One-Shot Ratio', fontsize=12)
ax3.set_ylabel('Silhouette', fontsize=12)
ax3.legend(fontsize=11, loc='best')
ax3.grid(alpha=0.3)
ax3.set_xticks(ONESHOT_RATIOS)

# --- SUBPLOT 4: % Pure Clusters ---
ax4 = fig.add_subplot(gs[1, 0])
for model in models:
    df_model = df_detailed[df_detailed['Model'] == model].sort_values('Ratio')
    color = COLORS_MODEL.get(model, '#969696')
    ax4.plot(df_model['Ratio'], df_model['pure_percentage'], marker='o', linewidth=2.5,
             markersize=10, color=color, label=model, alpha=0.85)
ax4.set_title('% Pure Clusters', fontweight='bold', fontsize=14)
ax4.set_xlabel('One-Shot Ratio', fontsize=12)
ax4.set_ylabel('% Pure', fontsize=12)
ax4.legend(fontsize=11, loc='best')
ax4.grid(alpha=0.3)
ax4.set_xticks(ONESHOT_RATIOS)

# --- SUBPLOT 5: Number of Pure Clusters ---
ax5 = fig.add_subplot(gs[1, 1])
for model in models:
    df_model = df_detailed[df_detailed['Model'] == model].sort_values('Ratio')
    color = COLORS_MODEL.get(model, '#969696')
    ax5.plot(df_model['Ratio'], df_model['n_pure_clusters'], marker='D', linewidth=2.5,
             markersize=10, color=color, label=model, alpha=0.85)
ax5.set_title('Number of Pure Clusters', fontweight='bold', fontsize=14)
ax5.set_xlabel('One-Shot Ratio', fontsize=12)
ax5.set_ylabel('# Pure Clusters', fontsize=12)
ax5.legend(fontsize=11, loc='best')
ax5.grid(alpha=0.3)
ax5.set_xticks(ONESHOT_RATIOS)

# --- SUBPLOT 6: Ranking by % Pure (Ratio 1.0) ---
ax6 = fig.add_subplot(gs[1, 2])
df_final = df_detailed[df_detailed['Ratio'] == 1.0].copy()
df_final = df_final.sort_values('pure_percentage', ascending=True)
colors = [COLORS_MODEL.get(m, '#969696') for m in df_final['Model']]
bars = ax6.barh(range(len(df_final)), df_final['pure_percentage'], 
                color=colors, edgecolor='black', alpha=0.8, linewidth=1.5)
ax6.set_yticks(range(len(df_final)))
ax6.set_yticklabels(df_final['Model'], fontsize=9)
ax6.set_xlabel('% Pure Clusters', fontsize=12)
ax6.set_title('Ranking: % Pure (Ratio 1.0)', fontweight='bold', fontsize=14)
ax6.grid(axis='x', alpha=0.3)
for i, v in enumerate(df_final['pure_percentage']):
    ax6.text(v+0.5, i, f'{v:.1f}%', va='center', ha='left', fontweight='bold', fontsize=9)

plt.savefig(PLOTS_DIR / 'oneshot_comparative_all_models.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# SUMMARY TABLE
# ============================================================================

print("\nGenerating summary table...")

# Table by ratio
summary_table = []
for ratio in ONESHOT_RATIOS:
    row = {'Ratio': ratio, 'OneShot_Classes': get_oneshot_classes_used(ratio)}
    for model in models:
        df_model_ratio = df_detailed[(df_detailed['Model'] == model) & 
                                      (df_detailed['Ratio'] == ratio)]
        if len(df_model_ratio) > 0:
            r = df_model_ratio.iloc[0]
            row[f'{model}_Pure'] = int(r['n_pure_clusters'])
            row[f'{model}_Mixed'] = int(r['n_mixed_clusters'])
            row[f'{model}_Pure%'] = f"{r['pure_percentage']:.1f}%"
            row[f'{model}_ARI'] = f"{r['ari']:.4f}"
    summary_table.append(row)

df_summary_table = pd.DataFrame(summary_table)
df_summary_table.to_csv(PLOTS_DIR / 'oneshot_summary_table.csv', index=False)

# ============================================================================
# FINAL REPORT
# ============================================================================

print()
print("="*80)
print("VISUALIZATIONS COMPLETED")
print("="*80)
print(f" Plots saved in: {PLOTS_DIR}")
print()
print("Generated files:")
for model in models:
    safe_name = model.replace('/', '-').replace(' ', '_')
    print(f"   oneshot_{safe_name}.png (6 subplots)")
print(f"   oneshot_comparative_all_models.png (6 comparative subplots)")
print(f"   oneshot_detailed_metrics.csv (complete metrics)")
print(f"   oneshot_summary_table.csv (summary table)")
print()
print("Summary of one-shot classes by ratio:")
for ratio in ONESHOT_RATIOS:
    n_oneshot = get_oneshot_classes_used(ratio)
    print(f"  - Ratio {ratio:.2f}: {n_oneshot} one-shot classes in test")
print()
print("="*80)






