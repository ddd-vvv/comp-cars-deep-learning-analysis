#!/usr/bin/env python3
"""
Visualization script for analysis of CLIP Min Images Study.
Generates progression plots by minimum number of images (4, 6, 8, 10, 12).
Based on visualize_clip_layers.py and visualize_clip_phase_order.py.

IMPORTANT: Replicates all plots from both scripts, adapted to the min_images experiment.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

REPO_ROOT = Path(__file__).resolve().parents[2]

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'figure.figsize': (12, 8),
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

COLOR_VISION = '#08519c'          # Very dark blue
COLOR_TEXT = '#6baed6'            # Medium blue
COLOR_RECALL1 = '#08519c'         # Very dark blue
COLOR_RECALL3 = '#3182bd'         # Medium blue
COLOR_RECALL5 = '#9ecae1'         # Light blue
COLOR_PURE = '#2171b5'            # Dark blue (pure clusters)
COLOR_MIXED = '#969696'           # Medium gray (mixed clusters)
COLOR_POSITIVE = '#2171b5'        # Dark blue (positive values)
COLOR_NEGATIVE = '#969696'        # Gray (negative values)
COLOR_HIGHLIGHT = '#08306b'       # Darkest blue (highlights)
COLOR_BASELINE = '#969696'        # Gray for baseline

CORR_COLORS = {
    'high': '#08519c',    # High correlation (>0.5)
    'medium': '#6baed6',  # Medium correlation (>0)
    'low': '#969696'      # Low/negative correlation
}

COLORS_MIN_IMAGES = {
    4: '#08306b',   # Very very dark blue (critical)
    6: '#08519c',   # Very dark blue (critical)
    8: '#3182bd',   # Medium blue (base/standard)
    10: '#6baed6',  # Light blue (rich)
    12: '#9ecae1'   # Very light blue (rich)
}

LABELS_MIN_IMAGES = {
    4: '4 images',
    6: '6 images',
    8: '8 images',
    10: '10 images',
    12: '12 images'
}

# Paths
RESULTS_DIR = REPO_ROOT / 'results/analysis'
PLOTS_DIR = REPO_ROOT / 'results/analysis/plots/clip_min_images'
TRAINING_PLOTS_DIR = REPO_ROOT / 'results/visualizations/training_history_plots'
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
TRAINING_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Load data
print("="*80)
print("GENERATING VISUALIZATIONS - CLIP MIN IMAGES STUDY")
print("="*80)
print()

df = pd.read_csv(RESULTS_DIR / 'data/clip_min_images/clip_min_images_results.csv')
print(f"Loaded data: {len(df)} configurations\n")

# Sort by min_images
df = df.sort_values('min_images')

# Create labels with view
def create_label(row):
    """Create labels with format: '4 img (F)' or '4 img (B)'"""
    min_imgs = int(row['min_images'])
    views = row['views_str']
    view_code = '(F)' if views == 'front' else '(B)'
    return f"{min_imgs} img {view_code}"

df['label'] = df.apply(create_label, axis=1)

# Load training histories
def load_training_histories():
    """Load training histories from original JSON files."""
    base_path = REPO_ROOT / 'results/models/clip/min_images_study/finetuning'
    histories = {}
    
    for min_imgs in [4, 6, 8, 10, 12]:
        for view_folder in ['front', 'front + rear']:
            results_file = base_path / f'{min_imgs} images' / view_folder / 'results.json'
            if results_file.exists():
                with open(results_file, 'r') as f:
                    data = json.load(f)
                    key = (min_imgs, view_folder)
                    histories[key] = data.get('finetuning', {}).get('training_history', {})
                break  # Use the first available view
    
    return histories

def load_embeddings_data():
    """Load embeddings/clustering data from original JSON files."""
    base_path = REPO_ROOT / 'results/models/clip/min_images_study/embeddings'
    embeddings_data = {}
    
    for min_imgs in [4, 6, 8, 10, 12]:
        results_file = base_path / f'{min_imgs}_images' / 'results.json'
        if results_file.exists():
            with open(results_file, 'r') as f:
                data = json.load(f)
                embeddings_data[min_imgs] = data
    
    return embeddings_data

histories = load_training_histories()
embeddings_data = load_embeddings_data()

# ============================================================================
# 1. CRITICAL vs BASE vs RICH SCENARIO COMPARISON
# ============================================================================
print("[1/10] Critical vs base vs rich scenario comparison...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('CLIP: Robustness Under Data Scarcity (Min Images Study)', fontsize=16, fontweight='bold')

# Average recall by scenario
ax = axes[0, 0]
x = np.arange(len(df))
width = 0.25

recall1 = df['finetuned_recall@1'].values
recall3 = df['finetuned_recall@3'].values
recall5 = df['finetuned_recall@5'].values

bars1 = ax.bar(x - width, recall1, width, label='Recall@1', color=COLOR_RECALL1, edgecolor='black', linewidth=0.5)
bars2 = ax.bar(x, recall3, width, label='Recall@3', color=COLOR_RECALL3, edgecolor='black', linewidth=0.5)
bars3 = ax.bar(x + width, recall5, width, label='Recall@5', color=COLOR_RECALL5, edgecolor='black', linewidth=0.5)

ax.set_title('Recall Final by Scenario', fontweight='bold')
ax.set_ylabel('Recall (%)')
ax.set_xlabel('Min Images')
ax.set_xticks(x)
ax.set_xticklabels(df['label'].tolist(), rotation=45, ha='right', fontsize=8)
ax.legend()
ax.grid(axis='y', alpha=0.3)

# ARI by scenario
ax = axes[0, 1]
ari_values = df['finetuned_ari'].values
colors = [COLORS_MIN_IMAGES.get(m, COLOR_VISION) for m in df['min_images']]
bars = ax.bar(x, ari_values, color=colors, edgecolor='black', linewidth=0.5)
ax.set_title('ARI by Data Scenario', fontweight='bold')
ax.set_ylabel('ARI')
ax.set_xlabel('Min Images')
ax.set_xticks(x)
ax.set_xticklabels(df['label'].tolist(), rotation=45, ha='right', fontsize=8)
ax.grid(axis='y', alpha=0.3)
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.3f}', ha='center', va='bottom', fontsize=8)

# % Pure clusters
ax = axes[1, 0]
pure_pct = df['finetuned_pure_percentage'].values
bars = ax.bar(x, pure_pct, color=colors, edgecolor='black', linewidth=0.5)
ax.set_title('% Pure Clusters by Scenario', fontweight='bold')
ax.set_ylabel('% Pure Clusters')
ax.set_xlabel('Min Images')
ax.set_xticks(x)
ax.set_xticklabels(df['label'].tolist(), rotation=45, ha='right', fontsize=8)
ax.grid(axis='y', alpha=0.3)
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f}%', ha='center', va='bottom', fontsize=7)

# Recall@1 improvement vs baseline
ax = axes[1, 1]
improvements = df['recall@1_improvement'].values
bar_colors = [COLOR_POSITIVE if imp > 0 else COLOR_NEGATIVE for imp in improvements]
bars = ax.bar(x, improvements, color=bar_colors, edgecolor='black', linewidth=0.5)
ax.set_title('Improvement Recall@1 (Fine-tuned - Baseline)', fontweight='bold')
ax.set_ylabel('Improvement (%)')
ax.set_xlabel('Min Images')
ax.set_xticks(x)
ax.set_xticklabels(df['label'].tolist(), rotation=45, ha='right', fontsize=8)
ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
ax.grid(axis='y', alpha=0.3)
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'+{height:.1f}' if height > 0 else f'{height:.1f}', 
            ha='center', va='bottom' if height > 0 else 'top', fontsize=7)

plt.tight_layout()
plt.savefig(PLOTS_DIR / '01_scenario_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# 2. RECALL PROGRESSION DURING TRAINING
# ============================================================================
print("[2/10] Recall progression during training...")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Recall Progression by Data Scenario', fontsize=16, fontweight='bold')

for idx, min_imgs in enumerate([4, 6, 8, 10, 12]):
    row_idx = idx // 3
    col_idx = idx % 3
    ax = axes[row_idx, col_idx]
    
    # Find history for this min_images (key is tuple)
    history = None
    for key, hist in histories.items():
        if isinstance(key, tuple) and key[0] == min_imgs:
            history = hist
            break
    
    if history is None:
        history = {}
    
    label = LABELS_MIN_IMAGES.get(min_imgs, f'{min_imgs} images')
    
    # Detect phase keys (can vary)
    phase_keys = list(history.keys())
    if len(phase_keys) >= 2:
        phase_1_key = phase_keys[0]
        phase_2_key = phase_keys[1]
    else:
        phase_1_key = 'phase_1_vision'
        phase_2_key = 'pahse_2_text'
    
    phase_1 = history.get(phase_1_key, {})
    phase_2 = history.get(phase_2_key, {})
    
    val_recall1_p1 = phase_1.get('val_recall@1', [])
    val_recall5_p1 = phase_1.get('val_recall@5', [])
    val_recall1_p2 = phase_2.get('val_recall@1', [])
    val_recall5_p2 = phase_2.get('val_recall@5', [])
    
    # Concatenate epochs
    all_recall1 = val_recall1_p1 + val_recall1_p2
    all_recall5 = val_recall5_p1 + val_recall5_p2
    epochs = list(range(1, len(all_recall1) + 1))
    
    if all_recall1:
        ax.plot(epochs, all_recall1, marker='o', linewidth=2, markersize=4, 
                label='Recall@1', color=COLOR_RECALL1)
    if all_recall5:
        ax.plot(epochs, all_recall5, marker='s', linewidth=2, markersize=4, 
                label='Recall@5', color=COLOR_RECALL5)
    
    # Vertical line for phase transition
    if val_recall1_p1:
        ax.axvline(x=len(val_recall1_p1) + 0.5, color='gray', linestyle=':', 
                   alpha=0.7, linewidth=2)
    
    ax.set_xlabel('Epoch', fontweight='bold')
    ax.set_ylabel('Recall (%)', fontweight='bold')
    ax.set_title(label, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

# Hide the empty subplot (position [1, 2])
axes[1, 2].axis('off')

plt.tight_layout()
plt.savefig(PLOTS_DIR / '02_recall_progression.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# 3. PROGRESSION BY MIN_IMAGES - RECALL (LINES)
# ============================================================================
print("[3/10] Recall progression by min_images...")

fig, ax = plt.subplots(figsize=(14, 7))
fig.suptitle('Recall Progression by Data Availability', fontsize=16, fontweight='bold')

min_images_vals = df['min_images'].values

ax.plot(min_images_vals, df['finetuned_recall@1'], 
        marker='o', linewidth=3, markersize=10, label='Recall@1', color=COLOR_RECALL1)
ax.plot(min_images_vals, df['finetuned_recall@3'], 
        marker='s', linewidth=3, markersize=10, label='Recall@3', color=COLOR_RECALL3)
ax.plot(min_images_vals, df['finetuned_recall@5'], 
        marker='^', linewidth=3, markersize=10, label='Recall@5', color=COLOR_RECALL5)

ax.set_xlabel('Minimum Number of Images per Class', fontweight='bold')
ax.set_ylabel('Recall (%)', fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)
ax.set_xticks([4, 6, 8, 10, 12])

# Mark the maximum
max_idx = df['finetuned_recall@1'].idxmax()
max_min_imgs = df.loc[max_idx, 'min_images']
max_recall = df.loc[max_idx, 'finetuned_recall@1']
ax.scatter([max_min_imgs], [max_recall], color=COLOR_HIGHLIGHT, s=200, zorder=5, marker='*', 
           edgecolors='black', linewidths=2)
ax.annotate(f'Maximum: {max_recall:.2f}%\n({int(max_min_imgs)} images)', 
            xy=(max_min_imgs, max_recall), xytext=(max_min_imgs+0.5, max_recall-2),
            fontweight='bold', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='#deebf7', alpha=0.8),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

plt.tight_layout()
plt.savefig(PLOTS_DIR / '03_recall_vs_min_images.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# 4. PROGRESSION BY MIN_IMAGES - ARI (LINES)
# ============================================================================
print("[4/10] ARI progression by min_images...")

fig, ax = plt.subplots(figsize=(14, 7))
fig.suptitle('ARI Progression by Data Availability', fontsize=16, fontweight='bold')

# Group by min_images and compute means
df_by_min_images = df.groupby('min_images').agg({
    'finetuned_ari': 'mean',
    'finetuned_recall@1': 'mean'
}).reset_index()

# Filter rows with available ARI
df_ari = df_by_min_images[df_by_min_images['finetuned_ari'].notna()].copy()
if len(df_ari) > 0:
    min_images_vals = df_ari['min_images'].tolist()
    ari_vals = df_ari['finetuned_ari'].tolist()
    
    ax.plot(min_images_vals, ari_vals, 
            marker='o', linewidth=3, markersize=12, color=COLOR_VISION, label='ARI')
    ax.fill_between(min_images_vals, ari_vals, alpha=0.3, color=COLOR_VISION)
    
    # Mark the maximum
    max_idx = df_ari['finetuned_ari'].idxmax()
    max_min_imgs = df_ari.loc[max_idx, 'min_images']
    max_ari = df_ari.loc[max_idx, 'finetuned_ari']
    ax.scatter([max_min_imgs], [max_ari], color=COLOR_HIGHLIGHT, s=200, zorder=5, marker='*', 
               edgecolors='black', linewidths=2)
    ax.annotate(f'Maximum: {max_ari:.4f}\n({int(max_min_imgs)} images)', 
                xy=(max_min_imgs, max_ari), xytext=(max_min_imgs+0.5, max_ari-0.02),
                fontweight='bold', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='#deebf7', alpha=0.8),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
else:
    ax.text(0.5, 0.5, 'ARI data not available', 
            ha='center', va='center', fontsize=14, transform=ax.transAxes)

ax.set_xlabel('Minimum Number of Images per Class', fontweight='bold')
ax.set_ylabel('ARI', fontweight='bold')
ax.grid(alpha=0.3)
ax.set_xticks([4, 6, 8, 10, 12])

plt.tight_layout()
plt.savefig(PLOTS_DIR / '04_ari_vs_min_images.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# 5. BASELINE vs FINETUNED BY SCENARIO
# ============================================================================
print("[5/10] Baseline vs fine-tuned by scenario...")

fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Baseline vs Fine-tuned: Comparison by Scenario', fontsize=16, fontweight='bold')

x = np.arange(len(df))
width = 0.35

# Recall@1
ax = axes[0, 0]
baseline_r1 = df['baseline_recall@1'].fillna(0).values
finetuned_r1 = df['finetuned_recall@1'].fillna(0).values

ax.bar(x - width/2, baseline_r1, width, label='Baseline', color=COLOR_BASELINE, edgecolor='black')
colors_ft = [COLORS_MIN_IMAGES.get(m, COLOR_VISION) for m in df['min_images']]
ax.bar(x + width/2, finetuned_r1, width, label='Fine-tuned', color=colors_ft, edgecolor='black')
ax.set_xlabel('Scenario', fontweight='bold')
ax.set_ylabel('Recall@1 (%)', fontweight='bold')
ax.set_title('Recall@1: Baseline vs Fine-tuned', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(df['label'].tolist(), rotation=45, ha='right', fontsize=8)
ax.legend()
ax.grid(axis='y', alpha=0.3)

# ARI
ax = axes[0, 1]
baseline_ari = df['baseline_ari'].fillna(0).values
finetuned_ari = df['finetuned_ari'].fillna(0).values

ax.bar(x - width/2, baseline_ari, width, label='Baseline', color=COLOR_BASELINE, edgecolor='black')
ax.bar(x + width/2, finetuned_ari, width, label='Fine-tuned', color=colors_ft, edgecolor='black')
ax.set_xlabel('Scenario', fontweight='bold')
ax.set_ylabel('ARI', fontweight='bold')
ax.set_title('ARI: Baseline vs Fine-tuned', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(df['label'].tolist(), rotation=45, ha='right', fontsize=8)
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Metric improvement
ax = axes[1, 0]
ari_improvement = finetuned_ari - baseline_ari
recall_improvement = df['recall@1_improvement'].fillna(0).values

x_pos = np.arange(len(df))
ax.bar(x_pos - width/2, recall_improvement, width, label=' Recall@1', color=COLOR_RECALL1, edgecolor='black')
ax.bar(x_pos + width/2, ari_improvement * 100, width, label='ARI x100', color=COLOR_RECALL5, edgecolor='black')
ax.set_xlabel('Scenario', fontweight='bold')
ax.set_ylabel('Improvement', fontweight='bold')
ax.set_title('Improvement: Fine-tuned - Baseline', fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(df['label'].tolist(), rotation=45, ha='right', fontsize=8)
ax.legend()
ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
ax.grid(axis='y', alpha=0.3)

# Degradation from best scenario
ax = axes[1, 1]
best_ari = df['finetuned_ari'].max()
best_min_imgs = df.loc[df['finetuned_ari'].idxmax(), 'min_images']
degradation = [(best_ari - ari) / best_ari * 100 for ari in df['finetuned_ari']]

bars = ax.bar(x, degradation, color=[COLOR_NEGATIVE if d > 0 else COLOR_POSITIVE for d in degradation], 
              edgecolor='black')
ax.axhline(y=0, color='black', linestyle='--', linewidth=2)
ax.set_xlabel('Scenario', fontweight='bold')
ax.set_ylabel('Degradation ARI (%)', fontweight='bold')
ax.set_title(f'Degradation from Best Scenario ({int(best_min_imgs)} images)', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(df['label'].tolist(), rotation=45, ha='right', fontsize=8)
ax.grid(axis='y', alpha=0.3)

for bar in bars:
    height = bar.get_height()
    if height != 0:
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom' if height > 0 else 'top', fontweight='bold')

plt.tight_layout()
plt.savefig(PLOTS_DIR / '05_baseline_vs_finetuned.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# 6. CLUSTERING QUALITY BY SCENARIO
# ============================================================================
print("[6/10] Clustering quality by scenario...")

fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Clustering Quality by Scenario of Data', fontsize=16, fontweight='bold')

x = np.arange(len(df))

# Pure vs mixed clusters
ax = axes[0, 0]
width = 0.35
pure_clusters = df['finetuned_n_pure_clusters'].fillna(0).values
mixed_clusters = df['finetuned_n_mixed_clusters'].fillna(0).values

bars1 = ax.bar(x - width/2, pure_clusters, width, label='Pure Clusters', color=COLOR_PURE, edgecolor='black')
bars2 = ax.bar(x + width/2, mixed_clusters, width, label='Mixed Clusters', color=COLOR_MIXED, edgecolor='black')
ax.set_xlabel('Min Images', fontweight='bold')
ax.set_ylabel('Number of Clusters')
ax.set_title('Pure Clusters vs Mixed', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(df['label'].tolist(), rotation=45, ha='right', fontsize=8)
ax.legend()
ax.grid(axis='y', alpha=0.3)

for bar in bars1:
    if bar.get_height() > 0:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{int(bar.get_height())}', ha='center', va='bottom', fontweight='bold')
for bar in bars2:
    if bar.get_height() > 0:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{int(bar.get_height())}', ha='center', va='bottom', fontweight='bold')

# Purity
ax = axes[0, 1]
purity_vals = df['finetuned_purity'].fillna(0).values * 100
bars = ax.bar(x, purity_vals, color=colors_ft, edgecolor='black')
ax.set_xlabel('Min Images', fontweight='bold')
ax.set_ylabel('Purity (%)', fontweight='bold')
ax.set_title('Purity Score', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(df['label'].tolist(), rotation=45, ha='right', fontsize=8)
ax.grid(axis='y', alpha=0.3)
for bar in bars:
    if bar.get_height() > 0:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{bar.get_height():.1f}%', ha='center', va='bottom', fontweight='bold')

# NMI
ax = axes[1, 0]
nmi_vals = df['finetuned_nmi'].fillna(0).values
bars = ax.bar(x, nmi_vals, color=colors_ft, edgecolor='black')
ax.set_xlabel('Min Images', fontweight='bold')
ax.set_ylabel('NMI', fontweight='bold')
ax.set_title('Normalized Mutual Information', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(df['label'].tolist(), rotation=45, ha='right', fontsize=8)
ax.grid(axis='y', alpha=0.3)
for bar in bars:
    if bar.get_height() > 0:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{bar.get_height():.4f}', ha='center', va='bottom', fontweight='bold')

# Problematic classes
ax = axes[1, 1]
overlapping = df['finetuned_n_overlapping_classes'].fillna(0).values
bars = ax.bar(x, overlapping, color=colors_ft, edgecolor='black')
ax.set_xlabel('Min Images', fontweight='bold')
ax.set_ylabel('Number of Classes', fontweight='bold')
ax.set_title('Classes in Multiple Clusters', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(df['label'].tolist(), rotation=45, ha='right', fontsize=8)
ax.grid(axis='y', alpha=0.3)
for bar in bars:
    if bar.get_height() > 0:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{int(bar.get_height())}', ha='center', va='bottom', fontweight='bold', fontsize=10)

plt.tight_layout()
plt.savefig(PLOTS_DIR / '06_clustering_quality.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# 7. MIN_IMAGES vs METRICS CORRELATION
# ============================================================================
print("[7/10] Correlation analysis: min_images vs metrics...")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Correlation: Data Availability vs Metrics', fontsize=16, fontweight='bold')

# Correlations with Recall@1 and ARI
ax = axes[0]
corr_recall = df[['min_images', 'finetuned_recall@1']].corr().iloc[0, 1]
corr_ari = df[['min_images', 'finetuned_ari']].corr().iloc[0, 1]
corr_pure = df[['min_images', 'finetuned_pure_percentage']].corr().iloc[0, 1]

metrics = ['Recall@1', 'ARI', '% Pure']
correlations = [corr_recall, corr_ari, corr_pure]
colors_corr = [CORR_COLORS['high'] if c > 0.5 else CORR_COLORS['medium'] if c > 0 else CORR_COLORS['low'] 
               for c in correlations]

bars = ax.barh(metrics, correlations, color=colors_corr, edgecolor='black', linewidth=2)
ax.set_xlabel('Correlation Coefficient', fontweight='bold')
ax.set_title('Correlation with Min Images', fontweight='bold')
ax.set_xlim([-1, 1])
ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
ax.grid(axis='x', alpha=0.3)

for i, (metric, corr) in enumerate(zip(metrics, correlations)):
    ax.text(corr + 0.05 if corr > 0 else corr - 0.05, i, 
            f'{corr:.3f}', va='center', ha='left' if corr > 0 else 'right',
            fontweight='bold', fontsize=12)

# Scatter plots
ax = axes[1]
ax2 = ax.twinx()

scatter1 = ax.scatter(df['min_images'], df['finetuned_recall@1'], 
                      s=150, c=COLOR_RECALL1, alpha=0.7, edgecolors='black', linewidths=2, 
                      label='Recall@1', marker='o')
scatter2 = ax2.scatter(df['min_images'], df['finetuned_ari'] * 100, 
                       s=150, c=COLOR_RECALL5, alpha=0.7, edgecolors='black', linewidths=2, 
                       label='ARI x100', marker='s')

ax.set_xlabel('Min Images', fontweight='bold')
ax.set_ylabel('Recall@1 (%)', fontweight='bold', color=COLOR_RECALL1)
ax2.set_ylabel('ARI x100', fontweight='bold', color=COLOR_RECALL5)
ax.set_title('Relationship Min Images vs Metrics', fontweight='bold')
ax.grid(alpha=0.3)
ax.set_xticks(range(len(df)))
ax.set_xticklabels(df['label'].tolist(), rotation=45, ha='right', fontsize=8)

# Combine legends
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc='lower right')

plt.tight_layout()
plt.savefig(PLOTS_DIR / '07_correlation_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# 8. EFFICIENCY: DATA vs PERFORMANCE
# ============================================================================
print("[8/10] Efficiency analysis: data vs performance...")

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Efficiency of Data - CLIP Min Images', fontsize=16, fontweight='bold')

# Training samples by scenario
ax = axes[0, 0]
train_samples = df['train_samples'].fillna(0).values
bars = ax.bar(x, train_samples, color=colors_ft, edgecolor='black', linewidth=0.5)
ax.set_title('Training Samples by Scenario', fontweight='bold')
ax.set_ylabel('Number of Samples')
ax.set_xlabel('Min Images')
ax.set_xticks(x)
ax.set_xticklabels(df['label'].tolist(), rotation=45, ha='right', fontsize=8)
ax.grid(axis='y', alpha=0.3)
for bar in bars:
    height = bar.get_height()
    if height > 0:
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontweight='bold')

# Training classes by scenario
ax = axes[0, 1]
num_classes = df['num_training_classes'].fillna(0).values
bars = ax.bar(x, num_classes, color=colors_ft, edgecolor='black', linewidth=0.5)
ax.set_title('Training Classes by Scenario', fontweight='bold')
ax.set_ylabel('Number of Classes')
ax.set_xlabel('Min Images')
ax.set_xticks(x)
ax.set_xticklabels(df['label'].tolist(), rotation=45, ha='right', fontsize=8)
ax.grid(axis='y', alpha=0.3)
for bar in bars:
    height = bar.get_height()
    if height > 0:
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontweight='bold')

# Efficiency: Recall per 1000 samples
ax = axes[1, 0]
recall_per_1k = [(r / s * 1000) if s > 0 else 0 
                 for r, s in zip(df['finetuned_recall@1'], df['train_samples'])]
bars = ax.bar(x, recall_per_1k, color=colors_ft, edgecolor='black', linewidth=0.5)
ax.set_title('Efficiency: Recall@1 per 1000 Samples', fontweight='bold')
ax.set_ylabel('Recall@1 / 1000 samples')
ax.set_xlabel('Min Images')
ax.set_xticks(x)
ax.set_xticklabels(df['label'].tolist(), rotation=45, ha='right', fontsize=8)
ax.grid(axis='y', alpha=0.3)
for bar in bars:
    height = bar.get_height()
    if height > 0:
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)

# Training time
ax = axes[1, 1]
durations = df['duration_minutes'].fillna(0).values
bars = ax.bar(x, durations, color=colors_ft, edgecolor='black', linewidth=0.5)
ax.set_title('Training Time by Scenario', fontweight='bold')
ax.set_ylabel('Duration (minutes)')
ax.set_xlabel('Min Images')
ax.set_xticks(x)
ax.set_xticklabels(df['label'].tolist(), rotation=45, ha='right', fontsize=8)
ax.grid(axis='y', alpha=0.3)
for bar in bars:
    height = bar.get_height()
    if height > 0:
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f} min', ha='center', va='bottom', fontweight='bold', fontsize=9)

plt.tight_layout()
plt.savefig(PLOTS_DIR / '08_data_efficiency.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# 9. BEST CONFIGURATION RANKING
# ============================================================================
print("[9/10] Ranking best configurations...")

fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Scenario Ranking - CLIP Min Images', fontsize=16, fontweight='bold')

# Ranking by ARI
ax = axes[0, 0]
df_sorted_ari = df.sort_values('finetuned_ari', ascending=True)
labels_ari = df_sorted_ari['label'].tolist()
colors_sorted = [COLORS_MIN_IMAGES.get(m, COLOR_VISION) for m in df_sorted_ari['min_images']]
bars = ax.barh(range(len(df_sorted_ari)), df_sorted_ari['finetuned_ari'].values, 
               color=colors_sorted, edgecolor='black')
ax.set_yticks(range(len(df_sorted_ari)))
ax.set_yticklabels(labels_ari)
ax.set_xlabel('ARI')
ax.set_title('Ranking by ARI', fontweight='bold')
ax.grid(axis='x', alpha=0.3)
for i, v in enumerate(df_sorted_ari['finetuned_ari'].values):
    ax.text(v + 0.005, i, f'{v:.4f}', va='center', fontsize=10, fontweight='bold')

# Ranking by Recall@1
ax = axes[0, 1]
df_sorted_recall = df.sort_values('finetuned_recall@1', ascending=True)
labels_recall = df_sorted_recall['label'].tolist()
colors_sorted_r = [COLORS_MIN_IMAGES.get(m, COLOR_VISION) for m in df_sorted_recall['min_images']]
bars = ax.barh(range(len(df_sorted_recall)), df_sorted_recall['finetuned_recall@1'].values, 
               color=colors_sorted_r, edgecolor='black')
ax.set_yticks(range(len(df_sorted_recall)))
ax.set_yticklabels(labels_recall)
ax.set_xlabel('Recall@1 (%)')
ax.set_title('Ranking by Recall@1', fontweight='bold')
ax.grid(axis='x', alpha=0.3)
for i, v in enumerate(df_sorted_recall['finetuned_recall@1'].values):
    ax.text(v + 0.5, i, f'{v:.2f}%', va='center', fontsize=10, fontweight='bold')

# Ranking by % Pure Clusters
ax = axes[1, 0]
df_sorted_pure = df.sort_values('finetuned_pure_percentage', ascending=True)
labels_pure = df_sorted_pure['label'].tolist()
colors_sorted_p = [COLORS_MIN_IMAGES.get(m, COLOR_VISION) for m in df_sorted_pure['min_images']]
bars = ax.barh(range(len(df_sorted_pure)), df_sorted_pure['finetuned_pure_percentage'].values, 
               color=colors_sorted_p, edgecolor='black')
ax.set_yticks(range(len(df_sorted_pure)))
ax.set_yticklabels(labels_pure)
ax.set_xlabel('% Pure Clusters')
ax.set_title('Ranking by % Pure Clusters', fontweight='bold')
ax.grid(axis='x', alpha=0.3)
for i, v in enumerate(df_sorted_pure['finetuned_pure_percentage'].values):
    ax.text(v + 0.5, i, f'{v:.1f}%', va='center', fontsize=10, fontweight='bold')

# Ranking by improvement
ax = axes[1, 1]
df_sorted_imp = df.sort_values('recall@1_improvement', ascending=True)
labels_imp = df_sorted_imp['label'].tolist()
colors_sorted_i = [COLORS_MIN_IMAGES.get(m, COLOR_VISION) for m in df_sorted_imp['min_images']]
bars = ax.barh(range(len(df_sorted_imp)), df_sorted_imp['recall@1_improvement'].values, 
               color=colors_sorted_i, edgecolor='black')
ax.set_yticks(range(len(df_sorted_imp)))
ax.set_yticklabels(labels_imp)
ax.set_xlabel('Improvement Recall@1 (%)')
ax.set_title('Ranking by Fine-tuning Improvement', fontweight='bold')
ax.grid(axis='x', alpha=0.3)
for i, v in enumerate(df_sorted_imp['recall@1_improvement'].values):
    ax.text(v + 0.1, i, f'+{v:.2f}%' if v > 0 else f'{v:.2f}%', va='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(PLOTS_DIR / '09_ranking_scenarios.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# 10. COMPLETE COMPARATIVE SUMMARY
# ============================================================================
print("[10/10] Complete comparative summary...")

fig = plt.figure(figsize=(18, 10))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

fig.suptitle('Complete Summary: CLIP Min Images Study', fontsize=18, fontweight='bold')

# 1. Best scenario
ax = fig.add_subplot(gs[0, 0])
best_idx = df['finetuned_ari'].idxmax()
best_row = df.loc[best_idx]
bars = ax.bar([f'{int(m)}' for m in df['min_images']], 
              df['finetuned_ari'].values, 
              color=[COLORS_MIN_IMAGES.get(m, COLOR_VISION) for m in df['min_images']], 
              edgecolor='black', linewidth=2)
ax.set_title('ARI by Scenario', fontweight='bold', fontsize=11)
ax.set_ylabel('ARI', fontweight='bold')
ax.set_xlabel('Min Images', fontweight='bold')
ax.grid(axis='y', alpha=0.3)
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.4f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

# 2. Average recall
ax = fig.add_subplot(gs[0, 1])
x_pos = np.arange(len(df))
width = 0.35
ax.bar(x_pos - width/2, df['finetuned_recall@1'].values, width, label='Recall@1', 
       color=COLOR_RECALL1, edgecolor='black')
ax.bar(x_pos + width/2, df['finetuned_recall@5'].values, width, label='Recall@5', 
       color=COLOR_RECALL5, edgecolor='black')
ax.set_title('Recall Final', fontweight='bold', fontsize=11)
ax.set_ylabel('Recall (%)', fontweight='bold')
ax.set_xlabel('Min Images', fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(df['label'].tolist(), rotation=45, ha='right', fontsize=8)
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)

# 3. Average clusters
ax = fig.add_subplot(gs[0, 2])
x_pos = np.arange(len(df))
ax.bar(x_pos - width/2, df['finetuned_n_pure_clusters'].values, width, 
       label='Pure', color=COLOR_PURE, edgecolor='black')
ax.bar(x_pos + width/2, df['finetuned_n_mixed_clusters'].values, width, 
       label='Mixed', color=COLOR_MIXED, edgecolor='black')
ax.set_title('Pure Clusters vs Mixed', fontweight='bold', fontsize=11)
ax.set_ylabel('Number of Clusters', fontweight='bold')
ax.set_xlabel('Min Images', fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(df['label'].tolist(), rotation=45, ha='right', fontsize=8)
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)

# 4. Complete progression (combined lines)
ax = fig.add_subplot(gs[1, :])
ax2 = ax.twinx()
ax3 = ax.twinx()
ax3.spines['right'].set_position(('outward', 60))

line1 = ax.plot(df['min_images'], df['finetuned_recall@1'], 
                marker='o', linewidth=3, markersize=8, label='Recall@1', color=COLOR_RECALL1)
line2 = ax2.plot(df['min_images'], df['finetuned_ari'], 
                 marker='s', linewidth=3, markersize=8, label='ARI', color=COLOR_RECALL3, linestyle='--')
line3 = ax3.plot(df['min_images'], df['finetuned_pure_percentage'], 
                 marker='^', linewidth=3, markersize=8, label='% Pure', color=COLOR_PURE, linestyle=':')

ax.set_xlabel('Minimum Number of Images per Class', fontweight='bold', fontsize=11)
ax.set_ylabel('Recall@1 (%)', fontweight='bold', fontsize=10, color=COLOR_RECALL1)
ax2.set_ylabel('ARI', fontweight='bold', fontsize=10, color=COLOR_RECALL3)
ax3.set_ylabel('% Pure Clusters', fontweight='bold', fontsize=10, color=COLOR_PURE)
ax.set_title('Complete Metric Progression by Data Availability', fontweight='bold', fontsize=12, pad=20)
ax.grid(alpha=0.3)
ax.set_xticks([4, 6, 8, 10, 12])

# Combine legends
lines = line1 + line2 + line3
labels = [l.get_label() for l in lines]
ax.legend(lines, labels, loc='upper left', bbox_to_anchor=(0, -0.1), ncol=3, fontsize=10)

# 5. Final statistics
ax = fig.add_subplot(gs[2, :])
ax.axis('off')

best_min_imgs = best_row['min_images']
worst_idx = df['finetuned_ari'].idxmin()
worst_row = df.loc[worst_idx]
worst_min_imgs = worst_row['min_images']

baseline_idx = df[df['min_images'] == 8].index[0] if 8 in df['min_images'].values else best_idx
baseline_row = df.loc[baseline_idx]

corr_ari = df[['min_images', 'finetuned_ari']].corr().iloc[0, 1]
corr_recall = df[['min_images', 'finetuned_recall@1']].corr().iloc[0, 1]

stats_text = f"""

                                                                         KEY STATISTICS


    BEST SCENARIO (MORE DATA)
      Min Images: {int(best_min_imgs)} images
      Recall@1: {best_row['finetuned_recall@1']:.2f}%
      ARI: {best_row['finetuned_ari']:.4f}
      % Pure Clusters: {best_row['finetuned_pure_percentage']:.2f}%
        Training classes: {int(best_row['num_training_classes'])}
        Training samples: {int(best_row['train_samples'])}

    WORST SCENARIO (LESS DATA)
      Min Images: {int(worst_min_imgs)} images
      Recall@1: {worst_row['finetuned_recall@1']:.2f}%
      ARI: {worst_row['finetuned_ari']:.4f}
      % Pure Clusters: {worst_row['finetuned_pure_percentage']:.2f}%
        Training classes: {int(worst_row['num_training_classes'])}
        Training samples: {int(worst_row['train_samples'])}

    DEGRADATION
        Recall@1 loss: {best_row['finetuned_recall@1'] - worst_row['finetuned_recall@1']:.2f}% ({((best_row['finetuned_recall@1'] - worst_row['finetuned_recall@1']) / best_row['finetuned_recall@1'] * 100):.1f}% loss)
        ARI loss: {best_row['finetuned_ari'] - worst_row['finetuned_ari']:.4f} ({((best_row['finetuned_ari'] - worst_row['finetuned_ari']) / best_row['finetuned_ari'] * 100):.1f}% loss)
        Pure% loss: {best_row['finetuned_pure_percentage'] - worst_row['finetuned_pure_percentage']:.2f}%

    CORRELATIONS (More data -> Better performance)
      Min Images  Recall@1: {corr_recall:.3f}
      Min Images  ARI: {corr_ari:.3f}

    BASE SCENARIO (8 IMAGES - STUDY STANDARD)
      Recall@1: {baseline_row['finetuned_recall@1']:.2f}%  |  ARI: {baseline_row['finetuned_ari']:.4f}
      Classes: {int(baseline_row['num_training_classes'])}  |  Samples: {int(baseline_row['train_samples'])}


"""

ax.text(0.5, 0.5, stats_text, ha='center', va='center', fontsize=10, 
        family='monospace', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3),
        transform=ax.transAxes)

plt.savefig(PLOTS_DIR / '10_complete_summary.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# TRAINING CURVES (SAVED IN training_history_plots)
# ============================================================================
print("\n[Extra] Generating training curves...")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('CLIP Min Images - Training Curves', fontsize=14, fontweight='bold')

for idx, min_imgs in enumerate([4, 6, 8, 10, 12]):
    row_idx = idx // 3
    col_idx = idx % 3
    ax = axes[row_idx, col_idx]
    
    history = histories.get(min_imgs, {})
    label = LABELS_MIN_IMAGES.get(min_imgs, f'{min_imgs} images')
    
    phase_keys = list(history.keys())
    if len(phase_keys) >= 2:
        phase_1_key = phase_keys[0]
        phase_2_key = phase_keys[1]
    else:
        phase_1_key = 'phase_1_vision'
        phase_2_key = 'pahse_2_text'
    
    phase_1 = history.get(phase_1_key, {})
    phase_2 = history.get(phase_2_key, {})
    
    train_loss_p1 = phase_1.get('train_loss', [])
    val_loss_p1 = phase_1.get('val_loss', [])
    train_loss_p2 = phase_2.get('train_loss', [])
    val_loss_p2 = phase_2.get('val_loss', [])
    
    all_train_loss = train_loss_p1 + train_loss_p2
    all_val_loss = val_loss_p1 + val_loss_p2
    epochs = list(range(1, len(all_train_loss) + 1))
    
    if all_train_loss:
        ax.plot(epochs, all_train_loss, '-o', color=COLOR_VISION, markersize=3, 
                linewidth=2, label='Train Loss')
    if all_val_loss:
        ax.plot(epochs, all_val_loss, '--s', color=COLOR_TEXT, markersize=3, 
                linewidth=2, label='Val Loss')
    
    if train_loss_p1:
        ax.axvline(x=len(train_loss_p1) + 0.5, color='gray', linestyle=':', 
                   alpha=0.7, linewidth=2)
    
    ax.set_xlabel('Epoch', fontweight='bold')
    ax.set_ylabel('Loss', fontweight='bold')
    ax.set_title(label, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(TRAINING_PLOTS_DIR / 'clip_min_images_training_curves.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  [OK] {TRAINING_PLOTS_DIR / 'clip_min_images_training_curves.png'}")

# ============================================================================
# END
# ============================================================================
print()
print("="*80)
print("VISUALIZATIONS COMPLETED")
print("="*80)
print(f" 10 plots saved in: {PLOTS_DIR}")
print(f" Training curves in: {TRAINING_PLOTS_DIR}")
print()
print("Generated files:")
for i in range(1, 11):
    png_files = list(PLOTS_DIR.glob(f'{i:02d}_*.png'))
    if png_files:
        print(f"   {png_files[0].name}")
print(f"   clip_min_images_training_curves.png (in training_history_plots/)")
print()






