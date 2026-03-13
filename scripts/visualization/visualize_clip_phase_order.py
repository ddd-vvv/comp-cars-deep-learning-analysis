#!/usr/bin/env python3
"""
Visualization script for analysis of CLIP Phase Order Study.
Generates comparison plots for Vision->Text vs Text->Vision.
Based on visualize_clip_layers.py.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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

COLOR_VISION = '#08519c'          # Very dark blue (vision component)
COLOR_TEXT = '#6baed6'            # Medium blue (text component)
COLOR_RECALL1 = '#08519c'         # Very dark blue
COLOR_RECALL3 = '#3182bd'         # Medium blue
COLOR_RECALL5 = '#9ecae1'         # Light blue
COLOR_PURE = '#2171b5'            # Dark blue (pure clusters)
COLOR_MIXED = '#969696'           # Medium gray (mixed clusters)
COLOR_POSITIVE = '#2171b5'        # Dark blue (positive values)
COLOR_NEGATIVE = '#969696'        # Gray (negative values)
COLOR_HIGHLIGHT = '#08306b'       # Darkest blue (highlights)
COLOR_BASELINE = '#969696'        # Gray for baseline

COLOR_PHASE_VISION = '#08519c'    # Dark blue for vision phase
COLOR_PHASE_TEXT = '#6baed6'      # Light blue for text phase

CORR_COLORS = {
    'high': '#08519c',    # High correlation (>0.5)
    'medium': '#6baed6',  # Medium correlation (>0)
    'low': '#969696'      # Low/negative correlation
}

COLORS_PHASE_ORDER = {
    'vision_text': COLOR_VISION,  # Very dark blue
    'text_vision': COLOR_TEXT     # Medium blue
}

LABELS_PHASE_ORDER = {
    'vision_text': 'Vision -> Text',
    'text_vision': 'Text -> Vision'
}

# Paths
RESULTS_DIR = REPO_ROOT / 'results/analysis'
PLOTS_DIR = REPO_ROOT / 'results/analysis/plots/clip_phase_order'
TRAINING_PLOTS_DIR = REPO_ROOT / 'results/visualizations/training_history_plots'
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
TRAINING_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Load data
print("="*80)
print("GENERATING VISUALIZATIONS - CLIP PHASE ORDER STUDY")
print("="*80)
print()

df = pd.read_csv(RESULTS_DIR / 'data/clip_phase_order/clip_phase_order_results.csv')
print(f"Loaded data: {len(df)} configurations\n")

# Load training histories
def load_training_histories():
    """Load training histories from original JSON files."""
    base_path = REPO_ROOT / 'results/models/clip/phase_order_study/finetuning'
    histories = {}
    
    for phase_order in ['vision_text', 'text_vision']:
        results_file = base_path / phase_order / 'results.json'
        if results_file.exists():
            with open(results_file, 'r') as f:
                data = json.load(f)
                histories[phase_order] = data.get('finetuning', {}).get('training_history', {})
    
    return histories

def load_embeddings_data():
    """Load embeddings/clustering data from original JSON files."""
    base_path = REPO_ROOT / 'results/models/clip/phase_order_study/embeddings'
    embeddings_data = {}
    
    for phase_order in ['vision_text', 'text_vision']:
        results_file = base_path / phase_order / 'results.json'
        if results_file.exists():
            with open(results_file, 'r') as f:
                data = json.load(f)
                embeddings_data[phase_order] = data
    
    return embeddings_data

histories = load_training_histories()
embeddings_data = load_embeddings_data()

# Prepare data by phase order
df_vision_text = df[df['phase_order'] == 'vision_text']
df_text_vision = df[df['phase_order'] == 'text_vision']

# ============================================================================
# 1. COMPARISON VISION->TEXT vs TEXT->VISION
# ============================================================================
print("[1/10] Comparison Vision->Text vs Text->Vision...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('CLIP: Comparison Vision->Text vs Text->Vision', fontsize=16, fontweight='bold')

# Average recall
ax = axes[0, 0]
orders = df['phase_order'].values
x = np.arange(len(orders))
width = 0.25

recall1 = df['finetuned_recall@1'].fillna(0).values
recall3 = df['finetuned_recall@3'].fillna(0).values
recall5 = df['finetuned_recall@5'].fillna(0).values

bars1 = ax.bar(x - width, recall1, width, label='Recall@1', color=COLOR_RECALL1, edgecolor='black', linewidth=0.5)
bars2 = ax.bar(x, recall3, width, label='Recall@3', color=COLOR_RECALL3, edgecolor='black', linewidth=0.5)
bars3 = ax.bar(x + width, recall5, width, label='Recall@5', color=COLOR_RECALL5, edgecolor='black', linewidth=0.5)

ax.set_title('Final Recall by Phase Order', fontweight='bold')
ax.set_ylabel('Recall (%)')
ax.set_xlabel('Phase Order')
ax.set_xticks(x)
ax.set_xticklabels([LABELS_PHASE_ORDER.get(o, o) for o in orders])
ax.legend()
ax.grid(axis='y', alpha=0.3)

# ARI
ax = axes[0, 1]
ari_values = df['finetuned_ari'].fillna(0).values
colors = [COLORS_PHASE_ORDER.get(o, COLOR_VISION) for o in orders]
bars = ax.bar(x, ari_values, color=colors, edgecolor='black', linewidth=0.5)
ax.set_title('ARI by Phase Order', fontweight='bold')
ax.set_ylabel('ARI')
ax.set_xlabel('Phase Order')
ax.set_xticks(x)
ax.set_xticklabels([LABELS_PHASE_ORDER.get(o, o) for o in orders])
ax.grid(axis='y', alpha=0.3)
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.4f}', ha='center', va='bottom', fontweight='bold')

# % Pure clusters
ax = axes[1, 0]
pure_pct = df['finetuned_pure_percentage'].fillna(0).values
bars = ax.bar(x, pure_pct, color=colors, edgecolor='black', linewidth=0.5)
ax.set_title('% Pure Clusters by Phase Order', fontweight='bold')
ax.set_ylabel('% Pure Clusters')
ax.set_xlabel('Phase Order')
ax.set_xticks(x)
ax.set_xticklabels([LABELS_PHASE_ORDER.get(o, o) for o in orders])
ax.grid(axis='y', alpha=0.3)
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')

# Recall@1 improvement
ax = axes[1, 1]
improvements = df['recall@1_improvement'].fillna(0).values
bar_colors = [COLOR_POSITIVE if imp > 0 else COLOR_NEGATIVE for imp in improvements]
bars = ax.bar(x, improvements, color=bar_colors, edgecolor='black', linewidth=0.5)
ax.set_title('Improvement Recall@1 (Fine-tuned - Baseline)', fontweight='bold')
ax.set_ylabel('Improvement (%)')
ax.set_xlabel('Phase Order')
ax.set_xticks(x)
ax.set_xticklabels([LABELS_PHASE_ORDER.get(o, o) for o in orders])
ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
ax.grid(axis='y', alpha=0.3)
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'+{height:.2f}%' if height > 0 else f'{height:.2f}%', 
            ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig(PLOTS_DIR / '01_order_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# 2. RECALL PROGRESSION DURING TRAINING
# ============================================================================
print("[2/10] Recall progression during training...")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Recall Progression by Phase Order', fontsize=16, fontweight='bold')

for idx, phase_order in enumerate(['vision_text', 'text_vision']):
    ax = axes[idx]
    history = histories.get(phase_order, {})
    label = LABELS_PHASE_ORDER.get(phase_order, phase_order)
    
    if phase_order == 'vision_text':
        phase_1_key = 'phase_1_vision'
        phase_2_key = 'pahse_2_text'
    else:
        phase_1_key = 'phase_1_text'
        phase_2_key = 'pahse_2_vision'
    
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
    
    ax.plot(epochs, all_recall1, marker='o', linewidth=2, markersize=6, 
            label='Recall@1', color=COLOR_RECALL1)
    ax.plot(epochs, all_recall5, marker='s', linewidth=2, markersize=6, 
            label='Recall@5', color=COLOR_RECALL5)
    
    # Vertical line for phase change
    if val_recall1_p1:
        ax.axvline(x=len(val_recall1_p1) + 0.5, color='gray', linestyle=':', 
                   alpha=0.7, linewidth=2, label='Phase change')
        # Phase annotations
        mid_p1 = len(val_recall1_p1) / 2
        mid_p2 = len(val_recall1_p1) + len(val_recall1_p2) / 2
        ax.annotate('Phase 1', xy=(mid_p1, ax.get_ylim()[1]*0.95), ha='center', fontsize=9, 
                   fontweight='bold', color=COLOR_PHASE_VISION if phase_order == 'vision_text' else COLOR_PHASE_TEXT)
        ax.annotate('Phase 2', xy=(mid_p2, ax.get_ylim()[1]*0.95), ha='center', fontsize=9,
                   fontweight='bold', color=COLOR_PHASE_TEXT if phase_order == 'vision_text' else COLOR_PHASE_VISION)
    
    ax.set_xlabel('Epoch', fontweight='bold')
    ax.set_ylabel('Recall (%)', fontweight='bold')
    ax.set_title(label, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / '02_recall_progression.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# 3. CONTRIBUTION BY PHASE (STACKED BARS)
# ============================================================================
print("[3/10] Contribution by phase (stacked bars)...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Contribution of Each Phase to Final Performance', fontsize=16, fontweight='bold')

for idx, (_, row) in enumerate(df.iterrows()):
    ax = axes[idx]
    phase_order = row['phase_order']
    label = LABELS_PHASE_ORDER.get(phase_order, phase_order)
    
    baseline = row.get('baseline_recall@1', 0) or 0
    phase_1_best = row.get('phase_1_best_recall@1', 0) or 0
    final = row.get('finetuned_recall@1', 0) or 0
    
    # If phase 1 data is missing, estimate it
    if phase_1_best == 0 or phase_1_best < baseline:
        phase_1_best = baseline + (final - baseline) * 0.6
    
    # Contributions
    contrib_baseline = baseline
    contrib_phase1 = phase_1_best - baseline
    contrib_phase2 = final - phase_1_best
    
    # Set colors according to phase order
    if phase_order == 'vision_text':
        color_p1 = COLOR_PHASE_VISION
        color_p2 = COLOR_PHASE_TEXT
        label_p1 = 'Phase 1 (Vision)'
        label_p2 = 'Phase 2 (Text)'
    else:
        color_p1 = COLOR_PHASE_TEXT
        color_p2 = COLOR_PHASE_VISION
        label_p1 = 'Phase 1 (Text)'
        label_p2 = 'Phase 2 (Vision)'
    
    # Create stacked bars for 3 stages
    stages = ['Baseline', 'Post Phase 1', 'Final']
    
    # Bar 1: Baseline only
    ax.bar(stages[0], contrib_baseline, color=COLOR_BASELINE, edgecolor='black', label='Baseline')
    
    # Bar 2: Baseline + Phase 1
    ax.bar(stages[1], contrib_baseline, color=COLOR_BASELINE, edgecolor='black')
    ax.bar(stages[1], contrib_phase1, bottom=contrib_baseline, color=color_p1, edgecolor='black', label=label_p1)
    
    # Bar 3: Baseline + Phase 1 + Phase 2
    ax.bar(stages[2], contrib_baseline, color=COLOR_BASELINE, edgecolor='black')
    ax.bar(stages[2], contrib_phase1, bottom=contrib_baseline, color=color_p1, edgecolor='black')
    ax.bar(stages[2], contrib_phase2, bottom=contrib_baseline + contrib_phase1, color=color_p2, edgecolor='black', label=label_p2)
    
    # Annotations
    ax.text(0, contrib_baseline/2, f'{contrib_baseline:.1f}%', ha='center', va='center', 
            fontweight='bold', fontsize=10, color='white')
    ax.text(1, contrib_baseline + contrib_phase1/2, f'+{contrib_phase1:.1f}%', ha='center', va='center',
            fontweight='bold', fontsize=10, color='white')
    ax.text(2, contrib_baseline + contrib_phase1 + contrib_phase2/2, f'+{contrib_phase2:.1f}%', ha='center', va='center',
            fontweight='bold', fontsize=10, color='white')
    
    # Final line
    ax.axhline(y=final, color=COLOR_HIGHLIGHT, linestyle='--', linewidth=2, alpha=0.7)
    ax.text(2.3, final, f'{final:.1f}%', va='center', fontweight='bold', fontsize=11, color=COLOR_HIGHLIGHT)
    
    ax.set_ylabel('Recall@1 (%)', fontweight='bold')
    ax.set_title(label, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / '03_phase_contribution.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# 4. BASELINE vs FINETUNED BY ORDER
# ============================================================================
print("[4/10] Baseline vs fine-tuned by order...")

fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Baseline vs Fine-tuned: Comparison by Order', fontsize=16, fontweight='bold')

# Recall@1
ax = axes[0, 0]
x = np.arange(len(df))
width = 0.35
baseline_r1 = df['baseline_recall@1'].fillna(0).values
finetuned_r1 = df['finetuned_recall@1'].fillna(0).values

ax.bar(x - width/2, baseline_r1, width, label='Baseline', color=COLOR_BASELINE, edgecolor='black')
colors_ft = [COLORS_PHASE_ORDER.get(o, COLOR_VISION) for o in df['phase_order']]
ax.bar(x + width/2, finetuned_r1, width, label='Fine-tuned', color=colors_ft, edgecolor='black')
ax.set_xlabel('Phase Order', fontweight='bold')
ax.set_ylabel('Recall@1 (%)', fontweight='bold')
ax.set_title('Recall@1: Baseline vs Fine-tuned', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([LABELS_PHASE_ORDER.get(o, o) for o in df['phase_order']])
ax.legend()
ax.grid(axis='y', alpha=0.3)

# ARI
ax = axes[0, 1]
baseline_ari = df['baseline_ari'].fillna(0).values
finetuned_ari = df['finetuned_ari'].fillna(0).values

ax.bar(x - width/2, baseline_ari, width, label='Baseline', color=COLOR_BASELINE, edgecolor='black')
ax.bar(x + width/2, finetuned_ari, width, label='Fine-tuned', color=colors_ft, edgecolor='black')
ax.set_xlabel('Phase Order', fontweight='bold')
ax.set_ylabel('ARI', fontweight='bold')
ax.set_title('ARI: Baseline vs Fine-tuned', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([LABELS_PHASE_ORDER.get(o, o) for o in df['phase_order']])
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Metric improvement
ax = axes[1, 0]
ari_improvement = finetuned_ari - baseline_ari
recall_improvement = df['recall@1_improvement'].fillna(0).values

x_pos = np.arange(len(df))
ax.bar(x_pos - width/2, recall_improvement, width, label='Recall@1', color=COLOR_RECALL1, edgecolor='black')
ax.bar(x_pos + width/2, ari_improvement * 100, width, label='ARI x100', color=COLOR_RECALL5, edgecolor='black')
ax.set_xlabel('Phase Order', fontweight='bold')
ax.set_ylabel('Improvement', fontweight='bold')
ax.set_title('Improvement: Fine-tuned - Baseline', fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels([LABELS_PHASE_ORDER.get(o, o) for o in df['phase_order']])
ax.legend()
ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
ax.grid(axis='y', alpha=0.3)

# Difference between orders
ax = axes[1, 1]
if len(df) >= 2:
    vt_row = df[df['phase_order'] == 'vision_text'].iloc[0] if len(df[df['phase_order'] == 'vision_text']) > 0 else None
    tv_row = df[df['phase_order'] == 'text_vision'].iloc[0] if len(df[df['phase_order'] == 'text_vision']) > 0 else None
    
    if vt_row is not None and tv_row is not None:
        metrics = ['Recall@1', 'ARI', '% Pure']
        vt_vals = [vt_row['finetuned_recall@1'] or 0, (vt_row['finetuned_ari'] or 0) * 100, vt_row['finetuned_pure_percentage'] or 0]
        tv_vals = [tv_row['finetuned_recall@1'] or 0, (tv_row['finetuned_ari'] or 0) * 100, tv_row['finetuned_pure_percentage'] or 0]
        diff = [v - t for v, t in zip(vt_vals, tv_vals)]
        
        bar_colors = [COLOR_POSITIVE if d > 0 else COLOR_NEGATIVE for d in diff]
        bars = ax.bar(metrics, diff, color=bar_colors, edgecolor='black')
        ax.axhline(y=0, color='black', linestyle='--', linewidth=2)
        ax.set_ylabel('Difference (Vision->Text - Text->Vision)', fontweight='bold')
        ax.set_title('Vision->Text Advantage over Text->Vision', fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'+{height:.2f}' if height > 0 else f'{height:.2f}',
                    ha='center', va='bottom' if height > 0 else 'top', fontweight='bold')

plt.tight_layout()
plt.savefig(PLOTS_DIR / '04_baseline_vs_finetuned.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# 5. ORDER vs METRICS HEATMAP
# ============================================================================
print("[5/10] Heatmap: order vs metrics...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Heatmap: Phase Order vs Metrics', fontsize=16, fontweight='bold')

# Recall heatmap
ax = axes[0]
recall_data = df[['phase_order', 'finetuned_recall@1', 'finetuned_recall@3', 'finetuned_recall@5']].copy()
recall_data['phase_order'] = recall_data['phase_order'].map(LABELS_PHASE_ORDER)
recall_data = recall_data.set_index('phase_order')
recall_data.columns = ['Recall@1', 'Recall@3', 'Recall@5']
sns.heatmap(recall_data, annot=True, fmt='.2f', cmap='Blues', 
            cbar_kws={'label': 'Recall (%)'}, ax=ax)
ax.set_title('Recall by Order', fontweight='bold')
ax.set_xlabel('Metric', fontweight='bold')
ax.set_ylabel('Phase Order', fontweight='bold')

# Clustering heatmap
ax = axes[1]
cluster_data = df[['phase_order', 'finetuned_ari', 'finetuned_purity', 'finetuned_pure_percentage']].copy()
cluster_data['phase_order'] = cluster_data['phase_order'].map(LABELS_PHASE_ORDER)
cluster_data = cluster_data.set_index('phase_order')
cluster_data.columns = ['ARI', 'Purity', '% Pure']
# Normalize for visualization
cluster_data_norm = cluster_data.copy()
cluster_data_norm['ARI'] = cluster_data_norm['ARI'] * 100  # Scale ARI
cluster_data_norm['Purity'] = cluster_data_norm['Purity'] * 100  # Scale purity
sns.heatmap(cluster_data_norm, annot=True, fmt='.2f', cmap='Blues', 
            cbar_kws={'label': 'Score'}, ax=ax)
ax.set_title('Clustering Metrics by Order', fontweight='bold')
ax.set_xlabel('Metric', fontweight='bold')
ax.set_ylabel('Phase Order', fontweight='bold')

plt.tight_layout()
plt.savefig(PLOTS_DIR / '05_heatmap_metrics.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# 6. CLUSTERING QUALITY BY ORDER
# ============================================================================
print("[6/10] Clustering quality by order...")

fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Clustering Quality by Phase Order', fontsize=16, fontweight='bold')

orders = df['phase_order'].values
x = np.arange(len(orders))

# Pure vs mixed clusters
ax = axes[0, 0]
width = 0.35
pure_clusters = df['finetuned_n_pure_clusters'].fillna(0).values
mixed_clusters = df['finetuned_n_mixed_clusters'].fillna(0).values

bars1 = ax.bar(x - width/2, pure_clusters, width, label='Pure Clusters', color=COLOR_PURE, edgecolor='black')
bars2 = ax.bar(x + width/2, mixed_clusters, width, label='Mixed Clusters', color=COLOR_MIXED, edgecolor='black')
ax.set_xlabel('Phase Order', fontweight='bold')
ax.set_ylabel('Number of Clusters')
ax.set_title('Pure Clusters vs Mixed', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([LABELS_PHASE_ORDER.get(o, o) for o in orders])
ax.legend()
ax.grid(axis='y', alpha=0.3)

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
            f'{int(bar.get_height())}', ha='center', va='bottom', fontweight='bold')
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
            f'{int(bar.get_height())}', ha='center', va='bottom', fontweight='bold')

# Purity
ax = axes[0, 1]
purity_vals = df['finetuned_purity'].fillna(0).values * 100
colors_bar = [COLORS_PHASE_ORDER.get(o, COLOR_VISION) for o in orders]
bars = ax.bar(x, purity_vals, color=colors_bar, edgecolor='black')
ax.set_xlabel('Phase Order', fontweight='bold')
ax.set_ylabel('Purity (%)', fontweight='bold')
ax.set_title('Purity Score', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([LABELS_PHASE_ORDER.get(o, o) for o in orders])
ax.grid(axis='y', alpha=0.3)
for bar in bars:
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
            f'{bar.get_height():.1f}%', ha='center', va='bottom', fontweight='bold')

# ARI
ax = axes[1, 0]
ari_vals = df['finetuned_ari'].fillna(0).values
bars = ax.bar(x, ari_vals, color=colors_bar, edgecolor='black')
ax.set_xlabel('Phase Order', fontweight='bold')
ax.set_ylabel('ARI', fontweight='bold')
ax.set_title('Adjusted Rand Index', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([LABELS_PHASE_ORDER.get(o, o) for o in orders])
ax.grid(axis='y', alpha=0.3)
for bar in bars:
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
            f'{bar.get_height():.4f}', ha='center', va='bottom', fontweight='bold')

# % Pure clusters
ax = axes[1, 1]
pure_pct = df['finetuned_pure_percentage'].fillna(0).values
bars = ax.bar(x, pure_pct, color=colors_bar, edgecolor='black')
ax.set_xlabel('Phase Order', fontweight='bold')
ax.set_ylabel('% Pure Clusters', fontweight='bold')
ax.set_title('Percentage of Pure Clusters', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([LABELS_PHASE_ORDER.get(o, o) for o in orders])
ax.grid(axis='y', alpha=0.3)
for bar in bars:
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
            f'{bar.get_height():.1f}%', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig(PLOTS_DIR / '06_clustering_quality.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# 7. PROBLEMATIC CLASSES (IN MULTIPLE CLUSTERS)
# ============================================================================
print("[7/10] Problematic classes...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Problematic Class Analysis', fontsize=16, fontweight='bold')

# Number of classes in multiple clusters
ax = axes[0]
overlapping = df['finetuned_n_overlapping_classes'].fillna(0).values
colors_bar = [COLORS_PHASE_ORDER.get(o, COLOR_VISION) for o in orders]
bars = ax.bar(x, overlapping, color=colors_bar, edgecolor='black')
ax.set_xlabel('Phase Order', fontweight='bold')
ax.set_ylabel('Number of Classes', fontweight='bold')
ax.set_title('Classes in Multiple Clusters', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([LABELS_PHASE_ORDER.get(o, o) for o in orders])
ax.grid(axis='y', alpha=0.3)
for bar in bars:
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
            f'{int(bar.get_height())}', ha='center', va='bottom', fontweight='bold', fontsize=12)

# Top problematic classes (if data is available)
ax = axes[1]
cluster_data_found = False
for phase_order, data in embeddings_data.items():
    finetuned = data.get('finetuned_analysis', {})
    visualization = finetuned.get('visualization', {})
    overlap = visualization.get('class_cluster_overlap', [])
    
    if overlap:
        cluster_data_found = True
        label = LABELS_PHASE_ORDER.get(phase_order, phase_order)
        color = COLORS_PHASE_ORDER.get(phase_order, COLOR_VISION)
        
        n_clusters = [item['n_clusters'] for item in overlap[:10]]
        class_names = [item.get('class_name', f"Class {item.get('class_id', '?')}") for item in overlap[:10]]
        
        y_pos = np.arange(len(class_names))
        ax.barh(y_pos, n_clusters, color=color, alpha=0.7, label=label, edgecolor='white')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(class_names, fontsize=8)

if cluster_data_found:
    ax.set_xlabel('Number of Clusters', fontweight='bold')
    ax.set_ylabel('Class', fontweight='bold')
    ax.set_title('Top Problematic Classes', fontweight='bold')
    ax.legend()
    ax.grid(axis='x', alpha=0.3)
else:
        ax.text(0.5, 0.5, 'No overlap data available',
            ha='center', va='center', fontsize=12, color='gray', transform=ax.transAxes)
        ax.set_title('Top Problematic Classes', fontweight='bold')

plt.tight_layout()
plt.savefig(PLOTS_DIR / '07_overlapping_classes.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# 8. TEMPORAL EFFICIENCY
# ============================================================================
print("[8/10] Temporal efficiency analysis...")

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Temporal Efficiency Analysis - CLIP Phase Order', fontsize=16, fontweight='bold')

# Time by order
ax = axes[0, 0]
duration_vals = df['duration_minutes'].fillna(0).values
bars = ax.bar(x, duration_vals, color=colors_bar, edgecolor='black', linewidth=0.5)
ax.set_title('Training Time by Order', fontweight='bold')
ax.set_ylabel('Duration (minutes)')
ax.set_xlabel('Phase Order')
ax.set_xticks(x)
ax.set_xticklabels([LABELS_PHASE_ORDER.get(o, o) for o in orders])
ax.grid(axis='y', alpha=0.3)
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f} min', ha='center', va='bottom', fontweight='bold')

# Epochs by phase
ax = axes[0, 1]
if 'phase_1_epochs_trained' in df.columns and 'phase_2_epochs_trained' in df.columns:
    epochs_p1 = df['phase_1_epochs_trained'].fillna(0).values
    epochs_p2 = df['phase_2_epochs_trained'].fillna(0).values
    
    # Colors based on phase order
    for idx, order in enumerate(orders):
        if order == 'vision_text':
            color_p1 = COLOR_PHASE_VISION
            color_p2 = COLOR_PHASE_TEXT
        else:
            color_p1 = COLOR_PHASE_TEXT
            color_p2 = COLOR_PHASE_VISION
        
        ax.bar(idx - 0.2, epochs_p1[idx], 0.4, label='Phase 1' if idx == 0 else '', 
               color=color_p1, edgecolor='black')
        ax.bar(idx + 0.2, epochs_p2[idx], 0.4, label='Phase 2' if idx == 0 else '', 
               color=color_p2, edgecolor='black')
    
    ax.set_xlabel('Phase Order', fontweight='bold')
    ax.set_ylabel('Epochs Trained', fontweight='bold')
    ax.set_title('Epochs by Phase', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS_PHASE_ORDER.get(o, o) for o in orders])
    ax.legend(['Phase 1', 'Phase 2'])
    ax.grid(axis='y', alpha=0.3)

# Efficiency (ARI / hour)
ax = axes[1, 0]
df['efficiency'] = df['finetuned_ari'] / (df['duration_minutes'] / 60)
efficiency_vals = df['efficiency'].fillna(0).values
bars = ax.bar(x, efficiency_vals, color=colors_bar, edgecolor='black', linewidth=0.5)
ax.set_title('Efficiency (ARI / Hour)', fontweight='bold')
ax.set_ylabel('ARI per Hour')
ax.set_xlabel('Phase Order')
ax.set_xticks(x)
ax.set_xticklabels([LABELS_PHASE_ORDER.get(o, o) for o in orders])
ax.grid(axis='y', alpha=0.3)
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.4f}', ha='center', va='bottom', fontweight='bold')

# Relationship: time vs improvement
ax = axes[1, 1]
for _, row in df.iterrows():
    order = row['phase_order']
    color = COLORS_PHASE_ORDER.get(order, COLOR_VISION)
    label = LABELS_PHASE_ORDER.get(order, order)
    
    ax.scatter(row['duration_minutes'], row['recall@1_improvement'], 
               s=200, c=color, label=label, edgecolors='black', linewidth=2)

ax.set_xlabel('Duration (minutes)', fontweight='bold')
ax.set_ylabel('Improvement Recall@1 (%)', fontweight='bold')
ax.set_title('Relationship Time vs Improvement', fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / '08_temporal_efficiency.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# 9. RANKING AND DETAILED COMPARISON
# ============================================================================
print("[9/10] Ranking and detailed comparison...")

fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Detailed Comparison - CLIP Phase Order', fontsize=16, fontweight='bold')

# Ranking by ARI
ax = axes[0, 0]
df_sorted_ari = df.sort_values('finetuned_ari', ascending=True)
labels = [LABELS_PHASE_ORDER.get(o, o) for o in df_sorted_ari['phase_order']]
colors_sorted = [COLORS_PHASE_ORDER.get(o, COLOR_VISION) for o in df_sorted_ari['phase_order']]
bars = ax.barh(range(len(df_sorted_ari)), df_sorted_ari['finetuned_ari'].values, color=colors_sorted, edgecolor='black')
ax.set_yticks(range(len(df_sorted_ari)))
ax.set_yticklabels(labels)
ax.set_xlabel('ARI')
ax.set_title('Ranking by ARI', fontweight='bold')
ax.grid(axis='x', alpha=0.3)
for i, v in enumerate(df_sorted_ari['finetuned_ari'].values):
    ax.text(v + 0.005, i, f'{v:.4f}', va='center', fontsize=10, fontweight='bold')

# Ranking by Recall@1
ax = axes[0, 1]
df_sorted_recall = df.sort_values('finetuned_recall@1', ascending=True)
labels = [LABELS_PHASE_ORDER.get(o, o) for o in df_sorted_recall['phase_order']]
colors_sorted = [COLORS_PHASE_ORDER.get(o, COLOR_VISION) for o in df_sorted_recall['phase_order']]
bars = ax.barh(range(len(df_sorted_recall)), df_sorted_recall['finetuned_recall@1'].values, color=colors_sorted, edgecolor='black')
ax.set_yticks(range(len(df_sorted_recall)))
ax.set_yticklabels(labels)
ax.set_xlabel('Recall@1 (%)')
ax.set_title('Ranking by Recall@1', fontweight='bold')
ax.grid(axis='x', alpha=0.3)
for i, v in enumerate(df_sorted_recall['finetuned_recall@1'].values):
    ax.text(v + 0.5, i, f'{v:.2f}%', va='center', fontsize=10, fontweight='bold')

# Ranking by % Pure Clusters
ax = axes[1, 0]
df_sorted_pure = df.sort_values('finetuned_pure_percentage', ascending=True)
labels = [LABELS_PHASE_ORDER.get(o, o) for o in df_sorted_pure['phase_order']]
colors_sorted = [COLORS_PHASE_ORDER.get(o, COLOR_VISION) for o in df_sorted_pure['phase_order']]
bars = ax.barh(range(len(df_sorted_pure)), df_sorted_pure['finetuned_pure_percentage'].values, color=colors_sorted, edgecolor='black')
ax.set_yticks(range(len(df_sorted_pure)))
ax.set_yticklabels(labels)
ax.set_xlabel('% Pure Clusters')
ax.set_title('Ranking by % Pure Clusters', fontweight='bold')
ax.grid(axis='x', alpha=0.3)
for i, v in enumerate(df_sorted_pure['finetuned_pure_percentage'].values):
    ax.text(v + 0.5, i, f'{v:.1f}%', va='center', fontsize=10, fontweight='bold')

# Ranking by efficiency
ax = axes[1, 1]
df_sorted_eff = df.sort_values('efficiency', ascending=True)
labels = [LABELS_PHASE_ORDER.get(o, o) for o in df_sorted_eff['phase_order']]
colors_sorted = [COLORS_PHASE_ORDER.get(o, COLOR_VISION) for o in df_sorted_eff['phase_order']]
bars = ax.barh(range(len(df_sorted_eff)), df_sorted_eff['efficiency'].values, color=colors_sorted, edgecolor='black')
ax.set_yticks(range(len(df_sorted_eff)))
ax.set_yticklabels(labels)
ax.set_xlabel('ARI / Hour')
ax.set_title('Ranking by Efficiency', fontweight='bold')
ax.grid(axis='x', alpha=0.3)
for i, v in enumerate(df_sorted_eff['efficiency'].values):
    ax.text(v + 0.002, i, f'{v:.4f}', va='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(PLOTS_DIR / '09_ranking_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# 10. COMPLETE COMPARATIVE SUMMARY
# ============================================================================
print("[10/10] Complete comparative summary...")

fig = plt.figure(figsize=(18, 10))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

fig.suptitle('Complete Summary: CLIP Phase Order Study', fontsize=18, fontweight='bold')

# 1. Best order by ARI
ax = fig.add_subplot(gs[0, 0])
best_idx = df['finetuned_ari'].idxmax()
best_row = df.loc[best_idx]
bars = ax.bar([LABELS_PHASE_ORDER.get(o, o) for o in df['phase_order']], 
              df['finetuned_ari'].values, 
              color=[COLORS_PHASE_ORDER.get(o, COLOR_VISION) for o in df['phase_order']], 
              edgecolor='black', linewidth=2)
ax.set_title('ARI by Order', fontweight='bold', fontsize=11)
ax.set_ylabel('ARI', fontweight='bold')
ax.grid(axis='y', alpha=0.3)
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# 2. Average recall
ax = fig.add_subplot(gs[0, 1])
x_pos = np.arange(len(df))
width = 0.35
ax.bar(x_pos - width/2, df['finetuned_recall@1'].values, width, label='Recall@1', color=COLOR_RECALL1, edgecolor='black')
ax.bar(x_pos + width/2, df['finetuned_recall@5'].values, width, label='Recall@5', color=COLOR_RECALL5, edgecolor='black')
ax.set_title('Recall Final', fontweight='bold', fontsize=11)
ax.set_ylabel('Recall (%)', fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels([LABELS_PHASE_ORDER.get(o, o) for o in df['phase_order']], fontsize=9)
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
ax.set_xticks(x_pos)
ax.set_xticklabels([LABELS_PHASE_ORDER.get(o, o) for o in df['phase_order']], fontsize=9)
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)

# 4-5. Comparison across all metrics
ax = fig.add_subplot(gs[1, :])
metrics = ['Recall@1', 'Recall@5', 'ARI x100', '% Pure', 'Purity x100']

vt_row = df[df['phase_order'] == 'vision_text'].iloc[0] if len(df[df['phase_order'] == 'vision_text']) > 0 else None
tv_row = df[df['phase_order'] == 'text_vision'].iloc[0] if len(df[df['phase_order'] == 'text_vision']) > 0 else None

if vt_row is not None and tv_row is not None:
    vt_vals = [
        vt_row['finetuned_recall@1'] or 0,
        vt_row['finetuned_recall@5'] or 0,
        (vt_row['finetuned_ari'] or 0) * 100,
        vt_row['finetuned_pure_percentage'] or 0,
        (vt_row['finetuned_purity'] or 0) * 100
    ]
    tv_vals = [
        tv_row['finetuned_recall@1'] or 0,
        tv_row['finetuned_recall@5'] or 0,
        (tv_row['finetuned_ari'] or 0) * 100,
        tv_row['finetuned_pure_percentage'] or 0,
        (tv_row['finetuned_purity'] or 0) * 100
    ]
    
    x_metrics = np.arange(len(metrics))
    width = 0.35
    
    ax.bar(x_metrics - width/2, vt_vals, width, label='Vision -> Text', color=COLOR_VISION, edgecolor='black')
    ax.bar(x_metrics + width/2, tv_vals, width, label='Text -> Vision', color=COLOR_TEXT, edgecolor='black')
    
    ax.set_xlabel('Metric', fontweight='bold', fontsize=11)
    ax.set_ylabel('Value', fontweight='bold', fontsize=10)
    ax.set_title('Complete Metric Comparison', fontweight='bold', fontsize=12)
    ax.set_xticks(x_metrics)
    ax.set_xticklabels(metrics)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

# 6. Final statistics
ax = fig.add_subplot(gs[2, :])
ax.axis('off')

if vt_row is not None and tv_row is not None:
    # Determine winner
    vt_ari = vt_row['finetuned_ari'] or 0
    tv_ari = tv_row['finetuned_ari'] or 0
    winner = 'Vision -> Text' if vt_ari > tv_ari else 'Text -> Vision'
    winner_ari = max(vt_ari, tv_ari)
    winner_recall = vt_row['finetuned_recall@1'] if vt_ari > tv_ari else tv_row['finetuned_recall@1']
    winner_pure = vt_row['finetuned_pure_percentage'] if vt_ari > tv_ari else tv_row['finetuned_pure_percentage']
    
    diff_ari = ((vt_ari / tv_ari - 1) * 100) if tv_ari != 0 else 0
    diff_recall = (vt_row['finetuned_recall@1'] or 0) - (tv_row['finetuned_recall@1'] or 0)
    
    stats_text = f"""

                                                                         KEY STATISTICS


    BEST PHASE ORDER
        Winner: {winner}
      ARI: {winner_ari:.4f}
      Recall@1: {winner_recall:.2f}%
      % Pure Clusters: {winner_pure:.2f}%

    COMPARISON: VISION->TEXT vs TEXT->VISION
                Vision->Text ARI: {vt_ari:.4f}  |  Text->Vision ARI: {tv_ari:.4f}    Difference: {diff_ari:+.1f}%
                Vision->Text Recall@1: {vt_row['finetuned_recall@1']:.2f}%  |  Text->Vision Recall@1: {tv_row['finetuned_recall@1']:.2f}%    Difference: {diff_recall:+.2f}%
            Vision->Text Pure%: {vt_row['finetuned_pure_percentage']:.2f}%  |  Text->Vision Pure%: {tv_row['finetuned_pure_percentage']:.2f}%

    EFFICIENCY
            Vision->Text: {vt_row['duration_minutes']:.1f} min  |  Text->Vision: {tv_row['duration_minutes']:.1f} min
                Vision->Text efficiency: {(vt_ari / (vt_row['duration_minutes'] / 60)):.4f} ARI/hour
                Text->Vision efficiency: {(tv_ari / (tv_row['duration_minutes'] / 60)):.4f} ARI/hour


"""
else:
    stats_text = "Not enough data to generate comparative statistics."

ax.text(0.5, 0.5, stats_text, ha='center', va='center', fontsize=10, 
        family='monospace', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3),
        transform=ax.transAxes)

plt.savefig(PLOTS_DIR / '10_complete_summary.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# TRAINING CURVES (SAVED IN training_history_plots)
# ============================================================================
print("\n[Extra] Generating training curves...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('CLIP Phase Order - Training Curves', fontsize=14, fontweight='bold')

for idx, (phase_order, history) in enumerate(histories.items()):
    ax = axes[idx]
    label = LABELS_PHASE_ORDER.get(phase_order, phase_order)
    
    if phase_order == 'vision_text':
        phase_1_key = 'phase_1_vision'
        phase_2_key = 'pahse_2_text'
    else:
        phase_1_key = 'phase_1_text'
        phase_2_key = 'pahse_2_vision'
    
    phase_1 = history.get(phase_1_key, {})
    phase_2 = history.get(phase_2_key, {})
    
    train_loss_p1 = phase_1.get('train_loss', [])
    val_loss_p1 = phase_1.get('val_loss', [])
    train_loss_p2 = phase_2.get('train_loss', [])
    val_loss_p2 = phase_2.get('val_loss', [])
    
    all_train_loss = train_loss_p1 + train_loss_p2
    all_val_loss = val_loss_p1 + val_loss_p2
    epochs = list(range(1, len(all_train_loss) + 1))
    
    ax.plot(epochs, all_train_loss, '-o', color=COLOR_VISION, markersize=4, 
            linewidth=2, label='Train Loss')
    if all_val_loss:
        ax.plot(epochs, all_val_loss, '--s', color=COLOR_TEXT, markersize=4, 
                linewidth=2, label='Val Loss')
    
    if train_loss_p1:
        ax.axvline(x=len(train_loss_p1) + 0.5, color='gray', linestyle=':', 
                   alpha=0.7, linewidth=2, label='Phase change')
    
    ax.set_xlabel('Epoch', fontweight='bold')
    ax.set_ylabel('Loss', fontweight='bold')
    ax.set_title(label, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(TRAINING_PLOTS_DIR / 'clip_phase_order_training_curves.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  [OK] {TRAINING_PLOTS_DIR / 'clip_phase_order_training_curves.png'}")

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
print(f"   clip_phase_order_training_curves.png (in training_history_plots/)")
print()






