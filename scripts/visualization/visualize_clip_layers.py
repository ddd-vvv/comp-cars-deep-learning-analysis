#!/usr/bin/env python3
"""
Visualization script for CLIP Layers Study analysis.
Generates progression plots by number of layers and vision-vs-text comparisons.
"""

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
COLOR_HIGHLIGHT = '#08306b'       # Darker blue (highlights)

CORR_COLORS = {
        'high': '#08519c',    # High correlation (>0.5)
        'medium': '#6baed6',  # Medium correlation (>0)
        'low': '#969696'      # Low/negative correlation
}

# Paths
RESULTS_DIR = REPO_ROOT / 'results/analysis'
PLOTS_DIR = REPO_ROOT / 'results/analysis/plots/clip_layers'
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Load data
print("="*80)
print("GENERATING VISUALIZATIONS - CLIP LAYERS STUDY")
print("="*80)
print()

df = pd.read_csv(RESULTS_DIR / 'data/clip_layers/clip_layers_results.csv')
print(f"Loaded data: {len(df)} configurations\n")

# ============================================================================
# 1. COMPARISON VISION vs TEXT COMPONENTS
# ============================================================================
print("[1/10] Comparison vision vs text components...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('CLIP: Comparison Vision vs Text Components', fontsize=16, fontweight='bold')

# Average recall
comp_stats = df.groupby('component_type')[['best_val_recall@1', 'best_val_recall@3', 'best_val_recall@5']].mean()
ax = axes[0, 0]
comp_stats.plot(kind='bar', ax=ax, color=[COLOR_RECALL1, COLOR_RECALL3, COLOR_RECALL5], edgecolor='black', linewidth=0.5)
ax.set_title('Recall Average by Component', fontweight='bold')
ax.set_ylabel('Recall (%)')
ax.set_xlabel('Component')
ax.legend(['Recall@1', 'Recall@3', 'Recall@5'])
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.grid(axis='y', alpha=0.3)

# ARI
ax = axes[0, 1]
ari_comp = df.groupby('component_type')[['finetuned_ari']].mean()
bars = ax.bar(ari_comp.index, ari_comp['finetuned_ari'], color=[COLOR_TEXT, COLOR_VISION], edgecolor='black', linewidth=0.5)
ax.set_title('ARI Average by Component', fontweight='bold')
ax.set_ylabel('ARI')
ax.set_xlabel('Component')
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.grid(axis='y', alpha=0.3)
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.4f}', ha='center', va='bottom', fontweight='bold')

# Pure clusters
ax = axes[1, 0]
pure_comp = df.groupby('component_type')[['finetuned_pure_percentage']].mean()
bars = ax.bar(pure_comp.index, pure_comp['finetuned_pure_percentage'], color=[COLOR_TEXT, COLOR_VISION], edgecolor='black', linewidth=0.5)
ax.set_title('% Pure Clusters by Component', fontweight='bold')
ax.set_ylabel('% Pure Clusters')
ax.set_xlabel('Component')
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.grid(axis='y', alpha=0.3)
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')

# Percentage difference
ax = axes[1, 1]
if 'vision' in ari_comp.index and 'text' in ari_comp.index:
    improvement = ((ari_comp.loc['vision', 'finetuned_ari'] - ari_comp.loc['text', 'finetuned_ari']) / 
                   ari_comp.loc['text', 'finetuned_ari'] * 100)
    bars = ax.bar(['Text', 'Vision'], 
                   [ari_comp.loc['text', 'finetuned_ari'], ari_comp.loc['vision', 'finetuned_ari']],
                   color=[COLOR_TEXT, COLOR_VISION], edgecolor='black', linewidth=0.5)
    ax.set_title(f'ARI: Vision is {improvement:.1f}% better than Text', fontweight='bold')
    ax.set_ylabel('ARI')
    ax.grid(axis='y', alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(PLOTS_DIR / '01_component_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# 2. RECALL PROGRESSION BY NUMBER OF LAYERS
# ============================================================================
print("[2/10] Recall progression by number of layers...")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Recall Progression by Number of Layers', fontsize=16, fontweight='bold')

# Vision component
ax = axes[0]
df_vision = df[df['component_type'] == 'vision'].sort_values('num_layers')
ax.plot(df_vision['num_layers'], df_vision['best_val_recall@1'], 
        marker='o', linewidth=2, markersize=8, label='Recall@1', color=COLOR_RECALL1)
ax.plot(df_vision['num_layers'], df_vision['best_val_recall@3'], 
        marker='s', linewidth=2, markersize=8, label='Recall@3', color=COLOR_RECALL3)
ax.plot(df_vision['num_layers'], df_vision['best_val_recall@5'], 
        marker='^', linewidth=2, markersize=8, label='Recall@5', color=COLOR_RECALL5)
ax.set_xlabel('Number of Layers', fontweight='bold')
ax.set_ylabel('Recall (%)', fontweight='bold')
ax.set_title('Vision Component', fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)
ax.set_xticks(range(1, 13))

# Text component
ax = axes[1]
df_text = df[df['component_type'] == 'text'].sort_values('num_layers')
ax.plot(df_text['num_layers'], df_text['best_val_recall@1'], 
        marker='o', linewidth=2, markersize=8, label='Recall@1', color=COLOR_RECALL1)
ax.plot(df_text['num_layers'], df_text['best_val_recall@3'], 
        marker='s', linewidth=2, markersize=8, label='Recall@3', color=COLOR_RECALL3)
ax.plot(df_text['num_layers'], df_text['best_val_recall@5'], 
        marker='^', linewidth=2, markersize=8, label='Recall@5', color=COLOR_RECALL5)
ax.set_xlabel('Number of Layers', fontweight='bold')
ax.set_ylabel('Recall (%)', fontweight='bold')
ax.set_title('Text Component', fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)
ax.set_xticks(range(1, 13))

plt.tight_layout()
plt.savefig(PLOTS_DIR / '02_recall_progression.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# 3. ARI PROGRESSION BY NUMBER OF LAYERS
# ============================================================================
print("[3/10] ARI progression by number of layers...")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('ARI Progression by Number of Layers', fontsize=16, fontweight='bold')

# Vision component
ax = axes[0]
ax.plot(df_vision['num_layers'], df_vision['finetuned_ari'], 
        marker='o', linewidth=3, markersize=10, color=COLOR_VISION, label='ARI')
ax.fill_between(df_vision['num_layers'], df_vision['finetuned_ari'], alpha=0.3, color=COLOR_VISION)
ax.set_xlabel('Number of Layers', fontweight='bold')
ax.set_ylabel('ARI', fontweight='bold')
ax.set_title('Vision Component', fontweight='bold')
ax.grid(alpha=0.3)
ax.set_xticks(range(1, 13))

# Highlight the maximum
max_idx = df_vision['finetuned_ari'].idxmax()
max_layers = df_vision.loc[max_idx, 'num_layers']
max_ari = df_vision.loc[max_idx, 'finetuned_ari']
ax.scatter([max_layers], [max_ari], color=COLOR_HIGHLIGHT, s=200, zorder=5, marker='*', 
           edgecolors='black', linewidths=2)
ax.annotate(f'Maximum: {max_ari:.4f}\n({int(max_layers)} layers)', 
            xy=(max_layers, max_ari), xytext=(max_layers+0.5, max_ari-0.02),
            fontweight='bold', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='#deebf7', alpha=0.8),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

# Text component
ax = axes[1]
ax.plot(df_text['num_layers'], df_text['finetuned_ari'], 
        marker='o', linewidth=3, markersize=10, color=COLOR_TEXT, label='ARI')
ax.fill_between(df_text['num_layers'], df_text['finetuned_ari'], alpha=0.3, color=COLOR_TEXT)
ax.set_xlabel('Number of Layers', fontweight='bold')
ax.set_ylabel('ARI', fontweight='bold')
ax.set_title('Text Component', fontweight='bold')
ax.grid(alpha=0.3)
ax.set_xticks(range(1, 13))

# Highlight the maximum
max_idx = df_text['finetuned_ari'].idxmax()
max_layers = df_text.loc[max_idx, 'num_layers']
max_ari = df_text.loc[max_idx, 'finetuned_ari']
ax.scatter([max_layers], [max_ari], color=COLOR_HIGHLIGHT, s=200, zorder=5, marker='*',
           edgecolors='black', linewidths=2)
ax.annotate(f'Maximum: {max_ari:.4f}\n({int(max_layers)} layers)', 
            xy=(max_layers, max_ari), xytext=(max_layers+0.5, max_ari+0.005),
            fontweight='bold', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='#deebf7', alpha=0.8),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

plt.tight_layout()
plt.savefig(PLOTS_DIR / '03_ari_progression.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# 4. VISION vs TEXT COMPARISON BY LAYERS
# ============================================================================
print("[4/10] Comparison: vision vs text by layers...")

fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Vision vs Text: Comparison by Number of Layers', fontsize=16, fontweight='bold')

# Recall@1
ax = axes[0, 0]
ax.plot(df_vision['num_layers'], df_vision['best_val_recall@1'], 
        marker='o', linewidth=2.5, markersize=8, label='Vision', color=COLOR_VISION)
ax.plot(df_text['num_layers'], df_text['best_val_recall@1'], 
        marker='s', linewidth=2.5, markersize=8, label='Text', color=COLOR_TEXT)
ax.set_xlabel('Number of Layers', fontweight='bold')
ax.set_ylabel('Recall@1 (%)', fontweight='bold')
ax.set_title('Recall@1 by Layers', fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)
ax.set_xticks(range(1, 13))

# ARI
ax = axes[0, 1]
ax.plot(df_vision['num_layers'], df_vision['finetuned_ari'], 
        marker='o', linewidth=2.5, markersize=8, label='Vision', color=COLOR_VISION)
ax.plot(df_text['num_layers'], df_text['finetuned_ari'], 
        marker='s', linewidth=2.5, markersize=8, label='Text', color=COLOR_TEXT)
ax.set_xlabel('Number of Layers', fontweight='bold')
ax.set_ylabel('ARI', fontweight='bold')
ax.set_title('ARI by Layers', fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)
ax.set_xticks(range(1, 13))

# % Pure clusters
ax = axes[1, 0]
ax.plot(df_vision['num_layers'], df_vision['finetuned_pure_percentage'], 
        marker='o', linewidth=2.5, markersize=8, label='Vision', color=COLOR_VISION)
ax.plot(df_text['num_layers'], df_text['finetuned_pure_percentage'], 
        marker='s', linewidth=2.5, markersize=8, label='Text', color=COLOR_TEXT)
ax.set_xlabel('Number of Layers', fontweight='bold')
ax.set_ylabel('% Pure Clusters', fontweight='bold')
ax.set_title('% Pure Clusters by Layers', fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)
ax.set_xticks(range(1, 13))

# ARI difference (Vision - Text)
ax = axes[1, 1]
diff_ari = df_vision.set_index('num_layers')['finetuned_ari'] - df_text.set_index('num_layers')['finetuned_ari']
colors = [COLOR_POSITIVE if d > 0 else COLOR_NEGATIVE for d in diff_ari.values]
bars = ax.bar(diff_ari.index, diff_ari.values, color=colors, alpha=0.7, edgecolor='black')
ax.axhline(y=0, color='black', linestyle='--', linewidth=2)
ax.set_xlabel('Number of Layers', fontweight='bold')
ax.set_ylabel('ARI Difference (Vision - Text)', fontweight='bold')
ax.set_title('Vision Advantage over Text', fontweight='bold')
ax.grid(axis='y', alpha=0.3)
ax.set_xticks(range(1, 13))

plt.tight_layout()
plt.savefig(PLOTS_DIR / '04_vision_vs_text_by_layers.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# 5. LAYER-METRIC CORRELATION
# ============================================================================
print("[5/10] Layer-metric correlation analysis...")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Correlation: Number of Layers vs Metrics', fontsize=16, fontweight='bold')

# Vision
ax = axes[0]
corr_vision_recall = df_vision[['num_layers', 'best_val_recall@1']].corr().iloc[0, 1]
corr_vision_ari = df_vision[['num_layers', 'finetuned_ari']].corr().iloc[0, 1]

metrics = ['Recall@1', 'ARI']
correlations = [corr_vision_recall, corr_vision_ari]
colors = [CORR_COLORS['high'] if c > 0.5 else CORR_COLORS['medium'] if c > 0 else CORR_COLORS['low'] for c in correlations]

bars = ax.barh(metrics, correlations, color=colors, edgecolor='black', linewidth=2)
ax.set_xlabel('Correlation Coefficient', fontweight='bold')
ax.set_title('Vision Component', fontweight='bold')
ax.set_xlim([-1, 1])
ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
ax.grid(axis='x', alpha=0.3)

for i, (metric, corr) in enumerate(zip(metrics, correlations)):
    ax.text(corr + 0.05 if corr > 0 else corr - 0.05, i, 
            f'{corr:.3f}', va='center', ha='left' if corr > 0 else 'right',
            fontweight='bold', fontsize=12)

# Text
ax = axes[1]
corr_text_recall = df_text[['num_layers', 'best_val_recall@1']].corr().iloc[0, 1]
corr_text_ari = df_text[['num_layers', 'finetuned_ari']].corr().iloc[0, 1]

correlations = [corr_text_recall, corr_text_ari]
colors = [CORR_COLORS['high'] if c > 0.5 else CORR_COLORS['medium'] if c > 0 else CORR_COLORS['low'] for c in correlations]

bars = ax.barh(metrics, correlations, color=colors, edgecolor='black', linewidth=2)
ax.set_xlabel('Correlation Coefficient', fontweight='bold')
ax.set_title('Text Component', fontweight='bold')
ax.set_xlim([-1, 1])
ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
ax.grid(axis='x', alpha=0.3)

for i, (metric, corr) in enumerate(zip(metrics, correlations)):
    ax.text(corr + 0.05 if corr > 0 else corr - 0.05, i, 
            f'{corr:.3f}', va='center', ha='left' if corr > 0 else 'right',
            fontweight='bold', fontsize=12)

plt.tight_layout()
plt.savefig(PLOTS_DIR / '05_layer_metric_correlation.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# 6. HEATMAP: COMPONENT x LAYERS x METRICS
# ============================================================================
print("[6/10] Heatmap: component and layers...")

fig, axes = plt.subplots(1, 2, figsize=(16, 8))
fig.suptitle('Heatmap: Component and Number of Layers', fontsize=16, fontweight='bold')

# Recall@1
pivot_recall = df.pivot_table(values='best_val_recall@1', 
                                index='num_layers', 
                                columns='component_type')
ax = axes[0]
sns.heatmap(pivot_recall, annot=True, fmt='.2f', cmap='Blues', 
            cbar_kws={'label': 'Recall@1 (%)'}, ax=ax)
ax.set_title('Recall@1', fontweight='bold')
ax.set_xlabel('Component', fontweight='bold')
ax.set_ylabel('Number of Layers', fontweight='bold')

# ARI
pivot_ari = df.pivot_table(values='finetuned_ari', 
                            index='num_layers', 
                            columns='component_type')
ax = axes[1]
sns.heatmap(pivot_ari, annot=True, fmt='.4f', cmap='Blues', 
            cbar_kws={'label': 'ARI'}, ax=ax, vmin=0, vmax=0.5)
ax.set_title('ARI', fontweight='bold')
ax.set_xlabel('Component', fontweight='bold')
ax.set_ylabel('Number of Layers', fontweight='bold')

plt.tight_layout()
plt.savefig(PLOTS_DIR / '06_heatmap_layers.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# 7. CLUSTERING QUALITY BY LAYERS
# ============================================================================
print("[7/10] Clustering quality by layers...")

fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Clustering Quality by Number of Layers', fontsize=16, fontweight='bold')

# Pure/mixed cluster distribution - Vision
ax = axes[0, 0]
x = df_vision['num_layers']
width = 0.35
ax.bar(x - width/2, df_vision['finetuned_n_pure_clusters'], width,
       label='Pure Clusters', color=COLOR_PURE, edgecolor='black')
ax.bar(x + width/2, df_vision['finetuned_n_mixed_clusters'], width,
       label='Mixed Clusters', color=COLOR_MIXED, edgecolor='black')
ax.set_xlabel('Number of Layers', fontweight='bold')
ax.set_ylabel('Number of Clusters')
ax.set_title('Vision Component: Pure Clusters vs Mixed', fontweight='bold')
ax.legend()
ax.grid(axis='y', alpha=0.3)
ax.set_xticks(range(1, 13))

# Pure/mixed cluster distribution - Text
ax = axes[0, 1]
x = df_text['num_layers']
ax.bar(x - width/2, df_text['finetuned_n_pure_clusters'], width,
       label='Pure Clusters', color=COLOR_PURE, edgecolor='black')
ax.bar(x + width/2, df_text['finetuned_n_mixed_clusters'], width,
       label='Mixed Clusters', color=COLOR_MIXED, edgecolor='black')
ax.set_xlabel('Number of Layers', fontweight='bold')
ax.set_ylabel('Number of Clusters')
ax.set_title('Text Component: Pure Clusters vs Mixed', fontweight='bold')
ax.legend()
ax.grid(axis='y', alpha=0.3)
ax.set_xticks(range(1, 13))

# Purity by layers
ax = axes[1, 0]
ax.plot(df_vision['num_layers'], df_vision['finetuned_purity'], 
        marker='o', linewidth=2.5, markersize=8, label='Vision', color=COLOR_VISION)
ax.plot(df_text['num_layers'], df_text['finetuned_purity'], 
        marker='s', linewidth=2.5, markersize=8, label='Text', color=COLOR_TEXT)
ax.set_xlabel('Number of Layers', fontweight='bold')
ax.set_ylabel('Purity', fontweight='bold')
ax.set_title('Purity Score by Layers', fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)
ax.set_xticks(range(1, 13))

# Problematic classes
ax = axes[1, 1]
ax.plot(df_vision['num_layers'], df_vision['finetuned_n_overlapping_classes'], 
        marker='o', linewidth=2.5, markersize=8, label='Vision', color=COLOR_VISION)
ax.plot(df_text['num_layers'], df_text['finetuned_n_overlapping_classes'], 
        marker='s', linewidth=2.5, markersize=8, label='Text', color=COLOR_TEXT)
ax.set_xlabel('Number of Layers', fontweight='bold')
ax.set_ylabel('Problematic Classes', fontweight='bold')
ax.set_title('Classes in Multiple Clusters', fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)
ax.set_xticks(range(1, 13))

plt.tight_layout()
plt.savefig(PLOTS_DIR / '07_clustering_quality_by_layers.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# 8. TEMPORAL EFFICIENCY
# ============================================================================
print("[8/10] Temporal efficiency analysis...")

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Temporal Efficiency Analysis - CLIP', fontsize=16, fontweight='bold')

# Time by component
ax = axes[0, 0]
comp_time = df.groupby('component_type')['duration_minutes'].mean()
bars = ax.bar(comp_time.index, comp_time.values, color=[COLOR_TEXT, COLOR_VISION], edgecolor='black', linewidth=0.5)
ax.set_title('Training Time by Component', fontweight='bold')
ax.set_ylabel('Duration (minutes)')
ax.set_xlabel('Component')
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.grid(axis='y', alpha=0.3)
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f} min', ha='center', va='bottom', fontweight='bold')

# Time by number of layers
ax = axes[0, 1]
layers_time = df.groupby('num_layers')['duration_minutes'].mean()
ax.plot(layers_time.index, layers_time.values, marker='o', linewidth=2.5, 
        markersize=8, color=COLOR_VISION)
ax.set_xlabel('Number of Layers', fontweight='bold')
ax.set_ylabel('Duration (minutes)', fontweight='bold')
ax.set_title('Time by Number of Layers', fontweight='bold')
ax.grid(alpha=0.3)
ax.set_xticks(range(1, 13))

# Efficiency (ARI / hour)
ax = axes[1, 0]
df['efficiency'] = df['finetuned_ari'] / (df['duration_minutes'] / 60)
comp_eff = df.groupby('component_type')['efficiency'].mean()
bars = ax.bar(comp_eff.index, comp_eff.values, color=[COLOR_TEXT, COLOR_VISION], edgecolor='black', linewidth=0.5)
ax.set_title('Efficiency (ARI / Hour)', fontweight='bold')
ax.set_ylabel('ARI per Hour')
ax.set_xlabel('Component')
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.grid(axis='y', alpha=0.3)
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.4f}', ha='center', va='bottom', fontweight='bold')

# Relationship: time vs ARI
ax = axes[1, 1]
scatter = ax.scatter(df[df['component_type']=='vision']['duration_minutes'], 
                     df[df['component_type']=='vision']['finetuned_ari'],
                     s=100, c=df[df['component_type']=='vision']['num_layers'], 
                     cmap='Blues', alpha=0.7, edgecolors='black', label='Vision')
scatter2 = ax.scatter(df[df['component_type']=='text']['duration_minutes'], 
                      df[df['component_type']=='text']['finetuned_ari'],
                      s=100, c=df[df['component_type']=='text']['num_layers'], 
                      cmap='Blues', alpha=0.7, edgecolors='black', marker='s', label='Text')
ax.set_xlabel('Duration (minutes)', fontweight='bold')
ax.set_ylabel('ARI', fontweight='bold')
ax.set_title('Relationship Time vs ARI', fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Number of Layers')

plt.tight_layout()
plt.savefig(PLOTS_DIR / '08_temporal_efficiency.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# 9. BEST CONFIGURATION RANKING
# ============================================================================
print("[9/10] Ranking best configurations...")

fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Configuration Ranking - CLIP Layers', fontsize=16, fontweight='bold')

# Top-10 by ARI
ax = axes[0, 0]
top_ari = df.nlargest(10, 'finetuned_ari')
labels = top_ari.apply(lambda x: f"{x['component_type'][:4]} - {int(x['num_layers'])} layers", axis=1)
colors = [COLOR_VISION if ct == 'vision' else COLOR_TEXT for ct in top_ari['component_type']]
bars = ax.barh(range(10), top_ari['finetuned_ari'].values, color=colors, edgecolor='black')
ax.set_yticks(range(10))
ax.set_yticklabels(labels, fontsize=9)
ax.set_xlabel('ARI')
ax.set_title('Top-10 by ARI', fontweight='bold')
ax.grid(axis='x', alpha=0.3)
for i, v in enumerate(top_ari['finetuned_ari'].values):
    ax.text(v + 0.005, i, f'{v:.4f}', va='center', fontsize=8)

# Top-10 by Recall@1
ax = axes[0, 1]
top_recall = df.nlargest(10, 'best_val_recall@1')
labels = top_recall.apply(lambda x: f"{x['component_type'][:4]} - {int(x['num_layers'])} layers", axis=1)
colors = [COLOR_VISION if ct == 'vision' else COLOR_TEXT for ct in top_recall['component_type']]
bars = ax.barh(range(10), top_recall['best_val_recall@1'].values, color=colors, edgecolor='black')
ax.set_yticks(range(10))
ax.set_yticklabels(labels, fontsize=9)
ax.set_xlabel('Recall@1 (%)')
ax.set_title('Top-10 by Recall@1', fontweight='bold')
ax.grid(axis='x', alpha=0.3)
for i, v in enumerate(top_recall['best_val_recall@1'].values):
    ax.text(v + 0.2, i, f'{v:.2f}', va='center', fontsize=8)

# Top-10 by % Pure Clusters
ax = axes[1, 0]
top_pure = df.nlargest(10, 'finetuned_pure_percentage')
labels = top_pure.apply(lambda x: f"{x['component_type'][:4]} - {int(x['num_layers'])} layers", axis=1)
colors = [COLOR_VISION if ct == 'vision' else COLOR_TEXT for ct in top_pure['component_type']]
bars = ax.barh(range(10), top_pure['finetuned_pure_percentage'].values, color=colors, edgecolor='black')
ax.set_yticks(range(10))
ax.set_yticklabels(labels, fontsize=9)
ax.set_xlabel('% Pure Clusters')
ax.set_title('Top-10 by % Pure Clusters', fontweight='bold')
ax.grid(axis='x', alpha=0.3)
for i, v in enumerate(top_pure['finetuned_pure_percentage'].values):
    ax.text(v + 0.5, i, f'{v:.1f}%', va='center', fontsize=8)

# Top-10 by Efficiency
ax = axes[1, 1]
top_eff = df.nlargest(10, 'efficiency')
labels = top_eff.apply(lambda x: f"{x['component_type'][:4]} - {int(x['num_layers'])} layers", axis=1)
colors = [COLOR_VISION if ct == 'vision' else COLOR_TEXT for ct in top_eff['component_type']]
bars = ax.barh(range(10), top_eff['efficiency'].values, color=colors, edgecolor='black')
ax.set_yticks(range(10))
ax.set_yticklabels(labels, fontsize=9)
ax.set_xlabel('ARI / Hour')
ax.set_title('Top-10 by Efficiency', fontweight='bold')
ax.grid(axis='x', alpha=0.3)
for i, v in enumerate(top_eff['efficiency'].values):
    ax.text(v + 0.005, i, f'{v:.4f}', va='center', fontsize=8)

plt.tight_layout()
plt.savefig(PLOTS_DIR / '09_configuration_ranking.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# 10. COMPLETE COMPARATIVE SUMMARY
# ============================================================================
print("[10/10] Complete comparative summary...")

fig = plt.figure(figsize=(18, 10))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

fig.suptitle('Complete Summary: CLIP Layers Study', fontsize=18, fontweight='bold')

# 1. Best by component
ax = fig.add_subplot(gs[0, 0])
best_by_comp = df.loc[df.groupby('component_type')['finetuned_ari'].idxmax()]
bars = ax.bar(best_by_comp['component_type'], best_by_comp['finetuned_ari'], 
              color=[COLOR_TEXT, COLOR_VISION], edgecolor='black', linewidth=2)
ax.set_title('Best ARI by Component', fontweight='bold', fontsize=11)
ax.set_ylabel('ARI', fontweight='bold')
ax.grid(axis='y', alpha=0.3)
for bar, layers in zip(bars, best_by_comp['num_layers']):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.4f}\n({int(layers)} layers)', ha='center', va='bottom', fontsize=9, fontweight='bold')

# 2. Average recall
ax = fig.add_subplot(gs[0, 1])
recall_avg = df.groupby('component_type')[['best_val_recall@1', 'best_val_recall@5']].mean()
x = np.arange(len(recall_avg))
width = 0.35
ax.bar(x - width/2, recall_avg['best_val_recall@1'], width, label='Recall@1', color=COLOR_RECALL1, edgecolor='black')
ax.bar(x + width/2, recall_avg['best_val_recall@5'], width, label='Recall@5', color=COLOR_RECALL5, edgecolor='black')
ax.set_title('Recall Average', fontweight='bold', fontsize=11)
ax.set_ylabel('Recall (%)', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(recall_avg.index)
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)

# 3. Average clusters
ax = fig.add_subplot(gs[0, 2])
cluster_avg = df.groupby('component_type')[['finetuned_n_pure_clusters', 'finetuned_n_mixed_clusters']].mean()
x = np.arange(len(cluster_avg))
ax.bar(x - width/2, cluster_avg['finetuned_n_pure_clusters'], width, 
       label='Pure', color=COLOR_PURE, edgecolor='black')
ax.bar(x + width/2, cluster_avg['finetuned_n_mixed_clusters'], width, 
       label='Mixed', color=COLOR_MIXED, edgecolor='black')
ax.set_title('Clusters Average', fontweight='bold', fontsize=11)
ax.set_ylabel('Number of Clusters', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(cluster_avg.index)
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)

# 4-6. Progression by layers (recall, ari, pure%)
ax = fig.add_subplot(gs[1, :])
ax2 = ax.twinx()
ax3 = ax.twinx()
ax3.spines['right'].set_position(('outward', 60))

# Recall@1
line1 = ax.plot(df_vision['num_layers'], df_vision['best_val_recall@1'], 
                marker='o', linewidth=2, markersize=6, label='Vision Recall@1', color=COLOR_VISION)
line2 = ax.plot(df_text['num_layers'], df_text['best_val_recall@1'], 
                marker='s', linewidth=2, markersize=6, label='Text Recall@1', color=COLOR_TEXT)

# ARI
line3 = ax2.plot(df_vision['num_layers'], df_vision['finetuned_ari'], 
                 marker='D', linewidth=2, markersize=6, label='Vision ARI', color=COLOR_RECALL1, linestyle='--')
line4 = ax2.plot(df_text['num_layers'], df_text['finetuned_ari'], 
                 marker='^', linewidth=2, markersize=6, label='Text ARI', color=COLOR_RECALL3, linestyle='--')

# Pure %
line5 = ax3.plot(df_vision['num_layers'], df_vision['finetuned_pure_percentage'], 
                 marker='*', linewidth=2, markersize=8, label='Vision Pure%', color=COLOR_PURE, linestyle=':')
line6 = ax3.plot(df_text['num_layers'], df_text['finetuned_pure_percentage'], 
                 marker='P', linewidth=2, markersize=6, label='Text Pure%', color=COLOR_RECALL5, linestyle=':')

ax.set_xlabel('Number of Layers', fontweight='bold', fontsize=11)
ax.set_ylabel('Recall@1 (%)', fontweight='bold', fontsize=10, color=COLOR_VISION)
ax2.set_ylabel('ARI', fontweight='bold', fontsize=10, color=COLOR_RECALL1)
ax3.set_ylabel('% Pure Clusters', fontweight='bold', fontsize=10, color=COLOR_PURE)
ax.set_title('Complete Progression by Number of Layers', fontweight='bold', fontsize=12, pad=20)
ax.grid(alpha=0.3)
ax.set_xticks(range(1, 13))

# Combine legends
lines = line1 + line2 + line3 + line4 + line5 + line6
labels = [l.get_label() for l in lines]
ax.legend(lines, labels, loc='upper left', bbox_to_anchor=(0, -0.1), ncol=3, fontsize=9)

# 7. Final statistics
ax = fig.add_subplot(gs[2, :])
ax.axis('off')

stats_text = f"""

                                                                                                                                                 KEY STATISTICS


        BEST GLOBAL CONFIGURATION
      Component: Vision - 11 layers
      ARI: {df_vision.loc[df_vision['finetuned_ari'].idxmax(), 'finetuned_ari']:.4f}
      Recall@1: {df_vision.loc[df_vision['finetuned_ari'].idxmax(), 'best_val_recall@1']:.2f}%
      % Pure Clusters: {df_vision.loc[df_vision['finetuned_ari'].idxmax(), 'finetuned_pure_percentage']:.2f}%

        VISION vs TEXT COMPARISON
                Mean Vision ARI: {df_vision['finetuned_ari'].mean():.4f}  |  Mean Text ARI: {df_text['finetuned_ari'].mean():.4f}    Vision is {((df_vision['finetuned_ari'].mean() / df_text['finetuned_ari'].mean() - 1) * 100):.1f}% better
                Vision Recall@1: {df_vision['best_val_recall@1'].mean():.2f}%  |  Text Recall@1: {df_text['best_val_recall@1'].mean():.2f}%    Difference: {df_vision['best_val_recall@1'].mean() - df_text['best_val_recall@1'].mean():.2f}%
      Vision Pure%: {df_vision['finetuned_pure_percentage'].mean():.2f}%  |  Text Pure%: {df_text['finetuned_pure_percentage'].mean():.2f}%

        CORRELATIONS
                Vision: Layers  Recall@1 = {corr_vision_recall:.3f}  |  Layers  ARI = {corr_vision_ari:.3f}
                Text:   Layers  Recall@1 = {corr_text_recall:.3f}  |  Layers  ARI = {corr_text_ari:.3f}

        EFFICIENCY
                Vision: {(df_vision['finetuned_ari'] / (df_vision['duration_minutes'] / 60)).mean():.4f} ARI/hour  |  Text: {(df_text['finetuned_ari'] / (df_text['duration_minutes'] / 60)).mean():.4f} ARI/hour
                Mean time: Vision {df_vision['duration_minutes'].mean():.1f} min  |  Text {df_text['duration_minutes'].mean():.1f} min


"""

ax.text(0.5, 0.5, stats_text, ha='center', va='center', fontsize=10, 
        family='monospace', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))

plt.savefig(PLOTS_DIR / '10_complete_summary.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# END
# ============================================================================
print()
print("="*80)
print("VISUALIZATIONS COMPLETED")
print("="*80)
print(f" 10 plots saved in: {PLOTS_DIR}")
print()
print("Generated files:")
for i in range(1, 11):
    png_file = list(PLOTS_DIR.glob(f'{i:02d}_*.png'))[0]
    print(f"   {png_file.name}")
print()






