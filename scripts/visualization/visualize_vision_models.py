#!/usr/bin/env python3
"""
Visualization script for analysis of Vision Models (ResNet50/ViT-B/32)
Generates comparative performance and clustering plots
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from matplotlib.patches import Patch
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

COLOR_RESNET = '#08519c'          # Very dark blue
COLOR_VIT = '#6baed6'             # Medium blue
COLOR_CLASSIFICATION = '#08306b'  # Darkest blue
COLOR_CLASS = '#08306b'           # Alias
COLOR_METRIC_LEARNING = '#4292c6' # Medium-dark blue
COLOR_ML = '#4292c6'              # Alias
COLOR_FRONT = '#2171b5'           # Dark blue
COLOR_FRONT_REAR = '#9ecae1'      # Light blue
COLOR_BASELINE = '#969696'        # Medium gray (ColorBrewer Greys)
COLOR_FINETUNED = '#2171b5'       # Dark blue
COLOR_RECALL1 = '#08519c'         # Very dark blue
COLOR_RECALL3 = '#3182bd'         # Medium blue
COLOR_RECALL5 = '#9ecae1'         # Light blue

CRITERION_BLUES = ['#c6dbef', '#9ecae1', '#6baed6', '#3182bd', '#08519c']

# Paths
RESULTS_DIR = REPO_ROOT / 'results/analysis/data/vision_models'
PLOTS_DIR = REPO_ROOT / 'results/analysis/plots/vision_models'
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Load data
print("="*80)
print("GENERATING VISUALIZATIONS - VISION MODELS")
print("="*80)
print()

df = pd.read_csv(RESULTS_DIR / 'vision_models_results.csv')
print(f"Loaded data: {len(df)} configurations\n")

# 1. COMPARISON BY MODEL
print("[1/14] Comparison by model...")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Comparison ResNet50 vs ViT-B/32', fontsize=14, fontweight='bold')

# Recall@1, @3, @5
model_stats = df.groupby('model_name')[['best_val_recall@1', 'best_val_recall@3', 'best_val_recall@5']].mean()
ax = axes[0, 0]
x = np.arange(len(model_stats))
width = 0.25
bars1 = ax.bar(x - width, model_stats['best_val_recall@1'], width, label='Recall@1', color=COLOR_RECALL1, edgecolor='black', linewidth=0.5)
bars2 = ax.bar(x, model_stats['best_val_recall@3'], width, label='Recall@3', color=COLOR_RECALL3, edgecolor='black', linewidth=0.5)
bars3 = ax.bar(x + width, model_stats['best_val_recall@5'], width, label='Recall@5', color=COLOR_RECALL5, edgecolor='black', linewidth=0.5)
ax.set_title('Recall Average by Model')
ax.set_ylabel('Recall')
ax.set_xlabel('Model')
ax.set_xticks(x)
ax.set_xticklabels(model_stats.index)
ax.legend(loc='upper right')
ax.grid(axis='y', alpha=0.3)

# ARI
ax = axes[0, 1]
ari_stats = df.groupby('model_name')[['finetuned_ari']].mean()
bars = ax.bar(ari_stats.index, ari_stats['finetuned_ari'], color=[COLOR_RESNET, COLOR_VIT], edgecolor='black', linewidth=0.5)
ax.set_title('ARI Average by Model')
ax.set_ylabel('ARI')
ax.set_xlabel('Model')
ax.grid(axis='y', alpha=0.3)
for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{height:.4f}', xy=(bar.get_x() + bar.get_width()/2., height),
                xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=9)

# Pure clusters
ax = axes[1, 0]
cluster_stats = df.groupby('model_name')[['finetuned_pure_percentage']].mean()
bars = ax.bar(cluster_stats.index, cluster_stats['finetuned_pure_percentage'], color=[COLOR_RESNET, COLOR_VIT], edgecolor='black', linewidth=0.5)
ax.set_title('% Pure Clusters Average by Model')
ax.set_ylabel('% Pure Clusters')
ax.set_xlabel('Model')
ax.grid(axis='y', alpha=0.3)
for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2., height),
                xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=9)

# Time
ax = axes[1, 1]
time_stats = df.groupby('model_name')[['duration_minutes']].mean()
bars = ax.bar(time_stats.index, time_stats['duration_minutes'], color=[COLOR_RESNET, COLOR_VIT], edgecolor='black', linewidth=0.5)
ax.set_title('Training Time Average')
ax.set_ylabel('Duration (minutes)')
ax.set_xlabel('Model')
ax.grid(axis='y', alpha=0.3)
for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{height:.0f} min', xy=(bar.get_x() + bar.get_width()/2., height),
                xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(PLOTS_DIR / '01_model_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. COMPARISON BY OBJECTIVE
print("[2/14] Comparison by objective...")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Comparison Classification vs Metric Learning', fontsize=14, fontweight='bold')

# Main metric
ax = axes[0, 0]
df_class = df[df['objective'] == 'classification']
df_ml = df[df['objective'] == 'metric_learning']
class_acc = df_class['best_val_accuracy'].mean() if len(df_class) > 0 else 0
ml_recall = df_ml['best_val_recall@1'].mean() * 100 if len(df_ml) > 0 else 0
x_labels = ['Classification\n(Accuracy %)', 'Metric Learning\n(Recall@1 %)']
values = [class_acc, ml_recall]
bars = ax.bar(x_labels, values, color=[COLOR_CLASSIFICATION, COLOR_METRIC_LEARNING], edgecolor='black', linewidth=0.5)
ax.set_title('Main Metric by Objective')
ax.set_ylabel('Percentage (%)')
ax.grid(axis='y', alpha=0.3)
for bar, val in zip(bars, values):
    height = bar.get_height()
    ax.annotate(f'{val:.2f}%', xy=(bar.get_x() + bar.get_width()/2., height),
                xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=9)

# ARI
ax = axes[0, 1]
ari_obj = df.groupby('objective')[['finetuned_ari']].mean()
bars = ax.bar(ari_obj.index, ari_obj['finetuned_ari'], color=[COLOR_CLASSIFICATION, COLOR_METRIC_LEARNING], edgecolor='black', linewidth=0.5)
ax.set_title('ARI Average by Objective')
ax.set_ylabel('ARI')
ax.set_xlabel('Objective')
ax.grid(axis='y', alpha=0.3)
for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{height:.4f}', xy=(bar.get_x() + bar.get_width()/2., height),
                xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=9)

# Pure clusters
ax = axes[1, 0]
pure_obj = df.groupby('objective')[['finetuned_pure_percentage']].mean()
bars = ax.bar(pure_obj.index, pure_obj['finetuned_pure_percentage'], color=[COLOR_CLASSIFICATION, COLOR_METRIC_LEARNING], edgecolor='black', linewidth=0.5)
ax.set_title('% Pure Clusters by Objective')
ax.set_ylabel('% Pure Clusters')
ax.set_xlabel('Objective')
ax.grid(axis='y', alpha=0.3)
for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2., height),
                xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=9)

# Number of configurations
ax = axes[1, 1]
config_count = df.groupby('objective').size()
bars = ax.bar(config_count.index, config_count.values, color=[COLOR_CLASSIFICATION, COLOR_METRIC_LEARNING], edgecolor='black', linewidth=0.5)
ax.set_title('Number of Configurations')
ax.set_ylabel('Count')
ax.set_xlabel('Objective')
ax.grid(axis='y', alpha=0.3)
for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{int(height)}', xy=(bar.get_x() + bar.get_width()/2., height),
                xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(PLOTS_DIR / '02_objective_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. COMPARISON BY LOSS FUNCTION (CRITERION) - 2x3
print("[3/14] Comparison by loss function...")

# Filter metric learning only
df_ml = df[df['objective'] == 'metric_learning'].copy()

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Loss Function Comparison (Metric Learning)', fontsize=14, fontweight='bold')

# Get blue colors for criteria
n_criteria = df_ml['criterion'].nunique()
criterion_colors = CRITERION_BLUES[:n_criteria]

# Recall@1
ax = axes[0, 0]
recall1_crit = df_ml.groupby('criterion')['best_val_recall@1'].mean().sort_values(ascending=True)
colors = [CRITERION_BLUES[list(recall1_crit.index).index(c) % len(CRITERION_BLUES)] for c in recall1_crit.index]
bars = ax.barh(range(len(recall1_crit)), recall1_crit.values, color=colors, edgecolor='black', linewidth=0.5)
ax.set_yticks(range(len(recall1_crit)))
ax.set_yticklabels(recall1_crit.index)
ax.set_xlabel('Recall@1')
ax.set_title('Recall@1 Average')
ax.grid(axis='x', alpha=0.3)
ax.set_xlim(0, recall1_crit.max() * 1.18)
for i, v in enumerate(recall1_crit.values):
    ax.text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=8)

# Recall@3
ax = axes[0, 1]
recall3_crit = df_ml.groupby('criterion')['best_val_recall@3'].mean().sort_values(ascending=True)
colors = [CRITERION_BLUES[list(recall3_crit.index).index(c) % len(CRITERION_BLUES)] for c in recall3_crit.index]
bars = ax.barh(range(len(recall3_crit)), recall3_crit.values, color=colors, edgecolor='black', linewidth=0.5)
ax.set_yticks(range(len(recall3_crit)))
ax.set_yticklabels(recall3_crit.index)
ax.set_xlabel('Recall@3')
ax.set_title('Recall@3 Average')
ax.grid(axis='x', alpha=0.3)
ax.set_xlim(0, recall3_crit.max() * 1.18)
for i, v in enumerate(recall3_crit.values):
    ax.text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=8)

# Recall@5
ax = axes[0, 2]
recall5_crit = df_ml.groupby('criterion')['best_val_recall@5'].mean().sort_values(ascending=True)
colors = [CRITERION_BLUES[list(recall5_crit.index).index(c) % len(CRITERION_BLUES)] for c in recall5_crit.index]
bars = ax.barh(range(len(recall5_crit)), recall5_crit.values, color=colors, edgecolor='black', linewidth=0.5)
ax.set_yticks(range(len(recall5_crit)))
ax.set_yticklabels(recall5_crit.index)
ax.set_xlabel('Recall@5')
ax.set_title('Recall@5 Average')
ax.grid(axis='x', alpha=0.3)
ax.set_xlim(0, recall5_crit.max() * 1.18)
for i, v in enumerate(recall5_crit.values):
    ax.text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=8)

# ARI
ax = axes[1, 0]
ari_crit = df_ml.groupby('criterion')['finetuned_ari'].mean().sort_values(ascending=True)
colors = [CRITERION_BLUES[list(ari_crit.index).index(c) % len(CRITERION_BLUES)] for c in ari_crit.index]
bars = ax.barh(range(len(ari_crit)), ari_crit.values, color=colors, edgecolor='black', linewidth=0.5)
ax.set_yticks(range(len(ari_crit)))
ax.set_yticklabels(ari_crit.index)
ax.set_xlabel('ARI')
ax.set_title('ARI Average')
ax.grid(axis='x', alpha=0.3)
ax.set_xlim(0, ari_crit.max() * 1.25)
for i, v in enumerate(ari_crit.values):
    ax.text(v + 0.005, i, f'{v:.4f}', va='center', fontsize=8)

# % Pure Clusters
ax = axes[1, 1]
pure_crit = df_ml.groupby('criterion')['finetuned_pure_percentage'].mean().sort_values(ascending=True)
colors = [CRITERION_BLUES[list(pure_crit.index).index(c) % len(CRITERION_BLUES)] for c in pure_crit.index]
bars = ax.barh(range(len(pure_crit)), pure_crit.values, color=colors, edgecolor='black', linewidth=0.5)
ax.set_yticks(range(len(pure_crit)))
ax.set_yticklabels(pure_crit.index)
ax.set_xlabel('% Pure Clusters')
ax.set_title('% Pure Clusters Average')
ax.grid(axis='x', alpha=0.3)
ax.set_xlim(0, pure_crit.max() * 1.18)
for i, v in enumerate(pure_crit.values):
    ax.text(v + 0.5, i, f'{v:.1f}%', va='center', fontsize=8)

# Number of clusters
ax = axes[1, 2]
clusters_crit = df_ml.groupby('criterion')['finetuned_n_clusters'].mean().sort_values(ascending=True)
colors = [CRITERION_BLUES[list(clusters_crit.index).index(c) % len(CRITERION_BLUES)] for c in clusters_crit.index]
bars = ax.barh(range(len(clusters_crit)), clusters_crit.values, color=colors, edgecolor='black', linewidth=0.5)
ax.set_yticks(range(len(clusters_crit)))
ax.set_yticklabels(clusters_crit.index)
ax.set_xlabel('Number of Clusters')
ax.set_title('Number of Clusters Average')
ax.grid(axis='x', alpha=0.3)
ax.set_xlim(0, clusters_crit.max() * 1.18)
for i, v in enumerate(clusters_crit.values):
    ax.text(v + 1, i, f'{v:.0f}', va='center', fontsize=8)

plt.tight_layout()
plt.savefig(PLOTS_DIR / '03_criterion_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# 4. COMPARISON FRONT vs FRONT+REAR - 2x2
print("[4/14] Comparison by views...")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Comparison Front vs Front+Rear', fontsize=14, fontweight='bold')

# Average accuracy by view
ax = axes[0, 0]
acc_views = df.groupby('views')['best_val_accuracy'].mean()
bars = ax.bar(acc_views.index, acc_views.values, color=[COLOR_FRONT, COLOR_FRONT_REAR], edgecolor='black', linewidth=0.5)
ax.set_title('Accuracy Average by View')
ax.set_ylabel('Accuracy')
ax.set_xlabel('View')
ax.grid(axis='y', alpha=0.3)
for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{height:.4f}', xy=(bar.get_x() + bar.get_width()/2., height),
                xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=9)

# Average recall by view
ax = axes[0, 1]
views_stats = df.groupby('views')[['best_val_recall@1', 'best_val_recall@3', 'best_val_recall@5']].mean()
x = np.arange(len(views_stats))
width = 0.25
bars1 = ax.bar(x - width, views_stats['best_val_recall@1'], width, label='Recall@1', color=COLOR_RECALL1, edgecolor='black', linewidth=0.5)
bars2 = ax.bar(x, views_stats['best_val_recall@3'], width, label='Recall@3', color=COLOR_RECALL3, edgecolor='black', linewidth=0.5)
bars3 = ax.bar(x + width, views_stats['best_val_recall@5'], width, label='Recall@5', color=COLOR_RECALL5, edgecolor='black', linewidth=0.5)
ax.set_title('Recall Average by View')
ax.set_ylabel('Recall')
ax.set_xlabel('View')
ax.set_xticks(x)
ax.set_xticklabels(views_stats.index)
ax.legend(loc='upper left')
ax.grid(axis='y', alpha=0.3)

# ARI
ax = axes[1, 0]
ari_views = df.groupby('views')[['finetuned_ari']].mean()
bars = ax.bar(ari_views.index, ari_views['finetuned_ari'], color=[COLOR_FRONT, COLOR_FRONT_REAR], edgecolor='black', linewidth=0.5)
ax.set_title('ARI Average by View')
ax.set_ylabel('ARI')
ax.set_xlabel('View')
ax.grid(axis='y', alpha=0.3)
for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{height:.4f}', xy=(bar.get_x() + bar.get_width()/2., height),
                xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=9)

# Pure clusters
ax = axes[1, 1]
pure_views = df.groupby('views')[['finetuned_pure_percentage']].mean()
bars = ax.bar(pure_views.index, pure_views['finetuned_pure_percentage'], color=[COLOR_FRONT, COLOR_FRONT_REAR], edgecolor='black', linewidth=0.5)
ax.set_title('% Pure Clusters by View')
ax.set_ylabel('% Pure Clusters')
ax.set_xlabel('View')
ax.grid(axis='y', alpha=0.3)
for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2., height),
                xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(PLOTS_DIR / '04_views_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# 5. HEATMAP MODEL x CRITERION x RECALL@1
print("[5/14] Heatmap model x criterion x recall@1...")

pivot_recall1 = df.pivot_table(values='best_val_recall@1', 
                                 index='criterion', 
                                 columns='model_name', 
                                 aggfunc='mean')

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(pivot_recall1, annot=True, fmt='.3f', cmap='Blues', 
            cbar_kws={'label': 'Recall@1'}, ax=ax, linewidths=0.5)
ax.set_title('Recall@1: Model x Criterion', fontsize=12, fontweight='bold', pad=15)
ax.set_xlabel('Model')
ax.set_ylabel('Criterion')
plt.tight_layout()
plt.savefig(PLOTS_DIR / '05_heatmap_recall1.png', dpi=300, bbox_inches='tight')
plt.close()

# 5b. HEATMAP MODEL x CRITERION x RECALL@3
print("[5b/14] Heatmap model x criterion x recall@3...")

pivot_recall3 = df.pivot_table(values='best_val_recall@3', 
                                 index='criterion', 
                                 columns='model_name', 
                                 aggfunc='mean')

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(pivot_recall3, annot=True, fmt='.3f', cmap='Blues', 
            cbar_kws={'label': 'Recall@3'}, ax=ax, linewidths=0.5)
ax.set_title('Recall@3: Model x Criterion', fontsize=12, fontweight='bold', pad=15)
ax.set_xlabel('Model')
ax.set_ylabel('Criterion')
plt.tight_layout()
plt.savefig(PLOTS_DIR / '05b_heatmap_recall3.png', dpi=300, bbox_inches='tight')
plt.close()

# 5c. HEATMAP MODEL x CRITERION x RECALL@5
print("[5c/14] Heatmap model x criterion x recall@5...")

pivot_recall5 = df.pivot_table(values='best_val_recall@5', 
                                 index='criterion', 
                                 columns='model_name', 
                                 aggfunc='mean')

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(pivot_recall5, annot=True, fmt='.3f', cmap='Blues', 
            cbar_kws={'label': 'Recall@5'}, ax=ax, linewidths=0.5)
ax.set_title('Recall@5: Model x Criterion', fontsize=12, fontweight='bold', pad=15)
ax.set_xlabel('Model')
ax.set_ylabel('Criterion')
plt.tight_layout()
plt.savefig(PLOTS_DIR / '05c_heatmap_recall5.png', dpi=300, bbox_inches='tight')
plt.close()

# 6. HEATMAP MODEL x CRITERION x ARI
print("[6/14] Heatmap model x criterion x ARI...")

pivot_ari = df.pivot_table(values='finetuned_ari', 
                            index='criterion', 
                            columns='model_name', 
                            aggfunc='mean')

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(pivot_ari, annot=True, fmt='.4f', cmap='Blues', 
            cbar_kws={'label': 'ARI'}, ax=ax, linewidths=0.5)
ax.set_title('ARI: Model x Criterion', fontsize=12, fontweight='bold', pad=15)
ax.set_xlabel('Model')
ax.set_ylabel('Criterion')
plt.tight_layout()
plt.savefig(PLOTS_DIR / '06_heatmap_ari.png', dpi=300, bbox_inches='tight')
plt.close()

# 7. DISTRIBUTION OF PURE VS MIXED CLUSTERS
print("[7/14] Distribution: pure vs mixed clusters...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Distribution of Clusters: Pure vs Mixed', fontsize=14, fontweight='bold')

# By model
ax = axes[0]
model_cluster_data = df.groupby('model_name')[['finetuned_n_pure_clusters', 'finetuned_n_mixed_clusters']].mean()
x = np.arange(len(model_cluster_data))
width = 0.35
bars1 = ax.bar(x - width/2, model_cluster_data['finetuned_n_pure_clusters'], width, 
               label='Pure Clusters', color='#2171b5', edgecolor='black', linewidth=0.5)
bars2 = ax.bar(x + width/2, model_cluster_data['finetuned_n_mixed_clusters'], width,
               label='Mixed Clusters', color='#969696', edgecolor='black', linewidth=0.5)
ax.set_xlabel('Model')
ax.set_ylabel('Number of Clusters')
ax.set_title('By Model')
ax.set_xticks(x)
ax.set_xticklabels(model_cluster_data.index, rotation=0)
ax.legend()
ax.grid(axis='y', alpha=0.3)

# By objective
ax = axes[1]
obj_cluster_data = df.groupby('objective')[['finetuned_n_pure_clusters', 'finetuned_n_mixed_clusters']].mean()
x = np.arange(len(obj_cluster_data))
bars1 = ax.bar(x - width/2, obj_cluster_data['finetuned_n_pure_clusters'], width,
               label='Pure Clusters', color='#2171b5', edgecolor='black', linewidth=0.5)
bars2 = ax.bar(x + width/2, obj_cluster_data['finetuned_n_mixed_clusters'], width,
               label='Mixed Clusters', color='#969696', edgecolor='black', linewidth=0.5)
ax.set_xlabel('Objective')
ax.set_ylabel('Number of Clusters')
ax.set_title('By Objective')
ax.set_xticks(x)
ax.set_xticklabels(obj_cluster_data.index, rotation=45)
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / '07_cluster_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# 8. CLUSTERING QUALITY: % PURE CLUSTERS
print("[8/14] Clustering quality...")

fig, ax = plt.subplots(figsize=(14, 10))

# Sort by % pure clusters
df_sorted = df.sort_values('finetuned_pure_percentage', ascending=True)

# Create more compact labels
labels = df_sorted.apply(lambda x: f"{x['model_name']} | {x['objective'][:6]} | {x['criterion'] if pd.notna(x['criterion']) else 'class'} | {x['views']}", axis=1)

# Color by model_name
colors = [COLOR_RESNET if m == 'resnet50' else COLOR_VIT for m in df_sorted['model_name']]

bars = ax.barh(range(len(df_sorted)), df_sorted['finetuned_pure_percentage'], color=colors, 
               edgecolor='black', linewidth=0.5, height=0.7)
ax.set_yticks(range(len(df_sorted)))
ax.set_yticklabels(labels, fontsize=7)
ax.set_xlabel('% Pure Clusters')
ax.set_title('Clustering Quality: % Pure Clusters by Configuration', fontsize=12, fontweight='bold')
ax.grid(axis='x', alpha=0.3)

# Adjust xlim so annotations fit
max_val = df_sorted['finetuned_pure_percentage'].max()
ax.set_xlim(0, max_val * 1.25)

# Add values with smaller font size
for i, (v, ari) in enumerate(zip(df_sorted['finetuned_pure_percentage'], df_sorted['finetuned_ari'])):
    ax.annotate(f'{v:.1f}% (ARI: {ari:.3f})', xy=(v, i), xytext=(3, 0), 
                textcoords='offset points', va='center', fontsize=6)

# Legend
legend_elements = [Patch(facecolor=COLOR_RESNET, label='ResNet50', edgecolor='black'),
                   Patch(facecolor=COLOR_VIT, label='ViT-B/32', edgecolor='black')]
ax.legend(handles=legend_elements, loc='lower right')

plt.tight_layout()
plt.savefig(PLOTS_DIR / '08_clustering_quality_ranking.png', dpi=300, bbox_inches='tight')
plt.close()

# 9. PROBLEMATIC CLASSES
print("[9/14] Problematic class analysis...")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Problematic Class Analysis (Overlap)', fontsize=14, fontweight='bold')

# By model
ax = axes[0, 0]
model_prob = df.groupby('model_name')['finetuned_n_overlapping_classes'].mean()
bars = ax.bar(model_prob.index, model_prob.values, color=[COLOR_RESNET, COLOR_VIT], edgecolor='black', linewidth=0.5)
ax.set_title('Classes in Multiple Clusters - By Model')
ax.set_ylabel('Number of Problematic Classes')
ax.set_xlabel('Model')
ax.grid(axis='y', alpha=0.3)
for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width()/2., height),
                xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=9)

# By objective
ax = axes[0, 1]
obj_prob = df.groupby('objective')['finetuned_n_overlapping_classes'].mean()
bars = ax.bar(obj_prob.index, obj_prob.values, color=[COLOR_CLASSIFICATION, COLOR_METRIC_LEARNING], edgecolor='black', linewidth=0.5)
ax.set_title('Problematic Classes - By Objective')
ax.set_ylabel('Number of Problematic Classes')
ax.set_xlabel('Objective')
ax.grid(axis='y', alpha=0.3)
for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width()/2., height),
                xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=9)

# By criterion (metric learning)
ax = axes[1, 0]
crit_prob = df_ml.groupby('criterion')['finetuned_n_overlapping_classes'].mean().sort_values()
colors_crit = [CRITERION_BLUES[list(crit_prob.index).index(c) % len(CRITERION_BLUES)] for c in crit_prob.index]
bars = ax.barh(range(len(crit_prob)), crit_prob.values, color=colors_crit, edgecolor='black', linewidth=0.5)
ax.set_yticks(range(len(crit_prob)))
ax.set_yticklabels(crit_prob.index)
ax.set_xlabel('Number of Problematic Classes')
ax.set_title('Problematic Classes - By Criterion')
ax.grid(axis='x', alpha=0.3)
ax.set_xlim(0, crit_prob.max() * 1.2)
for i, v in enumerate(crit_prob.values):
    ax.annotate(f'{v:.1f}', xy=(v, i), xytext=(3, 0), textcoords='offset points', va='center', fontsize=8)

# Relationship: ARI vs problematic classes
ax = axes[1, 1]
scatter = ax.scatter(df['finetuned_n_overlapping_classes'], df['finetuned_ari'], 
                     c=df['finetuned_pure_percentage'], s=100, cmap='Blues', 
                     alpha=0.7, edgecolors='black', linewidth=0.5)
ax.set_xlabel('Number of Problematic Classes')
ax.set_ylabel('ARI')
ax.set_title('Correlation: Problematic Classes vs ARI')
ax.grid(alpha=0.3)
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('% Pure Clusters')

# Compute correlation
corr = df[['finetuned_n_overlapping_classes', 'finetuned_ari']].corr().iloc[0, 1]
ax.text(0.05, 0.95, f'Correlation: {corr:.3f}', 
        transform=ax.transAxes, fontsize=9, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(PLOTS_DIR / '09_problematic_classes.png', dpi=300, bbox_inches='tight')
plt.close()
plt.close()

# 10. BASELINE vs FINETUNED IMPROVEMENT
print("[10/14] Baseline vs fine-tuned improvement analysis...")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Fine-tuning Improvement: Baseline x Finetuned', fontsize=14, fontweight='bold')

# Absolute improvement by model
ax = axes[0, 0]
model_improvement = df.groupby('model_name')['ari_improvement'].mean()
bars = ax.bar(model_improvement.index, model_improvement.values, color=[COLOR_RESNET, COLOR_VIT], edgecolor='black', linewidth=0.5)
ax.set_title('ARI Improvement - By Model')
ax.set_ylabel('ARI Improvement')
ax.set_xlabel('Model')
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.grid(axis='y', alpha=0.3)
for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{height:.4f}', xy=(bar.get_x() + bar.get_width()/2., height),
                xytext=(0, 3 if height > 0 else -10), textcoords='offset points', ha='center', fontsize=9)

# By objective
ax = axes[0, 1]
obj_improvement = df.groupby('objective')['ari_improvement'].mean()
bars = ax.bar(obj_improvement.index, obj_improvement.values, color=[COLOR_CLASSIFICATION, COLOR_METRIC_LEARNING], edgecolor='black', linewidth=0.5)
ax.set_title('ARI Improvement - By Objective')
ax.set_ylabel('ARI Improvement')
ax.set_xlabel('Objective')
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.grid(axis='y', alpha=0.3)
for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{height:.4f}', xy=(bar.get_x() + bar.get_width()/2., height),
                xytext=(0, 3 if height > 0 else -10), textcoords='offset points', ha='center', fontsize=9)

# Comparison baseline vs finetuned
ax = axes[1, 0]
comparison_data = df[['baseline_ari', 'finetuned_ari']].mean()
bars = ax.bar(['Baseline', 'Finetuned'], comparison_data.values, color=[COLOR_BASELINE, COLOR_FINETUNED], edgecolor='black', linewidth=0.5)
ax.set_title('ARI: Baseline vs Finetuned (Average)')
ax.set_ylabel('ARI')
ax.grid(axis='y', alpha=0.3)
for bar, val in zip(bars, comparison_data.values):
    height = bar.get_height()
    ax.annotate(f'{val:.4f}', xy=(bar.get_x() + bar.get_width()/2., height),
                xytext=(0, 3), textcoords='offset points', ha='center', fontsize=10, fontweight='bold')

# Improvement by view
ax = axes[1, 1]
views_improvement = df.groupby('views')['ari_improvement'].mean()
bars = ax.bar(views_improvement.index, views_improvement.values, color=[COLOR_FRONT, COLOR_FRONT_REAR], edgecolor='black', linewidth=0.5)
ax.set_title('ARI Improvement - By View')
ax.set_ylabel('ARI Improvement')
ax.set_xlabel('View')
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.grid(axis='y', alpha=0.3)
for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{height:.4f}', xy=(bar.get_x() + bar.get_width()/2., height),
                xytext=(0, 3 if height > 0 else -10), textcoords='offset points', ha='center', fontsize=9)

plt.tight_layout()
plt.savefig(PLOTS_DIR / '10_finetuning_improvement.png', dpi=300, bbox_inches='tight')
plt.close()

# 11. TEMPORAL EFFICIENCY
print("[11/14] Temporal efficiency analysis...")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Temporal Efficiency Analysis', fontsize=14, fontweight='bold')

# Time by model
ax = axes[0, 0]
model_time = df.groupby('model_name')['duration_minutes'].mean()
bars = ax.bar(model_time.index, model_time.values, color=[COLOR_RESNET, COLOR_VIT], edgecolor='black', linewidth=0.5)
ax.set_title('Training Time - By Model')
ax.set_ylabel('Duration (minutes)')
ax.set_xlabel('Model')
ax.grid(axis='y', alpha=0.3)
for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{height:.0f} min', xy=(bar.get_x() + bar.get_width()/2., height),
                xytext=(0, 3), textcoords='offset points', ha='center', fontsize=9)

# Time by objective
ax = axes[0, 1]
obj_time = df.groupby('objective')['duration_minutes'].mean()
bars = ax.bar(obj_time.index, obj_time.values, color=[COLOR_CLASSIFICATION, COLOR_METRIC_LEARNING], edgecolor='black', linewidth=0.5)
ax.set_title('Training Time - By Objective')
ax.set_ylabel('Duration (minutes)')
ax.set_xlabel('Objective')
ax.grid(axis='y', alpha=0.3)
for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{height:.0f} min', xy=(bar.get_x() + bar.get_width()/2., height),
                xytext=(0, 3), textcoords='offset points', ha='center', fontsize=9)

# Efficiency (ARI / time)
ax = axes[1, 0]
df['efficiency'] = df['finetuned_ari'] / (df['duration_minutes'] / 60)  # ARI per hour
model_eff = df.groupby('model_name')['efficiency'].mean()
bars = ax.bar(model_eff.index, model_eff.values, color=[COLOR_RESNET, COLOR_VIT], edgecolor='black', linewidth=0.5)
ax.set_title('Efficiency (ARI / Hour)')
ax.set_ylabel('ARI per Hour')
ax.set_xlabel('Model')
ax.grid(axis='y', alpha=0.3)
for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{height:.4f}', xy=(bar.get_x() + bar.get_width()/2., height),
                xytext=(0, 3), textcoords='offset points', ha='center', fontsize=9)

# Relationship: time vs ARI
ax = axes[1, 1]
scatter = ax.scatter(df['duration_minutes'], df['finetuned_ari'], 
                     c=df['finetuned_pure_percentage'], s=100, cmap='Blues',
                     alpha=0.7, edgecolors='black', linewidth=0.5)
ax.set_xlabel('Duration (minutes)')
ax.set_ylabel('ARI')
ax.set_title('Relationship Time vs ARI')
ax.grid(alpha=0.3)
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('% Pure Clusters')

# Compute correlation
corr = df[['duration_minutes', 'finetuned_ari']].corr().iloc[0, 1]
ax.text(0.05, 0.95, f'Correlation: {corr:.3f}', 
        transform=ax.transAxes, fontsize=9, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(PLOTS_DIR / '11_temporal_efficiency.png', dpi=300, bbox_inches='tight')
plt.close()

# 12. TOP-5 COMPARATIVE SUMMARY
print("[12/14] Top-5 comparative summary...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Top-5 Configurations by Metric', fontsize=14, fontweight='bold')

# Top-5 by ARI
ax = axes[0, 0]
top_ari = df.nlargest(5, 'finetuned_ari')
labels = top_ari.apply(lambda x: f"{x['model_name']} | {x['criterion'] if pd.notna(x['criterion']) else 'class'} | {x['views']}", axis=1)
colors = [COLOR_RESNET if 'resnet' in x.lower() else COLOR_VIT for x in top_ari['model_name']]
bars = ax.barh(range(5), top_ari['finetuned_ari'].values, color=colors, edgecolor='black', linewidth=0.5)
ax.set_yticks(range(5))
ax.set_yticklabels(labels, fontsize=8)
ax.set_xlabel('ARI')
ax.set_title('Top-5 by ARI')
ax.grid(axis='x', alpha=0.3)
ax.set_xlim(0, top_ari['finetuned_ari'].max() * 1.15)
for i, v in enumerate(top_ari['finetuned_ari'].values):
    ax.annotate(f'{v:.4f}', xy=(v, i), xytext=(3, 0), textcoords='offset points', va='center', fontsize=8, fontweight='bold')

# Top-5 by Recall@1
ax = axes[0, 1]
top_recall1 = df.nlargest(5, 'best_val_recall@1')
labels = top_recall1.apply(lambda x: f"{x['model_name']} | {x['criterion'] if pd.notna(x['criterion']) else 'class'} | {x['views']}", axis=1)
colors = [COLOR_RESNET if 'resnet' in x.lower() else COLOR_VIT for x in top_recall1['model_name']]
bars = ax.barh(range(5), top_recall1['best_val_recall@1'].values, color=colors, edgecolor='black', linewidth=0.5)
ax.set_yticks(range(5))
ax.set_yticklabels(labels, fontsize=8)
ax.set_xlabel('Recall@1')
ax.set_title('Top-5 by Recall@1')
ax.grid(axis='x', alpha=0.3)
ax.set_xlim(0, top_recall1['best_val_recall@1'].max() * 1.15)
for i, v in enumerate(top_recall1['best_val_recall@1'].values):
    ax.annotate(f'{v:.3f}', xy=(v, i), xytext=(3, 0), textcoords='offset points', va='center', fontsize=8, fontweight='bold')

# Top-5 by % Pure Clusters
ax = axes[1, 0]
top_pure = df.nlargest(5, 'finetuned_pure_percentage')
labels = top_pure.apply(lambda x: f"{x['model_name']} | {x['criterion'] if pd.notna(x['criterion']) else 'class'} | {x['views']}", axis=1)
colors = [COLOR_RESNET if 'resnet' in x.lower() else COLOR_VIT for x in top_pure['model_name']]
bars = ax.barh(range(5), top_pure['finetuned_pure_percentage'].values, color=colors, edgecolor='black', linewidth=0.5)
ax.set_yticks(range(5))
ax.set_yticklabels(labels, fontsize=8)
ax.set_xlabel('% Pure Clusters')
ax.set_title('Top-5 by % Pure Clusters')
ax.grid(axis='x', alpha=0.3)
ax.set_xlim(0, top_pure['finetuned_pure_percentage'].max() * 1.15)
for i, v in enumerate(top_pure['finetuned_pure_percentage'].values):
    ax.annotate(f'{v:.1f}%', xy=(v, i), xytext=(3, 0), textcoords='offset points', va='center', fontsize=8, fontweight='bold')

# Top-5 by efficiency
ax = axes[1, 1]
top_eff = df.nlargest(5, 'efficiency')
labels = top_eff.apply(lambda x: f"{x['model_name']} | {x['criterion'] if pd.notna(x['criterion']) else 'class'} | {x['views']}", axis=1)
colors = [COLOR_RESNET if 'resnet' in x.lower() else COLOR_VIT for x in top_eff['model_name']]
bars = ax.barh(range(5), top_eff['efficiency'].values, color=colors, edgecolor='black', linewidth=0.5)
ax.set_yticks(range(5))
ax.set_yticklabels(labels, fontsize=8)
ax.set_xlabel('ARI / Hour')
ax.set_title('Top-5 by Efficiency')
ax.grid(axis='x', alpha=0.3)
ax.set_xlim(0, top_eff['efficiency'].max() * 1.15)
for i, v in enumerate(top_eff['efficiency'].values):
    ax.annotate(f'{v:.4f}', xy=(v, i), xytext=(3, 0), textcoords='offset points', va='center', fontsize=8, fontweight='bold')

plt.tight_layout()
plt.savefig(PLOTS_DIR / '12_top5_summary.png', dpi=300, bbox_inches='tight')
plt.close()

print()
print("="*80)
print("VISUALIZATIONS COMPLETED")
print("="*80)
print(f"14 plots saved in: {PLOTS_DIR}")
print()
print("Generated files:")
for png_file in sorted(PLOTS_DIR.glob('*.png')):
    print(f"   {png_file.name}")
print()







