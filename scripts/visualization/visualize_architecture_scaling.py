#!/usr/bin/env python3
"""
Visualization script for Architecture Scaling.
Compares base vs scaled architectures:
- ResNet50 -> ResNet101
- ViT-B/32 -> ViT-B/16
- CLIP-ViT-B/32
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from pathlib import Path

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

COLOR_RESNET50 = '#08519c'
COLOR_RESNET101 = '#3182bd'
COLOR_VIT32 = '#6baed6'
COLOR_VIT16 = '#2171b5'
COLOR_CLIP = '#08306b'
COLOR_RECALL1 = '#08519c'
COLOR_RECALL3 = '#3182bd'
COLOR_RECALL5 = '#9ecae1'

COLORS_ARCH = {
    'ResNet50': COLOR_RESNET50,
    'ResNet101': COLOR_RESNET101,
    'ViT-B/32': COLOR_VIT32,
    'ViT-B/16': COLOR_VIT16,
    'CLIP-ViT-B/32': COLOR_CLIP,
}

# Paths
RESULTS_DIR = REPO_ROOT / 'results/analysis'
PLOTS_DIR = REPO_ROOT / 'results/analysis/plots/architecture_scaling'
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Load data
print("="*80)
print("GENERATING VISUALIZATIONS - ARCHITECTURE SCALING")
print("="*80)
print()

df = pd.read_csv(RESULTS_DIR / 'data/architecture_scaling/architecture_scaling_results.csv')
print(f"Loaded data: {len(df)} configurations\n")

# Normalize recall to percentage
df['recall@1_pct'] = df['best_val_recall@1'].apply(lambda x: x if x > 1 else x * 100 if pd.notna(x) else np.nan)
df['recall@3_pct'] = df['best_val_recall@3'].apply(lambda x: x if x > 1 else x * 100 if pd.notna(x) else np.nan)
df['recall@5_pct'] = df['best_val_recall@5'].apply(lambda x: x if x > 1 else x * 100 if pd.notna(x) else np.nan)

# ============================================================================
# 1. GENERAL COMPARISON - ALL ARCHITECTURES
# ============================================================================
print("[1/6] General architecture comparison...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Architecture Scaling: Model Comparison', fontsize=16, fontweight='bold')

# Recall@1
ax = axes[0, 0]
colors = [COLORS_ARCH.get(arch, '#969696') for arch in df['architecture']]
bars = ax.bar(df['architecture'], df['recall@1_pct'], color=colors, edgecolor='black', linewidth=0.5)
ax.set_title('Recall@1 by Architecture')
ax.set_ylabel('Recall@1 (%)')
ax.set_xticklabels(df['architecture'], rotation=45, ha='right')
ax.grid(axis='y', alpha=0.3)
for bar in bars:
    height = bar.get_height()
    if pd.notna(height):
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=9)

# ARI
ax = axes[0, 1]
bars = ax.bar(df['architecture'], df['finetuned_ari'], color=colors, edgecolor='black', linewidth=0.5)
ax.set_title('ARI by Architecture')
ax.set_ylabel('ARI')
ax.set_xticklabels(df['architecture'], rotation=45, ha='right')
ax.grid(axis='y', alpha=0.3)
for bar in bars:
    height = bar.get_height()
    if pd.notna(height):
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)

# % Pure Clusters
ax = axes[1, 0]
bars = ax.bar(df['architecture'], df['finetuned_pure_percentage'], color=colors, edgecolor='black', linewidth=0.5)
ax.set_title('% Pure Clusters by Architecture')
ax.set_ylabel('% Pure Clusters')
ax.set_xticklabels(df['architecture'], rotation=45, ha='right')
ax.grid(axis='y', alpha=0.3)
for bar in bars:
    height = bar.get_height()
    if pd.notna(height):
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=9)

# Training time
ax = axes[1, 1]
bars = ax.bar(df['architecture'], df['duration_minutes'], color=colors, edgecolor='black', linewidth=0.5)
ax.set_title('Training Time')
ax.set_ylabel('Duration (minutes)')
ax.set_xticklabels(df['architecture'], rotation=45, ha='right')
ax.grid(axis='y', alpha=0.3)
for bar in bars:
    height = bar.get_height()
    if pd.notna(height):
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=9)

plt.tight_layout()
plt.savefig(PLOTS_DIR / '01_general_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# 2. COMPARISON RESNET: 50 vs 101
# ============================================================================
print("[2/6] Comparison ResNet50 vs ResNet101...")

resnet_df = df[df['architecture'].str.contains('ResNet', na=False)].copy()

if len(resnet_df) == 2:
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('ResNet: Architecture Scaling (50 to 101)', fontsize=16, fontweight='bold')
    
    # Recall
    ax = axes[0, 0]
    x = np.arange(len(resnet_df))
    width = 0.25
    ax.bar(x - width, resnet_df['recall@1_pct'], width, label='Recall@1', color=COLOR_RECALL1, edgecolor='black')
    ax.bar(x, resnet_df['recall@3_pct'], width, label='Recall@3', color=COLOR_RECALL3, edgecolor='black')
    ax.bar(x + width, resnet_df['recall@5_pct'], width, label='Recall@5', color=COLOR_RECALL5, edgecolor='black')
    ax.set_title('Recall by Architecture')
    ax.set_ylabel('Recall (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(resnet_df['architecture'])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # ARI
    ax = axes[0, 1]
    bars = ax.bar(resnet_df['architecture'], resnet_df['finetuned_ari'], 
                  color=[COLOR_RESNET50, COLOR_RESNET101], edgecolor='black')
    ax.set_title('ARI: Clustering Quality')
    ax.set_ylabel('ARI')
    ax.grid(axis='y', alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # Clusters
    ax = axes[1, 0]
    x = np.arange(len(resnet_df))
    width = 0.35
    ax.bar(x - width/2, resnet_df['finetuned_n_pure_clusters'], width, label='Pure', 
           color=COLOR_RESNET50, edgecolor='black')
    ax.bar(x + width/2, resnet_df['finetuned_n_mixed_clusters'], width, label='Mixed',
           color='#969696', edgecolor='black')
    ax.set_title('Cluster Distribution')
    ax.set_ylabel('Number of Clusters')
    ax.set_xticks(x)
    ax.set_xticklabels(resnet_df['architecture'])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Efficiency (Recall/Time)
    ax = axes[1, 1]
    efficiency = resnet_df['recall@1_pct'] / resnet_df['duration_minutes']
    bars = ax.bar(resnet_df['architecture'], efficiency,
                  color=[COLOR_RESNET50, COLOR_RESNET101], edgecolor='black')
    ax.set_title('Efficiency: Recall@1 per Minute')
    ax.set_ylabel('Recall@1 / minute')
    ax.grid(axis='y', alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / '02_resnet_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

# ============================================================================
# 3. COMPARISON VIT: B/32 vs B/16
# ============================================================================
print("[3/6] Comparison ViT-B/32 vs ViT-B/16...")

vit_df = df[df['architecture'].str.match(r'^ViT-B/', na=False)].copy()

if len(vit_df) == 2:
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('ViT: Resolution scaling (B/32  B/16)', fontsize=16, fontweight='bold')
    
    # Recall
    ax = axes[0, 0]
    x = np.arange(len(vit_df))
    width = 0.25
    ax.bar(x - width, vit_df['recall@1_pct'], width, label='Recall@1', color=COLOR_RECALL1, edgecolor='black')
    ax.bar(x, vit_df['recall@3_pct'], width, label='Recall@3', color=COLOR_RECALL3, edgecolor='black')
    ax.bar(x + width, vit_df['recall@5_pct'], width, label='Recall@5', color=COLOR_RECALL5, edgecolor='black')
    ax.set_title('Recall by Architecture')
    ax.set_ylabel('Recall (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(vit_df['architecture'])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # ARI
    ax = axes[0, 1]
    bars = ax.bar(vit_df['architecture'], vit_df['finetuned_ari'],
                  color=[COLOR_VIT32, COLOR_VIT16], edgecolor='black')
    ax.set_title('ARI: Clustering Quality')
    ax.set_ylabel('ARI')
    ax.grid(axis='y', alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        if pd.notna(height):
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # Clusters
    ax = axes[1, 0]
    x = np.arange(len(vit_df))
    width = 0.35
    ax.bar(x - width/2, vit_df['finetuned_n_pure_clusters'], width, label='Pure',
           color=COLOR_VIT32, edgecolor='black')
    ax.bar(x + width/2, vit_df['finetuned_n_mixed_clusters'], width, label='Mixed',
           color='#969696', edgecolor='black')
    ax.set_title('Cluster Distribution')
    ax.set_ylabel('Number of Clusters')
    ax.set_xticks(x)
    ax.set_xticklabels(vit_df['architecture'])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Efficiency
    ax = axes[1, 1]
    efficiency = vit_df['recall@1_pct'] / vit_df['duration_minutes']
    bars = ax.bar(vit_df['architecture'], efficiency,
                  color=[COLOR_VIT32, COLOR_VIT16], edgecolor='black')
    ax.set_title('Efficiency: Recall@1 per Minute')
    ax.set_ylabel('Recall@1 / minute')
    ax.grid(axis='y', alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        if pd.notna(height):
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / '03_vit_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

# ============================================================================
# 4. FULL RANKING
# ============================================================================
print("[4/6] Metric rankings...")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Rankings: All Architectures', fontsize=16, fontweight='bold')

# Ranking Recall@1 (NaN at the end)
ax = axes[0]
df_sorted = df.sort_values('recall@1_pct', ascending=True, na_position='first')
colors_sorted = [COLORS_ARCH.get(arch, '#969696') for arch in df_sorted['architecture']]
bars = ax.barh(range(len(df_sorted)), df_sorted['recall@1_pct'], color=colors_sorted, edgecolor='black')
ax.set_yticks(range(len(df_sorted)))
ax.set_yticklabels(df_sorted['architecture'])
ax.set_xlabel('Recall@1 (%)')
ax.set_title('Ranking by Recall@1')
ax.grid(axis='x', alpha=0.3)
for i, v in enumerate(df_sorted['recall@1_pct']):
    if pd.notna(v):
        ax.text(v, i, f'{v:.2f}', va='center', ha='left', fontweight='bold')

# Ranking ARI (NaN at the end)
ax = axes[1]
df_sorted = df.sort_values('finetuned_ari', ascending=True, na_position='first')
colors_sorted = [COLORS_ARCH.get(arch, '#969696') for arch in df_sorted['architecture']]
bars = ax.barh(range(len(df_sorted)), df_sorted['finetuned_ari'], color=colors_sorted, edgecolor='black')
ax.set_yticks(range(len(df_sorted)))
ax.set_yticklabels(df_sorted['architecture'])
ax.set_xlabel('ARI')
ax.set_title('Ranking by ARI')
ax.grid(axis='x', alpha=0.3)
for i, v in enumerate(df_sorted['finetuned_ari']):
    if pd.notna(v):
        ax.text(v + 0.01, i, f'{v:.3f}', va='center', fontweight='bold', fontsize=9)

# Ranking Pure% (NaN at the end)
ax = axes[2]
df_sorted = df.sort_values('finetuned_pure_percentage', ascending=True, na_position='first')
colors_sorted = [COLORS_ARCH.get(arch, '#969696') for arch in df_sorted['architecture']]
bars = ax.barh(range(len(df_sorted)), df_sorted['finetuned_pure_percentage'], color=colors_sorted, edgecolor='black')
ax.set_yticks(range(len(df_sorted)))
ax.set_yticklabels(df_sorted['architecture'])
ax.set_xlabel('% Pure Clusters')
ax.set_title('Ranking by % Pure')
ax.grid(axis='x', alpha=0.3)
for i, v in enumerate(df_sorted['finetuned_pure_percentage']):
    if pd.notna(v):
        ax.text(v + 1, i, f'{v:.1f}%', va='center', fontweight='bold', fontsize=9)

plt.tight_layout()
plt.savefig(PLOTS_DIR / '04_rankings.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# 5. SCATTER: RECALL vs ARI
# ============================================================================
print("[5/6] Scatter Recall vs ARI...")

fig, ax = plt.subplots(figsize=(10, 7))
fig.suptitle('Relationship: Recall@1 vs ARI', fontsize=16, fontweight='bold')

for _, row in df.iterrows():
    arch = row['architecture']
    color = COLORS_ARCH.get(arch, '#969696')
    ax.scatter(row['recall@1_pct'], row['finetuned_ari'], s=200, c=color, 
               alpha=0.7, edgecolors='black', linewidths=2, label=arch)
    ax.annotate(arch, (row['recall@1_pct'], row['finetuned_ari']),
                xytext=(5, 5), textcoords='offset points', fontsize=9, fontweight='bold')

ax.set_xlabel('Recall@1 (%)', fontweight='bold')
ax.set_ylabel('ARI', fontweight='bold')
ax.grid(alpha=0.3)
ax.legend(loc='lower right')

plt.tight_layout()
plt.savefig(PLOTS_DIR / '05_recall_vs_ari.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# 6. COMPARATIVE HEATMAP (RAW VALUES)
# ============================================================================
print("[6/6] Metrics heatmap...")

fig, ax = plt.subplots(figsize=(10, 6))
fig.suptitle('Heatmap: Metrics by Architecture', fontsize=16, fontweight='bold')

# Select raw (non-normalized) metrics
metrics = ['recall@1_pct', 'finetuned_ari', 'finetuned_pure_percentage']
labels = ['Recall@1 (%)', 'ARI', '% Pure']

data_raw = df[['architecture'] + metrics].copy()
pivot = data_raw.set_index('architecture')[metrics].T
pivot.index = labels

# Heatmap with raw values
sns.heatmap(pivot, annot=True, fmt='.2f', cmap='Blues', cbar_kws={'label': 'Value'},
            ax=ax, linewidths=0.5, mask=pivot.isna())
ax.set_xlabel('Architecture')
ax.set_ylabel('Metric')

plt.tight_layout()
plt.savefig(PLOTS_DIR / '06_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

print()
print("="*80)
print("VISUALIZATIONS COMPLETED")
print("="*80)
print(f"6 plots saved in: {PLOTS_DIR}")
print()
print("Generated files:")
print("  [OK] 01_general_comparison.png")
print("  [OK] 02_resnet_comparison.png")
print("  [OK] 03_vit_comparison.png")
print("  [OK] 04_rankings.png")
print("  [OK] 05_recall_vs_ari.png")
print("  [OK] 06_heatmap.png")
print("="*80)







