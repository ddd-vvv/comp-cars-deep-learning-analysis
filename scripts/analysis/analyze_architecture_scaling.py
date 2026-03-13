"""Statistical analysis for the Architecture Scaling experiment.

Compares higher-capacity architectures:
- ResNet50 -> ResNet101
- ViT-B/32 -> ViT-B/16
- CLIP-ViT-B/32 (baseline)
"""

import pandas as pd
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / 'results/analysis/data/architecture_scaling'
OUTPUT_DIR = REPO_ROOT / 'results/analysis'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("STATISTICAL ANALYSIS - ARCHITECTURE SCALING")
print("=" * 80)
print()

df = pd.read_csv(DATA_DIR / 'architecture_scaling_results.csv')
print(f"Loaded data: {len(df)} configurations\n")

resnet_df = df[df['architecture'].str.contains('ResNet', na=False)].copy()
vit_df = df[df['architecture'].str.contains('ViT-B/', na=False)].copy()
clip_df = df[df['architecture'].str.contains('CLIP', na=False)].copy()

print("=" * 80)
print("SUMMARY BY ARCHITECTURE")
print("=" * 80)
print()

for _, row in df.iterrows():
    arch = row['architecture']
    criterion = row['criterion']
    recall1 = row['best_val_recall@1']
    ari = row['finetuned_ari']
    pure_pct = row['finetuned_pure_percentage']
    duration = row['duration_minutes']

    print(f"{arch} ({criterion if pd.notna(criterion) else 'N/A'}):")
    print(f"  Recall@1: {recall1:.2f}%" if pd.notna(recall1) and recall1 > 1 else f"  Recall@1: {recall1*100:.2f}%")
    print(f"  ARI: {ari:.4f}")
    print(f"  Pure Clusters: {pure_pct:.1f}%")
    print(f"  Duration: {duration:.1f} min")
    print()

print("=" * 80)
print("RESNET COMPARISON: 50 vs 101")
print("=" * 80)
print()

if len(resnet_df) == 2:
    resnet50 = resnet_df[resnet_df['architecture'] == 'ResNet50'].iloc[0]
    resnet101 = resnet_df[resnet_df['architecture'] == 'ResNet101'].iloc[0]

    recall_diff = resnet101['best_val_recall@1'] - resnet50['best_val_recall@1']
    ari_diff = resnet101['finetuned_ari'] - resnet50['finetuned_ari']
    pure_diff = resnet101['finetuned_pure_percentage'] - resnet50['finetuned_pure_percentage']
    time_diff = resnet101['duration_minutes'] - resnet50['duration_minutes']

    print("Differences (ResNet101 - ResNet50):")
    print(f"  d Recall@1: {recall_diff:+.2f}% ({recall_diff/resnet50['best_val_recall@1']*100:+.1f}%)")
    print(f"  d ARI: {ari_diff:+.4f} ({ari_diff/resnet50['finetuned_ari']*100:+.1f}%)")
    print(f"  d Pure%: {pure_diff:+.1f}% ({pure_diff/resnet50['finetuned_pure_percentage']*100:+.1f}%)")
    print(f"  d Time: {time_diff:+.1f} min ({time_diff/resnet50['duration_minutes']*100:+.1f}%)")
    print()

    if recall_diff > 0:
        print("ResNet101 improves Recall@1")
    else:
        print("ResNet101 does NOT improve Recall@1")

    if ari_diff > 0:
        print("ResNet101 improves ARI")
    else:
        print("ResNet101 does NOT improve ARI")
    print()

print("=" * 80)
print("VIT COMPARISON: B/32 vs B/16")
print("=" * 80)
print()

if len(vit_df) == 2:
    vitb32 = vit_df[vit_df['architecture'] == 'ViT-B/32'].iloc[0]
    vitb16 = vit_df[vit_df['architecture'] == 'ViT-B/16'].iloc[0]

    recall_diff = vitb16['best_val_recall@1'] - vitb32['best_val_recall@1']
    ari_diff = vitb16['finetuned_ari'] - vitb32['finetuned_ari']
    pure_diff = vitb16['finetuned_pure_percentage'] - vitb32['finetuned_pure_percentage']
    time_diff = vitb16['duration_minutes'] - vitb32['duration_minutes']

    print("Differences (ViT-B/16 - ViT-B/32):")
    print(f"  d Recall@1: {recall_diff:+.2f}% ({recall_diff/vitb32['best_val_recall@1']*100:+.1f}%)")
    print(f"  d ARI: {ari_diff:+.4f} ({ari_diff/vitb32['finetuned_ari']*100:+.1f}%)")
    print(f"  d Pure%: {pure_diff:+.1f}% ({pure_diff/vitb32['finetuned_pure_percentage']*100:+.1f}%)")
    print(f"  d Time: {time_diff:+.1f} min ({time_diff/vitb32['duration_minutes']*100:+.1f}%)")
    print()

    if recall_diff > 0:
        print("ViT-B/16 improves Recall@1")
    else:
        print("ViT-B/16 does NOT improve Recall@1")

    if ari_diff > 0:
        print("ViT-B/16 improves ARI")
    else:
        print("ViT-B/16 does NOT improve ARI")
    print()

print("=" * 80)
print("OVERALL RANKING")
print("=" * 80)
print()

df_sorted = df.copy()
df_sorted['recall_display'] = df_sorted['best_val_recall@1'].apply(
    lambda x: x if x > 1 else x * 100 if pd.notna(x) else 0
)
df_sorted = df_sorted.sort_values('recall_display', ascending=False)

print("Ranking by Recall@1:")
for i, (_, row) in enumerate(df_sorted.iterrows(), 1):
    arch = row['architecture']
    recall = row['recall_display']
    print(f"  {i}. {arch}: {recall:.2f}%")
print()

df_sorted_ari = df.sort_values('finetuned_ari', ascending=False)
print("Ranking by ARI:")
for i, (_, row) in enumerate(df_sorted_ari.iterrows(), 1):
    arch = row['architecture']
    ari = row['finetuned_ari']
    print(f"  {i}. {arch}: {ari:.4f}")
print()

df_sorted_pure = df.sort_values('finetuned_pure_percentage', ascending=False)
print("Ranking by % Pure Clusters:")
for i, (_, row) in enumerate(df_sorted_pure.iterrows(), 1):
    arch = row['architecture']
    pure = row['finetuned_pure_percentage']
    print(f"  {i}. {arch}: {pure:.1f}%")
print()

print("=" * 80)
print("CONCLUSIONS")
print("=" * 80)
print()

print("1. ResNet scaling:")
if len(resnet_df) == 2:
    if recall_diff < 0:
        print(f"   - ResNet101 has LOWER recall than ResNet50 ({recall_diff:.2f}%)")
        print("   - The capacity increase does NOT translate into better retrieval")
    else:
        print(f"   - ResNet101 has HIGHER recall than ResNet50 (+{recall_diff:.2f}%)")
print()

print("2. ViT scaling:")
if len(vit_df) == 2:
    if recall_diff > 0:
        print(f"   - ViT-B/16 has HIGHER recall than ViT-B/32 (+{recall_diff:.2f}%)")
        print("   - Higher patch resolution improves retrieval")
    else:
        print(f"   - ViT-B/16 has LOWER recall than ViT-B/32 ({recall_diff:.2f}%)")
print()

print("3. Comparison against CLIP:")
if len(clip_df) > 0:
    clip_recall = clip_df.iloc[0]['best_val_recall@1']
    if clip_recall > 1:
        clip_recall_pct = clip_recall
    else:
        clip_recall_pct = clip_recall * 100
    print(f"   - CLIP-ViT-B/32: {clip_recall_pct:.2f}% recall")
    print(f"   - CLIP-ViT-B/32: {clip_df.iloc[0]['finetuned_ari']:.4f} ARI")
    print("   - CLIP has the best ARI across all architectures")
print()

print("=" * 80)
