"""Statistical analysis for the CLIP Min Images study.

Analyzes the impact of data scarcity (4, 6, 8, 10, 12 images per class).
"""

import pandas as pd
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_data(csv_path: Path) -> pd.DataFrame:
    """Load the CLIP min-images results CSV."""
    if not csv_path.exists():
        raise FileNotFoundError(f"File not found: {csv_path}")

    df = pd.read_csv(csv_path)
    return df


def analyze_by_min_images(df: pd.DataFrame) -> pd.DataFrame:
    """Comparative analysis by minimum number of images."""
    print("\n" + "=" * 80)
    print("ANALYSIS BY MIN_IMAGES (Robustness Under Data Scarcity)")
    print("=" * 80)

    grouped = df.groupby('min_images').agg({
        'config_key': 'count',

        # Recall metrics
        'best_val_recall@1': ['mean', 'std', 'min', 'max'],
        'finetuned_recall@1': ['mean', 'std', 'min', 'max'],
        'finetuned_recall@3': ['mean', 'std', 'min', 'max'],
        'finetuned_recall@5': ['mean', 'std', 'min', 'max'],

        # Clustering
        'finetuned_ari': ['mean', 'std', 'min', 'max'],
        'finetuned_nmi': ['mean', 'std', 'min', 'max'],
        'finetuned_purity': ['mean', 'std', 'min', 'max'],

        # Cluster quality
        'finetuned_pure_percentage': ['mean', 'std', 'min', 'max'],
        'finetuned_n_pure_clusters': ['mean', 'std'],
        'finetuned_n_mixed_clusters': ['mean', 'std'],

        # Dataset stats
        'num_training_classes': 'mean',
        'train_samples': 'mean',
        'val_samples': 'mean',

        # Timing
        'duration_minutes': ['mean', 'std'],

        # Improvements
        'recall@1_improvement': ['mean', 'std'],
    }).round(4)

    grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]
    grouped = grouped.rename(columns={'config_key_count': 'num_experiments'})

    print("\n", grouped.to_string())

    print("\n" + "-" * 80)
    print("SCENARIO COMPARISON:")
    print("-" * 80)

    print("\nFinal Recall@1 by scenario:")
    for min_imgs in sorted(df['min_images'].unique()):
        recall = grouped.loc[min_imgs, 'finetuned_recall@1_mean']
        print(f"  {min_imgs} images: {recall:.4f}")

    print("\nARI by scenario:")
    for min_imgs in sorted(df['min_images'].unique()):
        ari = grouped.loc[min_imgs, 'finetuned_ari_mean']
        print(f"  {min_imgs} images: {ari:.4f}")

    print("\n% Pure Clusters by scenario:")
    for min_imgs in sorted(df['min_images'].unique()):
        pure_pct = grouped.loc[min_imgs, 'finetuned_pure_percentage_mean']
        print(f"  {min_imgs} images: {pure_pct:.2f}%")

    return grouped


def analyze_baseline_vs_standard(df: pd.DataFrame):
    """Compare baseline scenario (8 images) against other scenarios."""
    print("\n" + "=" * 80)
    print("ANALYSIS: BASELINE (8 IMAGES) vs OTHERS")
    print("=" * 80)

    if 8 not in df['min_images'].values:
        print("\n[WARNING] No data for min_images=8 (baseline)")
        return

    baseline = df[df['min_images'] == 8].iloc[0]

    print(f"\n--- BASELINE SCENARIO (8 images) ---")
    print(f"  Recall@1: {baseline['finetuned_recall@1']:.2f}%")
    print(f"  ARI: {baseline['finetuned_ari']:.4f}")
    print(f"  Pure %: {baseline['finetuned_pure_percentage']:.2f}%")
    print(f"  Training classes: {baseline['num_training_classes']}")
    print(f"  Training samples: {baseline['train_samples']}")

    print("\n--- COMPARISON WITH OTHER SCENARIOS ---")

    for min_imgs in sorted(df['min_images'].unique()):
        if min_imgs == 8:
            continue

        scenario = df[df['min_images'] == min_imgs].iloc[0]

        recall_diff = scenario['finetuned_recall@1'] - baseline['finetuned_recall@1']
        ari_diff = scenario['finetuned_ari'] - baseline['finetuned_ari']
        pure_diff = scenario['finetuned_pure_percentage'] - baseline['finetuned_pure_percentage']

        recall_pct = (recall_diff / baseline['finetuned_recall@1'] * 100) if baseline['finetuned_recall@1'] else 0
        ari_pct = (ari_diff / baseline['finetuned_ari'] * 100) if baseline['finetuned_ari'] else 0

        print(f"\n{min_imgs} images:")
        print(f"  Recall@1: {scenario['finetuned_recall@1']:.2f}% ({recall_diff:+.2f}%, {recall_pct:+.1f}%)")
        print(f"  ARI: {scenario['finetuned_ari']:.4f} ({ari_diff:+.4f}, {ari_pct:+.1f}%)")
        print(f"  Pure %: {scenario['finetuned_pure_percentage']:.2f}% ({pure_diff:+.2f}%)")
        print(f"  Classes: {scenario['num_training_classes']} ({int(scenario['num_training_classes'] - baseline['num_training_classes']):+d})")


def analyze_critical_vs_rich(df: pd.DataFrame):
    """Compare critical (4, 6) vs rich (10, 12) scenarios."""
    print("\n" + "=" * 80)
    print("ANALYSIS: CRITICAL vs RICH SCENARIOS")
    print("=" * 80)

    critical = df[df['min_images'].isin([4, 6])]
    rich = df[df['min_images'].isin([10, 12])]

    if critical.empty or rich.empty:
        print("\n[WARNING] Not enough data to compare")
        return

    print("\n--- CRITICAL SCENARIOS (4, 6 images) ---")
    critical_stats = critical.agg({
        'finetuned_recall@1': 'mean',
        'finetuned_ari': 'mean',
        'finetuned_pure_percentage': 'mean',
        'num_training_classes': 'mean',
        'train_samples': 'mean',
        'duration_minutes': 'mean',
    })
    print(f"  Mean Recall@1: {critical_stats['finetuned_recall@1']:.2f}%")
    print(f"  Mean ARI: {critical_stats['finetuned_ari']:.4f}")
    print(f"  Mean Pure %: {critical_stats['finetuned_pure_percentage']:.2f}%")
    print(f"  Mean classes: {critical_stats['num_training_classes']:.0f}")
    print(f"  Mean samples: {critical_stats['train_samples']:.0f}")

    print("\n--- RICH SCENARIOS (10, 12 images) ---")
    rich_stats = rich.agg({
        'finetuned_recall@1': 'mean',
        'finetuned_ari': 'mean',
        'finetuned_pure_percentage': 'mean',
        'num_training_classes': 'mean',
        'train_samples': 'mean',
        'duration_minutes': 'mean',
    })
    print(f"  Mean Recall@1: {rich_stats['finetuned_recall@1']:.2f}%")
    print(f"  Mean ARI: {rich_stats['finetuned_ari']:.4f}")
    print(f"  Mean Pure %: {rich_stats['finetuned_pure_percentage']:.2f}%")
    print(f"  Mean classes: {rich_stats['num_training_classes']:.0f}")
    print(f"  Mean samples: {rich_stats['train_samples']:.0f}")

    print("\n--- DIFFERENCES (Rich - Critical) ---")
    recall_gain = rich_stats['finetuned_recall@1'] - critical_stats['finetuned_recall@1']
    ari_gain = rich_stats['finetuned_ari'] - critical_stats['finetuned_ari']
    pure_gain = rich_stats['finetuned_pure_percentage'] - critical_stats['finetuned_pure_percentage']

    print(f"  Recall@1: +{recall_gain:.2f}%")
    print(f"  ARI: +{ari_gain:.4f}")
    print(f"  Pure %: +{pure_gain:.2f}%")


def analyze_data_efficiency(df: pd.DataFrame):
    """Analyze efficiency as a function of available data."""
    print("\n" + "=" * 80)
    print("DATA EFFICIENCY ANALYSIS")
    print("=" * 80)

    print("\nRecall@1 per training sample:")
    df_sorted = df.sort_values('min_images')
    for _, row in df_sorted.iterrows():
        train_samples = row['train_samples']
        recall = row['finetuned_recall@1']
        ari = row['finetuned_ari']

        if train_samples and train_samples > 0:
            recall_per_1k = (recall / train_samples) * 1000
            ari_per_1k = (ari / train_samples) * 1000

            print(f"  {int(row['min_images'])} images: {recall:.2f}% with {train_samples:.0f} samples -> {recall_per_1k:.4f} recall/1k samples")

    print("\nRuntime efficiency (ARI / hour):")
    for _, row in df_sorted.iterrows():
        if row['duration_minutes'] and row['duration_minutes'] > 0:
            efficiency = row['finetuned_ari'] / (row['duration_minutes'] / 60)
            print(f"  {int(row['min_images'])} images: {efficiency:.4f} ARI/hour ({row['duration_minutes']:.1f} min)")


def analyze_clustering_degradation(df: pd.DataFrame):
    """Analyze clustering degradation when data decreases."""
    print("\n" + "=" * 80)
    print("CLUSTERING DEGRADATION ANALYSIS")
    print("=" * 80)

    df_sorted = df.sort_values('min_images', ascending=False)

    print("\nDegradation from the richest scenario (12 images):")

    if 12 in df['min_images'].values:
        best_scenario = df[df['min_images'] == 12].iloc[0]
        best_ari = best_scenario['finetuned_ari']
        best_pure = best_scenario['finetuned_pure_percentage']
        best_recall = best_scenario['finetuned_recall@1']

        for _, row in df_sorted.iterrows():
            if row['min_images'] == 12:
                continue

            ari_loss = best_ari - row['finetuned_ari']
            pure_loss = best_pure - row['finetuned_pure_percentage']
            recall_loss = best_recall - row['finetuned_recall@1']

            ari_loss_pct = (ari_loss / best_ari * 100) if best_ari else 0
            recall_loss_pct = (recall_loss / best_recall * 100) if best_recall else 0

            print(f"\n  {int(row['min_images'])} images:")
            print(f"    Recall@1: -{recall_loss:.2f}% ({recall_loss_pct:.1f}% loss)")
            print(f"    ARI: -{ari_loss:.4f} ({ari_loss_pct:.1f}% loss)")
            print(f"    Pure %: -{pure_loss:.2f}%")
    else:
        print("\n[WARNING] No data for min_images=12 (best scenario)")


def analyze_training_stability(df: pd.DataFrame):
    """Analyze training stability by scenario."""
    print("\n" + "=" * 80)
    print("TRAINING STABILITY ANALYSIS")
    print("=" * 80)

    print("\nTrained epochs by scenario:")
    for min_imgs in sorted(df['min_images'].unique()):
        scenario = df[df['min_images'] == min_imgs].iloc[0]
        total_epochs = scenario['total_epochs']
        phase_1_epochs = scenario.get('phase_1_epochs_trained', 0)
        phase_2_epochs = scenario.get('phase_2_epochs_trained', 0)

        print(f"  {min_imgs} images: {total_epochs} total epochs (Phase 1: {phase_1_epochs}, Phase 2: {phase_2_epochs})")

    print("\nImprovement during fine-tuning:")
    for min_imgs in sorted(df['min_images'].unique()):
        scenario = df[df['min_images'] == min_imgs].iloc[0]
        baseline_recall = scenario.get('baseline_recall@1', 0)
        finetuned_recall = scenario.get('finetuned_recall@1', 0)
        improvement = scenario.get('recall@1_improvement', finetuned_recall - baseline_recall if baseline_recall else 0)

        print(f"  {min_imgs} images: {baseline_recall:.2f}% -> {finetuned_recall:.2f}% (+{improvement:.2f}%)")


def print_summary(df: pd.DataFrame):
    """Print a compact analysis summary."""
    print(f"\n{'='*80}")
    print("CLIP MIN IMAGES STUDY SUMMARY")
    print(f"{'='*80}")

    print(f"\nTotal scenarios: {len(df)}")
    print(f"Evaluated scenarios: {sorted(df['min_images'].unique())}")

    best_idx = df['finetuned_recall@1'].idxmax()
    best = df.loc[best_idx]

    print(f"\nBEST SCENARIO:")
    print(f"  Min images: {best['min_images']}")
    print(f"  Recall@1: {best['finetuned_recall@1']:.2f}%")
    if pd.notna(best.get('finetuned_ari')):
        print(f"  ARI: {best['finetuned_ari']:.4f}")
    if pd.notna(best.get('finetuned_pure_percentage')):
        print(f"  Pure %: {best['finetuned_pure_percentage']:.2f}%")
    print(f"  Classes: {best['num_training_classes']}")
    print(f"  Samples: {best['train_samples']}")

    worst_idx = df['finetuned_recall@1'].idxmin()
    worst = df.loc[worst_idx]

    print(f"\nWORST SCENARIO:")
    print(f"  Min images: {worst['min_images']}")
    print(f"  Recall@1: {worst['finetuned_recall@1']:.2f}%")
    if pd.notna(worst.get('finetuned_ari')):
        print(f"  ARI: {worst['finetuned_ari']:.4f}")
    if pd.notna(worst.get('finetuned_pure_percentage')):
        print(f"  Pure %: {worst['finetuned_pure_percentage']:.2f}%")
    print(f"  Classes: {worst['num_training_classes']}")
    print(f"  Samples: {worst['train_samples']}")

    recall_range = df['finetuned_recall@1'].max() - df['finetuned_recall@1'].min()

    print(f"\nRANGES:")
    print(f"  Recall@1: {recall_range:.2f}% difference")
    if df['finetuned_ari'].notna().any():
        ari_range = df['finetuned_ari'].max() - df['finetuned_ari'].min()
        print(f"  ARI: {ari_range:.4f} difference")


def main():
    """Main entrypoint for analysis."""
    print("=" * 80)
    print("CLIP MIN IMAGES STUDY ANALYSIS")
    print("=" * 80)

    csv_path = REPO_ROOT / 'results/analysis/data/clip_min_images/clip_min_images_results.csv'

    try:
        df = load_data(csv_path)
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        print("\nPlease run first: python scripts/extraction/extract_clip_min_images_results.py")
        return

    print(f"\nLoaded data: {len(df)} configurations")
    print(f"Scenarios: {sorted(df['min_images'].unique())}")

    analyze_by_min_images(df)
    analyze_baseline_vs_standard(df)
    analyze_critical_vs_rich(df)
    analyze_data_efficiency(df)
    analyze_clustering_degradation(df)
    analyze_training_stability(df)
    print_summary(df)

    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETED")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
