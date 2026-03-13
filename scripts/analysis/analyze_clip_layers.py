"""Statistical analysis for the CLIP layers study.

Analyzes the impact of number of layers (1-12) and component type
(vision vs text).
"""

import pandas as pd
import numpy as np
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_data(csv_path: Path) -> pd.DataFrame:
    """Load the CLIP layers results CSV."""
    if not csv_path.exists():
        raise FileNotFoundError(f"File not found: {csv_path}")

    df = pd.read_csv(csv_path)
    return df


def analyze_by_component(df: pd.DataFrame) -> pd.DataFrame:
    """Comparative analysis between vision and text components."""
    print("\n" + "=" * 80)
    print("ANALYSIS BY COMPONENT (Vision vs Text)")
    print("=" * 80)

    grouped = df.groupby('component_type').agg({
        'config_key': 'count',

        # Recall metrics
        'best_val_recall@1': ['mean', 'std', 'min', 'max'],
        'best_val_recall@3': ['mean', 'std', 'min', 'max'],
        'best_val_recall@5': ['mean', 'std', 'min', 'max'],

        # Clustering
        'finetuned_ari': ['mean', 'std', 'min', 'max'],
        'finetuned_nmi': ['mean', 'std', 'min', 'max'],
        'finetuned_purity': ['mean', 'std', 'min', 'max'],

        # Cluster quality
        'finetuned_pure_percentage': ['mean', 'std', 'min', 'max'],
        'finetuned_n_pure_clusters': ['mean', 'std'],
        'finetuned_n_mixed_clusters': ['mean', 'std'],

        # Timing
        'duration_minutes': ['mean', 'std'],
    }).round(4)

    grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]
    grouped = grouped.rename(columns={'config_key_count': 'num_experiments'})

    print("\n", grouped.to_string())

    # Direct comparison
    print("\n" + "-" * 80)
    print("VISION vs TEXT COMPARISON:")
    print("-" * 80)

    if 'vision' in grouped.index and 'text' in grouped.index:
        vision = grouped.loc['vision']
        text = grouped.loc['text']

        print(f"\nMean Recall@1:")
        print(f"  Vision: {vision['best_val_recall@1_mean']:.4f}")
        print(f"  Text:   {text['best_val_recall@1_mean']:.4f}")
        print(f"  Difference: {(vision['best_val_recall@1_mean'] - text['best_val_recall@1_mean']):.4f}")

        print(f"\nMean ARI:")
        print(f"  Vision: {vision['finetuned_ari_mean']:.4f}")
        print(f"  Text:   {text['finetuned_ari_mean']:.4f}")
        print(f"  Difference: {(vision['finetuned_ari_mean'] - text['finetuned_ari_mean']):.4f}")

        print(f"\nMean Pure %:")
        print(f"  Vision: {vision['finetuned_pure_percentage_mean']:.2f}%")
        print(f"  Text:   {text['finetuned_pure_percentage_mean']:.2f}%")
        print(f"  Difference: {(vision['finetuned_pure_percentage_mean'] - text['finetuned_pure_percentage_mean']):.2f}%")

    return grouped


def analyze_by_num_layers(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze the impact of number of layers (1-12)."""
    print("\n" + "=" * 80)
    print("ANALYSIS BY NUMBER OF LAYERS")
    print("=" * 80)

    grouped = df.groupby('num_layers').agg({
        'config_key': 'count',
        'component_type': lambda x: ', '.join(x.unique()),

        # Metrics
        'best_val_recall@1': ['mean', 'std'],
        'best_val_recall@3': ['mean', 'std'],
        'best_val_recall@5': ['mean', 'std'],
        'finetuned_ari': ['mean', 'std'],
        'finetuned_pure_percentage': ['mean', 'std'],

        # Timing
        'duration_minutes': ['mean', 'std'],
    }).round(4)

    grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]
    grouped = grouped.rename(columns={'config_key_count': 'num_experiments'})

    print("\n", grouped.to_string())

    print("\n" + "-" * 80)
    print("BEST NUMBER OF LAYERS BY METRIC:")
    print("-" * 80)

    metrics = {
        'Recall@1': 'best_val_recall@1_mean',
        'Recall@5': 'best_val_recall@5_mean',
    }

    for metric_name, col in metrics.items():
        if col in grouped.columns:
            best_layers = grouped[col].idxmax()
            if pd.notna(best_layers):
                value = grouped.loc[best_layers, col]
                print(f"  {metric_name:12s}: {best_layers} layers ({value:.4f})")

    return grouped


def analyze_component_x_layers(df: pd.DataFrame) -> pd.DataFrame:
    """Cross analysis: component x number of layers."""
    print("\n" + "=" * 80)
    print("COMPONENT x LAYERS ANALYSIS")
    print("=" * 80)

    print("\n--- VISION COMPONENT ---")
    df_vision = df[df['component_type'] == 'vision']

    if not df_vision.empty:
        vision_grouped = df_vision.groupby('num_layers').agg({
            'best_val_recall@1': ['mean', 'max'],
            'finetuned_ari': ['mean', 'max'],
            'finetuned_pure_percentage': ['mean', 'max'],
            'duration_minutes': 'mean',
        }).round(4)

        vision_grouped.columns = ['_'.join(col).strip() for col in vision_grouped.columns.values]
        print("\n", vision_grouped.to_string())

        best_vision_idx = df_vision['finetuned_ari'].idxmax()
        if pd.notna(best_vision_idx):
            best = df_vision.loc[best_vision_idx]
            print(f"\nBest configuration (ARI):")
            print(f"  Layers:   {best['num_layers']}")
            print(f"  Recall@1: {best['best_val_recall@1']:.4f}")
            print(f"  ARI:      {best['finetuned_ari']:.4f}")
            print(f"  Pure %:   {best['finetuned_pure_percentage']:.2f}%")

    print("\n--- TEXT COMPONENT ---")
    df_text = df[df['component_type'] == 'text']

    if not df_text.empty:
        text_grouped = df_text.groupby('num_layers').agg({
            'best_val_recall@1': ['mean', 'max'],
            'finetuned_ari': ['mean', 'max'],
            'finetuned_pure_percentage': ['mean', 'max'],
            'duration_minutes': 'mean',
        }).round(4)

        text_grouped.columns = ['_'.join(col).strip() for col in text_grouped.columns.values]
        print("\n", text_grouped.to_string())

        best_text_idx = df_text['finetuned_ari'].idxmax()
        if pd.notna(best_text_idx):
            best = df_text.loc[best_text_idx]
            print(f"\nBest configuration (ARI):")
            print(f"  Layers:   {best['num_layers']}")
            print(f"  Recall@1: {best['best_val_recall@1']:.4f}")
            print(f"  ARI:      {best['finetuned_ari']:.4f}")
            print(f"  Pure %:   {best['finetuned_pure_percentage']:.2f}%")

    return df_vision, df_text


def analyze_progression_by_layers(df: pd.DataFrame):
    """Analyze metric progression as the number of layers increases."""
    print("\n" + "=" * 80)
    print("LAYER PROGRESSION ANALYSIS")
    print("=" * 80)

    for component in ['vision', 'text']:
        df_comp = df[df['component_type'] == component]

        if df_comp.empty:
            continue

        print(f"\n--- {component.upper()} PROGRESSION ---")

        progression = df_comp.sort_values('num_layers')[
            ['num_layers', 'best_val_recall@1', 'finetuned_ari', 'finetuned_pure_percentage']
        ]

        print("\n", progression.to_string(index=False))

        corr_recall = df_comp[['num_layers', 'best_val_recall@1']].corr().iloc[0, 1]
        corr_ari = df_comp[['num_layers', 'finetuned_ari']].corr().iloc[0, 1]

        print(f"\nLayer-Recall@1 correlation: {corr_recall:.4f}")
        print(f"Layer-ARI correlation:      {corr_ari:.4f}")

        recall_values = df_comp.sort_values('num_layers')['best_val_recall@1'].values
        if len(recall_values) > 1:
            diffs = np.diff(recall_values)
            if len(diffs) > 0:
                avg_diff = np.mean(diffs[diffs > 0]) if len(diffs[diffs > 0]) > 0 else 0
                print(f"Average improvement per layer: {avg_diff:.4f}")


def analyze_clustering_quality(df: pd.DataFrame):
    """Clustering quality analysis for CLIP."""
    print("\n" + "=" * 80)
    print("CLUSTERING QUALITY ANALYSIS - CLIP")
    print("=" * 80)

    print("\n--- General Statistics ---")

    stats = {
        'Mean ARI': df['finetuned_ari'].mean(),
        'Max ARI': df['finetuned_ari'].max(),
        'Mean Purity': df['finetuned_purity'].mean(),
        'Mean % Pure Clusters': df['finetuned_pure_percentage'].mean(),
        'Mean # Pure Clusters': df['finetuned_n_pure_clusters'].mean(),
        'Mean # Mixed Clusters': df['finetuned_n_mixed_clusters'].mean(),
        'Models per Mixed Cluster': df['finetuned_avg_models_per_mixed'].mean(),
    }

    for key, value in stats.items():
        if pd.notna(value):
            print(f"  {key:30s}: {value:.4f}")

    print("\n--- Best Global Configuration (ARI) ---")
    best_idx = df['finetuned_ari'].idxmax()
    if pd.notna(best_idx):
        best = df.loc[best_idx]
        print(f"  Component:       {best['component_type']}")
        print(f"  Layers:          {best['num_layers']}")
        print(f"  Recall@1:        {best['best_val_recall@1']:.4f}")
        print(f"  ARI:             {best['finetuned_ari']:.4f}")
        print(f"  Purity:          {best['finetuned_purity']:.4f}")
        print(f"  Pure %:          {best['finetuned_pure_percentage']:.2f}%")
        print(f"  Pure clusters:   {best['finetuned_n_pure_clusters']:.0f}")
        print(f"  Mixed clusters:  {best['finetuned_n_mixed_clusters']:.0f}")

    print("\n--- Best Configuration by % Pure Clusters ---")
    best_pure_idx = df['finetuned_pure_percentage'].idxmax()
    if pd.notna(best_pure_idx):
        best = df.loc[best_pure_idx]
        print(f"  Component:       {best['component_type']}")
        print(f"  Layers:          {best['num_layers']}")
        print(f"  Pure %:          {best['finetuned_pure_percentage']:.2f}%")
        print(f"  ARI:             {best['finetuned_ari']:.4f}")
        print(f"  Pure clusters:   {best['finetuned_n_pure_clusters']:.0f}")

    print("\n--- Clustering Methods ---")

    clusterer_stats = df.groupby('finetuned_clusterer').agg({
        'config_key': 'count',
        'finetuned_ari': ['mean', 'std', 'max'],
        'finetuned_purity': ['mean', 'max'],
        'finetuned_pure_percentage': ['mean', 'max'],
    }).round(4)

    clusterer_stats.columns = ['_'.join(col).strip() for col in clusterer_stats.columns.values]
    clusterer_stats = clusterer_stats.rename(columns={'config_key_count': 'count'})

    print("\n", clusterer_stats.to_string())


def analyze_problematic_classes(df: pd.DataFrame):
    """Analysis of problematic classes in CLIP."""
    print("\n" + "=" * 80)
    print("PROBLEMATIC CLASS ANALYSIS - CLIP")
    print("=" * 80)

    print("\n--- General Statistics ---")

    stats = {
        '# problematic classes (mean)': df['finetuned_n_overlapping_classes'].mean(),
        '# problematic classes (max)': df['finetuned_n_overlapping_classes'].max(),
        'Clusters per problematic class': df['finetuned_avg_clusters_per_class'].mean(),
        'Max clusters per class': df['finetuned_max_clusters_per_class'].max(),
    }

    for key, value in stats.items():
        if pd.notna(value):
            print(f"  {key:40s}: {value:.2f}")

    print("\n--- Configuration with Fewer Problematic Classes ---")
    best_idx = df['finetuned_n_overlapping_classes'].idxmin()
    if pd.notna(best_idx):
        best = df.loc[best_idx]
        print(f"  Component:                 {best['component_type']}")
        print(f"  Layers:                    {best['num_layers']}")
        print(f"  Problematic classes:       {best['finetuned_n_overlapping_classes']:.0f}")
        print(f"  ARI:                       {best['finetuned_ari']:.4f}")
        print(f"  Pure %:                    {best['finetuned_pure_percentage']:.2f}%")

    print("\n--- Problematic Classes by Component ---")

    comp_stats = df.groupby('component_type').agg({
        'finetuned_n_overlapping_classes': ['mean', 'min', 'max'],
        'finetuned_avg_clusters_per_class': ['mean', 'max'],
    }).round(2)

    print("\n", comp_stats.to_string())


def analyze_efficiency(df: pd.DataFrame):
    """Runtime efficiency analysis."""
    print("\n" + "=" * 80)
    print("RUNTIME EFFICIENCY ANALYSIS - CLIP")
    print("=" * 80)

    print("\n--- Finetuning Time ---")
    print(f"Mean duration:       {df['duration_minutes'].mean():.2f} minutes")
    print(f"Min duration:        {df['duration_minutes'].min():.2f} minutes")
    print(f"Max duration:        {df['duration_minutes'].max():.2f} minutes")

    print("\n--- Time by Component ---")
    comp_time = df.groupby('component_type').agg({
        'duration_minutes': ['mean', 'std', 'min', 'max'],
    }).round(2)
    print("\n", comp_time.to_string())

    print("\n--- Time by Number of Layers ---")
    layers_time = df.groupby('num_layers').agg({
        'duration_minutes': ['mean', 'std'],
    }).round(2)
    print("\n", layers_time.to_string())

    print("\n--- Efficiency (ARI / Time) ---")
    df['efficiency'] = df['finetuned_ari'] / (df['duration_minutes'] / 60)

    print(f"\nMean efficiency: {df['efficiency'].mean():.4f} ARI/hour")

    best_eff_idx = df['efficiency'].idxmax()
    if pd.notna(best_eff_idx):
        best = df.loc[best_eff_idx]
        print(f"\nMost efficient configuration:")
        print(f"  Component:  {best['component_type']}")
        print(f"  Layers:     {best['num_layers']}")
        print(f"  ARI:        {best['finetuned_ari']:.4f}")
        print(f"  Time:       {best['duration_minutes']:.2f} min")
        print(f"  Efficiency: {best['efficiency']:.4f} ARI/hour")


def generate_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Generate summary table for CLIP configurations."""
    print("\n" + "=" * 80)
    print("SUMMARY TABLE - CLIP LAYERS")
    print("=" * 80)

    summary_cols = [
        'component_type',
        'num_layers',
        'best_val_recall@1',
        'best_val_recall@5',
        'finetuned_ari',
        'finetuned_pure_percentage',
        'finetuned_n_pure_clusters',
        'duration_minutes',
    ]

    available_cols = [col for col in summary_cols if col in df.columns]
    summary = df[available_cols].round(4)

    summary = summary.sort_values('finetuned_ari', ascending=False)

    print("\n", summary.to_string(index=False))

    return summary


def save_analysis_results(output_dir: Path, **dataframes):
    """Save analysis results to CSV files."""
    output_dir.mkdir(exist_ok=True, parents=True)

    for name, df in dataframes.items():
        if df is not None and isinstance(df, pd.DataFrame) and not df.empty:
            output_file = output_dir / f'{name}.csv'
            df.to_csv(output_file)
            print(f"[OK] {output_file}")


def main():
    """Main entrypoint for CLIP layers analysis."""
    csv_path = REPO_ROOT / 'results/analysis/data/clip_layers/clip_layers_results.csv'

    if not csv_path.exists():
        print(f"[ERROR] File not found: {csv_path}")
        print("Run first: python scripts/extraction/extract_clip_layers_results.py")
        return

    print("=" * 80)
    print("STATISTICAL ANALYSIS - CLIP LAYERS STUDY")
    print("=" * 80)

    df = load_data(csv_path)
    print(f"\nLoaded data: {len(df)} configurations")

    component_analysis = analyze_by_component(df)
    layers_analysis = analyze_by_num_layers(df)
    df_vision, df_text = analyze_component_x_layers(df)
    analyze_progression_by_layers(df)
    analyze_clustering_quality(df)
    analyze_problematic_classes(df)
    analyze_efficiency(df)
    summary = generate_summary_table(df)

    output_dir = REPO_ROOT / 'results/analysis/statistics'
    print(f"\n{'='*80}")
    print("SAVING RESULTS")
    print(f"{'='*80}")

    save_analysis_results(
        output_dir,
        clip_component_analysis=component_analysis,
        clip_layers_analysis=layers_analysis,
        clip_summary=summary,
    )

    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETED")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
