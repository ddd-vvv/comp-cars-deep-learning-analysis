"""Detailed statistical analysis for vision models (ResNet50 and ViT-B/32).

Generates comparative analysis across models, objectives, loss functions, and views.
"""

import pandas as pd
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_data(csv_path: Path) -> pd.DataFrame:
    """Load the vision-model results CSV."""
    if not csv_path.exists():
        raise FileNotFoundError(f"File not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    return df


def analyze_by_model(df: pd.DataFrame) -> pd.DataFrame:
    """Comparative analysis between ResNet50 and ViT-B/32."""
    print("\n" + "=" * 80)
    print("ANALYSIS BY MODEL (ResNet50 vs ViT-B/32)")
    print("=" * 80)
    
    grouped = df.groupby('model_name').agg({
        'config_key': 'count',
        
        # Finetuning metrics
        'best_val_accuracy': ['mean', 'std', 'min', 'max'],
        'best_val_recall@1': ['mean', 'std', 'min', 'max'],
        'best_val_recall@5': ['mean', 'std', 'min', 'max'],
        'duration_minutes': ['mean', 'std', 'min', 'max'],
        'total_epochs': ['mean', 'std'],
        
        # Clustering metrics
        'finetuned_ari': ['mean', 'std', 'min', 'max'],
        'finetuned_nmi': ['mean', 'std', 'min', 'max'],
        'finetuned_purity': ['mean', 'std', 'min', 'max'],
        'ari_improvement': ['mean', 'std'],
        
        # Cluster quality
        'finetuned_pure_percentage': ['mean', 'std', 'min', 'max'],
        'finetuned_n_pure_clusters': ['mean', 'std'],
        'finetuned_n_mixed_clusters': ['mean', 'std'],
        'finetuned_avg_models_per_mixed': ['mean', 'std'],
    }).round(4)
    
    grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]
    grouped = grouped.rename(columns={'config_key_count': 'num_experiments'})
    
    print("\n", grouped.to_string())
    
    return grouped


def analyze_by_objective(df: pd.DataFrame) -> pd.DataFrame:
    """Comparative analysis between classification and metric_learning."""
    print("\n" + "=" * 80)
    print("ANALYSIS BY OBJECTIVE (Classification vs Metric Learning)")
    print("=" * 80)
    
    grouped = df.groupby('objective').agg({
        'config_key': 'count',
        
        # Classification uses accuracy, metric_learning uses recall
        'best_val_accuracy': ['mean', 'std', 'min', 'max'],
        'best_val_recall@1': ['mean', 'std', 'min', 'max'],
        'best_val_recall@5': ['mean', 'std', 'min', 'max'],
        
        # Clustering
        'finetuned_ari': ['mean', 'std', 'max'],
        'finetuned_pure_percentage': ['mean', 'std', 'max'],
        'ari_improvement': ['mean', 'std'],
        
        # Timing
        'duration_minutes': ['mean', 'std'],
    }).round(4)
    
    grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]
    grouped = grouped.rename(columns={'config_key_count': 'num_experiments'})
    
    print("\n", grouped.to_string())
    
    return grouped


def analyze_by_criterion(df: pd.DataFrame) -> pd.DataFrame:
    """Comparative analysis across loss functions."""
    print("\n" + "=" * 80)
    print("ANALYSIS BY LOSS FUNCTION")
    print("=" * 80)
    
    grouped = df.groupby('criterion').agg({
        'config_key': 'count',
        'model_name': lambda x: ', '.join(x.unique()),
        
        # Main metrics
        'best_val_accuracy': ['mean', 'std', 'max'],
        'best_val_recall@1': ['mean', 'std', 'max'],
        'best_val_recall@5': ['mean', 'std', 'max'],
        
        # Clustering
        'finetuned_ari': ['mean', 'std', 'max'],
        'finetuned_purity': ['mean', 'std', 'max'],
        'finetuned_pure_percentage': ['mean', 'std', 'max'],
        'ari_improvement': ['mean', 'std'],
        
        # Efficiency
        'duration_minutes': ['mean', 'std'],
    }).round(4)
    
    grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]
    grouped = grouped.rename(columns={'config_key_count': 'num_experiments'})
    
    print("\n", grouped.to_string())
    
    # Find the best loss function by metric
    print("\n" + "-" * 80)
    print("BEST LOSS FUNCTIONS BY METRIC:")
    print("-" * 80)
    
    metrics = {
        'Recall@1': 'best_val_recall@1_max',
        'Recall@5': 'best_val_recall@5_max',
    }
    
    for metric_name, col in metrics.items():
        if col in grouped.columns:
            best = grouped[col].idxmax()
            if pd.notna(best):
                value = grouped.loc[best, col]
                print(f"  {metric_name:12s}: {best:20s} ({value:.4f})")
    
    return grouped


def analyze_by_views(df: pd.DataFrame) -> pd.DataFrame:
    """Comparative analysis between front and front+rear views."""
    print("\n" + "=" * 80)
    print("ANALYSIS BY VIEWS (Front vs Front+Rear)")
    print("=" * 80)
    
    grouped = df.groupby('views').agg({
        'config_key': 'count',
        
        # Metrics
        'best_val_accuracy': ['mean', 'std'],
        'best_val_recall@1': ['mean', 'std'],
        'best_val_recall@5': ['mean', 'std'],
        'finetuned_ari': ['mean', 'std'],
        'finetuned_pure_percentage': ['mean', 'std'],
        
        # Timing
        'duration_minutes': ['mean', 'std'],
    }).round(4)
    
    grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]
    grouped = grouped.rename(columns={'config_key_count': 'num_experiments'})
    
    print("\n", grouped.to_string())
    
    # Direct comparison
    print("\n" + "-" * 80)
    print("FRONT vs FRONT+REAR COMPARISON:")
    print("-" * 80)
    
    if 'front' in grouped.index and 'front + rear' in grouped.index:
        front = grouped.loc['front']
        rear = grouped.loc['front + rear']
        
        print(f"\nMean ARI:")
        print(f"  Front:       {front['finetuned_ari_mean']:.4f}")
        print(f"  Front+Rear:  {rear['finetuned_ari_mean']:.4f}")
        print(f"  Improvement: {(rear['finetuned_ari_mean'] - front['finetuned_ari_mean']):.4f}")
        
        print(f"\nMean Pure %:")
        print(f"  Front:       {front['finetuned_pure_percentage_mean']:.4f}%")
        print(f"  Front+Rear:  {rear['finetuned_pure_percentage_mean']:.4f}%")
        print(f"  Improvement: {(rear['finetuned_pure_percentage_mean'] - front['finetuned_pure_percentage_mean']):.4f}%")
    
    return grouped


def analyze_clustering_quality(df: pd.DataFrame) -> pd.DataFrame:
    """Detailed clustering-quality analysis."""
    print("\n" + "=" * 80)
    print("CLUSTERING QUALITY ANALYSIS")
    print("=" * 80)
    
    print("\n--- General Clustering Statistics ---")
    
    # General statistics
    stats = {
        'Mean ARI': df['finetuned_ari'].mean(),
        'Max ARI': df['finetuned_ari'].max(),
        'Min ARI': df['finetuned_ari'].min(),
        'Mean Purity': df['finetuned_purity'].mean(),
        'Mean % Pure Clusters': df['finetuned_pure_percentage'].mean(),
        'Mean # Pure Clusters': df['finetuned_n_pure_clusters'].mean(),
        'Mean # Mixed Clusters': df['finetuned_n_mixed_clusters'].mean(),
        'Models per Mixed Cluster': df['finetuned_avg_models_per_mixed'].mean(),
    }
    
    for key, value in stats.items():
        if pd.notna(value):
            print(f"  {key:30s}: {value:.4f}")
    
    # Best configuration by ARI
    print("\n--- Best Configuration by ARI ---")
    best_ari_idx = df['finetuned_ari'].idxmax()
    if pd.notna(best_ari_idx):
        best = df.loc[best_ari_idx]
        print(f"  Model:           {best['model_name']}")
        print(f"  Objective:       {best['objective']}")
        print(f"  Criterion:       {best['criterion']}")
        print(f"  Views:           {best['views']}")
        print(f"  ARI:             {best['finetuned_ari']:.4f}")
        print(f"  Purity:          {best['finetuned_purity']:.4f}")
        print(f"  Pure %:          {best['finetuned_pure_percentage']:.2f}%")
        print(f"  Pure clusters:   {best['finetuned_n_pure_clusters']:.0f}")
        print(f"  Mixed clusters:  {best['finetuned_n_mixed_clusters']:.0f}")
    
    # Best configuration by % pure clusters
    print("\n--- Best Configuration by % Pure Clusters ---")
    best_pure_idx = df['finetuned_pure_percentage'].idxmax()
    if pd.notna(best_pure_idx):
        best = df.loc[best_pure_idx]
        print(f"  Model:           {best['model_name']}")
        print(f"  Objective:       {best['objective']}")
        print(f"  Criterion:       {best['criterion']}")
        print(f"  Views:           {best['views']}")
        print(f"  Pure %:          {best['finetuned_pure_percentage']:.2f}%")
        print(f"  ARI:             {best['finetuned_ari']:.4f}")
        print(f"  Pure clusters:   {best['finetuned_n_pure_clusters']:.0f}")
    
    # Clustering method analysis
    print("\n--- Best Clustering Methods ---")
    
    clusterer_stats = df.groupby('finetuned_clusterer').agg({
        'config_key': 'count',
        'finetuned_ari': ['mean', 'std', 'max'],
        'finetuned_purity': ['mean', 'max'],
        'finetuned_pure_percentage': ['mean', 'max'],
    }).round(4)
    
    clusterer_stats.columns = ['_'.join(col).strip() for col in clusterer_stats.columns.values]
    clusterer_stats = clusterer_stats.rename(columns={'config_key_count': 'count'})
    
    print("\n", clusterer_stats.to_string())
    
    return clusterer_stats


def analyze_problematic_classes(df: pd.DataFrame):
    """Analysis of problematic (overlapping) classes."""
    print("\n" + "=" * 80)
    print("PROBLEMATIC CLASS ANALYSIS (Overlapping)")
    print("=" * 80)
    
    # Overlap statistics
    print("\n--- General Statistics ---")
    
    stats = {
        '# problematic classes (mean)': df['finetuned_n_overlapping_classes'].mean(),
        '# problematic classes (max)': df['finetuned_n_overlapping_classes'].max(),
        'Clusters per problematic class (mean)': df['finetuned_avg_clusters_per_class'].mean(),
        'Max clusters per class': df['finetuned_max_clusters_per_class'].max(),
    }
    
    for key, value in stats.items():
        if pd.notna(value):
            print(f"  {key:40s}: {value:.2f}")
    
    # Configuration with fewer problematic classes
    print("\n--- Configuration with Fewer Problematic Classes ---")
    best_idx = df['finetuned_n_overlapping_classes'].idxmin()
    if pd.notna(best_idx):
        best = df.loc[best_idx]
        print(f"  Model:                     {best['model_name']}")
        print(f"  Objective:                 {best['objective']}")
        print(f"  Criterion:                 {best['criterion']}")
        print(f"  Views:                     {best['views']}")
        print(f"  Problematic classes:       {best['finetuned_n_overlapping_classes']:.0f}")
        print(f"  ARI:                       {best['finetuned_ari']:.4f}")
        print(f"  Pure %:                    {best['finetuned_pure_percentage']:.2f}%")
    
    # Top configurations with more problematic classes
    print("\n--- Top 3 Configurations with Most Problematic Classes ---")
    
    # Filter and sort
    df_sorted = df.nlargest(3, 'finetuned_n_overlapping_classes')
    
    for i, (idx, row) in enumerate(df_sorted.iterrows(), 1):
        print(f"\n{i}. {row['model_name']} - {row['criterion']} - {row['views']}")
        print(f"   Problematic classes:  {row['finetuned_n_overlapping_classes']:.0f}")
        print(f"   Clusters per class:   {row['finetuned_avg_clusters_per_class']:.2f}")
        if pd.notna(row['finetuned_top_overlapping']):
            top_classes = str(row['finetuned_top_overlapping']).split(';')[:5]
            print(f"   Top classes: {', '.join(top_classes)}")


def analyze_baseline_vs_finetuned(df: pd.DataFrame):
    """Analyse improvements between baseline and finetuned runs."""
    print("\n" + "=" * 80)
    print("BASELINE vs FINETUNED ANALYSIS")
    print("=" * 80)
    
    # ARI improvements
    print("\n--- ARI Improvements ---")
    
    df['ari_improvement_pct'] = ((df['finetuned_ari'] - df['baseline_ari']) / df['baseline_ari'] * 100)
    
    print(f"\nMean ARI improvement:       {df['ari_improvement'].mean():.4f}")
    print(f"Mean percentage improvement:{df['ari_improvement_pct'].mean():.2f}%")
    print(f"Maximum ARI improvement:    {df['ari_improvement'].max():.4f}")
    print(f"Configurations improved:    {(df['ari_improvement'] > 0).sum()}/{len(df)}")
    
    # Top improvement
    print("\n--- Configuration with Largest ARI Improvement ---")
    best_imp_idx = df['ari_improvement'].idxmax()
    if pd.notna(best_imp_idx):
        best = df.loc[best_imp_idx]
        print(f"  Model:           {best['model_name']}")
        print(f"  Objective:       {best['objective']}")
        print(f"  Criterion:       {best['criterion']}")
        print(f"  Views:           {best['views']}")
        print(f"  Baseline ARI:    {best['baseline_ari']:.4f}")
        print(f"  Finetuned ARI:   {best['finetuned_ari']:.4f}")
        print(f"  Improvement:     {best['ari_improvement']:.4f} ({best['ari_improvement_pct']:.2f}%)")
    
    # Improvements by model
    print("\n--- Improvements by Model ---")
    model_imp = df.groupby('model_name').agg({
        'ari_improvement': ['mean', 'std'],
        'ari_improvement_pct': ['mean', 'std'],
    }).round(4)
    print("\n", model_imp.to_string())
    
    # Improvements by objective
    print("\n--- Improvements by Objective ---")
    obj_imp = df.groupby('objective').agg({
        'ari_improvement': ['mean', 'std'],
        'ari_improvement_pct': ['mean', 'std'],
    }).round(4)
    print("\n", obj_imp.to_string())


def analyze_efficiency(df: pd.DataFrame):
    """Analyze runtime efficiency."""
    print("\n" + "=" * 80)
    print("RUNTIME EFFICIENCY ANALYSIS")
    print("=" * 80)
    
    # Training times
    print("\n--- Finetuning Time ---")
    print(f"Mean duration:       {df['duration_minutes'].mean():.2f} minutes")
    print(f"Min duration:        {df['duration_minutes'].min():.2f} minutes")
    print(f"Max duration:        {df['duration_minutes'].max():.2f} minutes")
    print(f"Standard deviation:  {df['duration_minutes'].std():.2f} minutes")
    
    # Times by model
    print("\n--- Time by Model ---")
    model_time = df.groupby('model_name').agg({
        'duration_minutes': ['mean', 'std', 'min', 'max'],
    }).round(2)
    print("\n", model_time.to_string())
    
    # Times by objective
    print("\n--- Time by Objective ---")
    obj_time = df.groupby('objective').agg({
        'duration_minutes': ['mean', 'std'],
    }).round(2)
    print("\n", obj_time.to_string())
    
    # Efficiency: ARI / time
    print("\n--- Efficiency (ARI / Training Time) ---")
    df['efficiency'] = df['finetuned_ari'] / (df['duration_minutes'] / 60)  # ARI per hour
    
    print(f"\nMean efficiency: {df['efficiency'].mean():.4f} ARI/hour")
    
    best_eff_idx = df['efficiency'].idxmax()
    if pd.notna(best_eff_idx):
        best = df.loc[best_eff_idx]
        print(f"\nMost efficient configuration:")
        print(f"  Model:      {best['model_name']}")
        print(f"  Objective:  {best['objective']}")
        print(f"  Criterion:  {best['criterion']}")
        print(f"  Views:      {best['views']}")
        print(f"  ARI:        {best['finetuned_ari']:.4f}")
        print(f"  Time:       {best['duration_minutes']:.2f} min")
        print(f"  Efficiency: {best['efficiency']:.4f} ARI/hour")


def generate_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Generate a summary table with key metrics per configuration."""
    print("\n" + "=" * 80)
    print("CONFIGURATION SUMMARY TABLE")
    print("=" * 80)
    
    summary_cols = [
        'model_name',
        'objective',
        'criterion',
        'views',
        'best_val_accuracy',
        'best_val_recall@1',
        'finetuned_ari',
        'finetuned_pure_percentage',
        'finetuned_n_pure_clusters',
        'duration_minutes',
    ]
    
    available_cols = [col for col in summary_cols if col in df.columns]
    summary = df[available_cols].round(4)
    
    # Sort by ARI descending
    summary = summary.sort_values('finetuned_ari', ascending=False)
    
    print("\n", summary.to_string(index=False))
    
    return summary


def save_analysis_results(output_dir: Path, **dataframes):
    """Save analysis results to CSV files."""
    output_dir.mkdir(exist_ok=True, parents=True)
    
    for name, df in dataframes.items():
        if df is not None and not df.empty:
            output_file = output_dir / f'{name}.csv'
            df.to_csv(output_file)
            print(f"[OK] {output_file}")


def main():
    """Main entrypoint for vision-model analysis."""
    csv_path = REPO_ROOT / 'results/analysis/data/vision_models/vision_models_results.csv'
    
    if not csv_path.exists():
        print(f"[ERROR] File not found: {csv_path}")
        print("Run first: python scripts/extraction/extract_vision_results.py")
        return
    
    print("=" * 80)
    print("STATISTICAL ANALYSIS - VISION MODELS")
    print("=" * 80)
    
    # Load data
    df = load_data(csv_path)
    print(f"\nLoaded data: {len(df)} configurations")
    
    # Run analysis
    model_analysis = analyze_by_model(df)
    objective_analysis = analyze_by_objective(df)
    criterion_analysis = analyze_by_criterion(df)
    views_analysis = analyze_by_views(df)
    clustering_analysis = analyze_clustering_quality(df)
    analyze_problematic_classes(df)
    analyze_baseline_vs_finetuned(df)
    analyze_efficiency(df)
    summary = generate_summary_table(df)
    
    # Save outputs
    output_dir = REPO_ROOT / 'results/analysis/statistics/vision_models'
    print(f"\n{'='*80}")
    print("SAVING RESULTS")
    print(f"{'='*80}")
    
    save_analysis_results(
        output_dir,
        vision_model_analysis=model_analysis,
        vision_objective_analysis=objective_analysis,
        vision_criterion_analysis=criterion_analysis,
        vision_views_analysis=views_analysis,
        vision_clustering_analysis=clustering_analysis,
        vision_summary=summary,
    )
    
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETED")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
