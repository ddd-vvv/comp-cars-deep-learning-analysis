"""
Embeddings analysis pipeline.

This module provides a pipeline focused on the analysis and visualization
of embeddings, including dimensionality reduction and clustering.
"""

from __future__ import annotations

import json
import logging
import tempfile
import shutil
import warnings
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
import torch

from src.data.MyDataset import create_car_dataset
from src.utils.ClusteringAnalyzer import ClusteringAnalyzer
from src.utils.ClusterVisualizer import ClusterVisualizer
from src.utils.DimensionalityReducer import DimensionalityReducer
from src.utils.JsonUtils import safe_json_dump


warnings.filterwarnings('ignore')
logging.getLogger(__name__).addHandler(logging.NullHandler())

PROJECT_ROOT = Path(__file__).resolve().parents[2]

__all__ = [
    'EmbeddingsPipeline',
    'EmbeddingsPipelineError',
    'create_embeddings_pipeline',
]


class EmbeddingsPipelineError(Exception):
    """Custom exception for embeddings pipeline errors."""
    pass


class EmbeddingsPipeline:
    """Pipeline for analysing embeddings from vision models.

    Handles the complete embeddings analysis workflow: loading,
    dimensionality reduction, clustering, and visualisation.

    Note:
        - Uses percentage-based splits (train_ratio, val_ratio, test_ratio).
        - Uses ``'standard'`` sampling strategy for embeddings analysis.
        - Compatible with embeddings from models fine-tuned with pytorch-metric-learning.
    
    Example:
        >>> config = {
        ...     'min_images': 5,
        ...     'train_ratio': 0.7,
        ...     'val_ratio': 0.2,
        ...     'test_ratio': 0.1,
        ...     'views': ['front'],
        ...     'seed': 3
        ... }
        >>> pipeline = EmbeddingsPipeline(config, dataframe)
        >>> pipeline.create_dataset_for_labels()
        >>> pipeline.load_embeddings_from_files('baseline.pt')
        >>> results = pipeline.run_full_pipeline()
    """
    
    def __init__(
        self, 
        config: Dict[str, Any], 
        df: pd.DataFrame,
        experiment_name: Optional[str] = None
    ) -> None:
        """Initialise the embeddings analysis pipeline.

        Args:
            config: Pipeline configuration dictionary.
            df: DataFrame with dataset records.
            experiment_name: Experiment name; auto-generated when ``None``.
        """
        self.config = config
        self.df = df
        self.experiment_name = experiment_name or f"embeddings_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Pipeline components
        self.dataset_dict = None
        self.baseline_embeddings = None
        self.baseline_labels = None
        self.finetuned_embeddings = None

        # Loaded embeddings metadata
        self.baseline_info = None
        self.finetuned_info = None
        
        self.results = {
            'config': config.copy(),
            'experiment_name': self.experiment_name,
            'start_time': datetime.now().isoformat()
        }
        
        logging.info(f"Initialised EmbeddingsPipeline: {self.experiment_name}")
    
    def create_dataset_for_labels(self) -> None:
        """Create the dataset to obtain the correct labels for embedding analysis.

        Reads split ratios and view configuration from ``self.config``.
        After this call, ``self.baseline_labels`` is populated.
        """
        logging.info("Creating dataset to obtain labels...")
        
        self.dataset_dict = create_car_dataset(
            df=self.df,
            views=self.config.get('views', ['front']),
            min_images=self.config.get('min_images', 5),
            train_ratio=self.config.get('train_ratio', 0.7),
            val_ratio=self.config.get('val_ratio', 0.2),
            test_ratio=self.config.get('test_ratio', 0.1),
            batch_size=self.config.get('batch_size', 32),
            num_workers=self.config.get('num_workers', 4),
            seed=self.config.get('seed', 3),
            sampling_strategy='standard',
            model_type='vision',
            description_include=''
        )

        self.dataset_dict['dataset'].set_split('val')

        labels = []
        for sample_tuple in self.dataset_dict['dataset'].val_samples:
            model_year_tuple = sample_tuple[0]
            label_str = f"{model_year_tuple[0]}_{model_year_tuple[1]}"
            label = self.dataset_dict['dataset'].label_encoder.transform([label_str])[0]
            labels.append(label)
        
        self.baseline_labels = torch.tensor(labels)
        
        self.results['dataset_info'] = {
            'num_training_classes': self.dataset_dict['dataset'].get_num_classes_for_training(),
            'num_total_classes': self.dataset_dict['dataset'].get_total_num_classes(),
            'val_samples': len(self.dataset_dict['dataset'].val_samples),
            'views': self.dataset_dict['dataset'].views,
            'num_regular_classes': len(self.dataset_dict['dataset'].regular_classes),
            'num_oneshot_classes': len(self.dataset_dict['dataset'].oneshot_classes)
        }
        
        logging.info(f"Dataset created - val samples: {len(self.dataset_dict['dataset'].val_samples)}")
        logging.info(f"Labels extracted: {len(self.baseline_labels)} samples")
    
    def load_embeddings_from_files(
        self, 
        baseline_embeddings_path: Union[str, Path],
        finetuned_embeddings_path: Optional[Union[str, Path]] = None
    ) -> None:
        """Load embeddings from the specified ``.pt`` files.

        Args:
            baseline_embeddings_path: Path to the baseline embeddings file.
            finetuned_embeddings_path: Path to the fine-tuned embeddings file (optional).
        """
        logging.info("Loading embeddings from files...")        
        # Load baseline embeddings
        baseline_path = Path(baseline_embeddings_path)
        if not baseline_path.exists():
            raise EmbeddingsPipelineError(f"Baseline embeddings file not found: {baseline_path}")
        
        self.baseline_embeddings = torch.load(baseline_path, map_location='cpu')
        self.baseline_info = {
            'path': str(baseline_path),
            'shape': list(self.baseline_embeddings.shape),
            'device': str(self.baseline_embeddings.device)
        }
        
        logging.info(f"Baseline embeddings loaded: {self.baseline_embeddings.shape}")
        # Load fine-tuned embeddings if provided
        if finetuned_embeddings_path:
            finetuned_path = Path(finetuned_embeddings_path)
            if not finetuned_path.exists():
                raise EmbeddingsPipelineError(f"Fine-tuned embeddings file not found: {finetuned_path}")
            
            self.finetuned_embeddings = torch.load(finetuned_path, map_location='cpu')
            self.finetuned_info = {
                'path': str(finetuned_path),
                'shape': list(self.finetuned_embeddings.shape),
                'device': str(self.finetuned_embeddings.device)
            }
            
            logging.info(f"Fine-tuned embeddings loaded: {self.finetuned_embeddings.shape}")

            if self.baseline_embeddings.shape != self.finetuned_embeddings.shape:
                raise EmbeddingsPipelineError(
                    f"Embedding shapes do not match: "
                    f"baseline {self.baseline_embeddings.shape} vs "
                    f"finetuned {self.finetuned_embeddings.shape}"
                )
        
        self.results['embeddings_info'] = {
            'baseline': self.baseline_info,
            'finetuned': self.finetuned_info if finetuned_embeddings_path else None
        }
    
    def load_embeddings_from_zip(
        self,
        zip_path: Union[str, Path]
    ) -> None:
        """Load embeddings from a ZIP file.

        Args:
            zip_path: Path to the ZIP file containing the embeddings and configuration.
        """
        logging.info(f"Loading embeddings from ZIP: {zip_path}")

        # Extract and load files from the ZIP
        zip_path = Path(zip_path)
        if not zip_path.exists():
            raise EmbeddingsPipelineError(f"ZIP file not found: {zip_path}")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_path)
            
            baseline_files = list(temp_path.rglob("baseline_embeddings.pt"))
            finetuned_files = list(temp_path.rglob("finetuned_embeddings.pt"))
            
            if not baseline_files:
                raise EmbeddingsPipelineError("baseline_embeddings.pt not found in the ZIP")
            
            baseline_path = baseline_files[0]
            finetuned_path = finetuned_files[0] if finetuned_files else None
            
            self.load_embeddings_from_files(baseline_path, finetuned_path)
            
            config_files = list(temp_path.rglob("config.json"))
            if config_files:
                with open(config_files[0], 'r') as f:
                    loaded_config = json.load(f)
                    self.results['original_config'] = loaded_config
                    logging.info("Original configuration loaded from ZIP")
    
    def analyze_baseline_embeddings(self) -> Dict[str, Any]:
        """Analyse the loaded baseline embeddings.

        Returns:
            Dictionary with analysis results.
        """
        if self.baseline_embeddings is None:
            raise EmbeddingsPipelineError("Baseline embeddings not loaded. Run load_embeddings_from_files() first.")

        if self.baseline_labels is None:
            raise EmbeddingsPipelineError("Labels not available. Run create_dataset_for_labels() first.")

        logging.info("Analysing loaded baseline embeddings...")

        # Run analysis
        baseline_results = self._analyze_embeddings(
            embeddings=self.baseline_embeddings,
            labels=self.baseline_labels,
            phase='baseline'
        )
        
        # Save results
        self.results['baseline_analysis'] = baseline_results
        
        logging.info("Baseline analysis complete!")
        return baseline_results
    
    def analyze_finetuned_embeddings(self) -> Dict[str, Any]:
        """Analyse the loaded fine-tuned embeddings.

        Returns:
            Dictionary with analysis results.
        """
        if self.finetuned_embeddings is None:
            raise EmbeddingsPipelineError("Fine-tuned embeddings not loaded. Load them with load_embeddings_from_files().")

        if self.baseline_labels is None:
            raise EmbeddingsPipelineError("Labels not available. Run create_dataset_for_labels() first.")

        logging.info("Analysing loaded fine-tuned embeddings...")

        # Run analysis
        finetuned_results = self._analyze_embeddings(
            embeddings=self.finetuned_embeddings,
            labels=self.baseline_labels,
            phase='finetuned'
        )

        # Save results
        self.results['finetuned_analysis'] = finetuned_results
        
        logging.info("Fine-tuned analysis complete!")
        return finetuned_results
    
    def _analyze_embeddings(
        self, 
        embeddings: torch.Tensor, 
        labels: torch.Tensor, 
        phase: str
    ) -> Dict[str, Any]:
        """Perform complete analysis of dimensionality reduction, clustering,
        and visualisation for a given set of embeddings.

        Args:
            embeddings: Tensor of embeddings to analyse.
            labels: Tensor of true labels corresponding to the embeddings.
            phase: Analysis phase (``'baseline'`` or ``'finetuned'``).

        Returns:
            Dictionary with analysis results.
        """
        results = {}
        
        logging.info(f"Running dimensionality reduction - {phase}...")
        
        # Dimensionality reduction
        reducer = DimensionalityReducer(
            embeddings=embeddings,
            labels=labels,
            seed=self.config.get('seed', 3),
            optimizer_trials=self.config.get('reducer_trials', 50),
            available_methods=self.config.get('reducer_methods', ['pca', 'umap']),
            n_jobs=self.config.get('n_jobs', -1),
            use_incremental=self.config.get('use_incremental', True)
        )
        
        # Run dimensionality reduction
        reduction_scores = reducer.reduce_all()
        best_method, best_embeddings = reducer.get_best_result()
        
        # Save reduction results
        results['reduction'] = {
            'scores': reduction_scores,
            'best_method': best_method,
            'best_embeddings_shape': list(best_embeddings.shape)
        }
        
        logging.info(f"Running clustering - {phase}...")
        
        # Clustering
        clustering = ClusteringAnalyzer(
            embeddings=best_embeddings,
            true_labels=labels,
            seed=self.config.get('seed', 3),
            optimizer_trials=self.config.get('clustering_trials', 50),
            available_methods=self.config.get('clustering_methods', ['dbscan', 'hdbscan']),
            n_jobs=self.config.get('n_jobs', -1)
        )
        
        # Run clustering
        clustering_results = clustering.cluster_all()
        logging.info(f"Clustering results keys: {list(clustering_results.keys()) if clustering_results else 'Empty results'}")
        
        if not clustering_results:
            raise ValueError(f"Could not generate clustering results for {phase}")
        
        # Get best clustering method
        comparison_df = clustering.compare_methods()
        best_clustering_method, best_cluster_labels = clustering.get_best_result()
        
        # Save clustering results
        results['clustering'] = {
            'results': clustering_results,
            'comparison_df': comparison_df.to_dict('records'),
            'best_method': best_clustering_method
        }
        
        if self.config.get('generate_visualizations', True):
            logging.info(f"Generating visualizations - {phase}...")
            
            visualizer = ClusterVisualizer(
                embeddings=best_embeddings,
                cluster_labels=best_cluster_labels,
                true_labels=labels,
                val_samples=self.dataset_dict['dataset'].val_samples,
                label_encoder=self.dataset_dict['dataset'].label_encoder,
                seed=self.config.get('seed', 3)
            )
            
            # Store visualizer for use in save_results()
            if phase == 'baseline':
                self.baseline_visualizer = visualizer
            elif phase == 'finetuned':
                self.finetuned_visualizer = visualizer
            
            # Statistics and summary
            visualizer.print_cluster_statistics()
            summary_df = clustering.get_cluster_summary(best_clustering_method)

            # Number of clusters to visualize
            n_to_vis = int(self.config.get('n_clusters_to_visualize', 3))
            max_classes_per_cluster = int(self.config.get('max_classes_per_cluster_viz', 8))

            # Use adaptive visualization strategy
            try:
                visualizer.visualize_best_available_clusters(n_to_vis, max_classes_per_cluster)
            except Exception as e:
                logging.error(f"Error in adaptive cluster visualization: {e}")
                # Fallback: try individual visualizations
                try:
                    logging.info("Attempting pure cluster visualization as fallback...")
                    visualizer.visualize_good_clusters(n_to_vis, max_classes_per_cluster)
                except Exception as e2:
                    logging.warning(f"Error visualizing pure clusters: {e2}")

                try:
                    logging.info("Attempting mixed cluster visualization as fallback...")
                    visualizer.visualize_mixed_clusters(n_to_vis, max_classes_per_cluster)
                except Exception as e3:
                    logging.warning(f"Error visualizing mixed clusters: {e3}")

            # Get class overlap between clusters
            try:
                overlap_df = visualizer.get_class_cluster_overlap()
                overlap_records = overlap_df.to_dict('records') if isinstance(overlap_df, pd.DataFrame) else []
            except Exception as e:
                logging.warning(f"Error calculating class overlap: {e}")
                overlap_records = []

            # Get class overlap
            try:
                overlap_df = visualizer.get_class_cluster_overlap()
                overlap_records = overlap_df.to_dict('records') if not overlap_df.empty else []
            except Exception as e:
                logging.warning(f"Could not generate class overlap: {e}")
                overlap_records = []
            
            results['visualization'] = {
                'cluster_summary': summary_df.to_dict('records'),
                'cluster_analysis': visualizer.cluster_analysis,
                'class_cluster_overlap': overlap_records
            }
        
        logging.info(f"Analysis {phase} - Best reduction: {best_method}, Best clustering: {best_clustering_method}")
        
        return results
    
    def compare_results(self) -> Dict[str, Any]:
        """Compare results between baseline and fine-tuned embeddings.

        Returns:
            Dictionary with comparison results.
        """
        if 'baseline_analysis' not in self.results or 'finetuned_analysis' not in self.results:
            raise EmbeddingsPipelineError("Both analyses (baseline and finetuned) must be completed first.")
        
        logging.info("Comparing baseline vs fine-tuned results...")

        # Extract relevant results
        baseline = self.results['baseline_analysis']
        finetuned = self.results['finetuned_analysis']
        
        # Compare clustering
        baseline_clustering = baseline['clustering']['comparison_df']
        finetuned_clustering = finetuned['clustering']['comparison_df']
        
        # Build comparison
        comparison = {
            'reduction_methods': {
                'baseline_best': baseline['reduction']['best_method'],
                'finetuned_best': finetuned['reduction']['best_method']
            },
            'clustering_methods': {
                'baseline_best': baseline['clustering']['best_method'],
                'finetuned_best': finetuned['clustering']['best_method']
            },
            'clustering_metrics_comparison': self._compare_clustering_metrics(
                baseline_clustering, finetuned_clustering
            )
        }
        
        # Compute performance improvement based on ARI
        if  baseline_clustering and finetuned_clustering:
            baseline_best = max(baseline_clustering, key=lambda x: x.get('adjusted_rand_score', 0))
            finetuned_best = max(finetuned_clustering, key=lambda x: x.get('adjusted_rand_score', 0))
            
            comparison['performance_improvement'] = {
                'baseline_ari': baseline_best.get('adjusted_rand_score', 0),
                'finetuned_ari': finetuned_best.get('adjusted_rand_score', 0),
                'ari_improvement': finetuned_best.get('adjusted_rand_score', 0) - baseline_best.get('adjusted_rand_score', 0)
            }
        
        # Store comparison in results
        self.results['comparison'] = comparison
        
        logging.info("Comparison complete!")
        logging.info(f"Best baseline method: {comparison['clustering_methods']['baseline_best']}")
        logging.info(f"Best fine-tuned method: {comparison['clustering_methods']['finetuned_best']}")
        
        return comparison
    
    def _compare_clustering_metrics(
        self, 
        baseline_results: list, 
        finetuned_results: list
    ) -> Dict[str, Any]:
        """Compare clustering metrics between baseline and fine-tuned results.

        Args:
            baseline_results: Baseline clustering results.
            finetuned_results: Fine-tuned clustering results.

        Returns:
            Dictionary with the metric comparison.
        """
        if not baseline_results or not finetuned_results:
            return {}
        
        # Compare key metrics
        metrics = ['adjusted_rand_score', 'silhouette_score', 'calinski_harabasz_score']
        comparison = {}
        
        # Compute statistics for each metric
        for metric in metrics:
            baseline_values = [r.get(metric, 0) for r in baseline_results]
            finetuned_values = [r.get(metric, 0) for r in finetuned_results]
            
            if baseline_values and finetuned_values:
                comparison[metric] = {
                    'baseline_max': max(baseline_values),
                    'baseline_mean': np.mean(baseline_values),
                    'finetuned_max': max(finetuned_values),
                    'finetuned_mean': np.mean(finetuned_values),
                    'improvement_max': max(finetuned_values) - max(baseline_values),
                    'improvement_mean': np.mean(finetuned_values) - np.mean(baseline_values)
                }
        
        return comparison
    
    def save_results(self, save_dir: Union[str, Path] = "results") -> Path:
        """Save all pipeline results to the specified directory.

        Args:
            save_dir: Directory where the results will be saved.

        Returns:
            Path to the ZIP file containing the saved results.
        """
        # Create results directory
        save_dir = Path(save_dir)
        experiment_dir = save_dir / self.experiment_name
        experiment_dir.mkdir(parents=True, exist_ok=True)
        
        logging.info(f"Saving results to: {experiment_dir}")

        # Save configuration
        safe_json_dump(self.config, experiment_dir / "config.json")
        
        # Save results
        self.results['end_time'] = datetime.now().isoformat()
        results_to_save = self.results.copy()
        
        safe_json_dump(results_to_save, experiment_dir / "results.json")
        
        if self.baseline_embeddings is not None:
            np.save(experiment_dir / "baseline_embeddings.npy", self.baseline_embeddings.numpy())
        
        if self.finetuned_embeddings is not None:
            np.save(experiment_dir / "finetuned_embeddings.npy", self.finetuned_embeddings.numpy())
        
        if self.baseline_labels is not None:
            np.save(experiment_dir / "labels.npy", self.baseline_labels.numpy())
        
        if 'comparison' in self.results:
            comparison_df = pd.DataFrame([self.results['comparison']['performance_improvement']])
            comparison_df.to_csv(experiment_dir / "performance_comparison.csv", index=False)
        
        # Save baseline visualizations if available
        if hasattr(self, 'baseline_visualizer') and self.baseline_visualizer is not None:
            try:
                visualizations_dir = experiment_dir / "visualizations" / "baseline"
                logging.info("Saving baseline visualizations...")
                saved_viz_files = self.baseline_visualizer.save_visualizations(
                    visualizations_dir,
                    n_pure_clusters=5,
                    n_mixed_clusters=5,
                    n_images_per_cluster=6
                )
                logging.info(f"Baseline visualizations saved: {list(saved_viz_files.keys())}")
            except Exception as e:
                logging.warning(f"Could not save baseline visualizations: {e}")
        
        # Save fine-tuned visualizations if available
        if hasattr(self, 'finetuned_visualizer') and self.finetuned_visualizer is not None:
            try:
                visualizations_dir = experiment_dir / "visualizations" / "finetuned"
                logging.info("Saving fine-tuned visualizations...")
                saved_viz_files = self.finetuned_visualizer.save_visualizations(
                    visualizations_dir,
                    n_pure_clusters=5,
                    n_mixed_clusters=5,
                    n_images_per_cluster=6
                )
                logging.info(f"Fine-tuned visualizations saved: {list(saved_viz_files.keys())}")
            except Exception as e:
                logging.warning(f"Could not save fine-tuned visualizations: {e}")
        
        # Create ZIP file with all results
        zip_path = save_dir / f"{self.experiment_name}.zip"
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in experiment_dir.rglob("*"):
                if file_path.is_file():
                    arcname = file_path.relative_to(save_dir)
                    zipf.write(file_path, arcname)
        
        if experiment_dir.exists():
            shutil.rmtree(experiment_dir)
        
        logging.info(f"Results saved to ZIP: {zip_path}")
        return zip_path
    
    def run_baseline_analysis_from_zip(self, zip_path: Union[str, Path]) -> Dict[str, Any]:
        """Run baseline analysis loading embeddings from a ZIP file.

        Args:
            zip_path: Path to the ZIP file containing the embeddings and configuration.

        Returns:
            Dictionary with baseline analysis results.
        """
        logging.info(f"=== BASELINE ANALYSIS FROM ZIP: {self.experiment_name} ===")
        
        try:
            self.create_dataset_for_labels()
            self.load_embeddings_from_zip(zip_path)
            self.analyze_baseline_embeddings()
            
            zip_result_path = self.save_results()
            self.results['saved_to'] = str(zip_result_path)
            
            logging.info("=== BASELINE ANALYSIS COMPLETE ===")
            logging.info(f"Results saved to: {zip_result_path}")
            return self.results
            
        except Exception as e:
            logging.error(f"Error in baseline analysis: {e}")
            self.results['error'] = str(e)
            raise EmbeddingsPipelineError(f"Error in baseline analysis: {e}") from e
    
    def run_finetuned_analysis_from_zip(self, zip_path: Union[str, Path]) -> Dict[str, Any]:
        """Run fine-tuned analysis loading embeddings from a ZIP file.

        Args:
            zip_path: Path to the ZIP file containing the embeddings and configuration.

        Returns:
            Dictionary with fine-tuned analysis results.
        """
        logging.info(f"=== FINE-TUNED ANALYSIS FROM ZIP: {self.experiment_name} ===")
        
        try:
            if self.baseline_labels is None:
                self.create_dataset_for_labels()
            
            self.load_embeddings_from_zip(zip_path)
            
            if self.finetuned_embeddings is None:
                raise EmbeddingsPipelineError("No fine-tuned embeddings found in the ZIP")
            
            self.analyze_finetuned_embeddings()
            
            zip_result_path = self.save_results()
            self.results['saved_to'] = str(zip_result_path)
            
            logging.info("=== FINE-TUNED ANALYSIS COMPLETE ===")
            logging.info(f"Results saved to: {zip_result_path}")
            return self.results
            
        except Exception as e:
            logging.error(f"Error in fine-tuned analysis: {e}")
            self.results['error'] = str(e)
            raise EmbeddingsPipelineError(f"Error in fine-tuned analysis: {e}") from e
    
    def run_comparative_analysis_from_zip(self, zip_path: Union[str, Path]) -> Dict[str, Any]:
        """Run comparative analysis (baseline + fine-tuned) from a ZIP file.

        Args:
            zip_path: Path to the ZIP file containing the embeddings and configuration.

        Returns:
            Dictionary with comparative analysis results.
        """
        logging.info(f"=== COMPARATIVE ANALYSIS FROM ZIP: {self.experiment_name} ===")
        
        try:
            self.create_dataset_for_labels()
            self.load_embeddings_from_zip(zip_path)
            
            if self.finetuned_embeddings is None:
                raise EmbeddingsPipelineError("Fine-tuned embeddings are required for comparative analysis")
            
            self.analyze_baseline_embeddings()
            self.analyze_finetuned_embeddings()
            self.compare_results()
            
            zip_result_path = self.save_results()
            self.results['saved_to'] = str(zip_result_path)
            
            logging.info("=== COMPARATIVE ANALYSIS COMPLETE ===")
            logging.info(f"Results saved to: {zip_result_path}")
            return self.results
            
        except Exception as e:
            logging.error(f"Error in comparative analysis: {e}")
            self.results['error'] = str(e)
            raise EmbeddingsPipelineError(f"Error in comparative analysis: {e}") from e
    
    def run_full_analysis_from_zip(self, zip_path: Union[str, Path]) -> Dict[str, Any]:
        """Run the full analysis (baseline or comparative) from a ZIP file.

        Args:
            zip_path: Path to the ZIP file containing the embeddings and configuration.

        Returns:
            Dictionary with complete analysis results.
        """
        return self.run_comparative_analysis_from_zip(zip_path)
    
    def run_baseline_analysis_from_files(self, baseline_embeddings_path: Union[str, Path]) -> Dict[str, Any]:
        """Run baseline analysis loading embeddings from a file.

        Args:
            baseline_embeddings_path: Path to the baseline embeddings file.

        Returns:
            Dictionary with baseline analysis results.
        """
        logging.info(f"=== BASELINE ANALYSIS FROM FILES: {self.experiment_name} ===")
        
        try:
            self.create_dataset_for_labels()
            self.load_embeddings_from_files(baseline_embeddings_path, None)
            self.analyze_baseline_embeddings()
            
            zip_result_path = self.save_results()
            self.results['saved_to'] = str(zip_result_path)
            
            logging.info("=== BASELINE ANALYSIS COMPLETE ===")
            logging.info(f"Results saved to: {zip_result_path}")
            return self.results
            
        except Exception as e:
            logging.error(f"Error in baseline analysis: {e}")
            self.results['error'] = str(e)
            raise EmbeddingsPipelineError(f"Error in baseline analysis: {e}") from e
    
    def run_finetuned_analysis_from_files(self, finetuned_embeddings_path: Union[str, Path]) -> Dict[str, Any]:
        """Run fine-tuned analysis loading embeddings from a file.

        Args:
            finetuned_embeddings_path: Path to the fine-tuned embeddings file.

        Returns:
            Dictionary with fine-tuned analysis results.
        """
        logging.info(f"=== FINE-TUNED ANALYSIS FROM FILES: {self.experiment_name} ===")
        
        try:
            if self.baseline_labels is None:
                self.create_dataset_for_labels()
            
            baseline_path = Path(finetuned_embeddings_path).parent / "baseline_embeddings.pt"
            if not baseline_path.exists():
                raise EmbeddingsPipelineError(f"Baseline embeddings not found at {baseline_path}")
            
            self.load_embeddings_from_files(baseline_path, finetuned_embeddings_path)
            self.analyze_finetuned_embeddings()
            
            zip_result_path = self.save_results()
            self.results['saved_to'] = str(zip_result_path)
            
            logging.info("=== FINE-TUNED ANALYSIS COMPLETE ===")
            logging.info(f"Results saved to: {zip_result_path}")
            return self.results
            
        except Exception as e:
            logging.error(f"Error in fine-tuned analysis: {e}")
            self.results['error'] = str(e)
            raise EmbeddingsPipelineError(f"Error in fine-tuned analysis: {e}") from e
    
    def run_comparative_analysis_from_files(
        self,
        baseline_embeddings_path: Union[str, Path],
        finetuned_embeddings_path: Union[str, Path]
    ) -> Dict[str, Any]:
        """Run comparative analysis loading embeddings from separate files.

        Args:
            baseline_embeddings_path: Path to the baseline embeddings file.
            finetuned_embeddings_path: Path to the fine-tuned embeddings file.

        Returns:
            Dictionary with comparative analysis results.
        """
        logging.info(f"=== COMPARATIVE ANALYSIS FROM FILES: {self.experiment_name} ===")
        
        try:
            self.create_dataset_for_labels()
            self.load_embeddings_from_files(baseline_embeddings_path, finetuned_embeddings_path)
            self.analyze_baseline_embeddings()
            self.analyze_finetuned_embeddings()
            self.compare_results()
            
            zip_result_path = self.save_results()
            self.results['saved_to'] = str(zip_result_path)
            
            logging.info("=== COMPARATIVE ANALYSIS COMPLETE ===")
            logging.info(f"Results saved to: {zip_result_path}")
            return self.results
            
        except Exception as e:
            logging.error(f"Error in comparative analysis: {e}")
            self.results['error'] = str(e)
            raise EmbeddingsPipelineError(f"Error in comparative analysis: {e}") from e
    
    def run_full_analysis_from_files(
        self,
        baseline_embeddings_path: Union[str, Path],
        finetuned_embeddings_path: Optional[Union[str, Path]] = None
    ) -> Dict[str, Any]:
        """Run the full analysis (baseline or comparative) from embeddings files.

        Args:
            baseline_embeddings_path: Path to the baseline embeddings file.
            finetuned_embeddings_path: Path to the fine-tuned embeddings file (optional).

        Returns:
            Dictionary with complete analysis results.
        """
        if finetuned_embeddings_path is None:
            return self.run_baseline_analysis_from_files(baseline_embeddings_path)
        else:
            return self.run_comparative_analysis_from_files(baseline_embeddings_path, finetuned_embeddings_path)
    
    def load_results_from_zip(self, zip_path: Union[str, Path]) -> Dict[str, Any]:
        """Load saved results from a ZIP file.

        Args:
            zip_path: Path to the ZIP file containing the saved results.

        Returns:
            Dictionary with loaded results.
        """
        logging.info(f"Loading results from ZIP: {zip_path}")
        
        zip_path = Path(zip_path)
        if not zip_path.exists():
            raise EmbeddingsPipelineError(f"ZIP file not found: {zip_path}")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_path)
            
            results_files = list(temp_path.rglob("results.json"))
            
            if not results_files:
                raise EmbeddingsPipelineError("results.json not found in the ZIP")
            
            with open(results_files[0], 'r') as f:
                loaded_results = json.load(f)
            
            logging.info(f"Results loaded successfully from: {zip_path}")
            return loaded_results
    
    def compare_saved_results(
        self,
        baseline_zip_path: Union[str, Path],
        finetuned_zip_path: Union[str, Path],
        save_comparison: bool = True
    ) -> Dict[str, Any]:
        """Compare saved results between two ZIP archives.

        Args:
            baseline_zip_path: Path to the ZIP file with baseline results.
            finetuned_zip_path: Path to the ZIP file with fine-tuned results.
            save_comparison: Whether to save the comparison to JSON and CSV files.

        Returns:
            Dictionary with comparison results.
        """
        logging.info("=== COMPARISON OF SAVED RESULTS ===")
        logging.info(f"Baseline: {baseline_zip_path}")
        logging.info(f"Finetuned: {finetuned_zip_path}")
        
        try:
            baseline_results = self.load_results_from_zip(baseline_zip_path)
            finetuned_results = self.load_results_from_zip(finetuned_zip_path)
            
            if 'baseline_analysis' not in baseline_results:
                raise EmbeddingsPipelineError("The baseline ZIP does not contain a baseline analysis")
            
            if 'finetuned_analysis' not in finetuned_results:
                raise EmbeddingsPipelineError("The fine-tuned ZIP does not contain a fine-tuned analysis")
            
            baseline_analysis = baseline_results['baseline_analysis']
            finetuned_analysis = finetuned_results['finetuned_analysis']
            
            comparison = {
                'baseline_zip': str(baseline_zip_path),
                'finetuned_zip': str(finetuned_zip_path),
                'comparison_timestamp': datetime.now().isoformat(),
                'reduction_methods': {
                    'baseline_best': baseline_analysis['reduction']['best_method'],
                    'finetuned_best': finetuned_analysis['reduction']['best_method']
                },
                'clustering_methods': {
                    'baseline_best': baseline_analysis['clustering']['best_method'],
                    'finetuned_best': finetuned_analysis['clustering']['best_method']
                }
            }
            
            baseline_clustering = baseline_analysis['clustering']['comparison_df']
            finetuned_clustering = finetuned_analysis['clustering']['comparison_df']
            
            comparison['clustering_metrics_comparison'] = self._compare_clustering_metrics(
                baseline_clustering, finetuned_clustering
            )
            
            if baseline_clustering and finetuned_clustering:
                baseline_best = max(baseline_clustering, key=lambda x: x.get('adjusted_rand_score', 0))
                finetuned_best = max(finetuned_clustering, key=lambda x: x.get('adjusted_rand_score', 0))
                
                comparison['performance_improvement'] = {
                    'baseline_ari': baseline_best.get('adjusted_rand_score', 0),
                    'finetuned_ari': finetuned_best.get('adjusted_rand_score', 0),
                    'ari_improvement': finetuned_best.get('adjusted_rand_score', 0) - baseline_best.get('adjusted_rand_score', 0),
                    'baseline_silhouette': baseline_best.get('silhouette_score', 0),
                    'finetuned_silhouette': finetuned_best.get('silhouette_score', 0),
                    'silhouette_improvement': finetuned_best.get('silhouette_score', 0) - baseline_best.get('silhouette_score', 0),
                    'baseline_nmi': baseline_best.get('normalized_mutual_info', 0),
                    'finetuned_nmi': finetuned_best.get('normalized_mutual_info', 0),
                    'nmi_improvement': finetuned_best.get('normalized_mutual_info', 0) - baseline_best.get('normalized_mutual_info', 0)
                }
            
            logging.info("\n" + "="*80)
            logging.info("COMPARISON SUMMARY")
            logging.info("="*80)
            logging.info(f"Best baseline method: {comparison['clustering_methods']['baseline_best']}")
            logging.info(f"Best fine-tuned method: {comparison['clustering_methods']['finetuned_best']}")
            
            if 'performance_improvement' in comparison:
                perf = comparison['performance_improvement']
                logging.info("\nPERFORMANCE IMPROVEMENTS:")
                logging.info(f"  ARI: {perf['baseline_ari']:.4f} → {perf['finetuned_ari']:.4f} (Δ {perf['ari_improvement']:+.4f})")
                logging.info(f"  Silhouette: {perf['baseline_silhouette']:.4f} → {perf['finetuned_silhouette']:.4f} (Δ {perf['silhouette_improvement']:+.4f})")
                logging.info(f"  NMI: {perf['baseline_nmi']:.4f} → {perf['finetuned_nmi']:.4f} (Δ {perf['nmi_improvement']:+.4f})")
            logging.info("="*80 + "\n")
            
            if save_comparison:
                comparison_name = f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                save_dir = PROJECT_ROOT / "results"
                save_dir.mkdir(parents=True, exist_ok=True)
                
                comparison_path = save_dir / f"{comparison_name}.json"
                safe_json_dump(comparison, comparison_path)
                logging.info(f"Comparison saved to: {comparison_path}")
                
                if 'performance_improvement' in comparison:
                    comparison_df = pd.DataFrame([comparison['performance_improvement']])
                    csv_path = save_dir / f"{comparison_name}.csv"
                    comparison_df.to_csv(csv_path, index=False)
                    logging.info(f"Improvement CSV saved to: {csv_path}")
            
            return comparison
            
        except Exception as e:
            logging.error(f"Error comparing results: {e}")
            raise EmbeddingsPipelineError(f"Error comparing results: {e}") from e
    
    def __str__(self) -> str:
        """String representation of the pipeline."""
        return f"EmbeddingsPipeline(experiment={self.experiment_name})"
    
    def __repr__(self) -> str:
        """Detailed representation of the pipeline."""
        return (
            f"EmbeddingsPipeline("
            f"experiment={self.experiment_name}, "
            f"dataset={'✓' if self.dataset_dict else '✗'}, "
            f"baseline={'✓' if self.baseline_embeddings is not None else '✗'}, "
            f"finetuned={'✓' if self.finetuned_embeddings is not None else '✗'})"
        )


# Factory function
def create_embeddings_pipeline(
    config: Dict[str, Any],
    df: pd.DataFrame,
    experiment_name: Optional[str] = None
) -> EmbeddingsPipeline:
    """Create an :class:`EmbeddingsPipeline` with the given configuration.

    Args:
        config: Pipeline configuration dictionary.
        df: DataFrame with dataset records.
        experiment_name: Experiment name; auto-generated when ``None``.

    Returns:
        Configured :class:`EmbeddingsPipeline` instance.
    """
    return EmbeddingsPipeline(config, df, experiment_name)