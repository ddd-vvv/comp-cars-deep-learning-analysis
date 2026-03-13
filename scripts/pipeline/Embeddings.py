"""
Embeddings analysis runner script for vision and CLIP models.

For each fine-tuned experiment found under the results directory, computes
baseline and fine-tuned embeddings, runs dimensionality reduction and
clustering analysis, and saves the results to disk.
"""

import os


os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"


import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import pandas as pd
import torch
import yaml

from src.pipeline.EmbeddingsPipeline import create_embeddings_pipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve_repo_path(path_value: str | Path) -> Path:
    """Resolve a path relative to repository root unless absolute."""
    path = Path(path_value)
    return path if path.is_absolute() else REPO_ROOT / path


class EmbeddingsAnalysisRunner:
    """Orchestrates embedding analysis for all experiments in the results directory.

    Iterates over fine-tuning output directories, computes baseline and
    fine-tuned embeddings, runs clustering analysis, and persists results.
    Tracks successful, failed, skipped, and compared experiments.
    """

    def __init__(
        self,
        dataset_path: str = "dataset.csv",
        embeddings_config_path: str = "configs/embeddings.yaml",
        results_base_dir: str = "results/models"
    ) -> None:
        """Initialize the runner with dataset, config, and results paths.

        Args:
            dataset_path: Path to the dataset CSV file.
            embeddings_config_path: Path to the embeddings YAML config.
            results_base_dir: Root directory containing model results.
        """
        self.dataset_path = resolve_repo_path(dataset_path)
        self.embeddings_config_path = resolve_repo_path(embeddings_config_path)
        self.results_base_dir = resolve_repo_path(results_base_dir)

        self.dataset_df = self._load_dataset()
        self.embeddings_config = self._load_embeddings_config()

        self.successful_analyses: List[str] = []
        self.failed_analyses: List[Tuple[str, str]] = []
        self.comparisons_made: List[Dict[str, Any]] = []
        self.skipped_analyses: List[str] = []
    
    def _load_dataset(self) -> pd.DataFrame:
        """Load the dataset CSV from disk."""
        logger.info(f"Loading dataset from: {self.dataset_path}")
        return pd.read_csv(self.dataset_path)
    
    def _load_embeddings_config(self) -> Dict[str, Any]:
        """Load the embeddings YAML configuration."""
        with open(self.embeddings_config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _load_finetuning_config(self, finetuning_dir: Path) -> Dict[str, Any]:
        """Load the fine-tuning config.json from a result directory.

        Args:
            finetuning_dir: Directory containing config.json.

        Returns:
            Configuration dictionary.

        Raises:
            FileNotFoundError: If config.json does not exist in the directory.
        """
        config_path = finetuning_dir / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"config.json not found in {finetuning_dir}")
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _get_merged_config(self, finetuning_dir: Path) -> Dict[str, Any]:
        """Merge fine-tuning config with embeddings config.

        Args:
            finetuning_dir: Directory containing the fine-tuning config.json.

        Returns:
            Merged configuration dictionary (embeddings config takes precedence).
        """
        finetuning_config = self._load_finetuning_config(finetuning_dir)
        return {**finetuning_config, **self.embeddings_config}
    
    def _embeddings_dir_exists(self, embeddings_dir: Path) -> bool:
        """Check whether a complete embeddings output directory already exists.

        Args:
            embeddings_dir: Directory to check.

        Returns:
            True if the directory contains all required output files.
        """
        if not embeddings_dir.exists():
            return False
        required_files = ['results.json', 'config.json', 'labels.npy']
        return all((embeddings_dir / f).exists() for f in required_files)
    
    def _save_results_to_embeddings_dir(self, pipeline, embeddings_dir: Path, analysis_type: str) -> None:
        """Persist pipeline results and embeddings to an output directory.

        Args:
            pipeline: Completed EmbeddingsPipeline instance.
            embeddings_dir: Directory to write outputs to.
            analysis_type: Label stored in results.json (e.g. 'full', 'finetuned_only').
        """
        embeddings_dir.mkdir(parents=True, exist_ok=True)

        with open(embeddings_dir / "config.json", 'w', encoding='utf-8') as f:
            json.dump(pipeline.config, f, indent=2, default=str)

        results_to_save = self._make_json_serializable(pipeline.results.copy())
        results_to_save['analysis_type'] = analysis_type
        with open(embeddings_dir / "results.json", 'w', encoding='utf-8') as f:
            json.dump(results_to_save, f, indent=2, default=str)

        # Call .cpu() to ensure compatibility regardless of device
        if pipeline.baseline_embeddings is not None:
            np.save(embeddings_dir / "baseline_embeddings.npy", pipeline.baseline_embeddings.cpu().numpy())
        if pipeline.finetuned_embeddings is not None:
            np.save(embeddings_dir / "finetuned_embeddings.npy", pipeline.finetuned_embeddings.cpu().numpy())
        if pipeline.baseline_labels is not None:
            np.save(embeddings_dir / "labels.npy", pipeline.baseline_labels.cpu().numpy())

        if 'comparison' in pipeline.results and 'performance_improvement' in pipeline.results['comparison']:
            comparison_df = pd.DataFrame([pipeline.results['comparison']['performance_improvement']])
            comparison_df.to_csv(embeddings_dir / "performance_comparison.csv", index=False)

        logger.info(f"✓ Results saved to: {embeddings_dir}")
    
    def _make_json_serializable(self, obj: Any) -> Any:
        """Recursively convert numpy/torch types to JSON-serializable Python types.

        Args:
            obj: Any Python object that may contain numpy or torch values.

        Returns:
            A JSON-serializable version of the input object.
        """
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(v) for v in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.tolist()
        elif isinstance(obj, (np.bool_,)):
            return bool(obj)
        elif isinstance(obj, Path):
            return str(obj)
        else:
            return obj
    
    # -------------------------------------------------------------------------
    # Analysis methods
    # -------------------------------------------------------------------------

    def run_full_analysis(self, finetuning_dir: Path, embeddings_dir: Path, experiment_name: str) -> Optional[Dict[str, Any]]:
        """Run full baseline + fine-tuned analysis and comparison for one experiment.

        Args:
            finetuning_dir: Directory containing baseline and fine-tuned .pt embeddings.
            embeddings_dir: Directory to save output results.
            experiment_name: Human-readable experiment identifier.

        Returns:
            Results dictionary, or None if the analysis fails.
        """
        logger.info(f"\n--- FULL ANALYSIS: {experiment_name} ---")
        try:
            baseline_path = finetuning_dir / "baseline_embeddings.pt"
            finetuned_path = finetuning_dir / "finetuned_embeddings.pt"

            if not baseline_path.exists() or not finetuned_path.exists():
                raise FileNotFoundError(f"Missing .pt embeddings in {finetuning_dir}")
            
            merged_config = self._get_merged_config(finetuning_dir)
            pipeline = create_embeddings_pipeline(config=merged_config, df=self.dataset_df, experiment_name=experiment_name)
            
            pipeline.create_dataset_for_labels()
            pipeline.load_embeddings_from_files(baseline_path, finetuned_path)
            pipeline.analyze_baseline_embeddings()
            pipeline.analyze_finetuned_embeddings()
            pipeline.compare_results()
            
            self._save_results_to_embeddings_dir(pipeline, embeddings_dir, 'full')
            self.successful_analyses.append(experiment_name)
            return pipeline.results

        except Exception as e:
            self.failed_analyses.append((experiment_name, str(e)))
            logger.error(f"✗ Error in FULL {experiment_name}: {e}")
            return None
    
    def run_finetuned_only_analysis(
        self,
        finetuning_dir: Path,
        embeddings_dir: Path,
        experiment_name: str,
        baseline_results_dir: Optional[Path] = None
    ) -> Optional[Dict[str, Any]]:
        """Run fine-tuned-only analysis, optionally comparing against a saved baseline.

        Args:
            finetuning_dir: Directory containing finetuned_embeddings.pt.
            embeddings_dir: Directory to save output results.
            experiment_name: Human-readable experiment identifier.
            baseline_results_dir: Optional directory with a pre-computed baseline
                results.json to compare against.

        Returns:
            Results dictionary, or None if the analysis fails.
        """
        logger.info(f"\n--- FINETUNED-ONLY ANALYSIS: {experiment_name} ---")
        try:
            finetuned_path = finetuning_dir / "finetuned_embeddings.pt"
            if not finetuned_path.exists():
                raise FileNotFoundError(f"finetuned_embeddings.pt not found in {finetuning_dir}")
            
            merged_config = self._get_merged_config(finetuning_dir)
            pipeline = create_embeddings_pipeline(config=merged_config, df=self.dataset_df, experiment_name=experiment_name)
            
            pipeline.create_dataset_for_labels()
            pipeline.finetuned_embeddings = torch.load(finetuned_path, map_location='cpu')
            pipeline.finetuned_info = {'path': str(finetuned_path), 'shape': list(pipeline.finetuned_embeddings.shape)}
            
            # 1. Analyse fine-tuned embeddings
            pipeline.analyze_finetuned_embeddings()

            # 2. Optionally compare against a saved baseline
            if baseline_results_dir and baseline_results_dir.exists():
                baseline_results_path = baseline_results_dir / "results.json"
                if baseline_results_path.exists():
                    logger.info(f"Comparing against baseline: {baseline_results_dir.name}")
                    comparison = self._compare_with_baseline(pipeline.results, baseline_results_path, experiment_name)
                    if comparison:
                        pipeline.results['comparison'] = comparison
                        self.comparisons_made.append({'experiment': experiment_name, 'baseline': str(baseline_results_dir)})

            # 3. Save results
            self._save_results_to_embeddings_dir(pipeline, embeddings_dir, 'finetuned_only')
            self.successful_analyses.append(experiment_name)
            return pipeline.results

        except Exception as e:
            self.failed_analyses.append((experiment_name, str(e)))
            logger.error(f"✗ Error in FINETUNED-ONLY {experiment_name}: {e}")
            return None
    
    def _compare_with_baseline(
        self,
        finetuned_results: Dict[str, Any],
        baseline_results_path: Path,
        experiment_name: str,
    ) -> Optional[Dict[str, Any]]:
        """Compare fine-tuned results against a previously saved baseline.

        Args:
            finetuned_results: Results dict from the fine-tuned analysis.
            baseline_results_path: Path to the baseline results.json file.
            experiment_name: Experiment name used for logging.

        Returns:
            Comparison dictionary, or None if comparison cannot be computed.
        """
        try:
            with open(baseline_results_path, 'r', encoding='utf-8') as f:
                base_res = json.load(f)
            
            base_analysis = base_res.get('baseline_analysis')
            fine_analysis = finetuned_results.get('finetuned_analysis')
            
            if not base_analysis or not fine_analysis:
                return None
            
            comparison = {
                'baseline_source': str(baseline_results_path),
                'reduction_best': {
                    'base': base_analysis['reduction']['best_method'],
                    'fine': fine_analysis['reduction']['best_method']
                }
            }
            
            base_best = max(base_analysis['clustering']['comparison_df'], key=lambda x: x.get('adjusted_rand_score', 0))
            fine_best = max(fine_analysis['clustering']['comparison_df'], key=lambda x: x.get('adjusted_rand_score', 0))
            
            comparison['performance_improvement'] = {
                'baseline_ari': base_best.get('adjusted_rand_score', 0),
                'finetuned_ari': fine_best.get('adjusted_rand_score', 0),
                'ari_delta': fine_best.get('adjusted_rand_score', 0) - base_best.get('adjusted_rand_score', 0)
            }
            return comparison
        except Exception:
            return None

    # -------------------------------------------------------------------------
    # Model processing
    # -------------------------------------------------------------------------

    def process_vision_model(self, model_name: str) -> None:
        """Process all classification and metric-learning experiments for a vision model.

        Args:
            model_name: Name of the model directory (e.g. 'resnet50', 'vit_b_32').
        """
        model_dir = self.results_base_dir / model_name
        if not model_dir.exists():
            return

        logger.info(f"Checking model: {model_name}")
        
        class_dir = model_dir / "classification" / "finetuning"
        if class_dir.exists():
            for view_dir in class_dir.iterdir():
                if not view_dir.is_dir():
                    continue
                embeddings_dir = model_dir / "classification" / "embeddings" / view_dir.name
                exp_name = f"{model_name}_class_{view_dir.name}"
                if self._embeddings_dir_exists(embeddings_dir):
                    self.skipped_analyses.append(exp_name)
                else:
                    self.run_full_analysis(view_dir, embeddings_dir, exp_name)

        ml_dir = model_dir / "metric learning"
        if ml_dir.exists():
            for loss_dir in ml_dir.iterdir():
                if not loss_dir.is_dir():
                    continue
                ft_dir = loss_dir / "finetuning"
                if not ft_dir.exists():
                    continue
                for view_dir in ft_dir.iterdir():
                    if not view_dir.is_dir():
                        continue
                    embeddings_dir = loss_dir / "embeddings" / view_dir.name
                    exp_name = f"{model_name}_{loss_dir.name}_{view_dir.name}"
                    if self._embeddings_dir_exists(embeddings_dir):
                        self.skipped_analyses.append(exp_name)
                    else:
                        base_res_dir = model_dir / "classification" / "embeddings" / view_dir.name
                        self.run_finetuned_only_analysis(view_dir, embeddings_dir, exp_name, base_res_dir)

    def process_clip(self) -> None:
        """Process all CLIP experiments including fine-tuning and min-images study."""
        clip_dir = self.results_base_dir / "clip"
        if not clip_dir.exists():
            return

        logger.info("Checking model: CLIP")
        
        # Regular
        ft_dir = clip_dir / "finetuning"
        if ft_dir.exists():
            for view_dir in ft_dir.iterdir():
                if not view_dir.is_dir():
                    continue
                embeddings_dir = clip_dir / "embeddings" / view_dir.name
                exp_name = f"clip_{view_dir.name}"
                if self._embeddings_dir_exists(embeddings_dir):
                    self.skipped_analyses.append(exp_name)
                else:
                    self.run_full_analysis(view_dir, embeddings_dir, exp_name)

        min_images_dir = clip_dir / "min_images_study"
        if min_images_dir.exists():
            self._process_clip_min_images_study(min_images_dir)

    def _process_clip_layers_study_forced(self, layers_study_dir: Path, clip_base_dir: Path) -> None:
        """Process CLIP layers study using forced finetuned-only analysis.

        Args:
            layers_study_dir: Directory containing the layers study results.
            clip_base_dir: Base CLIP directory used to locate baseline embeddings.
        """
        ft_dir = layers_study_dir / "finetuning"
        if not ft_dir.exists():
            return

        logger.info(">>> Processing CLIP Layers Study (FORCED FINETUNED ONLY)")

        for modality_dir in ft_dir.iterdir():
            if not modality_dir.is_dir():
                continue
            modality = modality_dir.name

            baseline_dir = clip_base_dir / "embeddings" / modality
            if not (baseline_dir / "results.json").exists():
                logger.warning(f"  Warning: No baseline found for {modality}. Running WITHOUT comparison.")
                baseline_dir = None

            for layer_dir in modality_dir.iterdir():
                if not layer_dir.is_dir():
                    continue
                embeddings_dir = layers_study_dir / "embeddings" / modality / layer_dir.name
                exp_name = f"clip_layers_{modality}_{layer_dir.name}"

                if self._embeddings_dir_exists(embeddings_dir):
                    logger.info(f"  [SKIP] Already exists: {exp_name}")
                    self.skipped_analyses.append(exp_name)
                else:
                    logger.info(f"  -> Running: {exp_name}")
                    self.run_finetuned_only_analysis(
                        finetuning_dir=layer_dir,
                        embeddings_dir=embeddings_dir,
                        experiment_name=exp_name,
                        baseline_results_dir=baseline_dir
                    )

    def _process_clip_min_images_study(self, min_images_dir: Path) -> None:
        """Process CLIP min-images study using full analysis for each scenario.

        Expected directory layout::

            min_images_dir/finetuning/[N images]/[view]

        Args:
            min_images_dir: Directory containing the min-images study results.
        """
        logger.info(">>> Processing CLIP Min Images Study (FULL ANALYSIS)")

        FOLDERS_TO_PROCESS = ["12 images"]

        ft_base_dir = min_images_dir / "finetuning"
        if not ft_base_dir.exists():
            logger.warning(f"  Warning: finetuning directory not found in {min_images_dir}")
            return

        for n_imgs_dir in ft_base_dir.iterdir():
            if not n_imgs_dir.is_dir():
                continue

            if FOLDERS_TO_PROCESS and n_imgs_dir.name not in FOLDERS_TO_PROCESS:
                logger.info(f"  [SKIP] Ignoring folder: {n_imgs_dir.name}")
                continue

            for view_dir in n_imgs_dir.iterdir():
                if not view_dir.is_dir():
                    continue

                embeddings_dir = min_images_dir / "embeddings" / n_imgs_dir.name / view_dir.name
                exp_name = f"clip_min_imgs_{n_imgs_dir.name}_{view_dir.name}"

                if self._embeddings_dir_exists(embeddings_dir):
                    logger.info(f"  [SKIP] Already exists: {exp_name}")
                    self.skipped_analyses.append(exp_name)
                else:
                    logger.info(f"  -> Running FULL Analysis: {exp_name}")
                    self.run_full_analysis(
                        finetuning_dir=view_dir,
                        embeddings_dir=embeddings_dir,
                        experiment_name=exp_name
                    )
    
    def _process_clip_phase_order_study(self, phase_order_dir: Path) -> None:
        """Process CLIP phase-order study using full analysis for each order variant.

        Args:
            phase_order_dir: Directory containing the phase-order study results.
        """
        logger.info(">>>> Processing CLIP Phase Order Study (FULL ANALYSIS)")

        ft_dir_base = phase_order_dir / "finetuning"
        if not ft_dir_base.exists():
            return

        for order_dir in ft_dir_base.iterdir():
            if not order_dir.is_dir():
                continue

            embeddings_dir = phase_order_dir / "embeddings" / order_dir.name
            exp_name = f"clip_phase_order_{order_dir.name}"

            if self._embeddings_dir_exists(embeddings_dir):
                logger.info(f"  [SKIP] Already exists: {exp_name}")
                self.skipped_analyses.append(exp_name)
            else:
                logger.info(f"  -> Running FULL Analysis: {exp_name}")
                self.run_full_analysis(
                    finetuning_dir=order_dir,
                    embeddings_dir=embeddings_dir,
                    experiment_name=exp_name
                )


    def run_all(self) -> Dict[str, int]:
        """Run analysis for all configured experiments.

        Returns:
            Summary dictionary with counts of successful, failed, and skipped analyses.
        """
        self.process_clip()

        return {
            'successful': len(self.successful_analyses),
            'failed': len(self.failed_analyses),
            'skipped': len(self.skipped_analyses)
        }


if __name__ == "__main__":
    runner = EmbeddingsAnalysisRunner()
    summary = runner.run_all()
    logger.info(f"Final summary: {summary}")