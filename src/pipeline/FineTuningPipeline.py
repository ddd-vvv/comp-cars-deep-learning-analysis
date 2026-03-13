"""
Fine-tuning pipeline for vision models.

This module provides a pipeline focused exclusively on fine-tuning
vision models.
"""

from __future__ import annotations

import os
import logging
import shutil
import warnings
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

import pandas as pd
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from src.config.TransformConfig import create_standard_transform
from src.data.MyDataset import create_car_dataset
from src.models.MetricLearningLosses import create_metric_learning_loss, create_miner
from src.models.MyVisionModel import create_vision_model
from src.models.MyCLIPModel import create_clip_model
from src.utils.JsonUtils import safe_json_dump
from src.defaults import (
    CLIP_DEFAULT_FINETUNING_PHASES,
    DEFAULT_BACKBONE_LR,
    DEFAULT_BATCH_SIZE,
    DEFAULT_CLASS_GRANULARITY,
    DEFAULT_CLIP_MODEL_NAME,
    DEFAULT_DESCRIPTION_INCLUDE,
    DEFAULT_FINETUNE_CRITERION,
    DEFAULT_FINETUNE_EPOCHS,
    DEFAULT_FINETUNE_OPTIMIZER_TYPE,
    DEFAULT_GRADIENT_CLIP_VALUE,
    DEFAULT_GRAYSCALE,
    DEFAULT_HEAD_LR,
    DEFAULT_K,
    DEFAULT_MODEL_NAME,
    DEFAULT_MODEL_TYPE,
    DEFAULT_NUM_WORKERS,
    DEFAULT_OBJECTIVE,
    DEFAULT_P,
    DEFAULT_PATIENCE,
    DEFAULT_PIN_MEMORY,
    DEFAULT_RESIZE,
    DEFAULT_SCHEDULER_TYPE,
    DEFAULT_SEED,
    DEFAULT_USE_AMP,
    DEFAULT_USE_AUGMENT,
    DEFAULT_USE_BBOX,
    DEFAULT_USE_EARLY_STOPPING,
    DEFAULT_USE_SCHEDULER,
    DEFAULT_VIEWS,
    DEFAULT_WARMUP_EPOCHS,
    DEFAULT_WEIGHT_DECAY,
    DEFAULT_WEIGHTS,
)

warnings.filterwarnings('ignore')
logging.getLogger(__name__).addHandler(logging.NullHandler())

os.environ["TOKENIZERS_PARALLELISM"] = "false"

__all__ = [
    'FineTuningPipeline',
    'FineTuningPipelineError',
    'create_finetuning_pipeline',
]


class FineTuningPipelineError(Exception):
    """Custom exception for fine-tuning pipeline errors."""
    pass


class FineTuningPipeline:
    """Pipeline for fine-tuning vision and vision-language models.

    Handles the complete fine-tuning process: dataset creation, model
    initialisation, training, evaluation, and result saving.

    Note:
        - Supports percentage-based splits (train_ratio, val_ratio, test_ratio).
        - Integrates pytorch-metric-learning for losses.
        - Supported losses: triplet, contrastive, arcface, ntxent, multi_similarity.
        - Auto-configures sampling strategy based on objective.

    Attributes:
        config: Pipeline configuration dictionary.
        df: DataFrame with dataset data.
        dataset_dict: Dictionary with dataset, dataloaders, and sampler.
        model: Model for fine-tuning.
        results: Dictionary with training results.
        
    Example:
        >>> config = {
        ...     'min_images': 5,
        ...     'train_ratio': 0.7,
        ...     'val_ratio': 0.2,
        ...     'test_ratio': 0.1,
        ...     'objective': 'metric_learning',
        ...     'finetune_criterion': {'type': 'triplet', 'margin': 0.2}
        ... }
        >>> pipeline = FineTuningPipeline(config, dataframe)
        >>> results = pipeline.run_full_pipeline()
        >>> pipeline.save_results("experiment_name")
    """
    
    def __init__(
        self, 
        config: Dict[str, Any], 
        df: pd.DataFrame,
        experiment_name: Optional[str] = None,
        model_type: str = 'vision'
    ) -> None:
        """Initialise the fine-tuning pipeline.

        Args:
            config: Pipeline configuration dictionary.
            df: DataFrame with dataset data.
            experiment_name: Experiment name (optional).
            model_type: Model type (``'vision'`` or ``'clip'``).
        """
        self.config = config
        self.df = df
        self.experiment_name = experiment_name or f"finetune_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.model_type = model_type

        # Pipeline components
        self.dataset_dict = None
        self.model = DEFAULT_MODEL_NAME
        self.results = {
            'config': config.copy(),
            'experiment_name': self.experiment_name,
            'start_time': datetime.now().isoformat()
        }
        
        logging.info(f"Initialised FineTuningPipeline: {self.experiment_name}")
    
    def create_dataset(self) -> None:
        """Create the training dataset with percentage-based splits.

        Reads split ratios and sampling strategy from ``self.config``.

        Note:
            For ``objective='metric_learning'`` the sampling strategy is
            automatically set to ``'pk'``; otherwise ``'standard'`` is used.
        """
        # Create separate transforms for train and val/test
        augment = self.config.get('augment', DEFAULT_USE_AUGMENT)
        if augment:
            train_transform = create_standard_transform(
                size=tuple(self.config.get('image_size', DEFAULT_RESIZE)),
                grayscale=self.config.get('grayscale', DEFAULT_GRAYSCALE),
                use_bbox=self.config.get('use_bbox', DEFAULT_USE_BBOX),
                augment=augment
            )
        else: 
            train_transform = create_standard_transform(augment=augment)

        # Inference transform: no augmentation
        val_transform = create_standard_transform(
            size=tuple(self.config.get('image_size', [224, 224])),
            grayscale=self.config.get('grayscale', DEFAULT_GRAYSCALE),
            use_bbox=self.config.get('use_bbox', DEFAULT_USE_BBOX),
            augment=False
        )

        # Determine sampling strategy based on objective
        objective = self.config.get('objective', DEFAULT_OBJECTIVE)
        sampling_strategy = self.config.get('sampling_strategy', 'standard')
        
        # Auto-configure sampling if not explicitly set
        if sampling_strategy == 'standard':
            if objective == 'metric_learning':
                sampling_strategy = 'pk'
                logging.info(f"Auto-configuring sampling_strategy='pk' for objective='{objective}'")
        
        # Create the dataset with the given parameters
        self.dataset_dict = create_car_dataset(
            df=self.df,
            views=self.config.get('views', DEFAULT_VIEWS),
            min_images=self.config.get('min_images', 5),
            train_ratio=self.config.get('train_ratio', 0.7),
            val_ratio=self.config.get('val_ratio', 0.2),
            test_ratio=self.config.get('test_ratio', 0.1),
            train_transform=train_transform,
            val_transform=val_transform,
            batch_size=self.config.get('batch_size', DEFAULT_BATCH_SIZE),
            num_workers=self.config.get('num_workers', DEFAULT_NUM_WORKERS),
            seed=self.config.get('seed', DEFAULT_SEED),
            sampling_strategy=sampling_strategy,
            P=self.config.get('P', DEFAULT_P),
            K=self.config.get('K', DEFAULT_K),
            class_granularity=self.config.get('class_granularity', DEFAULT_CLASS_GRANULARITY),
            model_type=self.config.get('model_type', DEFAULT_MODEL_TYPE),
            description_include=self.config.get('description_include', DEFAULT_DESCRIPTION_INCLUDE),
            include_oneshot_in_test=self.config.get('include_oneshot_in_test', False),
            oneshot_ratio=self.config.get('oneshot_ratio', 0.1)
        )
        
        logging.info(f"Dataset created with objective='{objective}', sampling_strategy='{sampling_strategy}'")

        # Save dataset statistics
        dataset_stats = self.dataset_dict['dataset'].get_dataset_statistics()
        self.results['dataset_stats'] = dataset_stats
    
    def create_model(self) -> None:
        """Create and initialise the model."""
        if self.dataset_dict is None:
            raise FineTuningPipelineError("Dataset not created. Run create_dataset() first.")

        # Configure device
        device = torch.device(self.config.get('device', 'cpu'))

        # Create the model based on type
        if self.model_type == 'vision': # Vision model
            self.model = create_vision_model(
                name=f"{self.config.get('model_name', DEFAULT_MODEL_NAME)}_{self.experiment_name}",
                model_name=self.config.get('model_name', DEFAULT_MODEL_NAME),
                weights=self.config.get('weights', DEFAULT_WEIGHTS),
                device=device,
                objective=self.config.get('objective', DEFAULT_OBJECTIVE),
                dataset_dict=self.dataset_dict,
                batch_size=self.config.get('batch_size', DEFAULT_BATCH_SIZE),
                num_workers=self.config.get('num_workers', DEFAULT_NUM_WORKERS),
                pin_memory=self.config.get('pin_memory', DEFAULT_PIN_MEMORY)
            )

        else: # CLIP model
            self.model = create_clip_model(
                name=f"{self.config.get('model_name', DEFAULT_CLIP_MODEL_NAME)}_{self.experiment_name}",
                model_name=self.config.get('model_name', DEFAULT_CLIP_MODEL_NAME),
                device=device,
                objective=self.config.get('objective', 'CLIP'),
                dataset_dict=self.dataset_dict,
                batch_size=self.config.get('batch_size', DEFAULT_BATCH_SIZE),
                num_workers=self.config.get('num_workers', DEFAULT_NUM_WORKERS),
                pin_memory=self.config.get('pin_memory', DEFAULT_PIN_MEMORY)
            )
    
        # Save model information
        model_info = self.model.get_model_info()
        self.results['model_info'] = model_info
    
    def extract_baseline_embeddings(self) -> torch.Tensor:
        """Extract embeddings before fine-tuning.

        Returns:
            Baseline embeddings tensor.
        """
        if self.model is None:
            raise FineTuningPipelineError("Model not created. Run create_model() first.")

        logging.info("Extracting baseline embeddings...")
        baseline_embeddings = self.model.extract_val_embeddings()
        baseline_eval = self.model.evaluate()

        # Save baseline results
        self.results['baseline'] = {
            'embeddings_shape': list(baseline_embeddings.shape),
            'evaluation': baseline_eval
        }
        # Save embeddings for later use
        self.results['baseline_embeddings'] = baseline_embeddings

        logging.info(f"Baseline embeddings extracted: {baseline_embeddings.shape}")

        # Log based on metric type
        if 'accuracy' in baseline_eval:
            logging.info(f"Baseline accuracy: {baseline_eval['accuracy']:.4f}")
        elif 'recall@1' in baseline_eval:
            logging.info(f"Baseline Recall@1: {baseline_eval['recall@1']:.4f}")
            logging.info(f"Baseline Recall@5: {baseline_eval['recall@5']:.4f}")

        return baseline_embeddings
    
    def run_finetuning(self) -> Dict[str, Any]:
        """Run model fine-tuning using the configured criterion and optimiser.

        The criterion configuration is read from ``self.config['finetune_criterion']``,
        which can be a plain loss-type string or a dict with extra parameters,
        e.g. ``{'type': 'triplet', 'margin': 0.2, 'distance': 'euclidean'}``.

        Returns:
            Dictionary with the training history.

        Raises:
            FineTuningPipelineError: If the model has not been created.
        """
        if self.model is None:
            raise FineTuningPipelineError("Model not created. Run create_model() first.")

        if self.model_type == 'vision':
            # Configure loss criterion using MetricLearningLosses
            criterion_config = self.config.get('finetune_criterion', {})
            
            # If string, convert to dict with type key
            if isinstance(criterion_config, str):
                criterion_config = {'type': criterion_config}
            
            loss_type = criterion_config.get('type', DEFAULT_FINETUNE_CRITERION)
            
            # Validate that the loss is appropriate for the objective
            if loss_type in ['triplet', 'contrastive', 'ntxent', 'multi_similarity', 'arcface']:
                # Validate that the objective is 'metric_learning'
                if self.model.objective != 'metric_learning':
                    raise FineTuningPipelineError(
                        f"Loss '{loss_type}' requires objective='metric_learning', "
                        f"but the model has objective='{self.model.objective}'"
                    )
            
            # Create loss using the MetricLearningLosses factory
            if loss_type in ['triplet', 'contrastive', 'arcface', 'ntxent', 'multisimilarity', 'multi_similarity', 'CLIPLoss']:
                # Compute real embedding_dim accounting for number of views
                num_views = len(self.config.get('views', ['front']))
                actual_embedding_dim = self.model.embedding_dim * num_views
                
                # num_classes: regular training classes only
                num_training_classes = self.dataset_dict['dataset'].get_num_classes_for_training()
                
                criterion = create_metric_learning_loss(
                    loss_type=loss_type,
                    num_classes=num_training_classes,
                    embedding_dim=actual_embedding_dim,
                    **criterion_config  # Pass all additional parameters
                )
                logging.info(f"Criterion '{loss_type}' created using pytorch-metric-learning (classes: {num_training_classes})")
            else:
                # Fallback to standard torch.nn criteria (CrossEntropyLoss, etc.)
                if self.model.objective not in ['classification']:
                    logging.warning(
                        f"Using criterion '{loss_type}' with objective='{self.model.objective}'. "
                        "Consider using an appropriate metric-learning criterion instead."
                    )
                criterion_cls = getattr(torch.nn, loss_type)
                criterion = criterion_cls()
                logging.info(f"Criterion '{loss_type}' created using torch.nn")
            
            # Move criterion to the model device
            criterion.to(self.model.device)
            
            # Configure optimiser with separate parameter groups
            optimizer_type = self.config.get('finetune_optimizer_type', DEFAULT_FINETUNE_OPTIMIZER_TYPE)
            base_lr = float(self.config.get('finetune_backbone_lr', DEFAULT_BACKBONE_LR))
            head_lr = float(self.config.get('finetune_head_lr', DEFAULT_HEAD_LR))
            weight_decay = float(self.config.get('finetune_optimizer_weight_decay', DEFAULT_WEIGHT_DECAY))
            
            optimizer_cls = getattr(torch.optim, optimizer_type)
            
            backbone_params_with_decay = []
            backbone_params_no_decay = []
            head_params_with_decay = []
            head_params_no_decay = []
            
            for n, p in self.model.model.named_parameters():
                if not p.requires_grad:
                    continue
                if "bn" in n or "bias" in n or "norm" in n:
                    backbone_params_no_decay.append(p)
                else:
                    backbone_params_with_decay.append(p)
            
            for n, p in self.model.head_layer.named_parameters():
                if not p.requires_grad:
                    continue
                if "bn" in n or "bias" in n or "norm" in n:
                    head_params_no_decay.append(p)
                else:
                    head_params_with_decay.append(p)
            
            optimizer_params = [
                {"params": backbone_params_with_decay, "lr": base_lr, "weight_decay": weight_decay},
                {"params": backbone_params_no_decay, "lr": base_lr, "weight_decay": 0.0},
                {"params": head_params_with_decay, "lr": head_lr, "weight_decay": weight_decay},
                {"params": head_params_no_decay, "lr": head_lr, "weight_decay": 0.0}
            ]
            
            optimizer = optimizer_cls(optimizer_params)
            
            # Configure scheduler
            scheduler = None
            if self.config.get('use_scheduler', DEFAULT_USE_SCHEDULER):
                scheduler_type = self.config.get('scheduler_type', DEFAULT_SCHEDULER_TYPE)
                num_epochs = self.config.get('finetune_epochs', DEFAULT_FINETUNE_EPOCHS)
                warmup_epochs = self.config.get('warmup_epochs', DEFAULT_WARMUP_EPOCHS)
                
                if scheduler_type == 'cosine_warmup':                    
                    warmup_scheduler = LinearLR(
                        optimizer,
                        start_factor=0.1,
                        end_factor=1.0,
                        total_iters=warmup_epochs
                    )
                    
                    cosine_scheduler = CosineAnnealingLR(
                        optimizer,
                        T_max=num_epochs - warmup_epochs,
                        eta_min=base_lr * 0.01
                    )
                    
                    scheduler = SequentialLR(
                        optimizer,
                        schedulers=[warmup_scheduler, cosine_scheduler],
                        milestones=[warmup_epochs]
                    )
                    logging.info(f"Using cosine scheduler with warmup: {warmup_epochs} warmup epochs, {num_epochs-warmup_epochs} cosine epochs")
                
                elif scheduler_type == 'reduce_on_plateau':
                    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer,
                        mode='min',
                        factor=0.5,
                        patience=3,
                        verbose=True,
                        min_lr=base_lr * 0.001
                    )
                    logging.info("Using ReduceLROnPlateau scheduler")
                
                else:
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                        optimizer,
                        T_max=num_epochs,
                        eta_min=base_lr * 0.01
                    )
                    logging.info("Using standard CosineAnnealingLR scheduler")
            
            # Configure early stopping
            early_stopping = None
            if self.config.get('use_early_stopping', DEFAULT_USE_EARLY_STOPPING):
                early_stopping = {
                    'patience': self.config.get('patience', DEFAULT_PATIENCE),
                    'min_delta': 0.0001,
                    'restore_best_weights': True
                }
            
            # Configure gradient clipping
            gradient_clip_value = self.config.get('gradient_clip_value', DEFAULT_GRADIENT_CLIP_VALUE)
            
            # Configure AMP
            use_amp = self.config.get('use_amp', DEFAULT_USE_AMP)
            
            # Run fine-tuning
            training_history = self.model.finetune(
                criterion=criterion,
                optimizer=optimizer,
                epochs=self.config.get('finetune_epochs', DEFAULT_FINETUNE_EPOCHS),
                warmup_epochs=self.config.get('warmup_epochs', DEFAULT_WARMUP_EPOCHS),
                scheduler=scheduler,
                early_stopping=early_stopping,
                use_amp=use_amp,
                gradient_clip_value=gradient_clip_value
            )

            # Save fine-tuning results
            self.results['finetuning'] = {
                'training_history': training_history,
                'final_train_loss': training_history['train_loss'][-1],
                'final_val_loss': training_history['val_loss'][-1],
                'final_val_accuracy': training_history['val_accuracy'][-1],
                'best_val_accuracy': max(training_history['val_accuracy']),
                'total_epochs': len(training_history['train_loss'])
            }

        else:
            # CLIP fine-tuning by phases
            use_amp = self.config.get('use_amp', DEFAULT_USE_AMP)
            clip_finetuning_phases = self.config.get('clip_finetuning_phases', CLIP_DEFAULT_FINETUNING_PHASES)
            training_history = {}
            
            # Base optimiser configuration
            optimizer_type = self.config.get('finetune_optimizer_type', DEFAULT_FINETUNE_OPTIMIZER_TYPE)
            weight_decay = self.config.get('finetune_optimizer_weight_decay', DEFAULT_WEIGHT_DECAY)
            
            for phase_name, phase_params in clip_finetuning_phases.items():
                logging.info(f"\n{'='*60}")
                logging.info(f"Starting phase: {phase_name}")
                logging.info(f"{'='*60}")
                
                # Create phase-specific optimiser with its LR
                phase_type = phase_params.get('type', 'text')
                phase_lr = float(phase_params.get('lr', 1e-5))
                phase_num_vision_layers = int(phase_params.get('num_vision_layers', 1))
                phase_num_text_layers = int(phase_params.get('num_text_layers', -1))
                
                if phase_type == 'text':
                    self.model.unfreeze_text_encoder(num_layers=phase_num_text_layers)
                elif phase_type == 'projection':
                    self.model.unfreeze_projection_layers()
                elif phase_type == 'vision':
                    self.model.unfreeze_vision_final_layers(num_layers=phase_num_vision_layers)
                elif phase_type == 'projection_refine':
                    self.model.unfreeze_projection_layers()
                else:
                    raise ValueError(f"Phase type '{phase_type}' is not supported.")
                
                params_with_decay = []
                params_no_decay = []
                
                for n, p in self.model.model.named_parameters():
                    if not p.requires_grad:
                        continue
                    # Do not apply weight decay to biases, batch norm, layer norm, etc.
                    if "bias" in n or "bn" in n or "norm" in n or "ln" in n:
                        params_no_decay.append(p)
                    else:
                        params_with_decay.append(p)
                
                optimizer_params = [
                    {"params": params_with_decay, "lr": phase_lr, "weight_decay": weight_decay},
                    {"params": params_no_decay, "lr": phase_lr, "weight_decay": 0.0}
                ]
                
                optimizer_cls = getattr(torch.optim, optimizer_type)
                optimizer = optimizer_cls(optimizer_params)
                
                logging.info(f"Optimiser {optimizer_type} created with lr={phase_lr} for phase '{phase_type}'")
                logging.info(f"Parameters with weight_decay: {len(params_with_decay)}, without weight_decay: {len(params_no_decay)}")
                
                # Display trainable parameters
                params_info = self.model.get_trainable_params_count()
                logging.info(f"Phase '{phase_type}': {params_info['trainable']:,} trainable parameters out of {params_info['total']:,} total")
                
                training_history_phase = self.model.finetune(
                    optimizer=optimizer,
                    epochs=int(phase_params.get('epochs', 5)),
                    warmup_steps=int(phase_params.get('warmup_steps', 200)),
                    early_stopping=phase_params.get('early_stopping', None),
                    save_best=bool(phase_params.get('save_best', True)),
                    use_amp=use_amp
                )

                # Save this phase's history
                training_history[phase_name] = training_history_phase
                logging.info(f"Phase {phase_name} complete")
                
            # Save fine-tuning results
            self.results['finetuning'] = {
                'training_history': training_history,
            }
            
        return training_history
    
    def extract_finetuned_embeddings(self) -> torch.Tensor:
        """Extract embeddings after fine-tuning.

        Returns:
            Post fine-tuning embeddings tensor.
        """
        if self.model is None:
            raise FineTuningPipelineError("Model not created.")

        logging.info("Extracting post fine-tuning embeddings...")
        finetuned_embeddings = self.model.extract_val_embeddings()
        finetuned_eval = self.model.evaluate()

        # Save post fine-tuning results
        self.results['finetuned'] = {
            'embeddings_shape': list(finetuned_embeddings.shape),
            'evaluation': finetuned_eval
        }
        # Save embeddings for later use
        self.results['finetuned_embeddings'] = finetuned_embeddings

        logging.info(f"Post fine-tuning embeddings extracted: {finetuned_embeddings.shape}")

        # Log based on metric type
        if 'accuracy' in finetuned_eval:
            logging.info(f"Post fine-tuning accuracy: {finetuned_eval['accuracy']:.4f}")
        elif 'recall@1' in finetuned_eval:
            logging.info(f"Post fine-tuning Recall@1: {finetuned_eval['recall@1']:.4f}")
            logging.info(f"Post fine-tuning Recall@5: {finetuned_eval['recall@5']:.4f}")

        return finetuned_embeddings
    
    def save_results(self, save_dir: Union[str, Path] = "results") -> Path:
        """Save all results to a directory and create a ZIP file.

        Args:
            save_dir: Base directory for saving results.

        Returns:
            Path to the created ZIP file.
        """
        save_dir = Path(save_dir)
        experiment_dir = save_dir / self.experiment_name
        experiment_dir.mkdir(parents=True, exist_ok=True)
        
        logging.info(f"Saving results to: {experiment_dir}")

        # Save configuration and results as JSON
        safe_json_dump(self.config, experiment_dir / "config.json")
        
        # Finalise results with end timestamp
        self.results['end_time'] = datetime.now().isoformat()
        
        # Separate embeddings and history from the JSON payload
        results_for_json = self.results.copy()
        baseline_embeddings = None
        finetuned_embeddings = None
        training_history = None
        
        if 'baseline_embeddings' in results_for_json:
            baseline_embeddings = results_for_json.pop('baseline_embeddings')

        if 'finetuned_embeddings' in results_for_json:
            finetuned_embeddings = results_for_json.pop('finetuned_embeddings')
            
        if 'finetuning' in results_for_json and 'training_history' in results_for_json['finetuning']:
            training_history = results_for_json['finetuning']['training_history']
        
        safe_json_dump(results_for_json, experiment_dir / "results.json")
        
        # Save embeddings as PyTorch tensors
        if baseline_embeddings is not None:
            torch.save(baseline_embeddings, experiment_dir / "baseline_embeddings.pt")
            logging.info(f"Baseline embeddings saved: {baseline_embeddings.shape}")
            
        if finetuned_embeddings is not None:
            torch.save(finetuned_embeddings, experiment_dir / "finetuned_embeddings.pt")
            logging.info(f"Fine-tuned embeddings saved: {finetuned_embeddings.shape}")
        
        # Save training history as tensor/pickle
        if training_history is not None:
            torch.save(training_history, experiment_dir / "training_history.pt")
            logging.info("Training history saved")
        
        # Save model if it exists
        if self.model is not None:
            model_path = experiment_dir / "model.pth"
            self.model.save_model(model_path)
            logging.info(f"Model saved: {model_path}")
        
        # Save dataset statistics
        if 'dataset_stats' in self.results:
            # Flatten nested statistics for CSV
            stats_flat = {}
            for key, value in self.results['dataset_stats'].items():
                if isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        if isinstance(subvalue, dict):
                            for subsubkey, subsubvalue in subvalue.items():
                                stats_flat[f"{key}_{subkey}_{subsubkey}"] = subsubvalue
                        else:
                            stats_flat[f"{key}_{subkey}"] = subvalue
                else:
                    stats_flat[key] = value
            stats_df = pd.DataFrame([stats_flat])
            stats_df.to_csv(experiment_dir / "dataset_stats.csv", index=False)
        
        # Create ZIP file
        zip_path = save_dir / f"{self.experiment_name}.zip"
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in experiment_dir.rglob("*"):
                if file_path.is_file():
                    arcname = file_path.relative_to(save_dir)
                    zipf.write(file_path, arcname)
        
        # Clean up temporary directory
        if experiment_dir.exists():
            shutil.rmtree(experiment_dir)
        
        logging.info(f"Results saved to ZIP: {zip_path}")
        return zip_path
    
    def run_full_pipeline(self) -> Dict[str, Any]:
        """Run the complete fine-tuning pipeline end-to-end.

        Returns:
            Dictionary with all results.
        """
        logging.info(f"=== STARTING FULL PIPELINE: {self.experiment_name} ===")
        
        try:
            # 1. Create dataset
            self.create_dataset()
            
            # 2. Create model
            self.create_model()
            
            # 3. Extract baseline embeddings
            self.extract_baseline_embeddings()
            
            # 4. Fine-tuning
            self.run_finetuning()
            
            # 5. Extract post fine-tuning embeddings
            self.extract_finetuned_embeddings()
            
            # 6. Compute improvement based on metric type
            baseline_eval = self.results['baseline']['evaluation']
            finetuned_eval = self.results['finetuned']['evaluation']
            
            if 'accuracy' in baseline_eval:
                # Classification mode
                baseline_metric = baseline_eval['accuracy']
                finetuned_metric = finetuned_eval['accuracy']
                metric_name = 'accuracy'
            elif 'recall@1' in baseline_eval:
                # Metric learning mode
                baseline_metric = baseline_eval['recall@1']
                finetuned_metric = finetuned_eval['recall@1']
                metric_name = 'recall@1'
            else:
                raise FineTuningPipelineError("No valid metric found in evaluation results")
            
            improvement = finetuned_metric - baseline_metric
            improvement_percentage = (improvement / baseline_metric) * 100 if baseline_metric > 0 else 0.0
            
            self.results['summary'] = {
                'metric_name': metric_name,
                f'baseline_{metric_name}': baseline_metric,
                f'finetuned_{metric_name}': finetuned_metric,
                f'{metric_name}_improvement': improvement,
                'improvement_percentage': improvement_percentage,
                'objective': self.model.objective 
            }
            
            # Add extra metrics for metric_learning
            if 'recall@5' in baseline_eval:
                self.results['summary']['baseline_recall@5'] = baseline_eval['recall@5']
                self.results['summary']['finetuned_recall@5'] = finetuned_eval['recall@5']
                self.results['summary']['recall@5_improvement'] = finetuned_eval['recall@5'] - baseline_eval['recall@5']
            
            logging.info("=== PIPELINE COMPLETE ===")
            logging.info(f"{metric_name} improvement: {improvement:.4f} ({improvement_percentage:.2f}%)")
            
            # 7. Save results automatically
            zip_path = self.save_results()
            self.results['saved_to'] = str(zip_path)
            
            logging.info(f"Results saved to: {zip_path}")

            return self.results

        except Exception as e:
            logging.error(f"Error in pipeline: {e}")
            self.results['error'] = str(e)
            raise FineTuningPipelineError(f"Error running pipeline: {e}") from e
    
    def __str__(self) -> str:
        """String representation of the pipeline."""
        return f"FineTuningPipeline(experiment={self.experiment_name})"
    
    def __repr__(self) -> str:
        """Detailed representation of the pipeline."""
        return (
            f"FineTuningPipeline("
            f"experiment={self.experiment_name}, "
            f"dataset={'✓' if self.dataset_dict else '✗'}, "
            f"model={'✓' if self.model else '✗'})"
        )


# Factory function 
def create_finetuning_pipeline(
    config: Dict[str, Any],
    df: pd.DataFrame,
    experiment_name: Optional[str] = None,
    model_type: str = 'vision'
) -> FineTuningPipeline:
    """Create a :class:`FineTuningPipeline` with the given configuration.

    Args:
        config: Pipeline configuration dictionary.
        df: DataFrame with dataset data.
        experiment_name: Experiment name (optional).
        model_type: Model type (``'vision'`` or ``'clip'``).

    Returns:
        Configured :class:`FineTuningPipeline` instance.
    """
    return FineTuningPipeline(config, df, experiment_name, model_type)
