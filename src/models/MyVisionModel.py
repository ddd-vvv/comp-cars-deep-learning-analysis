"""
Vision model for multi-view vehicle clustering and classification.

Provides a wrapper around torchvision pre-trained backbones that supports
fine-tuning, multi-view embedding extraction, classification, and
metric-learning objectives.

Public API:
    VisionModelError: Domain-specific exception.
    MultiViewVisionModel: Multi-view vision model with fine-tuning support.
    create_vision_model: Convenience factory function.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from pytorch_metric_learning.distances import CosineSimilarity, LpDistance

from src.defaults import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_NUM_WORKERS,
    DEFAULT_WARMUP_EPOCHS,
    DEFAULT_OBJECTIVE,
    DEFAULT_PIN_MEMORY,
    DEFAULT_WEIGHTS,
    DEFAULT_FINETUNE_CRITERION,
    DEFAULT_FINETUNE_OPTIMIZER_TYPE,
    DEFAULT_FINETUNE_EPOCHS,
    DEFAULT_WEIGHTS_FILENAME,
    DEFAULT_USE_AMP,
    SUPPORTED_MODEL_ATTRIBUTES,
    MODEL_CONFIGS,
)

__all__ = [
    "VisionModelError",
    "MultiViewVisionModel",
    "create_vision_model",
]

warnings.filterwarnings('ignore')
logging.getLogger(__name__).addHandler(logging.NullHandler())


class VisionModelError(Exception):
    """Domain-specific exception for vision model errors."""
    pass


class MultiViewVisionModel(nn.Module):
    """Multi-view vision model for vehicle classification and metric learning.

    Wraps a torchvision pre-trained backbone and replaces its final layer
    so that it can be used either for classification (softmax head) or
    metric learning (identity head + external loss).

    Attributes:
        name: Descriptive name of the model instance.
        model_name: torchvision model identifier.
        weights: Pre-trained weights identifier.
        device: Compute device (CPU / GPU).
        objective: Task objective (``'classification'`` or ``'metric_learning'``).
        dataset: Dataset with train/val/test splits.
        batch_size: DataLoader batch size.
        model: torchvision backbone (final layer replaced with Identity).
        embedding_dim: Output dimension of the backbone.
        head_layer: Classification head or Identity for metric learning.
        train_loader: DataLoader for training.
        val_loader: DataLoader for validation.
        test_loader: DataLoader for testing.

    Example:
        >>> model = MultiViewVisionModel(
        ...     name="ResNet50_MultiView",
        ...     model_name="resnet50",
        ...     device=torch.device("cuda"),
        ...     dataset_dict=dataset_dict,
        ... )
        >>> history = model.finetune(criterion, optimizer, epochs=10)

    Raises:
        VisionModelError: For model-specific errors.
        ValueError: For invalid parameters.
    """

    def __init__(
        self,
        name: str,
        model_name: str,
        weights: str = DEFAULT_WEIGHTS,
        device: torch.device = None,
        objective: str = DEFAULT_OBJECTIVE,
        dataset_dict: dict = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        num_workers: int = DEFAULT_NUM_WORKERS,
        pin_memory: bool = DEFAULT_PIN_MEMORY
    ) -> None:
        """Initialise the multi-view vision model.

        Args:
            name: Descriptive model name.
            model_name: torchvision model identifier.
            weights: Pre-trained weights identifier.
            device: Compute device.
            objective: Task objective (``'classification'`` or ``'metric_learning'``).
            dataset_dict: Dict with ``'dataset'``, ``'train_loader'``,
                ``'val_loader'``, and ``'test_loader'``.
            batch_size: Batch size.
            num_workers: Number of DataLoader workers.
            pin_memory: Whether to pin memory in DataLoaders.

        Raises:
            VisionModelError: On initialisation errors.
            ValueError: If any parameter is invalid.
        """
        super().__init__()

        # Validate parameters
        self._validate_parameters(name, model_name, device, dataset_dict, batch_size)

        self.name = name
        self.model_name = model_name
        self.weights = weights
        self.device = device
        self.objective = objective
        self.dataset = dataset_dict['dataset']
        self.train_loader = dataset_dict['train_loader']
        self.val_loader = dataset_dict['val_loader']
        self.test_loader = dataset_dict['test_loader']
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.model = self._initialize_base_model()
        self.embedding_dim = self._extract_embedding_dimension()
        self._replace_final_layer()

        self.head_layer = None
        if self.objective == 'classification':
            self.head_layer = self._create_classification_layer()
        elif self.objective == 'metric_learning':
            self.head_layer = nn.Identity()
        else:
            raise ValueError(
                f"objective must be 'classification' or 'metric_learning', got: '{self.objective}'"
            )

        self.model.to(self.device)
        self.head_layer.to(self.device)

        logging.info(f"Initialised {self.__class__.__name__}: {self.name}")
        logging.info(f"Base model: {self.model_name} with weights: {self.weights}")
        logging.info(
            f"Objective: {self.objective} | Embedding dim: {self.embedding_dim} | "
            f"Training classes: {self.dataset.get_num_classes_for_training()} | "
            f"Total classes: {self.dataset.get_total_num_classes()}"
        )

    def _validate_parameters(
        self,
        name: str,
        model_name: str,
        device: torch.device,
        dataset_dict: dict,
        batch_size: int,
    ) -> None:
        """Validate constructor parameters.

        Raises:
            VisionModelError: On incompatible torchvision version or missing loaders.
            ValueError: If any parameter value is invalid.
        """
        if not name or not isinstance(name, str):
            raise ValueError("name must be a non-empty string")

        if not model_name or not isinstance(model_name, str):
            raise ValueError("model_name must be a non-empty string")

        if not hasattr(models, 'get_model'):
            raise VisionModelError("Incompatible torchvision version (get_model not found)")

        if not isinstance(device, torch.device):
            raise ValueError("device must be a torch.device instance")

        if batch_size <= 0:
            raise ValueError("batch_size must be greater than 0")

        if (
            dataset_dict['train_loader'] is None
            or dataset_dict['val_loader'] is None
            or dataset_dict['test_loader'] is None
        ):
            raise ValueError("dataset_dict must provide train_loader, val_loader, and test_loader")

        if not isinstance(dataset_dict['train_loader'], DataLoader):
            raise ValueError("train_loader must be a DataLoader")
        if not isinstance(dataset_dict['val_loader'], DataLoader):
            raise ValueError("val_loader must be a DataLoader")
        if not isinstance(dataset_dict['test_loader'], DataLoader):
            raise ValueError("test_loader must be a DataLoader")

    def _initialize_base_model(self) -> nn.Module:
        """Initialise the torchvision backbone.

        Returns:
            Initialised backbone module.

        Raises:
            VisionModelError: If the model cannot be loaded.
        """
        try:
            model = models.get_model(self.model_name, weights=self.weights)
            logging.info(f"Base model loaded: {self.model_name}")
            return model
        except Exception as exc:
            raise VisionModelError(
                f"Failed to initialise model '{self.model_name}': {exc}"
            ) from exc

    def _extract_embedding_dimension(self) -> int:
        """Determine the backbone's embedding dimensionality.

        Returns:
            Embedding dimensionality.

        Raises:
            VisionModelError: If the dimension cannot be determined.
        """
        for model_family, config in MODEL_CONFIGS.items():
            if model_family in self.model_name.lower():
                if hasattr(self.model, config['feature_attr']):
                    layer = getattr(self.model, config['feature_attr'])
                    if hasattr(layer, config['feature_key']):
                        return getattr(layer, config['feature_key'])
                    elif hasattr(layer, 'head') and hasattr(layer.head, config['feature_key']):
                        return getattr(layer.head, config['feature_key'])

        # Generic search across known final-layer attributes.
        for attr_name in SUPPORTED_MODEL_ATTRIBUTES:
            if hasattr(self.model, attr_name):
                layer = getattr(self.model, attr_name)
                if hasattr(layer, 'in_features'):
                    return layer.in_features
                elif hasattr(layer, 'head') and hasattr(layer.head, 'in_features'):
                    return layer.head.in_features
                elif isinstance(layer, nn.Sequential) and len(layer) > 0:
                    last_layer = layer[-1]
                    if hasattr(last_layer, 'in_features'):
                        return last_layer.in_features

        raise VisionModelError(
            f"Could not determine embedding dimension for '{self.model_name}'. "
            "Consider adding an entry to MODEL_CONFIGS."
        )

    def _replace_final_layer(self) -> None:
        """Replace the backbone's final classification layer with an Identity.

        Raises:
            VisionModelError: If the final layer cannot be replaced.
        """
        final_layer = nn.Identity()
        for config in MODEL_CONFIGS.values():
            if hasattr(self.model, config['feature_attr']):
                setattr(self.model, config['feature_attr'], final_layer)
                logging.debug("Final layer replaced (model-specific config)")
                return

        for attr_name in SUPPORTED_MODEL_ATTRIBUTES:
            if hasattr(self.model, attr_name):
                setattr(self.model, attr_name, final_layer)
                logging.debug("Final layer replaced (generic attribute search)")
                return

        raise VisionModelError(f"Could not replace the final layer for '{self.model_name}'")

    def _create_classification_layer(self) -> nn.Module:
        """Create the classification head.

        Returns:
            Linear layer mapping ``embedding_dim * num_views`` to the number
            of training classes.
        """
        input_dim = self.embedding_dim * self.dataset.num_views
        output_dim = self.dataset.get_num_classes_for_training()
        layer = nn.Linear(input_dim, output_dim)
        nn.init.xavier_uniform_(layer.weight)
        nn.init.zeros_(layer.bias)
        logging.debug(f"Classification head created: {input_dim} → {output_dim}")
        return layer

    def extract_embeddings(
        self,
        dataloader: DataLoader,
        apply_scaling: bool = True,
        show_progress: bool = True,
    ) -> torch.Tensor:
        """Extract embeddings from a DataLoader.

        Args:
            dataloader: DataLoader supplying the images.
            apply_scaling: Whether to apply ``StandardScaler`` to the output.
            show_progress: Whether to display a progress bar.

        Returns:
            Embedding tensor.

        Raises:
            VisionModelError: On extraction errors.
        """
        try:
            self.model.eval()
            embeddings = []

            with torch.no_grad():
                iterator = (
                    tqdm(dataloader, desc='Extracting embeddings', leave=False)
                    if show_progress
                    else dataloader
                )
                for batch in iterator:
                    images = batch['images'].to(self.device)
                    batch_embeddings = self.forward(images)
                    embeddings.append(batch_embeddings.cpu())

            all_embeddings = torch.cat(embeddings, dim=0)

            if apply_scaling:
                scaler = StandardScaler()
                scaled = scaler.fit_transform(all_embeddings.numpy())
                return torch.tensor(scaled, dtype=torch.float32)

            return all_embeddings

        except Exception as exc:
            raise VisionModelError(f"Failed to extract embeddings: {exc}") from exc

    def extract_val_embeddings(self, apply_scaling: bool = True) -> torch.Tensor:
        """Extract embeddings from the validation set.

        Args:
            apply_scaling: Whether to apply ``StandardScaler``.

        Returns:
            Validation embeddings.
        """
        return self.extract_embeddings(self.val_loader, apply_scaling=apply_scaling)

    def extract_test_embeddings(self, apply_scaling: bool = True) -> torch.Tensor:
        """Extract embeddings from the test set.

        Args:
            apply_scaling: Whether to apply ``StandardScaler``.

        Returns:
            Test embeddings.
        """
        return self.extract_embeddings(self.test_loader, apply_scaling=apply_scaling)

    def finetune(
        self,
        criterion: nn.Module = DEFAULT_FINETUNE_CRITERION,
        optimizer: torch.optim.Optimizer = DEFAULT_FINETUNE_OPTIMIZER_TYPE,
        epochs: int = DEFAULT_FINETUNE_EPOCHS,
        warmup_epochs: int = DEFAULT_WARMUP_EPOCHS,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        early_stopping: Optional[Dict[str, Any]] = None,
        save_best: bool = True,
        checkpoint_dir: Optional[Union[str, Path]] = None,
        use_amp: bool = DEFAULT_USE_AMP,
        gradient_clip_value: Optional[float] = None,
    ) -> Dict[str, List[float]]:
        """Fine-tune the model.

        Args:
            criterion: Loss function.
            optimizer: Optimiser.
            epochs: Total training epochs.
            warmup_epochs: Number of warm-up epochs for learning-rate scheduling.
            scheduler: Optional custom LR scheduler (applied after warm-up).
            early_stopping: Early-stopping config dict with optional keys
                ``'patience'`` and ``'restore_best_weights'``.
            save_best: Whether to save the best checkpoint to disk.
            checkpoint_dir: Directory for checkpoints.
            use_amp: Whether to use Automatic Mixed Precision.
            gradient_clip_value: Max gradient norm for clipping. ``None`` to disable.

        Returns:
            Dict with training history (loss and metric lists per epoch).

        Raises:
            VisionModelError: On training errors.
        """
        try:
            history = {
                'train_loss': [], 
                'val_loss': [], 
                'val_accuracy': [],
                'val_recall@1': [],
                'val_recall@3': [],
                'val_recall@5': []
            }
            
            best_val_loss = float('inf')
            best_epoch = 0
            patience_counter = 0

            best_model_state = None
            best_head_state = None

            warmup_scheduler = None
            if warmup_epochs > 0:
                warmup_scheduler = self._create_warmup_scheduler(optimizer, warmup_epochs)

            early_stop_patience = early_stopping.get('patience', 10) if early_stopping else None
            restore_best_weights = early_stopping.get('restore_best_weights', True) if early_stopping else save_best

            scaler = None
            device_type = 'cuda' if self.device.type == 'cuda' else 'cpu'
            if use_amp and device_type == 'cuda':
                scaler = torch.amp.GradScaler('cuda')
                logging.info("Automatic Mixed Precision enabled")
            elif use_amp and device_type != 'cuda':
                logging.warning("AMP requested but no GPU available; disabling AMP")
                use_amp = False

            logging.info(f"Starting fine-tuning: epochs={epochs}, warmup={warmup_epochs}, AMP={use_amp}")

            for epoch in range(epochs):
                train_loss = self._train_epoch(
                    criterion, optimizer, epoch, epochs, 
                    use_amp=use_amp, scaler=scaler,
                    gradient_clip_value=gradient_clip_value
                )
                
                if warmup_scheduler and epoch < warmup_epochs:
                    warmup_scheduler.step()
                elif scheduler:
                    scheduler.step()

                val_loss, val_accuracy, val_recalls = self._validate_epoch(criterion, epoch, epochs)

                # Save history
                history['train_loss'].append(train_loss)
                history['val_loss'].append(val_loss)
                history['val_accuracy'].append(val_accuracy)
                history['val_recall@1'].append(val_recalls['top1'] if val_recalls else None)
                history['val_recall@3'].append(val_recalls['top3'] if val_recalls else None)
                history['val_recall@5'].append(val_recalls['top5'] if val_recalls else None)

                log_msg = (
                    f'Epoch {epoch+1}/{epochs} | '
                    f'Train Loss: {train_loss:.4f} | '
                    f'Val Loss: {val_loss:.4f}'
                )
                
                if self.objective in ['classification']:
                    log_msg += f' | Val Acc: {val_accuracy:.2f}%'
                
                if val_recalls:
                    log_msg += f" | Recall@1: {val_recalls['top1']:.2f}% | Recall@3: {val_recalls['top3']:.2f}% | Recall@5: {val_recalls['top5']:.2f}%"
                
                logging.info(log_msg)
            
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_epoch = epoch
                    patience_counter = 0

                    if restore_best_weights:
                        best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                        best_head_state = {k: v.cpu().clone() for k, v in self.head_layer.state_dict().items()}
                        logging.info(f"Best model saved to memory (epoch {epoch+1}, val_loss={val_loss:.4f})")

                    if save_best and checkpoint_dir:
                        self._save_checkpoint(checkpoint_dir, epoch, val_loss, 'best')
                        logging.info(f"Checkpoint saved: {checkpoint_dir}/checkpoint_epoch_{epoch}_best.pth")

                else:
                    if early_stopping:
                        patience_counter += 1
                        logging.debug(f"No improvement (patience: {patience_counter}/{early_stop_patience})")

                # Early stopping check
                if early_stopping and patience_counter >= early_stop_patience:
                    logging.info(f"Early stopping at epoch {epoch+1}")
                    logging.info(f"   Best val_loss: {best_val_loss:.4f} at epoch {best_epoch+1}")
                    logging.info(f"   Epochs without improvement: {patience_counter}")
                    break

            # Save the best model at the end of training
            if restore_best_weights and best_model_state is not None:
                logging.info(f"Restoring best model weights (epoch {best_epoch+1}, val_loss={best_val_loss:.4f})")
                self.model.load_state_dict(best_model_state)
                self.head_layer.load_state_dict(best_head_state)
                self.model.to(self.device)
                self.head_layer.to(self.device)
            else:
                logging.info(f"Using final epoch weights (epoch {epoch+1})")

            logging.info("Fine-tuning complete.")
            return history

        except Exception as exc:
            raise VisionModelError(f"Fine-tuning failed: {exc}") from exc

    def _create_warmup_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
    ) -> torch.optim.lr_scheduler.LambdaLR:
        """Create a linear warm-up LR scheduler.

        Args:
            optimizer: Optimiser to schedule.
            warmup_epochs: Number of warm-up epochs.

        Returns:
            ``LambdaLR`` warm-up scheduler.
        """
        def lr_lambda(current_epoch: int) -> float:
            return float(current_epoch + 1) / warmup_epochs if current_epoch < warmup_epochs else 1.0
        
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    def _train_epoch(
        self,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        total_epochs: int,
        use_amp: bool = False,
        scaler: Optional[torch.amp.GradScaler] = None,
        gradient_clip_value: Optional[float] = None,
    ) -> float:
        """Run one training epoch.

        Args:
            criterion: Loss function.
            optimizer: Optimiser.
            epoch: Current epoch index (0-based).
            total_epochs: Total number of epochs.
            use_amp: Whether to use Automatic Mixed Precision.
            scaler: ``GradScaler`` for AMP.
            gradient_clip_value: Max gradient norm for clipping.

        Returns:
            Average loss for the epoch.

        Raises:
            VisionModelError: On training errors.
        """
        self.model.train()

        total_loss = 0.0
        num_batches = len(self.train_loader)
        with tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{total_epochs} [Train]', leave=False) as pbar:
            for batch in pbar:
                images = batch['images'].to(self.device)
                labels = batch['labels'].to(self.device)

                optimizer.zero_grad()

                with torch.amp.autocast(device_type='cuda' if self.device.type == 'cuda' else 'cpu', enabled=use_amp):
                    # Handle multiple views
                    if images.ndim == 4:
                        images = images.unsqueeze(1)

                    B, V, C, H, W = images.shape
                    images_reshaped = images.view(B * V, C, H, W)
                    features = self.model(images_reshaped)
                    features = features.view(B, V, -1)
                    embeddings = features.flatten(start_dim=1)

                    if self.objective == 'classification':
                        outputs = self.head_layer(embeddings)
                        loss = criterion(outputs, labels)
                    elif self.objective == 'metric_learning':
                        loss = criterion(embeddings, labels)
                    else:
                        raise VisionModelError(f"Unknown objective: '{self.objective}'")

                if use_amp and scaler is not None:
                    scaler.scale(loss).backward()
                    if gradient_clip_value is not None:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip_value)
                        torch.nn.utils.clip_grad_norm_(self.head_layer.parameters(), gradient_clip_value)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if gradient_clip_value is not None:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip_value)
                        torch.nn.utils.clip_grad_norm_(self.head_layer.parameters(), gradient_clip_value)
                    optimizer.step()

                total_loss += loss.item()
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        return total_loss / num_batches
    
    def _validate_epoch(
        self,
        criterion: nn.Module,
        epoch: int,
        total_epochs: int,
    ) -> Union[float, float, Optional[Dict[str, float]]]:
        """Run one validation epoch.

        Args:
            criterion: Loss function.
            epoch: Current epoch index (0-based).
            total_epochs: Total number of epochs.

        Returns:
            Tuple of ``(avg_loss, accuracy, recall_metrics)``.
        """
        self.model.eval()
        if self.head_layer is not None:
            self.head_layer.eval()

        total_loss = 0.0
        correct_cls = 0
        total_cls_samples = 0

        # Running buffers for metric-learning recall computation.
        running_emb = []
        running_lbl = []

        with torch.no_grad():
            with tqdm(self.val_loader, desc=f"Epoch {epoch+1}/{total_epochs} [Val]", leave=False) as pbar:
                for batch in pbar:

                    images = batch["images"].to(self.device)
                    labels = batch["labels"].to(self.device)

                    # Handle multiple views
                    if images.ndim == 4:
                        images = images.unsqueeze(1)

                    B, V, C, H, W = images.shape
                    flat_img = images.view(B * V, C, H, W)

                    feats = self.model(flat_img)
                    feats = feats.view(B, V, -1)
                    emb = feats.flatten(start_dim=1)

                    if self.objective == "classification":
                        outputs = self.head_layer(emb)
                        loss = criterion(outputs, labels)
                        total_loss += loss.item()

                        preds = outputs.argmax(1)
                        correct_cls += (preds == labels).sum().item()
                        total_cls_samples += labels.size(0)

                        acc = 100 * correct_cls / max(1, total_cls_samples)
                        pbar.set_postfix({"loss": f"{(total_loss/ (pbar.n+1)):.4f}",
                                      "acc": f"{acc:.2f}%"})

                    else:
                        loss = criterion(emb, labels)
                        total_loss += loss.item()

                        running_emb.append(emb.cpu())
                        running_lbl.append(labels.cpu())

                        # Compute interim recall if we have enough batches.
                        if len(running_emb) > 1:

                            all_e = F.normalize(torch.cat(running_emb, dim=0), p=2, dim=1)
                            all_l = torch.cat(running_lbl, dim=0)

                            # All-pairs distance matrix; mask self-distances.
                            dist = torch.cdist(all_e, all_e, p=2)
                            dist.fill_diagonal_(float("inf"))

                            # Top-5 nearest neighbours.
                            _, idx = torch.topk(dist, k=5, largest=False)

                            # Calculate recalls
                            recalls = {}
                            for k in [1, 3, 5]:
                                topk = all_l[idx[:, :k]]
                                correct_k = (topk == all_l.unsqueeze(1)).any(dim=1).float().mean().item()
                                recalls[f"top{k}"] = correct_k * 100

                            pbar.set_postfix({
                                "loss": f"{(total_loss / (pbar.n + 1)):.4f}",
                                "top1": f"{recalls['top1']:.2f}%",
                                "top3": f"{recalls['top3']:.2f}%",
                                "top5": f"{recalls['top5']:.2f}%"
                            })

            # Final results
            avg_loss = total_loss / len(self.val_loader)

            if self.objective == "classification":
                final_acc = 100 * correct_cls / total_cls_samples
                return avg_loss, final_acc, None

            else:
                all_e = F.normalize(torch.cat(running_emb, dim=0), p=2, dim=1)
                all_l = torch.cat(running_lbl, dim=0)

                dist = torch.cdist(all_e, all_e, p=2)
                dist.fill_diagonal_(float("inf"))
                _, idx = torch.topk(dist, k=5, largest=False)

                # Calculate final recalls on the full validation set
                final_recalls = {}
                for k in [1, 3, 5]:
                    topk = all_l[idx[:, :k]]
                    correct_k = (topk == all_l.unsqueeze(1)).any(dim=1).float().mean().item()
                    final_recalls[f"top{k}"] = correct_k * 100

                return avg_loss, 0.0, final_recalls

    def evaluate(
        self,
        dataloader: Optional[DataLoader] = None,
        distance_metric: str = 'cosine',
        k_values: List[int] = [1, 3, 5],
    ) -> Dict[str, float]:
        """Evaluate the model.

        For ``classification``: standard accuracy.
        For ``metric_learning``: KNN recall\@K using pytorch-metric-learning inference.

        Args:
            dataloader: DataLoader to evaluate. Defaults to ``val_loader``.
            distance_metric: Distance metric (``'cosine'`` or ``'euclidean'``).
            k_values: K values for recall\@K.

        Returns:
            Dict with evaluation metrics.
        """
        if dataloader is None:
            dataloader = self.val_loader

        self.model.eval()

        if self.head_layer is not None:
            self.head_layer.eval()

        # Extract query embeddings (val / test set)
        query_embeddings = []
        query_labels = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc='Extracting query embeddings', leave=False):
                images = batch['images'].to(self.device)
                labels = batch['labels']

                # Handle multiple views
                if images.ndim == 4:
                    images = images.unsqueeze(1)

                B, V, C, H, W = images.shape
                images_reshaped = images.view(B * V, C, H, W)
                features = self.model(images_reshaped)
                features = features.view(B, V, -1)
                embeddings = features.flatten(start_dim=1)

                query_embeddings.append(embeddings.cpu())
                query_labels.append(labels)

        query_embeddings = torch.cat(query_embeddings, dim=0)
        query_labels = torch.cat(query_labels, dim=0)

        # Evaluate based on objective
        if self.objective == 'classification':
            outputs = self.head_layer(query_embeddings.to(self.device))
            _, predicted = outputs.max(1)
            predicted = predicted.cpu()

            correct = (predicted == query_labels).sum().item()
            accuracy = 100 * correct / len(query_labels)
            
            return {
                'accuracy': accuracy,
                'correct': correct,
                'total': len(query_labels),
                'method': 'classification'
            }
        
        elif self.objective == 'metric_learning':
            if distance_metric == 'cosine':
                distance = CosineSimilarity()
            elif distance_metric == 'euclidean':
                distance = LpDistance(p=2, normalize_embeddings=False)
            else:
                raise ValueError(f"Unsupported distance metric: '{distance_metric}'")

            # Extract reference embeddings (train set).
            reference_embeddings = []
            reference_labels = []
            
            with torch.no_grad():
                for batch in tqdm(self.train_loader, desc='Extracting reference embeddings', leave=False):
                    images = batch['images'].to(self.device)
                    labels = batch['labels']
                    if images.ndim == 4:
                        images = images.unsqueeze(1)
                    B, V, C, H, W = images.shape
                    images_reshaped = images.view(B * V, C, H, W)
                    features = self.model(images_reshaped)
                    features = features.view(B, V, -1)
                    embeddings = features.flatten(start_dim=1)
                    reference_embeddings.append(embeddings.cpu())
                    reference_labels.append(labels)

            reference_embeddings = torch.cat(reference_embeddings, dim=0)
            reference_labels = torch.cat(reference_labels, dim=0)

            # Normalise embeddings for cosine similarity.
            if distance_metric == 'cosine':
                query_embeddings = F.normalize(query_embeddings, p=2, dim=1)
                reference_embeddings = F.normalize(reference_embeddings, p=2, dim=1)

            dist_matrix = distance(query_embeddings, reference_embeddings)
            results = {
                'method': 'metric_learning_inference',
                'distance_metric': distance_metric,
                'num_reference': len(reference_labels),
                'num_query': len(query_labels)
            }
            
            max_k = max(k_values)
            _, indices = torch.topk(dist_matrix, k=max_k, dim=1, largest=False)
            
            for k in k_values:
                topk_indices = indices[:, :k]
                topk_labels = reference_labels[topk_indices]
                correct = (topk_labels == query_labels.unsqueeze(1)).any(dim=1).float()
                recall = correct.mean().item() * 100
                results[f'recall@{k}'] = recall
            
            results['accuracy'] = results.get('recall@1', 0.0)
            return results 

        else:
            raise VisionModelError(f"Unknown objective: '{self.objective}'")

    def save_model(self, save_path: Union[str, Path], **kwargs) -> None:
        """Save the full model checkpoint to disk.

        Args:
            save_path: Directory in which to save the checkpoint.
            **kwargs: Extra metadata stored in the checkpoint.
        """
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        model_path = save_path / DEFAULT_WEIGHTS_FILENAME

        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'head_layer_state_dict': self.head_layer.state_dict(),
            'model_config': {
                'name': self.name,
                'model_name': self.model_name,
                'weights': self.weights,
                'embedding_dim': self.embedding_dim,
                'num_classes': self.dataset.get_num_classes_for_training(),
                'num_views': self.dataset.num_views
            },
            'metadata': kwargs
        }

        torch.save(checkpoint, model_path)
        logging.info(f"Model saved to: {model_path}")

    def load_model(self, load_path: Union[str, Path]) -> Dict[str, Any]:
        """Load a previously saved checkpoint.

        Args:
            load_path: Directory containing the checkpoint file.

        Returns:
            Checkpoint metadata dict.

        Raises:
            VisionModelError: If the checkpoint cannot be found or loaded.
        """
        load_path = Path(load_path)
        model_path = load_path / DEFAULT_WEIGHTS_FILENAME

        if not model_path.exists():
            raise VisionModelError(f"No checkpoint found at: {model_path}")

        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.head_layer.load_state_dict(checkpoint['head_layer_state_dict'])

            config = checkpoint.get('model_config', {})
            num_training_classes = self.dataset.get_num_classes_for_training()
            if config.get('num_classes') != num_training_classes:
                logging.warning(
                    f"Class count mismatch: checkpoint={config.get('num_classes')}, "
                    f"current={num_training_classes}"
                )

            logging.info(f"Model loaded from: {model_path}")
            return checkpoint.get('metadata', {})

        except Exception as exc:
            raise VisionModelError(f"Failed to load model: {exc}") from exc

    def _save_checkpoint(
        self,
        checkpoint_dir: Union[str, Path],
        epoch: int,
        accuracy: float,
        suffix: str = '',
    ) -> None:
        """Save a training checkpoint.

        Args:
            checkpoint_dir: Directory in which to write the checkpoint.
            epoch: Current epoch index.
            accuracy: Metric value (loss or accuracy) to record.
            suffix: Optional filename suffix.
        """
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"checkpoint_epoch_{epoch}_{suffix}.pth" if suffix else f"checkpoint_epoch_{epoch}.pth"
        checkpoint_path = checkpoint_dir / filename

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'head_layer_state_dict': self.head_layer.state_dict(),
            'accuracy': accuracy,
            'model_config': {
                'name': self.name,
                'model_name': self.model_name,
                'embedding_dim': self.embedding_dim
            }
        }

        torch.save(checkpoint, checkpoint_path)

    def forward(self, images: Union[List, torch.Tensor]) -> torch.Tensor:
        """Forward pass — extract and normalise multi-view embeddings.

        Args:
            images: Input images as a 4-D ``[B, C, H, W]`` or
                5-D ``[B, V, C, H, W]`` tensor.

        Returns:
            L2-normalised embeddings ``[B, D]`` averaged over views.
        """
        if images.ndim == 4:
            images = images.unsqueeze(1)

        B, V, C, H, W = images.shape
        images_reshaped = images.view(B * V, C, H, W)

        self.model.eval()

        with torch.no_grad():
            embeddings = self.model(images_reshaped)

        embeddings = embeddings.view(B, V, -1)
        embeddings = embeddings.mean(dim=1)  # average over views
        embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)

        return embeddings
    
    def get_model_info(self) -> Dict[str, Any]:
        """Return a summary dict of the model's configuration and size."""
        return {
            'name': self.name,
            'model_name': self.model_name,
            'weights': self.weights,
            'device': str(self.device),
            'batch_size': self.batch_size,
            'embedding_dim': self.embedding_dim,
            'num_classes': self.dataset.get_num_classes_for_training(),
            'num_total_classes': self.dataset.get_total_num_classes(),
            'num_views': self.dataset.num_views,
            'total_parameters': sum(p.numel() for p in self.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad),
            'model_size_mb': sum(p.numel() * p.element_size() for p in self.parameters()) / (1024 ** 2)
        }

    def __str__(self) -> str:
        """Return a human-readable summary of the model."""
        info = self.get_model_info()
        return (
            f"MultiViewVisionModel(\n"
            f"  name='{info['name']}',\n"
            f"  base_model='{info['model_name']}',\n"
            f"  device='{info['device']}',\n"
            f"  classes={info['num_classes']},\n"
            f"  views={info['num_views']},\n"
            f"  batch_size={info['batch_size']},\n"
            f"  parameters={info['total_parameters']:,}\n"
            f")"
        )

    def __repr__(self) -> str:
        """Return a concise string representation for debugging."""
        return f"MultiViewVisionModel(name='{self.name}', model='{self.model_name}')"


# Factory function
def create_vision_model(
    model_name: str,
    dataset_dict: dict,
    device: Optional[torch.device] = None,
    weights: str = 'DEFAULT',
    batch_size: int = DEFAULT_BATCH_SIZE,
    **kwargs
) -> MultiViewVisionModel:
    """Create and return a :class:`MultiViewVisionModel` instance.

    Args:
        model_name: torchvision model identifier.
        dataset_dict: Dict with ``'dataset'``, ``'train_loader'``,
            ``'val_loader'``, and ``'test_loader'``.
        device: Compute device. Defaults to CUDA if available, else CPU.
        weights: Pre-trained weights identifier.
        batch_size: Batch size.
        **kwargs: Extra arguments forwarded to :class:`MultiViewVisionModel`.

    Returns:
        Initialised :class:`MultiViewVisionModel`.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    name = kwargs.pop('name', f"{model_name.title()}_MultiView")

    return MultiViewVisionModel(
        name=name,
        model_name=model_name,
        weights=weights,
        device=device,
        dataset_dict=dataset_dict,
        batch_size=batch_size,
        **kwargs
    )
