"""
CLIP model for multi-view vehicle clustering with textual descriptions.

Provides a wrapper around HuggingFace CLIP models adapted for multi-view
vehicle clustering and classification, with phase-based fine-tuning and
multimodal embedding extraction.

Public API:
    CLIPModelError: Domain-specific exception.
    MultiViewCLIPModel: Multi-view CLIP model with phase fine-tuning support.
    create_clip_model: Convenience factory function.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor, get_linear_schedule_with_warmup

from .MetricLearningLosses import CLIPLoss

from src.defaults import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_NUM_WORKERS,
    DEFAULT_PIN_MEMORY,
    DEFAULT_FINETUNE_EPOCHS,
    DEFAULT_USE_AMP,
    CLIP_CONFIGS,
)

__all__ = [
    "CLIPModelError",
    "MultiViewCLIPModel",
    "create_clip_model",
]

warnings.filterwarnings('ignore')
logging.getLogger(__name__).addHandler(logging.NullHandler())


class CLIPModelError(Exception):
    """Domain-specific exception for CLIP model errors."""
    pass


class MultiViewCLIPModel(nn.Module):
    """Multi-view CLIP model for vehicle classification and clustering.

    Wraps a HuggingFace CLIP backbone and supports phase-based fine-tuning,
    multimodal embedding extraction, and cross-modal retrieval evaluation.

    Fine-tuning follows a phased strategy:

    1. ``'text'``: Fine-tune the text encoder.
    2. ``'projection'``: Fine-tune image/text projection layers.
    3. ``'vision'``: Fine-tune the final vision encoder layers.
    4. ``'projection_refine'``: Final projection refinement.

    Attributes:
        name: Descriptive model name.
        model_name: HuggingFace CLIP model key (from ``CLIP_CONFIGS``).
        device: Compute device (CPU / GPU).
        dataset: Dataset with train/val/test splits.
        batch_size: DataLoader batch size.
        model: HuggingFace ``CLIPModel``.
        processor: ``CLIPProcessor`` for images and text.
        train_loader: DataLoader for training.
        val_loader: DataLoader for validation.
        test_loader: DataLoader for testing.

    Example:
        >>> model = MultiViewCLIPModel(
        ...     name="CLIP_MultiView",
        ...     model_name="clip-vit-base-patch32",
        ...     device=torch.device("cuda"),
        ...     dataset_dict=dataset_dict,
        ... )
        >>> model.finetune_phase("text", optimizer, epochs=5)

    Raises:
        CLIPModelError: For model-specific errors.
        ValueError: For invalid parameters.
    """
    def __init__(
        self,
        name: str,
        model_name: str = 'clip-vit-base-patch32',
        device: torch.device = None,
        objective: str = 'CLIP',
        dataset_dict: dict = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        num_workers: int = DEFAULT_NUM_WORKERS,
        pin_memory: bool = DEFAULT_PIN_MEMORY
    ) -> None:
        """Initialise the multi-view CLIP model.

        Args:
            name: Descriptive model name.
            model_name: CLIP model key (must be a key in ``CLIP_CONFIGS``).
            device: Compute device. Defaults to CUDA if available, else CPU.
            objective: Model objective (currently only ``'CLIP'`` is supported).
            dataset_dict: Dict with ``'dataset'``, ``'train_loader'``,
                ``'val_loader'``, and ``'test_loader'``.
            batch_size: Batch size.
            num_workers: Number of DataLoader workers.
            pin_memory: Whether to pin memory in DataLoaders.

        Raises:
            CLIPModelError: On initialisation errors.
            ValueError: If any parameter is invalid.
        """
        super().__init__()
        
        self._validate_parameters(name, model_name, device, dataset_dict, batch_size)

        self.name = name
        self.model_name = model_name
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.objective = objective
        self.dataset = dataset_dict['dataset']
        self.train_loader = dataset_dict['train_loader']
        self.val_loader = dataset_dict['val_loader']
        self.test_loader = dataset_dict['test_loader']
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.model, self.processor = self._initialize_clip_model()
        self.contrastive_criterion = CLIPLoss()

        self.model.to(self.device)

        logging.info(f"Initialised {self.__class__.__name__}: {self.name}")
        logging.info(f"Base model: {self.model_name}")

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
            ValueError: If any parameter value is invalid.
        """
        if not name or not isinstance(name, str):
            raise ValueError("name must be a non-empty string")

        if model_name not in CLIP_CONFIGS:
            raise ValueError(f"model_name must be one of: {list(CLIP_CONFIGS.keys())}")

        if device is not None and not isinstance(device, torch.device):
            raise ValueError("device must be a torch.device instance or None")

        if batch_size <= 0:
            raise ValueError("batch_size must be greater than 0")

        if dataset_dict is None:
            raise ValueError("dataset_dict cannot be None")

        if (
            dataset_dict.get('train_loader') is None
            or dataset_dict.get('val_loader') is None
            or dataset_dict.get('test_loader') is None
        ):
            raise ValueError("dataset_dict must contain 'train_loader', 'val_loader', and 'test_loader'")

        if not isinstance(dataset_dict['train_loader'], DataLoader):
            raise ValueError("train_loader must be a DataLoader instance")
        if not isinstance(dataset_dict['val_loader'], DataLoader):
            raise ValueError("val_loader must be a DataLoader instance")
        if not isinstance(dataset_dict['test_loader'], DataLoader):
            raise ValueError("test_loader must be a DataLoader instance")

    def _initialize_clip_model(self) -> Tuple[CLIPModel, CLIPProcessor]:
        """Initialise the CLIP model and processor.

        Returns:
            Tuple of ``(CLIPModel, CLIPProcessor)``.

        Raises:
            CLIPModelError: On initialisation errors.
        """
        try:
            config = CLIP_CONFIGS[self.model_name]
            hf_model_name = config['model_name']
            model = CLIPModel.from_pretrained(hf_model_name, use_safetensors=True).to(self.device)
            processor = CLIPProcessor.from_pretrained(hf_model_name)
            logging.info(f"CLIP model loaded: {hf_model_name}")
            return model, processor
        except Exception as exc:
            raise CLIPModelError(f"Failed to initialise CLIP model '{self.model_name}': {exc}") from exc

    def extract_image_embeddings(self, images: torch.Tensor) -> torch.Tensor:
        """Extract multi-view image embeddings.

        Args:
            images: Tensor ``[B, V, C, H, W]`` or ``[B, C, H, W]`` for a single view.

        Returns:
            L2-normalised image embeddings ``[B, D]`` averaged over views.
        """
        if images.ndim == 4:
            images = images.unsqueeze(1)

        B, V, C, H, W = images.shape
        images_reshaped = images.view(B * V, C, H, W)
        image_embeddings = self.model.get_image_features(pixel_values=images_reshaped)
        image_embeddings = image_embeddings.view(B, V, -1)
        image_embeddings = image_embeddings.mean(dim=1)  # average over views
        image_embeddings = image_embeddings / image_embeddings.norm(dim=1, keepdim=True)

        return image_embeddings
    
    def extract_text_embeddings(self, texts: List[str]) -> torch.Tensor:
        """Extract text embeddings from a list of descriptions.

        Args:
            texts: List of text descriptions.

        Returns:
            L2-normalised text embeddings.
        """
        text_inputs = self.processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77
        ).to(self.device)
        text_embeddings = self.model.get_text_features(**text_inputs)
        text_embeddings = text_embeddings / text_embeddings.norm(dim=1, keepdim=True)

        return text_embeddings

    def extract_joint_embeddings(self, images: torch.Tensor, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract image and text embeddings for similarity computation.

        Args:
            images: Tensor ``[B, V, C, H, W]``.
            texts: List of text descriptions.

        Returns:
            Tuple of ``(image_embeddings, text_embeddings)``.
        """
        image_embeddings = self.extract_image_embeddings(images=images)
        text_embeddings = self.extract_text_embeddings(texts=texts)

        return image_embeddings, text_embeddings
    
    def extract_embeddings(self, dataloader: DataLoader) -> torch.Tensor:
        """Extract combined image-text embeddings from a DataLoader.

        Args:
            dataloader: DataLoader supplying image-text pairs.

        Returns:
            Combined embeddings ``[N, 2 * embedding_dim]``
            (first half image, second half text).
        """
        self.model.eval()

        all_image_embeddings = []
        all_text_embeddings = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc='Extracting embeddings', leave=False):
                images = batch['images'].to(self.device)
                texts = batch['text_description']
                image_embeddings, text_embeddings = self.extract_joint_embeddings(images=images, texts=texts)
                all_image_embeddings.append(image_embeddings.cpu())
                all_text_embeddings.append(text_embeddings.cpu())

        all_image_embeddings = torch.cat(all_image_embeddings, dim=0)
        all_text_embeddings = torch.cat(all_text_embeddings, dim=0)
        combined_embeddings = torch.cat([all_image_embeddings, all_text_embeddings], dim=1)

        return combined_embeddings
    
    def extract_val_embeddings(self) -> torch.Tensor:
        """Extract embeddings from the validation set."""
        return self.extract_embeddings(self.val_loader)

    def extract_test_embeddings(self) -> torch.Tensor:
        """Extract embeddings from the test set."""
        return self.extract_embeddings(self.test_loader)
    
    # -------------------------------------------------------------------------
    # Layer freeze / unfreeze helpers
    # -------------------------------------------------------------------------

    def freeze_all(self) -> None:
        """Freeze all CLIP model parameters."""
        for param in self.model.parameters():
            param.requires_grad = False
        logging.debug("All CLIP model parameters frozen")

    def unfreeze_text_encoder(self, num_layers: int = -1) -> None:
        """Unfreeze the text encoder.

        Args:
            num_layers: Number of final layers to unfreeze.
                ``-1`` unfreezes all text encoder layers.
        """
        self.freeze_all()

        if num_layers == -1:
            for param in self.model.text_model.parameters():
                param.requires_grad = True
            logging.debug("Text encoder fully unfrozen")
        else:
            text_layers = self.model.text_model.encoder.layers
            for layer in text_layers[-num_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True
            logging.debug(f"Last {num_layers} text encoder layer(s) unfrozen")

        for param in self.model.text_model.final_layer_norm.parameters():
            param.requires_grad = True
        for param in self.model.text_projection.parameters():
            param.requires_grad = True
        logging.debug("Text final LayerNorm and text projection enabled for fine-tuning")

    def unfreeze_projection_layers(self) -> None:
        """Unfreeze the image and text projection layers."""
        self.freeze_all()
        for param in self.model.text_projection.parameters():
            param.requires_grad = True
        for param in self.model.visual_projection.parameters():
            param.requires_grad = True
        logging.debug("Projection layers unfrozen")

    def unfreeze_vision_final_layers(self, num_layers: int = 1) -> None:
        """Unfreeze the final layers of the vision encoder.

        Args:
            num_layers: Number of final layers to unfreeze.
                ``-1`` unfreezes all vision encoder layers.
        """
        self.freeze_all()
        vision_layers = self.model.vision_model.encoder.layers

        if num_layers == -1:
            for param in self.model.vision_model.parameters():
                param.requires_grad = True
            logging.info(f"All vision encoder layers unfrozen ({len(vision_layers)} layers)")
        else:
            for layer in vision_layers[-num_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True
            logging.debug(f"Last {num_layers} vision encoder layer(s) unfrozen")

        for param in self.model.vision_model.post_layernorm.parameters():
            param.requires_grad = True
        for param in self.model.visual_projection.parameters():
            param.requires_grad = True
        logging.debug("Vision post-LayerNorm and visual projection enabled for fine-tuning")

    def get_trainable_params_count(self) -> Dict[str, int]:
        """Count the current trainable parameters.

        Returns:
            Dict with keys ``'total'``, ``'trainable'``, and ``'frozen'``.
        """
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            'total': total_params,
            'trainable': trainable_params,
            'frozen': total_params - trainable_params
        }

    # -------------------------------------------------------------------------
    # Fine-tuning
    # -------------------------------------------------------------------------

    def finetune(
        self,
        optimizer: torch.optim.Optimizer,
        epochs: int = DEFAULT_FINETUNE_EPOCHS,
        warmup_steps: int = 0,
        early_stopping: Optional[Dict[str, Any]] = None,
        save_best: bool = True,
        checkpoint_dir: Optional[Union[str, Path]] = None,
        temperature: float = 0.07,
        use_amp: bool = DEFAULT_USE_AMP
    ) -> Dict[str, List[float]]:
        """Fine-tune the model.

        Trains for the requested number of epochs, saves the best checkpoint
        to memory, and restores it after training.

        Args:
            optimizer: Optimiser.
            epochs: Total training epochs.
            warmup_steps: Warm-up steps for the LR scheduler.
            early_stopping: Early-stopping config dict with optional keys
                ``'patience'``, ``'min_delta'``, and ``'restore_best_weights'``.
            save_best: Whether to save the best checkpoint to disk.
            checkpoint_dir: Directory for checkpoints.
            temperature: Logit temperature scaling value.
            use_amp: Whether to use Automatic Mixed Precision.

        Returns:
            Dict with training history (loss and recall lists per epoch).

        Raises:
            CLIPModelError: On training errors.
        """
        total_steps = epochs * len(self.train_loader)

        try:
            history = {
                'train_loss': [],
                'val_loss': [],
                'val_recall@1': [],
                'val_recall@3': [],
                'val_recall@5': []
            }

            best_val_metric = 0.0
            best_epoch = 0
            patience_counter = 0
            best_model_state = None

            warmup_scheduler = None
            if warmup_steps > 0:
                warmup_scheduler = get_linear_schedule_with_warmup(
                    optimizer=optimizer,
                    num_warmup_steps=warmup_steps,
                    num_training_steps=total_steps
                )
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

            logging.info(f"Starting fine-tuning: epochs={epochs}, warmup={warmup_steps}, AMP={use_amp}")

            for epoch in range(epochs):
                train_loss = self._train_epoch(optimizer, epoch, epochs, temperature, use_amp=use_amp, scaler=scaler)

                if warmup_scheduler:
                    warmup_scheduler.step()

                val_loss, val_recalls = self._validate_epoch(epoch, epochs, temperature)

                # Save history
                history['train_loss'].append(train_loss)
                history['val_loss'].append(val_loss)
                history['val_recall@1'].append(val_recalls[1])
                history['val_recall@3'].append(val_recalls[3])
                history['val_recall@5'].append(val_recalls[5])

                # Current metric for early stopping and best-model saving
                current_metric = val_recalls[1]

                log_msg = (
                        f'Epoch {epoch+1}/{epochs} | '
                        f'Train Loss: {train_loss:.4f} | '
                        f'Val Loss: {val_loss:.4f} | '
                        f'Recall@1: {val_recalls[1]:.2f}% | Recall@3: {val_recalls[3]:.2f}% | Recall@5: {val_recalls[5]:.2f}%'
                    )
                
                logging.info(log_msg)

                # Save the best model based on the current metric
                if current_metric > best_val_metric:
                    best_val_metric = current_metric
                    best_epoch = epoch
                    patience_counter = 0

                    if restore_best_weights:
                        best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                        logging.info(f"Best model saved to memory (epoch {epoch+1}, recall@1={current_metric:.2f}%)")

                    if save_best and checkpoint_dir:
                        self._save_checkpoint(checkpoint_dir, epoch, current_metric, 'best')
                        logging.info(f"Checkpoint saved: {checkpoint_dir}/checkpoint_epoch{epoch}_best.pth")
                else:
                    if early_stopping:
                        patience_counter += 1
                        logging.debug(f"No improvement (patience: {patience_counter}/{early_stop_patience})")
                
                if early_stopping and patience_counter >= early_stop_patience:
                    logging.info(f"Early stopping at epoch {epoch+1}")
                    logging.info(f"Best recall@1: {best_val_metric:.2f}% at epoch {best_epoch+1}")
                    logging.info(f"Epochs without improvement: {patience_counter}")
                    break

            if restore_best_weights and best_model_state is not None:
                logging.info(f"Restoring best model (epoch {best_epoch+1}, recall@1={best_val_metric:.2f}%)")
                self.model.load_state_dict(best_model_state)
                self.model.to(self.device)
            else:
                logging.info(f"Using final epoch weights (epoch {epoch+1})")

            logging.info('Fine-tuning complete.')
            return history

        except Exception as exc:
            raise CLIPModelError(f'Fine-tuning failed: {exc}') from exc
        
    def _train_epoch(
        self,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        total_epochs: int,
        temperature: float = 0.07,
        use_amp: bool = False,
        scaler: Optional[torch.amp.GradScaler] = None,
    ) -> float:
        """Run one training epoch.

        Args:
            optimizer: Optimiser.
            epoch: Current epoch index (0-based).
            total_epochs: Total number of epochs.
            temperature: Logit temperature scaling value.
            use_amp: Whether to use Automatic Mixed Precision.
            scaler: ``GradScaler`` for AMP.

        Returns:
            Average loss for the epoch.
        """
        self.model.train()

        total_loss = 0.0
        num_batches = 0
        device_type = 'cuda' if self.device.type == 'cuda' else 'cpu'
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{total_epochs} [Train]", leave=False)
        
        for batch in pbar:
            images = batch['images'].to(self.device)
            texts = batch['text_description']

            optimizer.zero_grad()
        
            # Forward pass 
            with torch.amp.autocast(device_type=device_type, enabled=use_amp):
                image_embeddings, text_embeddings = self.extract_joint_embeddings(images, texts)

                # Compute similarity and loss
                logits_per_image = (image_embeddings @ text_embeddings.t()) / temperature
                logits_per_text = logits_per_image.t()

                loss = self.contrastive_criterion(logits_per_image, logits_per_text)

            # Backward pass con GradScaler
            if use_amp and scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'loss': loss.item()})
        
        return total_loss / num_batches

    def _validate_epoch(
        self,
        epoch: int,
        total_epochs: int,
        temperature: float,
    ) -> Tuple[float, Dict[int, float]]:
        """Run one validation epoch.

        Args:
            epoch: Current epoch index (0-based).
            total_epochs: Total number of epochs.
            temperature: Logit temperature scaling value.

        Returns:
            Tuple of ``(avg_loss, recall_metrics)``.
        """
        self.model.eval()

        total_loss = 0.0
        num_batches = 0

        all_image_embeddings = []
        all_text_embeddings = []
        all_labels = []
        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch+1}/{total_epochs} [Val]", leave=False)
        
        with torch.no_grad():
            for batch in pbar:
                images = batch['images'].to(self.device)
                texts = batch['text_description']
                labels = batch['labels'].to(self.device)

                image_embeddings, text_embeddings = self.extract_joint_embeddings(images=images, texts=texts)

                logits_per_image = (image_embeddings @ text_embeddings.t()) / temperature
                logits_per_text = logits_per_image.t()

                loss = self.contrastive_criterion(logits_per_image, logits_per_text)

                total_loss += loss.item()
                num_batches += 1

                all_image_embeddings.append(image_embeddings.cpu())
                all_text_embeddings.append(text_embeddings.cpu())
                all_labels.append(labels.cpu())

                pbar.set_postfix({'loss': loss.item()})

        all_image_embeddings = torch.cat(all_image_embeddings, dim=0)
        all_text_embeddings = torch.cat(all_text_embeddings, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        recalls = self._calculate_recall_metrics(
            all_image_embeddings,
            all_text_embeddings,
            all_labels,
        )

        avg_loss = total_loss / num_batches

        return avg_loss, recalls
    
    def finetune_phase(
        self,
        phase: str,
        optimizer: torch.optim.Optimizer,
        epochs: int = 5,
        warmup_steps: int = 200,
        early_stopping: Optional[Dict[str, Any]] = None,
        save_best: bool = True,
        checkpoint_dir: Optional[Union[str, Path]] = None,
        use_amp: bool = DEFAULT_USE_AMP,
        num_vision_layers: int = 1,
        num_text_layers: int = -1,
    ) -> Dict[str, List[float]]:
        """Run fine-tuning for a specific phase.

        Configures layer freezing according to ``phase``, then calls
        :meth:`finetune`.

        Args:
            phase: Fine-tuning phase
                (``'text'``, ``'projection'``, ``'vision'``, or
                ``'projection_refine'``).
            optimizer: Optimiser.
            epochs: Number of epochs for this phase.
            warmup_steps: Warm-up steps for the LR scheduler.
            early_stopping: Early-stopping config dict.
            save_best: Whether to save the best checkpoint to disk.
            checkpoint_dir: Directory for checkpoints.
            use_amp: Whether to use Automatic Mixed Precision.
            num_vision_layers: Vision encoder layers to unfreeze (``-1`` for all).
            num_text_layers: Text encoder layers to unfreeze (``-1`` for all).

        Returns:
            Dict with training history.

        Raises:
            ValueError: If ``phase`` is not recognised.
        """
        if phase == 'text':
            self.unfreeze_text_encoder(num_layers=num_text_layers)
        elif phase == 'projection':  # deprecated
            self.unfreeze_projection_layers()
        elif phase == 'vision':
            self.unfreeze_vision_final_layers(num_layers=num_vision_layers)
        elif phase == 'projection_refine':  # deprecated
            self.unfreeze_projection_layers()
        else:
            raise ValueError(
                f"Unsupported phase '{phase}'. "
                "Use 'text', 'projection', 'vision', or 'projection_refine'."
            )

        params_info = self.get_trainable_params_count()
        logging.info(
            f"Phase '{phase}': {params_info['trainable']:,} trainable params "
            f"out of {params_info['total']:,} total"
        )
        
        # Execute training
        return self.finetune(
            optimizer=optimizer,
            epochs=epochs,
            warmup_steps=warmup_steps,
            early_stopping=early_stopping,
            save_best=save_best,
            checkpoint_dir=checkpoint_dir,
            use_amp=use_amp
        )

    def _calculate_recall_metrics(
        self,
        image_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
        labels: torch.Tensor,
        k_values: List[int] = [1, 3, 5],
    ) -> Dict[int, float]:
        """Compute Recall\@K for image-to-text retrieval.

        Args:
            image_embeddings: Shape ``[N, D]``.
            text_embeddings: Shape ``[N, D]``.
            labels: Class labels, shape ``[N]``.
            k_values: K values for Recall\@K.

        Returns:
            Dict mapping each K to its recall percentage.
        """
        similarity = image_embeddings @ text_embeddings.t()

        recalls = {}
        for k in k_values:
            correct = 0
            for i in range(len(labels)):
                top_k_indices = similarity[i].topk(k).indices
                if labels[i] in labels[top_k_indices]:
                    correct += 1
            recalls[k] = 100.0 * correct / len(labels)

        return recalls

    def evaluate(
        self,
        dataloader: Optional[DataLoader] = None,
        temperature: float = 0.07,
    ) -> Dict[str, float]:
        """Evaluate the model using retrieval metrics.

        Args:
            dataloader: DataLoader to evaluate. Defaults to ``val_loader``.
            temperature: Logit temperature scaling value.

        Returns:
            Dict with ``loss``, ``recall@1``, ``recall@3``, ``recall@5``.
        """
        if dataloader is None:
            dataloader = self.val_loader

        self.model.eval()

        total_loss = 0.0
        num_batches = 0

        all_image_embeddings = []
        all_text_embeddings = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc='Evaluating', leave=False):
                images = batch['images'].to(self.device)
                texts = batch['text_description']
                labels = batch['labels'].to(self.device)

                image_embeddings, text_embeddings = self.extract_joint_embeddings(images=images, texts=texts)

                logits_per_image = (image_embeddings @ text_embeddings.t()) / temperature
                logits_per_text = logits_per_image.t()
                loss = self.contrastive_criterion(logits_per_image, logits_per_text)

                total_loss += loss.item()
                num_batches += 1

                all_image_embeddings.append(image_embeddings.cpu())
                all_text_embeddings.append(text_embeddings.cpu())
                all_labels.append(labels.cpu())

        all_image_embeddings = torch.cat(all_image_embeddings, dim=0)
        all_text_embeddings = torch.cat(all_text_embeddings, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        recalls = self._calculate_recall_metrics(
            all_image_embeddings,
            all_text_embeddings,
            all_labels,
        )

        avg_loss = total_loss / num_batches
        
        results = {
            'loss': avg_loss,
            'recall@1': recalls.get(1, 0.0),
            'recall@3': recalls.get(3, 0.0),
            'recall@5': recalls.get(5, 0.0),
        }
        
        logging.info(
            f"Evaluation: Loss = {avg_loss:.4f} | "
            f"Recall@1 = {results['recall@1']:.2f}% | "
            f"Recall@3 = {results['recall@3']:.2f}% | "
            f"Recall@5 = {results['recall@5']:.2f}% | "
        )
        return results

    def save_model(self, save_path: Union[str, Path], **kwargs) -> None:
        """Save the model checkpoint to disk.

        Args:
            save_path: File path for the checkpoint.
            **kwargs: Extra metadata stored in the checkpoint.
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'model_name': self.model_name,
            'model_state_dict': self.model.state_dict(),
            **kwargs
        }
        
        torch.save(checkpoint, save_path)
        logging.info(f"Model saved to: {save_path}")

    def load_model(self, load_path: Union[str, Path]) -> Dict[str, Any]:
        """Load a previously saved checkpoint.

        Args:
            load_path: Path to the checkpoint file.

        Returns:
            Checkpoint dict.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        load_path = Path(load_path)
        
        if not load_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {load_path}")

        checkpoint = torch.load(load_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        logging.info(f"Model loaded from: {load_path}")
        return checkpoint

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
            accuracy: Metric value to record.
            suffix: Optional filename suffix.
        """
        checkpoint_dir = Path(checkpoint_dir)
        filename = f"checkpoint_epoch{epoch}_{suffix}.pth" if suffix else f"checkpoint_epoch{epoch}.pth"
        self.save_model(checkpoint_dir / filename, epoch=epoch, accuracy=accuracy)

    def get_model_info(self) -> Dict[str, Any]:
        """Return a summary dict of the model's configuration and size."""
        params_info = self.get_trainable_params_count()
        
        return {
            'name': self.name,
            'model_name': self.model_name,
            'num_classes': self.dataset.get_num_classes_for_training(),
            'num_total_classes': self.dataset.get_total_num_classes(),
            'device': str(self.device),
            'total_parameters': params_info['total'],
            'trainable_parameters': params_info['trainable'],
            'frozen_parameters': params_info['frozen']
        }

    def __str__(self) -> str:
        """Return a human-readable summary of the model."""
        info = self.get_model_info()

        return (
            f"{self.__class__.__name__}(\n"
            f"  name='{info['name']}',\n"
            f"  model='{info['model_name']}',\n"
            f"  num_classes={info['num_classes']},\n"
            f"  trainable_params={info['trainable_parameters']:,}\n"
            f")"
        )

    def __repr__(self) -> str:
        """Return a concise string representation for debugging."""
        return self.__str__()


# Factory function
def create_clip_model(
    model_name: str,
    dataset_dict: dict,
    device: Optional[torch.device] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    **kwargs
) -> MultiViewCLIPModel:
    """Create and return a :class:`MultiViewCLIPModel` instance.

    Args:
        model_name: CLIP model key (must be a key in ``CLIP_CONFIGS``).
        dataset_dict: Dict with ``'dataset'``, ``'train_loader'``,
            ``'val_loader'``, and ``'test_loader'``.
        device: Compute device. Defaults to CUDA if available, else CPU.
        batch_size: Batch size.
        **kwargs: Extra arguments forwarded to :class:`MultiViewCLIPModel`.

    Returns:
        Initialised :class:`MultiViewCLIPModel`.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    name = kwargs.pop('name', f"CLIP_{model_name.upper()}_MultiView")

    return MultiViewCLIPModel(
        name=name,
        model_name=model_name,
        device=device,
        dataset_dict=dataset_dict,
        batch_size=batch_size,
        **kwargs
    )
