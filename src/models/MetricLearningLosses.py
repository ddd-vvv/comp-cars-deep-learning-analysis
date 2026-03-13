"""
Metric learning loss functions for the CompCars project.

Thin wrappers around pytorch-metric-learning losses and a bidirectional CLIP
contrastive loss, together with factory helpers for building loss functions
and hard-example miners.

Public API:
    TripletLossWrapper: Triplet margin loss.
    ContrastiveLossWrapper: Contrastive loss.
    ArcFaceLossWrapper: ArcFace angular margin loss.
    NTXentLossWrapper: NT-Xent / InfoNCE loss.
    MultiSimilarityLossWrapper: Multi-similarity loss.
    CLIPLoss: Bidirectional image-text contrastive loss.
    create_metric_learning_loss: Factory to select and configure a loss.
    create_miner: Factory to select and configure a hard-example miner.
"""

from __future__ import annotations

import logging
import warnings
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_metric_learning import losses, miners, distances, reducers

from src.defaults import (
    DEFAULT_TRIPLET_MARGIN,
    DEFAULT_CONTRASTIVE_MARGIN,
    DEFAULT_ARCFACE_SCALE,
    DEFAULT_ARCFACE_MARGIN,
)

__all__ = [
    "TripletLossWrapper",
    "ContrastiveLossWrapper",
    "ArcFaceLossWrapper",
    "NTXentLossWrapper",
    "MultiSimilarityLossWrapper",
    "CLIPLoss",
    "create_metric_learning_loss",
    "create_miner",
]

warnings.filterwarnings('ignore')
logging.getLogger(__name__).addHandler(logging.NullHandler())


class TripletLossWrapper(nn.Module):
    """Triplet margin loss wrapper for pytorch-metric-learning.

    Uses the library's optimised implementation, which supports multiple
    triplet-mining strategies and better numerical stability.

    Args:
        margin: Minimum margin between positive and negative distances.
        distance: Distance metric (``'euclidean'``, ``'cosine'``, ``'dot'``).
        reducer: How to reduce per-triplet losses (``'mean'`` or ``'sum'``).

    Example:
        >>> loss_fn = TripletLossWrapper(margin=0.2)
        >>> loss = loss_fn(embeddings, labels)
    """
    
    def __init__(
        self, 
        margin: float = DEFAULT_TRIPLET_MARGIN,
        distance: str = 'euclidean',
        reducer: str = 'mean'
        ):
        """Initialise the triplet loss wrapper.

        Args:
            margin: Minimum margin between positive and negative distances.
            distance: Distance metric name.
            reducer: Reduction strategy for per-triplet losses.
        """
        super(TripletLossWrapper, self).__init__()
        
        if distance == 'euclidean':
            distance_obj = distances.LpDistance(p=2, normalize_embeddings=True)
        elif distance == 'cosine':
            distance_obj = distances.CosineSimilarity()
        elif distance == 'dot':
            distance_obj = distances.DotProductSimilarity()
        else:
            raise ValueError(f"Unsupported distance: '{distance}'")

        self.loss_fn = losses.TripletMarginLoss(
            margin=margin,
            distance=distance_obj,
            reducer=self._get_reducer(reducer),
        )
        
        self.margin = margin
        self.distance_type = distance
        
        logging.info(f"TripletLossWrapper: margin={margin}, distance={distance}")
    
    def _get_reducer(self, reducer: str):
        """Return a pytorch-metric-learning reducer object.

        Args:
            reducer: Reduction type (``'mean'`` or ``'sum'``).

        Returns:
            Corresponding reducer instance. Defaults to ``MeanReducer``.
        """
        if reducer == 'mean':
            return reducers.MeanReducer()
        elif reducer == 'sum':
            return reducers.SumReducer()
        else:
            return reducers.MeanReducer()
    
    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute the triplet margin loss.

        Args:
            embeddings: Feature embeddings ``[batch_size, embedding_dim]``.
            labels: Class labels ``[batch_size]``.

        Returns:
            Scalar triplet loss.
        """
        return self.loss_fn(embeddings, labels)


class ContrastiveLossWrapper(nn.Module):
    """Contrastive loss wrapper for pytorch-metric-learning.

    Minimises the distance between same-class pairs and maximises it
    between different-class pairs.

    Args:
        pos_margin: Margin for positive pairs.
        neg_margin: Margin for negative pairs.
        distance: Distance metric (``'euclidean'`` or ``'cosine'``).

    Example:
        >>> loss_fn = ContrastiveLossWrapper(pos_margin=0, neg_margin=1.0)
        >>> loss = loss_fn(embeddings, labels)
    """
    
    def __init__(
        self,
        pos_margin: float = 0,
        neg_margin: float = DEFAULT_CONTRASTIVE_MARGIN,
        distance: str = 'euclidean'
    ):
        """Initialise the contrastive loss wrapper.

        Args:
            pos_margin: Margin for positive pairs.
            neg_margin: Margin for negative pairs.
            distance: Distance metric name.
        """
        super(ContrastiveLossWrapper, self).__init__()
        
        if distance == 'euclidean':
            distance_obj = distances.LpDistance(p=2, normalize_embeddings=True)
        elif distance == 'cosine':
            distance_obj = distances.CosineSimilarity()
        else:
            raise ValueError(f"Unsupported distance: '{distance}'")

        self.loss_fn = losses.ContrastiveLoss(
            pos_margin=pos_margin,
            neg_margin=neg_margin,
            distance=distance_obj
        )
        
        self.pos_margin = pos_margin
        self.neg_margin = neg_margin
        
        logging.info(f"ContrastiveLossWrapper: neg_margin={neg_margin}")
    
    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute the contrastive loss.

        Args:
            embeddings: Feature embeddings ``[batch_size, embedding_dim]``.
            labels: Class labels ``[batch_size]``.

        Returns:
            Scalar contrastive loss.
        """
        return self.loss_fn(embeddings, labels)


class ArcFaceLossWrapper(nn.Module):
    """ArcFace angular margin loss wrapper for pytorch-metric-learning.

    Adds an angular margin in the feature space to improve inter-class
    discriminability.

    Args:
        num_classes: Number of classes.
        embedding_dim: Embedding dimensionality.
        margin: Angular margin in degrees.
        scale: Scale factor applied to the logits.

    Example:
        >>> loss_fn = ArcFaceLossWrapper(num_classes=100, embedding_dim=512)
        >>> loss = loss_fn(embeddings, labels)
    """
    
    def __init__(
        self,
        num_classes: int,
        embedding_dim: int,
        margin: float = DEFAULT_ARCFACE_MARGIN,
        scale: float = DEFAULT_ARCFACE_SCALE
    ):
        """Initialise the ArcFace loss wrapper.

        Args:
            num_classes: Number of target classes.
            embedding_dim: Dimensionality of embeddings.
            margin: Angular margin in radians or degrees.
            scale: Logit scaling factor.
        """
        super(ArcFaceLossWrapper, self).__init__()
        
        # Accept radians (< 1) or degrees and normalise to degrees.
        margin_degrees = math.degrees(margin) if margin < 1 else margin

        self.loss_fn = losses.ArcFaceLoss(
            num_classes=num_classes,
            embedding_size=embedding_dim,
            margin=margin_degrees,
            scale=scale
        )
        
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.margin = margin_degrees
        self.scale = scale
        
        logging.info(
            f"ArcFaceLossWrapper: classes={num_classes}, embedding_dim={embedding_dim}, "
            f"margin={margin_degrees}°, scale={scale}"
        )
    
    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute the ArcFace loss.

        Args:
            embeddings: Feature embeddings ``[batch_size, embedding_dim]``.
            labels: Class labels ``[batch_size]``.

        Returns:
            Scalar ArcFace loss.
        """
        return self.loss_fn(embeddings, labels)

    def get_logits(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Compute class logits for inference (no labels required).

        Args:
            embeddings: Feature embeddings ``[batch_size, embedding_dim]``.

        Returns:
            Logits ``[batch_size, num_classes]``.
        """
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)
        weight = F.normalize(self.loss_fn.W, p=2, dim=1)
        weight_t = weight.t()  # [embedding_dim, num_classes]
        logits = F.linear(embeddings_norm, weight_t) * self.scale
        
        return logits


class NTXentLossWrapper(nn.Module):
    """NT-Xent (InfoNCE) loss wrapper for pytorch-metric-learning.

    Also known as InfoNCE loss; used in SimCLR and CLIP.
    Well-suited for contrastive learning with large batches.

    Args:
        temperature: Temperature parameter for the softmax distribution.

    Example:
        >>> loss_fn = NTXentLossWrapper(temperature=0.07)
        >>> loss = loss_fn(embeddings, labels)
    """
    
    def __init__(self, temperature: float = 0.07):
        """Initialise the NT-Xent loss wrapper.

        Args:
            temperature: Softmax temperature used by InfoNCE.
        """
        super(NTXentLossWrapper, self).__init__()
        
        self.loss_fn = losses.NTXentLoss(temperature=temperature)
        self.temperature = temperature
        
        logging.info(f"NTXentLossWrapper: temperature={temperature}")
    
    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute the NT-Xent loss.

        Args:
            embeddings: Feature embeddings ``[batch_size, embedding_dim]``.
            labels: Class labels ``[batch_size]``.

        Returns:
            Scalar NT-Xent loss.
        """
        return self.loss_fn(embeddings, labels)


class MultiSimilarityLossWrapper(nn.Module):
    """Multi-similarity loss wrapper for pytorch-metric-learning.

    Considers multiple pair similarities simultaneously for better
    training convergence.

    Args:
        alpha: Weight for positive pairs.
        beta: Weight for negative pairs.
        base: Similarity base threshold.

    Example:
        >>> loss_fn = MultiSimilarityLossWrapper()
        >>> loss = loss_fn(embeddings, labels)
    """
    
    def __init__(self, alpha: float = 2.0, beta: float = 50.0, base: float = 0.5):
        """Initialise the multi-similarity loss wrapper.

        Args:
            alpha: Positive-pair weighting factor.
            beta: Negative-pair weighting factor.
            base: Similarity base threshold.
        """
        super(MultiSimilarityLossWrapper, self).__init__()
        
        self.loss_fn = losses.MultiSimilarityLoss(alpha=alpha, beta=beta, base=base)
        
        logging.info(f"MultiSimilarityLossWrapper: alpha={alpha}, beta={beta}")
    
    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute the multi-similarity loss.

        Args:
            embeddings: Feature embeddings ``[batch_size, embedding_dim]``.
            labels: Class labels ``[batch_size]``.

        Returns:
            Scalar multi-similarity loss.
        """
        return self.loss_fn(embeddings, labels)


class CLIPLoss(nn.Module):
    """Bidirectional image-text contrastive loss (CLIP-style).

    Computes symmetric cross-entropy between image-to-text and
    text-to-image logits. Each image must correspond to its paired
    text at the same batch position.

    Args:
        reduction: Loss reduction mode (``'mean'``, ``'sum'``, ``'none'``).

    Example:
        >>> loss_fn = CLIPLoss()
        >>> loss = loss_fn(logits_per_image, logits_per_text)
    """
    
    def __init__(self, reduction: str = 'mean'):
        """Initialise CLIP bidirectional contrastive loss.

        Args:
            reduction: Reduction mode for cross-entropy terms.
        """
        super(CLIPLoss, self).__init__()
        self.reduction = reduction
    
    def forward(
        self,
        logits_per_image: torch.Tensor,
        logits_per_text: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the bidirectional CLIP contrastive loss.

        Args:
            logits_per_image: Image-to-text logits ``[batch_size, batch_size]``.
            logits_per_text: Text-to-image logits ``[batch_size, batch_size]``.

        Returns:
            Scalar CLIP loss averaged over both directions.
        """
        batch_size = logits_per_image.shape[0]
        # Diagonal ground-truth: image i matches text i.
        ground_truth = torch.arange(batch_size, dtype=torch.long, device=logits_per_image.device)
        loss_i2t = F.cross_entropy(logits_per_image, ground_truth, reduction=self.reduction)
        loss_t2i = F.cross_entropy(logits_per_text, ground_truth, reduction=self.reduction)
        return (loss_i2t + loss_t2i) / 2.0


def create_metric_learning_loss(
    loss_type: str,
    num_classes: Optional[int] = None,
    embedding_dim: Optional[int] = None,
    **kwargs,
) -> nn.Module:
    """Build a metric learning loss module.

    Args:
        loss_type: One of ``'triplet'``, ``'contrastive'``, ``'arcface'``,
            ``'ntxent'``, ``'multisimilarity'``, ``'clip'``.
        num_classes: Number of classes — required for ``'arcface'``.
        embedding_dim: Embedding dimension — required for ``'arcface'``.
        **kwargs: Extra keyword arguments forwarded to the chosen loss class.

    Returns:
        Configured loss module.

    Raises:
        ValueError: If ``loss_type`` is unsupported or required parameters
            are missing.

    Example:
        >>> loss = create_metric_learning_loss('triplet', margin=0.2)
        >>> loss = create_metric_learning_loss('arcface', num_classes=100, embedding_dim=512)
        >>> loss = create_metric_learning_loss('clip')
    """
    loss_type_lower = loss_type.lower()
    
    if loss_type_lower in ['triplet', 'tripletloss']:
        margin = kwargs.get('margin', DEFAULT_TRIPLET_MARGIN)
        distance = kwargs.get('distance', 'euclidean')
        return TripletLossWrapper(margin=margin, distance=distance)
    
    elif loss_type_lower in ['contrastive', 'contrastiveloss']:
        neg_margin = kwargs.get('margin', DEFAULT_CONTRASTIVE_MARGIN)
        pos_margin = kwargs.get('pos_margin', 0)
        distance = kwargs.get('distance', 'euclidean')
        return ContrastiveLossWrapper(
            pos_margin=pos_margin, 
            neg_margin=neg_margin, 
            distance=distance
        )
    
    elif loss_type_lower in ['arcface', 'arcfaceloss']:
        if num_classes is None or embedding_dim is None:
            raise ValueError("num_classes and embedding_dim are required for ArcFace")
        margin = kwargs.get('margin', DEFAULT_ARCFACE_MARGIN)
        scale = kwargs.get('scale', DEFAULT_ARCFACE_SCALE)
        return ArcFaceLossWrapper(
            num_classes=num_classes,
            embedding_dim=embedding_dim,
            margin=margin,
            scale=scale
        )
    
    elif loss_type_lower in ['ntxent', 'ntxentloss', 'infonce']:
        temperature = kwargs.get('temperature', 0.07)
        return NTXentLossWrapper(temperature=temperature)
    
    elif loss_type_lower in ['multisimilarity', 'multisimilarityloss']:
        alpha = kwargs.get('alpha', 2.0)
        beta = kwargs.get('beta', 50.0)
        base = kwargs.get('base', 0.5)
        return MultiSimilarityLossWrapper(alpha=alpha, beta=beta, base=base)
    
    elif loss_type_lower in ['clip', 'cliploss']:
        return CLIPLoss()
    
    else:
        raise ValueError(
            f"Unsupported loss type: '{loss_type}'. "
            f"Choose from: 'triplet', 'contrastive', 'arcface', 'ntxent', 'multisimilarity', 'clip'"
        )


def create_miner(miner_type: str = 'all', **kwargs):
    """Build a hard-example miner for pytorch-metric-learning.

    Miners automatically select difficult pairs or triplets during
    training to improve convergence.

    Args:
        miner_type: One of ``'all'``, ``'hard'``, ``'semihard'``,
            ``'distance'``, ``'multisimilarity'``.
        **kwargs: Extra keyword arguments forwarded to the chosen miner class.

    Returns:
        Configured miner, or ``None`` when ``miner_type='all'``.

    Example:
        >>> miner = create_miner('semihard', margin=0.2)
        >>> hard_pairs = miner(embeddings, labels)
    """
    miner_type_lower = miner_type.lower()
    
    if miner_type_lower == 'all':
        return None

    elif miner_type_lower == 'hard':
        return miners.BatchHardMiner()

    elif miner_type_lower == 'semihard':
        margin = kwargs.get('margin', DEFAULT_TRIPLET_MARGIN)
        return miners.TripletMarginMiner(margin=margin, type_of_triplets='semihard')
    
    elif miner_type_lower == 'distance':
        # Distance-based mining
        cutoff = kwargs.get('cutoff', 0.5)
        return miners.DistanceWeightedMiner(cutoff=cutoff)
    
    elif miner_type_lower == 'multisimilarity':
        # Multi-similarity mining
        epsilon = kwargs.get('epsilon', 0.1)
        return miners.MultiSimilarityMiner(epsilon=epsilon)
    
    else:
        logging.warning(f"Unrecognised miner type '{miner_type}', falling back to None (all pairs).")
        return None
