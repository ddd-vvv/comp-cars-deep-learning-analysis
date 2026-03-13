"""
Models package for the CompCars project.

Contains deep-learning models for multi-view vehicle classification and
metric learning using pytorch-metric-learning.

Classes:
    MultiViewVisionModel: Multi-view vision model with fine-tuning support.
    MultiViewCLIPModel: Multi-view CLIP model for image-text tasks.
    TripletLossWrapper: Wrapper for pytorch-metric-learning Triplet Loss.
    ContrastiveLossWrapper: Wrapper for pytorch-metric-learning Contrastive Loss.
    ArcFaceLossWrapper: Wrapper for pytorch-metric-learning ArcFace Loss.
    NTXentLossWrapper: Wrapper for NT-Xent Loss (InfoNCE).
    MultiSimilarityLossWrapper: Wrapper for MultiSimilarity Loss.
    CLIPLoss: CLIP contrastive loss for image-text pairs.
    VisionModelError: Exception for vision model errors.
    CLIPModelError: Exception for CLIP model errors.

Functions:
    create_vision_model: Factory function for vision models.
    create_clip_model: Factory function for CLIP models.
    create_metric_learning_loss: Factory function for metric-learning losses.
    create_miner: Factory function for pytorch-metric-learning miners.
"""

from __future__ import annotations

# Import public symbols
from .MyVisionModel import MultiViewVisionModel, VisionModelError, create_vision_model
from .MyCLIPModel import MultiViewCLIPModel, CLIPModelError, create_clip_model
from .MetricLearningLosses import (
    create_metric_learning_loss,
    TripletLossWrapper,
    ContrastiveLossWrapper,
    ArcFaceLossWrapper,
    NTXentLossWrapper,
    MultiSimilarityLossWrapper,
    CLIPLoss,
    create_miner
)

__all__ = [
    # Vision model
    'MultiViewVisionModel',
    # CLIP model
    'MultiViewCLIPModel',
    # Metric learning loss wrappers
    'TripletLossWrapper',
    'ContrastiveLossWrapper',
    'ArcFaceLossWrapper',
    'NTXentLossWrapper',
    'MultiSimilarityLossWrapper',
    'CLIPLoss',
    # Exceptions
    'VisionModelError',
    'CLIPModelError',
    # Factory functions
    'create_vision_model',
    'create_clip_model',
    'create_metric_learning_loss',
    'create_miner'
]

# Package metadata
__version__ = '1.0.0'
__author__ = 'Diego Vega'

# Suppress log output from library code by default.
import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())