"""
CompCars - Deep Learning Vehicle Analysis.

This project implements computer-vision models for vehicle classification
using the CompCars dataset, with a focus on fine-tuning,
embeddings analysis, clustering, and multimodal models.

Packages:
    config: Transform configuration and hyperparameters
    data: Dataset processing and management
    models: Vision and multimodal models
    pipeline: Training and analysis pipelines
    utils: Analysis and visualisation utilities

Main features:
    - Fine-tuning of pre-trained models (ResNet, ViT, CLIP, etc.)
    - Embeddings analysis before and after fine-tuning
    - Clustering and visualisation of learned representations
    - Multimodal models (vision + text) with CLIP
    - Multi-view support (front/rear) for vehicles
    - Reproducible pipelines with JSON configuration
"""

from __future__ import annotations

# Import main submodules
from . import config
from . import data
from . import models
from . import pipeline
from . import utils

# Import most-used classes for direct access
from .data import CarDataset, DataFrameMaker, create_car_dataset
from .models import MultiViewVisionModel, create_vision_model
from .pipeline import FineTuningPipeline, EmbeddingsPipeline
from .config import TransformConfig, create_standard_transform

# Define what is exported when doing "from src import *"
__all__ = [
    # Submodules
    'config',
    'data', 
    'models',
    'pipeline',
    'utils',
    # Main classes 
    'CarDataset',
    'DataFrameMaker',
    'MultiViewVisionModel',
    'FineTuningPipeline',
    'EmbeddingsPipeline',
    'TransformConfig',
    # Factory functions
    'create_car_dataset',
    'create_vision_model',
    'create_standard_transform',
]

# Project information
__version__ = '1.0.0'
__author__ = 'Diego Vega'
__email__ = 'diegovega2000@live.com'
__description__ = 'Vehicle analysis with deep learning using the CompCars dataset'

# Logging
import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())