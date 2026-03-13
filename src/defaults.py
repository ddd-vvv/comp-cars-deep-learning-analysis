"""
Default constants and configuration values for the CompCars analysis project.

This module centralises every project-wide default so that individual modules
can import what they need without hard-coding magic numbers.
"""

from __future__ import annotations

from typing import Dict, Final, FrozenSet, List, Optional, Set, Tuple

__all__ = [
    # Reproducibility
    "DEFAULT_SEED",
    "DEFAULT_N_JOBS",
    # ImageNet normalisation
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "GRAYSCALE_MEAN",
    "GRAYSCALE_STD",
    # Augmentation
    "DEFAULT_USE_AUGMENT",
    "DEFAULT_COLOR_JITTER_BRIGHTNESS",
    "DEFAULT_COLOR_JITTER_CONTRAST",
    "DEFAULT_COLOR_JITTER_SATURATION",
    "DEFAULT_COLOR_JITTER_HUE",
    "DEFAULT_ROTATION_DEGREES",
    "DEFAULT_RANDOM_ERASING_P",
    # Transform initialisation
    "DEFAULT_GRAYSCALE",
    "DEFAULT_RESIZE",
    "DEFAULT_NORMALIZE",
    "DEFAULT_USE_BBOX",
    "DEFAULT_AUGMENT",
    # Dataset – viewpoints
    "DEFAULT_VIEWPOINT_FILTER",
    "VIEWPOINT_MAPPING",
    "UNKNOWN_TYPE",
    "DEFAULT_COLUMNS_TO_KEEP",
    # Dataset – file names
    "ATTRIBUTES_FILE",
    "MAKE_MODEL_FILE",
    "CAR_TYPE_FILE",
    # Dataset – splits and sampling
    "DEFAULT_VIEWS",
    "DEFAULT_CLASS_GRANULARITY",
    "CLASS_GRANULARITY_OPTIONS",
    "DEFAULT_MIN_IMAGES",
    "DEFAULT_TRAIN_RATIO",
    "DEFAULT_VAL_RATIO",
    "DEFAULT_TEST_RATIO",
    "DEFAULT_TRANSFORM",
    "DEFAULT_ONESHOT",
    "DEFAULT_ONESHOT_RATIO",
    "DEFAULT_P",
    "DEFAULT_K",
    # Model types and descriptions
    "MODEL_TYPES",
    "DEFAULT_MODEL_TYPE",
    "DESCRIPTION_OPTIONS",
    "DEFAULT_DESCRIPTION_INCLUDE",
    # DataLoader
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_NUM_WORKERS",
    "DEFAULT_VERBOSE",
    "UNKNOWN_VALUES",
    # Metric learning losses
    "DEFAULT_OUTPUT_EMBEDDING_DIM",
    "DEFAULT_TRIPLET_MARGIN",
    "DEFAULT_CONTRASTIVE_MARGIN",
    "DEFAULT_ARCFACE_SCALE",
    "DEFAULT_ARCFACE_MARGIN",
    # Vision model
    "SUPPORTED_MODEL_ATTRIBUTES",
    "MODEL_CONFIGS",
    "DEFAULT_MODEL_NAME",
    "DEFAULT_WARMUP_EPOCHS",
    "DEFAULT_WEIGHTS_FILENAME",
    "DEFAULT_WEIGHTS",
    "DEFAULT_OBJECTIVE",
    "DEFAULT_PIN_MEMORY",
    # CLIP model
    "DEFAULT_CLIP_MODEL_NAME",
    "CLIP_EMBEDDING_MODES",
    "DEFAULT_CLIP_EMBEDDING_MODE",
    "CLIP_DEFAULT_FINETUNING_PHASES",
    "CLIP_CONFIGS",
    # Dimensionality reduction
    "DEFAULT_DIMENSIONALITY_REDUCER_OPTUNA_TRIALS",
    "DEFAULT_REDUCER_AVAILABLE_METHODS",
    "DEFAULT_USE_INCREMENTAL_PCA",
    "DEFAULT_INCREMENTAL_BATCH_SIZE",
    # Clustering
    "DEFAULT_CLUSTERING_OPTUNA_TRIALS",
    "DEFAULT_CLUSTERING_AVAILABLE_METHODS",
    # Fine-tuning
    "DEFAULT_FINETUNE_CRITERION",
    "DEFAULT_FINETUNE_OPTIMIZER_TYPE",
    "DEFAULT_BACKBONE_LR",
    "DEFAULT_HEAD_LR",
    "DEFAULT_WEIGHT_DECAY",
    "DEFAULT_USE_SCHEDULER",
    "DEFAULT_SCHEDULER_TYPE",
    "DEFAULT_USE_EARLY_STOPPING",
    "DEFAULT_PATIENCE",
    "DEFAULT_FINETUNE_EPOCHS",
    "DEFAULT_GRADIENT_CLIP_VALUE",
    # AMP
    "DEFAULT_USE_AMP",
    # Error messages
    "ERROR_INVALID_MODEL_TYPE",
    "ERROR_INVALID_DESCRIPTION",
]

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
DEFAULT_SEED: Final[int] = 3  # Lucky number
DEFAULT_N_JOBS: Final[int] = -1

# ---------------------------------------------------------------------------
# ImageNet normalisation constants
# ---------------------------------------------------------------------------
IMAGENET_MEAN: Final[List[float]] = [0.485, 0.456, 0.406]
IMAGENET_STD: Final[List[float]] = [0.229, 0.224, 0.225]
GRAYSCALE_MEAN: Final[List[float]] = [0.5]
GRAYSCALE_STD: Final[List[float]] = [0.5]

# ---------------------------------------------------------------------------
# Augmentation defaults
# ---------------------------------------------------------------------------
DEFAULT_USE_AUGMENT: Final[bool] = False
DEFAULT_COLOR_JITTER_BRIGHTNESS: Final[float] = 0.15
DEFAULT_COLOR_JITTER_CONTRAST: Final[float] = 0.15
DEFAULT_COLOR_JITTER_SATURATION: Final[float] = 0.1
DEFAULT_COLOR_JITTER_HUE: Final[float] = 0.02
DEFAULT_ROTATION_DEGREES: Final[int] = 3
DEFAULT_RANDOM_ERASING_P: Final[float] = 0.05

# ---------------------------------------------------------------------------
# Transform initialisation defaults
# ---------------------------------------------------------------------------
DEFAULT_GRAYSCALE: Final[bool] = True
DEFAULT_RESIZE: Final[Tuple[int, int]] = (224, 224)
DEFAULT_NORMALIZE: Final[bool] = True
DEFAULT_USE_BBOX: Final[bool] = True
DEFAULT_AUGMENT: Final[bool] = True

# ---------------------------------------------------------------------------
# Dataset – viewpoint constants
# ---------------------------------------------------------------------------
DEFAULT_VIEWPOINT_FILTER: Final[FrozenSet[int]] = frozenset({1, 2})  # front and rear
VIEWPOINT_MAPPING: Final[Dict[int, str]] = {
    1: 'front', 2: 'rear', 3: 'side', 4: 'frontside', 5: 'rearside'
}
UNKNOWN_TYPE: Final[str] = 'Unknown'
DEFAULT_COLUMNS_TO_KEEP: Final[List[str]] = [
    'image_name', 'image_path', 'released_year', 'viewpoint',
    'bbox', 'make', 'model', 'type',
]

# ---------------------------------------------------------------------------
# Dataset – file names
# ---------------------------------------------------------------------------
ATTRIBUTES_FILE: Final[str] = 'attributes.txt'
MAKE_MODEL_FILE: Final[str] = 'make_model_name.mat'
CAR_TYPE_FILE: Final[str] = 'car_type.mat'

# ---------------------------------------------------------------------------
# Dataset – splits and sampling
# ---------------------------------------------------------------------------
DEFAULT_VIEWS: Final[List[str]] = ['front']
DEFAULT_CLASS_GRANULARITY: Final[str] = 'model+year'
CLASS_GRANULARITY_OPTIONS: Final[Set[str]] = {'model', 'model+year'}
DEFAULT_MIN_IMAGES: Final[int] = 5
DEFAULT_TRAIN_RATIO: Final[float] = 0.7
DEFAULT_VAL_RATIO: Final[float] = 0.2
DEFAULT_TEST_RATIO: Final[float] = 0.1
DEFAULT_TRANSFORM: Final[Optional[object]] = None
DEFAULT_ONESHOT: Final[bool] = True
DEFAULT_ONESHOT_RATIO: Final[float] = 1.0
DEFAULT_P: Final[int] = 16
DEFAULT_K: Final[int] = 4

# ---------------------------------------------------------------------------
# Model types and text-description options
# ---------------------------------------------------------------------------
MODEL_TYPES: Final[Set[str]] = {'vision', 'textual', 'multimodal'}
DEFAULT_MODEL_TYPE: Final[str] = 'vision'
DESCRIPTION_OPTIONS: Final[Set[str]] = {'', 'make', 'model', 'full'}
DEFAULT_DESCRIPTION_INCLUDE: Final[str] = ''

# ---------------------------------------------------------------------------
# DataLoader defaults
# ---------------------------------------------------------------------------
DEFAULT_BATCH_SIZE: Final[int] = 64
DEFAULT_NUM_WORKERS: Final[int] = 0
DEFAULT_VERBOSE: Final[bool] = True
UNKNOWN_VALUES: Final[Set[object]] = {'unknown', 'Unknown', '', None}

# ---------------------------------------------------------------------------
# Metric-learning loss defaults
# ---------------------------------------------------------------------------
DEFAULT_OUTPUT_EMBEDDING_DIM: Final[int] = 512
DEFAULT_TRIPLET_MARGIN: Final[float] = 1.0
DEFAULT_CONTRASTIVE_MARGIN: Final[float] = 1.0
DEFAULT_ARCFACE_SCALE: Final[float] = 30.0
DEFAULT_ARCFACE_MARGIN: Final[float] = 0.50

# ---------------------------------------------------------------------------
# Vision model defaults
# ---------------------------------------------------------------------------
SUPPORTED_MODEL_ATTRIBUTES: Final[Set[str]] = {'fc', 'heads', 'classifier', 'head'}
MODEL_CONFIGS: Final[Dict[str, Dict[str, str]]] = {
    'resnet':      {'feature_attr': 'fc',         'feature_key': 'in_features'},
    'densenet':    {'feature_attr': 'classifier',  'feature_key': 'in_features'},
    'efficientnet':{'feature_attr': 'classifier',  'feature_key': 'in_features'},
    'vit':         {'feature_attr': 'heads',       'feature_key': 'in_features'},
    'swin':        {'feature_attr': 'head',        'feature_key': 'in_features'},
}
DEFAULT_MODEL_NAME: Final[str] = 'vit_b_32'
DEFAULT_WARMUP_EPOCHS: Final[int] = 5
DEFAULT_WEIGHTS_FILENAME: Final[str] = 'vision_model.pth'
DEFAULT_WEIGHTS: Final[str] = 'IMAGENET1K_V1'
DEFAULT_OBJECTIVE: Final[str] = 'metric_learning'
DEFAULT_PIN_MEMORY: Final[bool] = True

# ---------------------------------------------------------------------------
# CLIP model defaults
# ---------------------------------------------------------------------------
DEFAULT_CLIP_MODEL_NAME: Final[str] = 'clip-vit-base-patch32'
CLIP_EMBEDDING_MODES: Final[Set[str]] = {'image', 'text', 'joint'}
DEFAULT_CLIP_EMBEDDING_MODE: Final[str] = 'joint'
CLIP_DEFAULT_FINETUNING_PHASES: Final[Dict] = {
    'phase 1': {
        'type': 'text',
        'lr': 1e-5,
        'epochs': 15,
        'early_stopping': None,
        'save_best': True,
        'warmup_steps': 200,
        'num_text_layers': -1,  # -1 to unfreeze all text-encoder layers
    },
    'phase 2': {
        'type': 'vision',
        'lr': 1e-5,
        'epochs': 15,
        'early_stopping': None,
        'save_best': True,
        'warmup_steps': 200,
        'num_vision_layers': 1,  # Number of final vision-encoder layers to unfreeze
    },
}

# Known CLIP model configurations
CLIP_CONFIGS: Final[Dict[str, Dict[str, str]]] = {
    'clip-vit-base-patch32': {'model_name': 'openai/clip-vit-base-patch32'},
    'clip-vit-base-patch16': {'model_name': 'openai/clip-vit-base-patch16'},
    'clip-vit-large-patch14': {'model_name': 'openai/clip-vit-large-patch14'},
}

# ---------------------------------------------------------------------------
# Dimensionality reducer defaults
# ---------------------------------------------------------------------------
DEFAULT_DIMENSIONALITY_REDUCER_OPTUNA_TRIALS: Final[int] = 20
DEFAULT_REDUCER_AVAILABLE_METHODS: Final[List[str]] = ['pca', 'tsne', 'umap']
DEFAULT_USE_INCREMENTAL_PCA: Final[bool] = True
DEFAULT_INCREMENTAL_BATCH_SIZE: Final[int] = 1000

# ---------------------------------------------------------------------------
# Clustering analyzer defaults
# ---------------------------------------------------------------------------
DEFAULT_CLUSTERING_OPTUNA_TRIALS: Final[int] = 200
DEFAULT_CLUSTERING_AVAILABLE_METHODS: Final[List[str]] = [
    'dbscan', 'hdbscan', 'agglomerative', 'optics'
]

# ---------------------------------------------------------------------------
# Vision fine-tuning defaults
# ---------------------------------------------------------------------------
DEFAULT_FINETUNE_CRITERION: Final[str] = 'ContrastiveLoss'
DEFAULT_FINETUNE_OPTIMIZER_TYPE: Final[str] = 'AdamW'
DEFAULT_BACKBONE_LR: Final[float] = 1e-4
DEFAULT_HEAD_LR: Final[float] = 1e-3
DEFAULT_WEIGHT_DECAY: Final[float] = 5e-3
DEFAULT_USE_SCHEDULER: Final[bool] = True
DEFAULT_SCHEDULER_TYPE: Final[str] = 'cosine_warmup'
DEFAULT_USE_EARLY_STOPPING: Final[bool] = True
DEFAULT_PATIENCE: Final[int] = 10
DEFAULT_FINETUNE_EPOCHS: Final[int] = 50
DEFAULT_GRADIENT_CLIP_VALUE: Final[float] = 1.0

# ---------------------------------------------------------------------------
# Automatic Mixed Precision
# ---------------------------------------------------------------------------
DEFAULT_USE_AMP: Final[bool] = True

# ---------------------------------------------------------------------------
# Error message templates
# ---------------------------------------------------------------------------
ERROR_INVALID_MODEL_TYPE: Final[str] = "model_type must be one of: {}"
ERROR_INVALID_DESCRIPTION: Final[str] = "description_include must be one of: {}"
