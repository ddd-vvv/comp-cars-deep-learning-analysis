"""
Pipeline module for the CompCars project.

This package contains the main pipelines for fine-tuning and embeddings
analysis in the CompCars project.

Classes:
    FineTuningPipeline: Full fine-tuning pipeline with evaluation
    EmbeddingsPipeline: Pipeline for embeddings analysis and clustering
    FineTuningPipelineError: Exception for fine-tuning pipeline errors
    EmbeddingsPipelineError: Exception for embeddings pipeline errors

Functions:
    create_finetuning_pipeline: Factory function to create a fine-tuning pipeline
    create_embeddings_pipeline: Factory function to create an embeddings pipeline
"""

from __future__ import annotations

# Import main classes from the module
from .FineTuningPipeline import (
    FineTuningPipeline,
    FineTuningPipelineError,
    create_finetuning_pipeline,
)
# from .CLIPFineTuningPipeline import (
#     CLIPFineTuningPipeline,
#     CLIPFineTuningPipelineError,
#     create_clip_finetuning_pipeline,
# )
from .EmbeddingsPipeline import (
    EmbeddingsPipeline,
    EmbeddingsPipelineError,
    create_embeddings_pipeline,
)

# Define what is exported when doing "from src.pipeline import *"
__all__ = [
    # Pipeline classes
    'FineTuningPipeline',
    'EmbeddingsPipeline',
    # Exceptions
    'FineTuningPipelineError',
    'EmbeddingsPipelineError',
    # Factory functions
    'create_finetuning_pipeline',
    'create_embeddings_pipeline',
]

# Module information
__version__ = '0.1.0'
__author__ = 'Diego Vega'

# Logging
import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())