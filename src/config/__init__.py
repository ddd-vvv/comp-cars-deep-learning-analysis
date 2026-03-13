"""
Configuration package for the CompCars project.

This package contains classes and factory functions for configuring image
transformations, hyperparameters, and other project-wide settings.

Classes:
    TransformConfig: Image-transformation configuration dataclass.

Functions:
    create_standard_transform: Factory function for creating standard transforms.
"""

from __future__ import annotations

from .TransformConfig import TransformConfig, create_standard_transform
__all__ = [
    # Base Class
    'TransformConfig',
    # Factory Function
    'create_standard_transform'
]

# Package metadata
__version__ = '1.0.0'
__author__ = 'Diego Vega'

# Attach a NullHandler so the library never emits warnings about missing handlers
import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())