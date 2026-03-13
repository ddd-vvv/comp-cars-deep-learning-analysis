"""
Data package for the CompCars project.

This package contains classes and factory functions for processing and
managing CompCars datasets, including CSV DataFrame generation and
custom PyTorch datasets.

Classes:
    DataFrameMaker: Builds a structured CSV from the raw CompCars dataset.
    CarDataset: Custom multi-view vehicle dataset with percentage-based splits.
    MPerClassSamplerWrapper: Wrapper around pytorch-metric-learning's MPerClassSampler.
    CompCarsDatasetError: Exception for CompCars dataset errors.
    CarDatasetError: Exception for vehicle dataset errors.

Functions:
    create_compcars_dataset: Factory function to build a CompCars CSV dataset.
    create_car_dataset: Factory function to create a CarDataset with DataLoaders.
"""

from __future__ import annotations

from .DataFrameMaker import DataFrameMaker, CompCarsDatasetError, create_compcars_dataset
from .MyDataset import CarDataset, MPerClassSamplerWrapper, CarDatasetError, create_car_dataset


__all__ = [
    # Classes
    'DataFrameMaker',
    'CarDataset', 
    'MPerClassSamplerWrapper',
    # Exceptions
    'CompCarsDatasetError',
    'CarDatasetError',
    # Factory functions
    'create_compcars_dataset',
    'create_car_dataset'
]

# Package metadata
__version__ = '1.0.0'
__author__ = 'Diego Vega'

import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())