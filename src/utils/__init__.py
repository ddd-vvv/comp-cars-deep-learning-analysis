"""
Utility package for the CompCars project.

Contains tools for clustering analysis, dimensionality reduction,
visualisation, and JSON serialisation.

Classes:
    ClusteringAnalyzer: Multi-algorithm clustering with Optuna optimisation.
    DimensionalityReducer: PCA / t-SNE / UMAP reduction with Optuna optimisation.
    ClusterVisualizer: Cluster image-grid visualisation and purity analysis.
    NumpyJSONEncoder: Custom JSON encoder for NumPy / PyTorch types.

Functions:
    fast_purity_calculation: Numba-compiled clustering purity metric.
    convert_numpy_keys: Convert NumPy integer dict keys to Python strings.
    safe_json_dump: Write an object to a JSON file handling NumPy types.
    safe_json_dumps: Serialise an object to a JSON string handling NumPy types.
"""

from __future__ import annotations

from .ClusteringAnalyzer import ClusteringAnalyzer, fast_purity_calculation
from .DimensionalityReducer import DimensionalityReducer
from .ClusterVisualizer import ClusterVisualizer
from .JsonUtils import (
    NumpyJSONEncoder,
    convert_numpy_keys,
    safe_json_dump,
    safe_json_dumps,
)

__all__ = [
    # Analysis Classes
    'ClusteringAnalyzer',
    'DimensionalityReducer',
    'ClusterVisualizer',
    # JSON Utilities
    'NumpyJSONEncoder',
    'convert_numpy_keys',
    'safe_json_dump',
    'safe_json_dumps',
    # Analysis Functions
    'fast_purity_calculation',
]

__version__ = '1.0.0'
__author__ = 'Diego Vega'

import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())