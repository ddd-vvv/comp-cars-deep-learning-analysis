"""
JSON serialisation utilities for numpy and torch types.

Provides helpers for serialising numpy and PyTorch objects to JSON,
including key conversion and a custom encoder.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import torch

__all__ = ["NumpyJSONEncoder", "convert_numpy_keys", "safe_json_dump", "safe_json_dumps"]


def convert_numpy_keys(obj: Any) -> Any:
    """
    Recursively convert numpy integer dict keys to plain Python strings.

    Args:
        obj: Object to convert (dict, list, or any scalar).

    Returns:
        Object with numpy integer keys replaced by ``str``.
    """
    if isinstance(obj, dict):
        return {
            str(k) if isinstance(k, (np.integer, np.int64, np.int32, np.int16, np.int8)) else k: convert_numpy_keys(v)
            for k, v in obj.items()
        }
    elif isinstance(obj, list):
        return [convert_numpy_keys(item) for item in obj]
    else:
        return obj


class NumpyJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles NumPy and PyTorch types.

    Note:
        - ``numpy.integer``  → ``int``
        - ``numpy.floating`` → ``float``
        - ``numpy.ndarray``  → ``list``
        - ``torch.Tensor``   → ``list`` (when ≤ 1 000 elements) or shape string
        - Any object with ``.item()`` (numpy scalars)
    """
    
    def default(self, obj: Any) -> Any:
        if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().tolist() if obj.numel() <= 1000 else str(obj.shape)
        elif hasattr(obj, 'item'):  
            return obj.item()
        return super().default(obj)


def safe_json_dump(obj: Any, file_path: str, indent: int = 2) -> None:
    """
    Write *obj* to a JSON file, automatically handling numpy types.

    Args:
        obj: Object to serialise.
        file_path: Destination file path.
        indent: JSON indentation width (default: 2).
    """
    with open(file_path, "w") as f:
        json.dump(convert_numpy_keys(obj), f, indent=indent, cls=NumpyJSONEncoder)


def safe_json_dumps(obj: Any, indent: int = 2) -> str:
    """
    Serialise *obj* to a JSON string, automatically handling numpy types.

    Args:
        obj: Object to serialise.
        indent: JSON indentation width (default: 2).

    Returns:
        JSON-encoded string.
    """
    return json.dumps(convert_numpy_keys(obj), indent=indent, cls=NumpyJSONEncoder)