"""
Shared pytest fixtures for the CompCars test suite.

All fixtures here are automatically available to every test module — no
explicit import is needed.  Session-scoped fixtures are set up once per
``pytest`` run, which avoids repeating expensive I/O (reading the CSV,
loading images, constructing dataset objects).

Notes:
    Tests that access real images (``integration`` marker) require the
    dataset to be present at ``<project_root>/dataset/``.  Tests that only
    need the pre-built CSV require ``<project_root>/dataset.csv``.

    ``CarDataset`` stores image paths exactly as they appear in the
    DataFrame that is passed to it.  To make those paths valid regardless
    of the working directory, the ``df_abs`` fixture rewrites every
    ``image_path`` value to an absolute path before the dataset is
    constructed.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

# --------------------------------------------------------------------------- #
# Project-level paths                                                          #
# --------------------------------------------------------------------------- #

_PROJECT_ROOT = Path(__file__).parent.parent


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Absolute path to the repository root."""
    return _PROJECT_ROOT


@pytest.fixture(scope="session")
def dataset_root(project_root: Path) -> Path:
    """Absolute path to the raw CompCars dataset directory.

    Skips the entire test session if the directory does not exist, so
    integration tests are silently skipped on machines without the dataset.
    """
    root = project_root / "dataset"
    if not root.exists():
        pytest.skip("dataset/ directory not found — skipping all integration tests")
    return root


@pytest.fixture(scope="session")
def dataset_csv(project_root: Path) -> Path:
    """Absolute path to the pre-built ``dataset.csv``.

    Skips if the file does not exist.
    """
    csv = project_root / "dataset.csv"
    if not csv.exists():
        pytest.skip("dataset.csv not found — skipping tests that need the CSV")
    return csv


# --------------------------------------------------------------------------- #
# DataFrame fixtures                                                           #
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="session")
def df(dataset_csv: Path) -> pd.DataFrame:
    """Loaded ``dataset.csv`` as a DataFrame (session-scoped, read once)."""
    return pd.read_csv(dataset_csv)


@pytest.fixture(scope="session")
def df_abs(df: pd.DataFrame, project_root: Path) -> pd.DataFrame:
    """DataFrame with *absolute* ``image_path`` values.

    ``CarDataset`` passes image paths directly to ``PIL.Image.open``.
    Making them absolute means ``CarDataset`` works correctly regardless of
    the working directory when ``pytest`` is invoked.
    """
    out = df.copy()
    out["image_path"] = out["image_path"].apply(lambda p: str(project_root / p))
    return out


# --------------------------------------------------------------------------- #
# Image fixtures                                                               #
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="session")
def sample_image_path(df: pd.DataFrame, project_root: Path) -> Path:
    """Absolute path to the first image in the CSV that actually exists on disk.

    Skips if no valid image is found.
    """
    for _, row in df.iterrows():
        candidate = project_root / row["image_path"]
        if candidate.exists():
            return candidate
    pytest.skip("No valid image file found in dataset.csv")


@pytest.fixture(scope="session")
def sample_image(sample_image_path: Path) -> Image.Image:
    """A real RGB PIL Image read from the dataset (session-scoped).

    The image is fully loaded into memory (via ``.copy()``) so that the file
    handle is released and the fixture remains valid for the whole session.
    """
    with Image.open(sample_image_path) as img:
        return img.convert("RGB").copy()
