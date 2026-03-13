"""
Custom multi-view vehicle dataset with percentage-based train/val/test splits.

This module provides dataset classes for multi-view vehicle classification,
using traditional percentage-based splits and robust handling of long-tail
class distributions.

It also handles the creation of textual labels for multimodal models and
includes an MPerClassSampler wrapper for pytorch-metric-learning.

Public API:
    CarDatasetError: Domain-specific exception.
    CarDataset: Custom PyTorch Dataset with percentage-based splits.
    MPerClassSamplerWrapper: Wrapper for pytorch-metric-learning's MPerClassSampler.
    robust_collate_fn: Collate function that filters out failed samples.
    create_car_dataset: Factory function to build a CarDataset with DataLoaders.
"""

from __future__ import annotations

import copy
import logging
import random
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from PIL import Image
from pytorch_metric_learning.samplers import MPerClassSampler
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset, default_collate

from src.config.TransformConfig import create_standard_transform
from src.defaults import (
    CLASS_GRANULARITY_OPTIONS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_CLASS_GRANULARITY,
    DEFAULT_DESCRIPTION_INCLUDE,
    DEFAULT_K,
    DEFAULT_MIN_IMAGES,
    DEFAULT_MODEL_TYPE,
    DEFAULT_NUM_WORKERS,
    DEFAULT_ONESHOT,
    DEFAULT_ONESHOT_RATIO,
    DEFAULT_P,
    DEFAULT_SEED,
    DEFAULT_TEST_RATIO,
    DEFAULT_TRAIN_RATIO,
    DEFAULT_TRANSFORM,
    DEFAULT_VAL_RATIO,
    DEFAULT_VIEWS,
    DESCRIPTION_OPTIONS,
    ERROR_INVALID_DESCRIPTION,
    ERROR_INVALID_MODEL_TYPE,
    MODEL_TYPES,
)

__all__ = [
    "CarDatasetError",
    "CarDataset",
    "MPerClassSamplerWrapper",
    "robust_collate_fn",
    "create_car_dataset",
]

warnings.filterwarnings('ignore')
logging.getLogger(__name__).addHandler(logging.NullHandler())


class CarDatasetError(Exception):
    """Domain-specific exception for vehicle dataset errors."""
    pass


class CarDataset(Dataset):
    """Custom multi-view vehicle dataset with percentage-based train/val/test splits.

    Manages vehicle datasets with multiple viewpoints using traditional
    percentage-based splits. Long-tail distributions are handled by
    separating *regular* classes (enough images for training) from
    *one-shot* classes (too few images, used only in the test set).

    Class strategy:
        - Regular classes (>= ``min_images``): used in train / val / test.
        - One-shot classes (< ``min_images``): test-only for generalisation evaluation.
        - All classes receive encoded labels (for clustering / evaluation).

    Text-description strategy:
        - Train: detailed description according to ``description_include``.
        - Val / Test: partial description (make + type only) to simulate zero-shot.

    Reproducibility:
        The ``seed`` parameter guarantees identical splits across runs when
        the same configuration is used.

    Attributes:
        df: DataFrame holding the dataset records.
        views: List of viewpoints to include.
        num_views: Number of configured views.
        class_granularity: ``'model'`` or ``'model+year'``.
        min_images: Minimum images required per class for train/val/test.
        train_ratio: Fraction of data allocated to train.
        val_ratio: Fraction of data allocated to validation.
        test_ratio: Fraction of data allocated to test.
        seed: Random seed for reproducibility.
        transform: torchvision transforms applied to images.
        model_type: Output modality (``'vision'``, ``'textual'``, ``'multimodal'``).
        description_include: Text detail level for training descriptions.
        include_oneshot_in_test: Whether to add one-shot classes to the test set.
        regular_classes: Classes with enough images for splits.
        oneshot_classes: Classes with too few images (< ``min_images``).
        label_encoder: Encoder for training classes only.
        label_encoder_full: Encoder for all classes (regular + one-shot).
        train_samples: Training samples (regular classes only).
        val_samples: Validation samples (regular classes only).
        test_samples: Test samples (regular + one-shot when enabled).
        oneshot_samples: One-shot samples (reference only).
        current_split: Active split (``'train'``, ``'val'``, or ``'test'``).
    """
    def __init__(
        self,
        df: pd.DataFrame,
        views: List[str] = DEFAULT_VIEWS,
        class_granularity: str = DEFAULT_CLASS_GRANULARITY,
        min_images: int = DEFAULT_MIN_IMAGES,
        train_ratio: float = DEFAULT_TRAIN_RATIO,
        val_ratio: float = DEFAULT_VAL_RATIO,
        test_ratio: float = DEFAULT_TEST_RATIO,
        seed: int = DEFAULT_SEED,
        transform: Optional[Any] = DEFAULT_TRANSFORM,
        model_type: str = DEFAULT_MODEL_TYPE,
        description_include: str = DEFAULT_DESCRIPTION_INCLUDE,
        include_oneshot_in_test: bool = DEFAULT_ONESHOT,
        oneshot_ratio: Optional[float] = DEFAULT_ONESHOT_RATIO,
    ) -> None:
        """Initialise the vehicle dataset with percentage-based splits.

        Args:
            df: DataFrame with columns: ``model``, ``released_year``,
                ``viewpoint``, ``image_path``, ``make``, ``type``.
            views: Viewpoints to include.
            class_granularity: ``'model'`` (group years) or ``'model+year'`` (split years).
            min_images: Minimum images per class to include in train/val/test.
            train_ratio: Fraction allocated to train.
            val_ratio: Fraction allocated to validation.
            test_ratio: Fraction allocated to test.
            seed: Random seed for reproducibility.
            transform: torchvision transform applied to images.
            model_type: Output modality (``'vision'``, ``'textual'``, ``'multimodal'``).
            description_include: Text detail level: ``''``, ``'make'``, ``'model'``, ``'full'``.
            include_oneshot_in_test: Add one-shot classes to the test set.
            oneshot_ratio: Fraction of one-shot classes to include in test.

        Raises:
            CarDatasetError: On configuration or data errors.
            ValueError: If any parameter value is invalid.
        """
        self._validate_parameters(
            df, views, class_granularity, min_images,
            train_ratio, val_ratio, test_ratio,
            model_type, description_include
        )

        self.df = df.copy()
        self.views = views if views is not None else DEFAULT_VIEWS.copy()
        self.num_views = len(self.views)
        self.seed = seed
        self.class_granularity = class_granularity
        self.min_images = min_images
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.transform = transform
        self.model_type = model_type
        self.description_include = description_include
        self.include_oneshot_in_test = include_oneshot_in_test
        self.oneshot_ratio = oneshot_ratio if oneshot_ratio is not None else 1.0
        self.current_split = None

        total_ratio = train_ratio + val_ratio + test_ratio
        if not np.isclose(total_ratio, 1.0, atol=1e-6):
            raise ValueError(f"Ratios must sum to 1.0, got: {total_ratio}")

        random.seed(self.seed)
        np.random.seed(self.seed)

        logging.info(f"Initialising CarDataset with {len(self.df)} records")
        logging.info(f"Class granularity: {self.class_granularity}")
        logging.info(f"Configured views: {self.views}")
        logging.info(f"Minimum images per class: {self.min_images}")
        logging.info(
            f"Splits: train={train_ratio:.1%}, val={val_ratio:.1%}, test={test_ratio:.1%}"
        )
        logging.info(
            f"One-shot in test: {'yes' if self.include_oneshot_in_test else 'no'}"
        )

        self._identify_classes()
        self.label_encoder = self._initialize_label_encoder()
        self.df = self._filter_dataframe()
        self._create_data_splits()

        logging.info(
            f"Samples — Train: {len(self.train_samples)}, "
            f"Val: {len(self.val_samples)}, Test: {len(self.test_samples)}"
        )
        logging.info(
            f"One-shot classes: {len(self.oneshot_classes)} "
            f"({len(self.oneshot_samples)} samples)"
        )

    def _validate_parameters(
        self,
        df: pd.DataFrame,
        views: Optional[List[str]],
        class_granularity: str,
        min_images: int,
        train_ratio: float,
        val_ratio: float,
        test_ratio: float,
        model_type: str,
        description_include: str,
    ) -> None:
        """Validate all constructor parameters.

        Raises:
            CarDatasetError: On DataFrame or view errors.
            ValueError: If any parameter value is invalid.
        """
        required_columns = {'model', 'released_year', 'viewpoint', 'image_path', 'make'}
        if not required_columns.issubset(df.columns):
            missing = required_columns - set(df.columns)
            raise CarDatasetError(f"Missing columns in DataFrame: {missing}")

        if df.empty:
            raise CarDatasetError("DataFrame must not be empty")

        if class_granularity not in CLASS_GRANULARITY_OPTIONS:
            raise ValueError(f"class_granularity must be one of {CLASS_GRANULARITY_OPTIONS}")

        if min_images < 2:
            raise ValueError("min_images must be >= 2")

        if not (0 < train_ratio < 1):
            raise ValueError("train_ratio must be between 0 and 1")
        if not (0 < val_ratio < 1):
            raise ValueError("val_ratio must be between 0 and 1")
        if not (0 < test_ratio < 1):
            raise ValueError("test_ratio must be between 0 and 1")

        if model_type not in MODEL_TYPES:
            raise ValueError(ERROR_INVALID_MODEL_TYPE.format(MODEL_TYPES))

        if description_include not in DESCRIPTION_OPTIONS:
            raise ValueError(ERROR_INVALID_DESCRIPTION.format(DESCRIPTION_OPTIONS))

        if views is not None:
            if not views or not isinstance(views, list):
                raise ValueError("views must be a non-empty list")
            available_views = set(df['viewpoint'].unique())
            invalid_views = set(views) - available_views
            if invalid_views:
                raise CarDatasetError(
                    f"Views not present in dataset: {invalid_views}. "
                    f"Available: {available_views}"
                )

    def _identify_classes(self) -> None:
        """Partition classes into *regular* (>= ``min_images``) and *one-shot* (< ``min_images``).

        Regular classes are used for train/val/test.
        One-shot classes are reserved for evaluating generalisation in the test set.

        Raises:
            CarDatasetError: If no regular classes can be found.
        """
        logging.info(f"Identifying classes with granularity '{self.class_granularity}'...")

        group_cols = ['model', 'released_year'] if self.class_granularity == 'model+year' else ['model']
        group_cols_with_view = group_cols + ['viewpoint']

        counts = self.df.groupby(group_cols_with_view).size().unstack(fill_value=0)
        available_views = [v for v in self.views if v in counts.columns]

        if not available_views:
            raise CarDatasetError(
                f"None of the specified views exist in the dataset: {self.views}"
            )

        valid_classes = [
            class_key
            for class_key in counts.index
            if all(counts.loc[class_key, available_views] >= 1)
        ]

        logging.info(
            f"Classes with >= 1 image in all views: {len(valid_classes)}"
        )

        self.regular_classes = []
        self.oneshot_classes = []

        for class_key in valid_classes:
            if all(counts.loc[class_key, available_views] >= self.min_images):
                self.regular_classes.append(class_key)
            else:
                self.oneshot_classes.append(class_key)

        logging.info(
            f"Regular classes (>= {self.min_images} imgs): {len(self.regular_classes)}"
        )
        logging.info(
            f"One-shot classes (< {self.min_images} imgs): {len(self.oneshot_classes)}"
        )

        if not self.regular_classes:
            raise CarDatasetError(
                f"No classes with at least {self.min_images} images in all views. "
                "Consider lowering min_images or adding more data."
            )

    def _initialize_label_encoder(self) -> LabelEncoder:
        """Initialise two label encoders.

        1. ``label_encoder_full`` — all classes (regular + one-shot) for evaluation.
        2. ``label_encoder_training`` — regular classes only for training.

        Returns:
            The training-only encoder (also stored as ``self.label_encoder_training``).

        Raises:
            CarDatasetError: If there are no regular classes.
        """
        label_encoder_full = LabelEncoder()
        all_classes = self.regular_classes + self.oneshot_classes

        if self.class_granularity == 'model+year':
            class_strings_full = [f"{model}_{year}" for model, year in all_classes]
        else:
            class_strings_full = list(all_classes)

        label_encoder_full.fit(class_strings_full)

        label_encoder_training = LabelEncoder()

        if self.class_granularity == 'model+year':
            class_strings_training = [
                f"{model}_{year}" for model, year in self.regular_classes
            ]
        else:
            class_strings_training = list(self.regular_classes)

        label_encoder_training.fit(class_strings_training)

        logging.info(
            f"Full LabelEncoder: {len(all_classes)} classes total "
            f"({len(self.regular_classes)} regular + {len(self.oneshot_classes)} one-shot)"
        )
        logging.info(
            f"Training LabelEncoder: {len(self.regular_classes)} regular classes"
        )

        self.label_encoder_full = label_encoder_full
        return label_encoder_training

    def _filter_dataframe(self) -> pd.DataFrame:
        """Filter the DataFrame to include only valid classes and views.

        Returns:
            Filtered DataFrame.

        Raises:
            CarDatasetError: If the DataFrame is empty after filtering.
        """
        initial_size = len(self.df)

        all_valid_classes = set(self.regular_classes + self.oneshot_classes)

        df_filtered = self.df.copy()

        if self.class_granularity == 'model+year':
            df_filtered['class_key'] = list(
                zip(df_filtered['model'], df_filtered['released_year'])
            )
        else:
            df_filtered['class_key'] = df_filtered['model']

        filtered_df = df_filtered[
            (df_filtered['class_key'].isin(all_valid_classes))
            & (df_filtered['viewpoint'].isin(self.views))
        ].drop('class_key', axis=1)

        final_size = len(filtered_df)
        logging.info(f"DataFrame filtered: {initial_size} → {final_size} records")

        return filtered_df

    def _create_text_descriptor(self, image_paths: Union[str, List[str]]) -> str:
        """Build a textual description for one or more image paths.

        Uses ``self.current_split`` to determine the level of detail:

        - **Train**: detailed description controlled by ``self.description_include``.
        - **Val / Test / One-shot**: partial description (make + type only).

        Args:
            image_paths: Image path(s) to describe.

        Returns:
            Text description of the vehicle.
        """
        if isinstance(image_paths, str):
            image_paths = [image_paths]

        try:
            rows = []
            for path in image_paths:
                matching_rows = self.df[self.df['image_path'] == path]
                if matching_rows.empty:
                    raise CarDatasetError(f"Image not found in dataset: {path}")
                rows.append(matching_rows.iloc[0])
        except Exception as exc:
            raise CarDatasetError(
                f"Error retrieving image information: {exc}"
            ) from exc

        make = rows[0]['make'].strip()
        type_str = rows[0].get('type', '').strip()
        model = rows[0]['model'].strip()
        year = str(rows[0]['released_year']).strip()
        viewpoints = [row['viewpoint'] for row in rows]

        if len(viewpoints) == 1:
            view_prefix = f"The {viewpoints[0]} view image of a"
        else:
            viewpoint_text = " and ".join(viewpoints)
            view_prefix = f"The {viewpoint_text} view images of a"

        if self.current_split in ['val', 'test', 'oneshot']:
            if type_str:
                return f"{view_prefix} {make} vehicle type {type_str}."
            return f"{view_prefix} {make} vehicle."

        if self.current_split == 'train':
            if self.description_include == 'full':
                if type_str:
                    return f"{view_prefix} {make} {model} vehicle from {year} type {type_str}."
                return f"{view_prefix} {make} {model} vehicle from {year}."

            if self.description_include == 'model':
                if type_str:
                    return f"{view_prefix} {make} {model} vehicle type {type_str}."
                return f"{view_prefix} {make} {model} vehicle."

            if self.description_include == 'make':
                if type_str:
                    return f"{view_prefix} {make} vehicle type {type_str}."
                return f"{view_prefix} {make} vehicle."

            # description_include == ''
            if type_str:
                return f"{view_prefix} vehicle type {type_str}."
            return f"{view_prefix} vehicle."

        raise ValueError(f"Invalid split: {self.current_split}")

    def _create_data_splits(self) -> None:
        """Create train / val / test splits by percentage.

        Raises:
            CarDatasetError: If the splits cannot be created correctly.
        """
        logging.info("Creating percentage-based splits...")

        all_samples: List = []
        for class_key in self.regular_classes:
            all_samples.extend(self._create_samples_for_class(class_key))

        random.shuffle(all_samples)

        n_total = len(all_samples)
        n_train = int(n_total * self.train_ratio)
        n_val = int(n_total * self.val_ratio)

        self.train_samples = all_samples[:n_train]
        self.val_samples = all_samples[n_train:n_train + n_val]
        self.test_samples = all_samples[n_train + n_val:]

        self.oneshot_samples = []
        for class_key in self.oneshot_classes:
            self.oneshot_samples.extend(self._create_samples_for_class(class_key))

        if self.include_oneshot_in_test and self.oneshot_samples:
            n_oneshot = int(len(self.oneshot_samples) * self.oneshot_ratio)
            selected_oneshot = random.sample(self.oneshot_samples, n_oneshot)
            self.test_samples.extend(selected_oneshot)
            logging.info(f"Added {n_oneshot} one-shot samples to test set")
    
    def _create_samples_for_class(self, class_key: Union[Tuple[str, Any], str]) -> List:
        """Build the list of samples for a single class.

        Args:
            class_key: Class identifier — a ``(model, year)`` tuple for
                ``'model+year'`` granularity, or a plain model string otherwise.

        Returns:
            List of ``(class_key, image_paths)`` sample tuples.
        """
        if self.class_granularity == 'model+year':
            model, year = class_key
            class_data = self.df[
                (self.df['model'] == model)
                & (self.df['released_year'] == year)
                & (self.df['viewpoint'].isin(self.views))
            ]
        else:
            model = class_key
            class_data = self.df[
                (self.df['model'] == model)
                & (self.df['viewpoint'].isin(self.views))
            ]

        grouped = class_data.groupby('viewpoint')
        view_images: Dict[str, List[str]] = {}
        for view in self.views:
            if view in grouped.groups:
                paths = list(grouped.get_group(view)['image_path'])
                random.shuffle(paths)
                view_images[view] = paths
            else:
                view_images[view] = []

        min_images = min(
            (len(view_images[v]) for v in self.views if view_images[v]),
            default=0,
        )
        if min_images == 0:
            return []

        return [
            (class_key, [view_images[v][i] for v in self.views if i < len(view_images[v])])
            for i in range(min_images)
            if len([v for v in self.views if i < len(view_images[v])]) == len(self.views)
        ]

    def set_split(self, split: str) -> None:
        """Activate a dataset split.

        Args:
            split: One of ``'train'``, ``'val'``, or ``'test'``.

        Raises:
            ValueError: If ``split`` is not a valid split name.
        """
        if split not in ('train', 'val', 'test'):
            raise ValueError(
                f"split must be 'train', 'val' or 'test', got: '{split}'"
            )
        self.current_split = split
        logging.debug(f"Dataset switched to '{split}' split")

    def get_num_classes_for_training(self) -> int:
        """Return the number of *regular* classes used during training.

        This is the value that should be used for the model's output-layer
        dimension, since only regular classes have enough samples for
        train / val splits.

        Returns:
            Number of regular classes.
        """
        return len(self.regular_classes)

    def get_total_num_classes(self) -> int:
        """Return the total number of classes (regular + one-shot).

        Useful for clustering evaluation metrics that consider every class
        in the dataset.

        Returns:
            Total number of classes.
        """
        return len(self.regular_classes) + len(self.oneshot_classes)

    def get_label_encoder_for_evaluation(self) -> LabelEncoder:
        """Return the full label encoder (all classes) for evaluation / clustering.

        Use this encoder when evaluating embeddings or running clustering that
        includes one-shot classes.

        Returns:
            :class:`LabelEncoder` fitted on all classes (regular + one-shot).
        """
        return self.label_encoder_full

    def get_current_samples(self) -> List:
        """Return the samples for the currently active split.

        Returns:
            List of samples for the active split.

        Raises:
            ValueError: If no split has been activated yet.
        """
        if self.current_split == 'train':
            return self.train_samples
        elif self.current_split == 'val':
            return self.val_samples
        elif self.current_split == 'test':
            return self.test_samples
        else:
            raise ValueError(f"Invalid split: '{self.current_split}'")

    def __len__(self) -> int:
        """Return the number of samples in the active split."""
        return len(self.get_current_samples())

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Return a processed sample from the active split.

        Args:
            idx: Sample index.

        Returns:
            Dict with keys ``images``, ``labels``, ``class_key``,
            ``image_paths``, and optionally ``text_description``.
        """
        try:
            return self._load_sample(idx)
        except Exception as exc:
            logging.error(f"Error loading sample {idx}: {exc}")
            return None

    def _load_sample(self, idx: int) -> Dict[str, Any]:
        """Load and process a single sample from the active split.

        Args:
            idx: Sample index.

        Returns:
            Processed sample dict.

        Raises:
            ValueError: If a one-shot class is found outside the test split.
        """
        samples = self.get_current_samples()
        class_key, image_paths = samples[idx]

        if self.class_granularity == 'model+year':
            model, year = class_key
            label_str = f"{model}_{year}"
        else:
            label_str = class_key

        # Encode label: regular classes use the training encoder;
        # one-shot classes in the test set use the full encoder.
        if class_key in self.regular_classes:
            label = self.label_encoder.transform([label_str])[0]
        elif self.current_split == 'test' and self.include_oneshot_in_test:
            label = self.label_encoder_full.transform([label_str])[0]
        else:
            raise ValueError(
                f"One-shot class {class_key} encountered outside the test split"
            )

        images = []
        for img_path in image_paths:
            try:
                with Image.open(img_path) as img:
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    if self.transform:
                        img = self.transform(img)
                    images.append(img)
            except Exception as exc:
                logging.error(f"Error loading image {img_path}: {exc}")
                raise

        images_tensor = images[0] if len(images) == 1 else torch.stack(images)

        sample: Dict[str, Any] = {
            'images': images_tensor,
            'labels': torch.tensor(label, dtype=torch.long),
            'class_key': class_key,
            'image_paths': image_paths,
        }

        if self.model_type in ('textual', 'multimodal'):
            sample['text_description'] = self._create_text_descriptor(image_paths)

        return sample

    def get_dataset_statistics(self) -> Dict[str, Any]:
        """Return a statistical summary of the dataset.

        Returns:
            Dict with sample counts, class counts, split information,
            and description strategy details.
        """
        stats = {
            'total_samples': len(self.train_samples) + len(self.val_samples) + len(self.test_samples),
            'num_training_classes': len(self.regular_classes),   # output-layer size
            'num_total_classes': len(self.regular_classes) + len(self.oneshot_classes),  # for evaluation
            'num_regular_classes': len(self.regular_classes),
            'num_oneshot_classes': len(self.oneshot_classes),
            'num_views': self.num_views,
            'views': self.views,
            'class_granularity': self.class_granularity,
            'min_images': self.min_images,
            'seed': self.seed,
            'description_strategy': {
                'train': f'detailed ({self.description_include})',
                'val': 'partial (make + type)',
                'test': 'partial (make + type)'
            },
            'splits': {
                'train': {
                    'samples': len(self.train_samples),
                    'ratio': self.train_ratio,
                    'classes': 'regular only'
                },
                'val': {
                    'samples': len(self.val_samples),
                    'ratio': self.val_ratio,
                    'classes': 'regular only'
                },
                'test': {
                    'samples': len(self.test_samples),
                    'ratio': self.test_ratio,
                    'includes_oneshot': self.include_oneshot_in_test,
                    'oneshot_samples': len(self.oneshot_samples) if self.include_oneshot_in_test else 0,
                    'classes': 'regular + oneshot' if self.include_oneshot_in_test else 'regular only'
                }
            }
        }
        
        return stats

    def __str__(self) -> str:
        """Return a human-readable summary of the dataset."""
        stats = self.get_dataset_statistics()
        lines = []
        lines.append("="*70)
        lines.append(f"CarDataset Summary")
        lines.append("="*70)
        lines.append(f"Classes: {stats['num_training_classes']} training | "
                    f"{stats['num_total_classes']} total (+ {stats['num_oneshot_classes']} oneshot)")
        lines.append(f"Granularity: {stats['class_granularity']}")
        lines.append(f"Views: {stats['views']} (n={stats['num_views']})")
        lines.append(f"Min images per class: {stats['min_images']}")
        lines.append(f"Seed: {stats['seed']} (reproducible)")
        lines.append(f"\nText Descriptions:")
        lines.append(f"  Train: {stats['description_strategy']['train']}")
        lines.append(f"  Val:   {stats['description_strategy']['val']}")
        lines.append(f"  Test:  {stats['description_strategy']['test']}")
        lines.append(f"\nData Splits:")
        lines.append(f"  Train:  {stats['splits']['train']['samples']:6d} samples "
                    f"({stats['splits']['train']['ratio']:.1%}) - {stats['splits']['train']['classes']}")
        lines.append(f"  Val:    {stats['splits']['val']['samples']:6d} samples "
                    f"({stats['splits']['val']['ratio']:.1%}) - {stats['splits']['val']['classes']}")
        lines.append(f"  Test:   {stats['splits']['test']['samples']:6d} samples "
                    f"({stats['splits']['test']['ratio']:.1%}) - {stats['splits']['test']['classes']}")
        if stats['splits']['test']['includes_oneshot']:
            lines.append(f"    └─ includes {stats['splits']['test']['oneshot_samples']} one-shot samples")
        lines.append("="*70)
        return "\n".join(lines)

    def __repr__(self) -> str:
        """Return a concise string representation for debugging."""
        return (
            f"CarDataset(regular_classes={len(self.regular_classes)}, "
            f"oneshot_classes={len(self.oneshot_classes)}, "
            f"views={len(self.views)})"
        )


class MPerClassSamplerWrapper:
    """Wrap :class:`pytorch_metric_learning.samplers.MPerClassSampler` for use with :class:`CarDataset`.

    The original sampler requires a flat list of labels; this wrapper derives
    that list automatically from the ``(class_key, image_paths)`` sample tuples
    produced by :class:`CarDataset`.
    """

    def __init__(
        self,
        samples: List[Tuple],
        m: int = DEFAULT_K,
        batch_size: Optional[int] = None,
        length_before_new_iter: Optional[int] = None,
    ):
        """Initialise the sampler wrapper.

        Args:
            samples: List of ``(class_key, image_paths)`` tuples from
                :class:`CarDataset`.
            m: Number of samples per class per batch (equivalent to *K*).
            batch_size: Total batch size. Defaults to
                ``min(4, num_classes) * m`` when ``None``.
            length_before_new_iter: Dataset length before restarting the
                iterator. Defaults to ``len(samples)`` when ``None``.
        """
        self.samples = samples
        self.labels: List[int] = []
        self.class_key_to_label: Dict = {}

        current_label = 0
        for sample in samples:
            class_key = sample[0]
            if class_key not in self.class_key_to_label:
                self.class_key_to_label[class_key] = current_label
                current_label += 1
            self.labels.append(self.class_key_to_label[class_key])

        num_classes = len(set(self.labels))
        if batch_size is None:
            batch_size = min(4, num_classes) * m
        if length_before_new_iter is None:
            length_before_new_iter = len(self.labels)

        self.sampler = MPerClassSampler(
            labels=self.labels,
            m=m,
            batch_size=batch_size,
            length_before_new_iter=length_before_new_iter,
        )

        logging.info(
            f"MPerClassSamplerWrapper: {num_classes} classes, "
            f"{len(self.labels)} samples, m={m}, batch_size={batch_size}"
        )

    def __iter__(self):
        """Yield individual sample indices."""
        return iter(self.sampler)

    def __len__(self) -> int:
        return len(self.sampler)


def robust_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate function that silently drops failed (``None``) samples.

    Args:
        batch: List of samples returned by :meth:`CarDataset.__getitem__`.

    Returns:
        Collated batch with invalid samples removed.

    Raises:
        RuntimeError: If every sample in the batch failed to load.
    """
    valid_batch = [s for s in batch if s is not None]

    if not valid_batch:
        raise RuntimeError("All samples in the batch failed to load")

    if len(valid_batch) < len(batch):
        logging.warning(
            f"Dropped {len(batch) - len(valid_batch)} invalid sample(s) from batch"
        )

    return default_collate(valid_batch)

def create_car_dataset(
    df: pd.DataFrame,
    views: List[str] = DEFAULT_VIEWS,
    min_images: int = DEFAULT_MIN_IMAGES,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    val_ratio: float = DEFAULT_VAL_RATIO,
    test_ratio: float = DEFAULT_TEST_RATIO,
    train_transform=None,
    val_transform=None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_workers: int = DEFAULT_NUM_WORKERS,
    seed: int = DEFAULT_SEED,
    sampling_strategy: str = 'standard',
    P: int = DEFAULT_P,
    K: int = DEFAULT_K,
    **dataset_kwargs,
) -> Dict[str, Any]:
    """Create a :class:`CarDataset` with percentage-based splits and DataLoaders.

    Args:
        df: DataFrame with dataset records.
        views: Viewpoints to include.
        min_images: Minimum images per class for regular classes.
        train_ratio: Fraction allocated to training.
        val_ratio: Fraction allocated to validation.
        test_ratio: Fraction allocated to test.
        train_transform: Transform applied to training images. Defaults to
            augmented :func:`create_standard_transform`.
        val_transform: Transform applied to val / test images. Defaults to
            non-augmented :func:`create_standard_transform`.
        batch_size: DataLoader batch size.
        num_workers: Number of DataLoader worker processes.
        seed: Random seed.
        sampling_strategy: ``'standard'`` (shuffle) or ``'pk'`` (P classes × K
            samples per batch, for metric learning).
        P: Classes per batch — used only with ``'pk'`` strategy.
        K: Samples per class — used only with ``'pk'`` strategy.
        **dataset_kwargs: Extra keyword arguments forwarded to :class:`CarDataset`.

    Returns:
        Dict with keys ``'dataset'``, ``'train_loader'``, ``'val_loader'``,
        ``'test_loader'``, and ``'train_sampler'``.

    Raises:
        ValueError: If ``sampling_strategy`` is not ``'standard'`` or ``'pk'``.
    """
    valid_strategies = ('standard', 'pk')
    if sampling_strategy not in valid_strategies:
        raise ValueError(f"sampling_strategy must be one of {valid_strategies}")
    
    base_dataset = CarDataset(
        df=df,
        views=views,
        min_images=min_images,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        transform=None,
        seed=seed,
        **dataset_kwargs
    )

    if train_transform is None:
        train_transform = create_standard_transform(augment=True)
    if val_transform is None:
        val_transform = create_standard_transform(augment=False)

    train_dataset = copy.deepcopy(base_dataset)
    train_dataset.transform = train_transform
    train_dataset.set_split('train')

    if sampling_strategy == 'standard':
        train_sampler = None
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=robust_collate_fn,
        )
        logging.info(f"Train loader: standard shuffle, batch_size={batch_size}")

    else:  # 'pk'
        train_sampler = MPerClassSamplerWrapper(
            base_dataset.train_samples,
            m=K,
            batch_size=P * K,
            length_before_new_iter=len(base_dataset.train_samples),
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=P * K,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=robust_collate_fn,
        )
        logging.info(
            f"Train loader: P×K sampling (MPerClassSampler), "
            f"P={P}, K={K}, batch_size={P * K}"
        )

    val_dataset = copy.deepcopy(base_dataset)
    val_dataset.transform = val_transform
    val_dataset.set_split('val')

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=robust_collate_fn,
    )

    test_dataset = copy.deepcopy(base_dataset)
    test_dataset.transform = val_transform
    test_dataset.set_split('test')

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=robust_collate_fn,
    )

    logging.info("Transforms — train: with augmentation | val/test: without augmentation")
    
    return {
        "dataset": base_dataset,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "train_sampler": train_sampler
    }
