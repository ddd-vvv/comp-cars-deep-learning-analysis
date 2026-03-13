"""
CSV dataset builder for the CompCars dataset.

This module provides a class to construct and save a structured dataset from
the Comprehensive Cars (CompCars) dataset, processing images, metadata, and
labels to produce a clean CSV file.

Public API:
    CompCarsDatasetError: Domain-specific exception.
    DataFrameMaker: Builds the CompCars CSV dataset.
    create_compcars_dataset: Convenience factory function.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union
import warnings

import pandas as pd
import scipy.io

from src.defaults import (
    ATTRIBUTES_FILE,
    CAR_TYPE_FILE,
    DEFAULT_COLUMNS_TO_KEEP,
    DEFAULT_VIEWPOINT_FILTER,
    MAKE_MODEL_FILE,
    UNKNOWN_TYPE,
    VIEWPOINT_MAPPING,
)

__all__ = ["CompCarsDatasetError", "DataFrameMaker", "create_compcars_dataset"]

warnings.filterwarnings('ignore')
logging.getLogger(__name__).addHandler(logging.NullHandler())


class CompCarsDatasetError(Exception):
    """Domain-specific exception for CompCars dataset errors."""
    pass


class DataFrameMaker:
    """Build a structured CSV dataset from the raw CompCars dataset.

    Processes vehicle images, metadata, and bounding-box annotations to
    produce a clean pandas DataFrame that can be saved as a CSV file.

    Attributes:
        base_path: Root directory of the CompCars dataset.
        images_folder: Sub-directory containing the images.
        labels_folder: Sub-directory containing the label annotations.
        misc_folder: Sub-directory containing metadata files.
        attributes_df: DataFrame with per-model vehicle attributes.
        make_mapping: Dict mapping make IDs to make names.
        model_mapping: Dict mapping model IDs to model names.
        type_mapping: Dict mapping type IDs to type names.
        dataset_df: Final processed DataFrame.

    Example:
        >>> maker = DataFrameMaker("/path/to/CompCarsDataset")
        >>> maker.load_metadata()
        >>> dataset = maker.build_dataset()
        >>> maker.save_dataset("output.csv")

    Raises:
        CompCarsDatasetError: For dataset-specific errors.
        FileNotFoundError: When required files or directories are missing.
    """
    
    def __init__(self, base_path: Union[str, Path]) -> None:
        """Initialise the dataset builder.

        Args:
            base_path: Root directory of the CompCars dataset.

        Raises:
            CompCarsDatasetError: If ``base_path`` does not exist.
        """
        self.base_path = Path(base_path)
        if not self.base_path.exists():
            raise CompCarsDatasetError(f"Base path does not exist: {self.base_path}")

        self.images_folder = self.base_path / 'image'
        self.labels_folder = self.base_path / 'label'
        self.misc_folder = self.base_path / 'misc'

        self._validate_folder_structure()

        self.attributes_df: Optional[pd.DataFrame] = None
        self.make_mapping: Optional[Dict[str, str]] = None
        self.model_mapping: Optional[Dict[str, str]] = None
        self.type_mapping: Optional[Dict[str, str]] = None
        self.dataset_df: Optional[pd.DataFrame] = None

        logging.info(f"DataFrameMaker initialised for: {self.base_path}")

    def _validate_folder_structure(self) -> None:
        """Check that all required dataset sub-directories exist.

        Raises:
            CompCarsDatasetError: If any required folder is missing.
        """
        required_folders = [self.images_folder, self.labels_folder, self.misc_folder]
        missing_folders = [f for f in required_folders if not f.exists()]

        if missing_folders:
            missing_names = [f.name for f in missing_folders]
            raise CompCarsDatasetError(
                f"Missing required dataset folders: {missing_names}"
            )

    def _load_metadata(self) -> None:
        """Load the metadata files needed to build the dataset.

        Reads the vehicle attributes CSV and the make/model/type mappings
        from the ``misc`` directory.

        Raises:
            CompCarsDatasetError: If loading fails for any reason.
            FileNotFoundError: If required metadata files are absent.
        """
        try:
            logging.info("Loading dataset metadata...")

            attributes_path = self.misc_folder / ATTRIBUTES_FILE
            if not attributes_path.exists():
                raise FileNotFoundError(f"Attributes file not found: {attributes_path}")

            self.attributes_df = pd.read_csv(attributes_path, sep=' ')
            self.attributes_df['model_id'] = self.attributes_df['model_id'].astype(str)

            self._load_mat_mappings()

            logging.info("Metadata loaded successfully")

        except Exception as exc:
            raise CompCarsDatasetError(f"Error loading metadata: {exc}") from exc

    def _load_mat_mappings(self) -> None:
        """Load make, model, and type name mappings from .mat files.

        Raises:
            FileNotFoundError: If any required .mat file is missing.
        """
        make_model_path = self.misc_folder / MAKE_MODEL_FILE
        car_type_path = self.misc_folder / CAR_TYPE_FILE

        if not make_model_path.exists():
            raise FileNotFoundError(f"File not found: {make_model_path}")

        if not car_type_path.exists():
            raise FileNotFoundError(f"File not found: {car_type_path}")

        make_model_mat = scipy.io.loadmat(make_model_path)
        car_type_mat = scipy.io.loadmat(car_type_path)

        make_names = self._extract_mat_strings(make_model_mat['make_names'])
        model_names = self._extract_mat_strings(make_model_mat['model_names'], allow_empty=True)
        car_types = [str(x[0]) for x in car_type_mat['types'][0]]

        self.make_mapping = {str(i + 1): name for i, name in enumerate(make_names)}
        self.model_mapping = {str(i + 1): name for i, name in enumerate(model_names)}
        self.type_mapping = {str(i + 1): name for i, name in enumerate(car_types)}
        self.type_mapping['0'] = UNKNOWN_TYPE

    def _extract_mat_strings(self, mat_array: Any, allow_empty: bool = False) -> List[str]:
        """Extract strings from a MATLAB cell array.

        Args:
            mat_array: MATLAB array containing string cells.
            allow_empty: If ``True``, preserve empty strings instead of
                substituting ``'Unknown'``.

        Returns:
            List of extracted strings.
        """
        result = []
        for x in mat_array:
            if len(x) > 0 and len(x[0]) > 0:
                result.append(str(x[0][0]))
            elif allow_empty:
                result.append('')
            else:
                result.append('Unknown')
        return result

    def _process_image_data(
        self,
        image_path: Path,
        viewpoint_filter: Set[int] = DEFAULT_VIEWPOINT_FILTER
    ) -> Optional[Dict[str, Any]]:
        """Process metadata for a single image.

        Args:
            image_path: Path to the image file.
            viewpoint_filter: Set of allowed viewpoint IDs.

        Returns:
            Dict with the image data, or ``None`` if the image should be
            filtered out.

        Raises:
            CompCarsDatasetError: If a critical processing error occurs.
        """
        try:
            label_path = self._get_label_path(image_path)
            if not label_path.exists():
                logging.warning(f"Label file not found: {label_path}")
                return None

            relative_path = image_path.relative_to(self.images_folder)
            path_parts = relative_path.parts

            if len(path_parts) != 4:  # expected: make_id/model_id/year/image_name
                logging.warning(f"Unexpected path structure: {relative_path}")
                return None

            make_id, model_id, released_year, image_name = path_parts

            viewpoint, bbox = self._parse_label_file(label_path)

            if viewpoint not in viewpoint_filter:
                return None

            return {
                'image_name': image_name.replace('.jpg', ''),
                'image_path': str(image_path),
                'make_id': make_id,
                'model_id': model_id,
                'released_year': released_year,
                'viewpoint': viewpoint,
                'bbox': bbox,
            }

        except Exception as exc:
            logging.warning(f"Error processing image {image_path}: {exc}")
            return None

    def _get_label_path(self, image_path: Path) -> Path:
        """Return the label file path that corresponds to an image path.

        Args:
            image_path: Path to the image file.

        Returns:
            Path to the corresponding ``.txt`` label file.
        """
        return Path(
            str(image_path)
            .replace(str(self.images_folder), str(self.labels_folder))
            .replace('.jpg', '.txt')
        )

    def _parse_label_file(self, label_path: Path) -> tuple[int, List[float]]:
        """Parse a label file and return the viewpoint and bounding box.

        Args:
            label_path: Path to the ``.txt`` label file.

        Returns:
            Tuple of ``(viewpoint_id, bbox_coordinates)``.
        """
        with open(label_path, 'r', encoding='utf-8') as f:
            viewpoint = int(f.readline().strip())
            f.readline()  
            bb_coords = f.readline().strip()

        bbox = [float(coord) for coord in bb_coords.split()]
        return viewpoint, bbox

    def build_dataset(
        self,
        viewpoint_filter: Set[int] = DEFAULT_VIEWPOINT_FILTER,
        columns_to_keep: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Build the full dataset by processing every image in the dataset.

        Args:
            viewpoint_filter: Set of viewpoint IDs to include.
            columns_to_keep: Columns to retain in the final DataFrame.

        Returns:
            Processed DataFrame.

        Raises:
            CompCarsDatasetError: If building fails or no valid images are found.
        """
        if self.attributes_df is None:
            self._load_metadata()

        if columns_to_keep is None:
            columns_to_keep = DEFAULT_COLUMNS_TO_KEEP

        logging.info("Building dataset...")

        dataset_list = []

        for image_path in self.images_folder.rglob('*.jpg'):
            data = self._process_image_data(image_path, viewpoint_filter)
            if data:
                dataset_list.append(data)

        if not dataset_list:
            raise CompCarsDatasetError("No valid images found in the dataset")

        self.dataset_df = pd.DataFrame(dataset_list)
        self._apply_mappings()
        self.dataset_df = self.dataset_df[columns_to_keep]
        self._log_dataset_stats()

        logging.info(f"Dataset built successfully with {len(self.dataset_df)} records")
        return self.dataset_df

    def _apply_mappings(self) -> None:
        """Map numeric IDs to human-readable names in ``dataset_df``.

        Raises:
            CompCarsDatasetError: If the mappings have not been loaded yet.
        """
        self.dataset_df['viewpoint'] = self.dataset_df['viewpoint'].map(VIEWPOINT_MAPPING)

        self.dataset_df = self.dataset_df.merge(
            self.attributes_df, on='model_id', how='left'
        )

        self.dataset_df['make'] = self.dataset_df['make_id'].map(self.make_mapping)
        self.dataset_df['model'] = self.dataset_df['model_id'].map(self.model_mapping)
        self.dataset_df['type'] = (
            self.dataset_df['type']
            .astype(int)
            .astype(str)
            .map(self.type_mapping)
        )

    def _log_dataset_stats(self) -> None:
        """Log summary statistics for the built dataset."""
        logging.info(f"Final dataset: {len(self.dataset_df)} images")

        viewpoint_counts = self.dataset_df['viewpoint'].value_counts().to_dict()
        logging.info(f"Viewpoint distribution: {viewpoint_counts}")

        type_counts = self.dataset_df['type'].value_counts().head(5).to_dict()
        logging.info(f"Top-5 vehicle types: {type_counts}")

    def save_dataset(self, output_path: Union[str, Path], **kwargs) -> None:
        """Save the dataset to a CSV file.

        Args:
            output_path: Destination path for the CSV file.
            **kwargs: Extra keyword arguments forwarded to ``pandas.DataFrame.to_csv()``.

        Raises:
            CompCarsDatasetError: If no dataset has been built yet.
        """
        if self.dataset_df is None:
            raise CompCarsDatasetError(
                "No dataset to save. Call build_dataset() first."
            )

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        csv_params = {'index': False, 'encoding': 'utf-8'}
        csv_params.update(kwargs)

        self.dataset_df.to_csv(output_path, **csv_params)
        logging.info(f"Dataset saved to: {output_path}")

    def get_dataset_summary(self) -> Dict[str, Any]:
        """Return a statistical summary of the built dataset.

        Returns:
            Dict with summary statistics.

        Raises:
            CompCarsDatasetError: If no dataset has been built yet.
        """
        if self.dataset_df is None:
            raise CompCarsDatasetError(
                "No dataset built. Call build_dataset() first."
            )

        return {
            'total_images': len(self.dataset_df),
            'viewpoint_distribution': self.dataset_df['viewpoint'].value_counts().to_dict(),
            'make_distribution': self.dataset_df['make'].value_counts().head(10).to_dict(),
            'type_distribution': self.dataset_df['type'].value_counts().to_dict(),
            'year_range': (
                self.dataset_df['released_year'].min(),
                self.dataset_df['released_year'].max(),
            ),
            'columns': list(self.dataset_df.columns),
        }

    def __repr__(self) -> str:
        """Return an unambiguous string representation for debugging."""
        status = "built" if self.dataset_df is not None else "not built"
        size = len(self.dataset_df) if self.dataset_df is not None else 0
        return f"DataFrameMaker(base_path={self.base_path}, dataset={status}, size={size})"


def create_compcars_dataset(
    base_path: Union[str, Path],
    output_path: Union[str, Path],
    viewpoint_filter: Set[int] = DEFAULT_VIEWPOINT_FILTER,
) -> pd.DataFrame:
    """Convenience function to build a full CompCars CSV dataset in one call.

    Args:
        base_path: Root directory of the CompCars dataset.
        output_path: Destination path for the output CSV.
        viewpoint_filter: Viewpoint IDs to include.

    Returns:
        Processed DataFrame.
    """
    maker = DataFrameMaker(base_path)
    dataset = maker.build_dataset(viewpoint_filter=viewpoint_filter)
    maker.save_dataset(output_path)
    return dataset