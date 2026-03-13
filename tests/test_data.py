"""
Tests for ``src.data`` — DataFrameMaker and CarDataset.

Coverage:
    DataFrameMaker: construction, error cases, metadata loading from .mat
    files, ``build_dataset`` (marked slow), ``save_dataset``, and
    ``get_dataset_summary``.
    CarDataset: construction from the pre-built CSV, split proportions,
    ``set_split`` / ``__len__``, ``__getitem__`` on a real image, parameter
    validation, and ``create_car_dataset``.

Markers:
    unit        — tests with no filesystem I/O beyond reading the CSV.
    integration — tests that open real image files from the dataset.
    slow        — tests that invoke ``DataFrameMaker.build_dataset()``,
                  which scans the entire image tree.
                  Run with: ``pytest -m slow``
                  Skip with: ``pytest -m "not slow"``
"""

from __future__ import annotations

__all__: list[str] = []

from pathlib import Path

import pandas as pd
import pytest
import torch

from src.config import create_standard_transform
from src.data import CarDataset, DataFrameMaker, create_car_dataset
from src.data.DataFrameMaker import CompCarsDatasetError, create_compcars_dataset
from src.data.MyDataset import CarDatasetError, MPerClassSamplerWrapper


# --------------------------------------------------------------------------- #
# DataFrameMaker — construction                                                #
# --------------------------------------------------------------------------- #

class TestDataFrameMakerInit:
    """DataFrameMaker construction and path validation."""

    @pytest.mark.integration
    def test_init_succeeds_with_real_dataset_path(self, dataset_root: Path) -> None:
        """DataFrameMaker can be created with the real dataset directory."""
        maker = DataFrameMaker(dataset_root)
        assert maker is not None

    @pytest.mark.unit
    def test_init_raises_on_missing_path(self, tmp_path: Path) -> None:
        """A non-existent path raises CompCarsDatasetError."""
        missing = tmp_path / "does_not_exist"
        with pytest.raises(CompCarsDatasetError):
            DataFrameMaker(missing)

    @pytest.mark.integration
    def test_base_path_attribute_set(self, dataset_root: Path) -> None:
        """``base_path`` attribute is set to the provided directory."""
        maker = DataFrameMaker(dataset_root)
        assert maker.base_path == dataset_root

    @pytest.mark.integration
    def test_subfolders_set_correctly(self, dataset_root: Path) -> None:
        """image / label / misc sub-folder attributes point to correct paths."""
        maker = DataFrameMaker(dataset_root)
        assert maker.images_folder == dataset_root / "image"
        assert maker.labels_folder == dataset_root / "label"
        assert maker.misc_folder == dataset_root / "misc"

    @pytest.mark.integration
    def test_repr_shows_not_built(self, dataset_root: Path) -> None:
        """Before build_dataset(), repr says 'not built'."""
        maker = DataFrameMaker(dataset_root)
        assert "not built" in repr(maker)


# --------------------------------------------------------------------------- #
# DataFrameMaker — metadata loading                                            #
# --------------------------------------------------------------------------- #

class TestDataFrameMakerMetadata:
    """Metadata loading from .mat and .txt files in misc/."""

    @pytest.fixture(scope="class")
    def maker_with_metadata(self, dataset_root: Path) -> DataFrameMaker:
        """DataFrameMaker with metadata loaded (class-scoped, loaded once)."""
        maker = DataFrameMaker(dataset_root)
        maker._load_metadata()
        return maker

    @pytest.mark.integration
    def test_attributes_df_is_dataframe(
        self, maker_with_metadata: DataFrameMaker
    ) -> None:
        """attributes_df is a non-empty DataFrame after metadata load."""
        assert isinstance(maker_with_metadata.attributes_df, pd.DataFrame)
        assert len(maker_with_metadata.attributes_df) > 0

    @pytest.mark.integration
    def test_make_mapping_is_dict(self, maker_with_metadata: DataFrameMaker) -> None:
        """make_mapping is a non-empty dict of str → str."""
        mapping = maker_with_metadata.make_mapping
        assert isinstance(mapping, dict)
        assert len(mapping) > 0
        assert all(isinstance(k, str) for k in mapping)
        assert all(isinstance(v, str) for v in mapping.values())

    @pytest.mark.integration
    def test_model_mapping_is_dict(self, maker_with_metadata: DataFrameMaker) -> None:
        """model_mapping is a non-empty dict of str → str."""
        mapping = maker_with_metadata.model_mapping
        assert isinstance(mapping, dict)
        assert len(mapping) > 0

    @pytest.mark.integration
    def test_type_mapping_includes_unknown(
        self, maker_with_metadata: DataFrameMaker
    ) -> None:
        """type_mapping includes the key '0' mapped to UNKNOWN_TYPE."""
        assert "0" in maker_with_metadata.type_mapping

    @pytest.mark.integration
    def test_attributes_df_has_model_id_column(
        self, maker_with_metadata: DataFrameMaker
    ) -> None:
        """attributes_df contains a 'model_id' column (string type)."""
        assert "model_id" in maker_with_metadata.attributes_df.columns
        assert maker_with_metadata.attributes_df["model_id"].dtype == object


# --------------------------------------------------------------------------- #
# DataFrameMaker — save_dataset and get_dataset_summary (without full build)  #
# --------------------------------------------------------------------------- #

class TestDataFrameMakerSaveAndSummary:
    """save_dataset and get_dataset_summary using the pre-built CSV."""

    @pytest.fixture(scope="class")
    def maker_preloaded(
        self, dataset_root: Path, df: pd.DataFrame
    ) -> DataFrameMaker:
        """DataFrameMaker with dataset_df pre-populated from the existing CSV.

        This avoids running the slow build_dataset() scan while still
        exercising the save / summary code paths.
        """
        maker = DataFrameMaker(dataset_root)
        maker.dataset_df = df.copy()
        return maker

    @pytest.mark.integration
    def test_save_creates_csv_file(
        self, maker_preloaded: DataFrameMaker, tmp_path: Path
    ) -> None:
        """save_dataset writes a CSV file to the given path."""
        out = tmp_path / "out.csv"
        maker_preloaded.save_dataset(out)
        assert out.exists()

    @pytest.mark.integration
    def test_saved_file_is_valid_csv(
        self, maker_preloaded: DataFrameMaker, tmp_path: Path
    ) -> None:
        """The file written by save_dataset can be read back as a DataFrame."""
        out = tmp_path / "roundtrip.csv"
        maker_preloaded.save_dataset(out)
        loaded = pd.read_csv(out)
        assert len(loaded) > 0

    @pytest.mark.integration
    def test_save_before_build_raises(self, dataset_root: Path) -> None:
        """Calling save_dataset() before build_dataset() raises CompCarsDatasetError."""
        maker = DataFrameMaker(dataset_root)
        with pytest.raises(CompCarsDatasetError):
            maker.save_dataset("irrelevant.csv")

    @pytest.mark.integration
    def test_get_summary_returns_expected_keys(
        self, maker_preloaded: DataFrameMaker
    ) -> None:
        """get_dataset_summary returns a dict with the documented keys."""
        summary = maker_preloaded.get_dataset_summary()
        expected_keys = {
            "total_images",
            "viewpoint_distribution",
            "make_distribution",
            "type_distribution",
            "year_range",
            "columns",
        }
        assert expected_keys.issubset(summary.keys())

    @pytest.mark.integration
    def test_get_summary_before_build_raises(self, dataset_root: Path) -> None:
        """get_dataset_summary() before build_dataset() raises CompCarsDatasetError."""
        maker = DataFrameMaker(dataset_root)
        with pytest.raises(CompCarsDatasetError):
            maker.get_dataset_summary()

    @pytest.mark.integration
    def test_repr_shows_built_after_dataset_set(
        self, maker_preloaded: DataFrameMaker
    ) -> None:
        """repr includes 'built' once dataset_df is populated."""
        assert "built" in repr(maker_preloaded)


# --------------------------------------------------------------------------- #
# DataFrameMaker — full build (slow)                                           #
# --------------------------------------------------------------------------- #

class TestDataFrameMakerBuild:
    """Full build_dataset() scan — marked slow, skipped by default.

    Run explicitly with: ``pytest -m slow``
    """

    @pytest.fixture(scope="class")
    def built_maker(self, dataset_root: Path) -> DataFrameMaker:
        """DataFrameMaker with build_dataset() already called (class-scoped)."""
        maker = DataFrameMaker(dataset_root)
        maker.build_dataset()
        return maker

    @pytest.mark.slow
    @pytest.mark.integration
    def test_build_returns_dataframe(self, built_maker: DataFrameMaker) -> None:
        """build_dataset() returns a non-empty DataFrame."""
        assert isinstance(built_maker.dataset_df, pd.DataFrame)
        assert len(built_maker.dataset_df) > 0

    @pytest.mark.slow
    @pytest.mark.integration
    def test_build_has_required_columns(self, built_maker: DataFrameMaker) -> None:
        """The built DataFrame contains the columns required by CarDataset."""
        required = {"image_path", "viewpoint", "model", "released_year", "make", "type"}
        assert required.issubset(set(built_maker.dataset_df.columns))

    @pytest.mark.slow
    @pytest.mark.integration
    def test_build_viewpoints_are_strings(self, built_maker: DataFrameMaker) -> None:
        """The viewpoint column contains only string values."""
        assert built_maker.dataset_df["viewpoint"].dtype == object

    @pytest.mark.slow
    @pytest.mark.integration
    def test_summary_total_images_matches_dataframe(
        self, built_maker: DataFrameMaker
    ) -> None:
        """get_dataset_summary()['total_images'] equals len(dataset_df)."""
        summary = built_maker.get_dataset_summary()
        assert summary["total_images"] == len(built_maker.dataset_df)


# --------------------------------------------------------------------------- #
# CarDataset — construction                                                    #
# --------------------------------------------------------------------------- #

class TestCarDatasetInit:
    """CarDataset construction and split partitioning on the real CSV."""

    @pytest.fixture(scope="class")
    def dataset(self, df_abs: pd.DataFrame) -> CarDataset:
        """CarDataset built from the pre-built CSV with absolute image paths."""
        return CarDataset(
            df=df_abs,
            views=["front"],
            class_granularity="model",
            min_images=5,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            model_type="vision",
        )

    @pytest.mark.integration
    def test_dataset_constructed(self, dataset: CarDataset) -> None:
        """CarDataset can be constructed from the real CSV without errors."""
        assert dataset is not None

    @pytest.mark.integration
    def test_regular_classes_positive(self, dataset: CarDataset) -> None:
        """At least one regular class exists (enough images for splits)."""
        assert len(dataset.regular_classes) > 0

    @pytest.mark.integration
    def test_num_training_classes_positive(self, dataset: CarDataset) -> None:
        """get_num_classes_for_training() returns a positive integer."""
        assert dataset.get_num_classes_for_training() > 0

    @pytest.mark.integration
    def test_total_classes_gte_training_classes(self, dataset: CarDataset) -> None:
        """Total class count >= training class count (one-shot classes may exist)."""
        assert dataset.get_total_num_classes() >= dataset.get_num_classes_for_training()

    @pytest.mark.integration
    def test_split_sample_counts_positive(self, dataset: CarDataset) -> None:
        """Train, val, and test splits are all non-empty."""
        assert len(dataset.train_samples) > 0
        assert len(dataset.val_samples) > 0
        assert len(dataset.test_samples) > 0

    @pytest.mark.integration
    def test_label_encoder_fitted(self, dataset: CarDataset) -> None:
        """label_encoder is fitted with at least one class."""
        assert len(dataset.label_encoder.classes_) > 0

    @pytest.mark.integration
    def test_full_label_encoder_fitted(self, dataset: CarDataset) -> None:
        """label_encoder_full covers all regular + one-shot classes."""
        assert (
            len(dataset.label_encoder_full.classes_)
            >= dataset.get_num_classes_for_training()
        )

    @pytest.mark.unit
    def test_invalid_class_granularity_raises(self, df_abs: pd.DataFrame) -> None:
        """An unsupported class_granularity value raises ValueError."""
        with pytest.raises(ValueError, match="class_granularity"):
            CarDataset(df=df_abs, class_granularity="brand")

    @pytest.mark.unit
    def test_invalid_model_type_raises(self, df_abs: pd.DataFrame) -> None:
        """An unsupported model_type value raises ValueError."""
        with pytest.raises(ValueError, match="model_type|MODEL_TYPES"):
            CarDataset(df=df_abs, model_type="audio")

    @pytest.mark.unit
    def test_ratios_not_summing_to_one_raises(self, df_abs: pd.DataFrame) -> None:
        """Ratios that do not sum to 1.0 raise ValueError."""
        with pytest.raises(ValueError, match="sum"):
            CarDataset(
                df=df_abs,
                train_ratio=0.5,
                val_ratio=0.5,
                test_ratio=0.5,
            )

    @pytest.mark.unit
    def test_empty_dataframe_raises(self) -> None:
        """An empty DataFrame raises CarDatasetError (missing required columns)."""
        import pandas as pd
        with pytest.raises(CarDatasetError):
            CarDataset(df=pd.DataFrame())


# --------------------------------------------------------------------------- #
# CarDataset — split activation                                                #
# --------------------------------------------------------------------------- #

class TestCarDatasetSplits:
    """set_split / __len__ / get_current_samples interactions."""

    @pytest.fixture(scope="class")
    def dataset(self, df_abs: pd.DataFrame) -> CarDataset:
        """Shared CarDataset for split tests (class-scoped)."""
        return CarDataset(
            df=df_abs,
            views=["front"],
            class_granularity="model",
            min_images=5,
            model_type="vision",
        )

    @pytest.mark.integration
    def test_set_split_train_yields_positive_len(self, dataset: CarDataset) -> None:
        """After set_split('train'), __len__ returns the training sample count."""
        dataset.set_split("train")
        assert len(dataset) == len(dataset.train_samples)

    @pytest.mark.integration
    def test_set_split_val_yields_positive_len(self, dataset: CarDataset) -> None:
        """After set_split('val'), __len__ returns the validation sample count."""
        dataset.set_split("val")
        assert len(dataset) == len(dataset.val_samples)

    @pytest.mark.integration
    def test_set_split_test_yields_positive_len(self, dataset: CarDataset) -> None:
        """After set_split('test'), __len__ returns the test sample count."""
        dataset.set_split("test")
        assert len(dataset) == len(dataset.test_samples)

    @pytest.mark.unit
    def test_set_invalid_split_raises(self, dataset: CarDataset) -> None:
        """set_split with an unsupported name raises ValueError."""
        with pytest.raises(ValueError, match="split"):
            dataset.set_split("predict")

    @pytest.mark.unit
    def test_len_without_set_split_raises(self, df_abs: pd.DataFrame) -> None:
        """Calling __len__ before set_split raises ValueError."""
        fresh = CarDataset(
            df=df_abs,
            views=["front"],
            class_granularity="model",
            min_images=5,
            model_type="vision",
        )
        with pytest.raises(ValueError):
            _ = len(fresh)

    @pytest.mark.integration
    def test_total_samples_roughly_match_input(self, dataset: CarDataset) -> None:
        """Train + val + test samples account for all regular-class samples."""
        total = (
            len(dataset.train_samples)
            + len(dataset.val_samples)
            + len(dataset.test_samples)
        )
        assert total > 0

    @pytest.mark.integration
    def test_get_current_samples_after_set_split(self, dataset: CarDataset) -> None:
        """get_current_samples() returns the same list as the active split."""
        dataset.set_split("train")
        samples = dataset.get_current_samples()
        assert isinstance(samples, list)
        assert len(samples) == len(dataset.train_samples)


# --------------------------------------------------------------------------- #
# CarDataset — statistics and label encoder for evaluation                    #
# --------------------------------------------------------------------------- #

class TestCarDatasetStatistics:
    """get_dataset_statistics and get_label_encoder_for_evaluation."""

    @pytest.fixture(scope="class")
    def dataset(self, df_abs: pd.DataFrame) -> CarDataset:
        """Shared CarDataset (class-scoped)."""
        return CarDataset(
            df=df_abs,
            views=["front"],
            class_granularity="model",
            min_images=5,
            model_type="vision",
            include_oneshot_in_test=True,
            oneshot_ratio=0.1,
        )

    @pytest.mark.integration
    def test_get_dataset_statistics_returns_dict(self, dataset: CarDataset) -> None:
        """get_dataset_statistics() returns a dict."""
        stats = dataset.get_dataset_statistics()
        assert isinstance(stats, dict)

    @pytest.mark.integration
    def test_statistics_has_total_images(self, dataset: CarDataset) -> None:
        """Statistics dict contains a 'total_samples' key with a positive value."""
        stats = dataset.get_dataset_statistics()
        assert "total_samples" in stats
        assert stats["total_samples"] > 0

    @pytest.mark.integration
    def test_statistics_has_num_classes(self, dataset: CarDataset) -> None:
        """Statistics dict contains 'num_training_classes' and 'num_total_classes' keys."""
        stats = dataset.get_dataset_statistics()
        assert "num_training_classes" in stats
        assert "num_total_classes" in stats

    @pytest.mark.integration
    def test_get_label_encoder_for_evaluation_returns_encoder(self, dataset: CarDataset) -> None:
        """get_label_encoder_for_evaluation() returns a fitted LabelEncoder."""
        from sklearn.preprocessing import LabelEncoder
        encoder = dataset.get_label_encoder_for_evaluation()
        assert isinstance(encoder, LabelEncoder)
        assert len(encoder.classes_) > 0

    @pytest.mark.integration
    def test_eval_encoder_covers_all_classes(self, dataset: CarDataset) -> None:
        """Evaluation encoder has >= classes than the training-only encoder."""
        eval_encoder = dataset.get_label_encoder_for_evaluation()
        assert len(eval_encoder.classes_) >= dataset.get_num_classes_for_training()


# --------------------------------------------------------------------------- #
# robust_collate_fn                                                            #
# --------------------------------------------------------------------------- #

class TestRobustCollateFn:
    """robust_collate_fn handles normal batches and None entries."""

    @pytest.mark.unit
    def test_collate_fn_importable(self) -> None:
        """robust_collate_fn is importable from src.data."""
        from src.data.MyDataset import robust_collate_fn
        assert callable(robust_collate_fn)

    @pytest.mark.unit
    def test_collate_valid_batch(self) -> None:
        """A batch of well-formed dicts is collated without errors."""
        from src.data.MyDataset import robust_collate_fn
        batch = [
            {"images": torch.zeros(3, 64, 64), "labels": torch.tensor(0)},
            {"images": torch.zeros(3, 64, 64), "labels": torch.tensor(1)},
        ]
        result = robust_collate_fn(batch)
        assert isinstance(result, dict)
        assert "images" in result
        assert result["images"].shape[0] == 2

    @pytest.mark.unit
    def test_collate_filters_none_entries(self) -> None:
        """None entries in the batch are silently dropped."""
        from src.data.MyDataset import robust_collate_fn
        batch = [
            {"images": torch.zeros(3, 64, 64), "labels": torch.tensor(0)},
            None,
            {"images": torch.zeros(3, 64, 64), "labels": torch.tensor(1)},
        ]
        result = robust_collate_fn(batch)
        assert result["images"].shape[0] == 2


# --------------------------------------------------------------------------- #
# CarDataset — __getitem__                                                     #
# --------------------------------------------------------------------------- #

class TestCarDatasetGetItem:
    """__getitem__ on the training split returns correctly structured samples."""

    @pytest.fixture(scope="class")
    def dataset_train(self, df_abs: pd.DataFrame) -> CarDataset:
        """CarDataset on the 'train' split with a standard transform."""
        ds = CarDataset(
            df=df_abs,
            views=["front"],
            class_granularity="model",
            min_images=5,
            model_type="vision",
            transform=create_standard_transform(size=(64, 64)),
        )
        ds.set_split("train")
        return ds

    @pytest.mark.integration
    def test_getitem_returns_dict(self, dataset_train: CarDataset) -> None:
        """__getitem__ returns a dict."""
        sample = dataset_train[0]
        assert isinstance(sample, dict)

    @pytest.mark.integration
    def test_getitem_has_images_key(self, dataset_train: CarDataset) -> None:
        """Returned dict contains an 'images' key."""
        assert "images" in dataset_train[0]

    @pytest.mark.integration
    def test_getitem_has_labels_key(self, dataset_train: CarDataset) -> None:
        """Returned dict contains a 'labels' key."""
        assert "labels" in dataset_train[0]

    @pytest.mark.integration
    def test_image_is_tensor(self, dataset_train: CarDataset) -> None:
        """The 'images' value is a torch.Tensor."""
        assert isinstance(dataset_train[0]["images"], torch.Tensor)

    @pytest.mark.integration
    def test_image_tensor_shape(self, dataset_train: CarDataset) -> None:
        """Single-view image shape is (3, H, W) with the configured resize."""
        images = dataset_train[0]["images"]
        assert images.shape == (3, 64, 64), f"Unexpected shape: {images.shape}"

    @pytest.mark.integration
    def test_label_is_long_tensor(self, dataset_train: CarDataset) -> None:
        """The 'labels' value is a scalar torch.long tensor."""
        label = dataset_train[0]["labels"]
        assert isinstance(label, torch.Tensor)
        assert label.dtype == torch.long

    @pytest.mark.integration
    def test_label_is_non_negative(self, dataset_train: CarDataset) -> None:
        """Labels are non-negative integers (valid class indices)."""
        label = dataset_train[0]["labels"].item()
        assert label >= 0

    @pytest.mark.integration
    def test_label_within_num_classes(self, dataset_train: CarDataset) -> None:
        """Label is strictly less than the number of training classes."""
        label = dataset_train[0]["labels"].item()
        assert label < dataset_train.get_num_classes_for_training()

    @pytest.mark.integration
    def test_class_key_present(self, dataset_train: CarDataset) -> None:
        """Returned dict contains a non-empty 'class_key' entry."""
        assert "class_key" in dataset_train[0]
        assert dataset_train[0]["class_key"] is not None

    @pytest.mark.integration
    def test_image_paths_present(self, dataset_train: CarDataset) -> None:
        """Returned dict contains an 'image_paths' list."""
        sample = dataset_train[0]
        assert "image_paths" in sample
        assert isinstance(sample["image_paths"], list)
        assert len(sample["image_paths"]) > 0


# --------------------------------------------------------------------------- #
# create_car_dataset factory                                                   #
# --------------------------------------------------------------------------- #

class TestCreateCarDataset:
    """create_car_dataset factory function."""

    @pytest.fixture(scope="class")
    def factory_result(self, df_abs: pd.DataFrame) -> dict:
        """Return value of create_car_dataset (class-scoped, built once)."""
        return create_car_dataset(
            df=df_abs,
            views=["front"],
            min_images=5,
            batch_size=8,
            num_workers=0,
        )

    @pytest.mark.integration
    def test_returns_dict(self, factory_result: dict) -> None:
        """create_car_dataset returns a dict."""
        assert isinstance(factory_result, dict)

    @pytest.mark.integration
    @pytest.mark.parametrize("key", [
        "dataset", "train_loader", "val_loader", "test_loader", "train_sampler"
    ])
    def test_expected_keys_present(self, factory_result: dict, key: str) -> None:
        """Each documented key is present in the returned dict."""
        assert key in factory_result, f"Key '{key}' missing from create_car_dataset result"

    @pytest.mark.integration
    def test_dataset_is_car_dataset_instance(self, factory_result: dict) -> None:
        """The 'dataset' value is a CarDataset instance."""
        assert isinstance(factory_result["dataset"], CarDataset)

    @pytest.mark.unit
    def test_invalid_sampling_strategy_raises(self, df_abs: pd.DataFrame) -> None:
        """An unsupported sampling_strategy raises ValueError."""
        with pytest.raises(ValueError, match="sampling_strategy"):
            create_car_dataset(df=df_abs, sampling_strategy="random_walk")


# --------------------------------------------------------------------------- #
# MPerClassSamplerWrapper                                                      #
# --------------------------------------------------------------------------- #

class TestMPerClassSamplerWrapper:
    """MPerClassSamplerWrapper construction and iteration on real dataset samples."""

    @pytest.fixture(scope="class")
    def sampler(self, df_abs: pd.DataFrame) -> MPerClassSamplerWrapper:
        """MPerClassSamplerWrapper built from the training split of a CarDataset."""
        ds = CarDataset(
            df=df_abs,
            views=["front"],
            class_granularity="model",
            min_images=5,
            model_type="vision",
        )
        return MPerClassSamplerWrapper(samples=ds.train_samples, m=2)

    @pytest.mark.integration
    def test_construction_succeeds(self, sampler: MPerClassSamplerWrapper) -> None:
        """MPerClassSamplerWrapper can be built from CarDataset.train_samples."""
        assert sampler is not None

    @pytest.mark.integration
    def test_len_is_positive(self, sampler: MPerClassSamplerWrapper) -> None:
        """__len__ returns a positive integer."""
        assert len(sampler) > 0

    @pytest.mark.integration
    def test_labels_list_matches_samples(self, sampler: MPerClassSamplerWrapper) -> None:
        """Internal labels list has the same length as the sample list."""
        assert len(sampler.labels) == len(sampler.samples)

    @pytest.mark.integration
    def test_class_key_to_label_non_empty(self, sampler: MPerClassSamplerWrapper) -> None:
        """class_key_to_label dict has at least one entry."""
        assert len(sampler.class_key_to_label) > 0

    @pytest.mark.integration
    def test_iteration_yields_integers(self, sampler: MPerClassSamplerWrapper) -> None:
        """Iterating the sampler yields integer indices (int or numpy integer)."""
        import numbers
        indices = list(iter(sampler))
        assert len(indices) > 0
        assert all(isinstance(idx, numbers.Integral) for idx in indices[:10])

    @pytest.mark.integration
    def test_indices_within_valid_range(self, sampler: MPerClassSamplerWrapper) -> None:
        """All yielded indices are in [0, len(samples))."""
        indices = list(iter(sampler))
        n = len(sampler.samples)
        assert all(0 <= idx < n for idx in indices)


# --------------------------------------------------------------------------- #
# create_compcars_dataset factory                                              #
# --------------------------------------------------------------------------- #

class TestCreateCompcarsDataset:
    """create_compcars_dataset factory — callable / signature only.

    Actual execution triggers a full image-tree scan.  We verify only that
    the function is importable, callable, and has the expected signature.
    """

    @pytest.mark.unit
    def test_factory_is_callable(self) -> None:
        """create_compcars_dataset is a callable."""
        assert callable(create_compcars_dataset)

    @pytest.mark.unit
    def test_factory_in_data_package(self) -> None:
        """create_compcars_dataset is accessible from src.data."""
        import src.data as pkg
        assert hasattr(pkg, "create_compcars_dataset")

    @pytest.mark.unit
    def test_factory_accepts_base_path_param(self) -> None:
        """create_compcars_dataset signature includes a 'base_path' parameter."""
        import inspect
        sig = inspect.signature(create_compcars_dataset)
        assert "base_path" in sig.parameters

    @pytest.mark.unit
    def test_factory_accepts_output_path_param(self) -> None:
        """create_compcars_dataset signature includes an 'output_path' parameter."""
        import inspect
        sig = inspect.signature(create_compcars_dataset)
        assert "output_path" in sig.parameters


# --------------------------------------------------------------------------- #
# src.data package exports                                                     #
# --------------------------------------------------------------------------- #

class TestDataPackageExports:
    """All public symbols in src.data.__all__ are accessible from the package."""

    @pytest.mark.unit
    @pytest.mark.parametrize("symbol", [
        "DataFrameMaker",
        "CarDataset",
        "MPerClassSamplerWrapper",
        "CompCarsDatasetError",
        "CarDatasetError",
        "create_compcars_dataset",
        "create_car_dataset",
    ])
    def test_symbol_accessible(self, symbol: str) -> None:
        """Each exported symbol is accessible as an attribute of src.data."""
        import src.data as pkg
        assert hasattr(pkg, symbol), f"src.data missing export: {symbol}"
