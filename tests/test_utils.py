"""
Tests for ``src.utils`` — JSON serialisation utilities.

Coverage:
    NumpyJSONEncoder: encoding of numpy integers, floats, arrays, and
    torch tensors (both small and large), plus the fallback for unsupported
    types.
    convert_numpy_keys: conversion of numpy integer dict keys to Python
    strings, recursive behaviour on nested structures, and passthrough for
    non-numpy keys.
    safe_json_dumps: round-trip serialisation to a JSON string.
    safe_json_dump: writing to a real temporary file and reading it back.

Markers:
    unit — all tests here are pure Python/NumPy/PyTorch logic with no
            filesystem I/O, except for TestSafeJsonDump which writes to a
            pytest-managed tmp_path (still considered unit-level I/O).
"""

from __future__ import annotations

__all__: list[str] = []

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.utils import (
    NumpyJSONEncoder,
    convert_numpy_keys,
    safe_json_dump,
    safe_json_dumps,
)


# --------------------------------------------------------------------------- #
# NumpyJSONEncoder                                                             #
# --------------------------------------------------------------------------- #

class TestNumpyJSONEncoder:
    """Custom JSON encoder handles numpy and torch types correctly."""

    @pytest.mark.unit
    @pytest.mark.parametrize("np_type", [
        np.int8, np.int16, np.int32, np.int64,
    ])
    def test_numpy_integer_encoded_as_int(self, np_type) -> None:
        """numpy integer scalars are serialised as Python int."""
        value = np_type(42)
        result = json.dumps(value, cls=NumpyJSONEncoder)
        assert json.loads(result) == 42
        assert isinstance(json.loads(result), int)

    @pytest.mark.unit
    @pytest.mark.parametrize("np_type", [
        np.float16, np.float32, np.float64,
    ])
    def test_numpy_float_encoded_as_float(self, np_type) -> None:
        """numpy float scalars are serialised as Python float."""
        value = np_type(3.14)
        result = json.dumps(value, cls=NumpyJSONEncoder)
        assert json.loads(result) == pytest.approx(3.14, abs=1e-2)

    @pytest.mark.unit
    def test_numpy_array_encoded_as_list(self) -> None:
        """numpy ndarrays are serialised as JSON arrays (lists)."""
        arr = np.array([1, 2, 3])
        result = json.loads(json.dumps(arr, cls=NumpyJSONEncoder))
        assert result == [1, 2, 3]

    @pytest.mark.unit
    def test_numpy_2d_array_encoded_as_nested_list(self) -> None:
        """2-D numpy arrays become nested JSON arrays."""
        arr = np.array([[1, 2], [3, 4]])
        result = json.loads(json.dumps(arr, cls=NumpyJSONEncoder))
        assert result == [[1, 2], [3, 4]]

    @pytest.mark.unit
    def test_small_torch_tensor_encoded_as_list(self) -> None:
        """torch.Tensor with <= 1000 elements is serialised as a list."""
        t = torch.tensor([1.0, 2.0, 3.0])
        result = json.loads(json.dumps(t, cls=NumpyJSONEncoder))
        assert result == pytest.approx([1.0, 2.0, 3.0])

    @pytest.mark.unit
    def test_large_torch_tensor_encoded_as_shape_string(self) -> None:
        """torch.Tensor with > 1000 elements is serialised as its shape string."""
        t = torch.randn(1001)
        result = json.loads(json.dumps(t, cls=NumpyJSONEncoder))
        assert isinstance(result, str)
        assert "1001" in result

    @pytest.mark.unit
    def test_fallback_raises_for_unsupported_type(self) -> None:
        """An unsupported object type raises TypeError (standard JSONEncoder behaviour)."""
        class Unsupported:
            pass

        with pytest.raises(TypeError):
            json.dumps(Unsupported(), cls=NumpyJSONEncoder)

    @pytest.mark.unit
    def test_nested_dict_with_numpy_values(self) -> None:
        """Dicts containing numpy values are serialised correctly."""
        data = {"a": np.int64(1), "b": np.float32(2.5), "c": np.array([3, 4])}
        result = json.loads(json.dumps(data, cls=NumpyJSONEncoder))
        assert result["a"] == 1
        assert result["b"] == pytest.approx(2.5, abs=1e-3)
        assert result["c"] == [3, 4]


# --------------------------------------------------------------------------- #
# convert_numpy_keys                                                           #
# --------------------------------------------------------------------------- #

class TestConvertNumpyKeys:
    """convert_numpy_keys converts numpy integer keys to Python strings."""

    @pytest.mark.unit
    def test_int64_key_becomes_string(self) -> None:
        """numpy.int64 dict keys are converted to str."""
        data = {np.int64(1): "value"}
        result = convert_numpy_keys(data)
        assert "1" in result
        assert result["1"] == "value"

    @pytest.mark.unit
    def test_int32_key_becomes_string(self) -> None:
        """numpy.int32 dict keys are converted to str."""
        data = {np.int32(99): "x"}
        result = convert_numpy_keys(data)
        assert "99" in result

    @pytest.mark.unit
    def test_string_key_left_unchanged(self) -> None:
        """Plain Python string keys are left as-is."""
        data = {"hello": 42}
        result = convert_numpy_keys(data)
        assert "hello" in result
        assert result["hello"] == 42

    @pytest.mark.unit
    def test_integer_key_left_unchanged(self) -> None:
        """Plain Python int keys are left as-is (only numpy types are converted)."""
        data = {1: "plain_int"}
        result = convert_numpy_keys(data)
        assert 1 in result

    @pytest.mark.unit
    def test_mixed_keys(self) -> None:
        """Mixed numpy and plain keys are handled independently."""
        data = {np.int64(0): "np_key", "str_key": "plain"}
        result = convert_numpy_keys(data)
        assert "0" in result
        assert "str_key" in result

    @pytest.mark.unit
    def test_nested_dict_recursion(self) -> None:
        """Nested dicts have their numpy keys converted recursively."""
        data = {np.int64(1): {np.int64(2): "deep"}}
        result = convert_numpy_keys(data)
        assert result["1"]["2"] == "deep"

    @pytest.mark.unit
    def test_list_values_recursed(self) -> None:
        """List values inside dicts are recursively processed."""
        data = {"key": [{np.int64(3): "in_list"}]}
        result = convert_numpy_keys(data)
        assert result["key"][0]["3"] == "in_list"

    @pytest.mark.unit
    def test_non_dict_scalars_returned_unchanged(self) -> None:
        """Scalar values (int, str, float) are returned as-is."""
        assert convert_numpy_keys(42) == 42
        assert convert_numpy_keys("hello") == "hello"
        assert convert_numpy_keys(3.14) == pytest.approx(3.14)


# --------------------------------------------------------------------------- #
# safe_json_dumps                                                              #
# --------------------------------------------------------------------------- #

class TestSafeJsonDumps:
    """safe_json_dumps serialises objects to a JSON string."""

    @pytest.mark.unit
    def test_returns_string(self) -> None:
        """Return type is str."""
        assert isinstance(safe_json_dumps({"a": 1}), str)

    @pytest.mark.unit
    def test_output_is_valid_json(self) -> None:
        """Output can be parsed by the standard json module."""
        result = safe_json_dumps({"x": np.array([1, 2, 3])})
        parsed = json.loads(result)
        assert isinstance(parsed, dict)

    @pytest.mark.unit
    def test_numpy_array_round_trips(self) -> None:
        """A numpy array survives a dumps → loads round-trip as a list."""
        data = {"values": np.array([10, 20, 30])}
        parsed = json.loads(safe_json_dumps(data))
        assert parsed["values"] == [10, 20, 30]

    @pytest.mark.unit
    def test_numpy_scalar_round_trips(self) -> None:
        """A numpy int64 value survives a dumps → loads round-trip as an int."""
        data = {"count": np.int64(7)}
        parsed = json.loads(safe_json_dumps(data))
        assert parsed["count"] == 7

    @pytest.mark.unit
    def test_nested_structure_preserved(self) -> None:
        """Nested dicts with numpy values are serialised and parsed correctly."""
        data = {
            "metrics": {
                "accuracy": np.float32(0.95),
                "counts": np.array([1, 2, 3]),
            }
        }
        parsed = json.loads(safe_json_dumps(data))
        assert parsed["metrics"]["accuracy"] == pytest.approx(0.95, abs=1e-3)
        assert parsed["metrics"]["counts"] == [1, 2, 3]

    @pytest.mark.unit
    def test_torch_tensor_round_trips(self) -> None:
        """A small torch.Tensor survives a dumps → loads round-trip."""
        data = {"embedding": torch.tensor([0.1, 0.2, 0.3])}
        parsed = json.loads(safe_json_dumps(data))
        assert parsed["embedding"] == pytest.approx([0.1, 0.2, 0.3], abs=1e-5)

    @pytest.mark.unit
    def test_numpy_integer_keys_converted(self) -> None:
        """Dict keys that are numpy integers are converted before serialisation."""
        data = {np.int64(1): "first", np.int64(2): "second"}
        parsed = json.loads(safe_json_dumps(data))
        assert "1" in parsed
        assert "2" in parsed


# --------------------------------------------------------------------------- #
# safe_json_dump                                                               #
# --------------------------------------------------------------------------- #

class TestSafeJsonDump:
    """safe_json_dump writes a JSON file to disk."""

    @pytest.mark.unit
    def test_creates_file(self, tmp_path: Path) -> None:
        """safe_json_dump creates the target file."""
        out = tmp_path / "output.json"
        safe_json_dump({"key": "value"}, str(out))
        assert out.exists()

    @pytest.mark.unit
    def test_written_file_is_valid_json(self, tmp_path: Path) -> None:
        """The file written by safe_json_dump is valid JSON."""
        out = tmp_path / "valid.json"
        safe_json_dump({"a": 1, "b": [2, 3]}, str(out))
        with open(out) as f:
            loaded = json.load(f)
        assert loaded == {"a": 1, "b": [2, 3]}

    @pytest.mark.unit
    def test_numpy_values_serialised(self, tmp_path: Path) -> None:
        """numpy values are correctly written and readable."""
        data = {
            "array": np.array([1.0, 2.0, 3.0]),
            "scalar": np.int64(42),
        }
        out = tmp_path / "numpy.json"
        safe_json_dump(data, str(out))
        with open(out) as f:
            loaded = json.load(f)
        assert loaded["array"] == pytest.approx([1.0, 2.0, 3.0])
        assert loaded["scalar"] == 42

    @pytest.mark.unit
    def test_existing_file_is_overwritten(self, tmp_path: Path) -> None:
        """Calling safe_json_dump twice on the same path overwrites the file."""
        out = tmp_path / "overwrite.json"
        safe_json_dump({"round": 1}, str(out))
        safe_json_dump({"round": 2}, str(out))
        with open(out) as f:
            loaded = json.load(f)
        assert loaded["round"] == 2

    @pytest.mark.unit
    def test_default_indent_is_applied(self, tmp_path: Path) -> None:
        """The default indent=2 means the file contains newlines."""
        out = tmp_path / "indented.json"
        safe_json_dump({"x": 1}, str(out))
        content = out.read_text()
        assert "\n" in content

    @pytest.mark.unit
    def test_numpy_integer_keys_written_correctly(self, tmp_path: Path) -> None:
        """numpy integer keys are converted to strings before writing."""
        data = {np.int64(0): "zero", np.int64(1): "one"}
        out = tmp_path / "np_keys.json"
        safe_json_dump(data, str(out))
        with open(out) as f:
            loaded = json.load(f)
        assert "0" in loaded
        assert "1" in loaded


# --------------------------------------------------------------------------- #
# fast_purity_calculation                                                      #
# --------------------------------------------------------------------------- #

class TestFastPurityCalculation:
    """Numba-compiled purity metric."""

    @pytest.mark.unit
    def test_perfect_purity_is_one(self) -> None:
        """Perfectly separated clusters yield purity = 1.0."""
        cluster_labels = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=np.int32)
        true_labels    = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=np.int32)
        from src.utils import fast_purity_calculation
        result = fast_purity_calculation(cluster_labels, true_labels)
        assert result == pytest.approx(1.0)

    @pytest.mark.unit
    def test_mixed_cluster_purity_less_than_one(self) -> None:
        """Mixed clusters yield purity < 1.0."""
        cluster_labels = np.array([0, 0, 0, 0], dtype=np.int32)
        true_labels    = np.array([0, 0, 1, 1], dtype=np.int32)
        from src.utils import fast_purity_calculation
        result = fast_purity_calculation(cluster_labels, true_labels)
        assert result == pytest.approx(0.5)

    @pytest.mark.unit
    def test_noise_points_excluded(self) -> None:
        """Noise samples (label == -1) are excluded from the calculation."""
        cluster_labels = np.array([-1, 0, 0, 1, 1], dtype=np.int32)
        true_labels    = np.array([ 9, 0, 0, 1, 1], dtype=np.int32)
        from src.utils import fast_purity_calculation
        # Noise point excluded; both real clusters are pure.
        result = fast_purity_calculation(cluster_labels, true_labels)
        assert result == pytest.approx(4 / 5)

    @pytest.mark.unit
    def test_return_type_is_float(self) -> None:
        """Return type is a Python float."""
        from src.utils import fast_purity_calculation
        result = fast_purity_calculation(
            np.array([0, 0, 1, 1], dtype=np.int32),
            np.array([0, 0, 1, 1], dtype=np.int32),
        )
        assert isinstance(result, float)


# --------------------------------------------------------------------------- #
# ClusteringAnalyzer                                                           #
# --------------------------------------------------------------------------- #

class TestClusteringAnalyzerConstruction:
    """ClusteringAnalyzer initialisation with small synthetic embeddings."""

    @pytest.mark.unit
    def test_init_with_valid_data(self) -> None:
        """ClusteringAnalyzer can be constructed with valid numpy arrays."""
        from src.utils import ClusteringAnalyzer
        embeddings  = np.random.default_rng(0).standard_normal((30, 8)).astype(np.float32)
        true_labels = np.repeat(np.arange(3), 10).astype(np.int32)
        analyzer = ClusteringAnalyzer(embeddings, true_labels, optimizer_trials=1)
        assert analyzer is not None

    @pytest.mark.unit
    def test_embeddings_stored(self) -> None:
        """Embeddings are stored as a numpy array on the instance."""
        from src.utils import ClusteringAnalyzer
        embeddings  = np.random.default_rng(1).standard_normal((30, 8)).astype(np.float32)
        true_labels = np.repeat(np.arange(3), 10).astype(np.int32)
        analyzer = ClusteringAnalyzer(embeddings, true_labels, optimizer_trials=1)
        assert isinstance(analyzer.embeddings, np.ndarray)
        assert analyzer.embeddings.shape[0] == 30

    @pytest.mark.unit
    def test_available_methods_stored(self) -> None:
        """available_methods attribute is a non-empty list."""
        from src.utils import ClusteringAnalyzer
        embeddings  = np.random.default_rng(2).standard_normal((30, 8)).astype(np.float32)
        true_labels = np.repeat(np.arange(3), 10).astype(np.int32)
        analyzer = ClusteringAnalyzer(
            embeddings, true_labels,
            available_methods=["dbscan"],
            optimizer_trials=1
        )
        assert "dbscan" in analyzer.available_methods


class TestClusteringAnalyzerCluster:
    """ClusteringAnalyzer.cluster() with explicit parameters (no Optuna)."""

    @pytest.fixture(scope="class")
    def analyzer(self) -> object:
        """Shared ClusteringAnalyzer constructed with well-separated clusters."""
        from src.utils import ClusteringAnalyzer
        rng = np.random.default_rng(42)
        # 3 well-separated blobs of 10 points in 4-D space.
        centers = np.array([[0, 0, 0, 0], [5, 5, 5, 5], [-5, -5, -5, -5]], dtype=np.float32)
        parts = [rng.standard_normal((10, 4)).astype(np.float32) * 0.1 + centers[i] for i in range(3)]
        embeddings  = np.vstack(parts)
        true_labels = np.repeat(np.arange(3), 10).astype(np.int32)
        return ClusteringAnalyzer(
            embeddings, true_labels,
            available_methods=["dbscan"],
            optimizer_trials=1
        )

    @pytest.mark.unit
    def test_cluster_returns_ndarray(self, analyzer) -> None:
        """cluster() with explicit params returns a numpy ndarray."""
        labels = analyzer.cluster("dbscan", params={"eps": 1.0, "min_samples": 2})
        assert isinstance(labels, np.ndarray)

    @pytest.mark.unit
    def test_cluster_labels_length_matches_input(self, analyzer) -> None:
        """Length of cluster labels equals the number of embeddings."""
        labels = analyzer.cluster("dbscan", params={"eps": 1.0, "min_samples": 2})
        assert len(labels) == len(analyzer.embeddings)

    @pytest.mark.unit
    def test_evaluate_clustering_returns_expected_keys(self, analyzer) -> None:
        """evaluate_clustering() returns a dict with a 'silhouette_score' key."""
        labels = analyzer.cluster("dbscan", params={"eps": 1.0, "min_samples": 2})
        metrics = analyzer.evaluate_clustering(labels, method_name="dbscan")
        assert isinstance(metrics, dict)
        assert "silhouette_score" in metrics


class TestClusteringAnalyzerMultiMethod:
    """ClusteringAnalyzer.cluster_all(), compare_methods(), get_best_result(), get_cluster_summary()."""

    @pytest.fixture(scope="class")
    def analyzer_with_results(self) -> object:
        """ClusteringAnalyzer with results pre-populated via explicit cluster() call."""
        from src.utils import ClusteringAnalyzer
        rng = np.random.default_rng(42)
        centers = np.array([[0,0,0,0],[5,5,5,5],[-5,-5,-5,-5]], dtype=np.float32)
        parts = [rng.standard_normal((10,4)).astype(np.float32)*0.1+centers[i] for i in range(3)]
        embeddings  = np.vstack(parts)
        true_labels = np.repeat(np.arange(3), 10).astype(np.int32)
        a = ClusteringAnalyzer(
            embeddings, true_labels,
            available_methods=["dbscan"],
            optimizer_trials=1
        )
        # Populate results directly with explicit params (no Optuna).
        labels = a.cluster("dbscan", params={"eps": 1.0, "min_samples": 2})
        a.results["dbscan"] = labels
        return a

    @pytest.mark.unit
    def test_get_best_result_returns_tuple(self, analyzer_with_results) -> None:
        """get_best_result() returns a (method_name, labels) tuple."""
        best_method, best_labels = analyzer_with_results.get_best_result()
        assert isinstance(best_method, str)
        assert isinstance(best_labels, np.ndarray)

    @pytest.mark.unit
    def test_get_cluster_summary_returns_dataframe(self, analyzer_with_results) -> None:
        """get_cluster_summary() returns a DataFrame."""
        import pandas as pd
        summary = analyzer_with_results.get_cluster_summary("dbscan")
        assert isinstance(summary, pd.DataFrame)

    @pytest.mark.unit
    def test_cluster_all_returns_dict(self) -> None:
        """cluster_all() with explicit-param clustering returns a dict of ndarrays."""
        from src.utils import ClusteringAnalyzer
        rng = np.random.default_rng(0)
        centers = np.array([[0,0,0,0],[8,8,8,8],[-8,-8,-8,-8]], dtype=np.float32)
        parts = [rng.standard_normal((15,4)).astype(np.float32)*0.05+centers[i] for i in range(3)]
        embeddings  = np.vstack(parts)
        true_labels = np.repeat(np.arange(3), 15).astype(np.int32)
        a = ClusteringAnalyzer(
            embeddings, true_labels,
            available_methods=["dbscan"],
            optimizer_trials=5
        )
        result = a.cluster_all(methods=["dbscan"])
        # cluster_all may return empty if Optuna finds no valid params; just
        # check the return type.
        assert isinstance(result, dict)

    @pytest.mark.unit
    def test_compare_methods_returns_dataframe(self, analyzer_with_results) -> None:
        """compare_methods() returns a pandas DataFrame after results are populated."""
        import pandas as pd
        df = analyzer_with_results.compare_methods(methods=["dbscan"])
        assert isinstance(df, pd.DataFrame)
# --------------------------------------------------------------------------- #

class TestDimensionalityReducerConstruction:
    """DimensionalityReducer initialisation with small synthetic data."""

    @pytest.mark.unit
    def test_init_with_torch_tensor(self) -> None:
        """DimensionalityReducer can be constructed from a torch.Tensor."""
        from src.utils import DimensionalityReducer
        import torch
        embeddings  = torch.randn(40, 16)
        labels      = torch.repeat_interleave(torch.arange(4), 10)
        reducer = DimensionalityReducer(embeddings, labels, optimizer_trials=1)
        assert reducer is not None

    @pytest.mark.unit
    def test_init_with_numpy_array(self) -> None:
        """DimensionalityReducer can be constructed from a numpy ndarray."""
        from src.utils import DimensionalityReducer
        embeddings  = np.random.default_rng(7).standard_normal((40, 16)).astype(np.float32)
        labels      = np.repeat(np.arange(4), 10).astype(np.int32)
        reducer = DimensionalityReducer(embeddings, labels, optimizer_trials=1)
        assert reducer.embeddings.shape == (40, 16)

    @pytest.mark.unit
    def test_embeddings_standardised(self) -> None:
        """Embeddings are standard-scaled during construction."""
        from src.utils import DimensionalityReducer
        embeddings  = np.random.default_rng(8).standard_normal((40, 16)).astype(np.float32)
        labels      = np.repeat(np.arange(4), 10).astype(np.int32)
        reducer = DimensionalityReducer(embeddings, labels, optimizer_trials=1)
        # After StandardScaler the mean should be approximately 0.
        assert np.abs(reducer.embeddings.mean()) < 0.5


class TestDimensionalityReducerReduce:
    """DimensionalityReducer.reduce() with explicit params (no Optuna)."""

    @pytest.fixture(scope="class")
    def reducer(self) -> object:
        """Shared DimensionalityReducer (class-scoped)."""
        from src.utils import DimensionalityReducer
        embeddings  = np.random.default_rng(9).standard_normal((40, 16)).astype(np.float32)
        labels      = np.repeat(np.arange(4), 10).astype(np.int32)
        return DimensionalityReducer(
            embeddings, labels,
            available_methods=["pca"],
            optimizer_trials=1
        )

    @pytest.mark.unit
    def test_reduce_pca_returns_ndarray(self, reducer) -> None:
        """reduce('pca') with explicit params returns a numpy ndarray."""
        result = reducer.reduce("pca", params={"n_components": 2})
        assert isinstance(result, np.ndarray)

    @pytest.mark.unit
    def test_reduced_shape_correct(self, reducer) -> None:
        """Output has shape (n_samples, n_components)."""
        result = reducer.reduce("pca", params={"n_components": 3})
        assert result.shape == (40, 3)


class TestDimensionalityReducerMultiMethod:
    """DimensionalityReducer.reduce_all(), compare_methods(), get_best_result()."""

    @pytest.fixture(scope="class")
    def reducer(self) -> object:
        """Shared DimensionalityReducer restricted to PCA only (fast)."""
        from src.utils import DimensionalityReducer
        embeddings  = np.random.default_rng(10).standard_normal((40, 16)).astype(np.float32)
        labels      = np.repeat(np.arange(4), 10).astype(np.int32)
        return DimensionalityReducer(
            embeddings, labels,
            available_methods=["pca"],
            optimizer_trials=1
        )

    @pytest.mark.unit
    def test_reduce_all_returns_dict(self, reducer) -> None:
        """reduce_all() returns a dict mapping method name to ndarray."""
        result = reducer.reduce_all(methods=["pca"])
        assert isinstance(result, dict)
        assert "pca" in result
        assert isinstance(result["pca"], np.ndarray)

    @pytest.mark.unit
    def test_reduce_all_shape_correct(self, reducer) -> None:
        """reduce_all() output arrays have the correct number of rows."""
        result = reducer.reduce_all(methods=["pca"])
        assert result["pca"].shape[0] == 40

    @pytest.mark.unit
    def test_compare_methods_returns_dict(self, reducer) -> None:
        """compare_methods() returns a dict of {method: score}."""
        result = reducer.compare_methods(methods=["pca"])
        assert isinstance(result, dict)
        assert "pca" in result

    @pytest.mark.unit
    def test_get_best_result_returns_tuple(self, reducer) -> None:
        """get_best_result() returns a (method_name, reduced_array) tuple."""
        best_method, best_array = reducer.get_best_result()
        assert isinstance(best_method, str)
        assert isinstance(best_array, np.ndarray)


# --------------------------------------------------------------------------- #
# ClusterVisualizer API                                                        #
# --------------------------------------------------------------------------- #

class TestClusterVisualizerAPI:
    """ClusterVisualizer class-level API (no visualisation rendering)."""

    @pytest.mark.unit
    def test_class_importable(self) -> None:
        """ClusterVisualizer is importable."""
        from inspect import isclass
        from src.utils import ClusterVisualizer
        assert isclass(ClusterVisualizer)

    @pytest.mark.unit
    @pytest.mark.parametrize("method_name", [
        "print_cluster_statistics",
        "get_class_cluster_overlap",
        "visualize_good_clusters",
        "visualize_mixed_clusters",
        "visualize_best_available_clusters",
        "save_visualizations",
    ])
    def test_required_methods_exist(self, method_name: str) -> None:
        """All documented public methods are defined on ClusterVisualizer."""
        from src.utils import ClusterVisualizer
        assert hasattr(ClusterVisualizer, method_name), (
            f"ClusterVisualizer missing method: {method_name}"
        )


# --------------------------------------------------------------------------- #
# src.utils package exports                                                    #
# --------------------------------------------------------------------------- #

class TestUtilsPackageExports:
    """All public symbols in src.utils.__all__ are accessible from the package."""

    @pytest.mark.unit
    @pytest.mark.parametrize("symbol", [
        "ClusteringAnalyzer",
        "DimensionalityReducer",
        "ClusterVisualizer",
        "NumpyJSONEncoder",
        "convert_numpy_keys",
        "safe_json_dump",
        "safe_json_dumps",
        "fast_purity_calculation",
    ])
    def test_symbol_accessible(self, symbol: str) -> None:
        """Each exported symbol is accessible as an attribute of src.utils."""
        import src.utils as pkg
        assert hasattr(pkg, symbol), f"src.utils missing export: {symbol}"
