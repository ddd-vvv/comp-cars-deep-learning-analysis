"""
Tests for ``src.pipeline`` — FineTuningPipeline and EmbeddingsPipeline.

Coverage:
    Both pipeline classes are tested at the API / contract level only.
    Actual pipeline execution (model training, embedding extraction, etc.)
    is intentionally excluded — it requires pretrained model weights,
    GPU resources, and multi-minute runs that are inappropriate for a
    standard test suite.

    What is verified:
    - Both classes and their error classes are importable and correctly typed.
    - All documented public methods are defined on the class.
    - The error classes are proper ``Exception`` subclasses.
    - Factory functions (``create_finetuning_pipeline``,
      ``create_embeddings_pipeline``) are callable.

Markers:
    unit — all tests here are pure import / introspection checks.
"""

from __future__ import annotations

__all__: list[str] = []

from inspect import isclass

import pytest

from src.pipeline import EmbeddingsPipeline, FineTuningPipeline
from src.pipeline.EmbeddingsPipeline import (
    EmbeddingsPipelineError,
    create_embeddings_pipeline,
)
from src.pipeline.FineTuningPipeline import (
    FineTuningPipelineError,
    create_finetuning_pipeline,
)


# --------------------------------------------------------------------------- #
# FineTuningPipeline                                                           #
# --------------------------------------------------------------------------- #

class TestFineTuningPipelineAPI:
    """FineTuningPipeline class-level API verification."""

    @pytest.mark.unit
    def test_class_is_importable(self) -> None:
        """FineTuningPipeline is importable and is a class."""
        assert isclass(FineTuningPipeline)

    @pytest.mark.unit
    @pytest.mark.parametrize("method_name", [
        "create_dataset",
        "create_model",
        "extract_baseline_embeddings",
        "run_finetuning",
        "extract_finetuned_embeddings",
        "save_results",
        "run_full_pipeline",
    ])
    def test_required_methods_exist(self, method_name: str) -> None:
        """All documented public methods are present on FineTuningPipeline."""
        assert hasattr(FineTuningPipeline, method_name), (
            f"FineTuningPipeline missing method: {method_name}"
        )

    @pytest.mark.unit
    def test_error_class_is_exception(self) -> None:
        """FineTuningPipelineError is a proper Exception subclass."""
        assert issubclass(FineTuningPipelineError, Exception)

    @pytest.mark.unit
    def test_error_class_can_be_raised(self) -> None:
        """FineTuningPipelineError can be raised and caught."""
        with pytest.raises(FineTuningPipelineError):
            raise FineTuningPipelineError("test error")

    @pytest.mark.unit
    def test_factory_function_is_callable(self) -> None:
        """create_finetuning_pipeline is a callable."""
        assert callable(create_finetuning_pipeline)

    @pytest.mark.unit
    def test_factory_accepts_config_param(self) -> None:
        """create_finetuning_pipeline signature includes a 'config' parameter."""
        import inspect
        sig = inspect.signature(create_finetuning_pipeline)
        assert "config" in sig.parameters

    @pytest.mark.unit
    def test_str_method_exists(self) -> None:
        """FineTuningPipeline defines __str__."""
        assert hasattr(FineTuningPipeline, "__str__")

    @pytest.mark.unit
    def test_repr_method_exists(self) -> None:
        """FineTuningPipeline defines __repr__."""
        assert hasattr(FineTuningPipeline, "__repr__")


# --------------------------------------------------------------------------- #
# EmbeddingsPipeline                                                           #
# --------------------------------------------------------------------------- #

class TestEmbeddingsPipelineAPI:
    """EmbeddingsPipeline class-level API verification."""

    @pytest.mark.unit
    def test_class_is_importable(self) -> None:
        """EmbeddingsPipeline is importable and is a class."""
        assert isclass(EmbeddingsPipeline)

    @pytest.mark.unit
    @pytest.mark.parametrize("method_name", [
        "create_dataset_for_labels",
        "load_embeddings_from_files",
        "load_embeddings_from_zip",
        "analyze_baseline_embeddings",
        "analyze_finetuned_embeddings",
        "compare_results",
        "save_results",
        "run_baseline_analysis_from_zip",
        "run_finetuned_analysis_from_zip",
        "run_full_analysis_from_zip",
        "run_comparative_analysis_from_zip",
        "run_baseline_analysis_from_files",
        "run_finetuned_analysis_from_files",
        "run_full_analysis_from_files",
        "run_comparative_analysis_from_files",
        "load_results_from_zip",
        "compare_saved_results",
        "__str__",
        "__repr__",
    ])
    def test_required_methods_exist(self, method_name: str) -> None:
        """All documented public methods are present on EmbeddingsPipeline."""
        assert hasattr(EmbeddingsPipeline, method_name), (
            f"EmbeddingsPipeline missing method: {method_name}"
        )

    @pytest.mark.unit
    def test_error_class_is_exception(self) -> None:
        """EmbeddingsPipelineError is a proper Exception subclass."""
        assert issubclass(EmbeddingsPipelineError, Exception)

    @pytest.mark.unit
    def test_error_class_can_be_raised(self) -> None:
        """EmbeddingsPipelineError can be raised and caught."""
        with pytest.raises(EmbeddingsPipelineError):
            raise EmbeddingsPipelineError("test error")

    @pytest.mark.unit
    def test_factory_function_is_callable(self) -> None:
        """create_embeddings_pipeline is a callable."""
        assert callable(create_embeddings_pipeline)

    @pytest.mark.unit
    def test_factory_accepts_config_param(self) -> None:
        """create_embeddings_pipeline signature includes a 'config' parameter."""
        import inspect
        sig = inspect.signature(create_embeddings_pipeline)
        assert "config" in sig.parameters


# --------------------------------------------------------------------------- #
# Top-level pipeline package re-exports                                        #
# --------------------------------------------------------------------------- #

class TestPipelinePackageExports:
    """Verify that src.pipeline re-exports both pipeline classes."""

    @pytest.mark.unit
    def test_fine_tuning_pipeline_in_package(self) -> None:
        """FineTuningPipeline is accessible from src.pipeline."""
        import src.pipeline as pkg
        assert hasattr(pkg, "FineTuningPipeline")

    @pytest.mark.unit
    def test_embeddings_pipeline_in_package(self) -> None:
        """EmbeddingsPipeline is accessible from src.pipeline."""
        import src.pipeline as pkg
        assert hasattr(pkg, "EmbeddingsPipeline")
