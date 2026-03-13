"""
Tests for ``src.models`` — metric-learning losses, miners, and model APIs.

Coverage:
    Loss wrappers: construction with default and custom parameters, forward
    pass shape/type, value range checks, and error cases for unsupported
    arguments.
    Factory functions: create_metric_learning_loss for every supported loss
    type, create_miner for every supported miner type.
    Model APIs (MultiViewVisionModel, MultiViewCLIPModel): class existence,
    required method presence, and exception hierarchy — no heavyweight model
    instantiation is performed here (that would require downloading pretrained
    weights and is out of scope for unit tests).

Markers:
    unit — all tests here are pure PyTorch / Python logic with no I/O.
"""

from __future__ import annotations

__all__: list[str] = []

import pytest
import torch
import torch.nn as nn

from src.models import (
    ArcFaceLossWrapper,
    CLIPLoss,
    CLIPModelError,
    ContrastiveLossWrapper,
    MultiSimilarityLossWrapper,
    MultiViewCLIPModel,
    MultiViewVisionModel,
    NTXentLossWrapper,
    TripletLossWrapper,
    VisionModelError,
    create_clip_model,
    create_miner,
    create_metric_learning_loss,
    create_vision_model,
)
from src.defaults import (
    DEFAULT_ARCFACE_MARGIN,
    DEFAULT_ARCFACE_SCALE,
    DEFAULT_CONTRASTIVE_MARGIN,
    DEFAULT_TRIPLET_MARGIN,
)


# --------------------------------------------------------------------------- #
# Shared helpers                                                               #
# --------------------------------------------------------------------------- #

def _random_embeddings(batch: int = 16, dim: int = 64) -> torch.Tensor:
    """Return L2-normalised random embeddings of shape (batch, dim)."""
    x = torch.randn(batch, dim)
    return torch.nn.functional.normalize(x, p=2, dim=1)


def _random_labels(batch: int = 16, num_classes: int = 4) -> torch.Tensor:
    """Return integer labels in [0, num_classes) of shape (batch,)."""
    return torch.randint(0, num_classes, (batch,))


# --------------------------------------------------------------------------- #
# TripletLossWrapper                                                           #
# --------------------------------------------------------------------------- #

class TestTripletLossWrapper:
    """TripletLossWrapper construction and forward pass."""

    @pytest.mark.unit
    def test_default_init(self) -> None:
        """TripletLossWrapper can be created with no arguments."""
        loss_fn = TripletLossWrapper()
        assert loss_fn is not None

    @pytest.mark.unit
    def test_default_margin_stored(self) -> None:
        """Default margin attribute matches DEFAULT_TRIPLET_MARGIN."""
        loss_fn = TripletLossWrapper()
        assert loss_fn.margin == pytest.approx(DEFAULT_TRIPLET_MARGIN)

    @pytest.mark.unit
    def test_is_nn_module(self) -> None:
        """TripletLossWrapper is a torch.nn.Module subclass."""
        assert isinstance(TripletLossWrapper(), nn.Module)

    @pytest.mark.unit
    def test_forward_returns_scalar_tensor(self) -> None:
        """forward() returns a scalar torch.Tensor."""
        loss_fn = TripletLossWrapper()
        embeddings = _random_embeddings()
        labels = _random_labels()
        result = loss_fn(embeddings, labels)
        assert isinstance(result, torch.Tensor)
        assert result.ndim == 0  # scalar

    @pytest.mark.unit
    def test_forward_value_is_non_negative(self) -> None:
        """Triplet loss must be >= 0."""
        loss_fn = TripletLossWrapper()
        result = loss_fn(_random_embeddings(), _random_labels())
        assert result.item() >= 0.0

    @pytest.mark.unit
    def test_cosine_distance_variant(self) -> None:
        """distance='cosine' creates successfully and produces a scalar."""
        loss_fn = TripletLossWrapper(distance="cosine")
        result = loss_fn(_random_embeddings(), _random_labels())
        assert result.ndim == 0

    @pytest.mark.unit
    def test_dot_distance_variant(self) -> None:
        """distance='dot' creates successfully and produces a scalar."""
        loss_fn = TripletLossWrapper(distance="dot")
        result = loss_fn(_random_embeddings(), _random_labels())
        assert result.ndim == 0

    @pytest.mark.unit
    def test_unsupported_distance_raises(self) -> None:
        """An unsupported distance string raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported distance"):
            TripletLossWrapper(distance="manhattan")


# --------------------------------------------------------------------------- #
# ContrastiveLossWrapper                                                       #
# --------------------------------------------------------------------------- #

class TestContrastiveLossWrapper:
    """ContrastiveLossWrapper construction and forward pass."""

    @pytest.mark.unit
    def test_default_init(self) -> None:
        """ContrastiveLossWrapper can be created with default arguments."""
        assert ContrastiveLossWrapper() is not None

    @pytest.mark.unit
    def test_default_margins_stored(self) -> None:
        """Default neg_margin and pos_margin attributes are set correctly."""
        loss_fn = ContrastiveLossWrapper()
        assert loss_fn.neg_margin == pytest.approx(DEFAULT_CONTRASTIVE_MARGIN)
        assert loss_fn.pos_margin == pytest.approx(0.0)

    @pytest.mark.unit
    def test_forward_returns_scalar(self) -> None:
        """forward() returns a scalar tensor."""
        result = ContrastiveLossWrapper()(_random_embeddings(), _random_labels())
        assert result.ndim == 0

    @pytest.mark.unit
    def test_forward_non_negative(self) -> None:
        """Contrastive loss must be >= 0."""
        result = ContrastiveLossWrapper()(_random_embeddings(), _random_labels())
        assert result.item() >= 0.0

    @pytest.mark.unit
    def test_unsupported_distance_raises(self) -> None:
        """An unsupported distance string raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported distance"):
            ContrastiveLossWrapper(distance="chebyshev")


# --------------------------------------------------------------------------- #
# NTXentLossWrapper                                                            #
# --------------------------------------------------------------------------- #

class TestNTXentLossWrapper:
    """NTXentLossWrapper (InfoNCE) construction and forward pass."""

    @pytest.mark.unit
    def test_default_init(self) -> None:
        """NTXentLossWrapper can be created with no arguments."""
        assert NTXentLossWrapper() is not None

    @pytest.mark.unit
    def test_forward_returns_scalar(self) -> None:
        """forward() returns a scalar tensor."""
        loss_fn = NTXentLossWrapper()
        result = loss_fn(_random_embeddings(), _random_labels())
        assert result.ndim == 0

    @pytest.mark.unit
    def test_custom_temperature(self) -> None:
        """A custom temperature does not prevent construction or a forward pass."""
        loss_fn = NTXentLossWrapper(temperature=0.1)
        result = loss_fn(_random_embeddings(), _random_labels())
        assert result.ndim == 0


# --------------------------------------------------------------------------- #
# MultiSimilarityLossWrapper                                                   #
# --------------------------------------------------------------------------- #

class TestMultiSimilarityLossWrapper:
    """MultiSimilarityLossWrapper construction and forward pass."""

    @pytest.mark.unit
    def test_default_init(self) -> None:
        """MultiSimilarityLossWrapper can be created with no arguments."""
        assert MultiSimilarityLossWrapper() is not None

    @pytest.mark.unit
    def test_forward_returns_scalar(self) -> None:
        """forward() returns a scalar tensor."""
        result = MultiSimilarityLossWrapper()(_random_embeddings(), _random_labels())
        assert result.ndim == 0


# --------------------------------------------------------------------------- #
# ArcFaceLossWrapper                                                           #
# --------------------------------------------------------------------------- #

class TestArcFaceLossWrapper:
    """ArcFaceLossWrapper construction, attributes, and forward pass."""

    @pytest.mark.unit
    def test_default_init(self) -> None:
        """ArcFaceLossWrapper can be created with required arguments."""
        loss_fn = ArcFaceLossWrapper(num_classes=10, embedding_dim=64)
        assert loss_fn is not None

    @pytest.mark.unit
    def test_is_nn_module(self) -> None:
        """ArcFaceLossWrapper is a torch.nn.Module subclass."""
        assert isinstance(ArcFaceLossWrapper(num_classes=10, embedding_dim=64), nn.Module)

    @pytest.mark.unit
    def test_num_classes_stored(self) -> None:
        """num_classes attribute is set correctly."""
        loss_fn = ArcFaceLossWrapper(num_classes=10, embedding_dim=64)
        assert loss_fn.num_classes == 10

    @pytest.mark.unit
    def test_embedding_dim_stored(self) -> None:
        """embedding_dim attribute is set correctly."""
        loss_fn = ArcFaceLossWrapper(num_classes=10, embedding_dim=64)
        assert loss_fn.embedding_dim == 64

    @pytest.mark.unit
    def test_default_scale_stored(self) -> None:
        """Default scale attribute matches DEFAULT_ARCFACE_SCALE."""
        loss_fn = ArcFaceLossWrapper(num_classes=10, embedding_dim=64)
        assert loss_fn.scale == pytest.approx(DEFAULT_ARCFACE_SCALE)

    @pytest.mark.unit
    def test_custom_margin_stored(self) -> None:
        """A custom radian margin is converted to degrees and stored."""
        import math
        margin_rad = 0.3
        loss_fn = ArcFaceLossWrapper(num_classes=10, embedding_dim=64, margin=margin_rad)
        assert loss_fn.margin == pytest.approx(math.degrees(margin_rad), rel=1e-5)

    @pytest.mark.unit
    def test_forward_returns_scalar_tensor(self) -> None:
        """forward() returns a scalar torch.Tensor."""
        loss_fn = ArcFaceLossWrapper(num_classes=4, embedding_dim=64)
        embeddings = _random_embeddings(batch=16, dim=64)
        labels = _random_labels(batch=16, num_classes=4)
        result = loss_fn(embeddings, labels)
        assert isinstance(result, torch.Tensor)
        assert result.ndim == 0

    @pytest.mark.unit
    def test_forward_value_is_non_negative(self) -> None:
        """ArcFace loss must be >= 0."""
        loss_fn = ArcFaceLossWrapper(num_classes=4, embedding_dim=64)
        result = loss_fn(_random_embeddings(dim=64), _random_labels())
        assert result.item() >= 0.0

    @pytest.mark.unit
    def test_get_logits_returns_correct_shape(self) -> None:
        """get_logits() returns a tensor of shape (batch, num_classes)."""
        num_classes = 8
        dim = 64
        loss_fn = ArcFaceLossWrapper(num_classes=num_classes, embedding_dim=dim)
        embeddings = _random_embeddings(batch=16, dim=dim)
        logits = loss_fn.get_logits(embeddings)
        assert logits.shape == (16, num_classes), (
            f"Expected ({16}, {num_classes}), got {logits.shape}"
        )


# --------------------------------------------------------------------------- #
# CLIPLoss                                                                     #
# --------------------------------------------------------------------------- #

class TestCLIPLoss:
    """Bidirectional CLIP contrastive loss."""

    @pytest.mark.unit
    def test_default_init(self) -> None:
        """CLIPLoss can be created with default reduction."""
        assert CLIPLoss() is not None

    @pytest.mark.unit
    def test_forward_returns_scalar(self) -> None:
        """forward() with square logit matrices returns a scalar tensor."""
        loss_fn = CLIPLoss()
        n = 8
        logits_i = torch.randn(n, n)
        logits_t = logits_i.T
        result = loss_fn(logits_i, logits_t)
        assert result.ndim == 0

    @pytest.mark.unit
    def test_perfect_diagonal_alignment_gives_low_loss(self) -> None:
        """Very large diagonal logits (perfect alignment) give near-zero loss."""
        loss_fn = CLIPLoss()
        n = 8
        # Each image matches only its paired text.
        logits = torch.eye(n) * 100.0
        result = loss_fn(logits, logits.T)
        assert result.item() < 0.1, f"Expected low loss, got {result.item():.4f}"

    @pytest.mark.unit
    def test_loss_is_symmetric(self) -> None:
        """Swapping logits_per_image and logits_per_text does not change the loss
        when the matrices are transposed of each other."""
        loss_fn = CLIPLoss()
        n = 6
        logits_i = torch.randn(n, n)
        logits_t = logits_i.T
        loss_a = loss_fn(logits_i, logits_t)
        loss_b = loss_fn(logits_t, logits_i)
        assert loss_a.item() == pytest.approx(loss_b.item(), abs=1e-5)


# --------------------------------------------------------------------------- #
# create_metric_learning_loss factory                                          #
# --------------------------------------------------------------------------- #

class TestCreateMetricLearningLoss:
    """create_metric_learning_loss factory for all supported loss types."""

    @pytest.mark.unit
    @pytest.mark.parametrize("loss_type", [
        "triplet", "contrastive", "ntxent", "multisimilarity", "clip",
        # Aliases
        "TripletLoss", "ContrastiveLoss", "NTXentLoss", "InfoNCE",
    ])
    def test_creates_nn_module(self, loss_type: str) -> None:
        """Each supported loss type returns a torch.nn.Module."""
        loss_fn = create_metric_learning_loss(loss_type)
        assert isinstance(loss_fn, nn.Module)

    @pytest.mark.unit
    def test_creates_arcface_with_required_params(self) -> None:
        """ArcFace can be created when num_classes and embedding_dim are given."""
        loss_fn = create_metric_learning_loss(
            "arcface", num_classes=10, embedding_dim=64
        )
        assert isinstance(loss_fn, nn.Module)

    @pytest.mark.unit
    def test_arcface_without_num_classes_raises(self) -> None:
        """ArcFace without num_classes raises ValueError."""
        with pytest.raises(ValueError, match="num_classes"):
            create_metric_learning_loss("arcface", embedding_dim=64)

    @pytest.mark.unit
    def test_arcface_without_embedding_dim_raises(self) -> None:
        """ArcFace without embedding_dim raises ValueError."""
        with pytest.raises(ValueError, match="embedding_dim"):
            create_metric_learning_loss("arcface", num_classes=10)

    @pytest.mark.unit
    def test_unsupported_type_raises(self) -> None:
        """An unsupported loss type string raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported loss type"):
            create_metric_learning_loss("softmax_ce")

    @pytest.mark.unit
    def test_custom_triplet_margin_forwarded(self) -> None:
        """Extra kwargs are forwarded to the underlying loss wrapper."""
        loss_fn = create_metric_learning_loss("triplet", margin=0.5)
        assert isinstance(loss_fn, TripletLossWrapper)
        assert loss_fn.margin == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# create_miner factory                                                         #
# --------------------------------------------------------------------------- #

class TestCreateMiner:
    """create_miner factory for all supported miner types."""

    @pytest.mark.unit
    def test_all_returns_none(self) -> None:
        """miner_type='all' returns None (use all pairs, no mining)."""
        assert create_miner("all") is None

    @pytest.mark.unit
    @pytest.mark.parametrize("miner_type", ["hard", "semihard", "multisimilarity"])
    def test_named_types_return_objects(self, miner_type: str) -> None:
        """Supported miner types return a non-None object."""
        miner = create_miner(miner_type)
        assert miner is not None

    @pytest.mark.unit
    def test_unknown_miner_falls_back_to_none(self) -> None:
        """An unknown miner type falls back to None with a warning (no exception)."""
        result = create_miner("random_hard_swapping")
        assert result is None


# --------------------------------------------------------------------------- #
# MultiViewVisionModel API                                                     #
# --------------------------------------------------------------------------- #

class TestMultiViewVisionModelAPI:
    """MultiViewVisionModel class-level API (no model instantiation)."""

    @pytest.mark.unit
    def test_class_exists(self) -> None:
        """MultiViewVisionModel is importable and is a class."""
        from inspect import isclass
        assert isclass(MultiViewVisionModel)

    @pytest.mark.unit
    def test_is_nn_module_subclass(self) -> None:
        """MultiViewVisionModel is a subclass of torch.nn.Module."""
        assert issubclass(MultiViewVisionModel, nn.Module)

    @pytest.mark.unit
    @pytest.mark.parametrize("method_name", [
        "forward",
        "extract_embeddings",
        "extract_val_embeddings",
        "extract_test_embeddings",
        "finetune",
        "evaluate",
        "save_model",
        "load_model",
        "get_model_info",
        "__str__",
        "__repr__",
    ])
    def test_required_methods_exist(self, method_name: str) -> None:
        """Required public methods are defined on the class."""
        assert hasattr(MultiViewVisionModel, method_name), (
            f"MultiViewVisionModel missing method: {method_name}"
        )

    @pytest.mark.unit
    def test_vision_model_error_is_exception(self) -> None:
        """VisionModelError is a proper Exception subclass."""
        assert issubclass(VisionModelError, Exception)

    @pytest.mark.unit
    def test_vision_model_error_can_be_raised(self) -> None:
        """VisionModelError can be raised and caught."""
        with pytest.raises(VisionModelError):
            raise VisionModelError("test error")


# --------------------------------------------------------------------------- #
# MultiViewCLIPModel API                                                       #
# --------------------------------------------------------------------------- #

class TestMultiViewCLIPModelAPI:
    """MultiViewCLIPModel class-level API (no model instantiation)."""

    @pytest.mark.unit
    def test_class_exists(self) -> None:
        """MultiViewCLIPModel is importable and is a class."""
        from inspect import isclass
        assert isclass(MultiViewCLIPModel)

    @pytest.mark.unit
    def test_is_nn_module_subclass(self) -> None:
        """MultiViewCLIPModel is a subclass of torch.nn.Module."""
        assert issubclass(MultiViewCLIPModel, nn.Module)

    @pytest.mark.unit
    @pytest.mark.parametrize("method_name", [
        "freeze_all",
        "unfreeze_text_encoder",
        "unfreeze_projection_layers",
        "unfreeze_vision_final_layers",
        "finetune_phase",
        "extract_embeddings",
        "extract_val_embeddings",
        "extract_test_embeddings",
        "extract_image_embeddings",
        "extract_text_embeddings",
        "extract_joint_embeddings",
        "evaluate",
        "save_model",
        "load_model",
        "get_trainable_params_count",
        "get_model_info",
        "__str__",
        "__repr__",
    ])
    def test_required_methods_exist(self, method_name: str) -> None:
        """All documented public methods are present on the class."""
        assert hasattr(MultiViewCLIPModel, method_name), (
            f"MultiViewCLIPModel missing method: {method_name}"
        )

    @pytest.mark.unit
    def test_clip_model_error_is_exception(self) -> None:
        """CLIPModelError is a proper Exception subclass."""
        assert issubclass(CLIPModelError, Exception)

    @pytest.mark.unit
    def test_clip_model_error_can_be_raised(self) -> None:
        """CLIPModelError can be raised and caught."""
        with pytest.raises(CLIPModelError):
            raise CLIPModelError("test error")

    @pytest.mark.unit
    def test_clip_configs_dict_accessible(self) -> None:
        """CLIP_CONFIGS constant is accessible and non-empty."""
        from src.defaults import CLIP_CONFIGS
        assert isinstance(CLIP_CONFIGS, dict)
        assert len(CLIP_CONFIGS) > 0

    @pytest.mark.unit
    def test_clip_configs_entries_have_required_keys(self) -> None:
        """Every entry in CLIP_CONFIGS has a 'model_name' key."""
        from src.defaults import CLIP_CONFIGS
        for name, cfg in CLIP_CONFIGS.items():
            assert "model_name" in cfg, f"Missing 'model_name' for {name}"
            assert isinstance(cfg["model_name"], str)
            assert len(cfg["model_name"]) > 0


# --------------------------------------------------------------------------- #
# create_vision_model factory                                                  #
# --------------------------------------------------------------------------- #

class TestCreateVisionModelAPI:
    """create_vision_model factory — callable / signature checks only.

    Actual instantiation downloads pretrained torchvision weights and is
    therefore excluded from the unit-test suite.
    """

    @pytest.mark.unit
    def test_factory_is_callable(self) -> None:
        """create_vision_model is a callable."""
        assert callable(create_vision_model)

    @pytest.mark.unit
    def test_factory_in_models_package(self) -> None:
        """create_vision_model is accessible from src.models."""
        import src.models as pkg
        assert hasattr(pkg, "create_vision_model")

    @pytest.mark.unit
    def test_factory_accepts_model_name_param(self) -> None:
        """create_vision_model signature includes a 'model_name' parameter."""
        import inspect
        sig = inspect.signature(create_vision_model)
        assert "model_name" in sig.parameters


# --------------------------------------------------------------------------- #
# create_clip_model factory                                                    #
# --------------------------------------------------------------------------- #

class TestCreateCLIPModelAPI:
    """create_clip_model factory — callable / signature checks only.

    Actual instantiation downloads HuggingFace CLIP weights and is therefore
    excluded from the unit-test suite.
    """

    @pytest.mark.unit
    def test_factory_is_callable(self) -> None:
        """create_clip_model is a callable."""
        assert callable(create_clip_model)

    @pytest.mark.unit
    def test_factory_in_models_package(self) -> None:
        """create_clip_model is accessible from src.models."""
        import src.models as pkg
        assert hasattr(pkg, "create_clip_model")

    @pytest.mark.unit
    def test_factory_accepts_model_name_param(self) -> None:
        """create_clip_model signature includes a 'model_name' parameter."""
        import inspect
        sig = inspect.signature(create_clip_model)
        assert "model_name" in sig.parameters


# --------------------------------------------------------------------------- #
# src.models package exports                                                   #
# --------------------------------------------------------------------------- #

class TestModelsPackageExports:
    """All public symbols in src.models.__all__ are accessible from the package."""

    @pytest.mark.unit
    @pytest.mark.parametrize("symbol", [
        "MultiViewVisionModel",
        "MultiViewCLIPModel",
        "TripletLossWrapper",
        "ContrastiveLossWrapper",
        "ArcFaceLossWrapper",
        "NTXentLossWrapper",
        "MultiSimilarityLossWrapper",
        "CLIPLoss",
        "VisionModelError",
        "CLIPModelError",
        "create_vision_model",
        "create_clip_model",
        "create_metric_learning_loss",
        "create_miner",
    ])
    def test_symbol_accessible(self, symbol: str) -> None:
        """Each exported symbol is accessible as an attribute of src.models."""
        import src.models as pkg
        assert hasattr(pkg, symbol), f"src.models missing export: {symbol}"
