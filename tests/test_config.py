"""
Tests for ``src.config`` — TransformConfig and create_standard_transform.

Coverage:
    TransformConfig: default attribute values, parameter validation,
    ``__call__`` on a real dataset image, bounding-box crop, grayscale
    conversion, augmentation, and ``__repr__``.
    create_standard_transform: return type, custom size, and application
    to a real image.

Markers:
    unit        — tests that only exercise Python / PyTorch logic (no I/O).
    integration — tests that open a real image from the dataset.
"""

from __future__ import annotations

__all__: list[str] = []

import pytest
import torch
from PIL import Image

from src.config import TransformConfig, create_standard_transform
from src.defaults import (
    DEFAULT_AUGMENT,
    DEFAULT_GRAYSCALE,
    DEFAULT_NORMALIZE,
    DEFAULT_RESIZE,
    DEFAULT_USE_BBOX,
)


# --------------------------------------------------------------------------- #
# TestTransformConfigDefaults                                                  #
# --------------------------------------------------------------------------- #

class TestTransformConfigDefaults:
    """Verify that a zero-argument ``TransformConfig()`` mirrors ``src.defaults``."""

    @pytest.mark.unit
    def test_grayscale_default(self) -> None:
        """grayscale attribute matches DEFAULT_GRAYSCALE."""
        assert TransformConfig().grayscale == DEFAULT_GRAYSCALE

    @pytest.mark.unit
    def test_resize_default(self) -> None:
        """resize attribute matches DEFAULT_RESIZE."""
        assert TransformConfig().resize == DEFAULT_RESIZE

    @pytest.mark.unit
    def test_normalize_default(self) -> None:
        """normalize attribute matches DEFAULT_NORMALIZE."""
        assert TransformConfig().normalize == DEFAULT_NORMALIZE

    @pytest.mark.unit
    def test_use_bbox_default(self) -> None:
        """use_bbox attribute matches DEFAULT_USE_BBOX."""
        assert TransformConfig().use_bbox == DEFAULT_USE_BBOX

    @pytest.mark.unit
    def test_augment_default(self) -> None:
        """augment attribute matches DEFAULT_AUGMENT."""
        assert TransformConfig().augment == DEFAULT_AUGMENT


# --------------------------------------------------------------------------- #
# TestTransformConfigValidation                                                #
# --------------------------------------------------------------------------- #

class TestTransformConfigValidation:
    """Parameter validation: wrong types must raise ``ValueError``."""

    @pytest.mark.unit
    def test_invalid_grayscale_type_raises(self) -> None:
        """Passing a non-bool for grayscale raises ValueError."""
        with pytest.raises(ValueError, match="grayscale"):
            TransformConfig(grayscale="yes")  # type: ignore[arg-type]

    @pytest.mark.unit
    def test_invalid_resize_type_raises(self) -> None:
        """Passing a plain int instead of a tuple for resize raises ValueError."""
        with pytest.raises(ValueError, match="resize"):
            TransformConfig(resize=224)  # type: ignore[arg-type]

    @pytest.mark.unit
    def test_invalid_normalize_type_raises(self) -> None:
        """Passing a non-bool for normalize raises ValueError."""
        with pytest.raises(ValueError, match="normalize"):
            TransformConfig(normalize=1)  # type: ignore[arg-type]

    @pytest.mark.unit
    def test_invalid_use_bbox_type_raises(self) -> None:
        """Passing a non-bool for use_bbox raises ValueError."""
        with pytest.raises(ValueError, match="use_bbox"):
            TransformConfig(use_bbox="true")  # type: ignore[arg-type]

    @pytest.mark.unit
    def test_invalid_augment_type_raises(self) -> None:
        """Passing a non-bool for augment raises ValueError."""
        with pytest.raises(ValueError, match="augment"):
            TransformConfig(augment=0)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# TestTransformConfigCall                                                      #
# --------------------------------------------------------------------------- #

class TestTransformConfigCall:
    """``__call__`` on a real dataset image returns a correctly shaped tensor."""

    @pytest.mark.integration
    def test_returns_torch_tensor(self, sample_image: Image.Image) -> None:
        """Applying the default config to a real image returns a torch.Tensor."""
        config = TransformConfig()
        result = config(sample_image)
        assert isinstance(result, torch.Tensor)

    @pytest.mark.integration
    def test_output_shape_channels_height_width(self, sample_image: Image.Image) -> None:
        """Output tensor has shape (3, H, W) matching the configured resize."""
        h, w = DEFAULT_RESIZE
        config = TransformConfig()
        result = config(sample_image)
        assert result.shape == (3, h, w), f"Expected (3, {h}, {w}), got {result.shape}"

    @pytest.mark.integration
    def test_output_dtype_is_float(self, sample_image: Image.Image) -> None:
        """Output tensor has a floating-point dtype (normalise is on by default)."""
        result = TransformConfig()(sample_image)
        assert result.is_floating_point()

    @pytest.mark.integration
    def test_grayscale_still_produces_3_channels(self, sample_image: Image.Image) -> None:
        """grayscale=True keeps 3 output channels for pretrained-model compatibility."""
        config = TransformConfig(grayscale=True, resize=(64, 64))
        result = config(sample_image)
        assert result.shape[0] == 3

    @pytest.mark.integration
    def test_no_resize_preserves_image_dimensions(self, sample_image: Image.Image) -> None:
        """resize=None skips resizing; the spatial dimensions match the source image."""
        config = TransformConfig(resize=None, normalize=False, augment=False)
        result = config(sample_image)
        w, h = sample_image.size
        assert result.shape == (3, h, w)

    @pytest.mark.integration
    def test_augment_still_produces_correct_shape(self, sample_image: Image.Image) -> None:
        """augment=True applies extra transforms but still returns the right shape."""
        h, w = DEFAULT_RESIZE
        config = TransformConfig(augment=True)
        result = config(sample_image)
        assert result.shape == (3, h, w)


# --------------------------------------------------------------------------- #
# TestTransformConfigBbox                                                      #
# --------------------------------------------------------------------------- #

class TestTransformConfigBbox:
    """Bounding-box crop behaviour."""

    @pytest.mark.integration
    def test_bbox_crop_applied_when_use_bbox_is_true(
        self, sample_image: Image.Image
    ) -> None:
        """use_bbox=True with a valid bbox crops the image before transforms."""
        # Use a bbox that covers the central 50 % of the image.
        w, h = sample_image.size
        bbox = [w * 0.25, h * 0.25, w * 0.75, h * 0.75]

        config_no_bbox = TransformConfig(
            use_bbox=False, resize=(128, 128), normalize=False, augment=False
        )
        config_bbox = TransformConfig(
            use_bbox=True, resize=(128, 128), normalize=False, augment=False
        )

        result_no_bbox = config_no_bbox(sample_image)
        result_bbox = config_bbox(sample_image, bbox=bbox)

        # Both are tensors with the same output shape (resize normalises them).
        assert result_no_bbox.shape == result_bbox.shape
        # But their pixel values should differ since one was cropped.
        assert not torch.allclose(result_no_bbox, result_bbox), (
            "Cropped and uncropped results should differ"
        )

    @pytest.mark.integration
    def test_bbox_string_format_accepted(self, sample_image: Image.Image) -> None:
        """Bbox can be passed as a string representation of a list."""
        w, h = sample_image.size
        bbox_str = f"[0, 0, {w}, {h}]"  # full-image bbox as a string

        config = TransformConfig(use_bbox=True, resize=(64, 64), normalize=False)
        result = config(sample_image, bbox=bbox_str)
        assert isinstance(result, torch.Tensor)

    @pytest.mark.integration
    def test_malformed_bbox_falls_back_gracefully(
        self, sample_image: Image.Image
    ) -> None:
        """A malformed bbox is silently ignored and the full image is used."""
        config = TransformConfig(use_bbox=True, resize=(64, 64), normalize=False)
        result = config(sample_image, bbox="not_a_valid_bbox")
        assert isinstance(result, torch.Tensor)

    @pytest.mark.integration
    def test_bbox_ignored_when_use_bbox_is_false(
        self, sample_image: Image.Image
    ) -> None:
        """Passing a bbox when use_bbox=False has no effect on the output."""
        config_no_flag = TransformConfig(
            use_bbox=False, resize=(64, 64), normalize=False, augment=False
        )
        config_with_flag = TransformConfig(
            use_bbox=False, resize=(64, 64), normalize=False, augment=False
        )
        w, h = sample_image.size
        bbox = [0, 0, w // 2, h // 2]

        result_no_bbox = config_no_flag(sample_image)
        result_with_bbox = config_with_flag(sample_image, bbox=bbox)

        # Since use_bbox=False, the bbox is ignored and results must be identical.
        assert torch.allclose(result_no_bbox, result_with_bbox)


# --------------------------------------------------------------------------- #
# TestTransformConfigRepr                                                      #
# --------------------------------------------------------------------------- #

class TestTransformConfigRepr:
    """``__repr__`` returns a useful string for debugging."""

    @pytest.mark.unit
    def test_repr_contains_class_name(self) -> None:
        """repr starts with the class name."""
        r = repr(TransformConfig())
        assert "TransformConfig" in r

    @pytest.mark.unit
    def test_repr_contains_grayscale(self) -> None:
        """repr includes the grayscale flag."""
        r = repr(TransformConfig(grayscale=True))
        assert "grayscale=True" in r

    @pytest.mark.unit
    def test_repr_contains_resize(self) -> None:
        """repr includes the resize parameter."""
        r = repr(TransformConfig(resize=(128, 128)))
        assert "resize=(128, 128)" in r


# --------------------------------------------------------------------------- #
# TestCreateStandardTransform                                                  #
# --------------------------------------------------------------------------- #

class TestCreateStandardTransform:
    """Factory function ``create_standard_transform``."""

    @pytest.mark.unit
    def test_returns_transform_config_instance(self) -> None:
        """Return type is TransformConfig."""
        assert isinstance(create_standard_transform(), TransformConfig)

    @pytest.mark.unit
    def test_normalize_is_always_true(self) -> None:
        """The factory always enables normalisation."""
        assert create_standard_transform().normalize is True

    @pytest.mark.unit
    def test_custom_size_stored(self) -> None:
        """Custom size argument is reflected in the returned config."""
        cfg = create_standard_transform(size=(64, 64))
        assert cfg.resize == (64, 64)

    @pytest.mark.unit
    def test_grayscale_forwarded(self) -> None:
        """grayscale argument is forwarded to the TransformConfig."""
        assert create_standard_transform(grayscale=True).grayscale is True

    @pytest.mark.unit
    def test_use_bbox_forwarded(self) -> None:
        """use_bbox argument is forwarded to the TransformConfig."""
        assert create_standard_transform(use_bbox=True).use_bbox is True

    @pytest.mark.unit
    def test_augment_forwarded(self) -> None:
        """augment argument is forwarded to the TransformConfig."""
        assert create_standard_transform(augment=True).augment is True

    @pytest.mark.integration
    def test_applies_to_real_image(self, sample_image: Image.Image) -> None:
        """The returned config can be applied to a real PIL image."""
        result = create_standard_transform()(sample_image)
        assert isinstance(result, torch.Tensor)
        assert result.shape[0] == 3
