"""
Image-transformation configuration for the CompCars project.

This module provides a dataclass to configure and build torchvision transformation
pipelines. It includes optional augmentation to diversify training images and
improve generalisation, as well as bounding-box cropping support.

Public API:
    TransformConfig: Dataclass that builds and applies a transform pipeline.
    create_standard_transform: Factory function for common transform configurations.
"""

from __future__ import annotations

import ast
import logging
import warnings
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

from torchvision import transforms

from src.defaults import (
    DEFAULT_AUGMENT,
    DEFAULT_COLOR_JITTER_BRIGHTNESS,
    DEFAULT_COLOR_JITTER_CONTRAST,
    DEFAULT_COLOR_JITTER_HUE,
    DEFAULT_COLOR_JITTER_SATURATION,
    DEFAULT_GRAYSCALE,
    DEFAULT_NORMALIZE,
    DEFAULT_RANDOM_ERASING_P,
    DEFAULT_RESIZE,
    DEFAULT_ROTATION_DEGREES,
    DEFAULT_USE_BBOX,
    GRAYSCALE_MEAN,
    GRAYSCALE_STD,
    IMAGENET_MEAN,
    IMAGENET_STD,
)

__all__ = ["TransformConfig", "create_standard_transform"]

warnings.filterwarnings('ignore')
logging.getLogger(__name__).addHandler(logging.NullHandler())


@dataclass
class TransformConfig:
    """Configuration dataclass that builds and applies a torchvision transform pipeline.

    Attributes:
        grayscale: Convert the image to grayscale while keeping 3 output channels
            for compatibility with pretrained PyTorch models.
        resize: ``(height, width)`` target size. No resizing when ``None``.
        normalize: Apply normalisation using ImageNet or grayscale statistics.
        use_bbox: Enable bounding-box cropping. Requires a ``bbox`` argument
            to be passed at call time.
        augment: Apply training augmentation (horizontal flip, rotation, colour
            jitter, Gaussian blur, random erasing).

    Example:
        >>> config = TransformConfig(grayscale=True, resize=(256, 256), use_bbox=True, augment=True)
        >>> bbox = [100, 100, 400, 400]  # [x_min, y_min, x_max, y_max]
        >>> processed_image = config(image, bbox=bbox)
    """
    # Fields — default values come from src.defaults
    grayscale: bool = DEFAULT_GRAYSCALE
    resize: Optional[Tuple[int, int]] = DEFAULT_RESIZE
    normalize: bool = DEFAULT_NORMALIZE
    use_bbox: bool = DEFAULT_USE_BBOX
    augment: bool = DEFAULT_AUGMENT

    def __post_init__(self) -> None:
        """Validate parameters immediately after construction."""
        self._validate_parameters()

    def _validate_parameters(self) -> None:
        """Validate configuration parameter types.

        Raises:
            ValueError: If any parameter has an unexpected type.
        """
        if self.grayscale is not None and not isinstance(self.grayscale, bool):
            raise ValueError("grayscale must be a boolean")

        if self.resize is not None and not isinstance(self.resize, (tuple, list)):
            raise ValueError("resize must be a tuple or list of two integers")

        if self.normalize is not None and not isinstance(self.normalize, bool):
            raise ValueError("normalize must be a boolean")

        if self.use_bbox is not None and not isinstance(self.use_bbox, bool):
            raise ValueError("use_bbox must be a boolean")

        if self.augment is not None and not isinstance(self.augment, bool):
            raise ValueError("augment must be a boolean")
        
    def _get_transforms(self) -> transforms.Compose:
        """Build and return the composed transform pipeline.

        Returns:
            A ``transforms.Compose`` object with all configured steps.
        """
        transform_list = []

        # Convert to grayscale (3 output channels for pretrained-model compatibility)
        if self.grayscale:
            transform_list.append(transforms.Grayscale(num_output_channels=3))
        
        # Resize
        if self.resize is not None:
            transform_list.append(transforms.Resize(self.resize))

        # Training augmentation
        if self.augment:
            transform_list.extend([
                transforms.RandomHorizontalFlip(p=0.3),
                # Random rotation
                transforms.RandomApply(
                    [transforms.RandomRotation(
                        degrees=DEFAULT_ROTATION_DEGREES,
                        interpolation=transforms.InterpolationMode.BILINEAR,
                    )],
                    p=0.3,
                ),
                # Random affine translation + scale
                transforms.RandomApply(
                    [transforms.RandomAffine(
                        degrees=0,
                        translate=(0.03, 0.03),
                        scale=(0.97, 1.03),
                        shear=None,
                        interpolation=transforms.InterpolationMode.BILINEAR,
                    )],
                    p=0.3,
                ),
                # Colour jitter
                transforms.RandomApply(
                    [transforms.ColorJitter(
                        brightness=DEFAULT_COLOR_JITTER_BRIGHTNESS,
                        contrast=DEFAULT_COLOR_JITTER_CONTRAST,
                        saturation=DEFAULT_COLOR_JITTER_SATURATION,
                        hue=DEFAULT_COLOR_JITTER_HUE,
                    )],
                    p=0.5,
                ),
                # Gaussian blur
                transforms.RandomApply(
                    [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5))],
                    p=0.3,
                ),
            ])
        
        # Convert to tensor 
        transform_list.append(transforms.ToTensor())

        # Random erasing 
        if self.augment:
            transform_list.append(
                transforms.RandomErasing(
                    p=DEFAULT_RANDOM_ERASING_P,
                    scale=(0.02, 0.08),
                    ratio=(0.5, 2.0),
                    value='random',
                )
            )

        # Normalisation
        if self.normalize:
            if self.grayscale:
                transform_list.append(
                    transforms.Normalize(mean=GRAYSCALE_MEAN, std=GRAYSCALE_STD)
                )
            else:
                transform_list.append(
                    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
                )

        return transforms.Compose(transform_list)
    
    def _apply_bbox_crop(self, image: Any, bbox: Union[list, str]) -> Any:
        """Crop an image using a bounding box.

        Args:
            image: PIL Image to crop.
            bbox: Bounding box ``[x_min, y_min, x_max, y_max]`` as a list or
                a string representation of one.

        Returns:
            Cropped PIL Image when the bbox is valid; the original image otherwise.
        """
        try:
            if image is None:
                logging.warning("Image is None — skipping bbox crop")
                return image

            # Parse bbox from string when necessary
            if isinstance(bbox, str):
                try:
                    bbox = ast.literal_eval(bbox)
                except (ValueError, SyntaxError) as exc:
                    logging.warning(
                        f"Could not parse bbox string '{bbox}': {exc} — using full image"
                    )
                    return image

            if not isinstance(bbox, (list, tuple)):
                logging.warning(
                    f"bbox must be a list or tuple, got {type(bbox)} — using full image"
                )
                return image

            if len(bbox) != 4:
                logging.warning(
                    f"bbox must have 4 coordinates, got {len(bbox)} in {bbox} — using full image"
                )
                return image

            try:
                x_min, y_min, x_max, y_max = (float(v) for v in bbox)
            except (ValueError, TypeError) as exc:
                logging.warning(
                    f"Could not convert bbox coordinates {bbox}: {exc} — using full image"
                )
                return image

            try:
                img_width, img_height = image.size
            except AttributeError:
                logging.warning(
                    f"Image has no 'size' attribute (type: {type(image)}) — using full image"
                )
                return image

            # Clamp coordinates to image boundaries
            x_min = max(0, min(x_min, img_width - 1))
            y_min = max(0, min(y_min, img_height - 1))
            x_max = max(x_min + 1, min(x_max, img_width))   
            y_max = max(y_min + 1, min(y_max, img_height))  

            if x_min >= x_max or y_min >= y_max:
                logging.warning(
                    f"Invalid bbox after clamping — original={bbox}, "
                    f"adjusted=[{x_min}, {y_min}, {x_max}, {y_max}] — using full image"
                )
                return image

            return image.crop((int(x_min), int(y_min), int(x_max), int(y_max)))

        except Exception as exc:
            logging.warning(
                f"Unexpected error during bbox crop (bbox={bbox}): "
                f"{type(exc).__name__}: {exc} — using full image"
            )
            return image
    
    def __call__(self, image: Any, bbox: Optional[Union[list, str]] = None) -> Any:
        """Apply the configured transform pipeline to an image.

        Args:
            image: Input image (PIL Image, tensor, etc.).
            bbox: Optional bounding box ``[x_min, y_min, x_max, y_max]`` for
                cropping. Accepts a list or a string ``"[x1, y1, x2, y2]"``.

        Returns:
            Transformed image tensor.

        Raises:
            RuntimeError: If a critical error prevents the image from being
                processed (e.g. a corrupted image file).
        """
        try:
            if self.use_bbox and bbox is not None:
                image = self._apply_bbox_crop(image, bbox)

            transform = self._get_transforms()
            return transform(image)

        except Exception as exc:
            error_msg = (
                f"Critical error while applying transforms: {type(exc).__name__}: {exc}. "
                f"Config: grayscale={self.grayscale}, resize={self.resize}, "
                f"normalize={self.normalize}, use_bbox={self.use_bbox}, augment={self.augment}"
            )
            logging.error(error_msg)
            raise RuntimeError(error_msg) from exc
    
    def __repr__(self) -> str:
        """Return an unambiguous string representation for debugging."""
        return (
            f"{self.__class__.__name__}("
            f"grayscale={self.grayscale}, "
            f"resize={self.resize}, "
            f"normalize={self.normalize}, "
            f"use_bbox={self.use_bbox}, "
            f"augment={self.augment})"
        )


def create_standard_transform(
    size: Tuple[int, int] = DEFAULT_RESIZE,
    grayscale: bool = DEFAULT_GRAYSCALE,
    use_bbox: bool = DEFAULT_USE_BBOX,
    augment: bool = DEFAULT_AUGMENT,
) -> TransformConfig:
    """Create a standard :class:`TransformConfig` with sensible defaults.

    Args:
        size: Target ``(height, width)`` for resizing.
        grayscale: Whether to convert images to grayscale.
        use_bbox: Whether to enable bounding-box cropping.
        augment: Whether to apply training augmentation.

    Returns:
        A ready-to-use :class:`TransformConfig` instance.
    """
    return TransformConfig(
        grayscale=grayscale,
        resize=size,
        normalize=True,
        use_bbox=use_bbox,
        augment=augment,
    )