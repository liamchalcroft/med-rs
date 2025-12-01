"""Python bindings and utilities for medrs."""

from importlib import import_module
from importlib.util import find_spec

from ._medrs import (
    NiftiImage,
    PyTrainingDataLoader as TrainingDataLoader,
    TransformPipeline,
    clamp,
    crop_or_pad,
    load,
    load_cropped,
    load_cropped_to_jax,
    load_cropped_to_torch,
    load_label_aware_cropped,
    load_resampled,
    load_to_torch,
    reorient,
    resample,
    rescale_intensity,
    z_normalization,
    # Random augmentation functions
    random_flip,
    random_gaussian_noise,
    random_intensity_scale,
    random_intensity_shift,
    random_rotate_90,
    random_gamma,
    random_augment,
    # Crop region functions
    compute_crop_regions,
    compute_random_spatial_crops,
    compute_center_crop,
)
MedicalImage = NiftiImage
# Keep PyTrainingDataLoader as alias for backwards compatibility
PyTrainingDataLoader = TrainingDataLoader
from .exceptions import (
    LoadError,
    MedrsError,
    MemoryError,
    TransformError,
    ValidationError,
)
from .performance_profiler import PerformanceProfiler

__version__ = "0.1.0"
__author__ = "Liam Chalcroft"
__email__ = "liam.chalcroft.20@ucl.ac.uk"

__all__ = [
    # Image classes
    "MedicalImage",
    "NiftiImage",
    # Data loaders
    "TrainingDataLoader",
    "PyTrainingDataLoader",  # Backwards compatibility alias
    # Transform pipeline
    "TransformPipeline",
    # Basic transforms
    "clamp",
    "crop_or_pad",
    "reorient",
    "resample",
    "rescale_intensity",
    "z_normalization",
    # I/O functions
    "load",
    "load_cropped",
    "load_cropped_to_jax",
    "load_cropped_to_torch",
    "load_label_aware_cropped",
    "load_resampled",
    "load_to_torch",
    # Random augmentation
    "random_flip",
    "random_gaussian_noise",
    "random_intensity_scale",
    "random_intensity_shift",
    "random_rotate_90",
    "random_gamma",
    "random_augment",
    # Crop region functions
    "compute_crop_regions",
    "compute_random_spatial_crops",
    "compute_center_crop",
    # Exceptions
    "LoadError",
    "MedrsError",
    "MemoryError",
    "TransformError",
    "ValidationError",
    # Utilities
    "PerformanceProfiler",
]


def _load_optional(module: str, names: list[str]) -> None:
    """Import optional submodules when their dependencies are present."""
    try:
        if find_spec(f"{__name__}.{module}") is None:
            return
        mod = import_module(f"{__name__}.{module}")
        globals().update({name: getattr(mod, name) for name in names})
        __all__.extend(names)
    except ModuleNotFoundError:
        # Optional dependency not installed; skip exposing these helpers.
        return


_load_optional(
    "dictionary_transforms",
    [
        "SpatialNormalizer",
        "CoordinatedCropLoader",
        "MonaiCompatibleTransform",
        "create_multimodal_crop_transform",
        "create_monai_compatible_crop",
    ],
)

_load_optional(
    "metatensor_support",
    [
        "MedrsMetaTensorConverter",
        "MetaTensorLoader",
        "MetaTensorCoordinatedCropLoader",
        "MetaTensorCompatibleTransform",
        "create_metatensor_loader",
        "create_metatensor_crop_transform",
        "metatensor_from_medrs",
        "is_metatensor_supported",
        "enhance_dictionary_transforms_for_metatensor",
    ],
)
