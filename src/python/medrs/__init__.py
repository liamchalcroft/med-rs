"""High-performance medical imaging I/O and processing library.

medrs is designed for throughput-critical medical imaging workflows,
particularly deep learning pipelines that process large 3D volumes.

Key Features:
    - **Fast NIfTI I/O**: Memory-mapped reading, crop-first loading
      (read sub-volumes without loading entire files - up to 40x faster)
    - **Mixed Precision**: Native f16/bf16 support for 50% storage savings
    - **Transform Pipeline**: Lazy evaluation with SIMD acceleration
    - **Direct Tensor Creation**: Zero-copy PyTorch/JAX tensor loading
    - **MONAI Integration**: Drop-in replacements for MONAI transforms

Quick Start:
    >>> import medrs
    >>> img = medrs.load("brain.nii.gz")
    >>> processed = img.resample([1.0, 1.0, 1.0]).z_normalize().clamp(0, 1)
    >>> processed.save("output.nii.gz")

    # Load directly to PyTorch (most efficient)
    >>> import torch
    >>> tensor = medrs.load_to_torch("brain.nii.gz", dtype=torch.float16, device="cuda")

    # Crop-first loading (essential for training pipelines)
    >>> patch = medrs.load_cropped("volume.nii", [32, 32, 32], [64, 64, 64])

Performance Tips:
    1. Use `load_cropped()` or `load_cropped_to_torch()` for training
    2. Use `.with_dtype("bfloat16")` for 50% smaller files
    3. Use `.materialize()` before multiple transforms
    4. Use `TransformPipeline` for fused operations

For MONAI integration, see:
    >>> from medrs import monai_compat  # Drop-in replacements
    >>> from medrs import metatensor_support  # MetaTensor conversion
"""

from importlib import import_module
from importlib.util import find_spec
from typing import Any, Optional

from ._medrs import (
    NiftiImage,
    TrainingDataLoader,
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

# Alias for more intuitive naming
MedicalImage = NiftiImage

from .exceptions import (
    ConfigurationError,
    DeviceError,
    LoadError,
    MedrsError,
    MemoryError,
    TransformError,
    ValidationError,
)
from .performance_profiler import PerformanceProfiler

__version__ = "0.1.1"
__author__ = "Liam Chalcroft"
__email__ = "liam.chalcroft.20@ucl.ac.uk"


# Convenience functions

def get_info(path: str) -> dict[str, Any]:
    """Get image metadata without loading the full volume.

    This is useful for quickly inspecting file properties without
    the memory cost of loading all voxel data.

    Args:
        path: Path to NIfTI file

    Returns:
        Dictionary with keys: shape, spacing, affine, orientation, dtype

    Example:
        >>> info = medrs.get_info("brain.nii.gz")
        >>> print(f"Shape: {info['shape']}, Spacing: {info['spacing']}")
    """
    img = load(path)
    return {
        "shape": img.shape,
        "spacing": img.spacing,
        "affine": img.affine,
        "orientation": img.orientation,
        "dtype": img.dtype,
    }


def supports_monai() -> bool:
    """Check if MONAI integration is available.

    Returns True if MONAI is installed and the medrs MONAI
    compatibility modules are available.

    Returns:
        True if MONAI integration is available

    Example:
        >>> if medrs.supports_monai():
        ...     from medrs import monai_compat
        ...     loader = monai_compat.MedrsLoadImaged(keys=["image"])
    """
    try:
        import monai  # noqa: F401
        return find_spec("medrs.metatensor_support") is not None
    except ImportError:
        return False


def supports_torch() -> bool:
    """Check if PyTorch integration is available.

    Returns:
        True if PyTorch is installed
    """
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def supports_jax() -> bool:
    """Check if JAX integration is available.

    Returns:
        True if JAX is installed
    """
    try:
        import jax  # noqa: F401
        return True
    except ImportError:
        return False

__all__ = [
    # Image classes
    "MedicalImage",
    "NiftiImage",
    # Data loaders
    "TrainingDataLoader",
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
    # Convenience functions
    "get_info",
    "supports_monai",
    "supports_torch",
    "supports_jax",
    # Exceptions
    "ConfigurationError",
    "DeviceError",
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

_load_optional(
    "monai_compat",
    [
        # Load transforms (drop-in replacements)
        "MedrsLoadImage",
        "MedrsLoadImaged",
        # Save transforms
        "MedrsSaveImage",
        "MedrsSaveImaged",
        # Crop transforms (crop-first loading)
        "MedrsRandCropByPosNegLabeld",
        "MedrsRandSpatialCropd",
        "MedrsCenterSpatialCropd",
        # Spatial transforms
        "MedrsOrientation",
        "MedrsOrientationd",
        # Resample transforms
        "MedrsResample",
        "MedrsResampled",
    ],
)
