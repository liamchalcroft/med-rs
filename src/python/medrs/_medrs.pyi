"""Type stubs for the medrs Rust extension module.

This module provides high-performance medical imaging I/O and processing,
optimized for deep learning pipelines. Key features:

- **Fast NIfTI I/O**: Memory-mapped reading, crop-first loading
- **Transform Pipeline**: Lazy evaluation with SIMD acceleration
- **Direct Tensor Creation**: Zero-copy PyTorch/JAX tensor loading
- **Mixed Precision**: Native f16/bf16 support for 50% storage savings

Example:
    >>> import medrs
    >>> img = medrs.load("brain.nii.gz")
    >>> processed = img.resample([1.0, 1.0, 1.0]).z_normalize().clamp(0, 1)
    >>> processed.save("output.nii.gz")
"""

from typing import Any, Iterator, Literal, Optional, Sequence, TypeVar, overload

import numpy as np
import numpy.typing as npt

# Type aliases
Affine = tuple[
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    tuple[float, float, float, float],
]
AffineArray = list[list[float]]
Shape3D = tuple[int, int, int] | list[int] | Sequence[int]
Spacing3D = tuple[float, float, float] | list[float] | Sequence[float]
DType = Literal[
    "float32", "f32",
    "float64", "f64",
    "float16", "f16",
    "bfloat16", "bf16",
    "int8", "i8",
    "uint8", "u8",
    "int16", "i16",
    "uint16", "u16",
    "int32", "i32",
    "uint32", "u32",
    "int64", "i64",
    "uint64", "u64",
]
Orientation = Literal[
    "RAS", "LAS", "RPS", "LPS", "RSA", "LSA", "RPA", "LPA",
    "ARS", "ALS", "PRS", "PLS", "ASR", "ASL", "PSR", "PSL",
    "SAR", "SAL", "SPR", "SPL", "IAR", "IAL", "IPR", "IPL",
    "SRA", "SLA", "IRA", "ILA", "SRP", "SLP", "IRP", "ILP",
    "RAI", "LAI", "RPI", "LPI", "RSI", "LSI", "RII", "LII",
    "ARI", "ALI", "PRI", "PLI", "ASI", "ALI", "PSI", "PLI",
]
InterpolationMethod = Literal["trilinear", "linear", "nearest"]


class NiftiImage:
    """A NIfTI medical image with header metadata and voxel data.

    Supports method chaining for transform operations, enabling fluent APIs:

        >>> img = medrs.load("brain.nii.gz")
        >>> processed = img.resample([1.0, 1.0, 1.0]).z_normalize().clamp(0, 1)
        >>> processed.save("output.nii.gz")

    Attributes:
        shape: Image dimensions as (depth, height, width)
        ndim: Number of dimensions (typically 3 or 4)
        dtype: Data type string (e.g., "f32", "bf16", "i16")
        spacing: Voxel spacing in mm as [x, y, z]
        affine: 4x4 affine transformation matrix
        orientation: Orientation code (e.g., "RAS", "LPS")
        data: Raw voxel data as numpy array (float32)
    """

    def __init__(
        self,
        data: npt.NDArray[np.float32],
        affine: Optional[AffineArray] = None,
    ) -> None:
        """Create a NIfTI image from a numpy array.

        Args:
            data: Numpy array with at least 3 dimensions (D, H, W)
            affine: Optional 4x4 affine transformation matrix
        """
        ...

    @staticmethod
    def from_numpy(
        data: npt.NDArray[np.float32],
        affine: Optional[AffineArray] = None,
    ) -> "NiftiImage":
        """Create a NIfTI image from a numpy array.

        This is a convenience factory method equivalent to the constructor.

        Args:
            data: Numpy array with at least 3 dimensions (D, H, W)
            affine: Optional 4x4 affine transformation matrix

        Returns:
            New NiftiImage instance

        Example:
            >>> import numpy as np
            >>> data = np.random.rand(64, 64, 32).astype(np.float32)
            >>> img = NiftiImage.from_numpy(data)
        """
        ...

    # Properties
    @property
    def shape(self) -> list[int]:
        """Image dimensions as [depth, height, width]."""
        ...

    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        ...

    @property
    def dtype(self) -> str:
        """Data type as string (e.g., 'f32', 'bf16', 'i16')."""
        ...

    @property
    def spacing(self) -> list[float]:
        """Voxel spacing in mm as [x, y, z]."""
        ...

    @property
    def affine(self) -> list[list[float]]:
        """4x4 affine transformation matrix."""
        ...

    @affine.setter
    def affine(self, value: AffineArray) -> None:
        """Set the affine transformation matrix."""
        ...

    @property
    def orientation(self) -> str:
        """Orientation code (e.g., 'RAS', 'LPS')."""
        ...

    @property
    def data(self) -> npt.NDArray[np.float32]:
        """Raw voxel data as numpy array (float32)."""
        ...

    # Data conversion methods
    def to_numpy(self) -> npt.NDArray[np.float32]:
        """Get image data as float32 numpy array.

        Similar to nibabel's get_fdata(). Applies scaling factors if present.

        Returns:
            Numpy array with shape matching self.shape
        """
        ...

    def to_numpy_view(self) -> npt.NDArray[np.float32]:
        """Get image data as numpy view when possible.

        Attempts zero-copy access; falls back to copy if not contiguous.

        Returns:
            Numpy array (view or copy)
        """
        ...

    def to_numpy_native(self) -> npt.NDArray[Any]:
        """Get image data with native dtype.

        Half/bfloat16 are returned as float32 for numpy compatibility.

        Returns:
            Numpy array with native dtype
        """
        ...

    def to_torch(self) -> Any:
        """Convert to PyTorch tensor.

        Shares memory when possible (contiguous data).

        Returns:
            torch.Tensor with shape matching self.shape

        Raises:
            ImportError: If PyTorch is not installed
        """
        ...

    def to_torch_with_dtype_and_device(
        self,
        dtype: Optional[Any] = None,
        device: Optional[str] = None,
    ) -> Any:
        """Convert to PyTorch tensor with custom dtype and device.

        This is the most efficient way to load data into PyTorch with
        target precision and device placement.

        Args:
            dtype: PyTorch dtype (e.g., torch.float16)
            device: Device string (e.g., "cuda", "cuda:0", "cpu")

        Returns:
            torch.Tensor on specified device with specified dtype

        Example:
            >>> import torch
            >>> tensor = img.to_torch_with_dtype_and_device(
            ...     dtype=torch.float16, device="cuda"
            ... )
        """
        ...

    def to_jax(self) -> Any:
        """Convert to JAX array.

        Shares memory via numpy when possible.

        Returns:
            jax.Array with shape matching self.shape

        Raises:
            ImportError: If JAX is not installed
        """
        ...

    def to_jax_with_dtype_and_device(
        self,
        dtype: Optional[Any] = None,
        device: Optional[str] = None,
    ) -> Any:
        """Convert to JAX array with custom dtype and device.

        Args:
            dtype: JAX dtype (e.g., jax.numpy.bfloat16)
            device: Device string (e.g., "cuda:0", "cpu")

        Returns:
            jax.Array on specified device with specified dtype
        """
        ...

    # Transform methods (all support method chaining)
    def resample(
        self,
        spacing: Spacing3D,
        method: Optional[InterpolationMethod] = None,
    ) -> "NiftiImage":
        """Resample to target voxel spacing.

        Args:
            spacing: Target spacing as [x, y, z] in mm
            method: Interpolation method ("trilinear" or "nearest")

        Returns:
            New NiftiImage with resampled data
        """
        ...

    def resample_to_shape(
        self,
        shape: Shape3D,
        method: Optional[InterpolationMethod] = None,
    ) -> "NiftiImage":
        """Resample to target shape.

        Args:
            shape: Target shape as [depth, height, width]
            method: Interpolation method ("trilinear" or "nearest")

        Returns:
            New NiftiImage with resampled data
        """
        ...

    def reorient(self, orientation: Orientation) -> "NiftiImage":
        """Reorient to target orientation.

        Args:
            orientation: Target orientation code (e.g., "RAS", "LPS")

        Returns:
            New NiftiImage with reoriented data
        """
        ...

    def z_normalize(self) -> "NiftiImage":
        """Z-score normalization (zero mean, unit variance).

        Returns:
            New NiftiImage with normalized data
        """
        ...

    def rescale(self, out_min: float, out_max: float) -> "NiftiImage":
        """Rescale intensity to range [out_min, out_max].

        Args:
            out_min: Minimum output value
            out_max: Maximum output value

        Returns:
            New NiftiImage with rescaled data
        """
        ...

    def clamp(self, min: float, max: float) -> "NiftiImage":
        """Clamp intensity values to range [min, max].

        Args:
            min: Minimum value
            max: Maximum value

        Returns:
            New NiftiImage with clamped data
        """
        ...

    def crop_or_pad(self, target_shape: Shape3D) -> "NiftiImage":
        """Crop or pad to target shape.

        Args:
            target_shape: Target shape as [depth, height, width]

        Returns:
            New NiftiImage with cropped/padded data
        """
        ...

    def flip(self, axes: Sequence[int]) -> "NiftiImage":
        """Flip along specified axes.

        Args:
            axes: List of axes to flip (0=depth, 1=height, 2=width)

        Returns:
            New NiftiImage with flipped data
        """
        ...

    def with_dtype(self, dtype: DType) -> "NiftiImage":
        """Convert to a different data type.

        Useful for reducing file size when saving. Converting from float32
        to bfloat16 reduces storage by 50%.

        Args:
            dtype: Target dtype string (e.g., "bfloat16", "float16")

        Returns:
            New NiftiImage with converted dtype

        Example:
            >>> img_bf16 = img.with_dtype("bfloat16")
            >>> img_bf16.save("volume_bf16.nii.gz")  # 50% smaller
        """
        ...

    def materialize(self) -> "NiftiImage":
        """Convert mmap'd data to owned memory.

        Call this before running multiple transforms to avoid
        re-materializing data on each transform.

        Returns:
            New NiftiImage with data in memory
        """
        ...

    def is_materialized(self) -> bool:
        """Check if image data is already in memory.

        Returns:
            True if data is in memory, False if mmap'd from disk
        """
        ...

    def save(self, path: str) -> None:
        """Save image to file.

        Format determined by extension (.nii or .nii.gz).

        Args:
            path: Output file path
        """
        ...

    def __repr__(self) -> str: ...


# Alias for MedicalImage
MedicalImage = NiftiImage


class TrainingDataLoader:
    """High-performance training data loader with prefetching and caching.

    Optimized for loading training patches from multiple volumes with
    LRU caching and prefetching for maximum throughput.

    Example:
        >>> loader = medrs.TrainingDataLoader(
        ...     volumes=["vol1.nii", "vol2.nii"],
        ...     patch_size=[64, 64, 64],
        ...     patches_per_volume=4,
        ...     patch_overlap=[0, 0, 0],
        ...     randomize=True,
        ...     cache_size=1000
        ... )
        >>> for patch in loader:
        ...     tensor = patch.to_torch()
    """

    def __init__(
        self,
        volumes: list[str],
        patch_size: Shape3D,
        patches_per_volume: int,
        patch_overlap: Shape3D,
        randomize: bool,
        cache_size: Optional[int] = None,
    ) -> None:
        """Create a training data loader.

        Args:
            volumes: List of NIfTI file paths
            patch_size: Patch size to extract [d, h, w]
            patches_per_volume: Number of patches per volume
            patch_overlap: Overlap between patches [d, h, w] in voxels
            randomize: Whether to randomize patch positions
            cache_size: Maximum number of patches to cache (default: 1000)
        """
        ...

    def next_patch(self) -> NiftiImage:
        """Get next training patch.

        Returns:
            Next patch with automatic prefetching

        Raises:
            StopIteration: When all patches are processed
        """
        ...

    def reset(self) -> None:
        """Reset loader to start from the beginning."""
        ...

    def stats(self) -> str:
        """Get performance statistics."""
        ...

    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[NiftiImage]: ...
    def __next__(self) -> NiftiImage: ...


class TransformPipeline:
    """Composable transform pipeline with lazy evaluation.

    Supports method chaining for a fluent API similar to MONAI's Compose.
    Transforms are optimized and fused before execution.

    Example:
        >>> pipeline = medrs.TransformPipeline()
        >>> pipeline.z_normalize().clamp(-1.0, 1.0).resample_to_shape([64, 64, 64])
        >>> processed = pipeline.apply(img)
    """

    def __init__(self, lazy: bool = True) -> None:
        """Create a transform pipeline.

        Args:
            lazy: Enable lazy evaluation (default: True). Transforms are
                composed and optimized before execution.
        """
        ...

    def z_normalize(self) -> "TransformPipeline":
        """Add z-score normalization."""
        ...

    def rescale(self, out_min: float, out_max: float) -> "TransformPipeline":
        """Add intensity rescaling to [out_min, out_max]."""
        ...

    def clamp(self, min: float, max: float) -> "TransformPipeline":
        """Add intensity clamping to [min, max]."""
        ...

    def resample_to_spacing(self, spacing: Spacing3D) -> "TransformPipeline":
        """Add resampling to target spacing."""
        ...

    def resample_to_shape(self, shape: Shape3D) -> "TransformPipeline":
        """Add resampling to target shape."""
        ...

    def flip(self, axes: Sequence[int]) -> "TransformPipeline":
        """Add flip along axes."""
        ...

    def set_lazy(self, lazy: bool) -> "TransformPipeline":
        """Enable or disable lazy evaluation."""
        ...

    def apply(self, image: NiftiImage) -> NiftiImage:
        """Apply pipeline to an image."""
        ...

    def __repr__(self) -> str: ...


# I/O Functions

def load(path: str) -> NiftiImage:
    """Load a NIfTI image from file.

    Supports both .nii and .nii.gz formats.

    Args:
        path: Path to the NIfTI file

    Returns:
        NiftiImage instance

    Example:
        >>> img = medrs.load("brain.nii.gz")
    """
    ...


def load_to_torch(
    path: str,
    dtype: Optional[Any] = None,
    device: str = "cpu",
) -> Any:
    """Load a NIfTI image directly to PyTorch tensor.

    Most efficient way to load medical imaging data into PyTorch.
    Eliminates memory copies and supports half-precision directly.

    Args:
        path: Path to NIfTI file
        dtype: PyTorch dtype (default: torch.float32)
        device: PyTorch device (default: "cpu")

    Returns:
        PyTorch tensor

    Example:
        >>> import torch
        >>> tensor = medrs.load_to_torch("volume.nii", dtype=torch.float16, device="cuda")
    """
    ...


def load_cropped(
    path: str,
    crop_offset: Shape3D,
    crop_shape: Shape3D,
) -> NiftiImage:
    """Load only a cropped region without loading entire volume.

    Extremely efficient for training pipelines - load 64^3 patch
    from 256^3 volume without reading the full file.

    Args:
        path: Path to NIfTI file (must be uncompressed .nii)
        crop_offset: Starting coordinates [d, h, w]
        crop_shape: Size of crop region [d, h, w]

    Returns:
        NiftiImage with cropped data

    Example:
        >>> patch = medrs.load_cropped("volume.nii", [32, 32, 32], [64, 64, 64])
    """
    ...


def load_resampled(
    path: str,
    output_shape: Shape3D,
    target_spacing: Optional[Spacing3D] = None,
    target_orientation: Optional[Orientation] = None,
    output_offset: Optional[Shape3D] = None,
) -> NiftiImage:
    """Load with optional reorientation and resampling.

    Computes minimal region needed from raw file.

    Args:
        path: Path to NIfTI file
        output_shape: Desired output shape [d, h, w]
        target_spacing: Target voxel spacing [mm]
        target_orientation: Target orientation code
        output_offset: Offset in output space for non-centered crops

    Returns:
        NiftiImage with processed data
    """
    ...


def load_cropped_to_torch(
    path: str,
    output_shape: Shape3D,
    target_spacing: Optional[Spacing3D] = None,
    target_orientation: Optional[Orientation] = None,
    output_offset: Optional[Shape3D] = None,
    dtype: Optional[Any] = None,
    device: str = "cpu",
) -> Any:
    """Load cropped region directly to PyTorch tensor.

    Most efficient loading path: byte-exact cropping + direct tensor creation.

    Args:
        path: Path to NIfTI file
        output_shape: Desired output shape [d, h, w]
        target_spacing: Target voxel spacing [mm]
        target_orientation: Target orientation code
        output_offset: Offset in output space
        dtype: PyTorch dtype (default: torch.float32)
        device: PyTorch device (default: "cpu")

    Returns:
        PyTorch tensor

    Example:
        >>> import torch
        >>> tensor = medrs.load_cropped_to_torch(
        ...     "volume.nii",
        ...     output_shape=[64, 64, 64],
        ...     dtype=torch.float16,
        ...     device="cuda"
        ... )
    """
    ...


def load_cropped_to_jax(
    path: str,
    output_shape: Shape3D,
    target_spacing: Optional[Spacing3D] = None,
    target_orientation: Optional[Orientation] = None,
    output_offset: Optional[Shape3D] = None,
    dtype: Optional[Any] = None,
    device: str = "cpu",
) -> Any:
    """Load cropped region directly to JAX array.

    Args:
        path: Path to NIfTI file
        output_shape: Desired output shape [d, h, w]
        target_spacing: Target voxel spacing [mm]
        target_orientation: Target orientation code
        output_offset: Offset in output space
        dtype: JAX dtype (default: jax.numpy.float32)
        device: JAX device (default: "cpu")

    Returns:
        JAX array
    """
    ...


# Transform functions

def z_normalization(image: NiftiImage) -> NiftiImage:
    """Z-score normalize an image (zero mean, unit variance)."""
    ...


def rescale_intensity(
    image: NiftiImage,
    output_range: tuple[float, float] = (0.0, 1.0),
) -> NiftiImage:
    """Rescale intensity to the provided range."""
    ...


def clamp(image: NiftiImage, min_value: float, max_value: float) -> NiftiImage:
    """Clamp intensity values into a fixed range."""
    ...


def crop_or_pad(image: NiftiImage, target_shape: Shape3D) -> NiftiImage:
    """Crop or pad an image to the target shape."""
    ...


def resample(
    image: NiftiImage,
    target_spacing: tuple[float, float, float],
    method: Optional[InterpolationMethod] = None,
) -> NiftiImage:
    """Resample to target voxel spacing."""
    ...


def reorient(image: NiftiImage, orientation: Orientation) -> NiftiImage:
    """Reorient an image to the target orientation code."""
    ...


# Random augmentation functions

def random_flip(
    image: NiftiImage,
    axes: Sequence[int],
    prob: Optional[float] = None,
    seed: Optional[int] = None,
) -> NiftiImage:
    """Apply random flip along specified axes.

    Args:
        image: Input image
        axes: Axes that may be flipped (0=depth, 1=height, 2=width)
        prob: Probability of flipping each axis (default: 0.5)
        seed: Random seed for reproducibility

    Returns:
        Augmented image
    """
    ...


def random_gaussian_noise(
    image: NiftiImage,
    std: Optional[float] = None,
    seed: Optional[int] = None,
) -> NiftiImage:
    """Add random Gaussian noise.

    Args:
        image: Input image
        std: Standard deviation (default: 0.1)
        seed: Random seed

    Returns:
        Augmented image
    """
    ...


def random_intensity_scale(
    image: NiftiImage,
    scale_range: Optional[float] = None,
    seed: Optional[int] = None,
) -> NiftiImage:
    """Randomly scale image intensity.

    Multiplies by factor from [1-scale_range, 1+scale_range].

    Args:
        image: Input image
        scale_range: Range for scaling (default: 0.1 = 0.9-1.1x)
        seed: Random seed

    Returns:
        Augmented image
    """
    ...


def random_intensity_shift(
    image: NiftiImage,
    shift_range: Optional[float] = None,
    seed: Optional[int] = None,
) -> NiftiImage:
    """Randomly shift image intensity.

    Adds offset from [-shift_range, shift_range].

    Args:
        image: Input image
        shift_range: Range for shift (default: 0.1)
        seed: Random seed

    Returns:
        Augmented image
    """
    ...


def random_rotate_90(
    image: NiftiImage,
    axes: tuple[int, int],
    seed: Optional[int] = None,
) -> NiftiImage:
    """Randomly rotate by 90-degree increments.

    Args:
        image: Input image
        axes: Rotation plane as (axis1, axis2)
        seed: Random seed

    Returns:
        Augmented image
    """
    ...


def random_gamma(
    image: NiftiImage,
    gamma_range: Optional[tuple[float, float]] = None,
    seed: Optional[int] = None,
) -> NiftiImage:
    """Apply random gamma correction.

    output = input^gamma where gamma is randomly sampled.

    Args:
        image: Input image (should be normalized to [0, 1])
        gamma_range: Range as (min, max) (default: (0.7, 1.5))
        seed: Random seed

    Returns:
        Augmented image
    """
    ...


def random_augment(
    image: NiftiImage,
    seed: Optional[int] = None,
) -> NiftiImage:
    """Apply random combination of augmentations.

    Applies: random flip, intensity scale, intensity shift, Gaussian noise.

    Args:
        image: Input image
        seed: Random seed

    Returns:
        Augmented image
    """
    ...


# Crop region functions

def load_label_aware_cropped(
    image_path: str,
    label_path: str,
    patch_size: Shape3D,
    pos_neg_ratio: Optional[float] = None,
    min_pos_samples: Optional[int] = None,
    seed: Optional[int] = None,
) -> tuple[NiftiImage, NiftiImage]:
    """Load with byte-exact cropping for label-aware training.

    Combines MONAI's RandCropByPosNegLabeld with medrs's byte-exact loading.

    Args:
        image_path: Path to image file
        label_path: Path to label file
        patch_size: Target patch size [x, y, z]
        pos_neg_ratio: Ratio of positive to negative samples (default: 1.0)
        min_pos_samples: Minimum positive samples (default: 4)
        seed: Random seed

    Returns:
        Tuple of (cropped_image, cropped_label)
    """
    ...


def compute_crop_regions(
    image_path: str,
    label_path: str,
    patch_size: Shape3D,
    num_samples: int,
    pos_neg_ratio: Optional[float] = None,
    min_pos_samples: Optional[int] = None,
    seed: Optional[int] = None,
) -> list[dict[str, list[int]]]:
    """Compute crop regions for smart loading.

    Use with load_cropped() for maximum control in batch processing.

    Args:
        image_path: Path to image file
        label_path: Path to label file
        patch_size: Target patch size
        num_samples: Number of regions to compute
        pos_neg_ratio: Ratio of positive to negative samples
        min_pos_samples: Minimum positive samples
        seed: Random seed

    Returns:
        List of dicts with 'start', 'end', 'size' keys
    """
    ...


def compute_random_spatial_crops(
    image_path: str,
    patch_size: Shape3D,
    num_samples: int,
    seed: Optional[int] = None,
    allow_smaller: Optional[bool] = None,
) -> list[dict[str, list[int]]]:
    """Compute random spatial crop regions.

    Implements MONAI's RandSpatialCropd optimized for crop-first approach.

    Args:
        image_path: Path to image file
        patch_size: Target patch size
        num_samples: Number of regions
        seed: Random seed
        allow_smaller: Allow smaller crops at boundaries

    Returns:
        List of crop region dicts
    """
    ...


def compute_center_crop(
    image_path: str,
    patch_size: Shape3D,
) -> dict[str, list[int]]:
    """Compute center crop region.

    Implements MONAI's CenterSpatialCropd.

    Args:
        image_path: Path to image file
        patch_size: Target patch size

    Returns:
        Crop region dict with 'start', 'end', 'size'
    """
    ...
