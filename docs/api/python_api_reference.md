# medrs Python API Reference

This document provides comprehensive reference documentation for the medrs Python API.

## Table of Contents

- [Core Functions](#core-functions)
- [Image Classes](#image-classes)
- [Transform Pipeline](#transform-pipeline)
- [Random Augmentation](#random-augmentation)
- [Data Loading Classes](#data-loading-classes)
- [Transform Classes](#transform-classes)
- [Error Handling](#error-handling)
- [Type Definitions](#type-definitions)
- [Examples](#examples)

## Core Functions

### medrs.load()

Load a NIfTI image from file with automatic format detection.

```python
def load(path: Union[str, Path]) -> NiftiImage:
    """
    Load a NIfTI image from file.

    Args:
        path: Path to the NIfTI file (.nii or .nii.gz)

    Returns:
        NiftiImage instance containing loaded data

    Raises:
        LoadError: If file cannot be loaded
        ValidationError: If path is invalid

    Example:
        >>> import medrs
        >>> image = medrs.load("brain.nii.gz")
        >>> print(f"Shape: {image.shape()}")
        >>> print(f"Dtype: {image.dtype()}")
    """
```

### medrs.load_to_torch()

Load NIfTI data directly into PyTorch tensor with optimal performance.

```python
def load_to_torch(
    path: Union[str, Path],
    dtype: torch.dtype = torch.float32,
    device: str = "cpu"
) -> torch.Tensor:
    """
    Load NIfTI data directly into PyTorch tensor.

    This function provides zero-copy loading when possible and supports
    direct GPU placement, eliminating CPU-GPU transfer overhead.

    Args:
        path: Path to NIfTI file (.nii or .nii.gz)
        dtype: PyTorch tensor data type
        device: Target device ("cpu", "cuda", "cuda:0", etc.)

    Returns:
        PyTorch tensor with shape matching the image

    Raises:
        LoadError: If file cannot be loaded
        DeviceError: If device is not available
        ValidationError: If parameters are invalid

    Example:
        >>> import medrs
        >>> import torch
        >>>
        >>> # Load directly to GPU with half precision
        >>> tensor = medrs.load_to_torch(
        ...     "volume.nii.gz",
        ...     dtype=torch.float16,
        ...     device="cuda"
        ... )
        >>> print(f"Tensor: {tensor.shape}, {tensor.dtype}, {tensor.device}")
    """
```

### medrs.load_cropped()

Load only a cropped region from a NIfTI file for memory efficiency.

```python
def load_cropped(
    path: Union[str, Path],
    crop_offset: Sequence[int],
    crop_shape: Sequence[int]
) -> NiftiImage:
    """
    Load a cropped region from a NIfTI file without loading the entire volume.

    This is extremely efficient for training pipelines that load large volumes
    just to crop small patches (e.g., loading 64^3 patch from 256^3 volume).

    Args:
        path: Path to NIfTI file (.nii, must be uncompressed)
        crop_offset: Starting voxel indices [x, y, z]
        crop_shape: Size of crop region [x, y, z]

    Returns:
        NiftiImage with cropped data

    Raises:
        LoadError: If file cannot be loaded
        ValidationError: If crop parameters are invalid
        CropError: If crop region is invalid

    Example:
        >>> import medrs
        >>>
        >>> # Load 64x64x32 patch starting at offset [100, 100, 50]
        >>> patch = medrs.load_cropped(
        ...     "large_volume.nii",
        ...     crop_offset=[100, 100, 50],
        ...     crop_shape=[64, 64, 32]
        ... )
        >>> print(f"Patch shape: {patch.shape()}")
    """
```

### medrs.load_cropped_to_torch()

Load cropped NIfTI data directly into PyTorch tensor with maximum efficiency.

```python
def load_cropped_to_torch(
    volume_path: Union[str, Path],
    output_shape: Sequence[int],
    target_spacing: Optional[Sequence[float]] = None,
    target_orientation: Optional[str] = None,
    dtype: torch.dtype = torch.float32,
    device: str = "cpu"
) -> torch.Tensor:
    """
    Load cropped NIfTI data directly into PyTorch tensor.

    This is the most efficient way to load medical imaging data for PyTorch,
    combining crop-first I/O with direct tensor creation and device placement.

    Args:
        volume_path: Path to NIfTI file (.nii or .nii.gz)
        output_shape: Desired output shape [x, y, z]
        target_spacing: Target voxel spacing [x, y, z] in mm (optional)
        target_orientation: Target orientation (e.g., "RAS") (optional)
        dtype: PyTorch tensor data type
        device: Target device ("cpu", "cuda", "cuda:0", etc.)

    Returns:
        PyTorch tensor with specified shape and dtype

    Raises:
        LoadError: If file cannot be loaded
        ValidationError: If parameters are invalid
        DeviceError: If device is not available
        TransformError: If resampling/reorientation fails

    Example:
        >>> import medrs
        >>> import torch
        >>>
        >>> # Load 64x64x64 patch with isotropic 1mm spacing in RAS orientation
        >>> tensor = medrs.load_cropped_to_torch(
        ...     "brain.nii.gz",
        ...     output_shape=[64, 64, 64],
        ...     target_spacing=[1.0, 1.0, 1.0],
        ...     target_orientation="RAS",
        ...     dtype=torch.float16,
        ...     device="cuda"
        ... )
        >>> print(f"Loaded: {tensor.shape}, {tensor.dtype}, {tensor.device}")
    """
```

## Image Classes

### NiftiImage

Main class for representing NIfTI medical images.

```python
class NiftiImage:
    """
    A NIfTI image with header metadata and voxel data.

    This class provides efficient access to NIfTI files with support for
    various transforms and framework integrations.

    Attributes:
        _inner: Internal Rust NiftiImage object

    Example:
        >>> import medrs
        >>>
        >>> # Load image
        >>> img = medrs.load("brain.nii.gz")
        >>>
        >>> # Get properties
        >>> shape = img.shape()
        >>> affine = img.affine()
        >>>
        >>> # Apply transforms
        >>> processed = img.resample([1.0, 1.0, 1.0]).z_normalize().clamp(0, 1)
        >>>
        >>> # Save result
        >>> processed.save("processed.nii.gz")
    """

    def shape(self) -> Tuple[int, ...]:
        """Get the image shape."""

    def dtype(self) -> str:
        """Get the image data type."""

    def affine(self) -> np.ndarray:
        """Get the affine transformation matrix."""

    def spacing(self) -> Tuple[float, ...]:
        """Get voxel spacing in mm."""

    def to_numpy(self) -> np.ndarray:
        """Get image data as numpy array."""

    def to_numpy_view(self) -> np.ndarray:
        """Get zero-copy numpy view when possible."""

    def to_torch(
        self,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu"
    ) -> torch.Tensor:
        """Convert to PyTorch tensor."""

    def save(self, path: Union[str, Path]) -> None:
        """Save image to file."""

    def resample(self, target_spacing: Sequence[float]) -> "NiftiImage":
        """Resample image to target spacing."""

    def z_normalize(self) -> "NiftiImage":
        """Apply Z-score normalization."""

    def rescale(self, min_val: float, max_val: float) -> "NiftiImage":
        """Rescale intensity values to specified range."""

    def clamp(self, min_val: float, max_val: float) -> "NiftiImage":
        """Clamp intensity values to specified range."""

    @staticmethod
    def from_numpy(data: np.ndarray, affine: np.ndarray) -> "NiftiImage":
        """Create NiftiImage from numpy array."""
```

## Transform Pipeline

### TransformPipeline

Composable transform pipeline with lazy evaluation and automatic optimization.

```python
class TransformPipeline:
    """
    Build transformation chains that are optimized and applied efficiently.
    Supports method chaining for a fluent API.

    Args:
        lazy: Enable lazy evaluation (default: True). When True, transforms
              are composed and optimized before execution.

    Example:
        >>> import medrs
        >>>
        >>> # Create a reusable pipeline
        >>> pipeline = medrs.TransformPipeline()
        >>> pipeline.z_normalize()
        >>> pipeline.clamp(-1.0, 1.0)
        >>> pipeline.resample_to_shape([64, 64, 64])
        >>>
        >>> # Apply to multiple images
        >>> for path in image_paths:
        ...     img = medrs.load(path)
        ...     processed = pipeline.apply(img)
    """

    def __init__(self, lazy: bool = True):
        """Create a new transform pipeline."""

    def z_normalize(self) -> "TransformPipeline":
        """Add z-score normalization (zero mean, unit variance)."""

    def rescale(self, out_min: float, out_max: float) -> "TransformPipeline":
        """Add intensity rescaling to range [out_min, out_max]."""

    def clamp(self, min_val: float, max_val: float) -> "TransformPipeline":
        """Add intensity clamping to range [min, max]."""

    def resample_to_spacing(self, spacing: Sequence[float]) -> "TransformPipeline":
        """Add resampling to target voxel spacing [x, y, z] in mm."""

    def resample_to_shape(self, shape: Sequence[int]) -> "TransformPipeline":
        """Add resampling to target shape [depth, height, width]."""

    def flip(self, axes: Sequence[int]) -> "TransformPipeline":
        """Add flip along specified axes (0=depth, 1=height, 2=width)."""

    def set_lazy(self, lazy: bool) -> "TransformPipeline":
        """Enable or disable lazy evaluation."""

    def apply(self, image: NiftiImage) -> NiftiImage:
        """Apply the pipeline to an image."""
```

## Random Augmentation

Random augmentation functions for ML training with optional seeding for reproducibility.

### medrs.random_flip()

```python
def random_flip(
    image: NiftiImage,
    axes: Sequence[int],
    prob: Optional[float] = 0.5,
    seed: Optional[int] = None
) -> NiftiImage:
    """
    Apply random flip along specified axes with given probability.

    Args:
        image: Input image
        axes: Axes that may be flipped (0=depth, 1=height, 2=width)
        prob: Probability of flipping each axis (default: 0.5)
        seed: Optional random seed for reproducibility

    Returns:
        Augmented image

    Example:
        >>> augmented = medrs.random_flip(img, [0, 1, 2], prob=0.5, seed=42)
    """
```

### medrs.random_gaussian_noise()

```python
def random_gaussian_noise(
    image: NiftiImage,
    std: Optional[float] = 0.1,
    seed: Optional[int] = None
) -> NiftiImage:
    """
    Add random Gaussian noise to the image.

    Args:
        image: Input image
        std: Standard deviation of the noise (default: 0.1)
        seed: Optional random seed for reproducibility

    Returns:
        Augmented image
    """
```

### medrs.random_intensity_scale()

```python
def random_intensity_scale(
    image: NiftiImage,
    scale_range: Optional[float] = 0.1,
    seed: Optional[int] = None
) -> NiftiImage:
    """
    Randomly scale image intensity.

    Multiplies intensity by a random factor sampled from [1-scale_range, 1+scale_range].

    Args:
        image: Input image
        scale_range: Range for random scaling factor (default: 0.1, meaning 0.9-1.1)
        seed: Optional random seed for reproducibility

    Returns:
        Augmented image
    """
```

### medrs.random_intensity_shift()

```python
def random_intensity_shift(
    image: NiftiImage,
    shift_range: Optional[float] = 0.1,
    seed: Optional[int] = None
) -> NiftiImage:
    """
    Randomly shift image intensity.

    Adds a random offset sampled from [-shift_range, shift_range].

    Args:
        image: Input image
        shift_range: Range for random shift (default: 0.1)
        seed: Optional random seed for reproducibility

    Returns:
        Augmented image
    """
```

### medrs.random_rotate_90()

```python
def random_rotate_90(
    image: NiftiImage,
    axes: Tuple[int, int],
    seed: Optional[int] = None
) -> NiftiImage:
    """
    Randomly rotate the image by 90-degree increments.

    Performs random rotation in the specified plane by 0, 90, 180, or 270 degrees.

    Args:
        image: Input image
        axes: Tuple of two axes defining the rotation plane (e.g., (0, 1))
        seed: Optional random seed for reproducibility

    Returns:
        Augmented image
    """
```

### medrs.random_gamma()

```python
def random_gamma(
    image: NiftiImage,
    gamma_range: Optional[Tuple[float, float]] = (0.7, 1.5),
    seed: Optional[int] = None
) -> NiftiImage:
    """
    Apply random gamma correction to image intensity.

    Applies the transform: output = input^gamma where gamma is randomly sampled.

    Args:
        image: Input image (should be normalized to [0, 1] for best results)
        gamma_range: Range for gamma sampling as (min, max) (default: (0.7, 1.5))
        seed: Optional random seed for reproducibility

    Returns:
        Augmented image
    """
```

### medrs.random_augment()

```python
def random_augment(
    image: NiftiImage,
    seed: Optional[int] = None
) -> NiftiImage:
    """
    Apply a random combination of common augmentations.

    This is a convenience function that applies multiple augmentations:
    - Random flip (prob=0.5 per axis)
    - Random intensity scale
    - Random intensity shift
    - Random Gaussian noise

    Args:
        image: Input image
        seed: Optional random seed for reproducibility

    Returns:
        Augmented image

    Example:
        >>> augmented = medrs.random_augment(img, seed=42)
    """
```

## Data Loading Classes

### TrainingDataLoader

High-performance training data loader with intelligent caching.

```python
class TrainingDataLoader:
    """
    Optimized training data loader with prefetching and caching.

    This loader maintains an LRU cache of loaded patches and prefetches
    upcoming data to maximize I/O throughput.

    Args:
        volumes: List of NIfTI file paths
        patch_size: Size of patches to extract [x, y, z]
        patches_per_volume: Number of patches per volume
        randomize: Whether to randomize patch positions
        cache_size: Maximum number of patches to cache

    Example:
        >>> import medrs
        >>>
        >>> volumes = ["vol1.nii", "vol2.nii", "vol3.nii"]
        >>> loader = medrs.TrainingDataLoader(
        ...     volumes=volumes,
        ...     patch_size=[64, 64, 64],
        ...     patches_per_volume=8,
        ...     randomize=True,
        ...     cache_size=1000
        ... )
        >>>
        >>> # Training loop
        >>> while True:
        ...     try:
        ...         patch = loader.next_patch()
        ...         # Process patch...
        ...     except medrs.Error:
        ...         break  # No more patches
    """

    def next_patch(self) -> NiftiImage:
        """Get next training patch."""

    def reset(self) -> None:
        """Reset loader to start."""

    def stats(self) -> LoaderStats:
        """Get loader statistics."""
```

## Transform Classes

### Transform Base Classes

```python
class Transform(ABC):
    """Abstract base class for all transforms."""

    @abstractmethod
    def __call__(self, image: NiftiImage) -> NiftiImage:
        """Apply transform to image."""

class IntensityTransform(Transform):
    """Base class for intensity transforms."""

class SpatialTransform(Transform):
    """Base class for spatial transforms."""
```

### Built-in Transforms

#### ResampleTransform

```python
class ResampleTransform(SpatialTransform):
    """
    Resample image to target voxel spacing.

    Args:
        target_spacing: Target spacing [x, y, z] in mm
        interpolation: Interpolation method ('linear', 'nearest')

    Example:
        >>> transform = ResampleTransform([1.0, 1.0, 1.0])
        >>> resampled = transform(image)
    """
```

#### NormalizeTransform

```python
class NormalizeTransform(IntensityTransform):
    """
    Apply intensity normalization.

    Args:
        method: Normalization method ('zscore', 'minmax', 'robust')

    Example:
        >>> transform = NormalizeTransform(method='zscore')
        >>> normalized = transform(image)
    """
```

#### ClampTransform

```python
class ClampTransform(IntensityTransform):
    """
    Clamp intensity values to specified range.

    Args:
        min_val: Minimum intensity value
        max_val: Maximum intensity value

    Example:
        >>> transform = ClampTransform(0.0, 1.0)
        >>> clamped = transform(image)
    """
```

## Error Handling

### Exception Hierarchy

medrs provides a comprehensive exception hierarchy for better error handling:

```python
# Base exception
class MedrsError(Exception):
    """Base exception for all medrs operations."""

# Specific exceptions
class LoadError(MedrsError):
    """Raised when file loading fails."""

class ValidationError(MedrsError):
    """Raised when input validation fails."""

class DeviceError(MedrsError):
    """Raised when device operations fail."""

class TransformError(MedrsError):
    """Raised when transform operations fail."""

class MemoryError(MedrsError):
    """Raised when memory allocation fails."""

class ConfigurationError(MedrsError):
    """Raised when configuration is invalid."""
```

### Error Handling Best Practices

```python
try:
    # Load image with proper error handling
    image = medrs.load("brain.nii.gz")

    # Apply transforms
    processed = image.resample([1.0, 1.0, 1.0]).z_normalize()

except LoadError as e:
    print(f"Failed to load file: {e}")
    print(f"Suggestions: {e.suggestions}")

except ValidationError as e:
    print(f"Invalid parameter: {e}")

except DeviceError as e:
    print(f"Device error: {e}")

except MedrsError as e:
    print(f"medrs error: {e}")
    print(f"Details: {e.to_dict()}")
```

## Type Definitions

### Common Type Aliases

```python
from typing import Union, Sequence, Tuple

# Paths
PathLike = Union[str, Path]

# Shapes and dimensions
Shape3D = Tuple[int, int, int]
ShapeND = Tuple[int, ...]

# Device specifications
Device = str  # "cpu", "cuda", "cuda:0", etc.

# Data types
Dtype = Union[np.dtype, torch.dtype]
```

### Protocol Definitions

```python
@runtime_checkable
class ImageProtocol(Protocol):
    """Protocol for medical image data."""

    @property
    def shape(self) -> Tuple[int, ...]: ...

    @property
    def dtype(self) -> np.dtype: ...

    def get_fdata(self) -> np.ndarray: ...

@runtime_checkable
class DataLoaderProtocol(Protocol):
    """Protocol for data loader implementations."""

    def __iter__(self) -> Iterator[np.ndarray]: ...

    def reset(self) -> None: ...
```

## Examples

### Basic Usage

```python
import medrs
import torch
import numpy as np

# Load an image
image = medrs.load("brain.nii.gz")
print(f"Shape: {image.shape()}, Spacing: {image.spacing()}")

# Convert to numpy
data = image.to_numpy()
print(f"Numpy array: {data.shape}, {data.dtype}")

# Convert to PyTorch
tensor = image.to_torch(dtype=torch.float16, device="cuda")
print(f"PyTorch tensor: {tensor.shape}, {tensor.dtype}, {tensor.device}")
```

### High-Performance Training

```python
import medrs
from medrs import TrainingDataLoader

# Setup volumes
volumes = [
    "subject_01.nii.gz",
    "subject_02.nii.gz",
    "subject_03.nii.gz"
]

# Create optimized data loader
loader = TrainingDataLoader(
    volumes=volumes,
    patch_size=[64, 64, 64],
    patches_per_volume=8,
    randomize=True,
    cache_size=1000
)

# Training loop
for batch_idx in range(1000):
    try:
        # Get patch (40x memory efficient)
        patch = loader.next_patch()

        # Convert to tensor and process
        tensor = patch.to_torch(device="cuda", dtype=torch.float16)

        # Your training logic here
        loss = your_model(tensor)
        loss.backward()

        # Print progress
        if batch_idx % 100 == 0:
            stats = loader.stats()
            print(f"Batch {batch_idx}: {stats}")

    except medrs.Error:
        break  # No more patches

print(f"Training completed. Processed {loader.stats().patches_processed} patches")
```

### Framework Integration

```python
import medrs
import torch
from torch.utils.data import Dataset, DataLoader

class MedicalDataset(Dataset):
    def __init__(self, volume_paths, patch_size=(64, 64, 64)):
        self.volume_paths = volume_paths
        self.patch_size = patch_size

    def __len__(self):
        return len(self.volume_paths) * 8  # 8 patches per volume

    def __getitem__(self, idx):
        volume_idx = idx // 8
        patch_idx = idx % 8

        # Load patch with crop-first efficiency
        patch = medrs.load_cropped(
            self.volume_paths[volume_idx],
            crop_offset=[16, 16, 8],  # Example offset
            crop_shape=self.patch_size
        )

        # Convert to tensor directly
        tensor = patch.to_torch()
        return tensor

# Use with PyTorch DataLoader
dataset = MedicalDataset(volume_paths, patch_size=(64, 64, 64))
dataloader = DataLoader(dataset, batch_size=8, num_workers=4)

for batch in dataloader:
    # Process batch
    loss = model(batch)
    print(f"Batch shape: {batch.shape}, Loss: {loss.item()}")
```

### Advanced Processing Pipeline

```python
import medrs
from medrs.transforms import ResampleTransform, NormalizeTransform, ClampTransform

# Create transform pipeline
transforms = [
    ResampleTransform([1.0, 1.0, 1.0]),
    NormalizeTransform(method='zscore'),
    ClampTransform(0.0, 1.0)
]

# Load and process image
image = medrs.load("anatomic_scan.nii.gz")

# Apply transforms
for transform in transforms:
    image = transform(image)

# Save processed result
image.save("processed.nii.gz")

print(f"Processed image: {image.shape()}, {image.spacing()}")
```

## Performance Considerations

### Memory Efficiency

- **Always use crop-first loading** for training: `load_cropped()` instead of `load()`
- **Load directly to target framework**: `load_cropped_to_torch()` or `load_cropped_to_jax()`
- **Use TrainingDataLoader** for high-throughput pipelines with caching
- **Clear GPU cache** regularly: `torch.cuda.empty_cache()`

### Performance Monitoring

```python
from examples.base.performance_profiler import PerformanceProfiler, quick_profile

# Profile specific operations
@quick_profile("load_and_process")
def process_volume(path):
    image = medrs.load(path)
    return image.resample([1.0, 1.0, 1.0]).z_normalize()

# Comprehensive profiling
with PerformanceProfiler() as profiler:
    for path in volume_paths:
        result = process_volume(path)

# Get performance summary
summary = profiler.get_summary()
print(f"Performance summary: {summary}")
```

For more detailed examples and advanced usage patterns, see the [Advanced Features Guide](../guides/advanced_features.md).