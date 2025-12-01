"""
MONAI MetaTensor support for medrs.

Provides seamless integration with MONAI's MetaTensor data structure,
preserving NIfTI-style metadata (affine matrix, spacing, orientation)
alongside the tensor data for medical imaging workflows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Sequence, Any, Union
from pathlib import Path
import warnings

# Optional dependencies
try:
    from monai.data import MetaTensor
    from monai.transforms import EnsureChannelFirst
    MONAI_AVAILABLE = True
except ImportError:
    MONAI_AVAILABLE = False
    MetaTensor = None

try:
    import torch
    import numpy as np
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Import medrs core functions
try:
    from . import load, load_cropped, resample, reorient, crop_or_pad
    MEDRS_AVAILABLE = True
except ImportError:
    MEDRS_AVAILABLE = False


# ============================================================================
# METADATA HANDLING
# ============================================================================

@dataclass
class ImageMetadata:
    """Container for NIfTI-style image metadata."""
    affine: Optional[np.ndarray] = None
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    orientation: str = "RAS"
    origin: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    dtype: Optional[str] = None
    dim: int = 3
    voxel_size: Optional[float] = None


# ============================================================================
# METATENSOR CONVERSION
# ============================================================================

class MedrsMetaTensorConverter:
    """Converter between medrs MedicalImage and MONAI MetaTensor."""

    @staticmethod
    def medical_image_to_metatensor(medrs_image, device: Optional[str] = None) -> 'MetaTensor':
        """
        Convert medrs MedicalImage to MONAI MetaTensor with metadata.

        Args:
            medrs_image: medrs MedicalImage object
            device: Target device for tensor

        Returns:
            MONAI MetaTensor with preserved metadata
        """
        if not MONAI_AVAILABLE:
            raise ImportError("MONAI is not available. Install with: pip install monai")

        # Convert to tensor
        tensor = medrs_image.to_torch(device=device)

        # Extract metadata
        metadata = MedrsMetaTensorConverter._extract_metadata(medrs_image)

        # Create MetaTensor
        try:
            metatensor = MetaTensor(tensor, meta=metadata)
            return metatensor
        except Exception as e:
            # Fallback: create tensor without metadata if MetaTensor creation fails
            warnings.warn(f"Failed to create MetaTensor with metadata: {e}")
            return tensor

    @staticmethod
    def _extract_metadata(image) -> Dict[str, Any]:
        """Extract metadata from medrs image."""
        metadata = {}

        # Extract affine matrix
        affine = MedrsMetaTensorConverter._get_affine_matrix(image)
        if affine is not None:
            metadata['affine'] = affine

        # Extract spacing
        spacing = MedrsMetaTensorConverter._get_spacing(image)
        metadata['spacing'] = spacing

        # Extract orientation
        orientation = MedrsMetaTensorConverter._get_orientation(image)
        metadata['orientation'] = orientation

        # Extract origin
        origin = MedrsMetaTensorConverter._get_origin(image)
        metadata['original_origin'] = origin

        # Extract other properties
        if hasattr(image, 'data'):
            metadata['original_shape'] = image.data.shape
            metadata['dtype'] = str(image.data.dtype)

        return metadata

    @staticmethod
    def _get_affine_matrix(image) -> Optional[np.ndarray]:
        """Get affine matrix from image."""
        if hasattr(image, 'affine') and image.affine is not None:
            return image.affine

        # Try to construct from orientation and spacing
        if hasattr(image, 'orientation') and hasattr(image, 'spacing'):
            orientation = image.orientation
            spacing = image.spacing
            shape = getattr(image, 'data', (1, 1, 1)).shape

            # Construct a basic affine matrix
            affine = np.eye(4, dtype=np.float64)
            affine[0, 0] = spacing[0]
            affine[1, 1] = spacing[1]
            affine[2, 2] = spacing[2]

            # Center the image
            affine[:3, 3] = -(np.array(shape[:3]) * np.array(spacing)) / 2
            return affine

        return None

    @staticmethod
    def _get_spacing(image) -> Tuple[float, float, float]:
        """Get voxel spacing from image."""
        if hasattr(image, 'spacing'):
            return tuple(float(x) for x in image.spacing)
        return (1.0, 1.0, 1.0)

    @staticmethod
    def _get_orientation(image) -> str:
        """Get orientation from image."""
        if hasattr(image, 'orientation'):
            return str(image.orientation)
        return "RAS"

    @staticmethod
    def _get_origin(image) -> Tuple[float, float, float]:
        """Get origin from image."""
        if hasattr(image, 'origin'):
            return tuple(float(x) for x in image.origin)
        return (0.0, 0.0, 0.0)


# ============================================================================
# METATENSOR LOADER
# ============================================================================

class MetaTensorLoader:
    """
    Load NIfTI volumes directly into MONAI MetaTensor format with medrs performance.
    """

    def __init__(
        self,
        device: str = "cpu",
        dtype: Optional[Any] = None,
        preserve_metadata: bool = True,
        cache_metadata: bool = True
    ):
        """
        Initialize MetaTensor loader.

        Args:
            device: Target device for tensors
            dtype: Data type for tensors
            preserve_metadata: Whether to preserve NIfTI metadata
            cache_metadata: Whether to cache metadata for repeated access
        """
        if not MONAI_AVAILABLE:
            raise ImportError("MONAI is not available. Install with: pip install monai")

        self.device = device
        self.dtype = dtype
        self.preserve_metadata = preserve_metadata
        self.cache_metadata = cache_metadata
        self._metadata_cache = {} if cache_metadata else None

    def load(self, path: Union[str, Path]) -> 'MetaTensor':
        """
        Load NIfTI file into MONAI MetaTensor.

        Args:
            path: Path to NIfTI file

        Returns:
            MONAI MetaTensor with metadata
        """
        path = str(path)

        # Check cache
        if self._metadata_cache and path in self._metadata_cache:
            cached_metadata = self._metadata_cache[path]
        else:
            cached_metadata = None

        try:
            # Load with medrs
            medrs_image = load(path)

            # Convert to MetaTensor
            metatensor = MedrsMetaTensorConverter.medical_image_to_metatensor(
                medrs_image, device=self.device
            )

            # Cache metadata if enabled
            if self._metadata_cache is not None:
                self._metadata_cache[path] = metatensor.meta.copy() if hasattr(metatensor, 'meta') else {}

            # Apply dtype if specified
            if self.dtype is not None:
                metatensor = metatensor.to(self.dtype)

            return metatensor

        except Exception as e:
            raise RuntimeError(f"Failed to load {path}: {e}")

    def load_cropped(
        self,
        path: Union[str, Path],
        crop_offset: Sequence[int],
        crop_shape: Sequence[int]
    ) -> 'MetaTensor':
        """
        Load cropped NIfTI file into MONAI MetaTensor.

        Args:
            path: Path to NIfTI file
            crop_offset: Starting voxel offset [x, y, z]
            crop_shape: Crop shape [x, y, z]

        Returns:
            MONAI MetaTensor with metadata
        """
        path = str(path)

        try:
            # Load cropped with medrs
            medrs_image = load_cropped(path, crop_offset, crop_shape)

            # Convert to MetaTensor
            metatensor = MedrsMetaTensorConverter.medical_image_to_metatensor(
                medrs_image, device=self.device
            )

            # Apply dtype if specified
            if self.dtype is not None:
                metatensor = metatensor.to(self.dtype)

            return metatensor

        except Exception as e:
            raise RuntimeError(f"Failed to load cropped {path}: {e}")

    def load_multiple(
        self,
        paths: Sequence[Union[str, Path]],
        keys: Optional[Sequence[str]] = None
    ) -> Dict[str, 'MetaTensor']:
        """
        Load multiple NIfTI files into MetaTensors.

        Args:
            paths: Paths to NIfTI files
            keys: Optional keys for the loaded volumes

        Returns:
            Dictionary of MetaTensors
        """
        if keys is None:
            keys = [f"volume_{i}" for i in range(len(paths))]

        if len(paths) != len(keys):
            raise ValueError("Number of paths and keys must match")

        metatensors = {}
        for path, key in zip(paths, keys):
            metatensors[key] = self.load(path)

        return metatensors


# ============================================================================
# COORDINATED METATENSOR LOADING
# ============================================================================

class MetaTensorCoordinatedCropLoader:
    """
    Load and crop multiple volumes into MONAI MetaTensors with coordinated cropping.

    Ensures anatomical consistency across modalities while preserving metadata.
    """

    def __init__(
        self,
        keys: Sequence[str],
        crop_size: Tuple[int, int, int],
        spatial_normalizer,
        device: str = "cpu",
        dtype: Optional[Any] = None,
        preserve_metadata: bool = True,
        validation_mode: bool = False
    ):
        """
        Initialize coordinated crop loader for MetaTensors.

        Args:
            keys: Keys of volumes to load and crop
            crop_size: Desired crop size (x, y, z)
            spatial_normalizer: SpatialNormalizer instance
            device: Target device for tensors
            dtype: Data type for tensors
            preserve_metadata: Whether to preserve NIfTI metadata
            validation_mode: If True, uses deterministic center cropping
        """
        if not MONAI_AVAILABLE:
            raise ImportError("MONAI is not available. Install with: pip install monai")

        self.keys = list(keys)
        self.crop_size = crop_size
        self.spatial_normalizer = spatial_normalizer
        self.device = device
        self.dtype = dtype
        self.preserve_metadata = preserve_metadata
        self.validation_mode = validation_mode

        # Create MetaTensor loader
        self.metatensor_loader = MetaTensorLoader(
            device=device,
            dtype=dtype,
            preserve_metadata=preserve_metadata
        )

        # Import dictionary transforms for spatial normalization
        try:
            from .dictionary_transforms import CoordinatedCropLoader
            self._base_loader = CoordinatedCropLoader(
                keys=keys,
                crop_size=crop_size,
                spatial_normalizer=spatial_normalizer,
                device=device,
                dtype=dtype,
                random_center=not validation_mode
            )
        except ImportError:
            self._base_loader = None

    def _determine_crop_center(self, ref_shape: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """Determine crop center coordinates."""
        if self.validation_mode:
            # Deterministic center crop
            return tuple(dim // 2 for dim in ref_shape)
        else:
            # Could implement random cropping here
            # For now, use deterministic
            return tuple(dim // 2 for dim in ref_shape)

    def _calculate_crop_offsets(self, center: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """Calculate crop start offsets from center."""
        return tuple(
            max(0, coord - size // 2)
            for coord, size in zip(center, self.crop_size)
        )

    def __call__(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Load and crop volumes as MetaTensors with spatial coordination.

        Args:
            data_dict: Dictionary with keys pointing to volume file paths.

        Returns:
            Dictionary with MetaTensor volumes.
        """
        if not MEDRS_AVAILABLE:
            raise RuntimeError("medrs core not available")

        # Use base loader for spatial normalization if available
        if self._base_loader:
            normalized_dict = self._base_loader(data_dict)
            # Extract reference volume for crop calculation
            ref_key = self.spatial_normalizer.reference_key
            if ref_key in normalized_dict and hasattr(normalized_dict[ref_key], 'data'):
                ref_shape = normalized_dict[ref_key].data.shape
                crop_center = self._determine_crop_center(ref_shape)
                crop_offsets = self._calculate_crop_offsets(crop_center)
            else:
                crop_center = (64, 64, 64)  # Fallback
                crop_offsets = (32, 32, 32)
        else:
            # Simplified approach
            crop_center = (64, 64, 64)
            crop_offsets = (32, 32, 32)

        result_dict = {}

        for key in self.keys:
            if key not in data_dict:
                raise KeyError(f"Key '{key}' not found in data")

            try:
                # Determine if we have the path or a pre-loaded volume
                if key in data_dict and self._is_path_like(data_dict[key]):
                    # Load from file path
                    metatensor = self.metatensor_loader.load_cropped(
                        data_dict[key],
                        crop_offsets,
                        self.crop_size
                    )
                else:
                    # Try to find the volume in normalized dict
                    if hasattr(data_dict, 'get') and key in data_dict:
                        # Handle pre-loaded volume
                        volume = data_dict[key]
                        if hasattr(volume, 'data'):
                            # Crop the volume and convert to MetaTensor
                            if hasattr(volume, 'data'):
                                cropped_data = volume.data[
                                    crop_offsets[0]:crop_offsets[0] + self.crop_size[0],
                                    crop_offsets[1]:crop_offsets[1] + self.crop_size[1],
                                    crop_offsets[2]:crop_offsets[2] + self.crop_size[2]
                                ]

                                # Create MetaTensor with metadata
                                metadata = {}
                                if hasattr(volume, 'spacing'):
                                    metadata['spacing'] = volume.spacing
                                if hasattr(volume, 'orientation'):
                                    metadata['orientation'] = volume.orientation

                                tensor = volume.to_torch(device=self.device, dtype=self.dtype)
                                metatensor = MetaTensor(tensor, meta=metadata)
                            else:
                                # Fallback
                                metatensor = volume.to_torch(device=self.device, dtype=self.dtype)
                        else:
                            # Fallback to file loading
                            metatensor = self.metatensor_loader.load(str(volume))
                    else:
                        raise KeyError(f"Could not find volume for key '{key}'")

                result_dict[key] = metatensor

            except Exception as e:
                raise RuntimeError(f"Failed to load and crop {key}: {e}")

        # Pass through non-volume keys unchanged
        for key, value in data_dict.items():
            if key not in self.keys:
                result_dict[key] = value

        return result_dict

    def _is_path_like(self, value: Any) -> bool:
        """Check if value looks like a file path."""
        return isinstance(value, (str, Path)) and str(value).endswith(('.nii', '.nii.gz'))


# ============================================================================
# MONAI COMPATIBILITY
# ============================================================================

class MetaTensorCompatibleTransform:
    """
    MONAI-compatible wrapper for medrs transforms that returns MetaTensors.

    Usage:
        transform = MetaTensorCompatibleTransform(
            loader=MetaTensorCoordinatedCropLoader(keys, crop_size)
        )
        # In MONAI pipeline
        pipeline = Compose([transform, other_monai_transforms])
    """

    def __init__(self, loader):
        """Initialize MONAI-compatible transform."""
        self.loader = loader

    def __call__(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Apply transformation with MetaTensor support."""
        return self.loader(data_dict)


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_metatensor_loader(
    device: str = "cpu",
    dtype: Optional[Any] = None,
    preserve_metadata: bool = True,
    cache_metadata: bool = False
) -> MetaTensorLoader:
    """
    Create a MetaTensor loader with default settings.

    Args:
        device: Target device for tensors
        dtype: Data type for tensors
        preserve_metadata: Whether to preserve NIfTI metadata
        cache_metadata: Whether to cache metadata

    Returns:
        MetaTensorLoader instance
    """
    return MetaTensorLoader(
        device=device,
        dtype=dtype,
        preserve_metadata=preserve_metadata,
        cache_metadata=cache_metadata
    )


def create_metatensor_crop_transform(
    keys: Sequence[str],
    crop_size: Tuple[int, int, int],
    spatial_normalizer,
    device: str = "cpu",
    dtype: Optional[Any] = None,
    preserve_metadata: bool = True,
    **kwargs
) -> MetaTensorCompatibleTransform:
    """
    Create a MONAI-compatible transform that returns MetaTensors.

    Args:
        keys: Keys to crop
        crop_size: Desired crop size
        spatial_normalizer: SpatialNormalizer instance
        device: Target device
        dtype: Target data type
        preserve_metadata: Whether to preserve metadata
        **kwargs: Additional arguments for MetaTensorCoordinatedCropLoader

    Returns:
        MONAI-compatible transform
    """
    loader = MetaTensorCoordinatedCropLoader(
        keys=keys,
        crop_size=crop_size,
        spatial_normalizer=spatial_normalizer,
        device=device,
        dtype=dtype,
        preserve_metadata=preserve_metadata,
        **kwargs
    )
    return MetaTensorCompatibleTransform(loader)


def metatensor_from_medrs(medrs_image, device: Optional[str] = None) -> 'MetaTensor':
    """
    Convert a medrs MedicalImage to MONAI MetaTensor.

    Args:
        medrs_image: medrs MedicalImage object
        device: Target device for tensor

    Returns:
        MONAI MetaTensor with preserved metadata
    """
    return MedrsMetaTensorConverter.medical_image_to_metatensor(
        medrs_image, device=device
    )


def is_metatensor_supported() -> bool:
    """Check if MONAI MetaTensor is available."""
    return MONAI_AVAILABLE and MEDRS_AVAILABLE


def enhance_dictionary_transforms_for_metatensor():
    """
    Enhance dictionary transforms to support MetaTensor if MONAI is available.

    This function monkey-patches the dictionary transforms to support
    MetaTensor if MONAI is available.
    """
    if not is_metatensor_supported():
        return False

    try:
        from . import dictionary_transforms

        # Store original classes
        original_CoordinatedCropLoader = dictionary_transforms.CoordinatedCropLoader

        class MetaTensorAwareCoordinatedCropLoader(original_CoordinatedCropLoader):
            """Enhanced CoordinatedCropLoader that supports MetaTensor."""

            def __init__(self, *args, return_metatensor: bool = True, **kwargs):
                super().__init__(*args, **kwargs)
                self.return_metatensor = return_metatensor

                if return_metatensor:
                    self.metatensor_converter = MedrsMetaTensorConverter()

            def __call__(self, data_dict):
                result = super().__call__(data_dict)

                if not self.return_metatensor:
                    return result

                # Convert tensors to MetaTensors
                metatensor_result = {}
                for key, value in result.items():
                    if hasattr(value, 'shape') and hasattr(value, 'device'):  # It's a tensor
                        # Create MetaTensor with metadata
                        if hasattr(value, 'meta'):  # Already a MetaTensor
                            metatensor_result[key] = value
                        else:
                            # Convert to MetaTensor
                            try:
                                metatensor = MetaTensor(value)
                                metatensor_result[key] = metatensor
                            except Exception as e:
                                warnings.warn(f"Failed to convert {key} to MetaTensor: {e}")
                                metatensor_result[key] = value
                    else:
                        metatensor_result[key] = value

                return metatensor_result

        # Replace the original class
        dictionary_transforms.CoordinatedCropLoader = MetaTensorAwareCoordinatedCropLoader

        return True

    except ImportError:
        return False


# ============================================================================
# AVAILABILITY CHECKS
# ============================================================================

def check_availability() -> Dict[str, bool]:
    """
    Check availability of optional components.

    Returns:
        Dictionary mapping component names to availability status.
    """
    return {
        'monai': MONAI_AVAILABLE,
        'metatensor': is_metatensor_supported(),
        'torch': TORCH_AVAILABLE,
        'numpy': True,  # Always available with Python
        'medrs_core': MEDRS_AVAILABLE,
    }


def get_availability_report() -> str:
    """
    Get formatted availability report.

    Returns:
        Formatted string with availability information.
    """
    availability = check_availability()

    report_lines = ["medrs Component Availability:", "=" * 35]
    for component, available in availability.items():
        status = " Available" if available else " Not Available"
        report_lines.append(f"  {component}: {status}")

    return "\n".join(report_lines)


# Make availability info accessible at module level
_availability = check_availability()
_availability_report = get_availability_report()
