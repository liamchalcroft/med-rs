"""
Dictionary transforms for multi-modal medical imaging workflows.

Provides coordinated loading and cropping across multiple volumes,
automatic spatial normalization, and MONAI-compatible transforms.
"""

from __future__ import annotations

from typing import Dict, Tuple, Optional, Sequence, Any, Protocol
from pathlib import Path
from dataclasses import dataclass

import numpy as np

from . import load, resample, reorient


# ============================================================================
# TYPES AND PROTOCOLS
# ============================================================================

@dataclass
class SpatialProperties:
    """Container for spatial image properties."""
    spacing: Tuple[float, float, float]
    orientation: str
    shape: Tuple[int, ...]
    origin: Tuple[float, float, float] = (0.0, 0.0, 0.0)


class Transform(Protocol):
    """Protocol for dictionary transforms."""
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply the transform to the data dictionary."""
        ...


# ============================================================================
# SPATIAL NORMALIZATION
# ============================================================================

class SpatialNormalizer:
    """
    Normalize multiple volumes to the same spatial grid.

    This class handles the complex task of spatial normalization across
    multiple medical images with different voxel spacings and orientations.
    """

    # Standard orientation codes
    STANDARD_ORIENTATIONS = {"RAS", "LAS", "RPS", "LPS", "ASL", "AIL"}

    def __init__(
        self,
        target_spacing: Optional[Tuple[float, float, float]] = None,
        target_orientation: str = "RAS",
        reference_key: Optional[str] = None,
        auto_detect_spacing: bool = True,
        min_spacing: float = 0.1,
        max_spacing: float = 10.0,
    ):
        """
        Initialize spatial normalizer.

        Args:
            target_spacing: Target voxel spacing in mm. If None, auto-determines.
            target_orientation: Target orientation (e.g., "RAS", "LPS").
            reference_key: Key of reference volume to use as template.
            auto_detect_spacing: Whether to auto-determine optimal spacing.
            min_spacing: Minimum allowed spacing in mm.
            max_spacing: Maximum allowed spacing in mm.
        """
        if target_orientation not in self.STANDARD_ORIENTATIONS:
            raise ValueError(f"Invalid orientation. Must be one of: {self.STANDARD_ORIENTATIONS}")

        self.target_spacing = self._validate_spacing(target_spacing) if target_spacing else None
        self.target_orientation = target_orientation
        self.reference_key = reference_key
        self.auto_detect_spacing = auto_detect_spacing
        self.min_spacing = min_spacing
        self.max_spacing = max_spacing

    @staticmethod
    def _validate_spacing(spacing: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """Validate and return spacing within acceptable range."""
        return tuple(max(0.1, min(10.0, s)) for s in spacing)

    def _analyze_volume_properties(self, data_dict: Dict[str, Any]) -> Dict[str, SpatialProperties]:
        """Analyze spatial properties for all volumes."""
        properties = {}

        for key, value in data_dict.items():
            if self._is_volume_path(value):
                properties[key] = self._get_volume_properties(key, str(value))

        return properties

    def _is_volume_path(self, value: Any) -> bool:
        """Check if value represents a volume file path."""
        return isinstance(value, (str, Path)) and str(value).endswith(('.nii', '.nii.gz'))

    def _get_volume_properties(self, key: str, path: str) -> SpatialProperties:
        """Get spatial properties for a volume."""
        img = load(path)
        return SpatialProperties(
            spacing=getattr(img, 'spacing', (1.0, 1.0, 1.0)),
            orientation=getattr(img, 'orientation', 'RAS'),
            shape=getattr(img, 'data', (1, 1, 1)).shape,
            origin=getattr(img, 'origin', (0.0, 0.0, 0.0))
        )

    def _determine_target_properties(
        self,
        properties: Dict[str, SpatialProperties]
    ) -> Tuple[Tuple[float, float, float], str]:
        """Determine target spacing and orientation."""
        # Use reference volume if specified
        if self.reference_key and self.reference_key in properties:
            ref_props = properties[self.reference_key]
            target_spacing = self.target_spacing or ref_props.spacing
            return target_spacing, self.target_orientation

        # Auto-determine optimal spacing
        if self.auto_detect_spacing and self.target_spacing is None:
            target_spacing = self._compute_optimal_spacing(properties)
        else:
            target_spacing = self.target_spacing or (1.0, 1.0, 1.0)

        return target_spacing, self.target_orientation

    def _compute_optimal_spacing(self, properties: Dict[str, SpatialProperties]) -> Tuple[float, float, float]:
        """Compute optimal target spacing from all volumes."""
        if not properties:
            return (1.0, 1.0, 1.0)

        # Use smallest spacing in each dimension (highest resolution)
        all_spacings = [props.spacing for props in properties.values()]
        return tuple(min(s[i] for s in all_spacings) for i in range(3))

    def __call__(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize spatial properties of all volumes.

        Args:
            data_dict: Dictionary with volume file paths or volumes.

        Returns:
            Dictionary with spatially normalized volumes.
        """
        # Analyze spatial properties
        properties = self._analyze_volume_properties(data_dict)
        if not properties:
            return data_dict.copy()

        # Determine target properties
        target_spacing, target_orientation = self._determine_target_properties(properties)

        # Normalize all volumes
        normalized_dict = {}

        for key, value in data_dict.items():
            if key in properties:
                try:
                    # Load and normalize volume
                    img = load(str(value)) if isinstance(value, (str, Path)) else value
                    img = self._normalize_volume(img, target_spacing, target_orientation)
                    normalized_dict[key] = img
                except Exception as e:
                    raise RuntimeError(f"Failed to normalize {key}: {e}")
            else:
                # Pass through non-volume data
                normalized_dict[key] = value

        return normalized_dict

    def _normalize_volume(self, img, target_spacing, target_orientation):
        """Normalize a single volume to target properties."""
        # Resample to target spacing
        if hasattr(img, 'spacing') and img.spacing != target_spacing:
            img = resample(img, target_spacing)

        # Reorient to target orientation
        if hasattr(img, 'orientation') and img.orientation != target_orientation:
            img = reorient(img, target_orientation)

        return img


# ============================================================================
# COORDINATED CROPPING
# ============================================================================

class CoordinatedCropLoader:
    """
    Load and crop multiple volumes from the same spatial region.

    Ensures anatomical consistency across modalities by using the same
    crop parameters for all volumes.
    """

    def __init__(
        self,
        keys: Sequence[str],
        crop_size: Tuple[int, int, int],
        spatial_normalizer: Optional[SpatialNormalizer] = None,
        device: str = "cpu",
        dtype: Optional[Any] = None,
        random_center: bool = False,
        center_margin: int = 16,
        validation_mode: bool = False,
    ):
        """
        Initialize coordinated crop loader.

        Args:
            keys: Keys of volumes to load and crop.
            crop_size: Desired crop size (x, y, z).
            spatial_normalizer: SpatialNormalizer instance for preprocessing.
            device: Target device for tensors.
            dtype: Data type for tensors.
            random_center: Whether to use random crop center.
            center_margin: Margin from edges for random center selection.
            validation_mode: If True, uses deterministic center cropping.
        """
        self.keys = list(keys)
        self.crop_size = crop_size
        self.spatial_normalizer = spatial_normalizer or SpatialNormalizer()
        self.device = device
        self.dtype = dtype
        self.random_center = random_center and not validation_mode
        self.center_margin = center_margin
        self.validation_mode = validation_mode

    def _determine_crop_center(self, ref_shape: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """Determine crop center coordinates."""
        if self.validation_mode or not self.random_center:
            # Deterministic center crop
            return tuple(dim // 2 for dim in ref_shape)
        else:
            # Random center with margin
            min_coords = tuple(
                self.center_margin + size // 2
                for size in self.crop_size
            )
            max_coords = tuple(
                dim - self.center_margin - size // 2
                for dim, size in zip(ref_shape, self.crop_size)
            )

            # Validate bounds
            min_coords = tuple(max(0, coord) for coord in min_coords)
            max_coords = tuple(
                min(dim - size, coord) for dim, size, coord in zip(ref_shape, self.crop_size, max_coords)
            )

            import random
            return tuple(
                random.randint(min_coord, max_coord)
                for min_coord, max_coord in zip(min_coords, max_coords)
            )

    def _calculate_crop_offsets(self, center: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """Calculate crop start offsets from center."""
        return tuple(
            max(0, coord - size // 2)
            for coord, size in zip(center, self.crop_size)
        )

    def __call__(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Load and crop volumes with spatial coordination.

        Args:
            data_dict: Dictionary with keys pointing to volume file paths.

        Returns:
            Dictionary with cropped volumes/tensors.
        """
        # First, spatially normalize all volumes
        normalized_dict = self.spatial_normalizer(data_dict)

        # Determine crop center from reference volume
        if not hasattr(normalized_dict[self.spatial_normalizer.reference_key], 'data'):
            raise KeyError(f"Reference key '{self.spatial_normalizer.reference_key}' not found or invalid")

        ref_volume = normalized_dict[self.spatial_normalizer.reference_key]
        ref_shape = ref_volume.data.shape
        crop_center = self._determine_crop_center(ref_shape)
        crop_offsets = self._calculate_crop_offsets(crop_center)

        result_dict = {}

        for key in self.keys:
            if key not in normalized_dict:
                raise KeyError(f"Key '{key}' not found in normalized data")

            volume = normalized_dict[key]

            try:
                # Crop the volume
                if hasattr(volume, 'data'):
                    cropped_data = volume.data[
                        crop_offsets[0]:crop_offsets[0] + self.crop_size[0],
                        crop_offsets[1]:crop_offsets[1] + self.crop_size[1],
                        crop_offsets[2]:crop_offsets[2] + self.crop_size[2]
                    ]

                    cropped_volume = self._create_cropped_volume(cropped_data, volume)
                else:
                    raise TypeError(f"Unsupported volume type for key '{key}'")

                # Convert to tensor if requested
                if self._should_convert_to_tensor():
                    result_dict[key] = self._convert_to_tensor(cropped_volume)
                else:
                    result_dict[key] = cropped_volume

            except Exception as e:
                raise RuntimeError(f"Failed to crop {key}: {e}")

        # Pass through non-volume keys unchanged
        for key, value in data_dict.items():
            if key not in self.keys:
                result_dict[key] = value

        return result_dict

    def _should_convert_to_tensor(self) -> bool:
        """Check if conversion to tensor is requested."""
        return self.dtype is not None or self.device != "cpu"

    def _convert_to_tensor(self, volume):
        """Convert volume to tensor with specified parameters."""
        tensor = volume.to_torch(
            device=self.device,
            dtype=self.dtype
        )
        return tensor

    def _create_cropped_volume(self, cropped_data, original_volume):
        """Create a cropped volume preserving metadata."""
        affine = getattr(original_volume, "affine", None)
        data = np.asarray(cropped_data, dtype=np.float32, order="C")
        return original_volume.__class__(data, affine)


# ============================================================================
# MONAI COMPATIBILITY
# ============================================================================

class MonaiCompatibleTransform:
    """
    MONAI-compatible wrapper for medrs transforms.

    Usage:
        transform = MonaiCompatibleTransform(
            loader=CoordinatedCropLoader(keys, crop_size)
        )
        # In MONAI pipeline
        pipeline = Compose([transform, other_monai_transforms])
    """

    def __init__(self, loader: Transform):
        """Initialize MONAI-compatible transform."""
        self.loader = loader

    def __call__(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Apply transformation with MONAI compatibility."""
        return self.loader(data_dict)


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_multimodal_crop_transform(
    keys: Sequence[str],
    crop_size: Tuple[int, int, int],
    target_spacing: Optional[Tuple[float, float, float]] = None,
    device: str = "cpu",
    dtype: Optional[Any] = None,
    **kwargs
) -> CoordinatedCropLoader:
    """
    Create a transform for cropping multiple modalities.

    Args:
        keys: Keys of volumes to crop.
        crop_size: Desired crop size.
        target_spacing: Target spacing for spatial normalization.
        device: Target device.
        dtype: Target data type.
        **kwargs: Additional arguments for CoordinatedCropLoader.

    Returns:
        CoordinatedCropLoader instance.
    """
    normalizer = SpatialNormalizer(
        target_spacing=target_spacing,
        target_orientation="RAS"
    )

    return CoordinatedCropLoader(
        keys=keys,
        crop_size=crop_size,
        spatial_normalizer=normalizer,
        device=device,
        dtype=dtype,
        **kwargs
    )


def create_monai_compatible_crop(
    keys: Sequence[str],
    crop_size: Tuple[int, int, int],
    **kwargs
) -> MonaiCompatibleTransform:
    """
    Create a MONAI-compatible crop transform.

    Args:
        keys: Keys to crop.
        crop_size: Desired crop size.
        **kwargs: Additional arguments for CoordinatedCropLoader.

    Returns:
        MONAI-compatible transform.
    """
    loader = create_multimodal_crop_transform(keys, crop_size, **kwargs)
    return MonaiCompatibleTransform(loader)
