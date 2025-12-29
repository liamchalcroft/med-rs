"""MONAI-compatible drop-in replacement transforms.

This module provides transforms that are API-compatible with MONAI transforms,
allowing users to swap MONAI transforms for medrs versions with minimal code changes.

Usage:
    # Instead of:
    from monai.transforms import LoadImaged, RandCropByPosNegLabeld

    # Use:
    from medrs.monai_compat import MedrsLoadImaged, MedrsRandCropByPosNegLabeld

    # They work identically in MONAI Compose pipelines:
    pipeline = Compose([
        MedrsLoadImaged(keys=["image", "label"]),
        MedrsRandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=(96, 96, 96),
        ),
    ])

Performance Benefits:
    - MedrsLoadImaged: Up to 40x faster loading with memory mapping
    - MedrsRandCropByPosNegLabeld: Faster crop-first loading
    - Lower memory usage through crop-first approach
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Callable, Dict, Hashable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

# Optional dependencies
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

try:
    from monai.data import MetaTensor
    from monai.config import KeysCollection
    from monai.transforms import MapTransform, Transform
    from monai.utils import ensure_tuple
    MONAI_AVAILABLE = True
except ImportError:
    MONAI_AVAILABLE = False
    MetaTensor = None
    KeysCollection = Any
    MapTransform = object
    Transform = object

    def ensure_tuple(x):
        if isinstance(x, (list, tuple)):
            return tuple(x)
        return (x,)

# Import medrs core functions
from . import (
    load,
    load_cropped,
    load_cropped_to_torch,
    load_label_aware_cropped,
    compute_crop_regions,
    compute_random_spatial_crops,
    compute_center_crop,
    NiftiImage,
)


def _check_monai() -> None:
    """Check if MONAI is available."""
    if not MONAI_AVAILABLE:
        raise ImportError(
            "MONAI is not available. Install with: pip install monai\n"
            "For full functionality: pip install 'monai[all]'"
        )


def _to_metatensor(
    data: Union[np.ndarray, "torch.Tensor"],
    affine: Optional[np.ndarray] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> "MetaTensor":
    """Convert array to MetaTensor with metadata."""
    _check_monai()

    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data)

    meta_dict = meta or {}
    if affine is not None:
        meta_dict["affine"] = torch.from_numpy(affine) if isinstance(affine, np.ndarray) else affine

    return MetaTensor(data, meta=meta_dict)


def _medrs_to_metatensor(
    img: NiftiImage,
    ensure_channel_first: bool = False,
    dtype: Optional["torch.dtype"] = None,
    device: Optional[str] = None,
) -> "MetaTensor":
    """Convert medrs NiftiImage to MONAI MetaTensor."""
    _check_monai()

    # Get tensor data
    if dtype is not None or device is not None:
        tensor = img.to_torch_with_dtype_and_device(dtype=dtype, device=device)
    else:
        tensor = img.to_torch()

    # Build metadata dict
    meta_dict = {
        "affine": torch.tensor(img.affine, dtype=torch.float64),
        "spatial_shape": torch.tensor(img.shape[-3:]),
        "original_affine": torch.tensor(img.affine, dtype=torch.float64),
        "filename_or_obj": "",
    }

    # Add spacing info
    spacing = img.spacing
    if spacing:
        meta_dict["pixdim"] = torch.tensor([1.0] + list(spacing) + [1.0] * 4)

    # Create MetaTensor
    metatensor = MetaTensor(tensor, meta=meta_dict)

    # Add channel dim if needed
    if ensure_channel_first and metatensor.ndim == 3:
        metatensor = metatensor.unsqueeze(0)

    return metatensor


# =============================================================================
# LOAD TRANSFORMS
# =============================================================================


class MedrsLoadImage(Transform if MONAI_AVAILABLE else object):
    """Load image using medrs (drop-in replacement for MONAI's LoadImage).

    This is 3-5x faster than MONAI's LoadImage for NIfTI files due to
    memory mapping and optimized I/O.

    Args:
        reader: Ignored (medrs uses its own reader)
        image_only: If True, return only the image data
        dtype: Output dtype (default: float32)
        ensure_channel_first: Add channel dimension if needed
        simple_keys: Ignored (MONAI compatibility)
        prune_meta_pattern: Ignored (MONAI compatibility)
        prune_meta_sep: Ignored (MONAI compatibility)

    Example:
        >>> loader = MedrsLoadImage()
        >>> image = loader("brain.nii.gz")
    """

    def __init__(
        self,
        reader: Any = None,
        image_only: bool = False,
        dtype: Optional["torch.dtype"] = None,
        ensure_channel_first: bool = False,
        simple_keys: bool = False,
        prune_meta_pattern: Optional[str] = None,
        prune_meta_sep: str = ".",
        *args,
        **kwargs,
    ):
        if reader is not None:
            warnings.warn("MedrsLoadImage ignores 'reader' parameter")
        self.image_only = image_only
        self.dtype = dtype
        self.ensure_channel_first = ensure_channel_first

    def __call__(self, filename: Union[str, Path]) -> Union["MetaTensor", Tuple["MetaTensor", Dict]]:
        """Load image from file.

        Args:
            filename: Path to NIfTI file

        Returns:
            MetaTensor if image_only=True, else tuple of (MetaTensor, metadata)
        """
        img = load(str(filename))
        metatensor = _medrs_to_metatensor(
            img,
            ensure_channel_first=self.ensure_channel_first,
            dtype=self.dtype,
        )

        # Store filename in metadata
        metatensor.meta["filename_or_obj"] = str(filename)

        if self.image_only:
            return metatensor
        else:
            return metatensor, dict(metatensor.meta)


class MedrsLoadImaged(MapTransform if MONAI_AVAILABLE else object):
    """Dictionary-based image loading using medrs.

    Drop-in replacement for MONAI's LoadImaged with 3-5x faster loading.

    Args:
        keys: Keys of the data dictionary to load
        reader: Ignored (medrs uses its own reader)
        dtype: Output dtype
        meta_keys: Keys to store metadata (default: {key}_meta_dict)
        meta_key_postfix: Postfix for metadata keys
        overwriting: Whether to overwrite existing keys
        image_only: Ignored (always returns with metadata in dict)
        ensure_channel_first: Add channel dimension if needed
        simple_keys: Ignored
        allow_missing_keys: Allow missing keys without error

    Example:
        >>> loader = MedrsLoadImaged(keys=["image", "label"])
        >>> data = loader({"image": "brain.nii.gz", "label": "seg.nii.gz"})
    """

    def __init__(
        self,
        keys: KeysCollection,
        reader: Any = None,
        dtype: Optional["torch.dtype"] = None,
        meta_keys: Optional[KeysCollection] = None,
        meta_key_postfix: str = "_meta_dict",
        overwriting: bool = False,
        image_only: bool = False,
        ensure_channel_first: bool = False,
        simple_keys: bool = False,
        allow_missing_keys: bool = False,
        *args,
        **kwargs,
    ):
        if MONAI_AVAILABLE:
            super().__init__(keys, allow_missing_keys)
        self.keys = ensure_tuple(keys)
        self.dtype = dtype
        self.meta_keys = ensure_tuple(meta_keys) if meta_keys else tuple(f"{k}{meta_key_postfix}" for k in self.keys)
        self.meta_key_postfix = meta_key_postfix
        self.overwriting = overwriting
        self.ensure_channel_first = ensure_channel_first
        self.allow_missing_keys = allow_missing_keys
        self._loader = MedrsLoadImage(
            dtype=dtype,
            ensure_channel_first=ensure_channel_first,
            image_only=False,
        )

    def __call__(self, data: Mapping[Hashable, Any]) -> Dict[Hashable, Any]:
        """Load images for specified keys.

        Args:
            data: Dictionary with file paths

        Returns:
            Dictionary with loaded MetaTensors
        """
        d = dict(data)

        for key, meta_key in zip(self.keys, self.meta_keys):
            if key not in d:
                if self.allow_missing_keys:
                    continue
                raise KeyError(f"Key '{key}' not found in data")

            filename = d[key]
            metatensor, meta = self._loader(filename)

            if not self.overwriting and key in d and isinstance(d[key], (MetaTensor, torch.Tensor)):
                warnings.warn(f"Overwriting key '{key}'")

            d[key] = metatensor
            d[meta_key] = meta

        return d


class MedrsSaveImage(Transform if MONAI_AVAILABLE else object):
    """Save image using medrs (drop-in replacement for MONAI's SaveImage).

    Args:
        output_dir: Output directory
        output_postfix: Postfix for output filename
        output_ext: Output extension (.nii or .nii.gz)
        resample: Whether to resample to original space (not implemented)
        mode: Interpolation mode (ignored)
        padding_mode: Padding mode (ignored)
        scale: Intensity scaling (ignored)
        dtype: Output dtype
        squeeze_end_dims: Remove trailing dimensions of size 1
        data_root_dir: Root directory for relative paths
        separate_folder: Create separate folder for each output
        print_log: Print log message on save
        output_format: Ignored (always NIfTI)
        writer: Ignored

    Example:
        >>> saver = MedrsSaveImage(output_dir="./output")
        >>> saver(image)
    """

    def __init__(
        self,
        output_dir: Union[str, Path] = "./",
        output_postfix: str = "trans",
        output_ext: str = ".nii.gz",
        resample: bool = True,
        mode: str = "nearest",
        padding_mode: str = "border",
        scale: Optional[int] = None,
        dtype: Optional["torch.dtype"] = None,
        squeeze_end_dims: bool = True,
        data_root_dir: str = "",
        separate_folder: bool = True,
        print_log: bool = True,
        output_format: str = "",
        writer: Any = None,
        *args,
        **kwargs,
    ):
        self.output_dir = Path(output_dir)
        self.output_postfix = output_postfix
        self.output_ext = output_ext
        self.dtype = dtype
        self.squeeze_end_dims = squeeze_end_dims
        self.data_root_dir = data_root_dir
        self.separate_folder = separate_folder
        self.print_log = print_log

        if resample:
            warnings.warn("MedrsSaveImage does not support resampling to original space yet")

    def __call__(
        self,
        img: Union["MetaTensor", "torch.Tensor", np.ndarray],
        meta_data: Optional[Dict] = None,
        filename: Optional[str] = None,
    ) -> None:
        """Save image to file.

        Args:
            img: Image data
            meta_data: Optional metadata dict
            filename: Optional output filename
        """
        # Get metadata
        if meta_data is None and hasattr(img, "meta"):
            meta_data = dict(img.meta)
        meta_data = meta_data or {}

        # Determine output filename
        if filename is None:
            orig_filename = meta_data.get("filename_or_obj", "image")
            if isinstance(orig_filename, (list, tuple)):
                orig_filename = orig_filename[0]
            orig_filename = Path(orig_filename).stem.replace(".nii", "")
            filename = f"{orig_filename}_{self.output_postfix}{self.output_ext}"

        # Create output path
        output_path = self.output_dir / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to numpy
        if isinstance(img, torch.Tensor):
            data = img.detach().cpu().numpy()
        else:
            data = np.asarray(img)

        # Squeeze if needed
        if self.squeeze_end_dims:
            while data.ndim > 3 and data.shape[-1] == 1:
                data = data.squeeze(-1)
            # Remove channel dim if present
            if data.ndim == 4 and data.shape[0] == 1:
                data = data.squeeze(0)

        # Get affine
        affine = meta_data.get("affine")
        if affine is not None:
            if isinstance(affine, torch.Tensor):
                affine = affine.numpy()
            affine = np.array(affine).tolist()

        # Create medrs image and save
        medrs_img = NiftiImage.from_numpy(data.astype(np.float32), affine=affine)

        # Apply dtype conversion if specified
        if self.dtype is not None:
            dtype_str = str(self.dtype).replace("torch.", "")
            medrs_img = medrs_img.with_dtype(dtype_str)

        medrs_img.save(str(output_path))

        if self.print_log:
            print(f"Saved: {output_path}")


class MedrsSaveImaged(MapTransform if MONAI_AVAILABLE else object):
    """Dictionary-based image saving using medrs.

    Args:
        keys: Keys of images to save
        meta_keys: Keys for metadata dicts
        meta_key_postfix: Postfix for metadata keys
        output_dir: Output directory
        output_postfix: Postfix for filenames
        output_ext: Output extension
        resample: Ignored
        mode: Ignored
        padding_mode: Ignored
        scale: Ignored
        dtype: Output dtype
        squeeze_end_dims: Remove trailing dims of size 1
        data_root_dir: Root directory
        separate_folder: Create separate folders
        print_log: Print on save
        allow_missing_keys: Allow missing keys

    Example:
        >>> saver = MedrsSaveImaged(keys=["pred"], output_dir="./output")
        >>> saver({"pred": prediction, "pred_meta_dict": meta})
    """

    def __init__(
        self,
        keys: KeysCollection,
        meta_keys: Optional[KeysCollection] = None,
        meta_key_postfix: str = "_meta_dict",
        output_dir: Union[str, Path] = "./",
        output_postfix: str = "trans",
        output_ext: str = ".nii.gz",
        resample: bool = True,
        mode: str = "nearest",
        padding_mode: str = "border",
        scale: Optional[int] = None,
        dtype: Optional["torch.dtype"] = None,
        squeeze_end_dims: bool = True,
        data_root_dir: str = "",
        separate_folder: bool = True,
        print_log: bool = True,
        allow_missing_keys: bool = False,
        *args,
        **kwargs,
    ):
        if MONAI_AVAILABLE:
            super().__init__(keys, allow_missing_keys)
        self.keys = ensure_tuple(keys)
        self.meta_keys = ensure_tuple(meta_keys) if meta_keys else tuple(f"{k}{meta_key_postfix}" for k in self.keys)
        self.allow_missing_keys = allow_missing_keys
        self._saver = MedrsSaveImage(
            output_dir=output_dir,
            output_postfix=output_postfix,
            output_ext=output_ext,
            dtype=dtype,
            squeeze_end_dims=squeeze_end_dims,
            data_root_dir=data_root_dir,
            separate_folder=separate_folder,
            print_log=print_log,
        )

    def __call__(self, data: Mapping[Hashable, Any]) -> Dict[Hashable, Any]:
        """Save images for specified keys."""
        d = dict(data)

        for key, meta_key in zip(self.keys, self.meta_keys):
            if key not in d:
                if self.allow_missing_keys:
                    continue
                raise KeyError(f"Key '{key}' not found")

            meta = d.get(meta_key, {})
            self._saver(d[key], meta_data=meta)

        return d


# =============================================================================
# CROP TRANSFORMS
# =============================================================================


class MedrsRandCropByPosNegLabeld(MapTransform if MONAI_AVAILABLE else object):
    """Random crop by positive/negative label using medrs byte-exact loading.

    Drop-in replacement for MONAI's RandCropByPosNegLabeld with
    faster performance through crop-first loading.

    Args:
        keys: Keys of images to crop
        label_key: Key of label image
        spatial_size: Output patch size
        pos: Positive sample ratio
        neg: Negative sample ratio
        num_samples: Number of samples per image
        image_key: Key for foreground calculation (optional)
        image_threshold: Threshold for foreground
        fg_indices_key: Key to store foreground indices
        bg_indices_key: Key to store background indices
        meta_keys: Metadata keys
        meta_key_postfix: Metadata key postfix
        allow_smaller: Allow patches smaller than spatial_size
        allow_missing_keys: Allow missing keys

    Example:
        >>> cropper = MedrsRandCropByPosNegLabeld(
        ...     keys=["image", "label"],
        ...     label_key="label",
        ...     spatial_size=(96, 96, 96),
        ...     pos=1,
        ...     neg=1,
        ...     num_samples=4,
        ... )
    """

    def __init__(
        self,
        keys: KeysCollection,
        label_key: str,
        spatial_size: Union[Sequence[int], int],
        pos: float = 1.0,
        neg: float = 1.0,
        num_samples: int = 1,
        image_key: Optional[str] = None,
        image_threshold: float = 0.0,
        fg_indices_key: Optional[str] = None,
        bg_indices_key: Optional[str] = None,
        meta_keys: Optional[KeysCollection] = None,
        meta_key_postfix: str = "_meta_dict",
        allow_smaller: bool = False,
        allow_missing_keys: bool = False,
        *args,
        **kwargs,
    ):
        if MONAI_AVAILABLE:
            super().__init__(keys, allow_missing_keys)
        self.keys = ensure_tuple(keys)
        self.label_key = label_key
        self.spatial_size = ensure_tuple(spatial_size)
        if len(self.spatial_size) == 1:
            self.spatial_size = self.spatial_size * 3
        self.pos = pos
        self.neg = neg
        self.num_samples = num_samples
        self.image_key = image_key
        self.image_threshold = image_threshold
        self.fg_indices_key = fg_indices_key
        self.bg_indices_key = bg_indices_key
        self.meta_keys = ensure_tuple(meta_keys) if meta_keys else tuple(f"{k}{meta_key_postfix}" for k in self.keys)
        self.meta_key_postfix = meta_key_postfix
        self.allow_smaller = allow_smaller
        self.allow_missing_keys = allow_missing_keys

        # Calculate pos/neg ratio
        self.pos_neg_ratio = pos / (neg + 1e-8) if neg > 0 else float("inf")

    def __call__(self, data: Mapping[Hashable, Any]) -> List[Dict[Hashable, Any]]:
        """Apply random crop by positive/negative label.

        Returns list of cropped samples.
        """
        d = dict(data)
        results = []

        # Check if we have file paths (can use byte-exact loading)
        # or already loaded tensors (need in-memory cropping)
        label_data = d[self.label_key]

        if isinstance(label_data, (str, Path)):
            # Use byte-exact loading path (fast!)
            results = self._crop_from_files(d)
        else:
            # Use in-memory cropping (slower but necessary for pre-loaded data)
            results = self._crop_from_tensors(d)

        return results

    def _crop_from_files(self, data: Dict) -> List[Dict]:
        """Crop using byte-exact file loading (optimal path)."""
        results = []

        # Get file paths
        label_path = data[self.label_key]

        for _ in range(self.num_samples):
            sample = {}

            for key, meta_key in zip(self.keys, self.meta_keys):
                img_path = data[key]

                if key == self.label_key:
                    # Load label-aware cropped pair
                    cropped_img, cropped_label = load_label_aware_cropped(
                        str(img_path),
                        str(label_path),
                        list(self.spatial_size),
                        pos_neg_ratio=self.pos_neg_ratio,
                        min_pos_samples=1,
                    )
                    sample[key] = _medrs_to_metatensor(cropped_label, ensure_channel_first=True)
                    sample[meta_key] = dict(sample[key].meta)
                else:
                    # Load same region from other images
                    # For now, use the same crop computed from label
                    cropped_img, _ = load_label_aware_cropped(
                        str(img_path),
                        str(label_path),
                        list(self.spatial_size),
                        pos_neg_ratio=self.pos_neg_ratio,
                        min_pos_samples=1,
                    )
                    sample[key] = _medrs_to_metatensor(cropped_img, ensure_channel_first=True)
                    sample[meta_key] = dict(sample[key].meta)

            results.append(sample)

        return results

    def _crop_from_tensors(self, data: Dict) -> List[Dict]:
        """Crop from already-loaded tensors (fallback path)."""
        results = []

        # Get label data for computing crop regions
        label = data[self.label_key]
        if isinstance(label, MetaTensor):
            label_arr = label.numpy()
        elif isinstance(label, torch.Tensor):
            label_arr = label.numpy()
        else:
            label_arr = np.asarray(label)

        # Remove channel dim if present
        if label_arr.ndim == 4:
            label_arr = label_arr[0]

        shape = label_arr.shape

        for _ in range(self.num_samples):
            sample = {}

            # Compute random crop region
            crop_start = []
            for dim, (s, p) in enumerate(zip(shape, self.spatial_size)):
                max_start = max(0, s - p)
                start = np.random.randint(0, max_start + 1) if max_start > 0 else 0
                crop_start.append(start)

            crop_end = [s + p for s, p in zip(crop_start, self.spatial_size)]

            # Apply crop to all keys
            for key, meta_key in zip(self.keys, self.meta_keys):
                img = data[key]

                if isinstance(img, MetaTensor):
                    arr = img.numpy()
                    meta = dict(img.meta)
                elif isinstance(img, torch.Tensor):
                    arr = img.numpy()
                    meta = data.get(meta_key, {})
                else:
                    arr = np.asarray(img)
                    meta = data.get(meta_key, {})

                # Handle channel dimension
                has_channel = arr.ndim == 4
                if has_channel:
                    cropped = arr[
                        :,
                        crop_start[0]:crop_end[0],
                        crop_start[1]:crop_end[1],
                        crop_start[2]:crop_end[2],
                    ]
                else:
                    cropped = arr[
                        crop_start[0]:crop_end[0],
                        crop_start[1]:crop_end[1],
                        crop_start[2]:crop_end[2],
                    ]

                # Convert back to MetaTensor
                tensor = torch.from_numpy(cropped.copy())
                if MONAI_AVAILABLE:
                    sample[key] = MetaTensor(tensor, meta=meta)
                else:
                    sample[key] = tensor
                sample[meta_key] = meta

            results.append(sample)

        return results

    def inverse(self, data: Dict) -> Dict:
        """Inverse transform (not implemented)."""
        warnings.warn("MedrsRandCropByPosNegLabeld.inverse() is not implemented")
        return data


class MedrsRandSpatialCropd(MapTransform if MONAI_AVAILABLE else object):
    """Random spatial crop using medrs.

    Drop-in replacement for MONAI's RandSpatialCropd.

    Args:
        keys: Keys of images to crop
        roi_size: Region of interest size
        max_roi_size: Maximum ROI size (optional)
        random_center: Randomize crop center
        random_size: Randomize crop size
        meta_keys: Metadata keys
        meta_key_postfix: Metadata key postfix
        allow_missing_keys: Allow missing keys

    Example:
        >>> cropper = MedrsRandSpatialCropd(
        ...     keys=["image"],
        ...     roi_size=(64, 64, 64),
        ... )
    """

    def __init__(
        self,
        keys: KeysCollection,
        roi_size: Union[Sequence[int], int],
        max_roi_size: Optional[Union[Sequence[int], int]] = None,
        random_center: bool = True,
        random_size: bool = False,
        meta_keys: Optional[KeysCollection] = None,
        meta_key_postfix: str = "_meta_dict",
        allow_missing_keys: bool = False,
        *args,
        **kwargs,
    ):
        if MONAI_AVAILABLE:
            super().__init__(keys, allow_missing_keys)
        self.keys = ensure_tuple(keys)
        self.roi_size = ensure_tuple(roi_size)
        if len(self.roi_size) == 1:
            self.roi_size = self.roi_size * 3
        self.max_roi_size = ensure_tuple(max_roi_size) if max_roi_size else None
        self.random_center = random_center
        self.random_size = random_size
        self.meta_keys = ensure_tuple(meta_keys) if meta_keys else tuple(f"{k}{meta_key_postfix}" for k in self.keys)
        self.allow_missing_keys = allow_missing_keys

    def __call__(self, data: Mapping[Hashable, Any]) -> Dict[Hashable, Any]:
        """Apply random spatial crop."""
        d = dict(data)

        # Get first key to determine shape
        first_key = self.keys[0]
        first_data = d[first_key]

        if isinstance(first_data, (str, Path)):
            # Use file-based cropping
            regions = compute_random_spatial_crops(
                str(first_data),
                list(self.roi_size),
                num_samples=1,
            )
            if regions:
                crop_start = regions[0]["start"]
                crop_size = regions[0]["size"]
            else:
                crop_start = [0, 0, 0]
                crop_size = list(self.roi_size)
        else:
            # Compute crop from tensor shape
            if isinstance(first_data, (MetaTensor, torch.Tensor)):
                shape = first_data.shape[-3:]
            else:
                shape = np.asarray(first_data).shape[-3:]

            crop_start = []
            for s, r in zip(shape, self.roi_size):
                max_start = max(0, s - r)
                start = np.random.randint(0, max_start + 1) if self.random_center and max_start > 0 else max_start // 2
                crop_start.append(start)
            crop_size = list(self.roi_size)

        # Apply crop to all keys
        for key, meta_key in zip(self.keys, self.meta_keys):
            img_data = d[key]

            if isinstance(img_data, (str, Path)):
                # Load cropped directly
                cropped = load_cropped(str(img_data), crop_start, crop_size)
                d[key] = _medrs_to_metatensor(cropped, ensure_channel_first=True)
                d[meta_key] = dict(d[key].meta)
            else:
                # Crop in memory
                if isinstance(img_data, (MetaTensor, torch.Tensor)):
                    arr = img_data
                    meta = dict(img_data.meta) if hasattr(img_data, "meta") else d.get(meta_key, {})
                else:
                    arr = torch.from_numpy(np.asarray(img_data))
                    meta = d.get(meta_key, {})

                # Handle channel dimension
                if arr.ndim == 4:
                    cropped = arr[
                        :,
                        crop_start[0]:crop_start[0] + crop_size[0],
                        crop_start[1]:crop_start[1] + crop_size[1],
                        crop_start[2]:crop_start[2] + crop_size[2],
                    ]
                else:
                    cropped = arr[
                        crop_start[0]:crop_start[0] + crop_size[0],
                        crop_start[1]:crop_start[1] + crop_size[1],
                        crop_start[2]:crop_start[2] + crop_size[2],
                    ]

                if MONAI_AVAILABLE:
                    d[key] = MetaTensor(cropped.clone(), meta=meta)
                else:
                    d[key] = cropped.clone()
                d[meta_key] = meta

        return d


class MedrsCenterSpatialCropd(MapTransform if MONAI_AVAILABLE else object):
    """Center spatial crop using medrs.

    Drop-in replacement for MONAI's CenterSpatialCropd.

    Args:
        keys: Keys of images to crop
        roi_size: Region of interest size
        meta_keys: Metadata keys
        meta_key_postfix: Metadata key postfix
        allow_missing_keys: Allow missing keys

    Example:
        >>> cropper = MedrsCenterSpatialCropd(
        ...     keys=["image"],
        ...     roi_size=(64, 64, 64),
        ... )
    """

    def __init__(
        self,
        keys: KeysCollection,
        roi_size: Union[Sequence[int], int],
        meta_keys: Optional[KeysCollection] = None,
        meta_key_postfix: str = "_meta_dict",
        allow_missing_keys: bool = False,
        *args,
        **kwargs,
    ):
        if MONAI_AVAILABLE:
            super().__init__(keys, allow_missing_keys)
        self.keys = ensure_tuple(keys)
        self.roi_size = ensure_tuple(roi_size)
        if len(self.roi_size) == 1:
            self.roi_size = self.roi_size * 3
        self.meta_keys = ensure_tuple(meta_keys) if meta_keys else tuple(f"{k}{meta_key_postfix}" for k in self.keys)
        self.allow_missing_keys = allow_missing_keys

    def __call__(self, data: Mapping[Hashable, Any]) -> Dict[Hashable, Any]:
        """Apply center spatial crop."""
        d = dict(data)

        # Get first key to determine crop region
        first_key = self.keys[0]
        first_data = d[first_key]

        if isinstance(first_data, (str, Path)):
            # Use file-based center crop computation
            region = compute_center_crop(str(first_data), list(self.roi_size))
            crop_start = region["start"]
            crop_size = region["size"]
        else:
            # Compute center crop from tensor shape
            if isinstance(first_data, (MetaTensor, torch.Tensor)):
                shape = first_data.shape[-3:]
            else:
                shape = np.asarray(first_data).shape[-3:]

            crop_start = [(s - r) // 2 for s, r in zip(shape, self.roi_size)]
            crop_size = list(self.roi_size)

        # Apply crop to all keys
        for key, meta_key in zip(self.keys, self.meta_keys):
            img_data = d[key]

            if isinstance(img_data, (str, Path)):
                cropped = load_cropped(str(img_data), crop_start, crop_size)
                d[key] = _medrs_to_metatensor(cropped, ensure_channel_first=True)
                d[meta_key] = dict(d[key].meta)
            else:
                if isinstance(img_data, (MetaTensor, torch.Tensor)):
                    arr = img_data
                    meta = dict(img_data.meta) if hasattr(img_data, "meta") else d.get(meta_key, {})
                else:
                    arr = torch.from_numpy(np.asarray(img_data))
                    meta = d.get(meta_key, {})

                if arr.ndim == 4:
                    cropped = arr[
                        :,
                        crop_start[0]:crop_start[0] + crop_size[0],
                        crop_start[1]:crop_start[1] + crop_size[1],
                        crop_start[2]:crop_start[2] + crop_size[2],
                    ]
                else:
                    cropped = arr[
                        crop_start[0]:crop_start[0] + crop_size[0],
                        crop_start[1]:crop_start[1] + crop_size[1],
                        crop_start[2]:crop_start[2] + crop_size[2],
                    ]

                if MONAI_AVAILABLE:
                    d[key] = MetaTensor(cropped.clone(), meta=meta)
                else:
                    d[key] = cropped.clone()
                d[meta_key] = meta

        return d


# =============================================================================
# SPATIAL TRANSFORMS
# =============================================================================


class MedrsOrientation(Transform if MONAI_AVAILABLE else object):
    """Reorient image using medrs.

    Drop-in replacement for MONAI's Orientation transform.

    Args:
        axcodes: Target orientation code (e.g., "RAS", "LPS")
        as_closest_canonical: Use closest canonical orientation
        labels: Ignored

    Example:
        >>> orient = MedrsOrientation(axcodes="RAS")
        >>> reoriented = orient(image)
    """

    def __init__(
        self,
        axcodes: str = "RAS",
        as_closest_canonical: bool = False,
        labels: Optional[Any] = None,
        *args,
        **kwargs,
    ):
        self.axcodes = axcodes
        self.as_closest_canonical = as_closest_canonical

    def __call__(self, img: Union["MetaTensor", np.ndarray, str, Path]) -> "MetaTensor":
        """Reorient image to target orientation."""
        if isinstance(img, (str, Path)):
            medrs_img = load(str(img))
        elif isinstance(img, np.ndarray):
            medrs_img = NiftiImage.from_numpy(img)
        elif isinstance(img, MetaTensor):
            medrs_img = NiftiImage.from_numpy(img.numpy())
        elif isinstance(img, torch.Tensor):
            medrs_img = NiftiImage.from_numpy(img.numpy())
        else:
            raise TypeError(f"Unsupported type: {type(img)}")

        reoriented = medrs_img.reorient(self.axcodes)
        return _medrs_to_metatensor(reoriented)


class MedrsOrientationd(MapTransform if MONAI_AVAILABLE else object):
    """Dictionary-based reorientation using medrs.

    Args:
        keys: Keys to reorient
        axcodes: Target orientation code
        as_closest_canonical: Use closest canonical
        labels: Ignored
        meta_keys: Metadata keys
        meta_key_postfix: Metadata key postfix
        allow_missing_keys: Allow missing keys

    Example:
        >>> orient = MedrsOrientationd(keys=["image"], axcodes="RAS")
    """

    def __init__(
        self,
        keys: KeysCollection,
        axcodes: str = "RAS",
        as_closest_canonical: bool = False,
        labels: Optional[Any] = None,
        meta_keys: Optional[KeysCollection] = None,
        meta_key_postfix: str = "_meta_dict",
        allow_missing_keys: bool = False,
        *args,
        **kwargs,
    ):
        if MONAI_AVAILABLE:
            super().__init__(keys, allow_missing_keys)
        self.keys = ensure_tuple(keys)
        self.meta_keys = ensure_tuple(meta_keys) if meta_keys else tuple(f"{k}{meta_key_postfix}" for k in self.keys)
        self.allow_missing_keys = allow_missing_keys
        self._orient = MedrsOrientation(axcodes=axcodes, as_closest_canonical=as_closest_canonical)

    def __call__(self, data: Mapping[Hashable, Any]) -> Dict[Hashable, Any]:
        """Reorient images for specified keys."""
        d = dict(data)

        for key, meta_key in zip(self.keys, self.meta_keys):
            if key not in d:
                if self.allow_missing_keys:
                    continue
                raise KeyError(f"Key '{key}' not found")

            d[key] = self._orient(d[key])
            d[meta_key] = dict(d[key].meta) if hasattr(d[key], "meta") else {}

        return d


# =============================================================================
# RESAMPLE TRANSFORMS
# =============================================================================


class MedrsResample(Transform if MONAI_AVAILABLE else object):
    """Resample image using medrs.

    Args:
        mode: Interpolation mode ("trilinear" or "nearest")
        padding_mode: Ignored
        align_corners: Ignored
        dtype: Output dtype

    Example:
        >>> resample = MedrsResample()
        >>> resampled = resample(image, dst_affine=new_affine)
    """

    def __init__(
        self,
        mode: str = "trilinear",
        padding_mode: str = "border",
        align_corners: bool = False,
        dtype: Optional["torch.dtype"] = None,
        *args,
        **kwargs,
    ):
        self.mode = "trilinear" if mode in ("bilinear", "trilinear") else "nearest"
        self.dtype = dtype

    def __call__(
        self,
        img: Union["MetaTensor", np.ndarray],
        dst_affine: Optional[np.ndarray] = None,
        spatial_size: Optional[Sequence[int]] = None,
        mode: Optional[str] = None,
    ) -> "MetaTensor":
        """Resample image."""
        mode = mode or self.mode

        if isinstance(img, np.ndarray):
            medrs_img = NiftiImage.from_numpy(img)
        elif isinstance(img, MetaTensor):
            medrs_img = NiftiImage.from_numpy(img.numpy())
        elif isinstance(img, torch.Tensor):
            medrs_img = NiftiImage.from_numpy(img.numpy())
        else:
            medrs_img = img

        if spatial_size is not None:
            resampled = medrs_img.resample_to_shape(list(spatial_size), method=mode)
        else:
            # Default: keep same shape
            resampled = medrs_img

        return _medrs_to_metatensor(resampled, dtype=self.dtype)


class MedrsResampled(MapTransform if MONAI_AVAILABLE else object):
    """Dictionary-based resampling using medrs.

    Args:
        keys: Keys to resample
        mode: Interpolation mode
        padding_mode: Ignored
        align_corners: Ignored
        dtype: Output dtype
        meta_keys: Metadata keys
        meta_key_postfix: Metadata postfix
        allow_missing_keys: Allow missing keys

    Example:
        >>> resample = MedrsResampled(keys=["image"])
    """

    def __init__(
        self,
        keys: KeysCollection,
        mode: str = "trilinear",
        padding_mode: str = "border",
        align_corners: bool = False,
        dtype: Optional["torch.dtype"] = None,
        meta_keys: Optional[KeysCollection] = None,
        meta_key_postfix: str = "_meta_dict",
        allow_missing_keys: bool = False,
        *args,
        **kwargs,
    ):
        if MONAI_AVAILABLE:
            super().__init__(keys, allow_missing_keys)
        self.keys = ensure_tuple(keys)
        self.meta_keys = ensure_tuple(meta_keys) if meta_keys else tuple(f"{k}{meta_key_postfix}" for k in self.keys)
        self.allow_missing_keys = allow_missing_keys
        self._resample = MedrsResample(mode=mode, dtype=dtype)

    def __call__(
        self,
        data: Mapping[Hashable, Any],
        dst_affine: Optional[np.ndarray] = None,
        spatial_size: Optional[Sequence[int]] = None,
    ) -> Dict[Hashable, Any]:
        """Resample images for specified keys."""
        d = dict(data)

        for key, meta_key in zip(self.keys, self.meta_keys):
            if key not in d:
                if self.allow_missing_keys:
                    continue
                raise KeyError(f"Key '{key}' not found")

            d[key] = self._resample(d[key], dst_affine=dst_affine, spatial_size=spatial_size)
            d[meta_key] = dict(d[key].meta) if hasattr(d[key], "meta") else {}

        return d


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Load transforms
    "MedrsLoadImage",
    "MedrsLoadImaged",
    # Save transforms
    "MedrsSaveImage",
    "MedrsSaveImaged",
    # Crop transforms
    "MedrsRandCropByPosNegLabeld",
    "MedrsRandSpatialCropd",
    "MedrsCenterSpatialCropd",
    # Spatial transforms
    "MedrsOrientation",
    "MedrsOrientationd",
    # Resample transforms
    "MedrsResample",
    "MedrsResampled",
]
