//! Crop-first loading functions for Python bindings.
//!
//! This module provides efficient crop loading for training pipelines,
//! including MONAI-compatible label-aware cropping.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use super::image::PyNiftiImage;
use super::validation::{
    parse_shape3, to_py_err, validate_file_path, validate_shape, validate_spacing,
};
use crate::nifti;
use crate::transforms::crop::{
    compute_center_crop_regions, compute_label_aware_crop_regions,
    compute_random_spatial_crop_regions, RandCropByPosNegLabelConfig, SpatialCropConfig,
};
use crate::transforms::Orientation;

/// Load only a cropped region from a NIfTI file without loading entire volume.
///
/// This is extremely efficient for training pipelines that load large volumes
/// just to crop small patches (e.g., loading 64^3 patch from 256^3 volume).
///
/// Args:
///     path: Path to NIfTI file (must be uncompressed .nii)
///     crop_offset: Starting coordinates of crop region [d, h, w]
///     crop_shape: Size of crop region [d, h, w]
///
/// Returns:
///     NiftiImage instance with cropped data
///
/// Example:
///     >>> # Load 64^3 patch starting at (32, 32, 32)
///     >>> img = medrs.load_cropped("volume.nii", [32, 32, 32], [64, 64, 64])
#[pyfunction]
pub fn load_cropped(
    path: &str,
    crop_offset: [usize; 3],
    crop_shape: [usize; 3],
) -> PyResult<PyNiftiImage> {
    let validated_path = validate_file_path(path, "load_cropped")?;
    let path_str = validated_path
        .to_str()
        .ok_or_else(|| PyValueError::new_err("path contains invalid UTF-8"))?;

    nifti::load_cropped(path_str, crop_offset, crop_shape)
        .map(|inner| PyNiftiImage { inner })
        .map_err(|e| to_py_err(e, &format!("Failed to load cropped {}", path)))
}

/// Load a cropped region with optional reorientation and resampling.
///
/// Advanced version that computes minimal region needed from
/// raw file to achieve desired output after transforms.
///
/// Args:
///     path: Path to NIfTI file (must be uncompressed .nii)
///     output_shape: Desired output shape after all transforms [d, h, w]
///     target_spacing: Optional target voxel spacing [mm] (None = keep original)
///     target_orientation: Optional target orientation code (None = keep original)
///     output_offset: Optional offset in output space for non-centered crops [d, h, w]
///
/// Returns:
///     NiftiImage instance with processed cropped data
///
/// Example:
///     >>> # Load 64^3 RAS 1mm isotropic patch from any orientation/spacing
///     >>> img = medrs.load_resampled(
///     ...     "volume.nii",
///     ...     output_shape=[64, 64, 64],
///     ...     target_spacing=[1.0, 1.0, 1.0],
///     ...     target_orientation="RAS"
///     ... )
#[pyfunction]
#[pyo3(signature = (path, output_shape, target_spacing=None, target_orientation=None, output_offset=None))]
pub fn load_resampled(
    path: &str,
    output_shape: [usize; 3],
    target_spacing: Option<[f32; 3]>,
    target_orientation: Option<String>,
    output_offset: Option<[usize; 3]>,
) -> PyResult<PyNiftiImage> {
    let validated_path = validate_file_path(path, "load_resampled")?;
    let path_str = validated_path
        .to_str()
        .ok_or_else(|| PyValueError::new_err("path contains invalid UTF-8"))?;

    let orientation = match target_orientation {
        Some(s) => Some(
            s.parse::<Orientation>()
                .map_err(|e| PyValueError::new_err(format!("{}", e)))?,
        ),
        None => None,
    };

    let config = nifti::CropConfig {
        shape: output_shape,
        offset: output_offset,
        spacing: target_spacing,
        orientation,
    };

    nifti::load_with_crop(path_str, config)
        .map(|inner| PyNiftiImage { inner })
        .map_err(|e| to_py_err(e, &format!("Failed to load cropped {}", path)))
}

/// Load a cropped region directly into a PyTorch tensor without numpy intermediate.
///
/// This is the most efficient way to load medical imaging data into PyTorch.
/// Eliminates memory copies and supports half-precision tensors directly.
///
/// Args:
///     path: Path to NIfTI file (must be uncompressed .nii)
///     output_shape: Desired output shape after all transforms [d, h, w]
///     target_spacing: Optional target voxel spacing [mm] (None = keep original)
///     target_orientation: Optional target orientation code (None = keep original)
///     output_offset: Optional offset in output space for non-centered crops [d, h, w]
///     dtype: PyTorch dtype (default: torch.float32)
///     device: PyTorch device (default: "cpu")
///
/// Returns:
///     PyTorch tensor with shape [d, h, w]
///
/// Example:
///     >>> # Load 64^3 RAS 1mm isotropic patch directly as f16 tensor
///     >>> import torch
///     >>> tensor = medrs.load_cropped_to_torch(
///     ...     "volume.nii",
///     ...     output_shape=[64, 64, 64],
///     ...     target_spacing=[1.0, 1.0, 1.0],
///     ...     target_orientation="RAS",
///     ...     dtype=torch.float16,  # Direct f16 loading!
///     ...     device="cuda"
///     ... )
#[pyfunction]
#[pyo3(signature = (path, output_shape, target_spacing=None, target_orientation=None, output_offset=None, dtype=None, device="cpu"))]
#[allow(clippy::too_many_arguments)]
pub fn load_cropped_to_torch(
    py: Python<'_>,
    path: &str,
    output_shape: [usize; 3],
    target_spacing: Option<[f32; 3]>,
    target_orientation: Option<String>,
    output_offset: Option<[usize; 3]>,
    dtype: Option<PyObject>,
    device: &str,
) -> PyResult<PyObject> {
    // Validate inputs at Python boundary
    validate_shape(&output_shape, "output_shape")?;
    if let Some(ref spacing) = target_spacing {
        validate_spacing(spacing, "target_spacing")?;
    }

    // Load image using our I/O-optimized function
    let orientation = match target_orientation {
        Some(s) => Some(
            s.parse::<Orientation>()
                .map_err(|e| PyValueError::new_err(format!("{}", e)))?,
        ),
        None => None,
    };

    let config = nifti::CropConfig {
        shape: output_shape,
        offset: output_offset,
        spacing: target_spacing,
        orientation,
    };

    let validated_path = validate_file_path(path, "load_cropped_to_torch")?;
    let path_str = validated_path
        .to_str()
        .ok_or_else(|| PyValueError::new_err("path contains invalid UTF-8"))?;

    let img = nifti::load_with_crop(path_str, config)
        .map_err(|e| to_py_err(e, &format!("Failed to load cropped {}", path)))?;

    // Convert to PyTorch tensor directly using our optimized I/O + tensor conversion
    let py_img = PyNiftiImage { inner: img };
    py_img.to_torch_with_dtype_and_device(py, dtype, Some(device))
}

/// Load a cropped region directly into a JAX array without numpy intermediate.
///
/// This is the most efficient way to load medical imaging data into JAX.
/// Eliminates memory copies and supports bfloat16/f16 directly.
///
/// Args:
///     path: Path to NIfTI file (must be uncompressed .nii)
///     output_shape: Desired output shape after all transforms [d, h, w]
///     target_spacing: Optional target voxel spacing [mm] (None = keep original)
///     target_orientation: Optional target orientation code (None = keep original)
///     output_offset: Optional offset in output space for non-centered crops [d, h, w]
///     dtype: JAX dtype (default: jax.numpy.float32)
///     device: JAX device (default: "cpu")
///
/// Returns:
///     JAX array with shape [d, h, w]
///
/// Example:
///     >>> # Load 64^3 RAS 1mm isotropic patch directly as bfloat16
///     >>> import jax
///     >>> array = medrs.load_cropped_to_jax(
///     ...     "volume.nii",
///     ...     output_shape=[64, 64, 64],
///     ...     target_spacing=[1.0, 1.0, 1.0],
///     ...     target_orientation="RAS",
///     ...     dtype=jax.numpy.bfloat16,  # Direct bfloat16 loading!
///     ...     device="cuda:0"
///     ... )
#[pyfunction]
#[pyo3(signature = (path, output_shape, target_spacing=None, target_orientation=None, output_offset=None, dtype=None, device="cpu"))]
#[allow(clippy::too_many_arguments)]
pub fn load_cropped_to_jax(
    py: Python<'_>,
    path: &str,
    output_shape: [usize; 3],
    target_spacing: Option<[f32; 3]>,
    target_orientation: Option<String>,
    output_offset: Option<[usize; 3]>,
    dtype: Option<PyObject>,
    device: &str,
) -> PyResult<PyObject> {
    // Validate inputs at Python boundary
    validate_shape(&output_shape, "output_shape")?;
    if let Some(ref spacing) = target_spacing {
        validate_spacing(spacing, "target_spacing")?;
    }

    // Load image using our I/O-optimized function
    let orientation = match target_orientation {
        Some(s) => Some(
            s.parse::<Orientation>()
                .map_err(|e| PyValueError::new_err(format!("{}", e)))?,
        ),
        None => None,
    };

    let config = nifti::CropConfig {
        shape: output_shape,
        offset: output_offset,
        spacing: target_spacing,
        orientation,
    };

    let validated_path = validate_file_path(path, "load_cropped_to_jax")?;
    let path_str = validated_path
        .to_str()
        .ok_or_else(|| PyValueError::new_err("path contains invalid UTF-8"))?;

    let img = nifti::load_with_crop(path_str, config)
        .map_err(|e| to_py_err(e, &format!("Failed to load cropped {}", path)))?;

    // Convert to JAX array directly using our optimized I/O + array conversion
    let py_img = PyNiftiImage { inner: img };
    py_img.to_jax_with_dtype_and_device(py, dtype, Some(device))
}

/// Load a NIfTI file with byte-exact cropping for label-aware training.
///
/// This function combines MONAI's `RandCropByPosNegLabeld` with medrs's
/// byte-exact loading for maximum performance. It computes optimal crop
/// regions containing both positive and negative labels, then loads only
/// required bytes.
///
/// Args:
///     image_path: Path to image file
///     label_path: Path to label file
///     patch_size: Target patch size [x, y, z]
///     pos_neg_ratio: Ratio of positive to negative samples (default: 1.0)
///     min_pos_samples: Minimum positive samples per volume (default: 4)
///     seed: Random seed for reproducibility (optional)
///
/// Returns:
///     Tuple of (cropped_image, cropped_label) as PyNiftiImages
///
/// Example:
///     >>> image, label = medrs.load_label_aware_cropped(
///     ...     "mri.nii.gz",
///     ...     "seg.nii.gz",
///     ...     patch_size=[64, 64, 64],
///     ...     pos_neg_ratio=1.0,
///     ...     min_pos_samples=4,
///     ...     seed=42
///     ... )
#[pyfunction]
#[pyo3(signature = (image_path, label_path, patch_size, pos_neg_ratio=None, min_pos_samples=None, seed=None))]
pub fn load_label_aware_cropped(
    _py: Python<'_>,
    image_path: &str,
    label_path: &str,
    patch_size: Vec<usize>,
    pos_neg_ratio: Option<f64>,
    min_pos_samples: Option<usize>,
    seed: Option<u64>,
) -> PyResult<(PyNiftiImage, PyNiftiImage)> {
    let patch_size_arr = parse_shape3(&patch_size, "patch_size")?;

    // Load full images first (this could be optimized further)
    let image = nifti::load(image_path)
        .map_err(|e| to_py_err(e, &format!("Failed to load {}", image_path)))?;
    let label = nifti::load(label_path)
        .map_err(|e| to_py_err(e, &format!("Failed to load {}", label_path)))?;

    // Configure cropping
    let config = RandCropByPosNegLabelConfig {
        patch_size: patch_size_arr,
        pos_neg_ratio: pos_neg_ratio.unwrap_or(1.0) as f32,
        min_pos_samples: min_pos_samples.unwrap_or(4),
        seed,
        background_label: 0.0,
    };

    // Compute crop regions (this is fast, operates on labels only)
    let crop_regions = compute_label_aware_crop_regions(&config, &image, &label, 1)
        .map_err(|e| to_py_err(e, "compute_label_aware_crop_regions"))?;

    if crop_regions.is_empty() {
        return Err(PyValueError::new_err("No valid crop regions found"));
    }

    let region = &crop_regions[0];

    // Load cropped regions (this is a fast byte-exact loading)
    let cropped_image = nifti::load_cropped(image_path, region.start, region.size)
        .map_err(|e| to_py_err(e, "Failed to load cropped image"))?;
    let cropped_label = nifti::load_cropped(label_path, region.start, region.size)
        .map_err(|e| to_py_err(e, "Failed to load cropped label"))?;

    Ok((
        PyNiftiImage {
            inner: cropped_image,
        },
        PyNiftiImage {
            inner: cropped_label,
        },
    ))
}

/// Compute crop regions for smart loading without actually loading data.
///
/// This function allows users to compute optimal crop regions first,
/// then use them with `load_cropped()` for maximum control.
/// This is useful for batch processing and advanced training pipelines.
///
/// Args:
///     image_path: Path to image file
///     label_path: Path to label file
///     patch_size: Target patch size [x, y, z]
///     num_samples: Number of crop regions to compute
///     pos_neg_ratio: Ratio of positive to negative samples (default: 1.0)
///     min_pos_samples: Minimum positive samples per volume (default: 4)
///     seed: Random seed for reproducibility (optional)
///
/// Returns:
///     List of crop regions as dictionaries with 'start', 'end', and 'size' keys
///
/// Example:
///     >>> regions = medrs.compute_crop_regions(
///     ...     "mri.nii.gz", "seg.nii.gz",
///     ...     patch_size=[64, 64, 64],
///     ...     num_samples=10,
///     ...     seed=42
///     ... )
#[pyfunction]
#[pyo3(signature = (image_path, label_path, patch_size, num_samples, pos_neg_ratio=None, min_pos_samples=None, seed=None))]
#[allow(clippy::too_many_arguments)]
pub fn compute_crop_regions(
    py: Python<'_>,
    image_path: &str,
    label_path: &str,
    patch_size: Vec<usize>,
    num_samples: usize,
    pos_neg_ratio: Option<f64>,
    min_pos_samples: Option<usize>,
    seed: Option<u64>,
) -> PyResult<Vec<PyObject>> {
    let patch_size_arr = parse_shape3(&patch_size, "patch_size")?;

    // Load images to get shape information
    let image = nifti::load(image_path)
        .map_err(|e| to_py_err(e, &format!("Failed to load {}", image_path)))?;
    let label = nifti::load(label_path)
        .map_err(|e| to_py_err(e, &format!("Failed to load {}", label_path)))?;

    // Configure cropping
    let config = RandCropByPosNegLabelConfig {
        patch_size: patch_size_arr,
        pos_neg_ratio: pos_neg_ratio.unwrap_or(1.0) as f32,
        min_pos_samples: min_pos_samples.unwrap_or(4),
        seed,
        background_label: 0.0,
    };

    // Compute crop regions
    let crop_regions = compute_label_aware_crop_regions(&config, &image, &label, num_samples)
        .map_err(|e| to_py_err(e, "compute_label_aware_crop_regions"))?;

    // Convert to Python dictionaries
    let mut regions_py = Vec::new();
    for region in crop_regions {
        let region_dict = PyDict::new(py);
        region_dict.set_item("start", region.start.to_vec())?;
        region_dict.set_item("end", region.end.to_vec())?;
        region_dict.set_item("size", region.size.to_vec())?;
        regions_py.push(region_dict.unbind().into());
    }

    Ok(regions_py)
}

/// Compute random spatial crop regions.
///
/// This function implements MONAI's `RandSpatialCropd` functionality
/// optimized for medrs's crop-first approach.
///
/// Args:
///     image_path: Path to image file
///     patch_size: Target patch size [x, y, z]
///     num_samples: Number of crop regions to compute
///     seed: Random seed for reproducibility (optional)
///     allow_smaller: Whether to allow smaller crops at boundaries (default: false)
///
/// Returns:
///     List of crop regions as dictionaries
///
/// Example:
///     >>> regions = medrs.compute_random_spatial_crops(
///     ...     "volume.nii",
///     ...     patch_size=[64, 64, 64],
///     ...     num_samples=10,
///     ...     seed=42
///     ... )
#[pyfunction]
#[pyo3(signature = (image_path, patch_size, num_samples, seed=None, allow_smaller=None))]
pub fn compute_random_spatial_crops(
    py: Python<'_>,
    image_path: &str,
    patch_size: Vec<usize>,
    num_samples: usize,
    seed: Option<u64>,
    allow_smaller: Option<bool>,
) -> PyResult<Vec<PyObject>> {
    let patch_size_arr = parse_shape3(&patch_size, "patch_size")?;

    // Load image to get shape information
    let image = nifti::load(image_path)
        .map_err(|e| to_py_err(e, &format!("Failed to load {}", image_path)))?;

    // Configure cropping
    let config = SpatialCropConfig {
        patch_size: patch_size_arr,
        seed,
        allow_smaller: allow_smaller.unwrap_or(false),
    };

    // Compute crop regions
    let crop_regions = compute_random_spatial_crop_regions(&config, &image, num_samples);

    // Convert to Python dictionaries
    let mut regions_py = Vec::new();
    for region in crop_regions {
        let region_dict = PyDict::new(py);
        region_dict.set_item("start", region.start.to_vec())?;
        region_dict.set_item("end", region.end.to_vec())?;
        region_dict.set_item("size", region.size.to_vec())?;
        regions_py.push(region_dict.unbind().into());
    }

    Ok(regions_py)
}

/// Compute center crop region.
///
/// This function implements MONAI's `CenterSpatialCropd` functionality
/// optimized for medrs's crop-first approach.
///
/// Args:
///     image_path: Path to image file
///     patch_size: Target patch size [x, y, z]
///
/// Returns:
///     Crop region as dictionary
///
/// Example:
///     >>> region = medrs.compute_center_crop(
///     ...     "volume.nii",
///     ...     patch_size=[64, 64, 64]
///     ... )
#[pyfunction]
pub fn compute_center_crop(
    py: Python<'_>,
    image_path: &str,
    patch_size: Vec<usize>,
) -> PyResult<PyObject> {
    let patch_size_arr = parse_shape3(&patch_size, "patch_size")?;

    // Load image to get shape information
    let image = nifti::load(image_path)
        .map_err(|e| to_py_err(e, &format!("Failed to load {}", image_path)))?;

    // Compute center crop
    let region = compute_center_crop_regions(patch_size_arr, &image);

    // Convert to Python dictionary
    let region_dict = PyDict::new(py);
    region_dict.set_item("start", region.start.to_vec())?;
    region_dict.set_item("end", region.end.to_vec())?;
    region_dict.set_item("size", region.size.to_vec())?;

    Ok(region_dict.unbind().into())
}
