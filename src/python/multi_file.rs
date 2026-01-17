//! Multi-file loading functions for Python bindings.
//!
//! This module provides functions for loading related medical images
//! (e.g., MRI, CT, segmentation) with coordinated spatial processing.

use pyo3::prelude::*;
use pyo3::types::PyDict;

use super::image::PyNiftiImage;
use super::validation::to_py_err;
use crate::nifti;

/// Load multiple related files in parallel with spatial alignment.
///
/// This is the main entry point for loading related medical images (e.g., MRI, CT,
/// segmentation) that need to be processed together with consistent spatial operations.
///
/// All files are:
/// 1. Loaded in parallel using multiple threads
/// 2. Resampled to a common voxel spacing
/// 3. Cropped to the same region (if specified)
///
/// Labels/segmentations use nearest-neighbor interpolation to preserve discrete values.
///
/// Args:
///     files: List of file configurations, each containing:
///         - 'path': Path to the NIfTI file
///         - 'is_label': Whether this is a label/segmentation (uses nearest-neighbor)
///         - 'key': Optional name/key for this file
///     target_spacing: Target voxel spacing [x, y, z] (optional, uses reference if not provided)
///     crop_start: Crop region start [d, h, w] (optional)
///     crop_size: Crop region size [d, h, w] (optional)
///     reference_index: Index of reference file for spatial alignment (default: 0)
///     use_cache: Whether to use caching for gzipped files (default: true)
///
/// Returns:
///     List of NiftiImage instances in the same order as input files
///
/// Example:
///     >>> files = [
///     ...     {'path': 'mri.nii.gz', 'is_label': False, 'key': 'mri'},
///     ...     {'path': 'ct.nii.gz', 'is_label': False, 'key': 'ct'},
///     ...     {'path': 'seg.nii.gz', 'is_label': True, 'key': 'label'},
///     ... ]
///     >>> images = medrs.load_multi(
///     ...     files,
///     ...     target_spacing=[1.0, 1.0, 1.0],
///     ...     crop_start=[32, 32, 32],
///     ...     crop_size=[64, 64, 64]
///     ... )
#[pyfunction]
#[pyo3(signature = (files, target_spacing=None, crop_start=None, crop_size=None, reference_index=0, use_cache=true))]
#[allow(clippy::too_many_arguments)]
pub fn load_multi(
    py: Python<'_>,
    files: Vec<PyObject>,
    target_spacing: Option<Vec<f32>>,
    crop_start: Option<Vec<usize>>,
    crop_size: Option<Vec<usize>>,
    reference_index: usize,
    use_cache: bool,
) -> PyResult<Vec<PyNiftiImage>> {
    // Parse file configurations
    let mut file_configs = Vec::with_capacity(files.len());
    for file_obj in files {
        let dict = file_obj.downcast_bound::<PyDict>(py)?;

        let path: String = dict
            .get_item("path")?
            .ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err("Each file config must have a 'path' key")
            })?
            .extract()?;

        let is_label: bool = dict
            .get_item("is_label")?
            .is_some_and(|v| v.extract().unwrap_or(false));

        let key: Option<String> = dict.get_item("key")?.and_then(|v| v.extract().ok());

        let mut config = if is_label {
            nifti::FileConfig::label(&path)
        } else {
            nifti::FileConfig::image(&path)
        };

        if let Some(k) = key {
            config = config.with_key(&k);
        }

        file_configs.push(config);
    }

    // Build multi-file config
    let mut config = nifti::MultiFileConfig::default();
    config.reference_index = reference_index;
    config.use_cache = use_cache;

    if let Some(spacing) = target_spacing {
        if spacing.len() != 3 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "target_spacing must have 3 elements",
            ));
        }
        config.target_spacing = Some([spacing[0], spacing[1], spacing[2]]);
    }

    if let (Some(start), Some(size)) = (crop_start, crop_size) {
        if start.len() != 3 || size.len() != 3 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "crop_start and crop_size must have 3 elements each",
            ));
        }
        config.crop_start = Some([start[0], start[1], start[2]]);
        config.crop_size = Some([size[0], size[1], size[2]]);
    }

    // Load files
    let result =
        nifti::load_multi(&file_configs, config).map_err(|e| to_py_err(e, "load_multi"))?;

    // Convert to Python objects
    Ok(result
        .into_images()
        .into_iter()
        .map(|inner| PyNiftiImage { inner })
        .collect())
}

/// Load an image and label pair with coordinated spatial processing.
///
/// Convenience function for loading one image with its
/// corresponding segmentation label.
///
/// Args:
///     image_path: Path to image file
///     label_path: Path to label/segmentation file
///     target_spacing: Target voxel spacing [x, y, z] (optional)
///     crop_start: Crop region start [d, h, w] (optional)
///     crop_size: Crop region size [d, h, w] (optional)
///
/// Returns:
///     Tuple of (image, label) NiftiImage instances
///
/// Example:
///     >>> image, label = medrs.load_image_label_pair(
///     ...     "mri.nii.gz",
///     ...     "segmentation.nii.gz",
///     ...     target_spacing=[1.0, 1.0, 1.0],
///     ...     crop_start=[32, 32, 32],
///     ...     crop_size=[64, 64, 64]
///     ... )
#[pyfunction]
#[pyo3(signature = (image_path, label_path, target_spacing=None, crop_start=None, crop_size=None))]
pub fn load_image_label_pair(
    image_path: &str,
    label_path: &str,
    target_spacing: Option<Vec<f32>>,
    crop_start: Option<Vec<usize>>,
    crop_size: Option<Vec<usize>>,
) -> PyResult<(PyNiftiImage, PyNiftiImage)> {
    let spacing = target_spacing
        .map(|s| {
            if s.len() != 3 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "target_spacing must have 3 elements",
                ));
            }
            Ok([s[0], s[1], s[2]])
        })
        .transpose()?;

    let crop = match (crop_start, crop_size) {
        (Some(start), Some(size)) => {
            if start.len() != 3 || size.len() != 3 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "crop_start and crop_size must have 3 elements each",
                ));
            }
            Some(([start[0], start[1], start[2]], [size[0], size[1], size[2]]))
        }
        (None, None) => None,
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "crop_start and crop_size must both be provided or both omitted",
            ));
        }
    };

    let (image, label) = nifti::load_image_label_pair(image_path, label_path, spacing, crop)
        .map_err(|e| to_py_err(e, "load_image_label_pair"))?;

    Ok((PyNiftiImage { inner: image }, PyNiftiImage { inner: label }))
}
