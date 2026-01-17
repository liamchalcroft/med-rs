//! Basic I/O functions for Python bindings.
//!
//! This module provides fundamental NIfTI loading and saving operations
//! with caching support for repeated access.

use pyo3::prelude::*;

use super::validation::validate_file_path;
use crate::nifti;

/// Load a NIfTI image from file.
///
/// Supports both .nii and .nii.gz formats with automatic detection.
///
/// Args:
///     path: Path to the NIfTI file
///
/// Returns:
///     NiftiImage instance
///
/// Example:
///     >>> img = medrs.load("brain.nii.gz")
#[pyfunction]
pub fn load(path: &str) -> PyResult<super::image::PyNiftiImage> {
    let validated_path = validate_file_path(path, "load")?;
    let path_str = validated_path
        .to_str()
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("path contains invalid UTF-8"))?;
    nifti::load(path_str)
        .map(|inner| super::image::PyNiftiImage { inner })
        .map_err(|e| super::validation::to_py_err(e, &format!("Failed to load {}", path)))
}

/// Load a NIfTI image with caching for repeated access.
///
/// For gzipped files, this caches decompressed data so subsequent loads
/// of the same file are nearly instant (pseudo-zero-copy). For uncompressed
/// files, this behaves identically to `load()`.
///
/// This is particularly useful in training pipelines where the same volume
/// may be accessed multiple times across epochs.
///
/// Args:
///     path: Path to the NIfTI file
///
/// Returns:
///     NiftiImage instance
///
/// Example:
///     >>> # First load decompresses and caches
///     >>> img1 = medrs.load_cached("brain.nii.gz")
///     >>> # Second load returns cached data (very fast)
///     >>> img2 = medrs.load_cached("brain.nii.gz")
///     >>> # Clear cache when done to free memory
///     >>> medrs.clear_decompression_cache()
#[pyfunction]
pub fn load_cached(path: &str) -> PyResult<super::image::PyNiftiImage> {
    let validated_path = validate_file_path(path, "load_cached")?;
    let path_str = validated_path
        .to_str()
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("path contains invalid UTF-8"))?;
    nifti::load_cached(path_str)
        .map(|inner| super::image::PyNiftiImage { inner })
        .map_err(|e| super::validation::to_py_err(e, &format!("Failed to load_cached {}", path)))
}

/// Clear the global decompression cache.
///
/// Call this to free memory used by cached decompressed files from
/// `load_cached()` calls.
///
/// Example:
///     >>> medrs.clear_decompression_cache()
#[pyfunction]
pub fn clear_decompression_cache() {
    nifti::clear_decompression_cache();
}

/// Set the maximum size of the decompression cache.
///
/// Default is 10 entries. Set to 0 to disable caching.
///
/// Args:
///     max_entries: Maximum number of files to cache
///
/// Example:
///     >>> medrs.set_cache_size(20)  # Cache up to 20 files
///     >>> medrs.set_cache_size(0)   # Disable caching
#[pyfunction]
pub fn set_cache_size(max_entries: usize) {
    nifti::set_cache_size(max_entries);
}

/// Load a NIfTI image directly to a PyTorch tensor.
///
/// This is the most efficient way to load medical imaging data into PyTorch.
/// Eliminates memory copies and supports half-precision tensors directly.
///
/// Args:
///     path: Path to the NIfTI file
///     dtype: PyTorch dtype (default: torch.float32)
///     device: PyTorch device (default: "cpu")
///
/// Returns:
///     PyTorch tensor with shape matching the image
///
/// Example:
///     >>> import torch
///     >>> tensor = medrs.load_to_torch("volume.nii", dtype=torch.float16, device="cuda")
#[pyfunction]
#[pyo3(signature = (path, dtype=None, device="cpu"))]
pub fn load_to_torch(
    py: Python<'_>,
    path: &str,
    dtype: Option<PyObject>,
    device: &str,
) -> PyResult<PyObject> {
    // Load image using medrs
    let img = nifti::load(path).map_err(|e| {
        pyo3::exceptions::PyIOError::new_err(format!("Failed to load {}: {}", path, e))
    })?;

    // Convert to PyTorch tensor directly
    let py_img = super::image::PyNiftiImage { inner: img };
    py_img.to_torch_with_dtype_and_device(py, dtype, Some(device))
}

/// Save a NIfTI image in Mgzip format for parallel decompression.
///
/// Mgzip (multi-member gzip) stores data in independent blocks that can be
/// decompressed in parallel, providing 4-8x speedup on multi-core systems.
///
/// Args:
///     image: NiftiImage to save
///     path: Output file path (typically .nii.mgz)
///     num_threads: Number of compression threads (0 = auto-detect)
///
/// Example:
///     >>> img = medrs.load("brain.nii.gz")
///     >>> medrs.save_mgzip(img, "brain.nii.mgz")
#[pyfunction]
#[pyo3(signature = (image, path, num_threads=0))]
pub fn save_mgzip(
    image: &super::image::PyNiftiImage,
    path: &str,
    num_threads: usize,
) -> PyResult<()> {
    nifti::save_mgzip_with_threads(&image.inner, path, num_threads)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("save_mgzip failed: {}", e)))
}

/// Load a NIfTI image from Mgzip format with parallel decompression.
///
/// Provides 4-8x faster loading compared to standard gzip on multi-core systems.
/// The file should have been created with `save_mgzip()` for best performance.
///
/// Args:
///     path: Path to the Mgzip file
///     num_threads: Number of decompression threads (0 = auto-detect)
///
/// Returns:
///     NiftiImage instance
///
/// Example:
///     >>> img = medrs.load_mgzip("brain.nii.mgz")
#[pyfunction]
#[pyo3(signature = (path, num_threads=0))]
pub fn load_mgzip(path: &str, num_threads: usize) -> PyResult<super::image::PyNiftiImage> {
    let validated_path = validate_file_path(path, "load_mgzip")?;
    let path_str = validated_path
        .to_str()
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("path contains invalid UTF-8"))?;
    nifti::load_mgzip_with_threads(path_str, num_threads)
        .map(|inner| super::image::PyNiftiImage { inner })
        .map_err(|e| super::validation::to_py_err(e, &format!("Failed to load_mgzip {}", path)))
}

/// Convert a standard gzip NIfTI file to Mgzip format.
///
/// Reads an existing .nii.gz file and saves it as Mgzip for faster future loading.
/// The original file is not modified.
///
/// Args:
///     input_path: Path to input .nii.gz file
///     output_path: Path for output .nii.mgz file (None = auto-generate from input)
///     num_threads: Number of compression threads (0 = auto-detect)
///
/// Returns:
///     Path to the output file
///
/// Example:
///     >>> medrs.convert_to_mgzip("brain.nii.gz")  # Creates brain.nii.mgz
///     >>> medrs.convert_to_mgzip("brain.nii.gz", "output/brain.nii.mgz")
#[pyfunction]
#[pyo3(signature = (input_path, output_path=None, num_threads=0))]
pub fn convert_to_mgzip(
    input_path: &str,
    output_path: Option<&str>,
    num_threads: usize,
) -> PyResult<String> {
    let image = nifti::load(input_path).map_err(|e| {
        pyo3::exceptions::PyIOError::new_err(format!("Failed to load {}: {}", input_path, e))
    })?;

    let output = if let Some(p) = output_path { std::path::PathBuf::from(p) } else {
        let input = std::path::Path::new(input_path);
        let stem = input
            .file_stem()
            .and_then(|s| s.to_str())
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("invalid input path"))?;
        let stem = stem.strip_suffix(".nii").unwrap_or(stem);
        input.with_file_name(format!("{stem}.nii.mgz"))
    };

    nifti::save_mgzip_with_threads(&image, &output, num_threads).map_err(|e| {
        pyo3::exceptions::PyIOError::new_err(format!("convert_to_mgzip failed: {}", e))
    })?;

    Ok(output.display().to_string())
}

/// Check if a file appears to be in Mgzip (multi-member gzip) format.
///
/// Performs a quick heuristic check by looking for multiple gzip member signatures.
///
/// Args:
///     path: Path to the file to check
///
/// Returns:
///     True if file appears to be Mgzip format
///
/// Example:
///     >>> if medrs.is_mgzip("brain.nii.mgz"):
///     ...     img = medrs.load_mgzip("brain.nii.mgz")
#[pyfunction]
pub fn is_mgzip(path: &str) -> PyResult<bool> {
    nifti::is_mgzip(path)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("is_mgzip check failed: {}", e)))
}
