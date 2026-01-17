//! Transform functions and TransformPipeline for Python bindings.
//!
//! This module provides transform functions and the TransformPipeline class
//! for composable, lazy-evaluated image transformations.

use pyo3::prelude::*;
use pyo3::PyRefMut;

use super::image::PyNiftiImage;
use super::validation::to_py_err;
use crate::pipeline::TransformPipeline as RustTransformPipeline;
use crate::transforms::{self, Interpolation};

/// Z-score normalize an image (zero mean, unit variance).
#[pyfunction]
pub fn z_normalization(image: &PyNiftiImage) -> PyResult<PyNiftiImage> {
    Ok(PyNiftiImage {
        inner: transforms::z_normalization(&image.inner)
            .map_err(|e| to_py_err(e, "z_normalization"))?,
    })
}

/// Rescale intensity to the provided range.
#[pyfunction]
#[pyo3(signature = (image, output_range=(0.0, 1.0)))]
pub fn rescale_intensity(image: &PyNiftiImage, output_range: (f64, f64)) -> PyResult<PyNiftiImage> {
    let (out_min, out_max) = output_range;
    Ok(PyNiftiImage {
        inner: transforms::rescale_intensity(&image.inner, out_min, out_max)
            .map_err(|e| to_py_err(e, "rescale_intensity"))?,
    })
}

/// Clamp intensity values into a fixed range.
#[pyfunction]
pub fn clamp(image: &PyNiftiImage, min_value: f64, max_value: f64) -> PyResult<PyNiftiImage> {
    super::validation::validate_intensity_range(min_value, max_value, "clamp")?;
    Ok(PyNiftiImage {
        inner: transforms::clamp(&image.inner, min_value, max_value)
            .map_err(|e| to_py_err(e, "clamp"))?,
    })
}

/// Crop or pad an image to the target shape.
#[pyfunction]
pub fn crop_or_pad(image: &PyNiftiImage, target_shape: Vec<usize>) -> PyResult<PyNiftiImage> {
    if target_shape.len() != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "target_shape must be a 3-element sequence",
        ));
    }

    Ok(PyNiftiImage {
        inner: transforms::crop_or_pad(&image.inner, &target_shape)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?,
    })
}

/// Resample to target voxel spacing.
#[pyfunction]
#[pyo3(signature = (image, target_spacing, method=None))]
pub fn resample(
    image: &PyNiftiImage,
    target_spacing: (f32, f32, f32),
    method: Option<&str>,
) -> PyResult<PyNiftiImage> {
    let interp = match method.unwrap_or("trilinear") {
        "trilinear" | "linear" => Interpolation::Trilinear,
        "nearest" => Interpolation::Nearest,
        other => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "method must be 'trilinear' or 'nearest', got {}",
                other
            )))
        }
    };

    let spacing = [target_spacing.0, target_spacing.1, target_spacing.2];

    let resampled =
        transforms::resample_to_spacing(&image.inner, spacing, interp).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Resampling failed: {}", e))
        })?;
    Ok(PyNiftiImage { inner: resampled })
}

/// Reorient an image to the target orientation code (e.g., RAS or LPS).
#[pyfunction]
pub fn reorient(image: &PyNiftiImage, orientation: &str) -> PyResult<PyNiftiImage> {
    let target: crate::transforms::Orientation = orientation
        .parse()
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{}", e)))?;

    Ok(PyNiftiImage {
        inner: transforms::reorient(&image.inner, target).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Reorientation failed: {}", e))
        })?,
    })
}

/// Composable transform pipeline with lazy evaluation.
///
/// Build transformation chains that are optimized and applied efficiently.
/// Supports method chaining for a fluent API similar to MONAI's Compose.
///
/// Example:
///     >>> pipeline = medrs.TransformPipeline()
///     >>>     .z_normalize()
///     >>>     .clamp(-1.0, 1.0)
///     >>>     .resample_to_shape([64, 64, 64])
///     >>> processed = pipeline.apply(img)
#[pyclass(name = "TransformPipeline")]
pub struct PyTransformPipeline {
    inner: RustTransformPipeline,
}

#[pymethods]
impl PyTransformPipeline {
    /// Create a new transform pipeline.
    ///
    /// Args:
    ///     lazy: Enable lazy evaluation (default: True). When True, transforms
    ///           are composed and optimized before execution.
    #[new]
    #[pyo3(signature = (lazy=true))]
    fn new(lazy: bool) -> Self {
        let inner = if lazy {
            RustTransformPipeline::new()
        } else {
            RustTransformPipeline::new().lazy(false)
        };
        Self { inner }
    }

    /// Add z-score normalization (zero mean, unit variance).
    ///
    /// Returns:
    ///     Self for method chaining
    fn z_normalize(mut self_: PyRefMut<'_, Self>) -> PyRefMut<'_, Self> {
        self_.inner = std::mem::take(&mut self_.inner).z_normalize();
        self_
    }

    /// Add intensity rescaling to range [out_min, out_max].
    ///
    /// Args:
    ///     out_min: Minimum output value
    ///     out_max: Maximum output value
    ///
    /// Returns:
    ///     Self for method chaining
    fn rescale(mut self_: PyRefMut<'_, Self>, out_min: f32, out_max: f32) -> PyRefMut<'_, Self> {
        self_.inner = std::mem::take(&mut self_.inner).rescale(out_min, out_max);
        self_
    }

    /// Add intensity clamping to range [min, max].
    ///
    /// Args:
    ///     min: Minimum value
    ///     max: Maximum value
    ///
    /// Returns:
    ///     Self for method chaining
    fn clamp(mut self_: PyRefMut<'_, Self>, min: f32, max: f32) -> PyRefMut<'_, Self> {
        self_.inner = std::mem::take(&mut self_.inner).clamp(min, max);
        self_
    }

    /// Add resampling to target voxel spacing.
    ///
    /// Args:
    ///     spacing: Target spacing as [x, y, z] in mm
    ///
    /// Returns:
    ///     Self for method chaining
    fn resample_to_spacing(mut self_: PyRefMut<'_, Self>, spacing: [f32; 3]) -> PyRefMut<'_, Self> {
        self_.inner = std::mem::take(&mut self_.inner).resample_to_spacing(spacing);
        self_
    }

    /// Add resampling to target shape.
    ///
    /// Args:
    ///     shape: Target shape as [depth, height, width]
    ///
    /// Returns:
    ///     Self for method chaining
    fn resample_to_shape(mut self_: PyRefMut<'_, Self>, shape: [usize; 3]) -> PyRefMut<'_, Self> {
        self_.inner = std::mem::take(&mut self_.inner).resample_to_shape(shape);
        self_
    }

    /// Add flip along specified axes.
    ///
    /// Args:
    ///     axes: List of axes to flip (0=depth, 1=height, 2=width)
    ///
    /// Returns:
    ///     Self for method chaining
    fn flip(mut self_: PyRefMut<'_, Self>, axes: Vec<usize>) -> PyRefMut<'_, Self> {
        self_.inner = std::mem::take(&mut self_.inner).flip(&axes);
        self_
    }

    /// Enable or disable lazy evaluation.
    ///
    /// Args:
    ///     lazy: Whether to use lazy evaluation
    ///
    /// Returns:
    ///     Self for method chaining
    fn set_lazy(mut self_: PyRefMut<'_, Self>, lazy: bool) -> PyRefMut<'_, Self> {
        self_.inner = std::mem::take(&mut self_.inner).lazy(lazy);
        self_
    }

    /// Apply pipeline to an image.
    ///
    /// Args:
    ///     image: Input NiftiImage
    ///
    /// Returns:
    ///     Transformed NiftiImage
    ///
    /// Raises:
    ///     ValueError: If pipeline fails to apply
    fn apply(&self, image: &PyNiftiImage) -> PyResult<PyNiftiImage> {
        let result = self
            .inner
            .apply(&image.inner)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(PyNiftiImage { inner: result })
    }

    #[allow(clippy::unused_self)]
    fn __repr__(&self) -> String {
        "TransformPipeline(...)".to_string()
    }
}
