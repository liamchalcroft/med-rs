//! Random augmentation functions for Python bindings.
//!
//! This module provides reproducible random augmentations for medical imaging
//! with optional seeding for training workflows.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::image::PyNiftiImage;
use super::validation::validate_probability;
use crate::transforms::{self as transforms};

/// Apply random flip along specified axes with given probability.
///
/// Args:
///     image: Input image
///     axes: Axes that may be flipped (0=depth, 1=height, 2=width)
///     prob: Probability of flipping each axis (default: 0.5)
///     seed: Optional random seed for reproducibility
///
/// Returns:
///     Augmented image
///
/// Example:
///     >>> augmented = medrs.random_flip(img, [0, 1, 2], prob=0.5)
#[pyfunction]
#[pyo3(signature = (image, axes, prob=None, seed=None))]
pub fn random_flip(
    image: &PyNiftiImage,
    axes: Vec<usize>,
    prob: Option<f32>,
    seed: Option<u64>,
) -> PyResult<PyNiftiImage> {
    // Validate probability if provided
    if let Some(p) = prob {
        validate_probability(p as f64, "random_flip")?;
    }

    // Validate axes
    for &axis in &axes {
        if axis >= 3 {
            return Err(PyValueError::new_err(format!(
                "random_flip: axis {} is out of range (must be 0, 1, or 2)",
                axis
            )));
        }
    }

    Ok(PyNiftiImage {
        inner: transforms::random_flip(&image.inner, &axes, prob, seed)
            .map_err(|e| PyValueError::new_err(e.to_string()))?,
    })
}

/// Add random Gaussian noise to image.
///
/// Args:
///     image: Input image
///     std: Standard deviation of noise (default: 0.1)
///     seed: Optional random seed for reproducibility
///
/// Returns:
///     Augmented image
#[pyfunction]
#[pyo3(signature = (image, std=None, seed=None))]
pub fn random_gaussian_noise(
    image: &PyNiftiImage,
    std: Option<f32>,
    seed: Option<u64>,
) -> PyResult<PyNiftiImage> {
    Ok(PyNiftiImage {
        inner: transforms::random_gaussian_noise(&image.inner, std, seed)
            .map_err(|e| super::validation::to_py_err(e, "random_gaussian_noise"))?,
    })
}

/// Randomly scale image intensity.
///
/// Multiplies intensity by a random factor sampled from [1-scale_range, 1+scale_range].
///
/// Args:
///     image: Input image
///     scale_range: Range for random scaling factor (default: 0.1, meaning 0.9-1.1)
///     seed: Optional random seed for reproducibility
///
/// Returns:
///     Augmented image
#[pyfunction]
#[pyo3(signature = (image, scale_range=None, seed=None))]
pub fn random_intensity_scale(
    image: &PyNiftiImage,
    scale_range: Option<f32>,
    seed: Option<u64>,
) -> PyResult<PyNiftiImage> {
    Ok(PyNiftiImage {
        inner: transforms::random_intensity_scale(&image.inner, scale_range, seed)
            .map_err(|e| super::validation::to_py_err(e, "random_intensity_scale"))?,
    })
}

/// Randomly shift image intensity.
///
/// Adds a random offset sampled from [-shift_range, shift_range].
///
/// Args:
///     image: Input image
///     shift_range: Range for random shift (default: 0.1)
///     seed: Optional random seed for reproducibility
///
/// Returns:
///     Augmented image
#[pyfunction]
#[pyo3(signature = (image, shift_range=None, seed=None))]
pub fn random_intensity_shift(
    image: &PyNiftiImage,
    shift_range: Option<f32>,
    seed: Option<u64>,
) -> PyResult<PyNiftiImage> {
    Ok(PyNiftiImage {
        inner: transforms::random_intensity_shift(&image.inner, shift_range, seed)
            .map_err(|e| super::validation::to_py_err(e, "random_intensity_shift"))?,
    })
}

/// Randomly rotate image by 90-degree increments.
///
/// Performs random rotation in specified plane by 0, 90, 180, or 270 degrees.
///
/// Args:
///     image: Input image
///     axes: Tuple of two axes defining rotation plane (e.g., (0, 1) for depth-height plane)
///     seed: Optional random seed for reproducibility
///
/// Returns:
///     Augmented image
#[pyfunction]
#[pyo3(signature = (image, axes, seed=None))]
pub fn random_rotate_90(
    image: &PyNiftiImage,
    axes: (usize, usize),
    seed: Option<u64>,
) -> PyResult<PyNiftiImage> {
    Ok(PyNiftiImage {
        inner: transforms::random_rotate_90(&image.inner, axes, seed)
            .map_err(|e| PyValueError::new_err(e.to_string()))?,
    })
}

/// Apply random gamma correction to image intensity.
///
/// Applies the transform: output = input^gamma where gamma is randomly sampled.
///
/// Args:
///     image: Input image (should be normalized to [0, 1] for best results)
///     gamma_range: Range for gamma sampling as (min, max) (default: (0.7, 1.5))
///     seed: Optional random seed for reproducibility
///
/// Returns:
///     Augmented image
#[pyfunction]
#[pyo3(signature = (image, gamma_range=None, seed=None))]
pub fn random_gamma(
    image: &PyNiftiImage,
    gamma_range: Option<(f32, f32)>,
    seed: Option<u64>,
) -> PyResult<PyNiftiImage> {
    Ok(PyNiftiImage {
        inner: transforms::random_gamma(&image.inner, gamma_range, seed)
            .map_err(|e| super::validation::to_py_err(e, "random_gamma"))?,
    })
}

/// Apply a random combination of common augmentations.
///
/// This is a convenience function that applies multiple augmentations:
/// - Random flip (prob=0.5 per axis)
/// - Random intensity scale
/// - Random intensity shift
/// - Random Gaussian noise
///
/// Args:
///     image: Input image
///     seed: Optional random seed for reproducibility
///
/// Returns:
///     Augmented image
#[pyfunction]
#[pyo3(signature = (image, seed=None))]
pub fn random_augment(image: &PyNiftiImage, seed: Option<u64>) -> PyResult<PyNiftiImage> {
    Ok(PyNiftiImage {
        inner: transforms::random_augment(&image.inner, seed)
            .map_err(|e| PyValueError::new_err(e.to_string()))?,
    })
}
