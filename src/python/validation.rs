//! Validation helpers for Python boundary.
//!
//! This module provides validation functions for Python API inputs,
//! ensuring safe and predictable error messages at the Python interface.

use crate::error::Error as MedrsError;
use ndarray::{ArrayD, IxDyn, ShapeBuilder};
use pyo3::exceptions::{PyFileNotFoundError, PyValueError};

/// Convert a medrs Error to the appropriate Python exception.
///
/// This provides proper exception types for different error conditions.
pub fn to_py_err(e: MedrsError, context: &str) -> pyo3::PyErr {
    match &e {
        MedrsError::Io(io_err) => {
            // Map I/O errors to PyIOError
            pyo3::exceptions::PyIOError::new_err(format!("{}: {}", context, io_err))
        }
        MedrsError::MemoryAllocation(msg) => {
            pyo3::exceptions::PyMemoryError::new_err(format!("{}: {}", context, msg))
        }
        MedrsError::InvalidDimensions(msg)
        | MedrsError::InvalidAffine(msg)
        | MedrsError::InvalidCropRegion(msg)
        | MedrsError::ShapeMismatch(msg)
        | MedrsError::InvalidFileFormat(msg)
        | MedrsError::InvalidOrientation(msg)
        | MedrsError::NonContiguousArray(msg)
        | MedrsError::Configuration(msg)
        | MedrsError::Decompression(msg)
        | MedrsError::Exhausted(msg) => PyValueError::new_err(format!("{}: {}", context, msg)),
        MedrsError::InvalidMagic(magic) => PyValueError::new_err(format!(
            "{}: invalid NIfTI magic bytes {:?}",
            context, magic
        )),
        MedrsError::UnsupportedDataType(code) => {
            PyValueError::new_err(format!("{}: unsupported data type code {}", context, code))
        }
        MedrsError::DataTypeMismatch { expected, got } => PyValueError::new_err(format!(
            "{}: data type mismatch (expected {}, got {})",
            context, expected, got
        )),
        MedrsError::TransformError { operation, reason } => {
            PyValueError::new_err(format!("{}: {} failed: {}", context, operation, reason))
        }
    }
}

/// Validate shape array has positive dimensions.
pub fn validate_shape(shape: &[usize; 3], name: &str) -> pyo3::PyResult<()> {
    for (i, &dim) in shape.iter().enumerate() {
        if dim == 0 {
            return Err(PyValueError::new_err(format!(
                "{} dimension {} must be positive (got 0)",
                name, i
            )));
        }
    }
    Ok(())
}

/// Validate spacing array has positive values.
pub fn validate_spacing(spacing: &[f32; 3], name: &str) -> pyo3::PyResult<()> {
    for (i, &s) in spacing.iter().enumerate() {
        if s <= 0.0 {
            return Err(PyValueError::new_err(format!(
                "{} dimension {} must be positive (got {})",
                name, i, s
            )));
        }
        if !s.is_finite() {
            return Err(PyValueError::new_err(format!(
                "{} dimension {} must be finite (got {})",
                name, i, s
            )));
        }
    }
    Ok(())
}

/// Validate a 3-element shape vector and return it as an array.
pub fn parse_shape3(values: &[usize], name: &str) -> pyo3::PyResult<[usize; 3]> {
    if values.len() != 3 {
        return Err(PyValueError::new_err(format!(
            "{} must be a 3-element sequence (got {})",
            name,
            values.len()
        )));
    }
    let shape = [values[0], values[1], values[2]];
    validate_shape(&shape, name)?;
    Ok(shape)
}

/// Validate file path is safe and exists.
pub fn validate_file_path(path: &str, operation: &str) -> pyo3::PyResult<std::path::PathBuf> {
    if path.is_empty() {
        return Err(PyValueError::new_err(format!(
            "{}: file path cannot be empty",
            operation
        )));
    }

    // Check for null bytes (prevents injection in C APIs)
    if path.contains('\0') {
        return Err(PyValueError::new_err(format!(
            "{}: file path cannot contain null bytes",
            operation
        )));
    }

    let path_buf = std::path::PathBuf::from(path);

    // For loading operations, check if file exists
    if operation.contains("load") || operation.contains("read") {
        if !path_buf.exists() {
            return Err(PyFileNotFoundError::new_err(format!(
                "{}: file not found: {}",
                operation, path
            )));
        }

        if !path_buf.is_file() {
            return Err(PyValueError::new_err(format!(
                "{}: path is not a file: {}",
                operation, path
            )));
        }
    }

    // For saving operations, check if parent directory exists
    if operation.contains("save") || operation.contains("write") {
        if let Some(parent) = path_buf.parent() {
            if !parent.exists() {
                return Err(PyFileNotFoundError::new_err(format!(
                    "{}: parent directory does not exist: {}",
                    operation,
                    parent.display()
                )));
            }
        }
    }

    Ok(path_buf)
}

/// Validate intensity range parameters.
pub fn validate_intensity_range(min: f64, max: f64, param_name: &str) -> pyo3::PyResult<()> {
    if !min.is_finite() {
        return Err(PyValueError::new_err(format!(
            "{}: min value must be finite (got {})",
            param_name, min
        )));
    }
    if !max.is_finite() {
        return Err(PyValueError::new_err(format!(
            "{}: max value must be finite (got {})",
            param_name, max
        )));
    }
    if min > max {
        return Err(PyValueError::new_err(format!(
            "{}: min ({}) cannot be greater than max ({})",
            param_name, min, max
        )));
    }
    Ok(())
}

/// Validate probability value (0.0 to 1.0).
pub fn validate_probability(p: f64, param_name: &str) -> pyo3::PyResult<()> {
    if !p.is_finite() {
        return Err(PyValueError::new_err(format!(
            "{}: probability must be finite (got {})",
            param_name, p
        )));
    }
    if !(0.0..=1.0).contains(&p) {
        return Err(PyValueError::new_err(format!(
            "{}: probability must be between 0.0 and 1.0 (got {})",
            param_name, p
        )));
    }
    Ok(())
}

/// Shared helper for creating NiftiImage from numpy array.
///
/// Handles F-order conversion and validation.
pub fn create_nifti_from_numpy_array(
    arr: ndarray::ArrayViewD<'_, f32>,
    affine: Option<[[f32; 4]; 4]>,
) -> pyo3::PyResult<crate::nifti::NiftiImage> {
    let shape = arr.shape();

    if shape.len() < 3 {
        return Err(PyValueError::new_err(
            "Array must have at least 3 dimensions (D,H,W)",
        ));
    }

    // Validate that no dimension is zero
    for (i, &dim) in shape.iter().enumerate() {
        if dim == 0 {
            return Err(PyValueError::new_err(format!(
                "Array dimension {} cannot be 0",
                i
            )));
        }
    }

    // Check for integer overflow when casting to u16 (NIfTI header limitation)
    for (i, &dim) in shape.iter().enumerate() {
        if dim > u16::MAX as usize {
            return Err(PyValueError::new_err(format!(
                "Array dimension {} ({}) exceeds maximum NIfTI dimension size ({})",
                i,
                dim,
                u16::MAX
            )));
        }
    }

    // Create F-order array to match NIfTI convention
    // Use as_slice_memory_order to get data in physical layout
    let data_vec: Vec<f32> = if let Some(slice) = arr.as_slice_memory_order() {
        slice.to_vec()
    } else {
        // Fallback: iterate in logical order
        arr.iter().copied().collect()
    };

    // Determine if input is F-order
    let is_f_order = !arr.is_standard_layout() && arr.as_slice_memory_order().is_some();

    let array = if is_f_order {
        // Input was F-order, data_vec is in F-order
        ArrayD::from_shape_vec(IxDyn(shape).f(), data_vec)
            .map_err(|e| PyValueError::new_err(format!("Invalid array shape: {}", e)))?
    } else {
        // Input was C-order, convert to F-order
        let c_order = ArrayD::from_shape_vec(shape.to_vec(), data_vec)
            .map_err(|e| PyValueError::new_err(format!("Invalid array shape: {}", e)))?;
        let mut f_order = ArrayD::zeros(IxDyn(shape).f());
        f_order.assign(&c_order);
        f_order
    };

    let affine = affine.unwrap_or([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]);

    Ok(crate::nifti::NiftiImage::from_array(array, affine))
}
