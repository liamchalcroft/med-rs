//! Tensor conversion utilities for Python integration.
//!
//! This module provides efficient conversions between Rust NIfTI data
//! and Python tensor formats (numpy, PyTorch, JAX).

use pyo3::prelude::*;

use crate::nifti::image::ArrayData;
use crate::nifti::{DataType, NiftiImage as RustNiftiImage};
use memmap2::Mmap;
use numpy::{IntoPyArray, PyArrayDyn, PyArrayMethods};
use std::sync::Arc;

#[pyclass(frozen)]
struct MmapContainer {
    #[allow(dead_code)]
    mmap: Arc<Mmap>,
}

/// Convert ArrayData to a numpy array, preserving native dtype where possible.
/// The arrays already have correct shape, so no reshape is needed.
/// Uses into_pyarray for zero-copy transfer of owned data.
#[allow(clippy::unnecessary_wraps)]
pub fn arraydata_to_numpy(
    py: Python<'_>,
    data: &ArrayData,
    _shape: &[usize],
) -> PyResult<PyObject> {
    Ok(match data {
        // For standard types, clone and move into numpy (into_pyarray moves, no extra copy)
        ArrayData::U8(a) => a.to_owned().into_pyarray(py).unbind().into_any(),
        ArrayData::I8(a) => a.to_owned().into_pyarray(py).unbind().into_any(),
        ArrayData::I16(a) => a.to_owned().into_pyarray(py).unbind().into_any(),
        ArrayData::U16(a) => a.to_owned().into_pyarray(py).unbind().into_any(),
        ArrayData::I32(a) => a.to_owned().into_pyarray(py).unbind().into_any(),
        ArrayData::U32(a) => a.to_owned().into_pyarray(py).unbind().into_any(),
        ArrayData::I64(a) => a.to_owned().into_pyarray(py).unbind().into_any(),
        ArrayData::U64(a) => a.to_owned().into_pyarray(py).unbind().into_any(),
        // F16/BF16 not supported by numpy, convert to f32
        ArrayData::F16(a) => a.mapv(|v| v.to_f32()).into_pyarray(py).unbind().into_any(),
        ArrayData::BF16(a) => a.mapv(|v| v.to_f32()).into_pyarray(py).unbind().into_any(),
        ArrayData::F32(a) => a.to_owned().into_pyarray(py).unbind().into_any(),
        ArrayData::F64(a) => a.to_owned().into_pyarray(py).unbind().into_any(),
    })
}

/// Try to create a numpy array from a view of f32 data.
/// This still copies data (PyArrayDyn::from_array copies), but avoids
/// materializing non-f32 data to f32 in Rust first.
/// Returns None if image is not already f32 or not viewable.
pub fn to_numpy_view<'py>(
    py: Python<'py>,
    image: &RustNiftiImage,
) -> Option<Bound<'py, PyArrayDyn<f32>>> {
    let slope = image.header().scl_slope;
    let inter = image.header().scl_inter;
    let has_scaling = (slope != 0.0 && slope != 1.0) || inter != 0.0;
    if has_scaling {
        return None;
    }

    // Only use view path if data is already f32 and contiguous
    // This avoids double conversion (materialize + to_f32) for non-f32 data
    if let Some(view) = image.as_view_f32() {
        // Note: from_array copies data, but we avoid Rust-side to_f32() conversion
        // which would allocate a new array. For mmap'd f32 data this is optimal.
        let arr = PyArrayDyn::from_array(py, &view);
        return Some(arr);
    }
    None
}

/// Convert image to f32 numpy array (materializes if needed).
pub fn to_numpy_array<'py>(
    py: Python<'py>,
    image: &RustNiftiImage,
) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
    // Fast path: zero-copy for native f32 mmap data without scaling
    if image.can_zero_copy_f32() {
        return to_numpy_zero_copy(py, image);
    }

    // Standard path: materialize and convert
    let data = image
        .to_f32()
        .map_err(|e| super::validation::to_py_err(e, "to_f32"))?;

    // Data is already in F-order (column-major) from NIfTI
    // Use into_pyarray which moves data without copying for contiguous arrays
    // The array maintains its F-order layout in memory
    Ok(data.into_pyarray(py))
}

/// Create a zero-copy numpy array view of mmap'd data.
///
/// This creates a numpy array that directly references to mmap'd file data.
/// The array is read-only and shares memory with the file.
///
/// Returns error if zero-copy is not possible.
#[allow(unsafe_code)]
pub fn to_numpy_zero_copy<'py>(
    py: Python<'py>,
    image: &RustNiftiImage,
) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
    // Check all zero-copy preconditions
    if !image.can_zero_copy_f32() {
        let reasons = collect_zero_copy_failure_reasons(image);
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Zero-copy access not possible: {}. Use copy=True to get a copy of the data.",
            reasons.join(", ")
        )));
    }

    let view = image.as_view_f32().ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(
            "Zero-copy failed: data is not viewable as native f32",
        )
    })?;
    let mmap = image.mmap_arc().ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("Zero-copy failed: data is not mmap-backed")
    })?;

    let container = Py::new(py, MmapContainer { mmap })?;
    let container = container.into_bound(py).into_any();

    // SAFETY: `view` points into the mmap buffer and `container` keeps the Arc<Mmap>
    // alive for the lifetime of the numpy array. The data is not reallocated.
    let array = unsafe { PyArrayDyn::borrow_from_array(&view, container) };
    array.readwrite().make_nonwriteable();
    Ok(array)
}

/// Collect reasons why zero-copy is not possible (for error messages).
pub fn collect_zero_copy_failure_reasons(image: &RustNiftiImage) -> Vec<&'static str> {
    let mut reasons = Vec::new();

    // Check storage type
    if image.raw_bytes().is_none() {
        reasons.push("data is not mmap-backed (use uncompressed .nii files)");
    }

    // Check endianness
    let native_le = cfg!(target_endian = "little");
    if image.header().is_little_endian() != native_le {
        reasons.push("non-native endianness");
    }

    // Check dtype
    if image.dtype() != DataType::Float32 {
        reasons.push("data type is not float32");
    }

    // Check scaling
    let slope = image.header().scl_slope;
    let inter = image.header().scl_inter;
    if (slope != 0.0 && slope != 1.0) || inter != 0.0 {
        reasons.push("scaling factors present (slope != 1 or intercept != 0)");
    }

    if image.as_view_f32().is_none() {
        reasons.push("data is not viewable as native f32");
    }

    reasons
}

/// Try to get native-dtype numpy view of image data.
///
/// Returns None if data type not supported by numpy's from_array.
pub fn to_numpy_view_native(py: Python<'_>, image: &RustNiftiImage) -> Option<PyObject> {
    let arr_obj = match image.dtype() {
        DataType::UInt8 => image
            .as_view_t::<u8>()
            .map(|v| PyArrayDyn::from_array(py, &v).unbind().into_any()),
        DataType::Int8 => image
            .as_view_t::<i8>()
            .map(|v| PyArrayDyn::from_array(py, &v).unbind().into_any()),
        DataType::Int16 => image
            .as_view_t::<i16>()
            .map(|v| PyArrayDyn::from_array(py, &v).unbind().into_any()),
        DataType::UInt16 => image
            .as_view_t::<u16>()
            .map(|v| PyArrayDyn::from_array(py, &v).unbind().into_any()),
        DataType::Int32 => image
            .as_view_t::<i32>()
            .map(|v| PyArrayDyn::from_array(py, &v).unbind().into_any()),
        DataType::UInt32 => image
            .as_view_t::<u32>()
            .map(|v| PyArrayDyn::from_array(py, &v).unbind().into_any()),
        DataType::Int64 => image
            .as_view_t::<i64>()
            .map(|v| PyArrayDyn::from_array(py, &v).unbind().into_any()),
        DataType::UInt64 => image
            .as_view_t::<u64>()
            .map(|v| PyArrayDyn::from_array(py, &v).unbind().into_any()),
        // numpy crate lacks f16/bf16 Element impl, fall back to f32
        DataType::Float16 | DataType::BFloat16 => None,
        DataType::Float32 => image
            .as_view_t::<f32>()
            .map(|v| PyArrayDyn::from_array(py, &v).unbind().into_any()),
        DataType::Float64 => image
            .as_view_t::<f64>()
            .map(|v| PyArrayDyn::from_array(py, &v).unbind().into_any()),
    };
    arr_obj
}

/// Convert medrs DataType to PyTorch dtype.
///
/// Returns None if dtype is not supported by PyTorch.
pub fn torch_dtype(py: Python<'_>, dtype: DataType) -> Option<PyObject> {
    let torch = py.import("torch").ok()?;
    let dt = match dtype {
        DataType::UInt8 => "uint8",
        DataType::Int8 => "int8",
        DataType::Int16 => "int16",
        DataType::Int32 => "int32",
        DataType::Int64 => "int64",
        DataType::Float16 => "float16",
        DataType::BFloat16 => "bfloat16",
        DataType::Float32 => "float32",
        DataType::Float64 => "float64",
        // PyTorch doesn't support unsigned 16/32/64-bit integers
        DataType::UInt16 | DataType::UInt32 | DataType::UInt64 => return None,
    };
    torch.getattr(dt).ok().map(|o| o.unbind())
}
