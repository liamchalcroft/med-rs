#![allow(clippy::doc_markdown)]
//! Python bindings for medrs using PyO3.

use ndarray::{ArrayD, IxDyn};
use numpy::{IntoPyArray, PyArrayDyn, PyArrayMethods, PyReadonlyArrayDyn};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::IntoPyDict;

use crate::nifti::image::ArrayData;
use crate::nifti::DataType;
use crate::nifti::{self, NiftiImage as RustNiftiImage};
use crate::pipeline::TransformPipeline as RustTransformPipeline;
use crate::transforms::crop::{
    compute_center_crop_regions, compute_label_aware_crop_regions,
    compute_random_spatial_crop_regions,
};
use crate::transforms::{self, Interpolation, Orientation};

/// A NIfTI image with header metadata and voxel data.
///
/// Supports method chaining for transform operations.
///
/// Example:
///     >>> img = medrs.load("brain.nii.gz")
///     >>> processed = img.resample([1.0, 1.0, 1.0]).z_normalize().clamp(0, 1)
///     >>> processed.save("output.nii.gz")
#[pyclass(name = "NiftiImage")]
struct PyNiftiImage {
    inner: RustNiftiImage,
}

#[pymethods]
impl PyNiftiImage {
    /// Create a new NIfTI image from a numpy array (>=3D).
    ///
    /// Args:
    ///     data: numpy array of voxel values (last 3 dims are spatial)
    ///     affine: 4x4 affine transformation matrix (optional)
    #[new]
    #[pyo3(signature = (data, affine=None))]
    fn new<'py>(
        data: PyReadonlyArrayDyn<'py, f32>,
        affine: Option<[[f32; 4]; 4]>,
    ) -> PyResult<Self> {
        let arr = data.as_array();
        let shape = arr.shape();

        if shape.len() < 3 {
            return Err(PyValueError::new_err(
                "Array must have at least 3 dimensions (D,H,W)",
            ));
        }

        let data_vec: Vec<f32> = arr.iter().copied().collect();
        let array = ArrayD::from_shape_vec(shape.to_vec(), data_vec)
            .map_err(|e| PyValueError::new_err(format!("Invalid array shape: {}", e)))?;

        let affine = affine.unwrap_or([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]);

        Ok(Self {
            inner: RustNiftiImage::from_array(array, affine),
        })
    }

    /// Image shape as (depth, height, width).
    #[getter]
    fn shape(&self) -> Vec<usize> {
        self.inner.shape().to_vec()
    }

    /// Number of dimensions.
    #[getter]
    fn ndim(&self) -> usize {
        self.inner.ndim()
    }

    /// Data type as string.
    #[getter]
    fn dtype(&self) -> &'static str {
        self.inner.dtype().type_name()
    }

    /// Voxel spacing in mm.
    #[getter]
    fn spacing(&self) -> Vec<f32> {
        self.inner.spacing().to_vec()
    }

    /// 4x4 affine transformation matrix.
    #[getter]
    fn affine(&self) -> [[f32; 4]; 4] {
        self.inner.affine()
    }

    /// Set the affine transformation matrix.
    #[setter]
    fn set_affine(&mut self, affine: [[f32; 4]; 4]) {
        self.inner.set_affine(affine);
    }

    /// Image orientation code (e.g., "RAS", "LPS").
    #[getter]
    fn orientation(&self) -> String {
        let affine = self.inner.affine();
        transforms::orientation_from_affine(&affine).to_string()
    }

    /// Raw voxel data as a numpy array (float32).
    ///
    /// Uses zero-copy views when possible; may fall back to a copy for non-contiguous data.
    #[getter]
    fn data<'py>(&self, py: Python<'py>) -> Bound<'py, PyArrayDyn<f32>> {
        self.to_numpy(py)
    }

    /// Get image data as numpy array (float32).
    ///
    /// Similar to nibabel's get_fdata(). Applies scaling factors if present.
    /// Supports arbitrary ndim.
    fn to_numpy<'py>(&self, py: Python<'py>) -> Bound<'py, PyArrayDyn<f32>> {
        if let Some(arr) = to_numpy_view(py, &self.inner) {
            return arr;
        }
        to_numpy_array(py, &self.inner)
    }

    /// Get image data as numpy view when possible (may fall back to copy).
    fn to_numpy_view<'py>(&self, py: Python<'py>) -> Bound<'py, PyArrayDyn<f32>> {
        if let Some(arr) = to_numpy_view(py, &self.inner) {
            arr
        } else {
            to_numpy_array(py, &self.inner)
        }
    }

    /// Get image data as a torch tensor (shares memory when possible).
    fn to_torch<'py>(&self, py: Python<'py>) -> PyResult<PyObject> {
        let torch = py.import("torch")?;
        let dtype = torch_dtype(py, self.inner.dtype());
        let has_dtype = dtype.is_some();

        // Prefer view when dtype is torch-supported and contiguous
        let np_obj = if has_dtype {
            if let Some(np_view) = to_numpy_view_native(py, &self.inner) {
                np_view
            } else {
                arraydata_to_numpy(py, &self.inner.owned_data(), self.inner.shape())?
            }
        } else {
            to_numpy_array(py, &self.inner)
                .into_pyobject(py)?
                .into_any()
                .unbind()
        };

        let tensor = torch.getattr("from_numpy")?.call1((np_obj,))?;

        if let Some(dt) = dtype {
            let tensor = tensor.call_method1("to", (dt,))?;
            Ok(tensor.unbind())
        } else {
            Ok(tensor.unbind())
        }
    }

    /// Get image data as a JAX array (shares memory via numpy when possible).
    fn to_jax<'py>(&self, py: Python<'py>) -> PyResult<PyObject> {
        let jnp = py.import("jax.numpy")?;
        let np_obj = to_numpy_array(py, &self.inner);
        let arr = jnp.getattr("array")?.call1((np_obj,))?;
        Ok(arr.unbind())
    }

    /// Convert to PyTorch tensor with custom dtype and device.
    ///
    /// This is the most efficient way to load medical imaging data directly
    /// into PyTorch with the target precision and device placement.
    #[pyo3(signature = (dtype=None, device=None))]
    fn to_torch_with_dtype_and_device<'py>(
        &self,
        py: Python<'py>,
        dtype: Option<PyObject>,
        device: Option<&str>,
    ) -> PyResult<PyObject> {
        let device_str = device.unwrap_or("cpu");
        let torch = py.import("torch")?;

        // Get numpy array with maximum efficiency
        let np_obj = if let Some(np_view) = to_numpy_view_native(py, &self.inner) {
            np_view
        } else {
            to_numpy_array(py, &self.inner)
                .into_pyobject(py)?
                .into_any()
                .unbind()
        };

        // Convert to tensor
        let tensor = torch.getattr("from_numpy")?.call1((np_obj,))?;

        // Apply device placement
        let tensor = tensor.call_method1("to", (device_str,))?;

        // Apply dtype if specified
        if let Some(dt) = dtype {
            let tensor = tensor.call_method1("to", (dt,))?;
            Ok(tensor.unbind())
        } else {
            Ok(tensor.unbind())
        }
    }

    /// Convert to JAX array with custom dtype and device.
    ///
    /// This is the most efficient way to load medical imaging data directly
    /// into JAX with the target precision and device placement.
    #[pyo3(signature = (dtype=None, device=None))]
    fn to_jax_with_dtype_and_device<'py>(
        &self,
        py: Python<'py>,
        dtype: Option<PyObject>,
        device: Option<&str>,
    ) -> PyResult<PyObject> {
        let device_str = device.unwrap_or("cpu");
        let jax = py.import("jax")?;
        let _jnp = py.import("jax.numpy")?;

        // Get device object
        let device_obj = if device_str == "cpu" {
            jax.getattr("devices")?.call1((0,))?
        } else if device_str.starts_with("cuda") {
            let cuda_devices = jax.getattr("devices")?.call1(("cuda",))?;
            if device_str == "cuda" {
                cuda_devices.get_item(0)?
            } else {
                let device_id: usize = device_str
                    .strip_prefix("cuda:")
                    .unwrap_or("0")
                    .parse()
                    .unwrap_or(0);
                cuda_devices.get_item(device_id)?
            }
        } else {
            return Err(PyValueError::new_err(format!(
                "Unsupported device: {}",
                device_str
            )));
        };

        // Use optimized on-device array creation
        let jax = py.import("jax")?;
        let jnp = py.import("jax.numpy")?;

        // Check if we can use zero-copy numpy path for maximum efficiency
        if let Some(np_view) = to_numpy_view_native(py, &self.inner) {
            // Use numpy view for maximum efficiency
            let mut arr = jnp.getattr("array")?.call1((np_view,))?;

            // Apply dtype if specified
            if let Some(dt) = dtype {
                arr = jnp.getattr("astype")?.call1((dt,))?;
            }

            // Use jax.device_put for efficient async device placement
            let device_put = jax.getattr("device_put")?;
            let arr = device_put.call1((arr, &device_obj))?;

            return Ok(arr.into());
        }

        // Fallback to regular numpy array with efficient transfer
        let np_obj = to_numpy_array(py, &self.inner);
        let mut arr = jnp.getattr("array")?.call1((np_obj,))?;

        // Apply dtype if specified
        if let Some(dt) = dtype {
            arr = jnp.getattr("astype")?.call1((dt,))?;
        }

        // Use jax.device_put for efficient async device placement
        let device_put = jax.getattr("device_put")?;
        let arr = device_put.call1((arr, &device_obj))?;

        Ok(arr.into())
    }

    /// Get image data as a numpy array with native dtype.
    ///
    /// Half/bfloat16 are returned as float32 for compatibility.
    fn to_numpy_native<'py>(&self, py: Python<'py>) -> PyResult<PyObject> {
        Ok(arraydata_to_numpy(
            py,
            &self.inner.owned_data(),
            self.inner.shape(),
        )?)
    }

    /// Create a new NiftiImage from a numpy array.
    ///
    /// This is a convenience method for creating NIfTI images from numpy arrays.
    /// For more control, use the NiftiImage constructor directly.
    ///
    /// Args:
    ///     data: numpy array of voxel values (last 3 dims are spatial)
    ///     affine: 4x4 affine transformation matrix (optional)
    ///
    /// Returns:
    ///     NiftiImage instance
    ///
    /// Example:
    ///     >>> import numpy as np
    ///     >>> data = np.random.rand(64, 64, 32).astype(np.float32)
    ///     >>> img = NiftiImage.from_numpy(data)
    #[staticmethod]
    #[pyo3(signature = (data, affine=None))]
    fn from_numpy<'py>(
        _py: Python<'py>,
        data: PyReadonlyArrayDyn<'py, f32>,
        affine: Option<[[f32; 4]; 4]>,
    ) -> PyResult<Self> {
        use ndarray::ShapeBuilder;

        let arr = data.as_array();
        let shape = arr.shape();

        if shape.len() < 3 {
            return Err(PyValueError::new_err(
                "Array must have at least 3 dimensions (D,H,W)",
            ));
        }

        // Create F-order array to match NIfTI convention
        // Use as_slice_memory_order to get data in physical layout
        let data_vec: Vec<f32> = if let Some(slice) = arr.as_slice_memory_order() {
            slice.to_vec()
        } else {
            // Fallback: iterate in logical order and create C-order, then convert
            arr.iter().copied().collect()
        };

        // Determine if input is F-order
        let is_f_order = arr.is_standard_layout() == false && arr.as_slice_memory_order().is_some();

        let array = if is_f_order {
            // Input was F-order, data_vec is in F-order
            ArrayD::from_shape_vec(ndarray::IxDyn(shape).f(), data_vec)
                .map_err(|e| PyValueError::new_err(format!("Invalid array shape: {}", e)))?
        } else {
            // Input was C-order, convert to F-order
            let c_order = ArrayD::from_shape_vec(shape.to_vec(), data_vec)
                .map_err(|e| PyValueError::new_err(format!("Invalid array shape: {}", e)))?;
            let mut f_order = ArrayD::zeros(ndarray::IxDyn(shape).f());
            f_order.assign(&c_order);
            f_order
        };

        let affine = affine.unwrap_or([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]);

        Ok(Self {
            inner: RustNiftiImage::from_array(array, affine),
        })
    }

    /// Save image to file.
    ///
    /// Format is determined by extension (.nii or .nii.gz).
    fn save(&self, path: &str) -> PyResult<()> {
        nifti::save(&self.inner, path).map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Resample to target voxel spacing.
    ///
    /// Args:
    ///     spacing: Target spacing as [x, y, z] in mm
    ///     method: Interpolation method ("trilinear" or "nearest", default: "trilinear")
    ///
    /// Returns:
    ///     New NiftiImage (supports method chaining)
    #[pyo3(signature = (spacing, method=None))]
    fn resample(&self, spacing: [f32; 3], method: Option<&str>) -> PyResult<Self> {
        let method_str = method.unwrap_or("trilinear");
        let interp = match method_str {
            "trilinear" | "linear" => Interpolation::Trilinear,
            "nearest" => Interpolation::Nearest,
            _ => {
                return Err(PyValueError::new_err(
                    "method must be 'trilinear' or 'nearest'",
                ))
            }
        };

        Ok(Self {
            inner: transforms::resample_to_spacing(&self.inner, spacing, interp),
        })
    }

    /// Resample to target shape.
    ///
    /// Args:
    ///     shape: Target shape as [depth, height, width]
    ///     method: Interpolation method ("trilinear" or "nearest", default: "trilinear")
    ///
    /// Returns:
    ///     New NiftiImage (supports method chaining)
    #[pyo3(signature = (shape, method=None))]
    fn resample_to_shape(&self, shape: [usize; 3], method: Option<&str>) -> PyResult<Self> {
        let method_str = method.unwrap_or("trilinear");
        let interp = match method_str {
            "trilinear" | "linear" => Interpolation::Trilinear,
            "nearest" => Interpolation::Nearest,
            _ => {
                return Err(PyValueError::new_err(
                    "method must be 'trilinear' or 'nearest'",
                ))
            }
        };

        Ok(Self {
            inner: transforms::resample_to_shape(&self.inner, shape, interp),
        })
    }

    /// Reorient to target orientation.
    ///
    /// Args:
    ///     orientation: Target orientation code (e.g., "RAS", "LPS")
    ///
    /// Returns:
    ///     New NiftiImage (supports method chaining)
    fn reorient(&self, orientation: &str) -> PyResult<Self> {
        let target = Orientation::from_str(orientation).ok_or_else(|| {
            PyValueError::new_err(format!("Invalid orientation code: {}. Valid codes: RAS, LAS, LPI, RPI, ASL, ARS, ALI, ARI, IPL, SPR, IAR, SAR, IPR, SPL, IAL, SAL", orientation))
        })?;

        Ok(Self {
            inner: transforms::reorient(&self.inner, target),
        })
    }

    /// Z-score normalization (zero mean, unit variance).
    ///
    /// Returns:
    ///     New NiftiImage (supports method chaining)
    fn z_normalize(&self) -> Self {
        Self {
            inner: transforms::z_normalization(&self.inner),
        }
    }

    /// Rescale intensity to range [min, max].
    ///
    /// Args:
    ///     out_min: Minimum output value
    ///     out_max: Maximum output value
    ///
    /// Returns:
    ///     New NiftiImage (supports method chaining)
    fn rescale(&self, out_min: f64, out_max: f64) -> Self {
        Self {
            inner: transforms::rescale_intensity(&self.inner, out_min, out_max),
        }
    }

    /// Clamp intensity values to range [min, max].
    ///
    /// Args:
    ///     min: Minimum value
    ///     max: Maximum value
    ///
    /// Returns:
    ///     New NiftiImage (supports method chaining)
    fn clamp(&self, min: f64, max: f64) -> Self {
        Self {
            inner: transforms::clamp(&self.inner, min, max),
        }
    }

    /// Crop or pad to target shape.
    ///
    /// Args:
    ///     target_shape: Target shape as [depth, height, width]
    ///
    /// Returns:
    ///     New NiftiImage (supports method chaining)
    fn crop_or_pad(&self, target_shape: Vec<usize>) -> PyResult<Self> {
        Ok(Self {
            inner: transforms::crop_or_pad(&self.inner, &target_shape)
                .map_err(|e| PyValueError::new_err(e.to_string()))?,
        })
    }

    /// Flip along specified axes.
    ///
    /// Args:
    ///     axes: List of axes to flip (0=depth, 1=height, 2=width)
    ///
    /// Returns:
    ///     New NiftiImage (supports method chaining)
    fn flip(&self, axes: Vec<usize>) -> PyResult<Self> {
        Ok(Self {
            inner: transforms::flip(&self.inner, &axes)
                .map_err(|e| PyValueError::new_err(e.to_string()))?,
        })
    }

    /// Check if the image data is already materialized in memory.
    ///
    /// When False, the data is mmap'd from disk and will be materialized
    /// on each transform call.
    fn is_materialized(&self) -> bool {
        self.inner.is_materialized()
    }

    /// Convert mmap'd data to owned memory.
    ///
    /// Call this once before running multiple transforms to avoid
    /// re-materializing the data on each transform call.
    ///
    /// Returns:
    ///     New NiftiImage with data in memory (supports method chaining)
    ///
    /// Example:
    ///     >>> img = medrs.load("brain.nii.gz").materialize()
    ///     >>> # Now transforms are fast as data is in memory
    ///     >>> processed = img.z_normalize().rescale(0, 1).flip([0])
    fn materialize(&self) -> Self {
        Self {
            inner: self.inner.materialize(),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "NiftiImage(shape={:?}, dtype={}, spacing={:?}, orientation={})",
            self.shape(),
            self.dtype(),
            self.spacing(),
            self.orientation()
        )
    }
}

/// Z-score normalize an image (zero mean, unit variance).
#[pyfunction]
fn z_normalization(image: &PyNiftiImage) -> PyNiftiImage {
    PyNiftiImage {
        inner: transforms::z_normalization(&image.inner),
    }
}

/// Rescale intensity to the provided range.
#[pyfunction]
#[pyo3(signature = (image, output_range=(0.0, 1.0)))]
fn rescale_intensity(image: &PyNiftiImage, output_range: (f64, f64)) -> PyNiftiImage {
    let (out_min, out_max) = output_range;
    PyNiftiImage {
        inner: transforms::rescale_intensity(&image.inner, out_min, out_max),
    }
}

/// Clamp intensity values into a fixed range.
#[pyfunction]
fn clamp(image: &PyNiftiImage, min_value: f64, max_value: f64) -> PyNiftiImage {
    PyNiftiImage {
        inner: transforms::clamp(&image.inner, min_value, max_value),
    }
}

/// Crop or pad an image to the target shape.
#[pyfunction]
fn crop_or_pad(image: &PyNiftiImage, target_shape: Vec<usize>) -> PyResult<PyNiftiImage> {
    if target_shape.len() != 3 {
        return Err(PyValueError::new_err(
            "target_shape must be a 3-element sequence",
        ));
    }

    Ok(PyNiftiImage {
        inner: transforms::crop_or_pad(&image.inner, &target_shape)
            .map_err(|e| PyValueError::new_err(e.to_string()))?,
    })
}

/// Resample to target voxel spacing.
#[pyfunction]
#[pyo3(signature = (image, target_spacing, method=None))]
fn resample(
    image: &PyNiftiImage,
    target_spacing: (f32, f32, f32),
    method: Option<&str>,
) -> PyResult<PyNiftiImage> {
    let interp = match method.unwrap_or("trilinear") {
        "trilinear" | "linear" => Interpolation::Trilinear,
        "nearest" => Interpolation::Nearest,
        other => {
            return Err(PyValueError::new_err(format!(
                "method must be 'trilinear' or 'nearest', got {}",
                other
            )))
        }
    };

    let spacing = [target_spacing.0, target_spacing.1, target_spacing.2];

    Ok(PyNiftiImage {
        inner: transforms::resample_to_spacing(&image.inner, spacing, interp),
    })
}

/// Reorient an image to the target orientation code (e.g., RAS or LPS).
#[pyfunction]
fn reorient(image: &PyNiftiImage, orientation: &str) -> PyResult<PyNiftiImage> {
    let target = Orientation::from_str(orientation).ok_or_else(|| {
        PyValueError::new_err(format!(
            "Invalid orientation code: {}. Expected three-letter code like RAS or LPS",
            orientation
        ))
    })?;

    Ok(PyNiftiImage {
        inner: transforms::reorient(&image.inner, target),
    })
}

/// Load a NIfTI image from file.
///
/// Supports both .nii and .nii.gz formats.
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
fn load(path: &str) -> PyResult<PyNiftiImage> {
    nifti::load(path)
        .map(|inner| PyNiftiImage { inner })
        .map_err(|e| PyValueError::new_err(format!("Failed to load {}: {}", path, e)))
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
fn load_to_torch(
    py: Python<'_>,
    path: &str,
    dtype: Option<PyObject>,
    device: &str,
) -> PyResult<PyObject> {
    // Load image using medrs
    let img = nifti::load(path)
        .map_err(|e| PyValueError::new_err(format!("Failed to load {}: {}", path, e)))?;

    // Convert to PyTorch tensor directly
    let py_img = PyNiftiImage { inner: img };
    py_img.to_torch_with_dtype_and_device(py, dtype, Some(device))
}

/// Load only a cropped region from a NIfTI file without loading the entire volume.
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
fn load_cropped(
    path: &str,
    crop_offset: [usize; 3],
    crop_shape: [usize; 3],
) -> PyResult<PyNiftiImage> {
    nifti::load_cropped(path, crop_offset, crop_shape)
        .map(|inner| PyNiftiImage { inner })
        .map_err(|e| PyValueError::new_err(format!("Failed to load cropped {}: {}", path, e)))
}

/// Load a cropped region with optional reorientation and resampling.
///
/// Advanced version that computes the minimal region needed from the raw file
/// to achieve the desired output after transforms.
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
fn load_resampled(
    path: &str,
    output_shape: [usize; 3],
    target_spacing: Option<[f32; 3]>,
    target_orientation: Option<String>,
    output_offset: Option<[usize; 3]>,
) -> PyResult<PyNiftiImage> {
    use crate::transforms::Orientation;

    let orientation = match target_orientation {
        Some(s) => Some(
            Orientation::from_str(&s)
                .ok_or_else(|| PyValueError::new_err(format!("Invalid orientation: {}", s)))?,
        ),
        None => None,
    };

    let config = nifti::LoadCroppedConfig {
        output_shape,
        target_spacing,
        target_orientation: orientation,
        output_offset,
    };

    nifti::load_cropped_config(path, config)
        .map(|inner| PyNiftiImage { inner })
        .map_err(|e| PyValueError::new_err(format!("Failed to load cropped {}: {}", path, e)))
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
fn load_cropped_to_torch(
    py: Python<'_>,
    path: &str,
    output_shape: [usize; 3],
    target_spacing: Option<[f32; 3]>,
    target_orientation: Option<String>,
    output_offset: Option<[usize; 3]>,
    dtype: Option<PyObject>,
    device: &str,
) -> PyResult<PyObject> {
    use crate::transforms::Orientation;

    // Load image using our I/O-optimized function
    let orientation = match target_orientation {
        Some(s) => Some(
            Orientation::from_str(&s)
                .ok_or_else(|| PyValueError::new_err(format!("Invalid orientation: {}", s)))?,
        ),
        None => None,
    };

    let config = nifti::LoadCroppedConfig {
        output_shape,
        target_spacing,
        target_orientation: orientation,
        output_offset,
    };

    let img = nifti::load_cropped_config(path, config)
        .map_err(|e| PyValueError::new_err(format!("Failed to load cropped {}: {}", path, e)))?;

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
fn load_cropped_to_jax(
    py: Python<'_>,
    path: &str,
    output_shape: [usize; 3],
    target_spacing: Option<[f32; 3]>,
    target_orientation: Option<String>,
    output_offset: Option<[usize; 3]>,
    dtype: Option<PyObject>,
    device: &str,
) -> PyResult<PyObject> {
    use crate::transforms::Orientation;

    // Load image using our I/O-optimized function
    let orientation = match target_orientation {
        Some(s) => Some(
            Orientation::from_str(&s)
                .ok_or_else(|| PyValueError::new_err(format!("Invalid orientation: {}", s)))?,
        ),
        None => None,
    };

    let config = nifti::LoadCroppedConfig {
        output_shape,
        target_spacing,
        target_orientation: orientation,
        output_offset,
    };

    let img = nifti::load_cropped_config(path, config)
        .map_err(|e| PyValueError::new_err(format!("Failed to load cropped {}: {}", path, e)))?;

    // Convert to JAX array directly using our optimized I/O + array conversion
    let py_img = PyNiftiImage { inner: img };
    py_img.to_jax_with_dtype_and_device(py, dtype, Some(device))
}

/// High-performance training data loader with prefetching and caching.
///
/// This is the most efficient way to load training patches from multiple volumes.
/// Maintains an LRU cache and prefetches upcoming data to maximize throughput.
#[pyclass]
pub struct PyTrainingDataLoader {
    loader: nifti::TrainingDataLoader,
}

#[pymethods]
impl PyTrainingDataLoader {
    /// Create a new training data loader.
    ///
    /// Args:
    ///     volumes: List of NIfTI file paths
    ///     patch_size: Patch size to extract [d, h, w]
    ///     patches_per_volume: Number of patches per volume
    ///     patch_overlap: Overlap between patches [d, h, w] in voxels
    ///     randomize: Whether to randomize patch positions
    ///     cache_size: Maximum number of patches to cache
    ///
    /// Example:
    ///     >>> loader = medrs.TrainingDataLoader(
    ///     ...     volumes=["vol1.nii", "vol2.nii"],
    ///     ...     patch_size=[64, 64, 64],
    ///     ...     patches_per_volume=4,
    ///     ...     patch_overlap=[0, 0, 0],
    ///     ...     randomize=True,
    ///     ...     cache_size=1000
    ///     ... )
    ///     >>> patch = loader.next_patch()
    #[new]
    #[pyo3(signature = (volumes, patch_size, patches_per_volume, patch_overlap, randomize, cache_size=None))]
    fn new(
        volumes: Vec<String>,
        patch_size: [usize; 3],
        patches_per_volume: usize,
        patch_overlap: [usize; 3],
        randomize: bool,
        cache_size: Option<usize>,
    ) -> PyResult<Self> {
        for i in 0..3 {
            if patch_overlap[i] >= patch_size[i] {
                return Err(PyValueError::new_err(
                    "patch_overlap must be smaller than patch_size in all dimensions",
                ));
            }
        }

        let config = nifti::CropLoaderConfig {
            patch_size,
            patches_per_volume,
            patch_overlap,
            randomize,
        };

        let cache_size = cache_size.unwrap_or(1000);
        let loader = nifti::TrainingDataLoader::new(volumes, config, cache_size)
            .map_err(|e| PyValueError::new_err(format!("Failed to create loader: {}", e)))?;

        Ok(Self { loader })
    }

    /// Get next training patch.
    ///
    /// Returns the next training patch with automatic prefetching.
    /// Raises StopIteration when all patches are processed.
    fn next_patch(&mut self) -> PyResult<PyNiftiImage> {
        self.loader
            .next_patch()
            .map(|inner| PyNiftiImage { inner })
            .map_err(|e| PyValueError::new_err(format!("Failed to get next patch: {}", e)))
    }

    fn __len__(&self) -> usize {
        self.loader.volumes_len() * self.loader.patches_per_volume()
    }

    fn __iter__(mut slf: PyRefMut<'_, Self>) -> PyResult<PyRefMut<'_, Self>> {
        slf.loader
            .reset()
            .map_err(|e| PyValueError::new_err(format!("Failed to reset loader: {}", e)))?;
        Ok(slf)
    }

    fn __next__(mut slf: PyRefMut<'_, Self>) -> Option<PyNiftiImage> {
        slf.next_patch().ok()
    }

    /// Get performance statistics.
    fn stats(&self) -> String {
        format!("{}", self.loader.stats())
    }

    /// Reset the loader to start from the beginning.
    fn reset(&mut self) -> PyResult<()> {
        self.loader
            .reset()
            .map_err(|e| PyValueError::new_err(format!("Failed to reset loader: {}", e)))
    }
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

    /// Apply the pipeline to an image.
    ///
    /// Args:
    ///     image: Input NiftiImage
    ///
    /// Returns:
    ///     Transformed NiftiImage
    fn apply(&self, image: &PyNiftiImage) -> PyNiftiImage {
        PyNiftiImage {
            inner: self.inner.apply(&image.inner),
        }
    }

    fn __repr__(&self) -> String {
        "TransformPipeline(...)".to_string()
    }
}

// Random augmentation functions

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
fn random_flip(
    image: &PyNiftiImage,
    axes: Vec<usize>,
    prob: Option<f32>,
    seed: Option<u64>,
) -> PyResult<PyNiftiImage> {
    Ok(PyNiftiImage {
        inner: transforms::random_flip(&image.inner, &axes, prob, seed)
            .map_err(|e| PyValueError::new_err(e.to_string()))?,
    })
}

/// Add random Gaussian noise to the image.
///
/// Args:
///     image: Input image
///     std: Standard deviation of the noise (default: 0.1)
///     seed: Optional random seed for reproducibility
///
/// Returns:
///     Augmented image
#[pyfunction]
#[pyo3(signature = (image, std=None, seed=None))]
fn random_gaussian_noise(
    image: &PyNiftiImage,
    std: Option<f32>,
    seed: Option<u64>,
) -> PyNiftiImage {
    PyNiftiImage {
        inner: transforms::random_gaussian_noise(&image.inner, std, seed),
    }
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
fn random_intensity_scale(
    image: &PyNiftiImage,
    scale_range: Option<f32>,
    seed: Option<u64>,
) -> PyNiftiImage {
    PyNiftiImage {
        inner: transforms::random_intensity_scale(&image.inner, scale_range, seed),
    }
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
fn random_intensity_shift(
    image: &PyNiftiImage,
    shift_range: Option<f32>,
    seed: Option<u64>,
) -> PyNiftiImage {
    PyNiftiImage {
        inner: transforms::random_intensity_shift(&image.inner, shift_range, seed),
    }
}

/// Randomly rotate the image by 90-degree increments.
///
/// Performs random rotation in the specified plane by 0, 90, 180, or 270 degrees.
///
/// Args:
///     image: Input image
///     axes: Tuple of two axes defining the rotation plane (e.g., (0, 1) for depth-height plane)
///     seed: Optional random seed for reproducibility
///
/// Returns:
///     Augmented image
#[pyfunction]
#[pyo3(signature = (image, axes, seed=None))]
fn random_rotate_90(
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
fn random_gamma(
    image: &PyNiftiImage,
    gamma_range: Option<(f32, f32)>,
    seed: Option<u64>,
) -> PyNiftiImage {
    PyNiftiImage {
        inner: transforms::random_gamma(&image.inner, gamma_range, seed),
    }
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
fn random_augment(image: &PyNiftiImage, seed: Option<u64>) -> PyResult<PyNiftiImage> {
    Ok(PyNiftiImage {
        inner: transforms::random_augment(&image.inner, seed)
            .map_err(|e| PyValueError::new_err(e.to_string()))?,
    })
}

/// High-performance medical image I/O and processing.
///
/// Fast NIfTI reading/writing and transforms optimized for deep learning.
///
/// Example:
///     >>> import medrs
///     >>> img = medrs.load("brain.nii.gz")
///     >>> data = img.to_numpy()
///     >>> processed = img.resample([1.0, 1.0, 1.0]).z_normalize().clamp(0, 1)
///     >>> processed.save("output.nii.gz")
#[pymodule]
fn _medrs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Classes
    m.add_class::<PyNiftiImage>()?;
    m.add_class::<PyTrainingDataLoader>()?;
    m.add_class::<PyTransformPipeline>()?;

    // Basic transforms
    m.add_function(wrap_pyfunction!(z_normalization, m)?)?;
    m.add_function(wrap_pyfunction!(rescale_intensity, m)?)?;
    m.add_function(wrap_pyfunction!(clamp, m)?)?;
    m.add_function(wrap_pyfunction!(crop_or_pad, m)?)?;
    m.add_function(wrap_pyfunction!(resample, m)?)?;
    m.add_function(wrap_pyfunction!(reorient, m)?)?;

    // I/O functions
    m.add_function(wrap_pyfunction!(load, m)?)?;
    m.add_function(wrap_pyfunction!(load_to_torch, m)?)?;
    m.add_function(wrap_pyfunction!(load_cropped, m)?)?;
    m.add_function(wrap_pyfunction!(load_resampled, m)?)?;
    m.add_function(wrap_pyfunction!(load_cropped_to_torch, m)?)?;
    m.add_function(wrap_pyfunction!(load_cropped_to_jax, m)?)?;

    // Crop-first transform functions
    m.add_function(wrap_pyfunction!(load_label_aware_cropped, m)?)?;
    m.add_function(wrap_pyfunction!(compute_crop_regions, m)?)?;
    m.add_function(wrap_pyfunction!(compute_random_spatial_crops, m)?)?;
    m.add_function(wrap_pyfunction!(compute_center_crop, m)?)?;

    // Random augmentation functions
    m.add_function(wrap_pyfunction!(random_flip, m)?)?;
    m.add_function(wrap_pyfunction!(random_gaussian_noise, m)?)?;
    m.add_function(wrap_pyfunction!(random_intensity_scale, m)?)?;
    m.add_function(wrap_pyfunction!(random_intensity_shift, m)?)?;
    m.add_function(wrap_pyfunction!(random_rotate_90, m)?)?;
    m.add_function(wrap_pyfunction!(random_gamma, m)?)?;
    m.add_function(wrap_pyfunction!(random_augment, m)?)?;

    Ok(())
}

fn arraydata_to_numpy<'py>(
    py: Python<'py>,
    data: &ArrayData,
    shape: &[usize],
) -> PyResult<PyObject> {
    let dyn_shape = IxDyn(shape);
    Ok(match data {
        ArrayData::U8(a) => a
            .to_owned()
            .into_pyarray(py)
            .reshape(dyn_shape)?
            .unbind()
            .into_any(),
        ArrayData::I8(a) => a
            .to_owned()
            .into_pyarray(py)
            .reshape(dyn_shape)?
            .unbind()
            .into_any(),
        ArrayData::I16(a) => a
            .to_owned()
            .into_pyarray(py)
            .reshape(dyn_shape)?
            .unbind()
            .into_any(),
        ArrayData::U16(a) => a
            .to_owned()
            .into_pyarray(py)
            .reshape(dyn_shape)?
            .unbind()
            .into_any(),
        ArrayData::I32(a) => a
            .to_owned()
            .into_pyarray(py)
            .reshape(dyn_shape)?
            .unbind()
            .into_any(),
        ArrayData::U32(a) => a
            .to_owned()
            .into_pyarray(py)
            .reshape(dyn_shape)?
            .unbind()
            .into_any(),
        ArrayData::I64(a) => a
            .to_owned()
            .into_pyarray(py)
            .reshape(dyn_shape)?
            .unbind()
            .into_any(),
        ArrayData::U64(a) => a
            .to_owned()
            .into_pyarray(py)
            .reshape(dyn_shape)?
            .unbind()
            .into_any(),
        ArrayData::F16(a) => a
            .mapv(|v| v.to_f32())
            .into_pyarray(py)
            .reshape(dyn_shape)?
            .unbind()
            .into_any(),
        ArrayData::BF16(a) => a
            .mapv(|v| v.to_f32())
            .into_pyarray(py)
            .reshape(dyn_shape)?
            .unbind()
            .into_any(),
        ArrayData::F32(a) => a
            .to_owned()
            .into_pyarray(py)
            .reshape(dyn_shape)?
            .unbind()
            .into_any(),
        ArrayData::F64(a) => a
            .to_owned()
            .into_pyarray(py)
            .reshape(dyn_shape)?
            .unbind()
            .into_any(),
    })
}

fn to_numpy_view<'py>(
    py: Python<'py>,
    image: &RustNiftiImage,
) -> Option<Bound<'py, PyArrayDyn<f32>>> {
    if let Some(view) = image.as_view_f32() {
        // numpy will copy; this still avoids extra materialization in Rust
        let arr = PyArrayDyn::from_array(py, &view);
        return Some(arr);
    }
    None
}

fn to_numpy_array<'py>(py: Python<'py>, image: &RustNiftiImage) -> Bound<'py, PyArrayDyn<f32>> {
    let data = image.to_f32();
    let shape: Vec<usize> = data.shape().to_vec();

    // Data is in F-order (column-major) layout in memory
    // Use as_slice_memory_order to get bytes in physical order
    let slice = data.as_slice_memory_order().expect("Array should be contiguous");
    let np = py.import("numpy").expect("numpy import failed");

    // Create 1D array from F-order bytes
    let flat = slice.to_vec().into_pyarray(py);

    // Reshape with order='F' to interpret the flat data as F-order
    // This creates an F-contiguous array matching NIfTI convention
    np.call_method(
        "reshape",
        (flat, &shape),
        Some(&[("order", "F")].into_py_dict(py).expect("dict failed")),
    )
    .expect("reshape failed")
    .extract::<Bound<'py, PyArrayDyn<f32>>>()
    .expect("extract failed")
}

fn to_numpy_view_native<'py>(py: Python<'py>, image: &RustNiftiImage) -> Option<PyObject> {
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
        DataType::Float16 => None, // numpy crate lacks f16 Element impl, fall back to f32
        DataType::BFloat16 => None, // numpy lacks bf16
        DataType::Float32 => image
            .as_view_t::<f32>()
            .map(|v| PyArrayDyn::from_array(py, &v).unbind().into_any()),
        DataType::Float64 => image
            .as_view_t::<f64>()
            .map(|v| PyArrayDyn::from_array(py, &v).unbind().into_any()),
    };
    arr_obj
}

fn torch_dtype(py: Python<'_>, dtype: DataType) -> Option<PyObject> {
    let torch = py.import("torch").ok()?;
    let dt = match dtype {
        DataType::UInt8 => "uint8",
        DataType::Int8 => "int8",
        DataType::Int16 => "int16",
        DataType::UInt16 => return None,
        DataType::Int32 => "int32",
        DataType::UInt32 => return None,
        DataType::Int64 => "int64",
        DataType::UInt64 => return None,
        DataType::Float16 => "float16",
        DataType::BFloat16 => "bfloat16",
        DataType::Float32 => "float32",
        DataType::Float64 => "float64",
    };
    torch.getattr(dt).ok().map(|o| o.unbind())
}

/// Load a NIfTI file with byte-exact cropping for label-aware training.
///
/// This function combines MONAI's `RandCropByPosNegLabeld` with medrs's
/// byte-exact loading for maximum performance. It computes optimal crop
/// regions containing both positive and negative labels, then loads only
/// the required bytes.
///
/// Args:
///     image_path: Path to the image file
///     label_path: Path to the label file
///     patch_size: Target patch size [x, y, z]
///     pos_neg_ratio: Ratio of positive to negative samples (default: 1.0)
///     min_pos_samples: Minimum positive samples per volume (default: 4)
///     seed: Random seed for reproducibility (optional)
///
/// Returns:
///     Tuple of (cropped_image, cropped_label) as PyNiftiImages
#[pyfunction]
#[pyo3(signature = (image_path, label_path, patch_size, pos_neg_ratio=None, min_pos_samples=None, seed=None))]
fn load_label_aware_cropped(
    _py: Python<'_>,
    image_path: &str,
    label_path: &str,
    patch_size: Vec<usize>,
    pos_neg_ratio: Option<f64>,
    min_pos_samples: Option<usize>,
    seed: Option<u64>,
) -> PyResult<(PyNiftiImage, PyNiftiImage)> {
    // Load full images first (this could be optimized further)
    let image = nifti::load(image_path).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Failed to load image: {}", e))
    })?;
    let label = nifti::load(label_path).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Failed to load label: {}", e))
    })?;

    // Configure cropping
    let config = transforms::RandCropByPosNegLabelConfig {
        patch_size: [patch_size[0], patch_size[1], patch_size[2]],
        pos_neg_ratio: pos_neg_ratio.unwrap_or(1.0) as f32,
        min_pos_samples: min_pos_samples.unwrap_or(4),
        seed,
        background_label: 0.0,
    };

    // Compute crop regions (this is fast, operates on labels only)
    let crop_regions = compute_label_aware_crop_regions(&config, &image, &label, 1);

    if crop_regions.is_empty() {
        return Err(PyValueError::new_err("No valid crop regions found"));
    }

    let region = &crop_regions[0];

    // Load cropped regions (this is the fast byte-exact loading)
    let cropped_image =
        nifti::load_cropped(image_path, region.start, region.size).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!(
                "Failed to load cropped image: {}",
                e
            ))
        })?;
    let cropped_label =
        nifti::load_cropped(label_path, region.start, region.size).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!(
                "Failed to load cropped label: {}",
                e
            ))
        })?;

    Ok((
        PyNiftiImage {
            inner: cropped_image,
        },
        PyNiftiImage {
            inner: cropped_label,
        },
    ))
}

/// Compute crop regions for smart loading without actually loading the data.
///
/// This function allows users to compute optimal crop regions first,
/// then use them with `load_cropped()` for maximum control.
/// This is useful for batch processing and advanced training pipelines.
///
/// Args:
///     image_path: Path to the image file
///     label_path: Path to the label file
///     patch_size: Target patch size [x, y, z]
///     num_samples: Number of crop regions to compute
///     pos_neg_ratio: Ratio of positive to negative samples (default: 1.0)
///     min_pos_samples: Minimum positive samples per volume (default: 4)
///     seed: Random seed for reproducibility (optional)
///
/// Returns:
///     List of crop regions as dictionaries with 'start', 'end', and 'size' keys
#[pyfunction]
#[pyo3(signature = (image_path, label_path, patch_size, num_samples, pos_neg_ratio=None, min_pos_samples=None, seed=None))]
fn compute_crop_regions(
    py: Python<'_>,
    image_path: &str,
    label_path: &str,
    patch_size: Vec<usize>,
    num_samples: usize,
    pos_neg_ratio: Option<f64>,
    min_pos_samples: Option<usize>,
    seed: Option<u64>,
) -> PyResult<Vec<PyObject>> {
    // Load images to get shape information
    let image = nifti::load(image_path).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Failed to load image: {}", e))
    })?;
    let label = nifti::load(label_path).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Failed to load label: {}", e))
    })?;

    // Configure cropping
    let config = transforms::RandCropByPosNegLabelConfig {
        patch_size: [patch_size[0], patch_size[1], patch_size[2]],
        pos_neg_ratio: pos_neg_ratio.unwrap_or(1.0) as f32,
        min_pos_samples: min_pos_samples.unwrap_or(4),
        seed,
        background_label: 0.0,
    };

    // Compute crop regions
    let crop_regions = compute_label_aware_crop_regions(&config, &image, &label, num_samples);

    // Convert to Python dictionaries
    let mut regions_py = Vec::new();
    for region in crop_regions {
        let region_dict = pyo3::types::PyDict::new(py);
        region_dict.set_item("start", region.start.to_vec())?;
        region_dict.set_item("end", region.end.to_vec())?;
        region_dict.set_item("size", region.size.to_vec())?;
        regions_py.push(region_dict.into());
    }

    Ok(regions_py)
}

/// Compute random spatial crop regions.
///
/// This function implements MONAI's `RandSpatialCropd` functionality
/// optimized for medrs's crop-first approach.
///
/// Args:
///     image_path: Path to the image file
///     patch_size: Target patch size [x, y, z]
///     num_samples: Number of crop regions to compute
///     seed: Random seed for reproducibility (optional)
///     allow_smaller: Whether to allow smaller crops at boundaries (default: false)
///
/// Returns:
///     List of crop regions as dictionaries
#[pyfunction]
#[pyo3(signature = (image_path, patch_size, num_samples, seed=None, allow_smaller=None))]
fn compute_random_spatial_crops(
    py: Python<'_>,
    image_path: &str,
    patch_size: Vec<usize>,
    num_samples: usize,
    seed: Option<u64>,
    allow_smaller: Option<bool>,
) -> PyResult<Vec<PyObject>> {
    // Load image to get shape information
    let image = nifti::load(image_path).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Failed to load image: {}", e))
    })?;

    // Configure cropping
    let config = transforms::SpatialCropConfig {
        patch_size: [patch_size[0], patch_size[1], patch_size[2]],
        seed,
        allow_smaller: allow_smaller.unwrap_or(false),
    };

    // Compute crop regions
    let crop_regions = compute_random_spatial_crop_regions(&config, &image, num_samples);

    // Convert to Python dictionaries
    let mut regions_py = Vec::new();
    for region in crop_regions {
        let region_dict = pyo3::types::PyDict::new(py);
        region_dict.set_item("start", region.start.to_vec())?;
        region_dict.set_item("end", region.end.to_vec())?;
        region_dict.set_item("size", region.size.to_vec())?;
        regions_py.push(region_dict.into());
    }

    Ok(regions_py)
}

/// Compute center crop region.
///
/// This function implements MONAI's `CenterSpatialCropd` functionality
/// optimized for medrs's crop-first approach.
///
/// Args:
///     image_path: Path to the image file
///     patch_size: Target patch size [x, y, z]
///
/// Returns:
///     Crop region as dictionary
#[pyfunction]
fn compute_center_crop(
    py: Python<'_>,
    image_path: &str,
    patch_size: Vec<usize>,
) -> PyResult<PyObject> {
    // Load image to get shape information
    let image = nifti::load(image_path).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Failed to load image: {}", e))
    })?;

    // Compute center crop
    let region = compute_center_crop_regions([patch_size[0], patch_size[1], patch_size[2]], &image);

    // Convert to Python dictionary
    let region_dict = pyo3::types::PyDict::new(py);
    region_dict.set_item("start", region.start.to_vec())?;
    region_dict.set_item("end", region.end.to_vec())?;
    region_dict.set_item("size", region.size.to_vec())?;

    Ok(region_dict.into())
}
