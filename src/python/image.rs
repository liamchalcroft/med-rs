//! PyNiftiImage class for Python bindings.
//!
//! This module provides the main NiftiImage wrapper class that
//! exposes medrs image functionality to Python with method chaining.

use numpy::{PyArrayDyn, PyReadonlyArrayDyn};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::conversion::{
    self, to_numpy_array, to_numpy_view, to_numpy_view_native, to_numpy_zero_copy, torch_dtype,
};
use super::validation::to_py_err;

use crate::nifti::{self, NiftiImage as RustNiftiImage};
use crate::transforms::{self, Interpolation, Orientation};

fn scaling_required(image: &RustNiftiImage) -> bool {
    let slope = image.header().scl_slope;
    let inter = image.header().scl_inter;
    (slope != 0.0 && slope != 1.0) || inter != 0.0
}

fn numpy_for_torch(
    py: Python<'_>,
    image: &RustNiftiImage,
    allow_native: bool,
) -> PyResult<PyObject> {
    if !scaling_required(image) {
        if image.can_zero_copy_f32() {
            let arr = to_numpy_zero_copy(py, image)?;
            return Ok(arr.into_pyobject(py)?.into_any().unbind());
        }
        if allow_native {
            if let Some(np_view) = to_numpy_view_native(py, image) {
                return Ok(np_view);
            }
        }
    }

    let arr = to_numpy_array(py, image)?;
    Ok(arr.into_pyobject(py)?.into_any().unbind())
}

fn numpy_for_jax(py: Python<'_>, image: &RustNiftiImage) -> PyResult<PyObject> {
    if !scaling_required(image) {
        if image.can_zero_copy_f32() {
            let arr = to_numpy_zero_copy(py, image)?;
            return Ok(arr.into_pyobject(py)?.into_any().unbind());
        }
        if let Some(np_view) = to_numpy_view_native(py, image) {
            return Ok(np_view);
        }
    }

    let arr = to_numpy_array(py, image)?;
    Ok(arr.into_pyobject(py)?.into_any().unbind())
}

/// A NIfTI image with header metadata and voxel data.
///
/// Supports method chaining for transform operations.
///
/// Example:
///     >>> img = medrs.load("brain.nii.gz")
///     >>> processed = img.resample([1.0, 1.0, 1.0]).z_normalize().clamp(0, 1)
///     >>> processed.save("output.nii.gz")
#[pyclass(name = "NiftiImage")]
pub struct PyNiftiImage {
    pub(crate) inner: RustNiftiImage,
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
    fn new(data: PyReadonlyArrayDyn<'_, f32>, affine: Option<[[f32; 4]; 4]>) -> PyResult<Self> {
        let inner = super::validation::create_nifti_from_numpy_array(data.as_array(), affine)?;
        Ok(Self { inner })
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
        self.inner.spacing().clone()
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
    /// Always returns a copy of the data. For zero-copy access, use to_numpy(copy=False).
    #[getter]
    fn data<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
        self.to_numpy(py, true)
    }

    /// Get image data as numpy array (float32).
    ///
    /// Similar to nibabel's get_fdata(). Applies scaling factors if present.
    /// Supports arbitrary ndim.
    ///
    /// Args:
    ///     copy: If True (default), always returns a copy of the data.
    ///           If False, attempts zero-copy access and raises ValueError
    ///           if zero-copy is not possible.
    ///
    /// Zero-copy is possible when:
    /// - Data is from an uncompressed .nii file (mmap-backed)
    /// - Data type is float32
    /// - Native endianness (little-endian on x86/ARM)
    /// - No scaling required (slope=1 or 0, intercept=0)
    /// - Memory is properly aligned
    ///
    /// Returns:
    ///     numpy.ndarray: Image data as float32 array
    ///
    /// Raises:
    ///     ValueError: If copy=False and zero-copy is not possible
    #[pyo3(signature = (copy=true))]
    fn to_numpy<'py>(&self, py: Python<'py>, copy: bool) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
        if copy {
            // Standard copy path
            if let Some(arr) = to_numpy_view(py, &self.inner) {
                return Ok(arr);
            }
            to_numpy_array(py, &self.inner)
        } else {
            // Zero-copy path - must succeed or error
            to_numpy_zero_copy(py, &self.inner)
        }
    }

    /// Check if zero-copy numpy access is possible.
    ///
    /// Returns True if to_numpy(copy=False) would succeed.
    fn can_zero_copy(&self) -> bool {
        self.inner.can_zero_copy_f32()
    }

    /// Get image data as a torch tensor (shares memory when possible).
    ///
    /// Applies NIfTI scaling when present; zero-copy only applies when
    /// scaling is not needed and the data is mmap-backed float32.
    fn to_torch(&self, py: Python<'_>) -> PyResult<PyObject> {
        let torch = py.import("torch")?;
        let scaling = scaling_required(&self.inner);
        let torch_dt = torch_dtype(py, self.inner.dtype());
        let allow_native = torch_dt.is_some();
        let dtype = if scaling { None } else { torch_dt };
        let np_obj = numpy_for_torch(py, &self.inner, allow_native)?;

        let tensor = torch.getattr("from_numpy")?.call1((np_obj,))?;

        if let Some(dt) = dtype {
            let tensor = tensor.call_method1("to", (dt,))?;
            Ok(tensor.unbind())
        } else {
            Ok(tensor.unbind())
        }
    }

    /// Get image data as a JAX array (shares memory via numpy when possible).
    fn to_jax(&self, py: Python<'_>) -> PyResult<PyObject> {
        let jnp = py.import("jax.numpy")?;
        let np_obj = to_numpy_array(py, &self.inner)?;
        let arr = jnp.getattr("array")?.call1((np_obj,))?;
        Ok(arr.unbind())
    }

    /// Convert to PyTorch tensor with custom dtype and device.
    ///
    /// This is the most efficient way to load medical imaging data directly
    /// into PyTorch with the target precision and device placement.
    #[pyo3(signature = (dtype=None, device=None))]
    pub(crate) fn to_torch_with_dtype_and_device(
        &self,
        py: Python<'_>,
        dtype: Option<PyObject>,
        device: Option<&str>,
    ) -> PyResult<PyObject> {
        let device_str = device.unwrap_or("cpu");
        let torch = py.import("torch")?;
        let allow_native = torch_dtype(py, self.inner.dtype()).is_some();
        let np_obj = numpy_for_torch(py, &self.inner, allow_native)?;

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
    #[allow(clippy::useless_let_if_seq)]
    #[pyo3(signature = (dtype=None, device=None))]
    pub(crate) fn to_jax_with_dtype_and_device(
        &self,
        py: Python<'_>,
        dtype: Option<PyObject>,
        device: Option<&str>,
    ) -> PyResult<PyObject> {
        let device_str = device.unwrap_or("cpu");
        let jax = py.import("jax")?;
        let _jnp = py.import("jax.numpy")?;

        // Get device object using correct JAX API
        let device_obj = if device_str == "cpu" {
            let cpu_devices = jax.getattr("devices")?.call1(("cpu",))?;
            cpu_devices.get_item(0)?
        } else if device_str.starts_with("cuda") || device_str.starts_with("gpu") {
            // JAX uses "gpu" not "cuda" for CUDA devices
            let gpu_devices = jax.getattr("devices")?.call1(("gpu",))?;
            let device_id: usize = device_str
                .strip_prefix("cuda:")
                .or_else(|| device_str.strip_prefix("gpu:"))
                .unwrap_or("0")
                .parse()
                .unwrap_or(0);
            gpu_devices.get_item(device_id)?
        } else {
            return Err(PyValueError::new_err(format!(
                "Unsupported device: {}. Use 'cpu', 'cuda', 'cuda:N', 'gpu', or 'gpu:N'",
                device_str
            )));
        };

        // Use optimized on-device array creation
        let jax = py.import("jax")?;
        let jnp = py.import("jax.numpy")?;

        let np_obj = numpy_for_jax(py, &self.inner)?;
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
    fn to_numpy_native(&self, py: Python<'_>) -> PyResult<PyObject> {
        conversion::arraydata_to_numpy(
            py,
            &self
                .inner
                .owned_data()
                .map_err(|e| to_py_err(e, "to_numpy_native"))?,
            self.inner.shape(),
        )
    }

    /// Save image to file.
    ///
    /// Format is determined by extension (.nii or .nii.gz).
    fn save(&self, path: &str) -> PyResult<()> {
        let validated_path = super::validation::validate_file_path(path, "save")?;
        let path_str = validated_path
            .to_str()
            .ok_or_else(|| PyValueError::new_err("path contains invalid UTF-8"))?;
        nifti::save(&self.inner, path_str)
            .map_err(|e| to_py_err(e, &format!("Failed to save {}", path)))
    }

    /// Convert image to a different data type.
    ///
    /// This is useful for reducing file size when saving. For example,
    /// converting from float32 to bfloat16 reduces storage by 50%.
    ///
    /// Args:
    ///     dtype: Target dtype as string. Supported values:
    ///         - "float32", "f32" - 32-bit float (default)
    ///         - "float64", "f64" - 64-bit float
    ///         - "float16", "f16" - IEEE 754 half precision
    ///         - "bfloat16", "bf16" - Brain floating point 16-bit
    ///         - "int8", "i8" - Signed 8-bit integer
    ///         - "uint8", "u8" - Unsigned 8-bit integer
    ///         - "int16", "i16" - Signed 16-bit integer
    ///         - "uint16", "u16" - Unsigned 16-bit integer
    ///         - "int32", "i32" - Signed 32-bit integer
    ///         - "uint32", "u32" - Unsigned 32-bit integer
    ///         - "int64", "i64" - Signed 64-bit integer
    ///         - "uint64", "u64" - Unsigned 64-bit integer
    ///
    /// Returns:
    ///     New MedicalImage with converted dtype
    ///
    /// Example:
    ///     >>> img = medrs.load("volume.nii.gz")
    ///     >>> img_bf16 = img.with_dtype("bfloat16")
    ///     >>> img_bf16.save("volume_bf16.nii.gz")  # 50% smaller file
    fn with_dtype(&self, dtype: &str) -> PyResult<Self> {
        let target_dtype = match dtype.to_lowercase().as_str() {
            "float32" | "f32" => nifti::DataType::Float32,
            "float64" | "f64" => nifti::DataType::Float64,
            "float16" | "f16" => nifti::DataType::Float16,
            "bfloat16" | "bf16" => nifti::DataType::BFloat16,
            "int8" | "i8" => nifti::DataType::Int8,
            "uint8" | "u8" => nifti::DataType::UInt8,
            "int16" | "i16" => nifti::DataType::Int16,
            "uint16" | "u16" => nifti::DataType::UInt16,
            "int32" | "i32" => nifti::DataType::Int32,
            "uint32" | "u32" => nifti::DataType::UInt32,
            "int64" | "i64" => nifti::DataType::Int64,
            "uint64" | "u64" => nifti::DataType::UInt64,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "Unsupported dtype '{}'. Use: float32, float64, float16, bfloat16, int8, uint8, int16, uint16, int32, uint32, int64, uint64",
                    dtype
                )))
            }
        };

        Ok(Self {
            inner: self
                .inner
                .with_dtype(target_dtype)
                .map_err(|e| to_py_err(e, "with_dtype"))?,
        })
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

        let resampled = transforms::resample_to_spacing(&self.inner, spacing, interp)
            .map_err(|e| PyValueError::new_err(format!("Resampling failed: {}", e)))?;
        Ok(Self { inner: resampled })
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

        let resampled = transforms::resample_to_shape(&self.inner, shape, interp)
            .map_err(|e| PyValueError::new_err(format!("Resampling failed: {}", e)))?;
        Ok(Self { inner: resampled })
    }

    /// Reorient to target orientation.
    ///
    /// Args:
    ///     orientation: Target orientation code (e.g., "RAS", "LPS")
    ///
    /// Returns:
    ///     New NiftiImage (supports method chaining)
    fn reorient(&self, orientation: &str) -> PyResult<Self> {
        let target: Orientation = orientation
            .parse()
            .map_err(|e| PyValueError::new_err(format!("{}", e)))?;

        Ok(Self {
            inner: transforms::reorient(&self.inner, target)
                .map_err(|e| PyValueError::new_err(format!("Reorientation failed: {}", e)))?,
        })
    }

    /// Z-score normalization (zero mean, unit variance).
    ///
    /// Returns:
    ///     New NiftiImage (supports method chaining)
    fn z_normalize(&self) -> PyResult<Self> {
        Ok(Self {
            inner: transforms::z_normalization(&self.inner)
                .map_err(|e| to_py_err(e, "z_normalize"))?,
        })
    }

    /// Rescale intensity to range [min, max].
    ///
    /// Args:
    ///     out_min: Minimum output value
    ///     out_max: Maximum output value
    ///
    /// Returns:
    ///     New NiftiImage (supports method chaining)
    fn rescale(&self, out_min: f64, out_max: f64) -> PyResult<Self> {
        Ok(Self {
            inner: transforms::rescale_intensity(&self.inner, out_min, out_max)
                .map_err(|e| to_py_err(e, "rescale"))?,
        })
    }

    /// Clamp intensity values to range [min, max].
    ///
    /// Args:
    ///     min: Minimum value
    ///     max: Maximum value
    ///
    /// Returns:
    ///     New NiftiImage (supports method chaining)
    fn clamp(&self, min: f64, max: f64) -> PyResult<Self> {
        Ok(Self {
            inner: transforms::clamp(&self.inner, min, max).map_err(|e| to_py_err(e, "clamp"))?,
        })
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
    fn materialize(&self) -> PyResult<Self> {
        Ok(Self {
            inner: self
                .inner
                .materialize()
                .map_err(|e| to_py_err(e, "materialize"))?,
        })
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
