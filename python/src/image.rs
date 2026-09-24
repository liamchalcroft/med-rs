//! The `NiftiImage` class.

use crate::{parse_dtype, OrRaise};
use half::{bf16, f16};
use medrs::nifti::{
    Affine, NiftiElement, NiftiExtension, NiftiHeader, NiftiVersion, SpatialUnits, TemporalUnits,
};
use medrs::transforms::{self as t, Interpolation, Orientation};
use medrs::{DataType, NiftiImage};
use ndarray::{ArrayD, ArrayViewD, IxDyn, ShapeBuilder};
use numpy::{
    Element, IntoPyArray, PyArray2, PyArrayDescrMethods, PyArrayDyn, PyArrayMethods,
    PyReadonlyArray2, PyReadonlyArrayDyn, PyUntypedArray, PyUntypedArrayMethods,
};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList, PyTuple};
use std::path::PathBuf;

const IDENTITY: Affine = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
];

/// A medical image: voxel data plus its NIfTI header.
///
/// Images are immutable: every method returns a new image.
#[pyclass(name = "NiftiImage", module = "medrs", frozen)]
pub(crate) struct PyNiftiImage {
    pub(crate) inner: NiftiImage,
}

impl From<NiftiImage> for PyNiftiImage {
    fn from(inner: NiftiImage) -> Self {
        Self { inner }
    }
}

pub(crate) fn parse_interpolation(name: &str) -> PyResult<Interpolation> {
    name.parse()
        .map_err(|e: medrs::Error| PyValueError::new_err(e.to_string()))
}

pub(crate) fn parse_orientation(code: &str) -> PyResult<Orientation> {
    code.parse()
        .map_err(|e: medrs::Error| PyValueError::new_err(e.to_string()))
}

/// Read a 4x4 affine from anything `numpy.asarray` accepts.
pub(crate) fn affine_from_py(obj: &Bound<'_, PyAny>) -> PyResult<Affine> {
    let py = obj.py();
    let kwargs = PyDict::new(py);
    kwargs.set_item("dtype", "float64")?;
    let array = py
        .import("numpy")?
        .call_method("asarray", (obj,), Some(&kwargs))?;
    let array: PyReadonlyArray2<'_, f64> = array
        .extract()
        .map_err(|_| PyValueError::new_err("affine must be a 4x4 matrix"))?;
    let a = array.as_array();
    if a.shape() != [4, 4] {
        return Err(PyValueError::new_err(format!(
            "affine must be 4x4, got {:?}",
            a.shape()
        )));
    }
    let affine: Affine = std::array::from_fn(|i| std::array::from_fn(|j| a[[i, j]]));
    if affine[3] != [0.0, 0.0, 0.0, 1.0] || affine.iter().flatten().any(|v| !v.is_finite()) {
        return Err(PyValueError::new_err(
            "affine must be finite with last row [0, 0, 0, 1]",
        ));
    }
    Ok(affine)
}

fn affine_to_py<'py>(py: Python<'py>, a: &Affine) -> Bound<'py, PyArray2<f64>> {
    let values: Vec<f64> = a.iter().flatten().copied().collect();
    let array = ndarray::Array2::from_shape_vec((4, 4), values).unwrap_or_default();
    array.into_pyarray(py)
}

/// A copy of `view` in Fortran order, whatever the input layout.
fn fortran_owned<T: Clone + Default>(view: &ArrayViewD<'_, T>) -> ArrayD<T> {
    if view.t().is_standard_layout() {
        return view.to_owned();
    }
    let mut out = ArrayD::from_elem(IxDyn(view.shape()).f(), T::default());
    out.assign(view);
    out
}

fn image_from_view<T: NiftiElement>(
    view: &ArrayViewD<'_, T>,
    affine: Affine,
) -> PyResult<NiftiImage> {
    NiftiImage::from_array(fortran_owned(view), affine).or_raise()
}

/// Build an image from any array-like, keeping its dtype.
fn image_from_array(data: &Bound<'_, PyAny>, affine: Affine) -> PyResult<NiftiImage> {
    let py = data.py();
    let np = py.import("numpy")?;
    let mut array = np.call_method1("asarray", (data,))?;
    let dtype = array.cast::<PyUntypedArray>()?.dtype();
    if dtype.str()? == "bfloat16" {
        let bits = array.call_method1("view", ("uint16",))?;
        let bits: PyReadonlyArrayDyn<'_, u16> = bits.extract()?;
        let values = bits.as_array().mapv(bf16::from_bits);
        return image_from_view(&values.view(), affine);
    }
    if dtype.kind() == b'b' {
        array = array.call_method1("astype", ("uint8",))?;
    }
    // Native byte order and alignment, so the data can be read directly.
    let array = np.call_method1("require", (array, py.None(), vec!["A"]))?;
    let array = if array
        .cast::<PyUntypedArray>()?
        .dtype()
        .is_native_byteorder()
        == Some(false)
    {
        array.call_method1(
            "astype",
            (array
                .getattr("dtype")?
                .call_method1("newbyteorder", ("=",))?,),
        )?
    } else {
        array
    };
    let dtype = array.cast::<PyUntypedArray>()?.dtype();
    macro_rules! try_types {
        ($($t:ty),*) => {$(
            if dtype.is_equiv_to(&numpy::dtype::<$t>(py)) {
                let a: PyReadonlyArrayDyn<'_, $t> = array.extract()?;
                return image_from_view(&a.as_array(), affine);
            }
        )*};
    }
    try_types!(u8, i8, u16, i16, u32, i32, u64, i64, f16, f32, f64);
    Err(PyTypeError::new_err(format!(
        "unsupported array dtype {dtype}; expected a numeric type"
    )))
}

/// The dtype returned when none is requested: the stored type if values are
/// unscaled, otherwise float32 (like `numpy.asarray(nibabel_image.dataobj)`).
fn default_dtype(image: &NiftiImage) -> DataType {
    if image.header().has_scaling() {
        DataType::Float32
    } else {
        image.dtype()
    }
}

/// Scaled values as a new NumPy array of `dtype` (no bfloat16).
fn owned_numpy<'py>(
    py: Python<'py>,
    image: &NiftiImage,
    dtype: DataType,
) -> PyResult<Bound<'py, PyAny>> {
    fn convert<'py, T: NiftiElement + Element>(
        py: Python<'py>,
        image: &NiftiImage,
    ) -> PyResult<Bound<'py, PyAny>> {
        let array = py.detach(|| image.to_scaled::<T>()).or_raise()?;
        Ok(array.into_pyarray(py).into_any())
    }
    match dtype {
        DataType::UInt8 => convert::<u8>(py, image),
        DataType::Int8 => convert::<i8>(py, image),
        DataType::UInt16 => convert::<u16>(py, image),
        DataType::Int16 => convert::<i16>(py, image),
        DataType::UInt32 => convert::<u32>(py, image),
        DataType::Int32 => convert::<i32>(py, image),
        DataType::UInt64 => convert::<u64>(py, image),
        DataType::Int64 => convert::<i64>(py, image),
        DataType::Float16 => convert::<f16>(py, image),
        DataType::Float32 => convert::<f32>(py, image),
        DataType::Float64 => convert::<f64>(py, image),
        _ => Err(PyTypeError::new_err(format!(
            "cannot create a NumPy array of {dtype}"
        ))),
    }
}

/// Scaled values as bfloat16 bit patterns in an int16 array.
fn bf16_bits<'py>(py: Python<'py>, image: &NiftiImage) -> PyResult<Bound<'py, PyAny>> {
    let bits = py
        .detach(|| {
            image.to_scaled::<bf16>().map(|a| {
                let shape = a.shape().to_vec();
                let values: Vec<i16> = a.t().iter().map(|v| v.to_bits() as i16).collect();
                ArrayD::from_shape_vec(IxDyn(&shape).f(), values)
            })
        })
        .or_raise()?
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(bits.into_pyarray(py).into_any())
}

/// A read-only NumPy view of the stored values, if they can be shared.
fn numpy_view<'py>(
    slf: &Bound<'py, PyNiftiImage>,
    dtype: DataType,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    fn view<'py, T: NiftiElement + Element>(
        slf: &Bound<'py, PyNiftiImage>,
    ) -> PyResult<Option<Bound<'py, PyAny>>> {
        let Some(v) = slf.get().inner.view::<T>() else {
            return Ok(None);
        };
        // SAFETY: the view points into the image's buffer, which the frozen
        // `NiftiImage` object (the array's base) keeps alive and never changes.
        #[allow(unsafe_code)]
        let array = unsafe { PyArrayDyn::<T>::borrow_from_array(&v, slf.clone().into_any()) };
        array
            .try_readwrite()
            .map_err(|e| PyValueError::new_err(e.to_string()))?
            .make_nonwriteable();
        Ok(Some(array.into_any()))
    }
    match dtype {
        DataType::UInt8 => view::<u8>(slf),
        DataType::Int8 => view::<i8>(slf),
        DataType::UInt16 => view::<u16>(slf),
        DataType::Int16 => view::<i16>(slf),
        DataType::UInt32 => view::<u32>(slf),
        DataType::Int32 => view::<i32>(slf),
        DataType::UInt64 => view::<u64>(slf),
        DataType::Int64 => view::<i64>(slf),
        DataType::Float16 => view::<f16>(slf),
        DataType::Float32 => view::<f32>(slf),
        DataType::Float64 => view::<f64>(slf),
        _ => Ok(None),
    }
}

fn to_numpy_impl<'py>(
    slf: &Bound<'py, PyNiftiImage>,
    dtype: Option<&Bound<'py, PyAny>>,
    copy: Option<bool>,
) -> PyResult<Bound<'py, PyAny>> {
    let py = slf.py();
    let image = &slf.get().inner;
    let target = match dtype {
        Some(d) if !d.is_none() => parse_dtype(d)?,
        _ => match default_dtype(image) {
            DataType::BFloat16 => DataType::Float32,
            other => other,
        },
    };
    if copy != Some(true) && target == image.dtype() && !image.header().has_scaling() {
        if let Some(view) = numpy_view(slf, target)? {
            return Ok(view);
        }
    }
    if copy == Some(false) {
        return Err(PyValueError::new_err(
            "the values cannot be returned without a copy (they are scaled, converted, or stored byte-swapped)",
        ));
    }
    if target == DataType::BFloat16 {
        let ml = py.import("ml_dtypes").map_err(|_| {
            PyTypeError::new_err("NumPy needs the ml_dtypes package for bfloat16 arrays")
        })?;
        return bf16_bits(py, image)?.call_method1("view", (ml.getattr("bfloat16")?,));
    }
    owned_numpy(py, image, target)
}

/// The dtype `to_torch` uses by default: like NumPy, but unsigned types
/// without full PyTorch support are widened to signed ones.
fn torch_default(image: &NiftiImage) -> DataType {
    match default_dtype(image) {
        DataType::UInt16 => DataType::Int32,
        DataType::UInt32 | DataType::UInt64 => DataType::Int64,
        other => other,
    }
}

/// `"cpu"`, `"gpu"`, `"cuda:1"`, ... to a JAX device; anything else passes through.
fn jax_device<'py>(
    jax: &Bound<'py, PyModule>,
    device: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let Ok(name) = device.extract::<String>() else {
        return Ok(device.clone());
    };
    let (platform, index) = match name.split_once(':') {
        Some((p, i)) => (
            p.to_string(),
            i.parse::<usize>()
                .map_err(|_| PyValueError::new_err(format!("invalid device '{name}'")))?,
        ),
        None => (name.clone(), 0),
    };
    let platform = if platform == "cuda" {
        "gpu".to_string()
    } else {
        platform
    };
    let devices = jax.call_method1("devices", (platform,))?;
    devices
        .get_item(index)
        .map_err(|_| PyValueError::new_err(format!("no JAX device '{name}'")))
}

fn spatial_units_name(u: SpatialUnits) -> &'static str {
    match u {
        SpatialUnits::Meter => "m",
        SpatialUnits::Millimeter => "mm",
        SpatialUnits::Micrometer => "um",
        _ => "unknown",
    }
}

fn temporal_units_name(u: TemporalUnits) -> &'static str {
    match u {
        TemporalUnits::Second => "s",
        TemporalUnits::Millisecond => "ms",
        TemporalUnits::Microsecond => "us",
        TemporalUnits::Hertz => "hz",
        TemporalUnits::Ppm => "ppm",
        TemporalUnits::RadPerSecond => "rads",
        _ => "unknown",
    }
}

/// All header fields as a dictionary.
pub(crate) fn header_dict<'py>(py: Python<'py>, h: &NiftiHeader) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    let version = match h.version {
        NiftiVersion::Nifti1 => "nifti1",
        NiftiVersion::Nifti2 => "nifti2",
    };
    d.set_item("version", version)?;
    d.set_item("shape", PyTuple::new(py, h.shape())?)?;
    d.set_item("dtype", h.datatype.name())?;
    d.set_item("affine", affine_to_py(py, &h.affine()))?;
    d.set_item("qform", h.qform().map(|a| affine_to_py(py, &a)))?;
    d.set_item("sform", h.sform().map(|a| affine_to_py(py, &a)))?;
    d.set_item("qform_code", h.qform_code)?;
    d.set_item("sform_code", h.sform_code)?;
    d.set_item("pixdim", PyTuple::new(py, h.pixdim)?)?;
    d.set_item("spatial_units", spatial_units_name(h.spatial_units))?;
    d.set_item("temporal_units", temporal_units_name(h.temporal_units))?;
    d.set_item("scl_slope", h.scl_slope)?;
    d.set_item("scl_inter", h.scl_inter)?;
    d.set_item("cal_min", h.cal_min)?;
    d.set_item("cal_max", h.cal_max)?;
    d.set_item("intent_code", h.intent_code)?;
    d.set_item("intent_name", &h.intent_name)?;
    d.set_item("intent_p", PyTuple::new(py, h.intent_p)?)?;
    d.set_item("dim_info", h.dim_info)?;
    d.set_item("slice_code", h.slice_code)?;
    d.set_item("slice_start", h.slice_start)?;
    d.set_item("slice_end", h.slice_end)?;
    d.set_item("slice_duration", h.slice_duration)?;
    d.set_item("toffset", h.toffset)?;
    d.set_item("descrip", &h.descrip)?;
    d.set_item("aux_file", &h.aux_file)?;
    let extensions = PyList::empty(py);
    for e in &h.extensions {
        extensions.append((e.ecode, PyBytes::new(py, &e.data)))?;
    }
    d.set_item("extensions", extensions)?;
    Ok(d)
}

/// Apply `with_header` keyword updates.
fn update_header(h: &mut NiftiHeader, updates: &Bound<'_, PyDict>) -> PyResult<()> {
    for (key, value) in updates.iter() {
        let key: String = key.extract()?;
        match key.as_str() {
            "qform_code" => h.qform_code = value.extract()?,
            "sform_code" => h.sform_code = value.extract()?,
            "pixdim" => h.pixdim = value.extract()?,
            "spatial_units" => {
                h.spatial_units = match value.extract::<String>()?.as_str() {
                    "m" => SpatialUnits::Meter,
                    "mm" => SpatialUnits::Millimeter,
                    "um" => SpatialUnits::Micrometer,
                    "unknown" => SpatialUnits::Unknown,
                    other => return Err(PyValueError::new_err(format!("unknown spatial units '{other}'"))),
                }
            }
            "temporal_units" => {
                h.temporal_units = match value.extract::<String>()?.as_str() {
                    "s" => TemporalUnits::Second,
                    "ms" => TemporalUnits::Millisecond,
                    "us" => TemporalUnits::Microsecond,
                    "hz" => TemporalUnits::Hertz,
                    "ppm" => TemporalUnits::Ppm,
                    "rads" => TemporalUnits::RadPerSecond,
                    "unknown" => TemporalUnits::Unknown,
                    other => return Err(PyValueError::new_err(format!("unknown temporal units '{other}'"))),
                }
            }
            "scl_slope" => h.scl_slope = value.extract()?,
            "scl_inter" => h.scl_inter = value.extract()?,
            "cal_min" => h.cal_min = value.extract()?,
            "cal_max" => h.cal_max = value.extract()?,
            "intent_code" => h.intent_code = value.extract()?,
            "intent_name" => h.intent_name = value.extract()?,
            "intent_p" => h.intent_p = value.extract()?,
            "dim_info" => h.dim_info = value.extract()?,
            "slice_code" => h.slice_code = value.extract()?,
            "slice_start" => h.slice_start = value.extract()?,
            "slice_end" => h.slice_end = value.extract()?,
            "slice_duration" => h.slice_duration = value.extract()?,
            "toffset" => h.toffset = value.extract()?,
            "descrip" => h.descrip = value.extract()?,
            "aux_file" => h.aux_file = value.extract()?,
            "extensions" => {
                let items: Vec<(i32, Vec<u8>)> = value.extract()?;
                h.extensions = items.into_iter().map(|(code, data)| NiftiExtension::new(code, data)).collect();
            }
            "shape" | "dtype" | "affine" | "qform" | "sform" | "version" => {
                return Err(PyValueError::new_err(format!(
                    "'{key}' cannot be set with with_header; use the image methods (with_affine, with_dtype, ...)"
                )))
            }
            other => return Err(PyValueError::new_err(format!("unknown header field '{other}'"))),
        }
    }
    Ok(())
}

#[pymethods]
impl PyNiftiImage {
    /// Create an image from an array (any numeric dtype, indexed `[x, y, z, ...]`)
    /// and a 4x4 voxel-to-world affine (identity by default).
    #[new]
    #[pyo3(signature = (data, affine = None))]
    fn new(data: &Bound<'_, PyAny>, affine: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let affine = match affine {
            Some(a) if !a.is_none() => affine_from_py(a)?,
            _ => IDENTITY,
        };
        Ok(image_from_array(data, affine)?.into())
    }

    /// Image shape.
    #[getter]
    fn shape<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, self.inner.shape())
    }

    /// Number of dimensions.
    #[getter]
    fn ndim(&self) -> usize {
        self.inner.ndim()
    }

    /// Stored datatype name (`"uint8"`, `"float32"`, ...).
    #[getter]
    fn dtype(&self) -> &'static str {
        self.inner.dtype().name()
    }

    /// The 4x4 voxel-to-world affine (float64).
    #[getter]
    fn affine<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        affine_to_py(py, &self.inner.affine())
    }

    /// Voxel sizes along each axis.
    #[getter]
    fn spacing<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, self.inner.spacing())
    }

    /// Anatomical orientation of the voxel axes, such as `"RAS"`.
    #[getter]
    fn orientation(&self) -> String {
        self.inner.orientation().to_string()
    }

    /// All header fields as a dictionary.
    #[getter]
    fn header<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        header_dict(py, self.inner.header())
    }

    /// Values as a NumPy array indexed `[x, y, z, ...]` (Fortran order).
    ///
    /// Without `dtype`, the stored type is used if values are unscaled, and
    /// float32 otherwise. When possible the array shares memory with the image
    /// and is read-only; pass `copy=True` for a writable copy.
    #[pyo3(signature = (dtype = None, *, copy = false))]
    fn to_numpy<'py>(
        slf: &Bound<'py, Self>,
        dtype: Option<&Bound<'py, PyAny>>,
        copy: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        to_numpy_impl(slf, dtype, Some(copy).filter(|&c| c))
    }

    #[pyo3(signature = (dtype = None, copy = None))]
    fn __array__<'py>(
        slf: &Bound<'py, Self>,
        dtype: Option<&Bound<'py, PyAny>>,
        copy: Option<bool>,
    ) -> PyResult<Bound<'py, PyAny>> {
        to_numpy_impl(slf, dtype, copy)
    }

    /// Values as a new PyTorch tensor (never shares memory with the image).
    ///
    /// Without `dtype`, the stored type is used if values are unscaled
    /// (uint16/uint32/uint64 are widened to int32/int64), and float32
    /// otherwise. bfloat16 is supported.
    #[pyo3(signature = (dtype = None, device = None))]
    fn to_torch<'py>(
        &self,
        py: Python<'py>,
        dtype: Option<&Bound<'py, PyAny>>,
        device: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let torch = py.import("torch")?;
        let target = match dtype {
            Some(d) if !d.is_none() => parse_dtype(d)?,
            _ => torch_default(&self.inner),
        };
        let tensor = if target == DataType::BFloat16 {
            let bits = bf16_bits(py, &self.inner)?;
            torch
                .call_method1("from_numpy", (bits,))?
                .call_method1("view", (torch.getattr("bfloat16")?,))?
        } else {
            torch.call_method1("from_numpy", (owned_numpy(py, &self.inner, target)?,))?
        };
        match device {
            Some(d) if !d.is_none() => tensor.call_method1("to", (d,)),
            _ => Ok(tensor),
        }
    }

    /// Values as a JAX array, optionally placed on `device` (a `jax.Device`
    /// or a name such as `"cpu"` or `"gpu:0"`). dtype rules follow `to_numpy`.
    #[pyo3(signature = (dtype = None, device = None))]
    fn to_jax<'py>(
        &self,
        py: Python<'py>,
        dtype: Option<&Bound<'py, PyAny>>,
        device: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let jax = py.import("jax")?;
        let target = match dtype {
            Some(d) if !d.is_none() => parse_dtype(d)?,
            _ => default_dtype(&self.inner),
        };
        let array = if target == DataType::BFloat16 {
            let ml = py.import("ml_dtypes")?;
            bf16_bits(py, &self.inner)?.call_method1("view", (ml.getattr("bfloat16")?,))?
        } else {
            owned_numpy(py, &self.inner, target)?
        };
        match device {
            Some(d) if !d.is_none() => {
                jax.call_method1("device_put", (array, jax_device(&jax, d)?))
            }
            _ => jax.getattr("numpy")?.call_method1("asarray", (array,)),
        }
    }

    /// Save to `path`; the format follows the extension (`.nii`, `.nii.gz`,
    /// `.hdr`/`.img`, `.jvol`).
    ///
    /// `compression_level` (0-9) and `mgzip` apply to gzipped files: Mgzip
    /// output is still valid gzip and decompresses in parallel with medrs.
    /// `quality` (1-100) selects lossy `.jvol` compression (lossless when
    /// omitted) and `chunk_shape` sets the `.jvol` chunk size.
    #[pyo3(signature = (path, *, compression_level = None, mgzip = false, quality = None, chunk_shape = None))]
    fn save(
        &self,
        py: Python<'_>,
        path: PathBuf,
        compression_level: Option<u32>,
        mgzip: bool,
        quality: Option<u8>,
        chunk_shape: Option<[usize; 3]>,
    ) -> PyResult<()> {
        let is_jvol = path
            .extension()
            .is_some_and(|e| e.eq_ignore_ascii_case("jvol"));
        if is_jvol {
            if compression_level.is_some() || mgzip {
                return Err(PyValueError::new_err(
                    "compression_level and mgzip apply to gzipped NIfTI files, not .jvol",
                ));
            }
            let mut options = match quality {
                Some(q) => medrs::jvol::JvolOptions::lossy(q).or_raise()?,
                None => medrs::jvol::JvolOptions::lossless(),
            };
            if let Some(shape) = chunk_shape {
                options = options.with_chunk_shape(shape).or_raise()?;
            }
            return py
                .detach(|| medrs::jvol::save(&self.inner, &path, &options))
                .or_raise();
        }
        if quality.is_some() || chunk_shape.is_some() {
            return Err(PyValueError::new_err(
                "quality and chunk_shape apply to .jvol files",
            ));
        }
        let mut options = medrs::nifti::SaveOptions::default();
        if let Some(level) = compression_level {
            if level > 9 {
                return Err(PyValueError::new_err(format!(
                    "compression_level must be between 0 and 9, got {level}"
                )));
            }
            options.compression_level = level;
        }
        options.mgzip = mgzip;
        py.detach(|| medrs::nifti::save_with_options(&self.inner, &path, &options))
            .or_raise()
    }

    /// Resample to new voxel sizes, keeping the field of view.
    #[pyo3(signature = (spacing, interpolation = "trilinear"))]
    fn resample(&self, py: Python<'_>, spacing: [f64; 3], interpolation: &str) -> PyResult<Self> {
        let interp = parse_interpolation(interpolation)?;
        let img = &self.inner;
        Ok(py
            .detach(|| t::resample_to_spacing(img, spacing, interp))
            .or_raise()?
            .into())
    }

    /// Resample to a new grid shape, keeping the field of view.
    #[pyo3(signature = (shape, interpolation = "trilinear"))]
    fn resample_to_shape(
        &self,
        py: Python<'_>,
        shape: [usize; 3],
        interpolation: &str,
    ) -> PyResult<Self> {
        let interp = parse_interpolation(interpolation)?;
        let img = &self.inner;
        Ok(py
            .detach(|| t::resample_to_shape(img, shape, interp))
            .or_raise()?
            .into())
    }

    /// Resample onto the voxel grid of `reference` (shape and affine).
    #[pyo3(signature = (reference, interpolation = "trilinear"))]
    fn resample_like(
        &self,
        py: Python<'_>,
        reference: &Self,
        interpolation: &str,
    ) -> PyResult<Self> {
        let interp = parse_interpolation(interpolation)?;
        let (img, reference) = (&self.inner, &reference.inner);
        Ok(py
            .detach(|| t::resample_like(img, reference, interp))
            .or_raise()?
            .into())
    }

    /// Permute and flip axes to match `orientation` (such as `"RAS"`); exact.
    fn reorient(&self, py: Python<'_>, orientation: &str) -> PyResult<Self> {
        let target = parse_orientation(orientation)?;
        let img = &self.inner;
        Ok(py.detach(|| t::reorient(img, target)).or_raise()?.into())
    }

    /// Crop `shape` voxels starting at `offset` (first three axes).
    fn crop(&self, py: Python<'_>, offset: [usize; 3], shape: [usize; 3]) -> PyResult<Self> {
        let img = &self.inner;
        Ok(py.detach(|| t::crop(img, offset, shape)).or_raise()?.into())
    }

    /// Centre-crop or pad to `shape`, padding with `pad_value`.
    #[pyo3(signature = (shape, pad_value = 0.0))]
    fn crop_or_pad(&self, py: Python<'_>, shape: [usize; 3], pad_value: f64) -> PyResult<Self> {
        let img = &self.inner;
        Ok(py
            .detach(|| t::crop_or_pad(img, shape, pad_value))
            .or_raise()?
            .into())
    }

    /// Reverse the voxel order along `axes` (world-preserving).
    fn flip(&self, py: Python<'_>, axes: Vec<usize>) -> PyResult<Self> {
        let img = &self.inner;
        Ok(py.detach(|| t::flip(img, &axes)).or_raise()?.into())
    }

    /// Rotate by `k` × 90 degrees in the plane of `axes` (like `numpy.rot90`),
    /// world-preserving.
    #[pyo3(signature = (axes = (0, 1), k = 1), text_signature = "($self, axes=(0, 1), k=1)")]
    fn rotate_90(&self, py: Python<'_>, axes: (usize, usize), k: i32) -> PyResult<Self> {
        let img = &self.inner;
        Ok(py.detach(|| t::rotate_90(img, axes, k)).or_raise()?.into())
    }

    /// Normalize to zero mean and unit variance (over non-zero voxels only,
    /// keeping zeros, if `nonzero`).
    #[pyo3(signature = (*, nonzero = false))]
    fn z_normalize(&self, py: Python<'_>, nonzero: bool) -> PyResult<Self> {
        let img = &self.inner;
        let out = py.detach(|| {
            if nonzero {
                t::z_normalization_nonzero(img)
            } else {
                t::z_normalization(img)
            }
        });
        Ok(out.or_raise()?.into())
    }

    /// Linearly map the value range onto `[out_min, out_max]`.
    #[pyo3(signature = (out_min = 0.0, out_max = 1.0))]
    fn rescale(&self, py: Python<'_>, out_min: f64, out_max: f64) -> PyResult<Self> {
        let img = &self.inner;
        Ok(py
            .detach(|| t::rescale_intensity(img, out_min, out_max))
            .or_raise()?
            .into())
    }

    /// Percentiles of the values (NumPy's default linear interpolation), for
    /// each of `q` in `[0, 100]`. With `nonzero`, zero voxels are left out.
    #[pyo3(signature = (q, *, nonzero = false))]
    fn percentiles<'py>(
        &self,
        py: Python<'py>,
        q: Vec<f64>,
        nonzero: bool,
    ) -> PyResult<Bound<'py, PyTuple>> {
        let img = &self.inner;
        let values = py.detach(|| t::percentiles(img, &q, nonzero)).or_raise()?;
        PyTuple::new(py, values)
    }

    /// Clip to the `lower` and `upper` percentiles (of the non-zero voxels,
    /// if `nonzero`) and map that range linearly onto `[out_min, out_max]`.
    #[pyo3(signature = (lower, upper, *, nonzero = false, out_min = 0.0, out_max = 1.0))]
    fn rescale_percentiles(
        &self,
        py: Python<'_>,
        lower: f64,
        upper: f64,
        nonzero: bool,
        out_min: f64,
        out_max: f64,
    ) -> PyResult<Self> {
        let img = &self.inner;
        Ok(py
            .detach(|| t::rescale_percentiles(img, lower, upper, nonzero, out_min, out_max))
            .or_raise()?
            .into())
    }

    /// Clamp values to `[min, max]`.
    fn clamp(&self, py: Python<'_>, min: f64, max: f64) -> PyResult<Self> {
        let img = &self.inner;
        Ok(py.detach(|| t::clamp(img, min, max)).or_raise()?.into())
    }

    /// Range-preserving gamma adjustment.
    fn adjust_gamma(&self, py: Python<'_>, gamma: f64) -> PyResult<Self> {
        let img = &self.inner;
        Ok(py.detach(|| t::adjust_gamma(img, gamma)).or_raise()?.into())
    }

    /// Convert to another stored datatype (scaling applied, then reset;
    /// integers rounded and saturated).
    fn with_dtype(&self, py: Python<'_>, dtype: &Bound<'_, PyAny>) -> PyResult<Self> {
        let dtype = parse_dtype(dtype)?;
        let img = &self.inner;
        Ok(py.detach(|| img.with_dtype(dtype)).or_raise()?.into())
    }

    /// The same image with a new voxel-to-world affine (both sform and qform).
    fn with_affine(&self, affine: &Bound<'_, PyAny>) -> PyResult<Self> {
        let mut img = self.inner.clone();
        img.set_affine(affine_from_py(affine)?);
        Ok(img.into())
    }

    /// The same image with new voxel values (same shape; any dtype).
    fn with_data(&self, data: &Bound<'_, PyAny>) -> PyResult<Self> {
        let new = image_from_array(data, IDENTITY)?;
        if new.shape() != self.inner.shape() {
            return Err(PyValueError::new_err(format!(
                "new data has shape {:?}, the image has {:?}",
                new.shape(),
                self.inner.shape()
            )));
        }
        Ok(new.with_header(self.inner.header().clone()).into())
    }

    /// The same image with header fields changed, for example
    /// `img.with_header(descrip="denoised", sform_code=4)`.
    #[pyo3(signature = (**fields))]
    fn with_header(&self, fields: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let mut header = self.inner.header().clone();
        if let Some(fields) = fields {
            update_header(&mut header, fields)?;
        }
        Ok(self.inner.clone().with_header(header).into())
    }

    fn __repr__(&self) -> String {
        let spacing: Vec<String> = self
            .inner
            .spacing()
            .iter()
            .map(|s| format!("{s:.4}"))
            .collect();
        let shape: Vec<String> = self.inner.shape().iter().map(ToString::to_string).collect();
        let shape = if shape.len() == 1 {
            format!("{},", shape[0])
        } else {
            shape.join(", ")
        };
        format!(
            "NiftiImage(shape=({shape}), dtype={}, spacing=({}), orientation={})",
            self.inner.dtype().name(),
            spacing.join(", "),
            self.inner.orientation()
        )
    }

    fn __reduce__<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(Bound<'py, PyAny>, (Bound<'py, PyBytes>,))> {
        let bytes = py
            .detach(|| medrs::nifti::to_bytes(&self.inner, None))
            .or_raise()?;
        let constructor = py.import("medrs._medrs")?.getattr("_image_from_bytes")?;
        Ok((constructor, (PyBytes::new(py, &bytes),)))
    }

    fn __copy__(slf: Bound<'_, Self>) -> Bound<'_, Self> {
        slf
    }

    #[pyo3(signature = (_memo, /))]
    fn __deepcopy__<'py>(slf: Bound<'py, Self>, _memo: &Bound<'py, PyAny>) -> Bound<'py, Self> {
        slf
    }
}

/// Rebuild an image from `to_bytes` output (used by pickle).
#[pyfunction]
fn _image_from_bytes(py: Python<'_>, data: Vec<u8>) -> PyResult<PyNiftiImage> {
    Ok(py
        .detach(|| medrs::nifti::from_bytes(data))
        .or_raise()?
        .into())
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyNiftiImage>()?;
    m.add_function(wrap_pyfunction!(_image_from_bytes, m)?)?;
    Ok(())
}
