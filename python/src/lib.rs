//! Python bindings for medrs.

// PyO3 converts arguments into owned Rust values before the call.
#![allow(clippy::needless_pass_by_value)]

mod image;
mod io;
mod loader;
mod pipeline;

use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyString;

pyo3::create_exception!(
    medrs,
    FormatError,
    PyValueError,
    "A file is not a valid image of its format, or is corrupt."
);

/// Convert a medrs error into the matching Python exception.
pub(crate) fn to_py_err(e: medrs::Error) -> PyErr {
    use medrs::Error as E;
    let message = e.to_string();
    match e {
        E::Io(io) => PyErr::from(io),
        E::InvalidMagic(_)
        | E::UnsupportedDataType(_)
        | E::InvalidFileFormat(_)
        | E::Decompression(_) => FormatError::new_err(message),
        E::DataTypeMismatch { .. } => PyTypeError::new_err(message),
        E::Internal(_) => PyRuntimeError::new_err(message),
        _ => PyValueError::new_err(message),
    }
}

/// `Result<T, medrs::Error>` to `PyResult<T>`.
pub(crate) trait OrRaise<T> {
    fn or_raise(self) -> PyResult<T>;
}

impl<T> OrRaise<T> for medrs::Result<T> {
    fn or_raise(self) -> PyResult<T> {
        self.map_err(to_py_err)
    }
}

/// Parse a datatype name such as `"float32"` or a NumPy/PyTorch dtype object.
pub(crate) fn parse_dtype(obj: &Bound<'_, PyAny>) -> PyResult<medrs::DataType> {
    let name = if let Ok(s) = obj.cast::<PyString>() {
        s.to_string()
    } else if let Ok(name) = obj.getattr("__name__") {
        // Scalar types: numpy.float32, ml_dtypes.bfloat16, ...
        name.extract()?
    } else if let Ok(name) = obj.getattr("name") {
        // numpy.dtype
        name.extract()?
    } else {
        // torch.dtype prints as "torch.float32".
        let s = obj.str()?.to_string();
        s.strip_prefix("torch.").unwrap_or(&s).to_string()
    };
    name.parse()
        .map_err(|e: medrs::Error| PyValueError::new_err(e.to_string()))
}

/// Set the number of threads medrs uses for parallel work (0 restores the
/// default, the number of CPUs).
#[pyfunction]
fn set_num_threads(threads: usize) {
    medrs::set_num_threads(threads);
}

/// Number of threads medrs uses for parallel work.
#[pyfunction]
fn num_threads() -> usize {
    medrs::num_threads()
}

#[pymodule(gil_used = false)]
fn _medrs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("FormatError", m.py().get_type::<FormatError>())?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(set_num_threads, m)?)?;
    m.add_function(wrap_pyfunction!(num_threads, m)?)?;
    image::register(m)?;
    io::register(m)?;
    pipeline::register(m)?;
    loader::register(m)?;
    Ok(())
}
