//! Loading functions.

use crate::image::{header_dict, parse_interpolation, PyNiftiImage};
use crate::OrRaise;
use medrs::transforms::Interpolation;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::path::PathBuf;

/// Load an image (`.nii`, `.nii.gz`, `.hdr`/`.img`, or `.jvol`).
///
/// Uncompressed data is memory-mapped and read lazily. Gzip is detected from
/// the file contents, and block-compressed gzip (Mgzip, BGZF) is decompressed
/// in parallel. With `cache=True`, decompressed data is kept in a
/// process-wide cache (see `set_cache_limits`) so loading the same file again
/// skips decompression.
#[pyfunction]
#[pyo3(signature = (path, *, cache = false))]
fn load(py: Python<'_>, path: PathBuf, cache: bool) -> PyResult<PyNiftiImage> {
    let image = py.detach(|| {
        if cache {
            medrs::nifti::load_cached(&path)
        } else {
            medrs::nifti::load(&path)
        }
    });
    Ok(image.or_raise()?.into())
}

/// Load only `shape` voxels starting at `offset` (first three axes; any
/// further axes are kept whole).
///
/// Uncompressed files read only the region from disk, and `.jvol` files
/// decode only the chunks it touches. Gzipped files are decompressed once and
/// cached, so further crops of the same file are fast.
#[pyfunction]
fn load_cropped(
    py: Python<'_>,
    path: PathBuf,
    offset: [usize; 3],
    shape: [usize; 3],
) -> PyResult<PyNiftiImage> {
    Ok(py
        .detach(|| medrs::nifti::load_cropped(&path, offset, shape))
        .or_raise()?
        .into())
}

/// Read only the header of an image file, as a dictionary.
#[pyfunction]
fn load_header(py: Python<'_>, path: PathBuf) -> PyResult<Bound<'_, PyDict>> {
    let header = py.detach(|| medrs::nifti::load_header(&path)).or_raise()?;
    header_dict(py, &header)
}

/// Load several images and resample each onto the voxel grid of
/// `paths[reference]`.
///
/// `interpolation` is one name for all images or one per image (use
/// `"nearest"` for label maps). Images already on the reference grid are
/// returned unchanged.
#[pyfunction]
#[pyo3(signature = (paths, *, reference = 0, interpolation = None))]
fn load_multi(
    py: Python<'_>,
    paths: Vec<PathBuf>,
    reference: usize,
    interpolation: Option<&Bound<'_, PyAny>>,
) -> PyResult<Vec<PyNiftiImage>> {
    let interps: Vec<Interpolation> = match interpolation {
        None => vec![Interpolation::Trilinear; paths.len()],
        Some(obj) => {
            if let Ok(name) = obj.extract::<String>() {
                vec![parse_interpolation(&name)?; paths.len()]
            } else {
                let names: Vec<String> = obj.extract()?;
                if names.len() != paths.len() {
                    return Err(PyValueError::new_err(format!(
                        "{} interpolation names for {} paths",
                        names.len(),
                        paths.len()
                    )));
                }
                names
                    .iter()
                    .map(|n| parse_interpolation(n))
                    .collect::<PyResult<_>>()?
            }
        }
    };
    let files: Vec<(PathBuf, Interpolation)> = paths.into_iter().zip(interps).collect();
    let images = py
        .detach(|| medrs::nifti::load_multi(&files, reference))
        .or_raise()?;
    Ok(images.into_iter().map(PyNiftiImage::from).collect())
}

/// Remove every entry from the decompression cache.
#[pyfunction]
fn clear_cache() {
    medrs::nifti::clear_decompression_cache();
}

/// Limit the decompression cache used by `load(cache=True)` and
/// `load_cropped` (defaults: 16 files, 1 GiB).
#[pyfunction]
#[pyo3(signature = (*, max_entries = None, max_bytes = None))]
fn set_cache_limits(max_entries: Option<usize>, max_bytes: Option<usize>) {
    if let Some(n) = max_entries {
        medrs::nifti::set_cache_size(n);
    }
    if let Some(b) = max_bytes {
        medrs::nifti::set_cache_max_bytes(b);
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(load, m)?)?;
    m.add_function(wrap_pyfunction!(load_cropped, m)?)?;
    m.add_function(wrap_pyfunction!(load_header, m)?)?;
    m.add_function(wrap_pyfunction!(load_multi, m)?)?;
    m.add_function(wrap_pyfunction!(clear_cache, m)?)?;
    m.add_function(wrap_pyfunction!(set_cache_limits, m)?)?;
    Ok(())
}
