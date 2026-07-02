//! Python bindings for jvol volumetric compression.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::image::PyNiftiImage;
use super::validation::to_py_err;
use crate::jvol::{self, JvolOptions};
use crate::nifti::DataType;

/// Parse an optional dtype string (e.g. "bfloat16", "f32") into a `DataType`,
/// reusing the crate's `FromStr` parsing shared with `NiftiImage::with_dtype`.
fn parse_dtype(dtype: Option<&str>) -> PyResult<Option<DataType>> {
    dtype
        .map(str::parse::<DataType>)
        .transpose()
        .map_err(|e| PyValueError::new_err(format!("{e}")))
}

/// Save a NIfTI image in jvol format (wavelet + Rice-coded volumetric compression).
///
/// jvol typically compresses far smaller than gzip for medical volumes.
/// Loading is transparent through `medrs.load()` for paths ending in `.jvol`.
///
/// Args:
///     image: NiftiImage to save
///     output: Output file path (typically .jvol)
///     quality: Lossy quality level (1-100, higher is better). Ignored when lossless=True.
///     lossless: Use lossless encoding (default: False). Required for integer/label data.
///
/// Example:
///     >>> img = medrs.load("brain.nii.gz")
///     >>> medrs.save_jvol(img, "brain.jvol", quality=60)
///     >>> medrs.save_jvol(seg, "seg.jvol", lossless=True)
#[pyfunction]
#[pyo3(signature = (image, output, quality=60, lossless=false))]
pub fn save_jvol(
    py: Python<'_>,
    image: &super::image::PyNiftiImage,
    output: &str,
    quality: u8,
    lossless: bool,
) -> PyResult<()> {
    let inner = &image.inner;
    let options = JvolOptions { quality, lossless };
    py.allow_threads(|| jvol::save(inner, output, options))
        .map_err(|e| to_py_err(e, &format!("Failed to save_jvol {}", output)))
}

/// Convert a NIfTI file to jvol format.
///
/// Loads the input file and saves it as jvol (wavelet + Rice-coded compression).
///
/// Args:
///     input_path: Path to input NIfTI file (.nii or .nii.gz)
///     output_path: Path for output .jvol file
///     quality: Lossy quality level (1-100, higher is better). Ignored when lossless=True.
///     lossless: Use lossless encoding (default: False). Required for integer/label data.
///
/// Example:
///     >>> medrs.convert_to_jvol("brain.nii.gz", "brain.jvol", quality=60)
///     >>> medrs.convert_to_jvol("segmentation.nii.gz", "segmentation.jvol", lossless=True)
#[pyfunction]
#[pyo3(signature = (input_path, output_path, quality=60, lossless=false))]
pub fn convert_to_jvol(
    py: Python<'_>,
    input_path: &str,
    output_path: &str,
    quality: u8,
    lossless: bool,
) -> PyResult<()> {
    let options = JvolOptions { quality, lossless };
    py.allow_threads(|| {
        let image = crate::nifti::load(input_path)?;
        jvol::save(&image, output_path, options)
    })
    .map_err(|e| to_py_err(e, &format!("Failed to convert_to_jvol {}", input_path)))
}

/// Load a lossy jvol file at reduced resolution (progressive/multiresolution decode).
///
/// Decodes only the wavelet subbands coarser than the requested factor, so a
/// downsampled preview costs a fraction of a full load. The finest, most
/// numerous detail subbands are never entropy-decoded. The returned image has
/// its spacing scaled by ``factor`` and its origin shifted by the half-pixel
/// convention, so it stays registered with the full-resolution volume.
///
/// Only works on lossy files. Lossless files store a single block with no
/// multiresolution structure; use ``medrs.load()`` for those.
///
/// Args:
///     path: Path to a lossy .jvol file
///     factor: Downsample factor per axis; a power of two, at most 2**levels
///     dtype: Output dtype override (e.g. "bfloat16"). None keeps the stored dtype.
///
/// Returns:
///     NiftiImage at 1/factor of the stored resolution per axis
///
/// Example:
///     >>> preview = medrs.load_jvol_downsampled("brain.jvol", 4)
///     >>> preview.shape  # roughly 1/4 of the full shape per axis
#[pyfunction]
#[pyo3(signature = (path, factor, dtype=None))]
pub fn load_jvol_downsampled(
    py: Python<'_>,
    path: &str,
    factor: usize,
    dtype: Option<&str>,
) -> PyResult<PyNiftiImage> {
    let target = parse_dtype(dtype)?;
    let inner = py
        .allow_threads(|| jvol::load_downsampled_as(path, factor, target))
        .map_err(|e| to_py_err(e, &format!("Failed to load_jvol_downsampled {}", path)))?;
    Ok(PyNiftiImage { inner })
}

/// Load a `.jvol` file, optionally materializing it as a dtype other than the
/// one it was saved with.
///
/// Useful for mixed-precision pipelines: store a volume once as a compact
/// lossy `.jvol` and decode it directly to bfloat16/float16 at train time,
/// skipping the full-precision intermediate array.
///
/// Args:
///     path: Path to a .jvol file
///     dtype: Output dtype override (e.g. "bfloat16", "float32"). None
///         reproduces the file's stored dtype.
///
/// Returns:
///     NiftiImage materialized as `dtype` (or the stored dtype if None)
///
/// Example:
///     >>> img = medrs.load_jvol("brain.jvol", dtype="bfloat16")
#[pyfunction]
#[pyo3(signature = (path, dtype=None))]
pub fn load_jvol(py: Python<'_>, path: &str, dtype: Option<&str>) -> PyResult<PyNiftiImage> {
    let target = parse_dtype(dtype)?;
    let inner = py
        .allow_threads(|| jvol::load_as(path, target))
        .map_err(|e| to_py_err(e, &format!("Failed to load_jvol {}", path)))?;
    Ok(PyNiftiImage { inner })
}

/// Load a `.jvol` file, decoding once per (path, dtype) and reusing the
/// decoded result on subsequent calls.
///
/// Particularly useful in training pipelines that revisit the same volume
/// across epochs: the first call pays the wavelet/entropy decode cost, later
/// calls only pay an array copy out of the cache. A changed file on disk
/// (size or modification time) invalidates its cache entry.
///
/// Args:
///     path: Path to a .jvol file
///     dtype: Output dtype override (e.g. "bfloat16"). Part of the cache key,
///         so the same file cached at two dtypes decodes and caches each
///         independently.
///
/// Returns:
///     NiftiImage materialized as `dtype` (or the stored dtype if None)
///
/// Example:
///     >>> img1 = medrs.load_jvol_cached("brain.jvol", dtype="bfloat16")
///     >>> img2 = medrs.load_jvol_cached("brain.jvol", dtype="bfloat16")  # cache hit
///     >>> medrs.clear_jvol_cache()
#[pyfunction]
#[pyo3(signature = (path, dtype=None))]
pub fn load_jvol_cached(py: Python<'_>, path: &str, dtype: Option<&str>) -> PyResult<PyNiftiImage> {
    let target = parse_dtype(dtype)?;
    let inner = py
        .allow_threads(|| jvol::load_cached(path, target))
        .map_err(|e| to_py_err(e, &format!("Failed to load_jvol_cached {}", path)))?;
    Ok(PyNiftiImage { inner })
}

/// Clear the global jvol decoded-image cache.
///
/// Call this to free memory held by images cached via `load_jvol_cached()`.
///
/// Example:
///     >>> medrs.clear_jvol_cache()
#[pyfunction]
pub fn clear_jvol_cache() {
    jvol::clear_jvol_cache();
}

/// Set the maximum size of the jvol decoded-image cache.
///
/// Default is 10 entries. Set to 0 to disable caching.
///
/// Args:
///     max_entries: Maximum number of decoded images to cache
///
/// Example:
///     >>> medrs.set_jvol_cache_size(20)
///     >>> medrs.set_jvol_cache_size(0)  # disable
#[pyfunction]
pub fn set_jvol_cache_size(max_entries: usize) {
    jvol::set_jvol_cache_size(max_entries);
}

/// Decode a `.jvol` file and write it out as an uncompressed `.nii`.
///
/// A later `medrs.load()` of `output_path` memory-maps it with zero-copy
/// access instead of paying the wavelet/entropy decode cost on every load.
/// `output_path` must not end in `.gz` or `.jvol`.
///
/// Args:
///     jvol_path: Path to the input .jvol file
///     output_path: Path for the output uncompressed .nii file
///     dtype: Output dtype override (e.g. "float32"). None keeps the stored dtype.
///
/// Example:
///     >>> medrs.convert_jvol_to_nii("brain.jvol", "brain_cache.nii", dtype="float32")
///     >>> img = medrs.load("brain_cache.nii")  # zero-copy mmap
#[pyfunction]
#[pyo3(signature = (jvol_path, output_path, dtype=None))]
pub fn convert_jvol_to_nii(
    py: Python<'_>,
    jvol_path: &str,
    output_path: &str,
    dtype: Option<&str>,
) -> PyResult<()> {
    let target = parse_dtype(dtype)?;
    py.allow_threads(|| jvol::transcode_to_nii(jvol_path, output_path, target))
        .map_err(|e| to_py_err(e, &format!("Failed to convert_jvol_to_nii {}", jvol_path)))
}

/// Load a `.jvol` file via a decode-once mmap cache.
///
/// Transcodes to an uncompressed `.nii` under `cache_dir` on first use, then
/// memory-maps that file on this and subsequent calls, skipping the
/// transcode step when a cached `.nii` already exists and is newer than the
/// source `.jvol`. Trades one wavelet/entropy decode (on the first call) plus
/// disk space in `cache_dir` for zero-copy mmap access on every subsequent
/// call, which repeated-epoch training loops can amortize.
///
/// Args:
///     jvol_path: Path to the input .jvol file
///     cache_dir: Directory to store the transcoded .nii cache file in
///     dtype: Output dtype override (e.g. "float32"). None keeps the stored dtype.
///
/// Returns:
///     NiftiImage, mmap-backed after the first call
///
/// Example:
///     >>> img = medrs.load_jvol_via_mmap_cache("brain.jvol", "/tmp/jvol_cache")
#[pyfunction]
#[pyo3(signature = (jvol_path, cache_dir, dtype=None))]
pub fn load_jvol_via_mmap_cache(
    py: Python<'_>,
    jvol_path: &str,
    cache_dir: &str,
    dtype: Option<&str>,
) -> PyResult<PyNiftiImage> {
    let target = parse_dtype(dtype)?;
    let inner = py
        .allow_threads(|| jvol::load_via_mmap_cache(jvol_path, cache_dir, target))
        .map_err(|e| {
            to_py_err(
                e,
                &format!("Failed to load_jvol_via_mmap_cache {}", jvol_path),
            )
        })?;
    Ok(PyNiftiImage { inner })
}
