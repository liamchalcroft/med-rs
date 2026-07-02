//! Training data loaders for Python bindings.

use pyo3::exceptions::{PyRuntimeError, PyStopIteration, PyValueError};
use pyo3::prelude::*;

use super::image::PyNiftiImage;
use crate::nifti::{
    FastLoader as RustFastLoader, PatchConfig, TrainingDataLoader as RustTrainingDataLoader,
};

/// High-performance training data loader with prefetching and caching.
///
/// Loads training patches from multiple volumes, maintaining an LRU cache and
/// prefetching upcoming data to keep throughput high.
#[pyclass(name = "TrainingDataLoader")]
pub struct PyTrainingDataLoader {
    loader: RustTrainingDataLoader,
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
    ///     ```python
    ///     loader = medrs.TrainingDataLoader(
    ///         volumes=["vol1.nii", "vol2.nii"],
    ///         patch_size=[64, 64, 64],
    ///         patches_per_volume=4,
    ///         patch_overlap=[0, 0, 0],
    ///         randomize=True,
    ///         cache_size=1000
    ///     )
    ///     patch = loader.next_patch()
    ///     ```
    #[new]
    #[pyo3(signature = (volumes, patch_size, patches_per_volume, patch_overlap, randomize, cache_size=None, seed=None))]
    fn new(
        volumes: Vec<String>,
        patch_size: [usize; 3],
        patches_per_volume: usize,
        patch_overlap: [usize; 3],
        randomize: bool,
        cache_size: Option<usize>,
        seed: Option<u64>,
    ) -> PyResult<Self> {
        for i in 0..3 {
            if patch_overlap[i] >= patch_size[i] {
                return Err(PyValueError::new_err(
                    "patch_overlap must be smaller than patch_size in all dimensions",
                ));
            }
        }

        let config = PatchConfig {
            shape: patch_size,
            patches_per_volume,
            overlap: patch_overlap,
            randomize,
            seed,
        };

        let cache_size = cache_size.unwrap_or(1000);
        let loader = RustTrainingDataLoader::new(volumes, config, cache_size)
            .map_err(|e| PyValueError::new_err(format!("Failed to create loader: {}", e)))?;

        Ok(Self { loader })
    }

    /// Get next training patch.
    ///
    /// Returns next training patch with automatic prefetching.
    /// Raises StopIteration when all patches are processed.
    fn next_patch(&mut self, py: Python<'_>) -> PyResult<PyNiftiImage> {
        let loader = &mut self.loader;
        match py.allow_threads(|| loader.next_patch()) {
            Ok(inner) => Ok(PyNiftiImage { inner }),
            Err(crate::error::Error::Exhausted(msg)) => Err(PyStopIteration::new_err(msg)),
            Err(e) => Err(pyo3::exceptions::PyIOError::new_err(format!(
                "next_patch: {}",
                e
            ))),
        }
    }

    fn __len__(&self) -> usize {
        self.loader.volumes_len() * self.loader.patches_per_volume()
    }

    fn __iter__(mut slf: PyRefMut<'_, Self>) -> PyResult<PyRefMut<'_, Self>> {
        let py = slf.py();
        let loader = &mut slf.loader;
        py.allow_threads(|| loader.reset())
            .map_err(|e| PyValueError::new_err(format!("Failed to reset loader: {}", e)))?;
        Ok(slf)
    }

    fn __next__(&mut self, py: Python<'_>) -> PyResult<Option<PyNiftiImage>> {
        let loader = &mut self.loader;
        match py.allow_threads(|| loader.next_patch()) {
            Ok(img) => Ok(Some(PyNiftiImage { inner: img })),
            Err(crate::error::Error::Exhausted(_)) => Ok(None),
            Err(e) => Err(pyo3::exceptions::PyIOError::new_err(format!(
                "iterator: {}",
                e
            ))),
        }
    }

    /// Get performance statistics.
    fn stats(&self) -> String {
        format!("{}", self.loader.stats())
    }

    /// Reset loader to start from beginning.
    fn reset(&mut self, py: Python<'_>) -> PyResult<()> {
        let loader = &mut self.loader;
        py.allow_threads(|| loader.reset())
            .map_err(|e| PyValueError::new_err(format!("Failed to reset loader: {}", e)))
    }
}

/// Same-thread `Send` wrapper for a borrow of the loader.
///
/// `FastLoader` holds an `mpsc::Receiver` and is therefore not `Sync`, so a
/// plain `&FastLoader` cannot satisfy `allow_threads`'s `Send` bound. The
/// closure runs on the calling thread (the GIL is only released, not moved),
/// and `PyFastLoader` is `unsendable` and borrowed `&mut self` for the call, so
/// the receiver is never touched from another thread.
#[allow(unsafe_code)]
struct LoaderRef<'a>(&'a RustFastLoader);

#[allow(unsafe_code)]
unsafe impl Send for LoaderRef<'_> {}

impl LoaderRef<'_> {
    fn next(&self) -> Option<crate::error::Result<crate::nifti::NiftiImage>> {
        self.0.next()
    }
}

/// One-shot streaming loader over a set of volumes.
///
/// The loader drains a background worker channel exactly once. After the
/// underlying channel closes it is exhausted: iterating again raises
/// RuntimeError. Create a new FastLoader for each epoch.
#[pyclass(name = "FastLoader", unsendable)]
pub struct PyFastLoader {
    loader: RustFastLoader,
    consumed: bool,
}

#[pymethods]
impl PyFastLoader {
    #[new]
    #[pyo3(signature = (volumes, patch_shape, prefetch=16, workers=None, shuffle=true, seed=None, mgzip_threads=0))]
    fn new(
        volumes: Vec<String>,
        patch_shape: [usize; 3],
        prefetch: usize,
        workers: Option<usize>,
        shuffle: bool,
        seed: Option<u64>,
        mgzip_threads: usize,
    ) -> PyResult<Self> {
        let mut builder = RustFastLoader::new(volumes, patch_shape)
            .prefetch(prefetch)
            .shuffle(shuffle);

        if let Some(w) = workers {
            builder = builder.workers(w);
        }
        if let Some(s) = seed {
            builder = builder.seed(s);
        }
        if mgzip_threads > 0 {
            builder = builder.mgzip(mgzip_threads);
        }

        let loader = builder
            .build()
            .map_err(|e| PyValueError::new_err(format!("Failed to create FastLoader: {}", e)))?;

        Ok(Self {
            loader,
            consumed: false,
        })
    }

    fn __len__(&self) -> usize {
        self.loader.len()
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyResult<PyRef<'_, Self>> {
        if slf.consumed {
            return Err(PyRuntimeError::new_err(
                "FastLoader is one-shot and already exhausted; create a new FastLoader for another epoch",
            ));
        }
        Ok(slf)
    }

    fn __next__(&mut self, py: Python<'_>) -> PyResult<Option<PyNiftiImage>> {
        if self.consumed {
            return Ok(None);
        }
        let loader = LoaderRef(&self.loader);
        let next = py.allow_threads(move || loader.next());
        match next {
            Some(Ok(img)) => Ok(Some(PyNiftiImage { inner: img })),
            Some(Err(e)) => Err(pyo3::exceptions::PyIOError::new_err(format!("{}", e))),
            None => {
                self.consumed = true;
                Ok(None)
            }
        }
    }

    #[getter]
    fn patch_shape(&self) -> [usize; 3] {
        self.loader.patch_shape()
    }
}
