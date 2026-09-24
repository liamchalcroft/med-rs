//! The `FastLoader` class.

use crate::image::PyNiftiImage;
use crate::pipeline::PyPipeline;
use crate::OrRaise;
use medrs::loader::{Epoch, FastLoader, LoaderConfig};
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, PoisonError};

/// Streams fixed-size training patches from a list of volumes using worker
/// threads.
///
/// Patches are read crop-first (only the patch is read from uncompressed
/// files, and only the chunks it touches from `.jvol` files), padded to
/// `patch_shape` when a volume is smaller, and transformed by `pipeline` in
/// the workers. For a given seed the sequence of patches in an epoch does not
/// depend on the number of workers.
///
/// Iterating the loader runs the next epoch; `loader.epoch(n)` runs epoch `n`
/// explicitly.
#[pyclass(name = "FastLoader", module = "medrs", frozen)]
struct PyFastLoader {
    inner: FastLoader,
    next_epoch: AtomicU64,
}

#[pymethods]
impl PyFastLoader {
    #[new]
    #[pyo3(signature = (
        images,
        patch_shape,
        *,
        labels = None,
        patches_per_volume = 1,
        foreground_prob = None,
        pad_value = 0.0,
        pipeline = None,
        workers = None,
        prefetch = None,
        shuffle = true,
        seed = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        images: Vec<PathBuf>,
        patch_shape: [usize; 3],
        labels: Option<Vec<PathBuf>>,
        patches_per_volume: usize,
        foreground_prob: Option<f64>,
        pad_value: f64,
        pipeline: Option<&PyPipeline>,
        workers: Option<usize>,
        prefetch: Option<usize>,
        shuffle: bool,
        seed: Option<u64>,
    ) -> PyResult<Self> {
        let mut config = LoaderConfig::new(patch_shape);
        config.patches_per_volume = patches_per_volume;
        config.foreground_prob = foreground_prob;
        config.pad_value = pad_value;
        config.pipeline = pipeline.map(|p| p.inner.clone());
        if let Some(w) = workers {
            config.workers = w;
            config.prefetch = (2 * w).max(2);
        }
        if let Some(p) = prefetch {
            config.prefetch = p;
        }
        config.shuffle = shuffle;
        config.seed = seed;
        Ok(Self {
            inner: FastLoader::new(images, labels, config).or_raise()?,
            next_epoch: AtomicU64::new(0),
        })
    }

    /// Number of patches per epoch.
    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// The seed in use (the one given, or one picked at creation).
    #[getter]
    fn seed(&self) -> u64 {
        self.inner.seed()
    }

    /// Iterate over epoch `epoch`.
    fn epoch(&self, epoch: u64) -> PyEpoch {
        PyEpoch {
            inner: Mutex::new(Some(self.inner.epoch(epoch))),
        }
    }

    fn __iter__(&self) -> PyEpoch {
        self.epoch(self.next_epoch.fetch_add(1, Ordering::Relaxed))
    }

    fn __repr__(&self) -> String {
        let c = self.inner.config();
        format!(
            "FastLoader(patches={}, patch_shape={:?}, workers={}, seed={})",
            self.inner.len(),
            c.patch_shape,
            c.workers,
            self.inner.seed()
        )
    }
}

/// Iterator over the patches of one epoch.
///
/// Call `close()` (or use it as a context manager) to stop its worker threads
/// early.
#[pyclass(name = "Epoch", module = "medrs")]
struct PyEpoch {
    inner: Mutex<Option<Epoch>>,
}

impl PyEpoch {
    fn take(&self) -> Option<Epoch> {
        self.inner
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .take()
    }
}

#[pymethods]
impl PyEpoch {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&self, py: Python<'_>) -> PyResult<Option<PyPatch>> {
        let next = py.detach(|| {
            let mut guard = self.inner.lock().unwrap_or_else(PoisonError::into_inner);
            guard.as_mut().and_then(Iterator::next)
        });
        let Some(patch) = next else {
            return Ok(None);
        };
        let patch = patch.or_raise()?;
        Ok(Some(PyPatch {
            image: Py::new(py, PyNiftiImage::from(patch.image))?,
            label: patch
                .label
                .map(|l| Py::new(py, PyNiftiImage::from(l)))
                .transpose()?,
            volume: patch.volume,
            offset: patch.region.offset.into(),
            shape: patch.region.shape.into(),
        }))
    }

    /// Patches not yet returned.
    fn __length_hint__(&self) -> usize {
        self.inner
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .as_ref()
            .map_or(0, Epoch::remaining)
    }

    /// Stop the worker threads. Iteration ends.
    fn close(&self, py: Python<'_>) {
        let epoch = self.take();
        py.detach(|| drop(epoch));
    }

    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    #[pyo3(signature = (*_args))]
    fn __exit__(&self, py: Python<'_>, _args: &Bound<'_, PyTuple>) {
        self.close(py);
    }
}

impl Drop for PyEpoch {
    fn drop(&mut self) {
        // Joining workers can take as long as one volume load: do it without
        // blocking other Python threads.
        if let Some(epoch) = self
            .inner
            .get_mut()
            .unwrap_or_else(PoisonError::into_inner)
            .take()
        {
            Python::attach(|py| py.detach(|| drop(epoch)));
        }
    }
}

/// One training patch.
#[pyclass(name = "Patch", module = "medrs", frozen)]
struct PyPatch {
    /// The image patch.
    #[pyo3(get)]
    image: Py<PyNiftiImage>,
    /// The label patch, or `None` without labels.
    #[pyo3(get)]
    label: Option<Py<PyNiftiImage>>,
    /// Index of the source volume.
    #[pyo3(get)]
    volume: usize,
    /// First voxel of the region read from the source volume.
    #[pyo3(get)]
    offset: (usize, usize, usize),
    /// Shape of the region read (smaller than the patch where padded).
    #[pyo3(get)]
    shape: (usize, usize, usize),
}

#[pymethods]
impl PyPatch {
    fn __repr__(&self) -> String {
        format!(
            "Patch(volume={}, offset={:?}, shape={:?}, label={})",
            self.volume,
            self.offset,
            self.shape,
            self.label.is_some()
        )
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyFastLoader>()?;
    m.add_class::<PyEpoch>()?;
    m.add_class::<PyPatch>()?;
    Ok(())
}
