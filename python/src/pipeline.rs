//! The `Pipeline` class.

use crate::image::{parse_interpolation, parse_orientation, PyNiftiImage};
use crate::{parse_dtype, OrRaise};
use medrs::Pipeline;
use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

/// An ordered list of transforms applied with fused passes over the data.
///
/// Pipelines are immutable: each method returns a new pipeline with one more
/// step. Intensity steps are applied in a single pass and consecutive
/// spatial steps interpolate once. Random steps draw their parameters on each
/// `apply`; pass `seed` for reproducible results.
#[pyclass(name = "Pipeline", module = "medrs", frozen)]
pub(crate) struct PyPipeline {
    pub(crate) inner: Pipeline,
    /// `(method, args)` for each step, to rebuild the pipeline when unpickling.
    steps: Vec<(String, Py<PyTuple>)>,
}

impl PyPipeline {
    fn with(
        &self,
        name: &str,
        args: Bound<'_, PyTuple>,
        step: impl FnOnce(Pipeline) -> Pipeline,
    ) -> PyResult<Self> {
        let inner = step(self.inner.clone());
        inner.validate().or_raise()?;
        let py = args.py();
        let mut steps: Vec<(String, Py<PyTuple>)> = self
            .steps
            .iter()
            .map(|(n, a)| (n.clone(), a.clone_ref(py)))
            .collect();
        steps.push((name.to_string(), args.unbind()));
        Ok(Self { inner, steps })
    }
}

#[pymethods]
impl PyPipeline {
    #[new]
    fn new() -> Self {
        Self {
            inner: Pipeline::new(),
            steps: Vec::new(),
        }
    }

    /// Permute and flip axes to match `orientation` (such as `"RAS"`).
    fn reorient(&self, py: Python<'_>, orientation: &str) -> PyResult<Self> {
        let target = parse_orientation(orientation)?;
        self.with("reorient", PyTuple::new(py, [orientation])?, |p| {
            p.reorient(target)
        })
    }

    /// Resample to new voxel sizes.
    #[pyo3(signature = (spacing, interpolation = "trilinear"))]
    fn resample_to_spacing(
        &self,
        py: Python<'_>,
        spacing: [f64; 3],
        interpolation: &str,
    ) -> PyResult<Self> {
        let interp = parse_interpolation(interpolation)?;
        let args = (spacing, interpolation).into_pyobject(py)?;
        self.with("resample_to_spacing", args, |p| {
            p.resample_to_spacing(spacing, interp)
        })
    }

    /// Resample to a new grid shape.
    #[pyo3(signature = (shape, interpolation = "trilinear"))]
    fn resample_to_shape(
        &self,
        py: Python<'_>,
        shape: [usize; 3],
        interpolation: &str,
    ) -> PyResult<Self> {
        let interp = parse_interpolation(interpolation)?;
        let args = (shape, interpolation).into_pyobject(py)?;
        self.with("resample_to_shape", args, |p| {
            p.resample_to_shape(shape, interp)
        })
    }

    /// Crop `shape` voxels starting at `offset`.
    fn crop(&self, py: Python<'_>, offset: [usize; 3], shape: [usize; 3]) -> PyResult<Self> {
        let args = (offset, shape).into_pyobject(py)?;
        self.with("crop", args, |p| p.crop(offset, shape))
    }

    /// Centre-crop or pad to `shape`.
    #[pyo3(signature = (shape, pad_value = 0.0))]
    fn crop_or_pad(&self, py: Python<'_>, shape: [usize; 3], pad_value: f64) -> PyResult<Self> {
        let args = (shape, pad_value).into_pyobject(py)?;
        self.with("crop_or_pad", args, |p| p.crop_or_pad(shape, pad_value))
    }

    /// Reverse the voxel order along `axes`.
    fn flip(&self, py: Python<'_>, axes: Vec<usize>) -> PyResult<Self> {
        let args = (axes.clone(),).into_pyobject(py)?;
        self.with("flip", args, |p| p.flip(&axes))
    }

    /// Rotate by `k` × 90 degrees in the plane of `axes`.
    #[pyo3(signature = (axes = (0, 1), k = 1), text_signature = "($self, axes=(0, 1), k=1)")]
    fn rotate_90(&self, py: Python<'_>, axes: (usize, usize), k: i32) -> PyResult<Self> {
        let args = (axes, k).into_pyobject(py)?;
        self.with("rotate_90", args, |p| p.rotate_90(axes, k))
    }

    /// Clamp values to `[min, max]`.
    fn clamp(&self, py: Python<'_>, min: f64, max: f64) -> PyResult<Self> {
        let args = (min, max).into_pyobject(py)?;
        self.with("clamp", args, |p| p.clamp(min, max))
    }

    /// Normalize to zero mean and unit variance (over non-zero voxels only,
    /// keeping zeros, if `nonzero`).
    #[pyo3(signature = (nonzero = false))]
    fn z_normalize(&self, py: Python<'_>, nonzero: bool) -> PyResult<Self> {
        let args = (nonzero,).into_pyobject(py)?;
        self.with("z_normalize", args, |p| {
            if nonzero {
                p.z_normalize_nonzero()
            } else {
                p.z_normalize()
            }
        })
    }

    /// Linearly map the value range onto `[out_min, out_max]`.
    #[pyo3(signature = (out_min = 0.0, out_max = 1.0))]
    fn rescale(&self, py: Python<'_>, out_min: f64, out_max: f64) -> PyResult<Self> {
        let args = (out_min, out_max).into_pyobject(py)?;
        self.with("rescale", args, |p| p.rescale(out_min, out_max))
    }

    /// Clip to the `lower` and `upper` percentiles (of the non-zero voxels,
    /// if `nonzero`) and map that range linearly onto `[out_min, out_max]`.
    #[pyo3(signature = (lower, upper, nonzero = false, out_min = 0.0, out_max = 1.0))]
    fn rescale_percentiles(
        &self,
        py: Python<'_>,
        lower: f64,
        upper: f64,
        nonzero: bool,
        out_min: f64,
        out_max: f64,
    ) -> PyResult<Self> {
        let args = (lower, upper, nonzero, out_min, out_max).into_pyobject(py)?;
        self.with("rescale_percentiles", args, |p| {
            p.rescale_percentiles(lower, upper, nonzero, out_min, out_max)
        })
    }

    /// Range-preserving gamma adjustment.
    fn adjust_gamma(&self, py: Python<'_>, gamma: f64) -> PyResult<Self> {
        let args = (gamma,).into_pyobject(py)?;
        self.with("adjust_gamma", args, |p| p.adjust_gamma(gamma))
    }

    /// Convert to another stored datatype (for example `"float16"`).
    fn cast(&self, py: Python<'_>, dtype: &Bound<'_, PyAny>) -> PyResult<Self> {
        let parsed = parse_dtype(dtype)?;
        let args = (parsed.name(),).into_pyobject(py)?;
        self.with("cast", args, |p| p.cast(parsed))
    }

    /// Flip each of `axes` with probability `prob`.
    #[pyo3(
        signature = (axes = vec![0, 1, 2], prob = 0.5),
        text_signature = "($self, axes=(0, 1, 2), prob=0.5)"
    )]
    fn random_flip(&self, py: Python<'_>, axes: Vec<usize>, prob: f64) -> PyResult<Self> {
        let args = (axes.clone(), prob).into_pyobject(py)?;
        self.with("random_flip", args, |p| p.random_flip(&axes, prob))
    }

    /// Rotate by a random multiple of 90 degrees in the plane of `axes`.
    #[pyo3(signature = (axes = (0, 1)), text_signature = "($self, axes=(0, 1))")]
    fn random_rotate_90(&self, py: Python<'_>, axes: (usize, usize)) -> PyResult<Self> {
        let args = (axes,).into_pyobject(py)?;
        self.with("random_rotate_90", args, |p| p.random_rotate_90(axes))
    }

    /// Multiply intensities by `1 + f` with `f` uniform in `[-range, range]`.
    #[pyo3(signature = (range = 0.1))]
    fn random_intensity_scale(&self, py: Python<'_>, range: f64) -> PyResult<Self> {
        let args = (range,).into_pyobject(py)?;
        self.with("random_intensity_scale", args, |p| {
            p.random_intensity_scale(range)
        })
    }

    /// Add an offset uniform in `[-range, range]`.
    #[pyo3(signature = (range = 0.1))]
    fn random_intensity_shift(&self, py: Python<'_>, range: f64) -> PyResult<Self> {
        let args = (range,).into_pyobject(py)?;
        self.with("random_intensity_shift", args, |p| {
            p.random_intensity_shift(range)
        })
    }

    /// Add zero-mean Gaussian noise with standard deviation `std`.
    #[pyo3(signature = (std = 0.1))]
    fn random_gaussian_noise(&self, py: Python<'_>, std: f64) -> PyResult<Self> {
        let args = (std,).into_pyobject(py)?;
        self.with("random_gaussian_noise", args, |p| {
            p.random_gaussian_noise(std)
        })
    }

    /// Range-preserving gamma adjustment with gamma uniform in `range`.
    #[pyo3(signature = (range = (0.7, 1.5)), text_signature = "($self, range=(0.7, 1.5))")]
    fn random_gamma(&self, py: Python<'_>, range: (f64, f64)) -> PyResult<Self> {
        let args = (range,).into_pyobject(py)?;
        self.with("random_gamma", args, |p| p.random_gamma(range))
    }

    /// Apply the pipeline. With `label`, spatial steps are applied to both
    /// (nearest-neighbour for the label) with the same random draws, and
    /// `(image, label)` is returned.
    #[pyo3(signature = (image, label = None, *, seed = None))]
    fn apply<'py>(
        &self,
        py: Python<'py>,
        image: &PyNiftiImage,
        label: Option<&PyNiftiImage>,
        seed: Option<u64>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let mut rng = seed.map_or_else(rand::make_rng::<ChaCha8Rng>, ChaCha8Rng::seed_from_u64);
        let pipeline = &self.inner;
        match label {
            None => {
                let out = py
                    .detach(|| pipeline.apply_with_rng(&image.inner, &mut rng))
                    .or_raise()?;
                Ok(Bound::new(py, PyNiftiImage::from(out))?.into_any())
            }
            Some(label) => {
                let (a, b) = py
                    .detach(|| pipeline.apply_pair(&image.inner, &label.inner, &mut rng))
                    .or_raise()?;
                Ok((PyNiftiImage::from(a), PyNiftiImage::from(b))
                    .into_pyobject(py)?
                    .into_any())
            }
        }
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        let steps: Vec<String> = self
            .steps
            .iter()
            .map(|(name, args)| Ok(format!("{name}{}", args.bind(py).repr()?)))
            .collect::<PyResult<_>>()?;
        Ok(format!("Pipeline([{}])", steps.join(", ")))
    }

    fn __reduce__<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(Bound<'py, PyAny>, (Bound<'py, PyList>,))> {
        let steps = PyList::empty(py);
        for (name, args) in &self.steps {
            steps.append((name, args.bind(py)))?;
        }
        let constructor = py.import("medrs._medrs")?.getattr("_pipeline_from_steps")?;
        Ok((constructor, (steps,)))
    }
}

/// Rebuild a pipeline from its steps (used by pickle).
#[pyfunction]
fn _pipeline_from_steps<'py>(
    py: Python<'py>,
    steps: Vec<(String, Bound<'py, PyTuple>)>,
) -> PyResult<Bound<'py, PyAny>> {
    let mut pipeline = Bound::new(py, PyPipeline::new())?.into_any();
    for (name, args) in steps {
        pipeline = pipeline.call_method1(name.as_str(), args)?;
    }
    Ok(pipeline)
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyPipeline>()?;
    m.add_function(wrap_pyfunction!(_pipeline_from_steps, m)?)?;
    Ok(())
}
