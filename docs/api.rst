API Reference
=============

This section provides detailed API documentation for all medrs modules and functions.

Python API
----------

Core Functions
~~~~~~~~~~~~~~

.. autoclass:: medrs.MedicalImage
   :members:
   :undoc-members:
   :show-inheritance:

Loading Functions
~~~~~~~~~~~~~~~~~

.. autofunction:: medrs.load
.. autofunction:: medrs.load_cropped
.. autofunction:: medrs.load_cropped_to_torch
.. autofunction:: medrs.load_cropped_to_jax
.. autofunction:: medrs.load_resampled

Transform Functions
~~~~~~~~~~~~~~~~~~~

.. autofunction:: medrs.z_normalization
.. autofunction:: medrs.rescale_intensity
.. autofunction:: medrs.clamp
.. autofunction:: medrs.resample
.. autofunction:: medrs.reorient
.. autofunction:: medrs.crop_or_pad

Random Augmentation Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: medrs.random_flip
.. autofunction:: medrs.random_gaussian_noise
.. autofunction:: medrs.random_intensity_scale
.. autofunction:: medrs.random_intensity_shift
.. autofunction:: medrs.random_rotate_90
.. autofunction:: medrs.random_gamma
.. autofunction:: medrs.random_augment

Transform Pipeline
~~~~~~~~~~~~~~~~~~

.. autoclass:: medrs.TransformPipeline
   :members:
   :undoc-members:

Training Pipeline
~~~~~~~~~~~~~~~~~

.. autoclass:: medrs.PyTrainingDataLoader
   :members:
   :undoc-members:

Exception Hierarchy
~~~~~~~~~~~~~~~~~~~

.. automodule:: medrs.exceptions
   :members:
   :undoc-members:
   :show-inheritance:

Performance Profiling
~~~~~~~~~~~~~~~~~~~~~

.. automodule:: medrs.performance_profiler
   :members:
   :undoc-members:

Rust API
--------

Core Types
~~~~~~~~~~

Medical Image
^^^^^^^^^^^^^

.. rust:type:: medrs::core::MedicalImage<T>

The core medical image type that represents volumetric medical data with associated metadata.

**Fields**

- ``data``: ``Array3<T>`` - The volumetric data
- ``spacing``: ``[f64; 3]`` - Voxel spacing in mm [x, y, z]
- ``orientation``: ``Matrix3<f64>`` - Orientation matrix
- ``origin``: ``[f64; 3]`` - World coordinates origin

**Examples**

```rust
use medrs::nifti;
use medrs::transforms::z_normalization;

// Load a NIfTI file
let img = nifti::load("brain.nii.gz")?;

// Apply transforms
let normalized = z_normalization(&img);

println!("Image shape: {:?}", normalized.data.shape());
println!("Spacing: {:?}", normalized.spacing);
```

NIfTI Module
^^^^^^^^^^^^

.. rust:mod:: medrs::nifti

High-performance NIfTI I/O operations with byte-exact loading and crop-first optimization.

**Key Functions**

- ``nifti::load(path: &str) -> Result<MedicalImage<f32>>`` - Load NIfTI file
- ``nifti::save(img: &MedicalImage<T>, path: &str) -> Result<()>`` - Save NIfTI file
- ``nifti::load_cropped(path: &str, offset: [usize; 3], shape: [usize; 3]) -> Result<MedicalImage<T>>`` - Crop-first loading

Transforms Module
^^^^^^^^^^^^^^^^^

.. rust:mod:: medrs::transforms

High-performance image transformations with SIMD optimization.

**Key Functions**

- ``z_normalization(img: &MedicalImage<T>) -> MedicalImage<T>`` - Z-score normalization
- ``rescale_intensity(img: &MedicalImage<T>, min: T, max: T) -> MedicalImage<T>`` - Intensity rescaling
- ``resample_to_spacing(img: &MedicalImage<T>, spacing: [f64; 3], interp: Interpolation) -> MedicalImage<T>`` - Resampling

Pipeline Module
^^^^^^^^^^^^^^^

.. rust:mod:: medrs::pipeline

Memory-optimized data pipelines for training workflows.

**Key Types**

- ``TrainingDataLoader`` - High-throughput data loader with caching
- ``PatchSampler`` - Configurable patch sampling strategies

Performance Considerations
--------------------------

Memory Efficiency
~~~~~~~~~~~~~~~~~

medrs is designed for optimal memory usage:

- **Crop-first loading**: Only load the bytes you actually need (40x memory reduction)
- **Zero-copy operations**: Direct tensor creation without intermediate copies
- **Memory mapping**: Efficient access to large uncompressed files

Thread Safety
~~~~~~~~~~~~~

Most operations are thread-safe and designed for parallel processing:

- **SIMD optimization**: Vectorized operations for hot paths
- **Rayon integration**: Easy parallelization of batch operations
- **Lock-free data structures**: Minimized contention in multi-threaded scenarios

GPU Integration
~~~~~~~~~~~~~~~

Direct GPU memory operations:

```python
# Direct to GPU loading
tensor = medrs.load_cropped_to_torch(
    "volume.nii.gz",
    output_shape=[64, 64, 64],
    device="cuda",
    dtype=torch.float16
)
```

Error Handling
--------------

medrs uses a structured error hierarchy for actionable error messages:

.. code-block:: python

   try:
       img = medrs.load("invalid_file.nii.gz")
   except medrs.exceptions.FileNotFoundError as e:
       print(f"File not found: {e}")
       print("Suggestion:", e.suggestions[0])

For complete error handling documentation, see :doc:`../guides/error_handling`.

Type Safety
-----------

medrs uses Rust's type system and Python type hints for compile-time safety:

```rust
// Compile-time guarantee of valid operations
fn process_image<T: Float + Send + Sync>(img: &MedicalImage<T>) -> MedicalImage<T> {
    // Type-safe operations
}
```

```python
# Runtime type checking with Python hints
def load_cropped(
    path: str,
    output_shape: Sequence[int],
    dtype: Optional[torch.dtype] = None
) -> torch.Tensor:
    # Type-annotated implementation
```
