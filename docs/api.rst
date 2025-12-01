API Reference
=============

This section provides detailed API documentation for all medrs modules and functions.

Python API
----------

Core Functions
~~~~~~~~~~~~~~

.. py:class:: medrs.MedicalImage

   Core medical image class representing volumetric data with metadata.

   .. py:attribute:: shape
      :type: tuple

      Image dimensions (D, H, W) or (C, D, H, W).

   .. py:attribute:: spacing
      :type: tuple

      Voxel spacing in mm.

   .. py:attribute:: affine
      :type: numpy.ndarray

      4x4 affine transformation matrix.

   .. py:method:: to_numpy()

      Convert to numpy array.

   .. py:method:: to_torch(device=None, dtype=None)

      Convert to PyTorch tensor.

   .. py:method:: save(path)

      Save to NIfTI file.

Loading Functions
~~~~~~~~~~~~~~~~~

.. py:function:: medrs.load(path)

   Load a NIfTI file.

   :param path: Path to NIfTI file (.nii or .nii.gz)
   :type path: str
   :returns: Loaded medical image
   :rtype: MedicalImage

.. py:function:: medrs.load_cropped(path, crop_offset, crop_shape)

   Load only a cropped region from a NIfTI file.

   :param path: Path to NIfTI file
   :param crop_offset: Starting coordinates [x, y, z]
   :param crop_shape: Size of crop region [x, y, z]
   :returns: Cropped medical image

.. py:function:: medrs.load_to_torch(path, dtype=None, device="cpu")

   Load NIfTI directly to PyTorch tensor.

   :param path: Path to NIfTI file
   :param dtype: PyTorch dtype (default: float32)
   :param device: Target device
   :returns: PyTorch tensor

Transform Functions
~~~~~~~~~~~~~~~~~~~

.. py:function:: medrs.z_normalization(image)

   Z-score normalize an image (zero mean, unit variance).

.. py:function:: medrs.rescale_intensity(image, output_range=(0.0, 1.0))

   Rescale intensity to the provided range.

.. py:function:: medrs.clamp(image, min_value, max_value)

   Clamp intensity values to a range.

.. py:function:: medrs.resample(image, target_spacing, method=None)

   Resample to target voxel spacing.

.. py:function:: medrs.reorient(image, orientation)

   Reorient to target orientation (e.g., "RAS", "LPS").

.. py:function:: medrs.crop_or_pad(image, target_shape)

   Crop or pad to target shape.

Random Augmentation
~~~~~~~~~~~~~~~~~~~

.. py:function:: medrs.random_flip(image, axes, prob=0.5, seed=None)

   Random axis flipping.

.. py:function:: medrs.random_gaussian_noise(image, std=0.1, seed=None)

   Add random Gaussian noise.

.. py:function:: medrs.random_intensity_scale(image, scale_range=0.1, seed=None)

   Random intensity scaling.

.. py:function:: medrs.random_intensity_shift(image, shift_range=0.1, seed=None)

   Random intensity shift.

.. py:function:: medrs.random_rotate_90(image, axes, seed=None)

   Random 90-degree rotation.

.. py:function:: medrs.random_gamma(image, gamma_range=(0.7, 1.5), seed=None)

   Random gamma correction.

.. py:function:: medrs.random_augment(image, seed=None)

   Combined random augmentation pipeline.

Transform Pipeline
~~~~~~~~~~~~~~~~~~

.. py:class:: medrs.TransformPipeline(lazy=True)

   Composable transform pipeline with lazy evaluation.

   .. py:method:: z_normalize()

      Add z-score normalization.

   .. py:method:: clamp(min, max)

      Add intensity clamping.

   .. py:method:: resample_to_spacing(spacing)

      Add resampling to target spacing.

   .. py:method:: resample_to_shape(shape)

      Add resampling to target shape.

   .. py:method:: apply(image)

      Apply pipeline to an image.

Training Pipeline
~~~~~~~~~~~~~~~~~

.. py:class:: medrs.TrainingDataLoader(volumes, patch_size, patches_per_volume, patch_overlap, randomize, cache_size=None)

   High-performance training data loader with prefetching and caching.

   .. py:method:: next_patch()

      Get next training patch.

   .. py:method:: reset()

      Reset loader to beginning.

Rust API
--------

For Rust API documentation, see the generated docs at `docs.rs/medrs <https://docs.rs/medrs>`_.

Core modules:

- ``medrs::nifti`` - NIfTI I/O operations
- ``medrs::transforms`` - Image transformations
- ``medrs::pipeline`` - Transform pipelines

Example usage:

.. code-block:: rust

   use medrs::nifti;
   use medrs::transforms::z_normalization;

   fn main() -> medrs::Result<()> {
       let img = nifti::load("brain.nii.gz")?;
       let normalized = z_normalization(&img);
       nifti::save(&normalized, "output.nii.gz")?;
       Ok(())
   }
