
Changelog
=========

See ``CHANGELOG.md`` in the repository root for the full entry-by-entry history; this page summarizes the same releases.

Version 0.2.0
-------------
- Added optional ``.jvol`` volumetric compression (wavelet + Rice coding), vendored from `jvol-rust <https://github.com/fepegar/jvol-rust>`_ by Fernando Pérez-García (MIT licensed). See :doc:`guides/compression`.
- Fixed ``resample`` and ``reorient`` affine handling so world coordinates are preserved; resampling uses half-pixel-center sampling and records the achieved spacing.
- Z-normalization is two-pass and errors on non-finite statistics; ``clamp`` errors when ``min > max``. Several transforms now require exactly 3D input and return an error instead of panicking.
- Removed the internal memory pool and dead fusion code; transform pipeline fusion covers consecutive axis-aligned resamples and trailing intensity operations.
- SIMD kernels use portable ``wide::f32x8`` (SSE2 baseline, AVX2 with ``-C target-feature=+avx2`` / ``make build-native``).
- Python: GIL released around heavy operations; ``load_cached``, ``clear_decompression_cache``, ``set_cache_size``, ``load_multi``, ``load_image_label_pair`` now exported; ``__version__`` resolves from package metadata.
- Fixed potential panic when patch size exceeds volume dimensions in ``CropLoader`` and ``TrainingDataLoader``.
- Added dimension overflow validation in ``crop_or_pad`` and ``rotate_90`` transforms.
- Added regression tests for boundary condition handling.

Version 0.1.1
-------------
- Updated to F-order array handling throughout for NIfTI compatibility.
- Various bug fixes and performance improvements.

Version 0.1.0
-------------
- Initial public release.
- Rust NIfTI I/O with crop-first loading and save support.
- Python bindings for loading, transforms, and patch-based training with ``TrainingDataLoader``.
- Removed deprecated ``PyTrainingDataLoader`` alias; use ``TrainingDataLoader``.
- Dictionary transform helpers for multi-modal datasets.
- Performance profiling utilities.
