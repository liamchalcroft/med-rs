# Changelog

## 0.2.0 (2026-07-02)

- Added optional `.jvol` volumetric compression (wavelet + Rice coding), vendored from
  [jvol-rust](https://github.com/fepegar/jvol-rust) by Fernando Pérez-García (MIT licensed). Enabled by the
  `jvol` Cargo feature (implied by `python`); exposed in Python as `medrs.save_jvol` /
  `medrs.convert_to_jvol`, and transparently through `medrs.load()` for `.jvol` paths. Lossy encoding is
  rejected for integer/label dtypes.
- Fixed `resample` and `reorient` affine handling so world coordinates are preserved; resampling now uses
  half-pixel-center sampling (matching SimpleITK/MONAI conventions) and records the achieved spacing with a
  compensating origin shift when the requested spacing isn't exactly representable.
- Z-normalization is now two-pass and returns an error on non-finite statistics instead of silently
  producing NaNs; `clamp` returns an error when `min > max`. Several transforms now require exactly 3D
  input and return a `Result` instead of panicking or silently corrupting output.
- Removed the internal memory pool and dead fusion code path (`ops.rs`); the transform pipeline's
  "automatic fusion" is real for the cases it actually implements: consecutive axis-aligned resamples, and
  trailing intensity operations (z-normalize, scaling, clamping).
- SIMD kernels moved to portable `wide::f32x8`: two SSE registers on the x86-64 baseline, a single AVX2
  register only when built with `-C target-feature=+avx2` / `-C target-cpu=native` (see `make build-native`).
- Python: the GIL is now released around heavy Rust-side operations. `FastLoader` is documented as
  one-shot per epoch. `load_cached`, `clear_decompression_cache`, `set_cache_size`, `load_multi`, and
  `load_image_label_pair` are now exported from the top-level `medrs` package. `__version__` now resolves
  from installed package metadata, falling back to the hardcoded version string if metadata is unavailable.
- Fixed potential panic when patch size exceeds volume dimensions in `CropLoader` and `TrainingDataLoader`.
- Added dimension overflow validation in `crop_or_pad` and `rotate_90` transforms.
- Added regression tests for boundary condition handling and for the corrected resample/reorient affine math.

## 0.1.1

- Updated to F-order array handling throughout for NIfTI compatibility.
- Various bug fixes and performance improvements.

## 0.1.0

- Initial public release.
- Rust NIfTI I/O with crop-first loading and save support.
- Python bindings for loading, transforms, and patch-based training with `TrainingDataLoader`.
- Removed deprecated `PyTrainingDataLoader` alias; use `TrainingDataLoader`.
- Dictionary transform helpers for multi-modal datasets.
- Performance profiling utilities.
