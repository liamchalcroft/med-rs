# Changelog

## 0.3.0 (2026-09-25)

A rewrite focused on correctness, with a smaller, consistent API. The previous
release silently produced wrong results in several common cases (listed
under Fixes), so upgrading is strongly recommended.

### Breaking changes and migration

| 0.2 | 0.3 |
|---|---|
| `medrs.save_jvol(img, path, quality=q)` | `img.save("x.jvol", quality=q)` (lossless without `quality`) |
| `medrs.convert_to_mgzip(path)`, `medrs.load_mgzip` | `medrs.load(path).save(out, mgzip=True)`; `load` detects Mgzip itself. CLI: `medrs convert --to mgzip` |
| `TrainingDataLoader`, `CropLoader`, `BatchLoader` | `FastLoader(..., patches_per_volume=n, workers=0)` |
| `FastLoader(...).build()` / one-shot iteration | `FastLoader(images, patch_shape, ...)`; each iteration is a new epoch and yields `Patch` objects |
| `load_image_label_pair`, `load_multi(configs)` | `load_multi(paths, reference=0, interpolation=[...])`, which resamples onto the reference grid |
| `load_label_aware_cropped`, `compute_*_crop*` | `FastLoader(labels=..., foreground_prob=p)`; Rust: `transforms::sample_label_regions` |
| `TransformPipeline` (mutable) | `Pipeline` (immutable); `apply(image, label=None, seed=None)` |
| `random_*` functions and `random_augment` | Pipeline steps (`Pipeline().random_flip(...)`) |
| `load_to_torch(path)`, `load_cropped_to_torch`, ... | `medrs.load(path).to_torch()`, `medrs.load_cropped(...).to_torch()` |
| `img.data`, `to_numpy_native()` | `img.to_numpy()` (one dtype rule, see the guide) |
| `img.affine` as nested lists | `img.affine` as a float64 NumPy array; set with `img.with_affine(a)` |
| `medrs.monai_compat`, `metatensor_support`, `dictionary_transforms` | `medrs.monai.MedrsReader` with MONAI's own `LoadImage(d)` |
| `medrs.exceptions` classes | `medrs.FormatError` (a `ValueError`) and built-in exceptions |
| `flip` left the affine unchanged | All spatial transforms update the affine (world-preserving) |
| `.jvol` files written by 0.2 | Not readable (loading one says it was written by 0.2). Re-encode from the source images, or decode with medrs 0.2 and save with 0.3: losslessly to keep the decoded values exactly (about 12 times larger for lossy files), or lossily. 0.3's `quality` 60, 70, and 80 give about the size and error of 0.2's 70, 80, and 90 |
| Rust: `medrs::pipeline::compose::TransformPipeline`, `LazyImage`, `simd_kernels` | `medrs::Pipeline` |
| Rust: `python` Cargo feature | The bindings are a separate crate built by maturin |
| Rust: ndarray 0.16 in the API; Rust 1.81 or newer | ndarray 0.17 (for example `NiftiImage::to_f32`) and rand 0.10 (`Pipeline::apply_with_rng`, `transforms::random_*`); Rust 1.85 or newer |
| Alternative spellings such as `"f32"`, `"linear"`, `"ras"` | One name each: NumPy dtype names, `"nearest"`/`"trilinear"`, upper-case orientation codes |

### New

- Full NIfTI header round trips: extensions, intent, slice timing, units,
  calibration, and both transform codes. NIfTI-2, big-endian files, and
  `.hdr`/`.img` pairs are read and written.
- Parallel decompression of Mgzip/BGZF files in `load`; gzip is detected from
  the file contents.
- New `.jvol` container: chunked, checksummed, lossless for every datatype,
  crop-first decoding, parallel encode and decode.
- `resample_like`, `resample_to_grid`, `load_multi` onto a reference grid,
  4D support in every transform, `z_normalize(nonzero=True)`, `adjust_gamma`,
  `rotate_90`.
- `Pipeline` fuses intensity steps into one pass and spatial steps into one
  interpolation, and applies spatial randomness identically to image and label.
- `FastLoader`: all formats, crop-first reads, deterministic order for any
  number of workers, label-aware sampling, padding, per-patch pipelines.
  Patch shapes can be drawn per patch from weighted choices, volumes can be
  drawn with weights (with replacement), and `foreground_threshold` centres
  patches on bright voxels without labels.
- `percentiles` and `rescale_percentiles` (also a pipeline step): exact,
  NumPy-compatible percentiles, optionally over non-zero voxels only.
- Python: any numeric array dtype in `NiftiImage(...)`, `numpy.asarray(img)`,
  bfloat16 for PyTorch and JAX, pickling, header dictionaries, type stubs,
  free-threaded CPython support, and a `medrs` command-line tool.
- `medrs.set_num_threads`; parallel work is safe in forked processes.

### Fixes

- Flipped, cropped, padded, and rotated images were stored in the wrong memory
  order, so any later operation or save scrambled the voxels.
- `to_numpy()` returned garbage for int32/uint32 images and byte-swapped
  values for big-endian files.
- In-place operations on tensors from `to_torch()` crashed the interpreter
  (they wrote to read-only memory-mapped files).
- Saving over the file an image was loaded from crashed with SIGBUS; saving to
  a bare file name failed.
- The pipeline reordered and dropped intensity operations and did not update
  the affine when resampling.
- Nearest-neighbour resampling and reorientation converted label maps to
  float32. `clamp` ignored `scl_slope`/`scl_inter`. `reorient` accepted
  invalid codes such as `"RRR"` and could panic.
- `sform_code`, `qform_code`, and `pixdim` were reset by transforms; qforms
  were wrong for 180-degree rotations and left-handed affines.
- Patch sampling used a fixed default seed, had the positive/negative ratio
  inverted, and returned truncated patches at the border.
- `FastLoader` rejected `.nii` files and hung the interpreter when dropped
  mid-epoch; forked `DataLoader` workers hung after the parent used medrs.
- `.jvol` lossless mode was lossy for scaled images and 64-bit integers, lost
  most header fields, and could be crashed or made to exhaust memory by
  crafted files.
- `is_mgzip` missed Mgzip files, and `convert_to_mgzip` wrote files that
  `load` could not read.

### Removed

- The Sphinx documentation. medrs.readthedocs.io now shows the README, the
  user guide, and this changelog, and the Rust API is on docs.rs.
- The `benchmark` and `docs` workflows, and the profiling and compatibility
  helper modules.

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

## 0.1.2 (2025-12-29)

- Documentation and benchmark updates.

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
