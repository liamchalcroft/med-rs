# medrs

High-performance medical imaging I/O and processing library for Rust and Python.

[![Crates.io](https://img.shields.io/crates/v/medrs.svg)](https://crates.io/crates/medrs)
[![PyPI](https://img.shields.io/pypi/v/medrs.svg)](https://pypi.org/project/medrs/)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE)

## Overview

medrs is designed for throughput-critical medical imaging workflows, particularly deep learning pipelines that process large 3D volumes. It provides:

- **Fast NIfTI I/O**: Memory-mapped reading, crop-first loading (read sub-volumes without loading entire files)
- **Transform Pipeline**: Lazy evaluation that fuses consecutive axis-aligned resamples and trailing intensity operations (z-normalize, scaling, clamping) into a single pass, with portable SIMD acceleration
- **Mixed Precision**: Native f16/bf16 support for 50% storage savings
- **Volumetric Compression**: Optional `.jvol` codec (vendored from [jvol-rust](https://github.com/fepegar/jvol-rust)) for lossy wavelet compression of floating-point volumes
- **Random Augmentation**: Reproducible, GPU-friendly augmentations for ML training
- **Python Bindings**: Zero-copy numpy views, direct PyTorch/JAX tensor creation, GIL released around heavy operations
- **MONAI Integration**: Drop-in replacements for MONAI transforms

## Why medrs?

medrs memory-maps uncompressed `.nii` files, so opening a volume and materializing it to a numpy array is comparable to nibabel (which also memory-maps) and roughly 10-25x faster than MONAI and TorchIO, whose default loaders eagerly read the data and build a tensor. For gzipped `.nii.gz`, load time is bounded by the decompressor; medrs is competitive with nibabel and MONAI, and the parallel Mgzip format is the fast path for compressed data. All numbers below are reproducible with the scripts in `benchmarks/` (see "Reproducing benchmarks").

### Single File Loading

Load plus full materialization to a numpy array (median of repeated runs, so every library does the same work). Multiples are relative to medrs.

| Volume | Format | medrs | nibabel | MONAI | TorchIO | SimpleITK |
|--------|--------|-------|---------|-------|---------|-----------|
| 128³ | .nii | 1.3 ms | 1.4x | 14x | 9x | 4x |
| 256³ | .nii | 17 ms | 1.4x | 23x | 13x | 4x |
| 128³ | .nii.gz | 82 ms | 0.6x | 1.2x | 0.2x | 0.3x |
| 256³ | .nii.gz | 370 ms | 1.8x | 2.5x | 0.5x | 0.1x |

*Measured on Apple Silicon with `benchmarks/bench_load_comparison.py` (float32, structured data). For uncompressed volumes medrs and nibabel both memory-map, so they are close; the large multiples are against loaders that materialize eagerly. For gzipped volumes SimpleITK and TorchIO decompress faster than medrs's single-threaded path, which is why medrs also offers the parallel Mgzip format.*

### Compressed Storage

Optional lossy `.jvol` (wavelet codec, see the Compression Formats section) gives large storage and bandwidth savings on floating-point volumes. Measured on a 256³ float32 volume:

| Format | Size | vs raw | Decode |
|--------|------|--------|--------|
| raw .nii | 67 MB | 1x | fastest (mmap) |
| .nii.gz | 61 MB | 1.1x | baseline |
| jvol q80 | 3.1 MB | 22x smaller | about gzip speed |
| jvol q60 | 0.1-3 MB | 20-600x smaller | about gzip speed |

jvol decode is roughly gzip speed, so its benefit is storage and bandwidth, not decode CPU. Mixed-precision bf16/f16 saves a fixed 50% of f32 with no codec.

### Training Throughput (FastLoader)

The FastLoader prefetches patches across parallel worker threads with the GIL released. On 64³ random crops from gzipped volumes it delivers roughly 4x the throughput of a naive sequential load-and-crop loop (measured: 191 vs 49 patches/sec, 4 workers). Run your own with `python benchmarks/bench_fastloader.py`.

### Key Advantages

1. **Crop-First Loading**: Read only the cropped region from disk instead of loading the full volume, which matters most as volume size grows relative to patch size
2. **FastLoader**: Parallel patch prefetching with the GIL released, roughly 4x a naive sequential loader
3. **Mixed Precision**: Save in bf16/f16 for 50% smaller files, or lossy `.jvol` for 20-600x smaller
4. **MONAI Drop-in**: Replace MONAI I/O transforms with one import change
5. **Zero-Copy**: Direct tensor creation without intermediate numpy allocations

### Reproducing benchmarks

The numbers above are illustrative snapshots; `benchmarks/results/` is not checked into the repo (it's regenerated, not reproducible from git history alone). Regenerate them on your own hardware with:

```bash
pip install -e ".[examples]"

# Individual suites
python benchmarks/bench_medrs.py
python benchmarks/bench_nibabel.py
python benchmarks/bench_monai.py
python benchmarks/bench_torchio.py
python benchmarks/bench_mgzip.py
python benchmarks/bench_fastloader.py

# Or via cargo for the Rust-side microbenchmarks
cargo bench

# Plots and a combined comparison report
python benchmarks/compare_all.py
python benchmarks/plot_results.py
```

See `benchmarks/BENCHMARK_PLAN.md` for the full benchmark matrix (libraries, volume sizes, dtypes, formats).

## Installation

### Python

```bash
pip install medrs
```

### Rust

```toml
[dependencies]
medrs = "0.1"
```

### Development

```bash
git clone https://github.com/liamchalcroft/med-rs.git
cd med-rs
pip install -e ".[dev]"
maturin develop --features python
```

## Quick Start

**Python:**

```python
import medrs
import torch

# Load a NIfTI image
img = medrs.load("brain.nii.gz")
print(f"Shape: {img.shape}, Spacing: {img.spacing}")

# Method chaining for transforms
processed = img.resample([1.0, 1.0, 1.0]).z_normalize().clamp(-1, 1)
processed.save("output.nii.gz")

# Load directly to PyTorch tensor (most efficient)
tensor = medrs.load_to_torch("brain.nii.gz", dtype=torch.float16, device="cuda")
```

For training pipelines that repeatedly access the same files, use `load_cached()` for
faster subsequent loads (caches decompressed data for .nii.gz files).

**Rust:**

```rust
use medrs::nifti;
use medrs::transforms::{resample_to_spacing, Interpolation};

fn main() -> medrs::Result<()> {
    let img = nifti::load("brain.nii.gz")?;
    println!("Shape: {:?}, Spacing: {:?}", img.shape(), img.spacing());

    let resampled = resample_to_spacing(&img, [1.0, 1.0, 1.0], Interpolation::Trilinear)?;
    nifti::save(&resampled, "output.nii.gz")?;
    Ok(())
}
```

## Transform Pipeline

Build composable transform pipelines with lazy evaluation and automatic optimization:

**Python:**

```python
import medrs

# Create a reusable pipeline
pipeline = medrs.TransformPipeline()
pipeline.z_normalize()
pipeline.clamp(-1.0, 1.0)
pipeline.resample_to_shape([64, 64, 64])

# Apply to multiple images
for path in image_paths:
    img = medrs.load(path)
    processed = pipeline.apply(img)
```

**Rust:**

```rust
use medrs::pipeline::compose::TransformPipeline;

let pipeline = TransformPipeline::new()
    .z_normalize()
    .clamp(-1.0, 1.0)
    .resample_to_shape([64, 64, 64]);

let processed = pipeline.apply(&img);
```

## Random Augmentation

Reproducible augmentations for ML training with optional seeding:

**Python:**

```python
import medrs

img = medrs.load("brain.nii.gz")

# Individual augmentations
flipped = medrs.random_flip(img, axes=[0, 1, 2], prob=0.5, seed=42)
noisy = medrs.random_gaussian_noise(img, std=0.1, seed=42)
scaled = medrs.random_intensity_scale(img, scale_range=0.1, seed=42)
shifted = medrs.random_intensity_shift(img, shift_range=0.1, seed=42)
rotated = medrs.random_rotate_90(img, axes=(0, 1), seed=42)
gamma = medrs.random_gamma(img, gamma_range=(0.7, 1.5), seed=42)

# Combined augmentation (flip + noise + scale + shift)
augmented = medrs.random_augment(img, seed=42)
```

**Rust:**

```rust
use medrs::transforms::{random_flip, random_gaussian_noise, random_augment};

// Individual augmentations
let flipped = random_flip(&img, &[0, 1, 2], Some(0.5), Some(42))?;
let noisy = random_gaussian_noise(&img, Some(0.1), Some(42))?;

// Combined augmentation
let augmented = random_augment(&img, Some(42))?;
```

## Mgzip: Parallel Compressed Loading

For `.nii.gz` files, medrs supports the **Mgzip** (multi-member gzip) format for parallel decompression. Mgzip files are backwards-compatible with standard gzip; at 8 threads, decompression is 2-3x faster than medrs's own single-threaded libdeflate baseline. Single-threaded Mgzip is slightly slower than that baseline (channel and buffer overhead dominate below 2-3 threads), so it only pays off once you have a few cores to spare.

### Performance (256³ volume)

| Method | Time | vs nibabel |
|--------|------|------------|
| nibabel | 173ms | 1× |
| medrs.load() | 126ms | 1.4× |
| **medrs.load_mgzip(8 threads)** | **47ms** | **3.7×** |

### Usage

```python
import medrs

# Convert existing .nii.gz to Mgzip format (one-time)
medrs.convert_to_mgzip("brain.nii.gz", "brain.mgz.nii.gz", num_threads=8)

# Load with parallel decompression
img = medrs.load_mgzip("brain.mgz.nii.gz", num_threads=8)

# Save directly in Mgzip format
medrs.save_mgzip(img, "output.mgz.nii.gz", num_threads=8)

# Check if file is Mgzip format
if medrs.is_mgzip("file.nii.gz"):
    img = medrs.load_mgzip("file.nii.gz")
```

### Batch Conversion CLI

Convert entire datasets with the included CLI tool:

```bash
# Convert all .nii.gz files in a directory (recursive)
python -m medrs.cli convert-mgzip data/*.nii.gz -r -w 8 -v

# Options:
#   -r, --recursive    Search subdirectories
#   -w, --workers N    Parallel conversion threads (default: CPU count)
#   -v, --verbose      Show progress
#   --suffix .mgz      Output suffix (default: replaces .nii.gz with .mgz.nii.gz)
```

### When to Use Mgzip

- **Large compressed datasets** (100+ files, 256³+ volumes)
- **Multi-core systems** (4+ cores)
- **Repeated access** (training pipelines that load same files across epochs)

Mgzip files are ~1% larger than standard gzip but provide significant speedups. Standard gzip readers (nibabel, etc.) can still read Mgzip files.

## Crop-First Loading

Load only the data you need - essential for training pipelines:

```python
import medrs
import torch

# Load a 64^3 patch starting at position (32, 32, 32)
patch = medrs.load_cropped("volume.nii", [32, 32, 32], [64, 64, 64])

# Load with resampling and reorientation in one step
patch = medrs.load_resampled(
    "volume.nii",
    output_shape=[64, 64, 64],
    target_spacing=[1.0, 1.0, 1.0],
    target_orientation="RAS"
)

# Load directly to GPU tensor
tensor = medrs.load_cropped_to_torch(
    "volume.nii",
    output_shape=[64, 64, 64],
    target_spacing=[1.0, 1.0, 1.0],
    dtype=torch.float16,
    device="cuda"
)
```

## Training Data Loaders

### TrainingDataLoader

LRU-cached patch extraction with prefetching:

```python
import medrs

loader = medrs.TrainingDataLoader(
    volumes=["vol1.nii", "vol2.nii", "vol3.nii"],
    patch_size=[64, 64, 64],
    patches_per_volume=4,
    patch_overlap=[0, 0, 0],
    randomize=True,
    cache_size=1000
)

for patch in loader:
    tensor = patch.to_torch()
```

### FastLoader

Parallel prefetching loader for large .nii.gz datasets (100k+ files):

```python
import glob
import medrs

loader = medrs.FastLoader(
    volumes=glob.glob("data/*.nii.gz"),
    patch_shape=[64, 64, 64],
    prefetch=16,
    workers=4,
    shuffle=True,
    seed=42
)

for patch in loader:
    tensor = patch.to_torch()
```

## Available Transforms

### Intensity Transforms
- `z_normalize()` / `z_normalization()` - Zero mean, unit variance
- `rescale()` / `rescale_intensity()` - Scale to [min, max] range
- `clamp()` - Clamp values to range

### Spatial Transforms
- `resample()` - Resample to target spacing (also `TransformPipeline.resample_to_spacing()`; the Rust API exposes the same function as `resample_to_spacing`)
- `resample_to_shape()` - Resample to target shape
- `reorient()` - Reorient to standard orientation (RAS, LPS, etc.)
- `crop_or_pad()` - Crop or pad to target shape
- `flip()` - Flip along specified axes

### Random Augmentation
- `random_flip()` - Random axis flipping
- `random_gaussian_noise()` - Additive Gaussian noise
- `random_intensity_scale()` - Random intensity scaling
- `random_intensity_shift()` - Random intensity offset
- `random_rotate_90()` - Random 90-degree rotations
- `random_gamma()` - Random gamma correction
- `random_augment()` - Combined augmentation pipeline

## Performance

medrs uses several optimization strategies:

- **SIMD**: Hot loops use portable SIMD (`wide::f32x8`), which lowers to two SSE registers on the x86-64 baseline and a single AVX2 register only when built with `-C target-feature=+avx2` or `-C target-cpu=native` (see `make build-native`; the distributed wheels use the SSE2 baseline)
- **Parallel Processing**: Rayon-based parallelism for large volumes
- **Lazy Evaluation**: Transform pipelines fuse consecutive axis-aligned resamples and trailing intensity operations into a single pass before execution
- **Memory Mapping**: Large files are memory-mapped to avoid full loads
- **Parallel Decompression**: Mgzip format enables multi-threaded gzip decompression

## Compression Formats

| Format | Type | Best for |
|--------|------|----------|
| `.nii` | Uncompressed | Fastest load (memory-mapped), byte-exact crop-first reads |
| `.nii.gz` | gzip | Standard interchange; use `load_mgzip`/Mgzip for parallel decode |
| `.jvol` | Wavelet + Rice coding (optional `jvol` feature) | Storage- and bandwidth-bound workflows on floating-point volumes |

`.jvol` is an optional volumetric codec, vendored from [jvol-rust](https://github.com/fepegar/jvol-rust) by Fernando Pérez-García (MIT licensed; see [Credits](#credits)). It supports two modes:

- **Lossy** (`quality=1..100`): wavelet compression tuned for floating-point intensity volumes, typically 10x to 500x smaller than the source depending on quality. This is the mode `.jvol` is designed for.
- **Lossless**: exact round-trip, but only roughly gzip-parity in file size, and decode is more CPU-intensive than gzip or Mgzip. Prefer `.nii.gz` / `.nii.mgz` when load speed matters more than storage.

Lossy encoding is rejected outright for integer/label data (medrs returns an error) since wavelet quantization would silently corrupt segmentation values; use lossless mode for labels.

```python
import medrs

img = medrs.load("brain.nii.gz")

# Lossy, tuned for floating-point intensity volumes
medrs.save_jvol(img, "brain.jvol", quality=60)

# Lossless (required for label/segmentation volumes)
medrs.save_jvol(seg, "seg.jvol", lossless=True)

# .jvol loads transparently through the normal load() entry point
restored = medrs.load("brain.jvol")

# Convert an existing file directly
medrs.convert_to_jvol("brain.nii.gz", "brain.jvol", quality=60)
```

```rust
use medrs::jvol::{self, JvolOptions};

let img = medrs::nifti::load("brain.nii.gz")?;
jvol::save(&img, "brain.jvol", JvolOptions::lossy(60))?;
let restored = jvol::load("brain.jvol")?;
```

Build with the `jvol` cargo feature (`cargo build --features jvol`); the `python` feature enables it automatically, so the published Python wheel includes `.jvol` support. See `docs/guides/compression.rst` for the full format writeup.

## Examples

See the `examples/` directory for:
- `basic/` - Loading, transforms, and saving
- `integrations/` - PyTorch, MONAI, JAX integration
- `advanced/` - Async pipelines, custom transforms

## Testing

```bash
# Rust tests
cargo test
cargo test --features jvol

# Python tests
pytest tests/

# Benchmarks (see "Reproducing benchmarks" above)
python benchmarks/bench_medrs.py --quick
python benchmarks/bench_monai.py --quick
python benchmarks/bench_torchio.py --quick

# Generate benchmark plots
python benchmarks/plot_results.py
```

## License

medrs is dual-licensed under MIT and Apache-2.0. See [LICENSE](LICENSE) for details. The vendored `.jvol` codec (`src/jvol/codec/`) is MIT-licensed separately; see [LICENSE-jvol](LICENSE-jvol) and [Credits](#credits).

## Credits

The `.jvol` volumetric compression codec (`src/jvol/codec/`) is vendored from [jvol-rust](https://github.com/fepegar/jvol-rust) by [Fernando Pérez-García](https://github.com/fepegar), MIT licensed. Only the codec modules (wavelet lifting, entropy coding, subband management, type definitions) are vendored; medrs's own NIfTI I/O front-end and Python bindings wrap it. Full license text: [LICENSE-jvol](LICENSE-jvol).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Maintainer

Liam Chalcroft (liam.chalcroft.20@ucl.ac.uk)
