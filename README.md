# medrs

High-performance medical imaging I/O and processing library for Rust and Python.

[![Crates.io](https://img.shields.io/crates/v/medrs.svg)](https://crates.io/crates/medrs)
[![PyPI](https://img.shields.io/pypi/v/medrs.svg)](https://pypi.org/project/medrs/)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE)

## Overview

medrs is designed for throughput-critical medical imaging workflows, particularly deep learning pipelines that process large 3D volumes. It provides:

- **Fast NIfTI I/O**: Memory-mapped reading, crop-first loading (read sub-volumes without loading entire files)
- **Transform Pipeline**: Lazy evaluation with automatic operation fusion and SIMD acceleration
- **Mixed Precision**: Native f16/bf16 support for 50% storage savings
- **Random Augmentation**: Reproducible, GPU-friendly augmentations for ML training
- **Python Bindings**: Zero-copy numpy views, direct PyTorch/JAX tensor creation
- **MONAI Integration**: Drop-in replacements for MONAI transforms

## Why medrs?

### Performance vs MONAI & TorchIO (128³ volume)

| Operation | medrs | MONAI | TorchIO | vs MONAI |
|-----------|-------|-------|---------|----------|
| Load | 0.13ms | 4.55ms | 4.71ms | **35x** |
| Load Cropped (64³) | 0.41ms | 4.68ms | 9.86ms | **11x** |
| Load Resampled | 0.40ms | 6.88ms | 27.65ms | **17x** |
| To PyTorch | 0.49ms | 5.14ms | 10.22ms | **10x** |
| Load + Normalize | 0.60ms | 5.36ms | 12.26ms | **9x** |

*At larger volumes (512³), speedups increase dramatically: up to **38,000x** vs MONAI and **6,600x** vs TorchIO.*

### Storage Efficiency (128³ volume, compressed)

| Format | Size | vs f32 |
|--------|------|--------|
| float32 | 8.3 MB | 100% |
| **bfloat16** | **3.4 MB** | **41%** |
| **float16** | **4.1 MB** | **50%** |
| int16 | 1.2 MB | 15% |

### Key Advantages

1. **Crop-First Loading**: Load 64³ patch from 512³ volume without reading entire file - **6,600x faster** than MONAI
2. **Mixed Precision**: Save in bf16/f16 for 40-50% smaller files with minimal precision loss
3. **MONAI Drop-in**: Replace MONAI I/O transforms with one import change
4. **Zero-Copy**: Direct tensor creation without intermediate numpy allocations

<details>
<summary>📊 Detailed Benchmarks (click to expand)</summary>

### Comprehensive Benchmark Results

Benchmark results comparing medrs, MONAI, and TorchIO across multiple volume sizes and operations.

![Benchmark Summary](benchmarks/results/plots/summary.png)

#### Load Performance (Basic I/O)

| Size | medrs | MONAI | TorchIO | vs MONAI | vs TorchIO |
|------|-------|-------|---------|----------|------------|
| 64³ | 0.13ms | 1.34ms | 2.35ms | **10x** | **18x** |
| 128³ | 0.13ms | 4.55ms | 4.71ms | **35x** | **36x** |
| 256³ | 0.14ms | 159.11ms | 95.18ms | **1,136x** | **680x** |
| 512³ | 0.13ms | 5,006.76ms | 866.54ms | **38,513x** | **6,665x** |

#### Crop-First Loading (64³ patch)

| Source | medrs | MONAI | TorchIO | vs MONAI | vs TorchIO |
|--------|-------|-------|---------|----------|------------|
| 64³ | 0.27ms | 1.75ms | 6.00ms | **6x** | **22x** |
| 128³ | 0.41ms | 4.68ms | 9.86ms | **11x** | **24x** |
| 256³ | 0.55ms | 154.86ms | 104.48ms | **282x** | **190x** |
| 512³ | 0.76ms | 5,041.42ms | 1,076.89ms | **6,633x** | **1,417x** |

#### Load Resampled (Half resolution)

| Source | medrs | MONAI | TorchIO | vs MONAI | vs TorchIO |
|--------|-------|-------|---------|----------|------------|
| 64³ → 32³ | 0.18ms | 1.93ms | 5.45ms | **11x** | **30x** |
| 128³ → 64³ | 0.40ms | 6.88ms | 27.65ms | **17x** | **69x** |
| 256³ → 128³ | 2.02ms | 178.87ms | 363.85ms | **89x** | **180x** |
| 512³ → 256³ | 6.67ms | 5,960.93ms | 4,039.05ms | **894x** | **605x** |

#### Direct PyTorch Loading

| Source | medrs | MONAI | TorchIO | vs MONAI | vs TorchIO |
|--------|-------|-------|---------|----------|------------|
| 64³ | 0.34ms | 1.58ms | 5.37ms | **5x** | **16x** |
| 128³ | 0.49ms | 5.14ms | 10.22ms | **10x** | **21x** |
| 256³ | 0.60ms | 162.78ms | 53.70ms | **271x** | **90x** |
| 512³ | 0.84ms | 5,864.85ms | 1,223.24ms | **6,982x** | **1,456x** |

#### Load with Z-Normalization

| Source | medrs | MONAI | TorchIO | vs MONAI | vs TorchIO |
|--------|-------|-------|---------|----------|------------|
| 64³ | 0.49ms | 2.15ms | 7.04ms | **4x** | **14x** |
| 128³ | 0.60ms | 5.36ms | 12.26ms | **9x** | **20x** |
| 256³ | 0.73ms | 163.38ms | 53.59ms | **224x** | **73x** |
| 512³ | 1.01ms | 3,735.31ms | 1,092.25ms | **3,698x** | **1,081x** |

*Benchmarks run on Apple M1 Pro, 20 iterations, 3 warmup. Run your own: `python benchmarks/bench_medrs.py`*

</details>

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

## Training Data Loader

High-performance patch extraction for training:

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
    # Training loop
    tensor = patch.to_torch()
```

## Available Transforms

### Intensity Transforms
- `z_normalize()` / `z_normalization()` - Zero mean, unit variance
- `rescale()` / `rescale_intensity()` - Scale to [min, max] range
- `clamp()` - Clamp values to range

### Spatial Transforms
- `resample()` / `resample_to_spacing()` - Resample to target spacing
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

- **SIMD**: Trilinear interpolation uses AVX2/SSE for 8-way parallel processing
- **Parallel Processing**: Rayon-based parallelism for large volumes
- **Lazy Evaluation**: Transform pipelines compose operations before execution
- **Memory Mapping**: Large files are memory-mapped to avoid full loads
- **Buffer Pooling**: Reusable buffers reduce allocation overhead

## Examples

See the `examples/` directory for:
- `basic/` - Loading, transforms, and saving
- `integrations/` - PyTorch, MONAI, JAX integration
- `advanced/` - Async pipelines, custom transforms

## Testing

```bash
# Rust tests
cargo test

# Python tests
pytest tests/

# Benchmarks (requires torch, monai, torchio)
python benchmarks/bench_medrs.py --quick
python benchmarks/bench_monai.py --quick
python benchmarks/bench_torchio.py --quick

# Generate benchmark plots
python benchmarks/plot_results.py
```

## License

medrs is dual-licensed under MIT and Apache-2.0. See [LICENSE](LICENSE) for details.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Maintainer

Liam Chalcroft (liam.chalcroft.20@ucl.ac.uk)
