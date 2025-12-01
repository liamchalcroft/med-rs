# medrs

High-performance medical imaging I/O and processing library for Rust and Python.

[![Crates.io](https://img.shields.io/crates/v/medrs.svg)](https://crates.io/crates/medrs)
[![PyPI](https://img.shields.io/pypi/v/medrs.svg)](https://pypi.org/project/medrs/)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE)

## Overview

medrs is designed for throughput-critical medical imaging workflows, particularly deep learning pipelines that process large 3D volumes. It provides:

- **Fast NIfTI I/O**: Memory-mapped reading, crop-first loading (read sub-volumes without loading entire files)
- **Transform Pipeline**: Lazy evaluation with automatic operation fusion and SIMD acceleration
- **Random Augmentation**: Reproducible, GPU-friendly augmentations for ML training
- **Python Bindings**: Zero-copy numpy views, direct PyTorch/JAX tensor creation

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

### Python

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

### Rust

```rust
use medrs::nifti;
use medrs::transforms::{resample_to_spacing, Interpolation};

fn main() -> medrs::Result<()> {
    let img = nifti::load("brain.nii.gz")?;
    println!("Shape: {:?}, Spacing: {:?}", img.shape(), img.spacing());

    let resampled = resample_to_spacing(&img, [1.0, 1.0, 1.0], Interpolation::Trilinear);
    nifti::save(&resampled, "output.nii.gz")?;
    Ok(())
}
```

## Transform Pipeline

Build composable transform pipelines with lazy evaluation and automatic optimization:

### Python

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

### Rust

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

### Python

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

### Rust

```rust
use medrs::transforms::{random_flip, random_gaussian_noise, random_augment};

// Individual augmentations
let flipped = random_flip(&img, &[0, 1, 2], Some(0.5), Some(42))?;
let noisy = random_gaussian_noise(&img, Some(0.1), Some(42));

// Combined augmentation
let augmented = random_augment(&img, Some(42))?;
```

## Crop-First Loading

Load only the data you need - essential for training pipelines:

### Python

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

# Benchmarks
cargo bench
```

## License

medrs is dual-licensed under MIT and Apache-2.0. See [LICENSE](LICENSE) for details.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Maintainer

Liam Chalcroft (liam.chalcroft.20@ucl.ac.uk)
