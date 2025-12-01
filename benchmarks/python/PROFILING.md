# Profiling Guide

This directory contains profiling scripts for medrs operations.

## Quick Start

```bash
# Run all Rust profiles (32^3, 128^3, 256^3 volumes)
./profiling/run_profiles.sh

# Run Python profiles
./profiling/run_profiles.sh python
```

## Rust Profiling Examples

Individual operation profiles with configurable size and iterations:

```bash
# Profile z-normalization at 256^3 with 20 iterations
cargo run --example profile_z_norm --release -- 256 20

# Profile crop/pad at 512^3 with 10 iterations
cargo run --example profile_crop_pad --release -- 512 10

# Profile flip
cargo run --example profile_flip --release -- 256 20

# Profile rescale
cargo run --example profile_rescale --release -- 256 20

# Profile resample (expensive - use smaller sizes)
cargo run --example profile_resample --release -- 128 10

# Profile load/save
cargo run --example profile_load --release -- 256 20

# Profile all operations at multiple sizes
cargo run --example profile_all --release
cargo run --example profile_all --release -- --large  # Include 512^3
```

## Generating Flamegraphs

Requires [cargo-flamegraph](https://github.com/flamegraph-rs/flamegraph):

```bash
cargo install flamegraph

# Generate flamegraph for crop/pad
cargo flamegraph --example profile_crop_pad --release -o profiling/flamegraphs/crop_pad.svg -- 256 50

# Or use the helper script
./profiling/run_profiles.sh flamegraph
```

## macOS Profiling with Samply

[Samply](https://github.com/mstange/samply) provides excellent profiling on macOS:

```bash
cargo install samply

# Profile crop_pad at 256^3
./profiling/run_profiles.sh samply crop_pad 256

# Profile any operation
samply record cargo run --example profile_flip --release -- 256 50
```

## Python Profiling

### Basic timing

```bash
uv run python profiling/profile_ops.py
uv run python profiling/profile_ops.py --large  # Include 512^3
```

### Single operation (for detailed profiling)

```bash
uv run python profiling/profile_ops.py --size 256 --op crop --iterations 50
```

Available operations: `z_norm`, `rescale`, `crop`, `crop_expand`, `flip`, `flip_all`, `resample`

### With cProfile

```bash
python -m cProfile -s cumtime profiling/profile_ops.py --size 256 --op crop --iterations 50
```

### With py-spy

```bash
pip install py-spy
py-spy record -o profile.svg -- python profiling/profile_ops.py --size 256 --op crop --iterations 50
```

## Interpreting Results

Key metrics:
- **Mean/Median time**: Average operation time in milliseconds
- **Throughput (MB/s)**: Data processing rate (input volume size / time)

Performance targets (rough guidelines):
- Intensity ops (z-norm, rescale): >1000 MB/s
- Spatial ops (crop, flip): >500 MB/s
- Resample: >100 MB/s (depends on interpolation)
- Load .nii (mmap): Near-instant (<1ms for any size)
- Load .nii.gz: ~500 MB/s decompression

## Current Performance Issues

Based on benchmarks, these operations need optimization at large volumes:
- `crop_or_pad`: 50x slower than monai at 256^3
- `flip`: 7x slower than monai at 256^3
- `rescale_intensity`: Slightly slower than monai at 256^3

Focus areas:
1. Reduce unnecessary allocations
2. Improve SIMD utilization
3. Add rayon parallelization for large volumes
4. Optimize memory access patterns (cache-friendly iteration)
