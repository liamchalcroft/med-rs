# medrs Benchmark Plan

## Overview

Comprehensive benchmarking matrix to validate medrs performance claims across different scenarios, volume sizes, compression formats, and comparison libraries.

---

## 1. Comparison Libraries

| Library | Version | Purpose | Installation |
|---------|---------|---------|--------------|
| **medrs** | 0.1.2 | Our library | `pip install -e .` |
| **nibabel** | 5.x | Reference baseline (pure Python) | `pip install nibabel` |
| **MONAI** | 1.3+ | Production medical imaging framework | `pip install monai` |
| **TorchIO** | 0.19+ | Medical imaging augmentation library | `pip install torchio` |
| **SimpleITK** | 2.x | ITK-based I/O (C++ backend) | `pip install SimpleITK` |

### Why Each Library?

- **nibabel**: The standard Python NIfTI library. Baseline for compression benchmarks.
- **MONAI**: What most medical DL researchers use. Our primary competition.
- **TorchIO**: Popular for augmentation pipelines. Strong crop/resample performance.
- **SimpleITK**: C++ backend, represents "optimized C" performance ceiling for some ops.

---

## 2. Volume Sizes

| Size | Voxels | Uncompressed (f32) | Typical Use Case |
|------|--------|-------------------|------------------|
| 64³ | 262K | 1 MB | Patches, small ROIs |
| 128³ | 2.1M | 8 MB | Typical training patches |
| 256³ | 16.8M | 64 MB | Full brain MRI |
| 512³ | 134M | 512 MB | High-res CT/MRI |

### Anisotropic Volumes (Real-World)

| Size | Description | Use Case |
|------|-------------|----------|
| 512×512×30 | Typical CT slice stack | Chest/Abdomen CT |
| 256×256×170 | Brain MRI (1mm iso) | Neuroimaging |
| 192×192×192 | Common training size | nnU-Net default |

---

## 3. Data Types

| dtype | Bytes/voxel | Use Case |
|-------|-------------|----------|
| float32 | 4 | Default for training |
| float16 | 2 | Mixed precision training |
| int16 | 2 | Raw MRI/CT values |
| uint8 | 1 | Segmentation labels |

---

## 4. Compression Formats

| Format | Extension | Parallel Decompress? | Notes |
|--------|-----------|---------------------|-------|
| Uncompressed | .nii | N/A | Memory-mapped, fastest |
| Standard gzip | .nii.gz | ❌ No | Most common format |
| Mgzip | .mgz.nii.gz | ✅ Yes | medrs-specific optimization |

---

## 5. Benchmark Operations Matrix

### 5.1 Basic I/O

| Operation | medrs | nibabel | MONAI | TorchIO | SimpleITK |
|-----------|-------|---------|-------|---------|-----------|
| Load .nii | ✅ | ✅ | ✅ | ✅ | ✅ |
| Load .nii.gz | ✅ | ✅ | ✅ | ✅ | ✅ |
| Load .mgz.nii.gz | ✅ | ✅* | ❌ | ❌ | ❌ |
| Save .nii | ✅ | ✅ | ✅ | ✅ | ✅ |
| Save .nii.gz | ✅ | ✅ | ✅ | ✅ | ✅ |
| Header-only load | ✅ | ✅ | ❌ | ❌ | ✅ |

*nibabel can read Mgzip (backwards compatible) but sequentially.

### 5.2 Crop/ROI Operations

| Operation | medrs | nibabel | MONAI | TorchIO | SimpleITK |
|-----------|-------|---------|-------|---------|-----------|
| Byte-exact crop (.nii) | ✅ | ❌ | ❌ | ❌ | ❌ |
| Load-then-crop (.nii) | ✅ | ✅ | ✅ | ✅ | ✅ |
| Load-then-crop (.nii.gz) | ✅ | ✅ | ✅ | ✅ | ✅ |
| Random patch extraction | ✅ | ✅ | ✅ | ✅ | ❌ |

### 5.3 Resampling

| Operation | medrs | nibabel | MONAI | TorchIO | SimpleITK |
|-----------|-------|---------|-------|---------|-----------|
| Resample to spacing | ✅ | ❌* | ✅ | ✅ | ✅ |
| Resample to shape | ✅ | ❌* | ✅ | ✅ | ✅ |
| Load + Resample fused | ✅ | ❌ | ❌ | ❌ | ❌ |

*nibabel doesn't have built-in resampling.

### 5.4 Tensor Conversion

| Operation | medrs | nibabel | MONAI | TorchIO | SimpleITK |
|-----------|-------|---------|-------|---------|-----------|
| To NumPy | ✅ | ✅ | ✅ | ✅ | ✅ |
| To PyTorch (CPU) | ✅ | ✅ | ✅ | ✅ | ✅ |
| To PyTorch (CUDA) | ✅ | ✅ | ✅ | ✅ | ✅ |
| Zero-copy to NumPy | ✅ | ✅ | ❌ | ❌ | ❌ |

### 5.5 Intensity Operations

| Operation | medrs | nibabel | MONAI | TorchIO | SimpleITK |
|-----------|-------|---------|-------|---------|-----------|
| Z-normalization | ✅ | ✅* | ✅ | ✅ | ❌ |
| Rescale intensity | ✅ | ✅* | ✅ | ✅ | ✅ |
| Clamp | ✅ | ✅* | ✅ | ✅ | ✅ |

*Manual numpy operations.

### 5.6 Augmentation (Random)

| Operation | medrs | nibabel | MONAI | TorchIO | SimpleITK |
|-----------|-------|---------|-------|---------|-----------|
| Random flip | ✅ | ❌ | ✅ | ✅ | ❌ |
| Random noise | ✅ | ❌ | ✅ | ✅ | ❌ |
| Random rotate 90° | ✅ | ❌ | ✅ | ✅ | ❌ |
| Random gamma | ✅ | ❌ | ✅ | ✅ | ❌ |
| Random affine | ❌ | ❌ | ✅ | ✅ | ❌ |
| Random elastic | ❌ | ❌ | ✅ | ✅ | ❌ |

### 5.7 Training Pipeline Scenarios

| Scenario | Description | Libraries |
|----------|-------------|-----------|
| Single file load | Cold load, no caching | All |
| Repeated file load | Same file, test caching | All |
| Sequential epoch | Load N files in order | All |
| Shuffled epoch | Load N files randomly | All |
| Multi-worker DataLoader | PyTorch DataLoader integration | medrs, MONAI, TorchIO |
| FastLoader throughput | Parallel prefetch loader | medrs only |

---

## 6. Mgzip-Specific Benchmarks

### 6.1 Thread Scaling

| Threads | medrs load_mgzip | Expected Speedup |
|---------|------------------|------------------|
| 1 | baseline | 1.0× |
| 2 | measure | ~1.8× |
| 4 | measure | ~3.0× |
| 8 | measure | ~4.5× |
| 16 | measure | ~5.0× (diminishing) |

### 6.2 Format Comparison

| Format | Library | Load Time | Notes |
|--------|---------|-----------|-------|
| .nii | medrs | fastest | mmap |
| .nii.gz | nibabel | baseline | sequential gzip |
| .nii.gz | medrs | measure | libdeflate |
| .mgz.nii.gz | nibabel | measure | sequential (compat) |
| .mgz.nii.gz | medrs (1t) | measure | gzp single-thread |
| .mgz.nii.gz | medrs (8t) | measure | gzp parallel |

### 6.3 Conversion Overhead

| Operation | Time | Notes |
|-----------|------|-------|
| convert_to_mgzip (1 thread) | measure | One-time cost |
| convert_to_mgzip (8 threads) | measure | Parallel compression |
| File size increase | ~1% | Mgzip vs standard gzip |

---

## 7. Memory Benchmarks

| Metric | How to Measure |
|--------|----------------|
| Peak RSS during load | `psutil.Process().memory_info().rss` |
| Memory after load | Same, after GC |
| Memory with mmap | Should be ~0 for .nii |
| Memory with caching | `load_cached` repeated calls |

---

## 8. Real-World Dataset Benchmarks

### Datasets to Test

| Dataset | Files | Typical Size | Format |
|---------|-------|--------------|--------|
| Synthetic | Generated | 64³-512³ | .nii, .nii.gz |
| BraTS sample | 4 modalities × N | 240×240×155 | .nii.gz |
| AMOS sample | CT volumes | 512×512×~200 | .nii.gz |

### End-to-End Scenarios

| Scenario | Description | Metric |
|----------|-------------|--------|
| nnU-Net preprocessing | Load + resample + crop | Total time |
| Training epoch | 100 samples, random patches | Samples/sec |
| Inference | Load + predict + save | Total time |

---

## 9. Benchmark Execution Plan

### Phase 1: Core I/O (Priority: HIGH)
1. Load .nii (all sizes, all libraries)
2. Load .nii.gz (all sizes, all libraries)
3. Load Mgzip (medrs only, thread scaling)
4. Save .nii / .nii.gz (all libraries)

### Phase 2: Crop Operations (Priority: HIGH)
1. Byte-exact crop .nii (medrs only)
2. Load-then-crop comparison
3. Random patch extraction

### Phase 3: Transforms (Priority: MEDIUM)
1. Resample to spacing
2. Z-normalization
3. Combined load + transform

### Phase 4: Training Simulation (Priority: HIGH)
1. Single-worker throughput
2. Multi-worker DataLoader
3. FastLoader vs alternatives

### Phase 5: Memory Profiling (Priority: MEDIUM)
1. Peak memory during load
2. Memory efficiency with mmap
3. Caching effectiveness

---

## 10. Output Format

### JSON Structure
```json
{
  "metadata": {
    "timestamp": "2024-01-16T12:00:00",
    "platform": "darwin-arm64",
    "cpu": "Apple M2 Pro",
    "memory_gb": 16,
    "python_version": "3.10.0",
    "library_versions": {
      "medrs": "0.1.2",
      "nibabel": "5.2.0",
      "monai": "1.3.0",
      "torchio": "0.19.0"
    }
  },
  "results": [
    {
      "library": "medrs",
      "operation": "load",
      "format": ".nii",
      "size": [256, 256, 256],
      "dtype": "float32",
      "mean_ms": 0.45,
      "std_ms": 0.05,
      "median_ms": 0.44,
      "min_ms": 0.40,
      "max_ms": 0.55,
      "iterations": 20,
      "memory_mb": 64.0
    }
  ]
}
```

### Plots to Generate
1. **Bar chart**: Load time by library (grouped by size)
2. **Line chart**: Scaling with volume size
3. **Heatmap**: Operation × Library performance matrix
4. **Thread scaling**: Mgzip speedup vs thread count
5. **Memory**: Peak RSS by library and operation

---

## 11. Running Benchmarks

### Quick (CI/Development)
```bash
python benchmarks/bench_medrs.py --quick
python benchmarks/bench_monai.py --quick
python benchmarks/bench_torchio.py --quick
python benchmarks/compare_all.py --quick
```

### Full (Publication)
```bash
python benchmarks/bench_medrs.py --full
python benchmarks/bench_monai.py --full
python benchmarks/bench_torchio.py --full
python benchmarks/bench_nibabel.py --full  # NEW
python benchmarks/bench_simpleitk.py --full  # NEW
python benchmarks/compare_all.py --full
python benchmarks/plot_results.py
```

### Mgzip-Specific
```bash
python benchmarks/bench_mgzip.py --threads 1,2,4,8,16
```

---

## 12. TODO: New Benchmark Scripts Needed

| Script | Purpose | Priority |
|--------|---------|----------|
| `bench_nibabel.py` | Baseline comparison | HIGH |
| `bench_simpleitk.py` | C++ backend comparison | MEDIUM |
| `bench_mgzip.py` | Thread scaling analysis | HIGH |
| `bench_memory.py` | Memory profiling | MEDIUM |
| `bench_training.py` | End-to-end training sim | HIGH |
| `bench_fastloader.py` | FastLoader throughput | HIGH |

---

## 13. Expected Results Summary

### Load Performance (256³ .nii.gz → f32)

| Library | Expected Time | vs medrs |
|---------|---------------|----------|
| nibabel | ~170ms | baseline |
| medrs (gzip) | ~50ms | 3.4× faster |
| medrs (mgzip 8t) | ~25ms | 6.8× faster |
| MONAI | ~175ms | ~nibabel |
| TorchIO | ~180ms | ~nibabel |
| SimpleITK | ~80ms | 2× faster than nibabel |

### Crop Performance (64³ from 256³ .nii)

| Library | Expected Time | Notes |
|---------|---------------|-------|
| medrs (byte-exact) | <1ms | Only reads 1MB |
| Others | ~170ms | Must load full 64MB |

### Training Throughput (samples/sec, 64³ patches)

| Method | Expected | Notes |
|--------|----------|-------|
| medrs FastLoader (8 workers) | 500+ | Parallel prefetch |
| MONAI DataLoader (8 workers) | 50-100 | Standard approach |
| TorchIO Queue | 100-200 | Optimized queue |
