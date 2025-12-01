# medrs Examples

This directory contains comprehensive examples demonstrating medrs's capabilities and integrations with popular deep learning frameworks.

## Structure

```
examples/
+-- basic/                    # Core functionality examples
|   +-- quick_start.py       # Get started in 5 minutes
|   +-- load_cropped.py      # Crop-first loading demonstration
|   +-- transforms.py        # High-performance transforms
+-- integrations/            # Framework integrations
|   +-- pytorch_training.py  # PyTorch training pipeline
|   +-- monai_integration.py # MONAI transforms and workflows
|   +-- jax_integration.py   # JAX with JIT compilation
+-- advanced/                # Advanced production patterns
    +-- async_pipeline.py    # Async I/O and prefetching
```

## Getting Started

### Quick Start
```bash
cd examples/basic
python quick_start.py
```

### Prerequisites
Replace placeholder file paths (e.g., `"test_volume.nii.gz"`) with actual NIfTI files.

Install dependencies:
```bash
# Basic usage
pip install medrs

# With framework integrations
pip install medrs[torch,monai,jax]
```

## Examples Overview

### Basic Examples

**`quick_start.py`** - Perfect for beginners
- Demonstrates basic medrs usage
- Shows performance benefits vs traditional loading
- Covers essential operations

**`load_cropped.py`** - Core medrs feature
- Crop-first loading techniques
- Multiple loading methods
- Memory optimization

**`transforms.py`** - Image processing
- Normalization and rescaling
- Resampling and reorientation
- Performance benchmarks

### Framework Integrations

**`pytorch_training.py`** - PyTorch integration
- High-performance training pipeline
- Direct GPU loading
- Memory-efficient batching

**`monai_integration.py`** - MONAI compatibility
- Custom MONAI transforms
- Complete training workflow
- Performance comparison

**`jax_integration.py`** - JAX with JIT
- JAX array conversion
- JIT-compiled operations
- GPU acceleration

### Advanced Examples

**`async_pipeline.py`** - Production patterns
- Asynchronous data loading
- Prefetching strategies
- Concurrent I/O operations

## Key Features Demonstrated

### Performance
- **40x memory reduction** with crop-first loading
- **200-3500x faster** than traditional approaches
- **Direct GPU loading** eliminates CPU-GPU transfers

### Framework Integration
- **Zero-copy** tensor creation
- **Seamless MONAI** compatibility
- **JAX JIT compilation** support

### Production Ready
- **Async I/O** for high throughput
- **Memory pooling** and optimization
- **Error handling** and recovery

## Running Examples

1. **Choose your example** based on your needs:
   - New to medrs? Start with `basic/quick_start.py`
   - Using PyTorch? Try `integrations/pytorch_training.py`
   - Need maximum performance? See `advanced/async_pipeline.py`

2. **Replace file paths** in the examples with your NIfTI files

3. **Run the example**:
   ```bash
   python examples/path/to/example.py
   ```

## Tips for Best Results

- **Use crop-first loading** whenever possible
- **Load directly to GPU** for training workloads
- **Use half precision** (FP16) to save memory
- **Enable async loading** for large datasets
- **Profile your pipeline** to find bottlenecks

## Troubleshooting

### Common Issues

**"File not found" errors**
- Replace placeholder file paths with actual NIfTI files
- Check file permissions and paths

**CUDA out of memory**
- Reduce batch size or patch size
- Use half precision (`dtype=torch.float16`)
- Load smaller patches

**Slow loading**
- Ensure you're using crop-first loading
- Check that files are on fast storage
- Consider async loading for multiple volumes

### Performance Tips

1. **Always use crop-first loading** for training
2. **Load directly to target device** (avoid CPU intermediate)
3. **Use appropriate precision** (FP16 for training)
4. **Consider async loading** for multi-GPU training
5. **Profile your pipeline** regularly

## Learn More

- [medrs Documentation](https://medrs.readthedocs.io)
- [API Reference](../docs/api.rst)
- [Performance Optimization Guide](../docs/guides/performance_optimization.rst)
- [Framework Integration Guide](../docs/guides/advanced_features.rst)

## Contributing

Found a bug or have an improvement? Please:
1. Check existing issues
2. Create a new issue with details
3. Submit a pull request with your example

---

**medrs** - Ultra-high-performance medical imaging I/O for deep learning.