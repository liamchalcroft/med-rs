# medrs Examples

Examples demonstrating medrs capabilities and framework integrations.

## Structure

```
examples/
├── basic/                    # Core functionality
│   ├── quick_start.py       # Basic loading
│   ├── load_cropped.py      # Crop-first loading
│   └── transforms.py        # Transforms API
├── integrations/            # Framework integrations
│   ├── pytorch_training.py  # PyTorch pipeline
│   ├── monai_integration.py # MONAI transforms
│   └── jax_integration.py   # JAX integration
└── advanced/
    └── async_pipeline.py    # Async I/O
```

## Quick Start

```bash
# Install dependencies
pip install medrs torch

# Run an example (replace file paths with your NIfTI files)
python examples/basic/quick_start.py
```

## Examples

### basic/quick_start.py
Basic loading and normalization.

### basic/load_cropped.py
Crop-first loading - loads only the bytes you need from uncompressed .nii files.

### basic/transforms.py
Transform API: z-normalization, rescaling, resampling, crop/pad.

### integrations/pytorch_training.py
PyTorch training pipeline with direct GPU loading.

### integrations/monai_integration.py
MONAI-compatible transforms using medrs for I/O.

### integrations/jax_integration.py
JAX array conversion and JIT compilation.

### advanced/async_pipeline.py
Async data loading with prefetching.

## Notes

- Replace placeholder file paths with actual NIfTI files
- `load_cropped` requires uncompressed .nii files for byte-exact loading
- Set `num_workers=0` in DataLoader when using medrs (handles concurrency internally)
- Use `dtype=torch.float16` for memory efficiency on GPU
