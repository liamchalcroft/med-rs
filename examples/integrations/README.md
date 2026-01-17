# MONAI + medrs Integration Examples

Examples demonstrating medrs integration with MONAI for high-performance medical imaging workflows.

## Features

### Multi-Modal Dictionary Transforms
- Coordinated cropping across multiple modalities
- Automatic spatial normalization for different voxel spacings
- MONAI-compatible dictionary transforms

### MONAI MetaTensor Support
- NIfTI metadata preservation (affine matrix, spacing, orientation)
- Spatial accuracy for clinical applications
- Complete metadata for debugging

## Files Overview

### Core Integration Examples

- `monai_integration.py` - MONAI integration with dictionary transforms and MetaTensor support
- `monai_label_aware_training.py` - Label-aware training workflow
- `monai_performance_benchmark.py` - Performance benchmarking (uses synthetic data)

### Advanced Features

- `dictionary_transforms.py` - Multi-modal dictionary transforms for coordinated cropping
- `metatensor_example.py` - MONAI MetaTensor integration with metadata preservation

### Framework Integrations

- `pytorch_training.py` - PyTorch training pipeline
- `jax_integration.py` - JAX integration with JIT compilation

## Quick Start

```bash
# Verify installation
python quick_integration_test.py

# Run integration examples
python monai_crop_first_integration.py
python monai_label_aware_training.py
```

## Integration Pattern

### Traditional MONAI
```python
from monai.transforms import Compose, LoadImaged, RandCropByPosNegLabeld

# Loads FULL volumes into memory first
transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    RandCropByPosNegLabeld(
        keys=["image", "label"],
        spatial_size=(96, 96, 96),
        pos=1,
        neg=1
    )
])
```

### medrs Crop-First
```python
from monai.transforms import Compose, EnsureChannelFirstd, CastToTyped
import medrs

class MedrsLabelAwareCropd:
    def __init__(self, keys, patch_size, pos_neg_ratio=1.0, device="cpu"):
        self.keys = keys
        self.patch_size = patch_size
        self.pos_neg_ratio = pos_neg_ratio
        self.device = device

    def __call__(self, data):
        image_path = data["image"]
        label_path = data["label"]

        image_tensor, label_tensor = medrs.load_label_aware_cropped(
            image_path, label_path,
            patch_size=self.patch_size,
            pos_neg_ratio=self.pos_neg_ratio
        )

        return {"image": image_tensor, "label": label_tensor}

transforms = Compose([
    MedrsLabelAwareCropd(
        keys=["image", "label"],
        patch_size=(96, 96, 96),
        pos_neg_ratio=2.0,
        device="cuda:0"
    ),
    EnsureChannelFirstd(keys=["image", "label"]),
    CastToTyped(keys=["image", "label"], dtype=(torch.float32, torch.long)),
])
```

## Performance Comparison

| Operation | Traditional MONAI | medrs Crop-First | Improvement |
|-----------|------------------|------------------|-------------|
| Load 128³ volume | ~7.5ms | ~0.2ms | 41x |
| Load 64³ patch from 128³ | ~30ms | ~0.8ms | 39x |
| Memory usage | Full volume | Patch only | Reduced |
| To PyTorch (128³) | ~16ms | ~1.0ms | 16x |

## Installation

```bash
pip install monai torch numpy

# medrs (built from source)
cd /path/to/medrs
maturin develop --release
```

## Best Practices

### Device Management
```python
image = medrs.load_cropped_to_torch(
    "volume.nii",
    output_shape=(96, 96, 96),
    device="cuda:0",
    dtype=torch.float16
)
```

### Batch Processing
```python
from medrs import TrainingDataLoader

loader = TrainingDataLoader(
    volumes=image_paths,
    patch_size=(96, 96, 96),
    patches_per_volume=8,
    device="cuda:0"
)
```

### Pipeline Optimization
```python
transforms = Compose([
    medrs_load_transform,      # Fast I/O first
    monai_augment_transforms   # Then MONAI transforms
])
```

## When to Use Each

**medrs:**
- Loading cropped patches for training
- Label-aware sampling
- High-throughput pipelines
- Memory-constrained environments
- Direct GPU allocation

**MONAI:**
- Data augmentation (flips, rotations, noise)
- Normalization and scaling
- Complex transform pipelines
- Medical imaging specific transforms

## Troubleshooting

### medrs Not Found
```bash
maturin develop --release
python -c "import medrs; print('medrs imported successfully')"
```

### CUDA Issues
```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```
