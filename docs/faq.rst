Frequently Asked Questions (FAQ)
==================================

This page answers common questions about medrs and helps resolve typical issues.

General Questions
-----------------

**What is medrs?**

medrs is a high-performance medical imaging library that combines Rust's performance with Python's ecosystem. It provides ultra-fast NIfTI I/O operations with crop-first loading that achieves 40x memory reduction and 200-3500x speed improvements for deep learning workflows.

**What file formats does medrs support?**

Currently medrs supports:
- NIfTI-1 and NIfTI-2 (.nii, .nii.gz)
- All common data types (u8/i8/i16/u16/i32/u32/i64/u64/f16/bf16/f32/f64)
- Compressed and uncompressed files
- Memory mapping for uncompressed files

Future releases will add DICOM, PAR/REC, and Analyze format support.

**What Python versions are supported?**

medrs requires Python 3.10+ and is optimized for Python 3.11+. The library uses modern Python features and type hints extensively.

**Do I need Rust installed?**

No. For standard Python usage, you only need to install medrs via pip:

.. code-block:: bash

   pip install medrs

The Rust components are automatically compiled during installation.

**How does medrs achieve such high performance?**

medrs uses several optimization techniques:
- **Crop-first loading**: Only reads the exact bytes needed for your crop
- **Zero-copy operations**: Direct tensor creation without intermediate copies
- **SIMD optimization**: Vectorized operations for data processing
- **Memory pooling**: Intelligent buffer reuse to minimize allocations
- **Parallel I/O**: Multi-threaded file operations
- **GPU integration**: Direct GPU memory allocation

Performance Comparison
----------------------

**How does medrs compare to nibabel?**

medrs is significantly faster for typical deep learning workflows:

- **Full volume loading**: Similar performance to nibabel
- **Cropped loading**: 200-3500x faster (medrs reads only cropped region, nibabel loads entire volume)
- **Memory usage**: 40x reduction for training workflows
- **GPU integration**: Direct GPU loading vs CPU then GPU transfer

**How does medrs compare to MONAI?**

medrs complements MONAI by providing optimized I/O:

- **I/O operations**: medrs is 200-3500x faster for cropped loading
- **Transforms**: Use medrs for I/O + MONAI transforms for best performance
- **Memory**: 40x less memory usage for training pipelines
- **Integration**: Seamless MONAI compatibility with custom transforms

**How does medrs compare to torchio?**

Similar to nibabel, torchio loads entire volumes before cropping:

- **Speed**: medrs is 200-3500x faster for patch-based training
- **Memory**: medrs uses 40x less memory
- **Usage**: Replace torchio's `LoadImage` with medrs's crop-first loading
- **Augmentation**: Combine medrs I/O with torchio augmentations

Usage Questions
---------------

**How do I install medrs?**

.. code-block:: bash

   # Basic installation
   pip install medrs

   # With development dependencies
   pip install medrs[dev]

   # With framework integrations
   pip install medrs[monai]  # MONAI integration
   pip install medrs[jax]    # JAX integration

**How do I load a cropped patch?**

.. code-block:: python

   import medrs

   # Load a 64x64x64 patch
   patch = medrs.load_cropped(
       "volume.nii.gz",
       crop_offset=[32, 32, 16],  # Starting voxel
       crop_shape=[64, 64, 64]     # Patch size
   )

**How do I load directly to GPU?**

.. code-block:: python

   import torch
   import medrs

   # Direct GPU loading with half precision
   tensor = medrs.load_cropped_to_torch(
       "volume.nii.gz",
       output_shape=[64, 64, 64],
       device="cuda",
       dtype=torch.float16
   )

**How do I use medrs with PyTorch DataLoader?**

.. code-block:: python

   import medrs
   from torch.utils.data import DataLoader, Dataset

   class MedicalDataset(Dataset):
       def __init__(self, volume_paths, patch_size=(64, 64, 64)):
           self.volume_paths = volume_paths
           self.patch_size = patch_size

       def __getitem__(self, idx):
           patch = medrs.load_cropped(
               self.volume_paths[idx],
               crop_offset=[32, 32, 16],
               crop_shape=self.patch_size
           )
           return patch.to_torch(device="cuda")

   dataset = MedicalDataset(volume_paths)
   dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

**How do I integrate with MONAI?**

.. code-block:: python

   import medrs
   from monai.transforms import Compose
   from medrs.monai_integration import MedrsLoadCroppedd

   # Replace LoadImaged + RandCropByPosNegLabeld
   transform = Compose([
       MedrsLoadCroppedd(
           keys=["image", "label"],
           patch_size=(96, 96, 96),
           device="cuda"
       ),
       # Other MONAI transforms...
   ])

Troubleshooting
---------------

**Installation fails with "Could not find Rust"**

For development installation, you need Rust:

.. code-block:: bash

   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

For regular installation, try precompiled wheels:

.. code-block:: bash

   pip install medrs --only-binary=all

**"CUDA out of memory" error**

Reduce memory usage:

.. code-block:: python

   # Use smaller patches
   patch = medrs.load_cropped_to_torch(
       "volume.nii.gz",
       output_shape=[32, 32, 32],  # Smaller patch
       dtype=torch.float16         # Half precision
   )

   # Use CPU for very large operations
   tensor = medrs.load_cropped_to_torch("volume.nii.gz", device="cpu")

**Loading is slow for the first time**

First load includes compilation cache. Subsequent loads are much faster. To warm up:

.. code-block:: python

   # Warm up with a small patch
   _ = medrs.load_cropped("volume.nii.gz", [16, 16, 16], [16, 16, 16])

**"File not found" error**

Check file paths and formats:

.. code-block:: python

   from pathlib import Path
   import medrs

   path = "volume.nii.gz"
   if not Path(path).exists():
       print(f"File not found: {path}")
       print("Current directory:", Path.cwd())
       print("Files:", list(Path(".").glob("*.nii*")))

**Error with compressed files (.nii.gz)**

Ensure sufficient memory for decompression:

.. code-block:: python

   # For very large compressed files, consider decompressing first
   import gzip
   import shutil

   with gzip.open("large_volume.nii.gz", "rb") as f_in:
       with open("large_volume.nii", "wb") as f_out:
           shutil.copyfileobj(f_in, f_out)

Performance Questions
---------------------

**How can I optimize memory usage?**

Use crop-first loading and appropriate precision:

.. code-block:: python

   patch = medrs.load_cropped_to_torch(
       "volume.nii.gz",
       output_shape=[64, 64, 64],
       dtype=torch.float16  # Half precision
   )

   full_volume = medrs.load("volume.nii.gz")
   patch = full_volume[:, :, :64]

**How can I maximize loading speed?**

Use concurrent loading and caching:

.. code-block:: python

   import concurrent.futures

   def load_volumes_concurrent(volume_paths):
       with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
           futures = [
               executor.submit(
                   medrs.load_cropped_to_torch,
                   path, [64, 64, 64], device="cuda"
               )
               for path in volume_paths
           ]
           return [future.result() for future in futures]

   optimal_size = find_optimal_batch_size(
       input_shape=(1, 64, 64, 64),
       model=your_model,
       target_memory_gb=8.0
   )

**How do I profile performance?**

Use the built-in profiler:

.. code-block:: python

   from medrs.performance_profiler import PerformanceProfiler

   with PerformanceProfiler() as profiler:
       for i in range(100):
           patch = medrs.load_cropped("volume.nii.gz", [32, 32, 32], [64, 64, 64])
           result = model(patch)

   summary = profiler.get_summary()
   print(summary)

Advanced Usage
--------------

**Can I use custom data types?**

Yes, medrs supports all common medical imaging data types:

.. code-block:: python

   # Load with specific data type
   img_u16 = medrs.load("volume.nii.gz", dtype="uint16")
   img_f32 = medrs.load("volume.nii.gz", dtype="float32")

**How do I handle different orientations?**

.. code-block:: python

   # Reorient to standard RAS orientation
   img = medrs.load("volume.nii.gz")
   img_ras = medrs.reorient(img, "RAS")

   # Load with specific orientation
   tensor = medrs.load_resampled(
       "volume.nii.gz",
       target_spacing=[1.0, 1.0, 1.0],
       target_orientation="RAS"
   )

**Can I use medrs for inference?**

Yes, medrs is optimized for both training and inference:

.. code-block:: python

   # Fast inference with minimal preprocessing
   def predict_volume(model, volume_path, patch_size=128):
       # Load patch directly to GPU
       patch = medrs.load_cropped_to_torch(
           volume_path,
           output_shape=[patch_size, patch_size, patch_size],
           device="cuda"
       )

       with torch.no_grad():
           prediction = model(patch.unsqueeze(0))

       return prediction.squeeze(0)

**How do I handle multi-modal data?**

.. code-block:: python

   def load_multimodal_patch(t1_path, t2_path, patch_offset, patch_size):
       t1 = medrs.load_cropped(t1_path, patch_offset, patch_size)
       t2 = medrs.load_cropped(t2_path, patch_offset, patch_size)

       # Stack into multi-channel tensor
       combined = torch.stack([t1.to_torch(), t2.to_torch()], dim=0)
       return combined

Contributing
-----------

**How can I contribute to medrs?**

1. **Report issues**: File bug reports on GitHub
2. **Request features**: Open feature requests with use cases
3. **Submit PRs**: Fork the repository and submit pull requests
4. **Improve docs**: Help improve documentation and examples

See the `development/contributing` guide for details.

**What are the development requirements?**

.. code-block:: bash

   # Clone repository
   git clone https://github.com/medrs/medrs.git
   cd medrs

   # Install development dependencies
   pip install -e ".[dev]"

   # Run tests
   pytest tests/

**How do I run benchmarks?**

.. code-block:: bash

   # Run Python benchmarks
   python examples/performance/benchmark_comparison.py

   # Run Rust benchmarks
   cargo bench

Support and Community
---------------------

**Where can I get help?**

- **GitHub Issues**: https://github.com/medrs/medrs/issues
- **GitHub Discussions**: https://github.com/medrs/medrs/discussions
- **Documentation**: https://medrs.readthedocs.io/
- **Email**: medrs@example.com

**Is medrs production ready?**

Yes! medrs includes:
- Comprehensive error handling
- Performance monitoring
- Production deployment guides
- Docker containers
- CI/CD pipelines
- Extensive testing

**What license does medrs use?**

medrs is available under the MIT OR Apache-2.0 license, allowing flexible usage in both academic and commercial projects.

---

Still have questions? Feel free to ask on our `GitHub Discussions <https://github.com/medrs/medrs/discussions>`_ or open an issue!
