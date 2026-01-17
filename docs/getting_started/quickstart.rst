Quick Start Guide
=================

This guide walks through the basics of loading NIfTI data with medrs and using the Python bindings in a training pipeline.

Prerequisites
-------------
Install medrs following :doc:`installation` and ensure ``numpy`` is available. GPU support is optional but recommended for the PyTorch examples.

Load a Volume
-------------
.. code-block:: python

   import medrs

   img = medrs.load("brain.nii.gz")
   print("shape:", img.shape())
   print("spacing:", img.spacing())

   # Get a numpy array (copy by default)
   array = img.to_numpy(copy=True)
   print("dtype:", array.dtype)

Crop-First Loading
------------------
.. code-block:: python

   from medrs import load_cropped_to_torch
   import torch

   patch = load_cropped_to_torch(
       "brain.nii.gz",
       output_shape=[64, 64, 64],
       device="cuda" if torch.cuda.is_available() else "cpu",
       dtype=torch.float16,
   )
   print("patch:", patch.shape, patch.device, patch.dtype)

Common Transforms
-----------------
.. code-block:: python

   from medrs import z_normalization, rescale_intensity, resample, reorient, crop_or_pad

   img = medrs.load("brain.nii.gz")
   normalized = z_normalization(img)
   rescaled = rescale_intensity(normalized, 0.0, 1.0)
   iso = resample(rescaled, [1.0, 1.0, 1.0])
   ras = reorient(iso, "RAS")
   patched = crop_or_pad(ras, [128, 128, 128])

PyTorch DataLoader Example
--------------------------
.. code-block:: python

   import medrs
   import torch
   from torch.utils.data import DataLoader, Dataset

   class PatchDataset(Dataset):
       def __init__(self, volume_paths, patch_size=(64, 64, 64)):
           self.volume_paths = volume_paths
           self.patch_size = patch_size

       def __len__(self):
           return len(self.volume_paths)

       def __getitem__(self, idx):
           return medrs.load_cropped_to_torch(
               self.volume_paths[idx],
               output_shape=list(self.patch_size),
               device="cuda" if torch.cuda.is_available() else "cpu",
               dtype=torch.float16,
           )

   dataset = PatchDataset(["vol1.nii.gz", "vol2.nii.gz"])
   loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)

   for batch in loader:
       # Training step using tensors already on the right device
       pass

Next Steps
----------
- See the examples in ``examples/basic`` for more end-to-end scripts.
- The :doc:`../guides/advanced_features` guide covers asynchronous pipelines and memory reuse.
- The :doc:`../api` page documents all available functions and classes.
