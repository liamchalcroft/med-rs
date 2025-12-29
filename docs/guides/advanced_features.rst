Advanced Features
=================

This guide highlights utilities that build on top of the core loading and transform API.

Patch Sampling with ``TrainingDataLoader``
------------------------------------------
The Rust-backed loader extracts patches with caching and optional randomization.

.. code-block:: python

   import medrs

   loader = medrs.TrainingDataLoader(
       volumes=["vol1.nii.gz", "vol2.nii.gz"],
       patch_size=[64, 64, 64],
       patches_per_volume=4,
       patch_overlap=[0, 0, 0],
       randomize=True,
       cache_size=256,
   )

   patches = []
   try:
       while True:
           patches.append(loader.next_patch())
   except StopIteration:
       pass

   loader.reset()

Dictionary Transforms
---------------------
Use the pure-Python helpers to keep multi-modal dictionaries aligned when working with MONAI-style data structures.

.. code-block:: python

   from medrs.dictionary_transforms import SpatialNormalizer, CoordinatedCropLoader

   normalizer = SpatialNormalizer(target_orientation="RAS")
   crop_loader = CoordinatedCropLoader(
       keys=["image", "label"],
       crop_size=(96, 96, 96),
       randomize=True,
   )

   sample = {"image": "image.nii.gz", "label": "label.nii.gz"}
   sample = normalizer(sample)
   sample = crop_loader(sample)

Performance Profiling
---------------------
Track hotspots during training or preprocessing with ``PerformanceProfiler``.

.. code-block:: python

   from medrs.performance_profiler import PerformanceProfiler

   profiler = PerformanceProfiler()
   op_id = profiler.start_operation("load_patch")
   # Note: load_cropped requires uncompressed .nii files
   patch = medrs.load_cropped("brain.nii", [0, 0, 0], [64, 64, 64])
   profiler.end_operation(op_id, "load_patch")
   print(profiler.get_summary()["load_patch"])

Error Handling Helpers
----------------------
Wrap common patterns with explicit error handling to keep failures clear.

.. code-block:: python

   import medrs
   from medrs import LoadError

   try:
       image = medrs.load("brain.nii.gz")
   except LoadError as exc:
       raise RuntimeError(f"Failed to load brain.nii.gz") from exc
