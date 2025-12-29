Changelog
=========

Version 0.1.2
-------------
- Fixed potential panic when patch size exceeds volume dimensions in ``CropLoader`` and ``TrainingDataLoader``.
- Added dimension overflow validation in ``crop_or_pad`` and ``rotate_90`` transforms.
- Added regression tests for boundary condition handling.

Version 0.1.1
-------------
- Updated to F-order array handling throughout for NIfTI compatibility.
- Various bug fixes and performance improvements.

Version 0.1.0
-------------
- Initial public release.
- Rust NIfTI I/O with crop-first loading and save support.
- Python bindings for loading, transforms, and patch-based training with ``TrainingDataLoader``.
- Removed deprecated ``PyTrainingDataLoader`` alias; use ``TrainingDataLoader``.
- Dictionary transform helpers for multi-modal datasets.
- Performance profiling utilities.
