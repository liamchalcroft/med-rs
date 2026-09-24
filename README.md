# medrs

[![CI](https://github.com/liamchalcroft/med-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/liamchalcroft/med-rs/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/medrs.svg)](https://pypi.org/project/medrs/)
[![crates.io](https://img.shields.io/crates/v/medrs.svg)](https://crates.io/crates/medrs)
[![docs.rs](https://img.shields.io/docsrs/medrs)](https://docs.rs/medrs)

Fast medical image I/O, transforms, and training patch loading for Python and
Rust.

- **Complete NIfTI support.** NIfTI-1 and NIfTI-2, `.nii`, `.nii.gz`, and
  `.hdr`/`.img` pairs, either byte order, all twelve integer and float types.
  Every header field and extension survives a load/save round trip. The test
  suite checks results against nibabel.
- **Read only what you need.** Uncompressed files are memory-mapped, and
  `load_cropped` reads a single patch without loading the volume.
  Block-compressed gzip (Mgzip or BGZF) decompresses in parallel. The chunked
  `.jvol` format decodes only the chunks a patch touches.
- **Correct geometry.** Every spatial transform updates the affine, so saved
  results stay aligned with the anatomy, and label maps keep their datatype.
- **Fused pipelines.** A `Pipeline` applies a chain of intensity transforms in
  one pass over the data and a chain of spatial transforms with a single
  interpolation.
- **A training loader.** `FastLoader` streams fixed-size patches from worker
  threads. It gives the same patch order with any number of workers, can centre
  patches on label foreground, applies augmentation to image and label together,
  and is safe to use in forked `DataLoader` workers.

## Install

```bash
pip install medrs           # Python 3.10+
cargo add medrs             # Rust; add `--features jvol` for .jvol support
```

## Python

```python
import medrs

img = medrs.load("t1.nii.gz")
print(img.shape, img.dtype, img.spacing, img.orientation)

# Transforms return new images and keep the affine consistent.
img = img.reorient("RAS").resample((1.0, 1.0, 1.0)).z_normalize()
img.save("t1_1mm.nii.gz")

array = img.to_numpy()  # indexed [x, y, z], shares memory when possible
tensor = img.to_torch()  # always a new tensor
patch = medrs.load_cropped("t1.nii", offset=(64, 64, 32), shape=(96, 96, 96))
```

Training patches with augmentation applied to image and label together:

```python
pipeline = (
    medrs.Pipeline()
    .clamp(-1000, 1000)
    .z_normalize()
    .random_flip(axes=(0, 1, 2), prob=0.5)
    .random_gaussian_noise(std=0.05)
)
loader = medrs.FastLoader(
    images,
    patch_shape=(96, 96, 96),
    labels=labels,
    patches_per_volume=4,
    foreground_prob=0.5,
    pipeline=pipeline,
    seed=0,
)
for epoch in range(100):
    for patch in loader:  # each pass is a new epoch
        x = patch.image.to_torch()[None]  # (1, 96, 96, 96)
        y = patch.label.to_torch()[None]
```

With MONAI, use medrs as the image reader:

```python
from monai.transforms import LoadImaged
from medrs.monai import MedrsReader

load = LoadImaged(keys=["image", "label"], reader=MedrsReader())
```

## Rust

```rust
use medrs::transforms::{Interpolation, Orientation};
use medrs::Pipeline;

let image = medrs::load("t1.nii.gz")?;
let pipeline = Pipeline::new()
    .reorient(Orientation::RAS)
    .resample_to_spacing([1.0, 1.0, 1.0], Interpolation::Trilinear)
    .z_normalize();
medrs::save(&pipeline.apply(&image)?, "t1_1mm.nii.gz")?;
```

See the [API documentation](https://docs.rs/medrs) for the full Rust API.

## Formats

| Format | Read | Write | Notes |
|---|---|---|---|
| `.nii` | ✓ | ✓ | Memory-mapped; crops read only the region. |
| `.nii.gz` | ✓ | ✓ | Gzip is detected from the file contents, not the name. |
| Mgzip / BGZF `.nii.gz` | ✓ | ✓ (Mgzip) | Standard gzip that medrs decompresses in parallel: `img.save(path, mgzip=True)` or `medrs convert --to mgzip`. |
| `.hdr` / `.img` | ✓ | ✓ | Either file may be gzipped. |
| `.jvol` | ✓ | ✓ | Chunked compression, lossless by default or lossy (wavelet); crops decode only the chunks they overlap. medrs-specific. |

`float16` and `bfloat16` images use datatype codes that only medrs reads; use
them for caches, not for sharing files with other tools.

## Performance

`python benchmarks/compare.py` times each job with every installed library,
each in a separate process, always producing materialized arrays. The results
below are for a 192³ float32 volume on a 4-CPU x86-64 cloud VM (median of 9
runs, milliseconds):

| Job | medrs | nibabel | SimpleITK |
|---|---:|---:|---:|
| Load `.nii` | 2.5 | 2.8 | 20.7 |
| Load `.nii.gz` | 49.7 | 110.8 | 36.6 |
| Load Mgzip `.nii.gz` | 29.2 | – | – |
| Load `.jvol` (lossless) | 36.0 | – | – |
| Read a 96³ patch from `.nii` | 3.1 | 7.9 | – |
| Read a 96³ patch from `.nii.gz` | 30.2 | 86.5 | – |
| Reorient, resample to 1.5 mm, z-normalize | 6.5 | 265.8 (with scipy) | – |

A standard `.nii.gz` file is a single deflate stream that can only be
decompressed on one core. If you load the same files repeatedly, convert them
once to Mgzip or `.jvol`. Both decompress in parallel, and `.jvol` is also
smaller and supports crop-first reads.

## Documentation

- [User guide](docs/guide.md): formats, dtype rules, transforms, pipelines,
  training, and performance tips
- [Rust API](https://docs.rs/medrs), including the `.jvol` file format
- [Changelog](CHANGELOG.md)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Licensed under either of [Apache License 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT), at your option.

The `.jvol` name and its wavelet approach come from Fernando Pérez-García's
[jvol](https://github.com/fepegar/jvol).
