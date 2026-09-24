# medrs user guide

This guide covers the Python API. The Rust API mirrors it; see
[docs.rs/medrs](https://docs.rs/medrs).

- [Loading and saving](#loading-and-saving)
- [Images and arrays](#images-and-arrays)
- [Transforms](#transforms)
- [Pipelines](#pipelines)
- [Training](#training)
- [Performance](#performance)
- [Errors](#errors)

## Loading and saving

```python
img = medrs.load("scan.nii.gz")
img.save("copy.nii")
```

The format is chosen from the file name when saving, and from the file
contents when loading. A gzipped file named `.nii` still loads.

| Name | Saved as |
|---|---|
| `.nii` | Uncompressed single file |
| `.nii.gz` | gzip; `compression_level=0..9` (default 3), `mgzip=True` for block gzip |
| `.hdr`, `.img` (optionally `.gz`) | Two-file pair |
| `.jvol` | Chunked compression; `quality=1..100` for lossy, `chunk_shape=(x, y, z)` |

Saving writes a temporary file and renames it over the destination. An
interrupted save never leaves a partial file, and you can save over the file
an image was loaded from.

Headers are dictionaries. `load_header` reads only the header:

```python
h = medrs.load_header("scan.nii.gz")
h["shape"], h["affine"], h["sform_code"], h["descrip"], h["extensions"]
img = img.with_header(descrip="skull-stripped", intent_code=0)
img = img.with_affine(new_affine)
```

Every field and extension is written back on save. The shape, datatype, and
affine are properties of the image, so they are changed through
`with_dtype`, `with_affine`, and `with_data` rather than `with_header`.

**Crop-first reads.** `load_cropped(path, offset, shape)` returns `shape`
voxels starting at `offset`, keeping all volumes of 4D images. From `.nii` it
reads only those bytes, and from `.jvol` it decodes only the overlapping
chunks. Gzipped files have to be decompressed in full, so the decompressed
data is cached (see `set_cache_limits`) and further crops of the same file are
fast. `load(path, cache=True)` uses the same cache.

**Aligning images.** `load_multi` resamples several files onto the grid of a
reference, for example a segmentation onto its MRI. Files already on that grid
are returned as loaded:

```python
t1, flair, seg = medrs.load_multi(
    ["t1.nii.gz", "flair.nii.gz", "seg.nii.gz"],
    reference=0,
    interpolation=["trilinear", "trilinear", "nearest"],
)
```

## Images and arrays

Images are immutable. Arrays are indexed `[x, y, z, ...]` in the file's
(Fortran) memory order, like nibabel.

`to_numpy`, `to_torch`, `to_jax`, and `numpy.asarray(img)` all follow one
rule when no `dtype` is given:

- if the header has no scaling (`scl_slope`/`scl_inter`), you get the stored
  datatype;
- if it does, you get scaled `float32` values.

This is the same rule as `numpy.asarray(nibabel_image.dataobj)`. Pass `dtype`
to convert. Integers are rounded half to even and saturated.

| Method | Memory |
|---|---|
| `to_numpy()` | Shares the image's memory when it can; the array is then read-only |
| `to_numpy(copy=True)` | Always a new, writable array |
| `to_torch(dtype=None, device=None)` | Always a new tensor. `uint16/32/64` become `int32/int64` |
| `to_jax(dtype=None, device=None)` | JAX array; `device` is a `jax.Device` or a name such as `"gpu:0"` |

`bfloat16` works with PyTorch and JAX. NumPy has no bfloat16 type, so
`to_numpy()` returns `float32` for bfloat16 images unless you request
`ml_dtypes.bfloat16`.

## Transforms

Every transform returns a new image, following these conventions:

- **Spatial transforms preserve world position.** `crop`, `crop_or_pad`,
  `flip`, `rotate_90`, `reorient`, and all resampling update the affine. The
  voxel array changes, but each voxel keeps its position in scanner space, so
  saved results overlay the original in any viewer. This matches MONAI's
  `MetaTensor` behaviour.
- **Exact transforms keep the datatype.** Crops, flips, rotations,
  reorientation, and nearest-neighbour resampling copy stored values. A
  `uint8` label map stays `uint8`. Trilinear resampling produces `float32`.
- **Intensity transforms produce `float32`** in scaled units: `clamp`,
  `rescale`, `z_normalize`, `adjust_gamma`.
- **4D images** are transformed volume by volume along the first three axes.

| Method | Notes |
|---|---|
| `resample(spacing, interpolation)` | Keeps the field of view; `"trilinear"` or `"nearest"` |
| `resample_to_shape(shape, interpolation)` | Voxel centres follow the ITK/MONAI convention |
| `resample_like(reference, interpolation)` | Onto another image's grid, through both affines |
| `reorient("RAS")` | Exact permutation and flips; any of the 48 codes |
| `crop(offset, shape)`, `crop_or_pad(shape, pad_value)` | Centred for `crop_or_pad` |
| `flip(axes)`, `rotate_90(axes, k)` | `rotate_90` matches `numpy.rot90` |
| `clamp(min, max)`, `rescale(out_min, out_max)` | |
| `z_normalize(nonzero=False)` | `nonzero=True` uses only non-zero voxels and keeps zeros |
| `adjust_gamma(gamma)` | Preserves the value range (MONAI `AdjustContrast`) |
| `with_dtype(dtype)` | Applies and then resets scaling |

When resampling, target voxels whose centres fall outside the source image
are 0: stored 0 for nearest-neighbour, 0.0 for trilinear.

## Pipelines

A `Pipeline` records transforms and applies them with as few passes over the
data as possible:

```python
pipeline = (
    medrs.Pipeline()
    .reorient("RAS")
    .resample_to_spacing((1.0, 1.0, 1.0))
    .crop_or_pad((160, 192, 160))
    .clamp(-1000, 1000)
    .z_normalize()
    .random_flip(axes=(0,), prob=0.5)
    .random_intensity_scale(0.1)
)
out = pipeline.apply(img, seed=0)
out_img, out_seg = pipeline.apply(img, seg, seed=0)
```

- Consecutive intensity steps (clamps, rescaling, z-normalization, random
  scale and shift) become one pass. Steps that need statistics compute them
  from the pending values, without writing an intermediate image.
- Consecutive spatial steps become one grid change, so the image is
  interpolated once, from the original voxels.
- The result matches applying the transforms one at a time, apart from
  interpolating once instead of several times.
- Random steps draw their parameters on each `apply`. Pass `seed` for
  reproducible results.
- With a label, spatial steps use the same draws for both. The label uses
  nearest-neighbour interpolation and zero padding, and intensity steps skip
  it.
- Pipelines are immutable (each method returns a new pipeline), validate
  their arguments immediately, and can be pickled.

## Training

`FastLoader` produces fixed-size patches from a list of volumes, optionally
with label maps:

```python
loader = medrs.FastLoader(
    images,
    patch_shape=(96, 96, 96),
    labels=labels,  # optional, same order as images
    patches_per_volume=4,
    foreground_prob=0.5,  # needs labels; centre patches on label > 0
    pipeline=pipeline,  # runs in the worker threads
    workers=None,  # default: CPUs, at most 8; 0 = in this thread
    prefetch=None,  # patches in flight, default 2 * workers
    shuffle=True,
    seed=0,
)
for patch in loader:  # patch.image, patch.label, patch.volume, patch.offset
    ...
```

- Each pass over the loader is a new epoch with a new shuffle and new
  patches. `loader.epoch(n)` replays epoch `n`.
- For a given seed, the sequence of patches does not depend on `workers`.
- Volumes smaller than the patch are padded (images with `pad_value`,
  labels with 0), and `patch.shape` is the region actually read.
- A file that fails to load raises when its patches are reached; later files
  are still loaded if you keep iterating.
- An epoch iterator can be closed early with `close()`, or used as a context
  manager.

To feed a PyTorch `DataLoader`, wrap the loader in an `IterableDataset` and
let medrs do the parallel work:

```python
class Patches(torch.utils.data.IterableDataset):
    def __init__(self, loader):
        self.loader = loader

    def __iter__(self):
        for p in self.loader:
            yield p.image.to_torch()[None], p.label.to_torch()[None]


data = torch.utils.data.DataLoader(Patches(loader), batch_size=4, num_workers=0)
```

You can also call medrs inside `DataLoader` worker processes. It rebuilds its
thread pool after `fork()`, so forked workers do not hang. Use
`medrs.set_num_threads` to avoid oversubscribing the CPUs.

**MONAI.** `medrs.monai.MedrsReader` is a MONAI `ImageReader`. It produces
the same arrays as `NibabelReader`, with the same channel handling and the
metadata keys MONAI's transforms use, so it works with `LoadImage`,
`LoadImaged`, and the rest of MONAI:

```python
from medrs.monai import MedrsReader

LoadImaged(keys=["image", "label"], reader=MedrsReader())
```

## Performance

- **Uncompressed files are fastest.** They are memory-mapped, and patches
  read only their own bytes.
- **Standard `.nii.gz` decompresses on one core.** For datasets you read many
  times, convert once:

  ```bash
  medrs convert data/*.nii.gz --to mgzip -o data_mgzip/   # still standard .nii.gz
  medrs convert data/*.nii.gz --to jvol -o data_jvol/     # smaller, crop-first
  ```

- **`.jvol` lossless** is exact for every datatype, was 3–25% smaller than
  `.nii.gz` in our tests, and decompresses in parallel. **Lossy** `.jvol`
  (`quality=1..100`) is for intensity images, not label maps. It is 15 to 200
  times smaller than the uncompressed image, depending on `quality`. Values
  are clamped to their original range, background voxels stay exact, and
  decoded images are `float32`.
- **Threads.** medrs uses one thread per CPU by default;
  `medrs.set_num_threads(n)` changes it for the whole process.

## Errors

| Exception | When |
|---|---|
| `FileNotFoundError`, `PermissionError`, other `OSError` | File system errors |
| `medrs.FormatError` (a `ValueError`) | A file is not a valid image, or is corrupt or truncated |
| `ValueError` | Invalid arguments or unsuitable data (for example NaN when z-normalizing) |
| `TypeError` | Unsupported array dtypes |
