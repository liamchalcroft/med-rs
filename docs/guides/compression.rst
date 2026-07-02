
Compression Formats
====================

medrs reads and writes three volume formats. This guide covers what each is for, and the tradeoffs of the newer ``.jvol`` codec in particular.

.. list-table::
   :header-rows: 1
   :widths: 15 25 60

   * - Format
     - Type
     - Best for
   * - ``.nii``
     - Uncompressed
     - Fastest load (memory-mapped), byte-exact crop-first reads via ``load_cropped``
   * - ``.nii.gz``
     - gzip
     - Standard interchange; use ``load_mgzip`` / the Mgzip format for parallel decode on multi-core machines
   * - ``.jvol``
     - Wavelet + Rice coding (optional ``jvol`` feature)
     - Storage- and bandwidth-bound workflows on floating-point volumes

``.jvol``: wavelet volumetric compression
------------------------------------------

``.jvol`` is an optional codec vendored from `jvol-rust <https://github.com/fepegar/jvol-rust>`_ by Fernando Pérez-García (MIT licensed; see :ref:`compression-credits` below). It is built on a wavelet transform (LeGall 5/3 lifting) followed by Rice entropy coding of the subbands, and supports two encoding modes.

Lossy mode
~~~~~~~~~~

Lossy encoding (``quality=1..100``, higher is better) is what ``.jvol`` is designed for: wavelet quantization of floating-point intensity volumes, typically **10x to 500x smaller** than the source depending on the chosen quality. This is the mode to reach for when you are storage- or bandwidth-bound and can tolerate a controlled reconstruction error, for example archiving preprocessed training volumes or shipping data over a network.

Lossy encoding is **rejected for integer/label dtypes**. medrs returns an error rather than silently encoding a segmentation volume lossily, since wavelet quantization would corrupt discrete label values:

.. code-block:: python

   import medrs

   seg = medrs.load("segmentation.nii.gz")
   medrs.save_jvol(seg, "segmentation.jvol", quality=60)
   # raises: jvol lossy encoding is not supported for integer dtype ...

Use ``lossless=True`` for label/segmentation volumes (see below).

Lossless mode
~~~~~~~~~~~~~

Lossless ``.jvol`` gives an exact round trip, but the tradeoffs run the other way: file size is roughly gzip-parity (no material storage win over ``.nii.gz``), and decode is more CPU-intensive than gzip or Mgzip. If load speed matters more than storage, prefer ``.nii.gz`` or ``.nii.mgz`` (Mgzip) over lossless ``.jvol``. Lossless mode exists mainly so label volumes can share the same container format as their paired intensity volumes when that's convenient, not because it is the fastest way to store or load them.

Decode performance
------------------

Lossy decode cost is dominated by the inverse wavelet lifting and the Rice
entropy decode, both applied over the full voxel grid. Two decode paths cut that
cost.

**f32 fast path.** Lossy coefficients are already quantized, so a full-precision
f64 reconstruction is not needed to represent them faithfully. When the target
dtype is any non-f64 float (``float32``, ``float16``, ``bfloat16``), medrs
decodes entirely in f32: coefficients, dequantization, the inverse wavelet
lifting, and denormalization all stay in single precision, and the f64
intermediate array is skipped. On a 256\ :sup:`3` ``float32`` volume at
``quality=60`` this measured **1.8x faster** decode (203 ms to 112 ms, release
build), with the reconstruction matching the f64 path to well under 0.1% of the
value range. Lossless volumes and f64 targets keep the exact f64 path.

Progressive (multiresolution) decode
-------------------------------------

A multi-level 3D wavelet decomposition stores a small coarse approximation
subband plus detail subbands at each level. The finest detail subbands are the
largest and most numerous: at 256\ :sup:`3` with six levels, the level-0 detail
alone holds 7/8 of all coefficients. A reduced-resolution reconstruction does
not need them.

``load_downsampled(path, factor)`` in Rust, and ``load_jvol_downsampled(path,
factor)`` in Python, decode a lossy file at ``1 / factor`` of its stored
resolution per axis, where ``factor`` is a power of two no larger than
``2**levels``. Only the subbands at levels ``>= log2(factor)`` are Rice-decoded;
the finer levels are skipped entirely, and the inverse wavelet lifting runs for
only the coarse remaining levels. The result is a low-pass approximation of the
volume resampled by ``factor``.

Because the skipped subbands carry the bulk of the coefficients and the bulk of
the cost, the speedup is large:

.. list-table::
   :header-rows: 1
   :widths: 22 22 22 34

   * - Factor
     - Output shape
     - Decode time
     - Speedup vs full f64
   * - 1 (full, f64)
     - 256\ :sup:`3`
     - 203 ms
     - 1.0x
   * - 1 (full, f32)
     - 256\ :sup:`3`
     - 112 ms
     - 1.8x
   * - 2
     - 128\ :sup:`3`
     - 15 ms
     - 13x
   * - 4
     - 64\ :sup:`3`
     - 2.6 ms
     - 79x

(256\ :sup:`3` ``float32`` volume, ``quality=60``, release build, mean of five
runs.) Peak memory scales with the output grid, so a factor-4 preview also holds
roughly 1/64 of the working set of a full decode.

The returned image stays registered with the full-resolution volume: spacing is
scaled by ``factor`` and the origin is shifted by half a downsampled voxel, so
the new voxel ``(0,0,0)`` sits at the centre of the block of ``factor`` original
voxels it summarizes. The intensity scale is preserved by dividing out the
low-pass gain of the skipped levels, so a downsampled preview keeps the same
mean as the full-resolution decode.

Progressive decode is a lossy-only feature. Lossless files store a single
delta-coded block with no multiresolution subbands; calling the downsampled
loader on one returns an error.

Decode-as-dtype (mixed-precision decode)
-----------------------------------------

``load_as(path, dtype)`` in Rust, and the ``dtype`` argument on
``load_jvol``/``load_jvol_downsampled``/``load_jvol_cached`` in Python, decode
directly to a dtype other than the one the file was saved with. This is
useful for mixed-precision pipelines: a volume is stored once as a compact
lossy ``.jvol`` file and decoded straight to ``bfloat16``/``float16`` at train
time, without ever materializing the full-precision intermediate array.

A float target (``float32``, ``float16``, ``bfloat16``, ``float64``) is
always accepted. When the target is a non-f64 float, the f32 fast decode path
(see above) is used regardless of the file's stored dtype, since the decoded
coefficients are already quantized. Overriding to an integer dtype rounds the
decoded value to the nearest representable integer; this only makes sense for
lossless files, or when the caller already accepts the wavelet quantization
error and additionally wants the value snapped onto an integer grid.
Lossy-encoding of integer *source* data at save time is still rejected,
unaffected by this decode-time override.

``dtype=None`` reproduces the file's stored dtype, matching plain ``load``.

Decoded-image cache
--------------------

``load_cached(path, dtype)`` in Rust, and ``load_jvol_cached(path, dtype)`` in
Python, decode a ``.jvol`` file once per ``(path, dtype)`` pair and reuse the
result on subsequent calls, mirroring ``nifti::load_cached`` for gzip. This is
aimed at training pipelines that revisit the same volume across epochs: the
first call pays the wavelet/entropy decode cost, later calls only pay an
array copy out of the cache.

The cache key includes the requested output dtype, so caching the same file
at two dtypes (say ``float32`` for one branch of a model and ``bfloat16`` for
another) decodes and caches each independently rather than one silently
shadowing the other. A change to the underlying file on disk (size or
modification time) invalidates its entry, the same file-stamp mechanism used
by the gzip decompression cache.

``set_jvol_cache_size(n)`` controls the cache's maximum entry count (default
10; 0 disables caching), and ``clear_jvol_cache()`` frees everything it
holds.

Decode-once-to-mmap transcoding
---------------------------------

``.jvol``'s wavelet decode is CPU work that repeats on every load. For a
training loop that reloads the same volumes epoch after epoch,
``transcode_to_nii(jvol_path, out_path, dtype)`` in Rust, and
``convert_jvol_to_nii`` in Python, decode once and write the result out as an
uncompressed ``.nii`` file, so a later ``nifti::load`` of that file memory-maps
it directly instead of decoding again.

``load_via_mmap_cache(jvol_path, cache_dir, dtype)`` (Rust) /
``load_jvol_via_mmap_cache`` (Python) wraps this into a single call: it
transcodes to ``<cache_dir>/<stem>.nii`` on first use, skips the transcode
step when a cached ``.nii`` already exists and is newer than the source
``.jvol`` file, and returns the mmap-backed ``.nii`` load either way. A dtype
override produces a distinctly named cache file, so caching the same volume
at two dtypes does not collide.

The output path (or the file this function writes into ``cache_dir``) must be
an uncompressed ``.nii``; passing a ``.gz`` or ``.jvol`` output path is
rejected, since either would defeat the zero-copy mmap this function exists
to provide.

Usage
-----

Python
~~~~~~

.. code-block:: python

   import medrs

   img = medrs.load("brain.nii.gz")

   # Lossy, tuned for floating-point intensity volumes
   medrs.save_jvol(img, "brain.jvol", quality=60)

   # Lossless (required for label/segmentation volumes)
   seg = medrs.load("segmentation.nii.gz")
   medrs.save_jvol(seg, "segmentation.jvol", lossless=True)

   # .jvol loads transparently through the normal load() entry point,
   # dispatched by the .jvol extension
   restored = medrs.load("brain.jvol")

   # Fast reduced-resolution preview of a lossy file (progressive decode)
   preview = medrs.load_jvol_downsampled("brain.jvol", 4)  # ~1/4 shape per axis

   # Convert an existing NIfTI file directly
   medrs.convert_to_jvol("brain.nii.gz", "brain.jvol", quality=60)
   medrs.convert_to_jvol("segmentation.nii.gz", "segmentation.jvol", lossless=True)

   # Decode directly to a mixed-precision dtype, skipping the float32 intermediate
   bf16_img = medrs.load_jvol("brain.jvol", dtype="bfloat16")

   # Decode once, reuse across epochs (cache key includes the output dtype)
   medrs.set_jvol_cache_size(50)
   for _ in range(num_epochs):
       img = medrs.load_jvol_cached("brain.jvol", dtype="bfloat16")
   medrs.clear_jvol_cache()

   # Decode once to an uncompressed .nii for zero-copy mmap on every later load
   medrs.convert_jvol_to_nii("brain.jvol", "brain_cache.nii", dtype="float32")
   fast_img = medrs.load("brain_cache.nii")  # zero-copy mmap

   # Or let the mmap cache manage the transcode-and-reuse cycle
   fast_img = medrs.load_jvol_via_mmap_cache("brain.jvol", "/tmp/jvol_cache")

Rust
~~~~

.. code-block:: rust

   use medrs::jvol::{self, JvolOptions};

   let img = medrs::nifti::load("brain.nii.gz")?;

   // Lossy at quality 60
   jvol::save(&img, "brain.jvol", JvolOptions::lossy(60))?;

   // Lossless
   jvol::save(&img, "brain_lossless.jvol", JvolOptions::lossless())?;

   // medrs::nifti::load / save also dispatch to jvol automatically for
   // paths ending in .jvol
   let restored = medrs::nifti::load("brain.jvol")?;

   // Fast reduced-resolution preview of a lossy file (progressive decode)
   let preview = medrs::jvol::load_downsampled("brain.jvol", 4)?;

   // Decode directly to a mixed-precision dtype, skipping the float32 intermediate
   use medrs::nifti::DataType;
   let bf16_img = jvol::load_as("brain.jvol", Some(DataType::BFloat16))?;

   // Decode once, reuse across epochs (cache key includes the output dtype)
   jvol::set_jvol_cache_size(50);
   let cached = jvol::load_cached("brain.jvol", Some(DataType::BFloat16))?;
   jvol::clear_jvol_cache();

   // Decode once to an uncompressed .nii for zero-copy mmap on every later load
   jvol::transcode_to_nii("brain.jvol", "brain_cache.nii", Some(DataType::Float32))?;
   let fast_img = medrs::nifti::load("brain_cache.nii")?; // zero-copy mmap

   // Or let the mmap cache manage the transcode-and-reuse cycle
   let fast_img = jvol::load_via_mmap_cache("brain.jvol", "/tmp/jvol_cache", None)?;

Uncompressed mixed precision as an alternative
----------------------------------------------

Compression and memory-mapping are mutually exclusive: a ``.jvol`` (or ``.nii.gz``)
file is an encoded bitstream, not a voxel array, so its data can never be
memory-mapped with zero copy. When the goal is a smaller file that still loads at
true mmap speed, the simplest option is to store the volume uncompressed in a
lower-precision dtype rather than compressing it.

medrs memory-maps any NIfTI dtype, so a volume saved as ``bfloat16`` or ``int16``
is a real zero-copy mmap that is half the size of ``float32``, with only an upcast
at patch-extraction time (microseconds) instead of a decode:

.. code-block:: python

   import medrs

   # Half-size file, still true zero-copy mmap on load
   medrs.load("brain.nii").with_dtype("bfloat16").save("brain_bf16.nii")
   img = medrs.load("brain_bf16.nii")   # memory-mapped, no decode
   assert img.can_zero_copy()

Guidance:

- Use uncompressed ``bf16``/``int16`` when load latency and zero-copy access
  matter more than disk footprint, for example the hot training set on a fast
  local disk. It is 2x smaller than ``f32`` and keeps mmap speed.
- Use lossy ``.jvol`` when disk footprint or transfer bandwidth dominates (cloud
  storage, very large archives, network file systems); it is 20-600x smaller but
  pays a decode cost per load.
- The two combine: keep the archive as ``.jvol`` and transcode to an uncompressed
  ``bf16`` ``.nii`` cache on first use with ``load_via_mmap_cache`` for
  zero-copy access during training.

Building with ``.jvol`` support
--------------------------------

``.jvol`` is behind the ``jvol`` Cargo feature:

.. code-block:: bash

   cargo build --features jvol
   cargo test --features jvol

The ``python`` feature enables ``jvol`` automatically, so the published Python wheel includes ``.jvol`` support with no extra install step.

.. _compression-credits:

Attribution
-----------

The ``.jvol`` codec (wavelet lifting, entropy coding, subband management, and type definitions under ``src/jvol/codec/``) is vendored from `jvol-rust <https://github.com/fepegar/jvol-rust>`_ by `Fernando Pérez-García <https://github.com/fepegar>`_, MIT licensed. medrs vendors only the codec modules; the NIfTI I/O front-end and Python bindings in ``src/jvol/mod.rs`` and ``src/python/jvol.rs`` are medrs's own, built on top of that codec. The full upstream license text is bundled at ``LICENSE-jvol`` in the repository root.
