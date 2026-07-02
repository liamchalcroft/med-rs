
Benchmarks
==========

Performance measurements comparing medrs against nibabel, MONAI, TorchIO, and SimpleITK for common medical imaging operations, and the numbers behind medrs's own mixed-precision storage and FastLoader features.

.. _load-performance:

Single File Loading
--------------------

medrs memory-maps uncompressed ``.nii`` files, so opening and materializing a volume is comparable to nibabel (which also memory-maps) and roughly 10-25x faster than MONAI and TorchIO, whose default loaders eagerly read the data and build a tensor. For gzipped ``.nii.gz``, load time is bounded by the decompressor: medrs is competitive with nibabel and MONAI, while SimpleITK and TorchIO decompress faster than medrs's single-threaded path (the parallel Mgzip format below is medrs's fast path for compressed data).

The table below times load plus full materialization to a numpy array, so every library does the same work. Multiples are relative to medrs.

.. list-table::
   :header-rows: 1
   :widths: 12 14 12 12 12 12 14

   * - Volume
     - Format
     - medrs
     - nibabel
     - MONAI
     - TorchIO
     - SimpleITK
   * - 128³
     - .nii
     - 1.3 ms
     - 1.4x
     - 14x
     - 9x
     - 4x
   * - 256³
     - .nii
     - 17 ms
     - 1.4x
     - 23x
     - 13x
     - 4x
   * - 128³
     - .nii.gz
     - 82 ms
     - 0.6x
     - 1.2x
     - 0.2x
     - 0.3x
   * - 256³
     - .nii.gz
     - 370 ms
     - 1.8x
     - 2.5x
     - 0.5x
     - 0.1x

*Measured on Apple Silicon with* ``benchmarks/bench_load_comparison.py`` *(float32, structured data). For uncompressed volumes medrs and nibabel both memory-map, so they are close; the large multiples are against loaders that materialize eagerly.*

.. _mgzip-scaling:

Mgzip Thread Scaling
---------------------

Mgzip (multi-member gzip) splits a file into independently-decompressible blocks. Its own thread-scaling numbers, measured against medrs's single-threaded libdeflate baseline on a 256³ volume:

.. list-table::
   :header-rows: 1
   :widths: 20 20 30 30

   * - Threads
     - Time (ms)
     - Speedup vs 1 thread
     - Speedup vs libdeflate baseline
   * - 1
     - 206
     - 1.0x
     - 0.7x (slightly slower)
   * - 2
     - 111
     - 1.9x
     - 1.2x
   * - 4
     - 62
     - 3.3x
     - 2.2x
   * - 8
     - 44
     - 4.7x
     - 3.0x

Single-threaded Mgzip is slower than the libdeflate baseline (channel and per-block buffer overhead dominate); the crossover is around 2-3 threads. This is a known limitation, not a bug; a custom parallel decompressor is planned.

.. _storage-efficiency:

Storage Efficiency
-------------------

medrs supports mixed-precision storage with bf16/f16 for 40-50% file size reduction.

File Sizes (128³ Volume, Compressed)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25

   * - Data Type
     - File Size
     - vs float32
     - Read Speed (MB/s)
   * - float32
     - 8.3 MB
     - 100%
     - 147
   * - **bfloat16**
     - **3.4 MB**
     - **41%**
     - 100
   * - **float16**
     - **4.1 MB**
     - **50%**
     - 174
   * - int16
     - 1.2 MB
     - 15%
     - 47

Precision vs Error Trade-off
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For normalized data in [0, 1]:

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25

   * - Format
     - Max Error
     - Mean Error
     - Recommended Use
   * - bfloat16
     - 0.004
     - 0.0008
     - Training
   * - float16
     - 0.001
     - 0.0002
     - Inference
   * - int16
     - 0.00003
     - 0.000008
     - Storage/archival

.. _fastloader-throughput:

Training Throughput (FastLoader)
--------------------------------

The FastLoader prefetches patches across parallel worker threads with the GIL released, so patch loading overlaps decompression and disk I/O. On 64³ random crops from gzipped volumes it delivers roughly 4x the throughput of a naive sequential load-and-crop loop.

.. list-table::
   :header-rows: 1
   :widths: 40 20 20 20

   * - Loader
     - Workers
     - Patches/sec
     - vs naive
   * - medrs FastLoader
     - 4
     - 191
     - 3.9x
   * - naive sequential load + crop
     - 1
     - 49
     - 1x

Measured on Apple Silicon with structured 128³ volumes. Configuring a fair third-party DataLoader baseline (persistent workers, cached datasets) is workload-dependent; use ``benchmarks/bench_fastloader.py`` to compare against your own pipeline.

.. _running-benchmarks:

Reproducing Benchmarks
------------------------

``benchmarks/results/`` is not checked into the repository (it is regenerated output, not a source of truth). Run the suites yourself to get numbers for your own hardware:

.. code-block:: bash

   pip install -e ".[examples]"

   # Individual suites
   python benchmarks/bench_medrs.py
   python benchmarks/bench_nibabel.py
   python benchmarks/bench_monai.py
   python benchmarks/bench_torchio.py
   python benchmarks/bench_mgzip.py
   python benchmarks/bench_fastloader.py

   # Rust-side microbenchmarks
   cargo bench

   # Combined comparison report and plots
   python benchmarks/compare_all.py
   python benchmarks/plot_results.py

See ``benchmarks/BENCHMARK_PLAN.md`` for the full benchmark matrix (libraries, volume sizes, dtypes, formats).

.. _methodology:

Methodology Notes
-------------------

- medrs's uncompressed-``.nii`` load path memory-maps the file and parses the header; it does not eagerly copy voxel data into a fresh buffer. This is the main reason the uncompressed-``.nii`` numbers above look different in character from the ``.nii.gz`` numbers, where both sides decompress the full volume.
- The Mgzip thread-scaling table isolates Mgzip's own multi-threading efficiency against medrs's single-threaded libdeflate baseline; it is a different comparison from the load-plus-materialize numbers in the single-file-loading table above.
- Crop-first loading benefits scale with the ratio of patch size to volume size; a small patch pulled from a large volume benefits more than a patch close to the full volume size.
- Results are from synthetic data on Apple Silicon; real medical imaging data and different hardware will vary.
