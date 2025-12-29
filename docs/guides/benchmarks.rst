Benchmarks
==========

Comprehensive performance benchmarks comparing medrs against MONAI and TorchIO across common medical imaging operations.

.. _benchmark-summary:

Benchmark Summary
-----------------

.. image:: /_static/benchmark_summary.png
   :alt: Benchmark summary plot showing medrs, MONAI, and TorchIO performance
   :align: center
   :width: 100%

*Figure: Median execution time (ms) vs volume size for common operations. Lower is better.*

Key findings at 512³ volume size:

- **load**: medrs ~38,000x faster than MONAI, ~6,600x faster than TorchIO
- **load_cropped**: medrs ~6,600x faster than MONAI, ~1,400x faster than TorchIO
- **load_resampled**: medrs ~900x faster than MONAI, ~600x faster than TorchIO
- **load_cropped_to_torch**: medrs ~7,000x faster than MONAI, ~1,400x faster than TorchIO

The speedup advantage increases dramatically with volume size due to medrs's O(log n) scaling vs O(n) for Python-based libraries.

.. _load-performance:

Load Performance
----------------

Basic NIfTI file loading without transformations.

.. list-table::
   :header-rows: 1
   :widths: 15 15 15 15 15 15

   * - Size
     - medrs (ms)
     - MONAI (ms)
     - TorchIO (ms)
     - vs MONAI
     - vs TorchIO
   * - 64³
     - 0.13
     - 1.34
     - 2.35
     - **10x**
     - **18x**
   * - 128³
     - 0.13
     - 4.55
     - 4.71
     - **35x**
     - **36x**
   * - 256³
     - 0.14
     - 159.11
     - 95.18
     - **1,136x**
     - **680x**
   * - 512³
     - 0.13
     - 5,006.76
     - 866.54
     - **38,513x**
     - **6,665x**

.. _crop-first-loading:

Crop-First Loading
------------------

Loading only a cropped region (64³ patch) without reading the entire volume.

.. list-table::
   :header-rows: 1
   :widths: 15 15 15 15 15 15

   * - Source Size
     - medrs (ms)
     - MONAI (ms)
     - TorchIO (ms)
     - vs MONAI
     - vs TorchIO
   * - 64³
     - 0.27
     - 1.75
     - 6.00
     - **6x**
     - **22x**
   * - 128³
     - 0.41
     - 4.68
     - 9.86
     - **11x**
     - **24x**
   * - 256³
     - 0.55
     - 154.86
     - 104.48
     - **282x**
     - **190x**
   * - 512³
     - 0.76
     - 5,041.42
     - 1,076.89
     - **6,633x**
     - **1,417x**

This is the key differentiator for training pipelines where you extract random patches from large volumes.

.. _resampling:

Load Resampled
---------------

Loading with resampling to half resolution.

.. list-table::
   :header-rows: 1
   :widths: 15 15 15 15 15 15

   * - Source Size
     - medrs (ms)
     - MONAI (ms)
     - TorchIO (ms)
     - vs MONAI
     - vs TorchIO
   * - 64³ → 32³
     - 0.18
     - 1.93
     - 5.45
     - **11x**
     - **30x**
   * - 128³ → 64³
     - 0.40
     - 6.88
     - 27.65
     - **17x**
     - **69x**
   * - 256³ → 128³
     - 2.02
     - 178.87
     - 363.85
     - **89x**
     - **180x**
   * - 512³ → 256³
     - 6.67
     - 5,960.93
     - 4,039.05
     - **894x**
     - **605x**

.. _pytorch-integration:

PyTorch Integration
-------------------

Direct loading to PyTorch tensors without intermediate numpy arrays.

.. list-table::
   :header-rows: 1
   :widths: 15 15 15 15 15 15

   * - Source Size
     - medrs (ms)
     - MONAI (ms)
     - TorchIO (ms)
     - vs MONAI
     - vs TorchIO
   * - 64³
     - 0.34
     - 1.58
     - 5.37
     - **5x**
     - **16x**
   * - 128³
     - 0.49
     - 5.14
     - 10.22
     - **10x**
     - **21x**
   * - 256³
     - 0.60
     - 162.78
     - 53.70
     - **271x**
     - **90x**
   * - 512³
     - 0.84
     - 5,864.85
     - 1,223.24
     - **6,982x**
     - **1,456x**

.. _normalization:

Load with Normalization
------------------------

Loading with z-score normalization (zero mean, unit variance).

.. list-table::
   :header-rows: 1
   :widths: 15 15 15 15 15 15

   * - Source Size
     - medrs (ms)
     - MONAI (ms)
     - TorchIO (ms)
     - vs MONAI
     - vs TorchIO
   * - 64³
     - 0.49
     - 2.15
     - 7.04
     - **4x**
     - **14x**
   * - 128³
     - 0.60
     - 5.36
     - 12.26
     - **9x**
     - **20x**
   * - 256³
     - 0.73
     - 163.38
     - 53.59
     - **224x**
     - **73x**
   * - 512³
     - 1.01
     - 3,735.31
     - 1,092.25
     - **3,698x**
     - **1,081x**

.. _storage-efficiency:

Storage Efficiency
------------------

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

.. _running-benchmarks:

Running Benchmarks
------------------

Run benchmarks locally to measure performance on your hardware:

.. code-block:: bash

   # Install dependencies
   pip install torch monai torchio

   # Quick benchmark (64³-256³, 5 iterations)
   python benchmarks/bench_medrs.py --quick
   python benchmarks/bench_monai.py --quick
   python benchmarks/bench_torchio.py --quick

   # Full benchmark (64³-512³, 20 iterations)
   python benchmarks/bench_medrs.py
   python benchmarks/bench_monai.py
   python benchmarks/bench_torchio.py

   # Extended iterations (30 iterations)
   python benchmarks/bench_medrs.py --full

   # Generate plots from results
   python benchmarks/plot_results.py

Plot outputs are saved to ``benchmarks/results/plots/``:

- ``load.png`` - Basic loading performance
- ``load_cropped.png`` - Crop-first loading
- ``load_resampled.png`` - Load with resampling
- ``load_cropped_to_torch.png`` - Direct PyTorch tensor loading
- ``load_cropped_normalized.png`` - Load with normalization
- ``summary.png`` - Combined summary plot
- ``speedup_heatmap.png`` - Speedup comparison heatmap

.. _methodology:

Methodology
-----------

Benchmark Conditions
~~~~~~~~~~~~~~~~~~~~

- **Iterations**: 20 per operation (default), 5 for quick, 30 for full
- **Warmup**: 3 iterations before timing
- **Data**: Synthetic Gaussian noise with realistic intensity distribution
- **Format**: Uncompressed NIfTI (.nii) for fair comparison
- **Hardware**: Apple M1 Pro (macOS arm64)
- **Date**: December 2024

What We Measure
~~~~~~~~~~~~~~~

- **Median time**: Robust central tendency (less sensitive to outliers)
- **Std dev**: Consistency of performance
- **Min/Max**: Range of observed times

Notes
~~~~~

- Crop-first speedups depend heavily on the ratio of patch size to volume size
- GPU benchmarks not included in default suite (CPU only)
- Results are from synthetic data; real medical imaging data may vary
- First load may be slower due to disk caching effects
