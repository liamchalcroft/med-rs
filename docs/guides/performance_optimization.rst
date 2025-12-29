Performance Optimization
========================

This guide covers advanced performance optimization techniques for medrs applications to achieve maximum throughput and minimal memory usage.

Quick Reference: Performance vs MONAI
-------------------------------------

medrs delivers substantial speedups over MONAI for I/O-bound operations:

.. list-table::
   :header-rows: 1
   :widths: 30 20 20 20

   * - Operation
     - medrs
     - MONAI
     - Speedup
   * - Load (128³)
     - 0.2ms
     - 7.5ms
     - **41x**
   * - Load Cropped (64³ from 128³)
     - 0.8ms
     - 30ms
     - **39x**
   * - To PyTorch
     - 1.0ms
     - 16ms
     - **16x**
   * - Save (128³)
     - 110ms
     - 37ms
     - 3x slower
   * - Z-Normalize
     - 30ms
     - 20ms
     - 1.5x slower

*Note: medrs excels at I/O operations. MONAI is faster for compute-heavy transforms like resampling and normalization due to PyTorch's optimized kernels.*

Mixed-Precision Storage
-----------------------

medrs supports bf16 and f16 storage for 50% file size reduction with minimal precision loss.

Storage Efficiency
~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 25 25

   * - Format
     - Size (128³, compressed)
     - vs float32
   * - float32
     - 8.3 MB
     - 100%
   * - **bfloat16**
     - **3.4 MB**
     - **41%**
   * - **float16**
     - **4.1 MB**
     - **50%**
   * - int16
     - 1.2 MB
     - 15%

Using Mixed Precision
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import medrs

   # Load an image
   img = medrs.load("brain.nii.gz")

   # Convert to bf16 for training (better numerical range than f16)
   bf16_img = img.with_dtype("bfloat16")
   bf16_img.save("brain_bf16.nii.gz")  # 50% smaller file

   # Convert to f16 for inference
   f16_img = img.with_dtype("float16")
   f16_img.save("brain_f16.nii.gz")

   # Load directly to PyTorch with target precision
   import torch
   tensor = medrs.load_to_torch("brain.nii.gz", dtype=torch.bfloat16, device="cuda")

When to Use Mixed Precision
~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **bfloat16**: Recommended for training. Same dynamic range as float32, just reduced mantissa precision.
- **float16**: Best for inference or when hardware doesn't support bf16. Watch for overflow with large values.
- **int16**: Maximum compression for normalized data in [-1, 1] or [0, 1] range.

MONAI Integration for Maximum Performance
-----------------------------------------

medrs provides drop-in replacements for MONAI transforms that can dramatically improve I/O performance in existing pipelines.

Drop-in Replacement Usage
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Before (MONAI - slower)
   from monai.transforms import LoadImaged, RandCropByPosNegLabeld

   # After (medrs - up to 40x faster)
   from medrs.monai_compat import MedrsLoadImaged, MedrsRandCropByPosNegLabeld

   # Same API, just change the imports
   pipeline = Compose([
       MedrsLoadImaged(keys=["image", "label"], ensure_channel_first=True),
       MedrsRandCropByPosNegLabeld(
           keys=["image", "label"],
           label_key="label",
           spatial_size=(64, 64, 64),
           pos=1,
           neg=1,
           num_samples=4,
       ),
   ])

Mixing medrs and MONAI Transforms
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from medrs.monai_compat import MedrsLoadImaged, MedrsRandCropByPosNegLabeld
   from monai.transforms import Compose, RandFlipd, RandGaussianNoised, EnsureTyped

   # Use medrs for I/O-heavy operations, MONAI for augmentations
   train_transforms = Compose([
       # medrs: Fast loading (up to 40x faster)
       MedrsLoadImaged(keys=["image", "label"], ensure_channel_first=True),

       # medrs: Fast cropping (up to 40x faster)
       MedrsRandCropByPosNegLabeld(
           keys=["image", "label"],
           label_key="label",
           spatial_size=(64, 64, 64),
           pos=1, neg=1, num_samples=4,
       ),

       # MONAI: Standard augmentations (work seamlessly with medrs outputs)
       RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
       RandGaussianNoised(keys=["image"], prob=0.2, std=0.1),
       EnsureTyped(keys=["image", "label"]),
   ])

Available Drop-in Transforms
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``MedrsLoadImage`` / ``MedrsLoadImaged`` - Fast NIfTI loading
- ``MedrsSaveImage`` / ``MedrsSaveImaged`` - Fast NIfTI saving
- ``MedrsRandCropByPosNegLabeld`` - Label-aware cropping (up to 40x faster)
- ``MedrsRandSpatialCropd`` / ``MedrsCenterSpatialCropd`` - Spatial cropping
- ``MedrsOrientation`` / ``MedrsOrientationd`` - Reorientation
- ``MedrsResample`` / ``MedrsResampled`` - Resampling

Memory Optimization
------------------

Crop-First Loading Strategy
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The most significant performance gain comes from crop-first loading, which reduces memory usage (only patch data is loaded) and improves speed (only reads required bytes from disk).

.. code-block:: python

   import medrs
   import torch

   # Traditional approach: Load entire volume first
   def traditional_approach(volume_path: str, patch_size: tuple[int, int, int]):
       img = medrs.load(volume_path)  # Loads entire 200MB+ volume
       data = img.to_numpy()
       tensor = torch.from_numpy(data)  # Additional copy
       patch = tensor[:patch_size[0], :patch_size[1], :patch_size[2]]
       return patch

   # Crop-first approach: Load only what you need
   def crop_first_approach(volume_path: str, patch_size: tuple[int, int, int]):
       return medrs.load_cropped_to_torch(
           volume_path,
           output_shape=patch_size,
           device="cuda",  # Direct GPU placement
           dtype=torch.float16  # Half precision
       )

   # Performance comparison
   start_time = time.time()
   traditional_patch = traditional_approach("large_volume.nii.gz", (64, 64, 64))
   traditional_time = time.time() - start_time

   start_time = time.time()
   optimized_patch = crop_first_approach("large_volume.nii.gz", (64, 64, 64))
   optimized_time = time.time() - start_time

   print(f"Traditional: {traditional_time:.3f}s")
   print(f"Optimized: {optimized_time:.3f}s")
   print(f"Speedup: {traditional_time/optimized_time:.1f}x")

Memory Pool Management
~~~~~~~~~~~~~~~~~~~~~~

Implement intelligent memory pooling to avoid repeated allocations:

.. code-block:: python

   import torch
   from typing import Dict, List

   class TensorPool:
       def __init__(self, max_pool_size: int = 100):
           self.pools: Dict[tuple, List[torch.Tensor]] = {}
           self.max_pool_size = max_pool_size

       def get_tensor(self, shape: tuple[int, ...], device: str = "cpu") -> torch.Tensor:
           key = (shape, device)
           pool = self.pools.get(key, [])

           if pool:
               tensor = pool.pop()
               tensor.zero_()
               return tensor
           else:
               return torch.zeros(shape, device=device)

       def return_tensor(self, tensor: torch.Tensor):
           key = (tuple(tensor.shape), str(tensor.device))
           pool = self.pools.get(key, [])

           if len(pool) < self.max_pool_size:
               pool.append(tensor.detach())
               self.pools[key] = pool

   # Usage
   tensor_pool = TensorPool(max_pool_size=50)

   def pooled_batch_loading(volume_paths: List[str],
                           batch_size: int,
                           patch_size: tuple[int, int, int]) -> torch.Tensor:
       batch_tensors = []

       for path in volume_paths[:batch_size]:
           # Get tensor from pool or allocate new
           tensor = tensor_pool.get_tensor(patch_size, device="cuda")

           # Load directly into pooled tensor
           loaded = medrs.load_cropped_to_torch(path, patch_size, device="cuda")
           tensor.copy_(loaded)
           batch_tensors.append(tensor)

       batch = torch.stack(batch_tensors)

       # Return tensors to pool after use (implement with context manager)
       return batch

Precision Optimization
~~~~~~~~~~~~~~~~~~~~~

Use appropriate precision to reduce memory usage and improve speed:

.. code-block:: python

   def optimize_precision(tensor: torch.Tensor,
                         target_precision: str = "bfloat16") -> torch.Tensor:
       """Optimize tensor precision based on content and hardware support"""

       if target_precision == "float16":
           if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7:
               return tensor.half()  # FP16 on modern GPUs
           else:
               print("Warning: FP16 not supported, using FP32")
               return tensor.float()

       elif target_precision == "bfloat16":
           if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
               return tensor.bfloat16()
           else:
               print("Warning: BF16 not supported, using FP32")
               return tensor.float()

       return tensor.float()

   def precision_optimized_loading(volume_path: str,
                                   patch_size: tuple[int, int, int],
                                   precision: str = "bfloat16") -> torch.Tensor:
       """Load with automatic precision optimization"""

       # Load as float32 first
       tensor = medrs.load_cropped_to_torch(volume_path, patch_size, dtype=torch.float32)

       # Optimize precision based on hardware and content
       return optimize_precision(tensor, precision)

I/O Optimization
----------------

Concurrent Loading
~~~~~~~~~~~~~~~~~~

Utilize multiple processes for parallel I/O operations:

.. code-block:: python

   import concurrent.futures
   import multiprocessing as mp
   from typing import List

   class ConcurrentLoader:
       def __init__(self, max_workers: int = None):
           self.max_workers = max_workers or min(8, mp.cpu_count())

       def load_batch_concurrent(self,
                               volume_paths: List[str],
                               patch_size: tuple[int, int, int],
                               device: str = "cuda") -> List[torch.Tensor]:
           """Load multiple volumes concurrently"""

           def load_single_volume(path: str) -> torch.Tensor:
               return medrs.load_cropped_to_torch(
                   path, patch_size, device=device
               )

           with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
               futures = {
                   executor.submit(load_single_volume, path): path
                   for path in volume_paths
               }

               results = []
               for future in concurrent.futures.as_completed(futures):
                   path = futures[future]
                   try:
                       tensor = future.result()
                       results.append(tensor)
                   except Exception as e:
                       print(f"Failed to load {path}: {e}")
                       results.append(None)

               return [r for r in results if r is not None]

   # Usage
   loader = ConcurrentLoader(max_workers=4)
   batch = loader.load_batch_concurrent(
       volume_paths=["vol1.nii", "vol2.nii", "vol3.nii", "vol4.nii"],
       patch_size=(64, 64, 64),
       device="cuda"
   )

Asynchronous I/O Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~

Implement async I/O for non-blocking operations:

.. code-block:: python

   import asyncio
   import aiofiles
   from pathlib import Path

   class AsyncDataLoader:
       def __init__(self, prefetch_size: int = 10):
           self.prefetch_size = prefetch_size
           self.prefetch_queue = asyncio.Queue(maxsize=prefetch_size)

       async def prefetch_volumes(self, volume_paths: List[str],
                                 patch_size: tuple[int, int, int]):
           """Prefetch volumes asynchronously"""

           async def load_and_queue(path: str):
               try:
                   # Check if file exists
                   if not Path(path).exists():
                       print(f"File not found: {path}")
                       return

                   # Load in thread pool to avoid blocking
                   loop = asyncio.get_event_loop()
                   tensor = await loop.run_in_executor(
                       None,
                       medrs.load_cropped_to_torch,
                       path, patch_size
                   )

                   await self.prefetch_queue.put((path, tensor))

               except Exception as e:
                   print(f"Failed to prefetch {path}: {e}")

           # Start prefetching tasks
           tasks = [load_and_queue(path) for path in volume_paths]
           await asyncio.gather(*tasks)

       async def get_next_volume(self) -> tuple[str, torch.Tensor]:
           """Get next prefetched volume"""
           return await self.prefetch_queue.get()

   # Usage
   async def training_loop(volume_paths: List[str]):
       loader = AsyncDataLoader(prefetch_size=20)

       # Start prefetching
       prefetch_task = asyncio.create_task(
           loader.prefetch_volumes(volume_paths, (64, 64, 64))
       )

       # Process volumes as they become available
       processed_count = 0
       while processed_count < len(volume_paths):
           path, tensor = await loader.get_next_volume()

           # Process tensor
           output = model(tensor)
           processed_count += 1

           print(f"Processed {processed_count}/{len(volume_paths)}: {path}")

       await prefetch_task

GPU Optimization
---------------

Memory-Efficient GPU Operations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Optimize GPU memory usage for large-scale training:

.. code-block:: python

   class GPUMemoryManager:
       def __init__(self, memory_fraction: float = 0.9):
           self.memory_fraction = memory_fraction
           self.device = torch.cuda.current_device()

       def get_optimal_batch_size(self,
                                 input_shape: tuple[int, ...],
                                 model: torch.nn.Module,
                                 target_memory_gb: float = 10.0) -> int:
           """Find optimal batch size based on available GPU memory"""

           available_memory = torch.cuda.get_device_properties(self.device).total_memory
           target_memory = int(target_memory_gb * 1024**3)

           # Start with small batch size and increase until memory limit
           batch_size = 1
           max_batch_size = 64

           while batch_size <= max_batch_size:
               try:
                   # Clear cache
                   torch.cuda.empty_cache()

                   # Create dummy batch
                   dummy_input = torch.randn(batch_size, *input_shape, device="cuda")

                   # Forward pass
                   with torch.no_grad():
                       _ = model(dummy_input)

                   # Check memory usage
                   current_memory = torch.cuda.memory_allocated()
                   if current_memory > target_memory:
                       return batch_size // 2

                   batch_size *= 2

               except RuntimeError as e:
                   if "out of memory" in str(e):
                       return batch_size // 2
                   raise

           return batch_size // 2

       def monitor_gpu_memory(self):
           """Monitor GPU memory usage"""
           allocated = torch.cuda.memory_allocated() / 1024**3
           reserved = torch.cuda.memory_reserved() / 1024**3
           total = torch.cuda.get_device_properties(self.device).total_memory / 1024**3

           print(f"GPU Memory: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved, {total:.1f}GB total")

   # Usage
   memory_manager = GPUMemoryManager()

   # Find optimal batch size
   optimal_batch_size = memory_manager.get_optimal_batch_size(
       input_shape=(1, 64, 64, 64),  # Example input shape
       model=your_model
   )

   print(f"Optimal batch size: {optimal_batch_size}")

TensorRT Integration
~~~~~~~~~~~~~~~~~~~~

Use TensorRT for inference optimization:

.. code-block:: python

   import torch_tensorrt

   def optimize_model_with_tensorrt(model: torch.nn.Module,
                                   input_shape: tuple[int, ...],
                                   precision: str = "fp16") -> torch.nn.Module:
       """Optimize PyTorch model with TensorRT"""

       dummy_input = torch.randn(input_shape, device="cuda")

       # Convert to TensorRT
       trt_model = torch_tensorrt.compile(
           model,
           inputs=[dummy_input],
           enabled_precisions={torch.float16} if precision == "fp16" else {torch.float32},
           workspace_size=1 << 30,  # 1GB
           max_batch_size=32
       )

       return trt_model

   def benchmark_models(original_model: torch.nn.Module,
                        trt_model: torch.nn.Module,
                        input_shape: tuple[int, ...],
                        num_runs: int = 100) -> dict:
       """Benchmark original vs TensorRT model"""

       dummy_input = torch.randn(input_shape, device="cuda")

       # Warmup
       for _ in range(10):
           _ = original_model(dummy_input)
           _ = trt_model(dummy_input)

       # Benchmark original model
       torch.cuda.synchronize()
       start_time = time.time()
       for _ in range(num_runs):
           _ = original_model(dummy_input)
       torch.cuda.synchronize()
       original_time = time.time() - start_time

       # Benchmark TensorRT model
       torch.cuda.synchronize()
       start_time = time.time()
       for _ in range(num_runs):
           _ = trt_model(dummy_input)
       torch.cuda.synchronize()
       trt_time = time.time() - start_time

       return {
           'original_fps': num_runs / original_time,
           'trt_fps': num_runs / trt_time,
           'speedup': original_time / trt_time
       }

Training Pipeline Optimization
------------------------------

High-Throughput Data Loading
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Build a production-grade data loading pipeline:

.. code-block:: python

   import torch
   from torch.utils.data import DataLoader, Dataset
   import medrs

   class OptimizedMedicalDataset(Dataset):
       def __init__(self,
                    volume_paths: List[str],
                    patch_size: tuple[int, int, int],
                    transform=None,
                    cache_size: int = 1000):
           self.volume_paths = volume_paths
           self.patch_size = patch_size
           self.transform = transform
           self.cache_size = cache_size
           self.cache = {}
           self.access_order = []

       def __len__(self) -> int:
           return len(self.volume_paths)

       def __getitem__(self, idx: int) -> torch.Tensor:
           if idx in self.cache:
               # Move to end of access order (LRU)
               self.access_order.remove(idx)
               self.access_order.append(idx)
               return self.cache[idx]

           # Load new volume
           volume_path = self.volume_paths[idx]
           tensor = medrs.load_cropped_to_torch(
               volume_path,
               self.patch_size,
               device="cuda",  # Direct GPU loading
               dtype=torch.float16
           )

           # Apply transforms if provided
           if self.transform:
               tensor = self.transform(tensor)

           # Cache management
           if len(self.cache) >= self.cache_size:
               # Remove oldest item (LRU)
               oldest_idx = self.access_order.pop(0)
               del self.cache[oldest_idx]

           self.cache[idx] = tensor
           self.access_order.append(idx)

           return tensor

   def create_optimized_dataloader(volume_paths: List[str],
                                  batch_size: int,
                                  patch_size: tuple[int, int, int],
                                  num_workers: int = 4) -> DataLoader:
       """Create optimized DataLoader with best practices"""

       dataset = OptimizedMedicalDataset(
           volume_paths=volume_paths,
           patch_size=patch_size,
           cache_size=2000,  # Large cache for better hit rates
       )

       return DataLoader(
           dataset,
           batch_size=batch_size,
           shuffle=True,
           num_workers=num_workers,
           pin_memory=True,  # Faster GPU transfer
           persistent_workers=True,  # Keep workers alive between epochs
           prefetch_factor=2,  # Prefetch batches
           drop_last=True  # Consistent batch sizes
       )

Profiling and Monitoring
-----------------------

Performance Profiler
~~~~~~~~~~~~~~~~~~~~~

Comprehensive performance profiling for optimization:

.. code-block:: python

   import time
   import psutil
   import torch
   from contextlib import contextmanager
   from typing import Dict, List
   import numpy as np

   @contextmanager
   def profile_operation(operation_name: str):
       """Profile an operation with detailed metrics"""

       # System metrics
       process = psutil.Process()
       cpu_before = process.cpu_percent()
       memory_before = process.memory_info().rss / 1024**2  # MB

       # GPU metrics
       if torch.cuda.is_available():
           torch.cuda.synchronize()
           gpu_memory_before = torch.cuda.memory_allocated() / 1024**2  # MB
           gpu_utilization_before = torch.cuda.utilization()
       else:
           gpu_memory_before = gpu_utilization_before = 0

       start_time = time.time()

       try:
           yield
       finally:
           end_time = time.time()

           # System metrics after
           cpu_after = process.cpu_percent()
           memory_after = process.memory_info().rss / 1024**2  # MB

           # GPU metrics after
           if torch.cuda.is_available():
               torch.cuda.synchronize()
               gpu_memory_after = torch.cuda.memory_allocated() / 1024**2  # MB
               gpu_utilization_after = torch.cuda.utilization()
           else:
               gpu_memory_after = gpu_utilization_after = 0

           # Calculate deltas
           duration = end_time - start_time
           memory_delta = memory_after - memory_before
           gpu_memory_delta = gpu_memory_after - gpu_memory_before

           print(f"\n=== {operation_name} Profile ===")
           print(f"Duration: {duration:.3f}s")
           print(f"CPU: {cpu_before:.1f}% -> {cpu_after:.1f}%")
           print(f"RAM: {memory_before:.1f}MB -> {memory_after:.1f}MB (Delta{memory_delta:+.1f}MB)")

           if torch.cuda.is_available():
               print(f"GPU Memory: {gpu_memory_before:.1f}MB -> {gpu_memory_after:.1f}MB (Delta{gpu_memory_delta:+.1f}MB)")
               print(f"GPU Util: {gpu_utilization_before:.1f}% -> {gpu_utilization_after:.1f}%")

   # Usage
   def profile_training_pipeline(volume_paths: List[str]):
       with profile_operation("Complete Training Pipeline"):
           dataloader = create_optimized_dataloader(volume_paths, batch_size=8, patch_size=(64, 64, 64))

           for batch_idx, batch in enumerate(dataloader):
               with profile_operation(f"Batch {batch_idx} Processing"):
                   # Forward pass
                   output = model(batch)
                   loss = criterion(output, target)

                   # Backward pass
                   loss.backward()
                   optimizer.step()
                   optimizer.zero_grad()

               if batch_idx >= 10:  # Profile first 10 batches
                   break

Performance Benchmarking Framework
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a comprehensive benchmarking framework:

.. code-block:: python

   class PerformanceBenchmark:
       def __init__(self):
           self.results = {}

       def benchmark_loading_methods(self,
                                   volume_path: str,
                                   patch_size: tuple[int, int, int],
                                   num_runs: int = 100) -> dict:
           """Benchmark different loading methods"""

           methods = {
               'traditional_full_load': self._benchmark_traditional_loading,
               'crop_first_cpu': self._benchmark_crop_first_cpu,
               'crop_first_gpu': self._benchmark_crop_first_gpu,
               'crop_first_fp16': self._benchmark_crop_first_fp16
           }

           results = {}

           for method_name, method_func in methods.items():
               print(f"Benchmarking {method_name}...")
               times = []
               memory_usage = []

               for _ in range(num_runs):
                   start_time = time.time()
                   start_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

                   try:
                       result = method_func(volume_path, patch_size)
                       del result  # Free memory
                   except RuntimeError as e:
                       if "out of memory" in str(e):
                           print(f"OOM for {method_name}, skipping")
                           break
                       raise

                   if torch.cuda.is_available():
                       torch.cuda.empty_cache()

                   end_time = time.time()
                   end_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

                   times.append(end_time - start_time)
                   memory_usage.append(end_memory - start_memory)

               if times:
                   results[method_name] = {
                       'mean_time': np.mean(times),
                       'std_time': np.std(times),
                       'mean_memory': np.mean(memory_usage),
                       'throughput': 1.0 / np.mean(times)
                   }

           return results

       def _benchmark_traditional_loading(self, path: str, patch_size: tuple[int, int, int]):
           img = medrs.load(path)
           data = img.to_numpy()
           tensor = torch.from_numpy(data)
           return tensor[:patch_size[0], :patch_size[1], :patch_size[2]]

       def _benchmark_crop_first_cpu(self, path: str, patch_size: tuple[int, int, int]):
           return medrs.load_cropped_to_torch(path, patch_size, device="cpu")

       def _benchmark_crop_first_gpu(self, path: str, patch_size: tuple[int, int, int]):
           return medrs.load_cropped_to_torch(path, patch_size, device="cuda")

       def _benchmark_crop_first_fp16(self, path: str, patch_size: tuple[int, int, int]):
           return medrs.load_cropped_to_torch(path, patch_size, device="cuda", dtype=torch.float16)

       def generate_report(self, results: dict) -> str:
           """Generate comprehensive performance report"""

           report = ["=== PERFORMANCE BENCHMARK REPORT ===\n"]

           baseline_time = None
           baseline_memory = None

           for method, metrics in results.items():
               if baseline_time is None:
                   baseline_time = metrics['mean_time']
                   baseline_memory = metrics['mean_memory']

               speedup = baseline_time / metrics['mean_time']
               memory_ratio = metrics['mean_memory'] / baseline_memory if baseline_memory > 0 else 1.0

               report.append(f"{method}:")
               report.append(f"  Mean time: {metrics['mean_time']:.4f}s +/- {metrics['std_time']:.4f}s")
               report.append(f"  Throughput: {metrics['throughput']:.1f} ops/sec")
               report.append(f"  Memory usage: {metrics['mean_memory']:.1f}MB")
               report.append(f"  Speedup: {speedup:.1f}x")
               report.append(f"  Memory efficiency: {1/memory_ratio:.1f}x")
               report.append("")

           return "\n".join(report)

   # Usage
   benchmark = PerformanceBenchmark()
   results = benchmark.benchmark_loading_methods("test_volume.nii.gz", (64, 64, 64))
   print(benchmark.generate_report(results))

This comprehensive performance optimization guide provides the techniques needed to maximize medrs performance for production workloads.
