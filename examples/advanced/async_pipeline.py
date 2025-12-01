#!/usr/bin/env python3
"""
medrs Async Pipeline Example
===========================

This example demonstrates how to use medrs in asynchronous pipelines
for high-throughput data processing and training.
"""

import asyncio
import time
import torch
from concurrent.futures import ThreadPoolExecutor
import medrs


class AsyncMedicalDataLoader:
    """Asynchronous medical image data loader using medrs."""

    def __init__(self, volume_paths, max_concurrent_loads=4):
        self.volume_paths = volume_paths
        self.max_concurrent_loads = max_concurrent_loads
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent_loads)

    async def load_volume_async(self, path, patch_size=(64, 64, 64)):
        """Load a volume asynchronously."""
        loop = asyncio.get_event_loop()

        # Run medrs loading in thread pool
        tensor = await loop.run_in_executor(
            self.executor,
            medrs.load_cropped_to_torch,
            path,
            patch_size,
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        return tensor

    async def load_batch_async(self, batch_size, patch_size=(64, 64, 64)):
        """Load a batch of volumes concurrently."""
        batch_paths = self.volume_paths[:batch_size]

        # Create loading tasks
        tasks = [
            self.load_volume_async(path, patch_size)
            for path in batch_paths
        ]

        # Wait for all loads to complete
        batch = await asyncio.gather(*tasks)
        return torch.stack(batch)


class PrefetchDataLoader:
    """Data loader with automatic prefetching."""

    def __init__(self, volume_paths, prefetch_size=10, batch_size=4):
        self.volume_paths = volume_paths
        self.prefetch_size = prefetch_size
        self.batch_size = batch_size
        self.prefetch_queue = asyncio.Queue(maxsize=prefetch_size)
        self.load_semaphore = asyncio.Semaphore(3)  # Limit concurrent loads

    async def _load_single_volume(self, path):
        """Load a single volume with semaphore control."""
        async with self.load_semaphore:
            return await asyncio.get_event_loop().run_in_executor(
                None,
                medrs.load_cropped_to_torch,
                path,
                (64, 64, 64),
                "cuda" if torch.cuda.is_available() else "cpu"
            )

    async def prefetch_volumes(self):
        """Prefetch volumes into the queue."""
        for path in self.volume_paths:
            volume = await self._load_single_volume(path)
            await self.prefetch_queue.put(volume)

    async def get_batch(self):
        """Get a batch from prefetched volumes."""
        batch = []
        for _ in range(self.batch_size):
            volume = await self.prefetch_queue.get()
            batch.append(volume)
        return torch.stack(batch)


async def demonstrate_async_loading():
    """Demonstrate asynchronous loading capabilities."""

    print(" medrs Async Pipeline Example")
    print("=" * 35)

    # Mock volume paths (replace with real files)
    volume_paths = [f"volume_{i:03d}.nii.gz" for i in range(20)]

    print("\n1. Async Dataset Setup:")
    print(f"    Dataset size: {len(volume_paths)} volumes")

    # Create async loader
    loader = AsyncMedicalDataLoader(volume_paths, max_concurrent_loads=4)

    # Test concurrent loading
    print("\n2. Concurrent Loading Test:")

    start_time = time.time()

    # Load 8 volumes concurrently
    batch = await loader.load_batch_async(batch_size=8)

    concurrent_time = time.time() - start_time

    print(f"    Concurrent loading time: {concurrent_time:.3f}s")
    print(f"    Batch shape: {batch.shape}")
    print(f"    Device: {batch.device}")
    print(f"    Throughput: {8/concurrent_time:.1f} volumes/sec")

    # Compare with sequential loading
    print("\n3. Sequential Loading Comparison:")

    start_time = time.time()

    # Load sequentially (traditional approach)
    sequential_batch = []
    for path in volume_paths[:8]:
        tensor = medrs.load_cropped_to_torch(path, (64, 64, 64))
        sequential_batch.append(tensor)

    sequential_batch = torch.stack(sequential_batch)
    sequential_time = time.time() - start_time

    print(f"     Sequential loading time: {sequential_time:.3f}s")
    print(f"    Sequential batch shape: {sequential_batch.shape}")
    print(f"    Speedup: {sequential_time/concurrent_time:.1f}x")


async def demonstrate_prefetching():
    """Demonstrate prefetching for training pipelines."""

    print("\n4. Prefetching Pipeline:")

    # Create mock dataset
    volume_paths = [f"training_vol_{i:03d}.nii.gz" for i in range(50)]

    # Create prefetching loader
    prefetch_loader = PrefetchDataLoader(
        volume_paths,
        prefetch_size=15,
        batch_size=4
    )

    print(f"    Prefetch queue size: {prefetch_loader.prefetch_size}")
    print(f"    Batch size: {prefetch_loader.batch_size}")

    # Start prefetching in background
    prefetch_task = asyncio.create_task(prefetch_loader.prefetch_volumes())

    # Process batches as they become available
    print("\n5. Processing with Prefetch:")

    batch_times = []
    for batch_idx in range(5):  # Process 5 batches
        start_time = time.time()

        # Get batch (fast, already loaded)
        batch = await prefetch_loader.get_batch()

        batch_time = time.time() - start_time
        batch_times.append(batch_time)

        print(f"   Batch {batch_idx+1}: {batch_time:.4f}s, shape {batch.shape}")

        # Simulate some processing time
        await asyncio.sleep(0.1)

    # Wait for prefetching to complete
    await prefetch_task

    avg_batch_time = sum(batch_times) / len(batch_times)
    print(f"\n    Average batch time: {avg_batch_time:.4f}s")
    print(f"    Effective throughput: {4/avg_batch_time:.1f} batches/sec")


class AsyncTrainingPipeline:
    """Asynchronous training pipeline with medrs."""

    def __init__(self, model, optimizer, device="cuda"):
        self.model = model
        self.optimizer = optimizer
        self.device = device

    async def train_step_async(self, batch):
        """Perform a training step asynchronously."""
        # Move to device if needed
        if batch.device != self.device:
            batch = batch.to(self.device)

        # Create dummy labels
        labels = torch.rand(batch.shape[0], 1, device=self.device)

        # Forward pass
        self.optimizer.zero_grad()
        outputs = self.model(batch.unsqueeze(1))  # Add channel dim
        loss = torch.nn.functional.binary_cross_entropy_with_logits(outputs, labels)

        # Backward pass
        loss.backward()
        self.optimizer.step()

        return loss.item()


async def demonstrate_async_training():
    """Demonstrate asynchronous training pipeline."""

    print("\n6. Async Training Pipeline:")

    # Create simple model
    model = torch.nn.Sequential(
        torch.nn.Conv3d(1, 16, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool3d(1),
        torch.nn.Flatten(),
        torch.nn.Linear(16, 1),
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Create training pipeline
    pipeline = AsyncTrainingPipeline(model, optimizer)

    # Create async data loader
    volume_paths = [f"train_vol_{i:03d}.nii.gz" for i in range(100)]
    loader = AsyncMedicalDataLoader(volume_paths, max_concurrent_loads=6)

    print(f"    Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Simulate training with async data loading
    print("\n7. Async Training Loop:")

    total_time = 0
    batches_processed = 0

    for epoch in range(2):  # 2 epochs
        print(f"\n   Epoch {epoch + 1}:")

        for batch_idx in range(5):  # 5 batches per epoch
            start_time = time.time()

            # Load batch and train step concurrently
            load_task = loader.load_batch_async(batch_size=6)
            train_task = None  # Previous batch training

            # Wait for batch load
            batch = await load_task

            # Perform training step
            loss = await pipeline.train_step_async(batch)

            step_time = time.time() - start_time
            total_time += step_time
            batches_processed += 1

            print(f"     Batch {batch_idx + 1}: {step_time:.3f}s, loss: {loss:.4f}")

    avg_time = total_time / batches_processed
    print(f"\n    Average step time: {avg_time:.3f}s")
    print(f"    Training throughput: {6/avg_time:.1f} samples/sec")


async def demonstrate_memory_efficiency():
    """Demonstrate memory efficiency in async pipeline."""

    print("\n8. Memory Efficiency Analysis:")

    # Traditional approach would load all volumes into memory
    traditional_memory = 50 * 1600  # 50 volumes * 1600MB each
    medrs_memory = 50 * 40  # 50 patches * 40MB each

    print(f"    Traditional approach: {traditional_memory:,}MB")
    print(f"    medrs async approach: {medrs_memory:,}MB")
    print(f"    Memory reduction: {traditional_memory/medrs_memory:.0f}x")
    print(f"    Memory saved: {(traditional_memory - medrs_memory)/1024:.1f}GB")

    # Streaming analysis
    print("\n    Streaming Benefits:")
    print("    No upfront loading time")
    print("    Constant memory usage")
    print("    Scales to arbitrary dataset sizes")
    print("    Better cache locality")


async def main():
    """Main async demonstration."""
    try:
        await demonstrate_async_loading()
        await demonstrate_prefetching()
        await demonstrate_async_training()
        await demonstrate_memory_efficiency()

        print("\n Async pipeline example completed!")
        print("\n Key Benefits:")
        print("   - Concurrent I/O operations")
        print("   - Overlap loading and computation")
        print("   - Constant memory usage")
        print("   - Scales to large datasets")
        print("   - Automatic prefetching")

    except Exception as e:
        print(f"\n Error: {e}")
        print("Note: This example requires NIfTI files to be present")
        print("Replace placeholder file paths with actual medical images")


if __name__ == "__main__":
    asyncio.run(main())
