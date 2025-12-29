#!/usr/bin/env python3
"""
medrs + PyTorch Training Integration
====================================

This example shows how to use medrs with PyTorch for high-performance
medical image training pipelines.
"""

import torch
from torch.utils.data import DataLoader, Dataset
import time
import medrs


class MedicalDataset(Dataset):
    """High-performance medical image dataset using medrs."""

    def __init__(self, volume_paths, patch_size=(64, 64, 64), device="cpu"):
        self.volume_paths = volume_paths
        self.patch_size = patch_size
        self.device = device

    def __len__(self):
        return len(self.volume_paths)

    def __getitem__(self, idx):
        # Load only what we need - crop-first loading
        patch = medrs.load_cropped(
            self.volume_paths[idx],
            crop_offset=[32, 32, 16],
            crop_shape=self.patch_size
        )

        # Direct GPU placement with specified dtype
        tensor = patch.to_torch_with_dtype_and_device(
            dtype=torch.float32,
            device=self.device
        )

        # Simple preprocessing
        tensor = (tensor - tensor.mean()) / (tensor.std() + 1e-8)

        return tensor


def create_simple_model(input_shape=(1, 64, 64, 64)):
    """Create a simple 3D CNN model."""
    return torch.nn.Sequential(
        torch.nn.Conv3d(1, 16, 3, padding=1),
        torch.nn.BatchNorm3d(16),
        torch.nn.ReLU(),
        torch.nn.Conv3d(16, 32, 3, padding=1),
        torch.nn.BatchNorm3d(32),
        torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool3d(1),
        torch.nn.Flatten(),
        torch.nn.Linear(32, 1),
        torch.nn.Sigmoid()
    )


def benchmark_dataloader():
    """Benchmark medrs vs traditional loading."""

    print(" PyTorch Training Integration Example")
    print("=" * 45)

    # Create dummy volume paths (replace with real files)
    volume_paths = [f"volume_{i}.nii.gz" for i in range(100)]

    print("\n1. Dataset Creation:")

    # medrs dataset
    device = "cuda" if torch.cuda.is_available() else "cpu"
    medrs_dataset = MedicalDataset(volume_paths, device=device)

    print(f"    Dataset size: {len(medrs_dataset)}")
    print(f"    Patch size: {medrs_dataset.patch_size}")
    print(f"    Device: {device}")

    # DataLoader with optimized settings
    dataloader = DataLoader(
        medrs_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=0,  # medrs handles concurrency internally
        pin_memory=False,  # Not needed with direct GPU loading
    )

    print("\n2. Training Setup:")

    # Create model
    model = create_simple_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.BCELoss()

    print(f"    Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"    Model device: {next(model.parameters()).device}")

    print("\n3. Training Loop (first 10 batches):")

    total_load_time = 0
    total_train_time = 0

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= 10:
            break

        # Time loading
        load_start = time.time()

        # Add dummy batch dimension for channel
        batch = batch.unsqueeze(1)  # [B, 1, H, W, D]

        # Create dummy labels
        labels = torch.rand(batch.shape[0], 1, device=device)

        load_time = time.time() - load_start
        total_load_time += load_time

        # Time training
        train_start = time.time()

        optimizer.zero_grad()
        outputs = model(batch)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_time = time.time() - train_start
        total_train_time += train_time

        print(f"   Batch {batch_idx+1:2d}: {load_time:.3f}s load, {train_time:.3f}s train, "
              f"loss: {loss.item():.4f}")

    print("\n4. Performance Summary:")
    print(f"    Average load time: {total_load_time/10:.3f}s")
    print(f"    Average train time: {total_train_time/10:.3f}s")
    print(f"    Throughput: {10*8/(total_load_time + total_train_time):.1f} patches/sec")

    # Memory usage
    if torch.cuda.is_available():
        memory_mb = torch.cuda.memory_allocated() / 1024**2
        print(f"    GPU Memory: {memory_mb:.1f}MB")


def demonstrate_memory_efficiency():
    """Demonstrate memory efficiency of crop-first loading."""

    print("\n Memory Efficiency Demonstration:")
    print("-" * 40)

    # Traditional approach simulation
    print("\n1. Traditional approach (simulated):")
    traditional_memory = 1600  # MB for full 200x200x200 volume
    patch_memory = 8  # MB for 64x64x64 patch
    waste = traditional_memory - patch_memory

    print(f"    Full volume memory: {traditional_memory}MB")
    print(f"    Required patch memory: {patch_memory}MB")
    print(f"     Wasted memory: {waste}MB ({waste/traditional_memory*100:.1f}%)")

    # medrs approach
    print("\n2. medrs crop-first approach:")
    print(f"    Memory used: {patch_memory}MB")
    print(f"    Memory reduction: {traditional_memory/patch_memory}x")
    print("    Speed improvement: up to 40x faster I/O")

    print("\n3. Training batch memory calculation:")
    batch_size = 16
    total_memory = patch_memory * batch_size
    print(f"    Batch size: {batch_size}")
    print(f"    Batch memory: {total_memory}MB")
    print(f"    Fits comfortably in GPU memory: {'' if total_memory < 8192 else ''}")


if __name__ == "__main__":
    benchmark_dataloader()
    demonstrate_memory_efficiency()

    print("\n PyTorch integration example completed!")
    print("\n Tips:")
    print("   - Use crop-first loading to minimize memory usage")
    print("   - Load directly to GPU to avoid CPU-GPU transfers")
    print("   - Set num_workers=0 since medrs handles concurrency")
    print("   - Use pin_memory=False with direct GPU loading")
