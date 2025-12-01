#!/usr/bin/env python3
"""
medrs Crop-First Loading Example
================================

This example demonstrates medrs's signature crop-first loading capability,
which only loads the specific bytes you need from a NIfTI file.
"""

import torch
import medrs


def demonstrate_crop_loading():
    """Demonstrate different crop loading approaches."""

    print(" Crop-First Loading Examples")
    print("=" * 40)

    # Replace with your NIfTI file
    volume_path = "sample_volume.nii.gz"

    try:
        # Method 1: Load with specific offset and shape
        print("\n1. Load specific crop region:")
        crop = medrs.load_cropped(
            volume_path,
            crop_offset=[50, 50, 25],  # Starting voxel
            crop_shape=[64, 64, 32]    # Patch size
        )
        print(f"    Shape: {crop.data.shape}")
        print(f"    Spacing: {crop.spacing}")

        # Method 2: Load directly to PyTorch tensor
        print("\n2. Load directly to PyTorch:")
        tensor = medrs.load_cropped_to_torch(
            volume_path,
            output_shape=[96, 96, 96],
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        print(f"    Shape: {tensor.shape}")
        print(f"    Device: {tensor.device}")

        # Method 3: Load with half precision
        print("\n3. Load with memory optimization:")
        half_tensor = medrs.load_cropped_to_torch(
            volume_path,
            output_shape=[128, 128, 128],
            device="cuda" if torch.cuda.is_available() else "cpu",
            dtype=torch.float16
        )
        print(f"    Dtype: {half_tensor.dtype}")
        print(f"    Memory: ~{half_tensor.numel() * half_tensor.element_size() / 1024**2:.1f}MB")

        # Method 4: Multiple crops from same volume
        print("\n4. Load multiple crops:")
        patches = []
        crop_offsets = [[32, 32, 16], [64, 64, 32], [96, 96, 48]]

        for i, offset in enumerate(crop_offsets):
            patch = medrs.load_cropped(volume_path, offset, [32, 32, 32])
            patches.append(patch.to_torch())
            print(f"   Patch {i+1}: {patches[-1].shape}")

        stacked_patches = torch.stack(patches)
        print(f"    Stacked shape: {stacked_patches.shape}")

        # Performance comparison
        print("\n5. Performance comparison:")
        sizes = [32, 64, 96, 128]

        for size in sizes:
            start_time = time.time()
            patch = medrs.load_cropped(volume_path, [0, 0, 0], [size, size, size])
            load_time = time.time() - start_time
            memory_mb = patch.data.nbytes / 1024**2

            print(f"   {size}^3: {load_time:.4f}s, {memory_mb:.1f}MB")

    except FileNotFoundError:
        print(f" File not found: {volume_path}")
        print("Replace with a valid NIfTI file path")
    except Exception as e:
        print(f" Error: {e}")


if __name__ == "__main__":
    import time
    demonstrate_crop_loading()
