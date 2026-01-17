#!/usr/bin/env python3
"""medrs crop-first loading example."""

import time
import torch
import medrs


def main():
    volume_path = "sample_volume.nii"

    print("Crop-First Loading")
    print("=" * 40)

    try:
        print("\n1. Load specific region:")
        start = time.time()
        crop = medrs.load_cropped(
            volume_path,
            crop_offset=[50, 50, 25],
            crop_shape=[64, 64, 32],
        )
        load_time = time.time() - start
        print(f"   Shape: {crop.shape}")
        print(f"   Spacing: {crop.spacing}")
        print(f"   Time: {load_time * 1000:.2f}ms")

        print("\n2. Load directly to PyTorch:")
        start = time.time()
        tensor = medrs.load_cropped_to_torch(
            volume_path,
            output_shape=[96, 96, 96],
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        load_time = time.time() - start
        print(f"   Shape: {tuple(tensor.shape)}")
        print(f"   Device: {tensor.device}")
        print(f"   Time: {load_time * 1000:.2f}ms")

        print("\n3. Half precision:")
        start = time.time()
        half_tensor = medrs.load_cropped_to_torch(
            volume_path,
            output_shape=[128, 128, 128],
            device="cuda" if torch.cuda.is_available() else "cpu",
            dtype=torch.float16,
        )
        load_time = time.time() - start
        memory_mb = half_tensor.numel() * half_tensor.element_size() / 1024**2
        print(f"   Dtype: {half_tensor.dtype}")
        print(f"   Memory: {memory_mb:.1f}MB")
        print(f"   Time: {load_time * 1000:.2f}ms")

        print("\n4. Multiple patches:")
        offsets = [[32, 32, 16], [64, 64, 32], [96, 96, 48]]
        patches = []
        for offset in offsets:
            patch = medrs.load_cropped(volume_path, offset, [32, 32, 32])
            patches.append(patch.to_torch())
        stacked = torch.stack(patches)
        print(f"   Stacked shape: {tuple(stacked.shape)}")

        print("\n5. Scaling test:")
        for size in [32, 64, 96, 128]:
            start = time.time()
            patch = medrs.load_cropped(volume_path, [0, 0, 0], [size, size, size])
            load_time = time.time() - start
            memory_mb = patch.to_numpy().nbytes / 1024**2
            print(f"   {size}^3: {load_time * 1000:.2f}ms, {memory_mb:.1f}MB")

        print("\nDone.")

    except FileNotFoundError:
        print(f"\nFile not found: {volume_path}")
        print("Note: load_cropped requires uncompressed .nii files")
    except Exception as e:
        print(f"\nError: {e}")


if __name__ == "__main__":
    main()
