#!/usr/bin/env python3
"""medrs quick start - basic loading example."""

import time
import torch
import medrs


def main():
    volume_path = "test_volume.nii.gz"

    print("medrs Quick Start")
    print("=" * 40)

    try:
        print("\n1. Crop-first loading:")
        start = time.time()

        tensor = medrs.load_cropped_to_torch(
            volume_path,
            output_shape=[64, 64, 64],
            device="cuda" if torch.cuda.is_available() else "cpu",
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )

        load_time = time.time() - start

        print(f"   Shape: {tuple(tensor.shape)}")
        print(f"   Time: {load_time * 1000:.1f}ms")
        print(f"   Device: {tensor.device}")

        print("\n2. Value stats:")
        print(f"   Range: [{tensor.min().item():.3f}, {tensor.max().item():.3f}]")
        print(f"   Mean: {tensor.mean().item():.3f}")

        print("\n3. Z-normalize:")
        normalized = (tensor - tensor.mean()) / (tensor.std() + 1e-8)
        print(f"   Mean: {normalized.mean().item():.6f}")
        print(f"   Std: {normalized.std().item():.3f}")

        print("\nDone.")

    except FileNotFoundError:
        print(f"\nFile not found: {volume_path}")
    except Exception as e:
        print(f"\nError: {e}")


if __name__ == "__main__":
    main()
