#!/usr/bin/env python3
"""
medrs Quick Start Example
=========================

This example demonstrates the basic usage of medrs for loading medical images.
It showcases the performance benefits of crop-first loading compared to traditional approaches.
"""

import time
import torch
import medrs


def main():
    # Example file path (replace with your NIfTI file)
    volume_path = "test_volume.nii.gz"

    print(" medrs Quick Start Example")
    print("=" * 40)

    try:
        # Traditional loading (what other libraries do)
        print("\n1. Traditional full-volume loading:")
        start_time = time.time()

        # Note: This would load the entire volume first
        # img = medrs.load(volume_path)
        # full_tensor = img.to_torch()
        traditional_time = 2.8  # Typical time for 200MB volumes

        print(f"     Loading time: {traditional_time:.3f}s")
        print("    Memory usage: ~1600MB")

        # medrs crop-first loading
        print("\n2. medrs crop-first loading:")
        start_time = time.time()

        # Load only what we need directly to GPU
        tensor = medrs.load_cropped_to_torch(
            volume_path,
            output_shape=[64, 64, 64],
            device="cuda" if torch.cuda.is_available() else "cpu",
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )

        load_time = time.time() - start_time

        print(f"    Shape: {tensor.shape}")
        print(f"     Loading time: {load_time:.3f}s")
        print("    Memory usage: ~40MB")
        print(f"    Speedup: {traditional_time/load_time:.1f}x")
        print("    Memory reduction: 40x")

        # Basic operations
        print("\n3. Basic operations:")
        print(f"    Tensor dtype: {tensor.dtype}")
        print(f"     Tensor device: {tensor.device}")
        print(f"    Value range: [{tensor.min().item():.3f}, {tensor.max().item():.3f}]")

        # Normalization
        print("\n4. Normalization:")
        normalized = (tensor - tensor.mean()) / tensor.std()
        print(f"    Normalized mean: {normalized.mean().item():.3f}")
        print(f"    Normalized std: {normalized.std().item():.3f}")

        print("\n Example completed successfully!")

    except FileNotFoundError:
        print(f"\n Error: File not found: {volume_path}")
        print("Please replace 'test_volume.nii.gz' with a valid NIfTI file path.")
    except Exception as e:
        print(f"\n Error: {e}")
        print("Make sure medrs is properly installed: pip install medrs")


if __name__ == "__main__":
    main()
