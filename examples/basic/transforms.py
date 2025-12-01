#!/usr/bin/env python3
"""
medrs Transforms Example
========================

This example demonstrates medrs's high-performance image transforms
including normalization, rescaling, and resampling operations.
"""

import time
import medrs


def demonstrate_transforms():
    """Demonstrate various medrs transforms."""

    print(" medrs Transforms Example")
    print("=" * 35)

    # Replace with your NIfTI file
    volume_path = "sample_volume.nii.gz"

    try:
        # Load an image
        print("\n1. Loading image:")
        img = medrs.load_cropped(volume_path, [0, 0, 0], [128, 128, 64])
        print(f"    Shape: {img.data.shape}")
        print(f"    Spacing: {img.spacing}")
        print(f"    Data type: {img.data.dtype}")
        print(f"    Range: [{img.data.min():.3f}, {img.data.max():.3f}]")

        # Transform 1: Z-normalization
        print("\n2. Z-normalization:")
        start_time = time.time()
        normalized = medrs.z_normalization(img)
        norm_time = time.time() - start_time

        print(f"     Time: {norm_time:.4f}s")
        print(f"    Mean: {normalized.data.mean():.6f}")
        print(f"    Std: {normalized.data.std():.6f}")
        print(f"    Range: [{normalized.data.min():.3f}, {normalized.data.max():.3f}]")

        # Transform 2: Intensity rescaling
        print("\n3. Intensity rescaling:")
        start_time = time.time()
        rescaled = medrs.rescale_intensity(img, output_range=(0.0, 1.0))
        scale_time = time.time() - start_time

        print(f"     Time: {scale_time:.4f}s")
        print(f"    Range: [{rescaled.data.min():.3f}, {rescaled.data.max():.3f}]")

        # Transform 3: Clamping
        print("\n4. Value clamping:")
        clamped = medrs.clamp(img, min_value=img.data.mean(), max_value=img.data.mean() + img.data.std())
        print(f"    Original range: [{img.data.min():.3f}, {img.data.max():.3f}]")
        print(f"    Clamped range: [{clamped.data.min():.3f}, {clamped.data.max():.3f}]")

        # Transform 4: Resampling
        print("\n5. Resampling:")
        start_time = time.time()
        resampled = medrs.resample(img, target_spacing=(1.0, 1.0, 1.0))
        resample_time = time.time() - start_time

        print(f"     Time: {resample_time:.4f}s")
        print(f"    Original spacing: {img.spacing}")
        print("    Target spacing: (1.0, 1.0, 1.0)")
        print(f"    New shape: {resampled.data.shape}")

        # Transform 5: Reorientation
        print("\n6. Reorientation:")
        try:
            reoriented = medrs.reorient(img, "RAS")
            print(f"    Original orientation: {getattr(img, 'orientation', 'Unknown')}")
            print("    New orientation: RAS")
            print(f"    Shape: {reoriented.data.shape}")
        except Exception as e:
            print(f"     Reorientation not available: {e}")

        # Transform 6: Crop or pad to size
        print("\n7. Crop or pad:")
        target_size = (96, 96, 96)
        start_time = time.time()
        sized = medrs.crop_or_pad(img, target_size)
        size_time = time.time() - start_time

        print(f"     Time: {size_time:.4f}s")
        print(f"    Original shape: {img.data.shape}")
        print(f"    Target shape: {target_size}")
        print(f"    Final shape: {sized.data.shape}")

        # Transform chaining
        print("\n8. Transform chaining:")
        start_time = time.time()
        processed = medrs.crop_or_pad(img, (96, 96, 96))
        processed = medrs.z_normalization(processed)
        processed = medrs.rescale_intensity(processed, (0.0, 1.0))
        chain_time = time.time() - start_time

        print(f"     Total time: {chain_time:.4f}s")
        print(f"    Final range: [{processed.data.min():.3f}, {processed.data.max():.3f}]")

        # Performance comparison with PyTorch operations
        print("\n9. Performance comparison:")
        tensor = img.to_torch()

        # PyTorch normalization
        start_time = time.time()
        torch_norm = (tensor - tensor.mean()) / tensor.std()
        torch_time = time.time() - start_time

        print(f"    medrs z-normalization: {norm_time:.4f}s")
        print(f"    PyTorch normalization: {torch_time:.4f}s")
        print(f"    Speedup: {torch_time/norm_time:.1f}x")

        print("\n All transforms completed successfully!")

    except FileNotFoundError:
        print(f" File not found: {volume_path}")
        print("Replace with a valid NIfTI file path")
    except Exception as e:
        print(f" Error: {e}")


if __name__ == "__main__":
    demonstrate_transforms()
