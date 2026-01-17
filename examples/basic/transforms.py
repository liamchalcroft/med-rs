#!/usr/bin/env python3
"""medrs transforms example - normalization, resampling, etc."""

import time
import medrs


def main():
    volume_path = "sample_volume.nii.gz"

    print("medrs Transforms")
    print("=" * 35)

    try:
        print("\n1. Loading:")
        img = medrs.load(volume_path)
        data = img.to_numpy()
        print(f"   Shape: {img.shape}")
        print(f"   Spacing: {img.spacing}")
        print(f"   Range: [{data.min():.3f}, {data.max():.3f}]")

        print("\n2. Z-normalization:")
        start = time.time()
        normalized = img.z_normalize()
        norm_time = time.time() - start
        norm_data = normalized.to_numpy()
        print(f"   Time: {norm_time * 1000:.2f}ms")
        print(f"   Mean: {norm_data.mean():.6f}")
        print(f"   Std: {norm_data.std():.6f}")

        print("\n3. Rescale to [0, 1]:")
        start = time.time()
        rescaled = img.rescale(0.0, 1.0)
        scale_time = time.time() - start
        rescaled_data = rescaled.to_numpy()
        print(f"   Time: {scale_time * 1000:.2f}ms")
        print(f"   Range: [{rescaled_data.min():.3f}, {rescaled_data.max():.3f}]")

        print("\n4. Clamp:")
        clamped = img.clamp(data.mean(), data.mean() + data.std())
        clamped_data = clamped.to_numpy()
        print(f"   Original range: [{data.min():.3f}, {data.max():.3f}]")
        print(f"   Clamped range: [{clamped_data.min():.3f}, {clamped_data.max():.3f}]")

        print("\n5. Resample to 1mm isotropic:")
        start = time.time()
        resampled = img.resample([1.0, 1.0, 1.0])
        resample_time = time.time() - start
        print(f"   Time: {resample_time * 1000:.2f}ms")
        print(f"   Original spacing: {img.spacing}")
        print(f"   New shape: {resampled.shape}")

        print("\n6. Crop or pad to 96^3:")
        start = time.time()
        sized = img.crop_or_pad([96, 96, 96])
        size_time = time.time() - start
        print(f"   Time: {size_time * 1000:.2f}ms")
        print(f"   Original: {img.shape} -> Final: {sized.shape}")

        print("\n7. Method chaining:")
        start = time.time()
        processed = img.crop_or_pad([96, 96, 96]).z_normalize().rescale(0.0, 1.0)
        chain_time = time.time() - start
        proc_data = processed.to_numpy()
        print(f"   Time: {chain_time * 1000:.2f}ms")
        print(f"   Range: [{proc_data.min():.3f}, {proc_data.max():.3f}]")

        print("\nDone.")

    except FileNotFoundError:
        print(f"\nFile not found: {volume_path}")
    except Exception as e:
        print(f"\nError: {e}")


if __name__ == "__main__":
    main()
