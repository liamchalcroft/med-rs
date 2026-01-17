#!/usr/bin/env python3
"""Example: Using medrs as drop-in replacement for MONAI transforms.

This example demonstrates how to use medrs transforms as 1:1 replacements
for MONAI transforms in existing pipelines, achieving faster I/O performance.

Before (MONAI):
    from monai.transforms import LoadImaged, RandCropByPosNegLabeld

After (medrs - just change imports):
    from medrs.monai_compat import MedrsLoadImaged, MedrsRandCropByPosNegLabeld

Requirements:
    pip install medrs torch monai
"""

import tempfile
import time

import torch
from monai.transforms import Compose
import numpy as np

import medrs
from medrs import MedicalImage


def create_test_data(tmpdir: str) -> tuple[str, str]:
    """Create test NIfTI files for the example."""
    # Create synthetic brain volume (64^3)
    np.random.seed(42)
    image_data = np.random.randn(64, 64, 64).astype(np.float32)
    image_data = (image_data - image_data.min()) / (image_data.max() - image_data.min())

    # Create synthetic segmentation label
    label_data = np.zeros((64, 64, 64), dtype=np.float32)
    # Add some foreground regions
    label_data[20:40, 20:40, 20:40] = 1.0  # Central cube

    # Save as NIfTI
    image_path = f"{tmpdir}/image.nii"
    label_path = f"{tmpdir}/label.nii"

    MedicalImage.from_numpy(image_data).save(image_path)
    MedicalImage.from_numpy(label_data).save(label_path)

    return image_path, label_path


def example_basic_loading():
    """Example 1: Basic image loading with MedrsLoadImaged."""
    print("\n" + "=" * 60)
    print("Example 1: Basic Loading (MedrsLoadImaged)")
    print("=" * 60)

    from medrs.monai_compat import MedrsLoadImaged

    with tempfile.TemporaryDirectory() as tmpdir:
        image_path, label_path = create_test_data(tmpdir)

        # Create loader (same API as MONAI's LoadImaged)
        loader = MedrsLoadImaged(
            keys=["image", "label"],
            ensure_channel_first=True,
        )

        # Load data
        data = {"image": image_path, "label": label_path}
        result = loader(data)

        print(f"Image shape: {result['image'].shape}")
        print(f"Label shape: {result['label'].shape}")
        print(f"Image dtype: {result['image'].dtype}")
        print(f"Has affine: {'affine' in result['image'].meta}")


def example_label_aware_cropping():
    """Example 2: Label-aware cropping with MedrsRandCropByPosNegLabeld."""
    print("\n" + "=" * 60)
    print("Example 2: Label-Aware Cropping (crop-first loading)")
    print("=" * 60)

    from medrs.monai_compat import MedrsLoadImaged, MedrsRandCropByPosNegLabeld

    with tempfile.TemporaryDirectory() as tmpdir:
        image_path, label_path = create_test_data(tmpdir)

        # Create pipeline (same API as MONAI)
        pipeline = Compose(
            [
                MedrsLoadImaged(keys=["image", "label"], ensure_channel_first=True),
                MedrsRandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=(32, 32, 32),
                    pos=1,
                    neg=1,
                    num_samples=2,  # Generate 2 crops
                ),
            ]
        )

        # Run pipeline
        data = {"image": image_path, "label": label_path}
        results = pipeline(data)

        print(f"Number of samples: {len(results)}")
        for i, sample in enumerate(results):
            print(f"  Sample {i}: image={sample['image'].shape}, label={sample['label'].shape}")


def example_full_training_pipeline():
    """Example 3: Full training pipeline with multiple transforms."""
    print("\n" + "=" * 60)
    print("Example 3: Full Training Pipeline")
    print("=" * 60)

    from medrs.monai_compat import (
        MedrsLoadImaged,
        MedrsRandCropByPosNegLabeld,
        MedrsOrientationd,
    )
    from monai.transforms import (
        RandFlipd,
        RandGaussianNoised,
        EnsureTyped,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        image_path, label_path = create_test_data(tmpdir)

        # Build training pipeline mixing medrs and MONAI transforms
        train_transforms = Compose(
            [
                # medrs: Fast loading
                MedrsLoadImaged(keys=["image", "label"], ensure_channel_first=True),
                # medrs: Reorientation
                MedrsOrientationd(keys=["image", "label"], axcodes="RAS"),
                # medrs: Fast label-aware cropping
                MedrsRandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=(32, 32, 32),
                    pos=1,
                    neg=1,
                    num_samples=4,
                ),
                # MONAI: Standard augmentations (still work with medrs outputs)
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                RandGaussianNoised(keys=["image"], prob=0.2, std=0.1),
                # MONAI: Ensure correct tensor type
                EnsureTyped(keys=["image", "label"]),
            ]
        )

        # Run pipeline
        data = {"image": image_path, "label": label_path}
        results = train_transforms(data)

        print(f"Generated {len(results)} training samples")
        for i, sample in enumerate(results[:2]):  # Show first 2
            print(f"  Sample {i}:")
            print(f"    Image: {sample['image'].shape}, dtype={sample['image'].dtype}")
            print(f"    Label: {sample['label'].shape}, dtype={sample['label'].dtype}")


def example_performance_comparison():
    """Example 4: Performance comparison between medrs and MONAI."""
    print("\n" + "=" * 60)
    print("Example 4: Performance Comparison")
    print("=" * 60)

    from medrs.monai_compat import MedrsLoadImaged
    from monai.transforms import LoadImaged as MonaiLoadImaged

    with tempfile.TemporaryDirectory() as tmpdir:
        image_path, label_path = create_test_data(tmpdir)

        # Benchmark medrs
        medrs_loader = MedrsLoadImaged(keys=["image"], ensure_channel_first=True)
        data = {"image": image_path}

        # Warmup
        _ = medrs_loader(data)

        start = time.perf_counter()
        for _ in range(10):
            _ = medrs_loader(data)
        medrs_time = (time.perf_counter() - start) / 10

        # Benchmark MONAI
        monai_loader = MonaiLoadImaged(keys=["image"], ensure_channel_first=True)

        # Warmup
        _ = monai_loader(data)

        start = time.perf_counter()
        for _ in range(10):
            _ = monai_loader(data)
        monai_time = (time.perf_counter() - start) / 10

        speedup = monai_time / medrs_time
        print(f"medrs LoadImaged:  {medrs_time * 1000:.2f} ms")
        print(f"MONAI LoadImaged:  {monai_time * 1000:.2f} ms")
        print(f"Speedup: {speedup:.1f}x faster with medrs")


def example_saving():
    """Example 5: Saving results with MedrsSaveImaged."""
    print("\n" + "=" * 60)
    print("Example 5: Saving Results")
    print("=" * 60)

    from medrs.monai_compat import MedrsLoadImaged, MedrsSaveImaged

    with tempfile.TemporaryDirectory() as tmpdir:
        image_path, _ = create_test_data(tmpdir)
        output_dir = f"{tmpdir}/output"

        # Load, process, save
        pipeline = Compose(
            [
                MedrsLoadImaged(keys=["image"], ensure_channel_first=True),
                MedrsSaveImaged(
                    keys=["image"],
                    output_dir=output_dir,
                    output_postfix="processed",
                    output_ext=".nii.gz",
                    print_log=True,
                ),
            ]
        )

        data = {"image": image_path}
        pipeline(data)


if __name__ == "__main__":
    print("medrs MONAI Drop-in Replacement Examples")
    print("=========================================")
    print()
    print("These examples show how to use medrs transforms as 1:1")
    print("replacements for MONAI transforms with minimal code changes.")
    print()

    example_basic_loading()
    example_label_aware_cropping()
    example_full_training_pipeline()
    example_performance_comparison()
    example_saving()

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)
