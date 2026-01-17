#!/usr/bin/env python3
"""
medrs Dictionary Transforms Example

Demonstrates multi-modal dictionary transforms with coordinated cropping
and spatial normalization for different slice thickness and orientations.
"""

import torch
from monai.transforms import Compose, EnsureChannelFirst
from monai.data import DataLoader, Dataset

import medrs


def main():
    """Demonstrate dictionary transforms functionality."""

    print("medrs Dictionary Transforms Example")
    print("==================================")
    print()

    # Example 1: Spatial Normalization
    print("1. Spatial Normalization")
    data_dict = {
        "t1": "subject001_t1.nii.gz",  # 1mm isotropic, RAS
        "t2": "subject001_t2.nii.gz",  # 1.2x1.0x3.0mm, LAS
        "flair": "subject001_flair.nii.gz",  # 0.8x0.8x5.0mm, RAS
    }

    normalizer = medrs.SpatialNormalizer(
        target_spacing=(1.0, 1.0, 1.0), target_orientation="RAS", reference_key="t1"
    )

    print(f"   Input modalities: {list(data_dict.keys())}")
    print("   Target: 1mm isotropic, RAS orientation")
    print("   Reference: T1 volume")

    # Example 2: Coordinated Cropping
    print("\n2. Coordinated Cropping")
    crop_loader = medrs.CoordinatedCropLoader(
        keys=["t1", "t2", "flair", "seg"],
        crop_size=(96, 96, 96),
        reference_key="t1",
        spatial_normalizer=normalizer,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    print("   Crop size: 96x96x96")
    print("   Coordinated: Same anatomical region across modalities")

    # Example 3: MONAI Integration
    print("\n3. MONAI Integration")
    medrs_transform = medrs.MonaiCompatibleTransform(
        loader=medrs.CoordinatedCropLoader(
            keys=["image", "label"],
            crop_size=(96, 96, 96),
            spatial_normalizer=medrs.SpatialNormalizer(
                target_spacing=(1.0, 1.0, 1.0), target_orientation="RAS"
            ),
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
    )

    # Complete MONAI pipeline
    transform_pipeline = Compose(
        [
            medrs_transform,  # High-performance loading
            EnsureChannelFirst(keys=["image", "label"]),  # Add channel dim
        ]
    )

    print("   Pipeline: medrs loading + MONAI transforms")
    print("   Benefits: Reduced memory + faster I/O")

    # Example 4: Training Dataset
    print("\n4. Training Dataset")
    dataset_items = [
        {"image": f"subject_{i:03d}_t1.nii.gz", "label": f"subject_{i:03d}_seg.nii.gz"}
        for i in range(10)
    ]

    class MedrsDataset(Dataset):
        def __init__(self, data_items, transform=None):
            self.data_items = data_items
            self.transform = transform

        def __len__(self):
            return len(self.data_items)

        def __getitem__(self, index):
            item = self.data_items[index]
            if self.transform:
                item = self.transform(item)
            return item

    dataset = MedrsDataset(dataset_items, transform=transform_pipeline)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

    print(f"   Dataset: {len(dataset)} samples")
    print("   DataLoader: batch_size=4, num_workers=0 (medrs handles concurrency)")
    print("   Result: High-performance multi-modal training ready")

    print("\nKey Features:")
    print(" Handles different slice thickness automatically")
    print(" Coordinated cropping across all modalities")
    print(" Full MONAI compatibility")
    print(" Reduced memory, faster I/O")


if __name__ == "__main__":
    main()
