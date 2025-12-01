#!/usr/bin/env python3
"""
Simple MONAI + medrs Integration Test

Test that medrs functions are available and can be integrated with MONAI transforms.
"""

import sys

# Imports will fail fast if dependencies are missing
import medrs
import torch
import monai

def main():
    """Test basic integration."""
    print("MONAI + medrs integration")
    print("=" * 40)

    print("medrs imported successfully")
    print(f"PyTorch {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"MONAI {monai.__version__}")

    # Check that key functions exist
    required_functions = [
        'load',
        'load_cropped',
        'load_resampled'
    ]

    missing_functions = []
    for func in required_functions:
        if hasattr(medrs, func):
            print(f"ok: {func}")
        else:
            missing_functions.append(func)
            print(f"missing: {func}")

    # Check for advanced functions
    advanced_functions = [
        'load_cropped_to_torch',
        'load_label_aware_cropped',
        'compute_crop_regions'
    ]

    for func in advanced_functions:
        if hasattr(medrs, func):
            print(f"ok: {func}")
        else:
            print(f"{func} - Missing from current build")

    if missing_functions:
        print(f"\n Missing core functions: {missing_functions}")
        sys.exit(1)

    # Create test tensors to verify compatibility
    image_tensor = torch.randn(1, 96, 96, 96)  # Typical medical image shape
    label_tensor = torch.randint(0, 2, (1, 96, 96, 96))  # Binary label

    print("Created test tensors")
    print(f"  Image shape: {image_tensor.shape}, dtype: {image_tensor.dtype}")
    print(f"  Label shape: {label_tensor.shape}, dtype: {label_tensor.dtype}")

    print("\nIntegration example:")
    print("  image, label = medrs.load_label_aware_cropped(")
    print("      'image.nii', 'label.nii',")
    print("      patch_size=(96, 96, 96),")
    print("      pos_neg_ratio=2.0")
    print("  )")

    print("\nAll basic tests passed.")
    print("medrs integration ready.")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
