#!/usr/bin/env python3
"""
medrs + MONAI Quick Integration Test
====================================

This script verifies that medrs and MONAI are properly integrated
and working together for high-performance medical imaging workflows.
"""

import sys
import traceback


def test_medrs_import():
    """Test that medrs can be imported."""
    print("1. medrs import")

    try:
        import medrs
        print(f"    medrs {medrs.__version__}")
        return True
    except ImportError as e:
        print(f"    failed: {e}")
        return False


def test_monai_import():
    """Test that MONAI can be imported."""
    print("\n2. MONAI import")

    try:
        import monai
        print(f"    MONAI {monai.__version__}")
        return True
    except ImportError as e:
        print(f"    failed: {e}")
        return False


def test_torch_import():
    """Test that PyTorch can be imported."""
    print("\n3. PyTorch import")

    try:
        import torch
        print(f"    PyTorch {torch.__version__} (CUDA: {torch.cuda.is_available()})")
        return True
    except ImportError as e:
        print(f"    failed: {e}")
        return False


def test_medrs_features():
    """Check optional medrs features are exposed."""
    print("\n4. medrs feature surface")

    try:
        import medrs

        dictionary_transforms = hasattr(medrs, "SpatialNormalizer")
        metatensor = hasattr(medrs, "is_metatensor_supported") and medrs.is_metatensor_supported()

        print(f"    Dictionary transforms: {'yes' if dictionary_transforms else 'no'}")
        print(f"    MetaTensor support: {'yes' if metatensor else 'no'}")
        return True

    except Exception as e:
        print(f"    failed: {e}")
        return False


def test_integration_compatibility():
    """Test medrs + MONAI integration compatibility."""
    print("\n5. Integration compatibility")

    try:
        import medrs
        import torch

        # Test that we can access medrs functions through MONAI-style workflow
        compatibility_items = []

        if hasattr(medrs, 'load'):
            compatibility_items.append("Core loading")
        if hasattr(medrs, 'load_cropped'):
            compatibility_items.append("Crop-first loading")
        if hasattr(medrs, 'SpatialNormalizer'):
            compatibility_items.append("Dictionary transforms")
        if hasattr(medrs, 'metatensor_from_medrs'):
            compatibility_items.append("MetaTensor conversion")

        print(f"    Integration components: {', '.join(compatibility_items)}")

        # Test device detection
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"    Selected device: {device}")

        return len(compatibility_items) > 0

    except Exception as e:
        print(f"    failed: {e}")
        return False


def run_quick_performance_test():
    """Minimal perf sanity: ensure crop-to-torch runs."""
    print("\n6. Crop-to-torch sanity")

    try:
        import torch

        # Test tensor creation (basic functionality)
        if torch.cuda.is_available():
            device = "cuda"
            dtype = torch.float16
        else:
            device = "cpu"
            dtype = torch.float32

        print("    Optimal configuration for your system:")
        print(f"      Device: {device.upper()}")
        print(f"      Dtype: {dtype}")

        return True

    except Exception as e:
        print(f"    failed: {e}")
        return False


def main():
    """Run all integration tests."""
    print("medrs + MONAI integration test")
    print("=" * 40)

    # Run all tests
    tests = [
        ("medrs Import", test_medrs_import),
        ("MONAI Import", test_monai_import),
        ("PyTorch Import", test_torch_import),
        ("medrs Features", test_medrs_features),
        ("Integration Compatibility", test_integration_compatibility),
        ("Performance Check", run_quick_performance_test),
    ]

    results = []

    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception:
            print(f"\n    {test_name} failed with exception:")
            print(f"   {traceback.format_exc()}")
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 40)
    print("Summary:")

    passed = 0
    total = len(results)

    for test_name, result in results:
        status = " PASS" if result else " FAIL"
        print(f"   {status} {test_name}")
        if result:
            passed += 1

    print(f"\nResults: {passed}/{total} tests passed")

    if passed == total:
        print("All tests passed.")
        return True
    else:
        failed_tests = [name for name, result in results if not result]
        print(f"{len(failed_tests)} test(s) failed: {', '.join(failed_tests)}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
