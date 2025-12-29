#!/usr/bin/env python3
"""Tests for mixed-precision NIfTI save/load (f16, bf16).

This verifies:
1. Save/load consistency for all supported dtypes
2. Disk storage savings from using lower precision formats
3. Numerical precision loss is within expected bounds
"""

import os
import tempfile
import numpy as np
import pytest

import medrs
from medrs import MedicalImage


# Test dtypes and their properties
DTYPE_INFO = {
    "float32": {"bytes": 4, "rtol": 1e-6, "atol": 1e-6},
    "float64": {"bytes": 8, "rtol": 1e-12, "atol": 1e-12},
    "float16": {"bytes": 2, "rtol": 1e-3, "atol": 1e-3},
    "bfloat16": {"bytes": 2, "rtol": 1e-2, "atol": 1e-2},  # bf16 has less precision than f16
    "int16": {"bytes": 2, "rtol": 0, "atol": 1},  # Integer types have rounding
    "uint16": {"bytes": 2, "rtol": 0, "atol": 1},
    "int8": {"bytes": 1, "rtol": 0, "atol": 1},
    "uint8": {"bytes": 1, "rtol": 0, "atol": 1},
}


def create_test_volume(shape=(64, 64, 64), value_range=(-1.0, 1.0)):
    """Create a test volume with realistic values."""
    np.random.seed(42)
    data = np.random.uniform(value_range[0], value_range[1], shape).astype(np.float32)
    return data


class TestMixedPrecisionSaveLoad:
    """Test save/load consistency for various dtypes."""

    @pytest.fixture
    def test_volume(self):
        """Create a test volume."""
        return create_test_volume()

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    def test_save_load_float32(self, test_volume, temp_dir):
        """Test float32 save/load (baseline)."""
        img = MedicalImage.from_numpy(test_volume)
        path = os.path.join(temp_dir, "test_f32.nii.gz")

        img.save(path)
        loaded = medrs.load(path)

        loaded_data = loaded.to_numpy()
        np.testing.assert_allclose(loaded_data, test_volume, rtol=1e-6, atol=1e-6)
        assert loaded.dtype == "f32"

    def test_save_load_bfloat16(self, test_volume, temp_dir):
        """Test bfloat16 save/load."""
        img = MedicalImage.from_numpy(test_volume)
        img_bf16 = img.with_dtype("bfloat16")

        path = os.path.join(temp_dir, "test_bf16.nii.gz")
        img_bf16.save(path)

        # Verify dtype is set correctly
        assert img_bf16.dtype == "bf16"

        # Load and verify
        loaded = medrs.load(path)
        assert loaded.dtype == "bf16"

        # Check values are close (bf16 has ~3 decimal digits of precision)
        loaded_data = loaded.to_numpy()
        np.testing.assert_allclose(
            loaded_data, test_volume,
            rtol=1e-2, atol=1e-2,
            err_msg="bf16 precision loss exceeded expected bounds"
        )

    def test_save_load_float16(self, test_volume, temp_dir):
        """Test float16 (IEEE half) save/load."""
        img = MedicalImage.from_numpy(test_volume)
        img_f16 = img.with_dtype("float16")

        path = os.path.join(temp_dir, "test_f16.nii.gz")
        img_f16.save(path)

        assert img_f16.dtype == "f16"

        loaded = medrs.load(path)
        assert loaded.dtype == "f16"

        # f16 has ~4 decimal digits of precision
        loaded_data = loaded.to_numpy()
        np.testing.assert_allclose(
            loaded_data, test_volume,
            rtol=1e-3, atol=1e-3,
            err_msg="f16 precision loss exceeded expected bounds"
        )

    @pytest.mark.parametrize("dtype", ["int8", "uint8", "int16", "uint16"])
    def test_save_load_integer_types(self, temp_dir, dtype):
        """Test integer dtype save/load."""
        # Create volume with appropriate range for integer type
        if dtype.startswith("u"):
            data = create_test_volume(value_range=(0, 200))
        else:
            data = create_test_volume(value_range=(-100, 100))

        img = MedicalImage.from_numpy(data)
        img_int = img.with_dtype(dtype)

        path = os.path.join(temp_dir, f"test_{dtype}.nii.gz")
        img_int.save(path)

        loaded = medrs.load(path)
        loaded_data = loaded.to_numpy()

        # Integer types have rounding error
        np.testing.assert_allclose(
            loaded_data, np.round(data),
            rtol=0, atol=1,
            err_msg=f"{dtype} save/load failed"
        )


class TestDiskStorageSavings:
    """Benchmark disk storage savings from different precisions."""

    @pytest.fixture
    def large_volume(self):
        """Create a larger test volume for storage benchmarks."""
        return create_test_volume(shape=(128, 128, 128))

    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    def test_storage_savings_bf16(self, large_volume, temp_dir):
        """Verify bf16 achieves ~50% storage reduction vs f32."""
        img = MedicalImage.from_numpy(large_volume)

        path_f32 = os.path.join(temp_dir, "test_f32.nii.gz")
        path_bf16 = os.path.join(temp_dir, "test_bf16.nii.gz")

        img.save(path_f32)
        img.with_dtype("bfloat16").save(path_bf16)

        size_f32 = os.path.getsize(path_f32)
        size_bf16 = os.path.getsize(path_bf16)

        ratio = size_bf16 / size_f32
        print(f"\nbf16 storage: {size_bf16:,} bytes vs f32: {size_f32:,} bytes")
        print(f"Compression ratio: {ratio:.2%} (target: ~50%)")

        # bf16 should be roughly 50% the size (with some variation due to gzip)
        assert ratio < 0.7, f"bf16 should be <70% of f32 size, got {ratio:.2%}"

    def test_storage_savings_f16(self, large_volume, temp_dir):
        """Verify f16 achieves ~50% storage reduction vs f32."""
        img = MedicalImage.from_numpy(large_volume)

        path_f32 = os.path.join(temp_dir, "test_f32.nii.gz")
        path_f16 = os.path.join(temp_dir, "test_f16.nii.gz")

        img.save(path_f32)
        img.with_dtype("float16").save(path_f16)

        size_f32 = os.path.getsize(path_f32)
        size_f16 = os.path.getsize(path_f16)

        ratio = size_f16 / size_f32
        print(f"\nf16 storage: {size_f16:,} bytes vs f32: {size_f32:,} bytes")
        print(f"Compression ratio: {ratio:.2%} (target: ~50%)")

        assert ratio < 0.7, f"f16 should be <70% of f32 size, got {ratio:.2%}"

    def test_storage_comparison_all_dtypes(self, large_volume, temp_dir):
        """Compare storage for all floating point dtypes."""
        img = MedicalImage.from_numpy(large_volume)

        results = {}
        dtypes = ["float64", "float32", "float16", "bfloat16"]

        for dtype in dtypes:
            path = os.path.join(temp_dir, f"test_{dtype}.nii.gz")
            img.with_dtype(dtype).save(path)
            results[dtype] = os.path.getsize(path)

        print("\n=== Disk Storage Comparison ===")
        print(f"Volume shape: {large_volume.shape}")
        print(f"Raw voxels: {large_volume.size:,}")
        print()

        baseline = results["float32"]
        for dtype in dtypes:
            size = results[dtype]
            ratio = size / baseline
            bytes_per_voxel = DTYPE_INFO.get(dtype, {}).get("bytes", "?")
            theoretical = bytes_per_voxel / 4  # vs f32
            print(f"{dtype:10s}: {size:>10,} bytes ({ratio:>6.1%} of f32, theoretical: {theoretical:.0%})")


class TestPrecisionBounds:
    """Test that precision loss is within expected bounds."""

    @pytest.fixture
    def test_volume(self):
        return create_test_volume(shape=(32, 32, 32))

    def test_bf16_precision_typical_values(self, test_volume):
        """Test bf16 precision on typical medical imaging values."""
        img = MedicalImage.from_numpy(test_volume)
        img_bf16 = img.with_dtype("bfloat16")

        original = test_volume
        converted = img_bf16.to_numpy()

        # Compute relative error
        rel_error = np.abs(converted - original) / (np.abs(original) + 1e-10)
        max_rel_error = np.max(rel_error)
        mean_rel_error = np.mean(rel_error)

        print("\nbf16 precision analysis:")
        print(f"  Max relative error: {max_rel_error:.6f}")
        print(f"  Mean relative error: {mean_rel_error:.6f}")

        # bf16 has ~7 bits of mantissa = ~2 decimal digits
        assert max_rel_error < 0.01, f"bf16 max relative error too high: {max_rel_error}"

    def test_f16_precision_typical_values(self, test_volume):
        """Test f16 precision on typical medical imaging values."""
        img = MedicalImage.from_numpy(test_volume)
        img_f16 = img.with_dtype("float16")

        original = test_volume
        converted = img_f16.to_numpy()

        rel_error = np.abs(converted - original) / (np.abs(original) + 1e-10)
        max_rel_error = np.max(rel_error)
        mean_rel_error = np.mean(rel_error)

        print("\nf16 precision analysis:")
        print(f"  Max relative error: {max_rel_error:.6f}")
        print(f"  Mean relative error: {mean_rel_error:.6f}")

        # f16 has ~10 bits of mantissa = ~3 decimal digits
        # For values near zero, relative error can be higher, so we use 1% bound
        assert max_rel_error < 0.01, f"f16 max relative error too high: {max_rel_error}"


class TestRoundTrip:
    """Test round-trip consistency (save → load → compare)."""

    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
    def test_roundtrip_uncompressed(self, temp_dir, dtype):
        """Test save/load roundtrip with uncompressed .nii files."""
        data = create_test_volume(shape=(32, 32, 32))
        img = MedicalImage.from_numpy(data).with_dtype(dtype)

        path = os.path.join(temp_dir, f"test_{dtype}.nii")  # uncompressed
        img.save(path)
        loaded = medrs.load(path)

        # Values should be exactly equal after roundtrip (no additional precision loss)
        original_values = img.to_numpy()
        loaded_values = loaded.to_numpy()
        np.testing.assert_array_equal(
            original_values, loaded_values,
            err_msg=f"Roundtrip {dtype} values should be exactly equal"
        )

    @pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
    def test_roundtrip_compressed(self, temp_dir, dtype):
        """Test save/load roundtrip with gzip compressed .nii.gz files."""
        data = create_test_volume(shape=(32, 32, 32))
        img = MedicalImage.from_numpy(data).with_dtype(dtype)

        path = os.path.join(temp_dir, f"test_{dtype}.nii.gz")
        img.save(path)
        loaded = medrs.load(path)

        original_values = img.to_numpy()
        loaded_values = loaded.to_numpy()
        np.testing.assert_array_equal(
            original_values, loaded_values,
            err_msg=f"Roundtrip {dtype} (gzip) values should be exactly equal"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
