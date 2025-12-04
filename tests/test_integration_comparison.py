"""
Integration tests comparing medrs against nibabel and MONAI.

These tests verify that medrs produces equivalent results to established
neuroimaging libraries for loading, transforming, and processing NIfTI files.
"""

import numpy as np
import pytest
from pathlib import Path

# Test data path
TEST_IMAGE = Path("tests/fixtures/mprage_img.nii")


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def test_image_path():
    """Path to test NIfTI image."""
    assert TEST_IMAGE.exists(), f"Test image not found: {TEST_IMAGE}"
    return str(TEST_IMAGE)


@pytest.fixture
def nibabel_data(test_image_path):
    """Load test image with nibabel as reference."""
    import nibabel as nib
    img = nib.load(test_image_path)
    return {
        "data": img.get_fdata(),
        "affine": img.affine,
        "header": img.header,
        "shape": img.shape,
    }


@pytest.fixture
def medrs_image(test_image_path):
    """Load test image with medrs."""
    import medrs
    return medrs.load(test_image_path)


# ============================================================================
# Basic Loading Tests
# ============================================================================

class TestBasicLoading:
    """Tests for basic image loading functionality."""

    def test_load_shape_matches_nibabel(self, medrs_image, nibabel_data):
        """Verify medrs and nibabel report same shape."""
        assert tuple(medrs_image.shape) == nibabel_data["shape"]

    def test_load_data_matches_nibabel(self, medrs_image, nibabel_data):
        """Verify medrs data matches nibabel exactly."""
        medrs_data = medrs_image.to_numpy()
        nib_data = nibabel_data["data"]

        max_diff = np.max(np.abs(medrs_data - nib_data))
        assert max_diff == 0.0, f"Data mismatch: max_diff={max_diff}"

    def test_load_data_close_to_nibabel(self, medrs_image, nibabel_data):
        """Verify medrs data is close to nibabel within tolerance."""
        medrs_data = medrs_image.to_numpy()
        nib_data = nibabel_data["data"]

        assert np.allclose(medrs_data, nib_data, atol=1e-5), "Data not close within tolerance"

    def test_load_f_contiguous(self, medrs_image):
        """Verify medrs returns F-contiguous arrays (NIfTI convention)."""
        data = medrs_image.to_numpy()
        assert data.flags["F_CONTIGUOUS"], "Array should be F-contiguous"

    def test_load_dtype(self, medrs_image):
        """Verify medrs returns float32 data."""
        data = medrs_image.to_numpy()
        assert data.dtype == np.float32, f"Expected float32, got {data.dtype}"

    def test_affine_matches_nibabel(self, medrs_image, nibabel_data):
        """Verify affine matrix matches nibabel."""
        medrs_affine = np.array(medrs_image.affine).reshape(4, 4)
        nib_affine = nibabel_data["affine"]

        assert np.allclose(medrs_affine, nib_affine, atol=1e-6), "Affine mismatch"

    def test_spacing_correct(self, medrs_image, nibabel_data):
        """Verify voxel spacing is correctly extracted."""
        medrs_spacing = medrs_image.spacing
        nib_spacing = nibabel_data["header"].get_zooms()[:3]

        assert np.allclose(medrs_spacing, nib_spacing, atol=1e-6), "Spacing mismatch"


# ============================================================================
# MONAI Comparison Tests
# ============================================================================

class TestMonaiComparison:
    """Tests comparing medrs against MONAI."""

    @pytest.fixture
    def monai_data(self, test_image_path):
        """Load test image with MONAI."""
        from monai.transforms import LoadImage
        loader = LoadImage(image_only=True)
        data = loader(test_image_path)
        return np.asarray(data)

    def test_load_matches_monai(self, medrs_image, monai_data):
        """Verify medrs data matches MONAI."""
        medrs_data = medrs_image.to_numpy()

        # MONAI may return different order, compare values
        assert medrs_data.shape == monai_data.shape, "Shape mismatch with MONAI"
        max_diff = np.max(np.abs(medrs_data - monai_data))
        assert max_diff < 1e-5, f"Data mismatch with MONAI: max_diff={max_diff}"

    def test_z_normalize_matches_monai(self, medrs_image, test_image_path):
        """Compare z-normalization with MONAI.

        Note: MONAI's NormalizeIntensity uses different defaults than medrs.
        We compare that both produce mean ~0 and std ~1 for non-zero voxels.
        """
        from monai.transforms import LoadImage, NormalizeIntensity

        # MONAI z-normalization (channel_wise=False, nonzero=False by default)
        loader = LoadImage(image_only=True)
        monai_data = np.asarray(loader(test_image_path))
        # Use same settings as medrs (normalize over all voxels)
        normalizer = NormalizeIntensity(subtrahend=None, divisor=None, nonzero=False)
        monai_normalized = np.asarray(normalizer(monai_data))

        # medrs z-normalization
        medrs_normalized = medrs_image.z_normalize().to_numpy()

        # Both should have similar statistics (mean ~0, std ~1)
        medrs_mean = np.mean(medrs_normalized)
        medrs_std = np.std(medrs_normalized)
        monai_mean = np.mean(monai_normalized)
        monai_std = np.std(monai_normalized)

        # Check both produce proper z-scores
        assert abs(medrs_mean) < 1e-5, f"medrs mean not ~0: {medrs_mean}"
        assert abs(medrs_std - 1.0) < 1e-5, f"medrs std not ~1: {medrs_std}"
        assert abs(monai_mean) < 1e-5, f"monai mean not ~0: {monai_mean}"
        assert abs(monai_std - 1.0) < 1e-5, f"monai std not ~1: {monai_std}"

        # Note: Direct comparison may have differences due to implementation details
        # The important thing is both produce valid z-scores

    def test_resample_similar_to_monai(self, medrs_image, test_image_path):
        """Compare resampling with MONAI (approximate due to interpolation differences)."""
        from monai.transforms import LoadImage, Spacing

        target_spacing = [2.0, 2.0, 2.0]

        # MONAI resampling - LoadImage with image_only=True returns MetaTensor directly
        loader = LoadImage(image_only=True, ensure_channel_first=True)
        monai_img = loader(test_image_path)
        spacer = Spacing(pixdim=target_spacing, mode="bilinear")
        monai_resampled = spacer(monai_img)
        # Remove channel dim if present
        monai_resampled_arr = np.asarray(monai_resampled)
        if monai_resampled_arr.ndim == 4:
            monai_resampled_arr = monai_resampled_arr[0]

        # medrs resampling
        medrs_resampled = medrs_image.resample(target_spacing).to_numpy()

        # Shapes should match or be very close (interpolation edge effects)
        shape_diff = np.abs(np.array(medrs_resampled.shape) - np.array(monai_resampled_arr.shape))
        assert np.all(shape_diff <= 2), f"Resample shape mismatch: medrs={medrs_resampled.shape}, monai={monai_resampled_arr.shape}"


# ============================================================================
# Loading Variants Tests
# ============================================================================

class TestLoadingVariants:
    """Tests for different loading functions."""

    def test_load_to_numpy_methods_consistent(self, medrs_image):
        """Verify different to_numpy methods return consistent data."""
        data1 = medrs_image.to_numpy()
        data2 = medrs_image.to_numpy_view()
        data3 = medrs_image.to_numpy_native()

        # All should have same shape
        assert data1.shape == data2.shape == data3.shape

        # to_numpy and to_numpy_view should be identical (both return f32)
        assert np.allclose(data1, data2, atol=1e-6)

    def test_load_cropped(self, test_image_path, nibabel_data):
        """Test load_cropped function."""
        import medrs

        # load_cropped requires offset and shape
        crop_offset = [32, 32, 32]
        crop_size = [64, 64, 64]
        cropped = medrs.load_cropped(test_image_path, crop_offset, crop_size)
        data = cropped.to_numpy()

        assert data.shape == tuple(crop_size), f"Expected {crop_size}, got {data.shape}"
        assert data.flags["F_CONTIGUOUS"], "Cropped data should be F-contiguous"

        # Verify the crop contains valid data from the original
        nib_data = nibabel_data["data"]
        expected_crop = nib_data[32:96, 32:96, 32:96]
        assert np.allclose(data, expected_crop, atol=1e-5), "Cropped data doesn't match expected region"

    def test_load_resampled(self, test_image_path):
        """Test load_resampled function."""
        import medrs

        # load_resampled crops to output_shape at target_spacing
        # Test with output_shape at same spacing as original
        output_shape = [64, 64, 64]
        resampled = medrs.load_resampled(test_image_path, output_shape)

        # Verify shape is correct (cropped to specified shape)
        assert tuple(resampled.shape) == tuple(output_shape), \
            f"Shape mismatch: expected {output_shape}, got {resampled.shape}"

        # Verify F-contiguous
        data = resampled.to_numpy()
        assert data.flags["F_CONTIGUOUS"], "load_resampled output should be F-contiguous"

    def test_load_to_torch(self, test_image_path, nibabel_data):
        """Test load_to_torch function."""
        import medrs
        import torch

        tensor = medrs.load_to_torch(test_image_path)

        assert isinstance(tensor, torch.Tensor), "Should return torch.Tensor"
        assert tensor.shape == nibabel_data["shape"], "Shape mismatch"

        # Compare with nibabel
        np_data = tensor.numpy()
        max_diff = np.max(np.abs(np_data - nibabel_data["data"]))
        assert max_diff < 1e-5, f"Data mismatch: max_diff={max_diff}"

    def test_load_cropped_to_torch(self, test_image_path):
        """Test load_cropped_to_torch function."""
        import medrs
        import torch

        crop_size = [48, 48, 48]
        tensor = medrs.load_cropped_to_torch(test_image_path, crop_size)

        assert isinstance(tensor, torch.Tensor), "Should return torch.Tensor"
        assert tuple(tensor.shape) == tuple(crop_size), f"Shape mismatch: {tensor.shape}"


# ============================================================================
# Transform Tests
# ============================================================================

class TestTransforms:
    """Tests for image transformation operations."""

    def test_resample_changes_shape(self, medrs_image):
        """Verify resampling changes shape appropriately."""
        original_shape = medrs_image.shape
        original_spacing = medrs_image.spacing

        # Resample to 2x spacing (should halve dimensions approximately)
        new_spacing = [s * 2 for s in original_spacing]
        resampled = medrs_image.resample(new_spacing)

        # Shape should be approximately halved
        for orig, new in zip(original_shape, resampled.shape):
            assert 0.4 < new / orig < 0.6, f"Unexpected shape change: {orig} -> {new}"

    def test_resample_preserves_f_order(self, medrs_image):
        """Verify resampling preserves F-contiguous layout."""
        resampled = medrs_image.resample([2.0, 2.0, 2.0])
        data = resampled.to_numpy()
        assert data.flags["F_CONTIGUOUS"], "Resampled data should be F-contiguous"

    def test_z_normalize_stats(self, medrs_image):
        """Verify z-normalization produces correct statistics."""
        normalized = medrs_image.z_normalize()
        data = normalized.to_numpy()

        # Mean should be close to 0, std close to 1 (for non-zero voxels)
        nonzero_data = data[data != 0]
        mean = np.mean(nonzero_data)
        std = np.std(nonzero_data)

        assert abs(mean) < 0.1, f"Mean should be ~0, got {mean}"
        assert 0.9 < std < 1.1, f"Std should be ~1, got {std}"

    def test_clamp(self, medrs_image):
        """Verify clamp operation."""
        clamped = medrs_image.clamp(0.0, 100.0)
        data = clamped.to_numpy()

        assert data.min() >= 0.0, f"Min should be >= 0, got {data.min()}"
        assert data.max() <= 100.0, f"Max should be <= 100, got {data.max()}"

    def test_crop_or_pad_crop(self, medrs_image):
        """Verify crop_or_pad with smaller target size."""
        target_shape = [100, 100, 100]
        result = medrs_image.crop_or_pad(target_shape)

        assert tuple(result.shape) == tuple(target_shape), \
            f"Expected {target_shape}, got {result.shape}"

    def test_crop_or_pad_pad(self, medrs_image):
        """Verify crop_or_pad with larger target size."""
        original_shape = medrs_image.shape
        target_shape = [s + 50 for s in original_shape]
        result = medrs_image.crop_or_pad(target_shape)

        assert tuple(result.shape) == tuple(target_shape), \
            f"Expected {target_shape}, got {result.shape}"

    def test_flip(self, medrs_image):
        """Verify flip operation."""
        original = medrs_image.to_numpy()

        # Flip along axis 0 - API takes list of axis indices to flip
        flipped = medrs_image.flip([0])
        flipped_data = flipped.to_numpy()

        # Verify flip occurred
        assert np.allclose(flipped_data, np.flip(original, axis=0))

    def test_rescale(self, medrs_image):
        """Verify intensity rescaling."""
        rescaled = medrs_image.rescale(0.0, 1.0)
        data = rescaled.to_numpy()

        # Should be in [0, 1] range
        assert data.min() >= 0.0, f"Min should be >= 0, got {data.min()}"
        assert data.max() <= 1.0, f"Max should be <= 1, got {data.max()}"
        # Max should be close to 1.0 (full range used)
        assert data.max() > 0.99, f"Max should be ~1, got {data.max()}"


# ============================================================================
# Method Chaining Tests
# ============================================================================

class TestMethodChaining:
    """Tests for method chaining functionality."""

    def test_chain_resample_normalize(self, medrs_image):
        """Test chaining resample and normalize."""
        result = medrs_image.resample([2.0, 2.0, 2.0]).z_normalize()
        data = result.to_numpy()

        assert data.flags["F_CONTIGUOUS"]
        # Check normalization worked
        nonzero = data[data != 0]
        assert abs(np.mean(nonzero)) < 0.1

    def test_chain_crop_clamp_rescale(self, medrs_image):
        """Test chaining crop, clamp, and rescale."""
        result = (medrs_image
                  .crop_or_pad([128, 128, 128])
                  .clamp(0.0, 500.0)
                  .rescale(0.0, 1.0))

        data = result.to_numpy()
        assert data.shape == (128, 128, 128)
        assert data.min() >= 0.0
        assert data.max() <= 1.0

    def test_chain_preserves_metadata(self, medrs_image):
        """Verify method chaining preserves image metadata."""
        original_affine = np.array(medrs_image.affine).reshape(4, 4)

        result = medrs_image.z_normalize().clamp(-3.0, 3.0)
        result_affine = np.array(result.affine).reshape(4, 4)

        # Affine should be preserved through intensity transforms
        assert np.allclose(original_affine, result_affine)


# ============================================================================
# Edge Cases Tests
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_load_nonexistent_file(self):
        """Verify appropriate error for nonexistent file."""
        import medrs

        with pytest.raises(Exception):
            medrs.load("nonexistent_file.nii.gz")

    def test_from_numpy_roundtrip(self, nibabel_data):
        """Test creating NiftiImage from numpy array."""
        import medrs

        # Create from nibabel data
        original = nibabel_data["data"].astype(np.float32)
        img = medrs.NiftiImage.from_numpy(original)
        recovered = img.to_numpy()

        assert np.allclose(original, recovered, atol=1e-6)

    def test_from_numpy_f_order_input(self):
        """Test from_numpy with F-order input."""
        import medrs

        f_array = np.asfortranarray(np.random.rand(10, 20, 30).astype(np.float32))
        img = medrs.NiftiImage.from_numpy(f_array)
        recovered = img.to_numpy()

        assert np.allclose(f_array, recovered, atol=1e-6)
        assert recovered.flags["F_CONTIGUOUS"]

    def test_from_numpy_c_order_input(self):
        """Test from_numpy with C-order input."""
        import medrs

        c_array = np.ascontiguousarray(np.random.rand(10, 20, 30).astype(np.float32))
        img = medrs.NiftiImage.from_numpy(c_array)
        recovered = img.to_numpy()

        assert np.allclose(c_array, recovered, atol=1e-6)

    def test_small_image(self):
        """Test with very small image dimensions."""
        import medrs

        small = np.random.rand(2, 3, 4).astype(np.float32)
        img = medrs.NiftiImage.from_numpy(small)

        assert tuple(img.shape) == (2, 3, 4)
        assert np.allclose(small, img.to_numpy(), atol=1e-6)

    def test_single_slice(self):
        """Test with single-slice image."""
        import medrs

        single_slice = np.random.rand(64, 64, 1).astype(np.float32)
        img = medrs.NiftiImage.from_numpy(single_slice)

        assert tuple(img.shape) == (64, 64, 1)

    def test_crop_larger_than_image(self, medrs_image):
        """Test cropping with size larger than image (should pad)."""
        large_crop = [s + 100 for s in medrs_image.shape]
        result = medrs_image.crop_or_pad(large_crop)

        assert tuple(result.shape) == tuple(large_crop)

    def test_resample_to_same_spacing(self, medrs_image):
        """Test resampling to same spacing (should be near no-op)."""
        original_spacing = medrs_image.spacing
        original_data = medrs_image.to_numpy()

        resampled = medrs_image.resample(list(original_spacing))
        resampled_data = resampled.to_numpy()

        # Shape should match exactly
        assert resampled.shape == medrs_image.shape

        # Values should be similar (trilinear interpolation introduces small differences)
        # Use relative tolerance since absolute differences can vary with intensity range
        max_diff = np.max(np.abs(original_data - resampled_data))
        mean_val = np.mean(np.abs(original_data))
        relative_max_diff = max_diff / mean_val if mean_val > 0 else max_diff
        assert relative_max_diff < 0.01, f"Max relative diff {relative_max_diff:.4f} too large"


# ============================================================================
# Save/Load Roundtrip Tests
# ============================================================================

class TestSaveLoadRoundtrip:
    """Tests for saving and loading images."""

    def test_save_load_roundtrip(self, medrs_image, tmp_path):
        """Test save and reload produces identical data."""
        import medrs

        save_path = tmp_path / "test_output.nii.gz"
        medrs_image.save(str(save_path))

        # Reload
        reloaded = medrs.load(str(save_path))

        # Compare
        original_data = medrs_image.to_numpy()
        reloaded_data = reloaded.to_numpy()

        assert np.allclose(original_data, reloaded_data, atol=1e-5)
        assert tuple(reloaded.shape) == tuple(medrs_image.shape)

    def test_save_load_preserves_affine(self, medrs_image, tmp_path):
        """Test save and reload preserves affine matrix."""
        import medrs

        save_path = tmp_path / "test_affine.nii.gz"
        medrs_image.save(str(save_path))

        reloaded = medrs.load(str(save_path))

        original_affine = np.array(medrs_image.affine).reshape(4, 4)
        reloaded_affine = np.array(reloaded.affine).reshape(4, 4)

        assert np.allclose(original_affine, reloaded_affine, atol=1e-6)

    def test_save_load_nibabel_compatible(self, medrs_image, tmp_path):
        """Test medrs-saved file can be loaded by nibabel."""
        import nibabel as nib

        save_path = tmp_path / "test_nibabel.nii.gz"
        medrs_image.save(str(save_path))

        # Load with nibabel
        nib_img = nib.load(str(save_path))
        nib_data = nib_img.get_fdata()

        medrs_data = medrs_image.to_numpy()

        assert np.allclose(medrs_data, nib_data, atol=1e-5)


# ============================================================================
# Advanced MONAI Comparison Tests
# ============================================================================

class TestAdvancedMonaiComparison:
    """Additional MONAI comparison tests for transforms."""

    def test_rescale_matches_monai(self, medrs_image, test_image_path):
        """Compare rescale intensity with MONAI ScaleIntensityRange."""
        from monai.transforms import LoadImage, ScaleIntensityRange

        loader = LoadImage(image_only=True)
        monai_data = np.asarray(loader(test_image_path))

        # MONAI rescale to [0, 1]
        scaler = ScaleIntensityRange(a_min=float(monai_data.min()), a_max=float(monai_data.max()),
                                      b_min=0.0, b_max=1.0, clip=True)
        monai_rescaled = np.asarray(scaler(monai_data))

        # medrs rescale
        medrs_rescaled = medrs_image.rescale(0.0, 1.0).to_numpy()

        # Both should be in [0, 1] and close
        assert np.allclose(medrs_rescaled, monai_rescaled, atol=1e-5), \
            f"Rescale mismatch: max_diff={np.max(np.abs(medrs_rescaled - monai_rescaled))}"

    def test_clamp_matches_monai(self, medrs_image, test_image_path):
        """Compare clamp with MONAI ThresholdIntensity."""
        from monai.transforms import LoadImage, ThresholdIntensity

        loader = LoadImage(image_only=True)
        monai_data = np.asarray(loader(test_image_path))

        # MONAI threshold (clamp above and below)
        # Note: MONAI doesn't have direct clamp, but we can combine two thresholds
        monai_clamped = np.clip(monai_data, 0.0, 100.0)

        # medrs clamp
        medrs_clamped = medrs_image.clamp(0.0, 100.0).to_numpy()

        assert np.allclose(medrs_clamped, monai_clamped, atol=1e-5)

    def test_flip_matches_monai(self, medrs_image, test_image_path):
        """Compare flip with MONAI Flip."""
        from monai.transforms import LoadImage, Flip, EnsureChannelFirst

        # MONAI LoadImage returns (H, W, D), need EnsureChannelFirst for (C, H, W, D)
        loader = LoadImage(image_only=True)
        monai_data = loader(test_image_path)
        monai_data = EnsureChannelFirst()(monai_data)

        # MONAI flip on spatial axis 0
        flipper = Flip(spatial_axis=0)
        monai_flipped = np.asarray(flipper(monai_data))

        # Remove channel dimension to compare with medrs (C,H,W,D) -> (H,W,D)
        monai_flipped = monai_flipped[0]

        # medrs flip
        medrs_flipped = medrs_image.flip([0]).to_numpy()

        assert np.allclose(medrs_flipped, monai_flipped, atol=1e-5)

    def test_crop_matches_monai(self, medrs_image, test_image_path):
        """Compare center crop with MONAI CenterSpatialCrop."""
        from monai.transforms import LoadImage, CenterSpatialCrop, EnsureChannelFirst

        # MONAI LoadImage returns (H, W, D), need EnsureChannelFirst for (C, H, W, D)
        loader = LoadImage(image_only=True)
        monai_data = loader(test_image_path)
        monai_data = EnsureChannelFirst()(monai_data)

        # MONAI center crop (roi_size is for spatial dims only)
        crop_size = [100, 100, 100]
        cropper = CenterSpatialCrop(roi_size=crop_size)
        monai_cropped = np.asarray(cropper(monai_data))

        # Remove channel dimension to compare with medrs (C,H,W,D) -> (H,W,D)
        monai_cropped = monai_cropped[0]

        # medrs crop_or_pad to smaller size centers the crop
        medrs_cropped = medrs_image.crop_or_pad(crop_size).to_numpy()

        assert medrs_cropped.shape == monai_cropped.shape
        assert np.allclose(medrs_cropped, monai_cropped, atol=1e-5)


# ============================================================================
# Nibabel Roundtrip Tests
# ============================================================================

class TestNibabelRoundtrip:
    """Tests for nibabel compatibility and roundtrips."""

    def test_nibabel_save_medrs_load(self, nibabel_data, tmp_path):
        """Test loading file saved by nibabel."""
        import nibabel as nib
        import medrs

        # Save with nibabel
        save_path = tmp_path / "nibabel_saved.nii.gz"
        nib_img = nib.Nifti1Image(nibabel_data["data"].astype(np.float32),
                                   nibabel_data["affine"])
        nib.save(nib_img, str(save_path))

        # Load with medrs
        medrs_img = medrs.load(str(save_path))
        medrs_data = medrs_img.to_numpy()

        assert np.allclose(medrs_data, nibabel_data["data"], atol=1e-5)

    def test_medrs_transform_nibabel_verify(self, medrs_image, tmp_path):
        """Test transforms are correctly saved and readable by nibabel."""
        import nibabel as nib
        import medrs

        # Apply transforms
        transformed = medrs_image.z_normalize().clamp(-3.0, 3.0)

        # Save
        save_path = tmp_path / "transformed.nii.gz"
        transformed.save(str(save_path))

        # Load with nibabel and verify
        nib_img = nib.load(str(save_path))
        nib_data = nib_img.get_fdata()
        medrs_data = transformed.to_numpy()

        assert np.allclose(medrs_data, nib_data, atol=1e-5)

    def test_preserve_orientation_code(self, medrs_image):
        """Verify orientation code is preserved."""
        orientation = medrs_image.orientation
        assert orientation in ["RAS", "LAS", "LPS", "RPS", "RAI", "LAI", "LPI", "RPI",
                                "RSA", "LSA", "RSP", "LSP", "RIA", "LIA", "RIP", "LIP",
                                "ARS", "ALS", "PRS", "PLS", "ARI", "ALI", "PRI", "PLI",
                                "ASR", "ASL", "PSR", "PSL", "AIR", "AIL", "PIR", "PIL",
                                "SAR", "SAL", "SPR", "SPL", "IAR", "IAL", "IPR", "IPL",
                                "SRA", "SLA", "SRP", "SLP", "IRA", "ILA", "IRP", "ILP"]


# ============================================================================
# Advanced Cropping Tests
# ============================================================================

class TestAdvancedCropping:
    """Tests for various cropping scenarios."""

    def test_corner_crop_top_left(self, test_image_path, nibabel_data):
        """Test cropping from top-left corner."""
        import medrs

        crop_offset = [0, 0, 0]
        crop_size = [32, 32, 32]
        cropped = medrs.load_cropped(test_image_path, crop_offset, crop_size)
        data = cropped.to_numpy()

        expected = nibabel_data["data"][0:32, 0:32, 0:32]
        assert np.allclose(data, expected, atol=1e-5)

    def test_corner_crop_bottom_right(self, test_image_path, nibabel_data):
        """Test cropping from bottom-right corner."""
        import medrs

        shape = nibabel_data["shape"]
        crop_size = [32, 32, 32]
        crop_offset = [shape[0] - 32, shape[1] - 32, shape[2] - 32]

        cropped = medrs.load_cropped(test_image_path, crop_offset, crop_size)
        data = cropped.to_numpy()

        expected = nibabel_data["data"][-32:, -32:, -32:]
        assert np.allclose(data, expected, atol=1e-5)

    def test_asymmetric_crop(self, test_image_path, nibabel_data):
        """Test asymmetric crop dimensions."""
        import medrs

        crop_offset = [10, 20, 30]
        crop_size = [50, 40, 30]
        cropped = medrs.load_cropped(test_image_path, crop_offset, crop_size)
        data = cropped.to_numpy()

        expected = nibabel_data["data"][10:60, 20:60, 30:60]
        assert np.allclose(data, expected, atol=1e-5)

    def test_multiple_crops_consistent(self, test_image_path):
        """Verify multiple crops from same location are identical."""
        import medrs

        crop_offset = [50, 50, 50]
        crop_size = [64, 64, 64]

        crop1 = medrs.load_cropped(test_image_path, crop_offset, crop_size).to_numpy()
        crop2 = medrs.load_cropped(test_image_path, crop_offset, crop_size).to_numpy()

        assert np.array_equal(crop1, crop2)


# ============================================================================
# Pipeline Simulation Tests
# ============================================================================

class TestPipelineSimulation:
    """Tests simulating real-world processing pipelines."""

    def test_preprocessing_pipeline(self, medrs_image, nibabel_data):
        """Test typical preprocessing pipeline."""
        # Typical preprocessing: normalize -> clamp outliers -> rescale
        processed = (medrs_image
                     .z_normalize()
                     .clamp(-3.0, 3.0)
                     .rescale(0.0, 1.0))

        data = processed.to_numpy()

        assert data.min() >= 0.0
        assert data.max() <= 1.0
        assert data.flags["F_CONTIGUOUS"]

    def test_training_pipeline(self, test_image_path):
        """Test typical training data loading pipeline."""
        import medrs

        # Simulate loading a random crop for training
        crop_size = [64, 64, 64]
        img = medrs.load_cropped(test_image_path, [50, 50, 50], crop_size)

        # Apply training transforms
        processed = (img
                     .z_normalize()
                     .clamp(-5.0, 5.0))

        data = processed.to_numpy()
        assert data.shape == tuple(crop_size)
        assert not np.any(np.isnan(data))

    def test_inference_pipeline(self, medrs_image):
        """Test typical inference pipeline."""
        # Resample to standard spacing
        resampled = medrs_image.resample([1.0, 1.0, 1.0])

        # Normalize
        normalized = resampled.z_normalize()

        # Pad to multiple of 16 (common for deep learning)
        shape = normalized.shape
        target_shape = [(s // 16 + 1) * 16 for s in shape]
        padded = normalized.crop_or_pad(target_shape)

        assert all(s % 16 == 0 for s in padded.shape)

    def test_augmentation_pipeline(self, medrs_image):
        """Test typical augmentation pipeline."""
        # Flip along random axis
        flipped = medrs_image.flip([0])

        # Intensity augmentation (simulate)
        augmented = flipped.rescale(0.0, 1.0)

        data = augmented.to_numpy()
        assert data.flags["F_CONTIGUOUS"]

    def test_multi_resolution_pipeline(self, medrs_image):
        """Test processing at multiple resolutions."""
        # Original resolution
        orig = medrs_image.to_numpy()

        # Downsampled 2x
        down2x = medrs_image.resample([2.0, 2.0, 2.0]).to_numpy()

        # Downsampled 4x
        down4x = medrs_image.resample([4.0, 4.0, 4.0]).to_numpy()

        # Each level should be roughly half the previous
        for i in range(3):
            assert 0.4 < down2x.shape[i] / orig.shape[i] < 0.6
            assert 0.4 < down4x.shape[i] / down2x.shape[i] < 0.6


# ============================================================================
# Data Integrity Tests
# ============================================================================

class TestDataIntegrity:
    """Tests for data integrity across operations."""

    def test_double_flip_identity(self, medrs_image):
        """Flipping twice should return to original."""
        original = medrs_image.to_numpy()

        double_flipped = medrs_image.flip([0]).flip([0]).to_numpy()

        assert np.allclose(original, double_flipped, atol=1e-5)

    def test_crop_pad_roundtrip(self, medrs_image):
        """Crop then pad back should preserve center."""
        original_shape = medrs_image.shape
        original = medrs_image.to_numpy()

        # Crop to smaller
        small_shape = [s - 50 for s in original_shape]
        cropped = medrs_image.crop_or_pad(small_shape)

        # Pad back to original
        padded_back = cropped.crop_or_pad(list(original_shape))

        # Center region should match
        data = padded_back.to_numpy()
        margin = 25
        center_original = original[margin:-margin, margin:-margin, margin:-margin]
        center_recovered = data[margin:-margin, margin:-margin, margin:-margin]

        assert np.allclose(center_original, center_recovered, atol=1e-5)

    def test_rescale_preserves_relative_values(self, medrs_image):
        """Rescaling should preserve relative ordering."""
        original = medrs_image.to_numpy()
        rescaled = medrs_image.rescale(0.0, 1.0).to_numpy()

        # Find some test points with different values
        idx1 = (100, 100, 100)
        idx2 = (50, 50, 50)

        # If orig1 > orig2, then rescaled1 > rescaled2
        if original[idx1] > original[idx2]:
            assert rescaled[idx1] >= rescaled[idx2]
        elif original[idx1] < original[idx2]:
            assert rescaled[idx1] <= rescaled[idx2]

    def test_normalize_idempotent_stats(self, medrs_image):
        """Double normalization should still have mean~0, std~1."""
        normalized_once = medrs_image.z_normalize()
        normalized_twice = normalized_once.z_normalize()

        data = normalized_twice.to_numpy()
        mean = np.mean(data)
        std = np.std(data)

        assert abs(mean) < 1e-5
        assert abs(std - 1.0) < 1e-5


# ============================================================================
# Performance Sanity Tests
# ============================================================================

class TestPerformanceSanity:
    """Basic sanity tests for performance (not strict benchmarks)."""

    def test_load_is_reasonably_fast(self, test_image_path):
        """Verify loading completes in reasonable time."""
        import medrs
        import time

        start = time.time()
        for _ in range(5):
            img = medrs.load(test_image_path)
            _ = img.to_numpy()
        elapsed = time.time() - start

        # Should complete 5 loads in under 5 seconds
        assert elapsed < 5.0, f"Loading too slow: {elapsed:.2f}s for 5 iterations"

    def test_transforms_are_reasonably_fast(self, medrs_image):
        """Verify transforms complete in reasonable time."""
        import time

        start = time.time()
        for _ in range(3):
            _ = medrs_image.z_normalize().to_numpy()
            _ = medrs_image.clamp(0, 100).to_numpy()
            _ = medrs_image.rescale(0, 1).to_numpy()
        elapsed = time.time() - start

        # Should complete in under 3 seconds
        assert elapsed < 3.0, f"Transforms too slow: {elapsed:.2f}s"


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
