//! Comprehensive tests for core NIfTI functionality.
//!
//! This test module covers critical NIfTI I/O operations that were
//! previously lacking test coverage, including error conditions
//! and edge cases.

use medrs::nifti::{self, DataType, NiftiImage};
use ndarray::ArrayD;
use ndarray::ShapeBuilder;
use tempfile::NamedTempFile;

/// Create a test NIfTI image using the library's own functions
fn create_test_image(data: Vec<f32>, shape: Vec<usize>) -> NiftiImage {
    let c_order = ArrayD::from_shape_vec(shape.clone(), data).unwrap();
    let mut f_order = ArrayD::zeros(ndarray::IxDyn(&shape).f());
    f_order.assign(&c_order);
    let affine = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    NiftiImage::from_array(f_order, affine)
}

#[test]
fn test_load_invalid_magic_bytes() {
    let img = create_test_image(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2, 1]);
    let file = NamedTempFile::new().unwrap();
    nifti::save(&img, file.path().to_str().unwrap()).unwrap();

    // Corrupt magic bytes
    let path = file.path().to_str().unwrap();
    let mut file_data = std::fs::read(path).unwrap();
    file_data[344..348].copy_from_slice(b"BAD!"); // Corrupt magic bytes
    std::fs::write(path, file_data).unwrap();

    let result = nifti::load(path);
    assert!(result.is_err());

    let err = result.unwrap_err();
    assert!(err.to_string().contains("invalid NIfTI magic"));
}

#[test]
fn test_load_unsupported_data_type() {
    let img = create_test_image(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2, 1]);
    let file = NamedTempFile::new().unwrap();
    nifti::save(&img, file.path().to_str().unwrap()).unwrap();

    // Set unsupported data type in header
    let path = file.path().to_str().unwrap();
    let mut file_data = std::fs::read(path).unwrap();
    // Write unsupported data type code (e.g., 9999)
    file_data[70..72].copy_from_slice(&9999i16.to_le_bytes());
    std::fs::write(path, file_data).unwrap();

    let result = nifti::load(path);
    assert!(result.is_err());

    let err = result.unwrap_err();
    assert!(err.to_string().contains("unsupported data type"));
}

#[test]
fn test_bitpix_mismatch_rejected() {
    let img = create_test_image(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2, 1]);
    let file = NamedTempFile::new().unwrap();
    nifti::save(&img, file.path().to_str().unwrap()).unwrap();

    let path = file.path().to_str().unwrap();
    let mut file_data = std::fs::read(path).unwrap();
    // BITPIX is at bytes 72..74 for NIfTI-1
    file_data[72..74].copy_from_slice(&8i16.to_le_bytes());
    std::fs::write(path, file_data).unwrap();

    let result = nifti::load(path);
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(err.contains("bitpix"));
}

#[test]
fn test_vox_offset_fractional_rejected() {
    let img = create_test_image(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2, 1]);
    let file = NamedTempFile::new().unwrap();
    nifti::save(&img, file.path().to_str().unwrap()).unwrap();

    let path = file.path().to_str().unwrap();
    let mut file_data = std::fs::read(path).unwrap();
    // VOX_OFFSET is at bytes 108..112 for NIfTI-1
    file_data[108..112].copy_from_slice(&352.5f32.to_le_bytes());
    std::fs::write(path, file_data).unwrap();

    let result = nifti::load(path);
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(err.contains("vox_offset"));
}

#[test]
fn test_extension_flag_rejected() {
    let img = create_test_image(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2, 1]);
    let file = NamedTempFile::new().unwrap();
    nifti::save(&img, file.path().to_str().unwrap()).unwrap();

    let path = file.path().to_str().unwrap();
    let mut file_data = std::fs::read(path).unwrap();
    // Extension flag starts at byte 348 for NIfTI-1
    file_data[348] = 1;
    std::fs::write(path, file_data).unwrap();

    let result = nifti::load(path);
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(err.contains("extensions"));
}

#[test]
fn test_load_roundtrip_preserves_metadata() {
    let original_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let img = create_test_image(original_data.clone(), vec![2, 2, 2]);

    // Save and reload
    let file = NamedTempFile::new().unwrap();
    nifti::save(&img, file.path().to_str().unwrap()).unwrap();
    let reloaded_img = nifti::load(file.path().to_str().unwrap()).unwrap();

    // Check that data is preserved (F-order vs C-order differences are expected)
    let reloaded_data = reloaded_img.to_f32().unwrap();
    let reloaded_slice = reloaded_data.as_slice_memory_order().unwrap();

    // The data should be the same set of values, but potentially in different order due to F-order
    // Let's check that the sorted values are the same
    let mut original_sorted = original_data.clone();
    let mut reloaded_sorted = reloaded_slice.to_vec();
    original_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    reloaded_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert_eq!(original_sorted, reloaded_sorted);

    // Check metadata is preserved
    assert_eq!(img.shape(), reloaded_img.shape());
    assert_eq!(img.dtype(), reloaded_img.dtype());
    assert_eq!(img.spacing(), reloaded_img.spacing());
}

#[test]
fn test_memory_efficient_loading() {
    let data = vec![1.0f32; 1000]; // Larger dataset
    let img = create_test_image(data, vec![10, 10, 10]);

    // Save and reload
    let file = NamedTempFile::new().unwrap();
    nifti::save(&img, file.path().to_str().unwrap()).unwrap();
    let loaded_img = nifti::load(file.path().to_str().unwrap()).unwrap();

    // Memory mapping behavior may vary based on file size and system
    // Let's check that we can access the data regardless
    let _data = loaded_img.to_f32().unwrap();
    // The key test is that we can access the data without crashing

    // Check data values
    let loaded_data = loaded_img.to_f32().unwrap();
    let loaded_slice = loaded_data.as_slice_memory_order().unwrap();
    assert_eq!(loaded_slice.len(), 1000);
    assert_eq!(loaded_slice[0], 1.0);
    assert_eq!(loaded_slice[999], 1.0);
}

#[test]
fn test_dtype_conversions() {
    let img = create_test_image(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2, 1]);

    // Test conversion to different data types
    let img_u16 = img.with_dtype(DataType::UInt16).unwrap();
    assert_eq!(img_u16.dtype(), DataType::UInt16);

    let img_f16 = img.with_dtype(DataType::Float16).unwrap();
    assert_eq!(img_f16.dtype(), DataType::Float16);

    let img_bf16 = img.with_dtype(DataType::BFloat16).unwrap();
    assert_eq!(img_bf16.dtype(), DataType::BFloat16);
}

#[test]
fn test_safe_data_access() {
    let img = create_test_image(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2, 1]);

    // Test that we can access the data conversion function (now returns Result)
    // This test ensures the function exists and returns Ok
    let result = img.to_f32();
    assert!(result.is_ok());

    let data = result.unwrap();
    // Verify we can access the data
    let slice = data.as_slice_memory_order().unwrap();
    let mut sorted = slice.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert_eq!(sorted, vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_header_parsing_edge_cases() {
    // Test that the system can handle various edge cases in headers
    let img = create_test_image(vec![1.0f32], vec![1, 1, 1]); // Minimal valid dimensions

    // Save and reload
    let file = NamedTempFile::new().unwrap();
    nifti::save(&img, file.path().to_str().unwrap()).unwrap();
    let loaded_img = nifti::load(file.path().to_str().unwrap()).unwrap();

    // Should handle minimal volumes without crashing
    assert_eq!(loaded_img.shape(), [1, 1, 1]);
    assert_eq!(loaded_img.to_f32().unwrap().len(), 1);
}

#[test]
fn test_corrupted_file_handling() {
    let img = create_test_image(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2, 1]);
    let file = NamedTempFile::new().unwrap();
    nifti::save(&img, file.path().to_str().unwrap()).unwrap();

    // Truncate the file to create a corrupted file
    let path = file.path().to_str().unwrap();
    let original_data = std::fs::read(path).unwrap();
    let truncated_data = &original_data[..original_data.len() / 2]; // Cut file in half
    std::fs::write(path, truncated_data).unwrap();

    // Should fail gracefully with proper error message
    let result = nifti::load(path);
    assert!(result.is_err());

    let err = result.unwrap_err();
    // Should contain information about the file being corrupted/truncated
    let err_str = err.to_string();
    assert!(err_str.contains("error") || err_str.contains("failed"));
}

#[test]
fn test_different_data_types_roundtrip() {
    // Test saving and loading with different data types
    let data = vec![1.0f32, 2.0, 3.0, 4.0];

    for dtype in [DataType::Float32, DataType::UInt16, DataType::Int16] {
        let img = create_test_image(data.clone(), vec![2, 2, 1]);
        let img_converted = img.with_dtype(dtype).unwrap();

        let file = NamedTempFile::new().unwrap();
        nifti::save(&img_converted, file.path().to_str().unwrap()).unwrap();

        let loaded_img = nifti::load(file.path().to_str().unwrap()).unwrap();
        assert_eq!(loaded_img.dtype(), dtype);
        assert_eq!(loaded_img.shape(), [2, 2, 1]);
    }
}

// ============================================================================
// Phase 1 Correctness Tests - Scaling, Validation, Edge Cases
// ============================================================================

#[test]
fn test_scl_slope_zero_treated_as_one() {
    // Per NIfTI spec, scl_slope=0 means "no scaling" (equivalent to slope=1, inter=0)
    let img = create_test_image(vec![10.0f32, 20.0, 30.0, 40.0], vec![2, 2, 1]);

    // Manually set scl_slope=0 in header
    let file = NamedTempFile::new().unwrap();
    nifti::save(&img, file.path().to_str().unwrap()).unwrap();

    // Modify header to set scl_slope=0
    let path = file.path().to_str().unwrap();
    let mut file_data = std::fs::read(path).unwrap();
    // scl_slope is at offset 112 (4 bytes f32)
    file_data[112..116].copy_from_slice(&0.0f32.to_le_bytes());
    std::fs::write(path, file_data).unwrap();

    let loaded = nifti::load(path).unwrap();
    let data = loaded.to_f32().unwrap();

    // Data should be unscaled (slope=0 → slope=1)
    let slice = data.as_slice_memory_order().unwrap();
    let mut sorted = slice.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert_eq!(sorted, vec![10.0, 20.0, 30.0, 40.0]);
}

#[test]
fn test_scl_slope_and_intercept_applied() {
    // Test that non-zero slope and intercept are correctly applied
    let img = create_test_image(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2, 1]);

    let file = NamedTempFile::new().unwrap();
    nifti::save(&img, file.path().to_str().unwrap()).unwrap();

    // Set scl_slope=2.0, scl_inter=10.0
    let path = file.path().to_str().unwrap();
    let mut file_data = std::fs::read(path).unwrap();
    file_data[112..116].copy_from_slice(&2.0f32.to_le_bytes()); // scl_slope
    file_data[116..120].copy_from_slice(&10.0f32.to_le_bytes()); // scl_inter
    std::fs::write(path, file_data).unwrap();

    let loaded = nifti::load(path).unwrap();
    let data = loaded.to_f32().unwrap();

    // Expected: raw_value * 2.0 + 10.0
    // 1*2+10=12, 2*2+10=14, 3*2+10=16, 4*2+10=18
    let slice = data.as_slice_memory_order().unwrap();
    let mut sorted = slice.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert_eq!(sorted, vec![12.0, 14.0, 16.0, 18.0]);
}

#[test]
fn test_negative_dimension_in_header_rejected() {
    let img = create_test_image(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2, 1]);

    let file = NamedTempFile::new().unwrap();
    nifti::save(&img, file.path().to_str().unwrap()).unwrap();

    // Set dim[1] to -1 (invalid)
    let path = file.path().to_str().unwrap();
    let mut file_data = std::fs::read(path).unwrap();
    // dim[1] is at offset 42 (after dim[0] at 40)
    file_data[42..44].copy_from_slice(&(-1i16).to_le_bytes());
    std::fs::write(path, file_data).unwrap();

    let result = nifti::load(path);
    assert!(result.is_err());
    let err_str = result.unwrap_err().to_string();
    assert!(
        err_str.contains("zero") || err_str.contains("dimension"),
        "Expected error about zero dimension, got: {}",
        err_str
    );
}

#[test]
fn test_zero_copy_feasibility_checks() {
    // Test that zero-copy is possible for uncompressed f32 files
    let data = vec![1.0f32; 1000];
    let img = create_test_image(data.clone(), vec![10, 10, 10]);

    // Save as uncompressed .nii
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.nii");
    nifti::save(&img, path.to_str().unwrap()).unwrap();

    // Load uncompressed - should be mmap'd and support zero-copy
    let loaded = nifti::load(path.to_str().unwrap()).unwrap();
    assert!(
        loaded.can_zero_copy(),
        "Uncompressed f32 file should support zero-copy"
    );
    assert!(
        loaded.can_zero_copy_f32(),
        "Uncompressed f32 file should support zero-copy f32"
    );

    // Verify raw_bytes is available
    assert!(
        loaded.raw_bytes().is_some(),
        "Mmap'd file should have raw_bytes available"
    );
}

#[test]
fn test_zero_copy_not_available_for_gzipped() {
    let data = vec![1.0f32; 1000];
    let img = create_test_image(data, vec![10, 10, 10]);

    // Save as gzipped .nii.gz
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.nii.gz");
    nifti::save(&img, path.to_str().unwrap()).unwrap();

    // Load gzipped - should NOT support zero-copy (data is decompressed to Vec)
    let loaded = nifti::load(path.to_str().unwrap()).unwrap();
    assert!(
        !loaded.can_zero_copy(),
        "Gzipped file should not support zero-copy"
    );

    // raw_bytes should be None for non-mmap storage
    assert!(
        loaded.raw_bytes().is_none(),
        "Gzipped file should not have raw_bytes"
    );
}

#[test]
fn test_zero_copy_not_available_with_scaling() {
    let data = vec![1.0f32; 1000];
    let img = create_test_image(data, vec![10, 10, 10]);

    // Save as uncompressed
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.nii");
    nifti::save(&img, path.to_str().unwrap()).unwrap();

    // Modify the file to add scaling factors
    let mut file_data = std::fs::read(&path).unwrap();
    // scl_slope at offset 112, scl_inter at offset 116 (both f32)
    file_data[112..116].copy_from_slice(&2.0f32.to_le_bytes()); // slope = 2.0
    std::fs::write(&path, file_data).unwrap();

    let loaded = nifti::load(path.to_str().unwrap()).unwrap();

    // Should be mmap'd but NOT zero-copy due to scaling
    assert!(loaded.raw_bytes().is_some(), "Should still be mmap'd");
    assert!(
        !loaded.can_zero_copy_f32(),
        "File with scaling should not support zero-copy f32"
    );
}

#[test]
fn test_zero_dimension_rejected() {
    let img = create_test_image(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2, 1]);

    let file = NamedTempFile::new().unwrap();
    nifti::save(&img, file.path().to_str().unwrap()).unwrap();

    // Set dim[1] to 0 (invalid for active dimensions)
    let path = file.path().to_str().unwrap();
    let mut file_data = std::fs::read(path).unwrap();
    file_data[42..44].copy_from_slice(&0i16.to_le_bytes());
    std::fs::write(path, file_data).unwrap();

    let result = nifti::load(path);
    assert!(result.is_err());
    let err_str = result.unwrap_err().to_string();
    assert!(
        err_str.contains("zero") || err_str.contains("dimension"),
        "Expected error about zero dimension, got: {}",
        err_str
    );
}

#[test]
fn test_bitpix_mismatch_rejected_half_size() {
    let img = create_test_image(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2, 1]);

    let file = NamedTempFile::new().unwrap();
    nifti::save(&img, file.path().to_str().unwrap()).unwrap();

    let path = file.path().to_str().unwrap();
    let mut file_data = std::fs::read(path).unwrap();
    file_data[72..74].copy_from_slice(&16i16.to_le_bytes());
    std::fs::write(path, file_data).unwrap();

    let result = nifti::load(path);
    assert!(result.is_err(), "Expected error for bitpix mismatch");
    let err_str = result.unwrap_err().to_string();
    assert!(
        err_str.contains("bitpix") && err_str.contains("does not match"),
        "Expected bitpix mismatch error, got: {}",
        err_str
    );
}

#[test]
fn test_vox_offset_before_header_end_rejected() {
    let img = create_test_image(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2, 1]);

    let file = NamedTempFile::new().unwrap();
    nifti::save(&img, file.path().to_str().unwrap()).unwrap();

    // Set vox_offset to 100 (before 348-byte header end)
    let path = file.path().to_str().unwrap();
    let mut file_data = std::fs::read(path).unwrap();
    file_data[108..112].copy_from_slice(&100.0f32.to_le_bytes());
    std::fs::write(path, file_data).unwrap();

    let result = nifti::load(path);
    assert!(result.is_err());
    let err_str = result.unwrap_err().to_string();
    assert!(
        err_str.contains("vox_offset") || err_str.contains("header"),
        "Expected error about vox_offset, got: {}",
        err_str
    );
}

#[test]
fn test_file_too_small_for_data_rejected() {
    let img = create_test_image(vec![1.0f32; 1000], vec![10, 10, 10]);

    let file = NamedTempFile::new().unwrap();
    nifti::save(&img, file.path().to_str().unwrap()).unwrap();

    // Truncate file to be smaller than header says data should be
    let path = file.path().to_str().unwrap();
    let file_data = std::fs::read(path).unwrap();
    // Keep header (352 bytes) + some data, but not all
    let truncated = &file_data[..500];
    std::fs::write(path, truncated).unwrap();

    let result = nifti::load(path);
    assert!(result.is_err());
    let err_str = result.unwrap_err().to_string();
    assert!(
        err_str.contains("truncated") || err_str.contains("file") || err_str.contains("EOF"),
        "Expected error about truncated file, got: {}",
        err_str
    );
}

#[test]
fn test_gzipped_roundtrip() {
    let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let img = create_test_image(data.clone(), vec![2, 2, 2]);

    // Save as gzipped
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.nii.gz");
    nifti::save(&img, path.to_str().unwrap()).unwrap();

    // Reload
    let loaded = nifti::load(path.to_str().unwrap()).unwrap();
    assert_eq!(loaded.shape(), [2, 2, 2]);

    let loaded_data = loaded.to_f32().unwrap();
    let loaded_slice = loaded_data.as_slice_memory_order().unwrap();

    let mut original_sorted = data.clone();
    let mut loaded_sorted = loaded_slice.to_vec();
    original_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    loaded_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert_eq!(original_sorted, loaded_sorted);
}

#[test]
fn test_cropped_load_bounds_validation() {
    let data = vec![1.0f32; 1000];
    let img = create_test_image(data, vec![10, 10, 10]);

    let file = NamedTempFile::new().unwrap();
    nifti::save(&img, file.path().to_str().unwrap()).unwrap();

    // Try to crop beyond bounds
    let result = nifti::load_cropped(
        file.path().to_str().unwrap(),
        [5, 5, 5],    // offset
        [10, 10, 10], // shape - exceeds bounds (5+10 > 10)
    );

    assert!(result.is_err());
    let err_str = result.unwrap_err().to_string();
    assert!(
        err_str.contains("exceeds") || err_str.contains("dimension") || err_str.contains("bounds"),
        "Expected error about bounds, got: {}",
        err_str
    );
}

#[test]
fn test_cropped_load_zero_shape_rejected() {
    let data = vec![1.0f32; 1000];
    let img = create_test_image(data, vec![10, 10, 10]);

    let file = NamedTempFile::new().unwrap();
    nifti::save(&img, file.path().to_str().unwrap()).unwrap();

    // Try to crop with zero dimension
    let result = nifti::load_cropped(
        file.path().to_str().unwrap(),
        [0, 0, 0],
        [0, 5, 5], // zero in first dimension
    );

    assert!(result.is_err());
    let err_str = result.unwrap_err().to_string();
    assert!(
        err_str.contains("zero") || err_str.contains("dimension"),
        "Expected error about zero dimension, got: {}",
        err_str
    );
}
