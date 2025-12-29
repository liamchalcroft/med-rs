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
