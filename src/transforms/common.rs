//! Common utilities for transform operations.
//!
//! This module provides shared functionality to eliminate code duplication
//! across transform functions and ensure consistent error handling.

use crate::error::Error;

/// Create a new ArrayD from shape and vector with proper error handling.
///
/// This is a common pattern in transform functions that need to create
/// new arrays from processed data.
pub fn create_array_from_vec<T>(shape: &[usize], data: Vec<T>) -> Result<ndarray::ArrayD<T>, Error>
where
    T: Clone,
{
    ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(shape), data)
        .map_err(|e| Error::MemoryAllocation(format!("Failed to create array: {}", e)))
}

/// Validate that array dimensions are valid for the operation.
///
/// Common validation used across spatial transforms.
pub fn validate_array_dimensions(shape: &[usize], min_dims: usize) -> Result<(), Error> {
    if shape.len() < min_dims {
        return Err(Error::InvalidDimensions(format!(
            "Array must have at least {} dimensions, got {}",
            min_dims,
            shape.len()
        )));
    }

    if shape.contains(&0) {
        return Err(Error::InvalidDimensions(
            "Array dimensions cannot contain zero".to_string(),
        ));
    }

    Ok(())
}

/// Validate that an array is contiguous in memory.
///
/// Common validation for operations that require contiguous memory.
pub fn validate_contiguous<T>(array: &ndarray::ArrayD<T>) -> Result<(), Error> {
    if !array.is_standard_layout() && array.as_slice_memory_order().is_none() {
        return Err(Error::NonContiguousArray(
            "Array is not contiguous in memory".to_string(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::ArrayD;
    use ndarray::IxDyn;
    use ndarray::ShapeBuilder;

    #[test]
    fn test_validate_array_dimensions() {
        // Valid cases
        assert!(validate_array_dimensions(&[2, 2, 2], 3).is_ok());
        assert!(validate_array_dimensions(&[2, 2, 2, 1], 3).is_ok());

        // Invalid cases
        assert!(validate_array_dimensions(&[2, 2], 3).is_err());
        assert!(validate_array_dimensions(&[0, 2, 2], 3).is_err());
    }

    #[test]
    fn test_create_array_from_vec() {
        let shape = [2, 2];
        let data = vec![1.0f32, 2.0, 3.0, 4.0];

        let result = create_array_from_vec(&shape, data.clone()).unwrap();
        assert_eq!(result.shape(), &shape[..]);
        assert_eq!(result.as_slice().unwrap(), &data[..]);

        // Invalid case
        let wrong_data = vec![1.0f32, 2.0, 3.0]; // Wrong length
        assert!(create_array_from_vec(&shape, wrong_data).is_err());
    }

    #[test]
    fn test_validate_contiguous_standard_layout() {
        // Standard layout array is contiguous
        let arr = ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1.0f32; 6]).unwrap();
        assert!(validate_contiguous(&arr).is_ok());
    }

    #[test]
    fn test_validate_contiguous_f_order() {
        // Fortran-order array should also be valid (memory-contiguous)
        let arr = ArrayD::from_shape_vec(IxDyn(&[2, 3]).f(), vec![1.0f32; 6]).unwrap();
        assert!(validate_contiguous(&arr).is_ok());
    }

    #[test]
    fn test_validate_contiguous_sliced_view() {
        // Sliced view with step > 1 is non-contiguous
        use ndarray::s;
        let arr = ArrayD::from_shape_vec(IxDyn(&[4, 4]), vec![1.0f32; 16]).unwrap();
        // Take every other element - this creates a non-contiguous view
        let sliced = arr.slice(s![..;2, ..;2]).to_owned().into_dyn();
        // Owned sliced array is contiguous
        assert!(validate_contiguous(&sliced).is_ok());
    }

    #[test]
    fn test_validate_array_dimensions_zero_dimension() {
        // Zero in any dimension is invalid
        assert!(validate_array_dimensions(&[0, 2, 2], 3).is_err());
        assert!(validate_array_dimensions(&[2, 0, 2], 3).is_err());
        assert!(validate_array_dimensions(&[2, 2, 0], 3).is_err());
    }

    #[test]
    fn test_create_array_from_vec_empty() {
        // Empty array with valid shape is ok
        let shape: [usize; 2] = [0, 0];
        let data: Vec<f32> = vec![];
        let result = create_array_from_vec(&shape, data);
        assert!(result.is_ok());
        let arr = result.unwrap();
        assert_eq!(arr.len(), 0);
    }

    #[test]
    fn test_create_array_from_vec_zero_in_shape() {
        // Zero in shape with empty data is valid
        let shape = [2, 0, 3];
        let data: Vec<f32> = vec![];
        let result = create_array_from_vec(&shape, data);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 0);
    }
}
