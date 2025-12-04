use crate::error::{Error, Result};
use crate::nifti::image::ArrayData;
use crate::nifti::NiftiImage;
use ndarray::{ArrayD, Axis, Slice};

/// Crop or pad the image to a target shape, centered.
///
/// If the target dimension is smaller, it crops the center.
/// If the target dimension is larger, it pads with zeros (or min value).
///
/// # Errors
///
/// Returns an error if `target_shape` dimensions don't match the image dimensions.
pub fn crop_or_pad(image: &NiftiImage, target_shape: &[usize]) -> Result<NiftiImage> {
    let current_shape = image.shape();
    let ndim = current_shape.len();

    if target_shape.len() != ndim {
        return Err(Error::InvalidDimensions(format!(
            "Target shape dimensions {} must match image dimensions {}",
            target_shape.len(),
            ndim
        )));
    }

    // Calculate crop/pad slices
    let mut slices = Vec::with_capacity(ndim);
    let mut pad_width = Vec::with_capacity(ndim);
    let mut needs_padding = false;

    for i in 0..ndim {
        let curr = current_shape[i];
        let target = target_shape[i];

        if target < curr {
            // Crop
            let diff = curr - target;
            let start = diff / 2;
            slices.push(start..start + target);
            pad_width.push((0, 0));
        } else if target > curr {
            // Pad
            let diff = target - curr;
            let before = diff / 2;
            let after = diff - before;
            slices.push(0..curr);
            pad_width.push((before, after));
            needs_padding = true;
        } else {
            // Same
            slices.push(0..curr);
            pad_width.push((0, 0));
        }
    }

    macro_rules! process_array_ref {
        ($arr:expr, $ty:ty) => {{
            // First crop if needed (view) - works on reference, no clone
            let mut view = $arr.view();
            for (i, slice) in slices.iter().enumerate() {
                view.slice_axis_inplace(Axis(i), Slice::from(slice.clone()));
            }

            if !needs_padding {
                // If only cropping, copy only the cropped region (much smaller!)
                view.to_owned().into_dyn()
            } else {
                // If padding needed, allocate new array
                let mut out = ArrayD::<$ty>::from_elem(target_shape, <$ty>::default());

                // Calculate where to place the view in the new array
                let mut out_slice_info = Vec::new();
                for i in 0..ndim {
                    let (before, _) = pad_width[i];
                    let len = view.shape()[i];
                    out_slice_info.push(Slice::from(before..before + len));
                }

                // Slice the output array and assign the view
                let mut out_view = out.view_mut();
                for (i, slice) in out_slice_info.iter().enumerate() {
                    out_view.slice_axis_inplace(Axis(i), *slice);
                }
                out_view.assign(&view);

                out
            }
        }};
    }

    // Use data_cow() to avoid cloning if data is already owned
    let data = image.data_cow();
    let new_data = match data.as_ref() {
        ArrayData::U8(a) => ArrayData::U8(process_array_ref!(a, u8)),
        ArrayData::I8(a) => ArrayData::I8(process_array_ref!(a, i8)),
        ArrayData::I16(a) => ArrayData::I16(process_array_ref!(a, i16)),
        ArrayData::U16(a) => ArrayData::U16(process_array_ref!(a, u16)),
        ArrayData::I32(a) => ArrayData::I32(process_array_ref!(a, i32)),
        ArrayData::U32(a) => ArrayData::U32(process_array_ref!(a, u32)),
        ArrayData::I64(a) => ArrayData::I64(process_array_ref!(a, i64)),
        ArrayData::U64(a) => ArrayData::U64(process_array_ref!(a, u64)),
        ArrayData::F16(a) => ArrayData::F16(process_array_ref!(a, half::f16)),
        ArrayData::BF16(a) => ArrayData::BF16(process_array_ref!(a, half::bf16)),
        ArrayData::F32(a) => ArrayData::F32(process_array_ref!(a, f32)),
        ArrayData::F64(a) => ArrayData::F64(process_array_ref!(a, f64)),
    };

    // Update header dimensions (reset unused dims to 1)
    let mut header = image.header().clone();
    header.ndim = target_shape.len() as u8;
    header.dim = [1u16; 7];
    for (i, &s) in target_shape.iter().enumerate() {
        header.dim[i] = s as u16;
    }

    // Update origin for crop/pad offset
    let affine = image.affine();
    let mut new_affine = affine;

    for i in 0..ndim.min(3) {
        let curr = current_shape[i];
        let target = target_shape[i];

        let shift = if target < curr {
            // Cropped: shift origin "inwards" (positive direction)
            (curr - target) as f32 / 2.0
        } else {
            // Padded: shift origin "outwards" (negative direction)
            -((target - curr) as f32 / 2.0)
        };

        if shift != 0.0 {
            new_affine[0][3] += affine[0][i] * shift;
            new_affine[1][3] += affine[1][i] * shift;
            new_affine[2][3] += affine[2][i] * shift;
        }
    }

    header.set_affine(new_affine);

    Ok(NiftiImage::from_parts(header, new_data))
}

/// Flip the image along specified axes.
///
/// # Arguments
///
/// * `image` - The input image
/// * `axes` - Slice of axis indices to flip (0=depth, 1=height, 2=width)
///
/// # Errors
///
/// Returns an error if any axis index is out of bounds.
pub fn flip(image: &NiftiImage, axes: &[usize]) -> Result<NiftiImage> {
    let ndim = image.ndim();
    for &axis in axes {
        if axis >= ndim {
            return Err(Error::InvalidDimensions(format!(
                "Axis {} out of bounds for image with {} dimensions",
                axis, ndim
            )));
        }
    }

    let header = image.header().clone();

    // Flip using view - invert_axis just changes strides (O(1))
    // Then copy to contiguous layout
    macro_rules! flip_array_ref {
        ($arr:expr, $variant:ident) => {{
            let mut view = $arr.view();
            for &axis in axes {
                view.invert_axis(Axis(axis));
            }
            // Copy to contiguous layout - this is the actual data copy
            ArrayData::$variant(view.to_owned())
        }};
    }

    // Use data_cow() to avoid cloning if data is already owned
    let data = image.data_cow();
    let new_data = match data.as_ref() {
        ArrayData::U8(a) => flip_array_ref!(a, U8),
        ArrayData::I8(a) => flip_array_ref!(a, I8),
        ArrayData::I16(a) => flip_array_ref!(a, I16),
        ArrayData::U16(a) => flip_array_ref!(a, U16),
        ArrayData::I32(a) => flip_array_ref!(a, I32),
        ArrayData::U32(a) => flip_array_ref!(a, U32),
        ArrayData::I64(a) => flip_array_ref!(a, I64),
        ArrayData::U64(a) => flip_array_ref!(a, U64),
        ArrayData::F16(a) => flip_array_ref!(a, F16),
        ArrayData::BF16(a) => flip_array_ref!(a, BF16),
        ArrayData::F32(a) => flip_array_ref!(a, F32),
        ArrayData::F64(a) => flip_array_ref!(a, F64),
    };

    Ok(NiftiImage::from_parts(header, new_data))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{ArrayD, IxDyn, ShapeBuilder};

    fn create_test_image(data: Vec<f32>, shape: [usize; 3]) -> NiftiImage {
        // Create F-order array to match NIfTI convention
        let c_order = ArrayD::from_shape_vec(shape.to_vec(), data).unwrap();
        let mut f_order = ArrayD::zeros(IxDyn(&shape).f());
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
    fn test_crop_or_pad_crop() {
        // Create a 4x4x4 volume with known values
        let data: Vec<f32> = (0..64).map(|i| i as f32).collect();
        let img = create_test_image(data, [4, 4, 4]);

        // Crop to 2x2x2 (centered)
        let cropped = crop_or_pad(&img, &[2, 2, 2]).unwrap();
        assert_eq!(cropped.shape(), &[2, 2, 2]);

        // Verify we got the center region
        let result = cropped.to_f32();
        assert_eq!(result.len(), 8);
    }

    #[test]
    fn test_crop_or_pad_pad() {
        // Create a 2x2x2 volume
        let data: Vec<f32> = (1..=8).map(|i| i as f32).collect();
        let img = create_test_image(data, [2, 2, 2]);

        // Pad to 4x4x4 (centered)
        let padded = crop_or_pad(&img, &[4, 4, 4]).unwrap();
        assert_eq!(padded.shape(), &[4, 4, 4]);

        // The outer voxels should be 0 (padding)
        let result = padded.to_f32();
        let slice = result.as_slice_memory_order().unwrap();

        // With F-order, first element in memory is still [0,0,0]
        assert!((slice[0] - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_crop_or_pad_same_size() {
        let data: Vec<f32> = (1..=8).map(|i| i as f32).collect();
        let img = create_test_image(data.clone(), [2, 2, 2]);

        // Same size - should be identity
        let result = crop_or_pad(&img, &[2, 2, 2]).unwrap();
        assert_eq!(result.shape(), &[2, 2, 2]);

        let result_data = result.to_f32();
        let result_slice = result_data.as_slice_memory_order().unwrap();

        // Compare against original in same memory order
        let orig = img.to_f32();
        let orig_slice = orig.as_slice_memory_order().unwrap();

        for i in 0..result_slice.len() {
            assert!(
                (result_slice[i] - orig_slice[i]).abs() < 1e-5,
                "Value mismatch at index {}: expected {}, got {}",
                i,
                orig_slice[i],
                result_slice[i]
            );
        }
    }

    #[test]
    fn test_crop_or_pad_mixed() {
        // Test with different crop/pad per dimension
        let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let img = create_test_image(data, [2, 3, 4]);

        // 2x3x4 -> 4x2x4 (pad depth, crop height, same width)
        let result = crop_or_pad(&img, &[4, 2, 4]).unwrap();
        assert_eq!(result.shape(), &[4, 2, 4]);
    }

    #[test]
    fn test_crop_or_pad_dimension_mismatch() {
        let data: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let img = create_test_image(data, [2, 2, 2]);

        // Wrong number of dimensions
        let result = crop_or_pad(&img, &[2, 2]);
        assert!(result.is_err());
    }

    #[test]
    fn test_flip_single_axis() {
        // Create a 2x2x2 volume with distinct values
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let img = create_test_image(data, [2, 2, 2]);

        // Flip along axis 0 (depth)
        let flipped = flip(&img, &[0]).unwrap();
        let result = flipped.to_f32();

        // After flipping axis 0:
        // Original: [[[1,2],[3,4]], [[5,6],[7,8]]]
        // Flipped:  [[[5,6],[7,8]], [[1,2],[3,4]]]
        // Check using indexing
        assert!((result[[0, 0, 0]] - 5.0).abs() < 1e-5);
        assert!((result[[1, 0, 0]] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_flip_multiple_axes() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let img = create_test_image(data, [2, 2, 2]);

        // Flip along axes 0 and 2
        let flipped = flip(&img, &[0, 2]).unwrap();
        assert_eq!(flipped.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_flip_all_axes() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let img = create_test_image(data, [2, 2, 2]);

        // Flip along all axes
        let flipped = flip(&img, &[0, 1, 2]).unwrap();
        let result = flipped.to_f32();

        // Flipping all axes reverses the data
        // [0,0,0] should have what was at [1,1,1]
        assert!((result[[0, 0, 0]] - 8.0).abs() < 1e-5);
        // [1,1,1] should have what was at [0,0,0]
        assert!((result[[1, 1, 1]] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_flip_empty_axes() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let img = create_test_image(data.clone(), [2, 2, 2]);

        // No flip - should be identity
        let flipped = flip(&img, &[]).unwrap();
        let result = flipped.to_f32();

        // Check all positions match original
        assert!((result[[0, 0, 0]] - 1.0).abs() < 1e-5);
        assert!((result[[0, 0, 1]] - 2.0).abs() < 1e-5);
        assert!((result[[1, 1, 1]] - 8.0).abs() < 1e-5);
    }

    #[test]
    fn test_flip_out_of_bounds() {
        let data = vec![1.0; 8];
        let img = create_test_image(data, [2, 2, 2]);

        // Axis 3 is out of bounds for 3D image
        let result = flip(&img, &[3]);
        assert!(result.is_err());
    }

    #[test]
    fn test_flip_preserves_shape() {
        let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let img = create_test_image(data, [2, 3, 4]);

        let flipped = flip(&img, &[1]).unwrap();
        assert_eq!(flipped.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_flip_double_flip_identity() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let img = create_test_image(data.clone(), [2, 2, 2]);

        // Flip twice along same axis should be identity
        let flipped1 = flip(&img, &[0]).unwrap();
        let flipped2 = flip(&flipped1, &[0]).unwrap();

        let result = flipped2.to_f32();

        // Double flip should be identity - check key positions
        assert!(
            (result[[0, 0, 0]] - 1.0).abs() < 1e-5,
            "Double flip should be identity at [0,0,0]: expected 1.0, got {}",
            result[[0, 0, 0]]
        );
        assert!(
            (result[[0, 0, 1]] - 2.0).abs() < 1e-5,
            "Double flip should be identity at [0,0,1]: expected 2.0, got {}",
            result[[0, 0, 1]]
        );
        assert!(
            (result[[1, 1, 1]] - 8.0).abs() < 1e-5,
            "Double flip should be identity at [1,1,1]: expected 8.0, got {}",
            result[[1, 1, 1]]
        );
    }
}
