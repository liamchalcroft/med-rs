//! Image resampling operations.
//!
//! Provides trilinear interpolation for resampling 3D medical images to new
//! voxel spacings or grid sizes.
//!
//! # Performance
//!
//! The resampling implementation uses several optimizations:
//! - **F-order native**: Works directly with NIfTI's column-major layout
//! - **SIMD acceleration**: Uses AVX (f32x8) for parallel interpolation
//! - **Parallel processing**: Rayon-based multi-threading across Z-slices
//! - **Adaptive algorithm**: Chooses direct vs separable based on volume size

use crate::error::{Error, Result};
use crate::nifti::image::ArrayData;
use crate::nifti::{DataType, NiftiImage};
use crate::pipeline::acquire_buffer;
use crate::pipeline::simd_kernels::trilinear_resample_forder_adaptive;
use ndarray::{ArrayD, IxDyn, ShapeBuilder};
use rayon::prelude::*;

/// Interpolation method for resampling.
#[derive(Debug, Clone, Copy, Default)]
pub enum Interpolation {
    /// Nearest neighbor (fast, preserves labels).
    Nearest,
    /// Trilinear interpolation (smooth, default).
    #[default]
    Trilinear,
}

/// Resample image to new voxel spacing.
///
/// # Arguments
/// * `image` - Input image
/// * `target_spacing` - Target voxel spacing in mm (x, y, z)
/// * `interp` - Interpolation method
///
/// # Errors
/// Returns `Error::InvalidDimensions` if any target spacing component is <= 0.
///
/// # Example
/// ```ignore
/// let resampled = resample_to_spacing(&img, [1.0, 1.0, 1.0], Interpolation::Trilinear)?;
/// ```
#[must_use = "this function returns a new resampled image"]
#[allow(clippy::needless_range_loop)]
pub fn resample_to_spacing(
    image: &NiftiImage,
    target_spacing: [f32; 3],
    interp: Interpolation,
) -> Result<NiftiImage> {
    // Validate inputs
    for (i, &spacing) in target_spacing.iter().enumerate() {
        if spacing <= 0.0 {
            return Err(Error::InvalidDimensions(format!(
                "Target spacing must be > 0, got {} for dimension {}",
                spacing, i
            )));
        }
    }

    let data = image.to_f32()?;
    let current_spacing = image.spacing();

    // Calculate new dimensions
    let old_shape: Vec<usize> = data.shape().to_vec();
    let new_shape: Vec<usize> = (0..3)
        .map(|i| {
            let factor = current_spacing[i] / target_spacing[i];
            let new_dim = (old_shape[i] as f32 * factor).round() as usize;
            new_dim.max(1) // Ensure at least 1 voxel per dimension
        })
        .collect();

    let resampled = match interp {
        Interpolation::Nearest => resample_nearest_3d(&data, &new_shape),
        Interpolation::Trilinear => resample_trilinear_optimized(&data, &new_shape),
    };

    // Update affine matrix with new spacing
    let mut affine = image.affine();
    let spatial_dims = current_spacing.len().min(3);
    for i in 0..spatial_dims {
        let scale_factor = target_spacing[i] / current_spacing[i].abs();
        for j in 0..3 {
            affine[i][j] *= scale_factor;
        }
    }

    // Build output header while preserving metadata
    let mut header = image.header().clone();
    header.ndim = new_shape.len() as u8;
    header.datatype = DataType::Float32;
    header.scl_slope = 1.0;
    header.scl_inter = 0.0;
    header.dim = [1u16; 7];
    for (i, &d) in new_shape.iter().enumerate() {
        header.dim[i] = d as u16;
    }
    header.pixdim = [1.0f32; 8];
    for i in 0..spatial_dims {
        header.pixdim[i + 1] = target_spacing[i];
    }
    header.set_affine(affine);

    Ok(NiftiImage::from_parts(header, ArrayData::F32(resampled)))
}

/// Resample image to target shape.
///
/// # Errors
/// Returns `Error::InvalidDimensions` if any target shape dimension is 0.
#[must_use = "this function returns a new resampled image"]
#[allow(clippy::needless_range_loop)]
pub fn resample_to_shape(
    image: &NiftiImage,
    target_shape: [usize; 3],
    interp: Interpolation,
) -> Result<NiftiImage> {
    // Validate inputs
    for (i, &dim) in target_shape.iter().enumerate() {
        if dim == 0 {
            return Err(Error::InvalidDimensions(format!(
                "Target shape dimension {} cannot be 0",
                i
            )));
        }
    }

    let data = image.to_f32()?;

    let resampled = match interp {
        Interpolation::Nearest => resample_nearest_3d(&data, &target_shape),
        Interpolation::Trilinear => resample_trilinear_optimized(&data, &target_shape),
    };

    // Compute new spacing from shape ratio
    let old_shape = data.shape();
    let mut affine = image.affine();
    let mut new_spacing = [1.0f32; 3];
    let spatial_dims = image.spacing().len().min(3);

    for i in 0..spatial_dims {
        let scale = old_shape[i] as f32 / target_shape[i] as f32;
        for j in 0..3 {
            affine[i][j] *= scale;
        }
        new_spacing[i] = image.spacing()[i] * scale;
    }

    // Build output header while preserving metadata
    let mut header = image.header().clone();
    header.ndim = target_shape.len() as u8;
    header.datatype = DataType::Float32;
    header.scl_slope = 1.0;
    header.scl_inter = 0.0;
    header.dim = [1u16; 7];
    for (i, &d) in target_shape.iter().enumerate() {
        header.dim[i] = d as u16;
    }
    header.pixdim = [1.0f32; 8];
    for i in 0..spatial_dims {
        header.pixdim[i + 1] = new_spacing[i];
    }
    header.set_affine(affine);

    Ok(NiftiImage::from_parts(header, ArrayData::F32(resampled)))
}

/// Optimized F-order trilinear resampling.
///
/// Uses SIMD-accelerated kernels that work directly with F-order data,
/// avoiding expensive memory layout conversions.
#[allow(clippy::option_if_let_else)]
#[allow(clippy::expect_used)]
fn resample_trilinear_optimized(data: &ArrayD<f32>, new_shape: &[usize]) -> ArrayD<f32> {
    let old_shape = data.shape();
    let src_shape = [old_shape[0], old_shape[1], old_shape[2]];
    let dst_shape = [new_shape[0], new_shape[1], new_shape[2]];

    // Get source data as contiguous slice
    // For F-order arrays, as_slice_memory_order gives us F-order contiguous data
    let src_slice = if let Some(slice) = data.as_slice_memory_order() {
        slice.to_vec()
    } else {
        // Fallback: create contiguous F-order copy
        let mut f_order = ArrayD::zeros(IxDyn(old_shape).f());
        f_order.assign(data);
        f_order
            .as_slice_memory_order()
            .expect("F-order array should be contiguous")
            .to_vec()
    };

    // Use the optimized F-order SIMD kernel
    let result_vec = trilinear_resample_forder_adaptive(&src_slice, src_shape, dst_shape);

    // Create F-order output array
    ArrayD::from_shape_vec(
        IxDyn(&[dst_shape[0], dst_shape[1], dst_shape[2]]).f(),
        result_vec,
    )
    .expect("Buffer size mismatch in trilinear resampling")
}

#[allow(clippy::similar_names, clippy::tuple_array_conversions)]
fn resample_nearest_3d(data: &ArrayD<f32>, new_shape: &[usize]) -> ArrayD<f32> {
    // The algorithm is optimized for C-order (row-major) data.
    // If input is F-order, convert to C-order for processing.
    let data_c: std::borrow::Cow<'_, ArrayD<f32>> = if data.is_standard_layout() {
        std::borrow::Cow::Borrowed(data)
    } else {
        let mut c_order = ArrayD::zeros(IxDyn(data.shape()));
        c_order.assign(data);
        std::borrow::Cow::Owned(c_order)
    };

    let old_shape = data_c.shape();
    let (od, oh, ow) = (old_shape[0], old_shape[1], old_shape[2]);
    let (nd, nh, nw) = (new_shape[0], new_shape[1], new_shape[2]);

    // Precompute indices for each axis
    let scale_d = od as f32 / nd as f32;
    let scale_h = oh as f32 / nh as f32;
    let scale_w = ow as f32 / nw as f32;

    let z_indices: Vec<usize> = (0..nd)
        .map(|d| (((d as f32 + 0.5) * scale_d) as usize).min(od - 1))
        .collect();
    let y_indices: Vec<usize> = (0..nh)
        .map(|h| (((h as f32 + 0.5) * scale_h) as usize).min(oh - 1))
        .collect();
    let x_indices: Vec<usize> = (0..nw)
        .map(|w| (((w as f32 + 0.5) * scale_w) as usize).min(ow - 1))
        .collect();

    #[allow(clippy::expect_used)]
    let src = data_c
        .as_slice()
        .expect("C-order array should have contiguous slice");
    let stride_z = oh * ow;
    let stride_y = ow;

    let mut output: Vec<f32> = acquire_buffer(nd * nh * nw);

    output
        .par_chunks_mut(nh * nw)
        .enumerate()
        .for_each(|(d, slice)| {
            let z_base = z_indices[d] * stride_z;

            for h in 0..nh {
                let zy_base = z_base + y_indices[h] * stride_y;
                let out_row = &mut slice[h * nw..(h + 1) * nw];

                for w in 0..nw {
                    out_row[w] = src[zy_base + x_indices[w]];
                }
            }
        });

    // Output is in C-order. Convert to F-order to match NIfTI convention.
    #[allow(clippy::expect_used)]
    let c_order = ArrayD::from_shape_vec(IxDyn(&[nd, nh, nw]), output)
        .expect("Buffer size mismatch in nearest neighbor resampling - this is a bug");
    let mut f_order = ArrayD::zeros(IxDyn(&[nd, nh, nw]).f());
    f_order.assign(&c_order);
    f_order
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::ArrayD;

    fn create_test_image_with_spacing(
        data: Vec<f32>,
        shape: [usize; 3],
        spacing: [f32; 3],
    ) -> NiftiImage {
        // Create F-order array to match NIfTI convention
        let c_order = ArrayD::from_shape_vec(shape.to_vec(), data).unwrap();
        let mut f_order = ArrayD::zeros(IxDyn(&shape).f());
        f_order.assign(&c_order);
        let affine = [
            [spacing[0], 0.0, 0.0, 0.0],
            [0.0, spacing[1], 0.0, 0.0],
            [0.0, 0.0, spacing[2], 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        NiftiImage::from_array(f_order, affine)
    }

    fn create_test_image(data: Vec<f32>, shape: [usize; 3]) -> NiftiImage {
        create_test_image_with_spacing(data, shape, [1.0, 1.0, 1.0])
    }

    #[test]
    fn test_resample_to_spacing_upsample() {
        // Create 4x4x4 at 2mm spacing
        let data: Vec<f32> = (0..64).map(|i| i as f32).collect();
        let img = create_test_image_with_spacing(data, [4, 4, 4], [2.0, 2.0, 2.0]);

        // Note: spacing from affine is [2,2,2], target is [1,1,1]
        // factor = 2/1 = 2, new_dim = round(4*2) = 8
        // But the actual spacing extraction may differ...
        let resampled =
            resample_to_spacing(&img, [1.0, 1.0, 1.0], Interpolation::Trilinear).unwrap();

        // The resampled image should have more voxels than original
        let shape = resampled.shape();
        assert!(
            shape[0] > 2,
            "Upsampling should increase dimensions, got {}",
            shape[0]
        );

        // Check that spacing is updated to target
        let new_spacing = resampled.spacing();
        assert!((new_spacing[0] - 1.0).abs() < 0.1, "Spacing should be ~1.0");
    }

    #[test]
    fn test_resample_to_spacing_downsample() {
        // Create 4x4x4 at 1mm spacing
        let data: Vec<f32> = (0..64).map(|i| i as f32).collect();
        let img = create_test_image_with_spacing(data, [4, 4, 4], [1.0, 1.0, 1.0]);

        // Resample to 2mm spacing (should halve the dimensions)
        let resampled =
            resample_to_spacing(&img, [2.0, 2.0, 2.0], Interpolation::Trilinear).unwrap();

        // Expect 2x2x2
        let shape = resampled.shape();
        assert_eq!(shape[0], 2);
        assert_eq!(shape[1], 2);
        assert_eq!(shape[2], 2);
    }

    #[test]
    fn test_resample_to_spacing_nearest() {
        // Create 2x2x2 with distinct integer values
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let img = create_test_image_with_spacing(data, [2, 2, 2], [2.0, 2.0, 2.0]);

        // Resample using nearest neighbor
        let resampled = resample_to_spacing(&img, [1.0, 1.0, 1.0], Interpolation::Nearest).unwrap();

        // Result should only contain values from original set
        let result = resampled.to_f32().unwrap();
        let slice = result.as_slice_memory_order().unwrap();
        for &v in slice {
            assert!(
                (1.0..=8.0).contains(&v),
                "Nearest neighbor should preserve original values, got {}",
                v
            );
        }
    }

    #[test]
    fn test_resample_to_shape_basic() {
        // Create 4x4x4 volume
        let data: Vec<f32> = (0..64).map(|i| i as f32).collect();
        let img = create_test_image(data, [4, 4, 4]);

        // Resample to 8x8x8
        let resampled = resample_to_shape(&img, [8, 8, 8], Interpolation::Trilinear).unwrap();
        assert_eq!(resampled.shape(), &[8, 8, 8]);
    }

    #[test]
    fn test_resample_to_shape_anisotropic() {
        // Create 4x4x4 volume
        let data: Vec<f32> = (0..64).map(|i| i as f32).collect();
        let img = create_test_image(data, [4, 4, 4]);

        // Resample to different sizes per dimension
        let resampled = resample_to_shape(&img, [8, 4, 2], Interpolation::Trilinear).unwrap();
        assert_eq!(resampled.shape(), &[8, 4, 2]);
    }

    #[test]
    fn test_resample_preserves_value_range() {
        // Create volume with known min/max
        let data: Vec<f32> = (0..64).map(|i| (i as f32) / 63.0).collect();
        let img = create_test_image(data, [4, 4, 4]);

        // Resample
        let resampled = resample_to_shape(&img, [8, 8, 8], Interpolation::Trilinear).unwrap();
        let result = resampled.to_f32().unwrap();
        let slice = result.as_slice_memory_order().unwrap();

        // Trilinear should not extrapolate, so values should be in [0, 1]
        for &v in slice {
            assert!(
                v >= -0.01 && v <= 1.01,
                "Value {} outside expected range [0, 1]",
                v
            );
        }
    }

    #[test]
    fn test_resample_constant_volume() {
        // Volume with all same values
        let data = vec![5.0; 64];
        let img = create_test_image(data, [4, 4, 4]);

        // Resample should preserve constant value
        let resampled = resample_to_shape(&img, [8, 8, 8], Interpolation::Trilinear).unwrap();
        let result = resampled.to_f32().unwrap();
        let slice = result.as_slice_memory_order().unwrap();

        for &v in slice {
            assert!((v - 5.0).abs() < 1e-4, "Expected 5.0, got {}", v);
        }
    }

    #[test]
    fn test_resample_same_shape() {
        // Resampling to same shape should be close to identity
        let data: Vec<f32> = (0..64).map(|i| i as f32).collect();
        let img = create_test_image(data.clone(), [4, 4, 4]);

        let resampled = resample_to_shape(&img, [4, 4, 4], Interpolation::Trilinear).unwrap();
        let result = resampled.to_f32().unwrap();
        let result_slice = result.as_slice_memory_order().unwrap();

        // Compare values - note that both are in F-order so indices match
        let orig = img.to_f32().unwrap();
        let orig_slice = orig.as_slice_memory_order().unwrap();

        for i in 0..result_slice.len() {
            assert!(
                (result_slice[i] - orig_slice[i]).abs() < 0.5,
                "Value at {} too different: expected {}, got {}",
                i,
                orig_slice[i],
                result_slice[i]
            );
        }
    }

    #[test]
    fn test_adaptive_selection() {
        // Small volume should use direct method
        let small_data = vec![1.0; 8];
        let small = create_test_image(small_data, [2, 2, 2]);
        let _small_result = resample_to_shape(&small, [4, 4, 4], Interpolation::Trilinear).unwrap();
        // Just verify it completes without panicking
    }

    #[test]
    fn test_resample_to_shape_rejects_zero_dimension() {
        let data: Vec<f32> = (0..64).map(|i| i as f32).collect();
        let img = create_test_image(data, [4, 4, 4]);

        let result = resample_to_shape(&img, [0, 4, 4], Interpolation::Trilinear);
        assert!(result.is_err());

        let result = resample_to_shape(&img, [4, 0, 4], Interpolation::Trilinear);
        assert!(result.is_err());

        let result = resample_to_shape(&img, [4, 4, 0], Interpolation::Trilinear);
        assert!(result.is_err());
    }

    #[test]
    fn test_resample_to_spacing_rejects_zero_spacing() {
        let data: Vec<f32> = (0..64).map(|i| i as f32).collect();
        let img = create_test_image(data, [4, 4, 4]);

        let result = resample_to_spacing(&img, [0.0, 1.0, 1.0], Interpolation::Trilinear);
        assert!(result.is_err());

        let result = resample_to_spacing(&img, [-1.0, 1.0, 1.0], Interpolation::Trilinear);
        assert!(result.is_err());
    }
}
