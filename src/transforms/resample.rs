//! Image resampling operations.
//!
//! Provides trilinear interpolation for resampling 3D medical images to new
//! voxel spacings or grid sizes.

use crate::nifti::image::ArrayData;
use crate::nifti::{DataType, NiftiImage};
use crate::pipeline::acquire_buffer;
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
/// # Example
/// ```ignore
/// let resampled = resample_to_spacing(&img, [1.0, 1.0, 1.0], Interpolation::Trilinear);
/// ```
pub fn resample_to_spacing(
    image: &NiftiImage,
    target_spacing: [f32; 3],
    interp: Interpolation,
) -> NiftiImage {
    let data = image.to_f32();
    let current_spacing = image.spacing();

    // Calculate new dimensions
    let old_shape: Vec<usize> = data.shape().to_vec();
    let new_shape: Vec<usize> = (0..3)
        .map(|i| {
            let factor = current_spacing[i] / target_spacing[i];
            (old_shape[i] as f32 * factor).round() as usize
        })
        .collect();

    let resampled = match interp {
        Interpolation::Nearest => resample_nearest_3d(&data, &new_shape),
        Interpolation::Trilinear => resample_trilinear_adaptive(&data, &new_shape),
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
    header.pixdim = [1.0f32; 7];
    for i in 0..spatial_dims {
        header.pixdim[i] = target_spacing[i];
    }
    header.set_affine(affine);

    NiftiImage::from_parts(header, ArrayData::F32(resampled))
}

/// Resample image to target shape.
pub fn resample_to_shape(
    image: &NiftiImage,
    target_shape: [usize; 3],
    interp: Interpolation,
) -> NiftiImage {
    let data = image.to_f32();

    let resampled = match interp {
        Interpolation::Nearest => resample_nearest_3d(&data, &target_shape),
        Interpolation::Trilinear => resample_trilinear_adaptive(&data, &target_shape),
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
    header.pixdim = [1.0f32; 7];
    for i in 0..spatial_dims {
        header.pixdim[i] = new_spacing[i];
    }
    header.set_affine(affine);

    NiftiImage::from_parts(header, ArrayData::F32(resampled))
}

/// Precomputed interpolation parameters for one axis.
struct InterpParams {
    idx0: Vec<usize>, // Lower index
    idx1: Vec<usize>, // Upper index
    frac: Vec<f32>,   // Fractional part (weight for idx1)
}

impl InterpParams {
    fn new(new_size: usize, old_size: usize) -> Self {
        let scale = (old_size - 1) as f32 / (new_size - 1).max(1) as f32;
        let mut idx0 = Vec::with_capacity(new_size);
        let mut idx1 = Vec::with_capacity(new_size);
        let mut frac = Vec::with_capacity(new_size);

        for i in 0..new_size {
            let pos = i as f32 * scale;
            let i0 = pos.floor() as usize;
            let i1 = (i0 + 1).min(old_size - 1);
            idx0.push(i0);
            idx1.push(i1);
            frac.push(pos - i0 as f32);
        }

        Self { idx0, idx1, frac }
    }
}

#[allow(clippy::similar_names)]
fn resample_trilinear_3d(data: &ArrayD<f32>, new_shape: &[usize]) -> ArrayD<f32> {
    use crate::pipeline::simd_kernels::trilinear_row_simd;

    // The SIMD algorithm is optimized for C-order (row-major) data.
    // If input is F-order, convert to C-order for processing.
    let data_c: std::borrow::Cow<'_, ArrayD<f32>> = if data.is_standard_layout() {
        std::borrow::Cow::Borrowed(data)
    } else {
        // Convert F-order to C-order by creating a new array with standard layout
        let mut c_order = ArrayD::zeros(IxDyn(data.shape()));
        c_order.assign(data);
        std::borrow::Cow::Owned(c_order)
    };

    let old_shape = data_c.shape();
    let (od, oh, ow) = (old_shape[0], old_shape[1], old_shape[2]);
    let (nd, nh, nw) = (new_shape[0], new_shape[1], new_shape[2]);

    // Precompute interpolation parameters for each axis
    let z_params = InterpParams::new(nd, od);
    let y_params = InterpParams::new(nh, oh);
    let x_params = InterpParams::new(nw, ow);

    let src = data_c.as_slice().expect("C-order array should have contiguous slice");
    let stride_z = oh * ow;
    let stride_y = ow;

    let mut output: Vec<f32> = acquire_buffer(nd * nh * nw);

    // Parallel processing over depth slices
    output
        .par_chunks_mut(nh * nw)
        .enumerate()
        .for_each(|(d, slice)| {
            let z0 = z_params.idx0[d];
            let z1 = z_params.idx1[d];
            let zf = z_params.frac[d];

            for h in 0..nh {
                let y0 = y_params.idx0[h];
                let y1 = y_params.idx1[h];
                let yf = y_params.frac[h];

                let out_row = &mut slice[h * nw..(h + 1) * nw];

                // Use SIMD-optimized row interpolation
                trilinear_row_simd(
                    src,
                    stride_z,
                    stride_y,
                    z0,
                    z1,
                    y0,
                    y1,
                    zf,
                    yf,
                    &x_params.idx0,
                    &x_params.idx1,
                    &x_params.frac,
                    out_row,
                );
            }
        });

    // Output is in C-order. Convert to F-order to match NIfTI convention.
    let c_order = ArrayD::from_shape_vec(IxDyn(&[nd, nh, nw]), output).unwrap();
    let mut f_order = ArrayD::zeros(IxDyn(&[nd, nh, nw]).f());
    f_order.assign(&c_order);
    f_order
}

/// Separable trilinear resampling (cache-friendly approach).
///
/// Processes each axis independently in three passes:
/// 1. Resample along X (innermost) - best cache locality
/// 2. Resample along Y
/// 3. Resample along Z (outermost)
///
/// This approach has better cache behavior for large volumes.
fn resample_trilinear_separable(data: &ArrayD<f32>, new_shape: &[usize]) -> ArrayD<f32> {
    use crate::pipeline::simd_kernels::{lerp_1d_simd, SIMD_WIDTH};

    // The SIMD algorithm is optimized for C-order (row-major) data.
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

    // Pass 1: Resample along X (old shape: od x oh x ow -> od x oh x nw)
    let x_params = InterpParams::new(nw, ow);
    let mut temp1: Vec<f32> = acquire_buffer(od * oh * nw);

    let src_slice = data_c.as_slice().expect("C-order array should have contiguous slice");
    temp1
        .par_chunks_mut(nw)
        .enumerate()
        .for_each(|(idx, out_row)| {
            let z = idx / oh;
            let y = idx % oh;
            let src_base = z * oh * ow + y * ow;
            let src_row = &src_slice[src_base..src_base + ow];

            // SIMD interpolation along X - process 8 output values at a time
            let chunks = nw / SIMD_WIDTH;
            for chunk_i in 0..chunks {
                let base = chunk_i * SIMD_WIDTH;

                // Gather values and interpolate
                let mut vals = [0.0f32; 8];
                for i in 0..SIMD_WIDTH {
                    let w = base + i;
                    let x0 = x_params.idx0[w];
                    let x1 = x_params.idx1[w];
                    let f = x_params.frac[w];
                    vals[i] = src_row[x0] * (1.0 - f) + src_row[x1] * f;
                }
                out_row[base..base + SIMD_WIDTH].copy_from_slice(&vals);
            }

            // Scalar remainder
            for w in (chunks * SIMD_WIDTH)..nw {
                let x0 = x_params.idx0[w];
                let x1 = x_params.idx1[w];
                let f = x_params.frac[w];
                out_row[w] = src_row[x0] * (1.0 - f) + src_row[x1] * f;
            }
        });

    // Pass 2: Resample along Y (shape: od x oh x nw -> od x nh x nw)
    let y_params = InterpParams::new(nh, oh);
    let mut temp2: Vec<f32> = acquire_buffer(od * nh * nw);

    for z in 0..od {
        let z_base_in = z * oh * nw;
        let z_base_out = z * nh * nw;

        // Process each output row in parallel
        temp2[z_base_out..z_base_out + nh * nw]
            .par_chunks_mut(nw)
            .enumerate()
            .for_each(|(h, out_row)| {
                let y0 = y_params.idx0[h];
                let y1 = y_params.idx1[h];
                let f = y_params.frac[h];

                let row0 = &temp1[z_base_in + y0 * nw..z_base_in + y0 * nw + nw];
                let row1 = &temp1[z_base_in + y1 * nw..z_base_in + y1 * nw + nw];

                // Use centralized SIMD lerp function
                lerp_1d_simd(row0, row1, f, out_row);
            });
    }

    // Release temp1 early
    drop(temp1);

    // Pass 3: Resample along Z (shape: od x nh x nw -> nd x nh x nw)
    let z_params = InterpParams::new(nd, od);
    let mut output: Vec<f32> = acquire_buffer(nd * nh * nw);
    let slice_size = nh * nw;

    output
        .par_chunks_mut(slice_size)
        .enumerate()
        .for_each(|(d, out_slice)| {
            let z0 = z_params.idx0[d];
            let z1 = z_params.idx1[d];
            let f = z_params.frac[d];

            let slice0 = &temp2[z0 * slice_size..(z0 + 1) * slice_size];
            let slice1 = &temp2[z1 * slice_size..(z1 + 1) * slice_size];

            // Use centralized SIMD lerp function
            lerp_1d_simd(slice0, slice1, f, out_slice);
        });

    // Output is in C-order. Convert to F-order to match NIfTI convention.
    let c_order = ArrayD::from_shape_vec(IxDyn(&[nd, nh, nw]), output).unwrap();
    let mut f_order = ArrayD::zeros(IxDyn(&[nd, nh, nw]).f());
    f_order.assign(&c_order);
    f_order
}

/// Choose between direct and separable resampling based on volume size.
fn resample_trilinear_adaptive(data: &ArrayD<f32>, new_shape: &[usize]) -> ArrayD<f32> {
    let old_shape = data.shape();
    let total_voxels = old_shape[0] * old_shape[1] * old_shape[2];

    // Use separable for larger volumes (>64MB) where cache effects matter more
    if total_voxels > 16 * 1024 * 1024 {
        resample_trilinear_separable(data, new_shape)
    } else {
        resample_trilinear_3d(data, new_shape)
    }
}

#[allow(clippy::similar_names)]
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

    let src = data_c.as_slice().expect("C-order array should have contiguous slice");
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
    let c_order = ArrayD::from_shape_vec(IxDyn(&[nd, nh, nw]), output).unwrap();
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
        let resampled = resample_to_spacing(&img, [1.0, 1.0, 1.0], Interpolation::Trilinear);

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
        let resampled = resample_to_spacing(&img, [2.0, 2.0, 2.0], Interpolation::Trilinear);

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
        let resampled = resample_to_spacing(&img, [1.0, 1.0, 1.0], Interpolation::Nearest);

        // Result should only contain values from original set
        let result = resampled.to_f32();
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
        let resampled = resample_to_shape(&img, [8, 8, 8], Interpolation::Trilinear);
        assert_eq!(resampled.shape(), &[8, 8, 8]);
    }

    #[test]
    fn test_resample_to_shape_anisotropic() {
        // Create 4x4x4 volume
        let data: Vec<f32> = (0..64).map(|i| i as f32).collect();
        let img = create_test_image(data, [4, 4, 4]);

        // Resample to different sizes per dimension
        let resampled = resample_to_shape(&img, [8, 4, 2], Interpolation::Trilinear);
        assert_eq!(resampled.shape(), &[8, 4, 2]);
    }

    #[test]
    fn test_resample_preserves_value_range() {
        // Create volume with known min/max
        let data: Vec<f32> = (0..64).map(|i| (i as f32) / 63.0).collect();
        let img = create_test_image(data, [4, 4, 4]);

        // Resample
        let resampled = resample_to_shape(&img, [8, 8, 8], Interpolation::Trilinear);
        let result = resampled.to_f32();
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
        let resampled = resample_to_shape(&img, [8, 8, 8], Interpolation::Trilinear);
        let result = resampled.to_f32();
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

        let resampled = resample_to_shape(&img, [4, 4, 4], Interpolation::Trilinear);
        let result = resampled.to_f32();
        let result_slice = result.as_slice_memory_order().unwrap();

        // Compare values - note that both are in F-order so indices match
        let orig = img.to_f32();
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
    fn test_interp_params() {
        // Test interpolation parameter calculation
        let params = InterpParams::new(4, 2);

        // For 2->4, we expect indices and fractions for smooth interpolation
        assert_eq!(params.idx0.len(), 4);
        assert_eq!(params.idx1.len(), 4);
        assert_eq!(params.frac.len(), 4);

        // First point should map to 0
        assert_eq!(params.idx0[0], 0);

        // Last point should map to last index
        assert!(params.idx0[3] <= 1);
        assert!(params.idx1[3] <= 1);
    }

    #[test]
    fn test_adaptive_selection() {
        // Small volume should use direct method
        let small_data = vec![1.0; 8];
        let small = create_test_image(small_data, [2, 2, 2]);
        let _small_result = resample_to_shape(&small, [4, 4, 4], Interpolation::Trilinear);
        // Just verify it completes without panicking
    }
}
