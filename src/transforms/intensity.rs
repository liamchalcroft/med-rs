use crate::nifti::image::ArrayData;
use crate::nifti::{DataType, NiftiImage};
use crate::pipeline::acquire_buffer;
use crate::pipeline::simd_kernels::{parallel_linear_transform_f32, parallel_sum_and_sum_sq_f32};
use ndarray::{ArrayD, IxDyn, ShapeBuilder};
use rayon::prelude::*;

/// Normalize image intensity to zero mean and unit variance.
///
/// Formula: output = (input - mean) / std
pub fn z_normalization(image: &NiftiImage) -> NiftiImage {
    let mut header = image.header().clone();

    // Use data_cow() to avoid cloning if already owned
    let data = image.data_cow();

    // Fast path for f32 (most common case) - SIMD + parallel
    if let ArrayData::F32(a) = data.as_ref() {
        let slice = a.as_slice_memory_order().unwrap();
        let len = slice.len();

        // SIMD-accelerated statistics computation
        let (sum, sum_sq, _) = parallel_sum_and_sum_sq_f32(slice);

        let mean = (sum / len as f64) as f32;
        let variance = (sum_sq / len as f64) - (mean as f64 * mean as f64);
        let inv_std = if variance <= 0.0 {
            1.0f32
        } else {
            1.0 / (variance.sqrt() as f32)
        };

        // SIMD-accelerated transformation: output = (input - mean) * inv_std = input * inv_std - mean * inv_std
        // Use memory pool for buffer reuse in pipelines
        let mut output = acquire_buffer(len);
        let offset = -mean * inv_std;
        parallel_linear_transform_f32(slice, &mut output, inv_std, offset);

        // Return result in F-order to match NIfTI convention
        let out_array = ArrayD::from_shape_vec(IxDyn(a.shape()).f(), output).unwrap();
        header.datatype = DataType::Float32;
        header.scl_slope = 1.0;
        header.scl_inter = 0.0;
        return NiftiImage::from_parts(header, ArrayData::F32(out_array));
    }

    // Generic path for other types
    macro_rules! normalize {
        ($array:expr, $to_f64:expr, $from_f32:expr) => {{
            let slice = $array.as_slice_memory_order().unwrap();
            let len = slice.len();

            let (sum, sum_sq) = slice
                .par_iter()
                .map(|&v| {
                    let val = $to_f64(v);
                    (val, val * val)
                })
                .reduce(|| (0.0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1));

            let mean = sum / len as f64;
            let variance = (sum_sq / len as f64) - (mean * mean);
            let inv_std = if variance <= 0.0 {
                1.0
            } else {
                1.0 / variance.sqrt()
            };

            // Parallel transformation (use memory pool)
            let mut output = acquire_buffer(len);
            output
                .par_iter_mut()
                .zip(slice.par_iter())
                .for_each(|(out, &v)| {
                    let val = $to_f64(v);
                    *out = $from_f32((val - mean) * inv_std);
                });

            // F-order to match NIfTI convention
            let out_array = ArrayD::from_shape_vec(IxDyn($array.shape()).f(), output).unwrap();
            header.datatype = DataType::Float32;
            ArrayData::F32(out_array)
        }};
    }

    let new_data = match data.as_ref() {
        ArrayData::U8(a) => normalize!(a, |v: u8| v as f64, |v: f64| v as f32),
        ArrayData::I8(a) => normalize!(a, |v: i8| v as f64, |v: f64| v as f32),
        ArrayData::I16(a) => normalize!(a, |v: i16| v as f64, |v: f64| v as f32),
        ArrayData::U16(a) => normalize!(a, |v: u16| v as f64, |v: f64| v as f32),
        ArrayData::I32(a) => normalize!(a, |v: i32| v as f64, |v: f64| v as f32),
        ArrayData::U32(a) => normalize!(a, |v: u32| v as f64, |v: f64| v as f32),
        ArrayData::I64(a) => normalize!(a, |v: i64| v as f64, |v: f64| v as f32),
        ArrayData::U64(a) => normalize!(a, |v: u64| v as f64, |v: f64| v as f32),
        ArrayData::F16(a) => normalize!(a, |v: half::f16| v.to_f64(), |v: f64| v as f32),
        ArrayData::BF16(a) => normalize!(a, |v: half::bf16| v.to_f64(), |v: f64| v as f32),
        ArrayData::F32(_) => unreachable!(), // Handled above
        ArrayData::F64(a) => {
            let slice = a.as_slice_memory_order().unwrap();
            let len = slice.len();

            let (sum, sum_sq) = slice
                .par_iter()
                .map(|&v| (v, v * v))
                .reduce(|| (0.0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1));

            let mean = sum / len as f64;
            let variance = (sum_sq / len as f64) - (mean * mean);
            let inv_std = if variance <= 0.0 {
                1.0
            } else {
                1.0 / variance.sqrt()
            };

            let mut output = vec![0.0f64; len];
            output
                .par_iter_mut()
                .zip(slice.par_iter())
                .for_each(|(out, &v)| {
                    *out = (v - mean) * inv_std;
                });

            // F-order to match NIfTI convention
            let out_array = ArrayD::from_shape_vec(IxDyn(a.shape()).f(), output).unwrap();
            header.datatype = DataType::Float64;
            ArrayData::F64(out_array)
        }
    };

    header.scl_slope = 1.0;
    header.scl_inter = 0.0;

    NiftiImage::from_parts(header, new_data)
}

/// Rescale image intensity to a specific range.
///
/// Formula: output = (input - min) / (max - min) * (out_max - out_min) + out_min
pub fn rescale_intensity(image: &NiftiImage, out_min: f64, out_max: f64) -> NiftiImage {
    use crate::pipeline::simd_kernels::parallel_minmax_f32;

    let mut header = image.header().clone();

    // Use data_cow() to avoid cloning if already owned
    let data = image.data_cow();

    // Fast path for f32 - SIMD + parallel
    if let ArrayData::F32(a) = data.as_ref() {
        let slice = a.as_slice_memory_order().unwrap();

        // SIMD-accelerated min/max
        let (min, max) = parallel_minmax_f32(slice);

        // Precompute scale and offset: out = (v - min) * scale + out_min = v * scale + offset
        let range = if max - min == 0.0 { 1.0 } else { max - min };
        let scale = ((out_max - out_min) / range as f64) as f32;
        let offset = out_min as f32 - min * scale;

        // SIMD-accelerated rescaling (use memory pool)
        let mut output = acquire_buffer(slice.len());
        parallel_linear_transform_f32(slice, &mut output, scale, offset);

        // F-order to match NIfTI convention
        let out_array = ArrayD::from_shape_vec(IxDyn(a.shape()).f(), output).unwrap();
        header.datatype = DataType::Float32;
        header.scl_slope = 1.0;
        header.scl_inter = 0.0;
        return NiftiImage::from_parts(header, ArrayData::F32(out_array));
    }

    // Generic path for other types
    macro_rules! rescale {
        ($array:expr, $to_f64:expr) => {{
            let slice = $array.as_slice_memory_order().unwrap();
            let len = slice.len();

            let (min, max) = slice
                .par_iter()
                .map(|&v| {
                    let val = $to_f64(v);
                    (val, val)
                })
                .reduce(
                    || (f64::INFINITY, f64::NEG_INFINITY),
                    |a, b| (a.0.min(b.0), a.1.max(b.1)),
                );

            // Precompute scale and offset
            let range = if max - min == 0.0 { 1.0 } else { max - min };
            let scale = ((out_max - out_min) / range) as f32;
            let offset = (out_min - min * (out_max - out_min) / range) as f32;

            // Parallel transformation (use memory pool)
            let mut output = acquire_buffer(len);
            output
                .par_iter_mut()
                .zip(slice.par_iter())
                .for_each(|(out, &v)| {
                    let val = $to_f64(v) as f32;
                    *out = val * scale + offset;
                });

            // F-order to match NIfTI convention
            let out_array = ArrayD::from_shape_vec(IxDyn($array.shape()).f(), output).unwrap();
            header.datatype = DataType::Float32;
            ArrayData::F32(out_array)
        }};
    }

    let new_data = match data.as_ref() {
        ArrayData::U8(a) => rescale!(a, |v: u8| v as f64),
        ArrayData::I8(a) => rescale!(a, |v: i8| v as f64),
        ArrayData::I16(a) => rescale!(a, |v: i16| v as f64),
        ArrayData::U16(a) => rescale!(a, |v: u16| v as f64),
        ArrayData::I32(a) => rescale!(a, |v: i32| v as f64),
        ArrayData::U32(a) => rescale!(a, |v: u32| v as f64),
        ArrayData::I64(a) => rescale!(a, |v: i64| v as f64),
        ArrayData::U64(a) => rescale!(a, |v: u64| v as f64),
        ArrayData::F16(a) => rescale!(a, |v: half::f16| v.to_f64()),
        ArrayData::BF16(a) => rescale!(a, |v: half::bf16| v.to_f64()),
        ArrayData::F32(_) => unreachable!(), // Handled above
        ArrayData::F64(a) => {
            let slice = a.as_slice_memory_order().unwrap();
            let len = slice.len();

            let (min, max) = slice.par_iter().map(|&v| (v, v)).reduce(
                || (f64::INFINITY, f64::NEG_INFINITY),
                |a, b| (a.0.min(b.0), a.1.max(b.1)),
            );

            let range = if max - min == 0.0 { 1.0 } else { max - min };
            let scale = (out_max - out_min) / range;
            let offset = out_min - min * scale;

            let mut output = vec![0.0f64; len];
            output
                .par_iter_mut()
                .zip(slice.par_iter())
                .for_each(|(out, &v)| {
                    *out = v * scale + offset;
                });

            // F-order to match NIfTI convention
            let out_array = ArrayD::from_shape_vec(IxDyn(a.shape()).f(), output).unwrap();
            header.datatype = DataType::Float64;
            ArrayData::F64(out_array)
        }
    };

    header.scl_slope = 1.0;
    header.scl_inter = 0.0;

    NiftiImage::from_parts(header, new_data)
}

/// Clamp image intensity to a specific range.
pub fn clamp(image: &NiftiImage, min: f64, max: f64) -> NiftiImage {
    use crate::pipeline::simd_kernels::parallel_linear_transform_clamp_f32;

    let header = image.header().clone();

    // Use data_cow() to avoid cloning if already owned
    let data = image.data_cow();

    // Fast path for f32 - SIMD + parallel
    if let ArrayData::F32(a) = data.as_ref() {
        let slice = a.as_slice_memory_order().unwrap();
        let (min_f, max_f) = (min as f32, max as f32);

        // SIMD-accelerated clamping (identity transform with clamp, use memory pool)
        let mut output = acquire_buffer(slice.len());
        parallel_linear_transform_clamp_f32(slice, &mut output, 1.0, 0.0, min_f, max_f);

        // F-order to match NIfTI convention
        let out_array = ArrayD::from_shape_vec(IxDyn(a.shape()).f(), output).unwrap();
        return NiftiImage::from_parts(header, ArrayData::F32(out_array));
    }

    // Generic path - use mapv for simplicity (less common types)
    let new_data = match data.into_owned() {
        ArrayData::U8(a) => {
            let (min, max) = (min as u8, max as u8);
            ArrayData::U8(a.mapv(|v| v.clamp(min, max)))
        }
        ArrayData::I8(a) => {
            let (min, max) = (min as i8, max as i8);
            ArrayData::I8(a.mapv(|v| v.clamp(min, max)))
        }
        ArrayData::I16(a) => {
            let (min, max) = (min as i16, max as i16);
            ArrayData::I16(a.mapv(|v| v.clamp(min, max)))
        }
        ArrayData::U16(a) => {
            let (min, max) = (min as u16, max as u16);
            ArrayData::U16(a.mapv(|v| v.clamp(min, max)))
        }
        ArrayData::I32(a) => {
            let (min, max) = (min as i32, max as i32);
            ArrayData::I32(a.mapv(|v| v.clamp(min, max)))
        }
        ArrayData::U32(a) => {
            let (min, max) = (min as u32, max as u32);
            ArrayData::U32(a.mapv(|v| v.clamp(min, max)))
        }
        ArrayData::I64(a) => {
            let (min, max) = (min as i64, max as i64);
            ArrayData::I64(a.mapv(|v| v.clamp(min, max)))
        }
        ArrayData::U64(a) => {
            let (min, max) = (min as u64, max as u64);
            ArrayData::U64(a.mapv(|v| v.clamp(min, max)))
        }
        ArrayData::F16(a) => {
            let (min, max) = (half::f16::from_f64(min), half::f16::from_f64(max));
            ArrayData::F16(a.mapv(|v| v.clamp(min, max)))
        }
        ArrayData::BF16(a) => {
            let (min, max) = (half::bf16::from_f64(min), half::bf16::from_f64(max));
            ArrayData::BF16(a.mapv(|v| v.clamp(min, max)))
        }
        ArrayData::F32(_) => unreachable!(), // Handled above
        ArrayData::F64(a) => ArrayData::F64(a.mapv(|v| v.clamp(min, max))),
    };

    NiftiImage::from_parts(header, new_data)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::ArrayD;

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
    fn test_z_normalization_basic() {
        // Create a simple 2x2x2 volume
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let img = create_test_image(data, [2, 2, 2]);

        let normalized = z_normalization(&img);
        let result = normalized.to_f32();
        let result_slice = result.as_slice_memory_order().unwrap();

        // After z-normalization, mean should be ~0 and std should be ~1
        let mean: f32 = result_slice.iter().sum::<f32>() / result_slice.len() as f32;
        let variance: f32 = result_slice.iter().map(|v| (v - mean).powi(2)).sum::<f32>()
            / result_slice.len() as f32;
        let std = variance.sqrt();

        assert!(mean.abs() < 1e-5, "Mean should be ~0, got {}", mean);
        assert!((std - 1.0).abs() < 1e-5, "Std should be ~1, got {}", std);
    }

    #[test]
    fn test_z_normalization_constant_value() {
        // All same values - should not produce NaN
        let data = vec![5.0; 8];
        let img = create_test_image(data, [2, 2, 2]);

        let normalized = z_normalization(&img);
        let result = normalized.to_f32();
        let result_slice = result.as_slice_memory_order().unwrap();

        // With zero variance, should handle gracefully (no NaN)
        for &v in result_slice {
            assert!(!v.is_nan(), "Should not produce NaN");
        }
    }

    #[test]
    fn test_rescale_intensity_basic() {
        let data = vec![0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0];
        let img = create_test_image(data, [2, 2, 2]);

        let rescaled = rescale_intensity(&img, 0.0, 1.0);
        let result = rescaled.to_f32();
        let result_slice = result.as_slice_memory_order().unwrap();

        // After rescaling to [0, 1], min should be 0 and max should be 1
        let min = result_slice.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = result_slice
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);

        assert!((min - 0.0).abs() < 1e-5, "Min should be 0, got {}", min);
        assert!((max - 1.0).abs() < 1e-5, "Max should be 1, got {}", max);
    }

    #[test]
    fn test_rescale_intensity_custom_range() {
        let data = vec![0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0];
        let img = create_test_image(data, [2, 2, 2]);

        let rescaled = rescale_intensity(&img, -1.0, 1.0);
        let result = rescaled.to_f32();
        let result_slice = result.as_slice_memory_order().unwrap();

        let min = result_slice.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = result_slice
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);

        assert!((min - (-1.0)).abs() < 1e-5, "Min should be -1, got {}", min);
        assert!((max - 1.0).abs() < 1e-5, "Max should be 1, got {}", max);
    }

    #[test]
    fn test_rescale_intensity_constant() {
        // All same values - should handle gracefully
        let data = vec![5.0; 8];
        let img = create_test_image(data, [2, 2, 2]);

        let rescaled = rescale_intensity(&img, 0.0, 1.0);
        let result = rescaled.to_f32();
        let result_slice = result.as_slice_memory_order().unwrap();

        // With zero range, should not produce NaN
        for &v in result_slice {
            assert!(!v.is_nan(), "Should not produce NaN");
        }
    }

    #[test]
    fn test_clamp_basic() {
        let data = vec![-10.0, 0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0];
        let img = create_test_image(data, [2, 2, 2]);

        let clamped = clamp(&img, 0.0, 20.0);
        let result = clamped.to_f32();
        let result_slice = result.as_slice_memory_order().unwrap();

        for &v in result_slice {
            assert!(v >= 0.0, "Value {} should be >= 0", v);
            assert!(v <= 20.0, "Value {} should be <= 20", v);
        }

        // Check that values are properly clamped (don't rely on specific indices
        // since F-order changes the memory layout)
        let orig = img.to_f32();
        let orig_slice = orig.as_slice_memory_order().unwrap();
        for i in 0..result_slice.len() {
            let expected = orig_slice[i].max(0.0).min(20.0);
            assert!(
                (result_slice[i] - expected).abs() < 1e-5,
                "Value at {} should be clamped: expected {}, got {}",
                i,
                expected,
                result_slice[i]
            );
        }
    }

    #[test]
    fn test_clamp_negative_range() {
        let data = vec![-10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0, 25.0];
        let img = create_test_image(data, [2, 2, 2]);

        let clamped = clamp(&img, -3.0, 3.0);
        let result = clamped.to_f32();
        let result_slice = result.as_slice_memory_order().unwrap();

        for &v in result_slice {
            assert!(v >= -3.0, "Value {} should be >= -3", v);
            assert!(v <= 3.0, "Value {} should be <= 3", v);
        }
    }

    #[test]
    fn test_intensity_preserves_shape() {
        let data = vec![1.0; 24]; // 2x3x4 = 24
        let img = create_test_image(data, [2, 3, 4]);

        let z_norm = z_normalization(&img);
        assert_eq!(z_norm.shape(), &[2, 3, 4]);

        let rescaled = rescale_intensity(&img, 0.0, 1.0);
        assert_eq!(rescaled.shape(), &[2, 3, 4]);

        let clamped = clamp(&img, 0.0, 2.0);
        assert_eq!(clamped.shape(), &[2, 3, 4]);
    }
}
