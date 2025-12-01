//! SIMD-optimized kernels for transform operations.
//!
//! These kernels use the `wide` crate for portable SIMD across platforms.
//! Each function has both SIMD and scalar fallback paths.

use wide::f32x8;

/// SIMD width for f32 operations (8 floats = 256 bits = AVX).
pub const SIMD_WIDTH: usize = 8;

/// Apply linear transform: output = input * scale + offset
///
/// Uses SIMD for bulk of data, scalar for remainder.
#[inline]
pub fn linear_transform_f32(input: &[f32], output: &mut [f32], scale: f32, offset: f32) {
    assert_eq!(input.len(), output.len());
    let len = input.len();

    // SIMD constants
    let scale_vec = f32x8::splat(scale);
    let offset_vec = f32x8::splat(offset);

    // Process 8 elements at a time
    let chunks = len / SIMD_WIDTH;
    let remainder = len % SIMD_WIDTH;

    for i in 0..chunks {
        let base = i * SIMD_WIDTH;
        let in_vec = f32x8::from(&input[base..base + SIMD_WIDTH]);
        let out_vec = in_vec * scale_vec + offset_vec;

        // Store result
        let out_arr: [f32; 8] = out_vec.into();
        output[base..base + SIMD_WIDTH].copy_from_slice(&out_arr);
    }

    // Handle remainder with scalar
    let base = chunks * SIMD_WIDTH;
    for i in 0..remainder {
        output[base + i] = input[base + i] * scale + offset;
    }
}

/// Apply linear transform with clamping: output = clamp(input * scale + offset, min, max)
#[inline]
pub fn linear_transform_clamp_f32(
    input: &[f32],
    output: &mut [f32],
    scale: f32,
    offset: f32,
    min: f32,
    max: f32,
) {
    assert_eq!(input.len(), output.len());
    let len = input.len();

    let scale_vec = f32x8::splat(scale);
    let offset_vec = f32x8::splat(offset);
    let min_vec = f32x8::splat(min);
    let max_vec = f32x8::splat(max);

    let chunks = len / SIMD_WIDTH;
    let remainder = len % SIMD_WIDTH;

    for i in 0..chunks {
        let base = i * SIMD_WIDTH;
        let in_vec = f32x8::from(&input[base..base + SIMD_WIDTH]);
        let out_vec = (in_vec * scale_vec + offset_vec).max(min_vec).min(max_vec);

        let out_arr: [f32; 8] = out_vec.into();
        output[base..base + SIMD_WIDTH].copy_from_slice(&out_arr);
    }

    let base = chunks * SIMD_WIDTH;
    for i in 0..remainder {
        output[base + i] = (input[base + i] * scale + offset).clamp(min, max);
    }
}

/// Compute sum and sum of squares for mean/variance calculation.
///
/// Returns (sum, sum_sq, count).
#[inline]
pub fn sum_and_sum_sq_f32(input: &[f32]) -> (f64, f64, usize) {
    let len = input.len();
    let chunks = len / SIMD_WIDTH;
    let remainder = len % SIMD_WIDTH;

    // Use f64 accumulators for precision
    let mut sum_acc = [0.0f64; SIMD_WIDTH];
    let mut sq_acc = [0.0f64; SIMD_WIDTH];

    for i in 0..chunks {
        let base = i * SIMD_WIDTH;
        let in_vec = f32x8::from(&input[base..base + SIMD_WIDTH]);
        let sq_vec = in_vec * in_vec;

        let in_arr: [f32; 8] = in_vec.into();
        let sq_arr: [f32; 8] = sq_vec.into();

        for j in 0..SIMD_WIDTH {
            sum_acc[j] += in_arr[j] as f64;
            sq_acc[j] += sq_arr[j] as f64;
        }
    }

    // Sum across SIMD lanes
    let mut sum: f64 = sum_acc.iter().sum();
    let mut sum_sq: f64 = sq_acc.iter().sum();

    // Handle remainder
    let base = chunks * SIMD_WIDTH;
    for i in 0..remainder {
        let v = input[base + i] as f64;
        sum += v;
        sum_sq += v * v;
    }

    (sum, sum_sq, len)
}

/// Compute min and max values.
#[inline]
pub fn minmax_f32(input: &[f32]) -> (f32, f32) {
    if input.is_empty() {
        return (f32::INFINITY, f32::NEG_INFINITY);
    }

    let len = input.len();
    let chunks = len / SIMD_WIDTH;
    let remainder = len % SIMD_WIDTH;

    // Initialize with first element
    let mut min_vec = f32x8::splat(input[0]);
    let mut max_vec = f32x8::splat(input[0]);

    for i in 0..chunks {
        let base = i * SIMD_WIDTH;
        let in_vec = f32x8::from(&input[base..base + SIMD_WIDTH]);
        min_vec = min_vec.min(in_vec);
        max_vec = max_vec.max(in_vec);
    }

    // Reduce SIMD lanes
    let min_arr: [f32; 8] = min_vec.into();
    let max_arr: [f32; 8] = max_vec.into();

    let mut min_val = min_arr[0];
    let mut max_val = max_arr[0];
    for i in 1..SIMD_WIDTH {
        min_val = min_val.min(min_arr[i]);
        max_val = max_val.max(max_arr[i]);
    }

    // Handle remainder
    let base = chunks * SIMD_WIDTH;
    for i in 0..remainder {
        let v = input[base + i];
        min_val = min_val.min(v);
        max_val = max_val.max(v);
    }

    (min_val, max_val)
}

/// Clamp values in-place.
#[inline]
pub fn clamp_f32_inplace(data: &mut [f32], min: f32, max: f32) {
    let len = data.len();
    let chunks = len / SIMD_WIDTH;
    let remainder = len % SIMD_WIDTH;

    let min_vec = f32x8::splat(min);
    let max_vec = f32x8::splat(max);

    for i in 0..chunks {
        let base = i * SIMD_WIDTH;
        let in_vec = f32x8::from(&data[base..base + SIMD_WIDTH]);
        let out_vec = in_vec.max(min_vec).min(max_vec);

        let out_arr: [f32; 8] = out_vec.into();
        data[base..base + SIMD_WIDTH].copy_from_slice(&out_arr);
    }

    let base = chunks * SIMD_WIDTH;
    for i in 0..remainder {
        data[base + i] = data[base + i].clamp(min, max);
    }
}

/// Trilinear interpolation for a batch of output voxels.
///
/// This processes multiple output coordinates at once using SIMD.
///
/// # Arguments
/// * `input` - Input volume data in row-major order (z, y, x)
/// * `shape` - Input shape [z, y, x]
/// * `coords` - Output coordinates as (z, y, x) tuples
/// * `output` - Output buffer
#[inline]
#[allow(clippy::similar_names)]
pub fn trilinear_interp_batch_f32(
    input: &[f32],
    shape: [usize; 3],
    coords: &[(f32, f32, f32)],
    output: &mut [f32],
) {
    assert_eq!(coords.len(), output.len());

    let [sz, sy, sx] = shape;
    let stride_z = sy * sx;
    let stride_y = sx;

    for (i, &(z, y, x)) in coords.iter().enumerate() {
        // Handle out-of-bounds with zero padding
        if z < 0.0
            || y < 0.0
            || x < 0.0
            || z >= (sz - 1) as f32
            || y >= (sy - 1) as f32
            || x >= (sx - 1) as f32
        {
            output[i] = 0.0;
            continue;
        }

        // Integer indices
        let z0 = z as usize;
        let y0 = y as usize;
        let x0 = x as usize;
        let z1 = z0 + 1;
        let y1 = y0 + 1;
        let x1 = x0 + 1;

        // Fractional parts
        let fz = z - z0 as f32;
        let fy = y - y0 as f32;
        let fx = x - x0 as f32;

        // Fetch 8 corner values
        let c000 = input[z0 * stride_z + y0 * stride_y + x0];
        let c001 = input[z0 * stride_z + y0 * stride_y + x1];
        let c010 = input[z0 * stride_z + y1 * stride_y + x0];
        let c011 = input[z0 * stride_z + y1 * stride_y + x1];
        let c100 = input[z1 * stride_z + y0 * stride_y + x0];
        let c101 = input[z1 * stride_z + y0 * stride_y + x1];
        let c110 = input[z1 * stride_z + y1 * stride_y + x0];
        let c111 = input[z1 * stride_z + y1 * stride_y + x1];

        // Trilinear interpolation
        let c00 = c000 * (1.0 - fx) + c001 * fx;
        let c01 = c010 * (1.0 - fx) + c011 * fx;
        let c10 = c100 * (1.0 - fx) + c101 * fx;
        let c11 = c110 * (1.0 - fx) + c111 * fx;

        let c0 = c00 * (1.0 - fy) + c01 * fy;
        let c1 = c10 * (1.0 - fy) + c11 * fy;

        output[i] = c0 * (1.0 - fz) + c1 * fz;
    }
}

/// SIMD-optimized trilinear interpolation along X axis for a single row.
///
/// Processes 8 output X values at a time. All output voxels share the same
/// Y and Z coordinates, enabling efficient SIMD gather and interpolation.
///
/// # Arguments
/// * `src` - Input volume slice
/// * `stride_z` - Stride between Z slices
/// * `stride_y` - Stride between Y rows
/// * `z0`, `z1` - Z indices for interpolation
/// * `y0`, `y1` - Y indices for interpolation
/// * `zf`, `yf` - Z and Y fractional weights
/// * `x_params` - Precomputed X interpolation parameters (idx0, idx1, frac)
/// * `out_row` - Output row buffer
#[inline]
#[allow(clippy::too_many_arguments, clippy::similar_names)]
pub fn trilinear_row_simd(
    src: &[f32],
    stride_z: usize,
    stride_y: usize,
    z0: usize,
    z1: usize,
    y0: usize,
    y1: usize,
    zf: f32,
    yf: f32,
    x_idx0: &[usize],
    x_idx1: &[usize],
    x_frac: &[f32],
    out_row: &mut [f32],
) {
    let nw = out_row.len();
    let zf_inv = 1.0 - zf;
    let yf_inv = 1.0 - yf;

    // Precompute base offsets for the 4 corner rows
    let off_z0_y0 = z0 * stride_z + y0 * stride_y;
    let off_z0_y1 = z0 * stride_z + y1 * stride_y;
    let off_z1_y0 = z1 * stride_z + y0 * stride_y;
    let off_z1_y1 = z1 * stride_z + y1 * stride_y;

    // Precompute Y and Z interpolation weights
    let w00 = zf_inv * yf_inv; // z0, y0
    let w01 = zf_inv * yf; // z0, y1
    let w10 = zf * yf_inv; // z1, y0
    let w11 = zf * yf; // z1, y1

    let w00_vec = f32x8::splat(w00);
    let w01_vec = f32x8::splat(w01);
    let w10_vec = f32x8::splat(w10);
    let w11_vec = f32x8::splat(w11);

    // Process 8 X values at a time
    let chunks = nw / SIMD_WIDTH;

    for chunk_i in 0..chunks {
        let base = chunk_i * SIMD_WIDTH;

        // Gather values for x0 and x1 indices (8 pairs)
        // For each of the 4 corner rows, we need values at x0[i] and x1[i]
        let mut c000 = [0.0f32; 8];
        let mut c001 = [0.0f32; 8];
        let mut c010 = [0.0f32; 8];
        let mut c011 = [0.0f32; 8];
        let mut c100 = [0.0f32; 8];
        let mut c101 = [0.0f32; 8];
        let mut c110 = [0.0f32; 8];
        let mut c111 = [0.0f32; 8];
        let mut xf = [0.0f32; 8];

        for i in 0..SIMD_WIDTH {
            let w = base + i;
            let x0 = x_idx0[w];
            let x1 = x_idx1[w];
            xf[i] = x_frac[w];

            c000[i] = src[off_z0_y0 + x0];
            c001[i] = src[off_z0_y0 + x1];
            c010[i] = src[off_z0_y1 + x0];
            c011[i] = src[off_z0_y1 + x1];
            c100[i] = src[off_z1_y0 + x0];
            c101[i] = src[off_z1_y0 + x1];
            c110[i] = src[off_z1_y1 + x0];
            c111[i] = src[off_z1_y1 + x1];
        }

        // Convert to SIMD vectors
        let c000_v = f32x8::from(c000);
        let c001_v = f32x8::from(c001);
        let c010_v = f32x8::from(c010);
        let c011_v = f32x8::from(c011);
        let c100_v = f32x8::from(c100);
        let c101_v = f32x8::from(c101);
        let c110_v = f32x8::from(c110);
        let c111_v = f32x8::from(c111);
        let xf_v = f32x8::from(xf);
        let xf_inv_v = f32x8::splat(1.0) - xf_v;

        // Interpolate along X for each corner row
        let c00 = c000_v * xf_inv_v + c001_v * xf_v; // z0, y0
        let c01 = c010_v * xf_inv_v + c011_v * xf_v; // z0, y1
        let c10 = c100_v * xf_inv_v + c101_v * xf_v; // z1, y0
        let c11 = c110_v * xf_inv_v + c111_v * xf_v; // z1, y1

        // Combine Y and Z interpolation in one step
        let result = c00 * w00_vec + c01 * w01_vec + c10 * w10_vec + c11 * w11_vec;

        // Store result
        let result_arr: [f32; 8] = result.into();
        out_row[base..base + SIMD_WIDTH].copy_from_slice(&result_arr);
    }

    // Handle remainder with scalar code
    let base = chunks * SIMD_WIDTH;
    for w in base..nw {
        let x0 = x_idx0[w];
        let x1 = x_idx1[w];
        let xf = x_frac[w];
        let xf_inv = 1.0 - xf;

        let c000 = src[off_z0_y0 + x0];
        let c001 = src[off_z0_y0 + x1];
        let c010 = src[off_z0_y1 + x0];
        let c011 = src[off_z0_y1 + x1];
        let c100 = src[off_z1_y0 + x0];
        let c101 = src[off_z1_y0 + x1];
        let c110 = src[off_z1_y1 + x0];
        let c111 = src[off_z1_y1 + x1];

        let c00 = c000 * xf_inv + c001 * xf;
        let c01 = c010 * xf_inv + c011 * xf;
        let c10 = c100 * xf_inv + c101 * xf;
        let c11 = c110 * xf_inv + c111 * xf;

        out_row[w] = c00 * w00 + c01 * w01 + c10 * w10 + c11 * w11;
    }
}

/// Interpolate along a single dimension using SIMD.
///
/// Performs linear interpolation between two rows/slices.
#[inline]
pub fn lerp_1d_simd(src0: &[f32], src1: &[f32], frac: f32, output: &mut [f32]) {
    debug_assert_eq!(src0.len(), src1.len());
    debug_assert_eq!(src0.len(), output.len());

    let len = output.len();
    let chunks = len / SIMD_WIDTH;

    let f_vec = f32x8::splat(frac);
    let f_inv_vec = f32x8::splat(1.0 - frac);

    for chunk_i in 0..chunks {
        let base = chunk_i * SIMD_WIDTH;
        let v0 = f32x8::from(&src0[base..base + SIMD_WIDTH]);
        let v1 = f32x8::from(&src1[base..base + SIMD_WIDTH]);
        let result = v0 * f_inv_vec + v1 * f_vec;
        let arr: [f32; 8] = result.into();
        output[base..base + SIMD_WIDTH].copy_from_slice(&arr);
    }

    // Scalar remainder
    let base = chunks * SIMD_WIDTH;
    let f_inv = 1.0 - frac;
    for i in base..len {
        output[i] = src0[i] * f_inv + src1[i] * frac;
    }
}

/// Parallel SIMD linear transform using rayon.
///
/// Splits work across threads, each thread uses SIMD.
pub fn parallel_linear_transform_f32(input: &[f32], output: &mut [f32], scale: f32, offset: f32) {
    use rayon::prelude::*;

    const CHUNK_SIZE: usize = 8192; // Process 32KB chunks per thread

    output
        .par_chunks_mut(CHUNK_SIZE)
        .zip(input.par_chunks(CHUNK_SIZE))
        .for_each(|(out_chunk, in_chunk)| {
            linear_transform_f32(in_chunk, out_chunk, scale, offset);
        });
}

/// Parallel SIMD linear transform with clamping.
pub fn parallel_linear_transform_clamp_f32(
    input: &[f32],
    output: &mut [f32],
    scale: f32,
    offset: f32,
    min: f32,
    max: f32,
) {
    use rayon::prelude::*;

    const CHUNK_SIZE: usize = 8192;

    output
        .par_chunks_mut(CHUNK_SIZE)
        .zip(input.par_chunks(CHUNK_SIZE))
        .for_each(|(out_chunk, in_chunk)| {
            linear_transform_clamp_f32(in_chunk, out_chunk, scale, offset, min, max);
        });
}

/// Parallel sum and sum of squares using rayon.
pub fn parallel_sum_and_sum_sq_f32(input: &[f32]) -> (f64, f64, usize) {
    use rayon::prelude::*;

    const CHUNK_SIZE: usize = 16384;

    let (sum, sum_sq): (f64, f64) = input
        .par_chunks(CHUNK_SIZE)
        .map(|chunk| {
            let (s, sq, _) = sum_and_sum_sq_f32(chunk);
            (s, sq)
        })
        .reduce(|| (0.0, 0.0), |(s1, sq1), (s2, sq2)| (s1 + s2, sq1 + sq2));

    (sum, sum_sq, input.len())
}

/// Parallel min/max using rayon.
pub fn parallel_minmax_f32(input: &[f32]) -> (f32, f32) {
    use rayon::prelude::*;

    const CHUNK_SIZE: usize = 16384;

    input.par_chunks(CHUNK_SIZE).map(minmax_f32).reduce(
        || (f32::INFINITY, f32::NEG_INFINITY),
        |(min1, max1), (min2, max2)| (min1.min(min2), max1.max(max2)),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_transform() {
        let input: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let mut output = vec![0.0; 100];

        linear_transform_f32(&input, &mut output, 2.0, 1.0);

        for i in 0..100 {
            assert!((output[i] - (input[i] * 2.0 + 1.0)).abs() < 1e-6);
        }
    }

    #[test]
    fn test_trilinear_row_simd() {
        // Create a 4x4x4 volume with predictable values
        let mut src = vec![0.0f32; 4 * 4 * 4];
        for z in 0..4 {
            for y in 0..4 {
                for x in 0..4 {
                    src[z * 16 + y * 4 + x] = (z * 100 + y * 10 + x) as f32;
                }
            }
        }

        let stride_z = 16;
        let stride_y = 4;

        // Test interpolation at z=0.5, y=0.5 for various x
        let x_idx0 = vec![0, 0, 1, 1, 2, 2, 3, 3];
        let x_idx1 = vec![1, 1, 2, 2, 3, 3, 3, 3];
        let x_frac = vec![0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5];
        let mut output = vec![0.0f32; 8];

        trilinear_row_simd(
            &src,
            stride_z,
            stride_y,
            0,
            1, // z0=0, z1=1
            0,
            1, // y0=0, y1=1
            0.5,
            0.5, // zf=0.5, yf=0.5
            &x_idx0,
            &x_idx1,
            &x_frac,
            &mut output,
        );

        // Verify interpolation produces valid values
        for &v in &output {
            assert!(v >= 0.0 && v <= 400.0, "Output {} out of expected range", v);
        }

        // First value: x=0, no x interp, z=0.5, y=0.5
        // c000=0, c010=10, c100=100, c110=110
        // Expected: 0.25*0 + 0.25*10 + 0.25*100 + 0.25*110 = 55
        assert!(
            (output[0] - 55.0).abs() < 1e-3,
            "Expected 55.0, got {}",
            output[0]
        );
    }

    #[test]
    fn test_lerp_1d_simd() {
        let src0: Vec<f32> = (0..32).map(|i| i as f32).collect();
        let src1: Vec<f32> = (0..32).map(|i| (i + 100) as f32).collect();
        let mut output = vec![0.0f32; 32];

        // Interpolate with frac=0.25 (25% towards src1)
        lerp_1d_simd(&src0, &src1, 0.25, &mut output);

        for i in 0..32 {
            let expected = src0[i] * 0.75 + src1[i] * 0.25;
            assert!(
                (output[i] - expected).abs() < 1e-5,
                "At {}: expected {}, got {}",
                i,
                expected,
                output[i]
            );
        }
    }

    #[test]
    fn test_lerp_1d_simd_remainder() {
        // Test with non-multiple-of-8 length to exercise scalar remainder
        let src0: Vec<f32> = (0..13).map(|i| i as f32).collect();
        let src1: Vec<f32> = (0..13).map(|i| (i * 2) as f32).collect();
        let mut output = vec![0.0f32; 13];

        lerp_1d_simd(&src0, &src1, 0.5, &mut output);

        for i in 0..13 {
            let expected = (src0[i] + src1[i]) / 2.0;
            assert!(
                (output[i] - expected).abs() < 1e-5,
                "At {}: expected {}, got {}",
                i,
                expected,
                output[i]
            );
        }
    }

    #[test]
    fn test_linear_transform_clamp() {
        let input: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let mut output = vec![0.0; 100];

        linear_transform_clamp_f32(&input, &mut output, 1.0, 0.0, 10.0, 50.0);

        for i in 0..100 {
            let expected = (input[i]).clamp(10.0, 50.0);
            assert!((output[i] - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn test_sum_and_sum_sq() {
        let input: Vec<f32> = (1..=100).map(|i| i as f32).collect();
        let (sum, sum_sq, count) = sum_and_sum_sq_f32(&input);

        // Sum of 1..=100 = 5050
        assert!((sum - 5050.0).abs() < 1e-6);
        assert_eq!(count, 100);

        // Sum of squares = 338350
        let expected_sq: f64 = (1..=100).map(|i| (i * i) as f64).sum();
        assert!((sum_sq - expected_sq).abs() < 1e-3);
    }

    #[test]
    fn test_minmax() {
        let input: Vec<f32> = vec![-5.0, 3.0, 100.0, -200.0, 50.0, 0.0];
        let (min, max) = minmax_f32(&input);

        assert_eq!(min, -200.0);
        assert_eq!(max, 100.0);
    }
}
