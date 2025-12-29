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
        // Valid range is [0, size-1] for each dimension
        if z < 0.0
            || y < 0.0
            || x < 0.0
            || z > (sz - 1) as f32
            || y > (sy - 1) as f32
            || x > (sx - 1) as f32
        {
            output[i] = 0.0;
            continue;
        }

        // Integer indices - clamp upper indices to handle boundary exactly at size-1
        let z0 = z as usize;
        let y0 = y as usize;
        let x0 = x as usize;
        let z1 = (z0 + 1).min(sz - 1);
        let y1 = (y0 + 1).min(sy - 1);
        let x1 = (x0 + 1).min(sx - 1);

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
#[allow(
    clippy::too_many_arguments,
    clippy::similar_names,
    clippy::needless_range_loop
)]
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

    // 8192 f32 values = 32KB per chunk, sized to fit in L1 cache (typically 32-48KB)
    // This balances parallelism overhead against cache efficiency
    const CHUNK_SIZE: usize = 8192;

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

    // 8192 f32 values = 32KB per chunk (L1 cache optimal)
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

    // 16384 f32 values = 64KB per chunk (L2 cache optimal for reduction operations)
    // Larger than linear transform because reduction has lower memory bandwidth needs
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

// =============================================================================
// OPTIMIZED F-ORDER TRILINEAR RESAMPLING
// =============================================================================
//
// These functions work directly with F-order (column-major) data to avoid
// expensive memory layout conversions. F-order means the first index varies
// fastest in memory, i.e., for shape [X, Y, Z], elements at (x, y, z) and
// (x+1, y, z) are adjacent in memory.

/// Precomputed interpolation weights for a single axis.
/// Stores both indices and weights for efficient SIMD processing.
#[derive(Clone)]
pub struct AxisInterpWeights {
    /// Lower indices for each output position
    pub idx0: Vec<usize>,
    /// Upper indices for each output position
    pub idx1: Vec<usize>,
    /// Interpolation weights (fraction towards idx1)
    pub frac: Vec<f32>,
    /// Inverse weights (1 - frac), precomputed for SIMD
    pub frac_inv: Vec<f32>,
}

impl AxisInterpWeights {
    /// Create interpolation weights for resampling from old_size to new_size.
    ///
    /// # Panics
    /// Panics if `old_size` is 0. This is an invariant violation.
    pub fn new(new_size: usize, old_size: usize) -> Self {
        assert!(old_size > 0, "old_size must be > 0, got {}", old_size);

        // Handle edge case: if new_size is 0, return empty weights
        if new_size == 0 {
            return Self {
                idx0: Vec::new(),
                idx1: Vec::new(),
                frac: Vec::new(),
                frac_inv: Vec::new(),
            };
        }

        // Scale factor: map [0, new_size-1] to [0, old_size-1]
        // When new_size == 1, all output maps to center of input (scale = 0)
        let scale = if new_size > 1 && old_size > 1 {
            (old_size - 1) as f32 / (new_size - 1) as f32
        } else {
            0.0
        };

        let mut idx0 = Vec::with_capacity(new_size);
        let mut idx1 = Vec::with_capacity(new_size);
        let mut frac = Vec::with_capacity(new_size);
        let mut frac_inv = Vec::with_capacity(new_size);

        for i in 0..new_size {
            let pos = i as f32 * scale;
            let i0 = (pos.floor() as usize).min(old_size - 1);
            let i1 = (i0 + 1).min(old_size - 1);
            let f = pos - i0 as f32;

            idx0.push(i0);
            idx1.push(i1);
            frac.push(f);
            frac_inv.push(1.0 - f);
        }

        Self {
            idx0,
            idx1,
            frac,
            frac_inv,
        }
    }
}

/// F-order optimized trilinear resampling with SIMD.
///
/// Works directly with F-order data (X varies fastest), avoiding layout conversions.
/// Uses tiled processing for better cache utilization on large volumes.
///
/// # Arguments
/// * `src` - Source data in F-order [X, Y, Z]
/// * `src_shape` - Source shape [sx, sy, sz]
/// * `dst_shape` - Destination shape [dx, dy, dz]
///
/// # Returns
/// Resampled data in F-order
#[allow(clippy::similar_names)]
pub fn trilinear_resample_forder(
    src: &[f32],
    src_shape: [usize; 3],
    dst_shape: [usize; 3],
) -> Vec<f32> {
    use crate::pipeline::acquire_buffer;
    use rayon::prelude::*;

    let [sx, sy, sz] = src_shape;
    let [dx, dy, dz] = dst_shape;

    // Precompute interpolation weights for each axis
    let x_weights = AxisInterpWeights::new(dx, sx);
    let y_weights = AxisInterpWeights::new(dy, sy);
    let z_weights = AxisInterpWeights::new(dz, sz);

    // F-order strides: X varies fastest
    let src_stride_y = sx;
    let src_stride_z = sx * sy;

    let dst_stride_y = dx;
    let dst_stride_z = dx * dy;

    let total_voxels = dx * dy * dz;
    let mut dst: Vec<f32> = acquire_buffer(total_voxels);

    // Process in Z-slices for parallelization
    // Each thread processes one or more Z-slices
    dst.par_chunks_mut(dst_stride_z)
        .enumerate()
        .for_each(|(z_dst, z_slice)| {
            let z0 = z_weights.idx0[z_dst];
            let z1 = z_weights.idx1[z_dst];
            let wz = z_weights.frac[z_dst];
            let wz_inv = z_weights.frac_inv[z_dst];

            // Base offsets for the two Z planes
            let z0_base = z0 * src_stride_z;
            let z1_base = z1 * src_stride_z;

            for y_dst in 0..dy {
                let y0 = y_weights.idx0[y_dst];
                let y1 = y_weights.idx1[y_dst];
                let wy = y_weights.frac[y_dst];
                let wy_inv = y_weights.frac_inv[y_dst];

                // Precompute combined weights for the 4 Y-Z corner combinations
                let w00 = wz_inv * wy_inv; // z0, y0
                let w01 = wz_inv * wy; // z0, y1
                let w10 = wz * wy_inv; // z1, y0
                let w11 = wz * wy; // z1, y1

                // Base offsets for the 4 source rows
                let off_z0_y0 = z0_base + y0 * src_stride_y;
                let off_z0_y1 = z0_base + y1 * src_stride_y;
                let off_z1_y0 = z1_base + y0 * src_stride_y;
                let off_z1_y1 = z1_base + y1 * src_stride_y;

                let dst_row = &mut z_slice[y_dst * dst_stride_y..(y_dst + 1) * dst_stride_y];

                // SIMD processing along X axis
                trilinear_x_simd_forder(
                    src, &x_weights, off_z0_y0, off_z0_y1, off_z1_y0, off_z1_y1, w00, w01, w10,
                    w11, dst_row,
                );
            }
        });

    dst
}

/// SIMD-optimized X-axis interpolation for F-order data.
///
/// Processes 8 output X values at a time using AVX (f32x8).
#[inline]
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
fn trilinear_x_simd_forder(
    src: &[f32],
    x_weights: &AxisInterpWeights,
    off_z0_y0: usize,
    off_z0_y1: usize,
    off_z1_y0: usize,
    off_z1_y1: usize,
    w00: f32,
    w01: f32,
    w10: f32,
    w11: f32,
    dst_row: &mut [f32],
) {
    let dx = dst_row.len();
    let chunks = dx / SIMD_WIDTH;

    // SIMD weight vectors
    let w00_v = f32x8::splat(w00);
    let w01_v = f32x8::splat(w01);
    let w10_v = f32x8::splat(w10);
    let w11_v = f32x8::splat(w11);

    for chunk_i in 0..chunks {
        let base = chunk_i * SIMD_WIDTH;

        let mut v_z0_y0_0 = [0.0f32; 8];
        let mut v_z0_y0_1 = [0.0f32; 8];
        let mut v_z0_y1_0 = [0.0f32; 8];
        let mut v_z0_y1_1 = [0.0f32; 8];
        let mut v_z1_y0_0 = [0.0f32; 8];
        let mut v_z1_y0_1 = [0.0f32; 8];
        let mut v_z1_y1_0 = [0.0f32; 8];
        let mut v_z1_y1_1 = [0.0f32; 8];
        let mut xf = [0.0f32; 8];
        let mut xf_inv = [0.0f32; 8];

        for i in 0..SIMD_WIDTH {
            let xi = base + i;
            let x0 = x_weights.idx0[xi];
            let x1 = x_weights.idx1[xi];
            xf[i] = x_weights.frac[xi];
            xf_inv[i] = x_weights.frac_inv[xi];

            v_z0_y0_0[i] = src[off_z0_y0 + x0];
            v_z0_y0_1[i] = src[off_z0_y0 + x1];
            v_z0_y1_0[i] = src[off_z0_y1 + x0];
            v_z0_y1_1[i] = src[off_z0_y1 + x1];
            v_z1_y0_0[i] = src[off_z1_y0 + x0];
            v_z1_y0_1[i] = src[off_z1_y0 + x1];
            v_z1_y1_0[i] = src[off_z1_y1 + x0];
            v_z1_y1_1[i] = src[off_z1_y1 + x1];
        }

        let xf_v = f32x8::from(xf);
        let xf_inv_v = f32x8::from(xf_inv);

        let c_z0_y0 = f32x8::from(v_z0_y0_0) * xf_inv_v + f32x8::from(v_z0_y0_1) * xf_v;
        let c_z0_y1 = f32x8::from(v_z0_y1_0) * xf_inv_v + f32x8::from(v_z0_y1_1) * xf_v;
        let c_z1_y0 = f32x8::from(v_z1_y0_0) * xf_inv_v + f32x8::from(v_z1_y0_1) * xf_v;
        let c_z1_y1 = f32x8::from(v_z1_y1_0) * xf_inv_v + f32x8::from(v_z1_y1_1) * xf_v;

        let result = c_z0_y0 * w00_v + c_z0_y1 * w01_v + c_z1_y0 * w10_v + c_z1_y1 * w11_v;

        let result_arr: [f32; 8] = result.into();
        dst_row[base..base + SIMD_WIDTH].copy_from_slice(&result_arr);
    }

    // Scalar remainder
    let base = chunks * SIMD_WIDTH;
    for xi in base..dx {
        let x0 = x_weights.idx0[xi];
        let x1 = x_weights.idx1[xi];
        let xf = x_weights.frac[xi];
        let xf_inv = x_weights.frac_inv[xi];

        let c_z0_y0 = src[off_z0_y0 + x0] * xf_inv + src[off_z0_y0 + x1] * xf;
        let c_z0_y1 = src[off_z0_y1 + x0] * xf_inv + src[off_z0_y1 + x1] * xf;
        let c_z1_y0 = src[off_z1_y0 + x0] * xf_inv + src[off_z1_y0 + x1] * xf;
        let c_z1_y1 = src[off_z1_y1 + x0] * xf_inv + src[off_z1_y1 + x1] * xf;

        dst_row[xi] = c_z0_y0 * w00 + c_z0_y1 * w01 + c_z1_y0 * w10 + c_z1_y1 * w11;
    }
}

/// Separable trilinear resampling optimized for F-order data.
///
/// Uses 3-pass approach (X, Y, Z) for better cache locality on large volumes.
/// Each pass processes data along one axis, keeping memory accesses sequential.
#[allow(clippy::similar_names)]
pub fn trilinear_resample_forder_separable(
    src: &[f32],
    src_shape: [usize; 3],
    dst_shape: [usize; 3],
) -> Vec<f32> {
    use crate::pipeline::acquire_buffer;
    use rayon::prelude::*;

    let [sx, sy, sz] = src_shape;
    let [dx, dy, dz] = dst_shape;

    // Pass 1: Resample along X (sx, sy, sz) -> (dx, sy, sz)
    let x_weights = AxisInterpWeights::new(dx, sx);
    let temp1_size = dx * sy * sz;
    let mut temp1: Vec<f32> = acquire_buffer(temp1_size);

    // F-order: X varies fastest, so we process YZ slices in parallel
    temp1
        .par_chunks_mut(dx)
        .enumerate()
        .for_each(|(yz_idx, dst_row)| {
            let y = yz_idx % sy;
            let z = yz_idx / sy;
            let src_row_offset = y * sx + z * sx * sy;

            resample_1d_simd(
                &src[src_row_offset..src_row_offset + sx],
                &x_weights,
                dst_row,
            );
        });

    // Pass 2: Resample along Y (dx, sy, sz) -> (dx, dy, sz)
    let y_weights = AxisInterpWeights::new(dy, sy);
    let temp2_size = dx * dy * sz;
    let mut temp2: Vec<f32> = acquire_buffer(temp2_size);

    // For Y resampling in F-order, we need to handle non-contiguous access
    // Process each (X, Z) fiber in parallel
    let temp1_ref = &temp1;
    temp2
        .par_chunks_mut(dx * dy)
        .enumerate()
        .for_each(|(z, z_slice)| {
            let src_z_base = z * dx * sy;
            for x in 0..dx {
                for y_dst in 0..dy {
                    let y0 = y_weights.idx0[y_dst];
                    let y1 = y_weights.idx1[y_dst];
                    let f = y_weights.frac[y_dst];
                    let f_inv = y_weights.frac_inv[y_dst];

                    let v0 = temp1_ref[src_z_base + y0 * dx + x];
                    let v1 = temp1_ref[src_z_base + y1 * dx + x];
                    z_slice[y_dst * dx + x] = v0 * f_inv + v1 * f;
                }
            }
        });

    drop(temp1);

    // Pass 3: Resample along Z (dx, dy, sz) -> (dx, dy, dz)
    let z_weights = AxisInterpWeights::new(dz, sz);
    let dst_size = dx * dy * dz;
    let mut dst: Vec<f32> = acquire_buffer(dst_size);

    let xy_size = dx * dy;
    let temp2_ref = &temp2;

    dst.par_chunks_mut(xy_size)
        .enumerate()
        .for_each(|(z_dst, xy_slice)| {
            let z0 = z_weights.idx0[z_dst];
            let z1 = z_weights.idx1[z_dst];
            let f = z_weights.frac[z_dst];

            let src0 = &temp2_ref[z0 * xy_size..(z0 + 1) * xy_size];
            let src1 = &temp2_ref[z1 * xy_size..(z1 + 1) * xy_size];

            lerp_1d_simd(src0, src1, f, xy_slice);
        });

    dst
}

/// 1D SIMD resampling using precomputed weights.
#[inline]
fn resample_1d_simd(src: &[f32], weights: &AxisInterpWeights, dst: &mut [f32]) {
    let n = dst.len();
    let chunks = n / SIMD_WIDTH;

    for chunk_i in 0..chunks {
        let base = chunk_i * SIMD_WIDTH;

        let mut v0 = [0.0f32; 8];
        let mut v1 = [0.0f32; 8];
        let mut f = [0.0f32; 8];
        let mut f_inv = [0.0f32; 8];

        for i in 0..SIMD_WIDTH {
            let idx = base + i;
            v0[i] = src[weights.idx0[idx]];
            v1[i] = src[weights.idx1[idx]];
            f[i] = weights.frac[idx];
            f_inv[i] = weights.frac_inv[idx];
        }

        let v0_v = f32x8::from(v0);
        let v1_v = f32x8::from(v1);
        let f_v = f32x8::from(f);
        let f_inv_v = f32x8::from(f_inv);

        let result = v0_v * f_inv_v + v1_v * f_v;
        let result_arr: [f32; 8] = result.into();
        dst[base..base + SIMD_WIDTH].copy_from_slice(&result_arr);
    }

    // Scalar remainder
    for i in (chunks * SIMD_WIDTH)..n {
        let v0 = src[weights.idx0[i]];
        let v1 = src[weights.idx1[i]];
        dst[i] = v0 * weights.frac_inv[i] + v1 * weights.frac[i];
    }
}

/// Choose optimal resampling strategy based on volume size.
///
/// Automatically selects specialized kernels for common cases:
/// - 2x upsampling: Uses optimized kernel with contiguous memory access
/// - General case: Uses direct trilinear with scattered gathers
pub fn trilinear_resample_forder_adaptive(
    src: &[f32],
    src_shape: [usize; 3],
    dst_shape: [usize; 3],
) -> Vec<f32> {
    // Check for exact 2x upsampling (very common case)
    let is_2x_upsample = dst_shape[0] == 2 * src_shape[0] - 1
        && dst_shape[1] == 2 * src_shape[1] - 1
        && dst_shape[2] == 2 * src_shape[2] - 1;

    // Also check for approximate 2x (allowing for rounding)
    let is_approx_2x = (dst_shape[0] as f32 / src_shape[0] as f32 - 2.0).abs() < 0.1
        && (dst_shape[1] as f32 / src_shape[1] as f32 - 2.0).abs() < 0.1
        && (dst_shape[2] as f32 / src_shape[2] as f32 - 2.0).abs() < 0.1;

    if is_2x_upsample || is_approx_2x {
        trilinear_upsample_2x_forder(src, src_shape, dst_shape)
    } else {
        trilinear_resample_forder(src, src_shape, dst_shape)
    }
}

/// Optimized 2x upsampling using contiguous memory access.
///
/// This kernel exploits the regular pattern of 2x upsampling:
/// - Even output indices map exactly to input indices
/// - Odd output indices interpolate between adjacent input values
///
/// By processing along X axis (contiguous in F-order), we achieve
/// much better cache utilization than the general scattered-gather approach.
#[allow(clippy::similar_names, clippy::needless_range_loop)]
pub fn trilinear_upsample_2x_forder(
    src: &[f32],
    src_shape: [usize; 3],
    dst_shape: [usize; 3],
) -> Vec<f32> {
    use crate::pipeline::acquire_buffer;
    use rayon::prelude::*;

    let [sx, sy, sz] = src_shape;
    let [dx, dy, dz] = dst_shape;

    // Source strides (F-order: X varies fastest)
    let src_stride_y = sx;
    let src_stride_z = sx * sy;

    // Destination strides
    let dst_stride_y = dx;
    let dst_stride_z = dx * dy;

    let total_voxels = dx * dy * dz;
    let mut dst: Vec<f32> = acquire_buffer(total_voxels);

    // Precompute Y and Z interpolation weights
    // For 2x upsampling: even indices → frac=0, odd indices → frac=0.5
    let y_scale = (sy - 1) as f32 / (dy - 1).max(1) as f32;
    let z_scale = (sz - 1) as f32 / (dz - 1).max(1) as f32;

    // Process Z slices in parallel
    dst.par_chunks_mut(dst_stride_z)
        .enumerate()
        .for_each(|(z_dst, z_slice)| {
            // Z interpolation
            let z_pos = z_dst as f32 * z_scale;
            let z0 = (z_pos.floor() as usize).min(sz - 1);
            let z1 = (z0 + 1).min(sz - 1);
            let wz = z_pos - z0 as f32;
            let wz_inv = 1.0 - wz;

            let z0_base = z0 * src_stride_z;
            let z1_base = z1 * src_stride_z;

            for y_dst in 0..dy {
                // Y interpolation
                let y_pos = y_dst as f32 * y_scale;
                let y0 = (y_pos.floor() as usize).min(sy - 1);
                let y1 = (y0 + 1).min(sy - 1);
                let wy = y_pos - y0 as f32;
                let wy_inv = 1.0 - wy;

                // Combined Y-Z weights
                let w00 = wz_inv * wy_inv;
                let w01 = wz_inv * wy;
                let w10 = wz * wy_inv;
                let w11 = wz * wy;

                // Source row offsets for the 4 corners
                let off_z0_y0 = z0_base + y0 * src_stride_y;
                let off_z0_y1 = z0_base + y1 * src_stride_y;
                let off_z1_y0 = z1_base + y0 * src_stride_y;
                let off_z1_y1 = z1_base + y1 * src_stride_y;

                let dst_row = &mut z_slice[y_dst * dst_stride_y..(y_dst + 1) * dst_stride_y];

                // Use optimized X interpolation with contiguous access
                upsample_x_row_simd(
                    src, sx, dx, off_z0_y0, off_z0_y1, off_z1_y0, off_z1_y1, w00, w01, w10, w11,
                    dst_row,
                );
            }
        });

    dst
}

/// SIMD-optimized X-axis upsampling using contiguous reads.
///
/// Instead of gathering scattered values, this reads consecutive source
/// values and computes interpolated outputs efficiently.
#[inline]
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
fn upsample_x_row_simd(
    src: &[f32],
    sx: usize,
    dx: usize,
    off_z0_y0: usize,
    off_z0_y1: usize,
    off_z1_y0: usize,
    off_z1_y1: usize,
    w00: f32,
    w01: f32,
    w10: f32,
    w11: f32,
    dst_row: &mut [f32],
) {
    let x_scale = (sx - 1) as f32 / (dx - 1).max(1) as f32;

    // SIMD constants for Y-Z weights
    let w00_v = f32x8::splat(w00);
    let w01_v = f32x8::splat(w01);
    let w10_v = f32x8::splat(w10);
    let w11_v = f32x8::splat(w11);

    // Process 8 output values at a time
    let chunks = dx / SIMD_WIDTH;

    for chunk_i in 0..chunks {
        let base = chunk_i * SIMD_WIDTH;

        // Gather X-interpolated values for each of the 4 Y-Z corners
        let mut v_z0_y0 = [0.0f32; 8];
        let mut v_z0_y1 = [0.0f32; 8];
        let mut v_z1_y0 = [0.0f32; 8];
        let mut v_z1_y1 = [0.0f32; 8];

        for i in 0..SIMD_WIDTH {
            let x_dst = base + i;
            let x_pos = x_dst as f32 * x_scale;
            let x0 = (x_pos.floor() as usize).min(sx - 1);
            let x1 = (x0 + 1).min(sx - 1);
            let wx = x_pos - x0 as f32;
            let wx_inv = 1.0 - wx;

            // X-interpolate for each Y-Z corner
            v_z0_y0[i] = src[off_z0_y0 + x0] * wx_inv + src[off_z0_y0 + x1] * wx;
            v_z0_y1[i] = src[off_z0_y1 + x0] * wx_inv + src[off_z0_y1 + x1] * wx;
            v_z1_y0[i] = src[off_z1_y0 + x0] * wx_inv + src[off_z1_y0 + x1] * wx;
            v_z1_y1[i] = src[off_z1_y1 + x0] * wx_inv + src[off_z1_y1 + x1] * wx;
        }

        // SIMD weighted sum for Y-Z interpolation
        let v_z0_y0_v = f32x8::from(v_z0_y0);
        let v_z0_y1_v = f32x8::from(v_z0_y1);
        let v_z1_y0_v = f32x8::from(v_z1_y0);
        let v_z1_y1_v = f32x8::from(v_z1_y1);

        let result = v_z0_y0_v * w00_v + v_z0_y1_v * w01_v + v_z1_y0_v * w10_v + v_z1_y1_v * w11_v;
        let result_arr: [f32; 8] = result.into();
        dst_row[base..base + SIMD_WIDTH].copy_from_slice(&result_arr);
    }

    // Scalar remainder
    for x_dst in (chunks * SIMD_WIDTH)..dx {
        let x_pos = x_dst as f32 * x_scale;
        let x0 = (x_pos.floor() as usize).min(sx - 1);
        let x1 = (x0 + 1).min(sx - 1);
        let wx = x_pos - x0 as f32;
        let wx_inv = 1.0 - wx;

        let v_z0_y0 = src[off_z0_y0 + x0] * wx_inv + src[off_z0_y0 + x1] * wx;
        let v_z0_y1 = src[off_z0_y1 + x0] * wx_inv + src[off_z0_y1 + x1] * wx;
        let v_z1_y0 = src[off_z1_y0 + x0] * wx_inv + src[off_z1_y0 + x1] * wx;
        let v_z1_y1 = src[off_z1_y1 + x0] * wx_inv + src[off_z1_y1 + x1] * wx;

        dst_row[x_dst] = v_z0_y0 * w00 + v_z0_y1 * w01 + v_z1_y0 * w10 + v_z1_y1 * w11;
    }
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
