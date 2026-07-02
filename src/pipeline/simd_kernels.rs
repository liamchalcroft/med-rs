//! Portable-SIMD kernels for transform operations.
//!
//! Hot loops use `wide::f32x8`, an 8-wide f32 vector that lowers to whatever the
//! compile target supports: two SSE registers on the x86-64 baseline, a single
//! AVX register only when built with `-C target-feature=+avx2` (or
//! `-C target-cpu=native`). The scalar tails handle the trailing elements that
//! do not fill a full vector.

use wide::{f32x8, f64x4};

/// Number of f32 lanes processed per vector iteration.
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
/// Accumulates in f64 lanes so large volumes do not lose precision the way f32
/// accumulation does. Returns (sum, sum_sq, count).
#[inline]
pub fn sum_and_sum_sq_f32(input: &[f32]) -> (f64, f64, usize) {
    let len = input.len();
    let chunks = len / SIMD_WIDTH;
    let remainder = len % SIMD_WIDTH;

    let mut sum_lo = f64x4::splat(0.0);
    let mut sum_hi = f64x4::splat(0.0);
    let mut sq_lo = f64x4::splat(0.0);
    let mut sq_hi = f64x4::splat(0.0);

    for i in 0..chunks {
        let base = i * SIMD_WIDTH;
        let arr: [f32; 8] = f32x8::from(&input[base..base + SIMD_WIDTH]).into();
        let lo = f64x4::from([arr[0] as f64, arr[1] as f64, arr[2] as f64, arr[3] as f64]);
        let hi = f64x4::from([arr[4] as f64, arr[5] as f64, arr[6] as f64, arr[7] as f64]);
        sum_lo += lo;
        sum_hi += hi;
        sq_lo += lo * lo;
        sq_hi += hi * hi;
    }

    let sum_lo: [f64; 4] = sum_lo.into();
    let sum_hi: [f64; 4] = sum_hi.into();
    let sq_lo: [f64; 4] = sq_lo.into();
    let sq_hi: [f64; 4] = sq_hi.into();
    let mut sum: f64 = sum_lo.iter().chain(sum_hi.iter()).sum();
    let mut sum_sq: f64 = sq_lo.iter().chain(sq_hi.iter()).sum();

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

// =============================================================================
// DTYPE CONVERSION TO F32 WITH SCALING
// =============================================================================

/// Convert u8 bytes to f32 slice with scaling.
///
/// For 8-bit types (no swap needed), just applies scaling.
#[inline]
pub fn u8_to_f32_scaled(input: &[u8], output: &mut [f32], slope: f32, inter: f32) {
    debug_assert_eq!(input.len(), output.len());

    let len = output.len();
    let chunks = len / SIMD_WIDTH;

    let slope_vec = f32x8::splat(slope);
    let inter_vec = f32x8::splat(inter);

    for i in 0..chunks {
        let base = i * SIMD_WIDTH;
        // Convert 8 u8s to f32s
        let v: [f32; 8] = [
            input[base] as f32,
            input[base + 1] as f32,
            input[base + 2] as f32,
            input[base + 3] as f32,
            input[base + 4] as f32,
            input[base + 5] as f32,
            input[base + 6] as f32,
            input[base + 7] as f32,
        ];
        let in_vec = f32x8::from(v);
        let out_vec = in_vec * slope_vec + inter_vec;
        let out_arr: [f32; 8] = out_vec.into();
        output[base..base + SIMD_WIDTH].copy_from_slice(&out_arr);
    }

    // Scalar remainder
    let base = chunks * SIMD_WIDTH;
    for i in base..len {
        output[i] = input[i] as f32 * slope + inter;
    }
}

/// Generate a byte-slice to f32 conversion kernel for a 2-byte integer type,
/// parameterized over the byte-order decode (`from_ne_bytes` vs `from_be_bytes`).
macro_rules! define_int2_to_f32 {
    ($name:ident, $ty:ty, $from_bytes:ident) => {
        #[doc = concat!("Decode ", stringify!($ty), " byte data to f32 with scaling.")]
        #[inline]
        pub fn $name(input: &[u8], output: &mut [f32], slope: f32, inter: f32) {
            debug_assert_eq!(input.len(), output.len() * 2);
            let len = output.len();
            let chunks = len / SIMD_WIDTH;
            let slope_vec = f32x8::splat(slope);
            let inter_vec = f32x8::splat(inter);
            for i in 0..chunks {
                let base_in = i * SIMD_WIDTH * 2;
                let base_out = i * SIMD_WIDTH;
                let v: [f32; 8] = std::array::from_fn(|j| {
                    let o = base_in + j * 2;
                    <$ty>::$from_bytes([input[o], input[o + 1]]) as f32
                });
                let out_arr: [f32; 8] = (f32x8::from(v) * slope_vec + inter_vec).into();
                output[base_out..base_out + SIMD_WIDTH].copy_from_slice(&out_arr);
            }
            let base_out = chunks * SIMD_WIDTH;
            let base_in = base_out * 2;
            for i in 0..(len - base_out) {
                let idx = base_in + i * 2;
                let val = <$ty>::$from_bytes([input[idx], input[idx + 1]]) as f32;
                output[base_out + i] = val * slope + inter;
            }
        }
    };
}

/// Generate a byte-slice to f32 conversion kernel for a 4-byte integer type.
macro_rules! define_int4_to_f32 {
    ($name:ident, $ty:ty, $from_bytes:ident) => {
        #[doc = concat!("Decode ", stringify!($ty), " byte data to f32 with scaling.")]
        #[inline]
        pub fn $name(input: &[u8], output: &mut [f32], slope: f32, inter: f32) {
            debug_assert_eq!(input.len(), output.len() * 4);
            let len = output.len();
            let chunks = len / SIMD_WIDTH;
            let slope_vec = f32x8::splat(slope);
            let inter_vec = f32x8::splat(inter);
            for i in 0..chunks {
                let base_in = i * SIMD_WIDTH * 4;
                let base_out = i * SIMD_WIDTH;
                let v: [f32; 8] = std::array::from_fn(|j| {
                    let o = base_in + j * 4;
                    <$ty>::$from_bytes([input[o], input[o + 1], input[o + 2], input[o + 3]]) as f32
                });
                let out_arr: [f32; 8] = (f32x8::from(v) * slope_vec + inter_vec).into();
                output[base_out..base_out + SIMD_WIDTH].copy_from_slice(&out_arr);
            }
            let base_out = chunks * SIMD_WIDTH;
            let base_in = base_out * 4;
            for i in 0..(len - base_out) {
                let idx = base_in + i * 4;
                let val = <$ty>::$from_bytes([
                    input[idx],
                    input[idx + 1],
                    input[idx + 2],
                    input[idx + 3],
                ]) as f32;
                output[base_out + i] = val * slope + inter;
            }
        }
    };
}

/// Generate an f32 byte-slice conversion kernel (decode returns f32 directly).
macro_rules! define_f32_to_f32 {
    ($name:ident, $from_bytes:ident) => {
        #[doc = "Decode f32 byte data to f32 with scaling."]
        #[inline]
        pub fn $name(input: &[u8], output: &mut [f32], slope: f32, inter: f32) {
            debug_assert_eq!(input.len(), output.len() * 4);
            let len = output.len();
            let chunks = len / SIMD_WIDTH;
            let slope_vec = f32x8::splat(slope);
            let inter_vec = f32x8::splat(inter);
            for i in 0..chunks {
                let base_in = i * SIMD_WIDTH * 4;
                let base_out = i * SIMD_WIDTH;
                let v: [f32; 8] = std::array::from_fn(|j| {
                    let o = base_in + j * 4;
                    f32::$from_bytes([input[o], input[o + 1], input[o + 2], input[o + 3]])
                });
                let out_arr: [f32; 8] = (f32x8::from(v) * slope_vec + inter_vec).into();
                output[base_out..base_out + SIMD_WIDTH].copy_from_slice(&out_arr);
            }
            let base_out = chunks * SIMD_WIDTH;
            let base_in = base_out * 4;
            for i in 0..(len - base_out) {
                let idx = base_in + i * 4;
                let val =
                    f32::$from_bytes([input[idx], input[idx + 1], input[idx + 2], input[idx + 3]]);
                output[base_out + i] = val * slope + inter;
            }
        }
    };
}

// Native-endian decode uses `from_ne_bytes`; non-native ("swap") uses
// `from_be_bytes`, since NIfTI stores its declared endianness and the only
// non-native case reached here is big-endian data on a little-endian host.
define_int2_to_f32!(i16_native_to_f32_scaled, i16, from_ne_bytes);
define_int2_to_f32!(i16_swap_to_f32_scaled, i16, from_be_bytes);
define_int2_to_f32!(u16_swap_to_f32_scaled, u16, from_be_bytes);
define_int4_to_f32!(i32_swap_to_f32_scaled, i32, from_be_bytes);
define_int4_to_f32!(u32_swap_to_f32_scaled, u32, from_be_bytes);
define_f32_to_f32!(f32_native_scaled, from_ne_bytes);
define_f32_to_f32!(f32_swap_to_f32_scaled, from_be_bytes);

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
    /// Uses the half-pixel-center convention (SimpleITK/MONAI default):
    /// source position = (i + 0.5) * old/new - 0.5, clamped to [0, old-1].
    /// This matches the nearest-neighbour sampling path so image and label
    /// resampling stay aligned.
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

        let scale = old_size as f32 / new_size as f32;
        let max_idx = old_size - 1;

        let mut idx0 = Vec::with_capacity(new_size);
        let mut idx1 = Vec::with_capacity(new_size);
        let mut frac = Vec::with_capacity(new_size);
        let mut frac_inv = Vec::with_capacity(new_size);

        for i in 0..new_size {
            let pos = ((i as f32 + 0.5) * scale - 0.5).clamp(0.0, max_idx as f32);
            let i0 = (pos.floor() as usize).min(max_idx);
            let i1 = (i0 + 1).min(max_idx);
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
    let mut dst: Vec<f32> = vec![0.0f32; total_voxels];

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

    let w00_v = f32x8::splat(w00);
    let w01_v = f32x8::splat(w01);
    let w10_v = f32x8::splat(w10);
    let w11_v = f32x8::splat(w11);

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

    for chunk_i in 0..chunks {
        let base = chunk_i * SIMD_WIDTH;

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

/// Choose the resampling strategy for a given volume size.
///
/// The half-pixel-center convention makes exact 2x upsampling irregular
/// (source positions do not fall on clean even/odd boundaries), so the
/// general trilinear kernel is used for all shapes.
pub fn trilinear_resample_forder_adaptive(
    src: &[f32],
    src_shape: [usize; 3],
    dst_shape: [usize; 3],
) -> Vec<f32> {
    trilinear_resample_forder(src, src_shape, dst_shape)
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

    #[test]
    fn test_u8_to_f32_scaled() {
        let input: Vec<u8> = (0..20).collect();
        let mut output = vec![0.0f32; 20];

        u8_to_f32_scaled(&input, &mut output, 2.0, 10.0);

        for i in 0..20 {
            let expected = input[i] as f32 * 2.0 + 10.0;
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
    fn test_i16_swap_to_f32_scaled() {
        // Create big-endian i16 values: 256, 512, 768, ...
        // In BE: 256 = [0x01, 0x00], 512 = [0x02, 0x00], etc.
        let mut input = Vec::new();
        for i in 1..=16i16 {
            let val = i * 256;
            input.extend_from_slice(&val.to_be_bytes());
        }

        let mut output = vec![0.0f32; 16];
        i16_swap_to_f32_scaled(&input, &mut output, 1.0, 0.0);

        for i in 0..16 {
            let expected = ((i + 1) * 256) as f32;
            assert!(
                (output[i] - expected).abs() < 1e-5,
                "At {}: expected {}, got {}",
                i,
                expected,
                output[i]
            );
        }

        // Test with scaling
        let mut output2 = vec![0.0f32; 16];
        i16_swap_to_f32_scaled(&input, &mut output2, 0.5, 100.0);

        for i in 0..16 {
            let expected = ((i + 1) * 256) as f32 * 0.5 + 100.0;
            assert!(
                (output2[i] - expected).abs() < 1e-5,
                "At {}: expected {}, got {}",
                i,
                expected,
                output2[i]
            );
        }
    }

    #[test]
    fn test_f32_swap_to_f32_scaled() {
        // Create big-endian f32 values
        let values: Vec<f32> = (0..16).map(|i| i as f32 * 1.5).collect();
        let mut input = Vec::new();
        for v in &values {
            input.extend_from_slice(&v.to_be_bytes());
        }

        let mut output = vec![0.0f32; 16];
        f32_swap_to_f32_scaled(&input, &mut output, 1.0, 0.0);

        for i in 0..16 {
            assert!(
                (output[i] - values[i]).abs() < 1e-5,
                "At {}: expected {}, got {}",
                i,
                values[i],
                output[i]
            );
        }

        // Test with scaling
        let mut output2 = vec![0.0f32; 16];
        f32_swap_to_f32_scaled(&input, &mut output2, 2.0, -5.0);

        for i in 0..16 {
            let expected = values[i] * 2.0 - 5.0;
            assert!(
                (output2[i] - expected).abs() < 1e-5,
                "At {}: expected {}, got {}",
                i,
                expected,
                output2[i]
            );
        }
    }

    #[test]
    fn test_i16_native_to_f32_scaled() {
        // Create native-endian i16 values
        let values: Vec<i16> = (-8..8).collect();
        let mut input = Vec::new();
        for v in &values {
            input.extend_from_slice(&v.to_ne_bytes());
        }

        let mut output = vec![0.0f32; 16];
        i16_native_to_f32_scaled(&input, &mut output, 1.0, 0.0);

        for i in 0..16 {
            let expected = values[i] as f32;
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
    fn test_f32_native_scaled() {
        // Create native-endian f32 values
        let values: Vec<f32> = (0..16).map(|i| i as f32 * 3.14).collect();
        let mut input = Vec::new();
        for v in &values {
            input.extend_from_slice(&v.to_ne_bytes());
        }

        let mut output = vec![0.0f32; 16];
        f32_native_scaled(&input, &mut output, 1.0, 0.0);

        for i in 0..16 {
            assert!(
                (output[i] - values[i]).abs() < 1e-5,
                "At {}: expected {}, got {}",
                i,
                values[i],
                output[i]
            );
        }
    }

    #[test]
    fn test_u16_swap_to_f32_scaled() {
        // Create big-endian u16 values
        let values: Vec<u16> = (0..16).map(|i| i * 1000).collect();
        let mut input = Vec::new();
        for v in &values {
            input.extend_from_slice(&v.to_be_bytes());
        }

        let mut output = vec![0.0f32; 16];
        u16_swap_to_f32_scaled(&input, &mut output, 1.0, 0.0);

        for i in 0..16 {
            let expected = values[i] as f32;
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
    fn test_i32_swap_to_f32_scaled() {
        // Create big-endian i32 values
        let values: Vec<i32> = (-8..8).map(|i| i * 100000).collect();
        let mut input = Vec::new();
        for v in &values {
            input.extend_from_slice(&v.to_be_bytes());
        }

        let mut output = vec![0.0f32; 16];
        i32_swap_to_f32_scaled(&input, &mut output, 1.0, 0.0);

        for i in 0..16 {
            let expected = values[i] as f32;
            assert!(
                (output[i] - expected).abs() < 1.0, // f32 precision limits
                "At {}: expected {}, got {}",
                i,
                expected,
                output[i]
            );
        }
    }

    #[test]
    fn test_u32_swap_to_f32_scaled() {
        // Create big-endian u32 values
        let values: Vec<u32> = (0..16).map(|i| i * 50000).collect();
        let mut input = Vec::new();
        for v in &values {
            input.extend_from_slice(&v.to_be_bytes());
        }

        let mut output = vec![0.0f32; 16];
        u32_swap_to_f32_scaled(&input, &mut output, 1.0, 0.0);

        for i in 0..16 {
            let expected = values[i] as f32;
            assert!(
                (output[i] - expected).abs() < 1.0,
                "At {}: expected {}, got {}",
                i,
                expected,
                output[i]
            );
        }
    }

    #[test]
    fn test_conversion_with_remainder() {
        // Test that functions handle non-multiple-of-8 sizes correctly
        let input: Vec<u8> = (0..13).collect();
        let mut output = vec![0.0f32; 13];

        u8_to_f32_scaled(&input, &mut output, 1.0, 0.0);

        for i in 0..13 {
            assert!(
                (output[i] - i as f32).abs() < 1e-5,
                "At {}: expected {}, got {}",
                i,
                i as f32,
                output[i]
            );
        }
    }
}
