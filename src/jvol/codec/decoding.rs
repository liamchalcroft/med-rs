use ndarray::Array3;

use super::entropy::rice_decode_subband;
use super::subbands::{compute_subbands, inject_subband_i32, inject_subband_i32_f32};
use super::types::{EncodedSubband, JvolDtype};
use super::wavelet::{downsample_dc_gain, dwt3d_inverse, dwt3d_inverse_g, WaveletType};

/// Decode encoded subbands back into a 3D array.
#[allow(clippy::too_many_arguments)]
pub fn decode_array(
    subbands: &[EncodedSubband],
    shape: [usize; 3],
    wavelet: WaveletType,
    levels: usize,
    step: f64,
    intercept: f64,
    slope: f64,
    quality: u8,
    dtype: JvolDtype,
) -> Array3<f64> {
    if quality == 0 {
        decode_lossless(subbands, shape, dtype)
    } else {
        decode_lossy(
            subbands, shape, wavelet, levels, step, intercept, slope, dtype,
        )
    }
}

/// Lossless decode, dtype-aware: reverse byte-unshuffle + delta/XOR-delta decode.
fn decode_lossless(
    subbands: &[EncodedSubband],
    shape: [usize; 3],
    dtype: JvolDtype,
) -> Array3<f64> {
    assert_eq!(
        subbands.len(),
        1,
        "Lossless mode expects single encoded block"
    );
    let sub = &subbands[0];
    let [ni, nj, nk] = shape;

    let mut array = Array3::zeros((ni, nj, nk));

    match dtype {
        JvolDtype::U8 => {
            let mut vals = sub.data.clone();
            delta_decode_u8(&mut vals);
            let mut idx = 0;
            for k in 0..nk {
                for j in 0..nj {
                    for i in 0..ni {
                        array[[i, j, k]] = vals[idx] as f64;
                        idx += 1;
                    }
                }
            }
        }
        JvolDtype::U16 => {
            let unshuffled = byte_unshuffle(&sub.data, 2);
            let mut vals = from_le_bytes_u16(&unshuffled);
            delta_decode_u16(&mut vals);
            let mut idx = 0;
            for k in 0..nk {
                for j in 0..nj {
                    for i in 0..ni {
                        array[[i, j, k]] = vals[idx] as f64;
                        idx += 1;
                    }
                }
            }
        }
        JvolDtype::I16 => {
            let unshuffled = byte_unshuffle(&sub.data, 2);
            let mut vals = from_le_bytes_i16(&unshuffled);
            delta_decode_i16(&mut vals);
            let mut idx = 0;
            for k in 0..nk {
                for j in 0..nj {
                    for i in 0..ni {
                        array[[i, j, k]] = vals[idx] as f64;
                        idx += 1;
                    }
                }
            }
        }
        JvolDtype::I32 => {
            let unshuffled = byte_unshuffle(&sub.data, 4);
            let mut vals = from_le_bytes_i32(&unshuffled);
            delta_decode_i32(&mut vals);
            let mut idx = 0;
            for k in 0..nk {
                for j in 0..nj {
                    for i in 0..ni {
                        array[[i, j, k]] = vals[idx] as f64;
                        idx += 1;
                    }
                }
            }
        }
        JvolDtype::F32 => {
            // Raw f32 bytes in Fortran order
            let mut idx = 0;
            for k in 0..nk {
                for j in 0..nj {
                    for i in 0..ni {
                        let offset = idx * 4;
                        let v = f32::from_le_bytes([
                            sub.data[offset],
                            sub.data[offset + 1],
                            sub.data[offset + 2],
                            sub.data[offset + 3],
                        ]);
                        array[[i, j, k]] = v as f64;
                        idx += 1;
                    }
                }
            }
        }
        JvolDtype::F64 => {
            // Raw f64 bytes in Fortran order
            let mut idx = 0;
            for k in 0..nk {
                for j in 0..nj {
                    for i in 0..ni {
                        let offset = idx * 8;
                        let v = f64::from_le_bytes([
                            sub.data[offset],
                            sub.data[offset + 1],
                            sub.data[offset + 2],
                            sub.data[offset + 3],
                            sub.data[offset + 4],
                            sub.data[offset + 5],
                            sub.data[offset + 6],
                            sub.data[offset + 7],
                        ]);
                        array[[i, j, k]] = v;
                        idx += 1;
                    }
                }
            }
        }
    }

    array
}

// --- Byte unshuffle ---

/// Reverse byte-shuffle: N planes of 1-byte → N-byte elements.
fn byte_unshuffle(data: &[u8], elem_size: usize) -> Vec<u8> {
    let n = data.len() / elem_size;
    let mut out = vec![0u8; data.len()];
    for i in 0..n {
        for b in 0..elem_size {
            out[i * elem_size + b] = data[b * n + i];
        }
    }
    out
}

// --- Delta decode (prefix sum with wrapping arithmetic) ---

fn delta_decode_u8(data: &mut [u8]) {
    for i in 1..data.len() {
        data[i] = data[i].wrapping_add(data[i - 1]);
    }
}

fn delta_decode_u16(data: &mut [u16]) {
    for i in 1..data.len() {
        data[i] = data[i].wrapping_add(data[i - 1]);
    }
}

fn delta_decode_i16(data: &mut [i16]) {
    for i in 1..data.len() {
        data[i] = data[i].wrapping_add(data[i - 1]);
    }
}

fn delta_decode_i32(data: &mut [i32]) {
    for i in 1..data.len() {
        data[i] = data[i].wrapping_add(data[i - 1]);
    }
}

// --- Bytes-to-type conversion ---

fn from_le_bytes_u16(data: &[u8]) -> Vec<u16> {
    data.chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect()
}

fn from_le_bytes_i16(data: &[u8]) -> Vec<i16> {
    data.chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]))
        .collect()
}

fn from_le_bytes_i32(data: &[u8]) -> Vec<i32> {
    data.chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Lossy decode: Rice decode → inverse DWT → denormalize.
#[allow(clippy::too_many_arguments)]
fn decode_lossy(
    subbands: &[EncodedSubband],
    shape: [usize; 3],
    wavelet: WaveletType,
    levels: usize,
    step: f64,
    intercept: f64,
    slope: f64,
    dtype: JvolDtype,
) -> Array3<f64> {
    let subband_infos = compute_subbands(shape, levels);
    assert_eq!(
        subbands.len(),
        subband_infos.len(),
        "Subband count mismatch"
    );

    let mut data = Array3::zeros((shape[0], shape[1], shape[2]));

    for (encoded, info) in subbands.iter().zip(subband_infos.iter()) {
        let coefficients =
            rice_decode_subband(&encoded.data, encoded.num_values as usize, encoded.rice_k);
        inject_subband_i32(&mut data, info, &coefficients);
    }

    // Dequantize
    data.mapv_inplace(|v| v * step);

    // Inverse 3D DWT
    dwt3d_inverse(&mut data, wavelet, levels);

    // Denormalize
    if slope > 0.0 {
        let scale = slope / 255.0;
        let offset = 128.0 * scale + intercept;
        data.mapv_inplace(|v| v * scale + offset);
    }

    // Clip to dtype range
    if let Some((min_val, max_val)) = dtype.iinfo() {
        data.mapv_inplace(|v| v.max(min_val).min(max_val));
    }

    data
}

/// f32 lossy decode: same pipeline as [`decode_lossy`] but keeps coefficients,
/// dequantization, the inverse DWT and denormalization in f32 throughout,
/// avoiding the f64 intermediate. Lossy data is already quantized, so f32
/// precision is sufficient and the result matches the f64 path to within a
/// tiny fraction of the value range.
///
/// Only valid for lossy channels (`quality != 0`).
#[allow(clippy::too_many_arguments)]
pub fn decode_lossy_f32(
    subbands: &[EncodedSubband],
    shape: [usize; 3],
    wavelet: WaveletType,
    levels: usize,
    step: f64,
    intercept: f64,
    slope: f64,
    dtype: JvolDtype,
) -> Array3<f32> {
    let subband_infos = compute_subbands(shape, levels);
    assert_eq!(
        subbands.len(),
        subband_infos.len(),
        "Subband count mismatch"
    );

    let mut data = Array3::<f32>::zeros((shape[0], shape[1], shape[2]));
    let step_f32 = step as f32;

    // Rice decode + dequantize (fused into one pass per subband).
    for (encoded, info) in subbands.iter().zip(subband_infos.iter()) {
        let coefficients =
            rice_decode_subband(&encoded.data, encoded.num_values as usize, encoded.rice_k);
        inject_subband_i32_f32(&mut data, info, &coefficients, step_f32);
    }

    // Inverse 3D DWT in f32
    dwt3d_inverse_g::<f32>(&mut data, wavelet, levels);

    // Denormalize
    if slope > 0.0 {
        let scale = (slope / 255.0) as f32;
        let offset = (128.0 * (slope / 255.0) + intercept) as f32;
        data.mapv_inplace(|v| v * scale + offset);
    }

    // Clip to dtype range
    if let Some((min_val, max_val)) = dtype.iinfo() {
        let (mn, mx) = (min_val as f32, max_val as f32);
        data.mapv_inplace(|v| v.max(mn).min(mx));
    }

    data
}

/// Progressive/multiresolution lossy decode at `1 / factor` of the full
/// resolution per axis (`factor` a power of two, `factor <= 2^levels`).
///
/// A multi-level 3D DWT stores a coarse approximation subband plus detail
/// subbands per level. Reconstructing at `1/2^k` resolution needs only the
/// subbands at levels `>= k`, so the finest (largest, most numerous) detail
/// subbands are never Rice-decoded and the inverse DWT runs for only the coarse
/// `levels - k` levels. The result is a low-pass approximation of the volume
/// downsampled by `factor`, at a fraction of the full decode cost.
#[allow(clippy::too_many_arguments)]
pub fn decode_downsampled_f32(
    subbands: &[EncodedSubband],
    full_shape: [usize; 3],
    wavelet: WaveletType,
    levels: usize,
    step: f64,
    intercept: f64,
    slope: f64,
    factor: usize,
) -> Array3<f32> {
    let k = factor.trailing_zeros() as usize;

    // Extent at each level (level 0 = full shape).
    let mut extents = Vec::with_capacity(levels + 1);
    extents.push(full_shape);
    for _ in 0..levels {
        let p = *extents.last().unwrap();
        extents.push([p[0].div_ceil(2), p[1].div_ceil(2), p[2].div_ceil(2)]);
    }
    let small_shape = extents[k];

    let subband_infos = compute_subbands(full_shape, levels);
    let mut data = Array3::<f32>::zeros((small_shape[0], small_shape[1], small_shape[2]));
    let step_f32 = step as f32;

    // Inject only the subbands at levels >= k. Their offsets already lie inside
    // the top-left `small_shape` corner (coarser subbands nest there), so they
    // drop straight into the downsampled buffer. Finer subbands are skipped,
    // including their Rice decode.
    for (encoded, info) in subbands.iter().zip(subband_infos.iter()) {
        if info.level < k {
            continue;
        }
        let coefficients =
            rice_decode_subband(&encoded.data, encoded.num_values as usize, encoded.rice_k);
        inject_subband_i32_f32(&mut data, info, &coefficients, step_f32);
    }

    // Partial inverse DWT: only the coarse `levels - k` levels.
    dwt3d_inverse_g::<f32>(&mut data, wavelet, levels - k);

    // Undo the low-pass DC gain of the k skipped finest levels.
    let gain = downsample_dc_gain(wavelet, k);
    if gain != 1.0 {
        let inv = (1.0 / gain) as f32;
        data.mapv_inplace(|v| v * inv);
    }

    // Denormalize (same affine mapping as the full-resolution decode).
    if slope > 0.0 {
        let scale = (slope / 255.0) as f32;
        let offset = (128.0 * (slope / 255.0) + intercept) as f32;
        data.mapv_inplace(|v| v * scale + offset);
    }

    data
}

#[cfg(test)]
mod tests {
    use super::super::encoding::encode_array;
    use super::*;
    use ndarray::Array3;

    fn smooth_volume(n: usize) -> Array3<f64> {
        Array3::from_shape_fn((n, n, n), |(i, j, k)| {
            let x = i as f64 / n as f64;
            let y = j as f64 / n as f64;
            let z = k as f64 / n as f64;
            ((x * 6.0).sin() + (y * 5.0).cos() + (z * 7.0).sin()) * 300.0 + 500.0
        })
    }

    #[test]
    fn f32_lossy_decode_matches_f64() {
        let shape = [40, 36, 32];
        let vol = smooth_volume(40);
        let vol = vol.slice(ndarray::s![.., ..36, ..32]).to_owned();
        let res = encode_array(&vol.view(), 60, JvolDtype::F32);

        let f64_dec = decode_lossy(
            &res.subbands,
            shape,
            res.wavelet,
            res.levels,
            res.step,
            res.intercept,
            res.slope,
            JvolDtype::F32,
        );
        let f32_dec = decode_lossy_f32(
            &res.subbands,
            shape,
            res.wavelet,
            res.levels,
            res.step,
            res.intercept,
            res.slope,
            JvolDtype::F32,
        );

        let range = vol.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
            - vol.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_abs = f64_dec
            .iter()
            .zip(f32_dec.iter())
            .map(|(a, b)| (a - *b as f64).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_abs < 1e-3 * range,
            "f32 vs f64 decode diverged: {max_abs} (range {range})"
        );
    }

    #[test]
    fn downsampled_decode_shape_and_mean() {
        let n = 64;
        let shape = [n, n, n];
        let vol = smooth_volume(n);
        let res = encode_array(&vol.view(), 60, JvolDtype::F32);

        let full = decode_lossy_f32(
            &res.subbands,
            shape,
            res.wavelet,
            res.levels,
            res.step,
            res.intercept,
            res.slope,
            JvolDtype::F32,
        );
        let full_mean = full.iter().map(|&v| v as f64).sum::<f64>() / full.len() as f64;

        for factor in [2usize, 4] {
            let down = decode_downsampled_f32(
                &res.subbands,
                shape,
                res.wavelet,
                res.levels,
                res.step,
                res.intercept,
                res.slope,
                factor,
            );
            let expected = n / factor;
            assert_eq!(
                down.shape(),
                &[expected, expected, expected],
                "downsampled shape wrong for factor {factor}"
            );
            let down_mean = down.iter().map(|&v| v as f64).sum::<f64>() / down.len() as f64;
            let range = vol.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
                - vol.iter().cloned().fold(f64::INFINITY, f64::min);
            assert!(
                (down_mean - full_mean).abs() < 0.05 * range,
                "factor {factor}: downsampled mean {down_mean} vs full {full_mean} (range {range})"
            );
        }
    }

    // Release-only timing: `cargo test --release --features jvol -- --ignored --nocapture measure_decode`
    #[test]
    #[ignore]
    fn measure_decode_speedups() {
        use std::time::Instant;

        let n = 256;
        let shape = [n, n, n];
        let vol = smooth_volume(n);
        let res = encode_array(&vol.view(), 60, JvolDtype::F32);

        let (subs, wav, lvl, step, inter, slope) = (
            &res.subbands,
            res.wavelet,
            res.levels,
            res.step,
            res.intercept,
            res.slope,
        );
        let iters = 5;
        let bench = |mut f: Box<dyn FnMut()>| {
            f(); // warm up
            let t = Instant::now();
            for _ in 0..iters {
                f();
            }
            t.elapsed().as_secs_f64() * 1e3 / iters as f64
        };

        let f64_ms = bench(Box::new(|| {
            let _ = decode_lossy(subs, shape, wav, lvl, step, inter, slope, JvolDtype::F32);
        }));
        let f32_ms = bench(Box::new(|| {
            let _ = decode_lossy_f32(subs, shape, wav, lvl, step, inter, slope, JvolDtype::F32);
        }));
        let d2_ms = bench(Box::new(|| {
            let _ = decode_downsampled_f32(subs, shape, wav, lvl, step, inter, slope, 2);
        }));
        let d4_ms = bench(Box::new(|| {
            let _ = decode_downsampled_f32(subs, shape, wav, lvl, step, inter, slope, 4);
        }));

        println!("\n=== jvol 256^3 q60 decode timings (ms, mean of {iters}) ===");
        println!("f64 full decode      : {f64_ms:8.2}");
        println!(
            "f32 full decode      : {f32_ms:8.2}  ({:.2}x vs f64)",
            f64_ms / f32_ms
        );
        println!(
            "f32 downsample x2    : {d2_ms:8.2}  ({:.2}x vs f64 full)",
            f64_ms / d2_ms
        );
        println!(
            "f32 downsample x4    : {d4_ms:8.2}  ({:.2}x vs f64 full)",
            f64_ms / d4_ms
        );
    }
}
