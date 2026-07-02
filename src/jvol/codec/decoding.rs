use ndarray::Array3;

use super::entropy::rice_decode_subband;
use super::subbands::{compute_subbands, inject_subband_i32, inject_subband_i32_f32, SubbandInfo};
use super::types::{EncodedSubband, JvolDtype};
use super::wavelet::{downsample_dc_gain, dwt3d_inverse, dwt3d_inverse_g, WaveletType};

/// Error produced when a codec decode path encounters malformed or truncated
/// `.jvol` input. `src/jvol/mod.rs` maps this to `Error::InvalidFileFormat`
/// at the crate's public API boundary; a `.jvol` file is untrusted external
/// input, so every decode entry point that touches its bytes returns this
/// instead of panicking or indexing out of bounds.
#[derive(Debug, Clone)]
pub struct CodecError(pub String);

impl std::fmt::Display for CodecError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for CodecError {}

/// Result alias for codec decode paths; see [`CodecError`].
pub type CodecResult<T> = Result<T, CodecError>;

/// Sanity cap on the total voxel count of a single decoded channel, checked
/// before any array allocation. Chosen so the worst-case f64 allocation
/// (`decode_lossless`, `decode_lossy`) stays on the same order of magnitude
/// as the decompressed-file cap in `src/jvol/mod.rs` (8 GiB): 1024^3 voxels
/// is already an unusually large medical volume, and no legitimate `.jvol`
/// file comes close to this, so the cap only ever rejects hostile metadata.
const MAX_VOXEL_COUNT: usize = 1usize << 30; // 1,073,741,824 voxels (1024^3)

/// Sanity cap on the wavelet decomposition level count. A real `.jvol` file
/// never exceeds 6 (see `wavelet::compute_max_levels`); this cap is far more
/// generous but still bounds the per-level extent/subband bookkeeping
/// allocations against an absurd `levels` value in untrusted metadata.
const MAX_LEVELS: usize = 32;

/// Validate a volume shape from untrusted metadata before it drives any
/// allocation, returning the total voxel count on success.
fn validate_shape(shape: [usize; 3]) -> CodecResult<usize> {
    let [ni, nj, nk] = shape;
    let voxels = ni
        .checked_mul(nj)
        .and_then(|v| v.checked_mul(nk))
        .ok_or_else(|| CodecError(format!("jvol shape overflows: {ni}x{nj}x{nk}")))?;
    if voxels > MAX_VOXEL_COUNT {
        return Err(CodecError(format!(
            "jvol shape too large: {ni}x{nj}x{nk} = {voxels} voxels exceeds the {MAX_VOXEL_COUNT} cap"
        )));
    }
    Ok(voxels)
}

/// Validate a decomposition level count from untrusted metadata.
fn validate_levels(levels: usize) -> CodecResult<()> {
    if levels > MAX_LEVELS {
        return Err(CodecError(format!(
            "jvol levels too large: {levels} exceeds the {MAX_LEVELS} cap"
        )));
    }
    Ok(())
}

/// Validate that an encoded subband's declared `num_values` matches the
/// voxel count its `SubbandInfo` expects, before it is used to preallocate
/// the Rice-decoded coefficient buffer. `info.shape` is itself derived from
/// an already-[`validate_shape`]d volume shape, so the returned count is
/// bounded by [`MAX_VOXEL_COUNT`].
fn validated_num_values(encoded: &EncodedSubband, info: &SubbandInfo) -> CodecResult<usize> {
    let expected = info.shape[0] * info.shape[1] * info.shape[2];
    let got = encoded.num_values as usize;
    if got != expected {
        return Err(CodecError(format!(
            "jvol subband num_values mismatch: expected {expected}, got {got}"
        )));
    }
    Ok(expected)
}

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
) -> CodecResult<Array3<f64>> {
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
) -> CodecResult<Array3<f64>> {
    if subbands.len() != 1 {
        return Err(CodecError(format!(
            "jvol lossless mode expects exactly one encoded block, got {}",
            subbands.len()
        )));
    }
    let voxel_count = validate_shape(shape)?;
    let sub = &subbands[0];
    let [ni, nj, nk] = shape;

    let mut array = Array3::zeros((ni, nj, nk));

    match dtype {
        JvolDtype::U8 => {
            let mut vals = sub.data.clone();
            delta_decode_u8(&mut vals);
            require_len(vals.len(), voxel_count, "U8")?;
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
            require_len(vals.len(), voxel_count, "U16")?;
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
            require_len(vals.len(), voxel_count, "I16")?;
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
            require_len(vals.len(), voxel_count, "I32")?;
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
            let needed_bytes = voxel_count
                .checked_mul(4)
                .ok_or_else(|| CodecError("jvol F32 subband size overflows".to_string()))?;
            require_len(sub.data.len(), needed_bytes, "F32")?;
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
            let needed_bytes = voxel_count
                .checked_mul(8)
                .ok_or_else(|| CodecError("jvol F64 subband size overflows".to_string()))?;
            require_len(sub.data.len(), needed_bytes, "F64")?;
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

    Ok(array)
}

/// Reject a decoded/raw buffer shorter than the volume requires, naming the
/// dtype for a useful error message. Used throughout [`decode_lossless`] in
/// place of the direct indexing that would otherwise panic on truncated or
/// undersized subband data.
fn require_len(got: usize, needed: usize, what: &str) -> CodecResult<()> {
    if got < needed {
        return Err(CodecError(format!(
            "jvol lossless {what} subband too short: expected at least {needed}, got {got}"
        )));
    }
    Ok(())
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

/// Check that the decoded subband count matches what the (already-validated)
/// shape and levels imply, naming both counts for a useful error message.
fn require_subband_count(got: usize, expected: usize) -> CodecResult<()> {
    if got != expected {
        return Err(CodecError(format!(
            "jvol subband count mismatch: expected {expected}, got {got}"
        )));
    }
    Ok(())
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
) -> CodecResult<Array3<f64>> {
    validate_shape(shape)?;
    validate_levels(levels)?;
    let subband_infos = compute_subbands(shape, levels);
    require_subband_count(subbands.len(), subband_infos.len())?;

    let mut data = Array3::zeros((shape[0], shape[1], shape[2]));

    for (encoded, info) in subbands.iter().zip(subband_infos.iter()) {
        let num_values = validated_num_values(encoded, info)?;
        let coefficients = rice_decode_subband(&encoded.data, num_values, encoded.rice_k)?;
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

    Ok(data)
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
) -> CodecResult<Array3<f32>> {
    validate_shape(shape)?;
    validate_levels(levels)?;
    let subband_infos = compute_subbands(shape, levels);
    require_subband_count(subbands.len(), subband_infos.len())?;

    let mut data = Array3::<f32>::zeros((shape[0], shape[1], shape[2]));
    let step_f32 = step as f32;

    // Rice decode + dequantize (fused into one pass per subband).
    for (encoded, info) in subbands.iter().zip(subband_infos.iter()) {
        let num_values = validated_num_values(encoded, info)?;
        let coefficients = rice_decode_subband(&encoded.data, num_values, encoded.rice_k)?;
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

    Ok(data)
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
) -> CodecResult<Array3<f32>> {
    validate_shape(full_shape)?;
    validate_levels(levels)?;

    let k = factor.trailing_zeros() as usize;
    if k > levels {
        return Err(CodecError(format!(
            "jvol downsample level {k} (factor {factor}) exceeds available levels {levels}"
        )));
    }

    // Extent at each level (level 0 = full shape).
    let mut extents = Vec::with_capacity(levels + 1);
    extents.push(full_shape);
    for _ in 0..levels {
        let p = *extents
            .last()
            .ok_or_else(|| CodecError("jvol internal error: empty extents".to_string()))?;
        extents.push([p[0].div_ceil(2), p[1].div_ceil(2), p[2].div_ceil(2)]);
    }
    let small_shape = extents[k];

    let subband_infos = compute_subbands(full_shape, levels);
    require_subband_count(subbands.len(), subband_infos.len())?;

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
        let num_values = validated_num_values(encoded, info)?;
        let coefficients = rice_decode_subband(&encoded.data, num_values, encoded.rice_k)?;
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

    Ok(data)
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
        )
        .unwrap();
        let f32_dec = decode_lossy_f32(
            &res.subbands,
            shape,
            res.wavelet,
            res.levels,
            res.step,
            res.intercept,
            res.slope,
            JvolDtype::F32,
        )
        .unwrap();

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
        )
        .unwrap();
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
            )
            .unwrap();
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

    // --- Malicious/truncated-input hardening ---

    #[test]
    fn shape_overflow_is_rejected() {
        assert!(validate_shape([usize::MAX, 2, 2]).is_err());
    }

    #[test]
    fn shape_exceeding_voxel_cap_is_rejected() {
        assert!(validate_shape([1 << 20, 1 << 20, 1 << 20]).is_err());
    }

    #[test]
    fn shape_within_cap_is_accepted() {
        assert_eq!(validate_shape([4, 5, 6]).unwrap(), 120);
    }

    #[test]
    fn levels_exceeding_cap_is_rejected() {
        assert!(validate_levels(MAX_LEVELS + 1).is_err());
        assert!(validate_levels(MAX_LEVELS).is_ok());
    }

    #[test]
    fn decode_lossless_rejects_wrong_subband_count() {
        let subbands = vec![
            EncodedSubband {
                rice_k: 255,
                num_values: 8,
                data: vec![0u8; 8],
            },
            EncodedSubband {
                rice_k: 255,
                num_values: 8,
                data: vec![0u8; 8],
            },
        ];
        assert!(decode_lossless(&subbands, [2, 2, 2], JvolDtype::U8).is_err());
    }

    #[test]
    fn decode_lossless_rejects_absurd_shape() {
        let subbands = vec![EncodedSubband {
            rice_k: 255,
            num_values: 8,
            data: vec![0u8; 8],
        }];
        assert!(decode_lossless(&subbands, [1 << 20, 1 << 20, 1 << 20], JvolDtype::U8).is_err());
    }

    #[test]
    fn decode_lossless_rejects_truncated_subband_data() {
        // Metadata declares an 4x4x4 = 64-voxel volume, but the subband only
        // carries 4 bytes: the F64 raw-byte path must reject this instead of
        // indexing past the end of `sub.data`.
        let subbands = vec![EncodedSubband {
            rice_k: 255,
            num_values: 64,
            data: vec![0u8; 4],
        }];
        assert!(decode_lossless(&subbands, [4, 4, 4], JvolDtype::F64).is_err());
    }

    #[test]
    fn decode_lossy_rejects_subband_count_mismatch() {
        let shape = [8, 8, 8];
        let vol = smooth_volume(8);
        let res = encode_array(&vol.view(), 60, JvolDtype::F32);
        let mut bad_subbands = res.subbands.clone();
        bad_subbands.pop();
        assert!(decode_lossy(
            &bad_subbands,
            shape,
            res.wavelet,
            res.levels,
            res.step,
            res.intercept,
            res.slope,
            JvolDtype::F32,
        )
        .is_err());
    }

    #[test]
    fn decode_lossy_rejects_num_values_mismatch() {
        let shape = [8, 8, 8];
        let vol = smooth_volume(8);
        let res = encode_array(&vol.view(), 60, JvolDtype::F32);
        let mut bad_subbands = res.subbands.clone();
        bad_subbands[0].num_values += 1;
        assert!(decode_lossy(
            &bad_subbands,
            shape,
            res.wavelet,
            res.levels,
            res.step,
            res.intercept,
            res.slope,
            JvolDtype::F32,
        )
        .is_err());
    }

    #[test]
    fn decode_lossy_rejects_truncated_rice_data() {
        let shape = [16, 16, 16];
        let vol = smooth_volume(16);
        let res = encode_array(&vol.view(), 60, JvolDtype::F32);
        let mut bad_subbands = res.subbands.clone();
        // Truncate the largest subband's Rice-coded bytes so its declared
        // num_values can no longer be satisfied by the bitstream.
        let largest = bad_subbands
            .iter()
            .enumerate()
            .max_by_key(|(_, s)| s.data.len())
            .map(|(i, _)| i)
            .unwrap();
        let truncated_len = bad_subbands[largest].data.len() / 8;
        bad_subbands[largest].data.truncate(truncated_len);
        assert!(decode_lossy(
            &bad_subbands,
            shape,
            res.wavelet,
            res.levels,
            res.step,
            res.intercept,
            res.slope,
            JvolDtype::F32,
        )
        .is_err());
    }

    #[test]
    fn decode_downsampled_rejects_factor_exceeding_levels() {
        let shape = [16, 16, 16];
        let vol = smooth_volume(16);
        let res = encode_array(&vol.view(), 60, JvolDtype::F32);
        let huge_factor = 1usize << (res.levels + 4);
        assert!(decode_downsampled_f32(
            &res.subbands,
            shape,
            res.wavelet,
            res.levels,
            res.step,
            res.intercept,
            res.slope,
            huge_factor,
        )
        .is_err());
    }
}
