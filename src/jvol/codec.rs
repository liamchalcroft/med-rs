//! Chunk encoders and decoders.
//!
//! Every chunk payload starts with a one-byte kind:
//!
//! | kind | payload after the kind byte                                           |
//! |------|-----------------------------------------------------------------------|
//! | 0    | constant: one stored element (lossless) or one `f32` (lossy)          |
//! | 1    | lossless, no filter: zstd frame of the element bytes                  |
//! | 2    | lossless, byte shuffle: zstd frame                                    |
//! | 3    | lossless, delta along x then byte shuffle: zstd frame                 |
//! | 4    | lossy wavelet: `min: f32`, `max: f32`, `width: u8`, `coef_len: u32`,  |
//! |      | coefficient zstd frame, then the background-mask zstd frame           |
//!
//! Element bytes are little-endian in Fortran order within the chunk. The
//! delta filter subtracts the previous element along x, treating each element
//! as an unsigned integer of its width (wrapping), which is lossless for every
//! datatype. Every zstd frame carries a content checksum.

use super::wavelet;
use crate::error::{Error, Result};

const KIND_CONSTANT: u8 = 0;
const KIND_PLAIN: u8 = 1;
const KIND_SHUFFLE: u8 = 2;
const KIND_DELTA_SHUFFLE: u8 = 3;
const KIND_WAVELET: u8 = 4;

fn corrupt(what: &str) -> Error {
    Error::InvalidFileFormat(format!("corrupt .jvol chunk: {what}"))
}

fn zstd_compress(data: &[u8], level: i32) -> Result<Vec<u8>> {
    let mut c = zstd::bulk::Compressor::new(level)?;
    c.include_checksum(true)?;
    c.include_contentsize(true)?;
    Ok(c.compress(data)?)
}

/// Decompress a frame that must hold exactly `len` bytes.
fn zstd_decompress(frame: &[u8], len: usize) -> Result<Vec<u8>> {
    let out = zstd::bulk::decompress(frame, len).map_err(|e| corrupt(&e.to_string()))?;
    if out.len() == len {
        Ok(out)
    } else {
        Err(corrupt("decompressed size mismatch"))
    }
}

/// Transpose `[element][byte]` into `[byte][element]`.
fn shuffle(data: &[u8], width: usize) -> Vec<u8> {
    fn run<const W: usize>(data: &[u8]) -> Vec<u8> {
        let n = data.len() / W;
        let mut out = vec![0u8; data.len()];
        for (i, element) in data.chunks_exact(W).enumerate() {
            for b in 0..W {
                out[b * n + i] = element[b];
            }
        }
        out
    }
    match width {
        2 => run::<2>(data),
        4 => run::<4>(data),
        8 => run::<8>(data),
        _ => data.to_vec(),
    }
}

fn unshuffle(data: &[u8], width: usize) -> Vec<u8> {
    fn run<const W: usize>(data: &[u8]) -> Vec<u8> {
        let n = data.len() / W;
        let mut out = vec![0u8; data.len()];
        for (i, element) in out.chunks_exact_mut(W).enumerate() {
            for b in 0..W {
                element[b] = data[b * n + i];
            }
        }
        out
    }
    match width {
        2 => run::<2>(data),
        4 => run::<4>(data),
        8 => run::<8>(data),
        _ => data.to_vec(),
    }
}

fn read_uint(bytes: &[u8]) -> u64 {
    let mut v = [0u8; 8];
    v[..bytes.len()].copy_from_slice(bytes);
    u64::from_le_bytes(v)
}

/// Apply `f(previous, current) -> new` along each row of `row` elements,
/// treating elements as little-endian unsigned integers of their width.
macro_rules! delta_rows {
    ($data:expr, $width:expr, $row:expr, |$prev:ident, $cur:ident| $update:expr, $next_prev:expr) => {
        match $width {
            1 => delta_rows!(@ u8, $data, $row, |$prev, $cur| $update, $next_prev),
            2 => delta_rows!(@ u16, $data, $row, |$prev, $cur| $update, $next_prev),
            4 => delta_rows!(@ u32, $data, $row, |$prev, $cur| $update, $next_prev),
            _ => delta_rows!(@ u64, $data, $row, |$prev, $cur| $update, $next_prev),
        }
    };
    (@ $t:ty, $data:expr, $row:expr, |$prev:ident, $cur:ident| $update:expr, $next_prev:expr) => {{
        const W: usize = std::mem::size_of::<$t>();
        for r in $data.chunks_exact_mut($row * W) {
            let mut $prev: $t = 0;
            for e in r.chunks_exact_mut(W) {
                let $cur = <$t>::from_le_bytes(e.try_into().unwrap_or([0; W]));
                let new: $t = $update;
                e.copy_from_slice(&new.to_le_bytes());
                $prev = $next_prev(new, $cur);
            }
        }
    }};
}

/// Replace each element with its difference from the previous one along x.
fn delta_encode(data: &mut [u8], width: usize, row: usize) {
    delta_rows!(
        data,
        width,
        row,
        |prev, cur| cur.wrapping_sub(prev),
        |_new, cur| cur
    );
}

fn delta_decode(data: &mut [u8], width: usize, row: usize) {
    delta_rows!(
        data,
        width,
        row,
        |prev, cur| prev.wrapping_add(cur),
        |new, _cur| new
    );
}

fn is_constant(data: &[u8], width: usize) -> bool {
    let first = &data[..width];
    data.chunks_exact(width).all(|e| e == first)
}

/// Encode little-endian element bytes of a chunk whose rows along x have
/// `row` elements, keeping the smallest of the candidate filters.
pub(super) fn encode_lossless(
    data: &[u8],
    width: usize,
    row: usize,
    level: i32,
) -> Result<Vec<u8>> {
    if is_constant(data, width) {
        let mut out = vec![KIND_CONSTANT];
        out.extend_from_slice(&data[..width]);
        return Ok(out);
    }
    let mut best = vec![KIND_PLAIN];
    best.extend(zstd_compress(data, level)?);
    if width > 1 {
        let mut candidate = vec![KIND_SHUFFLE];
        candidate.extend(zstd_compress(&shuffle(data, width), level)?);
        if candidate.len() < best.len() {
            best = candidate;
        }
    }
    let mut delta = data.to_vec();
    delta_encode(&mut delta, width, row);
    let mut candidate = vec![KIND_DELTA_SHUFFLE];
    candidate.extend(zstd_compress(&shuffle(&delta, width), level)?);
    if candidate.len() < best.len() {
        best = candidate;
    }
    Ok(best)
}

/// Decode a lossless chunk of `len` bytes.
pub(super) fn decode_lossless(
    payload: &[u8],
    len: usize,
    width: usize,
    row: usize,
) -> Result<Vec<u8>> {
    let (&kind, body) = payload
        .split_first()
        .ok_or_else(|| corrupt("empty payload"))?;
    match kind {
        KIND_CONSTANT => {
            if body.len() != width {
                return Err(corrupt("bad constant"));
            }
            Ok(body.repeat(len / width))
        }
        KIND_PLAIN => zstd_decompress(body, len),
        KIND_SHUFFLE => Ok(unshuffle(&zstd_decompress(body, len)?, width)),
        KIND_DELTA_SHUFFLE => {
            let mut data = unshuffle(&zstd_decompress(body, len)?, width);
            delta_decode(&mut data, width, row);
            Ok(data)
        }
        _ => Err(corrupt(&format!("unknown lossless kind {kind}"))),
    }
}

/// Encode a lossy chunk of finite values.
pub(super) fn encode_lossy(
    values: &[f32],
    shape: [usize; 3],
    levels: usize,
    step: f32,
    level: i32,
) -> Result<Vec<u8>> {
    let (min, max) = values
        .iter()
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), &v| {
            (lo.min(v), hi.max(v))
        });
    if min == max {
        let mut out = vec![KIND_CONSTANT];
        out.extend_from_slice(&min.to_le_bytes());
        return Ok(out);
    }
    let mut coef = values.to_vec();
    wavelet::forward(&mut coef, shape, levels);
    let inv = 1.0 / step;
    let order = wavelet::subband_order(shape, levels);
    let q: Vec<u32> = order
        .iter()
        .map(|&i| {
            let c = coef[i as usize];
            // Dead-zone quantization, then zigzag to unsigned.
            let v = (c * inv).trunc() as i32;
            ((v << 1) ^ (v >> 31)) as u32
        })
        .collect();
    let peak = q.iter().copied().max().unwrap_or(0);
    let width: usize = if peak <= 0xFF {
        1
    } else if peak <= 0xFFFF {
        2
    } else {
        4
    };
    let mut packed = Vec::with_capacity(q.len() * width);
    for v in &q {
        packed.extend_from_slice(&v.to_le_bytes()[..width]);
    }
    let coef_frame = zstd_compress(&shuffle(&packed, width), level)?;
    // Voxels at the chunk minimum (typically background) are restored exactly.
    let mut mask = vec![0u8; values.len().div_ceil(8)];
    for (i, &v) in values.iter().enumerate() {
        if v == min {
            mask[i / 8] |= 1 << (i % 8);
        }
    }
    let mask_frame = zstd_compress(&mask, level)?;

    let mut out = Vec::with_capacity(14 + coef_frame.len() + mask_frame.len());
    out.push(KIND_WAVELET);
    out.extend_from_slice(&min.to_le_bytes());
    out.extend_from_slice(&max.to_le_bytes());
    out.push(width as u8);
    out.extend_from_slice(
        &u32::try_from(coef_frame.len())
            .map_err(|_| corrupt("chunk too large"))?
            .to_le_bytes(),
    );
    out.extend(coef_frame);
    out.extend(mask_frame);
    Ok(out)
}

/// Decode a lossy chunk of `shape` voxels.
pub(super) fn decode_lossy(
    payload: &[u8],
    shape: [usize; 3],
    levels: usize,
    step: f32,
) -> Result<Vec<f32>> {
    let n: usize = shape.iter().product();
    let (&kind, body) = payload
        .split_first()
        .ok_or_else(|| corrupt("empty payload"))?;
    match kind {
        KIND_CONSTANT => {
            let bytes: [u8; 4] = body.try_into().map_err(|_| corrupt("bad constant"))?;
            Ok(vec![f32::from_le_bytes(bytes); n])
        }
        KIND_WAVELET => {
            if body.len() < 13 {
                return Err(corrupt("truncated header"));
            }
            let f = |o: usize| f32::from_le_bytes([body[o], body[o + 1], body[o + 2], body[o + 3]]);
            let (min, max) = (f(0), f(4));
            let width = usize::from(body[8]);
            let valid = [1, 2, 4].contains(&width) && min <= max;
            if !valid {
                return Err(corrupt("bad parameters"));
            }
            let coef_len = u32::from_le_bytes([body[9], body[10], body[11], body[12]]) as usize;
            let frames = &body[13..];
            if coef_len > frames.len() {
                return Err(corrupt("truncated coefficients"));
            }
            let packed = unshuffle(&zstd_decompress(&frames[..coef_len], n * width)?, width);
            let mask = zstd_decompress(&frames[coef_len..], n.div_ceil(8))?;
            let order = wavelet::subband_order(shape, levels);
            let mut coef = vec![0f32; n];
            for (b, &i) in packed.chunks_exact(width).zip(&order) {
                let z = read_uint(b) as u32;
                let v = ((z >> 1) as i32) ^ -((z & 1) as i32);
                // Reconstruct at the middle of the quantization bin.
                if v != 0 {
                    coef[i as usize] = (v as f32 + 0.5 * v.signum() as f32) * step;
                }
            }
            wavelet::inverse(&mut coef, shape, levels);
            for (i, c) in coef.iter_mut().enumerate() {
                *c = if mask[i / 8] >> (i % 8) & 1 == 1 {
                    min
                } else {
                    c.clamp(min, max)
                };
            }
            Ok(coef)
        }
        _ => Err(corrupt(&format!("unknown lossy kind {kind}"))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lossless_roundtrip_every_width_and_filter() {
        for width in [1, 2, 4, 8] {
            let row = 7;
            let n = row * 5 * 3;
            let data: Vec<u8> = (0..n * width).map(|i| ((i * 131) % 251) as u8).collect();
            let enc = encode_lossless(&data, width, row, 3).unwrap();
            assert_eq!(decode_lossless(&enc, data.len(), width, row).unwrap(), data);
            let smooth: Vec<u8> = (0..n as u64)
                .flat_map(|i| (1000 + 3 * i).to_le_bytes()[..width].to_vec())
                .collect();
            let enc = encode_lossless(&smooth, width, row, 3).unwrap();
            assert_eq!(
                decode_lossless(&enc, smooth.len(), width, row).unwrap(),
                smooth
            );
        }
        let constant = vec![9u8; 64];
        let enc = encode_lossless(&constant, 2, 4, 3).unwrap();
        assert_eq!(enc.len(), 3);
        assert_eq!(decode_lossless(&enc, 64, 2, 4).unwrap(), constant);
    }

    #[test]
    fn corrupt_payloads_are_errors() {
        let data: Vec<u8> = (0..400).map(|i| (i % 7) as u8).collect();
        let enc = encode_lossless(&data, 4, 10, 3).unwrap();
        for i in [1, enc.len() / 2, enc.len() - 1] {
            let mut bad = enc.clone();
            bad[i] ^= 0x55;
            assert!(
                decode_lossless(&bad, data.len(), 4, 10).is_err(),
                "flip at {i}"
            );
        }
        assert!(decode_lossless(&[], 4, 4, 1).is_err());
        assert!(decode_lossless(&[9], 4, 4, 1).is_err());
        assert!(decode_lossy(&[4, 0, 0], [2, 2, 2], 1, 1.0).is_err());
    }

    #[test]
    fn lossy_error_bounded_range_clamped_and_background_exact() {
        let shape = [24, 20, 16];
        let n: usize = shape.iter().product();
        let values: Vec<f32> = (0..n)
            .map(|i| {
                let (x, y, z) = (i % 24, (i / 24) % 20, i / 480);
                if x < 4 {
                    0.0
                } else {
                    100.0 + 20.0 * ((x as f32) * 0.3).sin() + y as f32 + 0.5 * z as f32
                }
            })
            .collect();
        let step = 0.5;
        let enc = encode_lossy(&values, shape, 3, step, 3).unwrap();
        let dec = decode_lossy(&enc, shape, 3, step).unwrap();
        let rmse = (values
            .iter()
            .zip(&dec)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            / n as f32)
            .sqrt();
        assert!(rmse < step, "{rmse}");
        let hi = values.iter().copied().fold(0.0, f32::max);
        assert!(dec.iter().all(|v| (0.0..=hi).contains(v)));
        for (a, b) in values.iter().zip(&dec) {
            if *a == 0.0 {
                assert_eq!(*b, 0.0);
            }
        }
        assert!(enc.len() < n);
    }
}
