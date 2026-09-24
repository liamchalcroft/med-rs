//! gzip decoding and encoding for `.nii.gz`.
//!
//! Three kinds of gzip stream are handled:
//!
//! * **Standard single-member gzip** (what nibabel, FSL, ITK and dcm2niix
//!   write): decoded in one call to libdeflate into a buffer sized from the
//!   gzip trailer, falling back to a streaming decoder when the trailer is
//!   unreliable (multi-member streams, payloads over 4 GiB).
//! * **Block-indexed multi-member gzip** — Mgzip (`IG` extra subfield, as
//!   written by the Python `mgzip` package and [`save_mgzip`]) and BGZF
//!   (`BC` subfield). Each member records its own compressed size, so the
//!   members are located without decompressing and inflated in parallel on
//!   the rayon pool.
//! * **Other multi-member streams**, decoded sequentially.
//!
//! All three are valid gzip, so every reader can open files written by any of
//! the writers; only the decoding speed differs.
//!
//! [`save_mgzip`]: crate::nifti::save_mgzip

use crate::error::{Error, Result};
use flate2::bufread::MultiGzDecoder;
use gzp::deflate::{Gzip, Mgzip};
use gzp::par::compress::ParCompressBuilder;
use gzp::{Compression, ZWriter};
use libdeflater::{CompressionLvl, Compressor, DecompressionError, Decompressor};
use rayon::prelude::*;
use std::cell::RefCell;
use std::io::{Read, Write};

/// Upper bound on the deflate expansion ratio (1032:1 plus header slack),
/// used to reject size claims that no valid stream could satisfy.
const MAX_DEFLATE_RATIO: usize = 1032;

/// Payloads at least this large are compressed with the parallel encoder.
const PARALLEL_COMPRESS_THRESHOLD: usize = 4 << 20;

/// Uncompressed block size for Mgzip output.
pub(crate) const MGZIP_BLOCK_SIZE: usize = 1 << 20;

thread_local! {
    static DECOMPRESSOR: RefCell<Decompressor> = RefCell::new(Decompressor::new());
}

/// Whether `bytes` start with the gzip magic number.
pub(crate) fn is_gzip(bytes: &[u8]) -> bool {
    bytes.len() >= 2 && bytes[0] == 0x1f && bytes[1] == 0x8b
}

/// The block-indexed gzip dialect of a stream, if any.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BlockFormat {
    /// Mgzip: `IG` subfield holding the total member size.
    Mgzip,
    /// BGZF: `BC` subfield holding the total member size minus one.
    Bgzf,
}

/// Parse the block size from the extra field of the member starting at
/// `bytes[0]`. Returns the dialect and the total member size in bytes.
fn member_block_size(bytes: &[u8]) -> Option<(BlockFormat, usize)> {
    if bytes.len() < 12 || !is_gzip(bytes) || bytes[2] != 8 || bytes[3] & 0x04 == 0 {
        return None;
    }
    let xlen = usize::from(u16::from_le_bytes([bytes[10], bytes[11]]));
    let extra = bytes.get(12..12 + xlen)?;
    let mut i = 0;
    while i + 4 <= extra.len() {
        let (si1, si2) = (extra[i], extra[i + 1]);
        let len = usize::from(u16::from_le_bytes([extra[i + 2], extra[i + 3]]));
        let payload = extra.get(i + 4..i + 4 + len)?;
        match (si1, si2, len) {
            (b'I', b'G', 4) => {
                let size = u32::from_le_bytes([payload[0], payload[1], payload[2], payload[3]]);
                return Some((BlockFormat::Mgzip, size as usize));
            }
            (b'B', b'C', 2) => {
                let size = u16::from_le_bytes([payload[0], payload[1]]);
                return Some((BlockFormat::Bgzf, usize::from(size) + 1));
            }
            _ => {}
        }
        i += 4 + len;
    }
    None
}

/// Detect whether a stream is block-indexed from its first member.
pub(crate) fn block_format(bytes: &[u8]) -> Option<BlockFormat> {
    member_block_size(bytes).map(|(format, _)| format)
}

/// Split a block-indexed stream into `(start, len, uncompressed_len)` members.
///
/// Returns `None` if any member lacks a valid block size, in which case the
/// caller decodes sequentially. Trailing zero padding is tolerated.
fn block_members(bytes: &[u8]) -> Option<Vec<(usize, usize, usize)>> {
    let mut members = Vec::new();
    let mut pos = 0;
    while pos < bytes.len() {
        if bytes[pos..].iter().all(|&b| b == 0) {
            break;
        }
        let (_, size) = member_block_size(&bytes[pos..])?;
        // Smallest member: 12-byte header + extra + empty deflate + 8-byte trailer.
        if size < 20 || pos + size > bytes.len() {
            return None;
        }
        let trailer = &bytes[pos + size - 4..pos + size];
        let isize = u32::from_le_bytes([trailer[0], trailer[1], trailer[2], trailer[3]]) as usize;
        if isize > size.saturating_mul(MAX_DEFLATE_RATIO) {
            return None;
        }
        members.push((pos, size, isize));
        pos += size;
    }
    (!members.is_empty()).then_some(members)
}

/// Decompress an entire gzip stream (any of the three kinds).
pub(crate) fn decompress(bytes: &[u8]) -> Result<Vec<u8>> {
    if !is_gzip(bytes) {
        return Err(Error::Decompression("not a gzip stream".into()));
    }
    if let Some(members) = block_members(bytes) {
        if members.len() > 1 {
            return decompress_members(bytes, &members);
        }
    }
    decompress_single(bytes)
}

/// Inflate block members in parallel into one pre-sized buffer.
fn decompress_members(bytes: &[u8], members: &[(usize, usize, usize)]) -> Result<Vec<u8>> {
    let total: usize = members.iter().map(|m| m.2).sum();
    let mut out = vec![0u8; total];
    let mut slices = Vec::with_capacity(members.len());
    let mut rest: &mut [u8] = &mut out;
    for &(start, len, isize) in members {
        let (head, tail) = rest.split_at_mut(isize);
        slices.push((&bytes[start..start + len], head));
        rest = tail;
    }
    crate::parallel::install(|| {
        slices.into_par_iter().try_for_each(|(member, dst)| {
            DECOMPRESSOR.with(|d| {
                let written = d
                    .borrow_mut()
                    .gzip_decompress(member, dst)
                    .map_err(|e| Error::Decompression(format!("corrupt gzip block: {e}")))?;
                if written == dst.len() {
                    Ok(())
                } else {
                    Err(Error::Decompression(format!(
                        "gzip block decoded to {written} bytes, trailer says {}",
                        dst.len()
                    )))
                }
            })
        })
    })?;
    Ok(out)
}

/// Decode a (usually single-member) stream, trying libdeflate first.
fn decompress_single(bytes: &[u8]) -> Result<Vec<u8>> {
    let trailer = &bytes[bytes.len().saturating_sub(4)..];
    let isize = if trailer.len() == 4 {
        u32::from_le_bytes([trailer[0], trailer[1], trailer[2], trailer[3]]) as usize
    } else {
        0
    };
    let bound = bytes.len().saturating_mul(MAX_DEFLATE_RATIO);
    if isize > 0 && isize <= bound {
        let mut out = vec![0u8; isize];
        let result = DECOMPRESSOR.with(|d| d.borrow_mut().gzip_decompress(bytes, &mut out));
        match result {
            Ok(written) if written == isize => return Ok(out),
            // Wrong size hint (multi-member, >4 GiB, or trailing data):
            // decode sequentially instead.
            Ok(_) | Err(DecompressionError::InsufficientSpace | DecompressionError::BadData) => {}
        }
    }
    decompress_streaming(bytes, isize.min(bound))
}

/// Sequential decoder that handles any number of members.
fn decompress_streaming(bytes: &[u8], capacity_hint: usize) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(capacity_hint);
    let mut decoder = MultiGzDecoder::new(bytes);
    decoder
        .read_to_end(&mut out)
        .map_err(|e| Error::Decompression(format!("gzip stream is corrupt: {e}")))?;
    Ok(out)
}

/// Decompress just the first `limit` bytes of a gzip stream (for headers).
pub(crate) fn decompress_prefix(reader: impl Read, limit: usize) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(limit.min(1 << 20));
    MultiGzDecoder::new(std::io::BufReader::new(reader))
        .take(limit as u64)
        .read_to_end(&mut out)
        .map_err(|e| Error::Decompression(format!("gzip stream is corrupt: {e}")))?;
    Ok(out)
}

/// How to gzip-compress on save.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct GzipOptions {
    /// Compression level 0..=9 (libdeflate supports up to 12; clamped).
    pub level: u32,
    /// Write block-indexed Mgzip instead of a single member.
    pub mgzip: bool,
    /// Compression threads (0 = all available).
    pub threads: usize,
}

fn thread_count(threads: usize) -> usize {
    if threads == 0 {
        std::thread::available_parallelism().map_or(4, std::num::NonZeroUsize::get)
    } else {
        threads
    }
}

/// Compress the concatenation of `parts` as gzip, writing to `out`.
pub(crate) fn compress_to<W: Write + Send + 'static>(
    parts: &[&[u8]],
    out: W,
    opts: GzipOptions,
) -> Result<W> {
    let total: usize = parts.iter().map(|p| p.len()).sum();
    let level = opts.level.min(9);
    let threads = thread_count(opts.threads);

    if opts.mgzip {
        // gzp requires at least one worker thread.
        let mut writer = ParCompressBuilder::<Mgzip>::new()
            .compression_level(Compression::new(level))
            .num_threads(threads.max(1))
            .map_err(|e| gzp_err(&e))?
            .buffer_size(MGZIP_BLOCK_SIZE)
            .map_err(|e| gzp_err(&e))?
            .from_writer(out);
        for part in parts {
            writer.write_all(part)?;
        }
        return writer.finish().map_err(|e| gzp_err(&e));
    }

    if total >= PARALLEL_COMPRESS_THRESHOLD && threads > 1 {
        let mut writer = ParCompressBuilder::<Gzip>::new()
            .compression_level(Compression::new(level))
            .num_threads(threads)
            .map_err(|e| gzp_err(&e))?
            .from_writer(out);
        for part in parts {
            writer.write_all(part)?;
        }
        return writer.finish().map_err(|e| gzp_err(&e));
    }

    // Small payloads: a single libdeflate call.
    let mut joined = Vec::with_capacity(total);
    for part in parts {
        joined.extend_from_slice(part);
    }
    let lvl = CompressionLvl::new(level as i32).unwrap_or_default();
    let mut compressor = Compressor::new(lvl);
    let mut buf = vec![0u8; compressor.gzip_compress_bound(joined.len())];
    let n = compressor.gzip_compress(&joined, &mut buf).map_err(|e| {
        Error::Io(std::io::Error::other(format!(
            "gzip compression failed: {e}"
        )))
    })?;
    let mut out = out;
    out.write_all(&buf[..n])?;
    Ok(out)
}

fn gzp_err(e: &gzp::GzpError) -> Error {
    Error::Io(std::io::Error::other(format!(
        "gzip compression failed: {e}"
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use flate2::write::GzEncoder;

    fn payload(n: usize) -> Vec<u8> {
        (0..n).map(|i| ((i * 7919) % 251) as u8).collect()
    }

    fn gzip(data: &[u8]) -> Vec<u8> {
        let mut e = GzEncoder::new(Vec::new(), flate2::Compression::fast());
        e.write_all(data).unwrap();
        e.finish().unwrap()
    }

    #[test]
    fn single_member_roundtrip() {
        let data = payload(100_000);
        assert_eq!(decompress(&gzip(&data)).unwrap(), data);
    }

    #[test]
    fn plain_multi_member_and_trailing_zeros() {
        let data = payload(50_000);
        let mut multi = gzip(&data[..20_000]);
        multi.extend(gzip(&data[20_000..]));
        assert_eq!(decompress(&multi).unwrap(), data);
    }

    #[test]
    fn mgzip_roundtrip_is_detected_and_parallel() {
        let data = payload(3 * MGZIP_BLOCK_SIZE + 12_345);
        let opts = GzipOptions {
            level: 1,
            mgzip: true,
            threads: 2,
        };
        let out = compress_to(&[&data[..10], &data[10..]], Vec::new(), opts).unwrap();
        assert_eq!(block_format(&out), Some(BlockFormat::Mgzip));
        let members = block_members(&out).unwrap();
        assert_eq!(members.len(), 4);
        assert_eq!(decompress(&out).unwrap(), data);
        // Every gzip reader can still decode it sequentially.
        let mut seq = Vec::new();
        MultiGzDecoder::new(&out[..]).read_to_end(&mut seq).unwrap();
        assert_eq!(seq, data);
    }

    #[test]
    fn standard_outputs_are_single_member_gzip() {
        for n in [1000, PARALLEL_COMPRESS_THRESHOLD + 1] {
            let data = payload(n);
            let opts = GzipOptions {
                level: 1,
                mgzip: false,
                threads: 4,
            };
            let out = compress_to(&[&data], Vec::new(), opts).unwrap();
            assert_eq!(block_format(&out), None);
            assert_eq!(decompress(&out).unwrap(), data);
        }
    }

    #[test]
    fn corrupt_streams_error_without_panicking() {
        let data = payload(10_000);
        let mut gz = gzip(&data);
        let mid = gz.len() / 2;
        gz[mid] ^= 0xff;
        assert!(decompress(&gz).is_err());
        assert!(decompress(&gz[..gz.len() / 3]).is_err());
        assert!(decompress(b"not gzip").is_err());
        // A forged ISIZE trailer is reported as corruption without first
        // allocating the 4 GiB it claims.
        let mut bomb = gzip(&[0u8; 16]);
        let n = bomb.len();
        bomb[n - 4..].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(decompress(&bomb).is_err());
    }

    #[test]
    fn prefix_decoding() {
        let data = payload(10_000);
        let prefix = decompress_prefix(&gzip(&data)[..], 540).unwrap();
        assert_eq!(prefix, data[..540]);
    }
}
