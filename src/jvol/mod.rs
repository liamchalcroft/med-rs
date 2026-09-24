//! `.jvol`: a chunked, compressed volume format.
//!
//! A `.jvol` file stores a complete `NIfTI` header (every field and
//! extension) followed by the voxel data split into independently compressed
//! chunks. Because chunks are independent, files are encoded and decoded in
//! parallel, and [`load_cropped`](crate::nifti::load_cropped) or the
//! [`FastLoader`](crate::FastLoader) decode only the chunks a region touches.
//!
//! Two codecs are available:
//!
//! * **Lossless** ([`JvolOptions::lossless`], the default): exact for every
//!   datatype. Each chunk keeps whichever filter (none, byte shuffle, or delta
//!   and byte shuffle) compresses best with zstd.
//! * **Lossy** ([`JvolOptions::lossy`]): a 3D CDF 9/7 wavelet transform with
//!   uniform quantization, for intensity images. Decoded images are `f32` in
//!   scaled units. Values are clamped to each chunk's original range, and
//!   voxels equal to a chunk's minimum (usually background) are restored
//!   exactly. Label maps should use the lossless codec.
//!
//! The name and the wavelet approach come from Fernando Pérez-García's
//! [jvol](https://github.com/fepegar/jvol) project; the file format below is
//! specific to medrs.
//!
//! # File format (version 1)
//!
//! All integers are little-endian.
//!
//! | offset | size | field                                                     |
//! |--------|------|-----------------------------------------------------------|
//! | 0      | 8    | magic `\x89JVL\r\n\x1a\n`                                 |
//! | 8      | 2    | format version (1)                                        |
//! | 10     | 1    | codec: 0 lossless, 1 lossy                                |
//! | 11     | 1    | wavelet levels (lossy) or 0                               |
//! | 12     | 12   | chunk shape, three `u32`                                  |
//! | 24     | 4    | `h`: length of the header block                           |
//! | 28     | 4    | quantization step as `f32` (lossy) or 0                   |
//! | 32     | 1    | flags: bit 0 set if the image was NIfTI-1 (restored on load) |
//! | 33     | 3    | reserved (0)                                              |
//! | 36     | 4    | CRC-32 of bytes 0..36, the header block, and the table    |
//! | 40     | `h`  | `NIfTI` header block: a NIfTI-2 two-file (`ni2`) header with its extensions, so every field keeps full precision |
//! | …      | 16·n | chunk table: per chunk, file offset `u64`, length `u32`, 4 reserved bytes |
//! | …      |      | chunk payloads (see below)                                |
//!
//! Chunks tile the first three axes; images with more axes have one set of
//! chunks per volume. Chunks are ordered with x fastest, then y, z, and volume.
//! Edge chunks are smaller. Voxels within a chunk are in Fortran order.
//! Each payload starts with a kind byte:
//!
//! | kind | payload                                                               |
//! |------|-----------------------------------------------------------------------|
//! | 0    | constant: one stored element (lossless) or one `f32` (lossy)          |
//! | 1    | zstd frame of the element bytes                                       |
//! | 2    | zstd frame of the byte-shuffled element bytes                         |
//! | 3    | zstd frame of the byte-shuffled element deltas along x                |
//! | 4    | `min f32`, `max f32`, `width u8`, `n u32`, `n`-byte zstd frame of the quantized wavelet coefficients (zigzag, `width` bytes each, byte-shuffled), zstd frame of the minimum-value bitmask |
//!
//! Every zstd frame carries a content checksum, so corruption is detected.

mod codec;
mod wavelet;

use crate::error::{Error, Result};
use crate::nifti::element::{fortran_from_vec, ArrayData};
use crate::nifti::header::{translation, FileLayout, NiftiHeader};
use crate::nifti::image::Buffer;
use crate::nifti::io::{open_bytes, spatial_region, write_atomic};
use crate::nifti::{DataType, NiftiImage, NiftiVersion};
use crate::transforms::geometry::split_shape;
use rayon::prelude::*;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::Arc;

const MAGIC: &[u8; 8] = b"\x89JVL\r\n\x1a\n";
const VERSION: u16 = 1;
const PREAMBLE: usize = 40;
const TABLE_ENTRY: usize = 16;
/// Largest chunk (in voxels) accepted when reading.
const MAX_CHUNK_VOXELS: usize = 1 << 24;
/// Largest image (in voxels) accepted when reading.
const MAX_IMAGE_VOXELS: usize = 1 << 34;

/// How to encode a `.jvol` file.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct JvolOptions {
    lossy: Option<u8>,
    chunk_shape: [usize; 3],
    level: i32,
}

impl Default for JvolOptions {
    fn default() -> Self {
        Self::lossless()
    }
}

impl JvolOptions {
    /// Exact compression of the stored values (the default).
    pub fn lossless() -> Self {
        Self {
            lossy: None,
            chunk_shape: [64, 64, 64],
            level: 5,
        }
    }

    /// Wavelet compression with `quality` from 1 (smallest) to 100 (most
    /// accurate). The quantization step is `0.8 * 0.0025^((quality - 1) / 99)`
    /// of the image's robust intensity range (0.1st to 99.9th percentile):
    /// quality 100 quantizes at 0.2% of the range, quality 60 at about 2%.
    pub fn lossy(quality: u8) -> Result<Self> {
        if !(1..=100).contains(&quality) {
            return Err(Error::InvalidArgument(format!(
                "jvol quality must be between 1 and 100, got {quality}"
            )));
        }
        Ok(Self {
            lossy: Some(quality),
            chunk_shape: [64, 64, 64],
            level: 9,
        })
    }

    /// Chunk shape along the first three axes (default 64³). Smaller chunks
    /// make small crops cheaper; larger chunks compress slightly better.
    pub fn with_chunk_shape(mut self, chunk_shape: [usize; 3]) -> Result<Self> {
        let voxels = chunk_shape
            .iter()
            .try_fold(1usize, |a, &d| a.checked_mul(d));
        if chunk_shape.contains(&0) || voxels.is_none_or(|v| v > MAX_CHUNK_VOXELS) {
            return Err(Error::InvalidArgument(format!(
                "chunk shape must be positive with at most {MAX_CHUNK_VOXELS} voxels, got {chunk_shape:?}"
            )));
        }
        self.chunk_shape = chunk_shape;
        Ok(self)
    }

    /// zstd compression level (1 to 19; default 5 lossless, 9 lossy).
    /// Decoding speed does not depend on the level.
    pub fn with_level(mut self, level: i32) -> Result<Self> {
        if !(1..=19).contains(&level) {
            return Err(Error::InvalidArgument(format!(
                "zstd level must be between 1 and 19, got {level}"
            )));
        }
        self.level = level;
        Ok(self)
    }

    /// Whether this is the lossy codec.
    pub fn is_lossy(&self) -> bool {
        self.lossy.is_some()
    }
}

/// Wavelet levels used by the lossy codec.
const LOSSY_LEVELS: usize = 4;

/// Chunk layout of an image.
#[derive(Debug, Clone, Copy)]
struct Grid {
    spatial: [usize; 3],
    volumes: usize,
    chunk: [usize; 3],
    counts: [usize; 3],
}

impl Grid {
    fn new(shape: &[usize], chunk: [usize; 3]) -> Result<Self> {
        let (spatial, volumes) = split_shape(shape);
        let counts: [usize; 3] = std::array::from_fn(|i| spatial[i].div_ceil(chunk[i]));
        if counts
            .iter()
            .try_fold(volumes, |a, &c| a.checked_mul(c))
            .is_none_or(|n| n > MAX_IMAGE_VOXELS)
        {
            return Err(Error::InvalidDimensions("too many chunks".into()));
        }
        Ok(Self {
            spatial,
            volumes,
            chunk,
            counts,
        })
    }

    fn len(&self) -> usize {
        self.counts.iter().product::<usize>() * self.volumes
    }

    /// Volume index, start, and shape of chunk `i`.
    fn chunk(&self, i: usize) -> (usize, [usize; 3], [usize; 3]) {
        let per_volume: usize = self.counts.iter().product();
        let (volume, mut rest) = (i / per_volume, i % per_volume);
        let mut start = [0; 3];
        let mut shape = [0; 3];
        for axis in 0..3 {
            let c = rest % self.counts[axis];
            rest /= self.counts[axis];
            start[axis] = c * self.chunk[axis];
            shape[axis] = self.chunk[axis].min(self.spatial[axis] - start[axis]);
        }
        (volume, start, shape)
    }

    /// Index of the chunk at `coords` (chunk units) in `volume`.
    fn index(&self, volume: usize, coords: [usize; 3]) -> usize {
        let per_volume: usize = self.counts.iter().product();
        volume * per_volume + coords[0] + self.counts[0] * (coords[1] + self.counts[1] * coords[2])
    }
}

/// Copy the block at `start`/`shape` of volume `volume` out of `src`, which
/// holds `volumes` Fortran-ordered volumes of `spatial` shape.
fn gather<T: Copy>(
    src: &[T],
    spatial: [usize; 3],
    volume: usize,
    start: [usize; 3],
    shape: [usize; 3],
) -> Vec<T> {
    let (nx, ny) = (spatial[0], spatial[1]);
    let base = volume * nx * ny * spatial[2];
    let mut out = Vec::with_capacity(shape.iter().product());
    for z in 0..shape[2] {
        for y in 0..shape[1] {
            let row = base + start[0] + nx * (start[1] + y + ny * (start[2] + z));
            out.extend_from_slice(&src[row..row + shape[0]]);
        }
    }
    out
}

/// Robust intensity range: the 0.1st and 99.9th percentiles, or the full range
/// if they coincide.
fn robust_range(values: &[f32]) -> (f32, f32) {
    const BINS: usize = 4096;
    let (lo, hi) = crate::parallel::install(|| {
        values
            .par_chunks(1 << 16)
            .map(|c| {
                c.iter()
                    .fold((f32::INFINITY, f32::NEG_INFINITY), |(a, b), &v| {
                        (a.min(v), b.max(v))
                    })
            })
            .reduce(
                || (f32::INFINITY, f32::NEG_INFINITY),
                |a, b| (a.0.min(b.0), a.1.max(b.1)),
            )
    });
    if hi <= lo {
        return (lo, hi);
    }
    let scale = (BINS - 1) as f64 / f64::from(hi - lo);
    let bin = |v: f32| (f64::from(v - lo) * scale) as usize;
    let mut hist = vec![0usize; BINS];
    for &v in values {
        hist[bin(v).min(BINS - 1)] += 1;
    }
    let n = values.len();
    let cut = n / 1000;
    let mut acc = 0;
    let low = hist.iter().position(|&c| {
        acc += c;
        acc > cut
    });
    acc = 0;
    let high = hist.iter().rposition(|&c| {
        acc += c;
        acc > cut
    });
    match (low, high) {
        (Some(a), Some(b)) if b > a => {
            let width = f64::from(hi - lo) / (BINS - 1) as f64;
            (
                lo + (a as f64 * width) as f32,
                lo + (b as f64 * width) as f32,
            )
        }
        _ => (lo, hi),
    }
}

/// Save an image as `.jvol` (written to a temporary file and renamed into
/// place).
pub fn save<P: AsRef<Path>>(image: &NiftiImage, path: P, options: &JvolOptions) -> Result<()> {
    let path = path.as_ref();
    let grid = Grid::new(image.shape(), options.chunk_shape)?;
    let mut header = image.header().prepared_for_write(FileLayout::Pair);
    let was_nifti1 = header.version == NiftiVersion::Nifti1;
    header.version = NiftiVersion::Nifti2;
    header.validate()?;
    let header_block = header.encode(FileLayout::Pair)?;

    let (payloads, step) = match options.lossy {
        None => {
            let width = image.dtype().byte_size();
            let bytes = image.data_bytes_le()?;
            let payloads = crate::parallel::install(|| {
                (0..grid.len())
                    .into_par_iter()
                    .map(|i| {
                        let (volume, start, shape) = grid.chunk(i);
                        let block = gather_bytes(&bytes, width, &grid, volume, start, shape);
                        codec::encode_lossless(&block, width, shape[0], options.level)
                    })
                    .collect::<Result<Vec<_>>>()
            })?;
            (payloads, 0.0)
        }
        Some(quality) => {
            let values = image.f32_values()?;
            if values.iter().any(|v| !v.is_finite()) {
                return Err(Error::InvalidData(
                    "lossy .jvol requires finite values; use the lossless codec for images with NaN or infinity"
                        .into(),
                ));
            }
            let (lo, hi) = robust_range(&values);
            let fraction = 0.8 * 0.0025f64.powf(f64::from(quality - 1) / 99.0);
            let range = f64::from(hi - lo);
            let step = if range > 0.0 {
                (range * fraction) as f32
            } else {
                1.0
            };
            let payloads = crate::parallel::install(|| {
                (0..grid.len())
                    .into_par_iter()
                    .map(|i| {
                        let (volume, start, shape) = grid.chunk(i);
                        let block = gather(&values, grid.spatial, volume, start, shape);
                        codec::encode_lossy(&block, shape, LOSSY_LEVELS, step, options.level)
                    })
                    .collect::<Result<Vec<_>>>()
            })?;
            (payloads, step)
        }
    };

    let mut preamble = [0u8; PREAMBLE];
    preamble[..8].copy_from_slice(MAGIC);
    preamble[8..10].copy_from_slice(&VERSION.to_le_bytes());
    preamble[10] = u8::from(options.lossy.is_some());
    preamble[11] = if options.lossy.is_some() {
        LOSSY_LEVELS as u8
    } else {
        0
    };
    for (i, &c) in options.chunk_shape.iter().enumerate() {
        preamble[12 + 4 * i..16 + 4 * i].copy_from_slice(&(c as u32).to_le_bytes());
    }
    let header_len = u32::try_from(header_block.len())
        .map_err(|_| Error::InvalidArgument("header extensions are too large".into()))?;
    preamble[24..28].copy_from_slice(&header_len.to_le_bytes());
    preamble[28..32].copy_from_slice(&step.to_le_bytes());
    preamble[32] = u8::from(was_nifti1);

    let mut table = Vec::with_capacity(payloads.len() * TABLE_ENTRY);
    let mut offset = (PREAMBLE + header_block.len() + payloads.len() * TABLE_ENTRY) as u64;
    for p in &payloads {
        let len =
            u32::try_from(p.len()).map_err(|_| Error::InvalidArgument("chunk too large".into()))?;
        table.extend_from_slice(&offset.to_le_bytes());
        table.extend_from_slice(&len.to_le_bytes());
        table.extend_from_slice(&[0; 4]);
        offset += u64::from(len);
    }
    let crc = crc(&preamble[..36], &header_block, &table);
    preamble[36..40].copy_from_slice(&crc.to_le_bytes());

    write_atomic(path, |file| {
        let mut w = BufWriter::with_capacity(1 << 20, file);
        w.write_all(&preamble)?;
        w.write_all(&header_block)?;
        w.write_all(&table)?;
        for p in &payloads {
            w.write_all(p)?;
        }
        w.into_inner().map_err(|e| Error::Io(e.into_error()))?;
        Ok(())
    })
}

fn gather_bytes(
    bytes: &[u8],
    width: usize,
    grid: &Grid,
    volume: usize,
    start: [usize; 3],
    shape: [usize; 3],
) -> Vec<u8> {
    let (nx, ny) = (grid.spatial[0], grid.spatial[1]);
    let base = volume * nx * ny * grid.spatial[2];
    let mut out = Vec::with_capacity(shape.iter().product::<usize>() * width);
    for z in 0..shape[2] {
        for y in 0..shape[1] {
            let row = (base + start[0] + nx * (start[1] + y + ny * (start[2] + z))) * width;
            out.extend_from_slice(&bytes[row..row + shape[0] * width]);
        }
    }
    out
}

fn crc(preamble: &[u8], header: &[u8], table: &[u8]) -> u32 {
    let mut c = libdeflater::Crc::new();
    c.update(preamble);
    c.update(header);
    c.update(table);
    c.sum()
}

/// A `.jvol` file opened for reading. Regions decode only the chunks they
/// overlap.
pub struct Reader {
    buf: Buffer,
    header: NiftiHeader,
    grid: Grid,
    lossy: bool,
    levels: usize,
    step: f32,
    table: Vec<(usize, usize)>,
}

impl std::fmt::Debug for Reader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Reader")
            .field("shape", &self.header.shape())
            .field("chunk_shape", &self.grid.chunk)
            .field("lossy", &self.lossy)
            .finish_non_exhaustive()
    }
}

fn invalid(path: &Path, what: &str) -> Error {
    Error::InvalidFileFormat(format!("{}: {what}", path.display()))
}

impl Reader {
    /// Open a `.jvol` file, validating its header and chunk table.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        let buf = open_bytes(path)?;
        if buf.len() < PREAMBLE || &buf[..8] != MAGIC {
            return Err(invalid(path, "not a .jvol file (bad magic number)"));
        }
        let u32_at = |o: usize| u32::from_le_bytes([buf[o], buf[o + 1], buf[o + 2], buf[o + 3]]);
        let version = u16::from_le_bytes([buf[8], buf[9]]);
        if version != VERSION {
            return Err(invalid(
                path,
                &format!("unsupported .jvol version {version}"),
            ));
        }
        let lossy = match buf[10] {
            0 => false,
            1 => true,
            c => return Err(invalid(path, &format!("unknown codec {c}"))),
        };
        let levels = usize::from(buf[11]);
        let chunk: [usize; 3] = std::array::from_fn(|i| u32_at(12 + 4 * i) as usize);
        if chunk.contains(&0) || chunk.iter().product::<usize>() > MAX_CHUNK_VOXELS {
            return Err(invalid(path, "invalid chunk shape"));
        }
        let header_len = u32_at(24) as usize;
        let step = f32::from_le_bytes(
            buf[28..32]
                .try_into()
                .map_err(|_| invalid(path, "truncated"))?,
        );
        if lossy && !(step.is_finite() && step > 0.0) {
            return Err(invalid(path, "invalid quantization step"));
        }
        let header_end = PREAMBLE
            .checked_add(header_len)
            .filter(|&e| e <= buf.len())
            .ok_or_else(|| invalid(path, "truncated header"))?;
        let (mut header, _) = NiftiHeader::parse(&buf[PREAMBLE..header_end])
            .map_err(|e| invalid(path, &format!("embedded header: {e}")))?;
        if buf[32] & 1 == 1 {
            header.version = NiftiVersion::Nifti1;
        }
        if lossy {
            header.datatype = DataType::Float32;
            header.scl_slope = 1.0;
            header.scl_inter = 0.0;
        }
        let grid = Grid::new(&header.shape(), chunk)?;
        if header.num_voxels() > MAX_IMAGE_VOXELS {
            return Err(invalid(path, "image is too large"));
        }
        let table_end = grid
            .len()
            .checked_mul(TABLE_ENTRY)
            .and_then(|t| t.checked_add(header_end))
            .filter(|&e| e <= buf.len())
            .ok_or_else(|| invalid(path, "truncated chunk table"))?;
        let expected = crc(
            &buf[..36],
            &buf[PREAMBLE..header_end],
            &buf[header_end..table_end],
        );
        if expected != u32_at(36) {
            return Err(invalid(path, "header checksum mismatch (file is corrupt)"));
        }
        let table = buf[header_end..table_end]
            .chunks_exact(TABLE_ENTRY)
            .map(|e| {
                let offset = u64::from_le_bytes(e[..8].try_into().unwrap_or([0; 8]));
                let len = u32::from_le_bytes(e[8..12].try_into().unwrap_or([0; 4])) as usize;
                usize::try_from(offset)
                    .ok()
                    .filter(|&o| {
                        o >= table_end && o.checked_add(len).is_some_and(|end| end <= buf.len())
                    })
                    .map(|o| (o, len))
                    .ok_or_else(|| invalid(path, "chunk outside the file"))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            buf,
            header,
            grid,
            lossy,
            levels,
            step,
            table,
        })
    }

    /// The header of the decoded image (for lossy files: `f32` with identity
    /// scaling).
    pub fn header(&self) -> &NiftiHeader {
        &self.header
    }

    /// Chunk shape along the first three axes.
    pub fn chunk_shape(&self) -> [usize; 3] {
        self.grid.chunk
    }

    /// Whether the file uses the lossy codec.
    pub fn is_lossy(&self) -> bool {
        self.lossy
    }

    /// Decode the whole image.
    pub fn image(&self) -> Result<NiftiImage> {
        self.region([0; 3], self.grid.spatial)
    }

    /// Decode `shape` voxels at `offset` along the first three axes.
    pub fn region(&self, offset: [usize; 3], shape: [usize; 3]) -> Result<NiftiImage> {
        let full = self.header.shape();
        let (_, size) = spatial_region(&full, offset, shape)?;
        let out_spatial: [usize; 3] = std::array::from_fn(|i| size.get(i).copied().unwrap_or(1));
        let first: [usize; 3] = std::array::from_fn(|i| offset[i] / self.grid.chunk[i]);
        let last: [usize; 3] =
            std::array::from_fn(|i| (offset[i] + out_spatial[i] - 1) / self.grid.chunk[i]);
        let mut wanted = Vec::new();
        for v in 0..self.grid.volumes {
            for z in first[2]..=last[2] {
                for y in first[1]..=last[1] {
                    for x in first[0]..=last[0] {
                        wanted.push(self.grid.index(v, [x, y, z]));
                    }
                }
            }
        }
        let width = self.header.datatype.byte_size();
        let decoded = crate::parallel::install(|| {
            wanted
                .par_iter()
                .map(|&i| self.decode_chunk(i).map(|b| (i, b)))
                .collect::<Result<Vec<_>>>()
        })?;
        let out_voxels: usize = out_spatial.iter().product();
        let mut out = vec![0u8; out_voxels * self.grid.volumes * width];
        for (i, block) in decoded {
            let (volume, start, cshape) = self.grid.chunk(i);
            // Overlap of this chunk with the requested region, per axis.
            let lo: [usize; 3] = std::array::from_fn(|a| start[a].max(offset[a]));
            let hi: [usize; 3] =
                std::array::from_fn(|a| (start[a] + cshape[a]).min(offset[a] + out_spatial[a]));
            let run = (hi[0] - lo[0]) * width;
            for z in lo[2]..hi[2] {
                for y in lo[1]..hi[1] {
                    let src = (lo[0] - start[0]
                        + cshape[0] * (y - start[1] + cshape[1] * (z - start[2])))
                        * width;
                    let dst = (volume * out_voxels + lo[0] - offset[0]
                        + out_spatial[0] * (y - offset[1] + out_spatial[1] * (z - offset[2])))
                        * width;
                    out[dst..dst + run].copy_from_slice(&block[src..src + run]);
                }
            }
        }
        let mut header = self.header.clone();
        header.ndim = size.len() as u8;
        header.dim = [1; 7];
        for (d, &s) in header.dim.iter_mut().zip(&size) {
            *d = s as i64;
        }
        header.transform_voxels(&translation(offset.map(|o| o as f64)));
        if self.lossy {
            let values: Vec<f32> = out
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            Ok(NiftiImage::from_parts(
                header,
                ArrayData::F32(fortran_from_vec(&size, values)),
            ))
        } else {
            header.little_endian = true;
            NiftiImage::from_raw(header, Buffer::Heap(Arc::new(out)), 0)
        }
    }

    /// Decode chunk `i` into little-endian element bytes.
    fn decode_chunk(&self, i: usize) -> Result<Vec<u8>> {
        let (offset, len) = self.table[i];
        let payload = &self.buf[offset..offset + len];
        let (_, _, shape) = self.grid.chunk(i);
        let voxels: usize = shape.iter().product();
        if self.lossy {
            let values = codec::decode_lossy(payload, shape, self.levels, self.step)?;
            Ok(values.iter().flat_map(|v| v.to_le_bytes()).collect())
        } else {
            let width = self.header.datatype.byte_size();
            codec::decode_lossless(payload, voxels * width, width, shape[0])
        }
    }
}

/// Load a `.jvol` file (also available through [`crate::load`]).
pub fn load<P: AsRef<Path>>(path: P) -> Result<NiftiImage> {
    Reader::open(path)?.image()
}

#[cfg(test)]
mod tests;
