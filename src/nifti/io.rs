//! Reading and writing `NIfTI` files.
//!
//! * Compression is detected from the file contents (gzip magic), not the
//!   extension, so misnamed files load correctly. Block-indexed gzip (Mgzip,
//!   BGZF) is decompressed in parallel automatically.
//! * Uncompressed files are memory-mapped; nothing is copied until the data is
//!   used, and [`load_cropped`] copies only the requested region.
//! * Single files (`.nii`, `.nii.gz`) and two-file pairs (`.hdr`/`.img`,
//!   optionally gzipped) are supported for both reading and writing.
//! * Saving writes to a temporary file and renames it into place, so a crash
//!   never leaves a partial file and an image can be saved over the file it
//!   was memory-mapped from.

use super::gzip::{self, GzipOptions};
use super::header::{translation, FileLayout, NiftiHeader};
use super::image::{Buffer, NiftiImage};
use crate::error::{Error, Result};
use crate::transforms::Interpolation;
use memmap2::Mmap;
use rayon::prelude::*;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, LazyLock, Mutex, MutexGuard};
use std::time::SystemTime;

// ===========================================================================
// File formats
// ===========================================================================

/// The on-disk format of a path, decided from its name.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Format {
    /// Single file: `.nii`, `.nii.gz`, or anything else.
    Single,
    /// Two-file pair: header and image paths.
    Pair { header: PathBuf, image: PathBuf },
    /// `.jvol` volumetric compression.
    Jvol,
}

/// Lower-cased file name with a trailing `.gz` removed, plus whether it had one.
fn stem_and_gz(path: &Path) -> (String, bool) {
    let name = path
        .file_name()
        .map(|n| n.to_string_lossy().to_ascii_lowercase())
        .unwrap_or_default();
    match name.strip_suffix(".gz") {
        Some(stem) => (stem.to_string(), true),
        None => (name, false),
    }
}

/// Replace the extension of a pair file (`.hdr` <-> `.img`), keeping any
/// `.gz` suffix and the case of the original extension.
fn swap_pair_extension(path: &Path, to: &str) -> PathBuf {
    let name = path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    let (base, gz) = match name.len().checked_sub(3) {
        Some(i) if name[i..].eq_ignore_ascii_case(".gz") => (&name[..i], &name[i..]),
        _ => (name.as_str(), ""),
    };
    let stem = &base[..base.len().saturating_sub(4)];
    let ext = &base[base.len().saturating_sub(4)..];
    let new_ext = if ext.chars().skip(1).all(|c| c.is_ascii_uppercase()) {
        to.to_ascii_uppercase()
    } else {
        to.to_string()
    };
    path.with_file_name(format!("{stem}{new_ext}{gz}"))
}

/// Resolve the partner of a pair file, preferring an existing file (with or
/// without `.gz`) and falling back to the same compression as `path`.
fn pair_partner(path: &Path, to: &str) -> PathBuf {
    let same = swap_pair_extension(path, to);
    if same.exists() {
        return same;
    }
    let name = same.to_string_lossy().into_owned();
    let alternative = match name.len().checked_sub(3) {
        Some(i) if name[i..].eq_ignore_ascii_case(".gz") => PathBuf::from(&name[..i]),
        _ => PathBuf::from(format!("{name}.gz")),
    };
    if alternative.exists() {
        alternative
    } else {
        same
    }
}

// `stem_and_gz` lower-cases the name, so these comparisons ignore case.
#[allow(clippy::case_sensitive_file_extension_comparisons)]
fn detect_format(path: &Path) -> Format {
    let (stem, _) = stem_and_gz(path);
    if stem.ends_with(".jvol") {
        Format::Jvol
    } else if stem.ends_with(".hdr") {
        Format::Pair {
            header: path.to_path_buf(),
            image: pair_partner(path, ".img"),
        }
    } else if stem.ends_with(".img") {
        Format::Pair {
            header: pair_partner(path, ".hdr"),
            image: path.to_path_buf(),
        }
    } else {
        Format::Single
    }
}

// ===========================================================================
// Reading bytes
// ===========================================================================

/// Map a file read-only, or read it when it cannot be mapped.
#[allow(unsafe_code)]
pub(crate) fn open_bytes(path: &Path) -> Result<Buffer> {
    let file = File::open(path).map_err(|e| io_error(&e, path))?;
    let len = file.metadata().map_err(|e| io_error(&e, path))?.len();
    if len == 0 {
        return Err(Error::InvalidFileFormat(format!(
            "{} is empty",
            path.display()
        )));
    }
    // SAFETY: the map is read-only. Modifying or truncating the file while it
    // is mapped is undefined behaviour at the OS level (it can raise SIGBUS);
    // medrs itself never does so because saves go through a rename.
    if let Ok(map) = unsafe { Mmap::map(&file) } {
        return Ok(Buffer::Mmap(Arc::new(map)));
    }
    // Some files (pipes, some network file systems) cannot be mapped.
    let mut bytes = Vec::with_capacity(usize::try_from(len).unwrap_or(0));
    let mut file = file;
    file.read_to_end(&mut bytes)
        .map_err(|e| io_error(&e, path))?;
    Ok(Buffer::Heap(Arc::new(bytes)))
}

/// Attach the path to an I/O error while keeping its kind.
fn io_error(e: &std::io::Error, path: &Path) -> Error {
    Error::Io(std::io::Error::new(
        e.kind(),
        format!("{}: {e}", path.display()),
    ))
}

/// The bytes of a file with any gzip compression removed.
fn decoded_bytes(path: &Path, use_cache: bool) -> Result<Buffer> {
    if use_cache {
        if let Some(hit) = cache().get(path) {
            return Ok(Buffer::Heap(hit));
        }
    }
    // Stamp before reading so a concurrent rewrite is never cached under the
    // new file's stamp.
    let stamp = file_stamp(path);
    let raw = open_bytes(path)?;
    if !gzip::is_gzip(&raw) {
        return Ok(raw);
    }
    let data = Arc::new(gzip::decompress(&raw).map_err(|e| with_path(e, path))?);
    if let (true, Some(stamp)) = (use_cache, stamp) {
        cache().insert(path, Arc::clone(&data), stamp);
    }
    Ok(Buffer::Heap(data))
}

fn with_path(e: Error, path: &Path) -> Error {
    match e {
        Error::Decompression(msg) => Error::Decompression(format!("{}: {msg}", path.display())),
        Error::InvalidFileFormat(msg) => {
            Error::InvalidFileFormat(format!("{}: {msg}", path.display()))
        }
        other => other,
    }
}

// ===========================================================================
// Decompression cache
// ===========================================================================

/// Default maximum number of cached decompressed files.
pub const DEFAULT_CACHE_ENTRIES: usize = 16;
/// Default maximum total size of cached decompressed data (1 GiB).
pub const DEFAULT_CACHE_BYTES: usize = 1 << 30;

#[derive(Clone, PartialEq, Eq)]
struct FileStamp {
    len: u64,
    modified: Option<SystemTime>,
}

fn file_stamp(path: &Path) -> Option<FileStamp> {
    let meta = std::fs::metadata(path).ok()?;
    Some(FileStamp {
        len: meta.len(),
        modified: meta.modified().ok(),
    })
}

struct CacheEntry {
    data: Arc<Vec<u8>>,
    stamp: FileStamp,
    last_use: u64,
}

/// LRU cache of decompressed files, bounded by entry count and total bytes.
struct DecompressionCache {
    entries: HashMap<PathBuf, CacheEntry>,
    max_entries: usize,
    max_bytes: usize,
    bytes: usize,
    clock: u64,
}

impl DecompressionCache {
    fn key(path: &Path) -> PathBuf {
        path.canonicalize().unwrap_or_else(|_| path.to_path_buf())
    }

    fn get(&mut self, path: &Path) -> Option<Arc<Vec<u8>>> {
        let key = Self::key(path);
        let stamp = file_stamp(&key)?;
        let fresh = self.entries.get(&key).map(|e| e.stamp == stamp)?;
        if !fresh {
            self.remove(&key);
            return None;
        }
        self.clock += 1;
        let entry = self.entries.get_mut(&key)?;
        entry.last_use = self.clock;
        Some(Arc::clone(&entry.data))
    }

    fn insert(&mut self, path: &Path, data: Arc<Vec<u8>>, stamp: FileStamp) {
        if self.max_entries == 0 || data.len() > self.max_bytes {
            return;
        }
        let key = Self::key(path);
        self.remove(&key);
        self.clock += 1;
        self.bytes += data.len();
        self.entries.insert(
            key,
            CacheEntry {
                data,
                stamp,
                last_use: self.clock,
            },
        );
        self.evict();
    }

    fn remove(&mut self, key: &Path) {
        if let Some(e) = self.entries.remove(key) {
            self.bytes -= e.data.len();
        }
    }

    fn evict(&mut self) {
        while self.entries.len() > self.max_entries || self.bytes > self.max_bytes {
            let Some(oldest) = self
                .entries
                .iter()
                .min_by_key(|(_, e)| e.last_use)
                .map(|(k, _)| k.clone())
            else {
                break;
            };
            self.remove(&oldest);
        }
    }
}

static CACHE: LazyLock<Mutex<DecompressionCache>> = LazyLock::new(|| {
    Mutex::new(DecompressionCache {
        entries: HashMap::new(),
        max_entries: DEFAULT_CACHE_ENTRIES,
        max_bytes: DEFAULT_CACHE_BYTES,
        bytes: 0,
        clock: 0,
    })
});

/// Lock the cache, recovering from poisoning (the cache holds no invariants a
/// panic could break).
fn cache() -> MutexGuard<'static, DecompressionCache> {
    CACHE
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Remove every entry from the decompression cache.
pub fn clear_decompression_cache() {
    let mut c = cache();
    c.entries.clear();
    c.bytes = 0;
}

/// Set the maximum number of files kept in the decompression cache
/// (0 disables caching). Default [`DEFAULT_CACHE_ENTRIES`].
pub fn set_cache_size(max_entries: usize) {
    let mut c = cache();
    c.max_entries = max_entries;
    c.evict();
}

/// Set the maximum total size in bytes of the decompression cache.
/// Default [`DEFAULT_CACHE_BYTES`].
pub fn set_cache_max_bytes(max_bytes: usize) {
    let mut c = cache();
    c.max_bytes = max_bytes;
    c.evict();
}

/// Current `(entries, bytes)` held by the decompression cache.
pub fn cache_usage() -> (usize, usize) {
    let c = cache();
    (c.entries.len(), c.bytes)
}

// ===========================================================================
// Loading
// ===========================================================================

/// Load a `NIfTI` image.
///
/// Supports `.nii`, `.nii.gz` (including Mgzip/BGZF, decompressed in
/// parallel), `.hdr`/`.img` pairs (optionally gzipped), and `.jvol` (with the
/// `jvol` feature). Uncompressed data is memory-mapped and decoded lazily.
///
/// # Example
/// ```no_run
/// let img = medrs::nifti::load("brain.nii.gz")?;
/// let data = img.to_f32()?;
/// # Ok::<(), medrs::Error>(())
/// ```
pub fn load<P: AsRef<Path>>(path: P) -> Result<NiftiImage> {
    load_impl(path.as_ref(), false)
}

/// Like [`load`], but keeps decompressed `.nii.gz` data in a process-wide
/// LRU cache so repeated loads of the same file skip decompression.
///
/// Entries are invalidated when the file's size or modification time
/// changes. See [`set_cache_size`] and [`set_cache_max_bytes`].
pub fn load_cached<P: AsRef<Path>>(path: P) -> Result<NiftiImage> {
    load_impl(path.as_ref(), true)
}

fn load_impl(path: &Path, use_cache: bool) -> Result<NiftiImage> {
    Volume::open(path, use_cache)?.image()
}

fn read_pair_header(path: &Path, use_cache: bool) -> Result<(NiftiHeader, FileLayout)> {
    let buf = decoded_bytes(path, use_cache)?;
    let (header, layout) = NiftiHeader::parse(&buf).map_err(|e| with_path(e, path))?;
    if layout != FileLayout::Pair {
        return Err(Error::InvalidFileFormat(format!(
            "{} has single-file (n+1/n+2) magic but a .hdr name",
            path.display()
        )));
    }
    Ok((header, layout))
}

/// Read only the header (including extensions) of a `NIfTI` file.
///
/// For gzipped files only the start of the stream is decompressed.
pub fn load_header<P: AsRef<Path>>(path: P) -> Result<NiftiHeader> {
    let path = path.as_ref();
    match detect_format(path) {
        Format::Jvol => Volume::open(path, false).map(|v| v.header().clone()),
        Format::Pair { header, .. } => read_header_prefix(&header),
        Format::Single => read_header_prefix(path),
    }
}

/// Parse the header at the start of a (possibly gzipped) file, reading only
/// as many bytes as the header and its extensions need.
fn read_header_prefix(path: &Path) -> Result<NiftiHeader> {
    let mut file = File::open(path).map_err(|e| io_error(&e, path))?;
    let mut magic = [0u8; 2];
    let n = file.read(&mut magic).map_err(|e| io_error(&e, path))?;
    drop(file);
    let read = |limit: usize| -> Result<Vec<u8>> {
        let file = File::open(path).map_err(|e| io_error(&e, path))?;
        if n == 2 && gzip::is_gzip(&magic) {
            gzip::decompress_prefix(file, limit).map_err(|e| with_path(e, path))
        } else {
            let mut out = Vec::new();
            file.take(limit as u64)
                .read_to_end(&mut out)
                .map_err(|e| io_error(&e, path))?;
            Ok(out)
        }
    };
    let head = read(NiftiHeader::SIZE_V2 + 4)?;
    let (header, layout) = NiftiHeader::parse(&head).map_err(|e| with_path(e, path))?;
    let extensions_end = match layout {
        FileLayout::Single => header.vox_offset as usize,
        // A pair's header file holds only the header and extensions.
        FileLayout::Pair => usize::MAX,
    };
    let has_extensions = head.get(header.header_size()).is_some_and(|&f| f != 0);
    if !has_extensions || extensions_end <= head.len() {
        return Ok(header);
    }
    // Extensions are bounded in the parser; cap the read as well.
    let full = read(extensions_end.min(header.header_size() + 4 + (256 << 20)))?;
    NiftiHeader::from_bytes(&full).map_err(|e| with_path(e, path))
}

// ===========================================================================
// Crop-first loading
// ===========================================================================

/// Load only a spatial region of an image: `shape` voxels starting at
/// `offset` along the first three axes (all of any further axes are kept).
///
/// For uncompressed files only the region is read from disk, and for `.jvol`
/// files only the chunks that overlap it are decoded. Gzipped files must be
/// decompressed in full; the decompressed data is kept in the [`load_cached`]
/// cache so further crops of the same file are cheap. The result keeps the
/// stored datatype, scaling, and world coordinates.
pub fn load_cropped<P: AsRef<Path>>(
    path: P,
    offset: [usize; 3],
    shape: [usize; 3],
) -> Result<NiftiImage> {
    let path = path.as_ref();
    Volume::open(path, true)?.region(offset, shape)
}

/// An image file opened for reading: uncompressed data is memory-mapped,
/// gzipped data is decompressed once, and `.jvol` data is decoded chunk by
/// chunk on demand. Any number of regions can be read from it.
pub(crate) enum Volume {
    Raw {
        header: NiftiHeader,
        buf: Buffer,
        offset: usize,
    },
    #[cfg(feature = "jvol")]
    Jvol(crate::jvol::Reader),
}

impl Volume {
    pub(crate) fn open(path: &Path, use_cache: bool) -> Result<Self> {
        let (header, buf) = match detect_format(path) {
            #[cfg(feature = "jvol")]
            Format::Jvol => return crate::jvol::Reader::open(path).map(Self::Jvol),
            #[cfg(not(feature = "jvol"))]
            Format::Jvol => {
                return Err(Error::InvalidFileFormat(
                    "reading .jvol files requires the `jvol` cargo feature".into(),
                ))
            }
            Format::Single => {
                let buf = decoded_bytes(path, use_cache)?;
                let (header, layout) = NiftiHeader::parse(&buf).map_err(|e| with_path(e, path))?;
                if layout == FileLayout::Pair {
                    return Err(Error::InvalidFileFormat(format!(
                        "{} is the header of a two-file (.hdr/.img) dataset; \
                         rename it to .hdr or load the pair",
                        path.display()
                    )));
                }
                (header, buf)
            }
            Format::Pair { header, image } => {
                let (hdr, _) = read_pair_header(&header, use_cache)?;
                (hdr, decoded_bytes(&image, use_cache)?)
            }
        };
        let offset = usize::try_from(header.vox_offset).map_err(|_| {
            Error::InvalidFileFormat(format!("{}: invalid vox_offset", path.display()))
        })?;
        let needed = header.data_size();
        let available = buf.len().saturating_sub(offset);
        if available < needed {
            return Err(Error::InvalidFileFormat(format!(
                "{}: file truncated: voxel data needs {needed} bytes at offset {offset}, \
                 but only {available} are present",
                path.display()
            )));
        }
        Ok(Self::Raw {
            header,
            buf,
            offset,
        })
    }

    pub(crate) fn header(&self) -> &NiftiHeader {
        match self {
            Self::Raw { header, .. } => header,
            #[cfg(feature = "jvol")]
            Self::Jvol(r) => r.header(),
        }
    }

    /// The whole image (no copy for uncompressed and gzipped files).
    pub(crate) fn image(&self) -> Result<NiftiImage> {
        match self {
            Self::Raw {
                header,
                buf,
                offset,
            } => NiftiImage::from_raw(header.clone(), buf.clone(), *offset),
            #[cfg(feature = "jvol")]
            Self::Jvol(r) => r.image(),
        }
    }

    /// `shape` voxels at `offset` along the first three axes.
    pub(crate) fn region(&self, offset: [usize; 3], shape: [usize; 3]) -> Result<NiftiImage> {
        match self {
            Self::Raw {
                header,
                buf,
                offset: data_offset,
            } => {
                let full = header.shape();
                let (start, size) = spatial_region(&full, offset, shape)?;
                let data = &buf[*data_offset..*data_offset + header.data_size()];
                let bytes = copy_region(data, &full, header.datatype.byte_size(), &start, &size);
                let mut out = header.clone();
                out.ndim = size.len() as u8;
                out.dim = [1; 7];
                for (d, &s) in out.dim.iter_mut().zip(&size) {
                    *d = s as i64;
                }
                out.transform_voxels(&translation(offset.map(|o| o as f64)));
                NiftiImage::from_raw(out, Buffer::Heap(Arc::new(bytes)), 0)
            }
            #[cfg(feature = "jvol")]
            Self::Jvol(r) => r.region(offset, shape),
        }
    }
}

/// Load several images and resample each onto the voxel grid of
/// `files[reference]`, so that they can be processed voxel by voxel together
/// (for example an MRI, a CT, and a segmentation of the same subject).
///
/// Each file comes with the interpolation used if it needs resampling (use
/// [`Interpolation::Nearest`] for label maps). Images already on the
/// reference grid are returned as loaded, keeping their datatype. Files are
/// loaded in parallel.
pub fn load_multi<P: AsRef<Path> + Sync>(
    files: &[(P, Interpolation)],
    reference: usize,
) -> Result<Vec<NiftiImage>> {
    if reference >= files.len() {
        return Err(Error::InvalidArgument(format!(
            "reference index {reference} is out of range for {} files",
            files.len()
        )));
    }
    let images = crate::parallel::install(|| {
        files
            .par_iter()
            .map(|(path, _)| load(path))
            .collect::<Result<Vec<_>>>()
    })?;
    let target = &images[reference];
    images
        .iter()
        .zip(files)
        .map(|(image, (_, interp))| {
            if same_grid(image, target) {
                Ok(image.clone())
            } else {
                crate::transforms::resample_like(image, target, *interp)
            }
        })
        .collect()
}

/// Whether two images share spatial shape and (to within 1e-4 mm) affine.
fn same_grid(a: &NiftiImage, b: &NiftiImage) -> bool {
    let spatial = |i: &NiftiImage| crate::transforms::geometry::split_shape(i.shape()).0;
    let (fa, fb) = (a.affine(), b.affine());
    spatial(a) == spatial(b)
        && (0..3)
            .all(|i| (0..4).all(|j| (fa[i][j] - fb[i][j]).abs() <= 1e-4 * fb[i][j].abs().max(1.0)))
}

/// Validate a 3D region against an image shape and extend it to all axes.
///
/// Images with fewer than three axes are treated as having trailing axes of
/// length 1; axes beyond the third are kept whole.
pub(crate) fn spatial_region(
    full: &[usize],
    offset: [usize; 3],
    shape: [usize; 3],
) -> Result<(Vec<usize>, Vec<usize>)> {
    let nd = full.len().max(3);
    let mut start = vec![0; nd];
    let mut size = full.to_vec();
    size.resize(nd, 1);
    for axis in 0..3 {
        let extent = full.get(axis).copied().unwrap_or(1);
        if shape[axis] == 0 {
            return Err(Error::InvalidCropRegion(format!(
                "crop size along axis {axis} must be positive"
            )));
        }
        let end = offset[axis].checked_add(shape[axis]);
        if end.is_none_or(|e| e > extent) {
            return Err(Error::InvalidCropRegion(format!(
                "region {}..{} along axis {axis} exceeds the image extent {extent}",
                offset[axis],
                offset[axis].saturating_add(shape[axis])
            )));
        }
        start[axis] = offset[axis];
        size[axis] = shape[axis];
    }
    size.truncate(full.len().max(3));
    // Drop the padding axes again for images with fewer than three dimensions.
    if full.len() < 3 {
        start.truncate(full.len());
        size.truncate(full.len());
    }
    Ok((start, size))
}

/// Copy an axis-aligned region of a Fortran-ordered array stored as bytes.
///
/// `shape`, `start`, and `size` are per axis; the region must lie within the
/// array and `src` must hold the whole array. Leading axes that are copied
/// whole are merged into one contiguous run per copy.
pub(crate) fn copy_region(
    src: &[u8],
    shape: &[usize],
    elem: usize,
    start: &[usize],
    size: &[usize],
) -> Vec<u8> {
    let nd = shape.len();
    let mut stride = vec![0usize; nd];
    let mut acc = elem;
    for (s, &n) in stride.iter_mut().zip(shape) {
        *s = acc;
        acc *= n;
    }
    let mut full_axes = 0;
    while full_axes < nd && start[full_axes] == 0 && size[full_axes] == shape[full_axes] {
        full_axes += 1;
    }
    let (run, outer_from) = if full_axes == nd {
        (acc, nd)
    } else {
        (stride[full_axes] * size[full_axes], full_axes + 1)
    };
    let base: usize = (0..nd).map(|i| start[i] * stride[i]).sum();
    let outer = &size[outer_from..];
    let runs: usize = outer.iter().product();
    let mut out = vec![0u8; run * runs];
    if run == 0 {
        return out;
    }
    let copy = |(r, dst): (usize, &mut [u8])| {
        let mut rem = r;
        let mut off = base;
        for (k, &n) in outer.iter().enumerate() {
            off += (rem % n) * stride[outer_from + k];
            rem /= n;
        }
        dst.copy_from_slice(&src[off..off + run]);
    };
    if out.len() >= 1 << 20 {
        crate::parallel::install(|| out.par_chunks_mut(run).enumerate().for_each(copy));
    } else {
        out.chunks_mut(run).enumerate().for_each(copy);
    }
    out
}

// ===========================================================================
// Saving
// ===========================================================================

/// Options for [`save_with_options`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SaveOptions {
    /// gzip compression level, 0 (none) to 9 (smallest). Default 3.
    pub compression_level: u32,
    /// Write gzip output as block-indexed Mgzip, which medrs (and the Python
    /// `mgzip` package) can decompress in parallel. Still valid gzip for
    /// every other reader. Default `false`.
    pub mgzip: bool,
    /// Threads for compression; 0 uses all available cores. Default 0.
    pub threads: usize,
}

impl Default for SaveOptions {
    fn default() -> Self {
        Self {
            compression_level: 3,
            mgzip: false,
            threads: 0,
        }
    }
}

/// Save an image; the format follows the file name.
///
/// * `.nii` — uncompressed single file
/// * `.nii.gz` (any `.gz`) — gzipped single file
/// * `.hdr` / `.img` (optionally `.gz`) — two-file pair
/// * `.jvol` — volumetric compression (lossless; see [`crate::jvol`])
///
/// Every header field and extension is written back. The file is written to
/// a temporary name and renamed into place.
///
/// # Example
/// ```no_run
/// # let img = medrs::nifti::load("in.nii")?;
/// medrs::nifti::save(&img, "out.nii.gz")?;
/// # Ok::<(), medrs::Error>(())
/// ```
pub fn save<P: AsRef<Path>>(image: &NiftiImage, path: P) -> Result<()> {
    save_with_options(image, path, &SaveOptions::default())
}

/// Save an image in Mgzip format (block-indexed gzip, decompressed in
/// parallel by [`load`]).
pub fn save_mgzip<P: AsRef<Path>>(image: &NiftiImage, path: P) -> Result<()> {
    let options = SaveOptions {
        mgzip: true,
        ..SaveOptions::default()
    };
    save_with_options(image, path, &options)
}

/// Save an image with explicit compression options.
pub fn save_with_options<P: AsRef<Path>>(
    image: &NiftiImage,
    path: P,
    options: &SaveOptions,
) -> Result<()> {
    let path = path.as_ref();
    let gz = |p: &Path| stem_and_gz(p).1;
    let gzip = |p: &Path| {
        gz(p).then_some(GzipOptions {
            level: options.compression_level,
            mgzip: options.mgzip,
            threads: options.threads,
        })
    };
    match detect_format(path) {
        Format::Jvol => save_jvol(image, path),
        Format::Single => {
            let header = image.header().prepared_for_write(FileLayout::Single);
            header.validate()?;
            let prefix = header.encode(FileLayout::Single)?;
            let data = image.data_bytes_le()?;
            write_file(path, &[&prefix, &data], gzip(path))
        }
        Format::Pair {
            header: hdr_path, ..
        } => {
            // Derive the image file from the header name so both share the
            // same compression.
            let img_path = swap_pair_extension(&hdr_path, ".img");
            let header = image.header().prepared_for_write(FileLayout::Pair);
            header.validate()?;
            let prefix = header.encode(FileLayout::Pair)?;
            let data = image.data_bytes_le()?;
            write_file(&img_path, &[&data], gzip(&img_path))?;
            write_file(&hdr_path, &[&prefix], gzip(&hdr_path))
        }
    }
}

#[cfg(feature = "jvol")]
fn save_jvol(image: &NiftiImage, path: &Path) -> Result<()> {
    crate::jvol::save(image, path, &crate::jvol::JvolOptions::default())
}

#[cfg(not(feature = "jvol"))]
fn save_jvol(_image: &NiftiImage, _path: &Path) -> Result<()> {
    Err(Error::InvalidFileFormat(
        "writing .jvol files requires the `jvol` cargo feature".into(),
    ))
}

/// Write `parts` to `path` atomically, gzip-compressing when requested.
fn write_file(path: &Path, parts: &[&[u8]], gzip: Option<GzipOptions>) -> Result<()> {
    write_atomic(path, |file| {
        let writer = BufWriter::with_capacity(1 << 20, file);
        let mut writer = if let Some(opts) = gzip {
            gzip::compress_to(parts, writer, opts)?
        } else {
            let mut writer = writer;
            for part in parts {
                writer.write_all(part)?;
            }
            writer
        };
        // `into_inner` flushes, surfacing any deferred write error.
        writer.flush()?;
        writer.into_inner().map_err(|e| Error::Io(e.into_error()))?;
        Ok(())
    })
}

static TEMP_COUNTER: AtomicUsize = AtomicUsize::new(0);

/// Create `path` by writing a temporary file in the same directory and
/// renaming it over the destination on success.
pub(crate) fn write_atomic(path: &Path, write: impl FnOnce(File) -> Result<()>) -> Result<()> {
    let dir = match path.parent() {
        Some(p) if !p.as_os_str().is_empty() => p.to_path_buf(),
        _ => PathBuf::from("."),
    };
    let name = path
        .file_name()
        .ok_or_else(|| Error::InvalidArgument(format!("{} is not a file path", path.display())))?
        .to_string_lossy()
        .into_owned();
    let tmp = dir.join(format!(
        ".{name}.{}-{}.tmp",
        std::process::id(),
        TEMP_COUNTER.fetch_add(1, Ordering::Relaxed)
    ));
    let file = File::options()
        .write(true)
        .create_new(true)
        .open(&tmp)
        .map_err(|e| io_error(&e, path))?;
    let result = write(file).and_then(|()| {
        // Keep the permissions of a file being replaced.
        if let Ok(meta) = std::fs::metadata(path) {
            let _ = std::fs::set_permissions(&tmp, meta.permissions());
        }
        std::fs::rename(&tmp, path).map_err(|e| io_error(&e, path))
    });
    if result.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
    result
}

// ===========================================================================
// Mgzip helpers
// ===========================================================================

/// Whether a file is block-indexed gzip (Mgzip or BGZF), which [`load`]
/// decompresses in parallel. Only the first bytes of the file are read.
pub fn is_mgzip<P: AsRef<Path>>(path: P) -> Result<bool> {
    let path = path.as_ref();
    let mut head = [0u8; 64];
    let mut file = File::open(path).map_err(|e| io_error(&e, path))?;
    let mut n = 0;
    while n < head.len() {
        let read = file.read(&mut head[n..]).map_err(|e| io_error(&e, path))?;
        if read == 0 {
            break;
        }
        n += read;
    }
    Ok(gzip::block_format(&head[..n]).is_some())
}

/// `<name>.mgz.nii.gz` next to `input`.
fn mgzip_name(input: &Path) -> PathBuf {
    let name = input
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    let lower = name.to_ascii_lowercase();
    let stem = [".nii.gz", ".nii", ".gz"]
        .iter()
        .find_map(|ext| lower.strip_suffix(ext).map(|s| &name[..s.len()]))
        .unwrap_or(&name);
    input.with_file_name(format!("{stem}.mgz.nii.gz"))
}

/// Re-save a `NIfTI` file as Mgzip for parallel decompression.
///
/// Without `output`, writes `<name>.mgz.nii.gz` next to the input (still a
/// valid `.nii.gz` for every reader). Returns the output path.
pub fn convert_to_mgzip<P: AsRef<Path>>(input: P, output: Option<&Path>) -> Result<PathBuf> {
    let input = input.as_ref();
    let output = output.map_or_else(|| mgzip_name(input), Path::to_path_buf);
    let image = load(input)?;
    save_mgzip(&image, &output)?;
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nifti::{Affine, DataType, NiftiExtension};
    use ndarray::{s, ArrayD, IxDyn, ShapeBuilder};
    use tempfile::tempdir;

    fn ramp(shape: &[usize]) -> ArrayD<f32> {
        let n: usize = shape.iter().product();
        ArrayD::from_shape_vec(IxDyn(shape).f(), (0..n).map(|v| v as f32).collect()).unwrap()
    }

    fn affine() -> Affine {
        [
            [-2.0, 0.0, 0.0, 90.0],
            [0.0, 2.0, 0.0, -126.0],
            [0.0, 0.0, 2.5, -72.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    }

    #[test]
    fn roundtrip_every_format() {
        let dir = tempdir().unwrap();
        let mut img = NiftiImage::from_array(ramp(&[5, 6, 7, 2]), affine()).unwrap();
        img.header_mut()
            .extensions
            .push(NiftiExtension::new(6, b"hello".to_vec()));
        img.header_mut().descrip = "round trip".into();
        for name in [
            "a.nii",
            "b.nii.gz",
            "c.hdr",
            "d.img",
            "e.hdr.gz",
            "f.NII.GZ",
            "g.mgz.nii.gz",
        ] {
            let path = dir.path().join(name);
            save(&img, &path).unwrap();
            let back = load(&path).unwrap();
            assert_eq!(back.shape(), img.shape(), "{name}");
            assert_eq!(back.to_f32().unwrap(), img.to_f32().unwrap(), "{name}");
            assert_eq!(back.affine(), img.affine(), "{name}");
            assert_eq!(back.header().descrip, "round trip");
            assert_eq!(back.header().extensions[0].ecode, 6);
            assert_eq!(&back.header().extensions[0].data[..5], b"hello");
            let hdr = load_header(&path).unwrap();
            assert_eq!(hdr.extensions.len(), 1, "{name}");
        }
        assert!(dir.path().join("e.img.gz").exists());
    }

    #[test]
    fn gzip_is_detected_by_content() {
        let dir = tempdir().unwrap();
        let img = NiftiImage::from_array(ramp(&[4, 4, 4]), affine()).unwrap();
        let gz = dir.path().join("x.nii.gz");
        save(&img, &gz).unwrap();
        let misnamed = dir.path().join("x.nii");
        std::fs::copy(&gz, &misnamed).unwrap();
        assert_eq!(
            load(&misnamed).unwrap().to_f32().unwrap(),
            img.to_f32().unwrap()
        );
    }

    #[test]
    fn mgzip_roundtrip_detection_and_conversion() {
        let dir = tempdir().unwrap();
        let img = NiftiImage::from_array(ramp(&[64, 64, 80]), affine()).unwrap();
        let path = dir.path().join("x.nii.gz");
        save(&img, &path).unwrap();
        assert!(!is_mgzip(&path).unwrap());
        let out = convert_to_mgzip(&path, None).unwrap();
        assert_eq!(out, dir.path().join("x.mgz.nii.gz"));
        assert!(is_mgzip(&out).unwrap());
        assert_eq!(load(&out).unwrap().to_f32().unwrap(), img.to_f32().unwrap());
    }

    #[test]
    fn save_over_the_mapped_source_is_safe() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("same.nii");
        let img = NiftiImage::from_array(ramp(&[32, 32, 32]), affine()).unwrap();
        save(&img, &path).unwrap();
        let mut mapped = load(&path).unwrap();
        assert!(mapped.is_memory_mapped());
        mapped.header_mut().descrip = "edited".into();
        save(&mapped, &path).unwrap();
        // The old mapping stays valid and the new file is complete.
        assert_eq!(mapped.to_f32().unwrap(), img.to_f32().unwrap());
        let back = load(&path).unwrap();
        assert_eq!(back.header().descrip, "edited");
        assert_eq!(back.to_f32().unwrap(), img.to_f32().unwrap());
        // No temporary files are left behind.
        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 1);
    }

    #[test]
    fn save_to_bare_relative_filename() {
        let dir = tempdir().unwrap();
        let old = std::env::current_dir().unwrap();
        std::env::set_current_dir(dir.path()).unwrap();
        let img = NiftiImage::from_array(ramp(&[2, 2, 2]), affine()).unwrap();
        let result = save(&img, "bare.nii");
        std::env::set_current_dir(old).unwrap();
        result.unwrap();
        assert!(dir.path().join("bare.nii").exists());
    }

    #[test]
    fn crop_matches_slicing_for_all_layouts_and_4d() {
        let dir = tempdir().unwrap();
        let data = ramp(&[9, 8, 7, 3]);
        let img = NiftiImage::from_array(data.clone(), affine()).unwrap();
        for name in ["c.nii", "c.nii.gz", "c.hdr"] {
            let path = dir.path().join(name);
            save(&img, &path).unwrap();
            for (off, shp) in [
                ([0, 0, 0], [9, 8, 7]),
                ([0, 0, 2], [9, 8, 3]),
                ([0, 1, 2], [9, 5, 3]),
                ([2, 1, 3], [4, 5, 2]),
            ] {
                let crop = load_cropped(&path, off, shp).unwrap();
                let expected = data
                    .slice(s![
                        off[0]..off[0] + shp[0],
                        off[1]..off[1] + shp[1],
                        off[2]..off[2] + shp[2],
                        ..
                    ])
                    .to_owned();
                assert_eq!(
                    crop.to_f32().unwrap(),
                    expected.into_dyn(),
                    "{name} {off:?}"
                );
                // World position of the first cropped voxel is preserved.
                let a = crop.affine();
                let src = img.affine();
                for row in 0..3 {
                    let world: f64 =
                        (0..3).map(|j| src[row][j] * off[j] as f64).sum::<f64>() + src[row][3];
                    assert!((a[row][3] - world).abs() < 1e-9);
                }
            }
        }
    }

    #[test]
    fn crop_bounds_are_checked_without_overflow() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("b.nii");
        save(
            &NiftiImage::from_array(ramp(&[4, 4, 4]), affine()).unwrap(),
            &path,
        )
        .unwrap();
        assert!(load_cropped(&path, [usize::MAX, 0, 0], [2, 2, 2]).is_err());
        assert!(load_cropped(&path, [3, 0, 0], [2, 2, 2]).is_err());
        assert!(load_cropped(&path, [0, 0, 0], [0, 2, 2]).is_err());
    }

    #[test]
    fn crop_preserves_dtype_and_scaling() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("s.nii.gz");
        let arr =
            ArrayD::from_shape_vec(IxDyn(&[4, 4, 4]).f(), (0..64).map(|v| v as i16).collect())
                .unwrap();
        let mut img = NiftiImage::from_array(arr, affine()).unwrap();
        img.header_mut().scl_slope = 2.0;
        img.header_mut().scl_inter = -1024.0;
        save(&img, &path).unwrap();
        let crop = load_cropped(&path, [1, 1, 1], [2, 2, 2]).unwrap();
        assert_eq!(crop.dtype(), DataType::Int16);
        let v = crop.to_f32().unwrap();
        assert_eq!(v[[0, 0, 0]], (1 + 4 + 16) as f32 * 2.0 - 1024.0);
    }

    #[test]
    fn truncated_and_corrupt_files_error_cleanly() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("t.nii");
        save(
            &NiftiImage::from_array(ramp(&[8, 8, 8]), affine()).unwrap(),
            &path,
        )
        .unwrap();
        let bytes = std::fs::read(&path).unwrap();
        std::fs::write(&path, &bytes[..bytes.len() - 10]).unwrap();
        let err = load(&path).unwrap_err().to_string();
        assert!(err.contains("truncated"), "{err}");
        std::fs::write(&path, b"").unwrap();
        assert!(load(&path).is_err());
        let missing = load(dir.path().join("missing.nii")).unwrap_err();
        assert!(matches!(missing, Error::Io(ref e) if e.kind() == std::io::ErrorKind::NotFound));
    }

    #[test]
    fn cache_hits_invalidates_and_bounds() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("cached.nii.gz");
        let img = NiftiImage::from_array(ramp(&[16, 16, 16]), affine()).unwrap();
        save(&img, &path).unwrap();
        let a = load_cached(&path).unwrap();
        let b = load_cached(&path).unwrap();
        assert_eq!(a.to_f32().unwrap(), b.to_f32().unwrap());
        // Rewriting the file invalidates the entry.
        let img2 = NiftiImage::from_array(ramp(&[16, 16, 16]).mapv(|v| -v), affine()).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(20));
        save(&img2, &path).unwrap();
        assert_eq!(
            load_cached(&path).unwrap().to_f32().unwrap(),
            img2.to_f32().unwrap()
        );
    }

    #[test]
    fn copy_region_general_nd() {
        let shape = [3usize, 4, 5, 2];
        let src: Vec<u8> = (0..shape.iter().product::<usize>())
            .map(|v| v as u8)
            .collect();
        let arr = ArrayD::from_shape_vec(IxDyn(&shape).f(), src.clone()).unwrap();
        let out = copy_region(&src, &shape, 1, &[1, 0, 2, 0], &[2, 4, 3, 2]);
        let expected: Vec<u8> = arr
            .slice(s![1..3, .., 2..5, ..])
            .t()
            .iter()
            .copied()
            .collect();
        assert_eq!(out, expected);
    }
}
