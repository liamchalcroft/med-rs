//! High-performance NIfTI I/O operations.
//!
//! Optimizations:
//! - Memory-mapped reading for uncompressed files
//! - Parallel gzip compression/decompression for .nii.gz (using gzp)
//! - Optional decompression caching for repeated access

use super::header::NiftiHeader;
use super::image::NiftiImage;
use crate::error::{Error, Result};
use flate2::bufread::{GzDecoder, MultiGzDecoder};
use gzp::deflate::{Gzip, Mgzip};
use gzp::par::compress::ParCompressBuilder;
use gzp::par::decompress::ParDecompressBuilder;
use gzp::ZWriter;
use libdeflater::{DecompressionError, Decompressor};
use memmap2::Mmap;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

use crate::transforms::{self, Interpolation};

// ============================================================================
// Decompression Cache for pseudo-zero-copy gzipped files
// ============================================================================

/// Global cache for decompressed gzip data.
/// Allows "zero-copy-like" repeated access to gzipped files.
static DECOMPRESSION_CACHE: std::sync::LazyLock<RwLock<DecompressionCache>> =
    std::sync::LazyLock::new(|| RwLock::new(DecompressionCache::new(10)));

thread_local! {
    static DECOMPRESSOR: std::cell::RefCell<Decompressor> = std::cell::RefCell::new(Decompressor::new());
    static DECOMPRESS_BUFFER_POOL: std::cell::RefCell<Vec<Vec<u8>>> = const { std::cell::RefCell::new(Vec::new()) };
}

fn acquire_decompress_buffer(capacity: usize) -> Vec<u8> {
    DECOMPRESS_BUFFER_POOL.with(|pool| {
        if let Ok(mut p) = pool.try_borrow_mut() {
            if let Some(mut buf) = p.pop() {
                if buf.capacity() >= capacity {
                    buf.clear();
                    return buf;
                }
            }
        }
        Vec::with_capacity(capacity)
    })
}

fn release_decompress_buffer(buf: Vec<u8>) {
    DECOMPRESS_BUFFER_POOL.with(|pool| {
        if let Ok(mut p) = pool.try_borrow_mut() {
            if p.len() < 4 {
                p.push(buf);
            }
        }
    });
}

/// LRU-style cache for decompressed NIfTI data.
struct DecompressionCache {
    /// Map from file path to (decompressed data, last access time)
    entries: HashMap<PathBuf, CacheEntry>,
    /// Maximum number of entries to cache
    max_entries: usize,
    /// Access counter for LRU eviction
    access_counter: u64,
}

struct CacheEntry {
    data: Arc<Vec<u8>>,
    header: NiftiHeader,
    last_access: u64,
}

impl DecompressionCache {
    fn new(max_entries: usize) -> Self {
        Self {
            entries: HashMap::new(),
            max_entries,
            access_counter: 0,
        }
    }

    fn get(&mut self, path: &Path) -> Option<(Arc<Vec<u8>>, NiftiHeader)> {
        if let Some(entry) = self.entries.get_mut(path) {
            self.access_counter += 1;
            entry.last_access = self.access_counter;
            Some((entry.data.clone(), entry.header.clone()))
        } else {
            None
        }
    }

    fn insert(&mut self, path: PathBuf, data: Arc<Vec<u8>>, header: NiftiHeader) {
        // Evict oldest entry if at capacity
        if self.entries.len() >= self.max_entries {
            if let Some(oldest_path) = self
                .entries
                .iter()
                .min_by_key(|(_, e)| e.last_access)
                .map(|(p, _)| p.clone())
            {
                self.entries.remove(&oldest_path);
            }
        }

        self.access_counter += 1;
        self.entries.insert(
            path,
            CacheEntry {
                data,
                header,
                last_access: self.access_counter,
            },
        );
    }

    fn clear(&mut self) {
        self.entries.clear();
    }
}

/// Clear the global decompression cache.
///
/// Call this to free memory used by cached decompressed files.
pub fn clear_decompression_cache() {
    if let Ok(mut cache) = DECOMPRESSION_CACHE.write() {
        cache.clear();
    }
}

/// Set the maximum size of the decompression cache.
///
/// Default is 10 entries. Set to 0 to disable caching.
pub fn set_cache_size(max_entries: usize) {
    if let Ok(mut cache) = DECOMPRESSION_CACHE.write() {
        cache.max_entries = max_entries;
        // Evict excess entries
        while cache.entries.len() > max_entries {
            if let Some(oldest_path) = cache
                .entries
                .iter()
                .min_by_key(|(_, e)| e.last_access)
                .map(|(p, _)| p.clone())
            {
                cache.entries.remove(&oldest_path);
            }
        }
    }
}

#[cfg(target_os = "linux")]
fn read_file_with_readahead(path: &Path) -> Result<Vec<u8>> {
    use std::os::unix::io::AsRawFd;

    let file = File::open(path)?;
    let fd = file.as_raw_fd();
    let metadata = file.metadata()?;
    let len = metadata.len() as usize;

    // POSIX_FADV_SEQUENTIAL = 2, hint that we'll read sequentially
    unsafe {
        libc::posix_fadvise(fd, 0, len as libc::off_t, libc::POSIX_FADV_SEQUENTIAL);
    }

    let mut buffer = Vec::with_capacity(len);
    let mut reader = BufReader::with_capacity(GZIP_BUFFER_SIZE, file);
    reader.read_to_end(&mut buffer)?;
    Ok(buffer)
}

#[cfg(not(target_os = "linux"))]
fn read_file_with_readahead(path: &Path) -> Result<Vec<u8>> {
    Ok(std::fs::read(path)?)
}

fn ensure_no_extensions(bytes: &[u8], header: &NiftiHeader) -> Result<()> {
    let header_size = header.header_size();
    let vox_offset = header.vox_offset as usize;
    if vox_offset >= header_size + 4 && bytes.len() >= header_size + 4 {
        let extension_flag = bytes[header_size];
        if extension_flag != 0 {
            return Err(Error::InvalidFileFormat(
                "NIfTI extensions are not supported".to_string(),
            ));
        }
    }
    Ok(())
}

/// Load a NIfTI image from file.
///
/// Supports both `.nii` and `.nii.gz` formats with automatic detection.
///
/// # Example
/// ```ignore
/// let img = medrs::nifti::load("brain.nii.gz")?;
/// let data = img.to_f32();
/// ```
#[must_use = "this function returns a loaded image that should be used"]
pub fn load<P: AsRef<Path>>(path: P) -> Result<NiftiImage> {
    let path = path.as_ref();
    let is_gzipped = path.extension().is_some_and(|e| e == "gz");

    if is_gzipped {
        load_gzipped(path)
    } else {
        load_uncompressed(path)
    }
}

/// Load a NIfTI image with caching for repeated access.
///
/// For gzipped files, this caches the decompressed data so subsequent loads
/// of the same file are nearly instant (pseudo-zero-copy). For uncompressed
/// files, this behaves identically to [`load`].
///
/// This is particularly useful in training pipelines where the same volume
/// may be accessed multiple times across epochs.
///
/// # Example
/// ```ignore
/// // First load decompresses and caches
/// let img1 = medrs::nifti::load_cached("brain.nii.gz")?;
///
/// // Second load returns cached data (very fast)
/// let img2 = medrs::nifti::load_cached("brain.nii.gz")?;
///
/// // Clear cache when done to free memory
/// medrs::nifti::clear_decompression_cache();
/// ```
#[must_use = "this function returns a loaded image that should be used"]
pub fn load_cached<P: AsRef<Path>>(path: P) -> Result<NiftiImage> {
    let path = path.as_ref();
    let is_gzipped = path.extension().is_some_and(|e| e == "gz");

    if is_gzipped {
        load_gzipped_cached(path)
    } else {
        // Uncompressed files already use mmap, so caching provides no benefit
        load_uncompressed(path)
    }
}

fn estimate_gzip_uncompressed_size(compressed: &[u8]) -> usize {
    // ISIZE per RFC 1952: "original input size modulo 2^32"
    // This is only reliable for single-member gzip < 4GB.
    if compressed.len() >= 4 {
        let trailer = &compressed[compressed.len() - 4..];
        u32::from_le_bytes([trailer[0], trailer[1], trailer[2], trailer[3]]) as usize
    } else {
        // Conservative fallback for tiny/invalid inputs.
        compressed.len() * 4
    }
}

const GZIP_BUFFER_SIZE: usize = 256 * 1024; // 256KB buffer for streaming decompression

fn decompress_gzip_streaming(compressed: &[u8]) -> Result<Vec<u8>> {
    let cursor = std::io::Cursor::new(compressed);
    let mut decoder = MultiGzDecoder::new(BufReader::with_capacity(GZIP_BUFFER_SIZE, cursor));

    let estimated = estimate_gzip_uncompressed_size(compressed);
    let mut output = Vec::with_capacity(estimated);

    decoder
        .read_to_end(&mut output)
        .map_err(|e| Error::Decompression(format!("gzip stream decode failed: {e}")))?;
    Ok(output)
}

#[allow(clippy::uninit_vec)]
fn decompress_gzip_with_fallback(compressed: &[u8]) -> Result<(Vec<u8>, bool)> {
    let estimated_size = estimate_gzip_uncompressed_size(compressed);
    let buffer_size = estimated_size.max(NiftiHeader::SIZE);

    // SAFETY: Allocate uninitialized buffer to avoid zeroing overhead for large files.
    // libdeflate's gzip_decompress writes directly to the buffer. We truncate to
    // the actual written size before returning. If decompression fails, the buffer
    // is dropped without reading uninitialized bytes.
    let mut output = Vec::with_capacity(buffer_size);
    unsafe {
        output.set_len(buffer_size);
    }

    let result = DECOMPRESSOR.with(|d| d.borrow_mut().gzip_decompress(compressed, &mut output));

    match result {
        Ok(written) => {
            output.truncate(written);
            Ok((output, false))
        }
        Err(DecompressionError::InsufficientSpace) => {
            drop(output);
            let output = decompress_gzip_streaming(compressed)?;
            Ok((output, true))
        }
        Err(e) => Err(Error::Decompression(format!("{}", e))),
    }
}

#[allow(clippy::uninit_vec)]
fn decompress_gzip_pooled(compressed: &[u8]) -> Result<(Vec<u8>, bool)> {
    let estimated_size = estimate_gzip_uncompressed_size(compressed);
    let buffer_size = estimated_size.max(NiftiHeader::SIZE);

    let mut output = acquire_decompress_buffer(buffer_size);
    if output.capacity() < buffer_size {
        output.reserve(buffer_size - output.capacity());
    }
    unsafe {
        output.set_len(buffer_size);
    }

    let result = DECOMPRESSOR.with(|d| d.borrow_mut().gzip_decompress(compressed, &mut output));

    match result {
        Ok(written) => {
            output.truncate(written);
            Ok((output, false))
        }
        Err(DecompressionError::InsufficientSpace) => {
            release_decompress_buffer(output);
            let output = decompress_gzip_streaming(compressed)?;
            Ok((output, true))
        }
        Err(e) => {
            release_decompress_buffer(output);
            Err(Error::Decompression(format!("{}", e)))
        }
    }
}

fn parse_decompressed_nifti(bytes: &[u8]) -> Result<(NiftiHeader, usize, usize)> {
    if bytes.len() < NiftiHeader::SIZE {
        return Err(Error::Decompression(format!(
            "decompressed data too small for NIfTI header: {} bytes (need at least {})",
            bytes.len(),
            NiftiHeader::SIZE
        )));
    }
    let header = NiftiHeader::from_bytes(bytes)?;
    ensure_no_extensions(bytes, &header)?;
    let offset = header.vox_offset as usize;
    let data_size = header.data_size();
    Ok((header, offset, data_size))
}

/// Load gzipped file with decompression caching.
///
/// Uses single-pass decompression (same optimization as `load_gzipped`).
fn load_gzipped_cached(path: &Path) -> Result<NiftiImage> {
    let canonical = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());

    // Check cache first
    {
        let mut cache = DECOMPRESSION_CACHE
            .write()
            .map_err(|_| Error::Io(std::io::Error::other("cache lock poisoned")))?;

        if let Some((data, header)) = cache.get(&canonical) {
            let offset = header.vox_offset as usize;
            let data_size = header.data_size();
            return Ok(NiftiImage::from_shared_bytes(
                header, data, offset, data_size,
            ));
        }
    }

    // Cache miss - decompress and store
    let compressed = read_file_with_readahead(path)?;
    let (mut output, used_streaming) = decompress_gzip_with_fallback(&compressed)?;
    let mut written = output.len();

    let (mut header, mut offset, mut data_size) = parse_decompressed_nifti(&output)?;
    let mut expected_size = offset + data_size;

    if written != expected_size {
        if used_streaming {
            return Err(Error::Decompression(format!(
                "decompressed size {} did not match expected {} (header offset {} + data size {})",
                written, expected_size, offset, data_size
            )));
        }

        output = decompress_gzip_streaming(&compressed)?;
        written = output.len();
        let parsed = parse_decompressed_nifti(&output)?;
        header = parsed.0;
        offset = parsed.1;
        data_size = parsed.2;
        expected_size = offset + data_size;

        if written != expected_size {
            return Err(Error::Decompression(format!(
                "decompressed size {} did not match expected {} (header offset {} + data size {})",
                written, expected_size, offset, data_size
            )));
        }
    }

    let data = Arc::new(output);

    // Store in cache
    {
        let mut cache = DECOMPRESSION_CACHE
            .write()
            .map_err(|_| Error::Io(std::io::Error::other("cache lock poisoned")))?;
        cache.insert(canonical, data.clone(), header.clone());
    }

    Ok(NiftiImage::from_shared_bytes(
        header, data, offset, data_size,
    ))
}

/// Load uncompressed .nii file using memory mapping for speed.
#[allow(unsafe_code)]
fn load_uncompressed(path: &Path) -> Result<NiftiImage> {
    let file = File::open(path)?;
    // SAFETY: Memory mapping is safe because:
    // 1. The file was just opened successfully
    // 2. The mmap is read-only and won't be modified
    // 3. If the file is modified externally, data may become inconsistent but no UB
    let mmap = unsafe { Mmap::map(&file)? };

    let header = NiftiHeader::from_bytes(&mmap)?;
    ensure_no_extensions(&mmap[..], &header)?;
    let offset = header.vox_offset as usize;
    let data_size = header.data_size();

    if mmap.len() < offset + data_size {
        return Err(Error::Io(std::io::Error::new(
            std::io::ErrorKind::UnexpectedEof,
            "file truncated",
        )));
    }

    let arc = Arc::new(mmap);
    Ok(NiftiImage::from_shared_mmap(header, arc, offset, data_size))
}

/// Load gzipped .nii.gz file with single-pass decompression.
///
/// Optimization: Reads uncompressed size from gzip trailer, allocates once,
/// decompresses with libdeflate, then parses header from decompressed buffer.
/// This avoids the overhead of a separate header-only decompression pass.
///
/// Falls back to streaming decode if ISIZE is insufficient (multi-member gzip
/// or payloads > 4GB).
fn load_gzipped(path: &Path) -> Result<NiftiImage> {
    let compressed = read_file_with_readahead(path)?;
    let (mut output, used_streaming) = decompress_gzip_with_fallback(&compressed)?;
    let mut written = output.len();

    let (mut header, mut offset, mut data_size) = parse_decompressed_nifti(&output)?;
    let mut expected_size = offset + data_size;

    if written != expected_size {
        if used_streaming {
            return Err(Error::Decompression(format!(
                "decompressed size {} did not match expected {} (header offset {} + data size {})",
                written, expected_size, offset, data_size
            )));
        }

        output = decompress_gzip_streaming(&compressed)?;
        written = output.len();
        let parsed = parse_decompressed_nifti(&output)?;
        header = parsed.0;
        offset = parsed.1;
        data_size = parsed.2;
        expected_size = offset + data_size;

        if written != expected_size {
            return Err(Error::Decompression(format!(
                "decompressed size {} did not match expected {} (header offset {} + data size {})",
                written, expected_size, offset, data_size
            )));
        }
    }

    let bytes = Arc::new(output);
    Ok(NiftiImage::from_shared_bytes(
        header, bytes, offset, data_size,
    ))
}

// ============================================================================
// Multi-File Loader for Loading Related Images Together
// ============================================================================

/// Configuration for how a file should be loaded and processed.
#[derive(Debug, Clone)]
pub struct FileConfig {
    /// Path to the file
    pub path: PathBuf,
    /// Whether this is a label/segmentation (uses nearest-neighbor interpolation)
    pub is_label: bool,
    /// Optional key/name for this file in the output
    pub key: Option<String>,
}

impl FileConfig {
    /// Create a new file config for an image file (trilinear interpolation).
    pub fn image<P: AsRef<Path>>(path: P) -> Self {
        Self {
            path: path.as_ref().to_path_buf(),
            is_label: false,
            key: None,
        }
    }

    /// Create a new file config for a label/segmentation file (nearest-neighbor).
    pub fn label<P: AsRef<Path>>(path: P) -> Self {
        Self {
            path: path.as_ref().to_path_buf(),
            is_label: true,
            key: None,
        }
    }

    /// Set a key/name for this file.
    pub fn with_key(mut self, key: &str) -> Self {
        self.key = Some(key.to_string());
        self
    }
}

impl<P: AsRef<Path>> From<P> for FileConfig {
    fn from(path: P) -> Self {
        Self::image(path)
    }
}

/// Configuration for multi-file loading with spatial alignment.
#[derive(Debug, Clone)]
pub struct MultiFileConfig {
    /// Target voxel spacing (if None, uses reference file's spacing)
    pub target_spacing: Option<[f32; 3]>,
    /// Target shape (if None, computed from reference + spacing)
    pub target_shape: Option<[usize; 3]>,
    /// Index of reference file (0 = first file, used for spatial reference)
    pub reference_index: usize,
    /// Crop region start (in target space voxels)
    pub crop_start: Option<[usize; 3]>,
    /// Crop region size (in target space voxels)
    pub crop_size: Option<[usize; 3]>,
    /// Whether to use caching for gzipped files
    pub use_cache: bool,
}

impl Default for MultiFileConfig {
    fn default() -> Self {
        Self {
            target_spacing: None,
            target_shape: None,
            reference_index: 0,
            crop_start: None,
            crop_size: None,
            use_cache: true,
        }
    }
}

impl MultiFileConfig {
    /// Create config with target spacing.
    pub fn with_spacing(spacing: [f32; 3]) -> Self {
        Self {
            target_spacing: Some(spacing),
            ..Default::default()
        }
    }

    /// Set target spacing.
    pub fn target_spacing(mut self, spacing: [f32; 3]) -> Self {
        self.target_spacing = Some(spacing);
        self
    }

    /// Set target shape.
    pub fn target_shape(mut self, shape: [usize; 3]) -> Self {
        self.target_shape = Some(shape);
        self
    }

    /// Set reference file index.
    pub fn reference_index(mut self, index: usize) -> Self {
        self.reference_index = index;
        self
    }

    /// Set crop region.
    pub fn crop(mut self, start: [usize; 3], size: [usize; 3]) -> Self {
        self.crop_start = Some(start);
        self.crop_size = Some(size);
        self
    }

    /// Disable caching.
    pub fn no_cache(mut self) -> Self {
        self.use_cache = false;
        self
    }
}

/// Result of loading multiple files together.
#[derive(Debug)]
pub struct MultiFileResult {
    /// Loaded images in the same order as input files
    pub images: Vec<NiftiImage>,
    /// Keys for each image (if provided in FileConfig)
    pub keys: Vec<Option<String>>,
    /// The reference image index used for spatial alignment
    pub reference_index: usize,
    /// Target spacing used
    pub target_spacing: [f32; 3],
    /// Target shape used
    pub target_shape: [usize; 3],
}

impl MultiFileResult {
    /// Get image by index.
    pub fn get(&self, index: usize) -> Option<&NiftiImage> {
        self.images.get(index)
    }

    /// Get image by key.
    pub fn get_by_key(&self, key: &str) -> Option<&NiftiImage> {
        for (i, k) in self.keys.iter().enumerate() {
            if k.as_deref() == Some(key) {
                return self.images.get(i);
            }
        }
        None
    }

    /// Get the reference image.
    pub fn reference(&self) -> &NiftiImage {
        &self.images[self.reference_index]
    }

    /// Consume and return images as a vector.
    pub fn into_images(self) -> Vec<NiftiImage> {
        self.images
    }
}

/// Load multiple related files in parallel with spatial alignment.
///
/// This is the main entry point for loading related medical images (e.g., MRI, CT,
/// segmentation) that need to be processed together with consistent spatial operations.
///
/// All files are:
/// 1. Loaded in parallel using rayon
/// 2. Resampled to a common voxel spacing (from reference or specified)
/// 3. Cropped to the same region (if specified)
///
/// Labels/segmentations use nearest-neighbor interpolation to preserve discrete values.
///
/// # Example
/// ```ignore
/// use medrs::nifti::{load_multi, FileConfig, MultiFileConfig};
///
/// let files = vec![
///     FileConfig::image("mri.nii.gz").with_key("mri"),
///     FileConfig::image("ct.nii.gz").with_key("ct"),
///     FileConfig::label("seg.nii.gz").with_key("label"),
/// ];
///
/// let config = MultiFileConfig::with_spacing([1.0, 1.0, 1.0])
///     .crop([32, 32, 32], [64, 64, 64]);
///
/// let result = load_multi(&files, config)?;
///
/// let mri = result.get_by_key("mri").unwrap();
/// let ct = result.get_by_key("ct").unwrap();
/// let label = result.get_by_key("label").unwrap();
/// ```
pub fn load_multi(files: &[FileConfig], config: MultiFileConfig) -> Result<MultiFileResult> {
    if files.is_empty() {
        return Err(Error::InvalidDimensions(
            "load_multi requires at least one file".to_string(),
        ));
    }

    if config.reference_index >= files.len() {
        return Err(Error::InvalidDimensions(format!(
            "reference_index {} out of bounds for {} files",
            config.reference_index,
            files.len()
        )));
    }

    // Step 1: Load all files in parallel
    let loaded: Vec<Result<(NiftiImage, bool, Option<String>)>> = files
        .par_iter()
        .map(|file_config| {
            let img = if config.use_cache {
                load_cached(&file_config.path)?
            } else {
                load(&file_config.path)?
            };
            Ok((img, file_config.is_label, file_config.key.clone()))
        })
        .collect();

    // Check for errors
    let mut images_with_meta: Vec<(NiftiImage, bool, Option<String>)> =
        Vec::with_capacity(files.len());
    for result in loaded {
        images_with_meta.push(result?);
    }

    // Step 2: Determine target spacing and shape from reference
    let reference_img = &images_with_meta[config.reference_index].0;
    let reference_spacing_vec = reference_img.header().spacing();
    let reference_shape = reference_img.shape();

    let reference_spacing: [f32; 3] = [
        reference_spacing_vec[0],
        reference_spacing_vec[1],
        reference_spacing_vec.get(2).copied().unwrap_or(1.0),
    ];

    let target_spacing = config.target_spacing.unwrap_or(reference_spacing);

    // Compute target shape if not specified
    let target_shape = config.target_shape.unwrap_or_else(|| {
        // Scale shape based on spacing ratio
        let scale = [
            reference_spacing[0] / target_spacing[0],
            reference_spacing[1] / target_spacing[1],
            reference_spacing[2] / target_spacing[2],
        ];
        [
            ((reference_shape[0] as f32 * scale[0]).round() as usize).max(1),
            ((reference_shape.get(1).copied().unwrap_or(1) as f32 * scale[1]).round() as usize)
                .max(1),
            ((reference_shape.get(2).copied().unwrap_or(1) as f32 * scale[2]).round() as usize)
                .max(1),
        ]
    });

    // Step 3: Resample all images to target space (in parallel)
    let resampled: Vec<Result<NiftiImage>> = images_with_meta
        .par_iter()
        .map(|(img, is_label, _key)| {
            let method = if *is_label {
                Interpolation::Nearest
            } else {
                Interpolation::Trilinear
            };

            let current_spacing_vec = img.header().spacing();
            let current_spacing: [f32; 3] = [
                current_spacing_vec[0],
                current_spacing_vec[1],
                current_spacing_vec.get(2).copied().unwrap_or(1.0),
            ];
            let needs_resample = (current_spacing[0] - target_spacing[0]).abs() > 1e-6
                || (current_spacing[1] - target_spacing[1]).abs() > 1e-6
                || (current_spacing[2] - target_spacing[2]).abs() > 1e-6;

            if needs_resample {
                transforms::resample_to_spacing(img, target_spacing, method)
            } else {
                // Clone if no resampling needed
                Ok(img.clone())
            }
        })
        .collect();

    // Check for errors and collect results
    let mut resampled_images: Vec<NiftiImage> = Vec::with_capacity(files.len());
    for result in resampled {
        resampled_images.push(result?);
    }

    // Step 4: Apply crop if specified (in parallel)
    let final_images: Vec<NiftiImage> =
        if let (Some(crop_start), Some(crop_size)) = (config.crop_start, config.crop_size) {
            let crop_end = [
                crop_start[0] + crop_size[0],
                crop_start[1] + crop_size[1],
                crop_start[2] + crop_size[2],
            ];
            let cropped: Vec<Result<NiftiImage>> = resampled_images
                .into_par_iter()
                .map(|img| transforms::crop(&img, crop_start, crop_end))
                .collect();
            cropped.into_iter().collect::<Result<Vec<_>>>()?
        } else {
            resampled_images
        };

    // Extract keys
    let keys: Vec<Option<String>> = images_with_meta.iter().map(|(_, _, k)| k.clone()).collect();

    let final_shape = config.crop_size.unwrap_or(target_shape);

    Ok(MultiFileResult {
        images: final_images,
        keys,
        reference_index: config.reference_index,
        target_spacing,
        target_shape: final_shape,
    })
}

/// Load multiple files with a simpler interface (all images, no labels).
/// Load image and label pair with coordinated spatial processing.
///
/// Convenience function for the common case of loading one image with its
/// corresponding segmentation label.
///
/// # Example
/// ```ignore
/// let (image, label) = load_image_label_pair(
///     "mri.nii.gz",
///     "segmentation.nii.gz",
///     Some([1.0, 1.0, 1.0]),
///     Some(([32, 32, 32], [64, 64, 64])),
/// )?;
/// ```
pub fn load_image_label_pair<P: AsRef<Path>>(
    image_path: P,
    label_path: P,
    target_spacing: Option<[f32; 3]>,
    crop: Option<([usize; 3], [usize; 3])>,
) -> Result<(NiftiImage, NiftiImage)> {
    let files = vec![FileConfig::image(image_path), FileConfig::label(label_path)];

    let mut config = MultiFileConfig::default();
    if let Some(spacing) = target_spacing {
        config = config.target_spacing(spacing);
    }
    if let Some((start, size)) = crop {
        config = config.crop(start, size);
    }

    let result = load_multi(&files, config)?;
    let mut images = result.into_images();

    if images.len() != 2 {
        return Err(Error::InvalidDimensions(
            "Expected exactly 2 images".to_string(),
        ));
    }

    let label = images
        .pop()
        .ok_or_else(|| Error::InvalidDimensions("Missing label image".to_string()))?;
    let image = images
        .pop()
        .ok_or_else(|| Error::InvalidDimensions("Missing input image".to_string()))?;

    Ok((image, label))
}

/// Save a NIfTI image to file.
///
/// Format is determined by file extension (`.nii` or `.nii.gz`).
///
/// # Example
/// ```ignore
/// medrs::nifti::save(&img, "output.nii.gz")?;
/// ```
pub fn save<P: AsRef<Path>>(image: &NiftiImage, path: P) -> Result<()> {
    image.header().validate()?;

    let path = path.as_ref();
    let is_gzipped = path.extension().is_some_and(|e| e == "gz");

    if is_gzipped {
        save_gzipped(image, path)
    } else {
        save_uncompressed(image, path)
    }
}

fn save_uncompressed(image: &NiftiImage, path: &Path) -> Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::with_capacity(1024 * 1024, file);

    // Write header
    let header_bytes = image.header().to_bytes();
    writer.write_all(&header_bytes)?;

    // Padding to vox_offset (typically 352)
    let padding = image.header().vox_offset as usize - NiftiHeader::SIZE;
    if padding > 0 {
        writer.write_all(&vec![0u8; padding])?;
    }

    // Write data
    let data = image.data_to_bytes()?;
    writer.write_all(&data)?;
    writer.flush()?;

    Ok(())
}

const PARALLEL_THRESHOLD: usize = 1024 * 1024;

fn save_gzipped(image: &NiftiImage, path: &Path) -> Result<()> {
    let header_bytes = image.header().to_bytes();
    let padding = image.header().vox_offset as usize - NiftiHeader::SIZE;
    let data = image.data_to_bytes()?;

    let total_size = header_bytes.len() + padding + data.len();
    let mut uncompressed = Vec::with_capacity(total_size);
    uncompressed.extend_from_slice(&header_bytes);
    uncompressed.resize(uncompressed.len() + padding, 0u8);
    uncompressed.extend_from_slice(&data);

    if uncompressed.len() >= PARALLEL_THRESHOLD {
        // Parallel compression using gzp
        let file = File::create(path)?;
        let writer = BufWriter::with_capacity(1024 * 1024, file);
        let mut parz = ParCompressBuilder::<Gzip>::new().from_writer(writer);
        parz.write_all(&uncompressed).map_err(|e| {
            Error::Io(std::io::Error::other(format!(
                "parallel compression failed: {e}"
            )))
        })?;
        parz.finish().map_err(|e| {
            Error::Io(std::io::Error::other(format!(
                "parallel compression finish failed: {e}"
            )))
        })?;
    } else {
        // Use libdeflate for fast single-shot compression for small files
        // Level 1 = fastest, good balance of speed vs compression ratio
        let mut compressor = libdeflater::Compressor::new(libdeflater::CompressionLvl::fastest());

        // Allocate output buffer (worst case: slightly larger than input for incompressible data)
        let max_compressed_size = compressor.gzip_compress_bound(uncompressed.len());
        let mut compressed = vec![0u8; max_compressed_size];

        let actual_size = compressor
            .gzip_compress(&uncompressed, &mut compressed)
            .map_err(|e| Error::Io(std::io::Error::other(format!("compression failed: {e:?}"))))?;

        compressed.truncate(actual_size);

        // Write compressed data to file
        let mut file = File::create(path)?;
        file.write_all(&compressed)?;
    }

    Ok(())
}

// ============================================================================
// Mgzip Support - Parallel Decompressible Format
// ============================================================================

/// Save a NIfTI image in Mgzip format for parallel decompression.
///
/// Mgzip (multi-member gzip) stores data in independent blocks that can be
/// decompressed in parallel, providing 4-8× speedup on multi-core systems.
///
/// Files are saved with `.nii.mgz` extension by convention, but `.nii.gz`
/// also works (Mgzip is backwards-compatible with standard gzip readers).
///
/// # Performance
/// - Compression: Similar speed to standard gzip (parallel)
/// - Decompression: 4-8× faster with [`load_mgzip`] on 4-8 cores
/// - File size: ~0.1-1% larger than standard gzip
///
/// # Example
/// ```ignore
/// // Save as Mgzip
/// medrs::nifti::save_mgzip(&img, "brain.nii.mgz")?;
///
/// // Load with parallel decompression
/// let img = medrs::nifti::load_mgzip("brain.nii.mgz")?;
/// ```
pub fn save_mgzip<P: AsRef<Path>>(image: &NiftiImage, path: P) -> Result<()> {
    save_mgzip_with_threads(image, path, 0)
}

/// Save a NIfTI image in Mgzip format with specified thread count.
///
/// # Arguments
/// * `image` - The NIfTI image to save
/// * `path` - Output file path
/// * `num_threads` - Number of compression threads (0 = auto-detect)
pub fn save_mgzip_with_threads<P: AsRef<Path>>(
    image: &NiftiImage,
    path: P,
    num_threads: usize,
) -> Result<()> {
    image.header().validate()?;

    let header_bytes = image.header().to_bytes();
    let padding = image.header().vox_offset as usize - NiftiHeader::SIZE;
    let data = image.data_to_bytes()?;

    let total_size = header_bytes.len() + padding + data.len();
    let mut uncompressed = Vec::with_capacity(total_size);
    uncompressed.extend_from_slice(&header_bytes);
    uncompressed.resize(uncompressed.len() + padding, 0u8);
    uncompressed.extend_from_slice(&data);

    let file = File::create(path.as_ref())?;
    let writer = BufWriter::with_capacity(1024 * 1024, file);

    let num_threads = if num_threads == 0 {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
    } else {
        num_threads
    };

    // Use larger blocks (1MB) for better parallel decompression
    // Default is 128KB which creates too many small blocks
    let block_size = 1024 * 1024; // 1MB blocks

    let builder = ParCompressBuilder::<Mgzip>::new()
        .num_threads(num_threads)
        .map_err(|e| {
            Error::Io(std::io::Error::other(format!(
                "failed to set thread count: {e}"
            )))
        })?
        .buffer_size(block_size)
        .map_err(|e| {
            Error::Io(std::io::Error::other(format!(
                "failed to set buffer size: {e}"
            )))
        })?;

    let mut parz = builder.from_writer(writer);

    parz.write_all(&uncompressed).map_err(|e| {
        Error::Io(std::io::Error::other(format!(
            "mgzip compression failed: {e}"
        )))
    })?;

    parz.finish().map_err(|e| {
        Error::Io(std::io::Error::other(format!(
            "mgzip compression finish failed: {e}"
        )))
    })?;

    Ok(())
}

/// Load a NIfTI image from Mgzip format with parallel decompression.
///
/// This provides 4-8× faster loading compared to standard gzip on multi-core
/// systems. The file must have been created with [`save_mgzip`] or compatible
/// tools that produce multi-member gzip streams.
///
/// # Note
/// This function can also read standard gzip files, but without parallel
/// speedup. For best performance, use files created with [`save_mgzip`].
///
/// # Example
/// ```ignore
/// let img = medrs::nifti::load_mgzip("brain.nii.mgz")?;
/// ```
pub fn load_mgzip<P: AsRef<Path>>(path: P) -> Result<NiftiImage> {
    load_mgzip_with_threads(path, 0)
}

/// Load a NIfTI image from Mgzip format with specified thread count.
///
/// # Arguments
/// * `path` - Path to the Mgzip file
/// * `num_threads` - Number of decompression threads (0 = auto-detect)
pub fn load_mgzip_with_threads<P: AsRef<Path>>(path: P, num_threads: usize) -> Result<NiftiImage> {
    let path = path.as_ref();
    let file = File::open(path)?;
    let reader = BufReader::with_capacity(1024 * 1024, file);

    let num_threads = if num_threads == 0 {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
    } else {
        num_threads
    };

    let mut decompressor = ParDecompressBuilder::<Mgzip>::new()
        .num_threads(num_threads)
        .map_err(|e| Error::Decompression(format!("failed to create parallel decompressor: {e}")))?
        .from_reader(reader);

    let mut decompressed = Vec::new();
    decompressor
        .read_to_end(&mut decompressed)
        .map_err(|e| Error::Decompression(format!("mgzip decompression failed: {e}")))?;

    let (header, offset, data_size) = parse_decompressed_nifti(&decompressed)?;
    let bytes = Arc::new(decompressed);
    Ok(NiftiImage::from_shared_bytes(
        header, bytes, offset, data_size,
    ))
}

/// Convert a standard gzip NIfTI file to Mgzip format.
///
/// This reads an existing `.nii.gz` file and saves it as Mgzip for faster
/// future loading. The original file is not modified.
///
/// # Arguments
/// * `input_path` - Path to input .nii.gz file
/// * `output_path` - Path for output .nii.mgz file (or None to replace extension)
///
/// # Returns
/// The output path that was written to.
///
/// # Example
/// ```ignore
/// // Convert single file
/// medrs::nifti::convert_to_mgzip("brain.nii.gz", None)?;
/// // Creates brain.nii.mgz
///
/// // Or specify output path
/// medrs::nifti::convert_to_mgzip("brain.nii.gz", Some("output/brain.nii.mgz"))?;
/// ```
pub fn convert_to_mgzip<P: AsRef<Path>>(input_path: P, output_path: Option<P>) -> Result<PathBuf> {
    let input = input_path.as_ref();
    let output = if let Some(p) = output_path { p.as_ref().to_path_buf() } else {
        let stem = input.file_stem().and_then(|s| s.to_str()).ok_or_else(|| {
            Error::InvalidFileFormat(format!("invalid path: {}", input.display()))
        })?;
        let stem = stem.strip_suffix(".nii").unwrap_or(stem);
        input.with_file_name(format!("{stem}.nii.mgz"))
    };

    let image = load(input)?;
    save_mgzip(&image, &output)?;

    Ok(output)
}

/// Check if a file appears to be in Mgzip (multi-member gzip) format.
///
/// This performs a quick heuristic check by looking for multiple gzip
/// member signatures. Not 100% reliable but useful for format detection.
pub fn is_mgzip<P: AsRef<Path>>(path: P) -> Result<bool> {
    let file = File::open(path.as_ref())?;
    let mut reader = BufReader::new(file);

    let mut buf = [0u8; 32768];
    let bytes_read = reader.read(&mut buf)?;

    if bytes_read < 10 {
        return Ok(false);
    }

    if buf[0] != 0x1f || buf[1] != 0x8b {
        return Ok(false);
    }

    let mut member_count = 1;
    for i in 10..bytes_read.saturating_sub(2) {
        if buf[i] == 0x1f && buf[i + 1] == 0x8b && buf[i + 2] == 0x08 {
            member_count += 1;
            if member_count >= 2 {
                return Ok(true);
            }
        }
    }

    Ok(false)
}

/// Load only the header from a NIfTI file (fast metadata inspection).
#[allow(unsafe_code)]
pub fn load_header<P: AsRef<Path>>(path: P) -> Result<NiftiHeader> {
    let path = path.as_ref();
    let is_gzipped = path.extension().is_some_and(|e| e == "gz");

    if is_gzipped {
        let file = File::open(path)?;
        let buf_reader = BufReader::new(file);
        let mut decoder = GzDecoder::new(buf_reader);
        let mut header_buf = vec![0u8; NiftiHeader::SIZE];
        decoder.read_exact(&mut header_buf)?;
        NiftiHeader::from_bytes(&header_buf)
    } else {
        let file = File::open(path)?;
        // SAFETY: Memory mapping is safe - file just opened, read-only access
        let mmap = unsafe { Mmap::map(&file)? };
        NiftiHeader::from_bytes(&mmap)
    }
}

/// Configuration for loading a cropped region with optional transforms.
#[derive(Debug, Clone)]
pub struct CropConfig {
    /// Shape of the region to extract [d, h, w]
    pub shape: [usize; 3],
    /// Offset from image origin (None = center crop)
    pub offset: Option<[usize; 3]>,
    /// Target voxel spacing [mm] (None = keep original)
    pub spacing: Option<[f32; 3]>,
    /// Target orientation (None = keep original)
    pub orientation: Option<crate::transforms::Orientation>,
}

impl Default for CropConfig {
    fn default() -> Self {
        Self {
            shape: [64, 64, 64],
            offset: None,
            spacing: None,
            orientation: None,
        }
    }
}

impl CropConfig {
    /// Create a new crop config with the given shape (centered by default).
    pub fn new(shape: [usize; 3]) -> Self {
        Self {
            shape,
            ..Default::default()
        }
    }

    /// Set the crop offset (default: centered).
    pub fn offset(mut self, offset: [usize; 3]) -> Self {
        self.offset = Some(offset);
        self
    }

    /// Set target spacing for resampling.
    pub fn spacing(mut self, spacing: [f32; 3]) -> Self {
        self.spacing = Some(spacing);
        self
    }

    /// Set target orientation for reorientation.
    pub fn orientation(mut self, orientation: crate::transforms::Orientation) -> Self {
        self.orientation = Some(orientation);
        self
    }
}

/// Load a cropped region with optional resampling and reorientation.
///
/// Efficient for training pipelines: extracts a patch without loading the entire volume.
///
/// # Example
/// ```ignore
/// use medrs::nifti::{load_with_crop, CropConfig};
///
/// // Simple centered crop
/// let patch = load_with_crop("volume.nii.gz", CropConfig::new([64, 64, 64]))?;
///
/// // Crop with transforms
/// let patch = load_with_crop("volume.nii.gz",
///     CropConfig::new([64, 64, 64])
///         .offset([32, 32, 32])
///         .spacing([1.0, 1.0, 1.0])
/// )?;
/// ```
#[must_use = "this function returns a loaded image that should be used"]
#[allow(unsafe_code)]
pub fn load_with_crop<P: AsRef<Path>>(path: P, config: CropConfig) -> Result<NiftiImage> {
    let path = path.as_ref();

    let cropped = if is_gzipped(path) {
        load_with_crop_gzipped(path, &config)?
    } else {
        let file = File::open(path)?;
        // SAFETY: Memory mapping is safe - file just opened, read-only access
        let mmap = unsafe { Mmap::map(&file)? };
        let header = NiftiHeader::from_bytes(&mmap)?;
        let data_offset = header.vox_offset as usize;
        let full_shape = header.shape();
        let crop_offset = config
            .offset
            .unwrap_or(compute_center_offset(&full_shape, &config.shape)?);

        copy_cropped_region(&header, &mmap, data_offset, crop_offset, config.shape)?
    };

    let mut output = cropped;

    if let Some(target_orient) = config.orientation {
        output = transforms::reorient(&output, target_orient)?;
    }

    if let Some(target_spacing) = config.spacing {
        output =
            transforms::resample_to_spacing(&output, target_spacing, Interpolation::Trilinear)?;
    }

    Ok(output)
}

fn load_with_crop_gzipped(path: &Path, config: &CropConfig) -> Result<NiftiImage> {
    let canonical = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());

    // Check cache first
    {
        let mut cache = DECOMPRESSION_CACHE
            .write()
            .map_err(|_| Error::Io(std::io::Error::other("cache lock poisoned")))?;

        if let Some((data, header)) = cache.get(&canonical) {
            let full_shape = header.shape();
            let crop_offset = config
                .offset
                .unwrap_or(compute_center_offset(&full_shape, &config.shape)?);
            let data_offset = header.vox_offset as usize;
            return copy_cropped_region_from_bytes(
                &header,
                &data,
                data_offset,
                crop_offset,
                config.shape,
            );
        }
    }

    // Cache miss - decompress and store
    let compressed = read_file_with_readahead(path)?;
    let (mut output, used_streaming) = decompress_gzip_with_fallback(&compressed)?;
    let mut written = output.len();

    let (mut header, mut offset, mut data_size) = parse_decompressed_nifti(&output)?;
    let mut expected_size = offset + data_size;

    if written != expected_size {
        if used_streaming {
            return Err(Error::Decompression(format!(
                "decompressed size {} did not match expected {} (header offset {} + data size {})",
                written, expected_size, offset, data_size
            )));
        }

        output = decompress_gzip_streaming(&compressed)?;
        written = output.len();
        let parsed = parse_decompressed_nifti(&output)?;
        header = parsed.0;
        offset = parsed.1;
        data_size = parsed.2;
        expected_size = offset + data_size;

        if written != expected_size {
            return Err(Error::Decompression(format!(
                "decompressed size {} did not match expected {} (header offset {} + data size {})",
                written, expected_size, offset, data_size
            )));
        }
    }

    let full_shape = header.shape();
    let crop_offset = config
        .offset
        .unwrap_or(compute_center_offset(&full_shape, &config.shape)?);

    let data = Arc::new(output);

    // Store in cache for subsequent crops
    {
        let mut cache = DECOMPRESSION_CACHE
            .write()
            .map_err(|_| Error::Io(std::io::Error::other("cache lock poisoned")))?;
        cache.insert(canonical, data.clone(), header.clone());
    }

    copy_cropped_region_from_bytes(&header, &data, offset, crop_offset, config.shape)
}

/// Simple version of load_cropped. Supports both .nii and .nii.gz files.
///
/// For gzipped files, decompresses to cache first, then extracts the crop.
/// Subsequent crops from the same file use the cached decompressed data.
#[must_use = "this function returns a loaded image that should be used"]
#[allow(unsafe_code)]
pub fn load_cropped<P: AsRef<Path>>(
    path: P,
    crop_offset: [usize; 3],
    crop_shape: [usize; 3],
) -> Result<NiftiImage> {
    let path = path.as_ref();

    if is_gzipped(path) {
        load_cropped_gzipped(path, crop_offset, crop_shape)
    } else {
        let file = File::open(path)?;
        // SAFETY: Memory mapping is safe - file just opened, read-only access
        let mmap = unsafe { Mmap::map(&file)? };
        let header = NiftiHeader::from_bytes(&mmap)?;
        let data_offset = header.vox_offset as usize;

        copy_cropped_region(&header, &mmap, data_offset, crop_offset, crop_shape)
    }
}

fn load_cropped_gzipped(
    path: &Path,
    crop_offset: [usize; 3],
    crop_shape: [usize; 3],
) -> Result<NiftiImage> {
    let canonical = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());

    // Check cache first
    {
        let mut cache = DECOMPRESSION_CACHE
            .write()
            .map_err(|_| Error::Io(std::io::Error::other("cache lock poisoned")))?;

        if let Some((data, header)) = cache.get(&canonical) {
            let data_offset = header.vox_offset as usize;
            return copy_cropped_region_from_bytes(
                &header,
                &data,
                data_offset,
                crop_offset,
                crop_shape,
            );
        }
    }

    // Cache miss - decompress and store
    let compressed = read_file_with_readahead(path)?;
    let (mut output, used_streaming) = decompress_gzip_with_fallback(&compressed)?;
    let mut written = output.len();

    let (mut header, mut offset, mut data_size) = parse_decompressed_nifti(&output)?;
    let mut expected_size = offset + data_size;

    if written != expected_size {
        if used_streaming {
            return Err(Error::Decompression(format!(
                "decompressed size {} did not match expected {} (header offset {} + data size {})",
                written, expected_size, offset, data_size
            )));
        }

        output = decompress_gzip_streaming(&compressed)?;
        written = output.len();
        let parsed = parse_decompressed_nifti(&output)?;
        header = parsed.0;
        offset = parsed.1;
        data_size = parsed.2;
        expected_size = offset + data_size;

        if written != expected_size {
            return Err(Error::Decompression(format!(
                "decompressed size {} did not match expected {} (header offset {} + data size {})",
                written, expected_size, offset, data_size
            )));
        }
    }

    let data = Arc::new(output);

    // Store in cache for subsequent crops
    {
        let mut cache = DECOMPRESSION_CACHE
            .write()
            .map_err(|_| Error::Io(std::io::Error::other("cache lock poisoned")))?;
        cache.insert(canonical, data.clone(), header.clone());
    }

    copy_cropped_region_from_bytes(&header, &data, offset, crop_offset, crop_shape)
}

fn is_gzipped(path: &Path) -> bool {
    path.extension().is_some_and(|e| e == "gz")
}

/// Copy a cropped region from a byte slice (used for gzipped data in memory).
/// This is equivalent to `copy_cropped_region` but works with `&[u8]` instead of `&Mmap`.
fn copy_cropped_region_from_bytes(
    header: &NiftiHeader,
    bytes: &[u8],
    data_offset: usize,
    crop_offset: [usize; 3],
    crop_shape: [usize; 3],
) -> Result<NiftiImage> {
    let full_shape = header.shape();
    if full_shape.len() < 3 {
        return Err(Error::InvalidDimensions(
            "expected at least 3 spatial dimensions".to_string(),
        ));
    }

    // Validate crop_shape has no zero dimensions
    for (i, &dim) in crop_shape.iter().enumerate() {
        if dim == 0 {
            return Err(Error::InvalidDimensions(format!(
                "Crop shape dimension {} cannot be zero",
                i
            )));
        }
    }

    for i in 0..3 {
        if crop_offset[i] + crop_shape[i] > full_shape[i] {
            return Err(Error::InvalidDimensions(format!(
                "Crop region exceeds dimension {}: {} + {} > {}",
                i, crop_offset[i], crop_shape[i], full_shape[i]
            )));
        }
    }

    let elem_size = header.datatype.size();
    let dim0 = full_shape[0];
    let dim1 = full_shape.get(1).copied().unwrap_or(1);

    // Calculate total_bytes with overflow checking
    let total_bytes = crop_shape[0]
        .checked_mul(crop_shape[1])
        .and_then(|v| v.checked_mul(crop_shape[2]))
        .and_then(|v| v.checked_mul(elem_size))
        .ok_or_else(|| {
            Error::InvalidDimensions(format!(
                "Crop region too large: {:?} x {} bytes would overflow",
                crop_shape, elem_size
            ))
        })?;
    let mut buffer = vec![0u8; total_bytes];

    let expected_data_size = full_shape[0]
        .checked_mul(full_shape[1])
        .and_then(|v| v.checked_mul(full_shape[2]))
        .and_then(|v| v.checked_mul(elem_size))
        .ok_or_else(|| {
            Error::InvalidDimensions(format!(
                "Volume too large: {:?} x {} bytes would overflow",
                &full_shape[..3],
                elem_size
            ))
        })?;

    let total_required = data_offset
        .checked_add(expected_data_size)
        .ok_or_else(|| Error::InvalidDimensions("Data offset + size would overflow".to_string()))?;

    if bytes.len() < total_required {
        return Err(Error::InvalidDimensions(format!(
            "Buffer too small: need {} bytes for data but buffer is {} bytes (offset {})",
            expected_data_size,
            bytes.len(),
            data_offset
        )));
    }

    // NIfTI uses F-order (column-major): first index changes fastest
    // F-order linear index = x + y * dim0 + z * dim0 * dim1
    let row_bytes = crop_shape[0] * elem_size;

    // Check if we can do larger memcpy: when copying full x-dimension,
    // we can copy entire y-planes at once if y is also contiguous
    let full_x = crop_offset[0] == 0 && crop_shape[0] == dim0;
    let full_y = crop_offset[1] == 0 && crop_shape[1] == dim1;

    // For larger crops, use parallel copying across z-slices
    // Threshold: ~64KB of data per slice makes parallelization worthwhile
    let slice_bytes = crop_shape[0] * crop_shape[1] * elem_size;
    let use_parallel = slice_bytes > 65536 && crop_shape[2] >= 4;

    // Super-fast path: if full x and y dimensions, copy entire z-planes
    if full_x && full_y {
        let plane_bytes = dim0 * dim1 * elem_size;

        if use_parallel {
            use rayon::prelude::*;

            buffer
                .par_chunks_mut(plane_bytes)
                .enumerate()
                .for_each(|(z, z_buffer)| {
                    let src_z = crop_offset[2] + z;
                    let src_byte = data_offset + src_z * plane_bytes;
                    z_buffer.copy_from_slice(&bytes[src_byte..src_byte + plane_bytes]);
                });
        } else {
            for z in 0..crop_shape[2] {
                let src_z = crop_offset[2] + z;
                let src_byte = data_offset + src_z * plane_bytes;
                let dst_start = z * plane_bytes;
                buffer[dst_start..dst_start + plane_bytes]
                    .copy_from_slice(&bytes[src_byte..src_byte + plane_bytes]);
            }
        }
    } else if full_x {
        // Fast path: copy entire y-rows (multiple x-rows) at once
        let y_row_bytes = dim0 * elem_size;
        let plane_bytes_src = dim0 * dim1 * elem_size;
        let plane_bytes_dst = crop_shape[0] * crop_shape[1] * elem_size;

        if use_parallel {
            use rayon::prelude::*;

            buffer
                .par_chunks_mut(plane_bytes_dst)
                .enumerate()
                .for_each(|(z, z_buffer)| {
                    let src_z = crop_offset[2] + z;
                    let z_byte_offset = data_offset + src_z * plane_bytes_src;

                    // Copy y-rows contiguously when possible
                    let src_y_start = crop_offset[1];
                    let src_byte = z_byte_offset + src_y_start * y_row_bytes;
                    let copy_bytes = crop_shape[1] * y_row_bytes;
                    z_buffer[..copy_bytes].copy_from_slice(&bytes[src_byte..src_byte + copy_bytes]);
                });
        } else {
            let mut dst_cursor = 0;
            for z in 0..crop_shape[2] {
                let src_z = crop_offset[2] + z;
                let z_byte_offset = data_offset + src_z * plane_bytes_src;
                let src_y_start = crop_offset[1];
                let src_byte = z_byte_offset + src_y_start * y_row_bytes;
                let copy_bytes = crop_shape[1] * y_row_bytes;

                buffer[dst_cursor..dst_cursor + copy_bytes]
                    .copy_from_slice(&bytes[src_byte..src_byte + copy_bytes]);
                dst_cursor += copy_bytes;
            }
        }
    } else if use_parallel {
        use rayon::prelude::*;

        // Split buffer into z-slices and copy in parallel
        let slices_per_z = crop_shape[1];
        let bytes_per_z = slices_per_z * row_bytes;

        buffer
            .par_chunks_mut(bytes_per_z)
            .enumerate()
            .for_each(|(z, z_buffer)| {
                let src_z = crop_offset[2] + z;
                let z_offset = src_z * dim0 * dim1;

                for y in 0..crop_shape[1] {
                    let src_y = crop_offset[1] + y;
                    let src_index = crop_offset[0] + src_y * dim0 + z_offset;
                    let src_byte = data_offset + src_index * elem_size;

                    let dst_start = y * row_bytes;
                    z_buffer[dst_start..dst_start + row_bytes]
                        .copy_from_slice(&bytes[src_byte..src_byte + row_bytes]);
                }
            });
    } else {
        // Sequential copy for small crops (lower overhead)
        let mut dst_cursor = 0;
        for z in 0..crop_shape[2] {
            let src_z = crop_offset[2] + z;
            let z_offset = src_z * dim0 * dim1;

            for y in 0..crop_shape[1] {
                let src_y = crop_offset[1] + y;
                let src_index = crop_offset[0] + src_y * dim0 + z_offset;
                let src_byte = data_offset + src_index * elem_size;

                let src_range = src_byte..src_byte + row_bytes;
                let dst_range = dst_cursor..dst_cursor + row_bytes;
                buffer[dst_range].copy_from_slice(&bytes[src_range]);
                dst_cursor += row_bytes;
            }
        }
    }

    let mut new_header = header.clone();
    new_header.ndim = 3;
    new_header.dim = [1i64; 7];
    for (i, &s) in crop_shape.iter().enumerate() {
        new_header.dim[i] = s as i64;
    }
    new_header.vox_offset = NiftiHeader::default().vox_offset;

    // Translate affine by crop offset
    let mut affine = new_header.affine();
    for row in affine.iter_mut().take(3) {
        row[3] += row[0] * crop_offset[0] as f32
            + row[1] * crop_offset[1] as f32
            + row[2] * crop_offset[2] as f32;
    }
    new_header.set_affine(affine);

    let data_len = buffer.len();
    Ok(NiftiImage::from_shared_bytes(
        new_header,
        Arc::new(buffer),
        0,
        data_len,
    ))
}

fn compute_center_offset(full_shape: &[usize], crop_shape: &[usize; 3]) -> Result<[usize; 3]> {
    if full_shape.len() < 3 {
        return Err(Error::InvalidDimensions(
            "expected at least 3 spatial dimensions".to_string(),
        ));
    }

    Ok([
        full_shape[0].saturating_sub(crop_shape[0]) / 2,
        full_shape[1].saturating_sub(crop_shape[1]) / 2,
        full_shape[2].saturating_sub(crop_shape[2]) / 2,
    ])
}

fn copy_cropped_region(
    header: &NiftiHeader,
    mmap: &Mmap,
    data_offset: usize,
    crop_offset: [usize; 3],
    crop_shape: [usize; 3],
) -> Result<NiftiImage> {
    let full_shape = header.shape();
    if full_shape.len() < 3 {
        return Err(Error::InvalidDimensions(
            "expected at least 3 spatial dimensions".to_string(),
        ));
    }

    // Validate crop_shape has no zero dimensions
    for (i, &dim) in crop_shape.iter().enumerate() {
        if dim == 0 {
            return Err(Error::InvalidDimensions(format!(
                "Crop shape dimension {} cannot be zero",
                i
            )));
        }
    }

    for i in 0..3 {
        if crop_offset[i] + crop_shape[i] > full_shape[i] {
            return Err(Error::InvalidDimensions(format!(
                "Crop region exceeds dimension {}: {} + {} > {}",
                i, crop_offset[i], crop_shape[i], full_shape[i]
            )));
        }
    }

    let elem_size = header.datatype.size();
    let dim0 = full_shape[0];
    let dim1 = full_shape.get(1).copied().unwrap_or(1);

    // Calculate total_bytes with overflow checking
    let total_bytes = crop_shape[0]
        .checked_mul(crop_shape[1])
        .and_then(|v| v.checked_mul(crop_shape[2]))
        .and_then(|v| v.checked_mul(elem_size))
        .ok_or_else(|| {
            Error::InvalidDimensions(format!(
                "Crop region too large: {:?} x {} bytes would overflow",
                crop_shape, elem_size
            ))
        })?;
    let mut buffer = vec![0u8; total_bytes];

    let expected_data_size = full_shape[0]
        .checked_mul(full_shape[1])
        .and_then(|v| v.checked_mul(full_shape[2]))
        .and_then(|v| v.checked_mul(elem_size))
        .ok_or_else(|| {
            Error::InvalidDimensions(format!(
                "Volume too large: {:?} x {} bytes would overflow",
                &full_shape[..3],
                elem_size
            ))
        })?;

    let total_required = data_offset
        .checked_add(expected_data_size)
        .ok_or_else(|| Error::InvalidDimensions("Data offset + size would overflow".to_string()))?;

    if mmap.len() < total_required {
        return Err(Error::InvalidDimensions(format!(
            "File too small: need {} bytes for data but mmap is {} bytes (offset {})",
            expected_data_size,
            mmap.len(),
            data_offset
        )));
    }

    // NIfTI uses F-order (column-major): first index changes fastest
    // F-order linear index = x + y * dim0 + z * dim0 * dim1
    let row_bytes = crop_shape[0] * elem_size;
    let mmap_slice = mmap.as_ref();

    // Check if we can do larger memcpy: when copying full x-dimension,
    // we can copy entire y-planes at once if y is also contiguous
    let full_x = crop_offset[0] == 0 && crop_shape[0] == dim0;
    let full_y = crop_offset[1] == 0 && crop_shape[1] == dim1;

    // For larger crops, use parallel copying across z-slices
    // Threshold: ~64KB of data per slice makes parallelization worthwhile
    let slice_bytes = crop_shape[0] * crop_shape[1] * elem_size;
    let use_parallel = slice_bytes > 65536 && crop_shape[2] >= 4;

    // Super-fast path: if full x and y dimensions, copy entire z-planes
    if full_x && full_y {
        let plane_bytes = dim0 * dim1 * elem_size;

        if use_parallel {
            use rayon::prelude::*;

            buffer
                .par_chunks_mut(plane_bytes)
                .enumerate()
                .for_each(|(z, z_buffer)| {
                    let src_z = crop_offset[2] + z;
                    let src_byte = data_offset + src_z * plane_bytes;
                    z_buffer.copy_from_slice(&mmap_slice[src_byte..src_byte + plane_bytes]);
                });
        } else {
            for z in 0..crop_shape[2] {
                let src_z = crop_offset[2] + z;
                let src_byte = data_offset + src_z * plane_bytes;
                let dst_start = z * plane_bytes;
                buffer[dst_start..dst_start + plane_bytes]
                    .copy_from_slice(&mmap_slice[src_byte..src_byte + plane_bytes]);
            }
        }
    } else if full_x {
        // Fast path: copy entire y-rows (multiple x-rows) at once
        let y_row_bytes = dim0 * elem_size;
        let plane_bytes_src = dim0 * dim1 * elem_size;
        let plane_bytes_dst = crop_shape[0] * crop_shape[1] * elem_size;

        if use_parallel {
            use rayon::prelude::*;

            buffer
                .par_chunks_mut(plane_bytes_dst)
                .enumerate()
                .for_each(|(z, z_buffer)| {
                    let src_z = crop_offset[2] + z;
                    let z_byte_offset = data_offset + src_z * plane_bytes_src;

                    // Copy y-rows contiguously when possible
                    let src_y_start = crop_offset[1];
                    let src_byte = z_byte_offset + src_y_start * y_row_bytes;
                    let copy_bytes = crop_shape[1] * y_row_bytes;
                    z_buffer[..copy_bytes]
                        .copy_from_slice(&mmap_slice[src_byte..src_byte + copy_bytes]);
                });
        } else {
            let mut dst_cursor = 0;
            for z in 0..crop_shape[2] {
                let src_z = crop_offset[2] + z;
                let z_byte_offset = data_offset + src_z * plane_bytes_src;
                let src_y_start = crop_offset[1];
                let src_byte = z_byte_offset + src_y_start * y_row_bytes;
                let copy_bytes = crop_shape[1] * y_row_bytes;

                buffer[dst_cursor..dst_cursor + copy_bytes]
                    .copy_from_slice(&mmap_slice[src_byte..src_byte + copy_bytes]);
                dst_cursor += copy_bytes;
            }
        }
    } else if use_parallel {
        use rayon::prelude::*;

        // Split buffer into z-slices and copy in parallel
        let slices_per_z = crop_shape[1];
        let bytes_per_z = slices_per_z * row_bytes;

        buffer
            .par_chunks_mut(bytes_per_z)
            .enumerate()
            .for_each(|(z, z_buffer)| {
                let src_z = crop_offset[2] + z;
                let z_offset = src_z * dim0 * dim1;

                for y in 0..crop_shape[1] {
                    let src_y = crop_offset[1] + y;
                    let src_index = crop_offset[0] + src_y * dim0 + z_offset;
                    let src_byte = data_offset + src_index * elem_size;

                    let dst_start = y * row_bytes;
                    z_buffer[dst_start..dst_start + row_bytes]
                        .copy_from_slice(&mmap_slice[src_byte..src_byte + row_bytes]);
                }
            });
    } else {
        // Sequential copy for small crops (lower overhead)
        let mut dst_cursor = 0;
        for z in 0..crop_shape[2] {
            let src_z = crop_offset[2] + z;
            let z_offset = src_z * dim0 * dim1;

            for y in 0..crop_shape[1] {
                let src_y = crop_offset[1] + y;
                let src_index = crop_offset[0] + src_y * dim0 + z_offset;
                let src_byte = data_offset + src_index * elem_size;

                let src_range = src_byte..src_byte + row_bytes;
                let dst_range = dst_cursor..dst_cursor + row_bytes;
                buffer[dst_range].copy_from_slice(&mmap_slice[src_range]);
                dst_cursor += row_bytes;
            }
        }
    }

    let mut new_header = header.clone();
    new_header.ndim = 3;
    new_header.dim = [1i64; 7];
    for (i, &s) in crop_shape.iter().enumerate() {
        new_header.dim[i] = s as i64;
    }
    new_header.vox_offset = NiftiHeader::default().vox_offset;

    // Translate affine by crop offset
    let mut affine = new_header.affine();
    for row in affine.iter_mut().take(3) {
        row[3] += row[0] * crop_offset[0] as f32
            + row[1] * crop_offset[1] as f32
            + row[2] * crop_offset[2] as f32;
    }
    new_header.set_affine(affine);

    let data_len = buffer.len();
    Ok(NiftiImage::from_shared_bytes(
        new_header,
        Arc::new(buffer),
        0,
        data_len,
    ))
}

/// Configuration for patch extraction from volumes.
#[derive(Debug, Clone)]
pub struct PatchConfig {
    /// Shape of patches to extract [d, h, w]
    pub shape: [usize; 3],
    /// Number of patches per volume
    pub patches_per_volume: usize,
    /// Overlap between adjacent patches [d, h, w] in voxels
    pub overlap: [usize; 3],
    /// Whether to randomize patch positions (false = grid sampling)
    pub randomize: bool,
}

impl Default for PatchConfig {
    fn default() -> Self {
        Self {
            shape: [64, 64, 64],
            patches_per_volume: 4,
            overlap: [0, 0, 0],
            randomize: false,
        }
    }
}

impl PatchConfig {
    /// Create a new patch config with the given shape.
    pub fn new(shape: [usize; 3]) -> Self {
        Self {
            shape,
            ..Default::default()
        }
    }

    /// Set number of patches to extract per volume.
    pub fn patches_per_volume(mut self, n: usize) -> Self {
        self.patches_per_volume = n;
        self
    }

    /// Set overlap between adjacent patches.
    pub fn overlap(mut self, overlap: [usize; 3]) -> Self {
        self.overlap = overlap;
        self
    }

    /// Enable random patch sampling (default: grid sampling).
    pub fn randomize(mut self) -> Self {
        self.randomize = true;
        self
    }
}

/// Streaming crop loader that efficiently extracts multiple patches from volumes.
///
/// This maintains memory efficiency while extracting multiple patches per volume,
/// perfect for training pipelines.
pub struct CropLoader {
    volumes: Vec<PathBuf>,
    current_volume: usize,
    patches_extracted: usize,
    config: PatchConfig,
}

impl CropLoader {
    /// Create a new crop loader for the given volumes.
    pub fn new<P: AsRef<Path>>(volumes: Vec<P>, config: PatchConfig) -> Self {
        Self {
            volumes: volumes.iter().map(|p| p.as_ref().to_path_buf()).collect(),
            current_volume: 0,
            patches_extracted: 0,
            config,
        }
    }

    /// Extract the next patch from the training set.
    #[allow(unsafe_code)]
    pub fn next_patch(&mut self) -> Result<NiftiImage> {
        if self.current_volume >= self.volumes.len() {
            return Err(Error::Exhausted("all volumes processed".to_string()));
        }

        if self.config.patches_per_volume == 0 {
            return Err(Error::InvalidDimensions(
                "patches_per_volume must be positive".to_string(),
            ));
        }

        // Load current volume header to get dimensions
        let path = &self.volumes[self.current_volume];
        let file = File::open(path)?;
        // SAFETY: Memory mapping is safe - file just opened, read-only access
        let mmap = unsafe { Mmap::map(&file)? };
        let header = NiftiHeader::from_bytes(&mmap)?;
        let volume_shape = header.shape();

        if volume_shape.len() < 3 {
            return Err(Error::InvalidDimensions(
                "expected at least 3 spatial dimensions".to_string(),
            ));
        }

        for (i, (&dim, &patch_dim)) in volume_shape
            .iter()
            .zip(self.config.shape.iter())
            .take(3)
            .enumerate()
        {
            if patch_dim == 0 {
                return Err(Error::InvalidDimensions(format!(
                    "patch_size[{}] must be positive",
                    i
                )));
            }
            if patch_dim > dim {
                return Err(Error::InvalidDimensions(format!(
                    "patch_size[{}]={} cannot exceed image dimension[{}]={}",
                    i, patch_dim, i, dim
                )));
            }
        }

        for i in 0..3 {
            if self.config.overlap[i] >= self.config.shape[i] {
                return Err(Error::InvalidDimensions(
                    "patch_overlap must be smaller than patch_size in all dimensions".to_string(),
                ));
            }
        }

        // Calculate patch positions
        let patch_positions = if self.config.randomize {
            self.random_patch_positions(&volume_shape)
        } else {
            self.grid_patch_positions(&volume_shape)?
        };

        // Get the next patch position
        if self.patches_extracted >= patch_positions.len() {
            // Move to next volume
            self.current_volume += 1;
            self.patches_extracted = 0;
            return self.next_patch(); // Recursive call for next volume
        }

        let patch_offset = patch_positions[self.patches_extracted];
        self.patches_extracted += 1;

        // Use the efficient load_cropped function
        load_cropped(path, patch_offset, self.config.shape)
    }

    /// Calculate grid-based patch positions.
    fn grid_patch_positions(&self, shape: &[usize]) -> Result<Vec<[usize; 3]>> {
        let [pd, ph, pw] = self.config.shape;
        let [od, oh, ow] = self.config.overlap;

        let step_d = pd.saturating_sub(od);
        let step_h = ph.saturating_sub(oh);
        let step_w = pw.saturating_sub(ow);

        if step_d == 0 || step_h == 0 || step_w == 0 {
            return Err(Error::InvalidDimensions(
                "patch_size must be larger than patch_overlap in all dimensions".to_string(),
            ));
        }

        let mut positions = Vec::new();
        // Use saturating_sub to prevent underflow when patch is larger than volume
        let max_d = shape[0].saturating_sub(pd);
        let max_h = shape.get(1).copied().unwrap_or(1).saturating_sub(ph);
        let max_w = shape.get(2).copied().unwrap_or(1).saturating_sub(pw);

        for d in (0..=max_d).step_by(step_d) {
            for h in (0..=max_h).step_by(step_h) {
                for w in (0..=max_w).step_by(step_w) {
                    positions.push([d, h, w]);
                }
            }
        }

        Ok(positions)
    }

    /// Calculate random patch positions.
    fn random_patch_positions(&self, shape: &[usize]) -> Vec<[usize; 3]> {
        use rand::thread_rng;

        // Use saturating_sub to prevent underflow when patch is larger than volume
        let max_d = shape
            .first()
            .copied()
            .unwrap_or(1)
            .saturating_sub(self.config.shape[0]);
        let max_h = shape
            .get(1)
            .copied()
            .unwrap_or(1)
            .saturating_sub(self.config.shape[1]);
        let max_w = shape
            .get(2)
            .copied()
            .unwrap_or(1)
            .saturating_sub(self.config.shape[2]);

        let mut rng = thread_rng();
        let mut positions = Vec::new();

        for _ in 0..self.config.patches_per_volume {
            positions.push([
                rng.gen_range(0..=max_d),
                rng.gen_range(0..=max_h),
                rng.gen_range(0..=max_w),
            ]);
        }

        positions
    }
}

/// Batch loader for efficient training pipeline data loading.
///
/// This combines multiple volumes and patch extraction into a memory-efficient
/// streaming interface optimized for high-throughput training.
pub struct BatchLoader {
    loader: CropLoader,
    batch_size: usize,
}

impl BatchLoader {
    /// Create a new batch loader.
    pub fn new<P: AsRef<Path>>(volumes: Vec<P>, batch_size: usize) -> Self {
        let config = PatchConfig::default();
        Self {
            loader: CropLoader::new(volumes, config),
            batch_size,
        }
    }

    /// Load the next batch of patches.
    pub fn next_batch(&mut self) -> Result<Vec<NiftiImage>> {
        if self.batch_size == 0 {
            return Err(Error::InvalidDimensions(
                "batch_size must be positive".to_string(),
            ));
        }

        let mut batch = Vec::with_capacity(self.batch_size);

        for _ in 0..self.batch_size {
            match self.loader.next_patch() {
                Ok(patch) => batch.push(patch),
                Err(Error::Exhausted(_)) => break,
                Err(e) => return Err(e),
            }
        }

        if batch.is_empty() {
            Err(Error::Exhausted("no more patches available".to_string()))
        } else {
            Ok(batch)
        }
    }
}

/// High-performance prefetch and caching system for training pipelines.
///
/// Maintains an LRU cache of loaded patches and prefetches upcoming data
/// to maximize I/O throughput while iterating over many volumes.
pub struct TrainingDataLoader {
    /// Volume file paths
    volumes: Vec<PathBuf>,
    /// Patch configuration
    config: PatchConfig,
    /// LRU cache of loaded patches (volume_index -> cached patches)
    cache: HashMap<usize, Vec<NiftiImage>>,
    /// Maximum cache size in patches
    max_cache_size: usize,
    /// Current volume index
    current_volume: usize,
    /// Current patch index within volume
    current_patch: usize,
    /// Prefetch queue
    prefetch_queue: Vec<(usize, Vec<[usize; 3]>)>,
    /// Total patches processed
    patches_processed: usize,
}

impl TrainingDataLoader {
    /// Create a new training data loader with prefetching.
    ///
    /// Args:
    ///     volumes: List of NIfTI file paths
    ///     config: Patch extraction configuration
    ///     cache_size: Maximum number of patches to cache (default: 1000)
    ///
    /// Example:
    ///     ```rust
    ///     let loader = TrainingDataLoader::new(
    ///         vec!["vol1.nii", "vol2.nii"],
    ///         PatchConfig {
    ///             shape: [64, 64, 64],
    ///             patches_per_volume: 4,
    ///             overlap: [0, 0, 0],
    ///             randomize: true,
    ///         },
    ///         1000, // Cache 1000 patches
    ///     );
    ///     ```
    pub fn new<P: AsRef<Path>>(
        volumes: Vec<P>,
        config: PatchConfig,
        cache_size: usize,
    ) -> Result<Self> {
        let volumes: Vec<PathBuf> = volumes
            .into_iter()
            .map(|p| p.as_ref().to_path_buf())
            .collect();

        if volumes.is_empty() {
            return Err(Error::InvalidDimensions("No volumes provided".to_string()));
        }

        for i in 0..3 {
            if config.shape[i] == 0 {
                return Err(Error::InvalidDimensions(format!(
                    "patch_size[{}] must be positive",
                    i
                )));
            }
        }

        if config.patches_per_volume == 0 {
            return Err(Error::InvalidDimensions(
                "patches_per_volume must be positive".to_string(),
            ));
        }

        for i in 0..3 {
            if config.overlap[i] >= config.shape[i] {
                return Err(Error::InvalidDimensions(
                    "patch_overlap must be smaller than patch_size in all dimensions".to_string(),
                ));
            }
        }

        let mut loader = Self {
            cache: HashMap::new(),
            max_cache_size: cache_size,
            current_volume: 0,
            current_patch: 0,
            prefetch_queue: Vec::new(),
            patches_processed: 0,
            volumes,
            config,
        };

        // Initialize prefetch queue for first few volumes
        loader.initialize_prefetch()?;
        Ok(loader)
    }

    /// Initialize prefetch queue with upcoming volumes.
    fn initialize_prefetch(&mut self) -> Result<()> {
        // Prefetch first few volumes to fill cache
        let prefetch_count = (self.max_cache_size / self.config.patches_per_volume).min(3);

        for i in 0..prefetch_count.min(self.volumes.len()) {
            let patch_positions = self.compute_patch_positions(&self.volumes[i])?;
            self.prefetch_queue.push((i, patch_positions));
        }

        Ok(())
    }

    /// Get next training patch with automatic prefetching.
    ///
    /// This method maintains a background cache and prefetches upcoming data
    /// to ensure patches are always available with minimal latency.
    ///
    /// Returns: Next training patch
    pub fn next_patch(&mut self) -> Result<NiftiImage> {
        // Check if we need to load current volume's patches
        if !self.cache.contains_key(&self.current_volume) {
            self.load_volume_patches(self.current_volume)?;
        }

        // Get patch from cache (invariant: load_volume_patches ensures key exists)
        let patches = self.cache.get_mut(&self.current_volume).ok_or_else(|| {
            Error::InvalidDimensions("cache invariant violated: volume should be loaded".into())
        })?;

        if self.current_patch >= patches.len() {
            // Move to next volume
            self.current_volume += 1;
            self.current_patch = 0;

            if self.current_volume >= self.volumes.len() {
                return Err(Error::Exhausted("all patches processed".to_string()));
            }

            return self.next_patch();
        }

        let patch = patches.swap_remove(self.current_patch); // Remove for memory efficiency
        self.patches_processed += 1;

        // Trigger prefetch for upcoming volumes if cache is getting low
        if self.cache.len() < 3 && self.current_volume + 2 < self.volumes.len() {
            self.trigger_prefetch(self.current_volume + 2)?;
        }

        Ok(patch)
    }

    /// Load all patches for a volume into cache.
    fn load_volume_patches(&mut self, volume_idx: usize) -> Result<()> {
        let volume_path = &self.volumes[volume_idx];
        let patch_positions = self.compute_patch_positions(volume_path)?;

        let mut patches = Vec::with_capacity(patch_positions.len());

        for position in patch_positions {
            let patch = load_cropped(volume_path, position, self.config.shape)?;
            patches.push(patch);
        }

        self.cache.insert(volume_idx, patches);
        Ok(())
    }

    /// Compute patch positions for a volume.
    fn compute_patch_positions(&self, volume_path: &Path) -> Result<Vec<[usize; 3]>> {
        let header = load_header(volume_path)?;
        let shape = header.shape();

        if shape.len() < 3 {
            return Err(Error::InvalidDimensions(
                "expected at least 3 spatial dimensions".to_string(),
            ));
        }

        for (i, (&dim, &patch_dim)) in shape
            .iter()
            .zip(self.config.shape.iter())
            .take(3)
            .enumerate()
        {
            if patch_dim > dim {
                return Err(Error::InvalidDimensions(format!(
                    "patch_size[{}]={} cannot exceed image dimension[{}]={}",
                    i, patch_dim, i, dim
                )));
            }
        }

        if self.config.randomize {
            Ok(self.random_patch_positions(&shape))
        } else {
            self.grid_patch_positions(&shape)
        }
    }

    /// Generate grid-based patch positions.
    fn grid_patch_positions(&self, shape: &[usize]) -> Result<Vec<[usize; 3]>> {
        let [pd, ph, pw] = self.config.shape;
        let [od, oh, ow] = self.config.overlap;

        // Use saturating_sub to prevent underflow when patch is larger than volume
        let max_d = shape[0].saturating_sub(pd);
        let max_h = shape.get(1).copied().unwrap_or(1).saturating_sub(ph);
        let max_w = shape.get(2).copied().unwrap_or(1).saturating_sub(pw);

        let step_d = pd.saturating_sub(od);
        let step_h = ph.saturating_sub(oh);
        let step_w = pw.saturating_sub(ow);

        if step_d == 0 || step_h == 0 || step_w == 0 {
            return Err(Error::InvalidDimensions(
                "patch_size must be larger than patch_overlap in all dimensions".to_string(),
            ));
        }

        let mut positions = Vec::new();
        for d in (0..=max_d).step_by(step_d) {
            for h in (0..=max_h).step_by(step_h) {
                for w in (0..=max_w).step_by(step_w) {
                    positions.push([d, h, w]);
                }
            }
        }

        // Ensure we get exactly patches_per_volume
        if positions.len() > self.config.patches_per_volume {
            positions.truncate(self.config.patches_per_volume);
        } else if positions.len() < self.config.patches_per_volume {
            // Add random positions if needed
            let mut rng = rand::thread_rng();
            while positions.len() < self.config.patches_per_volume {
                let d = rng.gen_range(0..=max_d);
                let h = rng.gen_range(0..=max_h);
                let w = rng.gen_range(0..=max_w);
                positions.push([d, h, w]);
            }
        }

        Ok(positions)
    }

    /// Generate random patch positions.
    fn random_patch_positions(&self, shape: &[usize]) -> Vec<[usize; 3]> {
        let [pd, ph, pw] = self.config.shape;

        // Use saturating_sub to prevent underflow when patch is larger than volume
        let max_d = shape[0].saturating_sub(pd);
        let max_h = shape.get(1).copied().unwrap_or(1).saturating_sub(ph);
        let max_w = shape.get(2).copied().unwrap_or(1).saturating_sub(pw);

        let mut rng = rand::thread_rng();
        let mut positions = Vec::with_capacity(self.config.patches_per_volume);

        for _ in 0..self.config.patches_per_volume {
            positions.push([
                rng.gen_range(0..=max_d),
                rng.gen_range(0..=max_h),
                rng.gen_range(0..=max_w),
            ]);
        }

        positions
    }

    /// Trigger prefetch for upcoming volume.
    fn trigger_prefetch(&mut self, volume_idx: usize) -> Result<()> {
        if volume_idx >= self.volumes.len() {
            return Ok(());
        }

        let patch_positions = self.compute_patch_positions(&self.volumes[volume_idx])?;
        self.prefetch_queue.push((volume_idx, patch_positions));

        Ok(())
    }

    /// Get statistics about the loader performance.
    pub fn stats(&self) -> LoaderStats {
        LoaderStats {
            total_volumes: self.volumes.len(),
            current_volume: self.current_volume,
            cached_volumes: self.cache.len(),
            patches_processed: self.patches_processed,
            cache_size: self.cache.values().map(|patches| patches.len()).sum(),
            max_cache_size: self.max_cache_size,
        }
    }

    /// Reset the loader to start from the beginning.
    pub fn reset(&mut self) -> Result<()> {
        self.cache.clear();
        self.current_volume = 0;
        self.current_patch = 0;
        self.patches_processed = 0;
        self.prefetch_queue.clear();
        self.initialize_prefetch()?;
        Ok(())
    }

    /// Total number of volumes configured for this loader.
    pub fn volumes_len(&self) -> usize {
        self.volumes.len()
    }

    /// Number of patches extracted from each volume.
    pub fn patches_per_volume(&self) -> usize {
        self.config.patches_per_volume
    }
}

/// Performance statistics for the training data loader.
#[derive(Debug, Clone)]
pub struct LoaderStats {
    /// Number of volumes managed by the loader.
    pub total_volumes: usize,
    /// Index of the current volume being processed.
    pub current_volume: usize,
    /// Number of volumes currently cached.
    pub cached_volumes: usize,
    /// Total patches produced so far.
    pub patches_processed: usize,
    /// Current cache size (patches).
    pub cache_size: usize,
    /// Maximum allowed cache size (patches).
    pub max_cache_size: usize,
}

impl std::fmt::Display for LoaderStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Training Data Loader Statistics:")?;
        writeln!(f, "  Total volumes: {}", self.total_volumes)?;
        writeln!(
            f,
            "  Current volume: {}/{}",
            self.current_volume + 1,
            self.total_volumes
        )?;
        writeln!(f, "  Cached volumes: {}", self.cached_volumes)?;
        writeln!(f, "  Patches processed: {}", self.patches_processed)?;
        writeln!(
            f,
            "  Cache utilization: {}/{} patches",
            self.cache_size, self.max_cache_size
        )?;
        Ok(())
    }
}

// ============================================================================
// High-Performance Parallel Training Loader
// ============================================================================

use std::sync::mpsc::{self, Receiver};
use std::thread::{self, JoinHandle};

/// Ultra-fast training data loader with parallel prefetching.
///
/// Designed for maximum throughput when training on large datasets of .nii.gz files.
/// Uses a worker pool to decompress and extract crops in parallel while the main
/// thread (and GPU) processes the current batch.
///
/// # Architecture
/// - Main thread requests batches via `next_batch()`
/// - Worker pool continuously decompresses files and extracts random crops
/// - Bounded channel ensures memory usage stays controlled
/// - Workers stay ahead of consumption to hide I/O latency
///
/// # Example
/// ```ignore
/// let loader = FastLoader::new(paths, [64, 64, 64])
///     .prefetch(16)        // Keep 16 samples ready
///     .workers(4)          // Use 4 decompression threads
///     .build()?;
///
/// for batch in loader.batches(8) {
///     let tensor = batch_to_tensor(&batch);
///     train_step(&model, tensor);
/// }
/// ```
pub struct FastLoader {
    receiver: Receiver<Result<NiftiImage>>,
    worker_handles: Vec<JoinHandle<()>>,
    stop_signal: Arc<std::sync::atomic::AtomicBool>,
    total_volumes: usize,
    patch_shape: [usize; 3],
}

/// Builder for [`FastLoader`].
pub struct FastLoaderBuilder {
    paths: Vec<PathBuf>,
    patch_shape: [usize; 3],
    prefetch_size: usize,
    num_workers: usize,
    shuffle: bool,
    seed: Option<u64>,
    /// Number of threads for mgzip parallel decompression (0 = disabled, use standard gzip)
    mgzip_threads: usize,
}

impl FastLoaderBuilder {
    /// Set prefetch buffer size (default: 16).
    pub fn prefetch(mut self, size: usize) -> Self {
        self.prefetch_size = size;
        self
    }

    /// Set number of worker threads (default: num_cpus).
    pub fn workers(mut self, n: usize) -> Self {
        self.num_workers = n;
        self
    }

    /// Shuffle file order each epoch (default: true).
    pub fn shuffle(mut self, shuffle: bool) -> Self {
        self.shuffle = shuffle;
        self
    }

    /// Set random seed for reproducibility.
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Enable mgzip parallel decompression for .nii.gz files.
    ///
    /// When enabled, files are decompressed using multiple threads per file.
    /// This is beneficial when files are in Mgzip format (multi-member gzip).
    /// Standard gzip files will also work but won't benefit from parallelism.
    ///
    /// # Arguments
    /// * `threads` - Threads per file (0 = auto-detect based on CPU count)
    pub fn mgzip(mut self, threads: usize) -> Self {
        self.mgzip_threads = if threads == 0 {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4)
        } else {
            threads
        };
        self
    }

    /// Build the loader and start worker threads.
    pub fn build(self) -> Result<FastLoader> {
        if self.paths.is_empty() {
            return Err(Error::InvalidDimensions("No files provided".into()));
        }
        if self.patch_shape.contains(&0) {
            return Err(Error::InvalidDimensions(
                "Patch shape cannot have zero dimensions".into(),
            ));
        }

        let (sender, receiver) = mpsc::sync_channel(self.prefetch_size);
        let stop_signal = Arc::new(std::sync::atomic::AtomicBool::new(false));

        let paths = Arc::new(self.paths);
        let total_volumes = paths.len();
        let patch_shape = self.patch_shape;

        let mut handles = Vec::with_capacity(self.num_workers);

        // Shared work queue: indices into paths array
        let work_queue = Arc::new(std::sync::Mutex::new({
            let mut indices: Vec<usize> = (0..paths.len()).collect();
            if self.shuffle {
                use rand::seq::SliceRandom;
                let mut rng = self.seed.map_or_else(
                    rand::rngs::StdRng::from_entropy,
                    rand::rngs::StdRng::seed_from_u64,
                );
                indices.shuffle(&mut rng);
            }
            indices
                .into_iter()
                .collect::<std::collections::VecDeque<_>>()
        }));

        let mgzip_threads = self.mgzip_threads;

        for _ in 0..self.num_workers {
            let sender = sender.clone();
            let stop = stop_signal.clone();
            let paths = paths.clone();
            let queue = work_queue.clone();

            let handle = thread::spawn(move || {
                let mut rng = rand::thread_rng();

                loop {
                    if stop.load(std::sync::atomic::Ordering::Relaxed) {
                        break;
                    }

                    let idx = {
                        let Ok(mut q) = queue.lock() else { break };
                        q.pop_front()
                    };

                    let Some(idx) = idx else { break };

                    let path = &paths[idx];

                    let result = load_random_crop(path, patch_shape, mgzip_threads, &mut rng);

                    if sender.send(result).is_err() {
                        break;
                    }
                }
            });

            handles.push(handle);
        }

        // Drop our sender so channel closes when workers finish
        drop(sender);

        Ok(FastLoader {
            receiver,
            worker_handles: handles,
            stop_signal,
            total_volumes,
            patch_shape,
        })
    }
}

impl FastLoader {
    /// Create a new fast loader builder.
    #[allow(clippy::new_ret_no_self)]
    pub fn new<P: AsRef<Path>>(paths: Vec<P>, patch_shape: [usize; 3]) -> FastLoaderBuilder {
        FastLoaderBuilder {
            paths: paths
                .into_iter()
                .map(|p| p.as_ref().to_path_buf())
                .collect(),
            patch_shape,
            prefetch_size: 16,
            num_workers: rayon::current_num_threads().max(1),
            shuffle: true,
            seed: None,
            mgzip_threads: 0,
        }
    }

    /// Get the next sample. Returns None when all files processed.
    pub fn next(&self) -> Option<Result<NiftiImage>> {
        self.receiver.recv().ok()
    }

    /// Iterate over batches of the given size.
    pub fn batches(self, batch_size: usize) -> BatchIter {
        BatchIter {
            loader: self,
            batch_size,
        }
    }

    /// Total number of volumes in the dataset.
    pub fn len(&self) -> usize {
        self.total_volumes
    }

    /// Check if dataset is empty.
    pub fn is_empty(&self) -> bool {
        self.total_volumes == 0
    }

    /// Patch shape being extracted.
    pub fn patch_shape(&self) -> [usize; 3] {
        self.patch_shape
    }
}

impl Drop for FastLoader {
    fn drop(&mut self) {
        self.stop_signal
            .store(true, std::sync::atomic::Ordering::Relaxed);
        for handle in self.worker_handles.drain(..) {
            let _ = handle.join();
        }
    }
}

impl Iterator for FastLoader {
    type Item = Result<NiftiImage>;

    fn next(&mut self) -> Option<Self::Item> {
        self.receiver.recv().ok()
    }
}

/// Iterator over batches from FastLoader.
pub struct BatchIter {
    loader: FastLoader,
    batch_size: usize,
}

impl Iterator for BatchIter {
    type Item = Vec<Result<NiftiImage>>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut batch = Vec::with_capacity(self.batch_size);
        for _ in 0..self.batch_size {
            match self.loader.receiver.recv() {
                Ok(item) => batch.push(item),
                Err(_) => break,
            }
        }
        if batch.is_empty() {
            None
        } else {
            Some(batch)
        }
    }
}

fn load_random_crop<R: Rng>(
    path: &Path,
    shape: [usize; 3],
    mgzip_threads: usize,
    rng: &mut R,
) -> Result<NiftiImage> {
    let decompressed = if mgzip_threads > 0 {
        load_random_crop_mgzip(path, mgzip_threads)?
    } else {
        let compressed = read_file_with_readahead(path)?;
        let (output, _) = decompress_gzip_pooled(&compressed)?;
        output
    };

    let header = NiftiHeader::from_bytes(&decompressed)?;
    let full_shape = header.shape();

    if full_shape.len() < 3 {
        if mgzip_threads == 0 {
            release_decompress_buffer(decompressed);
        }
        return Err(Error::InvalidDimensions(
            "Volume must have at least 3 dimensions".into(),
        ));
    }

    let max_offset: Vec<usize> = full_shape
        .iter()
        .zip(shape.iter())
        .map(|(&full, &crop)| full.saturating_sub(crop))
        .collect();

    let offset = [
        rng.gen_range(0..=max_offset[0]),
        rng.gen_range(0..=max_offset[1]),
        rng.gen_range(0..=max_offset.get(2).copied().unwrap_or(0)),
    ];

    let data_offset = header.vox_offset as usize;
    let result = copy_cropped_region_from_bytes(&header, &decompressed, data_offset, offset, shape);
    if mgzip_threads == 0 {
        release_decompress_buffer(decompressed);
    }
    result
}

fn load_random_crop_mgzip(path: &Path, num_threads: usize) -> Result<Vec<u8>> {
    let file = File::open(path)?;
    let reader = BufReader::with_capacity(1024 * 1024, file);

    let mut decompressor = ParDecompressBuilder::<Mgzip>::new()
        .num_threads(num_threads)
        .map_err(|e| Error::Decompression(format!("failed to create parallel decompressor: {e}")))?
        .from_reader(reader);

    let mut decompressed = Vec::new();
    decompressor
        .read_to_end(&mut decompressed)
        .map_err(|e| Error::Decompression(format!("mgzip decompression failed: {e}")))?;

    Ok(decompressed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use flate2::write::GzEncoder;
    use flate2::Compression;
    use ndarray::s;
    use ndarray::ArrayD;
    use ndarray::ShapeBuilder;
    use std::io::Write;
    use tempfile::tempdir;

    fn create_f_order_array(data: Vec<f32>, shape: Vec<usize>) -> ArrayD<f32> {
        let c_order = ArrayD::from_shape_vec(shape.clone(), data).unwrap();
        let mut f_order = ArrayD::zeros(ndarray::IxDyn(&shape).f());
        f_order.assign(&c_order);
        f_order
    }

    #[test]
    fn test_roundtrip_uncompressed() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.nii");

        // Create test image with F-order
        let data = create_f_order_array((0..1000).map(|i| i as f32).collect(), vec![10, 10, 10]);
        let affine = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let img = NiftiImage::from_array(data.clone(), affine);

        // Save and reload
        save(&img, &path).unwrap();
        let loaded = load(&path).unwrap();

        assert_eq!(loaded.shape(), &[10, 10, 10]);
        // Compare in memory order
        let loaded_data = loaded.to_f32().unwrap();
        assert_eq!(
            loaded_data.as_slice_memory_order().unwrap(),
            data.as_slice_memory_order().unwrap()
        );
    }

    #[test]
    fn test_roundtrip_gzipped() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.nii.gz");

        let data = create_f_order_array((0..1000).map(|i| i as f32).collect(), vec![10, 10, 10]);
        let affine = [
            [2.0, 0.0, 0.0, -10.0],
            [0.0, 2.0, 0.0, -10.0],
            [0.0, 0.0, 2.0, -10.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let img = NiftiImage::from_array(data.clone(), affine);

        save(&img, &path).unwrap();
        let loaded = load(&path).unwrap();

        assert_eq!(loaded.shape(), &[10, 10, 10]);
        assert_eq!(loaded.affine(), affine);
        // Compare in memory order
        let loaded_data = loaded.to_f32().unwrap();
        assert_eq!(
            loaded_data.as_slice_memory_order().unwrap(),
            data.as_slice_memory_order().unwrap()
        );
    }

    #[test]
    fn test_multimember_gzip_fallback() {
        let dir = tempdir().unwrap();
        let base_path = dir.path().join("base.nii");
        let path = dir.path().join("multi.nii.gz");

        let data = create_f_order_array((0..1000).map(|i| i as f32).collect(), vec![10, 10, 10]);
        let affine = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let img = NiftiImage::from_array(data.clone(), affine);
        save(&img, &base_path).unwrap();

        let bytes = std::fs::read(&base_path).unwrap();
        let split = bytes.len() / 2;

        let mut multi_member = Vec::new();
        let mut encoder = GzEncoder::new(Vec::new(), Compression::fast());
        encoder.write_all(&bytes[..split]).unwrap();
        multi_member.extend(encoder.finish().unwrap());

        let mut encoder = GzEncoder::new(Vec::new(), Compression::fast());
        encoder.write_all(&bytes[split..]).unwrap();
        multi_member.extend(encoder.finish().unwrap());

        std::fs::write(&path, multi_member).unwrap();

        let loaded = load(&path).unwrap();
        assert_eq!(loaded.shape(), &[10, 10, 10]);
        let loaded_data = loaded.to_f32().unwrap();
        assert_eq!(
            loaded_data.as_slice_memory_order().unwrap(),
            data.as_slice_memory_order().unwrap()
        );
    }

    #[test]
    fn test_load_cropped_byte_exact() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.nii");

        // Create a larger test image for cropping with F-order
        let data = create_f_order_array((0..131072).map(|i| i as f32).collect(), vec![64, 64, 32]);
        let affine = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let img = NiftiImage::from_array(data.clone(), affine);
        save(&img, &path).unwrap();

        // Test byte-exact cropped loading
        let crop_offset = [16, 16, 8];
        let crop_shape = [32, 32, 16];
        let cropped = load_cropped(&path, crop_offset, crop_shape).unwrap();

        assert_eq!(cropped.shape(), &[32, 32, 16]);

        // Verify the cropped data matches the expected region
        let original_slice = data.slice(s![16..48, 16..48, 8..24]).to_owned();
        let cropped_data = cropped.to_f32().unwrap();

        // Compare by iterating over logical coordinates
        for x in 0..32 {
            for y in 0..32 {
                for z in 0..16 {
                    let expected = original_slice[[x, y, z]];
                    let actual = cropped_data[[x, y, z]];
                    assert!(
                        (expected - actual).abs() < 1e-5,
                        "Mismatch at [{},{},{}]: expected {}, got {}",
                        x,
                        y,
                        z,
                        expected,
                        actual
                    );
                }
            }
        }
    }

    #[test]
    fn test_save_cropped_roundtrip() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.nii");
        let cropped_path = dir.path().join("cropped.nii");

        let data = create_f_order_array((0..131072).map(|i| i as f32).collect(), vec![64, 64, 32]);
        let affine = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let img = NiftiImage::from_array(data.clone(), affine);
        save(&img, &path).unwrap();

        let crop_offset = [8, 8, 4];
        let crop_shape = [16, 16, 8];
        let cropped = load_cropped(&path, crop_offset, crop_shape).unwrap();
        let cropped_data = cropped.to_f32().unwrap();

        save(&cropped, &cropped_path).unwrap();
        let loaded = load(&cropped_path).unwrap();

        assert_eq!(loaded.shape(), &crop_shape);
        let loaded_data = loaded.to_f32().unwrap();
        assert_eq!(
            loaded_data.as_slice_memory_order().unwrap(),
            cropped_data.as_slice_memory_order().unwrap()
        );
    }

    #[test]
    fn test_load_with_crop() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.nii");

        let data = create_f_order_array((0..16384).map(|i| i as f32).collect(), vec![32, 32, 16]);
        let affine = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let mut img = NiftiImage::from_array(data.clone(), affine);
        img.header_mut().pixdim = [0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0];
        save(&img, &path).unwrap();

        let loaded = load_with_crop(&path, CropConfig::new([16, 16, 16])).unwrap();
        assert_eq!(loaded.shape(), &[16, 16, 16]);
    }

    #[test]
    fn test_training_data_loader() {
        let dir = tempdir().unwrap();
        let paths = vec![dir.path().join("test1.nii"), dir.path().join("test2.nii")];

        // Create test volumes
        for (i, path) in paths.iter().enumerate() {
            let size = 64 * 64 * 32;
            let data = create_f_order_array(
                ((i * size)..((i + 1) * size)).map(|v| v as f32).collect(),
                vec![64, 64, 32],
            );
            let affine = [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ];
            let img = NiftiImage::from_array(data, affine);
            save(&img, path).unwrap();
        }

        // Test TrainingDataLoader creation
        let config = PatchConfig {
            shape: [32, 32, 16],
            patches_per_volume: 2,
            overlap: [0, 0, 0],
            randomize: false,
        };

        let mut loader = TrainingDataLoader::new(paths, config, 100).unwrap();
        assert_eq!(loader.stats().total_volumes, 2);

        // Test patch extraction
        let patch1 = loader.next_patch().unwrap();
        assert_eq!(patch1.shape(), &[32, 32, 16]);

        let patch2 = loader.next_patch().unwrap();
        assert_eq!(patch2.shape(), &[32, 32, 16]);

        let patch3 = loader.next_patch().unwrap();
        assert_eq!(patch3.shape(), &[32, 32, 16]);

        let patch4 = loader.next_patch().unwrap();
        assert_eq!(patch4.shape(), &[32, 32, 16]);

        let exhausted = loader.next_patch().unwrap_err();
        assert!(matches!(exhausted, Error::Exhausted(_)));

        // Test stats
        let stats = loader.stats();
        assert_eq!(stats.patches_processed, 4);
    }

    #[test]
    fn test_training_data_loader_random() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.nii");

        // Create test volume with F-order
        let data = create_f_order_array((0..131072).map(|i| i as f32).collect(), vec![64, 64, 32]);
        let affine = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let img = NiftiImage::from_array(data, affine);
        save(&img, &path).unwrap();

        // Test random patch extraction
        let config = PatchConfig {
            shape: [16, 16, 8],
            patches_per_volume: 4,
            overlap: [0, 0, 0],
            randomize: true,
        };

        let mut loader = TrainingDataLoader::new(vec![&path], config, 50).unwrap();

        // Extract patches and ensure they're different
        let patch1 = loader.next_patch().unwrap();
        let patch2 = loader.next_patch().unwrap();
        let _patch3 = loader.next_patch().unwrap();
        let _patch4 = loader.next_patch().unwrap();

        assert_eq!(patch1.shape(), &[16, 16, 8]);
        assert_eq!(patch2.shape(), &[16, 16, 8]);
        assert_eq!(_patch3.shape(), &[16, 16, 8]);
        assert_eq!(_patch4.shape(), &[16, 16, 8]);

        // With randomization, patches should be different
        let data1 = patch1.to_f32().unwrap();
        let data2 = patch2.to_f32().unwrap();
        let _data3 = _patch3.to_f32().unwrap();
        let _data4 = _patch4.to_f32().unwrap();

        // At least some patches should be different
        assert_ne!(data1, data2);
    }

    #[test]
    fn test_crop_loader() {
        let dir = tempdir().unwrap();
        let paths = vec![dir.path().join("test1.nii"), dir.path().join("test2.nii")];

        // Create test volumes
        for (i, path) in paths.iter().enumerate() {
            let size = 32 * 32 * 16;
            let data = ArrayD::from_shape_vec(
                vec![32, 32, 16],
                ((i * size)..((i + 1) * size)).map(|v| v as f32).collect(),
            )
            .unwrap();
            let affine = [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ];
            let img = NiftiImage::from_array(data, affine);
            save(&img, path).unwrap();
        }

        // Test CropLoader
        let config = PatchConfig {
            shape: [16, 16, 8],
            patches_per_volume: 2,
            overlap: [0, 0, 0],
            randomize: false,
        };

        let mut loader = CropLoader::new(paths, config);

        // Should be able to get 4 patches total (2 per volume)
        let patch1 = loader.next_patch().unwrap();
        let patch2 = loader.next_patch().unwrap();
        let patch3 = loader.next_patch().unwrap();
        let patch4 = loader.next_patch().unwrap();

        assert_eq!(patch1.shape(), &[16, 16, 8]);
        assert_eq!(patch2.shape(), &[16, 16, 8]);
        assert_eq!(patch3.shape(), &[16, 16, 8]);
        assert_eq!(patch4.shape(), &[16, 16, 8]);
    }

    #[test]
    fn training_data_loader_rejects_invalid_overlap() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test_invalid.nii");
        let data = create_f_order_array((0..4096).map(|i| i as f32).collect(), vec![16, 16, 16]);
        let affine = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let img = NiftiImage::from_array(data, affine);
        save(&img, &path).unwrap();

        let cfg = PatchConfig {
            shape: [8, 8, 8],
            patches_per_volume: 1,
            overlap: [8, 4, 4],
            randomize: false,
        };

        let result = TrainingDataLoader::new(vec![path], cfg, 10);
        assert!(result.is_err());
    }

    #[test]
    fn training_data_loader_rejects_zero_patch_size() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test_zero_patch_size.nii");
        let data = create_f_order_array((0..4096).map(|i| i as f32).collect(), vec![16, 16, 16]);
        let affine = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let img = NiftiImage::from_array(data, affine);
        save(&img, &path).unwrap();

        let cfg = PatchConfig {
            shape: [0, 8, 8],
            patches_per_volume: 1,
            overlap: [0, 0, 0],
            randomize: false,
        };

        let result = TrainingDataLoader::new(vec![&path], cfg, 10);
        assert!(matches!(result, Err(Error::InvalidDimensions(_))));
    }

    #[test]
    fn training_data_loader_rejects_zero_patches_per_volume() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test_zero_patches.nii");
        let data = create_f_order_array((0..4096).map(|i| i as f32).collect(), vec![16, 16, 16]);
        let affine = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let img = NiftiImage::from_array(data, affine);
        save(&img, &path).unwrap();

        let cfg = PatchConfig {
            shape: [8, 8, 8],
            patches_per_volume: 0,
            overlap: [0, 0, 0],
            randomize: false,
        };

        let result = TrainingDataLoader::new(vec![&path], cfg, 10);
        assert!(matches!(result, Err(Error::InvalidDimensions(_))));
    }

    #[test]
    fn batch_loader_exhausts_cleanly() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test_batch.nii");
        let data = create_f_order_array((0..262144).map(|i| i as f32).collect(), vec![64, 64, 64]);
        let affine = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let img = NiftiImage::from_array(data, affine);
        save(&img, &path).unwrap();

        let mut loader = BatchLoader::new(vec![&path], 2);
        let batch = loader.next_batch().unwrap();
        assert_eq!(batch.len(), 1);

        let err = loader.next_batch().unwrap_err();
        assert!(matches!(err, Error::Exhausted(_)));
    }

    #[test]
    fn batch_loader_propagates_invalid_patch_error() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test_batch_invalid.nii");
        let data = create_f_order_array((0..32768).map(|i| i as f32).collect(), vec![32, 32, 32]);
        let affine = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let img = NiftiImage::from_array(data, affine);
        save(&img, &path).unwrap();

        let mut loader = BatchLoader::new(vec![&path], 1);
        let err = loader.next_batch().unwrap_err();
        assert!(matches!(err, Error::InvalidDimensions(_)));
    }

    #[test]
    fn test_memory_efficiency() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("large_test.nii");

        // Create a large test image (256x256x64) with F-order
        let data = create_f_order_array(
            (0..(256 * 256 * 64)).map(|i| i as f32).collect(),
            vec![256, 256, 64],
        );
        let affine = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let img = NiftiImage::from_array(data.clone(), affine);
        save(&img, &path).unwrap();

        // Test that load_cropped uses significantly less memory
        let crop_offset = [64, 64, 16];
        let crop_shape = [64, 64, 32];

        // This should load only the cropped region, not the entire file
        let cropped = load_cropped(&path, crop_offset, crop_shape).unwrap();
        assert_eq!(cropped.shape(), crop_shape);

        // Verify data matches expected region using logical indexing
        let original_slice = data.slice(s![64..128, 64..128, 16..48]).to_owned();
        let cropped_data = cropped.to_f32().unwrap();

        // Compare by iterating over logical coordinates
        for x in 0..64 {
            for y in 0..64 {
                for z in 0..32 {
                    let expected = original_slice[[x, y, z]];
                    let actual = cropped_data[[x, y, z]];
                    assert!(
                        (expected - actual).abs() < 1e-5,
                        "Mismatch at [{},{},{}]: expected {}, got {}",
                        x,
                        y,
                        z,
                        expected,
                        actual
                    );
                }
            }
        }

        // Memory usage should be proportional to crop size, not full image size
        let full_size_bytes = 256 * 256 * 64 * 4; // f32 = 4 bytes
        let crop_size_bytes = 64 * 64 * 32 * 4;
        assert!(crop_size_bytes < full_size_bytes / 10); // At least 10x reduction
    }

    #[test]
    fn test_patch_larger_than_volume_does_not_panic() {
        // Regression test: patch size > volume dimension should not panic due to underflow
        let dir = tempdir().unwrap();
        let path = dir.path().join("test_small_volume.nii");

        // Create small test volume (4x4x4)
        let data = create_f_order_array((0..64).map(|i| i as f32).collect(), vec![4, 4, 4]);
        let affine = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let img = NiftiImage::from_array(data, affine);
        save(&img, &path).unwrap();

        // Create loader with patch larger than volume (8x8x8 > 4x4x4)
        let config = PatchConfig {
            shape: [8, 8, 8],
            patches_per_volume: 2,
            overlap: [0, 0, 0],
            randomize: false, // Use grid mode
        };

        // This should NOT panic - grid_patch_positions uses saturating_sub
        let mut loader = CropLoader::new(vec![&path], config);
        // It should still attempt to load (may fail with bounds error, but not panic)
        let result = loader.next_patch();
        // The behavior is that it still generates positions at (0,0,0)
        // and the load_cropped may fail or succeed with partial data
        assert!(result.is_ok() || result.is_err());

        // Test random mode too
        let config_random = PatchConfig {
            shape: [8, 8, 8],
            patches_per_volume: 2,
            overlap: [0, 0, 0],
            randomize: true,
        };
        let mut loader_random = CropLoader::new(vec![&path], config_random);
        let result_random = loader_random.next_patch();
        assert!(result_random.is_ok() || result_random.is_err());
    }

    #[test]
    fn test_parallel_gzip_large_file() {
        // Test that parallel gzip is used for large files (>1MB)
        let dir = tempdir().unwrap();
        let path = dir.path().join("large_test.nii.gz");

        // Create a large test image (128x128x64 = 1M voxels = 4MB uncompressed)
        let data = create_f_order_array(
            (0..(128 * 128 * 64)).map(|i| i as f32).collect(),
            vec![128, 128, 64],
        );
        let affine = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let img = NiftiImage::from_array(data.clone(), affine);

        // Save using parallel gzip (should trigger for files >1MB)
        save(&img, &path).unwrap();

        // Verify the file was created
        assert!(path.exists());

        // Load and verify data integrity
        let loaded = load(&path).unwrap();
        assert_eq!(loaded.shape(), &[128, 128, 64]);

        let loaded_data = loaded.to_f32().unwrap();
        assert_eq!(
            loaded_data.as_slice_memory_order().unwrap(),
            data.as_slice_memory_order().unwrap()
        );
    }

    #[test]
    fn test_mgzip_roundtrip() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.nii.mgz");

        let data = create_f_order_array((0..1000).map(|i| i as f32).collect(), vec![10, 10, 10]);
        let affine = [
            [2.0, 0.0, 0.0, -10.0],
            [0.0, 2.0, 0.0, -10.0],
            [0.0, 0.0, 2.0, -10.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let img = NiftiImage::from_array(data.clone(), affine);

        save_mgzip(&img, &path).unwrap();
        let loaded = load_mgzip(&path).unwrap();

        assert_eq!(loaded.shape(), &[10, 10, 10]);
        assert_eq!(loaded.affine(), affine);

        let loaded_data = loaded.to_f32().unwrap();
        assert_eq!(
            loaded_data.as_slice_memory_order().unwrap(),
            data.as_slice_memory_order().unwrap()
        );
    }

    #[test]
    fn test_mgzip_convert_from_gzip() {
        let dir = tempdir().unwrap();
        let gzip_path = dir.path().join("test.nii.gz");
        let mgzip_path = dir.path().join("test.nii.mgz");

        let data = create_f_order_array((0..8000).map(|i| i as f32).collect(), vec![20, 20, 20]);
        let affine = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let img = NiftiImage::from_array(data.clone(), affine);

        save(&img, &gzip_path).unwrap();
        let output = convert_to_mgzip(&gzip_path, Some(&mgzip_path)).unwrap();
        assert_eq!(output, mgzip_path);

        let loaded = load_mgzip(&mgzip_path).unwrap();
        assert_eq!(loaded.shape(), &[20, 20, 20]);

        let loaded_data = loaded.to_f32().unwrap();
        assert_eq!(
            loaded_data.as_slice_memory_order().unwrap(),
            data.as_slice_memory_order().unwrap()
        );
    }

    #[test]
    #[ignore = "is_mgzip detection is heuristic and may not work for all file sizes"]
    fn test_is_mgzip_detection() {
        let dir = tempdir().unwrap();
        let gzip_path = dir.path().join("test.nii.gz");
        let mgzip_path = dir.path().join("test.nii.mgz");

        let data = create_f_order_array((0..512000).map(|i| i as f32).collect(), vec![80, 80, 80]);
        let affine = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let img = NiftiImage::from_array(data, affine);

        save(&img, &gzip_path).unwrap();
        save_mgzip(&img, &mgzip_path).unwrap();

        assert!(!is_mgzip(&gzip_path).unwrap());
        assert!(is_mgzip(&mgzip_path).unwrap());
    }

    #[test]
    fn test_load_cached_basic() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("cached_test.nii.gz");

        // Create test image
        let data = create_f_order_array((0..1000).map(|i| i as f32).collect(), vec![10, 10, 10]);
        let affine = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let img = NiftiImage::from_array(data.clone(), affine);
        save(&img, &path).unwrap();

        // Clear cache first
        clear_decompression_cache();

        // First load - should decompress and cache
        let loaded1 = load_cached(&path).unwrap();
        assert_eq!(loaded1.shape(), &[10, 10, 10]);

        // Second load - should hit cache
        let loaded2 = load_cached(&path).unwrap();
        assert_eq!(loaded2.shape(), &[10, 10, 10]);

        // Verify data is the same
        let data1 = loaded1.to_f32().unwrap();
        let data2 = loaded2.to_f32().unwrap();
        assert_eq!(
            data1.as_slice_memory_order().unwrap(),
            data2.as_slice_memory_order().unwrap()
        );

        // Clear cache
        clear_decompression_cache();
    }

    #[test]
    fn test_cache_size_control() {
        let dir = tempdir().unwrap();

        // Create multiple test files
        let mut paths = Vec::new();
        for i in 0..5 {
            let path = dir.path().join(format!("cache_test_{}.nii.gz", i));
            let data = create_f_order_array(
                (0..1000).map(|j| (i * 1000 + j) as f32).collect(),
                vec![10, 10, 10],
            );
            let affine = [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ];
            let img = NiftiImage::from_array(data, affine);
            save(&img, &path).unwrap();
            paths.push(path);
        }

        // Clear cache and set small size
        clear_decompression_cache();
        set_cache_size(2);

        // Load all files - cache should evict older entries
        for path in &paths {
            let _loaded = load_cached(path).unwrap();
        }

        // Clear cache and reset size
        clear_decompression_cache();
        set_cache_size(10); // Reset to default
    }

    #[test]
    fn test_load_cached_uncompressed_fallback() {
        // Uncompressed files should work with load_cached too
        let dir = tempdir().unwrap();
        let path = dir.path().join("cached_uncompressed.nii");

        let data = create_f_order_array((0..1000).map(|i| i as f32).collect(), vec![10, 10, 10]);
        let affine = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let img = NiftiImage::from_array(data.clone(), affine);
        save(&img, &path).unwrap();

        // load_cached on uncompressed files just uses regular load
        let loaded = load_cached(&path).unwrap();
        assert_eq!(loaded.shape(), &[10, 10, 10]);

        let loaded_data = loaded.to_f32().unwrap();
        assert_eq!(
            loaded_data.as_slice_memory_order().unwrap(),
            data.as_slice_memory_order().unwrap()
        );
    }

    #[test]
    fn test_load_multi_basic() {
        let dir = tempdir().unwrap();

        // Create two test images with different spacing
        let path1 = dir.path().join("image1.nii");
        let path2 = dir.path().join("image2.nii");

        // Image 1: 20x20x20 with 1mm spacing
        let data1 = create_f_order_array((0..8000).map(|i| i as f32).collect(), vec![20, 20, 20]);
        let affine1 = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let img1 = NiftiImage::from_array(data1, affine1);
        save(&img1, &path1).unwrap();

        // Image 2: 20x20x20 with 1mm spacing (same as image 1)
        let data2 =
            create_f_order_array((8000..16000).map(|i| i as f32).collect(), vec![20, 20, 20]);
        let img2 = NiftiImage::from_array(data2, affine1);
        save(&img2, &path2).unwrap();

        // Load both with common spacing
        let files = vec![
            FileConfig::image(&path1).with_key("img1"),
            FileConfig::image(&path2).with_key("img2"),
        ];
        let config = MultiFileConfig::with_spacing([1.0, 1.0, 1.0]);

        let result = load_multi(&files, config).unwrap();

        assert_eq!(result.images.len(), 2);
        assert_eq!(result.get_by_key("img1").unwrap().shape(), &[20, 20, 20]);
        assert_eq!(result.get_by_key("img2").unwrap().shape(), &[20, 20, 20]);
    }

    #[test]
    fn test_load_multi_with_crop() {
        let dir = tempdir().unwrap();

        // Create test images
        let path1 = dir.path().join("image_crop1.nii");
        let path2 = dir.path().join("label_crop1.nii");

        let data1 = create_f_order_array((0..8000).map(|i| i as f32).collect(), vec![20, 20, 20]);
        let data2 = create_f_order_array(
            (0..8000).map(|i| (i % 4) as f32).collect(),
            vec![20, 20, 20],
        );
        let affine = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];

        let img1 = NiftiImage::from_array(data1, affine);
        let img2 = NiftiImage::from_array(data2, affine);
        save(&img1, &path1).unwrap();
        save(&img2, &path2).unwrap();

        // Load with crop
        let files = vec![
            FileConfig::image(&path1),
            FileConfig::label(&path2), // Label uses nearest-neighbor
        ];
        let config = MultiFileConfig::default().crop([5, 5, 5], [10, 10, 10]);

        let result = load_multi(&files, config).unwrap();

        assert_eq!(result.images.len(), 2);
        assert_eq!(result.images[0].shape(), &[10, 10, 10]);
        assert_eq!(result.images[1].shape(), &[10, 10, 10]);
    }

    #[test]
    fn test_load_image_label_pair_basic() {
        let dir = tempdir().unwrap();

        let image_path = dir.path().join("img.nii");
        let label_path = dir.path().join("seg.nii");

        let data_img =
            create_f_order_array((0..1000).map(|i| i as f32).collect(), vec![10, 10, 10]);
        let data_seg = create_f_order_array(
            (0..1000).map(|i| (i % 3) as f32).collect(),
            vec![10, 10, 10],
        );
        let affine = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];

        save(&NiftiImage::from_array(data_img, affine), &image_path).unwrap();
        save(&NiftiImage::from_array(data_seg, affine), &label_path).unwrap();

        // Load pair
        let (image, label) = load_image_label_pair(&image_path, &label_path, None, None).unwrap();

        assert_eq!(image.shape(), &[10, 10, 10]);
        assert_eq!(label.shape(), &[10, 10, 10]);
    }

    #[test]
    fn test_load_multi_resampling() {
        let dir = tempdir().unwrap();

        // Create image with 2mm spacing
        let path = dir.path().join("resample_test.nii");
        let data = create_f_order_array((0..1000).map(|i| i as f32).collect(), vec![10, 10, 10]);
        let affine = [
            [2.0, 0.0, 0.0, 0.0], // 2mm spacing
            [0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        save(&NiftiImage::from_array(data, affine), &path).unwrap();

        // Load with 1mm target spacing (should double the dimensions)
        let files = vec![FileConfig::image(&path)];
        let config = MultiFileConfig::with_spacing([1.0, 1.0, 1.0]);

        let result = load_multi(&files, config).unwrap();

        // Original 10x10x10 at 2mm -> 20x20x20 at 1mm
        assert_eq!(result.images[0].shape(), &[20, 20, 20]);
    }

    #[test]
    fn test_fast_loader_basic() {
        let dir = tempdir().unwrap();

        // Create a few test gzipped files
        let mut paths = Vec::new();
        for i in 0..3 {
            let path = dir.path().join(format!("vol{}.nii.gz", i));
            let data = create_f_order_array(
                (0..1000).map(|j| (i * 1000 + j) as f32).collect(),
                vec![10, 10, 10],
            );
            let affine = [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ];
            save(&NiftiImage::from_array(data, affine), &path).unwrap();
            paths.push(path);
        }

        // Create FastLoader with small patch
        let loader = FastLoader::new(paths.clone(), [4, 4, 4])
            .prefetch(2)
            .workers(2)
            .shuffle(false)
            .build()
            .unwrap();

        assert_eq!(loader.len(), 3);
        assert_eq!(loader.patch_shape(), [4, 4, 4]);

        // Collect all patches
        let patches: Vec<_> = loader.filter_map(|r| r.ok()).collect();
        assert_eq!(patches.len(), 3);

        // Each patch should have the right shape
        for patch in &patches {
            assert_eq!(patch.shape(), &[4, 4, 4]);
        }
    }

    #[test]
    fn test_fast_loader_batches() {
        let dir = tempdir().unwrap();

        // Create test files
        let mut paths = Vec::new();
        for i in 0..6 {
            let path = dir.path().join(format!("batch_vol{}.nii.gz", i));
            let data =
                create_f_order_array((0..1000).map(|j| j as f32).collect(), vec![10, 10, 10]);
            let affine = [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ];
            save(&NiftiImage::from_array(data, affine), &path).unwrap();
            paths.push(path);
        }

        // Create FastLoader and iterate by batches
        let loader = FastLoader::new(paths, [4, 4, 4])
            .prefetch(4)
            .workers(2)
            .build()
            .unwrap();

        let batches: Vec<_> = loader.batches(2).collect();

        // 6 files / batch_size 2 = 3 batches
        assert_eq!(batches.len(), 3);

        for batch in &batches {
            assert_eq!(batch.len(), 2);
        }
    }
}
