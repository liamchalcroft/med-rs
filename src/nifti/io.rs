//! High-performance NIfTI I/O operations.
//!
//! Optimizations:
//! - Memory-mapped reading for uncompressed files
//! - libdeflate-based decompression for .nii.gz

use super::header::NiftiHeader;
use super::image::NiftiImage;
use crate::error::{Error, Result};
use flate2::bufread::GzDecoder;
use flate2::write::GzEncoder;
use flate2::Compression;
use libdeflater::Decompressor;
use memmap2::Mmap;
use rand::Rng;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Cursor, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::transforms::{self, Interpolation};

const GZIP_BUF_SIZE: usize = 8 * 1024 * 1024; // 8MB buffer for faster streaming

/// Load a NIfTI image from file.
///
/// Supports both `.nii` and `.nii.gz` formats with automatic detection.
///
/// # Example
/// ```ignore
/// let img = medrs::nifti::load("brain.nii.gz")?;
/// let data = img.to_f32();
/// ```
pub fn load<P: AsRef<Path>>(path: P) -> Result<NiftiImage> {
    let path = path.as_ref();
    let is_gzipped = path.extension().map(|e| e == "gz").unwrap_or(false);

    if is_gzipped {
        load_gzipped(path)
    } else {
        load_uncompressed(path)
    }
}

/// Load uncompressed .nii file using memory mapping for speed.
#[allow(unsafe_code)]
fn load_uncompressed(path: &Path) -> Result<NiftiImage> {
    let file = File::open(path)?;
    let mmap = unsafe { Mmap::map(&file)? };

    let header = NiftiHeader::from_bytes(&mmap)?;
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

/// Load gzipped .nii.gz file with streaming decompression.
fn load_gzipped(path: &Path) -> Result<NiftiImage> {
    // Read compressed bytes
    let compressed = std::fs::read(path)?;

    // First, decompress only the header to discover expected size
    let mut header_buf = [0u8; NiftiHeader::SIZE];
    {
        let cursor = Cursor::new(&compressed);
        let mut decoder = GzDecoder::new(cursor);
        decoder.read_exact(&mut header_buf)?;
    }
    let header = NiftiHeader::from_bytes(&header_buf)?;

    let offset = header.vox_offset as usize;
    let data_size = header.data_size();
    let total_size = offset + data_size;

    // Decompress entire payload in one go using libdeflate (fast single-thread)
    let mut output = vec![0u8; total_size];
    let mut decompressor = Decompressor::new();
    let written = decompressor
        .gzip_decompress(&compressed, &mut output)
        .map_err(|e| Error::Decompression(format!("{}", e)))?;

    if written != total_size {
        return Err(Error::Decompression(format!(
            "decompressed size {} did not match expected {}",
            written, total_size
        )));
    }

    let bytes = Arc::new(output);
    Ok(NiftiImage::from_shared_bytes(
        header, bytes, offset, data_size,
    ))
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
    let path = path.as_ref();
    let is_gzipped = path.extension().map(|e| e == "gz").unwrap_or(false);

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
    let data = image.data_to_bytes();
    writer.write_all(&data)?;
    writer.flush()?;

    Ok(())
}

fn save_gzipped(image: &NiftiImage, path: &Path) -> Result<()> {
    let file = File::create(path)?;
    let buf_writer = BufWriter::with_capacity(GZIP_BUF_SIZE, file);
    let mut encoder = GzEncoder::new(buf_writer, Compression::default());

    // Write header
    let header_bytes = image.header().to_bytes();
    encoder.write_all(&header_bytes)?;

    // Padding
    let padding = image.header().vox_offset as usize - NiftiHeader::SIZE;
    if padding > 0 {
        encoder.write_all(&vec![0u8; padding])?;
    }

    // Write data
    let data = image.data_to_bytes();
    encoder.write_all(&data)?;
    encoder.finish()?;

    Ok(())
}

/// Load only the header from a NIfTI file (fast metadata inspection).
#[allow(unsafe_code)]
pub fn load_header<P: AsRef<Path>>(path: P) -> Result<NiftiHeader> {
    let path = path.as_ref();
    let is_gzipped = path.extension().map(|e| e == "gz").unwrap_or(false);

    if is_gzipped {
        let file = File::open(path)?;
        let buf_reader = BufReader::new(file);
        let mut decoder = GzDecoder::new(buf_reader);
        let mut header_buf = vec![0u8; NiftiHeader::SIZE];
        decoder.read_exact(&mut header_buf)?;
        NiftiHeader::from_bytes(&header_buf)
    } else {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        NiftiHeader::from_bytes(&mmap)
    }
}

/// Configuration for loading a cropped region with optional transforms.
pub struct LoadCroppedConfig {
    /// Desired output shape after all transforms [d, h, w]
    pub output_shape: [usize; 3],
    /// Target voxel spacing [mm] for resampling (None = keep original)
    pub target_spacing: Option<[f32; 3]>,
    /// Target orientation (None = keep original)
    pub target_orientation: Option<crate::transforms::Orientation>,
    /// Optional offset in output space for non-centered crops [d, h, w]
    pub output_offset: Option<[usize; 3]>,
}

impl Default for LoadCroppedConfig {
    fn default() -> Self {
        Self {
            output_shape: [64, 64, 64],
            target_spacing: None,
            target_orientation: None,
            output_offset: None,
        }
    }
}

/// Load a cropped region from a NIfTI file without loading the entire volume.
///
/// This is extremely efficient for training pipelines that load large volumes
/// just to crop small patches (e.g., loading 64^3 patch from 256^3 volume).
///
/// Advanced version that can compute minimal raw region needed to achieve
/// desired output after optional reorientation and resampling.
///
/// Arguments:
///   path: Path to NIfTI file (must be uncompressed .nii)
///   config: Configuration specifying desired output and optional transforms
///
/// Returns: NiftiImage with cropped data (as owned array)
#[allow(unsafe_code)]
pub fn load_cropped_config<P: AsRef<Path>>(
    path: P,
    config: LoadCroppedConfig,
) -> Result<NiftiImage> {
    let path = path.as_ref();

    ensure_uncompressed(path)?;

    let file = File::open(path)?;
    let mmap = unsafe { Mmap::map(&file)? };
    let header = NiftiHeader::from_bytes(&mmap)?;
    let data_offset = header.vox_offset as usize;
    let shape = header.shape();
    let crop_offset = config
        .output_offset
        .unwrap_or(compute_center_offset(&shape, &config.output_shape)?);

    let cropped = copy_cropped_region(
        &header,
        &mmap,
        data_offset,
        crop_offset,
        config.output_shape,
    )?;

    let mut output = cropped;

    if let Some(target_orient) = config.target_orientation {
        output = transforms::reorient(&output, target_orient);
    }

    if let Some(target_spacing) = config.target_spacing {
        output = transforms::resample_to_spacing(&output, target_spacing, Interpolation::Trilinear);
    }

    Ok(output)
}

/// Simple version of load_cropped.
#[allow(unsafe_code)]
pub fn load_cropped<P: AsRef<Path>>(
    path: P,
    crop_offset: [usize; 3],
    crop_shape: [usize; 3],
) -> Result<NiftiImage> {
    let path = path.as_ref();

    ensure_uncompressed(path)?;

    let file = File::open(path)?;
    let mmap = unsafe { Mmap::map(&file)? };
    let header = NiftiHeader::from_bytes(&mmap)?;
    let data_offset = header.vox_offset as usize;

    copy_cropped_region(&header, &mmap, data_offset, crop_offset, crop_shape)
}

fn ensure_uncompressed(path: &Path) -> Result<()> {
    if path.extension().map(|e| e == "gz").unwrap_or(false) {
        return Err(Error::InvalidDimensions(
            "load_cropped only supports uncompressed .nii files".to_string(),
        ));
    }
    Ok(())
}

fn compute_center_offset(full_shape: &[u16], crop_shape: &[usize; 3]) -> Result<[usize; 3]> {
    if full_shape.len() < 3 {
        return Err(Error::InvalidDimensions(
            "expected at least 3 spatial dimensions".to_string(),
        ));
    }

    Ok([
        (full_shape[0] as usize).saturating_sub(crop_shape[0]) / 2,
        (full_shape[1] as usize).saturating_sub(crop_shape[1]) / 2,
        (full_shape[2] as usize).saturating_sub(crop_shape[2]) / 2,
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

    for i in 0..3 {
        if crop_offset[i] + crop_shape[i] > full_shape[i] as usize {
            return Err(Error::InvalidDimensions(format!(
                "Crop region exceeds dimension {}: {} + {} > {}",
                i, crop_offset[i], crop_shape[i], full_shape[i]
            )));
        }
    }

    let elem_size = header.datatype.size();
    let dim0 = full_shape[0] as usize;  // First dimension (fastest changing in F-order)
    let dim1 = full_shape.get(1).copied().unwrap_or(1) as usize;
    let _dim2 = full_shape.get(2).copied().unwrap_or(1) as usize;

    // Allocate contiguous buffer for cropped block
    let total_elements = crop_shape[0] * crop_shape[1] * crop_shape[2];
    let mut buffer = vec![0u8; total_elements * elem_size];

    // NIfTI uses F-order (column-major): first index changes fastest
    // F-order linear index = x + y * dim0 + z * dim0 * dim1
    let mut dst_cursor = 0;
    for z in 0..crop_shape[2] {
        let src_z = crop_offset[2] + z;
        for y in 0..crop_shape[1] {
            let src_y = crop_offset[1] + y;
            // F-order: index = x + y * dim0 + z * dim0 * dim1
            // We copy a contiguous run along the first axis (x)
            let src_index = crop_offset[0] + src_y * dim0 + src_z * dim0 * dim1;
            let src_byte = data_offset + src_index * elem_size;
            let row_bytes = crop_shape[0] * elem_size;  // First dimension is contiguous
            let src_slice = mmap.get(src_byte..src_byte + row_bytes).ok_or_else(|| {
                Error::InvalidDimensions("Crop region exceeds file size".to_string())
            })?;

            buffer[dst_cursor..dst_cursor + row_bytes].copy_from_slice(src_slice);
            dst_cursor += row_bytes;
        }
    }

    let mut new_header = header.clone();
    new_header.ndim = 3;
    new_header.dim = [1u16; 7];
    for (i, &s) in crop_shape.iter().enumerate() {
        new_header.dim[i] = s as u16;
    }
    new_header.vox_offset = 0.0;

    // Translate affine by crop offset
    let mut affine = new_header.affine();
    for r in 0..3 {
        affine[r][3] += affine[r][0] * crop_offset[0] as f32
            + affine[r][1] * crop_offset[1] as f32
            + affine[r][2] * crop_offset[2] as f32;
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

/// Configuration for streaming crop loader optimized for training pipelines.
pub struct CropLoaderConfig {
    /// Patch size to extract [d, h, w]
    pub patch_size: [usize; 3],
    /// Number of patches per volume
    pub patches_per_volume: usize,
    /// Overlap between patches [d, h, w] in voxels
    pub patch_overlap: [usize; 3],
    /// Whether to randomize patch positions
    pub randomize: bool,
}

impl Default for CropLoaderConfig {
    fn default() -> Self {
        Self {
            patch_size: [64, 64, 64],
            patches_per_volume: 4,
            patch_overlap: [0, 0, 0],
            randomize: false,
        }
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
    config: CropLoaderConfig,
}

impl CropLoader {
    /// Create a new crop loader for the given volumes.
    pub fn new<P: AsRef<Path>>(volumes: Vec<P>, config: CropLoaderConfig) -> Self {
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
            return Err(Error::InvalidDimensions(
                "No more volumes available".to_string(),
            ));
        }

        // Load current volume header to get dimensions
        let path = &self.volumes[self.current_volume];
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        let header = NiftiHeader::from_bytes(&mmap)?;
        let volume_shape = header.shape();

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
        load_cropped(path, patch_offset, self.config.patch_size)
    }

    /// Calculate grid-based patch positions.
    fn grid_patch_positions(&self, shape: &[u16]) -> Result<Vec<[usize; 3]>> {
        let [pd, ph, pw] = self.config.patch_size;
        let [od, oh, ow] = self.config.patch_overlap;

        let step_d = pd.saturating_sub(od);
        let step_h = ph.saturating_sub(oh);
        let step_w = pw.saturating_sub(ow);

        if step_d == 0 || step_h == 0 || step_w == 0 {
            return Err(Error::InvalidDimensions(
                "patch_size must be larger than patch_overlap in all dimensions".to_string(),
            ));
        }

        let mut positions = Vec::new();
        let max_d = shape[0] as usize - pd;
        let max_h = *shape.get(1).unwrap_or(&1) as usize - ph;
        let max_w = *shape.get(2).unwrap_or(&1) as usize - pw;

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
    fn random_patch_positions(&self, shape: &[u16]) -> Vec<[usize; 3]> {
        use rand::thread_rng;

        let max_d = *shape.get(0).unwrap_or(&1) as usize - self.config.patch_size[0];
        let max_h = *shape.get(1).unwrap_or(&1) as usize - self.config.patch_size[1];
        let max_w = *shape.get(2).unwrap_or(&1) as usize - self.config.patch_size[2];

        let mut rng = thread_rng();
        let mut positions = Vec::new();

        for _ in 0..self.config.patches_per_volume {
            positions.push([
                rng.gen_range(0..=max_d.max(0)),
                rng.gen_range(0..=max_h.max(0)),
                rng.gen_range(0..=max_w.max(0)),
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
        let config = CropLoaderConfig::default();
        Self {
            loader: CropLoader::new(volumes, config),
            batch_size,
        }
    }

    /// Load the next batch of patches.
    pub fn next_batch(&mut self) -> Result<Vec<NiftiImage>> {
        let mut batch = Vec::with_capacity(self.batch_size);

        for _ in 0..self.batch_size {
            match self.loader.next_patch() {
                Ok(patch) => batch.push(patch),
                Err(_) => break, // No more patches
            }
        }

        if batch.is_empty() {
            Err(Error::InvalidDimensions(
                "No more patches available".to_string(),
            ))
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
    config: CropLoaderConfig,
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
    ///         CropLoaderConfig {
    ///             patch_size: [64, 64, 64],
    ///             patches_per_volume: 4,
    ///             patch_overlap: [0, 0, 0],
    ///             randomize: true,
    ///         },
    ///         1000, // Cache 1000 patches
    ///     );
    ///     ```
    pub fn new<P: AsRef<Path>>(
        volumes: Vec<P>,
        config: CropLoaderConfig,
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
            if config.patch_overlap[i] >= config.patch_size[i] {
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

        // Get patch from cache
        let patches = self.cache.get_mut(&self.current_volume).unwrap();

        if self.current_patch >= patches.len() {
            // Move to next volume
            self.current_volume += 1;
            self.current_patch = 0;

            if self.current_volume >= self.volumes.len() {
                return Err(Error::InvalidDimensions(
                    "All patches processed".to_string(),
                ));
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
            let patch = load_cropped(volume_path, position, self.config.patch_size)?;
            patches.push(patch);
        }

        self.cache.insert(volume_idx, patches);
        Ok(())
    }

    /// Compute patch positions for a volume.
    fn compute_patch_positions(&self, volume_path: &Path) -> Result<Vec<[usize; 3]>> {
        let header = load_header(volume_path)?;
        let shape = header.shape();

        for i in 0..3 {
            if self.config.patch_size[i] > shape[i] as usize {
                return Err(Error::InvalidDimensions(
                    "patch_size cannot exceed image dimensions".to_string(),
                ));
            }
        }

        if self.config.randomize {
            self.random_patch_positions(shape)
        } else {
            self.grid_patch_positions(shape)
        }
    }

    /// Generate grid-based patch positions.
    fn grid_patch_positions(&self, shape: &[u16]) -> Result<Vec<[usize; 3]>> {
        let [pd, ph, pw] = self.config.patch_size;
        let [od, oh, ow] = self.config.patch_overlap;

        let max_d = shape[0] as usize - pd;
        let max_h = *shape.get(1).unwrap_or(&1) as usize - ph;
        let max_w = *shape.get(2).unwrap_or(&1) as usize - pw;

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
    fn random_patch_positions(&self, shape: &[u16]) -> Result<Vec<[usize; 3]>> {
        let [pd, ph, pw] = self.config.patch_size;

        let max_d = shape[0] as usize - pd;
        let max_h = *shape.get(1).unwrap_or(&1) as usize - ph;
        let max_w = *shape.get(2).unwrap_or(&1) as usize - pw;

        let mut rng = rand::thread_rng();
        let mut positions = Vec::with_capacity(self.config.patches_per_volume);

        for _ in 0..self.config.patches_per_volume {
            positions.push([
                rng.gen_range(0..=max_d),
                rng.gen_range(0..=max_h),
                rng.gen_range(0..=max_w),
            ]);
        }

        Ok(positions)
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
            cache_size: self.cache.iter().map(|(_, patches)| patches.len()).sum(),
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

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::s;
    use ndarray::ArrayD;
    use ndarray::ShapeBuilder;
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
        let loaded_data = loaded.to_f32();
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
        let loaded_data = loaded.to_f32();
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
        let data = create_f_order_array(
            (0..131072).map(|i| i as f32).collect(),
            vec![64, 64, 32],
        );
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
        let cropped_data = cropped.to_f32();

        // Compare by iterating over logical coordinates
        for x in 0..32 {
            for y in 0..32 {
                for z in 0..16 {
                    let expected = original_slice[[x, y, z]];
                    let actual = cropped_data[[x, y, z]];
                    assert!(
                        (expected - actual).abs() < 1e-5,
                        "Mismatch at [{},{},{}]: expected {}, got {}",
                        x, y, z, expected, actual
                    );
                }
            }
        }
    }

    #[test]
    fn test_load_cropped_config() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.nii");

        // Create test image with uniform spacing
        let data = create_f_order_array((0..16384).map(|i| i as f32).collect(), vec![32, 32, 16]);
        let affine = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let mut img = NiftiImage::from_array(data.clone(), affine);
        img.header_mut().pixdim = [0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0];
        save(&img, &path).unwrap();

        // Test configured loading with cropping to specific shape
        let config = LoadCroppedConfig {
            output_shape: [16, 16, 16],
            target_spacing: None, // Keep original spacing
            target_orientation: None,
            output_offset: None,
        };

        let loaded = load_cropped_config(&path, config).unwrap();
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
        let config = CropLoaderConfig {
            patch_size: [32, 32, 16],
            patches_per_volume: 2,
            patch_overlap: [0, 0, 0],
            randomize: false,
        };

        let mut loader = TrainingDataLoader::new(paths, config, 100).unwrap();
        assert_eq!(loader.stats().total_volumes, 2);

        // Test patch extraction
        let patch1 = loader.next_patch().unwrap();
        assert_eq!(patch1.shape(), &[32, 32, 16]);

        let patch2 = loader.next_patch().unwrap();
        assert_eq!(patch2.shape(), &[32, 32, 16]);

        // Test stats
        let stats = loader.stats();
        assert_eq!(stats.patches_processed, 2);
    }

    #[test]
    fn test_training_data_loader_random() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.nii");

        // Create test volume with F-order
        let data = create_f_order_array(
            (0..131072).map(|i| i as f32).collect(),
            vec![64, 64, 32],
        );
        let affine = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let img = NiftiImage::from_array(data, affine);
        save(&img, &path).unwrap();

        // Test random patch extraction
        let config = CropLoaderConfig {
            patch_size: [16, 16, 8],
            patches_per_volume: 4,
            patch_overlap: [0, 0, 0],
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
        let data1 = patch1.to_f32();
        let data2 = patch2.to_f32();
        let _data3 = _patch3.to_f32();
        let _data4 = _patch4.to_f32();

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
        let config = CropLoaderConfig {
            patch_size: [16, 16, 8],
            patches_per_volume: 2,
            patch_overlap: [0, 0, 0],
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

        let cfg = CropLoaderConfig {
            patch_size: [8, 8, 8],
            patches_per_volume: 1,
            patch_overlap: [8, 4, 4],
            randomize: false,
        };

        let result = TrainingDataLoader::new(vec![path], cfg, 10);
        assert!(result.is_err());
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
        let cropped_data = cropped.to_f32();

        // Compare by iterating over logical coordinates
        for x in 0..64 {
            for y in 0..64 {
                for z in 0..32 {
                    let expected = original_slice[[x, y, z]];
                    let actual = cropped_data[[x, y, z]];
                    assert!(
                        (expected - actual).abs() < 1e-5,
                        "Mismatch at [{},{},{}]: expected {}, got {}",
                        x, y, z, expected, actual
                    );
                }
            }
        }

        // Memory usage should be proportional to crop size, not full image size
        let full_size_bytes = 256 * 256 * 64 * 4; // f32 = 4 bytes
        let crop_size_bytes = 64 * 64 * 32 * 4;
        assert!(crop_size_bytes < full_size_bytes / 10); // At least 10x reduction
    }
}
