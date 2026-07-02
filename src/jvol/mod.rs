//! `.jvol` volumetric compression (wavelet + Rice coding), via the vendored
//! [jvol-rust](https://github.com/fepegar/jvol-rust) codec.
//!
//! `.jvol` is typically far smaller than gzip for medical volumes, at the cost
//! of a bespoke container. Use [`JvolOptions::lossless`] for exact round trips
//! (required for integer/label data); lossy encoding is reserved for float
//! intensity volumes where the wavelet quantization error is acceptable.
//!
//! See [`crate::nifti::load`] and [`crate::nifti::save`], which dispatch to
//! this module automatically for paths ending in `.jvol`.

mod codec;

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

use bincode::Options;
use ndarray::{Array3, ArrayD, Axis, Ix3, ShapeBuilder, Zip};
use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::nifti::{DataType, NiftiImage};
use codec::{
    decode_array, decode_downsampled_f32, decode_lossy_f32, encode_array, Affine4x4,
    EncodedChannel, EncodedVolume, JvolDtype, JvolMetadata, WaveletType,
};

const IDENTITY_AFFINE_F32: [[f32; 4]; 4] = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
];

/// Encoding options for `.jvol` volumetric compression.
#[derive(Debug, Clone, Copy)]
pub struct JvolOptions {
    /// Quality level (1-100, higher is better) for lossy encoding.
    /// Ignored when `lossless` is true.
    pub quality: u8,
    /// Encode losslessly (exact round trip). `quality` is ignored when true.
    /// Default: true.
    pub lossless: bool,
}

impl Default for JvolOptions {
    fn default() -> Self {
        Self {
            quality: 60,
            lossless: true,
        }
    }
}

impl JvolOptions {
    /// Lossless encoding options (exact round trip). Required for
    /// integer/label volumes.
    pub fn lossless() -> Self {
        Self {
            quality: 0,
            lossless: true,
        }
    }

    /// Lossy encoding options at the given quality (1-100, higher is
    /// better). Only valid for float dtypes.
    pub fn lossy(quality: u8) -> Self {
        Self {
            quality,
            lossless: false,
        }
    }
}

/// On-disk container for a medrs `.jvol` file: the jvol-rust encoded volume
/// plus the medrs-specific metadata needed to restore the exact original
/// `NiftiImage` (dtype and scaling), which jvol-rust's own container does not
/// track. Serialized with bincode and zstd, matching jvol-rust's own
/// container format.
#[derive(Serialize, Deserialize)]
struct MedrsJvolFile {
    encoded: EncodedVolume,
    original_datatype: i16,
    scl_slope: f64,
    scl_inter: f64,
}

/// Maximum decompressed size accepted for a single `.jvol` file's bincode
/// payload. A `.jvol` file is untrusted external input, so its zstd stream
/// must not be trusted to decompress to a bounded size; without this cap, a
/// small hostile file could decompress to an arbitrarily large buffer (a
/// decompression bomb) before its contents are even parsed. 8 GiB is
/// generous relative to any real encoded volume (the entropy-coded subband
/// bytes are always far smaller than the raw voxel data they represent) but
/// still bounds the allocation to a fixed size.
const MAX_JVOL_DECOMPRESSED: u64 = 8 * 1024 * 1024 * 1024;

/// Read, size-bounded zstd-decompress, and bincode-deserialize a `.jvol`
/// file's container. Shared by [`load_as`] and [`load_downsampled_as`] so
/// both paths get the same decompression-bomb and oversized-length-prefix
/// defenses: the zstd output is capped at [`MAX_JVOL_DECOMPRESSED`] bytes via
/// `Read::take`, and bincode itself is configured with a matching byte limit
/// so a corrupt/hostile length prefix inside the payload (e.g. a `Vec` field
/// claiming an enormous element count) is rejected instead of driving an
/// oversized allocation.
fn read_jvol_file(path: &Path) -> Result<MedrsJvolFile> {
    let input = File::open(path)?;
    let decoder = zstd::Decoder::new(BufReader::new(input))?;
    let mut limited = decoder.take(MAX_JVOL_DECOMPRESSED);
    let mut buf = Vec::new();
    limited.read_to_end(&mut buf)?;
    if buf.len() as u64 >= MAX_JVOL_DECOMPRESSED {
        return Err(Error::InvalidFileFormat(format!(
            "jvol file decompresses to at least the {MAX_JVOL_DECOMPRESSED}-byte cap; \
             refusing to load a possible decompression bomb"
        )));
    }

    bincode::DefaultOptions::new()
        .with_fixint_encoding()
        .allow_trailing_bytes()
        .with_limit(MAX_JVOL_DECOMPRESSED)
        .deserialize(&buf)
        .map_err(|e| Error::InvalidFileFormat(format!("jvol deserialization failed: {e}")))
}

fn is_integer_dtype(dtype: DataType) -> bool {
    matches!(
        dtype,
        DataType::UInt8
            | DataType::Int8
            | DataType::Int16
            | DataType::UInt16
            | DataType::Int32
            | DataType::UInt32
            | DataType::Int64
            | DataType::UInt64
    )
}

/// Map a medrs dtype to the closest jvol-rust dtype. jvol-rust only supports
/// six dtypes natively; anything else (i8, u32, i64, u64, f16, bf16) falls
/// back to `JvolDtype::F64`, which stores raw f64 bytes losslessly. The exact
/// original dtype is preserved separately in [`MedrsJvolFile::original_datatype`]
/// and restored on load.
fn jvol_dtype_for(dtype: DataType) -> Option<JvolDtype> {
    match dtype {
        DataType::UInt8 => Some(JvolDtype::U8),
        DataType::UInt16 => Some(JvolDtype::U16),
        DataType::Int16 => Some(JvolDtype::I16),
        DataType::Int32 => Some(JvolDtype::I32),
        DataType::Float32 => Some(JvolDtype::F32),
        DataType::Float64 => Some(JvolDtype::F64),
        _ => None,
    }
}

/// Effective scaling slope per the NIfTI spec: `scl_slope == 0` means
/// "no scaling", i.e. an effective slope of 1.0.
fn effective_slope(slope: f64) -> f64 {
    if slope == 0.0 {
        1.0
    } else {
        slope
    }
}

fn invert_scale(v: f64, slope: f64, inter: f64) -> f64 {
    (v - inter) / slope
}

/// Save a [`NiftiImage`] to a `.jvol` file.
///
/// Supports 3D (single channel) and 4D (channel-last) images. Integer/label
/// dtypes must use [`JvolOptions::lossless`]; lossy encoding of integer data
/// is rejected to avoid silently corrupting label values.
pub fn save<P: AsRef<Path>>(image: &NiftiImage, path: P, options: JvolOptions) -> Result<()> {
    let shape = image.shape().to_vec();
    if !(3..=4).contains(&shape.len()) {
        return Err(Error::InvalidDimensions(format!(
            "jvol::save requires a 3D or 4D image, got {}D (shape {:?})",
            shape.len(),
            shape
        )));
    }

    let quality = if options.lossless { 0 } else { options.quality };
    let dtype = image.dtype();
    if quality != 0 && is_integer_dtype(dtype) {
        return Err(Error::Configuration(format!(
            "jvol lossy encoding (quality={quality}) is not supported for integer dtype {dtype}; \
             use JvolOptions::lossless() for integer/label volumes to avoid corrupting values"
        )));
    }

    let jvol_dtype = jvol_dtype_for(dtype).unwrap_or(JvolDtype::F64);

    let data = image.to_f64()?;
    let channels_data: Vec<Array3<f64>> = if shape.len() == 3 {
        let arr3 = data
            .into_dimensionality::<Ix3>()
            .map_err(|e| Error::InvalidDimensions(e.to_string()))?;
        vec![arr3.as_standard_layout().into_owned()]
    } else {
        let nc = shape[3];
        let mut out = Vec::with_capacity(nc);
        for c in 0..nc {
            let owned = data
                .index_axis(Axis(3), c)
                .to_owned()
                .into_dimensionality::<Ix3>()
                .map_err(|e| Error::InvalidDimensions(e.to_string()))?;
            out.push(owned.as_standard_layout().into_owned());
        }
        out
    };

    let mut encoded_channels = Vec::with_capacity(channels_data.len());
    let mut wavelet = WaveletType::LeGall53;
    let mut levels = 0usize;
    for channel in &channels_data {
        let result = encode_array(&channel.view(), quality, jvol_dtype);
        wavelet = result.wavelet;
        levels = result.levels;
        encoded_channels.push(EncodedChannel {
            subbands: result.subbands,
            intercept: result.intercept,
            slope: result.slope,
            step: result.step,
        });
    }

    let spatial_shape = [shape[0], shape[1], shape[2]];
    let affine: Affine4x4 = image.header().affine_f64();
    let encoded = EncodedVolume {
        metadata: JvolMetadata {
            shape: spatial_shape,
            num_channels: channels_data.len(),
            ijk_to_ras: affine,
            dtype: jvol_dtype,
            wavelet,
            levels,
            quality,
        },
        channels: encoded_channels,
    };

    let file = MedrsJvolFile {
        encoded,
        original_datatype: dtype as i16,
        scl_slope: effective_slope(image.header().scl_slope),
        scl_inter: image.header().scl_inter,
    };

    let serialized = bincode::serialize(&file)
        .map_err(|e| Error::InvalidFileFormat(format!("jvol serialization failed: {e}")))?;

    let out = File::create(path.as_ref())?;
    // Higher zstd level for lossless (more data to compress, ratio matters more).
    let level = if quality == 0 { 12 } else { 6 };
    let mut encoder = zstd::Encoder::new(BufWriter::new(out), level)?;
    encoder.write_all(&serialized)?;
    encoder.finish()?;

    Ok(())
}

/// Load a [`NiftiImage`] from a `.jvol` file, materializing it as its stored
/// original dtype. Equivalent to [`load_as`] with `dtype: None`.
pub fn load<P: AsRef<Path>>(path: P) -> Result<NiftiImage> {
    load_as(path, None)
}

/// Load a [`NiftiImage`] from a `.jvol` file, optionally materializing it as a
/// dtype other than the one it was saved with.
///
/// This is useful for mixed-precision pipelines: a volume can be stored once
/// as a compact lossy `.jvol` and decoded directly to `bfloat16`/`float16` at
/// train time, skipping the intermediate full-precision array entirely.
///
/// `dtype: None` reproduces the stored `original_datatype`, matching [`load`].
/// A float target (`f32`/`f16`/`bf16`/`f64`) is always accepted. Overriding to
/// an integer dtype rounds the decoded value to the nearest representable
/// integer (via the same `invert_scale(..).round()` path used for the stored
/// dtype); this is only meaningful for lossless files or when the caller
/// accepts the wavelet quantization error rounding onto an integer grid.
/// Lossy encoding of integer *source* data is still rejected at [`save`] time,
/// unchanged by this override.
pub fn load_as<P: AsRef<Path>>(path: P, dtype: Option<DataType>) -> Result<NiftiImage> {
    let file = read_jvol_file(path.as_ref())?;

    let original_datatype = DataType::from_code(file.original_datatype)?;
    let target_datatype = dtype.unwrap_or(original_datatype);
    let meta = file.encoded.metadata.clone();

    // Lossy volumes whose target is a non-f64 float type decode entirely in f32:
    // the coefficients are already quantized, so f32 precision is sufficient and
    // it skips the f64 inverse DWT plus the f64 intermediate array. Lossless
    // volumes and f64 targets keep the exact f64 path.
    let mut image = if meta.quality != 0 && lossy_f32_target(target_datatype) {
        let mut channels = Vec::with_capacity(file.encoded.channels.len());
        for ch in &file.encoded.channels {
            channels.push(
                decode_lossy_f32(
                    &ch.subbands,
                    meta.shape,
                    meta.wavelet,
                    meta.levels,
                    ch.step,
                    ch.intercept,
                    ch.slope,
                    meta.dtype,
                )
                .map_err(|e| Error::InvalidFileFormat(e.to_string()))?,
            );
        }
        build_typed_image_f32(channels, target_datatype, file.scl_slope, file.scl_inter)?
    } else {
        let mut channels = Vec::with_capacity(file.encoded.channels.len());
        for ch in &file.encoded.channels {
            channels.push(
                decode_array(
                    &ch.subbands,
                    meta.shape,
                    meta.wavelet,
                    meta.levels,
                    ch.step,
                    ch.intercept,
                    ch.slope,
                    meta.quality,
                    meta.dtype,
                )
                .map_err(|e| Error::InvalidFileFormat(e.to_string()))?,
            );
        }
        build_typed_image(channels, target_datatype, file.scl_slope, file.scl_inter)?
    };
    image.header_mut().set_affine_f64(meta.ijk_to_ras);

    Ok(image)
}

/// Whether a lossy decode target can use the f32 fast path. F64 targets keep
/// the exact f64 path; integer targets never occur (lossy integer encoding is
/// rejected at save time).
fn lossy_f32_target(dtype: DataType) -> bool {
    matches!(
        dtype,
        DataType::Float32 | DataType::Float16 | DataType::BFloat16
    )
}

/// Load a lossy `.jvol` file at `1 / factor` of its stored resolution per axis.
///
/// `factor` must be a power of two no larger than `2^levels`. Only the wavelet
/// subbands coarser than the requested factor are decoded, so a downsampled
/// preview costs a fraction of a full [`load`] (the finest, most numerous
/// detail subbands are never entropy-decoded). The returned image has its
/// spacing scaled by `factor` and its origin shifted by the half-pixel
/// convention so it stays registered with the full-resolution volume.
///
/// Returns an error for lossless files, which store a single block with no
/// multiresolution structure; use [`load`] for those.
pub fn load_downsampled<P: AsRef<Path>>(path: P, factor: usize) -> Result<NiftiImage> {
    load_downsampled_as(path, factor, None)
}

/// [`load_downsampled`] with an optional output dtype override.
///
/// Follows the same semantics as [`load_as`]: `None` reproduces the stored
/// `original_datatype`, a float target is always accepted, and an integer
/// target rounds the decoded value.
pub fn load_downsampled_as<P: AsRef<Path>>(
    path: P,
    factor: usize,
    dtype: Option<DataType>,
) -> Result<NiftiImage> {
    if factor == 0 || (factor & (factor - 1)) != 0 {
        return Err(Error::Configuration(format!(
            "jvol load_downsampled: factor must be a power of two, got {factor}"
        )));
    }
    if factor == 1 {
        return load_as(path, dtype);
    }

    let file = read_jvol_file(path.as_ref())?;

    let original_datatype = DataType::from_code(file.original_datatype)?;
    let target_datatype = dtype.unwrap_or(original_datatype);
    let meta = file.encoded.metadata.clone();

    if meta.quality == 0 {
        return Err(Error::Configuration(
            "jvol load_downsampled requires a lossy .jvol file; lossless files have no \
             multiresolution subbands. Use load() instead."
                .to_string(),
        ));
    }

    let k = factor.trailing_zeros() as usize;
    if k > meta.levels {
        return Err(Error::Configuration(format!(
            "jvol load_downsampled: factor {factor} exceeds available detail levels ({}); \
             maximum factor for this file is {}",
            meta.levels,
            1usize << meta.levels
        )));
    }

    let mut channels = Vec::with_capacity(file.encoded.channels.len());
    for ch in &file.encoded.channels {
        channels.push(
            decode_downsampled_f32(
                &ch.subbands,
                meta.shape,
                meta.wavelet,
                meta.levels,
                ch.step,
                ch.intercept,
                ch.slope,
                factor,
            )
            .map_err(|e| Error::InvalidFileFormat(e.to_string()))?,
        );
    }

    let mut image =
        build_typed_image_f32(channels, target_datatype, file.scl_slope, file.scl_inter)?;
    let down_affine = downsample_affine(meta.ijk_to_ras, factor);
    image.header_mut().set_affine_f64(down_affine);

    Ok(image)
}

/// Adjust a voxel-to-world affine for an isotropic `factor` downsample: scale
/// the direction/spacing columns by `factor` and shift the origin so the new
/// voxel (0,0,0) sits at the centre of the block of `factor` original voxels it
/// summarizes (half-pixel convention).
fn downsample_affine(affine: Affine4x4, factor: usize) -> Affine4x4 {
    let f = factor as f64;
    let shift = (f - 1.0) / 2.0;
    let mut out = affine;
    for r in 0..3 {
        let column_sum = affine[r][0] + affine[r][1] + affine[r][2];
        out[r][3] = affine[r][3] + column_sum * shift;
        for c in 0..3 {
            out[r][c] = affine[r][c] * f;
        }
    }
    out
}

/// Reconstruct a typed [`NiftiImage`] from decoded (scaled, real-world) f64
/// channel data, inverting the stored `scl_slope`/`scl_inter` to recover the
/// raw on-disk values before casting to the original dtype.
fn build_typed_image(
    channels: Vec<Array3<f64>>,
    dtype: DataType,
    slope: f64,
    inter: f64,
) -> Result<NiftiImage> {
    if channels.is_empty() {
        return Err(Error::InvalidDimensions(
            "jvol file has no channels".to_string(),
        ));
    }

    let stack: ArrayD<f64> = if channels.len() == 1 {
        let only = channels
            .into_iter()
            .next()
            .ok_or_else(|| Error::InvalidDimensions("jvol file has no channels".to_string()))?;
        only.into_dyn()
    } else {
        let spatial_shape = channels[0].shape().to_vec();
        let mut full_shape = spatial_shape;
        full_shape.push(channels.len());
        let mut out = ArrayD::<f64>::zeros(ndarray::IxDyn(&full_shape));
        for (c, channel) in channels.into_iter().enumerate() {
            out.index_axis_mut(Axis(3), c).assign(&channel);
        }
        out
    };

    macro_rules! build_typed {
        ($ty:ty, $conv:expr) => {{
            // NIfTI's on-disk convention is F-order (column-major); building
            // directly in that layout means a later `crate::nifti::save` of
            // this image round-trips through mmap'd reload instead of
            // silently transposing the volume (the decoded stack above is
            // C-order, matching the vendored codec's own indexing).
            let mut typed = ArrayD::<$ty>::zeros(stack.raw_dim().f());
            Zip::from(&mut typed)
                .and(&stack)
                .for_each(|t, &s| *t = ($conv)(s));
            let mut img = NiftiImage::from_array(typed, IDENTITY_AFFINE_F32);
            img.header_mut().scl_slope = slope;
            img.header_mut().scl_inter = inter;
            img
        }};
    }

    let image = match dtype {
        DataType::UInt8 => build_typed!(u8, |v: f64| invert_scale(v, slope, inter).round() as u8),
        DataType::Int8 => build_typed!(i8, |v: f64| invert_scale(v, slope, inter).round() as i8),
        DataType::Int16 => {
            build_typed!(i16, |v: f64| invert_scale(v, slope, inter).round() as i16)
        }
        DataType::UInt16 => {
            build_typed!(u16, |v: f64| invert_scale(v, slope, inter).round() as u16)
        }
        DataType::Int32 => {
            build_typed!(i32, |v: f64| invert_scale(v, slope, inter).round() as i32)
        }
        DataType::UInt32 => {
            build_typed!(u32, |v: f64| invert_scale(v, slope, inter).round() as u32)
        }
        DataType::Int64 => {
            build_typed!(i64, |v: f64| invert_scale(v, slope, inter).round() as i64)
        }
        DataType::UInt64 => {
            build_typed!(u64, |v: f64| invert_scale(v, slope, inter).round() as u64)
        }
        DataType::Float32 => build_typed!(f32, |v: f64| invert_scale(v, slope, inter) as f32),
        DataType::Float64 => build_typed!(f64, |v: f64| invert_scale(v, slope, inter)),
        DataType::Float16 => {
            build_typed!(half::f16, |v: f64| half::f16::from_f64(invert_scale(
                v, slope, inter
            )))
        }
        DataType::BFloat16 => {
            build_typed!(half::bf16, |v: f64| half::bf16::from_f64(invert_scale(
                v, slope, inter
            )))
        }
    };

    Ok(image)
}

fn invert_scale_f32(v: f32, slope: f64, inter: f64) -> f32 {
    (v - inter as f32) / slope as f32
}

/// f32 counterpart of [`build_typed_image`]: reconstruct a typed image from
/// f32 channel data (produced by the lossy f32 / downsampled decode paths),
/// casting directly to the target dtype without an f64 intermediate.
fn build_typed_image_f32(
    channels: Vec<Array3<f32>>,
    dtype: DataType,
    slope: f64,
    inter: f64,
) -> Result<NiftiImage> {
    if channels.is_empty() {
        return Err(Error::InvalidDimensions(
            "jvol file has no channels".to_string(),
        ));
    }

    let stack: ArrayD<f32> = if channels.len() == 1 {
        let only = channels
            .into_iter()
            .next()
            .ok_or_else(|| Error::InvalidDimensions("jvol file has no channels".to_string()))?;
        only.into_dyn()
    } else {
        let spatial_shape = channels[0].shape().to_vec();
        let mut full_shape = spatial_shape;
        full_shape.push(channels.len());
        let mut out = ArrayD::<f32>::zeros(ndarray::IxDyn(&full_shape));
        for (c, channel) in channels.into_iter().enumerate() {
            out.index_axis_mut(Axis(3), c).assign(&channel);
        }
        out
    };

    macro_rules! build_typed {
        ($ty:ty, $conv:expr) => {{
            // See the matching comment in `build_typed_image`: construct
            // directly in F-order so this image round-trips through a real
            // `.nii` save/mmap-reload rather than transposing.
            let mut typed = ArrayD::<$ty>::zeros(stack.raw_dim().f());
            Zip::from(&mut typed)
                .and(&stack)
                .for_each(|t, &s| *t = ($conv)(s));
            let mut img = NiftiImage::from_array(typed, IDENTITY_AFFINE_F32);
            img.header_mut().scl_slope = slope;
            img.header_mut().scl_inter = inter;
            img
        }};
    }

    let image = match dtype {
        DataType::UInt8 => {
            build_typed!(u8, |v: f32| invert_scale_f32(v, slope, inter).round() as u8)
        }
        DataType::Int8 => {
            build_typed!(i8, |v: f32| invert_scale_f32(v, slope, inter).round() as i8)
        }
        DataType::Int16 => {
            build_typed!(i16, |v: f32| invert_scale_f32(v, slope, inter).round()
                as i16)
        }
        DataType::UInt16 => {
            build_typed!(u16, |v: f32| invert_scale_f32(v, slope, inter).round()
                as u16)
        }
        DataType::Int32 => {
            build_typed!(i32, |v: f32| invert_scale_f32(v, slope, inter).round()
                as i32)
        }
        DataType::UInt32 => {
            build_typed!(u32, |v: f32| invert_scale_f32(v, slope, inter).round()
                as u32)
        }
        DataType::Int64 => {
            build_typed!(i64, |v: f32| invert_scale_f32(v, slope, inter).round()
                as i64)
        }
        DataType::UInt64 => {
            build_typed!(u64, |v: f32| invert_scale_f32(v, slope, inter).round()
                as u64)
        }
        DataType::Float32 => build_typed!(f32, |v: f32| invert_scale_f32(v, slope, inter)),
        DataType::Float64 => {
            build_typed!(f64, |v: f32| invert_scale(v as f64, slope, inter))
        }
        DataType::Float16 => {
            build_typed!(half::f16, |v: f32| half::f16::from_f32(invert_scale_f32(
                v, slope, inter
            )))
        }
        DataType::BFloat16 => {
            build_typed!(half::bf16, |v: f32| half::bf16::from_f32(invert_scale_f32(
                v, slope, inter
            )))
        }
    };

    Ok(image)
}

// ============================================================================
// Decoded-image cache for repeated-epoch access
// ============================================================================

/// Identity stamp for a cached `.jvol` file, used to invalidate an entry when
/// the underlying file changes on disk. Mirrors the equivalent stamp in
/// `crate::nifti::io`'s gzip decompression cache.
#[derive(Clone, PartialEq, Eq)]
struct JvolFileStamp {
    len: u64,
    mtime: Option<std::time::SystemTime>,
}

fn jvol_file_stamp(path: &Path) -> JvolFileStamp {
    let meta = std::fs::metadata(path).ok();
    JvolFileStamp {
        len: meta.as_ref().map_or(0, std::fs::Metadata::len),
        mtime: meta.and_then(|m| m.modified().ok()),
    }
}

/// Cache key: a canonicalized path plus the requested output dtype. The dtype
/// is part of the key so that loading the same file at two different dtypes
/// caches each decode independently, rather than one silently shadowing the
/// other.
#[derive(Clone, PartialEq, Eq, Hash)]
struct JvolCacheKey {
    path: PathBuf,
    dtype: Option<i16>,
}

struct JvolCacheEntry {
    image: Arc<NiftiImage>,
    stamp: JvolFileStamp,
    last_access: u64,
}

/// LRU-style cache of decoded `.jvol` images, keyed by (path, output dtype).
struct JvolDecodedCache {
    entries: HashMap<JvolCacheKey, JvolCacheEntry>,
    max_entries: usize,
    access_counter: u64,
}

impl JvolDecodedCache {
    fn new(max_entries: usize) -> Self {
        Self {
            entries: HashMap::new(),
            max_entries,
            access_counter: 0,
        }
    }

    fn get(&mut self, key: &JvolCacheKey) -> Option<Arc<NiftiImage>> {
        let current = jvol_file_stamp(&key.path);
        if let Some(entry) = self.entries.get(key) {
            if entry.stamp != current {
                self.entries.remove(key);
                return None;
            }
        }
        if let Some(entry) = self.entries.get_mut(key) {
            self.access_counter += 1;
            entry.last_access = self.access_counter;
            Some(entry.image.clone())
        } else {
            None
        }
    }

    fn insert(&mut self, key: JvolCacheKey, image: Arc<NiftiImage>) {
        // A zero-size cache disables caching entirely.
        if self.max_entries == 0 {
            return;
        }

        if self.entries.len() >= self.max_entries && !self.entries.contains_key(&key) {
            if let Some(oldest_key) = self
                .entries
                .iter()
                .min_by_key(|(_, e)| e.last_access)
                .map(|(k, _)| k.clone())
            {
                self.entries.remove(&oldest_key);
            }
        }

        self.access_counter += 1;
        let stamp = jvol_file_stamp(&key.path);
        self.entries.insert(
            key,
            JvolCacheEntry {
                image,
                stamp,
                last_access: self.access_counter,
            },
        );
    }

    fn clear(&mut self) {
        self.entries.clear();
    }
}

static JVOL_CACHE: std::sync::LazyLock<RwLock<JvolDecodedCache>> =
    std::sync::LazyLock::new(|| RwLock::new(JvolDecodedCache::new(10)));

/// Acquire the jvol decoded-image cache for writing, recovering from a
/// poisoned lock so a panicking worker cannot permanently disable caching.
fn jvol_cache_write() -> std::sync::RwLockWriteGuard<'static, JvolDecodedCache> {
    JVOL_CACHE
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Clear the global jvol decoded-image cache.
///
/// Call this to free memory held by images cached via [`load_cached`].
pub fn clear_jvol_cache() {
    jvol_cache_write().clear();
}

/// Set the maximum size of the jvol decoded-image cache.
///
/// Default is 10 entries. Set to 0 to disable caching.
pub fn set_jvol_cache_size(max_entries: usize) {
    let mut cache = jvol_cache_write();
    cache.max_entries = max_entries;
    while cache.entries.len() > max_entries {
        let Some(oldest_key) = cache
            .entries
            .iter()
            .min_by_key(|(_, e)| e.last_access)
            .map(|(k, _)| k.clone())
        else {
            break;
        };
        cache.entries.remove(&oldest_key);
    }
    drop(cache);
}

/// Load a `.jvol` file, decoding once per (path, dtype) and reusing the
/// decoded result on subsequent calls.
///
/// Similar in spirit to [`crate::nifti::load_cached`] for gzip. Particularly
/// useful in training pipelines that revisit the same volume across epochs.
///
/// The cache key includes the requested output dtype (see [`load_as`]), so
/// loading the same file at two different dtypes decodes and caches each
/// independently. A change to the underlying file (size or modification time)
/// invalidates its entry.
///
/// The decoded volume is stored with shared (`Arc`-backed) byte storage, so a
/// cache hit clones the shared buffer rather than the whole voxel array. The
/// cost avoided on a hit is both the wavelet/entropy decode and the array copy.
pub fn load_cached<P: AsRef<Path>>(path: P, dtype: Option<DataType>) -> Result<NiftiImage> {
    let path_ref = path.as_ref();
    let canonical = path_ref
        .canonicalize()
        .unwrap_or_else(|_| path_ref.to_path_buf());
    let key = JvolCacheKey {
        path: canonical,
        dtype: dtype.map(|d| d as i16),
    };

    {
        let mut cache = jvol_cache_write();
        if let Some(image) = cache.get(&key) {
            return Ok((*image).clone());
        }
    }

    let image = load_as(path_ref, dtype)?;
    let shared = to_shared_image(&image)?;
    jvol_cache_write().insert(key, Arc::new(shared.clone()));
    Ok(shared)
}

/// Re-back a decoded image with shared (`Arc`) byte storage so later clones
/// share the buffer instead of copying the voxel array. The bytes come from
/// [`NiftiImage::data_to_bytes`], which emits the array in its native (little
/// endian on every supported target) F-order layout, exactly what
/// [`NiftiImage::from_shared_bytes`] materializes.
fn to_shared_image(image: &NiftiImage) -> Result<NiftiImage> {
    let bytes = image.data_to_bytes()?;
    let len = bytes.len();
    let mut header = image.header().clone();
    header.little_endian = true;
    Ok(NiftiImage::from_shared_bytes(
        header,
        Arc::new(bytes),
        0,
        len,
    ))
}

// ============================================================================
// Decode-once-to-mmap transcoding
// ============================================================================

/// Decode a `.jvol` file and write it out as an uncompressed `.nii`.
///
/// A later [`crate::nifti::load`] of `out_path` memory-maps it with zero-copy
/// access instead of paying the wavelet/entropy decode cost on every load.
///
/// `out_path` must not end in `.gz` or `.jvol`: those formats defeat the
/// zero-copy mmap this function exists to enable.
pub fn transcode_to_nii<P: AsRef<Path>, Q: AsRef<Path>>(
    jvol_path: P,
    out_path: Q,
    dtype: Option<DataType>,
) -> Result<()> {
    let out = out_path.as_ref();
    if out.extension().is_some_and(|e| e == "gz" || e == "jvol") {
        return Err(Error::Configuration(format!(
            "jvol::transcode_to_nii requires an uncompressed .nii output path for zero-copy \
             mmap loading, got {}",
            out.display()
        )));
    }

    let image = load_as(jvol_path, dtype)?;
    crate::nifti::save(&image, out)
}

/// Load a `.jvol` file via a decode-once mmap cache.
///
/// Transcodes to `<cache_dir>/<stem>.nii` (or `<cache_dir>/<stem>__<dtype>.nii`
/// for a dtype override) on first use, then memory-maps that file on this and
/// subsequent calls. Skips the transcode step when a cached `.nii` already
/// exists and is newer than the source `.jvol`.
///
/// This trades one wavelet/entropy decode (on the first call) plus disk space
/// for the cached `.nii` for true zero-copy mmap access on every subsequent
/// call, which repeated-epoch training loops can amortize.
pub fn load_via_mmap_cache<P: AsRef<Path>, Q: AsRef<Path>>(
    jvol_path: P,
    cache_dir: Q,
    dtype: Option<DataType>,
) -> Result<NiftiImage> {
    let jvol_path = jvol_path.as_ref();
    let stem = jvol_path
        .file_stem()
        .and_then(|s| s.to_str())
        .ok_or_else(|| {
            Error::InvalidFileFormat(format!("invalid jvol path: {}", jvol_path.display()))
        })?;

    std::fs::create_dir_all(cache_dir.as_ref())?;
    let suffix = dtype.map_or_else(String::new, |d| format!("__{}", d.type_name()));
    let cached_nii = cache_dir.as_ref().join(format!("{stem}{suffix}.nii"));

    let is_stale = match (
        std::fs::metadata(&cached_nii).and_then(|m| m.modified()),
        std::fs::metadata(jvol_path).and_then(|m| m.modified()),
    ) {
        (Ok(cached_mtime), Ok(src_mtime)) => cached_mtime < src_mtime,
        _ => true,
    };

    if is_stale {
        transcode_to_nii(jvol_path, &cached_nii, dtype)?;
    }

    crate::nifti::load(&cached_nii)
}
