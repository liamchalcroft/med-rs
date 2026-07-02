use serde::{Deserialize, Serialize};

pub use super::wavelet::WaveletType;

/// Affine transformation matrix (4x4) mapping voxel indices to RAS+ coordinates.
pub type Affine4x4 = [[f64; 4]; 4];

/// Metadata stored alongside the compressed data in a .jvol file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JvolMetadata {
    pub shape: [usize; 3],
    pub num_channels: usize,
    pub ijk_to_ras: Affine4x4,
    pub dtype: JvolDtype,
    pub wavelet: WaveletType,
    pub levels: usize,
    pub quality: u8, // 0 = lossless
}

/// Supported data types for volume arrays.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum JvolDtype {
    U8,
    U16,
    I16,
    I32,
    F32,
    F64,
}

impl JvolDtype {
    /// Get the min/max range for integer dtypes.
    pub fn iinfo(&self) -> Option<(f64, f64)> {
        match self {
            JvolDtype::U8 => Some((0.0, 255.0)),
            JvolDtype::U16 => Some((0.0, 65535.0)),
            JvolDtype::I16 => Some((-32768.0, 32767.0)),
            JvolDtype::I32 => Some((-2147483648.0, 2147483647.0)),
            JvolDtype::F32 | JvolDtype::F64 => None,
        }
    }
}

/// One Rice-coded subband within an encoded channel.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncodedSubband {
    /// Rice parameter k used for this subband.
    pub rice_k: u8,
    /// Number of coefficient values in this subband.
    pub num_values: u32,
    /// Rice-coded bitstream bytes.
    pub data: Vec<u8>,
}

/// One encoded channel of a volume.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncodedChannel {
    /// Per-subband Rice-coded data (ordered: coarsest detail first, approximation last).
    pub subbands: Vec<EncodedSubband>,
    /// Normalization intercept (lossy only; min value of original data).
    pub intercept: f64,
    /// Normalization slope (lossy only; max - min of original data).
    pub slope: f64,
    /// Quantization step (lossy only; computed from quality).
    pub step: f64,
}

/// The full encoded representation of a (possibly multi-channel) volume.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncodedVolume {
    pub metadata: JvolMetadata,
    pub channels: Vec<EncodedChannel>,
}
