//! `NIfTI` header parsing and representation.
//!
//! Supports both NIfTI-1 (348-byte header) and NIfTI-2 (540-byte header) formats
//! with automatic version detection and endianness handling.

use crate::error::{Error, Result};
use byteorder::{BigEndian, ByteOrder, LittleEndian};

/// NIfTI format version.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NiftiVersion {
    /// NIfTI-1 format (348-byte header, 16-bit dimensions)
    #[default]
    Nifti1,
    /// NIfTI-2 format (540-byte header, 64-bit dimensions)
    Nifti2,
}

impl NiftiVersion {
    /// Header size in bytes for this version.
    pub const fn header_size(self) -> usize {
        match self {
            Self::Nifti1 => 348,
            Self::Nifti2 => 540,
        }
    }

    /// Default vox_offset for this version (header size + padding).
    pub const fn default_vox_offset(self) -> i64 {
        match self {
            Self::Nifti1 => 352,
            Self::Nifti2 => 544,
        }
    }
}

/// NIfTI-1 header field byte offsets.
mod offsets_v1 {
    pub const SIZEOF_HDR: usize = 0;
    pub const DIM: usize = 40;
    pub const INTENT_CODE: usize = 68;
    pub const DATATYPE: usize = 70;
    pub const BITPIX: usize = 72;
    pub const PIXDIM: usize = 76;
    pub const VOX_OFFSET: usize = 108;
    pub const SCL_SLOPE: usize = 112;
    pub const SCL_INTER: usize = 116;
    pub const XYZT_UNITS: usize = 123;
    pub const DESCRIP: usize = 148;
    pub const AUX_FILE: usize = 228;
    pub const QFORM_CODE: usize = 252;
    pub const SFORM_CODE: usize = 254;
    pub const QUATERN_B: usize = 256;
    pub const QUATERN_C: usize = 260;
    pub const QUATERN_D: usize = 264;
    pub const QOFFSET_X: usize = 268;
    pub const QOFFSET_Y: usize = 272;
    pub const QOFFSET_Z: usize = 276;
    pub const SROW_X: usize = 280;
    pub const SROW_Y: usize = 296;
    pub const SROW_Z: usize = 312;
    pub const MAGIC: usize = 344;
}

/// NIfTI-2 header field byte offsets.
///
/// Note: Some constants (INTENT_P1-P3, CAL_MAX/MIN, SLICE_*, etc.) are defined
/// for NIfTI-2 spec completeness but not currently used in parsing/writing.
/// They are retained as documentation of the full header structure.
#[allow(dead_code)]
mod offsets_v2 {
    pub const SIZEOF_HDR: usize = 0;
    pub const MAGIC: usize = 4;
    pub const DATATYPE: usize = 12;
    pub const BITPIX: usize = 14;
    pub const DIM: usize = 16;
    pub const INTENT_P1: usize = 80;
    pub const INTENT_P2: usize = 88;
    pub const INTENT_P3: usize = 96;
    pub const PIXDIM: usize = 104;
    pub const VOX_OFFSET: usize = 168;
    pub const SCL_SLOPE: usize = 176;
    pub const SCL_INTER: usize = 184;
    pub const CAL_MAX: usize = 192;
    pub const CAL_MIN: usize = 200;
    pub const SLICE_DURATION: usize = 208;
    pub const TOFFSET: usize = 216;
    pub const SLICE_START: usize = 224;
    pub const SLICE_END: usize = 232;
    pub const DESCRIP: usize = 240;
    pub const AUX_FILE: usize = 320;
    pub const QFORM_CODE: usize = 344;
    pub const SFORM_CODE: usize = 348;
    pub const QUATERN_B: usize = 352;
    pub const QUATERN_C: usize = 360;
    pub const QUATERN_D: usize = 368;
    pub const QOFFSET_X: usize = 376;
    pub const QOFFSET_Y: usize = 384;
    pub const QOFFSET_Z: usize = 392;
    pub const SROW_X: usize = 400;
    pub const SROW_Y: usize = 432;
    pub const SROW_Z: usize = 464;
    pub const SLICE_CODE: usize = 496;
    pub const XYZT_UNITS: usize = 500;
    pub const INTENT_CODE: usize = 504;
    pub const INTENT_NAME: usize = 508;
    pub const DIM_INFO: usize = 524;
}

/// `NIfTI` data type codes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i16)]
pub enum DataType {
    /// Unsigned 8-bit integer
    UInt8 = 2,
    /// Signed 16-bit integer
    Int16 = 4,
    /// Signed 32-bit integer
    Int32 = 8,
    /// 32-bit floating point
    Float32 = 16,
    /// 64-bit floating point
    Float64 = 64,
    /// Signed 8-bit integer
    Int8 = 256,
    /// Unsigned 16-bit integer
    UInt16 = 512,
    /// Unsigned 32-bit integer
    UInt32 = 768,
    /// Signed 64-bit integer
    Int64 = 1024,
    /// Unsigned 64-bit integer
    UInt64 = 1280,
    /// IEEE 754 16-bit floating point (half precision)
    Float16 = 16384,
    /// Brain floating point 16-bit (bfloat16)
    BFloat16 = 16385,
}

impl DataType {
    /// Parse from `NIfTI` datatype code.
    pub fn from_code(code: i16) -> Result<Self> {
        match code {
            2 => Ok(Self::UInt8),
            4 => Ok(Self::Int16),
            8 => Ok(Self::Int32),
            16 => Ok(Self::Float32),
            64 => Ok(Self::Float64),
            256 => Ok(Self::Int8),
            512 => Ok(Self::UInt16),
            768 => Ok(Self::UInt32),
            1024 => Ok(Self::Int64),
            1280 => Ok(Self::UInt64),
            16384 => Ok(Self::Float16),
            16385 => Ok(Self::BFloat16),
            _ => Err(Error::UnsupportedDataType(code)),
        }
    }

    /// Size of each element in bytes.
    pub const fn byte_size(self) -> usize {
        match self {
            Self::UInt8 | Self::Int8 => 1,
            Self::Int16 | Self::UInt16 | Self::Float16 | Self::BFloat16 => 2,
            Self::Int32 | Self::UInt32 | Self::Float32 => 4,
            Self::Int64 | Self::UInt64 | Self::Float64 => 8,
        }
    }

    /// Size of each element in bytes (alias for consistency).
    pub const fn size(self) -> usize {
        self.byte_size()
    }

    /// Get the Rust type name for documentation.
    pub const fn type_name(self) -> &'static str {
        match self {
            Self::UInt8 => "u8",
            Self::Int8 => "i8",
            Self::Int16 => "i16",
            Self::UInt16 => "u16",
            Self::Int32 => "i32",
            Self::UInt32 => "u32",
            Self::Int64 => "i64",
            Self::UInt64 => "u64",
            Self::Float16 => "f16",
            Self::BFloat16 => "bf16",
            Self::Float32 => "f32",
            Self::Float64 => "f64",
        }
    }
}

impl std::fmt::Display for DataType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.type_name())
    }
}

impl std::str::FromStr for DataType {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "u8" | "uint8" => Ok(Self::UInt8),
            "i8" | "int8" => Ok(Self::Int8),
            "i16" | "int16" => Ok(Self::Int16),
            "u16" | "uint16" => Ok(Self::UInt16),
            "i32" | "int32" => Ok(Self::Int32),
            "u32" | "uint32" => Ok(Self::UInt32),
            "i64" | "int64" => Ok(Self::Int64),
            "u64" | "uint64" => Ok(Self::UInt64),
            "f16" | "float16" => Ok(Self::Float16),
            "bf16" | "bfloat16" => Ok(Self::BFloat16),
            "f32" | "float32" => Ok(Self::Float32),
            "f64" | "float64" => Ok(Self::Float64),
            _ => Err(Error::Configuration(format!(
                "unknown data type: '{}' (expected u8, i16, f32, etc.)",
                s
            ))),
        }
    }
}

/// Spatial units for voxel dimensions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SpatialUnits {
    #[default]
    /// Units are not specified.
    Unknown,
    /// Voxel dimensions expressed in meters.
    Meter,
    /// Voxel dimensions expressed in millimeters.
    Millimeter,
    /// Voxel dimensions expressed in micrometers.
    Micrometer,
}

impl SpatialUnits {
    fn from_code(code: u8) -> Self {
        match code & 0x07 {
            1 => Self::Meter,
            2 => Self::Millimeter,
            3 => Self::Micrometer,
            _ => Self::Unknown,
        }
    }

    fn to_code(self) -> u8 {
        match self {
            Self::Unknown => 0,
            Self::Meter => 1,
            Self::Millimeter => 2,
            Self::Micrometer => 3,
        }
    }
}

/// Temporal units for time dimensions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TemporalUnits {
    #[default]
    /// Temporal spacing unspecified.
    Unknown,
    /// Temporal spacing in seconds.
    Second,
    /// Temporal spacing in milliseconds.
    Millisecond,
    /// Temporal spacing in microseconds.
    Microsecond,
}

impl TemporalUnits {
    fn from_code(code: u8) -> Self {
        match code & 0x38 {
            0x08 => Self::Second,
            0x10 => Self::Millisecond,
            0x18 => Self::Microsecond,
            _ => Self::Unknown,
        }
    }

    fn to_code(self) -> u8 {
        match self {
            Self::Unknown => 0,
            Self::Second => 0x08,
            Self::Millisecond => 0x10,
            Self::Microsecond => 0x18,
        }
    }
}

/// Unified NIfTI header supporting both NIfTI-1 and NIfTI-2 formats.
///
/// Internally uses 64-bit dimensions and f64 precision to accommodate NIfTI-2.
/// When writing NIfTI-1, values are downcast (with validation that they fit).
#[derive(Debug, Clone)]
pub struct NiftiHeader {
    /// NIfTI format version.
    pub version: NiftiVersion,
    /// Number of dimensions (1-7).
    pub ndim: u8,
    /// Size along each dimension (64-bit for NIfTI-2 compatibility).
    pub dim: [i64; 7],
    /// Data type.
    pub datatype: DataType,
    /// Voxel sizes (pixdim[1..=ndim]) and qfac at index 0 (f64 for NIfTI-2).
    pub pixdim: [f64; 8],
    /// Data offset in file (i64 for NIfTI-2).
    pub vox_offset: i64,
    /// Data scaling slope (f64 for NIfTI-2).
    pub scl_slope: f64,
    /// Data scaling intercept (f64 for NIfTI-2).
    pub scl_inter: f64,
    /// Spatial units.
    pub spatial_units: SpatialUnits,
    /// Temporal units.
    pub temporal_units: TemporalUnits,
    /// Intent code (i32 for NIfTI-2).
    pub intent_code: i32,
    /// Description string.
    pub descrip: String,
    /// Auxiliary filename.
    pub aux_file: String,
    /// qform transform code (i32 for NIfTI-2).
    pub qform_code: i32,
    /// sform transform code (i32 for NIfTI-2).
    pub sform_code: i32,
    /// Quaternion parameters for qform (f64 for NIfTI-2).
    pub quatern: [f64; 3],
    /// Offset parameters for qform (f64 for NIfTI-2).
    pub qoffset: [f64; 3],
    /// First row of the sform affine matrix (f64 for NIfTI-2).
    pub srow_x: [f64; 4],
    /// Second row of the sform affine matrix (f64 for NIfTI-2).
    pub srow_y: [f64; 4],
    /// Third row of the sform affine matrix (f64 for NIfTI-2).
    pub srow_z: [f64; 4],
    /// File endianness (true = little endian).
    pub(crate) little_endian: bool,
}

impl Default for NiftiHeader {
    fn default() -> Self {
        Self {
            version: NiftiVersion::Nifti1,
            ndim: 3,
            dim: [1, 1, 1, 1, 1, 1, 1],
            datatype: DataType::Float32,
            pixdim: [1.0; 8],
            vox_offset: 352,
            scl_slope: 1.0,
            scl_inter: 0.0,
            spatial_units: SpatialUnits::Millimeter,
            temporal_units: TemporalUnits::Unknown,
            intent_code: 0,
            descrip: String::new(),
            aux_file: String::new(),
            qform_code: 0,
            sform_code: 1,
            quatern: [0.0; 3],
            qoffset: [0.0; 3],
            srow_x: [1.0, 0.0, 0.0, 0.0],
            srow_y: [0.0, 1.0, 0.0, 0.0],
            srow_z: [0.0, 0.0, 1.0, 0.0],
            little_endian: true,
        }
    }
}

impl NiftiHeader {
    /// Size of NIfTI-1 header in bytes.
    pub const SIZE: usize = 348;

    /// Size of NIfTI-2 header in bytes.
    pub const SIZE_V2: usize = 540;

    /// Returns the header size for this header's version.
    pub fn header_size(&self) -> usize {
        self.version.header_size()
    }

    /// Read header from bytes with automatic version and endianness detection.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < 4 {
            return Err(Error::Io(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "header too short to detect version",
            )));
        }

        // Detect version and endianness from sizeof_hdr field
        let sizeof_hdr_le = LittleEndian::read_i32(&bytes[0..4]);
        let sizeof_hdr_be = BigEndian::read_i32(&bytes[0..4]);

        let (version, little_endian) = if sizeof_hdr_le == 348 {
            (NiftiVersion::Nifti1, true)
        } else if sizeof_hdr_be == 348 {
            (NiftiVersion::Nifti1, false)
        } else if sizeof_hdr_le == 540 {
            (NiftiVersion::Nifti2, true)
        } else if sizeof_hdr_be == 540 {
            (NiftiVersion::Nifti2, false)
        } else {
            return Err(Error::InvalidMagic([
                bytes[0], bytes[1], bytes[2], bytes[3],
            ]));
        };

        let required_size = version.header_size();
        if bytes.len() < required_size {
            return Err(Error::Io(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                format!(
                    "header too short: got {} bytes, need {} for {:?}",
                    bytes.len(),
                    required_size,
                    version
                ),
            )));
        }

        match (version, little_endian) {
            (NiftiVersion::Nifti1, true) => Self::parse_v1::<LittleEndian>(bytes, true),
            (NiftiVersion::Nifti1, false) => Self::parse_v1::<BigEndian>(bytes, false),
            (NiftiVersion::Nifti2, true) => Self::parse_v2::<LittleEndian>(bytes, true),
            (NiftiVersion::Nifti2, false) => Self::parse_v2::<BigEndian>(bytes, false),
        }
    }

    /// Parse NIfTI-1 header.
    #[allow(clippy::wildcard_imports)]
    fn parse_v1<E: ByteOrder>(bytes: &[u8], little_endian: bool) -> Result<Self> {
        use offsets_v1::*;

        // Validate magic
        let magic = &bytes[MAGIC..MAGIC + 4];
        if magic != b"n+1\0" && magic != b"ni1\0" {
            return Err(Error::InvalidMagic([
                magic[0], magic[1], magic[2], magic[3],
            ]));
        }

        let ndim_raw = E::read_i16(&bytes[DIM..DIM + 2]);
        if !(1..=7).contains(&ndim_raw) {
            return Err(Error::InvalidDimensions(format!(
                "ndim must be 1..=7, got {}",
                ndim_raw
            )));
        }
        let ndim = ndim_raw as u8;

        let mut dim = [0i64; 7];
        for (i, dim_val) in dim.iter_mut().enumerate() {
            let offset = DIM + 2 + i * 2;
            let dim_raw = E::read_i16(&bytes[offset..offset + 2]);
            if dim_raw < 0 {
                return Err(Error::InvalidDimensions(format!(
                    "dimension {} has negative value: {}",
                    i, dim_raw
                )));
            }
            *dim_val = dim_raw as i64;
        }

        let datatype = DataType::from_code(E::read_i16(&bytes[DATATYPE..DATATYPE + 2]))?;
        let bitpix = E::read_i16(&bytes[BITPIX..BITPIX + 2]);
        let expected_bitpix = (datatype.byte_size() * 8) as i16;
        if bitpix != expected_bitpix {
            return Err(Error::InvalidDimensions(format!(
                "bitpix {} does not match datatype {} (expected {})",
                bitpix,
                datatype.type_name(),
                expected_bitpix
            )));
        }

        let mut pixdim = [0.0f64; 8];
        for (i, pix_val) in pixdim.iter_mut().enumerate() {
            let offset = PIXDIM + i * 4;
            *pix_val = E::read_f32(&bytes[offset..offset + 4]) as f64;
        }

        let xyzt_units = bytes[XYZT_UNITS];

        let descrip = String::from_utf8_lossy(&bytes[DESCRIP..AUX_FILE])
            .trim_end_matches('\0')
            .to_string();
        let aux_file = String::from_utf8_lossy(&bytes[AUX_FILE..QFORM_CODE])
            .trim_end_matches('\0')
            .to_string();

        let vox_offset_raw = E::read_f32(&bytes[VOX_OFFSET..VOX_OFFSET + 4]);
        if !vox_offset_raw.is_finite() {
            return Err(Error::InvalidDimensions(format!(
                "vox_offset must be finite, got {}",
                vox_offset_raw
            )));
        }
        if vox_offset_raw.fract() != 0.0 {
            return Err(Error::InvalidDimensions(format!(
                "vox_offset must be an integer, got {}",
                vox_offset_raw
            )));
        }

        let header = Self {
            version: NiftiVersion::Nifti1,
            ndim,
            dim,
            datatype,
            pixdim,
            vox_offset: vox_offset_raw as i64,
            scl_slope: E::read_f32(&bytes[SCL_SLOPE..SCL_SLOPE + 4]) as f64,
            scl_inter: E::read_f32(&bytes[SCL_INTER..SCL_INTER + 4]) as f64,
            spatial_units: SpatialUnits::from_code(xyzt_units),
            temporal_units: TemporalUnits::from_code(xyzt_units),
            intent_code: E::read_i16(&bytes[INTENT_CODE..INTENT_CODE + 2]) as i32,
            descrip,
            aux_file,
            qform_code: E::read_i16(&bytes[QFORM_CODE..QFORM_CODE + 2]) as i32,
            sform_code: E::read_i16(&bytes[SFORM_CODE..SFORM_CODE + 2]) as i32,
            quatern: [
                E::read_f32(&bytes[QUATERN_B..QUATERN_B + 4]) as f64,
                E::read_f32(&bytes[QUATERN_C..QUATERN_C + 4]) as f64,
                E::read_f32(&bytes[QUATERN_D..QUATERN_D + 4]) as f64,
            ],
            qoffset: [
                E::read_f32(&bytes[QOFFSET_X..QOFFSET_X + 4]) as f64,
                E::read_f32(&bytes[QOFFSET_Y..QOFFSET_Y + 4]) as f64,
                E::read_f32(&bytes[QOFFSET_Z..QOFFSET_Z + 4]) as f64,
            ],
            srow_x: [
                E::read_f32(&bytes[SROW_X..SROW_X + 4]) as f64,
                E::read_f32(&bytes[SROW_X + 4..SROW_X + 8]) as f64,
                E::read_f32(&bytes[SROW_X + 8..SROW_X + 12]) as f64,
                E::read_f32(&bytes[SROW_X + 12..SROW_X + 16]) as f64,
            ],
            srow_y: [
                E::read_f32(&bytes[SROW_Y..SROW_Y + 4]) as f64,
                E::read_f32(&bytes[SROW_Y + 4..SROW_Y + 8]) as f64,
                E::read_f32(&bytes[SROW_Y + 8..SROW_Y + 12]) as f64,
                E::read_f32(&bytes[SROW_Y + 12..SROW_Y + 16]) as f64,
            ],
            srow_z: [
                E::read_f32(&bytes[SROW_Z..SROW_Z + 4]) as f64,
                E::read_f32(&bytes[SROW_Z + 4..SROW_Z + 8]) as f64,
                E::read_f32(&bytes[SROW_Z + 8..SROW_Z + 12]) as f64,
                E::read_f32(&bytes[SROW_Z + 12..SROW_Z + 16]) as f64,
            ],
            little_endian,
        };

        header.validate()?;
        Ok(header)
    }

    /// Parse NIfTI-2 header.
    #[allow(clippy::wildcard_imports)]
    fn parse_v2<E: ByteOrder>(bytes: &[u8], little_endian: bool) -> Result<Self> {
        use offsets_v2::*;

        // Validate magic (at offset 4 in NIfTI-2)
        let magic = &bytes[MAGIC..MAGIC + 8];
        if magic != b"n+2\0\r\n\x1a\n" && magic != b"ni2\0\r\n\x1a\n" {
            return Err(Error::InvalidMagic([
                magic[0], magic[1], magic[2], magic[3],
            ]));
        }

        // dim[0] is ndim, stored as i64 at offset 16
        let ndim_raw = E::read_i64(&bytes[DIM..DIM + 8]);
        if !(1..=7).contains(&ndim_raw) {
            return Err(Error::InvalidDimensions(format!(
                "ndim must be 1..=7, got {}",
                ndim_raw
            )));
        }
        let ndim = ndim_raw as u8;

        let mut dim = [0i64; 7];
        for (i, dim_val) in dim.iter_mut().enumerate() {
            let offset = DIM + 8 + i * 8;
            *dim_val = E::read_i64(&bytes[offset..offset + 8]);
            if *dim_val < 0 {
                return Err(Error::InvalidDimensions(format!(
                    "dimension {} has negative value: {}",
                    i, *dim_val
                )));
            }
        }

        let datatype = DataType::from_code(E::read_i16(&bytes[DATATYPE..DATATYPE + 2]))?;
        let bitpix = E::read_i16(&bytes[BITPIX..BITPIX + 2]);
        let expected_bitpix = (datatype.byte_size() * 8) as i16;
        if bitpix != expected_bitpix {
            return Err(Error::InvalidDimensions(format!(
                "bitpix {} does not match datatype {} (expected {})",
                bitpix,
                datatype.type_name(),
                expected_bitpix
            )));
        }

        let mut pixdim = [0.0f64; 8];
        for (i, pix_val) in pixdim.iter_mut().enumerate() {
            let offset = PIXDIM + i * 8;
            *pix_val = E::read_f64(&bytes[offset..offset + 8]);
        }

        let xyzt_units_raw = E::read_i32(&bytes[XYZT_UNITS..XYZT_UNITS + 4]);
        let xyzt_units = xyzt_units_raw as u8;

        let descrip = String::from_utf8_lossy(&bytes[DESCRIP..DESCRIP + 80])
            .trim_end_matches('\0')
            .to_string();
        let aux_file = String::from_utf8_lossy(&bytes[AUX_FILE..AUX_FILE + 24])
            .trim_end_matches('\0')
            .to_string();

        let header = Self {
            version: NiftiVersion::Nifti2,
            ndim,
            dim,
            datatype,
            pixdim,
            vox_offset: E::read_i64(&bytes[VOX_OFFSET..VOX_OFFSET + 8]),
            scl_slope: E::read_f64(&bytes[SCL_SLOPE..SCL_SLOPE + 8]),
            scl_inter: E::read_f64(&bytes[SCL_INTER..SCL_INTER + 8]),
            spatial_units: SpatialUnits::from_code(xyzt_units),
            temporal_units: TemporalUnits::from_code(xyzt_units),
            intent_code: E::read_i32(&bytes[INTENT_CODE..INTENT_CODE + 4]),
            descrip,
            aux_file,
            qform_code: E::read_i32(&bytes[QFORM_CODE..QFORM_CODE + 4]),
            sform_code: E::read_i32(&bytes[SFORM_CODE..SFORM_CODE + 4]),
            quatern: [
                E::read_f64(&bytes[QUATERN_B..QUATERN_B + 8]),
                E::read_f64(&bytes[QUATERN_C..QUATERN_C + 8]),
                E::read_f64(&bytes[QUATERN_D..QUATERN_D + 8]),
            ],
            qoffset: [
                E::read_f64(&bytes[QOFFSET_X..QOFFSET_X + 8]),
                E::read_f64(&bytes[QOFFSET_Y..QOFFSET_Y + 8]),
                E::read_f64(&bytes[QOFFSET_Z..QOFFSET_Z + 8]),
            ],
            srow_x: [
                E::read_f64(&bytes[SROW_X..SROW_X + 8]),
                E::read_f64(&bytes[SROW_X + 8..SROW_X + 16]),
                E::read_f64(&bytes[SROW_X + 16..SROW_X + 24]),
                E::read_f64(&bytes[SROW_X + 24..SROW_X + 32]),
            ],
            srow_y: [
                E::read_f64(&bytes[SROW_Y..SROW_Y + 8]),
                E::read_f64(&bytes[SROW_Y + 8..SROW_Y + 16]),
                E::read_f64(&bytes[SROW_Y + 16..SROW_Y + 24]),
                E::read_f64(&bytes[SROW_Y + 24..SROW_Y + 32]),
            ],
            srow_z: [
                E::read_f64(&bytes[SROW_Z..SROW_Z + 8]),
                E::read_f64(&bytes[SROW_Z + 8..SROW_Z + 16]),
                E::read_f64(&bytes[SROW_Z + 16..SROW_Z + 24]),
                E::read_f64(&bytes[SROW_Z + 24..SROW_Z + 32]),
            ],
            little_endian,
        };

        header.validate()?;
        Ok(header)
    }

    /// Write header to bytes.
    ///
    /// Writes NIfTI-1 format by default. Use `to_bytes_v2()` for NIfTI-2.
    /// Returns error if dimensions exceed NIfTI-1 limits (32767).
    pub fn to_bytes(&self) -> Vec<u8> {
        match self.version {
            NiftiVersion::Nifti1 => self.to_bytes_v1(),
            NiftiVersion::Nifti2 => self.to_bytes_v2(),
        }
    }

    /// Write NIfTI-1 format header.
    #[allow(clippy::wildcard_imports)]
    fn to_bytes_v1(&self) -> Vec<u8> {
        use offsets_v1::*;

        let mut buf = vec![0u8; Self::SIZE];

        LittleEndian::write_i32(&mut buf[SIZEOF_HDR..SIZEOF_HDR + 4], 348);

        // dim
        LittleEndian::write_i16(&mut buf[DIM..DIM + 2], self.ndim as i16);
        for i in 0..7 {
            let offset = DIM + 2 + i * 2;
            let dim_val = self.dim[i].min(i16::MAX as i64) as i16;
            LittleEndian::write_i16(&mut buf[offset..offset + 2], dim_val);
        }

        // datatype and bitpix
        LittleEndian::write_i16(&mut buf[DATATYPE..DATATYPE + 2], self.datatype as i16);
        LittleEndian::write_i16(
            &mut buf[BITPIX..BITPIX + 2],
            (self.datatype.byte_size() * 8) as i16,
        );

        // pixdim (downcast from f64 to f32)
        for (i, &value) in self.pixdim.iter().enumerate() {
            let offset = PIXDIM + i * 4;
            LittleEndian::write_f32(&mut buf[offset..offset + 4], value as f32);
        }

        // vox_offset (downcast from i64 to f32)
        LittleEndian::write_f32(&mut buf[VOX_OFFSET..VOX_OFFSET + 4], self.vox_offset as f32);

        // scl_slope, scl_inter (downcast from f64 to f32)
        LittleEndian::write_f32(&mut buf[SCL_SLOPE..SCL_SLOPE + 4], self.scl_slope as f32);
        LittleEndian::write_f32(&mut buf[SCL_INTER..SCL_INTER + 4], self.scl_inter as f32);

        // xyzt_units
        buf[XYZT_UNITS] = self.spatial_units.to_code() | self.temporal_units.to_code();

        // descrip (80 bytes)
        let descrip_bytes = self.descrip.as_bytes();
        let len = descrip_bytes.len().min(79);
        buf[DESCRIP..DESCRIP + len].copy_from_slice(&descrip_bytes[..len]);

        // aux_file (24 bytes)
        let aux_bytes = self.aux_file.as_bytes();
        let len = aux_bytes.len().min(23);
        buf[AUX_FILE..AUX_FILE + len].copy_from_slice(&aux_bytes[..len]);

        // qform_code, sform_code (downcast from i32 to i16)
        LittleEndian::write_i16(&mut buf[QFORM_CODE..QFORM_CODE + 2], self.qform_code as i16);
        LittleEndian::write_i16(&mut buf[SFORM_CODE..SFORM_CODE + 2], self.sform_code as i16);

        // quatern (downcast from f64 to f32)
        LittleEndian::write_f32(&mut buf[QUATERN_B..QUATERN_B + 4], self.quatern[0] as f32);
        LittleEndian::write_f32(&mut buf[QUATERN_C..QUATERN_C + 4], self.quatern[1] as f32);
        LittleEndian::write_f32(&mut buf[QUATERN_D..QUATERN_D + 4], self.quatern[2] as f32);

        // qoffset (downcast from f64 to f32)
        LittleEndian::write_f32(&mut buf[QOFFSET_X..QOFFSET_X + 4], self.qoffset[0] as f32);
        LittleEndian::write_f32(&mut buf[QOFFSET_Y..QOFFSET_Y + 4], self.qoffset[1] as f32);
        LittleEndian::write_f32(&mut buf[QOFFSET_Z..QOFFSET_Z + 4], self.qoffset[2] as f32);

        // srow (downcast from f64 to f32)
        for (i, &v) in self.srow_x.iter().enumerate() {
            let offset = SROW_X + i * 4;
            LittleEndian::write_f32(&mut buf[offset..offset + 4], v as f32);
        }
        for (i, &v) in self.srow_y.iter().enumerate() {
            let offset = SROW_Y + i * 4;
            LittleEndian::write_f32(&mut buf[offset..offset + 4], v as f32);
        }
        for (i, &v) in self.srow_z.iter().enumerate() {
            let offset = SROW_Z + i * 4;
            LittleEndian::write_f32(&mut buf[offset..offset + 4], v as f32);
        }

        // magic
        buf[MAGIC..MAGIC + 4].copy_from_slice(b"n+1\0");

        buf
    }

    /// Write NIfTI-2 format header.
    #[allow(clippy::wildcard_imports)]
    pub fn to_bytes_v2(&self) -> Vec<u8> {
        use offsets_v2::*;

        let mut buf = vec![0u8; Self::SIZE_V2];

        LittleEndian::write_i32(&mut buf[SIZEOF_HDR..SIZEOF_HDR + 4], 540);

        // magic (8 bytes for NIfTI-2)
        buf[MAGIC..MAGIC + 8].copy_from_slice(b"n+2\0\r\n\x1a\n");

        // datatype and bitpix
        LittleEndian::write_i16(&mut buf[DATATYPE..DATATYPE + 2], self.datatype as i16);
        LittleEndian::write_i16(
            &mut buf[BITPIX..BITPIX + 2],
            (self.datatype.byte_size() * 8) as i16,
        );

        // dim (i64 array)
        LittleEndian::write_i64(&mut buf[DIM..DIM + 8], self.ndim as i64);
        for i in 0..7 {
            let offset = DIM + 8 + i * 8;
            LittleEndian::write_i64(&mut buf[offset..offset + 8], self.dim[i]);
        }

        // pixdim (f64 array)
        for (i, &value) in self.pixdim.iter().enumerate() {
            let offset = PIXDIM + i * 8;
            LittleEndian::write_f64(&mut buf[offset..offset + 8], value);
        }

        // vox_offset (i64)
        LittleEndian::write_i64(&mut buf[VOX_OFFSET..VOX_OFFSET + 8], self.vox_offset);

        // scl_slope, scl_inter (f64)
        LittleEndian::write_f64(&mut buf[SCL_SLOPE..SCL_SLOPE + 8], self.scl_slope);
        LittleEndian::write_f64(&mut buf[SCL_INTER..SCL_INTER + 8], self.scl_inter);

        // xyzt_units (i32 in NIfTI-2)
        let xyzt_code = (self.spatial_units.to_code() | self.temporal_units.to_code()) as i32;
        LittleEndian::write_i32(&mut buf[XYZT_UNITS..XYZT_UNITS + 4], xyzt_code);

        // descrip (80 bytes)
        let descrip_bytes = self.descrip.as_bytes();
        let len = descrip_bytes.len().min(79);
        buf[DESCRIP..DESCRIP + len].copy_from_slice(&descrip_bytes[..len]);

        // aux_file (24 bytes)
        let aux_bytes = self.aux_file.as_bytes();
        let len = aux_bytes.len().min(23);
        buf[AUX_FILE..AUX_FILE + len].copy_from_slice(&aux_bytes[..len]);

        // qform_code, sform_code (i32)
        LittleEndian::write_i32(&mut buf[QFORM_CODE..QFORM_CODE + 4], self.qform_code);
        LittleEndian::write_i32(&mut buf[SFORM_CODE..SFORM_CODE + 4], self.sform_code);

        // quatern (f64)
        LittleEndian::write_f64(&mut buf[QUATERN_B..QUATERN_B + 8], self.quatern[0]);
        LittleEndian::write_f64(&mut buf[QUATERN_C..QUATERN_C + 8], self.quatern[1]);
        LittleEndian::write_f64(&mut buf[QUATERN_D..QUATERN_D + 8], self.quatern[2]);

        // qoffset (f64)
        LittleEndian::write_f64(&mut buf[QOFFSET_X..QOFFSET_X + 8], self.qoffset[0]);
        LittleEndian::write_f64(&mut buf[QOFFSET_Y..QOFFSET_Y + 8], self.qoffset[1]);
        LittleEndian::write_f64(&mut buf[QOFFSET_Z..QOFFSET_Z + 8], self.qoffset[2]);

        // srow (f64 arrays)
        for (i, &v) in self.srow_x.iter().enumerate() {
            let offset = SROW_X + i * 8;
            LittleEndian::write_f64(&mut buf[offset..offset + 8], v);
        }
        for (i, &v) in self.srow_y.iter().enumerate() {
            let offset = SROW_Y + i * 8;
            LittleEndian::write_f64(&mut buf[offset..offset + 8], v);
        }
        for (i, &v) in self.srow_z.iter().enumerate() {
            let offset = SROW_Z + i * 8;
            LittleEndian::write_f64(&mut buf[offset..offset + 8], v);
        }

        // intent_code (i32)
        LittleEndian::write_i32(&mut buf[INTENT_CODE..INTENT_CODE + 4], self.intent_code);

        buf
    }

    /// Get the 4x4 affine transformation matrix (sform or qform).
    /// Returns f32 for API compatibility; use `affine_f64()` for full precision.
    pub fn affine(&self) -> [[f32; 4]; 4] {
        let aff64 = self.affine_f64();
        [
            [
                aff64[0][0] as f32,
                aff64[0][1] as f32,
                aff64[0][2] as f32,
                aff64[0][3] as f32,
            ],
            [
                aff64[1][0] as f32,
                aff64[1][1] as f32,
                aff64[1][2] as f32,
                aff64[1][3] as f32,
            ],
            [
                aff64[2][0] as f32,
                aff64[2][1] as f32,
                aff64[2][2] as f32,
                aff64[2][3] as f32,
            ],
            [
                aff64[3][0] as f32,
                aff64[3][1] as f32,
                aff64[3][2] as f32,
                aff64[3][3] as f32,
            ],
        ]
    }

    /// Get the 4x4 affine transformation matrix with f64 precision.
    pub fn affine_f64(&self) -> [[f64; 4]; 4] {
        if self.sform_code > 0 {
            [self.srow_x, self.srow_y, self.srow_z, [0.0, 0.0, 0.0, 1.0]]
        } else if self.qform_code > 0 {
            self.qform_to_affine_f64()
        } else {
            // Default: identity scaled by pixdim
            [
                [self.pixdim[1], 0.0, 0.0, 0.0],
                [0.0, self.pixdim[2], 0.0, 0.0],
                [0.0, 0.0, self.pixdim[3], 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        }
    }

    /// Set affine from 4x4 matrix (f32 input for API compatibility).
    pub fn set_affine(&mut self, affine: [[f32; 4]; 4]) {
        self.srow_x = [
            affine[0][0] as f64,
            affine[0][1] as f64,
            affine[0][2] as f64,
            affine[0][3] as f64,
        ];
        self.srow_y = [
            affine[1][0] as f64,
            affine[1][1] as f64,
            affine[1][2] as f64,
            affine[1][3] as f64,
        ];
        self.srow_z = [
            affine[2][0] as f64,
            affine[2][1] as f64,
            affine[2][2] as f64,
            affine[2][3] as f64,
        ];
        self.sform_code = 1;

        // Also set pixdim from affine column norms (voxel spacing)
        // pixdim[0] is qfac (usually 1.0), pixdim[1..4] are x/y/z spacing
        let spacing_x = (affine[0][0] * affine[0][0]
            + affine[1][0] * affine[1][0]
            + affine[2][0] * affine[2][0])
            .sqrt();
        let spacing_y = (affine[0][1] * affine[0][1]
            + affine[1][1] * affine[1][1]
            + affine[2][1] * affine[2][1])
            .sqrt();
        let spacing_z = (affine[0][2] * affine[0][2]
            + affine[1][2] * affine[1][2]
            + affine[2][2] * affine[2][2])
            .sqrt();

        self.pixdim[1] = spacing_x as f64;
        self.pixdim[2] = spacing_y as f64;
        self.pixdim[3] = spacing_z as f64;
    }

    /// Set affine from 4x4 matrix with f64 precision.
    pub fn set_affine_f64(&mut self, affine: [[f64; 4]; 4]) {
        self.srow_x = affine[0];
        self.srow_y = affine[1];
        self.srow_z = affine[2];
        self.sform_code = 1;

        // Also set pixdim from affine column norms (voxel spacing)
        let spacing_x = (affine[0][0] * affine[0][0]
            + affine[1][0] * affine[1][0]
            + affine[2][0] * affine[2][0])
            .sqrt();
        let spacing_y = (affine[0][1] * affine[0][1]
            + affine[1][1] * affine[1][1]
            + affine[2][1] * affine[2][1])
            .sqrt();
        let spacing_z = (affine[0][2] * affine[0][2]
            + affine[1][2] * affine[1][2]
            + affine[2][2] * affine[2][2])
            .sqrt();

        self.pixdim[1] = spacing_x;
        self.pixdim[2] = spacing_y;
        self.pixdim[3] = spacing_z;
    }

    /// Convert quaternion representation to affine matrix (f64 precision).
    #[allow(clippy::many_single_char_names)]
    fn qform_to_affine_f64(&self) -> [[f64; 4]; 4] {
        let [b, c, d] = self.quatern;
        let a = (1.0 - b * b - c * c - d * d).max(0.0).sqrt();

        let qfac = if self.pixdim[0] < 0.0 { -1.0 } else { 1.0 };
        let [i, j, k] = [self.pixdim[1].abs(), self.pixdim[2], self.pixdim[3] * qfac];

        [
            [
                (a * a + b * b - c * c - d * d) * i,
                2.0 * (b * c - a * d) * j,
                2.0 * (b * d + a * c) * k,
                self.qoffset[0],
            ],
            [
                2.0 * (b * c + a * d) * i,
                (a * a - b * b + c * c - d * d) * j,
                2.0 * (c * d - a * b) * k,
                self.qoffset[1],
            ],
            [
                2.0 * (b * d - a * c) * i,
                2.0 * (c * d + a * b) * j,
                (a * a - b * b - c * c + d * d) * k,
                self.qoffset[2],
            ],
            [0.0, 0.0, 0.0, 1.0],
        ]
    }

    /// Get image shape as a Vec<usize> (up to ndim elements).
    /// Returns owned Vec to accommodate both NIfTI-1 and NIfTI-2 dimensions.
    pub fn shape(&self) -> Vec<usize> {
        self.dim[..self.ndim as usize]
            .iter()
            .map(|&d| d as usize)
            .collect()
    }

    /// Get voxel spacing as a Vec<f32> (up to ndim elements).
    /// Returns owned Vec; spacing is stored at pixdim[1..=ndim].
    pub fn spacing(&self) -> Vec<f32> {
        let end = (self.ndim as usize + 1).min(self.pixdim.len());
        self.pixdim[1..end].iter().map(|&p| p as f32).collect()
    }

    /// Get voxel spacing with f64 precision.
    pub fn spacing_f64(&self) -> Vec<f64> {
        let end = (self.ndim as usize + 1).min(self.pixdim.len());
        self.pixdim[1..end].to_vec()
    }

    /// Total number of voxels.
    pub fn num_voxels(&self) -> usize {
        self.dim[..self.ndim as usize]
            .iter()
            .map(|&d| d as usize)
            .product()
    }

    /// Total size of image data in bytes.
    pub fn data_size(&self) -> usize {
        self.num_voxels() * self.datatype.byte_size()
    }

    /// Returns true if file is little endian.
    pub fn is_little_endian(&self) -> bool {
        self.little_endian
    }

    /// Returns true if this header requires NIfTI-2 format.
    /// NIfTI-2 is required if any dimension exceeds 32767.
    pub fn requires_nifti2(&self) -> bool {
        self.dim.iter().any(|&d| d > i16::MAX as i64)
    }

    /// Validate header fields for basic NIfTI invariants.
    pub fn validate(&self) -> Result<()> {
        if self.ndim == 0 || self.ndim > 7 {
            return Err(Error::InvalidDimensions(format!(
                "ndim must be 1..=7, got {}",
                self.ndim
            )));
        }

        for i in 0..self.ndim as usize {
            if self.dim[i] == 0 {
                return Err(Error::InvalidDimensions(format!("dimension {} is zero", i)));
            }
            let spacing = self.pixdim[i + 1];
            if !spacing.is_finite() || spacing <= 0.0 {
                return Err(Error::InvalidDimensions(format!(
                    "pixdim[{}] must be finite and > 0, got {}",
                    i + 1,
                    spacing
                )));
            }
        }

        let vox_offset_f64 = self.vox_offset as f64;
        if !vox_offset_f64.is_finite() {
            return Err(Error::InvalidDimensions(format!(
                "vox_offset must be finite, got {}",
                self.vox_offset
            )));
        }

        let min_offset = self.version.header_size() as i64;
        if self.vox_offset < min_offset {
            return Err(Error::InvalidDimensions(format!(
                "vox_offset {} before header end ({})",
                self.vox_offset, min_offset
            )));
        }

        // Check that voxel count and byte size don't overflow usize
        let mut voxels: usize = 1;
        for i in 0..self.ndim as usize {
            voxels = voxels
                .checked_mul(self.dim[i] as usize)
                .ok_or_else(|| Error::InvalidDimensions("dimension product overflow".into()))?;
        }

        voxels
            .checked_mul(self.datatype.byte_size())
            .ok_or_else(|| Error::InvalidDimensions("data size overflow".into()))?;

        // vox_offset should be aligned to element size for mmap compatibility
        let vox_offset_int = self.vox_offset as usize;
        let byte_size = self.datatype.byte_size();
        if vox_offset_int % byte_size != 0 {
            return Err(Error::InvalidDimensions(format!(
                "vox_offset {} not aligned to element size {}",
                self.vox_offset, byte_size
            )));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_temporal_units_from_code() {
        assert_eq!(TemporalUnits::from_code(0x08), TemporalUnits::Second);
        assert_eq!(TemporalUnits::from_code(0x10), TemporalUnits::Millisecond);
        assert_eq!(TemporalUnits::from_code(0x18), TemporalUnits::Microsecond);
        assert_eq!(TemporalUnits::from_code(0x00), TemporalUnits::Unknown);
        assert_eq!(TemporalUnits::from_code(0x0A), TemporalUnits::Second);
        assert_eq!(TemporalUnits::from_code(0x11), TemporalUnits::Millisecond);
    }

    #[test]
    fn test_temporal_units_to_code() {
        assert_eq!(TemporalUnits::Second.to_code(), 0x08);
        assert_eq!(TemporalUnits::Millisecond.to_code(), 0x10);
        assert_eq!(TemporalUnits::Microsecond.to_code(), 0x18);
        assert_eq!(TemporalUnits::Unknown.to_code(), 0x00);
    }

    #[test]
    fn test_temporal_units_roundtrip() {
        for unit in [
            TemporalUnits::Unknown,
            TemporalUnits::Second,
            TemporalUnits::Millisecond,
            TemporalUnits::Microsecond,
        ] {
            let code = unit.to_code();
            assert_eq!(TemporalUnits::from_code(code), unit);
        }
    }

    #[test]
    fn test_spatial_units_from_code() {
        assert_eq!(SpatialUnits::from_code(0x00), SpatialUnits::Unknown);
        assert_eq!(SpatialUnits::from_code(0x01), SpatialUnits::Meter);
        assert_eq!(SpatialUnits::from_code(0x02), SpatialUnits::Millimeter);
        assert_eq!(SpatialUnits::from_code(0x03), SpatialUnits::Micrometer);
    }

    #[test]
    fn test_spacing_returns_vec() {
        let mut header = NiftiHeader::default();
        header.ndim = 3;
        header.pixdim = [-1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0];
        assert_eq!(header.spacing(), vec![2.0f32, 3.0, 4.0]);
    }

    #[test]
    fn test_shape_returns_vec() {
        let mut header = NiftiHeader::default();
        header.ndim = 3;
        header.dim = [100, 200, 300, 1, 1, 1, 1];
        assert_eq!(header.shape(), vec![100usize, 200, 300]);
    }

    #[test]
    fn test_nifti1_roundtrip() {
        let mut header = NiftiHeader::default();
        header.ndim = 3;
        header.dim = [64, 64, 64, 1, 1, 1, 1];
        header.pixdim = [-1.0, 1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0];

        let bytes = header.to_bytes();
        assert_eq!(bytes.len(), 348);

        let parsed = NiftiHeader::from_bytes(&bytes).unwrap();
        assert_eq!(parsed.version, NiftiVersion::Nifti1);
        assert_eq!(parsed.ndim, 3);
        assert_eq!(parsed.dim[..3], [64, 64, 64]);
    }

    #[test]
    fn test_nifti2_roundtrip() {
        let mut header = NiftiHeader::default();
        header.version = NiftiVersion::Nifti2;
        header.ndim = 3;
        header.dim = [100000, 100000, 100, 1, 1, 1, 1]; // Exceeds NIfTI-1 limit
        header.pixdim = [-1.0, 0.5, 0.5, 1.0, 0.0, 0.0, 0.0, 0.0];
        header.vox_offset = 544;

        assert!(header.requires_nifti2());

        let bytes = header.to_bytes_v2();
        assert_eq!(bytes.len(), 540);

        let parsed = NiftiHeader::from_bytes(&bytes).unwrap();
        assert_eq!(parsed.version, NiftiVersion::Nifti2);
        assert_eq!(parsed.ndim, 3);
        assert_eq!(parsed.dim[0], 100000);
        assert_eq!(parsed.dim[1], 100000);
    }

    #[test]
    fn test_version_detection() {
        // NIfTI-1 header starts with 348
        let mut v1_bytes = vec![0u8; 348];
        LittleEndian::write_i32(&mut v1_bytes[0..4], 348);
        v1_bytes[344..348].copy_from_slice(b"n+1\0");
        // Set minimal valid header fields
        LittleEndian::write_i16(&mut v1_bytes[40..42], 3); // ndim
        LittleEndian::write_i16(&mut v1_bytes[42..44], 10); // dim[0]
        LittleEndian::write_i16(&mut v1_bytes[44..46], 10); // dim[1]
        LittleEndian::write_i16(&mut v1_bytes[46..48], 10); // dim[2]
        LittleEndian::write_i16(&mut v1_bytes[70..72], 16); // datatype = Float32
        LittleEndian::write_i16(&mut v1_bytes[72..74], 32); // bitpix = 32 (4 bytes * 8)
        LittleEndian::write_f32(&mut v1_bytes[80..84], 1.0); // pixdim[1]
        LittleEndian::write_f32(&mut v1_bytes[84..88], 1.0); // pixdim[2]
        LittleEndian::write_f32(&mut v1_bytes[88..92], 1.0); // pixdim[3]
        LittleEndian::write_f32(&mut v1_bytes[108..112], 352.0); // vox_offset

        let h1 = NiftiHeader::from_bytes(&v1_bytes).unwrap();
        assert_eq!(h1.version, NiftiVersion::Nifti1);

        // NIfTI-2 header starts with 540
        let mut v2_bytes = vec![0u8; 540];
        LittleEndian::write_i32(&mut v2_bytes[0..4], 540);
        v2_bytes[4..12].copy_from_slice(b"n+2\0\r\n\x1a\n");
        LittleEndian::write_i16(&mut v2_bytes[12..14], 16); // datatype = Float32
        LittleEndian::write_i16(&mut v2_bytes[14..16], 32); // bitpix = 32 (4 bytes * 8)
        LittleEndian::write_i64(&mut v2_bytes[16..24], 3); // ndim
        LittleEndian::write_i64(&mut v2_bytes[24..32], 10); // dim[0]
        LittleEndian::write_i64(&mut v2_bytes[32..40], 10); // dim[1]
        LittleEndian::write_i64(&mut v2_bytes[40..48], 10); // dim[2]
        LittleEndian::write_f64(&mut v2_bytes[112..120], 1.0); // pixdim[1]
        LittleEndian::write_f64(&mut v2_bytes[120..128], 1.0); // pixdim[2]
        LittleEndian::write_f64(&mut v2_bytes[128..136], 1.0); // pixdim[3]
        LittleEndian::write_i64(&mut v2_bytes[168..176], 544); // vox_offset

        let h2 = NiftiHeader::from_bytes(&v2_bytes).unwrap();
        assert_eq!(h2.version, NiftiVersion::Nifti2);
    }

    #[test]
    fn test_requires_nifti2() {
        let mut header = NiftiHeader::default();
        header.dim = [100, 100, 100, 1, 1, 1, 1];
        assert!(!header.requires_nifti2());

        header.dim[0] = 50000; // Exceeds i16::MAX (32767)
        assert!(header.requires_nifti2());
    }
}
