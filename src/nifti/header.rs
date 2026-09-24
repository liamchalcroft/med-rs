//! `NIfTI` header parsing, validation, and serialization.
//!
//! Supports NIfTI-1 (348-byte header) and NIfTI-2 (540-byte header), both the
//! single-file (`.nii`) and two-file (`.hdr`/`.img`) layouts, either byte
//! order, and header extensions. Every field defined by the standard survives a
//! load/save round trip; only the unused Analyze 7.5 legacy fields are dropped.
//!
//! Geometry follows the NIfTI-1 reference implementation (`nifti1_io.c`):
//! the sform is preferred over the qform, non-positive voxel sizes are treated
//! as 1 when building a qform matrix, and a matrix is converted back to a
//! quaternion through its closest orthogonal matrix (polar decomposition).

use crate::error::{Error, Result};
use byteorder::{BigEndian, ByteOrder, LittleEndian};

/// A 4x4 homogeneous affine matrix in row-major order.
pub type Affine = [[f64; 4]; 4];

/// `NIfTI` format version.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NiftiVersion {
    /// NIfTI-1: 348-byte header, 16-bit dimensions, 32-bit floats.
    #[default]
    Nifti1,
    /// NIfTI-2: 540-byte header, 64-bit dimensions, 64-bit floats.
    Nifti2,
}

impl NiftiVersion {
    /// Size of the fixed header in bytes.
    pub const fn header_size(self) -> usize {
        match self {
            Self::Nifti1 => 348,
            Self::Nifti2 => 540,
        }
    }
}

/// `NIfTI` datatype codes supported by medrs.
///
/// `Float16` (16384) and `BFloat16` (16385) are **medrs extensions**: they are
/// not part of the NIfTI standard, so files that use them can only be read by
/// medrs. Use them for caches and intermediate storage, not for interchange.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(i16)]
#[non_exhaustive]
pub enum DataType {
    /// Unsigned 8-bit integer (`DT_UINT8`).
    UInt8 = 2,
    /// Signed 16-bit integer (`DT_INT16`).
    Int16 = 4,
    /// Signed 32-bit integer (`DT_INT32`).
    Int32 = 8,
    /// 32-bit IEEE float (`DT_FLOAT32`).
    Float32 = 16,
    /// 64-bit IEEE float (`DT_FLOAT64`).
    Float64 = 64,
    /// Signed 8-bit integer (`DT_INT8`).
    Int8 = 256,
    /// Unsigned 16-bit integer (`DT_UINT16`).
    UInt16 = 512,
    /// Unsigned 32-bit integer (`DT_UINT32`).
    UInt32 = 768,
    /// Signed 64-bit integer (`DT_INT64`).
    Int64 = 1024,
    /// Unsigned 64-bit integer (`DT_UINT64`).
    UInt64 = 1280,
    /// 16-bit IEEE half-precision float (medrs extension, code 16384).
    Float16 = 16384,
    /// 16-bit brain float (medrs extension, code 16385).
    BFloat16 = 16385,
}

impl DataType {
    /// Parse a `NIfTI` datatype code.
    pub fn from_code(code: i16) -> Result<Self> {
        Ok(match code {
            2 => Self::UInt8,
            4 => Self::Int16,
            8 => Self::Int32,
            16 => Self::Float32,
            64 => Self::Float64,
            256 => Self::Int8,
            512 => Self::UInt16,
            768 => Self::UInt32,
            1024 => Self::Int64,
            1280 => Self::UInt64,
            16384 => Self::Float16,
            16385 => Self::BFloat16,
            _ => return Err(Error::UnsupportedDataType(code)),
        })
    }

    /// The `NIfTI` datatype code.
    pub const fn code(self) -> i16 {
        self as i16
    }

    /// Size of one element in bytes.
    pub const fn byte_size(self) -> usize {
        match self {
            Self::UInt8 | Self::Int8 => 1,
            Self::Int16 | Self::UInt16 | Self::Float16 | Self::BFloat16 => 2,
            Self::Int32 | Self::UInt32 | Self::Float32 => 4,
            Self::Int64 | Self::UInt64 | Self::Float64 => 8,
        }
    }

    /// Whether this is a floating-point type.
    pub const fn is_float(self) -> bool {
        matches!(
            self,
            Self::Float16 | Self::BFloat16 | Self::Float32 | Self::Float64
        )
    }

    /// Whether this is an integer type.
    pub const fn is_integer(self) -> bool {
        !self.is_float()
    }

    /// Name of the type, as in NumPy (`uint8`, `float32`, ...).
    pub const fn name(self) -> &'static str {
        match self {
            Self::UInt8 => "uint8",
            Self::Int8 => "int8",
            Self::Int16 => "int16",
            Self::UInt16 => "uint16",
            Self::Int32 => "int32",
            Self::UInt32 => "uint32",
            Self::Int64 => "int64",
            Self::UInt64 => "uint64",
            Self::Float16 => "float16",
            Self::BFloat16 => "bfloat16",
            Self::Float32 => "float32",
            Self::Float64 => "float64",
        }
    }
}

impl std::fmt::Display for DataType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

impl std::str::FromStr for DataType {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self> {
        ALL_TYPES
            .into_iter()
            .find(|t| t.name() == s)
            .ok_or_else(|| {
                let names: Vec<&str> = ALL_TYPES.iter().map(|t| t.name()).collect();
                Error::InvalidArgument(format!(
                    "unknown data type '{s}' (expected one of {})",
                    names.join(", ")
                ))
            })
    }
}

const ALL_TYPES: [DataType; 12] = [
    DataType::UInt8,
    DataType::Int8,
    DataType::UInt16,
    DataType::Int16,
    DataType::UInt32,
    DataType::Int32,
    DataType::UInt64,
    DataType::Int64,
    DataType::Float16,
    DataType::BFloat16,
    DataType::Float32,
    DataType::Float64,
];

/// Describe a datatype code medrs cannot load, for error messages.
pub(crate) fn describe_unsupported_code(code: i16) -> &'static str {
    match code {
        0 => "DT_UNKNOWN",
        1 => "DT_BINARY",
        32 => "DT_COMPLEX64",
        128 => "DT_RGB24",
        1536 => "DT_FLOAT128",
        1792 => "DT_COMPLEX128",
        2048 => "DT_COMPLEX256",
        2304 => "DT_RGBA32",
        _ => "an unknown code",
    }
}

/// Spatial units of the voxel sizes (low 3 bits of `xyzt_units`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum SpatialUnits {
    /// Units not specified.
    #[default]
    Unknown,
    /// Meters.
    Meter,
    /// Millimeters.
    Millimeter,
    /// Micrometers.
    Micrometer,
}

impl SpatialUnits {
    const fn from_code(code: u8) -> Self {
        match code & 0x07 {
            1 => Self::Meter,
            2 => Self::Millimeter,
            3 => Self::Micrometer,
            _ => Self::Unknown,
        }
    }

    const fn to_code(self) -> u8 {
        match self {
            Self::Unknown => 0,
            Self::Meter => 1,
            Self::Millimeter => 2,
            Self::Micrometer => 3,
        }
    }
}

/// Temporal units (bits 3..=5 of `xyzt_units`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum TemporalUnits {
    /// Units not specified.
    #[default]
    Unknown,
    /// Seconds.
    Second,
    /// Milliseconds.
    Millisecond,
    /// Microseconds.
    Microsecond,
    /// Hertz.
    Hertz,
    /// Parts per million.
    Ppm,
    /// Radians per second.
    RadPerSecond,
}

impl TemporalUnits {
    const fn from_code(code: u8) -> Self {
        match code & 0x38 {
            0x08 => Self::Second,
            0x10 => Self::Millisecond,
            0x18 => Self::Microsecond,
            0x20 => Self::Hertz,
            0x28 => Self::Ppm,
            0x30 => Self::RadPerSecond,
            _ => Self::Unknown,
        }
    }

    const fn to_code(self) -> u8 {
        match self {
            Self::Unknown => 0,
            Self::Second => 0x08,
            Self::Millisecond => 0x10,
            Self::Microsecond => 0x18,
            Self::Hertz => 0x20,
            Self::Ppm => 0x28,
            Self::RadPerSecond => 0x30,
        }
    }
}

/// A `NIfTI` header extension: an opaque payload tagged with an `ecode`
/// (for example 4 = AFNI, 6 = comment, 32 = CIFTI).
///
/// Extensions are preserved verbatim through load and save.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NiftiExtension {
    /// Extension code identifying the payload format.
    pub ecode: i32,
    /// Raw payload bytes (any trailing zero padding from the file is kept).
    pub data: Vec<u8>,
}

impl NiftiExtension {
    /// Create an extension from a code and payload.
    pub fn new(ecode: i32, data: impl Into<Vec<u8>>) -> Self {
        Self {
            ecode,
            data: data.into(),
        }
    }

    /// On-disk size including the 8-byte `esize`/`ecode` prefix, padded to 16.
    pub fn encoded_size(&self) -> usize {
        (self.data.len() + 8).div_ceil(16) * 16
    }
}

/// On-disk layout of a `NIfTI` dataset.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FileLayout {
    /// Header, extensions, and voxels in one file (`.nii`, magic `n+1`/`n+2`).
    Single,
    /// Header (and extensions) in `.hdr`, voxels in `.img` (magic `ni1`/`ni2`).
    Pair,
}

/// A complete `NIfTI` header.
///
/// Internally every field is stored at NIfTI-2 width (64-bit dimensions and
/// floats); NIfTI-1 files are widened on load and narrowed on save. A NIfTI-1
/// header whose dimensions no longer fit in 16 bits is written as NIfTI-2.
#[derive(Debug, Clone, PartialEq)]
pub struct NiftiHeader {
    /// Format version used when this header was read (and preferred on save).
    pub version: NiftiVersion,
    /// Number of dimensions (1..=7).
    pub ndim: u8,
    /// Size along each dimension (`dim[1..=7]` in the file).
    pub dim: [i64; 7],
    /// Voxel datatype.
    pub datatype: DataType,
    /// `pixdim[0]` is `qfac` (±1); `pixdim[1..=7]` are the grid spacings.
    pub pixdim: [f64; 8],
    /// Byte offset of the voxel data. Recomputed on save.
    pub vox_offset: i64,
    /// Data scaling slope (`0` means "no scaling", as in the standard).
    pub scl_slope: f64,
    /// Data scaling intercept.
    pub scl_inter: f64,
    /// Spatial units.
    pub spatial_units: SpatialUnits,
    /// Temporal units.
    pub temporal_units: TemporalUnits,
    /// Statistical or other intent code.
    pub intent_code: i32,
    /// Intent parameters `intent_p1..=intent_p3`.
    pub intent_p: [f64; 3],
    /// Intent name (up to 16 bytes).
    pub intent_name: String,
    /// MRI slice ordering: frequency, phase, and slice dimensions packed in 2-bit fields.
    pub dim_info: u8,
    /// Slice timing order code.
    pub slice_code: u8,
    /// First slice index.
    pub slice_start: i64,
    /// Last slice index.
    pub slice_end: i64,
    /// Time taken to acquire one slice.
    pub slice_duration: f64,
    /// Time axis offset.
    pub toffset: f64,
    /// Display range maximum.
    pub cal_max: f64,
    /// Display range minimum.
    pub cal_min: f64,
    /// Free-form description (up to 80 bytes).
    pub descrip: String,
    /// Auxiliary filename (up to 24 bytes).
    pub aux_file: String,
    /// qform transform code (`NIFTI_XFORM_*`).
    pub qform_code: i32,
    /// sform transform code (`NIFTI_XFORM_*`).
    pub sform_code: i32,
    /// qform quaternion parameters `b`, `c`, `d`.
    pub quatern: [f64; 3],
    /// qform translation.
    pub qoffset: [f64; 3],
    /// First row of the sform matrix.
    pub srow_x: [f64; 4],
    /// Second row of the sform matrix.
    pub srow_y: [f64; 4],
    /// Third row of the sform matrix.
    pub srow_z: [f64; 4],
    /// Header extensions, preserved verbatim.
    pub extensions: Vec<NiftiExtension>,
    /// Byte order of the file this header was read from (writes are always
    /// little-endian).
    pub(crate) little_endian: bool,
}

impl Default for NiftiHeader {
    fn default() -> Self {
        Self {
            version: NiftiVersion::Nifti1,
            ndim: 3,
            dim: [1; 7],
            datatype: DataType::Float32,
            pixdim: [1.0; 8],
            vox_offset: 352,
            scl_slope: 1.0,
            scl_inter: 0.0,
            spatial_units: SpatialUnits::Millimeter,
            temporal_units: TemporalUnits::Unknown,
            intent_code: 0,
            intent_p: [0.0; 3],
            intent_name: String::new(),
            dim_info: 0,
            slice_code: 0,
            slice_start: 0,
            slice_end: 0,
            slice_duration: 0.0,
            toffset: 0.0,
            cal_max: 0.0,
            cal_min: 0.0,
            descrip: String::new(),
            aux_file: String::new(),
            qform_code: 1,
            sform_code: 1,
            quatern: [0.0; 3],
            qoffset: [0.0; 3],
            srow_x: [1.0, 0.0, 0.0, 0.0],
            srow_y: [0.0, 1.0, 0.0, 0.0],
            srow_z: [0.0, 0.0, 1.0, 0.0],
            extensions: Vec::new(),
            little_endian: true,
        }
    }
}

/// NIfTI-1 field offsets.
mod v1 {
    pub const DIM_INFO: usize = 39;
    pub const DIM: usize = 40;
    pub const INTENT_P1: usize = 56;
    pub const INTENT_CODE: usize = 68;
    pub const DATATYPE: usize = 70;
    pub const BITPIX: usize = 72;
    pub const SLICE_START: usize = 74;
    pub const PIXDIM: usize = 76;
    pub const VOX_OFFSET: usize = 108;
    pub const SCL_SLOPE: usize = 112;
    pub const SCL_INTER: usize = 116;
    pub const SLICE_END: usize = 120;
    pub const SLICE_CODE: usize = 122;
    pub const XYZT_UNITS: usize = 123;
    pub const CAL_MAX: usize = 124;
    pub const CAL_MIN: usize = 128;
    pub const SLICE_DURATION: usize = 132;
    pub const TOFFSET: usize = 136;
    pub const DESCRIP: usize = 148;
    pub const AUX_FILE: usize = 228;
    pub const QFORM_CODE: usize = 252;
    pub const SFORM_CODE: usize = 254;
    pub const QUATERN_B: usize = 256;
    pub const QOFFSET_X: usize = 268;
    pub const SROW_X: usize = 280;
    pub const INTENT_NAME: usize = 328;
    pub const MAGIC: usize = 344;
}

/// NIfTI-2 field offsets.
mod v2 {
    pub const MAGIC: usize = 4;
    pub const DATATYPE: usize = 12;
    pub const BITPIX: usize = 14;
    pub const DIM: usize = 16;
    pub const INTENT_P1: usize = 80;
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
    pub const QOFFSET_X: usize = 376;
    pub const SROW_X: usize = 400;
    pub const SLICE_CODE: usize = 496;
    pub const XYZT_UNITS: usize = 500;
    pub const INTENT_CODE: usize = 504;
    pub const INTENT_NAME: usize = 508;
    pub const DIM_INFO: usize = 524;
}

const MAGIC_N1: &[u8; 4] = b"n+1\0";
const MAGIC_NI1: &[u8; 4] = b"ni1\0";
const MAGIC_N2: &[u8; 8] = b"n+2\0\r\n\x1a\n";
const MAGIC_NI2: &[u8; 8] = b"ni2\0\r\n\x1a\n";

/// Largest total size accepted for the extension block, as a sanity bound on
/// attacker-controlled `vox_offset` values.
const MAX_EXTENSION_BYTES: usize = 256 * 1024 * 1024;

/// Read a NUL-terminated fixed-width string field.
fn read_str(bytes: &[u8]) -> String {
    let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
    String::from_utf8_lossy(&bytes[..end]).into_owned()
}

/// Write a string into a fixed-width field, truncating on a UTF-8 boundary.
fn write_str(dst: &mut [u8], s: &str) {
    let mut len = s.len().min(dst.len());
    while len > 0 && !s.is_char_boundary(len) {
        len -= 1;
    }
    dst[..len].copy_from_slice(&s.as_bytes()[..len]);
}

/// Detect version, byte order, and layout from the start of a header.
pub(crate) fn sniff(bytes: &[u8]) -> Result<(NiftiVersion, bool)> {
    if bytes.len() < 4 {
        return Err(Error::InvalidFileFormat(
            "file too short to contain a NIfTI header".into(),
        ));
    }
    let le = LittleEndian::read_i32(&bytes[0..4]);
    let be = BigEndian::read_i32(&bytes[0..4]);
    match (le, be) {
        (348, _) => Ok((NiftiVersion::Nifti1, true)),
        (_, 348) => Ok((NiftiVersion::Nifti1, false)),
        (540, _) => Ok((NiftiVersion::Nifti2, true)),
        (_, 540) => Ok((NiftiVersion::Nifti2, false)),
        _ => {
            if bytes.starts_with(&[0x1f, 0x8b]) {
                return Err(Error::InvalidFileFormat(
                    "file is gzip-compressed but was read as uncompressed NIfTI".into(),
                ));
            }
            Err(Error::InvalidMagic([
                bytes[0], bytes[1], bytes[2], bytes[3],
            ]))
        }
    }
}

impl NiftiHeader {
    /// Size of a NIfTI-1 header in bytes.
    pub const SIZE_V1: usize = 348;
    /// Size of a NIfTI-2 header in bytes.
    pub const SIZE_V2: usize = 540;

    /// Size of the fixed header for this header's version.
    pub fn header_size(&self) -> usize {
        self.version.header_size()
    }

    /// Parse a header (and any extensions contained in `bytes`).
    ///
    /// `bytes` must start at the beginning of the file. Extensions are parsed
    /// when present and fully contained in `bytes`.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        Self::parse(bytes).map(|(header, _)| header)
    }

    /// Parse a header, also reporting the file layout implied by its magic.
    pub(crate) fn parse(bytes: &[u8]) -> Result<(Self, FileLayout)> {
        let (version, little_endian) = sniff(bytes)?;
        let size = version.header_size();
        if bytes.len() < size {
            return Err(Error::InvalidFileFormat(format!(
                "header truncated: {} bytes available, {size} required for {version:?}",
                bytes.len()
            )));
        }
        let (mut header, layout) = match (version, little_endian) {
            (NiftiVersion::Nifti1, true) => Self::parse_v1::<LittleEndian>(bytes)?,
            (NiftiVersion::Nifti1, false) => Self::parse_v1::<BigEndian>(bytes)?,
            (NiftiVersion::Nifti2, true) => Self::parse_v2::<LittleEndian>(bytes)?,
            (NiftiVersion::Nifti2, false) => Self::parse_v2::<BigEndian>(bytes)?,
        };
        header.little_endian = little_endian;
        header.normalize_vox_offset(layout)?;
        header.validate()?;
        header.extensions = parse_extensions(bytes, &header, layout);
        Ok((header, layout))
    }

    fn parse_v1<E: ByteOrder>(b: &[u8]) -> Result<(Self, FileLayout)> {
        use v1::{
            AUX_FILE, CAL_MAX, CAL_MIN, DATATYPE, DESCRIP, DIM, DIM_INFO, INTENT_CODE, INTENT_NAME,
            INTENT_P1, MAGIC, PIXDIM, QFORM_CODE, QOFFSET_X, QUATERN_B, SCL_INTER, SCL_SLOPE,
            SFORM_CODE, SLICE_CODE, SLICE_DURATION, SLICE_END, SLICE_START, SROW_X, TOFFSET,
            VOX_OFFSET, XYZT_UNITS,
        };
        let layout = match &b[MAGIC..MAGIC + 4] {
            m if m == MAGIC_N1 => FileLayout::Single,
            m if m == MAGIC_NI1 => FileLayout::Pair,
            m => return Err(bad_magic(m)),
        };

        let i16_at = |o: usize| E::read_i16(&b[o..o + 2]);
        let f32_at = |o: usize| f64::from(E::read_f32(&b[o..o + 4]));

        let ndim = parse_ndim(i64::from(i16_at(DIM)))?;
        let mut dim = [1i64; 7];
        for (i, d) in dim.iter_mut().enumerate() {
            *d = i64::from(i16_at(DIM + 2 + 2 * i));
        }
        // bitpix is redundant with datatype; like nibabel, trust datatype.
        let datatype = parse_datatype(i16_at(DATATYPE))?;

        let mut pixdim = [0.0; 8];
        for (i, p) in pixdim.iter_mut().enumerate() {
            *p = f32_at(PIXDIM + 4 * i);
        }

        let vox_offset = f32_at(VOX_OFFSET);
        if !vox_offset.is_finite() || vox_offset.fract() != 0.0 || vox_offset < 0.0 {
            return Err(Error::InvalidFileFormat(format!(
                "vox_offset must be a non-negative integer, got {vox_offset}"
            )));
        }

        let units = b[XYZT_UNITS];
        let srow =
            |row: usize| -> [f64; 4] { std::array::from_fn(|j| f32_at(SROW_X + 16 * row + 4 * j)) };

        Ok((
            Self {
                version: NiftiVersion::Nifti1,
                ndim,
                dim,
                datatype,
                pixdim,
                vox_offset: vox_offset as i64,
                scl_slope: f32_at(SCL_SLOPE),
                scl_inter: f32_at(SCL_INTER),
                spatial_units: SpatialUnits::from_code(units),
                temporal_units: TemporalUnits::from_code(units),
                intent_code: i32::from(i16_at(INTENT_CODE)),
                intent_p: std::array::from_fn(|i| f32_at(INTENT_P1 + 4 * i)),
                intent_name: read_str(&b[INTENT_NAME..INTENT_NAME + 16]),
                dim_info: b[DIM_INFO],
                slice_code: b[SLICE_CODE],
                slice_start: i64::from(i16_at(SLICE_START)),
                slice_end: i64::from(i16_at(SLICE_END)),
                slice_duration: f32_at(SLICE_DURATION),
                toffset: f32_at(TOFFSET),
                cal_max: f32_at(CAL_MAX),
                cal_min: f32_at(CAL_MIN),
                descrip: read_str(&b[DESCRIP..DESCRIP + 80]),
                aux_file: read_str(&b[AUX_FILE..AUX_FILE + 24]),
                qform_code: i32::from(i16_at(QFORM_CODE)),
                sform_code: i32::from(i16_at(SFORM_CODE)),
                quatern: std::array::from_fn(|i| f32_at(QUATERN_B + 4 * i)),
                qoffset: std::array::from_fn(|i| f32_at(QOFFSET_X + 4 * i)),
                srow_x: srow(0),
                srow_y: srow(1),
                srow_z: srow(2),
                extensions: Vec::new(),
                little_endian: true,
            },
            layout,
        ))
    }

    fn parse_v2<E: ByteOrder>(b: &[u8]) -> Result<(Self, FileLayout)> {
        use v2::{
            AUX_FILE, CAL_MAX, CAL_MIN, DATATYPE, DESCRIP, DIM, DIM_INFO, INTENT_CODE, INTENT_NAME,
            INTENT_P1, MAGIC, PIXDIM, QFORM_CODE, QOFFSET_X, QUATERN_B, SCL_INTER, SCL_SLOPE,
            SFORM_CODE, SLICE_CODE, SLICE_DURATION, SLICE_END, SLICE_START, SROW_X, TOFFSET,
            VOX_OFFSET, XYZT_UNITS,
        };
        let layout = match &b[MAGIC..MAGIC + 8] {
            m if m == MAGIC_N2 => FileLayout::Single,
            m if m == MAGIC_NI2 => FileLayout::Pair,
            m => return Err(bad_magic(m)),
        };

        let i32_at = |o: usize| E::read_i32(&b[o..o + 4]);
        let i64_at = |o: usize| E::read_i64(&b[o..o + 8]);
        let f64_at = |o: usize| E::read_f64(&b[o..o + 8]);

        let ndim = parse_ndim(i64_at(DIM))?;
        let mut dim = [1i64; 7];
        for (i, d) in dim.iter_mut().enumerate() {
            *d = i64_at(DIM + 8 + 8 * i);
        }
        // bitpix is redundant with datatype; like nibabel, trust datatype.
        let datatype = parse_datatype(E::read_i16(&b[DATATYPE..DATATYPE + 2]))?;

        let units = i32_at(XYZT_UNITS) as u8;
        let srow =
            |row: usize| -> [f64; 4] { std::array::from_fn(|j| f64_at(SROW_X + 32 * row + 8 * j)) };
        let vox_offset = i64_at(VOX_OFFSET);
        if vox_offset < 0 {
            return Err(Error::InvalidFileFormat(format!(
                "vox_offset must be non-negative, got {vox_offset}"
            )));
        }

        Ok((
            Self {
                version: NiftiVersion::Nifti2,
                ndim,
                dim,
                datatype,
                pixdim: std::array::from_fn(|i| f64_at(PIXDIM + 8 * i)),
                vox_offset,
                scl_slope: f64_at(SCL_SLOPE),
                scl_inter: f64_at(SCL_INTER),
                spatial_units: SpatialUnits::from_code(units),
                temporal_units: TemporalUnits::from_code(units),
                intent_code: i32_at(INTENT_CODE),
                intent_p: std::array::from_fn(|i| f64_at(INTENT_P1 + 8 * i)),
                intent_name: read_str(&b[INTENT_NAME..INTENT_NAME + 16]),
                dim_info: b[DIM_INFO],
                slice_code: i32_at(SLICE_CODE) as u8,
                slice_start: i64_at(SLICE_START),
                slice_end: i64_at(SLICE_END),
                slice_duration: f64_at(SLICE_DURATION),
                toffset: f64_at(TOFFSET),
                cal_max: f64_at(CAL_MAX),
                cal_min: f64_at(CAL_MIN),
                descrip: read_str(&b[DESCRIP..DESCRIP + 80]),
                aux_file: read_str(&b[AUX_FILE..AUX_FILE + 24]),
                qform_code: i32_at(QFORM_CODE),
                sform_code: i32_at(SFORM_CODE),
                quatern: std::array::from_fn(|i| f64_at(QUATERN_B + 8 * i)),
                qoffset: std::array::from_fn(|i| f64_at(QOFFSET_X + 8 * i)),
                srow_x: srow(0),
                srow_y: srow(1),
                srow_z: srow(2),
                extensions: Vec::new(),
                little_endian: true,
            },
            layout,
        ))
    }

    /// Apply the reference implementation's leniency for `vox_offset`.
    ///
    /// A single file must place its voxels after the header; a zero offset
    /// (written by some old tools) is interpreted as "immediately after the
    /// header and extension flag".
    fn normalize_vox_offset(&mut self, layout: FileLayout) -> Result<()> {
        if layout == FileLayout::Single {
            let size = self.header_size() as i64;
            if self.vox_offset == 0 {
                self.vox_offset = size + 4;
            } else if self.vox_offset < size {
                return Err(Error::InvalidFileFormat(format!(
                    "vox_offset {} lies inside the {size}-byte header",
                    self.vox_offset
                )));
            }
        }
        Ok(())
    }

    /// Check the invariants required to safely address the voxel data.
    ///
    /// Only properties that matter for memory safety and shape consistency are
    /// enforced; geometry fields (pixdim, quaternions, ...) are accepted as-is
    /// so that imperfect real-world files still load.
    pub fn validate(&self) -> Result<()> {
        if !(1..=7).contains(&self.ndim) {
            return Err(Error::InvalidDimensions(format!(
                "ndim must be in 1..=7, got {}",
                self.ndim
            )));
        }
        let mut voxels: usize = 1;
        for (i, &d) in self.dim[..self.ndim as usize].iter().enumerate() {
            if d < 1 {
                return Err(Error::InvalidDimensions(format!(
                    "dim[{}] must be >= 1, got {d}",
                    i + 1
                )));
            }
            let d = usize::try_from(d).map_err(|_| {
                Error::InvalidDimensions(format!("dim[{}] = {d} is too large", i + 1))
            })?;
            voxels = voxels
                .checked_mul(d)
                .ok_or_else(|| Error::InvalidDimensions("voxel count overflows".into()))?;
        }
        let bytes = voxels
            .checked_mul(self.datatype.byte_size())
            .ok_or_else(|| Error::InvalidDimensions("data size overflows".into()))?;
        if self.vox_offset < 0 {
            return Err(Error::InvalidFileFormat(format!(
                "vox_offset must be non-negative, got {}",
                self.vox_offset
            )));
        }
        usize::try_from(self.vox_offset)
            .ok()
            .and_then(|o| o.checked_add(bytes))
            .ok_or_else(|| Error::InvalidDimensions("vox_offset + data size overflows".into()))?;
        Ok(())
    }

    // ---------------------------------------------------------------------
    // Serialization
    // ---------------------------------------------------------------------

    /// Serialize the fixed-size header (348 or 540 bytes, little-endian) as a
    /// single-file (`n+1` / `n+2`) header with the current `vox_offset`.
    ///
    /// Returns an error for a NIfTI-1 header whose dimensions exceed the
    /// 16-bit limit; [`save`](crate::nifti::save) promotes such headers to
    /// NIfTI-2 automatically.
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        self.encode_fixed(FileLayout::Single)
    }

    /// Whether any field is out of range for NIfTI-1.
    pub fn requires_nifti2(&self) -> bool {
        let i16_range = i64::from(i16::MIN)..=i64::from(i16::MAX);
        self.dim.iter().any(|&d| d > i64::from(i16::MAX))
            || !i16_range.contains(&self.slice_start)
            || !i16_range.contains(&self.slice_end)
            || !i16_range.contains(&i64::from(self.intent_code))
            || !i16_range.contains(&i64::from(self.qform_code))
            || !i16_range.contains(&i64::from(self.sform_code))
    }

    /// Prepare a copy of this header for writing in `layout`: promote to
    /// NIfTI-2 if required and recompute `vox_offset` from the extensions.
    pub(crate) fn prepared_for_write(&self, layout: FileLayout) -> Self {
        let mut h = self.clone();
        if h.version == NiftiVersion::Nifti1 && h.requires_nifti2() {
            h.version = NiftiVersion::Nifti2;
        }
        h.little_endian = true;
        h.vox_offset = match layout {
            FileLayout::Single => {
                let ext: usize = h.extensions.iter().map(NiftiExtension::encoded_size).sum();
                (h.header_size() + 4 + ext).div_ceil(16) as i64 * 16
            }
            FileLayout::Pair => 0,
        };
        h
    }

    /// Encode header, extension flag, extensions, and padding exactly as they
    /// appear on disk. The header must come from [`Self::prepared_for_write`].
    pub(crate) fn encode(&self, layout: FileLayout) -> Result<Vec<u8>> {
        let mut out = self.encode_fixed(layout)?;
        let flag = u8::from(!self.extensions.is_empty());
        out.extend_from_slice(&[flag, 0, 0, 0]);
        for ext in &self.extensions {
            let size = ext.encoded_size();
            let esize = i32::try_from(size).map_err(|_| {
                Error::InvalidFileFormat(format!("extension of {size} bytes is too large"))
            })?;
            out.extend_from_slice(&esize.to_le_bytes());
            out.extend_from_slice(&ext.ecode.to_le_bytes());
            out.extend_from_slice(&ext.data);
            out.resize(out.len() + size - 8 - ext.data.len(), 0);
        }
        if layout == FileLayout::Single {
            let target = usize::try_from(self.vox_offset).unwrap_or(0);
            if out.len() > target {
                return Err(Error::InvalidFileFormat(format!(
                    "header and extensions ({} bytes) exceed vox_offset {target}",
                    out.len()
                )));
            }
            out.resize(target, 0);
        }
        Ok(out)
    }

    fn encode_fixed(&self, layout: FileLayout) -> Result<Vec<u8>> {
        match self.version {
            NiftiVersion::Nifti1 => self.encode_v1(layout),
            NiftiVersion::Nifti2 => Ok(self.encode_v2(layout)),
        }
    }

    fn encode_v1(&self, layout: FileLayout) -> Result<Vec<u8>> {
        use v1::{
            AUX_FILE, BITPIX, CAL_MAX, CAL_MIN, DATATYPE, DESCRIP, DIM, DIM_INFO, INTENT_CODE,
            INTENT_NAME, INTENT_P1, MAGIC, PIXDIM, QFORM_CODE, QOFFSET_X, QUATERN_B, SCL_INTER,
            SCL_SLOPE, SFORM_CODE, SLICE_CODE, SLICE_DURATION, SLICE_END, SLICE_START, SROW_X,
            TOFFSET, VOX_OFFSET, XYZT_UNITS,
        };
        type E = LittleEndian;
        if self.requires_nifti2() {
            return Err(Error::InvalidDimensions(format!(
                "header fields exceed NIfTI-1 limits (dims {:?}); write as NIfTI-2",
                &self.dim[..self.ndim as usize]
            )));
        }
        let mut b = vec![0u8; Self::SIZE_V1];
        let put_f32 = |b: &mut [u8], o: usize, v: f64| E::write_f32(&mut b[o..o + 4], v as f32);

        E::write_i32(&mut b[0..4], Self::SIZE_V1 as i32);
        b[DIM_INFO] = self.dim_info;
        E::write_i16(&mut b[DIM..DIM + 2], i16::from(self.ndim));
        for i in 0..7 {
            let d = if i < self.ndim as usize {
                self.dim[i]
            } else {
                1
            };
            E::write_i16(&mut b[DIM + 2 + 2 * i..DIM + 4 + 2 * i], d as i16);
        }
        for (i, &p) in self.intent_p.iter().enumerate() {
            put_f32(&mut b, INTENT_P1 + 4 * i, p);
        }
        E::write_i16(
            &mut b[INTENT_CODE..INTENT_CODE + 2],
            self.intent_code as i16,
        );
        E::write_i16(&mut b[DATATYPE..DATATYPE + 2], self.datatype.code());
        E::write_i16(
            &mut b[BITPIX..BITPIX + 2],
            (self.datatype.byte_size() * 8) as i16,
        );
        E::write_i16(
            &mut b[SLICE_START..SLICE_START + 2],
            self.slice_start as i16,
        );
        for (i, &p) in self.pixdim.iter().enumerate() {
            put_f32(&mut b, PIXDIM + 4 * i, p);
        }
        let vox_offset = match layout {
            FileLayout::Single => self.vox_offset,
            FileLayout::Pair => 0,
        };
        put_f32(&mut b, VOX_OFFSET, vox_offset as f64);
        put_f32(&mut b, SCL_SLOPE, self.scl_slope);
        put_f32(&mut b, SCL_INTER, self.scl_inter);
        E::write_i16(&mut b[SLICE_END..SLICE_END + 2], self.slice_end as i16);
        b[SLICE_CODE] = self.slice_code;
        b[XYZT_UNITS] = self.spatial_units.to_code() | self.temporal_units.to_code();
        put_f32(&mut b, CAL_MAX, self.cal_max);
        put_f32(&mut b, CAL_MIN, self.cal_min);
        put_f32(&mut b, SLICE_DURATION, self.slice_duration);
        put_f32(&mut b, TOFFSET, self.toffset);
        write_str(&mut b[DESCRIP..DESCRIP + 80], &self.descrip);
        write_str(&mut b[AUX_FILE..AUX_FILE + 24], &self.aux_file);
        E::write_i16(&mut b[QFORM_CODE..QFORM_CODE + 2], self.qform_code as i16);
        E::write_i16(&mut b[SFORM_CODE..SFORM_CODE + 2], self.sform_code as i16);
        for i in 0..3 {
            put_f32(&mut b, QUATERN_B + 4 * i, self.quatern[i]);
            put_f32(&mut b, QOFFSET_X + 4 * i, self.qoffset[i]);
        }
        for (row, values) in [self.srow_x, self.srow_y, self.srow_z].iter().enumerate() {
            for (j, &v) in values.iter().enumerate() {
                put_f32(&mut b, SROW_X + 16 * row + 4 * j, v);
            }
        }
        write_str(&mut b[INTENT_NAME..INTENT_NAME + 16], &self.intent_name);
        b[MAGIC..MAGIC + 4].copy_from_slice(match layout {
            FileLayout::Single => MAGIC_N1,
            FileLayout::Pair => MAGIC_NI1,
        });
        Ok(b)
    }

    fn encode_v2(&self, layout: FileLayout) -> Vec<u8> {
        use v2::{
            AUX_FILE, BITPIX, CAL_MAX, CAL_MIN, DATATYPE, DESCRIP, DIM, DIM_INFO, INTENT_CODE,
            INTENT_NAME, INTENT_P1, MAGIC, PIXDIM, QFORM_CODE, QOFFSET_X, QUATERN_B, SCL_INTER,
            SCL_SLOPE, SFORM_CODE, SLICE_CODE, SLICE_DURATION, SLICE_END, SLICE_START, SROW_X,
            TOFFSET, VOX_OFFSET, XYZT_UNITS,
        };
        type E = LittleEndian;
        let mut b = vec![0u8; Self::SIZE_V2];
        let put_f64 = |b: &mut [u8], o: usize, v: f64| E::write_f64(&mut b[o..o + 8], v);
        let put_i64 = |b: &mut [u8], o: usize, v: i64| E::write_i64(&mut b[o..o + 8], v);

        E::write_i32(&mut b[0..4], Self::SIZE_V2 as i32);
        b[MAGIC..MAGIC + 8].copy_from_slice(match layout {
            FileLayout::Single => MAGIC_N2,
            FileLayout::Pair => MAGIC_NI2,
        });
        E::write_i16(&mut b[DATATYPE..DATATYPE + 2], self.datatype.code());
        E::write_i16(
            &mut b[BITPIX..BITPIX + 2],
            (self.datatype.byte_size() * 8) as i16,
        );
        put_i64(&mut b, DIM, i64::from(self.ndim));
        for i in 0..7 {
            let d = if i < self.ndim as usize {
                self.dim[i]
            } else {
                1
            };
            put_i64(&mut b, DIM + 8 + 8 * i, d);
        }
        for (i, &p) in self.intent_p.iter().enumerate() {
            put_f64(&mut b, INTENT_P1 + 8 * i, p);
        }
        for (i, &p) in self.pixdim.iter().enumerate() {
            put_f64(&mut b, PIXDIM + 8 * i, p);
        }
        let vox_offset = match layout {
            FileLayout::Single => self.vox_offset,
            FileLayout::Pair => 0,
        };
        put_i64(&mut b, VOX_OFFSET, vox_offset);
        put_f64(&mut b, SCL_SLOPE, self.scl_slope);
        put_f64(&mut b, SCL_INTER, self.scl_inter);
        put_f64(&mut b, CAL_MAX, self.cal_max);
        put_f64(&mut b, CAL_MIN, self.cal_min);
        put_f64(&mut b, SLICE_DURATION, self.slice_duration);
        put_f64(&mut b, TOFFSET, self.toffset);
        put_i64(&mut b, SLICE_START, self.slice_start);
        put_i64(&mut b, SLICE_END, self.slice_end);
        write_str(&mut b[DESCRIP..DESCRIP + 80], &self.descrip);
        write_str(&mut b[AUX_FILE..AUX_FILE + 24], &self.aux_file);
        E::write_i32(&mut b[QFORM_CODE..QFORM_CODE + 4], self.qform_code);
        E::write_i32(&mut b[SFORM_CODE..SFORM_CODE + 4], self.sform_code);
        for i in 0..3 {
            put_f64(&mut b, QUATERN_B + 8 * i, self.quatern[i]);
            put_f64(&mut b, QOFFSET_X + 8 * i, self.qoffset[i]);
        }
        for (row, values) in [self.srow_x, self.srow_y, self.srow_z].iter().enumerate() {
            for (j, &v) in values.iter().enumerate() {
                put_f64(&mut b, SROW_X + 32 * row + 8 * j, v);
            }
        }
        E::write_i32(
            &mut b[SLICE_CODE..SLICE_CODE + 4],
            i32::from(self.slice_code),
        );
        E::write_i32(
            &mut b[XYZT_UNITS..XYZT_UNITS + 4],
            i32::from(self.spatial_units.to_code() | self.temporal_units.to_code()),
        );
        E::write_i32(&mut b[INTENT_CODE..INTENT_CODE + 4], self.intent_code);
        write_str(&mut b[INTENT_NAME..INTENT_NAME + 16], &self.intent_name);
        b[DIM_INFO] = self.dim_info;
        b
    }

    // ---------------------------------------------------------------------
    // Shape and scaling
    // ---------------------------------------------------------------------

    /// Image shape (`dim[1..=ndim]`).
    pub fn shape(&self) -> Vec<usize> {
        self.dim[..self.ndim as usize]
            .iter()
            .map(|&d| usize::try_from(d).unwrap_or(0))
            .collect()
    }

    /// Total number of voxels.
    pub fn num_voxels(&self) -> usize {
        self.shape().iter().product()
    }

    /// Size of the voxel data in bytes.
    pub fn data_size(&self) -> usize {
        self.num_voxels() * self.datatype.byte_size()
    }

    /// Whether the file was stored little-endian.
    pub fn is_little_endian(&self) -> bool {
        self.little_endian
    }

    /// Effective scaling `(slope, intercept)`: a slope of 0 (or a non-finite
    /// slope) means "no scaling", as specified by the standard.
    pub fn scaling(&self) -> (f64, f64) {
        if self.scl_slope == 0.0 || !self.scl_slope.is_finite() {
            (1.0, 0.0)
        } else {
            let inter = if self.scl_inter.is_finite() {
                self.scl_inter
            } else {
                0.0
            };
            (self.scl_slope, inter)
        }
    }

    /// Whether reading the data applies a non-identity scaling.
    pub fn has_scaling(&self) -> bool {
        self.scaling() != (1.0, 0.0)
    }

    // ---------------------------------------------------------------------
    // Geometry
    // ---------------------------------------------------------------------

    /// The voxel-to-world affine used for this image: the sform when
    /// `sform_code > 0`, else the qform when `qform_code > 0`, else a diagonal
    /// matrix of the voxel sizes.
    pub fn affine(&self) -> Affine {
        if self.sform_code > 0 {
            self.sform_matrix()
        } else if self.qform_code > 0 {
            self.qform_matrix()
        } else {
            self.fallback_matrix()
        }
    }

    /// The sform matrix, if `sform_code > 0`.
    pub fn sform(&self) -> Option<Affine> {
        (self.sform_code > 0).then(|| self.sform_matrix())
    }

    /// The qform matrix, if `qform_code > 0`.
    pub fn qform(&self) -> Option<Affine> {
        (self.qform_code > 0).then(|| self.qform_matrix())
    }

    fn sform_matrix(&self) -> Affine {
        [self.srow_x, self.srow_y, self.srow_z, [0.0, 0.0, 0.0, 1.0]]
    }

    /// Voxel sizes along the first three axes as used by the qform: values
    /// that are not positive are replaced by 1, following `nifti1_io.c`.
    fn qform_zooms(&self) -> [f64; 3] {
        std::array::from_fn(|i| {
            let p = self.pixdim[i + 1];
            if p > 0.0 && p.is_finite() {
                p
            } else {
                1.0
            }
        })
    }

    fn fallback_matrix(&self) -> Affine {
        let [x, y, z] = self.qform_zooms();
        [
            [x, 0.0, 0.0, 0.0],
            [0.0, y, 0.0, 0.0],
            [0.0, 0.0, z, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    }

    /// Build the qform matrix from the quaternion (`nifti_quatern_to_mat44`).
    fn qform_matrix(&self) -> Affine {
        let [mut b, mut c, mut d] = self.quatern;
        let mut a = 1.0 - (b * b + c * c + d * d);
        if a < 1e-7 {
            // 180-degree rotation: renormalise (b, c, d) and set a = 0.
            let norm = (b * b + c * c + d * d).sqrt();
            if norm > 0.0 {
                b /= norm;
                c /= norm;
                d /= norm;
            }
            a = 0.0;
        } else {
            a = a.sqrt();
        }
        let [xd, yd, mut zd] = self.qform_zooms();
        if self.pixdim[0] < 0.0 {
            zd = -zd;
        }
        [
            [
                (a * a + b * b - c * c - d * d) * xd,
                2.0 * (b * c - a * d) * yd,
                2.0 * (b * d + a * c) * zd,
                self.qoffset[0],
            ],
            [
                2.0 * (b * c + a * d) * xd,
                (a * a + c * c - b * b - d * d) * yd,
                2.0 * (c * d - a * b) * zd,
                self.qoffset[1],
            ],
            [
                2.0 * (b * d - a * c) * xd,
                2.0 * (c * d + a * b) * yd,
                (a * a + d * d - c * c - b * b) * zd,
                self.qoffset[2],
            ],
            [0.0, 0.0, 0.0, 1.0],
        ]
    }

    /// Set both the sform and the qform to `affine`.
    ///
    /// Existing positive `sform_code`/`qform_code` values are kept; a code of
    /// 0 becomes 1 (`NIFTI_XFORM_SCANNER_ANAT`). The qform stores the closest
    /// rigid-plus-scaling approximation of `affine`, and `pixdim[0..=3]` are
    /// updated to match it.
    pub fn set_affine(&mut self, affine: Affine) {
        if self.sform_code <= 0 {
            self.sform_code = 1;
        }
        if self.qform_code <= 0 {
            self.qform_code = 1;
        }
        self.set_sform_matrix(&affine);
        self.set_qform_matrix(&affine);
    }

    /// Set both transforms and their codes explicitly.
    pub fn set_affine_with_codes(&mut self, affine: Affine, sform_code: i32, qform_code: i32) {
        self.set_sform_matrix(&affine);
        self.set_qform_matrix(&affine);
        self.sform_code = sform_code;
        self.qform_code = qform_code;
    }

    fn set_sform_matrix(&mut self, affine: &Affine) {
        self.srow_x = affine[0];
        self.srow_y = affine[1];
        self.srow_z = affine[2];
    }

    /// Update the geometry after a voxel-space transform.
    ///
    /// `voxel_map` maps *new* voxel indices to *old* voxel indices
    /// (homogeneous 4x4, acting on the first three axes). Both the sform and
    /// the qform are updated so that every voxel keeps its world position and
    /// both transform codes are preserved. An image with neither transform
    /// gains an sform and qform (code 1) describing the transformed grid.
    pub fn transform_voxels(&mut self, voxel_map: &Affine) {
        let has_sform = self.sform_code > 0;
        let has_qform = self.qform_code > 0;
        if !has_sform && !has_qform {
            let affine = matmul(&self.fallback_matrix(), voxel_map);
            self.set_affine_with_codes(affine, 1, 1);
            return;
        }
        if has_sform {
            let sform = matmul(&self.sform_matrix(), voxel_map);
            self.set_sform_matrix(&sform);
            if !has_qform {
                let [x, y, z] = column_norms(&sform);
                self.pixdim[1] = x;
                self.pixdim[2] = y;
                self.pixdim[3] = z;
            }
        }
        if has_qform {
            let qform = matmul(&self.qform_matrix(), voxel_map);
            self.set_qform_matrix(&qform);
        }
    }

    /// Store `affine` as the qform (`nifti_mat44_to_quatern`), updating the
    /// quaternion, offsets, voxel sizes, and `qfac`.
    fn set_qform_matrix(&mut self, affine: &Affine) {
        let (quatern, zooms, qfac) = matrix_to_quaternion(affine);
        self.quatern = quatern;
        self.qoffset = [affine[0][3], affine[1][3], affine[2][3]];
        self.pixdim[0] = qfac;
        self.pixdim[1] = zooms[0];
        self.pixdim[2] = zooms[1];
        self.pixdim[3] = zooms[2];
    }

    /// Voxel sizes along each axis: the column norms of [`affine`](Self::affine)
    /// for the first three axes, and `pixdim` for any further axes.
    pub fn spacing(&self) -> Vec<f64> {
        let norms = column_norms(&self.affine());
        (0..self.ndim as usize)
            .map(|i| if i < 3 { norms[i] } else { self.pixdim[i + 1] })
            .collect()
    }
}

fn bad_magic(m: &[u8]) -> Error {
    if m.iter().all(|&b| b == 0) {
        Error::InvalidFileFormat(
            "missing NIfTI magic: this looks like an Analyze 7.5 header, which is not supported"
                .into(),
        )
    } else {
        Error::InvalidMagic([m[0], m[1], m[2], m[3]])
    }
}

fn parse_ndim(raw: i64) -> Result<u8> {
    if (1..=7).contains(&raw) {
        Ok(raw as u8)
    } else {
        Err(Error::InvalidDimensions(format!(
            "dim[0] (ndim) must be in 1..=7, got {raw}"
        )))
    }
}

fn parse_datatype(code: i16) -> Result<DataType> {
    DataType::from_code(code).map_err(|_| {
        Error::InvalidFileFormat(format!(
            "unsupported NIfTI datatype {code} ({})",
            describe_unsupported_code(code)
        ))
    })
}

/// Parse the extension block that follows the fixed header.
///
/// Parsing is lenient: a malformed extension ends the list instead of failing
/// the load, mirroring nibabel, because extensions never affect voxel data.
fn parse_extensions(bytes: &[u8], header: &NiftiHeader, layout: FileLayout) -> Vec<NiftiExtension> {
    let start = header.header_size();
    let limit = match layout {
        FileLayout::Single => usize::try_from(header.vox_offset).unwrap_or(0),
        FileLayout::Pair => bytes.len(),
    }
    .min(bytes.len())
    .min(start + 4 + MAX_EXTENSION_BYTES);

    let mut out = Vec::new();
    if limit < start + 4 || bytes[start] == 0 {
        return out;
    }
    let mut pos = start + 4;
    while pos + 8 <= limit {
        let (esize, ecode) = if header.little_endian {
            (
                LittleEndian::read_i32(&bytes[pos..pos + 4]),
                LittleEndian::read_i32(&bytes[pos + 4..pos + 8]),
            )
        } else {
            (
                BigEndian::read_i32(&bytes[pos..pos + 4]),
                BigEndian::read_i32(&bytes[pos + 4..pos + 8]),
            )
        };
        let Ok(esize) = usize::try_from(esize) else {
            break;
        };
        if esize < 8 || pos + esize > limit {
            break;
        }
        out.push(NiftiExtension {
            ecode,
            data: bytes[pos + 8..pos + esize].to_vec(),
        });
        pos += esize;
    }
    out
}

/// Row-major 4x4 matrix product `a * b`.
pub(crate) fn matmul(a: &Affine, b: &Affine) -> Affine {
    let mut out = [[0.0; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            out[i][j] = (0..4).map(|k| a[i][k] * b[k][j]).sum();
        }
    }
    out
}

/// A 4x4 translation by `t`.
pub(crate) fn translation(t: [f64; 3]) -> Affine {
    [
        [1.0, 0.0, 0.0, t[0]],
        [0.0, 1.0, 0.0, t[1]],
        [0.0, 0.0, 1.0, t[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

/// Euclidean norms of the first three columns of the 3x3 block.
pub(crate) fn column_norms(a: &Affine) -> [f64; 3] {
    std::array::from_fn(|j| (0..3).map(|i| a[i][j] * a[i][j]).sum::<f64>().sqrt())
}

/// Invert a 4x4 affine (with last row `[0, 0, 0, 1]`). Returns `None` when the
/// 3x3 block is singular.
pub(crate) fn invert_affine(a: &Affine) -> Option<Affine> {
    let m = [
        [a[0][0], a[0][1], a[0][2]],
        [a[1][0], a[1][1], a[1][2]],
        [a[2][0], a[2][1], a[2][2]],
    ];
    let inv = invert3(&m)?;
    let t = [a[0][3], a[1][3], a[2][3]];
    let mut out = [[0.0; 4]; 4];
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] = inv[i][j];
        }
        out[i][3] = -(0..3).map(|k| inv[i][k] * t[k]).sum::<f64>();
    }
    out[3][3] = 1.0;
    Some(out)
}

pub(crate) type Mat3 = [[f64; 3]; 3];

fn det3(m: &Mat3) -> f64 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

fn invert3(m: &Mat3) -> Option<Mat3> {
    let det = det3(m);
    if det == 0.0 || !det.is_finite() {
        return None;
    }
    let inv_det = 1.0 / det;
    Some([
        [
            (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * inv_det,
            (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * inv_det,
            (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * inv_det,
        ],
        [
            (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * inv_det,
            (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * inv_det,
            (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * inv_det,
        ],
        [
            (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * inv_det,
            (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * inv_det,
            (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * inv_det,
        ],
    ])
}

fn row_norm(m: &Mat3) -> f64 {
    m.iter()
        .map(|r| r.iter().map(|v| v.abs()).sum::<f64>())
        .fold(0.0, f64::max)
}

fn col_norm(m: &Mat3) -> f64 {
    (0..3)
        .map(|j| (0..3).map(|i| m[i][j].abs()).sum::<f64>())
        .fold(0.0, f64::max)
}

/// Closest orthogonal matrix (polar decomposition), as `nifti_mat33_polar`.
pub(crate) fn polar(a: &Mat3) -> Mat3 {
    let mut x = *a;
    let mut det = det3(&x);
    let mut guard = 0;
    while det == 0.0 && guard < 100 {
        let g = 1e-5 * (1e-3 + row_norm(&x));
        for (i, row) in x.iter_mut().enumerate() {
            row[i] += g;
        }
        det = det3(&x);
        guard += 1;
    }
    let mut diff = 1.0;
    let mut z = x;
    for _ in 0..100 {
        let Some(y) = invert3(&x) else { break };
        let (gam, gmi) = if diff > 0.3 {
            let alp = (row_norm(&x) * col_norm(&x)).sqrt();
            let bet = (row_norm(&y) * col_norm(&y)).sqrt();
            let gam = (bet / alp).sqrt();
            (gam, 1.0 / gam)
        } else {
            (1.0, 1.0)
        };
        for i in 0..3 {
            for j in 0..3 {
                z[i][j] = 0.5 * (gam * x[i][j] + gmi * y[j][i]);
            }
        }
        diff = (0..3)
            .flat_map(|i| (0..3).map(move |j| (i, j)))
            .map(|(i, j)| (z[i][j] - x[i][j]).abs())
            .sum();
        if diff < 3e-12 {
            break;
        }
        x = z;
    }
    z
}

/// Decompose an affine into `(quaternion b,c,d; voxel sizes; qfac)` following
/// `nifti_mat44_to_quatern`.
fn matrix_to_quaternion(affine: &Affine) -> ([f64; 3], [f64; 3], f64) {
    let mut r: Mat3 = [
        [affine[0][0], affine[0][1], affine[0][2]],
        [affine[1][0], affine[1][1], affine[1][2]],
        [affine[2][0], affine[2][1], affine[2][2]],
    ];
    let mut zooms = column_norms(affine);
    for j in 0..3 {
        if zooms[j] == 0.0 || !zooms[j].is_finite() {
            for (i, row) in r.iter_mut().enumerate() {
                row[j] = if i == j { 1.0 } else { 0.0 };
            }
            zooms[j] = 1.0;
        }
        for row in &mut r {
            row[j] /= zooms[j];
        }
    }
    let mut r = polar(&r);
    let qfac = if det3(&r) > 0.0 {
        1.0
    } else {
        for row in &mut r {
            row[2] = -row[2];
        }
        -1.0
    };

    let (r11, r12, r13) = (r[0][0], r[0][1], r[0][2]);
    let (r21, r22, r23) = (r[1][0], r[1][1], r[1][2]);
    let (r31, r32, r33) = (r[2][0], r[2][1], r[2][2]);
    let mut a = r11 + r22 + r33 + 1.0;
    let (mut b, mut c, mut d);
    if a > 0.5 {
        a = 0.5 * a.sqrt();
        b = 0.25 * (r32 - r23) / a;
        c = 0.25 * (r13 - r31) / a;
        d = 0.25 * (r21 - r12) / a;
    } else {
        let xd = 1.0 + r11 - (r22 + r33);
        let yd = 1.0 + r22 - (r11 + r33);
        let zd = 1.0 + r33 - (r11 + r22);
        if xd > 1.0 {
            b = 0.5 * xd.sqrt();
            c = 0.25 * (r12 + r21) / b;
            d = 0.25 * (r13 + r31) / b;
            a = 0.25 * (r32 - r23) / b;
        } else if yd > 1.0 {
            c = 0.5 * yd.sqrt();
            b = 0.25 * (r12 + r21) / c;
            d = 0.25 * (r23 + r32) / c;
            a = 0.25 * (r13 - r31) / c;
        } else {
            d = 0.5 * zd.sqrt();
            b = 0.25 * (r13 + r31) / d;
            c = 0.25 * (r23 + r32) / d;
            a = 0.25 * (r21 - r12) / d;
        }
        if a < 0.0 {
            b = -b;
            c = -c;
            d = -d;
        }
    }
    ([b, c, d], zooms, qfac)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_affine_close(a: &Affine, b: &Affine, tol: f64) {
        for i in 0..4 {
            for j in 0..4 {
                assert!(
                    (a[i][j] - b[i][j]).abs() <= tol,
                    "[{i}][{j}]: {} != {}\n{a:?}\n{b:?}",
                    a[i][j],
                    b[i][j]
                );
            }
        }
    }

    fn oblique() -> Affine {
        let (c, s) = (30f64.to_radians().cos(), 30f64.to_radians().sin());
        [
            [c * 1.5, -s * 2.0, 0.0, 10.0],
            [s * 1.5, c * 2.0, 0.0, -5.0],
            [0.0, 0.0, 3.0, 7.5],
            [0.0, 0.0, 0.0, 1.0],
        ]
    }

    #[test]
    fn qform_roundtrips_oblique_and_left_handed_affines() {
        let mut lh = oblique();
        for row in lh.iter_mut().take(3) {
            row[2] = -row[2];
        }
        for affine in [oblique(), lh] {
            let mut h = NiftiHeader::default();
            h.set_affine(affine);
            assert_affine_close(&h.qform_matrix(), &affine, 1e-9);
            assert_affine_close(&h.affine(), &affine, 1e-12);
        }
    }

    #[test]
    fn qform_handles_180_degree_rotations() {
        // Rotation by pi about x: quaternion a = 0.
        let affine = [
            [2.0, 0.0, 0.0, 1.0],
            [0.0, -2.0, 0.0, 2.0],
            [0.0, 0.0, -2.0, 3.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let mut h = NiftiHeader::default();
        h.set_affine(affine);
        assert_affine_close(&h.qform_matrix(), &affine, 1e-9);
    }

    #[test]
    fn qform_of_sheared_matrix_is_nearest_rigid() {
        let sheared = [
            [1.0, 0.2, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let mut h = NiftiHeader::default();
        h.set_affine(sheared);
        let q = h.qform_matrix();
        // Columns of the qform rotation are orthogonal.
        let dot: f64 = (0..3).map(|i| q[i][0] * q[i][1]).sum();
        assert!(dot.abs() < 1e-9, "qform columns not orthogonal: {dot}");
        // The sform keeps the exact matrix.
        assert_affine_close(&h.affine(), &sheared, 0.0);
    }

    #[test]
    fn transform_voxels_preserves_codes_and_both_spaces() {
        let mut h = NiftiHeader::default();
        h.set_affine_with_codes(oblique(), 4, 1);
        // Shift by one voxel along axis 0.
        let shift = [
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        h.transform_voxels(&shift);
        assert_eq!((h.sform_code, h.qform_code), (4, 1));
        let expected = matmul(&oblique(), &shift);
        assert_affine_close(&h.affine(), &expected, 1e-12);
        assert_affine_close(&h.qform_matrix(), &expected, 1e-9);
    }

    #[test]
    fn zero_pixdim_uses_unit_spacing_in_qform() {
        let h = NiftiHeader {
            sform_code: 0,
            qform_code: 1,
            pixdim: [1.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ..Default::default()
        };
        let q = h.affine();
        assert_eq!([q[0][0], q[1][1], q[2][2]], [2.0, 2.0, 1.0]);
    }

    #[test]
    fn units_roundtrip_including_frequency_units() {
        for t in [
            TemporalUnits::Unknown,
            TemporalUnits::Second,
            TemporalUnits::Millisecond,
            TemporalUnits::Microsecond,
            TemporalUnits::Hertz,
            TemporalUnits::Ppm,
            TemporalUnits::RadPerSecond,
        ] {
            assert_eq!(TemporalUnits::from_code(t.to_code() | 2), t);
        }
        for s in [
            SpatialUnits::Unknown,
            SpatialUnits::Meter,
            SpatialUnits::Millimeter,
            SpatialUnits::Micrometer,
        ] {
            assert_eq!(SpatialUnits::from_code(s.to_code() | 0x18), s);
        }
    }

    fn rich_header(version: NiftiVersion) -> NiftiHeader {
        let mut h = NiftiHeader {
            version,
            ndim: 4,
            dim: [5, 6, 7, 3, 1, 1, 1],
            datatype: DataType::Int16,
            scl_slope: 2.0,
            scl_inter: -1024.0,
            temporal_units: TemporalUnits::Second,
            intent_code: 3,
            intent_p: [12.0, 0.5, 0.0],
            intent_name: "t-test".into(),
            dim_info: 57,
            slice_code: 1,
            slice_start: 0,
            slice_end: 6,
            slice_duration: 0.5,
            toffset: 1.25,
            cal_max: 20.0,
            cal_min: 1.0,
            descrip: "medrs round trip".into(),
            aux_file: "aux".into(),
            extensions: vec![
                NiftiExtension::new(6, b"a comment".to_vec()),
                NiftiExtension::new(4, vec![7u8; 40]),
            ],
            ..Default::default()
        };
        h.set_affine_with_codes(oblique(), 4, 2);
        h.pixdim[4] = 2.5;
        h
    }

    #[test]
    fn every_field_roundtrips_in_both_versions_and_layouts() {
        for version in [NiftiVersion::Nifti1, NiftiVersion::Nifti2] {
            for layout in [FileLayout::Single, FileLayout::Pair] {
                let h = rich_header(version).prepared_for_write(layout);
                let bytes = h.encode(layout).unwrap();
                let (parsed, parsed_layout) = NiftiHeader::parse(&bytes).unwrap();
                assert_eq!(parsed_layout, layout);
                // Extension payloads are padded to 16-byte blocks on disk.
                let mut expected = h.clone();
                for ext in &mut expected.extensions {
                    let padded = ext.encoded_size() - 8;
                    ext.data.resize(padded, 0);
                }
                if version == NiftiVersion::Nifti1 {
                    assert_nifti1_close(&parsed, &expected);
                } else {
                    assert_eq!(parsed, expected);
                }
            }
        }
    }

    /// NIfTI-1 stores floats as f32, so compare with f32 precision.
    fn assert_nifti1_close(a: &NiftiHeader, b: &NiftiHeader) {
        let round = |h: &NiftiHeader| {
            let mut h = h.clone();
            let r = |v: f64| f64::from(v as f32);
            h.pixdim = h.pixdim.map(r);
            h.quatern = h.quatern.map(r);
            h.qoffset = h.qoffset.map(r);
            h.srow_x = h.srow_x.map(r);
            h.srow_y = h.srow_y.map(r);
            h.srow_z = h.srow_z.map(r);
            h.intent_p = h.intent_p.map(r);
            h
        };
        assert_eq!(round(a), round(b));
    }

    #[test]
    fn single_file_vox_offset_accounts_for_extensions() {
        let h = rich_header(NiftiVersion::Nifti1).prepared_for_write(FileLayout::Single);
        // 348 + 4 + 32 ("a comment": 8 + 9 bytes, padded) + 48 (8 + 40) = 432.
        assert_eq!(h.vox_offset, 432);
        let bytes = h.encode(FileLayout::Single).unwrap();
        assert_eq!(bytes.len(), 432);
    }

    #[test]
    fn large_dims_promote_to_nifti2() {
        let h = NiftiHeader {
            dim: [40_000, 2, 2, 1, 1, 1, 1],
            ..Default::default()
        };
        assert!(h.requires_nifti2());
        assert!(h.to_bytes().is_err());
        let p = h.prepared_for_write(FileLayout::Single);
        assert_eq!(p.version, NiftiVersion::Nifti2);
        assert_eq!(p.vox_offset, 544);
        let parsed = NiftiHeader::from_bytes(&p.encode(FileLayout::Single).unwrap()).unwrap();
        assert_eq!(parsed.dim[0], 40_000);
    }

    #[test]
    fn big_endian_headers_parse() {
        let h = rich_header(NiftiVersion::Nifti1).prepared_for_write(FileLayout::Single);
        let mut le = h.encode(FileLayout::Single).unwrap();
        // Byte-swap the fields we check to build a big-endian header by hand.
        let swap = |b: &mut [u8], o: usize, n: usize| b[o..o + n].reverse();
        swap(&mut le, 0, 4);
        swap(&mut le, v1::DIM, 2);
        for i in 0..7 {
            swap(&mut le, v1::DIM + 2 + 2 * i, 2);
        }
        swap(&mut le, v1::DATATYPE, 2);
        swap(&mut le, v1::BITPIX, 2);
        for i in 0..8 {
            swap(&mut le, v1::PIXDIM + 4 * i, 4);
        }
        swap(&mut le, v1::VOX_OFFSET, 4);
        // Drop the extensions: their sizes are little-endian in `le`.
        le[348] = 0;
        let parsed = NiftiHeader::from_bytes(&le).unwrap();
        assert!(!parsed.is_little_endian());
        assert_eq!(parsed.shape(), vec![5, 6, 7, 3]);
        assert_eq!(parsed.datatype, DataType::Int16);
    }

    #[test]
    fn lenient_parsing_of_real_world_quirks() {
        let mut h = NiftiHeader {
            pixdim: [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ..Default::default()
        };
        h.dim = [4, 4, 1, 1, 1, 1, 1];
        let mut bytes = h
            .prepared_for_write(FileLayout::Single)
            .encode(FileLayout::Single)
            .unwrap();
        // vox_offset = 0 in a single file means "right after the header".
        LittleEndian::write_f32(&mut bytes[v1::VOX_OFFSET..v1::VOX_OFFSET + 4], 0.0);
        // Wrong bitpix is ignored in favour of datatype.
        LittleEndian::write_i16(&mut bytes[v1::BITPIX..v1::BITPIX + 2], 8);
        let parsed = NiftiHeader::from_bytes(&bytes).unwrap();
        assert_eq!(parsed.vox_offset, 352);
        assert_eq!(parsed.spacing(), vec![1.0, 1.0, 1.0]);
    }

    #[test]
    fn corrupt_headers_are_rejected_with_clear_errors() {
        let good = NiftiHeader::default()
            .prepared_for_write(FileLayout::Single)
            .encode(FileLayout::Single)
            .unwrap();

        let mut b = good.clone();
        b[v1::MAGIC..v1::MAGIC + 4].copy_from_slice(b"XYZ\0");
        assert!(matches!(
            NiftiHeader::from_bytes(&b),
            Err(Error::InvalidMagic(_))
        ));

        let mut b = good.clone();
        b[v1::MAGIC..v1::MAGIC + 4].copy_from_slice(&[0; 4]);
        let msg = NiftiHeader::from_bytes(&b).unwrap_err().to_string();
        assert!(msg.contains("Analyze"), "{msg}");

        let mut b = good.clone();
        LittleEndian::write_i16(&mut b[v1::DATATYPE..v1::DATATYPE + 2], 128);
        let msg = NiftiHeader::from_bytes(&b).unwrap_err().to_string();
        assert!(msg.contains("DT_RGB24"), "{msg}");

        let mut b = good.clone();
        LittleEndian::write_i16(&mut b[v1::DIM + 2..v1::DIM + 4], -3);
        assert!(NiftiHeader::from_bytes(&b).is_err());

        let mut b = good.clone();
        LittleEndian::write_f32(&mut b[v1::VOX_OFFSET..v1::VOX_OFFSET + 4], 352.5);
        assert!(NiftiHeader::from_bytes(&b).is_err());

        let mut b = good;
        LittleEndian::write_f32(&mut b[v1::VOX_OFFSET..v1::VOX_OFFSET + 4], 100.0);
        assert!(NiftiHeader::from_bytes(&b).is_err());

        assert!(NiftiHeader::from_bytes(&[0x1f, 0x8b, 8, 0])
            .unwrap_err()
            .to_string()
            .contains("gzip"));
    }

    #[test]
    fn malformed_extensions_are_skipped_not_fatal() {
        let h = rich_header(NiftiVersion::Nifti1).prepared_for_write(FileLayout::Single);
        let mut bytes = h.encode(FileLayout::Single).unwrap();
        // Corrupt the second extension's esize so it runs past vox_offset.
        let second = 352 + h.extensions[0].encoded_size();
        LittleEndian::write_i32(&mut bytes[second..second + 4], 10_000);
        let parsed = NiftiHeader::from_bytes(&bytes).unwrap();
        assert_eq!(parsed.extensions.len(), 1);
        assert_eq!(parsed.extensions[0].ecode, 6);
    }

    #[test]
    fn strings_truncate_on_char_boundaries() {
        let h = NiftiHeader {
            intent_name: "é".repeat(10), // 20 bytes, field holds 16
            ..Default::default()
        };
        let parsed = NiftiHeader::from_bytes(
            &h.prepared_for_write(FileLayout::Single)
                .encode(FileLayout::Single)
                .unwrap(),
        )
        .unwrap();
        assert_eq!(parsed.intent_name, "é".repeat(8));
    }

    #[test]
    fn spacing_follows_the_affine() {
        let mut h = NiftiHeader::default();
        h.set_affine(oblique());
        let s = h.spacing();
        assert!((s[0] - 1.5).abs() < 1e-12 && (s[1] - 2.0).abs() < 1e-12);
        assert!((s[2] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn invert_affine_roundtrips() {
        let a = oblique();
        let inv = invert_affine(&a).unwrap();
        let id = matmul(&a, &inv);
        let eye = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        assert_affine_close(&id, &eye, 1e-12);
        assert!(invert_affine(&[[0.0; 4]; 4]).is_none());
    }
}
