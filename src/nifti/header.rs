//! `NIfTI` header parsing and representation.
//!
//! Supports `NIfTI`-1 format (348-byte header) with automatic endianness detection.

use crate::error::{Error, Result};
use byteorder::{BigEndian, ByteOrder, LittleEndian};

/// NIfTI-1 header field byte offsets.
///
/// These constants define the byte positions of each field in the 348-byte NIfTI-1 header.
/// See: <https://nifti.nimh.nih.gov/nifti-1/documentation/nifti1fields>
mod offsets {
    /// sizeof_hdr: must be 348 for NIfTI-1
    pub const SIZEOF_HDR: usize = 0;
    /// dim[0..7]: array dimensions (i16 × 8, but dim[0] is ndim)
    pub const DIM: usize = 40;
    /// intent_code: statistical intent (i16)
    pub const INTENT_CODE: usize = 68;
    /// datatype: data type code (i16)
    pub const DATATYPE: usize = 70;
    /// bitpix: bits per voxel (i16)
    pub const BITPIX: usize = 72;
    /// pixdim[0..7]: qfac + voxel dimensions (f32 × 8)
    pub const PIXDIM: usize = 76;
    /// vox_offset: byte offset to data (f32)
    pub const VOX_OFFSET: usize = 108;
    /// scl_slope: intensity scaling slope (f32)
    pub const SCL_SLOPE: usize = 112;
    /// scl_inter: intensity scaling intercept (f32)
    pub const SCL_INTER: usize = 116;
    /// xyzt_units: spatial/temporal units (u8)
    pub const XYZT_UNITS: usize = 123;
    /// descrip: description string (80 bytes)
    pub const DESCRIP: usize = 148;
    /// aux_file: auxiliary filename (24 bytes)
    pub const AUX_FILE: usize = 228;
    /// qform_code: qform transform code (i16)
    pub const QFORM_CODE: usize = 252;
    /// sform_code: sform transform code (i16)
    pub const SFORM_CODE: usize = 254;
    /// quatern_b: quaternion b parameter (f32)
    pub const QUATERN_B: usize = 256;
    /// quatern_c: quaternion c parameter (f32)
    pub const QUATERN_C: usize = 260;
    /// quatern_d: quaternion d parameter (f32)
    pub const QUATERN_D: usize = 264;
    /// qoffset_x: quaternion x offset (f32)
    pub const QOFFSET_X: usize = 268;
    /// qoffset_y: quaternion y offset (f32)
    pub const QOFFSET_Y: usize = 272;
    /// qoffset_z: quaternion z offset (f32)
    pub const QOFFSET_Z: usize = 276;
    /// srow_x: sform row x (f32 × 4)
    pub const SROW_X: usize = 280;
    /// srow_y: sform row y (f32 × 4)
    pub const SROW_Y: usize = 296;
    /// srow_z: sform row z (f32 × 4)
    pub const SROW_Z: usize = 312;
    /// magic: NIfTI magic bytes ("n+1\0" or "ni1\0")
    pub const MAGIC: usize = 344;
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
        // Temporal units are in bits 3-5 of xyzt_units (mask 0x38 = 0b00111000)
        match code & 0x38 {
            0x08 => Self::Second,      // NIFTI_UNITS_SEC
            0x10 => Self::Millisecond, // NIFTI_UNITS_MSEC
            0x18 => Self::Microsecond, // NIFTI_UNITS_USEC
            _ => Self::Unknown,
        }
    }

    fn to_code(self) -> u8 {
        match self {
            Self::Unknown => 0,
            Self::Second => 0x08,      // NIFTI_UNITS_SEC
            Self::Millisecond => 0x10, // NIFTI_UNITS_MSEC
            Self::Microsecond => 0x18, // NIFTI_UNITS_USEC
        }
    }
}

/// `NIfTI`-1 header (348 bytes).
#[derive(Debug, Clone)]
pub struct NiftiHeader {
    /// Number of dimensions (1-7).
    pub ndim: u8,
    /// Size along each dimension.
    pub dim: [u16; 7],
    /// Data type.
    pub datatype: DataType,
    /// Voxel sizes (pixdim[1..=ndim]) and qfac at index 0.
    pub pixdim: [f32; 8],
    /// Data offset in file.
    pub vox_offset: f32,
    /// Data scaling slope.
    pub scl_slope: f32,
    /// Data scaling intercept.
    pub scl_inter: f32,
    /// Spatial units.
    pub spatial_units: SpatialUnits,
    /// Temporal units.
    pub temporal_units: TemporalUnits,
    /// Intent code.
    pub intent_code: i16,
    /// Description string.
    pub descrip: String,
    /// Auxiliary filename.
    pub aux_file: String,
    /// qform transform code.
    pub qform_code: i16,
    /// sform transform code.
    pub sform_code: i16,
    /// Quaternion parameters for qform.
    pub quatern: [f32; 3],
    /// Offset parameters for qform.
    pub qoffset: [f32; 3],
    /// Affine matrix rows for sform (4x4, stored row-major, last row implicit [0,0,0,1]).
    pub srow_x: [f32; 4],
    /// Second row of the sform affine matrix.
    pub srow_y: [f32; 4],
    /// Third row of the sform affine matrix.
    pub srow_z: [f32; 4],
    /// File endianness (true = little endian).
    pub(crate) little_endian: bool,
}

impl Default for NiftiHeader {
    fn default() -> Self {
        Self {
            ndim: 3,
            dim: [1, 1, 1, 1, 1, 1, 1],
            datatype: DataType::Float32,
            pixdim: [1.0; 8],
            vox_offset: 352.0,
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
    /// Size of `NIfTI`-1 header in bytes.
    pub const SIZE: usize = 348;

    /// Read header from bytes with automatic endianness detection.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < Self::SIZE {
            return Err(Error::Io(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "header too short",
            )));
        }

        // Detect endianness from sizeof_hdr field (should be 348)
        let sizeof_hdr_le = LittleEndian::read_i32(&bytes[0..4]);
        let little_endian = sizeof_hdr_le == 348;

        if little_endian {
            Self::parse::<LittleEndian>(bytes, true)
        } else {
            Self::parse::<BigEndian>(bytes, false)
        }
    }

    #[allow(clippy::wildcard_imports)] // Local module with 20+ related constants
    fn parse<E: ByteOrder>(bytes: &[u8], little_endian: bool) -> Result<Self> {
        use offsets::*;

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
                "ndim must be 1..=7, got {} (raw i16 value)",
                ndim_raw
            )));
        }
        let ndim = ndim_raw as u8;
        let mut dim = [0u16; 7];
        for (i, dim_val) in dim.iter_mut().enumerate() {
            let offset = DIM + 2 + i * 2;
            let dim_raw = E::read_i16(&bytes[offset..offset + 2]);
            if dim_raw < 0 {
                return Err(Error::InvalidDimensions(format!(
                    "dimension {} has negative value: {}",
                    i, dim_raw
                )));
            }
            *dim_val = dim_raw as u16;
        }

        let datatype = DataType::from_code(E::read_i16(&bytes[DATATYPE..DATATYPE + 2]))?;

        let mut pixdim = [0.0f32; 8];
        for (i, pix_val) in pixdim.iter_mut().enumerate() {
            let offset = PIXDIM + i * 4;
            *pix_val = E::read_f32(&bytes[offset..offset + 4]);
        }

        let xyzt_units = bytes[XYZT_UNITS];

        let descrip = String::from_utf8_lossy(&bytes[DESCRIP..AUX_FILE])
            .trim_end_matches('\0')
            .to_string();
        let aux_file = String::from_utf8_lossy(&bytes[AUX_FILE..QFORM_CODE])
            .trim_end_matches('\0')
            .to_string();

        Ok(Self {
            ndim,
            dim,
            datatype,
            pixdim,
            vox_offset: E::read_f32(&bytes[VOX_OFFSET..VOX_OFFSET + 4]),
            scl_slope: E::read_f32(&bytes[SCL_SLOPE..SCL_SLOPE + 4]),
            scl_inter: E::read_f32(&bytes[SCL_INTER..SCL_INTER + 4]),
            spatial_units: SpatialUnits::from_code(xyzt_units),
            temporal_units: TemporalUnits::from_code(xyzt_units),
            intent_code: E::read_i16(&bytes[INTENT_CODE..INTENT_CODE + 2]),
            descrip,
            aux_file,
            qform_code: E::read_i16(&bytes[QFORM_CODE..QFORM_CODE + 2]),
            sform_code: E::read_i16(&bytes[SFORM_CODE..SFORM_CODE + 2]),
            quatern: [
                E::read_f32(&bytes[QUATERN_B..QUATERN_B + 4]),
                E::read_f32(&bytes[QUATERN_C..QUATERN_C + 4]),
                E::read_f32(&bytes[QUATERN_D..QUATERN_D + 4]),
            ],
            qoffset: [
                E::read_f32(&bytes[QOFFSET_X..QOFFSET_X + 4]),
                E::read_f32(&bytes[QOFFSET_Y..QOFFSET_Y + 4]),
                E::read_f32(&bytes[QOFFSET_Z..QOFFSET_Z + 4]),
            ],
            srow_x: [
                E::read_f32(&bytes[SROW_X..SROW_X + 4]),
                E::read_f32(&bytes[SROW_X + 4..SROW_X + 8]),
                E::read_f32(&bytes[SROW_X + 8..SROW_X + 12]),
                E::read_f32(&bytes[SROW_X + 12..SROW_X + 16]),
            ],
            srow_y: [
                E::read_f32(&bytes[SROW_Y..SROW_Y + 4]),
                E::read_f32(&bytes[SROW_Y + 4..SROW_Y + 8]),
                E::read_f32(&bytes[SROW_Y + 8..SROW_Y + 12]),
                E::read_f32(&bytes[SROW_Y + 12..SROW_Y + 16]),
            ],
            srow_z: [
                E::read_f32(&bytes[SROW_Z..SROW_Z + 4]),
                E::read_f32(&bytes[SROW_Z + 4..SROW_Z + 8]),
                E::read_f32(&bytes[SROW_Z + 8..SROW_Z + 12]),
                E::read_f32(&bytes[SROW_Z + 12..SROW_Z + 16]),
            ],
            little_endian,
        })
        .and_then(|h| {
            h.validate()?;
            Ok(h)
        })
    }

    /// Write header to bytes.
    #[allow(clippy::wildcard_imports)] // Local module with 20+ related constants
    pub fn to_bytes(&self) -> Vec<u8> {
        use offsets::*;

        let mut buf = vec![0u8; Self::SIZE];

        // sizeof_hdr
        LittleEndian::write_i32(&mut buf[SIZEOF_HDR..SIZEOF_HDR + 4], 348);

        // dim
        LittleEndian::write_i16(&mut buf[DIM..DIM + 2], self.ndim as i16);
        for i in 0..7 {
            let offset = DIM + 2 + i * 2;
            LittleEndian::write_i16(&mut buf[offset..offset + 2], self.dim[i] as i16);
        }

        // datatype and bitpix
        LittleEndian::write_i16(&mut buf[DATATYPE..DATATYPE + 2], self.datatype as i16);
        LittleEndian::write_i16(
            &mut buf[BITPIX..BITPIX + 2],
            (self.datatype.byte_size() * 8) as i16,
        );

        // pixdim
        for (i, &value) in self.pixdim.iter().enumerate() {
            let offset = PIXDIM + i * 4;
            LittleEndian::write_f32(&mut buf[offset..offset + 4], value);
        }

        // vox_offset
        LittleEndian::write_f32(&mut buf[VOX_OFFSET..VOX_OFFSET + 4], self.vox_offset);

        // scl_slope, scl_inter
        LittleEndian::write_f32(&mut buf[SCL_SLOPE..SCL_SLOPE + 4], self.scl_slope);
        LittleEndian::write_f32(&mut buf[SCL_INTER..SCL_INTER + 4], self.scl_inter);

        // xyzt_units
        buf[XYZT_UNITS] = self.spatial_units.to_code() | self.temporal_units.to_code();

        // descrip (80 bytes, from DESCRIP to AUX_FILE)
        let descrip_bytes = self.descrip.as_bytes();
        let len = descrip_bytes.len().min(79);
        buf[DESCRIP..DESCRIP + len].copy_from_slice(&descrip_bytes[..len]);

        // aux_file (24 bytes, from AUX_FILE to QFORM_CODE)
        let aux_bytes = self.aux_file.as_bytes();
        let len = aux_bytes.len().min(23);
        buf[AUX_FILE..AUX_FILE + len].copy_from_slice(&aux_bytes[..len]);

        // qform_code, sform_code
        LittleEndian::write_i16(&mut buf[QFORM_CODE..QFORM_CODE + 2], self.qform_code);
        LittleEndian::write_i16(&mut buf[SFORM_CODE..SFORM_CODE + 2], self.sform_code);

        // quatern
        LittleEndian::write_f32(&mut buf[QUATERN_B..QUATERN_B + 4], self.quatern[0]);
        LittleEndian::write_f32(&mut buf[QUATERN_C..QUATERN_C + 4], self.quatern[1]);
        LittleEndian::write_f32(&mut buf[QUATERN_D..QUATERN_D + 4], self.quatern[2]);

        // qoffset
        LittleEndian::write_f32(&mut buf[QOFFSET_X..QOFFSET_X + 4], self.qoffset[0]);
        LittleEndian::write_f32(&mut buf[QOFFSET_Y..QOFFSET_Y + 4], self.qoffset[1]);
        LittleEndian::write_f32(&mut buf[QOFFSET_Z..QOFFSET_Z + 4], self.qoffset[2]);

        // srow_x, srow_y, srow_z
        for (i, &v) in self.srow_x.iter().enumerate() {
            let offset = SROW_X + i * 4;
            LittleEndian::write_f32(&mut buf[offset..offset + 4], v);
        }
        for (i, &v) in self.srow_y.iter().enumerate() {
            let offset = SROW_Y + i * 4;
            LittleEndian::write_f32(&mut buf[offset..offset + 4], v);
        }
        for (i, &v) in self.srow_z.iter().enumerate() {
            let offset = SROW_Z + i * 4;
            LittleEndian::write_f32(&mut buf[offset..offset + 4], v);
        }

        // magic
        buf[MAGIC..MAGIC + 4].copy_from_slice(b"n+1\0");

        buf
    }

    /// Get the 4x4 affine transformation matrix (sform or qform).
    pub fn affine(&self) -> [[f32; 4]; 4] {
        if self.sform_code > 0 {
            [self.srow_x, self.srow_y, self.srow_z, [0.0, 0.0, 0.0, 1.0]]
        } else if self.qform_code > 0 {
            self.qform_to_affine()
        } else {
            // Default: identity scaled by pixdim (skip qfac at index 0)
            [
                [self.pixdim[1], 0.0, 0.0, 0.0],
                [0.0, self.pixdim[2], 0.0, 0.0],
                [0.0, 0.0, self.pixdim[3], 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        }
    }

    /// Set affine from 4x4 matrix.
    pub fn set_affine(&mut self, affine: [[f32; 4]; 4]) {
        self.srow_x = affine[0];
        self.srow_y = affine[1];
        self.srow_z = affine[2];
        self.sform_code = 1;
    }

    /// Convert quaternion representation to affine matrix.
    #[allow(clippy::many_single_char_names)]
    fn qform_to_affine(&self) -> [[f32; 4]; 4] {
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

    /// Get image shape as a slice (up to ndim elements).
    pub fn shape(&self) -> &[u16] {
        &self.dim[..self.ndim as usize]
    }

    /// Get voxel spacing as a slice (up to ndim elements).
    pub fn spacing(&self) -> &[f32] {
        let end = (self.ndim as usize + 1).min(self.pixdim.len());
        &self.pixdim[1.min(end)..end]
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

    /// Validate header fields for basic `NIfTI` invariants.
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

        if !self.vox_offset.is_finite() {
            return Err(Error::InvalidDimensions(format!(
                "vox_offset must be finite, got {}",
                self.vox_offset
            )));
        }

        if self.vox_offset.fract() != 0.0 {
            return Err(Error::InvalidDimensions(format!(
                "vox_offset must be an integer number of bytes, got {}",
                self.vox_offset
            )));
        }

        if self.vox_offset < Self::SIZE as f32 {
            return Err(Error::InvalidDimensions(format!(
                "vox_offset {} before header end ({})",
                self.vox_offset,
                Self::SIZE
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
        // Use integer modulo to avoid float precision issues
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
        // Test that temporal units are correctly parsed from xyzt_units byte
        // Temporal units are stored in bits 3-5 (mask 0x38)
        assert_eq!(TemporalUnits::from_code(0x08), TemporalUnits::Second);
        assert_eq!(TemporalUnits::from_code(0x10), TemporalUnits::Millisecond);
        assert_eq!(TemporalUnits::from_code(0x18), TemporalUnits::Microsecond);
        assert_eq!(TemporalUnits::from_code(0x00), TemporalUnits::Unknown);

        // Test combined with spatial units (spatial units in bits 0-2)
        // mm (0x02) + second (0x08) = 0x0A
        assert_eq!(TemporalUnits::from_code(0x0A), TemporalUnits::Second);
        // meter (0x01) + millisecond (0x10) = 0x11
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
    fn test_spacing_skips_qfac() {
        let mut header = NiftiHeader::default();
        header.ndim = 3;
        header.pixdim = [-1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0];
        assert_eq!(header.spacing(), &[2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_pixdim_roundtrip_includes_qfac_and_dim7() {
        let mut header = NiftiHeader::default();
        header.ndim = 7;
        header.pixdim = [-1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];

        let bytes = header.to_bytes();
        let parsed = NiftiHeader::from_bytes(&bytes).unwrap();

        assert_eq!(parsed.pixdim, header.pixdim);
    }
}
