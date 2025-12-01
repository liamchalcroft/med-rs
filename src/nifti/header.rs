//! `NIfTI` header parsing and representation.
//!
//! Supports `NIfTI`-1 format (348-byte header) with automatic endianness detection.

use crate::error::{Error, Result};
use byteorder::{BigEndian, ByteOrder, LittleEndian};

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
        match (code >> 3) & 0x07 {
            8 => Self::Second,
            16 => Self::Millisecond,
            24 => Self::Microsecond,
            _ => Self::Unknown,
        }
    }

    fn to_code(self) -> u8 {
        match self {
            Self::Unknown => 0,
            Self::Second => 8,
            Self::Millisecond => 16,
            Self::Microsecond => 24,
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
    /// Voxel sizes (pixdim[1..=ndim]).
    pub pixdim: [f32; 7],
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
            pixdim: [1.0; 7],
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

    fn parse<E: ByteOrder>(bytes: &[u8], little_endian: bool) -> Result<Self> {
        // Validate magic
        let magic = &bytes[344..348];
        if magic != b"n+1\0" && magic != b"ni1\0" {
            return Err(Error::InvalidMagic([
                magic[0], magic[1], magic[2], magic[3],
            ]));
        }

        let ndim = E::read_i16(&bytes[40..42]) as u8;
        let mut dim = [0u16; 7];
        for i in 0..7 {
            dim[i] = E::read_i16(&bytes[42 + i * 2..44 + i * 2]) as u16;
        }

        let datatype = DataType::from_code(E::read_i16(&bytes[70..72]))?;

        let mut pixdim = [0.0f32; 7];
        for i in 0..7 {
            // pixdim array starts at byte 76 (inclusive) with 8 floats
            pixdim[i] = E::read_f32(&bytes[76 + i * 4..80 + i * 4]);
        }

        let xyzt_units = bytes[123];

        let descrip = String::from_utf8_lossy(&bytes[148..228])
            .trim_end_matches('\0')
            .to_string();
        let aux_file = String::from_utf8_lossy(&bytes[228..252])
            .trim_end_matches('\0')
            .to_string();

        Ok(Self {
            ndim,
            dim,
            datatype,
            pixdim,
            vox_offset: E::read_f32(&bytes[108..112]),
            scl_slope: E::read_f32(&bytes[112..116]),
            scl_inter: E::read_f32(&bytes[116..120]),
            spatial_units: SpatialUnits::from_code(xyzt_units),
            temporal_units: TemporalUnits::from_code(xyzt_units),
            intent_code: E::read_i16(&bytes[68..70]),
            descrip,
            aux_file,
            qform_code: E::read_i16(&bytes[252..254]),
            sform_code: E::read_i16(&bytes[254..256]),
            quatern: [
                E::read_f32(&bytes[256..260]),
                E::read_f32(&bytes[260..264]),
                E::read_f32(&bytes[264..268]),
            ],
            qoffset: [
                E::read_f32(&bytes[268..272]),
                E::read_f32(&bytes[272..276]),
                E::read_f32(&bytes[276..280]),
            ],
            srow_x: [
                E::read_f32(&bytes[280..284]),
                E::read_f32(&bytes[284..288]),
                E::read_f32(&bytes[288..292]),
                E::read_f32(&bytes[292..296]),
            ],
            srow_y: [
                E::read_f32(&bytes[296..300]),
                E::read_f32(&bytes[300..304]),
                E::read_f32(&bytes[304..308]),
                E::read_f32(&bytes[308..312]),
            ],
            srow_z: [
                E::read_f32(&bytes[312..316]),
                E::read_f32(&bytes[316..320]),
                E::read_f32(&bytes[320..324]),
                E::read_f32(&bytes[324..328]),
            ],
            little_endian,
        })
        .and_then(|h| {
            h.validate()?;
            Ok(h)
        })
    }

    /// Write header to bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = vec![0u8; Self::SIZE];

        // sizeof_hdr
        LittleEndian::write_i32(&mut buf[0..4], 348);

        // dim
        LittleEndian::write_i16(&mut buf[40..42], self.ndim as i16);
        for i in 0..7 {
            LittleEndian::write_i16(&mut buf[42 + i * 2..44 + i * 2], self.dim[i] as i16);
        }

        // datatype and bitpix
        LittleEndian::write_i16(&mut buf[70..72], self.datatype as i16);
        LittleEndian::write_i16(&mut buf[72..74], (self.datatype.byte_size() * 8) as i16);

        // pixdim
        for i in 0..7 {
            LittleEndian::write_f32(&mut buf[76 + i * 4..80 + i * 4], self.pixdim[i]);
        }

        // vox_offset
        LittleEndian::write_f32(&mut buf[108..112], self.vox_offset);

        // scl_slope, scl_inter
        LittleEndian::write_f32(&mut buf[112..116], self.scl_slope);
        LittleEndian::write_f32(&mut buf[116..120], self.scl_inter);

        // xyzt_units
        buf[123] = self.spatial_units.to_code() | self.temporal_units.to_code();

        // descrip
        let descrip_bytes = self.descrip.as_bytes();
        let len = descrip_bytes.len().min(79);
        buf[148..148 + len].copy_from_slice(&descrip_bytes[..len]);

        // aux_file
        let aux_bytes = self.aux_file.as_bytes();
        let len = aux_bytes.len().min(23);
        buf[228..228 + len].copy_from_slice(&aux_bytes[..len]);

        // qform_code, sform_code
        LittleEndian::write_i16(&mut buf[252..254], self.qform_code);
        LittleEndian::write_i16(&mut buf[254..256], self.sform_code);

        // quatern
        LittleEndian::write_f32(&mut buf[256..260], self.quatern[0]);
        LittleEndian::write_f32(&mut buf[260..264], self.quatern[1]);
        LittleEndian::write_f32(&mut buf[264..268], self.quatern[2]);

        // qoffset
        LittleEndian::write_f32(&mut buf[268..272], self.qoffset[0]);
        LittleEndian::write_f32(&mut buf[272..276], self.qoffset[1]);
        LittleEndian::write_f32(&mut buf[276..280], self.qoffset[2]);

        // srow_x, srow_y, srow_z
        for (i, &v) in self.srow_x.iter().enumerate() {
            LittleEndian::write_f32(&mut buf[280 + i * 4..284 + i * 4], v);
        }
        for (i, &v) in self.srow_y.iter().enumerate() {
            LittleEndian::write_f32(&mut buf[296 + i * 4..300 + i * 4], v);
        }
        for (i, &v) in self.srow_z.iter().enumerate() {
            LittleEndian::write_f32(&mut buf[312 + i * 4..316 + i * 4], v);
        }

        // magic
        buf[344..348].copy_from_slice(b"n+1\0");

        buf
    }

    /// Get the 4x4 affine transformation matrix (sform or qform).
    pub fn affine(&self) -> [[f32; 4]; 4] {
        if self.sform_code > 0 {
            [self.srow_x, self.srow_y, self.srow_z, [0.0, 0.0, 0.0, 1.0]]
        } else if self.qform_code > 0 {
            self.qform_to_affine()
        } else {
            // Default: identity scaled by pixdim
            [
                [self.pixdim[0], 0.0, 0.0, 0.0],
                [0.0, self.pixdim[1], 0.0, 0.0],
                [0.0, 0.0, self.pixdim[2], 0.0],
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
        let [i, j, k] = [self.pixdim[0].abs(), self.pixdim[1], self.pixdim[2] * qfac];

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
        &self.pixdim[..self.ndim as usize]
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
        let byte_size = self.datatype.byte_size() as f32;
        if (self.vox_offset / byte_size).fract() != 0.0 {
            return Err(Error::InvalidDimensions(format!(
                "vox_offset {} not aligned to element size {}",
                self.vox_offset,
                self.datatype.byte_size()
            )));
        }

        Ok(())
    }
}
