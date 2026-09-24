//! Error type.

use thiserror::Error;

/// Result type used throughout medrs.
pub type Result<T> = std::result::Result<T, Error>;

/// Errors returned by medrs.
#[derive(Error, Debug)]
#[non_exhaustive]
pub enum Error {
    /// A file system operation failed. The [`std::io::ErrorKind`] is preserved.
    #[error(transparent)]
    Io(#[from] std::io::Error),

    /// The file does not start with a `NIfTI` header.
    #[error("not a NIfTI file (unrecognised header bytes {0:02x?})")]
    InvalidMagic([u8; 4]),

    /// The datatype code is valid `NIfTI` but not supported by medrs.
    #[error("unsupported NIfTI datatype code {0}")]
    UnsupportedDataType(i16),

    /// A file is malformed, truncated, or internally inconsistent.
    #[error("invalid file: {0}")]
    InvalidFileFormat(String),

    /// A compressed stream is corrupt.
    #[error("decompression failed: {0}")]
    Decompression(String),

    /// Image dimensions are invalid for the operation.
    #[error("invalid dimensions: {0}")]
    InvalidDimensions(String),

    /// Two arrays or images have incompatible shapes.
    #[error("shape mismatch: {0}")]
    ShapeMismatch(String),

    /// An affine matrix is singular or not finite.
    #[error("invalid affine: {0}")]
    InvalidAffine(String),

    /// An orientation code is not a valid axis permutation.
    #[error("invalid orientation: {0}")]
    InvalidOrientation(String),

    /// A crop region does not fit inside the image.
    #[error("invalid crop region: {0}")]
    InvalidCropRegion(String),

    /// The voxel values are unsuitable for the operation (for example NaN
    /// values where statistics are required).
    #[error("invalid data: {0}")]
    InvalidData(String),

    /// An argument or option is out of range.
    #[error("invalid argument: {0}")]
    InvalidArgument(String),

    /// The image does not hold the requested element type.
    #[error("datatype mismatch: expected {expected}, found {found}")]
    DataTypeMismatch {
        /// Requested element type.
        expected: &'static str,
        /// Stored element type.
        found: &'static str,
    },

    /// Internal invariant violation (a bug in medrs).
    #[error("internal error: {0}")]
    Internal(String),
}
