//! Error types for medrs.

use thiserror::Error;

/// Result type alias for medrs operations.
pub type Result<T> = std::result::Result<T, Error>;

/// Errors that can occur during medical image operations.
#[derive(Error, Debug)]
pub enum Error {
    /// I/O error during file operations.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Invalid `NIfTI` header magic bytes.
    #[error("invalid NIfTI magic: expected 'n+1' or 'ni1', got {0:?}")]
    InvalidMagic([u8; 4]),

    /// Unsupported `NIfTI` data type.
    #[error("unsupported data type code: {0}")]
    UnsupportedDataType(i16),

    /// Invalid image dimensions.
    #[error("invalid dimensions: {0}")]
    InvalidDimensions(String),

    /// Data type mismatch during conversion.
    #[error("data type mismatch: expected {expected}, got {got}")]
    DataTypeMismatch {
        /// Expected Rust/Python data type name.
        expected: &'static str,
        /// Actual data type encountered.
        got: &'static str,
    },

    /// Invalid affine matrix.
    #[error("invalid affine matrix: {0}")]
    InvalidAffine(String),

    /// Decompression error.
    #[error("decompression failed: {0}")]
    Decompression(String),

    /// Shape mismatch during operations.
    #[error("shape mismatch: {0}")]
    ShapeMismatch(String),

    /// Invalid crop region.
    #[error("invalid crop region: {0}")]
    InvalidCropRegion(String),

    /// Memory allocation error.
    #[error("memory allocation failed: {0}")]
    MemoryAllocation(String),

    /// File format error.
    #[error("invalid file format: {0}")]
    InvalidFileFormat(String),

    /// Transform operation error.
    #[error("transform error: {operation} failed: {reason}")]
    TransformError {
        /// Name of the transform operation.
        operation: &'static str,
        /// Human-readable reason for the failure.
        reason: String,
    },

    /// Configuration error.
    #[error("configuration error: {0}")]
    Configuration(String),
}
