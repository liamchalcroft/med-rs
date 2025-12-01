// Clippy configuration for the crate
// Allow these pedantic lints that are too noisy for this numeric/scientific codebase
#![allow(clippy::cast_possible_truncation)] // Numeric casts are intentional
#![allow(clippy::cast_precision_loss)] // Precision loss in f32/f64 casts is acceptable
#![allow(clippy::cast_sign_loss)] // Sign handling is managed explicitly
#![allow(clippy::cast_lossless)] // Using `as` for clarity in numeric code
#![allow(clippy::cast_possible_wrap)] // Wrap behavior is handled
#![allow(clippy::similar_names)] // c00, c01, c10, c11 naming is intentional for interpolation
#![allow(clippy::many_single_char_names)] // x, y, z naming is standard in 3D code
#![allow(clippy::too_many_lines)] // Some functions are necessarily long
#![allow(clippy::module_name_repetitions)] // NiftiImage in nifti module is clear
#![allow(clippy::must_use_candidate)] // Not all returns need #[must_use]
#![allow(clippy::return_self_not_must_use)] // Builder pattern returns don't need #[must_use]
#![allow(clippy::missing_errors_doc)] // Error docs would be redundant
#![allow(clippy::missing_panics_doc)] // Panics are being converted to Results
#![allow(clippy::suboptimal_flops)] // mul_add suggestions hurt readability
#![allow(clippy::redundant_pub_crate)] // pub(crate) is intentional for visibility
#![allow(clippy::missing_const_for_fn)] // const fn suggestions are premature optimization
#![allow(clippy::doc_markdown)] // NIfTI doesn't need backticks everywhere
#![allow(clippy::ptr_as_ptr)] // Pointer casts are intentional for binary parsing
#![allow(clippy::cast_ptr_alignment)] // Alignment is handled in binary parsing
#![allow(clippy::uninlined_format_args)] // format!("{}", x) is clearer than format!("{x}")
#![allow(clippy::use_self)] // Self vs TypeName is stylistic
#![allow(clippy::redundant_closure_for_method_calls)] // Closures can be clearer
#![allow(clippy::manual_is_multiple_of)] // % 0 == 0 is clearer than is_multiple_of
#![allow(clippy::option_map_or_none)] // map().unwrap_or is clearer
#![allow(clippy::needless_pass_by_value)] // Pass by value is intentional for small types
#![allow(clippy::float_cmp)] // Float comparison is intentional in some cases
#![allow(clippy::wildcard_enum_match_arm)] // Wildcard matches are intentional
#![allow(clippy::explicit_iter_loop)] // for x in iter is clearer than for x in &collection
#![allow(clippy::needless_borrow)] // Borrow is intentional for clarity
#![allow(clippy::manual_memcpy)] // Manual copy is clearer in some contexts

//! # medrs
//!
//! High-performance medical image I/O and processing library for Rust and Python.
//!
//! `medrs` is designed for throughput-critical medical imaging workflows,
//! particularly deep learning pipelines that process large 3D volumes.
//!
//! ## Key Features
//!
//! - **Fast `NIfTI` I/O**: Memory-mapped reading, crop-first loading, optimized gzip
//! - **Transform Pipeline**: Lazy evaluation with automatic fusion and SIMD acceleration
//! - **Random Augmentation**: GPU-friendly augmentations for ML training
//! - **Python Bindings**: Zero-copy numpy views, direct PyTorch/JAX tensor creation
//!
//! ## Quick Start (Rust)
//!
//! ```ignore
//! use medrs::nifti;
//! use medrs::transforms::{resample_to_spacing, Interpolation};
//!
//! // Load a NIfTI image
//! let img = nifti::load("brain.nii.gz")?;
//! println!("Shape: {:?}, Spacing: {:?}", img.shape(), img.spacing());
//!
//! // Resample to isotropic 1mm spacing
//! let resampled = resample_to_spacing(&img, [1.0, 1.0, 1.0], Interpolation::Trilinear);
//!
//! // Save result
//! nifti::save(&resampled, "output.nii.gz")?;
//! ```
//!
//! ## Transform Pipeline
//!
//! ```ignore
//! use medrs::pipeline::compose::TransformPipeline;
//!
//! let pipeline = TransformPipeline::new()
//!     .z_normalize()
//!     .clamp(-1.0, 1.0)
//!     .resample_to_shape([64, 64, 64]);
//!
//! let processed = pipeline.apply(&img);
//! ```
//!
//! ## Random Augmentation
//!
//! ```ignore
//! use medrs::transforms::{random_flip, random_gaussian_noise, random_augment};
//!
//! // Individual augmentations with reproducible seeds
//! let flipped = random_flip(&img, &[0, 1, 2], Some(0.5), Some(42))?;
//! let noisy = random_gaussian_noise(&img, Some(0.1), Some(42));
//!
//! // Combined augmentation pipeline
//! let augmented = random_augment(&img, Some(42))?;
//! ```
//!
//! ## Module Overview
//!
//! - [`nifti`]: `NIfTI` file I/O with memory mapping and crop-first loading
//! - [`transforms`]: Image transforms (resampling, intensity, spatial, augmentation)
//! - [`pipeline`]: Transform composition with lazy evaluation
//! - [`error`]: Error types for the library

pub mod error;
pub mod nifti;
pub mod pipeline;
pub mod transforms;

#[cfg(feature = "python")]
mod python;

pub use error::{Error, Result};

// Re-export commonly used items at crate root
pub use nifti::{load, save, NiftiImage};
