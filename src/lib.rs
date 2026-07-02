// Numeric-codebase lint allowances live in `[workspace.lints.clippy]` so the
// lib, tests, and benches share one policy.
//
// Deny panic-prone patterns in library code to prevent regressions. Test code
// may use unwrap/expect/panic freely for assertions.
#![cfg_attr(
    not(test),
    deny(clippy::panic, clippy::unwrap_used, clippy::expect_used)
)]
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used, clippy::panic))]

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
#[cfg(feature = "jvol")]
pub mod jvol;
pub mod nifti;
pub mod pipeline;
pub mod transforms;

#[cfg(feature = "python")]
#[path = "python/mod.rs"]
mod python;

pub use error::{Error, Result};

// Re-export commonly used items at crate root
pub use nifti::{load, save, NiftiImage};
