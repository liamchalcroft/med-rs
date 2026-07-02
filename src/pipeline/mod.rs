//! Transform pipeline with lazy evaluation and automatic optimization.
//!
//! This module provides infrastructure for composing transforms and executing them
//! efficiently. Key features:
//!
//! - **Lazy Evaluation**: Operations are recorded and optimized before execution
//! - **Affine Fusion**: Consecutive axis-aligned resamples compose into one pass
//! - **Intensity Fusion**: z-normalize, linear scaling, and clamping stats are
//!   resolved once at materialization and applied in a single pass
//! - **Portable SIMD**: Hot loops use `wide::f32x8`, which lowers to whatever the
//!   compile target supports (SSE2 on baseline x86-64, AVX2 only when built with
//!   `-C target-feature=+avx2` / `-C target-cpu=native`)
//!
//! # Transform Pipeline
//!
//! ```ignore
//! use medrs::pipeline::TransformPipeline;
//!
//! let pipeline = TransformPipeline::new()
//!     .z_normalize()
//!     .clamp(-1.0, 1.0)
//!     .resample_to_shape([64, 64, 64])
//!     .flip(&[0]);
//!
//! let processed = pipeline.apply(&img);
//! ```
//!
//! # Compose API
//!
//! For more control, use [`Compose`] directly:
//!
//! ```ignore
//! use medrs::pipeline::Compose;
//!
//! let pipeline = Compose::new()
//!     .push(MyCustomTransform)
//!     .push(AnotherTransform);
//!
//! let result = pipeline.apply(&img);
//! ```
//!
//! # Lazy Image
//!
//! [`LazyImage`] accumulates pending operations:
//!
//! ```ignore
//! use medrs::pipeline::{LazyImage, PendingOp};
//!
//! let mut lazy = LazyImage::from_image(img);
//! lazy.push_op(PendingOp::Clamp { min: 0.0, max: 1.0 });
//! let result = lazy.materialize()?;
//! ```

mod compose;
mod lazy;
pub mod simd_kernels;

pub use compose::{Compose, TransformPipeline};
pub use lazy::{LazyImage, LazyTransform, PendingOp};
