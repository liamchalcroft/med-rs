//! Transform pipeline with lazy evaluation and automatic optimization.
//!
//! This module provides infrastructure for composing transforms and executing them
//! efficiently. Key features:
//!
//! - **Lazy Evaluation**: Operations are recorded and optimized before execution
//! - **Affine Fusion**: Multiple spatial transforms compose into a single resample
//! - **Intensity Fusion**: Sequential intensity operations merge into single passes
//! - **SIMD Acceleration**: Critical paths use AVX2/SSE for 8-way parallelism
//! - **Memory Pooling**: Reusable buffers reduce allocation overhead
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
//!
//! # Memory Pool
//!
//! Buffer pooling for reduced allocations in hot paths:
//!
//! ```ignore
//! use medrs::pipeline::{acquire_buffer, release_buffer};
//!
//! let buffer: Vec<f32> = acquire_buffer(1024 * 1024);
//! // Use buffer...
//! release_buffer(buffer);
//! ```

mod compose;
mod lazy;
mod memory_pool;
mod ops;
pub mod simd_kernels;

pub use compose::{Compose, TransformPipeline};
pub use lazy::{LazyImage, LazyTransform, PendingOp};
pub use memory_pool::{
    acquire_buffer, clear_pool, pool_usage, release_buffer, MemoryPool, PooledBuffer,
};
pub use ops::{AffineOp, FusedIntensityOp};
