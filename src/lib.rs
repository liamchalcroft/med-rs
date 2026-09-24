//! Fast medical image I/O and transforms for deep learning.
//!
//! medrs reads and writes `NIfTI` images, transforms them, and feeds training
//! patches to deep learning frameworks. It is designed around a few ideas:
//!
//! * **Read only what you need.** Uncompressed files are memory-mapped and
//!   decoded lazily; [`load_cropped`](nifti::load_cropped) copies only the
//!   requested region; block-compressed files (Mgzip, `.jvol`) are decoded in
//!   parallel.
//! * **Correct geometry.** Every spatial transform updates the affine so that
//!   images stay aligned with their anatomy (see [`transforms`]), and every
//!   header field and extension survives a load/save round trip.
//! * **Fused pipelines.** A [`Pipeline`] applies a chain of transforms in as
//!   few passes over the data as possible.
//! * **Parallel loading.** [`FastLoader`] streams training patches from worker
//!   threads with a deterministic order.
//!
//! # Example
//!
//! ```no_run
//! use medrs::transforms::{self, Interpolation, Orientation};
//!
//! let image = medrs::load("brain.nii.gz")?;
//! println!("{:?} voxels of {:?} mm", image.shape(), image.spacing());
//!
//! let image = transforms::reorient(&image, Orientation::RAS)?;
//! let image = transforms::resample_to_spacing(&image, [1.0; 3], Interpolation::Trilinear)?;
//! let image = transforms::z_normalization(&image)?;
//! medrs::save(&image, "brain_1mm.nii.gz")?;
//! # Ok::<(), medrs::Error>(())
//! ```
//!
//! # Cargo features
//!
//! * `jvol`: the [`.jvol`](jvol) chunked volume format (lossless and lossy
//!   compression with crop-first decoding).

#![cfg_attr(
    not(test),
    deny(clippy::panic, clippy::unwrap_used, clippy::expect_used)
)]

pub mod error;
#[cfg(feature = "jvol")]
pub mod jvol;
pub mod loader;
pub mod nifti;
mod parallel;
pub mod pipeline;
pub mod transforms;

pub use error::{Error, Result};
pub use loader::{FastLoader, LoaderConfig, Patch};
pub use nifti::{load, save, DataType, NiftiHeader, NiftiImage};
pub use parallel::{num_threads, set_num_threads};
pub use pipeline::Pipeline;
