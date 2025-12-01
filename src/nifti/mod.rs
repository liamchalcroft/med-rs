//! `NIfTI` file format support.
//!
//! `NIfTI` (Neuroimaging Informatics Technology Initiative) is the standard format
//! for neuroimaging data. This module provides high-performance reading and writing
//! of `.nii` and `.nii.gz` files.
//!
//! # Quick Start
//!
//! ```ignore
//! use medrs::nifti;
//!
//! // Load an image
//! let img = nifti::load("brain.nii.gz")?;
//!
//! // Access data as f32 array
//! let data = img.to_f32();
//! println!("Shape: {:?}", img.shape());
//! println!("Affine:\n{:?}", img.affine());
//!
//! // Save modified image
//! nifti::save(&img, "output.nii.gz")?;
//! ```

pub(crate) mod header;
pub(crate) mod image;
mod io;

pub use header::{DataType, NiftiHeader, SpatialUnits, TemporalUnits};
pub use image::{NiftiElement, NiftiImage};
pub use io::{
    load, load_cropped, load_cropped_config, load_header, save, BatchLoader, CropLoader,
    CropLoaderConfig, LoadCroppedConfig, LoaderStats, TrainingDataLoader,
};
