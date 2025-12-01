//! Image transformation operations for medical imaging.
//!
//! This module provides high-performance transforms commonly used in medical image
//! processing and deep learning pipelines. All transforms operate on [`NiftiImage`]
//! and return new images (immutable design).
//!
//! # Categories
//!
//! ## Resampling
//! - [`resample_to_spacing`] - Resample to target voxel spacing
//! - [`resample_to_shape`] - Resample to target dimensions
//!
//! ## Orientation
//! - [`reorient`] - Reorient to standard orientation (RAS, LPS, etc.)
//! - [`orientation_from_affine`] - Detect orientation from affine matrix
//!
//! ## Intensity
//! - [`z_normalization`] - Zero mean, unit variance normalization
//! - [`rescale_intensity`] - Scale to [min, max] range
//! - [`clamp`] - Clamp values to range
//!
//! ## Spatial
//! - [`crop_or_pad`] - Crop or pad to target shape (centered)
//! - [`flip`] - Flip along specified axes
//!
//! ## Random Augmentation
//! - [`random_flip`] - Probabilistic axis flipping
//! - [`random_gaussian_noise`] - Additive Gaussian noise
//! - [`random_intensity_scale`] - Random intensity scaling
//! - [`random_intensity_shift`] - Random intensity offset
//! - [`random_rotate_90`] - Random 90-degree rotations
//! - [`random_gamma`] - Random gamma correction
//! - [`random_augment`] - Combined augmentation pipeline
//!
//! ## Crop-First Loading
//! - [`CropRegion`] - Region specification for crop operations
//! - [`compute_label_aware_crop_regions`] - MONAI-style positive/negative sampling
//! - [`compute_random_spatial_crop_regions`] - Random spatial crops
//! - [`compute_center_crop_regions`] - Center crop computation
//!
//! # Example
//!
//! ```ignore
//! use medrs::transforms::{resample_to_spacing, z_normalization, Interpolation};
//!
//! let img = medrs::load("brain.nii.gz")?;
//! let resampled = resample_to_spacing(&img, [1.0, 1.0, 1.0], Interpolation::Trilinear);
//! let normalized = z_normalization(&resampled);
//! ```
//!
//! [`NiftiImage`]: crate::nifti::NiftiImage

mod augment;
pub mod crop;
mod intensity;
mod orientation;
mod resample;
mod spatial;

pub use augment::{
    random_augment, random_flip, random_gamma, random_gaussian_noise, random_intensity_scale,
    random_intensity_shift, random_rotate_90,
};
pub use intensity::{clamp, rescale_intensity, z_normalization};
pub use orientation::{orientation_from_affine, reorient, AxisCode, Orientation};
pub use resample::{resample_to_shape, resample_to_spacing, Interpolation};
pub use spatial::{crop_or_pad, flip};

pub use crop::{
    compute_center_crop_regions, compute_label_aware_crop_regions,
    compute_random_spatial_crop_regions, CropRegion, ForegroundDetector,
    RandCropByPosNegLabelConfig, SpatialCropConfig,
};
