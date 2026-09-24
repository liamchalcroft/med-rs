//! Image transforms.
//!
//! Every transform takes a [`NiftiImage`](crate::NiftiImage) and returns a new
//! one. The conventions are the same throughout:
//!
//! * **Spatial transforms are world-preserving.** Crops, flips, rotations,
//!   reorientation, and resampling update the affine so each voxel keeps its
//!   position in scanner space. Saved results overlay the input correctly in
//!   any viewer.
//! * **Exact transforms keep the datatype.** Crops, flips, rotations,
//!   reorientation, and nearest-neighbour resampling copy stored values, so a
//!   `u8` label map stays a `u8` label map. Trilinear resampling produces
//!   `f32`.
//! * **Intensity transforms produce `f32`** in scaled units
//!   (`scl_slope`/`scl_inter` applied).
//! * **4D images** are transformed volume by volume along the first three axes.
//!
//! To run several transforms efficiently, use a [`Pipeline`](crate::Pipeline).

mod augment;
pub(crate) mod geometry;
pub(crate) mod intensity;
mod orientation;
pub(crate) mod resample;
mod sampling;
mod spatial;
mod stats;

pub use augment::{
    random_flip, random_gamma, random_gaussian_noise, random_intensity_scale,
    random_intensity_shift, random_rotate_90,
};
pub use intensity::{
    adjust_gamma, clamp, rescale_intensity, z_normalization, z_normalization_nonzero,
};
pub use orientation::{orientation_from_affine, reorient, AxisCode, Orientation};
pub use resample::{
    resample_like, resample_to_grid, resample_to_shape, resample_to_spacing, Interpolation,
};
pub use sampling::{center_region, random_region, region_around, sample_label_regions, Region};
pub use spatial::{crop, crop_or_pad, flip, rotate_90};

pub(crate) use augment::{add_noise, check_probability, sample_gamma, scale_map, shift_map};
pub(crate) use orientation::reorient_plan;
