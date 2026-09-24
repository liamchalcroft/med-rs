//! Transform pipelines.
//!
//! A [`Pipeline`] is an ordered list of transforms that produces the same
//! result as calling the corresponding functions in [`crate::transforms`] one
//! after another, but with less work:
//!
//! * Consecutive intensity transforms (clamps, linear rescaling,
//!   z-normalization, random scale/shift) collapse into one pass over the
//!   data. Transforms that need statistics compute them from the pending
//!   values without writing them out.
//! * Consecutive spatial transforms collapse into one grid change, so a chain
//!   such as reorient → resample → crop interpolates once, from the original
//!   voxels (like MONAI's lazy resampling).
//!
//! Random transforms draw their parameters when the pipeline is applied. With
//! [`Pipeline::apply_pair`] the same draws are used for an image and its label
//! map: spatial transforms apply to both (nearest-neighbour for the label),
//! intensity transforms to the image only.
//!
//! ```no_run
//! use medrs::transforms::{Interpolation, Orientation};
//! use medrs::Pipeline;
//!
//! let pipeline = Pipeline::new()
//!     .reorient(Orientation::RAS)
//!     .resample_to_spacing([1.0, 1.0, 1.0], Interpolation::Trilinear)
//!     .crop_or_pad([128, 128, 128], 0.0)
//!     .clamp(-1000.0, 1000.0)
//!     .z_normalize()
//!     .random_flip(&[0, 1, 2], 0.5);
//!
//! let image = medrs::load("ct.nii.gz")?;
//! let output = pipeline.apply(&image)?;
//! # Ok::<(), medrs::Error>(())
//! ```

use crate::error::{Error, Result};
use crate::nifti::header::matmul;
use crate::nifti::{DataType, NiftiImage};
use crate::transforms::geometry::{
    crop_or_pad_plan, crop_plan, flip_axes, flip_plan, rotate_plan, rotation_args, spatial_rank,
    split_shape, with_spatial_shape, GridChange,
};
use crate::transforms::intensity::{
    apply_map, check_range, rescale_map, z_normalize_map, PointMap,
};
use crate::transforms::resample::{apply_grid_change, shape_plan, spacing_plan};
use crate::transforms::{
    add_noise, adjust_gamma, crop_or_pad, orientation_from_affine, reorient_plan, sample_gamma,
    scale_map, shift_map, z_normalization_nonzero, Interpolation, Orientation,
};
use rand::Rng;

/// An ordered list of transforms, applied with fused passes.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Pipeline {
    steps: Vec<Step>,
}

#[derive(Debug, Clone, PartialEq)]
enum Step {
    Spatial(Spatial),
    Clamp(f64, f64),
    ZNormalize,
    ZNormalizeNonzero,
    Rescale(f64, f64),
    Gamma(f64),
    Cast(DataType),
    RandomFlip(Vec<usize>, f64),
    RandomRotate90((usize, usize)),
    RandomScale(f64),
    RandomShift(f64),
    RandomNoise(f64),
    RandomGamma((f64, f64)),
}

#[derive(Debug, Clone, PartialEq)]
enum Spatial {
    Reorient(Orientation),
    Spacing([f64; 3], Interpolation),
    Shape([usize; 3], Interpolation),
    Crop([usize; 3], [usize; 3]),
    CropOrPad([usize; 3], f64),
    Flip(Vec<usize>),
    Rotate90((usize, usize), i32),
}

/// A step with its random parameters drawn.
#[derive(Debug, Clone)]
enum Op {
    Spatial(Spatial),
    Map(PointMap),
    ZNormalize,
    ZNormalizeNonzero,
    Rescale(f64, f64),
    Gamma(f64),
    Noise(f64, u64),
    Cast(DataType),
}

impl Pipeline {
    /// An empty pipeline.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    fn push(mut self, step: Step) -> Self {
        self.steps.push(step);
        self
    }

    fn spatial(self, s: Spatial) -> Self {
        self.push(Step::Spatial(s))
    }

    /// See [`reorient`](crate::transforms::reorient).
    #[must_use]
    pub fn reorient(self, target: Orientation) -> Self {
        self.spatial(Spatial::Reorient(target))
    }

    /// See [`resample_to_spacing`](crate::transforms::resample_to_spacing).
    #[must_use]
    pub fn resample_to_spacing(self, spacing: [f64; 3], interp: Interpolation) -> Self {
        self.spatial(Spatial::Spacing(spacing, interp))
    }

    /// See [`resample_to_shape`](crate::transforms::resample_to_shape).
    #[must_use]
    pub fn resample_to_shape(self, shape: [usize; 3], interp: Interpolation) -> Self {
        self.spatial(Spatial::Shape(shape, interp))
    }

    /// See [`crop`](crate::transforms::crop).
    #[must_use]
    pub fn crop(self, offset: [usize; 3], shape: [usize; 3]) -> Self {
        self.spatial(Spatial::Crop(offset, shape))
    }

    /// See [`crop_or_pad`](crate::transforms::crop_or_pad).
    #[must_use]
    pub fn crop_or_pad(self, shape: [usize; 3], pad_value: f64) -> Self {
        self.spatial(Spatial::CropOrPad(shape, pad_value))
    }

    /// See [`flip`](crate::transforms::flip).
    #[must_use]
    pub fn flip(self, axes: &[usize]) -> Self {
        self.spatial(Spatial::Flip(axes.to_vec()))
    }

    /// See [`rotate_90`](crate::transforms::rotate_90).
    #[must_use]
    pub fn rotate_90(self, axes: (usize, usize), k: i32) -> Self {
        self.spatial(Spatial::Rotate90(axes, k))
    }

    /// See [`clamp`](crate::transforms::clamp).
    #[must_use]
    pub fn clamp(self, min: f64, max: f64) -> Self {
        self.push(Step::Clamp(min, max))
    }

    /// See [`z_normalization`](crate::transforms::z_normalization).
    #[must_use]
    pub fn z_normalize(self) -> Self {
        self.push(Step::ZNormalize)
    }

    /// See [`z_normalization_nonzero`](crate::transforms::z_normalization_nonzero).
    #[must_use]
    pub fn z_normalize_nonzero(self) -> Self {
        self.push(Step::ZNormalizeNonzero)
    }

    /// See [`rescale_intensity`](crate::transforms::rescale_intensity).
    #[must_use]
    pub fn rescale(self, out_min: f64, out_max: f64) -> Self {
        self.push(Step::Rescale(out_min, out_max))
    }

    /// See [`adjust_gamma`](crate::transforms::adjust_gamma).
    #[must_use]
    pub fn adjust_gamma(self, gamma: f64) -> Self {
        self.push(Step::Gamma(gamma))
    }

    /// Convert to another stored datatype (see [`NiftiImage::with_dtype`]).
    #[must_use]
    pub fn cast(self, dtype: DataType) -> Self {
        self.push(Step::Cast(dtype))
    }

    /// See [`random_flip`](crate::transforms::random_flip).
    #[must_use]
    pub fn random_flip(self, axes: &[usize], prob: f64) -> Self {
        self.push(Step::RandomFlip(axes.to_vec(), prob))
    }

    /// See [`random_rotate_90`](crate::transforms::random_rotate_90).
    #[must_use]
    pub fn random_rotate_90(self, axes: (usize, usize)) -> Self {
        self.push(Step::RandomRotate90(axes))
    }

    /// See [`random_intensity_scale`](crate::transforms::random_intensity_scale).
    #[must_use]
    pub fn random_intensity_scale(self, range: f64) -> Self {
        self.push(Step::RandomScale(range))
    }

    /// See [`random_intensity_shift`](crate::transforms::random_intensity_shift).
    #[must_use]
    pub fn random_intensity_shift(self, range: f64) -> Self {
        self.push(Step::RandomShift(range))
    }

    /// See [`random_gaussian_noise`](crate::transforms::random_gaussian_noise).
    #[must_use]
    pub fn random_gaussian_noise(self, std: f64) -> Self {
        self.push(Step::RandomNoise(std))
    }

    /// See [`random_gamma`](crate::transforms::random_gamma).
    #[must_use]
    pub fn random_gamma(self, range: (f64, f64)) -> Self {
        self.push(Step::RandomGamma(range))
    }

    /// Number of transforms.
    pub fn len(&self) -> usize {
        self.steps.len()
    }

    /// Whether the pipeline has no transforms.
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }

    /// Check the parameters that do not depend on the image.
    pub fn validate(&self) -> Result<()> {
        let mut rng = <rand_chacha::ChaCha8Rng as rand::SeedableRng>::seed_from_u64(0);
        self.resolve(&mut rng).map(|_| ())
    }

    /// Apply the pipeline, drawing random parameters from the thread-local
    /// RNG.
    pub fn apply(&self, image: &NiftiImage) -> Result<NiftiImage> {
        self.apply_with_rng(image, &mut rand::rng())
    }

    /// Apply the pipeline, drawing random parameters from `rng`.
    pub fn apply_with_rng<R: Rng + ?Sized>(
        &self,
        image: &NiftiImage,
        rng: &mut R,
    ) -> Result<NiftiImage> {
        run(image, &self.resolve(rng)?)
    }

    /// Apply the pipeline to an image and its label map with the same random
    /// draws. Spatial transforms apply to both (nearest-neighbour and zero
    /// padding for the label); intensity transforms and casts apply to the
    /// image only. The label must be on the same voxel grid as the image.
    pub fn apply_pair<R: Rng + ?Sized>(
        &self,
        image: &NiftiImage,
        label: &NiftiImage,
        rng: &mut R,
    ) -> Result<(NiftiImage, NiftiImage)> {
        if split_shape(image.shape()).0 != split_shape(label.shape()).0 {
            return Err(Error::ShapeMismatch(format!(
                "label shape {:?} does not match image shape {:?}",
                label.shape(),
                image.shape()
            )));
        }
        let ops = self.resolve(rng)?;
        let label_ops: Vec<Op> = ops
            .iter()
            .filter_map(|op| match op {
                Op::Spatial(s) => Some(Op::Spatial(s.for_label())),
                _ => None,
            })
            .collect();
        Ok((run(image, &ops)?, run(label, &label_ops)?))
    }

    fn resolve<R: Rng + ?Sized>(&self, rng: &mut R) -> Result<Vec<Op>> {
        self.steps
            .iter()
            .map(|step| {
                Ok(match step {
                    Step::Spatial(s) => {
                        s.validate()?;
                        Op::Spatial(s.clone())
                    }
                    Step::Clamp(lo, hi) => {
                        check_range("clamp", *lo, *hi)?;
                        Op::Map(PointMap::clamp(*lo, *hi))
                    }
                    Step::ZNormalize => Op::ZNormalize,
                    Step::ZNormalizeNonzero => Op::ZNormalizeNonzero,
                    Step::Rescale(lo, hi) => {
                        check_range("rescale", *lo, *hi)?;
                        Op::Rescale(*lo, *hi)
                    }
                    Step::Gamma(g) => {
                        if !(g.is_finite() && *g > 0.0) {
                            return Err(Error::InvalidArgument(format!(
                                "gamma must be finite and positive, got {g}"
                            )));
                        }
                        Op::Gamma(*g)
                    }
                    Step::Cast(dtype) => Op::Cast(*dtype),
                    Step::RandomFlip(axes, prob) => {
                        crate::transforms::check_probability(*prob)?;
                        flip_axes(3, axes)?;
                        let chosen = axes.iter().copied().filter(|_| rng.random_bool(*prob)).collect();
                        Op::Spatial(Spatial::Flip(chosen))
                    }
                    Step::RandomRotate90(axes) => {
                        Op::Spatial(Spatial::Rotate90(*axes, rng.random_range(0..4)))
                    }
                    Step::RandomScale(range) => Op::Map(scale_map(*range, rng)?),
                    Step::RandomShift(range) => Op::Map(shift_map(*range, rng)?),
                    Step::RandomNoise(std) => {
                        if !(std.is_finite() && *std >= 0.0) {
                            return Err(Error::InvalidArgument(format!(
                                "noise standard deviation must be finite and non-negative, got {std}"
                            )));
                        }
                        Op::Noise(*std, rng.random())
                    }
                    Step::RandomGamma(range) => Op::Gamma(sample_gamma(*range, rng)?),
                })
            })
            .collect()
    }
}

impl Spatial {
    fn validate(&self) -> Result<()> {
        let positive = |shape: &[usize; 3]| {
            if shape.contains(&0) {
                Err(Error::InvalidArgument(format!(
                    "shape must be positive, got {shape:?}"
                )))
            } else {
                Ok(())
            }
        };
        match self {
            Self::Spacing(s, _) => {
                if s.iter().all(|v| v.is_finite() && *v > 0.0) {
                    Ok(())
                } else {
                    Err(Error::InvalidArgument(format!(
                        "target spacing must be finite and positive, got {s:?}"
                    )))
                }
            }
            Self::Shape(shape, _) | Self::Crop(_, shape) => positive(shape),
            Self::CropOrPad(shape, pad) => {
                positive(shape)?;
                if pad.is_finite() {
                    Ok(())
                } else {
                    Err(Error::InvalidArgument(format!(
                        "pad value must be finite, got {pad}"
                    )))
                }
            }
            Self::Flip(axes) => flip_axes(3, axes).map(|_| ()),
            Self::Rotate90(axes, k) => rotation_args(3, *axes, *k).map(|_| ()),
            Self::Reorient(_) => Ok(()),
        }
    }

    /// The same transform for a label map.
    fn for_label(&self) -> Self {
        match self {
            Self::Spacing(s, _) => Self::Spacing(*s, Interpolation::Nearest),
            Self::Shape(s, _) => Self::Shape(*s, Interpolation::Nearest),
            Self::CropOrPad(s, _) => Self::CropOrPad(*s, 0.0),
            other => other.clone(),
        }
    }
}

/// Execution state: the base image, a pending grid change relative to it,
/// and a pending intensity map applied after the grid change.
struct Exec {
    image: NiftiImage,
    grid: Option<GridChange>,
    trilinear: bool,
    map: PointMap,
}

fn run(image: &NiftiImage, ops: &[Op]) -> Result<NiftiImage> {
    let mut x = Exec {
        image: image.clone(),
        grid: None,
        trilinear: false,
        map: PointMap::IDENTITY,
    };
    for op in ops {
        match op {
            Op::Spatial(s) => x.spatial(s)?,
            Op::Map(m) => x.map = x.map.then(m),
            Op::ZNormalize => {
                x.flush_grid()?;
                x.map = z_normalize_map(&x.image.f32_values()?, &x.map)?;
            }
            Op::Rescale(lo, hi) => {
                x.flush_grid()?;
                x.map = rescale_map(&x.image.f32_values()?, &x.map, *lo, *hi)?;
            }
            Op::ZNormalizeNonzero => {
                x.flush()?;
                x.image = z_normalization_nonzero(&x.image)?;
            }
            Op::Gamma(g) => {
                x.flush()?;
                x.image = adjust_gamma(&x.image, *g)?;
            }
            Op::Noise(std, seed) => {
                x.flush_grid()?;
                x.image = add_noise(&x.image, &x.map, *std, *seed)?;
                x.map = PointMap::IDENTITY;
            }
            Op::Cast(dtype) => {
                x.flush()?;
                x.image = x.image.with_dtype(*dtype)?;
            }
        }
    }
    x.flush()?;
    Ok(x.image)
}

impl Exec {
    fn flush_grid(&mut self) -> Result<()> {
        if let Some(change) = self.grid.take() {
            let interp = if self.trilinear {
                Interpolation::Trilinear
            } else {
                Interpolation::Nearest
            };
            self.image = apply_grid_change(&self.image, &change, interp)?;
            self.trilinear = false;
        }
        Ok(())
    }

    fn flush(&mut self) -> Result<()> {
        self.flush_grid()?;
        if !self.map.is_identity() {
            self.image = apply_map(&self.image, &self.map)?;
            self.map = PointMap::IDENTITY;
        }
        Ok(())
    }

    /// Spatial shape after the pending grid change.
    fn spatial_shape(&self) -> [usize; 3] {
        self.grid
            .map_or_else(|| split_shape(self.image.shape()).0, |g| g.shape)
    }

    fn spatial(&mut self, op: &Spatial) -> Result<()> {
        let spatial = self.spatial_shape();
        let rank = spatial_rank(&with_spatial_shape(self.image.shape(), spatial));
        let (change, trilinear) = match op {
            Spatial::Reorient(target) => {
                let affine = match &self.grid {
                    Some(g) => matmul(&self.image.affine(), &g.map),
                    None => self.image.affine(),
                };
                (
                    reorient_plan(orientation_from_affine(&affine), *target, spatial).change,
                    false,
                )
            }
            Spatial::Spacing(target, interp) => {
                let mut header = self.image.header().clone();
                if let Some(g) = &self.grid {
                    header.transform_voxels(&g.map);
                }
                (
                    spacing_plan(spatial, &header.spacing(), *target)?,
                    *interp == Interpolation::Trilinear,
                )
            }
            Spatial::Shape(target, interp) => (
                shape_plan(spatial, *target),
                *interp == Interpolation::Trilinear,
            ),
            Spatial::Crop(offset, shape) => (crop_plan(spatial, *offset, *shape)?, false),
            Spatial::CropOrPad(shape, pad) => {
                let (change, pads) = crop_or_pad_plan(spatial, *shape)?;
                if pads {
                    // New voxels take the pad value in the current units, so
                    // everything before this step must be applied first.
                    self.flush()?;
                    self.image = crop_or_pad(&self.image, *shape, *pad)?;
                    return Ok(());
                }
                (change, false)
            }
            Spatial::Flip(axes) => (flip_plan(spatial, flip_axes(rank, axes)?), false),
            Spatial::Rotate90(axes, k) => (
                rotate_plan(spatial, *axes, rotation_args(rank, *axes, *k)?),
                false,
            ),
        };
        // A pending clamp does not commute with interpolation: apply it first.
        if self.map.has_clamp() && (trilinear || self.trilinear) {
            self.flush()?;
        }
        self.grid = Some(match &self.grid {
            Some(g) => g.then(&change),
            None => change,
        });
        self.trilinear |= trilinear;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nifti::header::Affine;
    use crate::transforms as t;
    use ndarray::{ArrayD, IxDyn, ShapeBuilder};
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    fn affine() -> Affine {
        [
            [-1.5, 0.0, 0.0, 30.0],
            [0.0, 1.0, 0.1, -20.0],
            [0.0, 0.0, 2.0, 10.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    }

    fn image() -> NiftiImage {
        let shape = [12usize, 10, 8];
        let n: usize = shape.iter().product();
        let values = (0..n).map(|i| ((i * 37) % 101) as i16 - 20).collect();
        let arr = ArrayD::from_shape_vec(IxDyn(&shape).f(), values).unwrap();
        let mut img = NiftiImage::from_array(arr, affine()).unwrap();
        img.header_mut().scl_slope = 2.0;
        img.header_mut().scl_inter = 5.0;
        img
    }

    fn assert_close(a: &NiftiImage, b: &NiftiImage) {
        assert_eq!(a.shape(), b.shape());
        assert_eq!(a.dtype(), b.dtype());
        let (x, y) = (a.to_f32().unwrap(), b.to_f32().unwrap());
        for (p, q) in x.iter().zip(y.iter()) {
            assert!((p - q).abs() <= 1e-4 * (1.0 + q.abs()), "{p} vs {q}");
        }
        let (fa, fb) = (a.affine(), b.affine());
        for i in 0..4 {
            for j in 0..4 {
                assert!((fa[i][j] - fb[i][j]).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn fused_intensity_matches_eager_in_order() {
        let img = image();
        let p = Pipeline::new()
            .clamp(0.0, 150.0)
            .z_normalize()
            .clamp(-1.0, 1.0)
            .rescale(0.0, 10.0)
            .clamp(2.0, 8.0);
        let eager = t::clamp(&img, 0.0, 150.0).unwrap();
        let eager = t::z_normalization(&eager).unwrap();
        let eager = t::clamp(&eager, -1.0, 1.0).unwrap();
        let eager = t::rescale_intensity(&eager, 0.0, 10.0).unwrap();
        let eager = t::clamp(&eager, 2.0, 8.0).unwrap();
        assert_close(&p.apply(&img).unwrap(), &eager);
    }

    #[test]
    fn exact_spatial_chain_matches_eager_and_keeps_dtype() {
        let img = image();
        let p = Pipeline::new()
            .flip(&[0, 2])
            .rotate_90((0, 1), 1)
            .reorient(Orientation::LPS)
            .crop([1, 2, 1], [5, 6, 4])
            .crop_or_pad([4, 4, 4], 0.0);
        let e = t::flip(&img, &[0, 2]).unwrap();
        let e = t::rotate_90(&e, (0, 1), 1).unwrap();
        let e = t::reorient(&e, Orientation::LPS).unwrap();
        let e = t::crop(&e, [1, 2, 1], [5, 6, 4]).unwrap();
        let e = t::crop_or_pad(&e, [4, 4, 4], 0.0).unwrap();
        let out = p.apply(&img).unwrap();
        assert_eq!(out.dtype(), DataType::Int16);
        assert_eq!(out.header().scl_slope, 2.0);
        assert_close(&out, &e);
    }

    #[test]
    fn resample_chain_interpolates_once_on_the_final_grid() {
        let img = image();
        let p = Pipeline::new()
            .resample_to_shape([24, 20, 16], Interpolation::Trilinear)
            .resample_to_shape([6, 5, 4], Interpolation::Trilinear);
        let direct = t::resample_to_shape(&img, [6, 5, 4], Interpolation::Trilinear).unwrap();
        assert_close(&p.apply(&img).unwrap(), &direct);
    }

    #[test]
    fn clamp_before_trilinear_resample_is_applied_first() {
        let img = image();
        let p = Pipeline::new()
            .clamp(0.0, 50.0)
            .resample_to_shape([5, 7, 3], Interpolation::Trilinear);
        let e = t::clamp(&img, 0.0, 50.0).unwrap();
        let e = t::resample_to_shape(&e, [5, 7, 3], Interpolation::Trilinear).unwrap();
        assert_close(&p.apply(&img).unwrap(), &e);
    }

    #[test]
    fn stats_after_crop_use_the_cropped_region() {
        let img = image();
        let p = Pipeline::new().crop([2, 2, 2], [4, 4, 4]).z_normalize();
        let e = t::z_normalization(&t::crop(&img, [2, 2, 2], [4, 4, 4]).unwrap()).unwrap();
        assert_close(&p.apply(&img).unwrap(), &e);
    }

    #[test]
    fn padding_uses_the_value_in_current_units() {
        let img = image();
        let p = Pipeline::new().z_normalize().crop_or_pad([14, 10, 8], 0.0);
        let out = p.apply(&img).unwrap().to_f32().unwrap();
        assert_eq!(out[[0, 0, 0]], 0.0);
        assert_eq!(out[[13, 9, 7]], 0.0);
    }

    #[test]
    fn random_pipelines_are_reproducible_and_pairs_stay_aligned() {
        let img = image();
        let label = img.with_dtype(DataType::UInt8).unwrap();
        let p = Pipeline::new()
            .random_flip(&[0, 1, 2], 0.5)
            .random_rotate_90((0, 1))
            .resample_to_shape([6, 6, 6], Interpolation::Trilinear)
            .random_intensity_scale(0.1)
            .random_gaussian_noise(0.1);
        let run = |seed| {
            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            p.apply_pair(&img, &label, &mut rng).unwrap()
        };
        let (a, la) = run(3);
        let (b, lb) = run(3);
        assert_eq!(a.to_f32().unwrap(), b.to_f32().unwrap());
        assert_eq!(la.to_f32().unwrap(), lb.to_f32().unwrap());
        assert_eq!(la.dtype(), DataType::UInt8);
        assert_eq!(a.affine(), la.affine());
        assert_eq!(a.shape(), la.shape());
    }

    #[test]
    fn invalid_parameters_fail_validation() {
        assert!(Pipeline::new().clamp(1.0, 0.0).validate().is_err());
        assert!(Pipeline::new().flip(&[3]).validate().is_err());
        assert!(Pipeline::new()
            .resample_to_spacing([0.0, 1.0, 1.0], Interpolation::Nearest)
            .validate()
            .is_err());
        assert!(Pipeline::new().random_flip(&[0], 2.0).validate().is_err());
        assert!(Pipeline::new()
            .z_normalize()
            .random_gamma((0.5, 2.0))
            .validate()
            .is_ok());
    }
}
