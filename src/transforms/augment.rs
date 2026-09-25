//! Random augmentations.
//!
//! Each function draws its randomness from a caller-supplied RNG, so results
//! are reproducible from a seed. Spatial augmentations are world-preserving
//! (see [`flip`](super::flip)); intensity augmentations return `f32` images.

use super::intensity::{adjust_gamma, apply_map, PointMap};
use super::spatial::{flip, rotate_90};
use crate::error::{Error, Result};
use crate::nifti::element::{fortran_from_vec, ArrayData};
use crate::nifti::NiftiImage;
use rand::{Rng, RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rand_distr::StandardNormal;
use rayon::prelude::*;

/// Flip each of `axes` independently with probability `prob`.
pub fn random_flip<R: Rng + ?Sized>(
    image: &NiftiImage,
    axes: &[usize],
    prob: f64,
    rng: &mut R,
) -> Result<NiftiImage> {
    check_probability(prob)?;
    let chosen: Vec<usize> = axes
        .iter()
        .copied()
        .filter(|_| rng.random_bool(prob))
        .collect();
    flip(image, &chosen)
}

/// Rotate by a uniformly chosen multiple of 90 degrees in the plane of `axes`.
pub fn random_rotate_90<R: Rng + ?Sized>(
    image: &NiftiImage,
    axes: (usize, usize),
    rng: &mut R,
) -> Result<NiftiImage> {
    rotate_90(image, axes, rng.random_range(0..4))
}

/// Multiply intensities by `1 + f`, with `f` uniform in `[-range, range]`
/// (MONAI's `RandScaleIntensity`).
pub fn random_intensity_scale<R: Rng + ?Sized>(
    image: &NiftiImage,
    range: f64,
    rng: &mut R,
) -> Result<NiftiImage> {
    apply_map(image, &scale_map(range, rng)?)
}

/// Add an offset uniform in `[-range, range]` (MONAI's `RandShiftIntensity`).
pub fn random_intensity_shift<R: Rng + ?Sized>(
    image: &NiftiImage,
    range: f64,
    rng: &mut R,
) -> Result<NiftiImage> {
    apply_map(image, &shift_map(range, rng)?)
}

/// Add zero-mean Gaussian noise with standard deviation `std`.
///
/// The noise is generated in parallel from independent streams of one seed,
/// so it is identical for a given RNG state regardless of the thread count.
pub fn random_gaussian_noise<R: Rng + ?Sized>(
    image: &NiftiImage,
    std: f64,
    rng: &mut R,
) -> Result<NiftiImage> {
    add_noise(image, &PointMap::IDENTITY, std, rng.random())
}

/// Gamma contrast adjustment with `gamma` uniform in `range` (MONAI's
/// `RandAdjustContrast`); the value range is preserved.
pub fn random_gamma<R: Rng + ?Sized>(
    image: &NiftiImage,
    range: (f64, f64),
    rng: &mut R,
) -> Result<NiftiImage> {
    adjust_gamma(image, sample_gamma(range, rng)?)
}

pub(crate) fn check_probability(prob: f64) -> Result<()> {
    if (0.0..=1.0).contains(&prob) {
        Ok(())
    } else {
        Err(Error::InvalidArgument(format!(
            "probability must be in [0, 1], got {prob}"
        )))
    }
}

fn check_nonnegative(what: &str, v: f64) -> Result<()> {
    if v.is_finite() && v >= 0.0 {
        Ok(())
    } else {
        Err(Error::InvalidArgument(format!(
            "{what} must be finite and non-negative, got {v}"
        )))
    }
}

fn uniform<R: Rng + ?Sized>(rng: &mut R, range: f64) -> f64 {
    if range > 0.0 {
        rng.random_range(-range..=range)
    } else {
        0.0
    }
}

pub(crate) fn scale_map<R: Rng + ?Sized>(range: f64, rng: &mut R) -> Result<PointMap> {
    check_nonnegative("scale range", range)?;
    Ok(PointMap::linear(1.0 + uniform(rng, range), 0.0))
}

pub(crate) fn shift_map<R: Rng + ?Sized>(range: f64, rng: &mut R) -> Result<PointMap> {
    check_nonnegative("shift range", range)?;
    Ok(PointMap::linear(1.0, uniform(rng, range)))
}

pub(crate) fn sample_gamma<R: Rng + ?Sized>(range: (f64, f64), rng: &mut R) -> Result<f64> {
    let (lo, hi) = range;
    if !(lo.is_finite() && hi.is_finite() && lo > 0.0 && lo <= hi) {
        return Err(Error::InvalidArgument(format!(
            "gamma range must satisfy 0 < min <= max, got ({lo}, {hi})"
        )));
    }
    Ok(if lo < hi {
        rng.random_range(lo..=hi)
    } else {
        lo
    })
}

/// Apply `map` and add Gaussian noise generated from `seed` in one pass.
pub(crate) fn add_noise(
    image: &NiftiImage,
    map: &PointMap,
    std: f64,
    seed: u64,
) -> Result<NiftiImage> {
    /// Elements per independent noise stream.
    const CHUNK: usize = 1 << 14;
    check_nonnegative("noise standard deviation", std)?;
    if std == 0.0 {
        return apply_map(image, map);
    }
    let f = map.kernel();
    let std = std as f32;
    let src = image.f32_values()?;
    let mut out = vec![0f32; src.len()];
    crate::parallel::install(|| {
        out.par_chunks_mut(CHUNK)
            .zip(src.par_chunks(CHUNK))
            .enumerate()
            .for_each(|(i, (o, s))| {
                let mut rng = ChaCha8Rng::seed_from_u64(seed);
                rng.set_stream(i as u64);
                for (d, &v) in o.iter_mut().zip(s) {
                    let z: f32 = rng.sample(StandardNormal);
                    *d = f(v) + std * z;
                }
            });
    });
    Ok(image.with_array_data(ArrayData::F32(fortran_from_vec(image.shape(), out)), true))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nifti::header::Affine;
    use ndarray::{ArrayD, IxDyn, ShapeBuilder};

    const EYE: Affine = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];

    fn ramp(shape: &[usize]) -> NiftiImage {
        let n: usize = shape.iter().product();
        let arr =
            ArrayD::from_shape_vec(IxDyn(shape).f(), (0..n).map(|v| v as f32).collect()).unwrap();
        NiftiImage::from_array(arr, EYE).unwrap()
    }

    #[test]
    fn seeded_augmentations_are_reproducible() {
        let img = ramp(&[8, 8, 8]);
        let run = |seed| {
            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            let a = random_flip(&img, &[0, 1, 2], 0.5, &mut rng).unwrap();
            let b = random_rotate_90(&a, (0, 1), &mut rng).unwrap();
            let c = random_intensity_scale(&b, 0.2, &mut rng).unwrap();
            let d = random_intensity_shift(&c, 0.2, &mut rng).unwrap();
            let e = random_gaussian_noise(&d, 0.1, &mut rng).unwrap();
            random_gamma(&e, (0.7, 1.5), &mut rng)
                .unwrap()
                .to_f32()
                .unwrap()
        };
        assert_eq!(run(7), run(7));
        assert_ne!(run(7), run(8));
    }

    #[test]
    fn flip_probability_extremes() {
        let img = ramp(&[4, 4, 4]);
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let never = random_flip(&img, &[0, 1, 2], 0.0, &mut rng).unwrap();
        assert_eq!(never.to_f32().unwrap(), img.to_f32().unwrap());
        let always = random_flip(&img, &[0], 1.0, &mut rng).unwrap();
        assert_eq!(always.to_f32().unwrap()[[0, 0, 0]], 3.0);
        assert!(random_flip(&img, &[0], 1.5, &mut rng).is_err());
    }

    #[test]
    fn noise_statistics_and_thread_independence() {
        let img =
            NiftiImage::from_array(ArrayD::<f32>::zeros(IxDyn(&[64, 64, 64]).f()), EYE).unwrap();
        let noisy = add_noise(&img, &PointMap::IDENTITY, 2.0, 99).unwrap();
        let v = noisy.to_f32().unwrap();
        let n = v.len() as f64;
        let mean = v.iter().map(|&x| f64::from(x)).sum::<f64>() / n;
        let var = v
            .iter()
            .map(|&x| (f64::from(x) - mean).powi(2))
            .sum::<f64>()
            / n;
        assert!(mean.abs() < 0.01, "{mean}");
        assert!((var.sqrt() - 2.0).abs() < 0.01, "{}", var.sqrt());
        let single = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        let again = single.install(|| add_noise(&img, &PointMap::IDENTITY, 2.0, 99).unwrap());
        assert_eq!(again.to_f32().unwrap(), v);
    }

    #[test]
    fn invalid_parameters_are_rejected() {
        let img = ramp(&[2, 2, 2]);
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        assert!(random_intensity_scale(&img, -0.1, &mut rng).is_err());
        assert!(random_gaussian_noise(&img, f64::NAN, &mut rng).is_err());
        assert!(random_gamma(&img, (0.0, 1.0), &mut rng).is_err());
        assert!(random_gamma(&img, (2.0, 1.0), &mut rng).is_err());
    }
}
