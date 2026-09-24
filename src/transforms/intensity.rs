//! Intensity transforms.
//!
//! Intensity transforms work on scaled values (`scl_slope`/`scl_inter`
//! applied) and return `f32` images with identity scaling. Statistics are
//! accumulated in double precision.
//!
//! Chains of linear maps and clamps are represented in closed form by
//! [`PointMap`], which lets the pipeline apply any sequence of them in a
//! single pass over the data.

use super::stats::{min_max, moments};
use crate::error::{Error, Result};
use crate::nifti::element::{fortran_from_vec, ArrayData};
use crate::nifti::NiftiImage;
use rayon::prelude::*;

const CHUNK: usize = 1 << 16;

/// `y = clamp(scale · x + offset, lo, hi)`, the closed form of any chain of
/// linear maps and clamps. NaN inputs stay NaN.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct PointMap {
    scale: f64,
    offset: f64,
    lo: f64,
    hi: f64,
}

impl PointMap {
    pub const IDENTITY: Self = Self {
        scale: 1.0,
        offset: 0.0,
        lo: f64::NEG_INFINITY,
        hi: f64::INFINITY,
    };

    pub fn linear(scale: f64, offset: f64) -> Self {
        Self {
            scale,
            offset,
            ..Self::IDENTITY
        }
    }

    /// Clamp to `[lo, hi]`; callers ensure `lo <= hi` and neither is NaN.
    pub fn clamp(lo: f64, hi: f64) -> Self {
        Self {
            lo,
            hi,
            ..Self::IDENTITY
        }
    }

    const fn constant(value: f64) -> Self {
        Self {
            scale: 0.0,
            offset: value,
            lo: f64::NEG_INFINITY,
            hi: f64::INFINITY,
        }
    }

    pub fn is_identity(&self) -> bool {
        *self == Self::IDENTITY
    }

    pub fn has_clamp(&self) -> bool {
        self.lo > f64::NEG_INFINITY || self.hi < f64::INFINITY
    }

    /// `next ∘ self`: apply `self`, then `next`.
    pub fn then(&self, next: &Self) -> Self {
        // next(clamp(u, lo, hi)) with u = scale·x + offset.
        let s = next.scale;
        let (lo, hi) = if s > 0.0 {
            (s * self.lo + next.offset, s * self.hi + next.offset)
        } else if s < 0.0 {
            (s * self.hi + next.offset, s * self.lo + next.offset)
        } else {
            return Self::constant(next.offset.clamp(next.lo, next.hi));
        };
        let (lo, hi) = (lo.max(next.lo), hi.min(next.hi));
        if lo > hi {
            // The two clamp ranges do not overlap: every value lands on the
            // edge of `next`'s range that faces the previous range.
            let value = if hi == next.hi { next.hi } else { next.lo };
            return Self::constant(value);
        }
        Self {
            scale: s * self.scale,
            offset: s * self.offset + next.offset,
            lo,
            hi,
        }
    }

    /// The map as an `f32` function.
    pub fn kernel(&self) -> impl Fn(f32) -> f32 + Copy + Send + Sync {
        let (s, o) = (self.scale as f32, self.offset as f32);
        let (lo, hi) = (self.lo as f32, self.hi as f32);
        let clamp = self.has_clamp();
        move |x: f32| {
            let y = x * s + o;
            if clamp {
                y.clamp(lo, hi)
            } else {
                y
            }
        }
    }
}

/// Apply `f` to every scaled value, producing an `f32` image.
pub(crate) fn map_values<F>(image: &NiftiImage, f: F) -> Result<NiftiImage>
where
    F: Fn(f32) -> f32 + Sync + Send,
{
    let src = image.f32_values()?;
    let mut out = vec![0f32; src.len()];
    crate::parallel::install(|| {
        out.par_chunks_mut(CHUNK)
            .zip(src.par_chunks(CHUNK))
            .for_each(|(o, s)| {
                for (d, &v) in o.iter_mut().zip(s) {
                    *d = f(v);
                }
            });
    });
    Ok(image.with_array_data(ArrayData::F32(fortran_from_vec(image.shape(), out)), true))
}

/// Apply a [`PointMap`], producing an `f32` image.
pub(crate) fn apply_map(image: &NiftiImage, map: &PointMap) -> Result<NiftiImage> {
    map_values(image, map.kernel())
}

/// The map that z-normalizes `map(values)`.
pub(crate) fn z_normalize_map(values: &[f32], map: &PointMap) -> Result<PointMap> {
    let f = map.kernel();
    let m = moments(values, |v| Some(f(v)));
    let std = m.std();
    if !(m.mean.is_finite() && std.is_finite()) {
        return Err(Error::InvalidData(
            "cannot z-normalize: the image contains NaN or infinite values".into(),
        ));
    }
    let inv = if std > 0.0 { 1.0 / std } else { 1.0 };
    Ok(map.then(&PointMap::linear(inv, -m.mean * inv)))
}

/// The map that rescales `map(values)` from its range to `[out_min, out_max]`.
pub(crate) fn rescale_map(
    values: &[f32],
    map: &PointMap,
    out_min: f64,
    out_max: f64,
) -> Result<PointMap> {
    check_range("rescale", out_min, out_max)?;
    let f = map.kernel();
    let Some((lo, hi)) = min_max(values, f) else {
        // Every value is NaN: nothing to rescale.
        return Ok(*map);
    };
    let (lo, hi) = (f64::from(lo), f64::from(hi));
    let scale = if hi > lo {
        (out_max - out_min) / (hi - lo)
    } else {
        0.0
    };
    Ok(map.then(&PointMap::linear(scale, out_min - lo * scale)))
}

pub(crate) fn check_range(what: &str, lo: f64, hi: f64) -> Result<()> {
    if lo.is_nan() || hi.is_nan() || lo > hi || lo == f64::INFINITY || hi == f64::NEG_INFINITY {
        return Err(Error::InvalidArgument(format!(
            "{what} requires min <= max, got [{lo}, {hi}]"
        )));
    }
    Ok(())
}

/// Normalize to zero mean and unit (population) standard deviation.
///
/// A constant image becomes all zeros. Returns an error if the image contains
/// NaN or infinity.
pub fn z_normalization(image: &NiftiImage) -> Result<NiftiImage> {
    let values = image.f32_values()?;
    let map = z_normalize_map(&values, &PointMap::IDENTITY)?;
    apply_map(image, &map)
}

/// Like [`z_normalization`], but statistics use only non-zero voxels and zero
/// voxels stay zero (MONAI's `NormalizeIntensity(nonzero=True)`), the usual
/// choice for skull-stripped images.
pub fn z_normalization_nonzero(image: &NiftiImage) -> Result<NiftiImage> {
    let values = image.f32_values()?;
    let m = moments(&values, |v| (v != 0.0).then_some(v));
    if m.n == 0.0 {
        return map_values(image, |v| v);
    }
    let std = m.std();
    if !(m.mean.is_finite() && std.is_finite()) {
        return Err(Error::InvalidData(
            "cannot z-normalize: the image contains NaN or infinite values".into(),
        ));
    }
    let inv = if std > 0.0 { 1.0 / std } else { 1.0 };
    let (mean, inv) = (m.mean as f32, inv as f32);
    map_values(
        image,
        move |v| if v == 0.0 { 0.0 } else { (v - mean) * inv },
    )
}

/// Linearly map the data range `[min, max]` onto `[out_min, out_max]`.
///
/// NaN values are ignored when finding the range and stay NaN. A constant
/// image maps to `out_min`.
pub fn rescale_intensity(image: &NiftiImage, out_min: f64, out_max: f64) -> Result<NiftiImage> {
    let values = image.f32_values()?;
    let map = rescale_map(&values, &PointMap::IDENTITY, out_min, out_max)?;
    apply_map(image, &map)
}

/// Clamp values to `[min, max]`. NaN values stay NaN.
pub fn clamp(image: &NiftiImage, min: f64, max: f64) -> Result<NiftiImage> {
    check_range("clamp", min, max)?;
    apply_map(image, &PointMap::clamp(min, max))
}

/// Gamma contrast adjustment that preserves the value range (MONAI's
/// `AdjustContrast`): values are mapped to `[0, 1]`, raised to `gamma`, and
/// mapped back.
pub fn adjust_gamma(image: &NiftiImage, gamma: f64) -> Result<NiftiImage> {
    if !(gamma.is_finite() && gamma > 0.0) {
        return Err(Error::InvalidArgument(format!(
            "gamma must be finite and positive, got {gamma}"
        )));
    }
    let values = image.f32_values()?;
    let Some((lo, hi)) = min_max(&values, |v| v) else {
        return map_values(image, |v| v);
    };
    let range = hi - lo;
    if range <= 0.0 {
        return map_values(image, |v| v);
    }
    let g = gamma as f32;
    map_values(image, move |v| ((v - lo) / range).powf(g) * range + lo)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nifti::element::NiftiElement;
    use crate::nifti::header::Affine;
    use crate::nifti::DataType;
    use ndarray::{ArrayD, IxDyn, ShapeBuilder};

    const EYE: Affine = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];

    fn img<T: NiftiElement>(values: Vec<T>) -> NiftiImage {
        let n = values.len();
        NiftiImage::from_array(
            ArrayD::from_shape_vec(IxDyn(&[n, 1, 1]).f(), values).unwrap(),
            EYE,
        )
        .unwrap()
    }

    fn values(i: &NiftiImage) -> Vec<f32> {
        i.to_f32().unwrap().iter().copied().collect()
    }

    fn eval(m: &PointMap, x: f64) -> f64 {
        f64::from(m.kernel()(x as f32))
    }

    #[test]
    fn point_map_composition_matches_sequential_application() {
        let maps = [
            PointMap::linear(2.0, -1.0),
            PointMap::clamp(-0.5, 3.0),
            PointMap::linear(-1.5, 0.25),
            PointMap::clamp(0.0, 1.0),
            PointMap::clamp(5.0, 6.0),
            PointMap::linear(0.0, 7.0),
            PointMap::clamp(-2.0, -1.0),
            PointMap::linear(1.0, 10.0),
        ];
        for x in [-10.0, -1.0, -0.3, 0.0, 0.4, 1.0, 2.5, 100.0] {
            for start in 0..maps.len() {
                for end in start..=maps.len() {
                    let mut composed = PointMap::IDENTITY;
                    let mut seq = x;
                    for m in &maps[start..end] {
                        composed = composed.then(m);
                        seq = eval(m, seq);
                    }
                    let got = eval(&composed, x);
                    assert!(
                        (got - seq).abs() <= 1e-5 * (1.0 + seq.abs()),
                        "{start}..{end} at {x}: {got} vs {seq}"
                    );
                }
            }
        }
        assert!(eval(&PointMap::clamp(0.0, 1.0), f64::NAN).is_nan());
    }

    #[test]
    fn znorm_basic_constant_and_nan() {
        let z = values(&z_normalization(&img(vec![1.0f32, 2.0, 3.0, 4.0])).unwrap());
        let mean: f32 = z.iter().sum::<f32>() / 4.0;
        let var: f32 = z.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / 4.0;
        assert!(mean.abs() < 1e-6 && (var - 1.0).abs() < 1e-5);
        assert_eq!(
            values(&z_normalization(&img(vec![5.0f32; 4])).unwrap()),
            vec![0.0; 4]
        );
        assert!(z_normalization(&img(vec![1.0f32, f32::NAN])).is_err());
    }

    #[test]
    fn znorm_nonzero_keeps_background() {
        let z = values(&z_normalization_nonzero(&img(vec![0.0f32, 2.0, 4.0, 0.0])).unwrap());
        assert_eq!(z, vec![0.0, -1.0, 1.0, 0.0]);
    }

    #[test]
    fn rescale_ignores_nan_and_handles_constant() {
        let r =
            values(&rescale_intensity(&img(vec![0.0f32, 5.0, 10.0, f32::NAN]), -1.0, 1.0).unwrap());
        assert_eq!(&r[..3], &[-1.0, 0.0, 1.0]);
        assert!(r[3].is_nan());
        assert_eq!(
            values(&rescale_intensity(&img(vec![3.0f32; 3]), 0.0, 1.0).unwrap()),
            vec![0.0; 3]
        );
        assert!(rescale_intensity(&img(vec![1.0f32]), 1.0, 0.0).is_err());
    }

    #[test]
    fn clamp_uses_scaled_units() {
        // CT-style scaling: raw 0..4000 with intercept -1024.
        let mut ct = img(vec![0i16, 500, 1500, 4000]);
        ct.header_mut().scl_inter = -1024.0;
        let c = clamp(&ct, -500.0, 500.0).unwrap();
        assert_eq!(c.dtype(), DataType::Float32);
        assert_eq!(c.header().scl_inter, 0.0);
        assert_eq!(values(&c), vec![-500.0, -500.0, 476.0, 500.0]);
        assert!(clamp(&ct, 1.0, 0.0).is_err());
        assert!(clamp(&ct, f64::NAN, 0.0).is_err());
    }

    #[test]
    fn gamma_preserves_range() {
        let g = values(&adjust_gamma(&img(vec![-1.0f32, 0.0, 1.0]), 2.0).unwrap());
        assert_eq!(g, vec![-1.0, -0.5, 1.0]);
        assert!(adjust_gamma(&img(vec![1.0f32]), 0.0).is_err());
    }
}
