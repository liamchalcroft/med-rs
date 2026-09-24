//! Parallel, numerically stable summary statistics over `f32` data.

use rayon::prelude::*;

const CHUNK: usize = 1 << 16;

/// Count, mean, and sum of squared deviations (Welford / Chan et al.).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub(crate) struct Moments {
    pub n: f64,
    pub mean: f64,
    pub m2: f64,
}

impl Moments {
    fn push(&mut self, x: f64) {
        self.n += 1.0;
        let delta = x - self.mean;
        self.mean += delta / self.n;
        self.m2 += delta * (x - self.mean);
    }

    fn merge(a: Self, b: Self) -> Self {
        if a.n == 0.0 {
            return b;
        }
        if b.n == 0.0 {
            return a;
        }
        let n = a.n + b.n;
        let delta = b.mean - a.mean;
        Self {
            n,
            mean: a.mean + delta * b.n / n,
            m2: a.m2 + b.m2 + delta * delta * a.n * b.n / n,
        }
    }

    /// Population standard deviation.
    pub fn std(&self) -> f64 {
        if self.n > 0.0 {
            (self.m2 / self.n).sqrt()
        } else {
            0.0
        }
    }
}

/// Moments of `f(x)` over the values where `f` returns `Some`.
pub(crate) fn moments<F>(data: &[f32], f: F) -> Moments
where
    F: Fn(f32) -> Option<f32> + Sync,
{
    crate::parallel::install(|| {
        data.par_chunks(CHUNK)
            .map(|chunk| {
                let mut m = Moments::default();
                for &v in chunk {
                    if let Some(x) = f(v) {
                        m.push(f64::from(x));
                    }
                }
                m
            })
            .reduce(Moments::default, Moments::merge)
    })
}

/// Minimum and maximum of `f(x)`, ignoring NaN. `None` if every value is NaN.
pub(crate) fn min_max<F>(data: &[f32], f: F) -> Option<(f32, f32)>
where
    F: Fn(f32) -> f32 + Sync,
{
    let (lo, hi) = crate::parallel::install(|| {
        data.par_chunks(CHUNK)
            .map(|chunk| {
                chunk
                    .iter()
                    .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), &v| {
                        let x = f(v);
                        // `min`/`max` return the non-NaN operand.
                        (lo.min(x), hi.max(x))
                    })
            })
            .reduce(
                || (f32::INFINITY, f32::NEG_INFINITY),
                |a, b| (a.0.min(b.0), a.1.max(b.1)),
            )
    });
    (lo <= hi).then_some((lo, hi))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn moments_are_stable_with_large_offsets() {
        let data: Vec<f32> = (0..1_000_003).map(|i| 1e6 + (i % 7) as f32).collect();
        let m = moments(&data, Some);
        let mean = 1e6 + (0..1_000_003u64).map(|i| (i % 7) as f64).sum::<f64>() / 1_000_003.0;
        assert!((m.mean - mean).abs() < 1e-6);
        let var: f64 = data
            .iter()
            .map(|&v| (f64::from(v) - mean).powi(2))
            .sum::<f64>()
            / 1_000_003.0;
        assert!((m.std() - var.sqrt()).abs() < 1e-6);
    }

    #[test]
    fn filters_and_nan_handling() {
        let data = [0.0f32, 2.0, 0.0, 4.0, f32::NAN];
        let m = moments(&data[..4], |v| (v != 0.0).then_some(v));
        assert_eq!((m.n, m.mean), (2.0, 3.0));
        assert_eq!(min_max(&data, |v| v), Some((0.0, 4.0)));
        assert_eq!(min_max(&[f32::NAN], |v| v), None);
        assert!(moments(&data, Some).mean.is_nan());
    }
}
