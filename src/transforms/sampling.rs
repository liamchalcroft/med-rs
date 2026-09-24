//! Choosing patch regions for training.
//!
//! All functions return regions that lie inside the volume. When the patch is
//! larger than the volume along an axis, the region covers the whole axis and
//! the caller pads (the [`FastLoader`](crate::loader::FastLoader) does this
//! automatically).

use super::geometry::split_shape;
use crate::error::{Error, Result};
use crate::nifti::element::dispatch_dtype;
use crate::nifti::NiftiImage;
use rand::Rng;
use rayon::prelude::*;

/// A box of voxels: `shape` voxels starting at `offset` along the first three
/// axes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Region {
    /// First voxel of the region.
    pub offset: [usize; 3],
    /// Number of voxels along each axis.
    pub shape: [usize; 3],
}

impl Region {
    /// One past the last voxel along each axis.
    pub fn end(&self) -> [usize; 3] {
        std::array::from_fn(|i| self.offset[i] + self.shape[i])
    }
}

fn clipped(volume: [usize; 3], patch: [usize; 3]) -> Result<[usize; 3]> {
    if patch.contains(&0) || volume.contains(&0) {
        return Err(Error::InvalidArgument(format!(
            "patch {patch:?} and volume {volume:?} must be non-empty"
        )));
    }
    Ok(std::array::from_fn(|i| patch[i].min(volume[i])))
}

/// The region of `patch` voxels centred in `volume`.
pub fn center_region(volume: [usize; 3], patch: [usize; 3]) -> Result<Region> {
    let shape = clipped(volume, patch)?;
    Ok(Region {
        offset: std::array::from_fn(|i| (volume[i] - shape[i]) / 2),
        shape,
    })
}

/// A region of `patch` voxels at a uniformly random position in `volume`.
pub fn random_region<R: Rng + ?Sized>(
    volume: [usize; 3],
    patch: [usize; 3],
    rng: &mut R,
) -> Result<Region> {
    let shape = clipped(volume, patch)?;
    Ok(Region {
        offset: std::array::from_fn(|i| rng.random_range(0..=volume[i] - shape[i])),
        shape,
    })
}

/// The region of `patch` voxels centred as closely as possible on `center`
/// while staying inside `volume`.
pub fn region_around(center: [usize; 3], volume: [usize; 3], patch: [usize; 3]) -> Result<Region> {
    let shape = clipped(volume, patch)?;
    Ok(Region {
        offset: std::array::from_fn(|i| {
            center[i]
                .saturating_sub(shape[i] / 2)
                .min(volume[i] - shape[i])
        }),
        shape,
    })
}

/// Sample `count` regions for segmentation training (MONAI's
/// `RandCropByPosNegLabel`): each region is centred on a random foreground
/// voxel (non-zero in `label`) with probability `foreground_prob`, and on a
/// random background voxel otherwise. If the label has no foreground (or no
/// background), every region uses the class that exists.
///
/// This is shorthand for [`ForegroundSampler::from_label`] followed by
/// `count` calls to [`ForegroundSampler::sample`].
pub fn sample_label_regions<R: Rng + ?Sized>(
    label: &NiftiImage,
    patch: [usize; 3],
    count: usize,
    foreground_prob: f64,
    rng: &mut R,
) -> Result<Vec<Region>> {
    super::augment::check_probability(foreground_prob)?;
    let mut sampler = ForegroundSampler::from_label(label)?;
    (0..count)
        .map(|_| sampler.sample(patch, foreground_prob, rng))
        .collect()
}

/// Draws patch regions centred on foreground or background voxels.
///
/// The foreground is computed once, from a label map or from an intensity
/// threshold, after which any number of regions of any shape can be drawn.
/// Voxels are drawn by rejection sampling when their class is common and from
/// an index list when it is rare, so memory stays bounded by a small fraction
/// of the volume. For 4D images a voxel is foreground if it is foreground in
/// any volume.
pub struct ForegroundSampler {
    mask: ForegroundMask,
    volume: [usize; 3],
    lists: [Option<Vec<u32>>; 2],
}

impl std::fmt::Debug for ForegroundSampler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ForegroundSampler")
            .field("volume", &self.volume)
            .field("foreground", &self.mask.foreground)
            .finish_non_exhaustive()
    }
}

impl ForegroundSampler {
    /// Foreground voxels are those that are non-zero in `label` (after
    /// scaling).
    pub fn from_label(label: &NiftiImage) -> Result<Self> {
        let (slope, inter) = label.header().scaling();
        // The stored value that represents a scaled zero.
        let zero = -inter / slope;
        Self::new(label, move |stored| stored != zero)
    }

    /// Foreground voxels are those of `image` whose scaled value is greater
    /// than `threshold`, for example to favour anatomy over air.
    pub fn from_threshold(image: &NiftiImage, threshold: f64) -> Result<Self> {
        if threshold.is_nan() {
            return Err(Error::InvalidArgument(
                "foreground threshold must not be NaN".into(),
            ));
        }
        let (slope, inter) = image.header().scaling();
        Self::new(image, move |stored| stored * slope + inter > threshold)
    }

    fn new(image: &NiftiImage, is_foreground: impl Fn(f64) -> bool + Sync) -> Result<Self> {
        let (volume, _) = split_shape(image.shape());
        Ok(Self {
            mask: ForegroundMask::new(image, is_foreground)?,
            volume,
            lists: [None, None],
        })
    }

    /// Number of foreground voxels.
    pub fn foreground(&self) -> usize {
        self.mask.foreground
    }

    /// A region of `patch` voxels (clipped to the volume) centred on a
    /// foreground voxel with probability `foreground_prob`, and on a
    /// background voxel otherwise. If one class is empty, the other is used.
    pub fn sample<R: Rng + ?Sized>(
        &mut self,
        patch: [usize; 3],
        foreground_prob: f64,
        rng: &mut R,
    ) -> Result<Region> {
        super::augment::check_probability(foreground_prob)?;
        let volume = self.volume;
        clipped(volume, patch)?;
        let want_fg = rng.random_bool(foreground_prob);
        let flat = self.voxel(want_fg, rng);
        let center = [
            flat % volume[0],
            (flat / volume[0]) % volume[1],
            flat / (volume[0] * volume[1]),
        ];
        region_around(center, volume, patch)
    }

    fn voxel<R: Rng + ?Sized>(&mut self, want_fg: bool, rng: &mut R) -> usize {
        let fg = self.mask.foreground;
        let total = self.mask.spatial;
        let class = match (fg, total - fg) {
            (0, _) => false,
            (_, 0) => true,
            _ => want_fg,
        };
        let size = if class { fg } else { total - fg };
        if size as f64 >= RARE * total as f64 || u32::try_from(total).is_err() {
            loop {
                let i = rng.random_range(0..total);
                if self.mask.get(i) == class {
                    return i;
                }
            }
        }
        let mask = &self.mask;
        let list = self.lists[usize::from(class)].get_or_insert_with(|| {
            (0..total)
                .filter(|&i| mask.get(i) == class)
                .map(|i| i as u32)
                .collect()
        });
        list[rng.random_range(0..list.len())] as usize
    }
}

/// Below this fraction a class is sampled from an explicit index list.
const RARE: f64 = 1.0 / 64.0;

/// Per-voxel foreground bits over the spatial grid of an image.
struct ForegroundMask {
    bits: Vec<u64>,
    spatial: usize,
    foreground: usize,
}

impl ForegroundMask {
    /// `is_foreground` receives stored (unscaled) values.
    fn new(image: &NiftiImage, is_foreground: impl Fn(f64) -> bool + Sync) -> Result<Self> {
        let (_, volumes) = split_shape(image.shape());
        let spatial: usize = split_shape(image.shape()).0.iter().product();
        let words = spatial.div_ceil(64);
        let mut bits = vec![0u64; words];
        dispatch_dtype!(image.dtype(), T => {
            let values = image.elements::<T>()?.as_cow();
            crate::parallel::install(|| {
                bits.par_iter_mut().enumerate().for_each(|(w, word)| {
                    let start = w * 64;
                    let end = (start + 64).min(spatial);
                    for i in start..end {
                        let fg = (0..volumes).any(|v| {
                            let x: T = values[v * spatial + i];
                            is_foreground(crate::nifti::element::sealed::Sealed::to_f64(x))
                        });
                        *word |= u64::from(fg) << (i - start);
                    }
                });
            });
        });
        let foreground = bits.iter().map(|w| w.count_ones() as usize).sum();
        Ok(Self {
            bits,
            spatial,
            foreground,
        })
    }

    fn get(&self, i: usize) -> bool {
        self.bits[i / 64] >> (i % 64) & 1 == 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nifti::header::Affine;
    use ndarray::{ArrayD, IxDyn, ShapeBuilder};
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    const EYE: Affine = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];

    #[test]
    fn regions_fit_and_shift_at_borders() {
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        for _ in 0..100 {
            let r = random_region([10, 5, 7], [4, 8, 7], &mut rng).unwrap();
            assert_eq!(r.shape, [4, 5, 7]);
            assert!(r.end()[0] <= 10 && r.offset[1] == 0 && r.offset[2] == 0);
        }
        assert_eq!(
            center_region([10, 10, 10], [4, 4, 4]).unwrap().offset,
            [3, 3, 3]
        );
        let edge = region_around([0, 9, 5], [10, 10, 10], [4, 4, 4]).unwrap();
        assert_eq!(edge.offset, [0, 6, 3]);
        assert!(random_region([4, 4, 4], [0, 1, 1], &mut rng).is_err());
    }

    fn label_with_cube() -> NiftiImage {
        // 32^3 volume with a 2^3 foreground cube at (20..22)^3: 0.02% foreground.
        let mut arr = ArrayD::<u8>::zeros(IxDyn(&[32, 32, 32]).f());
        for x in 20..22 {
            for y in 20..22 {
                for z in 20..22 {
                    arr[[x, y, z]] = 3;
                }
            }
        }
        NiftiImage::from_array(arr, EYE).unwrap()
    }

    #[test]
    fn foreground_probability_is_respected() {
        let label = label_with_cube();
        let mut rng = ChaCha8Rng::seed_from_u64(5);
        let contains_fg = |r: &Region| (0..3).all(|i| r.offset[i] <= 21 && r.end()[i] > 20);
        for (prob, lo, hi) in [(1.0, 400, 400), (0.0, 0, 20), (0.8, 290, 350)] {
            let regions = sample_label_regions(&label, [8, 8, 8], 400, prob, &mut rng).unwrap();
            let n = regions.iter().filter(|r| contains_fg(r)).count();
            assert!((lo..=hi).contains(&n), "prob {prob}: {n}");
            assert!(regions.iter().all(|r| r.shape == [8, 8, 8]));
        }
    }

    #[test]
    fn labels_without_foreground_fall_back_to_background() {
        let empty =
            NiftiImage::from_array(ArrayD::<u8>::zeros(IxDyn(&[8, 8, 8]).f()), EYE).unwrap();
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let r = sample_label_regions(&empty, [4, 4, 4], 5, 1.0, &mut rng).unwrap();
        assert_eq!(r.len(), 5);
        let full = NiftiImage::from_array(ArrayD::<u8>::ones(IxDyn(&[8, 8, 8]).f()), EYE).unwrap();
        assert_eq!(
            sample_label_regions(&full, [4, 4, 4], 5, 0.0, &mut rng)
                .unwrap()
                .len(),
            5
        );
    }

    #[test]
    fn threshold_sampler_centres_on_bright_voxels_with_any_shape() {
        // A bright 4^3 block in a dark 40x40x30 image, stored scaled.
        let mut arr = ArrayD::<i16>::zeros(IxDyn(&[40, 40, 30]).f());
        for x in 30..34 {
            for y in 5..9 {
                for z in 10..14 {
                    arr[[x, y, z]] = 500;
                }
            }
        }
        let mut image = NiftiImage::from_array(arr, EYE).unwrap();
        image.header_mut().scl_slope = 0.5;
        let mut rng = ChaCha8Rng::seed_from_u64(3);
        // 500 * 0.5 = 250 is above 200 but not above 300.
        let mut sampler = ForegroundSampler::from_threshold(&image, 200.0).unwrap();
        assert_eq!(sampler.foreground(), 64);
        for patch in [[8, 8, 8], [16, 4, 2], [40, 40, 40]] {
            let r = sampler.sample(patch, 1.0, &mut rng).unwrap();
            assert_eq!(
                r.shape,
                std::array::from_fn(|i| patch[i].min([40, 40, 30][i]))
            );
            assert!(r.offset[0] < 34 && r.end()[0] > 30, "{r:?}");
            assert!(r.offset[2] < 14 && r.end()[2] > 10, "{r:?}");
        }
        let none = ForegroundSampler::from_threshold(&image, 300.0).unwrap();
        assert_eq!(none.foreground(), 0);
        assert!(ForegroundSampler::from_threshold(&image, f64::NAN).is_err());
    }
}
