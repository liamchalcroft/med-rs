//! Parallel patch loading for training.
//!
//! [`FastLoader`] turns a list of volumes (optionally paired with label maps)
//! into a stream of fixed-size patches:
//!
//! * Patches are read crop-first: uncompressed files are memory-mapped and only
//!   the patch is copied, `.jvol` files decode only the chunks a patch touches,
//!   and gzipped files are decompressed once per volume per epoch.
//! * Worker threads load volumes ahead of the consumer, up to a fixed number
//!   of patches in flight.
//! * Output order is deterministic: for a given seed and epoch the sequence of
//!   patches is identical regardless of the number of workers.
//! * Patch shapes can be fixed or drawn per patch from weighted choices.
//!   Patches smaller than their shape (because the volume is smaller) are
//!   padded to it.
//! * Patches can be centred on foreground voxels, taken from label maps or
//!   from an intensity threshold, and volumes can be drawn with weights.
//! * An optional [`Pipeline`] runs in the workers, with spatial transforms
//!   applied identically to image and label.
//!
//! ```no_run
//! use medrs::loader::{FastLoader, LoaderConfig};
//!
//! let mut config = LoaderConfig::new([96, 96, 96]);
//! config.patches_per_volume = 4;
//! config.foreground_prob = Some(0.5);
//! config.seed = Some(0);
//! let loader = FastLoader::new(
//!     vec!["ct_000.nii.gz".into(), "ct_001.nii.gz".into()],
//!     Some(vec!["seg_000.nii.gz".into(), "seg_001.nii.gz".into()]),
//!     config,
//! )?;
//! for epoch in 0..10 {
//!     for patch in loader.epoch(epoch) {
//!         let patch = patch?;
//!         let (image, label) = (patch.image, patch.label);
//!     }
//! }
//! # Ok::<(), medrs::Error>(())
//! ```

use crate::error::{Error, Result};
use crate::nifti::io::Volume;
use crate::nifti::NiftiImage;
use crate::pipeline::Pipeline;
use crate::transforms::geometry::split_shape;
use crate::transforms::{crop, crop_or_pad, random_region, ForegroundSampler, Region};
use rand::distr::weighted::WeightedIndex;
use rand::distr::Distribution;
use rand::seq::SliceRandom;
use rand::{Rng, RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver};
use std::sync::{Arc, Condvar, Mutex, PoisonError};
use std::thread::JoinHandle;

/// Options for a [`FastLoader`].
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct LoaderConfig {
    /// Patch shapes along the first three axes. Each patch takes one of them,
    /// drawn with `patch_shape_weights`. Default: the shape given to
    /// [`LoaderConfig::new`].
    pub patch_shapes: Vec<[usize; 3]>,
    /// Relative probabilities of `patch_shapes`. `None` weighs them equally.
    pub patch_shape_weights: Option<Vec<f64>>,
    /// Patches drawn from each volume per epoch. Default 1.
    pub patches_per_volume: usize,
    /// Probability that a patch is centred on a foreground voxel rather than
    /// a background voxel. The foreground is given by `foreground_threshold`
    /// if set, and otherwise by the label maps (non-zero voxels). `None`
    /// draws patch positions uniformly. Default `None`.
    pub foreground_prob: Option<f64>,
    /// Makes the foreground the image voxels whose scaled value is greater
    /// than this, so `foreground_prob` works without label maps. Default
    /// `None`.
    pub foreground_threshold: Option<f64>,
    /// Sampling weight of each volume. When set, each epoch draws
    /// `volumes_per_epoch` volumes with replacement, in proportion to these
    /// weights (`shuffle` then has no effect). Default `None`.
    pub volume_weights: Option<Vec<f64>>,
    /// Volumes drawn per epoch. `None` uses each volume once per epoch (or,
    /// with `volume_weights`, draws as many volumes as there are). When set
    /// without `volume_weights`, volumes are drawn uniformly with
    /// replacement. Default `None`.
    pub volumes_per_epoch: Option<usize>,
    /// Value for padding patches of volumes smaller than `patch_shape`
    /// (labels are padded with 0). Default 0.
    pub pad_value: f64,
    /// Transforms applied to each patch in the workers. Default none.
    pub pipeline: Option<Pipeline>,
    /// Worker threads; 0 loads in the calling thread. Default: available
    /// CPUs, at most 8.
    pub workers: usize,
    /// Maximum number of patches loaded ahead of the consumer. Default
    /// `2 * workers` (at least 2).
    pub prefetch: usize,
    /// Shuffle the volume order each epoch. Default `true`.
    pub shuffle: bool,
    /// Seed for shuffling, patch positions, and random transforms. `None`
    /// picks a random seed when the loader is created. Default `None`.
    pub seed: Option<u64>,
}

impl LoaderConfig {
    /// Default options for patches of `patch_shape`.
    pub fn new(patch_shape: [usize; 3]) -> Self {
        let workers = std::thread::available_parallelism().map_or(1, |n| n.get().min(8));
        Self {
            patch_shapes: vec![patch_shape],
            patch_shape_weights: None,
            patches_per_volume: 1,
            foreground_prob: None,
            foreground_threshold: None,
            volume_weights: None,
            volumes_per_epoch: None,
            pad_value: 0.0,
            pipeline: None,
            workers,
            prefetch: (2 * workers).max(2),
            shuffle: true,
            seed: None,
        }
    }
}

/// One training patch.
#[derive(Debug, Clone)]
pub struct Patch {
    /// The image patch (after the pipeline, if any).
    pub image: NiftiImage,
    /// The label patch, if the loader has labels.
    pub label: Option<NiftiImage>,
    /// Index of the source volume in the loader's list.
    pub volume: usize,
    /// Region of the source volume the patch was read from (before padding
    /// and transforms).
    pub region: Region,
}

#[derive(Debug)]
struct Shared {
    images: Vec<PathBuf>,
    labels: Option<Vec<PathBuf>>,
    config: LoaderConfig,
    seed: u64,
    shape_weights: Option<WeightedIndex<f64>>,
    volume_weights: Option<WeightedIndex<f64>>,
}

/// Sampling weights for `n` choices, or an error naming `what`.
fn weights(what: &str, weights: Option<&Vec<f64>>, n: usize) -> Result<Option<WeightedIndex<f64>>> {
    let Some(w) = weights else {
        return Ok(None);
    };
    if w.len() != n {
        return Err(Error::InvalidArgument(format!(
            "{what} needs {n} weights, got {}",
            w.len()
        )));
    }
    if w.iter().any(|v| !v.is_finite()) {
        return Err(Error::InvalidArgument(format!("{what} must be finite")));
    }
    WeightedIndex::new(w).map(Some).map_err(|_| {
        Error::InvalidArgument(format!(
            "{what} must be non-negative with a positive sum, got {w:?}"
        ))
    })
}

/// Loads training patches in parallel. See the [module docs](self).
#[derive(Debug, Clone)]
pub struct FastLoader {
    shared: Arc<Shared>,
}

impl FastLoader {
    /// Create a loader over `images`, optionally paired with `labels` (same
    /// length and order).
    pub fn new(
        images: Vec<PathBuf>,
        labels: Option<Vec<PathBuf>>,
        config: LoaderConfig,
    ) -> Result<Self> {
        if images.is_empty() {
            return Err(Error::InvalidArgument(
                "the loader needs at least one volume".into(),
            ));
        }
        if let Some(labels) = &labels {
            if labels.len() != images.len() {
                return Err(Error::InvalidArgument(format!(
                    "{} images but {} labels",
                    images.len(),
                    labels.len()
                )));
            }
        }
        if config.patch_shapes.is_empty() {
            return Err(Error::InvalidArgument(
                "at least one patch shape is needed".into(),
            ));
        }
        if let Some(shape) = config.patch_shapes.iter().find(|s| s.contains(&0)) {
            return Err(Error::InvalidArgument(format!(
                "patch shape must be positive, got {shape:?}"
            )));
        }
        let shape_weights = weights(
            "patch_shape_weights",
            config.patch_shape_weights.as_ref(),
            config.patch_shapes.len(),
        )?;
        let volume_weights = weights(
            "volume_weights",
            config.volume_weights.as_ref(),
            images.len(),
        )?;
        if config.volumes_per_epoch == Some(0) {
            return Err(Error::InvalidArgument(
                "volumes_per_epoch must be at least 1".into(),
            ));
        }
        if config.patches_per_volume == 0 {
            return Err(Error::InvalidArgument(
                "patches_per_volume must be at least 1".into(),
            ));
        }
        if let Some(p) = config.foreground_prob {
            if labels.is_none() && config.foreground_threshold.is_none() {
                return Err(Error::InvalidArgument(
                    "foreground_prob requires label maps or a foreground_threshold".into(),
                ));
            }
            crate::transforms::check_probability(p)?;
        }
        if config.foreground_threshold.is_some_and(f64::is_nan) {
            return Err(Error::InvalidArgument(
                "foreground_threshold must not be NaN".into(),
            ));
        }
        if !config.pad_value.is_finite() {
            return Err(Error::InvalidArgument(format!(
                "pad value must be finite, got {}",
                config.pad_value
            )));
        }
        if let Some(pipeline) = &config.pipeline {
            pipeline.validate()?;
        }
        let seed = config.seed.unwrap_or_else(rand::random);
        Ok(Self {
            shared: Arc::new(Shared {
                images,
                labels,
                config,
                seed,
                shape_weights,
                volume_weights,
            }),
        })
    }

    /// Number of patches per epoch.
    pub fn len(&self) -> usize {
        self.shared.volumes_per_epoch() * self.shared.config.patches_per_volume
    }

    /// Always `false`: a loader has at least one volume.
    pub fn is_empty(&self) -> bool {
        false
    }

    /// The seed in use (the configured one, or the one picked at creation).
    pub fn seed(&self) -> u64 {
        self.shared.seed
    }

    /// The loader's options.
    pub fn config(&self) -> &LoaderConfig {
        &self.shared.config
    }

    /// Iterate over one epoch. Different epochs shuffle and sample
    /// differently; the same epoch number always gives the same patches.
    ///
    /// Dropping the iterator stops its workers; it waits at most for the
    /// volumes currently being loaded.
    pub fn epoch(&self, epoch: u64) -> Epoch {
        Epoch::new(Arc::clone(&self.shared), epoch)
    }
}

impl Shared {
    fn volumes_per_epoch(&self) -> usize {
        self.config.volumes_per_epoch.unwrap_or(self.images.len())
    }

    /// The patch shape for the next patch.
    fn patch_shape<R: Rng>(&self, rng: &mut R) -> [usize; 3] {
        let shapes = &self.config.patch_shapes;
        match (&self.shape_weights, shapes.len()) {
            (_, 1) => shapes[0],
            (Some(w), _) => shapes[w.sample(rng)],
            (None, n) => shapes[rng.random_range(0..n)],
        }
    }
}

fn epoch_rng(seed: u64, epoch: u64) -> ChaCha8Rng {
    ChaCha8Rng::seed_from_u64(seed ^ epoch.wrapping_mul(0x9E37_79B9_7F4A_7C15))
}

/// Seeded RNG for the volume at `position` in an epoch's order.
fn unit_rng(seed: u64, epoch: u64, position: usize) -> ChaCha8Rng {
    let mut rng = epoch_rng(seed, epoch);
    rng.set_stream(position as u64 + 1);
    rng
}

/// The volumes of an epoch, in order.
fn epoch_order(shared: &Shared, epoch: u64) -> Vec<usize> {
    let mut rng = epoch_rng(shared.seed, epoch);
    let n = shared.images.len();
    let count = shared.volumes_per_epoch();
    match (&shared.volume_weights, shared.config.volumes_per_epoch) {
        (Some(w), _) => (0..count).map(|_| w.sample(&mut rng)).collect(),
        (None, Some(_)) => (0..count).map(|_| rng.random_range(0..n)).collect(),
        (None, None) => {
            let mut order: Vec<usize> = (0..n).collect();
            if shared.config.shuffle {
                order.shuffle(&mut rng);
            }
            order
        }
    }
}

/// Load all patches of the volume at `position` in the epoch's order.
fn load_volume(shared: &Shared, epoch: u64, position: usize, volume: usize) -> Vec<Result<Patch>> {
    match try_load_volume(shared, epoch, position, volume) {
        Ok(patches) => patches.into_iter().map(Ok).collect(),
        Err(e) => vec![Err(e)],
    }
}

fn try_load_volume(
    shared: &Shared,
    epoch: u64,
    position: usize,
    volume: usize,
) -> Result<Vec<Patch>> {
    let config = &shared.config;
    let mut rng = unit_rng(shared.seed, epoch, position);
    let image = Volume::open(&shared.images[volume], false)?;
    let label = match &shared.labels {
        Some(labels) => Some(Volume::open(&labels[volume], false)?),
        None => None,
    };
    let (spatial, _) = split_shape(&image.header().shape());
    if let Some(label) = &label {
        let (label_spatial, _) = split_shape(&label.header().shape());
        if label_spatial != spatial {
            return Err(Error::ShapeMismatch(format!(
                "{}: label shape {label_spatial:?} does not match image shape {spatial:?}",
                shared
                    .labels
                    .as_ref()
                    .map_or_else(String::new, |l| l[volume].display().to_string())
            )));
        }
    }
    // Whole images already in memory (for foreground sampling); patches are
    // cropped from them instead of being read again.
    let (mut whole_image, mut whole_label) = (None, None);
    let mut sampler = match (config.foreground_prob, config.foreground_threshold, &label) {
        (Some(_), Some(threshold), _) => {
            let full = image.image()?;
            let sampler = ForegroundSampler::from_threshold(&full, threshold)?;
            whole_image = Some(full);
            Some(sampler)
        }
        (Some(_), None, Some(label)) => {
            let full = label.image()?;
            let sampler = ForegroundSampler::from_label(&full)?;
            whole_label = Some(full);
            Some(sampler)
        }
        _ => None,
    };
    let draws = (0..config.patches_per_volume)
        .map(|_| {
            let shape = shared.patch_shape(&mut rng);
            let region = match (&mut sampler, config.foreground_prob) {
                (Some(s), Some(p)) => s.sample(shape, p, &mut rng)?,
                _ => random_region(spatial, shape, &mut rng)?,
            };
            Ok((shape, region))
        })
        .collect::<Result<Vec<_>>>()?;
    let read = |volume: &Volume, whole: &Option<NiftiImage>, region: Region| match whole {
        Some(full) => crop(full, region.offset, region.shape),
        None => volume.region(region.offset, region.shape),
    };
    draws
        .into_iter()
        .map(|(patch, region)| {
            let mut img = read(&image, &whole_image, region)?;
            let mut lab = match &label {
                Some(l) => Some(read(l, &whole_label, region)?),
                None => None,
            };
            if region.shape != patch {
                img = crop_or_pad(&img, patch, config.pad_value)?;
                lab = lab.map(|l| crop_or_pad(&l, patch, 0.0)).transpose()?;
            }
            if let Some(pipeline) = &config.pipeline {
                match lab {
                    Some(l) => {
                        let (a, b) = pipeline.apply_pair(&img, &l, &mut rng)?;
                        img = a;
                        lab = Some(b);
                    }
                    None => img = pipeline.apply_with_rng(&img, &mut rng)?,
                }
            }
            Ok(Patch {
                image: img,
                label: lab,
                volume,
                region,
            })
        })
        .collect()
}

/// Worker coordination: how many volumes have been claimed, and how many the
/// consumer has taken (which bounds how far ahead workers may run).
#[derive(Default)]
struct Window {
    claimed: usize,
    consumed: usize,
}

struct Workers {
    window: Arc<(Mutex<Window>, Condvar)>,
    stop: Arc<AtomicBool>,
    receiver: Option<Receiver<(usize, Vec<Result<Patch>>)>>,
    handles: Vec<JoinHandle<()>>,
    pending: HashMap<usize, Vec<Result<Patch>>>,
}

/// Iterator over the patches of one epoch, from [`FastLoader::epoch`].
pub struct Epoch {
    shared: Arc<Shared>,
    number: u64,
    order: Vec<usize>,
    next: usize,
    current: VecDeque<Result<Patch>>,
    workers: Option<Workers>,
}

impl Epoch {
    fn new(shared: Arc<Shared>, epoch: u64) -> Self {
        let order = epoch_order(&shared, epoch);
        let workers = (shared.config.workers > 0).then(|| spawn_workers(&shared, epoch, &order));
        Self {
            shared,
            number: epoch,
            order,
            next: 0,
            current: VecDeque::new(),
            workers,
        }
    }

    /// Patches not yet returned in this epoch.
    pub fn remaining(&self) -> usize {
        (self.order.len() - self.next) * self.shared.config.patches_per_volume + self.current.len()
    }

    /// Stop the workers and wait for them to finish their current volume.
    fn stop(&mut self) {
        if let Some(mut w) = self.workers.take() {
            w.stop.store(true, Ordering::Relaxed);
            let (_, cvar) = &*w.window;
            cvar.notify_all();
            // Workers blocked on sending see a closed channel and exit.
            w.receiver.take();
            for handle in w.handles.drain(..) {
                let _ = handle.join();
            }
        }
    }

    fn next_volume(&mut self) -> Option<Vec<Result<Patch>>> {
        let position = self.next;
        let volume = *self.order.get(position)?;
        self.next += 1;
        let Some(w) = &mut self.workers else {
            return Some(load_volume(&self.shared, self.number, position, volume));
        };
        let patches = loop {
            if let Some(p) = w.pending.remove(&position) {
                break p;
            }
            match w.receiver.as_ref()?.recv() {
                Ok((pos, p)) if pos == position => break p,
                Ok((pos, p)) => {
                    w.pending.insert(pos, p);
                }
                Err(_) => {
                    return Some(vec![Err(Error::Internal(
                        "loader worker exited unexpectedly".into(),
                    ))])
                }
            }
        };
        let (lock, cvar) = &*w.window;
        lock.lock().unwrap_or_else(PoisonError::into_inner).consumed = position + 1;
        cvar.notify_all();
        Some(patches)
    }
}

fn spawn_workers(shared: &Arc<Shared>, epoch: u64, order: &[usize]) -> Workers {
    let config = &shared.config;
    let ahead = config.prefetch.div_ceil(config.patches_per_volume).max(1);
    let window = Arc::new((Mutex::new(Window::default()), Condvar::new()));
    let stop = Arc::new(AtomicBool::new(false));
    let (sender, receiver) = mpsc::channel();
    let order: Arc<[usize]> = order.into();
    let handles = (0..config.workers.min(order.len()))
        .map(|i| {
            let (shared, window, stop, sender, order) = (
                Arc::clone(shared),
                Arc::clone(&window),
                Arc::clone(&stop),
                sender.clone(),
                Arc::clone(&order),
            );
            std::thread::Builder::new()
                .name(format!("medrs-loader-{i}"))
                .spawn(move || loop {
                    let position = {
                        let (lock, cvar) = &*window;
                        let mut w = lock.lock().unwrap_or_else(PoisonError::into_inner);
                        loop {
                            if stop.load(Ordering::Relaxed) || w.claimed >= order.len() {
                                return;
                            }
                            if w.claimed < w.consumed + ahead {
                                w.claimed += 1;
                                break w.claimed - 1;
                            }
                            w = cvar.wait(w).unwrap_or_else(PoisonError::into_inner);
                        }
                    };
                    let patches = load_volume(&shared, epoch, position, order[position]);
                    if sender.send((position, patches)).is_err() {
                        return;
                    }
                })
        })
        .filter_map(std::io::Result::ok)
        .collect();
    Workers {
        window,
        stop,
        receiver: Some(receiver),
        handles,
        pending: HashMap::new(),
    }
}

impl Iterator for Epoch {
    type Item = Result<Patch>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(patch) = self.current.pop_front() {
                return Some(patch);
            }
            let patches = self.next_volume()?;
            self.current.extend(patches);
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(self.remaining()))
    }
}

impl Drop for Epoch {
    fn drop(&mut self) {
        self.stop();
    }
}

impl std::fmt::Debug for Epoch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Epoch")
            .field("epoch", &self.number)
            .field("remaining", &self.remaining())
            .field(
                "workers",
                &self.workers.as_ref().map_or(0, |w| w.handles.len()),
            )
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nifti::header::Affine;
    use crate::nifti::DataType;
    use ndarray::{ArrayD, IxDyn, ShapeBuilder};
    use tempfile::TempDir;

    const EYE: Affine = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];

    /// Volumes whose voxel values encode (volume, x, y, z), with labels that
    /// mark one corner.
    fn dataset(dir: &TempDir, n: usize, shape: [usize; 3]) -> (Vec<PathBuf>, Vec<PathBuf>) {
        let mut images = Vec::new();
        let mut labels = Vec::new();
        for v in 0..n {
            let len: usize = shape.iter().product();
            let values: Vec<f32> = (0..len).map(|i| (v * 1_000_000 + i) as f32).collect();
            let arr = ArrayD::from_shape_vec(IxDyn(&shape).f(), values).unwrap();
            let img = NiftiImage::from_array(arr, EYE).unwrap();
            let ext = if v % 2 == 0 { "nii" } else { "nii.gz" };
            let path = dir.path().join(format!("img{v}.{ext}"));
            crate::nifti::save(&img, &path).unwrap();
            images.push(path);
            let mut lab = ArrayD::<u8>::zeros(IxDyn(&shape).f());
            lab[[0, 0, 0]] = 1;
            let lpath = dir.path().join(format!("lab{v}.nii"));
            crate::nifti::save(&NiftiImage::from_array(lab, EYE).unwrap(), &lpath).unwrap();
            labels.push(lpath);
        }
        (images, labels)
    }

    fn collect(loader: &FastLoader, epoch: u64) -> Vec<(usize, Region, Vec<f32>)> {
        loader
            .epoch(epoch)
            .map(|p| {
                let p = p.unwrap();
                (
                    p.volume,
                    p.region,
                    p.image.to_f32().unwrap().iter().copied().collect(),
                )
            })
            .collect()
    }

    #[test]
    fn order_is_deterministic_across_worker_counts() {
        let dir = TempDir::new().unwrap();
        let (images, _) = dataset(&dir, 7, [10, 9, 8]);
        let make = |workers| {
            let mut c = LoaderConfig::new([4, 4, 4]);
            c.patches_per_volume = 3;
            c.workers = workers;
            c.prefetch = 2;
            c.seed = Some(11);
            FastLoader::new(images.clone(), None, c).unwrap()
        };
        let inline = collect(&make(0), 1);
        assert_eq!(inline.len(), 21);
        assert_eq!(collect(&make(1), 1), inline);
        assert_eq!(collect(&make(4), 1), inline);
        assert_ne!(collect(&make(4), 2), inline);
        // Each patch is the region it claims to be.
        for (v, region, values) in &inline {
            let x = region.offset;
            let expected = (*v * 1_000_000 + x[0] + 10 * (x[1] + 9 * x[2])) as f32;
            assert_eq!(values[0], expected);
        }
    }

    #[test]
    fn padding_labels_and_pipeline() {
        let dir = TempDir::new().unwrap();
        let (images, labels) = dataset(&dir, 2, [6, 5, 4]);
        let mut c = LoaderConfig::new([8, 8, 8]);
        c.foreground_prob = Some(1.0);
        c.pipeline = Some(
            Pipeline::new()
                .random_flip(&[0, 1, 2], 0.5)
                .cast(DataType::Float16),
        );
        c.workers = 2;
        let loader = FastLoader::new(images, Some(labels), c).unwrap();
        for patch in loader.epoch(0) {
            let patch = patch.unwrap();
            assert_eq!(patch.image.shape(), &[8, 8, 8]);
            assert_eq!(patch.image.dtype(), DataType::Float16);
            let label = patch.label.unwrap();
            assert_eq!(label.shape(), &[8, 8, 8]);
            assert_eq!(label.dtype(), DataType::UInt8);
            assert_eq!(patch.region.shape, [6, 5, 4]);
            // The labelled corner voxel is still in the patch.
            assert_eq!(
                label
                    .to_f32()
                    .unwrap()
                    .iter()
                    .filter(|&&v| v == 1.0)
                    .count(),
                1
            );
        }
    }

    #[test]
    fn dropping_a_partial_epoch_does_not_hang() {
        let dir = TempDir::new().unwrap();
        let (images, _) = dataset(&dir, 6, [8, 8, 8]);
        let mut c = LoaderConfig::new([4, 4, 4]);
        c.workers = 3;
        c.prefetch = 1;
        let loader = FastLoader::new(images, None, c).unwrap();
        let mut epoch = loader.epoch(0);
        assert!(epoch.next().is_some());
        drop(epoch);
        assert_eq!(loader.epoch(1).count(), 6);
    }

    #[test]
    fn errors_are_reported_per_volume() {
        let dir = TempDir::new().unwrap();
        let (mut images, _) = dataset(&dir, 2, [8, 8, 8]);
        images.insert(1, dir.path().join("missing.nii"));
        let mut c = LoaderConfig::new([4, 4, 4]);
        c.shuffle = false;
        let loader = FastLoader::new(images, None, c).unwrap();
        let results: Vec<_> = loader.epoch(0).collect();
        assert_eq!(results.len(), 3);
        assert!(results[0].is_ok() && results[1].is_err() && results[2].is_ok());
    }

    #[test]
    fn patch_shapes_are_drawn_with_their_weights() {
        let dir = TempDir::new().unwrap();
        let (images, _) = dataset(&dir, 3, [12, 10, 9]);
        let make = |workers, weights: Option<Vec<f64>>| {
            let mut c = LoaderConfig::new([4, 4, 4]);
            c.patch_shapes = vec![[4, 4, 4], [8, 3, 2], [20, 20, 20]];
            c.patch_shape_weights = weights;
            c.patches_per_volume = 100;
            c.workers = workers;
            c.seed = Some(5);
            FastLoader::new(images.clone(), None, c).unwrap()
        };
        let loader = make(0, Some(vec![3.0, 1.0, 0.0]));
        let mut counts = HashMap::new();
        for patch in loader.epoch(0) {
            let shape = patch.unwrap().image.shape().to_vec();
            *counts.entry(shape).or_insert(0) += 1;
        }
        let (small, flat) = (counts[&vec![4, 4, 4]], counts[&vec![8, 3, 2]]);
        assert_eq!(small + flat, 300, "{counts:?}");
        assert!((190..=260).contains(&small), "{counts:?}");
        // Unweighted choices include the shape larger than the volume, padded.
        let shapes: Vec<Vec<usize>> = make(2, None)
            .epoch(0)
            .map(|p| p.unwrap().image.shape().to_vec())
            .collect();
        assert!(shapes.contains(&vec![20, 20, 20]));
        assert_eq!(collect(&make(0, None), 3), collect(&make(3, None), 3));
    }

    #[test]
    fn volumes_are_drawn_with_replacement_by_weight() {
        let dir = TempDir::new().unwrap();
        let (images, _) = dataset(&dir, 3, [12, 12, 12]);
        let make = |workers| {
            let mut c = LoaderConfig::new([4, 4, 4]);
            c.volume_weights = Some(vec![1.0, 0.0, 3.0]);
            c.volumes_per_epoch = Some(80);
            c.workers = workers;
            c.seed = Some(9);
            FastLoader::new(images.clone(), None, c).unwrap()
        };
        let loader = make(0);
        assert_eq!(loader.len(), 80);
        let patches = collect(&loader, 0);
        assert_eq!(patches.len(), 80);
        let from = |v| patches.iter().filter(|p| p.0 == v).count();
        assert_eq!(from(1), 0);
        assert!((40..=75).contains(&from(2)), "{}", from(2));
        // Repeated draws of a volume get different patches.
        let regions: std::collections::HashSet<_> =
            patches.iter().filter(|p| p.0 == 2).map(|p| p.1).collect();
        assert!(regions.len() > 10);
        assert_eq!(collect(&make(4), 0), patches);

        // Without weights, `volumes_per_epoch` draws uniformly.
        let mut c = LoaderConfig::new([4, 4, 4]);
        c.volumes_per_epoch = Some(7);
        c.patches_per_volume = 2;
        assert_eq!(
            FastLoader::new(images, None, c).unwrap().epoch(0).count(),
            14
        );
    }

    #[test]
    fn threshold_foreground_needs_no_labels() {
        let dir = TempDir::new().unwrap();
        let mut arr = ArrayD::<f32>::zeros(IxDyn(&[30, 30, 30]).f());
        for x in 20..24 {
            for y in 3..7 {
                for z in 10..13 {
                    arr[[x, y, z]] = 100.0;
                }
            }
        }
        let path = dir.path().join("bright.nii.gz");
        crate::nifti::save(&NiftiImage::from_array(arr, EYE).unwrap(), &path).unwrap();
        let mut c = LoaderConfig::new([6, 6, 6]);
        c.foreground_prob = Some(1.0);
        c.foreground_threshold = Some(50.0);
        c.patches_per_volume = 20;
        let loader = FastLoader::new(vec![path], None, c).unwrap();
        for patch in loader.epoch(0) {
            let patch = patch.unwrap();
            assert!(patch.label.is_none());
            let max = patch.image.to_f32().unwrap().fold(0.0f32, |m, &v| m.max(v));
            assert_eq!(max, 100.0, "{:?}", patch.region);
        }
    }

    #[test]
    fn invalid_configuration_is_rejected() {
        let path = vec![PathBuf::from("a.nii")];
        assert!(FastLoader::new(vec![], None, LoaderConfig::new([4, 4, 4])).is_err());
        assert!(FastLoader::new(path.clone(), None, LoaderConfig::new([0, 4, 4])).is_err());
        let mut c = LoaderConfig::new([4, 4, 4]);
        c.foreground_prob = Some(0.5);
        assert!(FastLoader::new(path.clone(), None, c).is_err());
        assert!(FastLoader::new(path.clone(), Some(vec![]), LoaderConfig::new([4, 4, 4])).is_err());
        let rejected = |f: &dyn Fn(&mut LoaderConfig)| {
            let mut c = LoaderConfig::new([4, 4, 4]);
            f(&mut c);
            FastLoader::new(path.clone(), None, c).is_err()
        };
        assert!(rejected(&|c| c.patch_shapes.clear()));
        assert!(rejected(&|c| c.patch_shape_weights = Some(vec![1.0, 1.0])));
        assert!(rejected(&|c| c.volume_weights = Some(vec![0.0])));
        assert!(rejected(&|c| c.volume_weights = Some(vec![-1.0])));
        assert!(rejected(&|c| c.volume_weights = Some(vec![f64::NAN])));
        assert!(rejected(&|c| c.volumes_per_epoch = Some(0)));
        assert!(rejected(&|c| c.foreground_threshold = Some(f64::NAN)));
        assert!(!rejected(&|c| {
            c.foreground_prob = Some(0.5);
            c.foreground_threshold = Some(0.0);
        }));
    }
}
