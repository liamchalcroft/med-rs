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
//! * Patches smaller than the requested shape (because the volume is smaller)
//!   are padded to it, so every patch has the same shape.
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
use crate::transforms::{crop_or_pad, random_region, sample_label_regions, Region};
use rand::seq::SliceRandom;
use rand::SeedableRng;
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
    /// Shape of every patch along the first three axes.
    pub patch_shape: [usize; 3],
    /// Patches drawn from each volume per epoch. Default 1.
    pub patches_per_volume: usize,
    /// With labels: probability that a patch is centred on a foreground
    /// (non-zero) label voxel rather than a background voxel. `None` draws
    /// patch positions uniformly. Default `None`.
    pub foreground_prob: Option<f64>,
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
            patch_shape,
            patches_per_volume: 1,
            foreground_prob: None,
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
        if config.patch_shape.contains(&0) {
            return Err(Error::InvalidArgument(format!(
                "patch shape must be positive, got {:?}",
                config.patch_shape
            )));
        }
        if config.patches_per_volume == 0 {
            return Err(Error::InvalidArgument(
                "patches_per_volume must be at least 1".into(),
            ));
        }
        if let Some(p) = config.foreground_prob {
            if labels.is_none() {
                return Err(Error::InvalidArgument(
                    "foreground_prob requires label maps".into(),
                ));
            }
            crate::transforms::check_probability(p)?;
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
            }),
        })
    }

    /// Number of patches per epoch.
    pub fn len(&self) -> usize {
        self.shared.images.len() * self.shared.config.patches_per_volume
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

/// Seeded RNG for one (epoch, volume) pair.
fn volume_rng(seed: u64, epoch: u64, volume: usize) -> ChaCha8Rng {
    let mut rng = ChaCha8Rng::seed_from_u64(seed ^ epoch.wrapping_mul(0x9E37_79B9_7F4A_7C15));
    rng.set_stream(volume as u64 + 1);
    rng
}

fn epoch_order(shared: &Shared, epoch: u64) -> Vec<usize> {
    let mut order: Vec<usize> = (0..shared.images.len()).collect();
    if shared.config.shuffle {
        let mut rng =
            ChaCha8Rng::seed_from_u64(shared.seed ^ epoch.wrapping_mul(0x9E37_79B9_7F4A_7C15));
        order.shuffle(&mut rng);
    }
    order
}

/// Load all patches of one volume.
fn load_volume(shared: &Shared, epoch: u64, volume: usize) -> Vec<Result<Patch>> {
    match try_load_volume(shared, epoch, volume) {
        Ok(patches) => patches.into_iter().map(Ok).collect(),
        Err(e) => vec![Err(e)],
    }
}

fn try_load_volume(shared: &Shared, epoch: u64, volume: usize) -> Result<Vec<Patch>> {
    let config = &shared.config;
    let mut rng = volume_rng(shared.seed, epoch, volume);
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
    let patch = config.patch_shape;
    let regions = match (config.foreground_prob, &label) {
        (Some(p), Some(label)) => sample_label_regions(
            &label.image()?,
            patch,
            config.patches_per_volume,
            p,
            &mut rng,
        )?,
        _ => (0..config.patches_per_volume)
            .map(|_| random_region(spatial, patch, &mut rng))
            .collect::<Result<_>>()?,
    };
    regions
        .into_iter()
        .map(|region| {
            let mut img = image.region(region.offset, region.shape)?;
            let mut lab = match &label {
                Some(l) => Some(l.region(region.offset, region.shape)?),
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
            return Some(load_volume(&self.shared, self.number, volume));
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
                    let patches = load_volume(&shared, epoch, order[position]);
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
    fn invalid_configuration_is_rejected() {
        let path = vec![PathBuf::from("a.nii")];
        assert!(FastLoader::new(vec![], None, LoaderConfig::new([4, 4, 4])).is_err());
        assert!(FastLoader::new(path.clone(), None, LoaderConfig::new([0, 4, 4])).is_err());
        let mut c = LoaderConfig::new([4, 4, 4]);
        c.foreground_prob = Some(0.5);
        assert!(FastLoader::new(path.clone(), None, c).is_err());
        assert!(FastLoader::new(path.clone(), Some(vec![]), LoaderConfig::new([4, 4, 4])).is_err());
    }
}
