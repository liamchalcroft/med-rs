//! Smart cropping transforms for crop-first medical imaging I/O.
//!
//! This module provides intelligent cropping strategies that integrate with
//! medrs's byte-exact loading to eliminate the need for full volume loading.
//!
//! Key features:
//! - Label-aware cropping for segmentation training
//! - Random spatial cropping for general training
//! - Center cropping for inference/validation
//! - Foreground detection for automatic region selection

use crate::nifti::NiftiImage;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::cmp;

/// Configuration for label-aware cropping (RandCropByPosNegLabel equivalent).
#[derive(Debug, Clone)]
pub struct RandCropByPosNegLabelConfig {
    /// Target patch size [x, y, z]
    pub patch_size: [usize; 3],
    /// Ratio of positive to negative samples
    pub pos_neg_ratio: f32,
    /// Minimum positive samples per volume
    pub min_pos_samples: usize,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
    /// Background label value
    pub background_label: f32,
}

impl Default for RandCropByPosNegLabelConfig {
    fn default() -> Self {
        Self {
            patch_size: [64, 64, 64],
            pos_neg_ratio: 1.0,
            min_pos_samples: 4,
            seed: None,
            background_label: 0.0,
        }
    }
}

/// Configuration for random spatial cropping.
#[derive(Debug, Clone)]
pub struct SpatialCropConfig {
    /// Target patch size [x, y, z]
    pub patch_size: [usize; 3],
    /// Random seed for reproducibility
    pub seed: Option<u64>,
    /// Whether to allow smaller crops at image boundaries
    pub allow_smaller: bool,
}

impl Default for SpatialCropConfig {
    fn default() -> Self {
        Self {
            patch_size: [64, 64, 64],
            seed: None,
            allow_smaller: false,
        }
    }
}

/// Compute crop regions for label-based training (crop-first approach).
///
/// This function implements MONAI's `RandCropByPosNegLabeld` functionality
/// optimized for medrs's byte-exact loading. It returns crop regions
/// that can be used directly with `load_cropped()` to load only required bytes.
pub fn compute_label_aware_crop_regions(
    config: &RandCropByPosNegLabelConfig,
    _image: &NiftiImage,
    label: &NiftiImage,
    num_samples: usize,
) -> Vec<CropRegion> {
    let label_data = label.to_f32();
    let volume_shape_slice = label_data.shape();
    let volume_shape = if volume_shape_slice.len() == 3 {
        [
            volume_shape_slice[0],
            volume_shape_slice[1],
            volume_shape_slice[2],
        ]
    } else {
        // Skip processing if not 3D
        return Vec::new();
    };

    // Initialize random number generator
    let mut rng = ChaCha8Rng::seed_from_u64(config.seed.unwrap_or(42));

    // Find all positive and negative voxels
    let positive_voxels = find_positive_voxels(&label_data, config.background_label);
    let negative_voxels = find_negative_voxels(&label_data, config.background_label);

    if positive_voxels.is_empty() {
        // Fallback to random cropping if no positive voxels
        return compute_random_regions(config, &volume_shape, num_samples);
    }

    let mut regions = Vec::with_capacity(num_samples);
    let pos_per_batch = (num_samples as f32 / (1.0 + config.pos_neg_ratio)) as usize;
    let neg_per_batch = num_samples - pos_per_batch;

    // Sample positive regions
    for _ in 0..pos_per_batch.min(config.min_pos_samples) {
        if let Some(region) = sample_positive_region(
            &positive_voxels,
            &volume_shape,
            &config.patch_size,
            &mut rng,
        ) {
            regions.push(region);
        }
    }

    // Sample negative regions
    for _ in 0..neg_per_batch {
        if let Some(region) = sample_negative_region(
            &negative_voxels,
            &volume_shape,
            &config.patch_size,
            &mut rng,
        ) {
            regions.push(region);
        }
    }

    // Fill remaining slots with balanced sampling
    while regions.len() < num_samples {
        if rng.gen::<f32>() < 0.5 && !positive_voxels.is_empty() {
            if let Some(region) = sample_positive_region(
                &positive_voxels,
                &volume_shape,
                &config.patch_size,
                &mut rng,
            ) {
                regions.push(region);
            }
        } else if !negative_voxels.is_empty() {
            if let Some(region) = sample_negative_region(
                &negative_voxels,
                &volume_shape,
                &config.patch_size,
                &mut rng,
            ) {
                regions.push(region);
            }
        }
    }

    regions.truncate(num_samples);
    regions
}

/// Compute random spatial crop regions.
///
/// This function implements MONAI's `RandSpatialCropd` functionality
/// optimized for medrs's byte-exact loading.
pub fn compute_random_spatial_crop_regions(
    config: &SpatialCropConfig,
    image: &NiftiImage,
    num_samples: usize,
) -> Vec<CropRegion> {
    let volume_shape_slice = image.shape();
    let volume_shape = if volume_shape_slice.len() == 3 {
        [
            volume_shape_slice[0],
            volume_shape_slice[1],
            volume_shape_slice[2],
        ]
    } else {
        [0, 0, 0] // Default empty shape
    };
    compute_random_regions_for_size(config, &volume_shape, num_samples, config.patch_size)
}

/// Compute center crop regions.
///
/// This function implements MONAI's `CenterSpatialCropd` functionality
/// optimized for medrs's byte-exact loading.
pub fn compute_center_crop_regions(patch_size: [usize; 3], image: &NiftiImage) -> CropRegion {
    let volume_shape_slice = image.shape();
    let volume_shape = if volume_shape_slice.len() == 3 {
        [
            volume_shape_slice[0],
            volume_shape_slice[1],
            volume_shape_slice[2],
        ]
    } else {
        [0, 0, 0] // Default empty shape
    };

    let start = [
        volume_shape[0].saturating_sub(patch_size[0]) / 2,
        volume_shape[1].saturating_sub(patch_size[1]) / 2,
        volume_shape[2].saturating_sub(patch_size[2]) / 2,
    ];

    let end = [
        start[0] + patch_size[0],
        start[1] + patch_size[1],
        start[2] + patch_size[2],
    ];

    CropRegion {
        start,
        end,
        size: patch_size,
    }
}

// Helper functions
fn find_positive_voxels(
    label_data: &ndarray::ArrayD<f32>,
    background_label: f32,
) -> Vec<[usize; 3]> {
    let threshold = background_label;
    label_data
        .indexed_iter()
        .filter_map(|(idx, &value)| {
            if label_data.ndim() == 3 {
                let coords = [idx[0], idx[1], idx[2]];
                if value > threshold {
                    Some(coords)
                } else {
                    None
                }
            } else {
                None
            }
        })
        .collect()
}

fn find_negative_voxels(
    label_data: &ndarray::ArrayD<f32>,
    background_label: f32,
) -> Vec<[usize; 3]> {
    let threshold = background_label;
    label_data
        .indexed_iter()
        .filter_map(|(idx, &value)| {
            if label_data.ndim() == 3 {
                let coords = [idx[0], idx[1], idx[2]];
                if value <= threshold {
                    Some(coords)
                } else {
                    None
                }
            } else {
                None
            }
        })
        .collect()
}

fn sample_positive_region(
    positive_voxels: &[[usize; 3]],
    volume_shape: &[usize; 3],
    patch_size: &[usize; 3],
    rng: &mut ChaCha8Rng,
) -> Option<CropRegion> {
    if positive_voxels.is_empty() {
        return None;
    }

    let center_idx = rng.gen_range(0..positive_voxels.len());
    let center = positive_voxels[center_idx];

    Some(compute_region_from_center(center, volume_shape, patch_size))
}

fn sample_negative_region(
    negative_voxels: &[[usize; 3]],
    volume_shape: &[usize; 3],
    patch_size: &[usize; 3],
    rng: &mut ChaCha8Rng,
) -> Option<CropRegion> {
    if negative_voxels.is_empty() {
        return None;
    }

    let center_idx = rng.gen_range(0..negative_voxels.len());
    let center = negative_voxels[center_idx];

    Some(compute_region_from_center(center, volume_shape, patch_size))
}

fn compute_region_from_center(
    center: [usize; 3],
    volume_shape: &[usize; 3],
    patch_size: &[usize; 3],
) -> CropRegion {
    let half_size = [patch_size[0] / 2, patch_size[1] / 2, patch_size[2] / 2];

    let start = [
        center[0].saturating_sub(half_size[0]),
        center[1].saturating_sub(half_size[1]),
        center[2].saturating_sub(half_size[2]),
    ];

    let end = [
        cmp::min(start[0] + patch_size[0], volume_shape[0]),
        cmp::min(start[1] + patch_size[1], volume_shape[1]),
        cmp::min(start[2] + patch_size[2], volume_shape[2]),
    ];

    CropRegion {
        start,
        end,
        size: [end[0] - start[0], end[1] - start[1], end[2] - start[2]],
    }
}

fn compute_random_regions_for_size(
    config: &SpatialCropConfig,
    volume_shape: &[usize; 3],
    num_samples: usize,
    patch_size: [usize; 3],
) -> Vec<CropRegion> {
    let mut rng = ChaCha8Rng::seed_from_u64(config.seed.unwrap_or(42));
    let mut regions = Vec::with_capacity(num_samples);

    for _ in 0..num_samples {
        let max_start_x = if config.allow_smaller {
            volume_shape[0]
        } else {
            volume_shape[0].saturating_sub(patch_size[0])
        };

        let max_start_y = if config.allow_smaller {
            volume_shape[1]
        } else {
            volume_shape[1].saturating_sub(patch_size[1])
        };

        let max_start_z = if config.allow_smaller {
            volume_shape[2]
        } else {
            volume_shape[2].saturating_sub(patch_size[2])
        };

        let start = [
            rng.gen_range(0..max_start_x),
            rng.gen_range(0..max_start_y),
            rng.gen_range(0..max_start_z),
        ];

        let end = [
            cmp::min(start[0] + patch_size[0], volume_shape[0]),
            cmp::min(start[1] + patch_size[1], volume_shape[1]),
            cmp::min(start[2] + patch_size[2], volume_shape[2]),
        ];

        let actual_size = [end[0] - start[0], end[1] - start[1], end[2] - start[2]];

        regions.push(CropRegion {
            start,
            end,
            size: actual_size,
        });
    }

    regions
}

fn compute_random_regions(
    config: &RandCropByPosNegLabelConfig,
    volume_shape: &[usize; 3],
    num_samples: usize,
) -> Vec<CropRegion> {
    let spatial_config = SpatialCropConfig {
        patch_size: config.patch_size,
        seed: config.seed,
        allow_smaller: true,
    };

    compute_random_regions_for_size(
        &spatial_config,
        volume_shape,
        num_samples,
        config.patch_size,
    )
}

/// Represents a crop region within a volume.
#[derive(Debug, Clone)]
pub struct CropRegion {
    /// Starting voxel coordinates [x, y, z]
    pub start: [usize; 3],
    /// Ending voxel coordinates [x, y, z] (exclusive)
    pub end: [usize; 3],
    /// Region size [x, y, z]
    pub size: [usize; 3],
}

impl CropRegion {
    /// Create a new crop region.
    pub fn new(start: [usize; 3], size: [usize; 3]) -> Self {
        let end = [start[0] + size[0], start[1] + size[1], start[2] + size[2]];
        Self { start, end, size }
    }

    /// Check if this region is valid for the given volume shape.
    pub fn is_valid(&self, volume_shape: &[usize; 3]) -> bool {
        self.end[0] <= volume_shape[0]
            && self.end[1] <= volume_shape[1]
            && self.end[2] <= volume_shape[2]
    }

    /// Clamp this region to fit within the given volume shape.
    pub fn clamp_to_volume(&self, volume_shape: &[usize; 3]) -> Self {
        let start = [
            self.start[0].min(volume_shape[0].saturating_sub(1)),
            self.start[1].min(volume_shape[1].saturating_sub(1)),
            self.start[2].min(volume_shape[2].saturating_sub(1)),
        ];

        let end = [
            self.end[0].min(volume_shape[0]),
            self.end[1].min(volume_shape[1]),
            self.end[2].min(volume_shape[2]),
        ];

        let size = [end[0] - start[0], end[1] - start[1], end[2] - start[2]];

        Self { start, end, size }
    }
}

/// Intelligent foreground detector for automatic cropping.
pub struct ForegroundDetector {
    /// Threshold for determining foreground voxels
    threshold: f32,
    /// Minimum foreground volume ratio
    min_foreground_ratio: f32,
    /// Morphological operations radius
    morph_radius: usize,
}

impl ForegroundDetector {
    /// Create a new detector that finds bright regions above `threshold`.
    ///
    /// `min_foreground_ratio` defines the minimum fraction of voxels that must exceed
    /// the threshold; `morph_radius` controls the morphological smoothing window.
    pub fn new(threshold: f32, min_foreground_ratio: f32, morph_radius: usize) -> Self {
        Self {
            threshold,
            min_foreground_ratio,
            morph_radius,
        }
    }

    /// Find the bounding box of foreground voxels.
    pub fn find_foreground_bbox(&self, image: &NiftiImage) -> Option<CropRegion> {
        let data = image.to_f32();
        let volume_shape = data.shape();
        let mut min_coords = [volume_shape[0], volume_shape[1], volume_shape[2]];
        let mut max_coords = [0, 0, 0];
        let mut foreground_count = 0;

        // Find bounding box of foreground voxels
        for (idx, &value) in data.indexed_iter() {
            if data.ndim() == 3 {
                let (i, j, k) = (idx[0], idx[1], idx[2]);
                if value > self.threshold {
                    min_coords[0] = min_coords[0].min(i);
                    min_coords[1] = min_coords[1].min(j);
                    min_coords[2] = min_coords[2].min(k);

                    max_coords[0] = max_coords[0].max(i);
                    max_coords[1] = max_coords[1].max(j);
                    max_coords[2] = max_coords[2].max(k);

                    foreground_count += 1;
                }
            }
        }

        // Check if we have enough foreground
        let total_voxels = volume_shape[0] * volume_shape[1] * volume_shape[2];
        let foreground_ratio = foreground_count as f32 / total_voxels as f32;

        if foreground_ratio < self.min_foreground_ratio {
            return None;
        }

        // Expand bounding box slightly
        let padding = self.morph_radius;
        let start = [
            min_coords[0].saturating_sub(padding),
            min_coords[1].saturating_sub(padding),
            min_coords[2].saturating_sub(padding),
        ];

        let end = [
            (max_coords[0] + padding + 1).min(volume_shape[0]),
            (max_coords[1] + padding + 1).min(volume_shape[1]),
            (max_coords[2] + padding + 1).min(volume_shape[2]),
        ];

        let size = [end[0] - start[0], end[1] - start[1], end[2] - start[2]];

        Some(CropRegion { start, end, size })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::ArrayD;

    #[test]
    fn test_crop_region_creation() {
        let region = CropRegion::new([10, 20, 30], [64, 64, 32]);
        assert_eq!(region.start, [10, 20, 30]);
        assert_eq!(region.end, [74, 84, 62]);
        assert_eq!(region.size, [64, 64, 32]);
    }

    #[test]
    fn test_crop_region_bounds() {
        let region = CropRegion::new([64, 64, 64], [64, 64, 64]);
        let volume_shape = [128, 128, 128];

        assert!(region.is_valid(&volume_shape));

        let invalid_volume = [100, 100, 100];
        assert!(!region.is_valid(&invalid_volume));
    }

    #[test]
    fn test_foreground_detector() {
        let detector = ForegroundDetector::new(0.5, 0.01, 2); // Lower threshold to 1%

        // Create test data with foreground region
        let mut data = ArrayD::from_elem(vec![10, 10, 10], 0.0f32)
            .into_dimensionality()
            .unwrap();

        // Add foreground voxels in center
        data.slice_mut(ndarray::s![4..7, 4..7, 4..7]).fill(1.0);

        let affine = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let image = NiftiImage::from_array(data.clone(), affine);

        let bbox = detector.find_foreground_bbox(&image);
        assert!(bbox.is_some());

        let region = bbox.unwrap();
        assert_eq!(region.start, [2, 2, 2]); // Expanded by morph_radius=2
        assert_eq!(region.end, [9, 9, 9]); // max_coords (6) + morph_radius (2) + 1 = 9
    }
}
