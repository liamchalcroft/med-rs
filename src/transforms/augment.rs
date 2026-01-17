//! Random augmentation transforms for data augmentation in ML training.
//!
//! These transforms apply random perturbations to images for training
//! neural networks on medical imaging data.

use crate::error::{Error, Result};
use crate::nifti::image::ArrayData;
use crate::nifti::{DataType, NiftiImage};
use crate::pipeline::acquire_buffer;
use ndarray::{ArrayD, IxDyn, ShapeBuilder};
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;

/// Random number generator with optional seeding for reproducibility.
#[allow(clippy::option_if_let_else)] // match is clearer than map_or_else here
fn get_rng(seed: Option<u64>) -> ChaCha8Rng {
    match seed {
        Some(s) => ChaCha8Rng::seed_from_u64(s),
        None => ChaCha8Rng::from_entropy(),
    }
}

/// Randomly flip the image along specified axes with given probability.
///
/// # Arguments
///
/// * `image` - Input image
/// * `axes` - Axes that may be flipped (0=depth, 1=height, 2=width)
/// * `prob` - Probability of flipping each axis (default: 0.5)
/// * `seed` - Optional random seed for reproducibility
///
/// # Example
///
/// ```ignore
/// let augmented = random_flip(&img, &[0, 1, 2], Some(0.5), None);
/// ```
#[must_use = "this function returns a new image and does not modify the original"]
pub fn random_flip(
    image: &NiftiImage,
    axes: &[usize],
    prob: Option<f32>,
    seed: Option<u64>,
) -> Result<NiftiImage> {
    let prob = prob.unwrap_or(0.5);
    let mut rng = get_rng(seed);

    let flip_axes: Vec<usize> = axes
        .iter()
        .filter(|_| rng.gen::<f32>() < prob)
        .copied()
        .collect();

    if flip_axes.is_empty() {
        // No flip - return clone
        Ok(image.clone())
    } else {
        crate::transforms::flip(image, &flip_axes)
    }
}

/// Add random Gaussian noise to the image.
///
/// # Arguments
///
/// * `image` - Input image
/// * `std` - Standard deviation of the noise (default: 0.1)
/// * `seed` - Optional random seed for reproducibility
///
/// # Example
///
/// ```ignore
/// let noisy = random_gaussian_noise(&img, Some(0.05), None)?;
/// ```
#[must_use = "this function returns a Result and does not modify the original"]
pub fn random_gaussian_noise(
    image: &NiftiImage,
    std: Option<f32>,
    seed: Option<u64>,
) -> Result<NiftiImage> {
    const CHUNK_SIZE: usize = 8192;

    let std = std.unwrap_or(0.1);
    let base_seed = seed.unwrap_or_else(rand::random);

    let data = image.to_f32()?;
    let slice = data.as_slice_memory_order().ok_or_else(|| {
        Error::NonContiguousArray("Array must be contiguous for noise operation".to_string())
    })?;
    let len = slice.len();

    let mut output = acquire_buffer(len);
    let n_chunks = len.div_ceil(CHUNK_SIZE);

    if n_chunks > 1 {
        output
            .par_chunks_mut(CHUNK_SIZE)
            .enumerate()
            .for_each(|(chunk_idx, out_chunk)| {
                let chunk_seed = base_seed.wrapping_add(chunk_idx as u64);
                let mut rng = ChaCha8Rng::seed_from_u64(chunk_seed);
                let start = chunk_idx * CHUNK_SIZE;

                for (i, out) in out_chunk.iter_mut().enumerate() {
                    let u1: f32 = rng.gen::<f32>().max(1e-10);
                    let u2: f32 = rng.gen();
                    let noise =
                        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos() * std;
                    *out = slice[start + i] + noise;
                }
            });
    } else {
        let mut rng = ChaCha8Rng::seed_from_u64(base_seed);
        for (i, out) in output.iter_mut().enumerate() {
            let u1: f32 = rng.gen::<f32>().max(1e-10);
            let u2: f32 = rng.gen();
            let noise = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos() * std;
            *out = slice[i] + noise;
        }
    }

    // F-order to match NIfTI convention
    let shape = data.shape();
    let out_array = ArrayD::from_shape_vec(IxDyn(shape).f(), output)
        .map_err(|e| Error::MemoryAllocation(format!("Failed to create output array: {}", e)))?;
    let mut header = image.header().clone();
    header.datatype = DataType::Float32;
    header.scl_slope = 1.0;
    header.scl_inter = 0.0;

    Ok(NiftiImage::from_parts(header, ArrayData::F32(out_array)))
}

/// Randomly scale image intensity.
///
/// Multiplies intensity by a random factor sampled from [1-scale_range, 1+scale_range].
///
/// # Arguments
///
/// * `image` - Input image
/// * `scale_range` - Range for random scaling factor (default: 0.1, meaning 0.9-1.1)
/// * `seed` - Optional random seed for reproducibility
#[must_use = "this function returns a Result and does not modify the original"]
pub fn random_intensity_scale(
    image: &NiftiImage,
    scale_range: Option<f32>,
    seed: Option<u64>,
) -> Result<NiftiImage> {
    let scale_range = scale_range.unwrap_or(0.1);
    let mut rng = get_rng(seed);

    // Sample scale factor uniformly from [1-range, 1+range]
    let scale = 1.0 + rng.gen_range(-scale_range..=scale_range);

    let data = image.to_f32()?;
    let slice = data.as_slice_memory_order().ok_or_else(|| {
        Error::NonContiguousArray("Array must be contiguous for scale operation".to_string())
    })?;

    let mut output = acquire_buffer(slice.len());
    output
        .par_iter_mut()
        .zip(slice.par_iter())
        .for_each(|(out, &v)| {
            *out = v * scale;
        });

    // F-order to match NIfTI convention
    let shape = data.shape();
    let out_array = ArrayD::from_shape_vec(IxDyn(shape).f(), output)
        .map_err(|e| Error::MemoryAllocation(format!("Failed to create output array: {}", e)))?;
    let mut header = image.header().clone();
    header.datatype = DataType::Float32;
    header.scl_slope = 1.0;
    header.scl_inter = 0.0;

    Ok(NiftiImage::from_parts(header, ArrayData::F32(out_array)))
}

/// Randomly shift image intensity.
///
/// Adds a random offset sampled from [-shift_range, shift_range].
///
/// # Arguments
///
/// * `image` - Input image
/// * `shift_range` - Range for random shift (default: 0.1)
/// * `seed` - Optional random seed for reproducibility
#[must_use = "this function returns a Result and does not modify the original"]
pub fn random_intensity_shift(
    image: &NiftiImage,
    shift_range: Option<f32>,
    seed: Option<u64>,
) -> Result<NiftiImage> {
    let shift_range = shift_range.unwrap_or(0.1);
    let mut rng = get_rng(seed);

    // Sample shift uniformly from [-range, range]
    let shift = rng.gen_range(-shift_range..=shift_range);

    let data = image.to_f32()?;
    let slice = data.as_slice_memory_order().ok_or_else(|| {
        Error::NonContiguousArray("Array must be contiguous for shift operation".to_string())
    })?;

    let mut output = acquire_buffer(slice.len());
    output
        .par_iter_mut()
        .zip(slice.par_iter())
        .for_each(|(out, &v)| {
            *out = v + shift;
        });

    // F-order to match NIfTI convention
    let shape = data.shape();
    let out_array = ArrayD::from_shape_vec(IxDyn(shape).f(), output)
        .map_err(|e| Error::MemoryAllocation(format!("Failed to create output array: {}", e)))?;
    let mut header = image.header().clone();
    header.datatype = DataType::Float32;
    header.scl_slope = 1.0;
    header.scl_inter = 0.0;

    Ok(NiftiImage::from_parts(header, ArrayData::F32(out_array)))
}

/// Randomly rotate the image by 90-degree increments.
///
/// Performs random rotation in the specified plane by 0, 90, 180, or 270 degrees.
///
/// # Arguments
///
/// * `image` - Input image
/// * `axes` - Tuple of two axes defining the rotation plane (e.g., (0, 1) for depth-height plane)
/// * `seed` - Optional random seed for reproducibility
#[must_use = "this function returns a new image and does not modify the original"]
pub fn random_rotate_90(
    image: &NiftiImage,
    axes: (usize, usize),
    seed: Option<u64>,
) -> Result<NiftiImage> {
    let mut rng = get_rng(seed);

    // Choose random number of 90-degree rotations (0, 1, 2, or 3)
    let k: usize = rng.gen_range(0..4);

    if k == 0 {
        return Ok(image.clone());
    }

    rotate_90(image, axes, k)
}

/// Rotate the image by k * 90 degrees in the specified plane.
fn rotate_90(image: &NiftiImage, axes: (usize, usize), k: usize) -> Result<NiftiImage> {
    let ndim = image.ndim();
    if axes.0 >= ndim || axes.1 >= ndim {
        return Err(Error::InvalidDimensions(format!(
            "Rotation axes ({}, {}) out of bounds for {}D image",
            axes.0, axes.1, ndim
        )));
    }

    if axes.0 == axes.1 {
        return Err(Error::InvalidDimensions(
            "Rotation axes must be different".into(),
        ));
    }

    let k = k % 4;
    if k == 0 {
        return Ok(image.clone());
    }

    // Use ndarray's built-in rotation
    macro_rules! rotate_array {
        ($arr:expr, $variant:ident) => {{
            use ndarray::Axis;

            let mut arr = $arr.clone();

            for _ in 0..k {
                // Swap axes and reverse one of them to achieve 90-degree rotation
                arr.swap_axes(axes.0, axes.1);
                arr.invert_axis(Axis(axes.1));
            }

            // Convert to contiguous layout
            ArrayData::$variant(arr.as_standard_layout().to_owned())
        }};
    }

    let data = image.data_cow()?;
    let new_data = match data.as_ref() {
        ArrayData::U8(a) => rotate_array!(a, U8),
        ArrayData::I8(a) => rotate_array!(a, I8),
        ArrayData::I16(a) => rotate_array!(a, I16),
        ArrayData::U16(a) => rotate_array!(a, U16),
        ArrayData::I32(a) => rotate_array!(a, I32),
        ArrayData::U32(a) => rotate_array!(a, U32),
        ArrayData::I64(a) => rotate_array!(a, I64),
        ArrayData::U64(a) => rotate_array!(a, U64),
        ArrayData::F16(a) => rotate_array!(a, F16),
        ArrayData::BF16(a) => rotate_array!(a, BF16),
        ArrayData::F32(a) => rotate_array!(a, F32),
        ArrayData::F64(a) => rotate_array!(a, F64),
    };

    // Update header with rotated dimensions
    let mut header = image.header().clone();
    let old_shape = image.shape();
    let mut new_shape = old_shape.to_vec();

    // For odd k (90 or 270 degrees), swap dimensions
    if k % 2 == 1 {
        new_shape.swap(axes.0, axes.1);
    }

    header.ndim = new_shape.len() as u8;
    header.dim = [1i64; 7];
    for (i, &s) in new_shape.iter().enumerate() {
        header.dim[i] = s as i64;
    }

    Ok(NiftiImage::from_parts(header, new_data))
}

/// Apply random gamma correction to image intensity.
///
/// Applies the transform: output = input^gamma where gamma is randomly sampled.
///
/// # Arguments
///
/// * `image` - Input image (should be normalized to [0, 1] for best results)
/// * `gamma_range` - Range for gamma sampling (default: (0.7, 1.5))
/// * `seed` - Optional random seed for reproducibility
#[must_use = "this function returns a Result and does not modify the original"]
pub fn random_gamma(
    image: &NiftiImage,
    gamma_range: Option<(f32, f32)>,
    seed: Option<u64>,
) -> Result<NiftiImage> {
    let (gamma_min, gamma_max) = gamma_range.unwrap_or((0.7, 1.5));
    let mut rng = get_rng(seed);

    let gamma = rng.gen_range(gamma_min..=gamma_max);

    let data = image.to_f32()?;
    let slice = data.as_slice_memory_order().ok_or_else(|| {
        Error::NonContiguousArray("Array must be contiguous for gamma operation".to_string())
    })?;

    let mut output = acquire_buffer(slice.len());
    output
        .par_iter_mut()
        .zip(slice.par_iter())
        .for_each(|(out, &v)| {
            // Clamp to non-negative for gamma correction
            let v_clamped = v.max(0.0);
            *out = v_clamped.powf(gamma);
        });

    // F-order to match NIfTI convention
    let shape = data.shape();
    let out_array = ArrayD::from_shape_vec(IxDyn(shape).f(), output)
        .map_err(|e| Error::MemoryAllocation(format!("Failed to create output array: {}", e)))?;
    let mut header = image.header().clone();
    header.datatype = DataType::Float32;
    header.scl_slope = 1.0;
    header.scl_inter = 0.0;

    Ok(NiftiImage::from_parts(header, ArrayData::F32(out_array)))
}

/// Apply a random combination of augmentations commonly used in medical imaging.
///
/// This is a convenience function that applies multiple augmentations:
/// - Random flip (prob=0.5 per axis)
/// - Random intensity scale
/// - Random intensity shift
/// - Random Gaussian noise
///
/// For more control, use `RandomAugmentBuilder`.
#[must_use = "this function returns a Result and does not modify the original"]
pub fn random_augment(image: &NiftiImage, seed: Option<u64>) -> Result<NiftiImage> {
    RandomAugmentBuilder::new().seed_opt(seed).apply(image)
}

/// Builder for configurable random augmentation pipelines.
#[derive(Debug, Clone)]
#[allow(clippy::struct_excessive_bools)]
pub struct RandomAugmentBuilder {
    flip_axes: Vec<usize>,
    flip_prob: f32,
    scale_range: f32,
    shift_range: f32,
    noise_std: f32,
    gamma_range: Option<(f32, f32)>,
    enable_flip: bool,
    enable_scale: bool,
    enable_shift: bool,
    enable_noise: bool,
    enable_gamma: bool,
    seed: Option<u64>,
}

impl Default for RandomAugmentBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl RandomAugmentBuilder {
    /// Create a new augmentation builder with default settings.
    pub fn new() -> Self {
        Self {
            flip_axes: vec![0, 1, 2],
            flip_prob: 0.5,
            scale_range: 0.1,
            shift_range: 0.1,
            noise_std: 0.05,
            gamma_range: None,
            enable_flip: true,
            enable_scale: true,
            enable_shift: true,
            enable_noise: true,
            enable_gamma: false,
            seed: None,
        }
    }

    /// Set the random seed for reproducibility.
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Set an optional seed.
    pub fn seed_opt(mut self, seed: Option<u64>) -> Self {
        self.seed = seed;
        self
    }

    /// Set axes for random flipping.
    pub fn flip_axes(mut self, axes: &[usize]) -> Self {
        self.flip_axes = axes.to_vec();
        self
    }

    /// Set flip probability per axis.
    pub fn flip_prob(mut self, prob: f32) -> Self {
        self.flip_prob = prob;
        self
    }

    /// Set intensity scale range (± this value).
    pub fn scale_range(mut self, range: f32) -> Self {
        self.scale_range = range;
        self
    }

    /// Set intensity shift range (± this value).
    pub fn shift_range(mut self, range: f32) -> Self {
        self.shift_range = range;
        self
    }

    /// Set Gaussian noise standard deviation.
    pub fn noise_std(mut self, std: f32) -> Self {
        self.noise_std = std;
        self
    }

    /// Enable gamma correction with specified range.
    pub fn gamma(mut self, min: f32, max: f32) -> Self {
        self.gamma_range = Some((min, max));
        self.enable_gamma = true;
        self
    }

    /// Disable random flipping.
    pub fn no_flip(mut self) -> Self {
        self.enable_flip = false;
        self
    }

    /// Disable intensity scaling.
    pub fn no_scale(mut self) -> Self {
        self.enable_scale = false;
        self
    }

    /// Disable intensity shifting.
    pub fn no_shift(mut self) -> Self {
        self.enable_shift = false;
        self
    }

    /// Disable Gaussian noise.
    pub fn no_noise(mut self) -> Self {
        self.enable_noise = false;
        self
    }

    /// Apply the augmentation pipeline to an image.
    pub fn apply(&self, image: &NiftiImage) -> Result<NiftiImage> {
        let mut rng = get_rng(self.seed);
        let mut result = image.clone();

        if self.enable_flip && !self.flip_axes.is_empty() {
            let flip_seed: u64 = rng.gen();
            result = random_flip(
                &result,
                &self.flip_axes,
                Some(self.flip_prob),
                Some(flip_seed),
            )?;
        }

        if self.enable_scale && self.scale_range > 0.0 {
            let scale_seed: u64 = rng.gen();
            result = random_intensity_scale(&result, Some(self.scale_range), Some(scale_seed))?;
        }

        if self.enable_shift && self.shift_range > 0.0 {
            let shift_seed: u64 = rng.gen();
            result = random_intensity_shift(&result, Some(self.shift_range), Some(shift_seed))?;
        }

        if self.enable_noise && self.noise_std > 0.0 {
            let noise_seed: u64 = rng.gen();
            result = random_gaussian_noise(&result, Some(self.noise_std), Some(noise_seed))?;
        }

        if self.enable_gamma {
            let gamma_seed: u64 = rng.gen();
            result = random_gamma(&result, self.gamma_range, Some(gamma_seed))?;
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::ArrayD;

    fn create_test_image(data: Vec<f32>, shape: [usize; 3]) -> NiftiImage {
        use ndarray::ShapeBuilder;
        let c_order = ArrayD::from_shape_vec(shape.to_vec(), data).unwrap();
        let mut f_order = ArrayD::zeros(ndarray::IxDyn(&shape).f());
        f_order.assign(&c_order);
        let affine = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        NiftiImage::from_array(f_order, affine)
    }

    #[test]
    fn test_random_flip_with_seed() {
        let data: Vec<f32> = (0..64).map(|i| i as f32).collect();
        let img = create_test_image(data, [4, 4, 4]);

        // Same seed should produce same result
        let result1 = random_flip(&img, &[0, 1, 2], Some(0.5), Some(42)).unwrap();
        let result2 = random_flip(&img, &[0, 1, 2], Some(0.5), Some(42)).unwrap();

        let d1 = result1.to_f32().unwrap();
        let d2 = result2.to_f32().unwrap();
        assert_eq!(d1, d2);
    }

    #[test]
    fn test_random_gaussian_noise() {
        let data = vec![0.5; 64];
        let img = create_test_image(data.clone(), [4, 4, 4]);

        let noisy = random_gaussian_noise(&img, Some(0.1), Some(42)).unwrap();
        let result = noisy.to_f32().unwrap();

        // Noise should change values
        let original_sum: f32 = data.iter().sum();
        let result_sum: f32 = result.iter().sum();
        assert!((original_sum - result_sum).abs() > 0.01);

        // Shape should be preserved
        assert_eq!(result.shape(), &[4, 4, 4]);
    }

    #[test]
    fn test_random_intensity_scale() {
        let data: Vec<f32> = (1..=64).map(|i| i as f32).collect();
        let img = create_test_image(data, [4, 4, 4]);

        let scaled = random_intensity_scale(&img, Some(0.2), Some(42)).unwrap();
        let result = scaled.to_f32().unwrap();

        // Shape should be preserved
        assert_eq!(result.shape(), &[4, 4, 4]);

        // Values should be scaled (not identical to original)
        let original = img.to_f32().unwrap();
        assert_ne!(result, original);
    }

    #[test]
    fn test_random_intensity_shift() {
        let data: Vec<f32> = (1..=64).map(|i| i as f32).collect();
        let img = create_test_image(data, [4, 4, 4]);

        let shifted = random_intensity_shift(&img, Some(0.5), Some(42)).unwrap();
        let result = shifted.to_f32().unwrap();

        // Shape should be preserved
        assert_eq!(result.shape(), &[4, 4, 4]);
    }

    #[test]
    fn test_random_rotate_90() {
        let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let img = create_test_image(data, [2, 3, 4]);

        // Test rotation in different planes
        let rotated = random_rotate_90(&img, (0, 1), Some(42)).unwrap();

        // Shape may change (dimensions might swap)
        let shape = rotated.shape();
        assert!(shape.iter().product::<usize>() == 24); // Total elements preserved
    }

    #[test]
    fn test_rotate_90_k0() {
        let data: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let img = create_test_image(data.clone(), [2, 2, 2]);

        // k=0 should be identity
        let rotated = rotate_90(&img, (0, 1), 0).unwrap();
        assert_eq!(rotated.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_rotate_90_invalid_axes() {
        let data: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let img = create_test_image(data, [2, 2, 2]);

        // Out of bounds
        let result = rotate_90(&img, (0, 5), 1);
        assert!(result.is_err());

        // Same axis
        let result = rotate_90(&img, (1, 1), 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_random_gamma() {
        let data: Vec<f32> = (0..64).map(|i| (i as f32) / 63.0).collect();
        let img = create_test_image(data, [4, 4, 4]);

        let gamma_corrected = random_gamma(&img, Some((0.5, 2.0)), Some(42)).unwrap();
        let result = gamma_corrected.to_f32().unwrap();

        // Shape should be preserved
        assert_eq!(result.shape(), &[4, 4, 4]);

        // Values should be non-negative (gamma preserves non-negativity)
        for &v in result.iter() {
            assert!(v >= 0.0);
        }
    }

    #[test]
    fn test_random_augment() {
        let data: Vec<f32> = (0..64).map(|i| i as f32).collect();
        let img = create_test_image(data, [4, 4, 4]);

        let augmented = random_augment(&img, Some(42)).unwrap();

        // Shape should be preserved
        assert_eq!(augmented.shape(), &[4, 4, 4]);
    }

    #[test]
    fn test_reproducibility() {
        let data: Vec<f32> = (0..64).map(|i| i as f32).collect();
        let img = create_test_image(data, [4, 4, 4]);

        // Same seed should produce same result
        let aug1 = random_augment(&img, Some(12345)).unwrap();
        let aug2 = random_augment(&img, Some(12345)).unwrap();

        let d1 = aug1.to_f32().unwrap();
        let d2 = aug2.to_f32().unwrap();

        for (v1, v2) in d1.iter().zip(d2.iter()) {
            assert!((v1 - v2).abs() < 1e-6);
        }
    }
}
