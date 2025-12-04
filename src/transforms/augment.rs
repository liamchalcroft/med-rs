//! Random augmentation transforms for data augmentation in ML training.
//!
//! These transforms apply random perturbations to images for training
//! neural networks on medical imaging data.

use crate::error::Result;
use crate::nifti::image::ArrayData;
use crate::nifti::{DataType, NiftiImage};
use crate::pipeline::acquire_buffer;
use ndarray::{ArrayD, IxDyn, ShapeBuilder};
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;

/// Random number generator with optional seeding for reproducibility.
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
/// let noisy = random_gaussian_noise(&img, Some(0.05), None);
/// ```
pub fn random_gaussian_noise(
    image: &NiftiImage,
    std: Option<f32>,
    seed: Option<u64>,
) -> NiftiImage {
    let std = std.unwrap_or(0.1);
    let mut rng = get_rng(seed);

    let data = image.to_f32();
    let slice = data.as_slice_memory_order().unwrap();
    let len = slice.len();

    // Generate noise in parallel chunks for large arrays
    let mut output = acquire_buffer(len);

    // Use Box-Muller transform for Gaussian noise
    for (i, out) in output.iter_mut().enumerate() {
        let u1: f32 = rng.gen();
        let u2: f32 = rng.gen();
        // Avoid log(0)
        let u1 = u1.max(1e-10);
        let noise = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos() * std;
        *out = slice[i] + noise;
    }

    // F-order to match NIfTI convention
    let out_array = ArrayD::from_shape_vec(IxDyn(data.shape()).f(), output).unwrap();
    let mut header = image.header().clone();
    header.datatype = DataType::Float32;
    header.scl_slope = 1.0;
    header.scl_inter = 0.0;

    NiftiImage::from_parts(header, ArrayData::F32(out_array))
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
pub fn random_intensity_scale(
    image: &NiftiImage,
    scale_range: Option<f32>,
    seed: Option<u64>,
) -> NiftiImage {
    let scale_range = scale_range.unwrap_or(0.1);
    let mut rng = get_rng(seed);

    // Sample scale factor uniformly from [1-range, 1+range]
    let scale = 1.0 + rng.gen_range(-scale_range..=scale_range);

    let data = image.to_f32();
    let slice = data.as_slice_memory_order().unwrap();

    let mut output = acquire_buffer(slice.len());
    output
        .par_iter_mut()
        .zip(slice.par_iter())
        .for_each(|(out, &v)| {
            *out = v * scale;
        });

    // F-order to match NIfTI convention
    let out_array = ArrayD::from_shape_vec(IxDyn(data.shape()).f(), output).unwrap();
    let mut header = image.header().clone();
    header.datatype = DataType::Float32;
    header.scl_slope = 1.0;
    header.scl_inter = 0.0;

    NiftiImage::from_parts(header, ArrayData::F32(out_array))
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
pub fn random_intensity_shift(
    image: &NiftiImage,
    shift_range: Option<f32>,
    seed: Option<u64>,
) -> NiftiImage {
    let shift_range = shift_range.unwrap_or(0.1);
    let mut rng = get_rng(seed);

    // Sample shift uniformly from [-range, range]
    let shift = rng.gen_range(-shift_range..=shift_range);

    let data = image.to_f32();
    let slice = data.as_slice_memory_order().unwrap();

    let mut output = acquire_buffer(slice.len());
    output
        .par_iter_mut()
        .zip(slice.par_iter())
        .for_each(|(out, &v)| {
            *out = v + shift;
        });

    // F-order to match NIfTI convention
    let out_array = ArrayD::from_shape_vec(IxDyn(data.shape()).f(), output).unwrap();
    let mut header = image.header().clone();
    header.datatype = DataType::Float32;
    header.scl_slope = 1.0;
    header.scl_inter = 0.0;

    NiftiImage::from_parts(header, ArrayData::F32(out_array))
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
    use crate::error::Error;

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

    let data = image.data_cow();
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
    header.dim = [1u16; 7];
    for (i, &s) in new_shape.iter().enumerate() {
        header.dim[i] = s as u16;
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
pub fn random_gamma(
    image: &NiftiImage,
    gamma_range: Option<(f32, f32)>,
    seed: Option<u64>,
) -> NiftiImage {
    let (gamma_min, gamma_max) = gamma_range.unwrap_or((0.7, 1.5));
    let mut rng = get_rng(seed);

    let gamma = rng.gen_range(gamma_min..=gamma_max);

    let data = image.to_f32();
    let slice = data.as_slice_memory_order().unwrap();

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
    let out_array = ArrayD::from_shape_vec(IxDyn(data.shape()).f(), output).unwrap();
    let mut header = image.header().clone();
    header.datatype = DataType::Float32;
    header.scl_slope = 1.0;
    header.scl_inter = 0.0;

    NiftiImage::from_parts(header, ArrayData::F32(out_array))
}

/// Apply a random combination of augmentations commonly used in medical imaging.
///
/// This is a convenience function that applies multiple augmentations:
/// - Random flip (prob=0.5 per axis)
/// - Random intensity scale
/// - Random intensity shift
/// - Random Gaussian noise
///
/// # Arguments
///
/// * `image` - Input image
/// * `seed` - Optional random seed for reproducibility
pub fn random_augment(image: &NiftiImage, seed: Option<u64>) -> Result<NiftiImage> {
    let mut rng = get_rng(seed);

    // Generate seeds for each sub-operation
    let flip_seed = rng.gen();
    let scale_seed = rng.gen();
    let shift_seed = rng.gen();
    let noise_seed = rng.gen();

    // Apply augmentations in sequence
    let result = random_flip(image, &[0, 1, 2], Some(0.5), Some(flip_seed))?;
    let result = random_intensity_scale(&result, Some(0.1), Some(scale_seed));
    let result = random_intensity_shift(&result, Some(0.1), Some(shift_seed));
    let result = random_gaussian_noise(&result, Some(0.05), Some(noise_seed));

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::ArrayD;

    fn create_test_image(data: Vec<f32>, shape: [usize; 3]) -> NiftiImage {
        let array = ArrayD::from_shape_vec(shape.to_vec(), data).unwrap();
        let affine = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        NiftiImage::from_array(array, affine)
    }

    #[test]
    fn test_random_flip_with_seed() {
        let data: Vec<f32> = (0..64).map(|i| i as f32).collect();
        let img = create_test_image(data, [4, 4, 4]);

        // Same seed should produce same result
        let result1 = random_flip(&img, &[0, 1, 2], Some(0.5), Some(42)).unwrap();
        let result2 = random_flip(&img, &[0, 1, 2], Some(0.5), Some(42)).unwrap();

        let d1 = result1.to_f32();
        let d2 = result2.to_f32();
        assert_eq!(d1, d2);
    }

    #[test]
    fn test_random_gaussian_noise() {
        let data = vec![0.5; 64];
        let img = create_test_image(data.clone(), [4, 4, 4]);

        let noisy = random_gaussian_noise(&img, Some(0.1), Some(42));
        let result = noisy.to_f32();

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

        let scaled = random_intensity_scale(&img, Some(0.2), Some(42));
        let result = scaled.to_f32();

        // Shape should be preserved
        assert_eq!(result.shape(), &[4, 4, 4]);

        // Values should be scaled (not identical to original)
        let original = img.to_f32();
        assert_ne!(result, original);
    }

    #[test]
    fn test_random_intensity_shift() {
        let data: Vec<f32> = (1..=64).map(|i| i as f32).collect();
        let img = create_test_image(data, [4, 4, 4]);

        let shifted = random_intensity_shift(&img, Some(0.5), Some(42));
        let result = shifted.to_f32();

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

        let gamma_corrected = random_gamma(&img, Some((0.5, 2.0)), Some(42));
        let result = gamma_corrected.to_f32();

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

        let d1 = aug1.to_f32();
        let d2 = aug2.to_f32();

        for (v1, v2) in d1.iter().zip(d2.iter()) {
            assert!((v1 - v2).abs() < 1e-6);
        }
    }
}
