//! Python module definition.

use pyo3::prelude::*;

use super::{augmentation, crops, image, loader, multi_file, python_io, transforms};
#[pymodule]
fn _medrs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<image::PyNiftiImage>()?;
    m.add_class::<loader::PyTrainingDataLoader>()?;
    m.add_class::<loader::PyFastLoader>()?;
    m.add_class::<transforms::PyTransformPipeline>()?;

    // Basic transforms
    m.add_function(wrap_pyfunction!(transforms::z_normalization, m)?)?;
    m.add_function(wrap_pyfunction!(transforms::rescale_intensity, m)?)?;
    m.add_function(wrap_pyfunction!(transforms::clamp, m)?)?;
    m.add_function(wrap_pyfunction!(transforms::crop_or_pad, m)?)?;
    m.add_function(wrap_pyfunction!(transforms::resample, m)?)?;
    m.add_function(wrap_pyfunction!(transforms::reorient, m)?)?;

    // I/O functions
    m.add_function(wrap_pyfunction!(python_io::load, m)?)?;
    m.add_function(wrap_pyfunction!(python_io::load_cached, m)?)?;
    m.add_function(wrap_pyfunction!(python_io::load_to_torch, m)?)?;
    m.add_function(wrap_pyfunction!(python_io::clear_decompression_cache, m)?)?;
    m.add_function(wrap_pyfunction!(python_io::set_cache_size, m)?)?;

    m.add_function(wrap_pyfunction!(python_io::save_mgzip, m)?)?;
    m.add_function(wrap_pyfunction!(python_io::load_mgzip, m)?)?;
    m.add_function(wrap_pyfunction!(python_io::convert_to_mgzip, m)?)?;
    m.add_function(wrap_pyfunction!(python_io::is_mgzip, m)?)?;

    // Multi-file loading functions
    m.add_function(wrap_pyfunction!(multi_file::load_multi, m)?)?;
    m.add_function(wrap_pyfunction!(multi_file::load_image_label_pair, m)?)?;

    // Crop-first transform functions
    m.add_function(wrap_pyfunction!(crops::load_cropped, m)?)?;
    m.add_function(wrap_pyfunction!(crops::load_resampled, m)?)?;
    m.add_function(wrap_pyfunction!(crops::load_cropped_to_torch, m)?)?;
    m.add_function(wrap_pyfunction!(crops::load_cropped_to_jax, m)?)?;
    m.add_function(wrap_pyfunction!(crops::load_label_aware_cropped, m)?)?;
    m.add_function(wrap_pyfunction!(crops::compute_crop_regions, m)?)?;
    m.add_function(wrap_pyfunction!(crops::compute_random_spatial_crops, m)?)?;
    m.add_function(wrap_pyfunction!(crops::compute_center_crop, m)?)?;

    // Random augmentation functions
    m.add_function(wrap_pyfunction!(augmentation::random_flip, m)?)?;
    m.add_function(wrap_pyfunction!(augmentation::random_gaussian_noise, m)?)?;
    m.add_function(wrap_pyfunction!(augmentation::random_intensity_scale, m)?)?;
    m.add_function(wrap_pyfunction!(augmentation::random_intensity_shift, m)?)?;
    m.add_function(wrap_pyfunction!(augmentation::random_rotate_90, m)?)?;
    m.add_function(wrap_pyfunction!(augmentation::random_gamma, m)?)?;
    m.add_function(wrap_pyfunction!(augmentation::random_augment, m)?)?;

    Ok(())
}
