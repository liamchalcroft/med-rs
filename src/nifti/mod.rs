//! `NIfTI` file format support.
//!
//! `NIfTI` (Neuroimaging Informatics Technology Initiative) is a standard format
//! for neuroimaging data. This module provides high-performance reading and writing
//! of `.nii` and `.nii.gz` files.

pub(crate) mod header;
pub(crate) mod image;
pub mod io;

pub use header::{DataType, NiftiHeader, SpatialUnits, TemporalUnits};
pub use image::{NiftiElement, NiftiImage};
pub use io::{
    clear_decompression_cache, convert_to_mgzip, is_mgzip, load, load_cached, load_cropped,
    load_header, load_image_label_pair, load_mgzip, load_mgzip_with_threads, load_multi,
    load_with_crop, save, save_mgzip, save_mgzip_with_threads, set_cache_size, BatchIter,
    BatchLoader, CropConfig, CropLoader, FastLoader, FastLoaderBuilder, FileConfig, LoaderStats,
    MultiFileConfig, MultiFileResult, PatchConfig, TrainingDataLoader,
};
