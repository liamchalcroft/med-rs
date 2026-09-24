//! `NIfTI` reading and writing.
//!
//! Supports NIfTI-1 and NIfTI-2, single files (`.nii`, `.nii.gz`) and
//! two-file pairs (`.hdr`/`.img`), either byte order, and header extensions.
//! See [`load`] and [`save`].

pub(crate) mod element;
pub(crate) mod gzip;
pub(crate) mod header;
pub(crate) mod image;
pub(crate) mod io;

pub use element::NiftiElement;
pub use header::{
    Affine, DataType, NiftiExtension, NiftiHeader, NiftiVersion, SpatialUnits, TemporalUnits,
};
pub use image::{HeaderMut, NiftiImage};
pub use io::{
    cache_usage, clear_decompression_cache, from_bytes, is_mgzip, load, load_cached, load_cropped,
    load_header, load_multi, save, save_mgzip, save_with_options, set_cache_max_bytes,
    set_cache_size, to_bytes, SaveOptions, DEFAULT_CACHE_BYTES, DEFAULT_CACHE_ENTRIES,
};
