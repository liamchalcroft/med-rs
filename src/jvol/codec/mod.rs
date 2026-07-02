// Vendored from github.com/fepegar/jvol-rust (MIT, (c) Fernando Perez-Garcia).
// Codec modules only; see LICENSE-jvol.
//
// The six files under this module (types, entropy, subbands, wavelet, encoding,
// decoding) are copied verbatim from jvol-rust's src/, with only the internal
// `use crate::...` paths rewritten to `use super::...` so they resolve as a
// submodule of medrs instead of as their own crate root. No algorithmic changes
// were made.
#![allow(
    clippy::all,
    clippy::pedantic,
    clippy::nursery,
    clippy::cargo,
    clippy::unwrap_used,
    clippy::panic,
    clippy::expect_used,
    missing_docs,
    unsafe_code,
    dead_code
)] // vendored upstream code, not linted; dead_code fires because this module
   // is crate-private and only a subset of its public API is used by the
   // medrs bridge in src/jvol/mod.rs

pub mod decoding;
pub mod encoding;
pub mod entropy;
pub mod subbands;
pub mod types;
pub mod wavelet;

pub use decoding::{decode_array, decode_downsampled_f32, decode_lossy_f32};
pub use encoding::encode_array;
pub use types::{Affine4x4, EncodedChannel, EncodedVolume, JvolDtype, JvolMetadata};
pub use wavelet::WaveletType;
