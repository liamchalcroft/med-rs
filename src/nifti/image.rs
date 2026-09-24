//! `NIfTI` image: a header plus voxel data.
//!
//! Voxel data is either *owned* (a typed array) or *raw* (the bytes of the
//! file, memory-mapped or decompressed, decoded on demand). Raw storage keeps
//! loads cheap: nothing is copied until the data is actually needed.
//!
//! Shape, datatype, and byte order are properties of the data, so the image
//! keeps them itself; the header is kept consistent with them (see
//! [`NiftiImage::header_mut`]).

use super::element::{
    contiguous, dispatch_dtype, fortran_from_vec, scaled_as, scaled_f32, scaled_f64,
    try_fortran_from_vec, with_array, ArrayData, Elements, NiftiElement,
};
use super::header::{Affine, DataType, NiftiHeader};
use crate::error::{Error, Result};
use half::{bf16, f16};
use memmap2::Mmap;
use ndarray::{ArrayD, ArrayViewD, IxDyn, ShapeBuilder};
use std::borrow::Cow;
use std::fmt;
use std::ops::{Deref, DerefMut};
use std::sync::Arc;

/// Immutable shared byte buffer backing raw image data.
#[derive(Clone)]
pub(crate) enum Buffer {
    /// Heap buffer (for example decompressed gzip data).
    Heap(Arc<Vec<u8>>),
    /// Read-only memory map of an uncompressed file.
    Mmap(Arc<Mmap>),
}

impl Deref for Buffer {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        match self {
            Self::Heap(v) => v,
            Self::Mmap(m) => m,
        }
    }
}

/// Raw voxel bytes plus everything needed to decode them.
#[derive(Clone)]
pub(crate) struct RawData {
    pub(crate) buf: Buffer,
    pub(crate) offset: usize,
    pub(crate) dtype: DataType,
    pub(crate) little_endian: bool,
}

#[derive(Clone)]
pub(crate) enum Storage {
    Owned(ArrayData),
    Raw(RawData),
}

/// A `NIfTI` image: header metadata plus voxel data.
///
/// Arrays are indexed `[x, y, z, ...]` like the file, and stored in the file's
/// (Fortran) order.
#[derive(Clone)]
pub struct NiftiImage {
    header: NiftiHeader,
    storage: Storage,
    shape: Vec<usize>,
}

/// Mutable access to an image header.
///
/// Dereferences to [`NiftiHeader`]. When the guard is dropped, `ndim`, `dim`,
/// and `datatype` are restored from the voxel data, which is authoritative
/// for them: use [`NiftiImage::with_dtype`] to change the stored type.
pub struct HeaderMut<'a> {
    image: &'a mut NiftiImage,
}

impl Deref for HeaderMut<'_> {
    type Target = NiftiHeader;

    fn deref(&self) -> &NiftiHeader {
        &self.image.header
    }
}

impl DerefMut for HeaderMut<'_> {
    fn deref_mut(&mut self) -> &mut NiftiHeader {
        &mut self.image.header
    }
}

impl Drop for HeaderMut<'_> {
    fn drop(&mut self) {
        let dtype = self.image.dtype();
        sync_header(&mut self.image.header, &self.image.shape, dtype);
    }
}

/// Make the header's structural fields describe `shape` and `dtype`.
fn sync_header(header: &mut NiftiHeader, shape: &[usize], dtype: DataType) {
    header.ndim = shape.len() as u8;
    header.dim = [1; 7];
    for (d, &s) in header.dim.iter_mut().zip(shape) {
        *d = s as i64;
    }
    header.datatype = dtype;
}

fn check_ndim(shape: &[usize]) -> Result<()> {
    if shape.is_empty() || shape.len() > 7 {
        return Err(Error::InvalidDimensions(format!(
            "NIfTI images have 1 to 7 dimensions, got {}",
            shape.len()
        )));
    }
    if shape.contains(&0) {
        return Err(Error::InvalidDimensions(format!(
            "image dimensions must be non-zero, got {shape:?}"
        )));
    }
    Ok(())
}

impl NiftiImage {
    /// Create an image from an array (any memory layout) and an affine.
    ///
    /// The array is indexed `[x, y, z, ...]`; it is stored in Fortran order,
    /// copying only if it is not already in that layout.
    pub fn from_array<T: NiftiElement>(array: ArrayD<T>, affine: Affine) -> Result<Self> {
        let mut header = NiftiHeader::default();
        header.set_affine(affine);
        Self::from_array_with_header(array, header)
    }

    /// Create an image from an array and a template header.
    ///
    /// Geometry, intent, and other metadata come from `header`; its shape and
    /// datatype are replaced by the array's, and scaling is reset to identity.
    pub fn from_array_with_header<T: NiftiElement>(
        array: ArrayD<T>,
        mut header: NiftiHeader,
    ) -> Result<Self> {
        check_ndim(array.shape())?;
        header.scl_slope = 1.0;
        header.scl_inter = 0.0;
        Ok(Self::from_parts(header, ArrayData::new(array)))
    }

    /// Assemble an image from a header and owned data. The data is normalized
    /// to Fortran order and the header's shape and datatype are synced to it.
    pub(crate) fn from_parts(mut header: NiftiHeader, data: ArrayData) -> Self {
        let data = if data.is_fortran() {
            data
        } else {
            super::element::map_array!(data, |a| a)
        };
        let shape = data.shape().to_vec();
        sync_header(&mut header, &shape, data.dtype());
        Self {
            header,
            storage: Storage::Owned(data),
            shape,
        }
    }

    /// Wrap raw file bytes starting at `offset` without copying.
    ///
    /// Fails if the buffer is too short for the header's shape and datatype.
    pub(crate) fn from_raw(header: NiftiHeader, buf: Buffer, offset: usize) -> Result<Self> {
        let shape = header.shape();
        check_ndim(&shape)?;
        let needed = header.data_size();
        let available = buf.len().saturating_sub(offset);
        if available < needed {
            return Err(Error::InvalidFileFormat(format!(
                "file truncated: voxel data needs {needed} bytes at offset {offset}, \
                 but only {available} are present"
            )));
        }
        let raw = RawData {
            buf,
            offset,
            dtype: header.datatype,
            little_endian: header.little_endian,
        };
        Ok(Self {
            header,
            storage: Storage::Raw(raw),
            shape,
        })
    }

    // ------------------------------------------------------------------
    // Metadata
    // ------------------------------------------------------------------

    /// The header.
    pub fn header(&self) -> &NiftiHeader {
        &self.header
    }

    /// Mutable access to the header (see [`HeaderMut`]).
    pub fn header_mut(&mut self) -> HeaderMut<'_> {
        HeaderMut { image: self }
    }

    /// Replace the header's metadata, keeping this image's shape and datatype.
    #[must_use]
    pub fn with_header(mut self, header: NiftiHeader) -> Self {
        let dtype = self.dtype();
        self.header = header;
        sync_header(&mut self.header, &self.shape, dtype);
        self
    }

    /// Image shape.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Number of dimensions.
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Number of voxels.
    pub fn len(&self) -> usize {
        self.shape.iter().product()
    }

    /// Whether the image has no voxels (never true for a valid image).
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Stored voxel datatype.
    pub fn dtype(&self) -> DataType {
        match &self.storage {
            Storage::Owned(d) => d.dtype(),
            Storage::Raw(r) => r.dtype,
        }
    }

    /// Voxel-to-world affine (see [`NiftiHeader::affine`]).
    pub fn affine(&self) -> Affine {
        self.header.affine()
    }

    /// Set the affine (see [`NiftiHeader::set_affine`]).
    pub fn set_affine(&mut self, affine: Affine) {
        self.header.set_affine(affine);
    }

    /// Voxel sizes (see [`NiftiHeader::spacing`]).
    pub fn spacing(&self) -> Vec<f64> {
        self.header.spacing()
    }

    /// Anatomical orientation of the voxel axes, e.g. `RAS`.
    pub fn orientation(&self) -> crate::transforms::Orientation {
        crate::transforms::orientation_from_affine(&self.affine())
    }

    // ------------------------------------------------------------------
    // Storage
    // ------------------------------------------------------------------

    /// Whether the data is held as an owned typed array.
    pub fn is_materialized(&self) -> bool {
        matches!(self.storage, Storage::Owned(_))
    }

    /// Return an image whose data is an owned typed array (the stored values,
    /// unscaled). Cheap if already materialized.
    pub fn materialize(&self) -> Result<Self> {
        match &self.storage {
            Storage::Owned(_) => Ok(self.clone()),
            Storage::Raw(_) => Ok(Self {
                header: self.header.clone(),
                storage: Storage::Owned(self.raw_data()?.into_owned()),
                shape: self.shape.clone(),
            }),
        }
    }

    /// Whether the data is a read-only memory map of the source file.
    pub fn is_memory_mapped(&self) -> bool {
        matches!(
            self.storage,
            Storage::Raw(RawData {
                buf: Buffer::Mmap(_),
                ..
            })
        )
    }

    /// The stored bytes, if the data is still raw file data. Their byte order
    /// is the file's ([`NiftiHeader::is_little_endian`]).
    pub fn raw_bytes(&self) -> Option<&[u8]> {
        match &self.storage {
            Storage::Raw(r) => Some(self.raw_slice(r)),
            Storage::Owned(_) => None,
        }
    }

    fn raw_slice<'a>(&self, r: &'a RawData) -> &'a [u8] {
        let len = self.len() * r.dtype.byte_size();
        &r.buf[r.offset..r.offset + len]
    }

    /// The stored values as elements of type `T` (which must match the dtype).
    pub(crate) fn elements<T: NiftiElement>(&self) -> Result<Elements<'_, T>> {
        if T::DATA_TYPE != self.dtype() {
            return Err(Error::DataTypeMismatch {
                expected: T::DATA_TYPE.type_name(),
                found: self.dtype().type_name(),
            });
        }
        Ok(match &self.storage {
            Storage::Owned(data) => {
                let array = data
                    .as_array::<T>()
                    .ok_or_else(|| Error::Internal("stored dtype does not match data".into()))?;
                Elements::Slice(contiguous(array)?)
            }
            Storage::Raw(r) => Elements::from_bytes(self.raw_slice(r), r.little_endian),
        })
    }

    /// The stored (unscaled) values as a typed array, borrowing when owned.
    pub(crate) fn raw_data(&self) -> Result<Cow<'_, ArrayData>> {
        match &self.storage {
            Storage::Owned(d) => Ok(Cow::Borrowed(d)),
            Storage::Raw(_) => {
                let data = dispatch_dtype!(self.dtype(), T => {
                    let v = self.elements::<T>()?.to_vec();
                    ArrayData::new(fortran_from_vec(&self.shape, v))
                });
                Ok(Cow::Owned(data))
            }
        }
    }

    /// Zero-copy view of the stored (unscaled) values as `T`.
    ///
    /// Returns `None` if `T` is not the stored type, or if raw file data is
    /// byte-swapped or misaligned for `T`. Note that the view ignores
    /// `scl_slope`/`scl_inter`; see [`NiftiHeader::has_scaling`].
    #[allow(unsafe_code)]
    pub fn view<T: NiftiElement>(&self) -> Option<ArrayViewD<'_, T>> {
        match self.elements::<T>().ok()? {
            Elements::Slice(slice) => {
                // SAFETY: `slice` has exactly `shape.product()` elements (checked
                // when the image was built), is properly aligned and
                // native-endian (it came from `bytemuck::try_cast_slice` or an
                // owned array), and lives as long as `&self`.
                let view =
                    unsafe { ArrayViewD::from_shape_ptr(IxDyn(&self.shape).f(), slice.as_ptr()) };
                Some(view)
            }
            Elements::Bytes { .. } => None,
        }
    }

    /// Borrow the owned array if the data is materialized as `T`.
    pub fn as_array<T: NiftiElement>(&self) -> Option<&ArrayD<T>> {
        match &self.storage {
            Storage::Owned(d) => d.as_array::<T>(),
            Storage::Raw(_) => None,
        }
    }

    // ------------------------------------------------------------------
    // Conversions (scaling applied)
    // ------------------------------------------------------------------

    /// Voxel values with `scl_slope`/`scl_inter` applied, as `f32`
    /// (the equivalent of nibabel's `get_fdata(dtype=np.float32)`).
    pub fn to_f32(&self) -> Result<ArrayD<f32>> {
        let (slope, inter) = self.header.scaling();
        let v =
            dispatch_dtype!(self.dtype(), T => scaled_f32(&self.elements::<T>()?, slope, inter));
        try_fortran_from_vec(&self.shape, v)
    }

    /// Voxel values with scaling applied, as `f64`.
    pub fn to_f64(&self) -> Result<ArrayD<f64>> {
        let (slope, inter) = self.header.scaling();
        let v =
            dispatch_dtype!(self.dtype(), T => scaled_f64(&self.elements::<T>()?, slope, inter));
        try_fortran_from_vec(&self.shape, v)
    }

    /// Voxel values with scaling applied, as `f16`.
    pub fn to_f16(&self) -> Result<ArrayD<f16>> {
        self.to_scaled::<f16>()
    }

    /// Voxel values with scaling applied, as `bf16`.
    pub fn to_bf16(&self) -> Result<ArrayD<bf16>> {
        self.to_scaled::<bf16>()
    }

    /// Voxel values with scaling applied, converted to `T` (integers are
    /// rounded to nearest and saturated; NaN becomes 0).
    pub fn to_scaled<T: NiftiElement>(&self) -> Result<ArrayD<T>> {
        let (slope, inter) = self.header.scaling();
        let v = dispatch_dtype!(self.dtype(), S => scaled_as::<S, T>(&self.elements::<S>()?, slope, inter));
        try_fortran_from_vec(&self.shape, v)
    }

    /// Consume the image, returning the scaled values as `T`, reusing the
    /// buffer when the data is already an unscaled owned `T` array.
    pub fn into_array<T: NiftiElement>(self) -> Result<ArrayD<T>> {
        if !self.header.has_scaling() {
            if let Storage::Owned(data) = self.storage {
                return match data.into_array::<T>() {
                    Ok(a) => Ok(a),
                    Err(data) => Self {
                        header: self.header,
                        storage: Storage::Owned(data),
                        shape: self.shape,
                    }
                    .to_scaled(),
                };
            }
        }
        self.to_scaled()
    }

    /// Scaled values as `f32` in file order, borrowing when the stored data is
    /// already unscaled, native-endian `f32` (owned or memory-mapped).
    pub(crate) fn f32_values(&self) -> Result<Cow<'_, [f32]>> {
        let (slope, inter) = self.header.scaling();
        if self.dtype() == DataType::Float32 && slope == 1.0 && inter == 0.0 {
            if let Elements::Slice(s) = self.elements::<f32>()? {
                return Ok(Cow::Borrowed(s));
            }
        }
        let v =
            dispatch_dtype!(self.dtype(), T => scaled_f32(&self.elements::<T>()?, slope, inter));
        Ok(Cow::Owned(v))
    }

    /// Convert to another stored datatype, applying and then resetting the
    /// scaling (integers are rounded and saturated).
    pub fn with_dtype(&self, dtype: DataType) -> Result<Self> {
        if dtype == self.dtype() && !self.header.has_scaling() {
            return self.materialize();
        }
        let data = dispatch_dtype!(dtype, T => ArrayData::new(self.to_scaled::<T>()?));
        let mut header = self.header.clone();
        header.scl_slope = 1.0;
        header.scl_inter = 0.0;
        Ok(Self::from_parts(header, data))
    }

    /// Replace the voxel data, keeping this image's metadata.
    ///
    /// The new array must have the same shape. Scaling is reset to identity
    /// because the values are taken as-is.
    pub fn with_data<T: NiftiElement>(&self, array: ArrayD<T>) -> Result<Self> {
        if array.shape() != self.shape() {
            return Err(Error::ShapeMismatch(format!(
                "new data has shape {:?}, image has {:?}",
                array.shape(),
                self.shape()
            )));
        }
        Self::from_array_with_header(array, self.header.clone())
    }

    /// Replace owned data after an operation that keeps the geometry.
    pub(crate) fn with_array_data(&self, data: ArrayData, reset_scaling: bool) -> Self {
        let mut header = self.header.clone();
        if reset_scaling {
            header.scl_slope = 1.0;
            header.scl_inter = 0.0;
        }
        Self::from_parts(header, data)
    }

    /// Voxel bytes in little-endian file order, borrowing when possible.
    pub(crate) fn data_bytes_le(&self) -> Result<Cow<'_, [u8]>> {
        let native_le = cfg!(target_endian = "little");
        match &self.storage {
            Storage::Raw(r) => {
                let bytes = self.raw_slice(r);
                if r.little_endian || r.dtype.byte_size() == 1 {
                    Ok(Cow::Borrowed(bytes))
                } else {
                    Ok(Cow::Owned(swapped(bytes, r.dtype.byte_size())))
                }
            }
            Storage::Owned(data) => {
                let bytes: &[u8] = with_array!(data, |a| bytemuck::cast_slice(contiguous(a)?));
                let size = data.dtype().byte_size();
                if native_le || size == 1 {
                    Ok(Cow::Borrowed(bytes))
                } else {
                    Ok(Cow::Owned(swapped(bytes, size)))
                }
            }
        }
    }
}

fn swapped(bytes: &[u8], size: usize) -> Vec<u8> {
    let mut out = bytes.to_vec();
    for chunk in out.chunks_exact_mut(size) {
        chunk.reverse();
    }
    out
}

impl fmt::Debug for NiftiImage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NiftiImage")
            .field("shape", &self.shape)
            .field("dtype", &self.dtype())
            .field("spacing", &self.spacing())
            .field("materialized", &self.is_materialized())
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Axis, IxDyn};

    fn eye() -> Affine {
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    }

    #[test]
    fn from_array_accepts_any_layout() {
        let c =
            ArrayD::from_shape_vec(IxDyn(&[2, 3, 4]), (0..24).map(|v| v as f32).collect()).unwrap();
        let mut flipped = c.clone();
        flipped.invert_axis(Axis(0));
        for arr in [c.clone(), flipped.clone()] {
            let img = NiftiImage::from_array(arr.clone(), eye()).unwrap();
            assert_eq!(img.to_f32().unwrap(), arr);
            assert!(img.view::<f32>().is_some());
        }
    }

    #[test]
    fn header_edits_cannot_desync_structure() {
        let arr = ArrayD::<u8>::zeros(IxDyn(&[2, 2, 2]));
        let mut img = NiftiImage::from_array(arr, eye()).unwrap();
        {
            let mut h = img.header_mut();
            h.dim[0] = 99;
            h.datatype = DataType::Float64;
            h.scl_slope = 2.0;
        }
        assert_eq!(img.header().dim[0], 2);
        assert_eq!(img.header().datatype, DataType::UInt8);
        assert_eq!(img.header().scl_slope, 2.0);
    }

    #[test]
    fn raw_storage_decodes_big_endian_and_scaling() {
        let mut header = NiftiHeader {
            datatype: DataType::Int16,
            dim: [2, 2, 1, 1, 1, 1, 1],
            scl_slope: 2.0,
            scl_inter: -1.0,
            ..Default::default()
        };
        header.little_endian = false;
        let values: [i16; 4] = [1, -2, 3, 400];
        let mut bytes = vec![0u8; 3]; // odd offset => unaligned
        bytes.extend(values.iter().flat_map(|v| v.to_be_bytes()));
        let img = NiftiImage::from_raw(header, Buffer::Heap(Arc::new(bytes)), 3).unwrap();
        assert!(img.view::<i16>().is_none());
        let f = img.to_f32().unwrap();
        assert_eq!(f.as_slice_memory_order().unwrap(), &[1.0, -5.0, 5.0, 799.0]);
        let m = img.materialize().unwrap();
        assert_eq!(m.to_f32().unwrap(), f);
        let le = img.data_bytes_le().unwrap();
        assert_eq!(&le[..2], &1i16.to_le_bytes());
    }

    #[test]
    fn truncated_raw_data_is_an_error() {
        let header = NiftiHeader {
            dim: [4, 4, 4, 1, 1, 1, 1],
            ..Default::default()
        };
        let err = NiftiImage::from_raw(header, Buffer::Heap(Arc::new(vec![0; 100])), 0)
            .unwrap_err()
            .to_string();
        assert!(err.contains("truncated"), "{err}");
    }

    #[test]
    fn with_dtype_rounds_saturates_and_resets_scaling() {
        let arr = ArrayD::from_shape_vec(IxDyn(&[4, 1, 1]), vec![-1.6f32, 0.4, 2.5, 1e6]).unwrap();
        let img = NiftiImage::from_array(arr, eye()).unwrap();
        let u8img = img.with_dtype(DataType::UInt8).unwrap();
        assert_eq!(u8img.dtype(), DataType::UInt8);
        assert_eq!(
            u8img
                .as_array::<u8>()
                .unwrap()
                .as_slice_memory_order()
                .unwrap(),
            &[0, 0, 3, 255]
        );
        let bf = img.with_dtype(DataType::BFloat16).unwrap();
        assert_eq!(bf.header().datatype, DataType::BFloat16);
    }

    #[test]
    fn into_array_reuses_owned_buffer() {
        let arr = ArrayD::<i32>::from_elem(IxDyn(&[3, 3, 3]), 7);
        let img = NiftiImage::from_array(arr.clone(), eye()).unwrap();
        assert_eq!(img.clone().into_array::<i32>().unwrap(), arr);
        assert_eq!(img.into_array::<f64>().unwrap()[[0, 0, 0]], 7.0);
    }

    #[test]
    fn rejects_empty_and_too_many_dimensions() {
        assert!(NiftiImage::from_array(ArrayD::<f32>::zeros(IxDyn(&[0, 2])), eye()).is_err());
        assert!(NiftiImage::from_array(ArrayD::<f32>::zeros(IxDyn(&[1; 8])), eye()).is_err());
        assert!(NiftiImage::from_array(ArrayD::<f32>::zeros(IxDyn(&[])), eye()).is_err());
    }
}
