//! NIfTI image representation with type-safe array access.

#![allow(private_interfaces)]

use super::header::{DataType, NiftiHeader};
use crate::error::{Error, Result};
use byteorder::ByteOrder;
use half::{bf16, f16};
use memmap2::Mmap;
use ndarray::{ArrayD, ArrayViewD, IxDyn, ShapeBuilder};
use num_traits::NumCast;
use rayon::prelude::*;
use std::fmt;
use std::sync::Arc;

/// Threshold for parallel materialization (1MB)
const PARALLEL_THRESHOLD: usize = 1024 * 1024;

/// Check if file endianness matches system endianness.
#[inline]
fn is_native_endian(little_endian: bool) -> bool {
    #[cfg(target_endian = "little")]
    {
        little_endian
    }
    #[cfg(target_endian = "big")]
    {
        !little_endian
    }
}

/// A NIfTI image containing header metadata and voxel data.
#[derive(Clone)]
pub struct NiftiImage {
    header: NiftiHeader,
    storage: DataStorage,
    shape_cache: Vec<usize>,
}

/// Type-erased array data storage.
#[derive(Clone)]
pub(crate) enum ArrayData {
    U8(ArrayD<u8>),
    I8(ArrayD<i8>),
    I16(ArrayD<i16>),
    U16(ArrayD<u16>),
    I32(ArrayD<i32>),
    U32(ArrayD<u32>),
    I64(ArrayD<i64>),
    U64(ArrayD<u64>),
    F16(ArrayD<f16>),
    BF16(ArrayD<bf16>),
    F32(ArrayD<f32>),
    F64(ArrayD<f64>),
}

/// Backing storage for voxel data.
#[derive(Clone)]
pub(crate) enum DataStorage {
    /// Owned, typed array data.
    Owned(ArrayData),
    /// Shared raw bytes (e.g., mmap or shared buffer).
    SharedBytes {
        bytes: Arc<Vec<u8>>,
        offset: usize,
        len: usize,
    },
    /// Shared mmap-backed bytes.
    SharedMmap {
        mmap: Arc<Mmap>,
        offset: usize,
        len: usize,
    },
}

impl NiftiImage {
    /// Create a new image backed by shared bytes (e.g., decompressed buffer).
    pub(crate) fn from_shared_bytes(
        header: NiftiHeader,
        bytes: Arc<Vec<u8>>,
        offset: usize,
        len: usize,
    ) -> Self {
        let shape: Vec<usize> = header.shape().iter().map(|&d| d as usize).collect();
        Self {
            header,
            storage: DataStorage::SharedBytes { bytes, offset, len },
            shape_cache: shape,
        }
    }

    /// Create a new image backed by mmap (no copy).
    pub(crate) fn from_shared_mmap(
        header: NiftiHeader,
        mmap: Arc<Mmap>,
        offset: usize,
        len: usize,
    ) -> Self {
        let shape: Vec<usize> = header.shape().iter().map(|&d| d as usize).collect();
        Self {
            header,
            storage: DataStorage::SharedMmap { mmap, offset, len },
            shape_cache: shape,
        }
    }

    /// Create a new image from header and pre-parsed array data.
    ///
    /// This is more efficient than cloning an entire image when you only need
    /// the header metadata and have new data to associate with it.
    pub(crate) fn from_parts(header: NiftiHeader, data: ArrayData) -> Self {
        let shape: Vec<usize> = header.shape().iter().map(|&d| d as usize).collect();
        Self {
            header,
            storage: DataStorage::Owned(data),
            shape_cache: shape,
        }
    }

    /// Create a new image from an ndarray with automatic dtype inference.
    pub fn from_array<T>(array: ArrayD<T>, affine: [[f32; 4]; 4]) -> Self
    where
        T: NiftiElement + Clone,
    {
        let shape: Vec<u16> = array.shape().iter().map(|&d| d as u16).collect();
        let shape_cache: Vec<usize> = array.shape().iter().map(|&d| d as usize).collect();
        let mut dim = [1u16; 7];
        dim[..shape.len()].copy_from_slice(&shape);

        let mut header = NiftiHeader {
            ndim: shape.len() as u8,
            dim,
            datatype: T::DATA_TYPE,
            ..Default::default()
        };
        header.set_affine(affine);

        Self {
            header,
            storage: DataStorage::Owned(T::wrap_array(array)),
            shape_cache,
        }
    }

    /// Reference to the header.
    pub fn header(&self) -> &NiftiHeader {
        &self.header
    }

    /// Mutable reference to the header.
    pub fn header_mut(&mut self) -> &mut NiftiHeader {
        &mut self.header
    }

    /// Get owned array data (materializes shared storage if needed).
    /// Note: This clones even if data is already owned; prefer `data_cow()` to avoid copies.
    pub(crate) fn owned_data(&self) -> ArrayData {
        self.materialize_owned()
    }

    /// Get a reference to owned data, or materialize and return owned.
    /// This avoids cloning when data is already owned.
    pub(crate) fn data_cow(&self) -> std::borrow::Cow<'_, ArrayData> {
        match &self.storage {
            DataStorage::Owned(d) => std::borrow::Cow::Borrowed(d),
            _ => std::borrow::Cow::Owned(self.materialize_owned()),
        }
    }

    /// Check if the image data is already materialized (owned).
    pub fn is_materialized(&self) -> bool {
        matches!(&self.storage, DataStorage::Owned(_))
    }

    /// Convert mmap'd or shared data to owned data.
    ///
    /// Call this once before running multiple transforms to avoid
    /// re-materializing the data on each transform call.
    ///
    /// Returns a new NiftiImage with owned data storage.
    pub fn materialize(&self) -> Self {
        if self.is_materialized() {
            return self.clone();
        }
        Self {
            header: self.header.clone(),
            storage: DataStorage::Owned(self.materialize_owned()),
            shape_cache: self.shape_cache.clone(),
        }
    }

    /// Borrow a view of the data as f32 if storage is contiguous.
    #[allow(unsafe_code)]
    pub fn as_view_f32(&self) -> Option<ArrayViewD<'_, f32>> {
        match &self.storage {
            DataStorage::Owned(ArrayData::F32(a)) => Some(a.view()),
            DataStorage::SharedBytes { bytes, offset, len } => {
                let slice = &bytes[*offset..*offset + (*len).min(bytes.len() - *offset)];
                let elems = slice.len() / std::mem::size_of::<f32>();
                if elems != self.shape_cache.iter().product::<usize>() {
                    return None;
                }
                if (slice.as_ptr() as usize) % std::mem::align_of::<f32>() != 0 {
                    return None;
                }
                let ptr = slice.as_ptr() as *const f32;
                // NIfTI data is stored in F-order (column-major)
                let view = unsafe {
                    ndarray::ArrayView::from_shape_ptr(IxDyn(&self.shape_cache).f(), ptr)
                };
                Some(view)
            }
            DataStorage::SharedMmap { mmap, offset, len } => {
                let slice = &mmap[*offset..*offset + (*len).min(mmap.len() - *offset)];
                let elems = slice.len() / std::mem::size_of::<f32>();
                if elems != self.shape_cache.iter().product::<usize>() {
                    return None;
                }
                if (slice.as_ptr() as usize) % std::mem::align_of::<f32>() != 0 {
                    return None;
                }
                let ptr = slice.as_ptr() as *const f32;
                // NIfTI data is stored in F-order (column-major)
                let view = unsafe {
                    ndarray::ArrayView::from_shape_ptr(IxDyn(&self.shape_cache).f(), ptr)
                };
                Some(view)
            }
            _ => None,
        }
    }

    /// Borrow a view of the data for a specific dtype if contiguous.
    pub fn as_view_t<T: NiftiElement>(&self) -> Option<ArrayViewD<'_, T>> {
        match (&self.storage, T::DATA_TYPE) {
            (DataStorage::Owned(data), _) => T::extract_array(data).map(|a| a.view()),
            (DataStorage::SharedBytes { bytes, offset, len }, dtype) => {
                self.shared_view(bytes, *offset, *len, dtype)
            }
            (DataStorage::SharedMmap { mmap, offset, len }, dtype) => {
                self.shared_view_mmap(mmap, *offset, *len, dtype)
            }
        }
    }

    fn shared_view<T: NiftiElement>(
        &self,
        bytes: &Arc<Vec<u8>>,
        offset: usize,
        len: usize,
        dtype: DataType,
    ) -> Option<ArrayViewD<'_, T>> {
        let slice = &bytes[offset..offset + len.min(bytes.len() - offset)];
        self.build_view(slice, dtype)
    }

    fn shared_view_mmap<T: NiftiElement>(
        &self,
        mmap: &Arc<Mmap>,
        offset: usize,
        len: usize,
        dtype: DataType,
    ) -> Option<ArrayViewD<'_, T>> {
        let slice = &mmap[offset..offset + len.min(mmap.len() - offset)];
        self.build_view(slice, dtype)
    }

    #[allow(unsafe_code)]
    fn build_view<T: NiftiElement>(
        &self,
        slice: &[u8],
        dtype: DataType,
    ) -> Option<ArrayViewD<'_, T>> {
        // Only support native matching dtype
        if dtype != T::DATA_TYPE {
            return None;
        }

        let elem_size = std::mem::size_of::<T>();
        if slice.len() / elem_size != self.shape_cache.iter().product::<usize>() {
            return None;
        }
        if (slice.as_ptr() as usize) % std::mem::align_of::<T>() != 0 {
            return None;
        }

        // Endianness mismatch => no view
        let native_le = cfg!(target_endian = "little");
        if self.header.is_little_endian() != native_le {
            return None;
        }

        let ptr = slice.as_ptr() as *const T;
        // NIfTI data is stored in F-order (column-major)
        let view =
            unsafe { ndarray::ArrayView::from_shape_ptr(IxDyn(&self.shape_cache).f(), ptr) };
        Some(view)
    }

    /// Image shape.
    pub fn shape(&self) -> &[usize] {
        &self.shape_cache
    }

    /// Number of dimensions.
    pub fn ndim(&self) -> usize {
        self.shape_cache.len()
    }

    /// Data type.
    pub fn dtype(&self) -> DataType {
        self.header.datatype
    }

    /// 4x4 affine transformation matrix.
    pub fn affine(&self) -> [[f32; 4]; 4] {
        self.header.affine()
    }

    /// Set the affine transformation matrix.
    pub fn set_affine(&mut self, affine: [[f32; 4]; 4]) {
        self.header.set_affine(affine);
    }

    /// Voxel spacing.
    pub fn spacing(&self) -> &[f32] {
        self.header.spacing()
    }

    /// Get data as f32 array with scaling applied.
    ///
    /// This is the primary way to access image data, similar to nibabel's `get_fdata()`.
    pub fn to_f32(&self) -> ArrayD<f32> {
        let slope = if self.header.scl_slope == 0.0 {
            1.0
        } else {
            self.header.scl_slope
        };
        let inter = self.header.scl_inter;

        macro_rules! convert_owned {
            ($arr:expr) => {
                $arr.mapv(|v| v as f32 * slope + inter)
            };
        }

        match &self.storage {
            DataStorage::Owned(d) => match d {
                ArrayData::U8(a) => convert_owned!(a),
                ArrayData::I8(a) => convert_owned!(a),
                ArrayData::I16(a) => convert_owned!(a),
                ArrayData::U16(a) => convert_owned!(a),
                ArrayData::I32(a) => convert_owned!(a),
                ArrayData::U32(a) => convert_owned!(a),
                ArrayData::I64(a) => convert_owned!(a),
                ArrayData::U64(a) => convert_owned!(a),
                ArrayData::F16(a) => a.mapv(|v| v.to_f32() * slope + inter),
                ArrayData::BF16(a) => a.mapv(|v| v.to_f32() * slope + inter),
                ArrayData::F32(a) => {
                    if slope == 1.0 && inter == 0.0 {
                        a.clone()
                    } else {
                        a.mapv(|v| v * slope + inter)
                    }
                }
                ArrayData::F64(a) => a.mapv(|v| (v * slope as f64 + inter as f64) as f32),
            },
            DataStorage::SharedBytes { bytes, offset, len } => {
                let slice = &bytes[*offset..*offset + (*len).min(bytes.len() - *offset)];
                self.shared_to_f32_slice(slice, slope, inter)
                    .expect("failed to read shared bytes")
            }
            DataStorage::SharedMmap { mmap, offset, len } => {
                let slice = &mmap[*offset..*offset + (*len).min(mmap.len() - *offset)];
                self.shared_to_f32_slice(slice, slope, inter)
                    .expect("failed to read shared mmap")
            }
        }
    }

    /// Get data as f64 array with scaling applied.
    pub fn to_f64(&self) -> ArrayD<f64> {
        let slope = if self.header.scl_slope == 0.0 {
            1.0
        } else {
            self.header.scl_slope
        } as f64;
        let inter = self.header.scl_inter as f64;

        macro_rules! convert {
            ($arr:expr) => {
                $arr.mapv(|v| v as f64 * slope + inter)
            };
        }

        match self.materialize_owned() {
            ArrayData::U8(a) => convert!(a),
            ArrayData::I8(a) => convert!(a),
            ArrayData::I16(a) => convert!(a),
            ArrayData::U16(a) => convert!(a),
            ArrayData::I32(a) => convert!(a),
            ArrayData::U32(a) => convert!(a),
            ArrayData::I64(a) => convert!(a),
            ArrayData::U64(a) => convert!(a),
            ArrayData::F16(a) => a.mapv(|v| v.to_f64() * slope + inter),
            ArrayData::BF16(a) => a.mapv(|v| v.to_f64() * slope + inter),
            ArrayData::F32(a) => convert!(a),
            ArrayData::F64(a) => {
                if slope == 1.0 && inter == 0.0 {
                    a.clone()
                } else {
                    a.mapv(|v| v * slope + inter)
                }
            }
        }
    }

    /// Get typed array reference if data matches type T.
    pub fn as_array<T: NiftiElement>(&self) -> Option<&ArrayD<T>> {
        match &self.storage {
            DataStorage::Owned(d) => T::extract_array(d),
            _ => None,
        }
    }

    /// Convert data to type T (with potential loss of precision).
    pub fn into_array<T: NiftiElement + NumCast>(self) -> ArrayD<T> {
        let slope = if self.header.scl_slope == 0.0 {
            1.0
        } else {
            self.header.scl_slope
        } as f64;
        let inter = self.header.scl_inter as f64;

        macro_rules! convert {
            ($arr:expr) => {
                $arr.mapv(|v| {
                    let scaled = v as f64 * slope + inter;
                    T::from(scaled).unwrap_or_else(|| {
                        if scaled > 0.0 {
                            T::max_value()
                        } else {
                            T::min_value()
                        }
                    })
                })
            };
        }

        let owned = self.materialize_owned();

        match owned {
            ArrayData::U8(a) => convert!(a),
            ArrayData::I8(a) => convert!(a),
            ArrayData::I16(a) => convert!(a),
            ArrayData::U16(a) => convert!(a),
            ArrayData::I32(a) => convert!(a),
            ArrayData::U32(a) => convert!(a),
            ArrayData::I64(a) => convert!(a),
            ArrayData::U64(a) => convert!(a),
            ArrayData::F16(a) => a.mapv(|v| {
                let scaled = v.to_f64() * slope + inter;
                T::from(scaled).unwrap_or_else(|| {
                    if scaled > 0.0 {
                        T::max_value()
                    } else {
                        T::min_value()
                    }
                })
            }),
            ArrayData::BF16(a) => a.mapv(|v| {
                let scaled = v.to_f64() * slope + inter;
                T::from(scaled).unwrap_or_else(|| {
                    if scaled > 0.0 {
                        T::max_value()
                    } else {
                        T::min_value()
                    }
                })
            }),
            ArrayData::F32(a) => convert!(a),
            ArrayData::F64(a) => convert!(a),
        }
    }

    /// Serialize image data to bytes (for writing).
    /// Uses memory order (F-order for NIfTI convention) to write data.
    pub(crate) fn data_to_bytes(&self) -> Vec<u8> {
        use byteorder::{ByteOrder, LittleEndian};

        // Helper macro to serialize using memory order
        macro_rules! serialize_memory_order {
            ($arr:expr, $elem_size:expr, $write_fn:expr) => {{
                let slice = $arr.as_slice_memory_order()
                    .expect("Array should be contiguous in memory");
                let mut buf = vec![0u8; slice.len() * $elem_size];
                for (i, &v) in slice.iter().enumerate() {
                    $write_fn(&mut buf[i * $elem_size..(i + 1) * $elem_size], v);
                }
                buf
            }};
        }

        match self.materialize_owned() {
            ArrayData::U8(a) => a.as_slice_memory_order()
                .expect("Array should be contiguous").to_vec(),
            ArrayData::I8(a) => a.as_slice_memory_order()
                .expect("Array should be contiguous")
                .iter().map(|&v| v as u8).collect(),
            ArrayData::I16(a) => serialize_memory_order!(a, 2, LittleEndian::write_i16),
            ArrayData::U16(a) => serialize_memory_order!(a, 2, LittleEndian::write_u16),
            ArrayData::I32(a) => serialize_memory_order!(a, 4, LittleEndian::write_i32),
            ArrayData::U32(a) => serialize_memory_order!(a, 4, LittleEndian::write_u32),
            ArrayData::I64(a) => serialize_memory_order!(a, 8, LittleEndian::write_i64),
            ArrayData::U64(a) => serialize_memory_order!(a, 8, LittleEndian::write_u64),
            ArrayData::F16(a) => {
                let slice = a.as_slice_memory_order()
                    .expect("Array should be contiguous");
                let mut buf = vec![0u8; slice.len() * 2];
                for (i, &v) in slice.iter().enumerate() {
                    LittleEndian::write_u16(&mut buf[i * 2..(i + 1) * 2], v.to_bits());
                }
                buf
            }
            ArrayData::BF16(a) => {
                let slice = a.as_slice_memory_order()
                    .expect("Array should be contiguous");
                let mut buf = vec![0u8; slice.len() * 2];
                for (i, &v) in slice.iter().enumerate() {
                    LittleEndian::write_u16(&mut buf[i * 2..(i + 1) * 2], v.to_bits());
                }
                buf
            }
            ArrayData::F32(a) => serialize_memory_order!(a, 4, LittleEndian::write_f32),
            ArrayData::F64(a) => serialize_memory_order!(a, 8, LittleEndian::write_f64),
        }
    }

    fn materialize_owned(&self) -> ArrayData {
        match &self.storage {
            DataStorage::Owned(d) => d.clone(),
            DataStorage::SharedBytes { bytes, offset, len } => {
                let slice = &bytes[*offset..*offset + (*len).min(bytes.len() - *offset)];
                self.materialize_native_from_slice(slice)
                    .expect("failed to materialize shared bytes")
            }
            DataStorage::SharedMmap { mmap, offset, len } => {
                let slice = &mmap[*offset..*offset + (*len).min(mmap.len() - *offset)];
                self.materialize_native_from_slice(slice)
                    .expect("failed to materialize shared mmap")
            }
        }
    }

    /// Fast materialization that preserves native dtype.
    /// Uses zero-copy reinterpretation for native-endian data.
    pub(crate) fn materialize_native_from_slice(&self, bytes: &[u8]) -> Result<ArrayData> {
        let shape = IxDyn(&self.shape_cache);
        let is_native = is_native_endian(self.header.is_little_endian());

        // Fast path: native-endian data can be reinterpreted directly
        if is_native {
            return self.materialize_native_fast(bytes, shape);
        }

        // Slow path: need endianness conversion
        self.materialize_with_byteswap(bytes, shape)
    }

    /// Fast path for native-endian data - reinterpret bytes directly.
    #[allow(unsafe_code)]
    fn materialize_native_fast(&self, bytes: &[u8], shape: IxDyn) -> Result<ArrayData> {
        macro_rules! reinterpret_copy {
            ($ty:ty, $variant:ident) => {{
                let elem_size = std::mem::size_of::<$ty>();
                let align = std::mem::align_of::<$ty>();
                let num_elems = bytes.len() / elem_size;

                // Check alignment - if unaligned, use safe copy
                if (bytes.as_ptr() as usize) % align != 0 {
                    // Unaligned: copy byte by byte (still faster than endian-checking)
                    let vec = if bytes.len() >= PARALLEL_THRESHOLD {
                        // Parallel path for large arrays
                        bytes
                            .par_chunks_exact(elem_size)
                            .map(|chunk| {
                                let mut arr = [0u8; std::mem::size_of::<$ty>()];
                                arr.copy_from_slice(chunk);
                                <$ty>::from_ne_bytes(arr)
                            })
                            .collect()
                    } else {
                        // Sequential path for small arrays
                        bytes
                            .chunks_exact(elem_size)
                            .map(|chunk| {
                                let mut arr = [0u8; std::mem::size_of::<$ty>()];
                                arr.copy_from_slice(chunk);
                                <$ty>::from_ne_bytes(arr)
                            })
                            .collect()
                    };
                    ArrayD::from_shape_vec(shape.f(), vec)
                        .map(ArrayData::$variant)
                        .map_err(|e| Error::InvalidDimensions(e.to_string()))
                } else {
                    // Aligned: direct reinterpretation (fastest path)
                    // Safety: data is aligned and native-endian
                    let ptr = bytes.as_ptr() as *const $ty;
                    let slice = unsafe { std::slice::from_raw_parts(ptr, num_elems) };
                    let vec: Vec<$ty> = slice.to_vec();

                    ArrayD::from_shape_vec(shape.f(), vec)
                        .map(ArrayData::$variant)
                        .map_err(|e| Error::InvalidDimensions(e.to_string()))
                }
            }};
        }

        match self.header.datatype {
            DataType::UInt8 => {
                let vec = bytes.to_vec();
                ArrayD::from_shape_vec(shape.f(), vec)
                    .map(ArrayData::U8)
                    .map_err(|e| Error::InvalidDimensions(e.to_string()))
            }
            DataType::Int8 => {
                let vec: Vec<i8> = bytes.iter().map(|&b| b as i8).collect();
                ArrayD::from_shape_vec(shape.f(), vec)
                    .map(ArrayData::I8)
                    .map_err(|e| Error::InvalidDimensions(e.to_string()))
            }
            DataType::Int16 => reinterpret_copy!(i16, I16),
            DataType::UInt16 => reinterpret_copy!(u16, U16),
            DataType::Int32 => reinterpret_copy!(i32, I32),
            DataType::UInt32 => reinterpret_copy!(u32, U32),
            DataType::Int64 => reinterpret_copy!(i64, I64),
            DataType::UInt64 => reinterpret_copy!(u64, U64),
            DataType::Float16 => reinterpret_copy!(f16, F16),
            DataType::BFloat16 => reinterpret_copy!(bf16, BF16),
            DataType::Float32 => reinterpret_copy!(f32, F32),
            DataType::Float64 => reinterpret_copy!(f64, F64),
        }
    }

    /// Slow path with byte swapping for non-native endian data.
    fn materialize_with_byteswap(&self, bytes: &[u8], shape: IxDyn) -> Result<ArrayData> {
        let is_le = self.header.is_little_endian();
        let use_parallel = bytes.len() >= PARALLEL_THRESHOLD;

        macro_rules! byteswap_copy {
            ($ty:ty, $variant:ident, $read_le:expr, $read_be:expr, $elem_size:expr) => {{
                let vec: Vec<$ty> = if use_parallel {
                    bytes
                        .par_chunks_exact($elem_size)
                        .map(|chunk| {
                            if is_le {
                                $read_le(chunk)
                            } else {
                                $read_be(chunk)
                            }
                        })
                        .collect()
                } else {
                    bytes
                        .chunks_exact($elem_size)
                        .map(|chunk| {
                            if is_le {
                                $read_le(chunk)
                            } else {
                                $read_be(chunk)
                            }
                        })
                        .collect()
                };
                ArrayD::from_shape_vec(shape.f(), vec)
                    .map(ArrayData::$variant)
                    .map_err(|e| Error::InvalidDimensions(e.to_string()))
            }};
        }

        match self.header.datatype {
            DataType::UInt8 => {
                let vec = bytes.to_vec();
                ArrayD::from_shape_vec(shape.f(), vec)
                    .map(ArrayData::U8)
                    .map_err(|e| Error::InvalidDimensions(e.to_string()))
            }
            DataType::Int8 => {
                let vec: Vec<i8> = if use_parallel {
                    bytes.par_iter().map(|&b| b as i8).collect()
                } else {
                    bytes.iter().map(|&b| b as i8).collect()
                };
                ArrayD::from_shape_vec(shape.f(), vec)
                    .map(ArrayData::I8)
                    .map_err(|e| Error::InvalidDimensions(e.to_string()))
            }
            DataType::Int16 => byteswap_copy!(
                i16,
                I16,
                byteorder::LittleEndian::read_i16,
                byteorder::BigEndian::read_i16,
                2
            ),
            DataType::UInt16 => byteswap_copy!(
                u16,
                U16,
                byteorder::LittleEndian::read_u16,
                byteorder::BigEndian::read_u16,
                2
            ),
            DataType::Int32 => byteswap_copy!(
                i32,
                I32,
                byteorder::LittleEndian::read_i32,
                byteorder::BigEndian::read_i32,
                4
            ),
            DataType::UInt32 => byteswap_copy!(
                u32,
                U32,
                byteorder::LittleEndian::read_u32,
                byteorder::BigEndian::read_u32,
                4
            ),
            DataType::Int64 => byteswap_copy!(
                i64,
                I64,
                byteorder::LittleEndian::read_i64,
                byteorder::BigEndian::read_i64,
                8
            ),
            DataType::UInt64 => byteswap_copy!(
                u64,
                U64,
                byteorder::LittleEndian::read_u64,
                byteorder::BigEndian::read_u64,
                8
            ),
            DataType::Float16 => {
                let vec: Vec<f16> = if use_parallel {
                    bytes
                        .par_chunks_exact(2)
                        .map(|chunk| {
                            let bits = if is_le {
                                byteorder::LittleEndian::read_u16(chunk)
                            } else {
                                byteorder::BigEndian::read_u16(chunk)
                            };
                            f16::from_bits(bits)
                        })
                        .collect()
                } else {
                    bytes
                        .chunks_exact(2)
                        .map(|chunk| {
                            let bits = if is_le {
                                byteorder::LittleEndian::read_u16(chunk)
                            } else {
                                byteorder::BigEndian::read_u16(chunk)
                            };
                            f16::from_bits(bits)
                        })
                        .collect()
                };
                ArrayD::from_shape_vec(shape.f(), vec)
                    .map(ArrayData::F16)
                    .map_err(|e| Error::InvalidDimensions(e.to_string()))
            }
            DataType::BFloat16 => {
                let vec: Vec<bf16> = if use_parallel {
                    bytes
                        .par_chunks_exact(2)
                        .map(|chunk| {
                            let bits = if is_le {
                                byteorder::LittleEndian::read_u16(chunk)
                            } else {
                                byteorder::BigEndian::read_u16(chunk)
                            };
                            bf16::from_bits(bits)
                        })
                        .collect()
                } else {
                    bytes
                        .chunks_exact(2)
                        .map(|chunk| {
                            let bits = if is_le {
                                byteorder::LittleEndian::read_u16(chunk)
                            } else {
                                byteorder::BigEndian::read_u16(chunk)
                            };
                            bf16::from_bits(bits)
                        })
                        .collect()
                };
                ArrayD::from_shape_vec(shape.f(), vec)
                    .map(ArrayData::BF16)
                    .map_err(|e| Error::InvalidDimensions(e.to_string()))
            }
            DataType::Float32 => byteswap_copy!(
                f32,
                F32,
                byteorder::LittleEndian::read_f32,
                byteorder::BigEndian::read_f32,
                4
            ),
            DataType::Float64 => byteswap_copy!(
                f64,
                F64,
                byteorder::LittleEndian::read_f64,
                byteorder::BigEndian::read_f64,
                8
            ),
        }
    }

    #[allow(unsafe_code)]
    fn shared_to_f32_slice(&self, bytes: &[u8], slope: f32, inter: f32) -> Result<ArrayD<f32>> {
        let shape = &self.shape_cache;
        let is_le = self.header.is_little_endian();
        let is_native = is_native_endian(is_le);
        let use_parallel = bytes.len() >= PARALLEL_THRESHOLD;
        let identity_scale = slope == 1.0 && inter == 0.0;

        // Fast path: Float32 with native endianness and identity scaling
        if self.header.datatype == DataType::Float32 && is_native && identity_scale {
            let num_elems = bytes.len() / 4;
            let align = std::mem::align_of::<f32>();
            if (bytes.as_ptr() as usize) % align == 0 {
                // Aligned: direct reinterpretation
                let ptr = bytes.as_ptr() as *const f32;
                let slice = unsafe { std::slice::from_raw_parts(ptr, num_elems) };
                return ArrayD::from_shape_vec(IxDyn(shape).f(), slice.to_vec())
                    .map_err(|e| Error::InvalidDimensions(e.to_string()));
            }
        }

        // Parallel/sequential conversion macros
        macro_rules! convert_chunks {
            ($elem_size:expr, $read_le:expr, $read_be:expr) => {{
                if use_parallel {
                    bytes
                        .par_chunks_exact($elem_size)
                        .map(|chunk| {
                            let v = if is_le {
                                $read_le(chunk)
                            } else {
                                $read_be(chunk)
                            };
                            v as f32 * slope + inter
                        })
                        .collect()
                } else {
                    bytes
                        .chunks_exact($elem_size)
                        .map(|chunk| {
                            let v = if is_le {
                                $read_le(chunk)
                            } else {
                                $read_be(chunk)
                            };
                            v as f32 * slope + inter
                        })
                        .collect()
                }
            }};
        }

        let out: Vec<f32> = match self.header.datatype {
            DataType::UInt8 => {
                if use_parallel {
                    bytes
                        .par_iter()
                        .map(|&b| b as f32 * slope + inter)
                        .collect()
                } else {
                    bytes.iter().map(|&b| b as f32 * slope + inter).collect()
                }
            }
            DataType::Int8 => {
                if use_parallel {
                    bytes
                        .par_iter()
                        .map(|&b| (b as i8) as f32 * slope + inter)
                        .collect()
                } else {
                    bytes
                        .iter()
                        .map(|&b| (b as i8) as f32 * slope + inter)
                        .collect()
                }
            }
            DataType::Int16 => convert_chunks!(
                2,
                byteorder::LittleEndian::read_i16,
                byteorder::BigEndian::read_i16
            ),
            DataType::UInt16 => convert_chunks!(
                2,
                byteorder::LittleEndian::read_u16,
                byteorder::BigEndian::read_u16
            ),
            DataType::Int32 => convert_chunks!(
                4,
                byteorder::LittleEndian::read_i32,
                byteorder::BigEndian::read_i32
            ),
            DataType::UInt32 => convert_chunks!(
                4,
                byteorder::LittleEndian::read_u32,
                byteorder::BigEndian::read_u32
            ),
            DataType::Int64 => convert_chunks!(
                8,
                byteorder::LittleEndian::read_i64,
                byteorder::BigEndian::read_i64
            ),
            DataType::UInt64 => convert_chunks!(
                8,
                byteorder::LittleEndian::read_u64,
                byteorder::BigEndian::read_u64
            ),
            DataType::Float16 => {
                if use_parallel {
                    bytes
                        .par_chunks_exact(2)
                        .map(|chunk| {
                            let bits = if is_le {
                                byteorder::LittleEndian::read_u16(chunk)
                            } else {
                                byteorder::BigEndian::read_u16(chunk)
                            };
                            f16::from_bits(bits).to_f32() * slope + inter
                        })
                        .collect()
                } else {
                    bytes
                        .chunks_exact(2)
                        .map(|chunk| {
                            let bits = if is_le {
                                byteorder::LittleEndian::read_u16(chunk)
                            } else {
                                byteorder::BigEndian::read_u16(chunk)
                            };
                            f16::from_bits(bits).to_f32() * slope + inter
                        })
                        .collect()
                }
            }
            DataType::BFloat16 => {
                if use_parallel {
                    bytes
                        .par_chunks_exact(2)
                        .map(|chunk| {
                            let bits = if is_le {
                                byteorder::LittleEndian::read_u16(chunk)
                            } else {
                                byteorder::BigEndian::read_u16(chunk)
                            };
                            bf16::from_bits(bits).to_f32() * slope + inter
                        })
                        .collect()
                } else {
                    bytes
                        .chunks_exact(2)
                        .map(|chunk| {
                            let bits = if is_le {
                                byteorder::LittleEndian::read_u16(chunk)
                            } else {
                                byteorder::BigEndian::read_u16(chunk)
                            };
                            bf16::from_bits(bits).to_f32() * slope + inter
                        })
                        .collect()
                }
            }
            DataType::Float32 => {
                if use_parallel {
                    bytes
                        .par_chunks_exact(4)
                        .map(|chunk| {
                            let v = if is_le {
                                byteorder::LittleEndian::read_f32(chunk)
                            } else {
                                byteorder::BigEndian::read_f32(chunk)
                            };
                            v * slope + inter
                        })
                        .collect()
                } else {
                    bytes
                        .chunks_exact(4)
                        .map(|chunk| {
                            let v = if is_le {
                                byteorder::LittleEndian::read_f32(chunk)
                            } else {
                                byteorder::BigEndian::read_f32(chunk)
                            };
                            v * slope + inter
                        })
                        .collect()
                }
            }
            DataType::Float64 => {
                if use_parallel {
                    bytes
                        .par_chunks_exact(8)
                        .map(|chunk| {
                            let v = if is_le {
                                byteorder::LittleEndian::read_f64(chunk)
                            } else {
                                byteorder::BigEndian::read_f64(chunk)
                            };
                            (v * slope as f64 + inter as f64) as f32
                        })
                        .collect()
                } else {
                    bytes
                        .chunks_exact(8)
                        .map(|chunk| {
                            let v = if is_le {
                                byteorder::LittleEndian::read_f64(chunk)
                            } else {
                                byteorder::BigEndian::read_f64(chunk)
                            };
                            (v * slope as f64 + inter as f64) as f32
                        })
                        .collect()
                }
            }
        };

        ArrayD::from_shape_vec(IxDyn(shape).f(), out)
            .map_err(|e| Error::InvalidDimensions(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_as_view_owned() {
        let data: Vec<u16> = (0..8).collect();
        // Create F-order array to match NIfTI convention
        let c_order = ArrayD::from_shape_vec(IxDyn(&[2, 2, 2]), data.clone()).unwrap();
        let mut f_order: ArrayD<u16> = ArrayD::zeros(IxDyn(&[2, 2, 2]).f());
        f_order.assign(&c_order);
        let img = NiftiImage::from_array(
            f_order,
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        );
        let view = img.as_view_t::<u16>().expect("view should be available");
        assert_eq!(view.len(), 8);
        // F-order view - check via memory order slice
        let view_slice = view.as_slice_memory_order().unwrap();
        let orig = img.materialize_owned();
        if let ArrayData::U16(arr) = orig {
            let orig_slice = arr.as_slice_memory_order().unwrap();
            assert_eq!(view_slice, orig_slice);
        }
    }

    #[test]
    fn test_as_view_shared_mmap() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("view.nii");
        let data: Vec<f32> = (0..8).map(|v| v as f32).collect();
        // Create F-order array to match NIfTI convention
        let c_order = ArrayD::from_shape_vec(IxDyn(&[2, 2, 2]), data.clone()).unwrap();
        let mut f_order: ArrayD<f32> = ArrayD::zeros(IxDyn(&[2, 2, 2]).f());
        f_order.assign(&c_order);
        let img = NiftiImage::from_array(
            f_order.clone(),
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        );
        crate::nifti::save(&img, &path).unwrap();
        let loaded = crate::nifti::load(&path).unwrap();
        let view = loaded
            .as_view_f32()
            .expect("view should exist for mmap f32");
        assert_eq!(view.len(), 8);
        // F-order view - check via memory order slice
        let view_slice = view.as_slice_memory_order().unwrap();
        let orig_slice = f_order.as_slice_memory_order().unwrap();
        assert_eq!(view_slice, orig_slice);
    }
}

impl fmt::Debug for NiftiImage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NiftiImage")
            .field("shape", &self.shape())
            .field("dtype", &self.dtype())
            .field("spacing", &self.spacing())
            .finish()
    }
}

/// Trait for types that can be stored in NIfTI images.
///
/// This trait is sealed and cannot be implemented for types outside this crate.
pub trait NiftiElement: Clone + Copy + 'static + sealed::Sealed {
    /// The NIfTI data type code for this element type.
    const DATA_TYPE: DataType;
    #[doc(hidden)]
    fn wrap_array(arr: ArrayD<Self>) -> ArrayData;
    #[doc(hidden)]
    fn extract_array(data: &ArrayData) -> Option<&ArrayD<Self>>;
    #[doc(hidden)]
    fn max_value() -> Self;
    #[doc(hidden)]
    fn min_value() -> Self;
}

mod sealed {
    use half::{bf16, f16};
    pub trait Sealed {}
    impl Sealed for u8 {}
    impl Sealed for i8 {}
    impl Sealed for i16 {}
    impl Sealed for u16 {}
    impl Sealed for i32 {}
    impl Sealed for u32 {}
    impl Sealed for i64 {}
    impl Sealed for u64 {}
    impl Sealed for f16 {}
    impl Sealed for bf16 {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
}

macro_rules! impl_nifti_element {
    ($ty:ty, $dtype:ident, $variant:ident) => {
        impl NiftiElement for $ty {
            const DATA_TYPE: DataType = DataType::$dtype;

            fn wrap_array(arr: ArrayD<Self>) -> ArrayData {
                ArrayData::$variant(arr)
            }

            fn extract_array(data: &ArrayData) -> Option<&ArrayD<Self>> {
                match data {
                    ArrayData::$variant(a) => Some(a),
                    _ => None,
                }
            }

            fn max_value() -> Self {
                <$ty>::MAX
            }

            fn min_value() -> Self {
                <$ty>::MIN
            }
        }
    };
}

impl_nifti_element!(u8, UInt8, U8);
impl_nifti_element!(i8, Int8, I8);
impl_nifti_element!(i16, Int16, I16);
impl_nifti_element!(u16, UInt16, U16);
impl_nifti_element!(i32, Int32, I32);
impl_nifti_element!(u32, UInt32, U32);
impl_nifti_element!(i64, Int64, I64);
impl_nifti_element!(u64, UInt64, U64);

impl NiftiElement for f16 {
    const DATA_TYPE: DataType = DataType::Float16;

    fn wrap_array(arr: ArrayD<Self>) -> ArrayData {
        ArrayData::F16(arr)
    }

    fn extract_array(data: &ArrayData) -> Option<&ArrayD<Self>> {
        match data {
            ArrayData::F16(a) => Some(a),
            _ => None,
        }
    }

    fn max_value() -> Self {
        f16::MAX
    }

    fn min_value() -> Self {
        f16::MIN
    }
}

impl NiftiElement for bf16 {
    const DATA_TYPE: DataType = DataType::BFloat16;

    fn wrap_array(arr: ArrayD<Self>) -> ArrayData {
        ArrayData::BF16(arr)
    }

    fn extract_array(data: &ArrayData) -> Option<&ArrayD<Self>> {
        match data {
            ArrayData::BF16(a) => Some(a),
            _ => None,
        }
    }

    fn max_value() -> Self {
        bf16::MAX
    }

    fn min_value() -> Self {
        bf16::MIN
    }
}

impl NiftiElement for f32 {
    const DATA_TYPE: DataType = DataType::Float32;

    fn wrap_array(arr: ArrayD<Self>) -> ArrayData {
        ArrayData::F32(arr)
    }

    fn extract_array(data: &ArrayData) -> Option<&ArrayD<Self>> {
        match data {
            ArrayData::F32(a) => Some(a),
            _ => None,
        }
    }

    fn max_value() -> Self {
        f32::MAX
    }

    fn min_value() -> Self {
        f32::MIN
    }
}

impl NiftiElement for f64 {
    const DATA_TYPE: DataType = DataType::Float64;

    fn wrap_array(arr: ArrayD<Self>) -> ArrayData {
        ArrayData::F64(arr)
    }

    fn extract_array(data: &ArrayData) -> Option<&ArrayD<Self>> {
        match data {
            ArrayData::F64(a) => Some(a),
            _ => None,
        }
    }

    fn max_value() -> Self {
        f64::MAX
    }

    fn min_value() -> Self {
        f64::MIN
    }
}
