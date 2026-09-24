//! Voxel element types and type-erased array storage.
//!
//! [`NiftiElement`] is implemented for the twelve voxel types medrs supports.
//! [`ArrayData`] erases the element type; the `with_array!`, `map_array!` and
//! `dispatch_dtype!` macros turn a runtime [`DataType`] back into a generic
//! call so each operation is written once instead of once per type.
//!
//! Every owned array stored in an [`ArrayData`] is Fortran-contiguous with
//! positive strides (the on-disk `NIfTI` order), so its memory-order slice is
//! the voxel sequence of the file. [`ArrayData::into_fortran`] establishes
//! that invariant; all constructors in this crate go through it.

use super::header::DataType;
use crate::error::{Error, Result};
use half::{bf16, f16};
use ndarray::{ArrayD, ArrayViewD, IxDyn, ShapeBuilder};
use rayon::prelude::*;
use std::any::Any;

/// Element count above which element-wise conversions run in parallel.
pub(crate) const PARALLEL_ELEMENTS: usize = 1 << 18;
/// Chunk size (in elements) for parallel element-wise conversions.
const CHUNK_ELEMENTS: usize = 1 << 16;

pub(crate) mod sealed {
    /// Numeric conversions for [`NiftiElement`](super::NiftiElement).
    pub trait Sealed: Sized {
        /// Whether every value converts to `f32` exactly (so scaling can be
        /// done in single precision without extra rounding).
        const EXACT_IN_F32: bool;
        /// Convert to `f64` (exact except for 64-bit integers beyond 2^53).
        fn to_f64(self) -> f64;
        /// Convert to `f32`.
        fn to_f32(self) -> f32;
        /// Convert from `f64`: integers round half to even and saturate, NaN
        /// becomes 0.
        fn from_f64(v: f64) -> Self;
    }
}

/// A voxel type that can be stored in a [`NiftiImage`](super::NiftiImage).
///
/// Sealed: implemented for `u8`, `i8`, `u16`, `i16`, `u32`, `i32`, `u64`,
/// `i64`, `f16`, `bf16`, `f32`, and `f64`.
pub trait NiftiElement:
    bytemuck::Pod + Default + PartialOrd + Send + Sync + std::fmt::Debug + sealed::Sealed + 'static
{
    /// The `NIfTI` datatype for this element type.
    const DATA_TYPE: DataType;
}

macro_rules! impl_int_element {
    ($ty:ty, $dt:ident, $exact:expr) => {
        impl NiftiElement for $ty {
            const DATA_TYPE: DataType = DataType::$dt;
        }
        impl sealed::Sealed for $ty {
            const EXACT_IN_F32: bool = $exact;
            #[inline]
            #[allow(clippy::cast_lossless)]
            fn to_f64(self) -> f64 {
                self as f64
            }
            #[inline]
            #[allow(clippy::cast_lossless)]
            fn to_f32(self) -> f32 {
                self as f32
            }
            #[inline]
            fn from_f64(v: f64) -> Self {
                // Round half to even (like `numpy.rint`); `as` saturates and
                // maps NaN to 0.
                v.round_ties_even() as $ty
            }
        }
    };
}

impl_int_element!(u8, UInt8, true);
impl_int_element!(i8, Int8, true);
impl_int_element!(u16, UInt16, true);
impl_int_element!(i16, Int16, true);
impl_int_element!(u32, UInt32, false);
impl_int_element!(i32, Int32, false);
impl_int_element!(u64, UInt64, false);
impl_int_element!(i64, Int64, false);

macro_rules! impl_float_element {
    ($ty:ty, $dt:ident, $exact:expr, $to64:expr, $to32:expr, $from64:expr) => {
        impl NiftiElement for $ty {
            const DATA_TYPE: DataType = DataType::$dt;
        }
        impl sealed::Sealed for $ty {
            const EXACT_IN_F32: bool = $exact;
            #[inline]
            fn to_f64(self) -> f64 {
                ($to64)(self)
            }
            #[inline]
            fn to_f32(self) -> f32 {
                ($to32)(self)
            }
            #[inline]
            fn from_f64(v: f64) -> Self {
                ($from64)(v)
            }
        }
    };
}

impl_float_element!(f16, Float16, true, f16::to_f64, f16::to_f32, f16::from_f64);
impl_float_element!(
    bf16,
    BFloat16,
    true,
    bf16::to_f64,
    bf16::to_f32,
    bf16::from_f64
);
impl_float_element!(
    f32,
    Float32,
    true,
    f64::from,
    std::convert::identity,
    |v: f64| v as f32
);
impl_float_element!(
    f64,
    Float64,
    false,
    std::convert::identity,
    |v: f64| v as f32,
    std::convert::identity
);

/// Type-erased voxel array. Owned arrays are always Fortran-contiguous.
#[derive(Clone, Debug, PartialEq)]
#[allow(missing_docs)]
pub(crate) enum ArrayData {
    U8(ArrayD<u8>),
    I8(ArrayD<i8>),
    U16(ArrayD<u16>),
    I16(ArrayD<i16>),
    U32(ArrayD<u32>),
    I32(ArrayD<i32>),
    U64(ArrayD<u64>),
    I64(ArrayD<i64>),
    F16(ArrayD<f16>),
    BF16(ArrayD<bf16>),
    F32(ArrayD<f32>),
    F64(ArrayD<f64>),
}

/// Evaluate `$body` with `$a` bound to the inner array of an `ArrayData`
/// (by value or reference, depending on `$data`).
macro_rules! with_array {
    ($data:expr, |$a:ident| $body:expr) => {
        match $data {
            $crate::nifti::element::ArrayData::U8($a) => $body,
            $crate::nifti::element::ArrayData::I8($a) => $body,
            $crate::nifti::element::ArrayData::U16($a) => $body,
            $crate::nifti::element::ArrayData::I16($a) => $body,
            $crate::nifti::element::ArrayData::U32($a) => $body,
            $crate::nifti::element::ArrayData::I32($a) => $body,
            $crate::nifti::element::ArrayData::U64($a) => $body,
            $crate::nifti::element::ArrayData::I64($a) => $body,
            $crate::nifti::element::ArrayData::F16($a) => $body,
            $crate::nifti::element::ArrayData::BF16($a) => $body,
            $crate::nifti::element::ArrayData::F32($a) => $body,
            $crate::nifti::element::ArrayData::F64($a) => $body,
        }
    };
}

/// Map an `ArrayData` to another `ArrayData` of the same element type,
/// normalizing the result to Fortran order.
macro_rules! map_array {
    ($data:expr, |$a:ident| $body:expr) => {
        match $data {
            $crate::nifti::element::ArrayData::U8($a) => {
                $crate::nifti::element::ArrayData::new($body)
            }
            $crate::nifti::element::ArrayData::I8($a) => {
                $crate::nifti::element::ArrayData::new($body)
            }
            $crate::nifti::element::ArrayData::U16($a) => {
                $crate::nifti::element::ArrayData::new($body)
            }
            $crate::nifti::element::ArrayData::I16($a) => {
                $crate::nifti::element::ArrayData::new($body)
            }
            $crate::nifti::element::ArrayData::U32($a) => {
                $crate::nifti::element::ArrayData::new($body)
            }
            $crate::nifti::element::ArrayData::I32($a) => {
                $crate::nifti::element::ArrayData::new($body)
            }
            $crate::nifti::element::ArrayData::U64($a) => {
                $crate::nifti::element::ArrayData::new($body)
            }
            $crate::nifti::element::ArrayData::I64($a) => {
                $crate::nifti::element::ArrayData::new($body)
            }
            $crate::nifti::element::ArrayData::F16($a) => {
                $crate::nifti::element::ArrayData::new($body)
            }
            $crate::nifti::element::ArrayData::BF16($a) => {
                $crate::nifti::element::ArrayData::new($body)
            }
            $crate::nifti::element::ArrayData::F32($a) => {
                $crate::nifti::element::ArrayData::new($body)
            }
            $crate::nifti::element::ArrayData::F64($a) => {
                $crate::nifti::element::ArrayData::new($body)
            }
        }
    };
}

/// Evaluate `$body` with the type alias `$T` set to the Rust element type of
/// a runtime [`DataType`].
macro_rules! dispatch_dtype {
    ($dtype:expr, $T:ident => $body:expr) => {
        match $dtype {
            $crate::nifti::DataType::UInt8 => {
                #[allow(unused_imports)]
                use $crate::nifti::element::sealed::Sealed as _;
                type $T = u8;
                $body
            }
            $crate::nifti::DataType::Int8 => {
                #[allow(unused_imports)]
                use $crate::nifti::element::sealed::Sealed as _;
                type $T = i8;
                $body
            }
            $crate::nifti::DataType::UInt16 => {
                #[allow(unused_imports)]
                use $crate::nifti::element::sealed::Sealed as _;
                type $T = u16;
                $body
            }
            $crate::nifti::DataType::Int16 => {
                #[allow(unused_imports)]
                use $crate::nifti::element::sealed::Sealed as _;
                type $T = i16;
                $body
            }
            $crate::nifti::DataType::UInt32 => {
                #[allow(unused_imports)]
                use $crate::nifti::element::sealed::Sealed as _;
                type $T = u32;
                $body
            }
            $crate::nifti::DataType::Int32 => {
                #[allow(unused_imports)]
                use $crate::nifti::element::sealed::Sealed as _;
                type $T = i32;
                $body
            }
            $crate::nifti::DataType::UInt64 => {
                #[allow(unused_imports)]
                use $crate::nifti::element::sealed::Sealed as _;
                type $T = u64;
                $body
            }
            $crate::nifti::DataType::Int64 => {
                #[allow(unused_imports)]
                use $crate::nifti::element::sealed::Sealed as _;
                type $T = i64;
                $body
            }
            $crate::nifti::DataType::Float16 => {
                #[allow(unused_imports)]
                use $crate::nifti::element::sealed::Sealed as _;
                type $T = half::f16;
                $body
            }
            $crate::nifti::DataType::BFloat16 => {
                #[allow(unused_imports)]
                use $crate::nifti::element::sealed::Sealed as _;
                type $T = half::bf16;
                $body
            }
            $crate::nifti::DataType::Float32 => {
                #[allow(unused_imports)]
                use $crate::nifti::element::sealed::Sealed as _;
                type $T = f32;
                $body
            }
            $crate::nifti::DataType::Float64 => {
                #[allow(unused_imports)]
                use $crate::nifti::element::sealed::Sealed as _;
                type $T = f64;
                $body
            }
        }
    };
}

pub(crate) use {dispatch_dtype, map_array, with_array};

impl ArrayData {
    /// Wrap an owned array, normalizing it to Fortran order.
    pub(crate) fn new<T: NiftiElement>(array: ArrayD<T>) -> Self {
        let mut slot = Some(into_fortran(array));
        let any: &mut dyn Any = &mut slot;
        macro_rules! pick {
            ($($variant:ident: $t:ty),*) => {$(
                if let Some(a) = any.downcast_mut::<Option<ArrayD<$t>>>().and_then(Option::take) {
                    return Self::$variant(a);
                }
            )*};
        }
        pick!(U8: u8, I8: i8, U16: u16, I16: i16, U32: u32, I32: i32, U64: u64, I64: i64,
              F16: f16, BF16: bf16, F32: f32, F64: f64);
        unreachable!("NiftiElement is sealed to the ArrayData element types")
    }

    /// Borrow the array if it holds elements of type `T`.
    pub(crate) fn as_array<T: NiftiElement>(&self) -> Option<&ArrayD<T>> {
        with_array!(self, |a| (a as &dyn Any).downcast_ref::<ArrayD<T>>())
    }

    /// Take the array if it holds elements of type `T`.
    pub(crate) fn into_array<T: NiftiElement>(self) -> std::result::Result<ArrayD<T>, Self> {
        macro_rules! take {
            ($($variant:ident),*) => {
                match self {
                    $(Self::$variant(a) => {
                        let mut slot = Some(a);
                        let any: &mut dyn Any = &mut slot;
                        if let Some(a) = any.downcast_mut::<Option<ArrayD<T>>>().and_then(Option::take) {
                            return Ok(a);
                        }
                        match slot {
                            Some(a) => Err(Self::$variant(a)),
                            None => unreachable!("slot is only emptied on a successful downcast"),
                        }
                    })*
                }
            };
        }
        take!(U8, I8, U16, I16, U32, I32, U64, I64, F16, BF16, F32, F64)
    }

    /// Element datatype.
    pub(crate) fn dtype(&self) -> DataType {
        fn dt<T: NiftiElement>(_: &ArrayD<T>) -> DataType {
            T::DATA_TYPE
        }
        with_array!(self, |a| dt(a))
    }

    /// Array shape.
    pub(crate) fn shape(&self) -> &[usize] {
        with_array!(self, |a| a.shape())
    }

    /// Whether the array is Fortran-contiguous with positive strides.
    pub(crate) fn is_fortran(&self) -> bool {
        with_array!(self, |a| is_fortran(a))
    }
}

/// Whether `a` is Fortran-contiguous with positive strides.
pub(crate) fn is_fortran<T>(a: &ArrayD<T>) -> bool {
    a.t().is_standard_layout()
}

/// Copy `a` into Fortran order unless it already is (then it is moved).
pub(crate) fn into_fortran<T: Clone + Default>(a: ArrayD<T>) -> ArrayD<T> {
    if is_fortran(&a) {
        a
    } else {
        to_fortran(&a.view())
    }
}

/// Copy any view into a new Fortran-ordered array.
pub(crate) fn to_fortran<T: Clone + Default>(view: &ArrayViewD<'_, T>) -> ArrayD<T> {
    if view.t().is_standard_layout() {
        // Already Fortran-contiguous: a straight copy keeps the layout.
        return view.to_owned();
    }
    // `assign` walks the output in memory order with tight strided inner
    // loops, far faster than collecting a generic element iterator.
    let mut out = ArrayD::from_elem(IxDyn(view.shape()).f(), T::default());
    out.assign(view);
    out
}

/// Build a Fortran-ordered array from data already in Fortran order.
///
/// # Panics
/// Never in practice: callers pass `data.len() == shape.product()`; a mismatch
/// is an internal bug.
pub(crate) fn fortran_from_vec<T>(shape: &[usize], data: Vec<T>) -> ArrayD<T> {
    debug_assert_eq!(shape.iter().product::<usize>(), data.len());
    match ArrayD::from_shape_vec(IxDyn(shape).f(), data) {
        Ok(a) => a,
        Err(e) => unreachable!("fortran_from_vec: shape/data mismatch: {e}"),
    }
}

/// The elements of an owned array in memory order.
///
/// Owned arrays in medrs are always contiguous; failure here is a bug.
pub(crate) fn contiguous<T>(a: &ArrayD<T>) -> Result<&[T]> {
    a.as_slice_memory_order()
        .ok_or_else(|| Error::Internal("owned voxel array is not contiguous".into()))
}

/// Like [`fortran_from_vec`] but returns an error instead of panicking.
pub(crate) fn try_fortran_from_vec<T>(shape: &[usize], data: Vec<T>) -> Result<ArrayD<T>> {
    ArrayD::from_shape_vec(IxDyn(shape).f(), data)
        .map_err(|e| Error::ShapeMismatch(format!("{e} (shape {shape:?})")))
}

// ---------------------------------------------------------------------------
// Element sources: owned slices and raw (possibly unaligned or byte-swapped)
// file bytes, both in Fortran order.
// ---------------------------------------------------------------------------

/// A read-only sequence of voxel values in file (Fortran) order.
pub(crate) enum Elements<'a, T> {
    /// Native-endian, aligned values.
    Slice(&'a [T]),
    /// Raw bytes that may be unaligned and/or need byte swapping.
    Bytes {
        /// The raw bytes (`len == count * size_of::<T>()`).
        bytes: &'a [u8],
        /// Whether each element must be byte-swapped.
        swap: bool,
    },
}

#[inline]
fn read_elem<T: bytemuck::Pod>(chunk: &[u8], swap: bool) -> T {
    let mut v: T = bytemuck::pod_read_unaligned(chunk);
    if swap {
        bytemuck::bytes_of_mut(&mut v).reverse();
    }
    v
}

impl<'a, T: NiftiElement> Elements<'a, T> {
    /// View raw bytes as elements, borrowing them directly when they are
    /// native-endian and suitably aligned.
    pub(crate) fn from_bytes(bytes: &'a [u8], little_endian: bool) -> Self {
        let size = std::mem::size_of::<T>();
        let swap = size > 1 && little_endian != cfg!(target_endian = "little");
        if !swap {
            if let Ok(slice) = bytemuck::try_cast_slice::<u8, T>(bytes) {
                return Self::Slice(slice);
            }
        }
        Self::Bytes { bytes, swap }
    }

    /// Number of elements.
    pub(crate) fn len(&self) -> usize {
        match self {
            Self::Slice(s) => s.len(),
            Self::Bytes { bytes, .. } => bytes.len() / std::mem::size_of::<T>(),
        }
    }

    /// Write `f(value)` for every element into `out` (same length).
    pub(crate) fn map_into<U, F>(&self, out: &mut [U], f: F)
    where
        U: Send,
        F: Fn(T) -> U + Sync + Send,
    {
        debug_assert_eq!(out.len(), self.len());
        let parallel = out.len() >= PARALLEL_ELEMENTS;
        match *self {
            Self::Slice(src) => {
                let run = |(o, s): (&mut [U], &[T])| {
                    for (dst, &v) in o.iter_mut().zip(s) {
                        *dst = f(v);
                    }
                };
                if parallel {
                    crate::parallel::install(|| {
                        out.par_chunks_mut(CHUNK_ELEMENTS)
                            .zip(src.par_chunks(CHUNK_ELEMENTS))
                            .for_each(run);
                    });
                } else {
                    run((out, src));
                }
            }
            Self::Bytes { bytes, swap } => {
                let size = std::mem::size_of::<T>();
                let run = |(o, s): (&mut [U], &[u8])| {
                    if swap {
                        for (dst, c) in o.iter_mut().zip(s.chunks_exact(size)) {
                            *dst = f(read_elem::<T>(c, true));
                        }
                    } else {
                        for (dst, c) in o.iter_mut().zip(s.chunks_exact(size)) {
                            *dst = f(read_elem::<T>(c, false));
                        }
                    }
                };
                if parallel {
                    crate::parallel::install(|| {
                        out.par_chunks_mut(CHUNK_ELEMENTS)
                            .zip(bytes.par_chunks(CHUNK_ELEMENTS * size))
                            .for_each(run);
                    });
                } else {
                    run((out, bytes));
                }
            }
        }
    }

    /// Map every element through `f` into a new vector.
    pub(crate) fn map_vec<U, F>(&self, f: F) -> Vec<U>
    where
        U: Send + Default + Clone,
        F: Fn(T) -> U + Sync + Send,
    {
        let mut out = vec![U::default(); self.len()];
        self.map_into(&mut out, f);
        out
    }

    /// Borrow the values when they are an aligned native slice, else copy.
    pub(crate) fn as_cow(&self) -> std::borrow::Cow<'a, [T]> {
        match *self {
            Self::Slice(s) => std::borrow::Cow::Borrowed(s),
            Self::Bytes { .. } => std::borrow::Cow::Owned(self.to_vec()),
        }
    }

    /// Copy the values into an owned, native-endian vector.
    #[allow(unsafe_code)]
    pub(crate) fn to_vec(&self) -> Vec<T> {
        match *self {
            Self::Slice(s) => s.to_vec(),
            Self::Bytes { bytes, swap: false } => {
                let n = bytes.len() / std::mem::size_of::<T>();
                let mut v: Vec<T> = Vec::with_capacity(n);
                // SAFETY: `T: Pod`, so any byte pattern is a valid `T`; the
                // destination has capacity for `n` elements, which is exactly
                // `bytes.len()` bytes; source and destination do not overlap.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        bytes.as_ptr(),
                        v.as_mut_ptr().cast::<u8>(),
                        n * std::mem::size_of::<T>(),
                    );
                    v.set_len(n);
                }
                v
            }
            Self::Bytes { .. } => self.map_vec(std::convert::identity),
        }
    }
}

/// Values scaled as `v * slope + inter`, converted to `f32`.
///
/// Types that are exact in `f32` are scaled in single precision; wider types
/// are scaled in double precision and rounded once.
pub(crate) fn scaled_f32<T: NiftiElement>(
    src: &Elements<'_, T>,
    slope: f64,
    inter: f64,
) -> Vec<f32> {
    // Plain multiply-add rather than `mul_add`: without hardware FMA in the
    // target baseline, `mul_add` becomes a slow libm call per element.
    let identity = slope == 1.0 && inter == 0.0;
    if identity && T::DATA_TYPE == DataType::Float32 {
        // `T` is f32 here, so this is a plain copy.
        return match bytemuck::allocation::try_cast_vec::<T, f32>(src.to_vec()) {
            Ok(v) => v,
            Err((_, v)) => v.into_iter().map(T::to_f32).collect(),
        };
    }
    if identity {
        src.map_vec(T::to_f32)
    } else if T::EXACT_IN_F32 {
        let (s, i) = (slope as f32, inter as f32);
        src.map_vec(move |v| v.to_f32() * s + i)
    } else {
        src.map_vec(move |v| (v.to_f64() * slope + inter) as f32)
    }
}

/// Values scaled as `v * slope + inter` in double precision.
pub(crate) fn scaled_f64<T: NiftiElement>(
    src: &Elements<'_, T>,
    slope: f64,
    inter: f64,
) -> Vec<f64> {
    if slope == 1.0 && inter == 0.0 {
        src.map_vec(T::to_f64)
    } else {
        src.map_vec(move |v| v.to_f64() * slope + inter)
    }
}

/// Values scaled and converted to `U` (rounding and saturating for integers).
pub(crate) fn scaled_as<T: NiftiElement, U: NiftiElement>(
    src: &Elements<'_, T>,
    slope: f64,
    inter: f64,
) -> Vec<U> {
    if T::DATA_TYPE == U::DATA_TYPE && slope == 1.0 && inter == 0.0 {
        // Same type and no scaling: a plain copy.
        match bytemuck::allocation::try_cast_vec::<T, U>(src.to_vec()) {
            Ok(v) => return v,
            Err((_, v)) => return v.into_iter().map(|x| U::from_f64(x.to_f64())).collect(),
        }
    }
    src.map_vec(move |v| U::from_f64(v.to_f64() * slope + inter))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{s, Axis};

    #[test]
    fn into_fortran_normalizes_every_layout() {
        let c = ArrayD::from_shape_vec(IxDyn(&[2, 3, 4]), (0..24).collect::<Vec<i32>>()).unwrap();
        let expected: Vec<i32> = c.t().iter().copied().collect();

        let f = into_fortran(c.clone());
        assert!(is_fortran(&f));
        assert_eq!(f.as_slice_memory_order().unwrap(), &expected[..]);
        assert_eq!(f, c);

        let mut flipped = f.clone();
        flipped.invert_axis(Axis(1));
        assert!(!is_fortran(&flipped));
        let normalized = into_fortran(flipped.clone());
        assert!(is_fortran(&normalized));
        assert_eq!(normalized, flipped);

        let sliced = c.slice(s![.., 1.., ..;2]).to_owned().into_dyn();
        let normalized = into_fortran(sliced.clone());
        assert!(is_fortran(&normalized));
        assert_eq!(normalized, sliced);
    }

    #[test]
    fn elements_from_unaligned_and_swapped_bytes() {
        let values: Vec<i16> = vec![1, -2, 300, -32768, 32767];
        let mut le: Vec<u8> = vec![0];
        le.extend(values.iter().flat_map(|v| v.to_le_bytes()));
        let be: Vec<u8> = values.iter().flat_map(|v| v.to_be_bytes()).collect();

        let unaligned = Elements::<i16>::from_bytes(&le[1..], true);
        assert_eq!(unaligned.to_vec(), values);
        let swapped = Elements::<i16>::from_bytes(&be, false);
        assert!(matches!(swapped, Elements::Bytes { swap: true, .. }));
        assert_eq!(swapped.to_vec(), values);
    }

    #[test]
    fn scaled_conversions() {
        let v: Vec<u8> = vec![0, 1, 255];
        let e = Elements::Slice(&v[..]);
        assert_eq!(scaled_f32(&e, 2.0, -1.0), vec![-1.0, 1.0, 509.0]);
        assert_eq!(scaled_f64(&e, 0.5, 0.0), vec![0.0, 0.5, 127.5]);
        let as_i8: Vec<i8> = scaled_as(&e, 1.0, -100.0);
        assert_eq!(as_i8, vec![-100, -99, 127]);
        let big: Vec<i64> = vec![i64::MAX];
        let as_f32 = scaled_f32(&Elements::Slice(&big[..]), 1.0, 0.0);
        assert_eq!(as_f32, vec![9.223_372e18]);
        assert_eq!(<u8 as sealed::Sealed>::from_f64(f64::NAN), 0);
        assert_eq!(<i16 as sealed::Sealed>::from_f64(1e9), i16::MAX);
        assert_eq!(<i16 as sealed::Sealed>::from_f64(-2.5), -2);
        assert_eq!(<i16 as sealed::Sealed>::from_f64(3.5), 4);
    }

    #[test]
    fn parallel_path_matches_sequential() {
        let n = PARALLEL_ELEMENTS + 12_345;
        let v: Vec<u16> = (0..n).map(|i| (i % 65_536) as u16).collect();
        let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_be_bytes()).collect();
        let e = Elements::<u16>::from_bytes(&bytes, false);
        let out = scaled_f32(&e, 1.0, 0.5);
        assert!(out
            .iter()
            .enumerate()
            .all(|(i, &x)| x == (i % 65_536) as f32 + 0.5));
    }
}
