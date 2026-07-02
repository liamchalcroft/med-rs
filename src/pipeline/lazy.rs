//! Lazy evaluation infrastructure for transform pipelines.
//!
//! Lazy transforms accumulate operations without executing them immediately.
//! When the data is finally needed, all pending operations are composed and
//! executed in a single optimized pass.

use crate::error::{Error, Result};
use crate::nifti::image::ArrayData;
use crate::nifti::DataType;
use crate::nifti::NiftiImage;
use crate::pipeline::simd_kernels::{
    parallel_linear_transform_clamp_f32, parallel_linear_transform_f32,
    parallel_sum_and_sum_sq_f32, trilinear_resample_forder_adaptive,
};
use crate::transforms::Interpolation as TransformsInterpolation;
use ndarray::{ArrayD, IxDyn};
use rayon::prelude::*;

/// Chunk size for the second z-normalization pass.
const ZNORM_CHUNK_SIZE: usize = 4096;

/// A pending operation that can be lazily evaluated.
#[derive(Clone, Debug)]
pub enum PendingOp {
    /// Affine spatial transformation (4x4 matrix).
    /// Multiple affine transforms can be composed by matrix multiplication.
    Affine {
        /// Homogeneous transform matrix applied to voxel coordinates.
        matrix: [[f32; 4]; 4],
        /// Optional output shape override (preallocations/shape change).
        output_shape: Option<[usize; 3]>,
        /// Interpolation strategy to use for resampling.
        interpolation: Interpolation,
    },
    /// Intensity normalization to zero mean and unit variance. Statistics are
    /// deferred to materialization time so the op can be fused with clamping and
    /// linear scaling in a single pass.
    ZNormalize,
    /// Linear intensity transform: output = input * scale + offset
    /// This can represent rescaling, clamping bounds, etc.
    LinearIntensity {
        /// Multiplicative factor.
        scale: f32,
        /// Additive offset.
        offset: f32,
    },
    /// Clamp to range
    Clamp {
        /// Minimum allowed value.
        min: f32,
        /// Maximum allowed value.
        max: f32,
    },
    /// Flip along axes (stored as bitmask: bit 0 = axis 0, etc.)
    Flip {
        /// Bitmask of axes to flip (bit 0 = depth, 1 = height, 2 = width).
        axes: u8,
    },
}

/// Interpolation mode for resampling operations.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Interpolation {
    Nearest,
    #[default]
    Trilinear,
}

impl PendingOp {
    /// Check if this operation can be fused with another.
    pub fn can_fuse_with(&self, other: &PendingOp) -> bool {
        match (self, other) {
            // Affine transforms can be composed if same interpolation mode
            (
                PendingOp::Affine {
                    interpolation: i1, ..
                },
                PendingOp::Affine {
                    interpolation: i2, ..
                },
            ) => i1 == i2,
            // Fuseable linear intensity operations
            (PendingOp::LinearIntensity { .. }, PendingOp::LinearIntensity { .. })
            | (PendingOp::LinearIntensity { .. }, PendingOp::Clamp { .. }) => true,
            _ => false,
        }
    }

    /// Fuse this operation with another, returning the combined operation.
    pub fn fuse_with(&self, other: &PendingOp) -> Option<PendingOp> {
        match (self, other) {
            (
                PendingOp::Affine {
                    matrix: m1,
                    interpolation,
                    ..
                },
                PendingOp::Affine {
                    matrix: m2,
                    output_shape,
                    ..
                },
            ) => {
                // Compose affine matrices: result = m2 * m1
                let composed = compose_affine(m1, m2);
                Some(PendingOp::Affine {
                    matrix: composed,
                    output_shape: *output_shape,
                    interpolation: *interpolation,
                })
            }
            (
                PendingOp::LinearIntensity {
                    scale: s1,
                    offset: o1,
                },
                PendingOp::LinearIntensity {
                    scale: s2,
                    offset: o2,
                },
            ) => {
                // (s1*x + o1) * s2 + o2 = s1*s2*x + o1*s2 + o2
                Some(PendingOp::LinearIntensity {
                    scale: s1 * s2,
                    offset: o1 * s2 + o2,
                })
            }
            (PendingOp::LinearIntensity { .. }, PendingOp::Clamp { min, max }) => {
                Some(PendingOp::Clamp {
                    min: *min,
                    max: *max,
                })
            }
            _ => None,
        }
    }
}

/// Compose two 4x4 affine matrices: result = b * a
fn compose_affine(a: &[[f32; 4]; 4], b: &[[f32; 4]; 4]) -> [[f32; 4]; 4] {
    let mut result = [[0.0f32; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            for k in 0..4 {
                result[i][j] += b[i][k] * a[k][j];
            }
        }
    }
    result
}

/// Append a pending operation, fusing it into the previous one when possible.
pub(crate) fn push_pending(pending: &mut Vec<PendingOp>, op: PendingOp) {
    if let Some(last) = pending.last() {
        if last.can_fuse_with(&op) {
            if let Some(fused) = last.fuse_with(&op) {
                pending.pop();
                pending.push(fused);
                return;
            }
        }
    }
    pending.push(op);
}

/// A lazy image that accumulates pending operations.
#[derive(Clone)]
pub struct LazyImage {
    /// The underlying image data (may be None if not yet loaded).
    pub(crate) image: Option<NiftiImage>,
    /// Path to load image from (for deferred loading).
    pub(crate) path: Option<String>,
    /// Pending operations to apply.
    pub(crate) pending: Vec<PendingOp>,
}

impl LazyImage {
    /// Create a new lazy image from an existing NiftiImage.
    pub fn from_image(image: NiftiImage) -> Self {
        Self {
            image: Some(image),
            path: None,
            pending: Vec::new(),
        }
    }

    /// Create a lazy image from a file path (deferred loading).
    pub fn from_path(path: impl Into<String>) -> Self {
        Self {
            image: None,
            path: Some(path.into()),
            pending: Vec::new(),
        }
    }

    /// Add a pending operation.
    pub fn push_op(&mut self, op: PendingOp) {
        push_pending(&mut self.pending, op);
    }

    /// Check if there are pending operations.
    pub fn has_pending(&self) -> bool {
        !self.pending.is_empty()
    }

    /// Get the number of pending operations.
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }

    /// Execute all pending operations and return the materialized image.
    pub fn materialize(self) -> Result<NiftiImage> {
        let image = if let Some(img) = self.image {
            img
        } else if let Some(path) = &self.path {
            crate::nifti::load(path)?
        } else {
            return Err(Error::InvalidDimensions(
                "LazyImage has no image or path".into(),
            ));
        };

        materialize_ops(&image, &self.pending)
    }

    /// Get a reference to the pending operations.
    pub fn pending_ops(&self) -> &[PendingOp] {
        &self.pending
    }
}

/// Apply a chain of pending operations to an image, fusing intensity ops.
pub(crate) fn materialize_ops(image: &NiftiImage, pending: &[PendingOp]) -> Result<NiftiImage> {
    if let Some(fused) = execute_fused_intensity(image, pending)? {
        return Ok(fused);
    }

    let mut result: Option<NiftiImage> = None;
    for op in pending {
        let input = result.as_ref().unwrap_or(image);
        result = Some(execute_op(input, op)?);
    }
    result.map_or_else(|| Ok(image.clone()), Ok)
}

/// Execute a single pending operation on an image.
fn execute_op(image: &NiftiImage, op: &PendingOp) -> Result<NiftiImage> {
    use crate::transforms;

    match op {
        PendingOp::Affine {
            matrix,
            output_shape,
            interpolation,
        } => {
            let shape = output_shape.unwrap_or_else(|| {
                let shp = image.shape();
                [shp[0], shp[1], shp[2]]
            });
            let interp = match interpolation {
                Interpolation::Nearest => TransformsInterpolation::Nearest,
                Interpolation::Trilinear => TransformsInterpolation::Trilinear,
            };
            apply_affine(image, matrix, shape, interp)
        }
        PendingOp::ZNormalize => transforms::z_normalization(image),
        PendingOp::LinearIntensity { scale, offset } => {
            apply_linear_intensity(image, *scale, *offset)
        }
        PendingOp::Clamp { min, max } => transforms::clamp(image, *min as f64, *max as f64),
        PendingOp::Flip { axes } => {
            let axes_vec: Vec<usize> = (0..3).filter(|&i| (axes >> i) & 1 == 1).collect();
            transforms::flip(image, &axes_vec)
        }
    }
}

/// Borrow the image data as a contiguous f32 slice, converting only when the
/// storage is not already owned identity-scaled f32.
fn f32_input<'a>(image: &'a NiftiImage, owned: &'a mut Option<ArrayD<f32>>) -> Result<&'a [f32]> {
    if let Some(slice) = image.as_f32_slice() {
        return Ok(slice);
    }
    let arr = owned.insert(image.to_f32()?);
    arr.as_slice_memory_order()
        .ok_or_else(|| Error::NonContiguousArray("Array not contiguous".into()))
}

/// Execute a chain of intensity ops in a single fused pass.
///
/// Returns `Ok(None)` when the pending list contains a non-intensity op (the
/// caller falls back to per-op execution), and `Err` when z-normalization
/// statistics are not finite.
pub(crate) fn execute_fused_intensity(
    image: &NiftiImage,
    pending: &[PendingOp],
) -> Result<Option<NiftiImage>> {
    use ndarray::ShapeBuilder;

    if pending.is_empty() {
        return Ok(None);
    }

    let mut do_znorm = false;
    let mut scale = 1.0f32;
    let mut offset = 0.0f32;
    let mut clamp: Option<(f32, f32)> = None;

    for op in pending {
        match op {
            PendingOp::ZNormalize => do_znorm = true,
            PendingOp::LinearIntensity {
                scale: s,
                offset: o,
            } => {
                offset = offset * s + o;
                scale *= s;
            }
            PendingOp::Clamp { min, max } => clamp = Some((*min, *max)),
            _ => return Ok(None),
        }
    }

    let mut owned = None;
    let slice = f32_input(image, &mut owned)?;

    if do_znorm {
        let (mean, inv_std) = znorm_stats(slice)?;
        // Compose z-normalization (applied first) with the accumulated linear op:
        // ((x * inv_std) - mean * inv_std) * scale + offset
        offset += -mean * inv_std * scale;
        scale *= inv_std;
    }

    let shape = image.shape();
    let mut output = vec![0.0f32; slice.len()];
    match clamp {
        Some((min, max)) => {
            parallel_linear_transform_clamp_f32(slice, &mut output, scale, offset, min, max);
        }
        None => parallel_linear_transform_f32(slice, &mut output, scale, offset),
    }

    let out_array = ArrayD::from_shape_vec(IxDyn(shape).f(), output)
        .map_err(|e| Error::InvalidDimensions(format!("Shape mismatch: {}", e)))?;
    let mut header = image.header().clone();
    header.datatype = DataType::Float32;
    header.scl_slope = 1.0;
    header.scl_inter = 0.0;
    Ok(Some(NiftiImage::from_parts(
        header,
        ArrayData::F32(out_array),
    )))
}

/// Two-pass mean and inverse standard deviation, matching the eager
/// `z_normalization` transform (mean, then squared deviations in f64).
fn znorm_stats(slice: &[f32]) -> Result<(f32, f32)> {
    let len = slice.len();
    if len == 0 {
        return Err(Error::InvalidDimensions(
            "z-normalization on empty array".into(),
        ));
    }

    let (sum, _, _) = parallel_sum_and_sum_sq_f32(slice);
    let mean = sum / len as f64;

    let sum_sq_dev: f64 = slice
        .par_chunks(ZNORM_CHUNK_SIZE)
        .map(|chunk| {
            chunk
                .iter()
                .map(|&v| {
                    let d = v as f64 - mean;
                    d * d
                })
                .sum::<f64>()
        })
        .sum();
    let variance = sum_sq_dev / len as f64;

    if !mean.is_finite() || !variance.is_finite() {
        return Err(Error::Configuration(
            "z-normalization statistics are not finite (input contains NaN or infinity)".into(),
        ));
    }

    let inv_std = if variance <= 0.0 {
        1.0f32
    } else {
        1.0 / (variance.sqrt() as f32)
    };
    Ok((mean as f32, inv_std))
}

/// Apply a linear intensity transformation: output = input * scale + offset
fn apply_linear_intensity(image: &NiftiImage, scale: f32, offset: f32) -> Result<NiftiImage> {
    use ndarray::ShapeBuilder;

    let mut owned = None;
    let slice = f32_input(image, &mut owned)?;
    let mut output = vec![0.0f32; slice.len()];
    parallel_linear_transform_f32(slice, &mut output, scale, offset);

    let shape = image.shape();
    let out_array = ArrayD::from_shape_vec(IxDyn(shape).f(), output)
        .map_err(|e| Error::InvalidDimensions(format!("Shape mismatch: {}", e)))?;
    let mut header = image.header().clone();
    header.datatype = DataType::Float32;
    header.scl_slope = 1.0;
    header.scl_inter = 0.0;
    Ok(NiftiImage::from_parts(header, ArrayData::F32(out_array)))
}

#[allow(clippy::similar_names)]
fn apply_affine(
    image: &NiftiImage,
    matrix: &[[f32; 4]; 4],
    output_shape: [usize; 3],
    interpolation: TransformsInterpolation,
) -> Result<NiftiImage> {
    use ndarray::ShapeBuilder;

    let mut owned = None;
    let src = f32_input(image, &mut owned)?;
    let shape = image.shape();
    let (id, ih, iw) = (shape[0], shape[1], shape[2]);
    let stride_z = ih * iw;
    let stride_y = iw;

    let (od, oh, ow) = (output_shape[0], output_shape[1], output_shape[2]);

    // Axis-aligned positive scaling (produced by the built-in resample
    // transforms) goes through the shared half-pixel trilinear kernel so fused
    // and unfused resamples agree. Translation and off-diagonal terms must be
    // zero; flips carry a translation and fall through to the general path.
    let is_pure_scale = matrix[0][1] == 0.0
        && matrix[0][2] == 0.0
        && matrix[0][3] == 0.0
        && matrix[1][0] == 0.0
        && matrix[1][2] == 0.0
        && matrix[1][3] == 0.0
        && matrix[2][0] == 0.0
        && matrix[2][1] == 0.0
        && matrix[2][3] == 0.0
        && matrix[0][0] > 0.0
        && matrix[1][1] > 0.0
        && matrix[2][2] > 0.0;

    let out = if is_pure_scale && matches!(interpolation, TransformsInterpolation::Trilinear) {
        trilinear_resample_forder_adaptive(src, [id, ih, iw], [od, oh, ow])
    } else {
        let mut out = vec![0.0f32; od * oh * ow];
        out.par_chunks_mut(oh * ow)
            .enumerate()
            .for_each(|(z, slab)| {
                let oz = z as f32;
                for y in 0..oh {
                    let oy = y as f32;
                    for x in 0..ow {
                        let ox = x as f32;
                        let sx = matrix[0][0] * ox
                            + matrix[0][1] * oy
                            + matrix[0][2] * oz
                            + matrix[0][3];
                        let sy = matrix[1][0] * ox
                            + matrix[1][1] * oy
                            + matrix[1][2] * oz
                            + matrix[1][3];
                        let sz = matrix[2][0] * ox
                            + matrix[2][1] * oy
                            + matrix[2][2] * oz
                            + matrix[2][3];

                        let dst = &mut slab[y * ow + x];

                        if sx < 0.0
                            || sy < 0.0
                            || sz < 0.0
                            || sx > (iw - 1) as f32
                            || sy > (ih - 1) as f32
                            || sz > (id - 1) as f32
                        {
                            *dst = 0.0;
                            continue;
                        }

                        match interpolation {
                            TransformsInterpolation::Nearest => {
                                let xi = (sx.round() as usize).min(iw - 1);
                                let yi = (sy.round() as usize).min(ih - 1);
                                let zi = (sz.round() as usize).min(id - 1);
                                *dst = src[zi * stride_z + yi * stride_y + xi];
                            }
                            TransformsInterpolation::Trilinear => {
                                let x0 = sx.floor() as usize;
                                let y0 = sy.floor() as usize;
                                let z0 = sz.floor() as usize;
                                let x1 = (x0 + 1).min(iw - 1);
                                let y1 = (y0 + 1).min(ih - 1);
                                let z1 = (z0 + 1).min(id - 1);

                                let fx = sx - x0 as f32;
                                let fy = sy - y0 as f32;
                                let fz = sz - z0 as f32;

                                let c000 = src[z0 * stride_z + y0 * stride_y + x0];
                                let c001 = src[z0 * stride_z + y0 * stride_y + x1];
                                let c010 = src[z0 * stride_z + y1 * stride_y + x0];
                                let c011 = src[z0 * stride_z + y1 * stride_y + x1];
                                let c100 = src[z1 * stride_z + y0 * stride_y + x0];
                                let c101 = src[z1 * stride_z + y0 * stride_y + x1];
                                let c110 = src[z1 * stride_z + y1 * stride_y + x0];
                                let c111 = src[z1 * stride_z + y1 * stride_y + x1];

                                let c00 = c000 * (1.0 - fx) + c001 * fx;
                                let c01 = c010 * (1.0 - fx) + c011 * fx;
                                let c10 = c100 * (1.0 - fx) + c101 * fx;
                                let c11 = c110 * (1.0 - fx) + c111 * fx;
                                let c0 = c00 * (1.0 - fy) + c01 * fy;
                                let c1 = c10 * (1.0 - fy) + c11 * fy;
                                *dst = c0 * (1.0 - fz) + c1 * fz;
                            }
                        }
                    }
                }
            });
        out
    };

    let out_array = ArrayD::from_shape_vec(IxDyn(&[od, oh, ow]).f(), out)
        .map_err(|e| Error::InvalidDimensions(format!("Shape mismatch: {}", e)))?;
    let mut header = image.header().clone();
    header.ndim = 3;
    header.dim = [1i64; 7];
    header.dim[0] = od as i64;
    header.dim[1] = oh as i64;
    header.dim[2] = ow as i64;
    header.datatype = DataType::Float32;
    header.scl_slope = 1.0;
    header.scl_inter = 0.0;
    Ok(NiftiImage::from_parts(header, ArrayData::F32(out_array)))
}

/// Trait for transforms that support lazy evaluation.
pub trait LazyTransform {
    /// Return the pending operation(s) for this transform, given the current
    /// (already materialized) image. Returns `None` when the transform cannot
    /// be lazily evaluated.
    fn to_pending_op(&self, image: &NiftiImage) -> Option<Vec<PendingOp>>;

    /// Whether this transform requires the actual image data.
    /// If false, the transform can be lazily composed.
    fn requires_data(&self) -> bool {
        false
    }
}
