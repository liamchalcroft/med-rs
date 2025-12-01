//! Lazy evaluation infrastructure for transform pipelines.
//!
//! Lazy transforms accumulate operations without executing them immediately.
//! When the data is finally needed, all pending operations are composed and
//! executed in a single optimized pass.

use crate::nifti::image::ArrayData;
use crate::nifti::DataType;
use crate::nifti::NiftiImage;
use crate::pipeline::acquire_buffer;
use crate::pipeline::simd_kernels::{
    parallel_linear_transform_f32, parallel_minmax_f32, parallel_sum_and_sum_sq_f32,
};
use crate::transforms::Interpolation as TransformsInterpolation;
use ndarray::{ArrayD, IxDyn};

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
    /// Intensity normalization: output = (input - mean) * inv_std
    ZNormalize {
        /// Mean value used for centering.
        mean: f32,
        /// Inverse standard deviation for scaling.
        inv_std: f32,
    },
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
        use PendingOp::*;
        match (self, other) {
            // Affine transforms can be composed
            (
                Affine { .. },
                Affine {
                    interpolation: i2, ..
                },
            ) => {
                if let Affine {
                    interpolation: i1, ..
                } = self
                {
                    i1 == i2 // Same interpolation mode
                } else {
                    false
                }
            }
            // Linear intensity ops can be composed: (ax+b)*c+d = acx + (bc+d)
            (LinearIntensity { .. }, LinearIntensity { .. }) => true,
            // Z-normalize followed by linear intensity
            (ZNormalize { .. }, LinearIntensity { .. }) => true,
            // Z-normalize followed by clamp
            (ZNormalize { .. }, Clamp { .. }) => true,
            // Linear intensity then clamp
            (LinearIntensity { .. }, Clamp { .. }) => true,
            _ => false,
        }
    }

    /// Fuse this operation with another, returning the combined operation.
    pub fn fuse_with(&self, other: &PendingOp) -> Option<PendingOp> {
        use PendingOp::*;
        match (self, other) {
            (
                Affine {
                    matrix: m1,
                    interpolation,
                    ..
                },
                Affine {
                    matrix: m2,
                    output_shape,
                    ..
                },
            ) => {
                // Compose affine matrices: result = m2 * m1
                let composed = compose_affine(m1, m2);
                Some(Affine {
                    matrix: composed,
                    output_shape: *output_shape,
                    interpolation: *interpolation,
                })
            }
            (
                LinearIntensity {
                    scale: s1,
                    offset: o1,
                },
                LinearIntensity {
                    scale: s2,
                    offset: o2,
                },
            ) => {
                // (s1*x + o1) * s2 + o2 = s1*s2*x + o1*s2 + o2
                Some(LinearIntensity {
                    scale: s1 * s2,
                    offset: o1 * s2 + o2,
                })
            }
            (ZNormalize { mean, inv_std }, LinearIntensity { scale, offset }) => {
                // ((x - mean) * inv_std) * scale + offset
                // = x * (inv_std * scale) + (-mean * inv_std * scale + offset)
                Some(LinearIntensity {
                    scale: inv_std * scale,
                    offset: -mean * inv_std * scale + offset,
                })
            }
            (ZNormalize { mean, inv_std }, Clamp { min, max }) => {
                // Turn into LinearIntensity + clamp
                Some(Clamp {
                    min: *min,
                    max: *max,
                })
                .and_then(|cl| {
                    Some(LinearIntensity {
                        scale: *inv_std,
                        offset: -mean * inv_std,
                    })
                    .and_then(|lin| lin.fuse_with(&cl))
                })
            }
            (LinearIntensity { .. }, Clamp { min, max }) => Some(Clamp {
                min: *min,
                max: *max,
            }),
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
        // Try to fuse with the last operation
        if let Some(last) = self.pending.last() {
            if last.can_fuse_with(&op) {
                if let Some(fused) = last.fuse_with(&op) {
                    self.pending.pop();
                    self.pending.push(fused);
                    return;
                }
            }
        }
        self.pending.push(op);
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
    pub fn materialize(self) -> crate::error::Result<NiftiImage> {
        let mut image = if let Some(img) = self.image {
            img
        } else if let Some(path) = &self.path {
            crate::nifti::load(path)?
        } else {
            return Err(crate::error::Error::InvalidDimensions(
                "LazyImage has no image or path".into(),
            ));
        };

        // Group and execute pending operations
        for op in &self.pending {
            image = execute_op(image, op)?;
        }

        Ok(image)
    }

    /// Get a reference to the pending operations.
    pub fn pending_ops(&self) -> &[PendingOp] {
        &self.pending
    }
}

/// Execute a single pending operation on an image.
fn execute_op(image: NiftiImage, op: &PendingOp) -> crate::error::Result<NiftiImage> {
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
            Ok(apply_affine(&image, matrix, shape, interp))
        }
        PendingOp::ZNormalize { .. } => {
            // Fallback: eager
            Ok(transforms::z_normalization(&image))
        }
        PendingOp::LinearIntensity { scale, offset } => {
            // Apply linear transform: output = input * scale + offset
            Ok(apply_linear_intensity(&image, *scale, *offset))
        }
        PendingOp::Clamp { min, max } => Ok(transforms::clamp(&image, *min as f64, *max as f64)),
        PendingOp::Flip { axes } => {
            let axes_vec: Vec<usize> = (0..3).filter(|&i| (axes >> i) & 1 == 1).collect();
            transforms::flip(&image, &axes_vec)
        }
    }
}

/// Execute a fused chain of intensity ops if possible.
pub fn execute_fused_intensity(image: &NiftiImage, pending: &[PendingOp]) -> Option<NiftiImage> {
    let mut do_znorm = false;
    let rescale: Option<(f32, f32)> = None;
    let mut clamp: Option<(f32, f32)> = None;

    for op in pending {
        match op {
            PendingOp::ZNormalize { .. } => do_znorm = true,
            PendingOp::LinearIntensity {
                scale: _,
                offset: _,
            } => {
                // We can't distinguish arbitrary linear intents here; skip fusion.
                return None;
            }
            PendingOp::Clamp { min, max } => clamp = Some((*min, *max)),
            _ => return None,
        }
    }

    Some(fuse_intensity_ops(image, do_znorm, rescale, clamp))
}

/// Apply a linear intensity transformation: output = input * scale + offset
/// Uses SIMD-accelerated parallel processing with memory pool for buffer reuse.
fn apply_linear_intensity(image: &NiftiImage, scale: f32, offset: f32) -> NiftiImage {
    use super::acquire_buffer;
    use super::simd_kernels::parallel_linear_transform_f32;
    use crate::nifti::image::ArrayData;
    use crate::nifti::DataType;
    use ndarray::{ArrayD, IxDyn};

    let header = image.header().clone();

    // Fast path for f32 with SIMD
    if let ArrayData::F32(a) = image.owned_data() {
        let slice = a.as_slice_memory_order().unwrap();
        let mut output = acquire_buffer(slice.len());

        parallel_linear_transform_f32(slice, &mut output, scale, offset);

        let out_array = ArrayD::from_shape_vec(IxDyn(a.shape()), output).unwrap();
        return NiftiImage::from_parts(header, ArrayData::F32(out_array));
    }

    // Generic path: convert to f32
    let data = image.to_f32();
    let slice = data.as_slice_memory_order().unwrap();
    let mut output = acquire_buffer(slice.len());

    parallel_linear_transform_f32(slice, &mut output, scale, offset);

    let out_array = ArrayD::from_shape_vec(IxDyn(data.shape()), output).unwrap();
    let mut new_header = header;
    new_header.datatype = DataType::Float32;
    new_header.scl_slope = 1.0;
    new_header.scl_inter = 0.0;
    NiftiImage::from_parts(new_header, ArrayData::F32(out_array))
}

/// Fuse z-normalize, rescale, and clamp into one pass when possible.
#[allow(clippy::similar_names)]
pub fn fuse_intensity_ops(
    image: &NiftiImage,
    do_znorm: bool,
    rescale: Option<(f32, f32)>,
    clamp: Option<(f32, f32)>,
) -> NiftiImage {
    // Work in f32
    let data = image.to_f32();
    let slice = data.as_slice_memory_order().unwrap();

    let mut scale = 1.0f32;
    let mut offset = 0.0f32;

    // z-norm stats
    if do_znorm {
        let (sum, sum_sq, count) = parallel_sum_and_sum_sq_f32(slice);
        let mean = (sum / count as f64) as f32;
        let variance = (sum_sq / count as f64) - (mean as f64 * mean as f64);
        let inv_std = if variance <= 0.0 {
            1.0
        } else {
            1.0 / (variance.sqrt() as f32)
        };
        scale *= inv_std;
        offset += -mean * inv_std;
    }

    // rescale to range
    let mut clamp_min = None;
    let mut clamp_max = None;

    if let Some((out_min, out_max)) = rescale {
        let (min, max) = parallel_minmax_f32(slice);
        let range = if max - min == 0.0 { 1.0 } else { max - min };
        let r_scale = (out_max - out_min) / range;
        let r_offset = out_min - min * r_scale;
        scale *= r_scale;
        offset = offset * r_scale + r_offset;
    }

    if let Some((min, max)) = clamp {
        clamp_min = Some(min);
        clamp_max = Some(max);
    }

    // Apply fused op
    let mut output = acquire_buffer(slice.len());

    match (clamp_min, clamp_max) {
        (Some(min), Some(max)) => {
            super::simd_kernels::parallel_linear_transform_clamp_f32(
                slice,
                &mut output,
                scale,
                offset,
                min,
                max,
            );
        }
        (Some(min), None) => {
            super::simd_kernels::parallel_linear_transform_clamp_f32(
                slice,
                &mut output,
                scale,
                offset,
                min,
                f32::MAX,
            );
        }
        (None, Some(max)) => {
            super::simd_kernels::parallel_linear_transform_clamp_f32(
                slice,
                &mut output,
                scale,
                offset,
                f32::MIN,
                max,
            );
        }
        (None, None) => {
            parallel_linear_transform_f32(slice, &mut output, scale, offset);
        }
    }

    let out_array = ArrayD::from_shape_vec(IxDyn(data.shape()), output).unwrap();
    let mut header = image.header().clone();
    header.datatype = DataType::Float32;
    header.scl_slope = 1.0;
    header.scl_inter = 0.0;
    NiftiImage::from_parts(header, ArrayData::F32(out_array))
}

#[allow(clippy::similar_names)]
fn apply_affine(
    image: &NiftiImage,
    matrix: &[[f32; 4]; 4],
    output_shape: [usize; 3],
    interpolation: TransformsInterpolation,
) -> NiftiImage {
    let data = image.to_f32();
    let shape = data.shape();
    let (id, ih, iw) = (shape[0], shape[1], shape[2]);
    let src = data.as_slice_memory_order().unwrap();
    let stride_z = ih * iw;
    let stride_y = iw;

    let (od, oh, ow) = (output_shape[0], output_shape[1], output_shape[2]);
    let mut out = vec![0.0f32; od * oh * ow];

    for z in 0..od {
        for y in 0..oh {
            for x in 0..ow {
                let ox = x as f32;
                let oy = y as f32;
                let oz = z as f32;
                let sx = matrix[0][0] * ox + matrix[0][1] * oy + matrix[0][2] * oz + matrix[0][3];
                let sy = matrix[1][0] * ox + matrix[1][1] * oy + matrix[1][2] * oz + matrix[1][3];
                let sz = matrix[2][0] * ox + matrix[2][1] * oy + matrix[2][2] * oz + matrix[2][3];

                let idx = z * oh * ow + y * ow + x;

                if sx < 0.0
                    || sy < 0.0
                    || sz < 0.0
                    || sx >= (iw - 1) as f32
                    || sy >= (ih - 1) as f32
                    || sz >= (id - 1) as f32
                {
                    out[idx] = 0.0;
                    continue;
                }

                match interpolation {
                    TransformsInterpolation::Nearest => {
                        let xi = sx.round() as usize;
                        let yi = sy.round() as usize;
                        let zi = sz.round() as usize;
                        out[idx] = src[zi * stride_z + yi * stride_y + xi];
                    }
                    TransformsInterpolation::Trilinear => {
                        let x0 = sx.floor() as usize;
                        let y0 = sy.floor() as usize;
                        let z0 = sz.floor() as usize;
                        let x1 = x0 + 1;
                        let y1 = y0 + 1;
                        let z1 = z0 + 1;

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
                        out[idx] = c0 * (1.0 - fz) + c1 * fz;
                    }
                }
            }
        }
    }

    let out_array = ArrayD::from_shape_vec(IxDyn(&[od, oh, ow]), out).unwrap();
    let mut header = image.header().clone();
    header.ndim = 3;
    header.dim = [1u16; 7];
    header.dim[0] = od as u16;
    header.dim[1] = oh as u16;
    header.dim[2] = ow as u16;
    header.datatype = DataType::Float32;
    header.scl_slope = 1.0;
    header.scl_inter = 0.0;
    // Keep spacing/affine from the source for now; caller may overwrite
    NiftiImage::from_parts(header, ArrayData::F32(out_array))
}
/// Trait for transforms that support lazy evaluation.
pub trait LazyTransform {
    /// Return the pending operation(s) for this transform.
    /// If the transform cannot be lazily evaluated, returns None.
    fn to_pending_op(&self, image: &LazyImage) -> Option<Vec<PendingOp>>;

    /// Whether this transform requires the actual image data.
    /// If false, the transform can be lazily composed.
    fn requires_data(&self) -> bool {
        false
    }
}
