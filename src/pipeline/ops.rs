//! Fused operations for optimized execution.
//!
//! These operations combine multiple transforms into single passes
//! to minimize memory traffic and improve cache utilization.

use ndarray::{ArrayD, IxDyn};

/// A fused intensity operation that combines multiple linear transforms.
///
/// Represents: output = clamp(input * scale + offset, min, max)
#[derive(Clone, Copy, Debug)]
pub struct FusedIntensityOp {
    /// Multiplicative factor applied to input values.
    pub scale: f32,
    /// Additive offset applied after scaling.
    pub offset: f32,
    /// Optional lower clamp bound.
    pub clamp_min: Option<f32>,
    /// Optional upper clamp bound.
    pub clamp_max: Option<f32>,
}

impl FusedIntensityOp {
    /// Create an identity operation.
    pub fn identity() -> Self {
        Self {
            scale: 1.0,
            offset: 0.0,
            clamp_min: None,
            clamp_max: None,
        }
    }

    /// Create a z-normalization operation.
    pub fn z_normalize(mean: f32, std: f32) -> Self {
        let inv_std = if std <= 0.0 { 1.0 } else { 1.0 / std };
        Self {
            scale: inv_std,
            offset: -mean * inv_std,
            clamp_min: None,
            clamp_max: None,
        }
    }

    /// Create a rescale operation.
    pub fn rescale(in_min: f32, in_max: f32, out_min: f32, out_max: f32) -> Self {
        let in_range = if in_max - in_min == 0.0 {
            1.0
        } else {
            in_max - in_min
        };
        let scale = (out_max - out_min) / in_range;
        let offset = out_min - in_min * scale;
        Self {
            scale,
            offset,
            clamp_min: None,
            clamp_max: None,
        }
    }

    /// Chain another linear operation: (self)(x) then (other)
    pub fn then_linear(self, scale: f32, offset: f32) -> Self {
        // (self.scale * x + self.offset) * scale + offset
        // = self.scale * scale * x + self.offset * scale + offset
        Self {
            scale: self.scale * scale,
            offset: self.offset * scale + offset,
            clamp_min: self.clamp_min,
            clamp_max: self.clamp_max,
        }
    }

    /// Add clamping to this operation.
    pub fn with_clamp(mut self, min: f32, max: f32) -> Self {
        self.clamp_min = Some(min);
        self.clamp_max = Some(max);
        self
    }

    /// Apply this fused operation to an f32 array using SIMD + parallel.
    pub fn apply_f32(&self, input: &[f32], output: &mut [f32]) {
        use super::simd_kernels::{
            parallel_linear_transform_clamp_f32, parallel_linear_transform_f32,
        };

        let scale = self.scale;
        let offset = self.offset;

        match (self.clamp_min, self.clamp_max) {
            (Some(min), Some(max)) => {
                parallel_linear_transform_clamp_f32(input, output, scale, offset, min, max);
            }
            (Some(min), None) => {
                // Clamp with only min: use very large max
                parallel_linear_transform_clamp_f32(input, output, scale, offset, min, f32::MAX);
            }
            (None, Some(max)) => {
                // Clamp with only max: use very small min
                parallel_linear_transform_clamp_f32(input, output, scale, offset, f32::MIN, max);
            }
            (None, None) => {
                parallel_linear_transform_f32(input, output, scale, offset);
            }
        }
    }

    /// Apply this fused operation to an f32 array, allocating output.
    /// Uses memory pool for buffer reuse in pipelines.
    pub fn apply_f32_alloc(&self, input: &[f32]) -> Vec<f32> {
        use super::acquire_buffer;
        let mut output = acquire_buffer(input.len());
        self.apply_f32(input, &mut output);
        output
    }

    /// Apply to an ndarray, returning a new array.
    pub fn apply_array(&self, input: &ArrayD<f32>) -> ArrayD<f32> {
        let slice = input.as_slice_memory_order().unwrap();
        let output = self.apply_f32_alloc(slice);
        ArrayD::from_shape_vec(IxDyn(input.shape()), output).unwrap()
    }
}

/// An affine spatial operation (4x4 matrix).
#[derive(Clone, Copy, Debug)]
pub struct AffineOp {
    /// Homogeneous transform matrix applied to voxel coordinates.
    pub matrix: [[f32; 4]; 4],
}

impl AffineOp {
    /// Create an identity affine.
    pub fn identity() -> Self {
        Self {
            matrix: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }
    }

    /// Create a scaling affine.
    pub fn scale(sx: f32, sy: f32, sz: f32) -> Self {
        Self {
            matrix: [
                [sx, 0.0, 0.0, 0.0],
                [0.0, sy, 0.0, 0.0],
                [0.0, 0.0, sz, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }
    }

    /// Create a translation affine.
    pub fn translate(tx: f32, ty: f32, tz: f32) -> Self {
        Self {
            matrix: [
                [1.0, 0.0, 0.0, tx],
                [0.0, 1.0, 0.0, ty],
                [0.0, 0.0, 1.0, tz],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }
    }

    /// Create a flip affine (flip along specified axes).
    pub fn flip(flip_x: bool, flip_y: bool, flip_z: bool, shape: [usize; 3]) -> Self {
        let sx = if flip_x { -1.0 } else { 1.0 };
        let sy = if flip_y { -1.0 } else { 1.0 };
        let sz = if flip_z { -1.0 } else { 1.0 };

        // Flip needs translation to keep center fixed
        let tx = if flip_x { (shape[0] - 1) as f32 } else { 0.0 };
        let ty = if flip_y { (shape[1] - 1) as f32 } else { 0.0 };
        let tz = if flip_z { (shape[2] - 1) as f32 } else { 0.0 };

        Self {
            matrix: [
                [sx, 0.0, 0.0, tx],
                [0.0, sy, 0.0, ty],
                [0.0, 0.0, sz, tz],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }
    }

    /// Compose this affine with another: result = other * self
    pub fn then(self, other: &AffineOp) -> Self {
        let a = &self.matrix;
        let b = &other.matrix;
        let mut result = [[0.0f32; 4]; 4];

        for i in 0..4 {
            for j in 0..4 {
                for k in 0..4 {
                    result[i][j] += b[i][k] * a[k][j];
                }
            }
        }

        Self { matrix: result }
    }

    /// Apply this affine to a 3D point.
    #[inline]
    pub fn transform_point(&self, x: f32, y: f32, z: f32) -> (f32, f32, f32) {
        let m = &self.matrix;
        let nx = m[0][0] * x + m[0][1] * y + m[0][2] * z + m[0][3];
        let ny = m[1][0] * x + m[1][1] * y + m[1][2] * z + m[1][3];
        let nz = m[2][0] * x + m[2][1] * y + m[2][2] * z + m[2][3];
        (nx, ny, nz)
    }

    /// Invert this affine (for going from output to input coordinates).
    #[allow(clippy::many_single_char_names)]
    pub fn inverse(&self) -> Option<Self> {
        // For a 4x4 affine matrix, we can use the fact that the bottom row is [0,0,0,1]
        // The inverse of [R t; 0 1] is [R^-1  -R^-1*t; 0 1]
        let m = &self.matrix;

        // Extract 3x3 rotation/scale part
        let a = m[0][0];
        let b = m[0][1];
        let c = m[0][2];
        let d = m[1][0];
        let e = m[1][1];
        let f = m[1][2];
        let g = m[2][0];
        let h = m[2][1];
        let i = m[2][2];

        // Determinant of 3x3
        let det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);

        if det.abs() < 1e-10 {
            return None;
        }

        let inv_det = 1.0 / det;

        // Inverse of 3x3
        let r00 = (e * i - f * h) * inv_det;
        let r01 = (c * h - b * i) * inv_det;
        let r02 = (b * f - c * e) * inv_det;
        let r10 = (f * g - d * i) * inv_det;
        let r11 = (a * i - c * g) * inv_det;
        let r12 = (c * d - a * f) * inv_det;
        let r20 = (d * h - e * g) * inv_det;
        let r21 = (b * g - a * h) * inv_det;
        let r22 = (a * e - b * d) * inv_det;

        // -R^-1 * t
        let tx = m[0][3];
        let ty = m[1][3];
        let tz = m[2][3];
        let t0 = -(r00 * tx + r01 * ty + r02 * tz);
        let t1 = -(r10 * tx + r11 * ty + r12 * tz);
        let t2 = -(r20 * tx + r21 * ty + r22 * tz);

        Some(Self {
            matrix: [
                [r00, r01, r02, t0],
                [r10, r11, r12, t1],
                [r20, r21, r22, t2],
                [0.0, 0.0, 0.0, 1.0],
            ],
        })
    }
}
