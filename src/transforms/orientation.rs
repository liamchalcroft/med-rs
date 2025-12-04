//! Image orientation and reorientation.
//!
//! Medical images can be stored in various orientations. This module provides
//! tools to detect orientation and reorient to standard coordinate systems.

use crate::nifti::image::ArrayData;
use crate::nifti::{DataType, NiftiImage};
use ndarray::{ArrayD, IxDyn, ShapeBuilder};

/// Standard anatomical orientation codes.
///
/// Each axis is labeled by which direction increases along that axis:
/// - R/L: Right/Left
/// - A/P: Anterior/Posterior
/// - S/I: Superior/Inferior
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AxisCode {
    /// Increasing axis moves to the subject's right.
    R,
    /// Increasing axis moves to the subject's left.
    L, // Right, Left
    /// Increasing axis moves toward the anterior direction.
    A,
    /// Increasing axis moves toward the posterior direction.
    P, // Anterior, Posterior
    /// Increasing axis moves toward the superior direction.
    S,
    /// Increasing axis moves toward the inferior direction.
    I, // Superior, Inferior
}

impl AxisCode {
    #[allow(dead_code)]
    fn opposite(self) -> Self {
        match self {
            Self::R => Self::L,
            Self::L => Self::R,
            Self::A => Self::P,
            Self::P => Self::A,
            Self::S => Self::I,
            Self::I => Self::S,
        }
    }

    fn axis_index(self) -> usize {
        match self {
            Self::R | Self::L => 0,
            Self::A | Self::P => 1,
            Self::S | Self::I => 2,
        }
    }

    #[allow(dead_code)]
    fn is_positive(self) -> bool {
        matches!(self, Self::R | Self::A | Self::S)
    }
}

/// Orientation as a tuple of three axis codes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Orientation(pub AxisCode, pub AxisCode, pub AxisCode);

impl Orientation {
    /// RAS (Right-Anterior-Superior) - standard neuroimaging orientation.
    pub const RAS: Self = Self(AxisCode::R, AxisCode::A, AxisCode::S);
    /// LAS (Left-Anterior-Superior) - radiological convention.
    pub const LAS: Self = Self(AxisCode::L, AxisCode::A, AxisCode::S);
    /// LPS (Left-Posterior-Superior) - DICOM convention.
    pub const LPS: Self = Self(AxisCode::L, AxisCode::P, AxisCode::S);

    /// Parse from 3-character string like "RAS", "LPS", etc.
    pub fn from_str(s: &str) -> Option<Self> {
        let chars: Vec<char> = s.chars().collect();
        if chars.len() != 3 {
            return None;
        }

        let parse_code = |c: char| -> Option<AxisCode> {
            match c.to_ascii_uppercase() {
                'R' => Some(AxisCode::R),
                'L' => Some(AxisCode::L),
                'A' => Some(AxisCode::A),
                'P' => Some(AxisCode::P),
                'S' => Some(AxisCode::S),
                'I' => Some(AxisCode::I),
                _ => None,
            }
        };

        Some(Self(
            parse_code(chars[0])?,
            parse_code(chars[1])?,
            parse_code(chars[2])?,
        ))
    }

    /// Get codes as array.
    pub fn codes(&self) -> [AxisCode; 3] {
        [self.0, self.1, self.2]
    }
}

impl std::fmt::Display for Orientation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let code_char = |c: AxisCode| -> char {
            match c {
                AxisCode::R => 'R',
                AxisCode::L => 'L',
                AxisCode::A => 'A',
                AxisCode::P => 'P',
                AxisCode::S => 'S',
                AxisCode::I => 'I',
            }
        };
        write!(
            f,
            "{}{}{}",
            code_char(self.0),
            code_char(self.1),
            code_char(self.2)
        )
    }
}

/// Detect the orientation of an image from its affine matrix.
pub fn orientation_from_affine(affine: &[[f32; 4]; 4]) -> Orientation {
    // Extract rotation/scaling part
    let mut codes = [AxisCode::R, AxisCode::A, AxisCode::S];

    for i in 0..3 {
        // Find which anatomical axis this image axis corresponds to
        let col = [affine[0][i], affine[1][i], affine[2][i]];
        let abs_col = [col[0].abs(), col[1].abs(), col[2].abs()];

        // Find dominant axis
        let max_idx = if abs_col[0] >= abs_col[1] && abs_col[0] >= abs_col[2] {
            0
        } else if abs_col[1] >= abs_col[2] {
            1
        } else {
            2
        };

        let positive = col[max_idx] > 0.0;

        codes[i] = match max_idx {
            0 => {
                if positive {
                    AxisCode::R
                } else {
                    AxisCode::L
                }
            }
            1 => {
                if positive {
                    AxisCode::A
                } else {
                    AxisCode::P
                }
            }
            2 => {
                if positive {
                    AxisCode::S
                } else {
                    AxisCode::I
                }
            }
            _ => unreachable!(),
        };
    }

    Orientation(codes[0], codes[1], codes[2])
}

/// Reorient an image to target orientation.
///
/// # Example
/// ```ignore
/// use medrs::transforms::{Orientation, reorient};
///
/// // Reorient to standard RAS orientation
/// let ras_img = reorient(&img, Orientation::RAS);
/// ```
pub fn reorient(image: &NiftiImage, target: Orientation) -> NiftiImage {
    let current = orientation_from_affine(&image.affine());

    if current == target {
        return image.clone();
    }

    let data = image.to_f32();
    let affine = image.affine();
    let shape = image.shape();

    // Compute axis permutation and flips
    let current_codes = current.codes();
    let target_codes = target.codes();

    let mut perm = [0usize; 3];
    let mut flip = [false; 3];

    for (i, &target_code) in target_codes.iter().enumerate() {
        // Find which current axis maps to this target axis
        for (j, &current_code) in current_codes.iter().enumerate() {
            if current_code.axis_index() == target_code.axis_index() {
                perm[i] = j;
                flip[i] = current_code != target_code;
                break;
            }
        }
    }

    // Apply permutation and flips
    let new_shape: Vec<usize> = perm.iter().map(|&p| shape[p]).collect();
    let mut new_data = permute_axes(&data, &perm, &new_shape);

    // Apply flips
    for (i, &should_flip) in flip.iter().enumerate() {
        if should_flip {
            new_data = flip_axis(&new_data, i);
        }
    }

    // Update affine matrix
    let mut new_affine = [[0.0f32; 4]; 4];
    new_affine[3][3] = 1.0;

    for i in 0..3 {
        let src_axis = perm[i];
        let flip_sign = if flip[i] { -1.0 } else { 1.0 };

        for j in 0..3 {
            new_affine[j][i] = affine[j][src_axis] * flip_sign;
        }

        // Adjust origin if flipping
        if flip[i] {
            let extent = (new_shape[i] - 1) as f32;
            for j in 0..3 {
                new_affine[j][3] += new_affine[j][i] * extent;
            }
        }
    }

    // Copy origin
    for j in 0..3 {
        new_affine[j][3] += affine[j][3];
    }

    // Build output header while preserving metadata
    let mut header = image.header().clone();
    header.ndim = new_shape.len() as u8;
    header.datatype = DataType::Float32;
    header.scl_slope = 1.0;
    header.scl_inter = 0.0;
    header.dim = [1u16; 7];
    for (i, &d) in new_shape.iter().enumerate() {
        header.dim[i] = d as u16;
    }
    header.pixdim = [1.0f32; 7];
    for i in 0..3 {
        header.pixdim[i] = image.spacing()[perm[i]];
    }
    header.set_affine(new_affine);

    NiftiImage::from_parts(header, ArrayData::F32(new_data))
}

fn permute_axes(data: &ArrayD<f32>, perm: &[usize; 3], new_shape: &[usize]) -> ArrayD<f32> {
    let old_shape = data.shape();

    // Convert F-order to C-order for processing if needed
    let data_c: std::borrow::Cow<'_, ArrayD<f32>> = if data.is_standard_layout() {
        std::borrow::Cow::Borrowed(data)
    } else {
        let mut c_order = ArrayD::zeros(IxDyn(old_shape));
        c_order.assign(data);
        std::borrow::Cow::Owned(c_order)
    };

    let src = data_c.as_slice().expect("C-order array should have contiguous slice");
    let mut output = vec![0.0f32; new_shape.iter().product()];
    let (nd, nh, nw) = (new_shape[0], new_shape[1], new_shape[2]);
    let old_strides = [old_shape[1] * old_shape[2], old_shape[2], 1];

    for d in 0..nd {
        for h in 0..nh {
            for w in 0..nw {
                let new_coords = [d, h, w];
                let old_coords = [
                    new_coords[perm.iter().position(|&p| p == 0).unwrap()],
                    new_coords[perm.iter().position(|&p| p == 1).unwrap()],
                    new_coords[perm.iter().position(|&p| p == 2).unwrap()],
                ];

                let old_idx = old_coords[0] * old_strides[0]
                    + old_coords[1] * old_strides[1]
                    + old_coords[2] * old_strides[2];
                let new_idx = d * nh * nw + h * nw + w;

                output[new_idx] = src[old_idx];
            }
        }
    }

    // Output is in C-order. Convert to F-order to match NIfTI convention.
    let c_order = ArrayD::from_shape_vec(IxDyn(new_shape), output).unwrap();
    let mut f_order = ArrayD::zeros(IxDyn(new_shape).f());
    f_order.assign(&c_order);
    f_order
}

fn flip_axis(data: &ArrayD<f32>, axis: usize) -> ArrayD<f32> {
    let shape = data.shape();

    // Convert F-order to C-order for processing if needed
    let data_c: std::borrow::Cow<'_, ArrayD<f32>> = if data.is_standard_layout() {
        std::borrow::Cow::Borrowed(data)
    } else {
        let mut c_order = ArrayD::zeros(IxDyn(shape));
        c_order.assign(data);
        std::borrow::Cow::Owned(c_order)
    };

    let src = data_c.as_slice().expect("C-order array should have contiguous slice");
    let mut output = vec![0.0f32; src.len()];
    let (d, h, w) = (shape[0], shape[1], shape[2]);

    for di in 0..d {
        for hi in 0..h {
            for wi in 0..w {
                let (new_d, new_h, new_w) = match axis {
                    0 => (d - 1 - di, hi, wi),
                    1 => (di, h - 1 - hi, wi),
                    2 => (di, hi, w - 1 - wi),
                    _ => unreachable!(),
                };

                let old_idx = di * h * w + hi * w + wi;
                let new_idx = new_d * h * w + new_h * w + new_w;
                output[new_idx] = src[old_idx];
            }
        }
    }

    // Output is in C-order. Convert to F-order to match NIfTI convention.
    let c_order = ArrayD::from_shape_vec(IxDyn(shape), output).unwrap();
    let mut f_order = ArrayD::zeros(IxDyn(shape).f());
    f_order.assign(&c_order);
    f_order
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_orientation_parse() {
        assert_eq!(Orientation::from_str("RAS"), Some(Orientation::RAS));
        assert_eq!(Orientation::from_str("LPS"), Some(Orientation::LPS));
        assert_eq!(Orientation::from_str("XYZ"), None);
    }

    #[test]
    fn test_orientation_display() {
        assert_eq!(format!("{}", Orientation::RAS), "RAS");
        assert_eq!(format!("{}", Orientation::LPS), "LPS");
    }
}
