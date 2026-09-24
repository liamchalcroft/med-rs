//! Anatomical orientation and reorientation.
//!
//! An orientation such as `RAS` names, for each voxel axis, the anatomical
//! direction in which that axis *increases*: `R`ight/`L`eft,
//! `A`nterior/`P`osterior, `S`uperior/`I`nferior.

use super::geometry::{split_shape, with_spatial_shape, GridChange};
use crate::error::{Error, Result};
use crate::nifti::element::{map_array, to_fortran};
use crate::nifti::header::{polar, Affine};
use crate::nifti::NiftiImage;
use ndarray::{Axis, Order};
use std::str::FromStr;

/// Direction in which a voxel axis increases.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AxisCode {
    /// Towards the subject's right.
    R,
    /// Towards the subject's left.
    L,
    /// Towards anterior.
    A,
    /// Towards posterior.
    P,
    /// Towards superior.
    S,
    /// Towards inferior.
    I,
}

impl AxisCode {
    /// World axis (0 = x/left-right, 1 = y/anterior-posterior, 2 = z/superior-inferior).
    pub const fn world_axis(self) -> usize {
        match self {
            Self::R | Self::L => 0,
            Self::A | Self::P => 1,
            Self::S | Self::I => 2,
        }
    }

    /// Whether the code points along the positive RAS world axis.
    pub const fn is_positive(self) -> bool {
        matches!(self, Self::R | Self::A | Self::S)
    }

    const fn from_axis(axis: usize, positive: bool) -> Self {
        match (axis, positive) {
            (0, true) => Self::R,
            (0, false) => Self::L,
            (1, true) => Self::A,
            (1, false) => Self::P,
            (_, true) => Self::S,
            (_, false) => Self::I,
        }
    }

    const fn as_char(self) -> char {
        match self {
            Self::R => 'R',
            Self::L => 'L',
            Self::A => 'A',
            Self::P => 'P',
            Self::S => 'S',
            Self::I => 'I',
        }
    }
}

/// Orientation of the three voxel axes, e.g. `RAS`. Always a valid
/// permutation: each world axis appears exactly once.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Orientation([AxisCode; 3]);

impl Orientation {
    /// Right-Anterior-Superior (nibabel / MNI convention).
    pub const RAS: Self = Self([AxisCode::R, AxisCode::A, AxisCode::S]);
    /// Left-Posterior-Superior (DICOM / ITK convention).
    pub const LPS: Self = Self([AxisCode::L, AxisCode::P, AxisCode::S]);
    /// Left-Anterior-Superior (radiological).
    pub const LAS: Self = Self([AxisCode::L, AxisCode::A, AxisCode::S]);

    /// Build an orientation, rejecting codes that repeat a world axis.
    pub fn new(codes: [AxisCode; 3]) -> Result<Self> {
        let mut seen = [false; 3];
        for code in codes {
            let axis = code.world_axis();
            if seen[axis] {
                return Err(Error::InvalidOrientation(format!(
                    "{} uses the same anatomical axis twice",
                    codes.iter().map(|c| c.as_char()).collect::<String>()
                )));
            }
            seen[axis] = true;
        }
        Ok(Self(codes))
    }

    /// The three axis codes.
    pub const fn codes(&self) -> [AxisCode; 3] {
        self.0
    }
}

impl FromStr for Orientation {
    type Err = Error;

    /// Parse a three-letter code such as `"RAS"` or `"lps"`.
    fn from_str(s: &str) -> Result<Self> {
        let bad = || {
            Error::InvalidOrientation(format!(
                "invalid orientation '{s}': expected three letters from R/L, A/P, S/I \
                 using each pair once (e.g. 'RAS', 'LPS')"
            ))
        };
        let chars: Vec<char> = s.chars().collect();
        if chars.len() != 3 {
            return Err(bad());
        }
        let mut codes = [AxisCode::R; 3];
        for (code, c) in codes.iter_mut().zip(chars) {
            *code = match c.to_ascii_uppercase() {
                'R' => AxisCode::R,
                'L' => AxisCode::L,
                'A' => AxisCode::A,
                'P' => AxisCode::P,
                'S' => AxisCode::S,
                'I' => AxisCode::I,
                _ => return Err(bad()),
            };
        }
        Self::new(codes).map_err(|_| bad())
    }
}

impl std::fmt::Display for Orientation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for c in self.0 {
            write!(f, "{}", c.as_char())?;
        }
        Ok(())
    }
}

/// Orientation of the voxel axes of an affine (nibabel's `io_orientation`).
///
/// The rotation part is orthogonalised first, and axes are assigned greedily,
/// so every non-degenerate affine yields a valid orientation even when it is
/// oblique. Degenerate axes (zero columns) keep their `RAS` default.
pub fn orientation_from_affine(affine: &Affine) -> Orientation {
    let mut rs = [[0.0f64; 3]; 3];
    for j in 0..3 {
        let norm = (0..3)
            .map(|i| affine[i][j] * affine[i][j])
            .sum::<f64>()
            .sqrt();
        let norm = if norm > 0.0 { norm } else { 1.0 };
        for i in 0..3 {
            rs[i][j] = affine[i][j] / norm;
        }
    }
    let mut r = polar(&rs);
    let mut codes = [AxisCode::R, AxisCode::A, AxisCode::S];
    let mut assigned = [false; 3];
    for (in_axis, code) in codes.iter_mut().enumerate() {
        let best = (0..3)
            .filter(|&row| !assigned[row])
            .max_by(|&a, &b| r[a][in_axis].abs().total_cmp(&r[b][in_axis].abs()));
        if let Some(row) = best.filter(|&row| r[row][in_axis].abs() > 1e-12) {
            *code = AxisCode::from_axis(row, r[row][in_axis] > 0.0);
            assigned[row] = true;
            r[row] = [0.0; 3];
        }
    }
    // Degenerate columns may leave a duplicate default; fill with unused axes.
    let mut used = [false; 3];
    for code in &codes {
        used[code.world_axis()] = true;
    }
    if used.iter().any(|u| !u) {
        let mut free = (0..3).filter(|&a| !used[a]);
        let mut seen = [false; 3];
        for code in &mut codes {
            let axis = code.world_axis();
            if seen[axis] {
                if let Some(a) = free.next() {
                    *code = AxisCode::from_axis(a, true);
                }
            }
            seen[code.world_axis()] = true;
        }
    }
    Orientation(codes)
}

/// How to reach `target` from `current`: for each new axis, the old axis it
/// reads and whether it is reversed.
pub(crate) struct Reorientation {
    pub perm: [usize; 3],
    pub flip: [bool; 3],
    pub change: GridChange,
}

pub(crate) fn reorient_plan(
    current: Orientation,
    target: Orientation,
    spatial: [usize; 3],
) -> Reorientation {
    let cur = current.codes();
    let mut perm = [0usize; 3];
    let mut flip = [false; 3];
    for (i, want) in target.codes().iter().enumerate() {
        // Both orientations are permutations, so the axis always exists.
        let j = (0..3)
            .find(|&j| cur[j].world_axis() == want.world_axis())
            .unwrap_or(i);
        perm[i] = j;
        flip[i] = cur[j] != *want;
    }
    // New index v -> old index u with u[perm[i]] = v[i] or n - 1 - v[i].
    let mut map = [[0.0; 4]; 4];
    map[3][3] = 1.0;
    for i in 0..3 {
        let j = perm[i];
        map[j][i] = if flip[i] { -1.0 } else { 1.0 };
        map[j][3] = if flip[i] {
            (spatial[j] - 1) as f64
        } else {
            0.0
        };
    }
    Reorientation {
        perm,
        flip,
        change: GridChange {
            map,
            shape: [spatial[perm[0]], spatial[perm[1]], spatial[perm[2]]],
        },
    }
}

/// Reorient an image so its voxel axes follow `target`.
///
/// This only permutes and flips axes, so it is exact: the datatype, scaling,
/// and values are unchanged and world coordinates are preserved (both the
/// sform and qform are updated). Axes beyond the third are kept.
pub fn reorient(image: &NiftiImage, target: Orientation) -> Result<NiftiImage> {
    let current = image.orientation();
    if current == target {
        return Ok(image.clone());
    }
    let (spatial, _) = split_shape(image.shape());
    let Reorientation { perm, flip, change } = reorient_plan(current, target, spatial);
    let ndim = image.ndim();
    let out_shape = with_spatial_shape(image.shape(), change.shape);
    let data = image.raw_data()?;
    let data = map_array!(data.as_ref(), |a| {
        // Work on at least three axes (1D/2D images are padded with 1s).
        let mut padded = a.shape().to_vec();
        padded.resize(ndim.max(3), 1);
        let view = a
            .view()
            .into_shape_with_order((padded, Order::ColumnMajor))
            .map_err(|e| Error::Internal(format!("cannot pad axes for reorient: {e}")))?;
        let mut axes: Vec<usize> = (0..ndim.max(3)).collect();
        axes[..3].copy_from_slice(&perm);
        let mut view = view.permuted_axes(axes);
        for (i, &f) in flip.iter().enumerate() {
            if f {
                view.invert_axis(Axis(i));
            }
        }
        to_fortran(&view)
            .into_shape_with_order((out_shape.clone(), Order::ColumnMajor))
            .map_err(|e| Error::Internal(format!("cannot restore axes after reorient: {e}")))?
    });
    let mut header = image.header().clone();
    header.transform_voxels(&change.map);
    Ok(NiftiImage::from_parts(header, data))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{ArrayD, IxDyn, ShapeBuilder};

    fn diag(d: [f64; 3]) -> Affine {
        [
            [d[0], 0.0, 0.0, 1.0],
            [0.0, d[1], 0.0, 2.0],
            [0.0, 0.0, d[2], 3.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    }

    #[test]
    fn parse_validates_axes() {
        assert_eq!("ras".parse::<Orientation>().unwrap(), Orientation::RAS);
        assert_eq!("LPS".parse::<Orientation>().unwrap().to_string(), "LPS");
        for bad in ["RRR", "RLA", "XYZ", "RA", "RASS"] {
            assert!(bad.parse::<Orientation>().is_err(), "{bad}");
        }
        // All 48 valid orientations parse.
        let mut n = 0;
        for a in "RLAPSI".chars() {
            for b in "RLAPSI".chars() {
                for c in "RLAPSI".chars() {
                    n += usize::from(format!("{a}{b}{c}").parse::<Orientation>().is_ok());
                }
            }
        }
        assert_eq!(n, 48);
    }

    #[test]
    fn orientation_of_oblique_and_permuted_affines() {
        assert_eq!(
            orientation_from_affine(&diag([1.0, 1.0, 1.0])),
            Orientation::RAS
        );
        assert_eq!(
            orientation_from_affine(&diag([-1.0, -1.0, 1.0])),
            Orientation::LPS
        );
        // 45 degrees exactly: greedy assignment still gives a permutation.
        let s = std::f64::consts::FRAC_1_SQRT_2;
        let tie = [
            [s, -s, 0.0, 0.0],
            [s, s, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let o = orientation_from_affine(&tie);
        assert!(Orientation::new(o.codes()).is_ok());
        // Axis permutation: voxel x runs along world z.
        let perm = [
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        assert_eq!(orientation_from_affine(&perm).to_string(), "SRP");
        // Degenerate affine does not panic and gives a valid permutation.
        let o = orientation_from_affine(&[[0.0; 4]; 4]);
        assert!(Orientation::new(o.codes()).is_ok());
    }

    #[test]
    fn reorient_is_exact_and_world_preserving() {
        let shape = [4usize, 5, 6, 2];
        let n: usize = shape.iter().product();
        let arr =
            ArrayD::from_shape_vec(IxDyn(&shape).f(), (0..n).map(|v| v as i16).collect()).unwrap();
        let img = NiftiImage::from_array(arr, diag([-1.0, 2.0, 3.0])).unwrap();
        assert_eq!(img.orientation().to_string(), "LAS");
        for target in ["RAS", "LPS", "SPR", "IAL"] {
            let t: Orientation = target.parse().unwrap();
            let r = reorient(&img, t).unwrap();
            assert_eq!(r.orientation(), t);
            assert_eq!(r.dtype(), img.dtype());
            assert_eq!(r.ndim(), 4);
            // Every voxel keeps its world position: compare via a round trip
            // back to the original orientation.
            let back = reorient(&r, img.orientation()).unwrap();
            assert_eq!(back.to_f32().unwrap(), img.to_f32().unwrap());
            let (a, b) = (back.affine(), img.affine());
            for i in 0..3 {
                for j in 0..4 {
                    assert!((a[i][j] - b[i][j]).abs() < 1e-9, "{target}: {a:?} vs {b:?}");
                }
            }
        }
    }

    #[test]
    fn reorient_value_at_world_point() {
        let arr = ArrayD::from_shape_vec(IxDyn(&[3, 1, 1]).f(), vec![10.0f32, 20.0, 30.0]).unwrap();
        let img = NiftiImage::from_array(arr, diag([-1.0, 1.0, 1.0])).unwrap();
        let r = reorient(&img, Orientation::RAS).unwrap();
        // x reversed; first RAS voxel is the old last voxel.
        let a = r.as_array::<f32>().unwrap();
        assert_eq!(a[[0, 0, 0]], 30.0);
        // Its world x equals the old last voxel's world x: 1 - 2 = -1.
        assert!((r.affine()[0][3] - -1.0).abs() < 1e-12);
    }
}
