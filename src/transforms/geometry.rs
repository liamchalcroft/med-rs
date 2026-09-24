//! Voxel-grid bookkeeping shared by the spatial transforms and the pipeline.
//!
//! Spatial transforms act on the first three axes; any further axes (time,
//! channels) are treated as a stack of volumes and carried through unchanged.
//! Images with fewer than three axes behave as if padded with length-1 axes.
//!
//! Every spatial transform is described by a [`GridChange`]: the voxel map
//! from new to old voxel indices plus the new spatial shape. Eager transforms
//! and the pipeline both derive their header updates from it, and the pipeline
//! composes consecutive changes into a single resampling pass.

use crate::error::{Error, Result};
use crate::nifti::header::{matmul, translation, Affine};

/// Split a shape into its spatial part (first three axes, padded with 1) and
/// the number of volumes (the product of any further axes).
pub(crate) fn split_shape(shape: &[usize]) -> ([usize; 3], usize) {
    let spatial = std::array::from_fn(|i| shape.get(i).copied().unwrap_or(1));
    let volumes = shape.iter().skip(3).product();
    (spatial, volumes)
}

/// Replace the spatial part of `shape`, keeping trailing axes. The rank grows
/// only if a padded spatial axis becomes longer than 1.
pub(crate) fn with_spatial_shape(shape: &[usize], spatial: [usize; 3]) -> Vec<usize> {
    let used = spatial.iter().rposition(|&d| d != 1).map_or(0, |i| i + 1);
    let rank = shape.len().max(used).max(1);
    let mut out = shape.to_vec();
    out.resize(rank, 1);
    for (dst, &d) in out.iter_mut().zip(&spatial) {
        *dst = d;
    }
    out
}

/// Number of spatial axes an image has (at most 3).
pub(crate) fn spatial_rank(shape: &[usize]) -> usize {
    shape.len().min(3)
}

/// Apply an affine to a voxel coordinate.
#[cfg(test)]
pub(crate) fn apply(a: &Affine, v: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|r| (0..3).map(|c| a[r][c] * v[c]).sum::<f64>() + a[r][3])
}

/// A spatial transform as a change of voxel grid: new voxel `v` takes the
/// value at old voxel `map · v`, and the new grid has `shape` spatial voxels.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct GridChange {
    pub map: Affine,
    pub shape: [usize; 3],
}

impl GridChange {
    /// Apply `self` and then `next`.
    pub fn then(&self, next: &Self) -> Self {
        Self {
            map: matmul(&self.map, &next.map),
            shape: next.shape,
        }
    }
}

/// Crop `shape` voxels at `offset`.
pub(crate) fn crop_plan(
    spatial: [usize; 3],
    offset: [usize; 3],
    shape: [usize; 3],
) -> Result<GridChange> {
    for axis in 0..3 {
        let end = offset[axis].checked_add(shape[axis]);
        if shape[axis] == 0 || end.is_none_or(|e| e > spatial[axis]) {
            return Err(Error::InvalidCropRegion(format!(
                "crop of {shape:?} at {offset:?} does not fit spatial shape {spatial:?}"
            )));
        }
    }
    Ok(GridChange {
        map: translation(offset.map(|o| o as f64)),
        shape,
    })
}

/// Centred crop or pad to `target`. Returns the change and whether any axis
/// is padded (so that new voxels need a fill value).
pub(crate) fn crop_or_pad_plan(
    spatial: [usize; 3],
    target: [usize; 3],
) -> Result<(GridChange, bool)> {
    if target.contains(&0) {
        return Err(Error::InvalidDimensions(format!(
            "target shape must be positive, got {target:?}"
        )));
    }
    let mut shift = [0.0; 3];
    let mut pads = false;
    for axis in 0..3 {
        let (n, t) = (spatial[axis], target[axis]);
        shift[axis] = if t <= n {
            ((n - t) / 2) as f64
        } else {
            pads = true;
            -(((t - n) / 2) as f64)
        };
    }
    Ok((
        GridChange {
            map: translation(shift),
            shape: target,
        },
        pads,
    ))
}

/// Voxel map that reverses axis `axis` of length `n`.
fn flip_axis_map(axis: usize, n: usize) -> Affine {
    let mut m = translation([0.0; 3]);
    m[axis][axis] = -1.0;
    m[axis][3] = (n - 1) as f64;
    m
}

/// Validate flip axes and return which spatial axes are flipped.
pub(crate) fn flip_axes(rank: usize, axes: &[usize]) -> Result<[bool; 3]> {
    let mut selected = [false; 3];
    for &axis in axes {
        if axis >= rank {
            return Err(Error::InvalidArgument(format!(
                "cannot flip axis {axis}: the image has {rank} spatial axes"
            )));
        }
        selected[axis] = true;
    }
    Ok(selected)
}

/// Reverse the selected axes.
pub(crate) fn flip_plan(spatial: [usize; 3], selected: [bool; 3]) -> GridChange {
    let mut map = translation([0.0; 3]);
    for (axis, &s) in selected.iter().enumerate() {
        if s {
            map = matmul(&map, &flip_axis_map(axis, spatial[axis]));
        }
    }
    GridChange {
        map,
        shape: spatial,
    }
}

/// Validate rotation axes and normalise `k` to `0..4`.
pub(crate) fn rotation_args(rank: usize, axes: (usize, usize), k: i32) -> Result<i32> {
    let (a, b) = axes;
    if a >= rank || b >= rank || a == b {
        return Err(Error::InvalidArgument(format!(
            "rotation axes {axes:?} must be two different spatial axes (< {rank})"
        )));
    }
    Ok(k.rem_euclid(4))
}

/// `numpy.rot90(a, k, axes)` as a grid change: one step maps new index `v` to
/// old index `u` with `u[a] = v[b]` and `u[b] = n_b - 1 - v[a]`.
pub(crate) fn rotate_plan(spatial: [usize; 3], axes: (usize, usize), k: i32) -> GridChange {
    let (a, b) = axes;
    let mut change = GridChange {
        map: translation([0.0; 3]),
        shape: spatial,
    };
    for _ in 0..k.rem_euclid(4) {
        let mut m = [[0.0; 4]; 4];
        m[3][3] = 1.0;
        for axis in (0..3).filter(|&x| x != a && x != b) {
            m[axis][axis] = 1.0;
        }
        m[a][b] = 1.0;
        m[b][a] = -1.0;
        m[b][3] = (change.shape[b] - 1) as f64;
        let mut shape = change.shape;
        shape.swap(a, b);
        change = change.then(&GridChange { map: m, shape });
    }
    change
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shapes() {
        assert_eq!(split_shape(&[4, 5]), ([4, 5, 1], 1));
        assert_eq!(split_shape(&[4, 5, 6, 2, 3]), ([4, 5, 6], 6));
        assert_eq!(with_spatial_shape(&[4, 5], [2, 3, 1]), vec![2, 3]);
        assert_eq!(with_spatial_shape(&[4, 5], [2, 3, 7]), vec![2, 3, 7]);
        assert_eq!(
            with_spatial_shape(&[4, 5, 6, 2], [1, 1, 1]),
            vec![1, 1, 1, 2]
        );
        assert_eq!(with_spatial_shape(&[4], [3, 1, 1]), vec![3]);
    }

    #[test]
    fn four_rotations_are_identity() {
        let r = rotate_plan([3, 5, 7], (0, 2), 4);
        assert_eq!(r.shape, [3, 5, 7]);
        assert_eq!(r.map, translation([0.0; 3]));
        let once = rotate_plan([3, 5, 7], (0, 2), 1);
        assert_eq!(once.shape, [7, 5, 3]);
    }

    #[test]
    fn crop_and_pad_plans() {
        assert!(crop_plan([4, 4, 4], [3, 0, 0], [2, 1, 1]).is_err());
        assert!(crop_plan([4, 4, 4], [usize::MAX, 0, 0], [2, 1, 1]).is_err());
        let (c, pads) = crop_or_pad_plan([4, 5, 6], [6, 3, 6]).unwrap();
        assert!(pads);
        assert_eq!(c.map[0][3], -1.0);
        assert_eq!(c.map[1][3], 1.0);
        assert_eq!(c.map[2][3], 0.0);
    }
}
