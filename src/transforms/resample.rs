//! Resampling onto a new voxel grid.
//!
//! All resampling goes through [`resample_to_grid`]: given the affine and
//! shape of a target grid, every target voxel centre is mapped into the
//! source image through the two affines and interpolated there, so the output
//! occupies the same world space as the input. The convenience functions
//! ([`resample_to_spacing`], [`resample_to_shape`], [`resample_like`]) only
//! differ in how they choose the target grid.
//!
//! * Trilinear interpolation produces `f32` values (with scaling applied).
//! * Nearest-neighbour interpolation copies stored values, so it keeps the
//!   datatype and the scaling of the input: resampling a `u8` label map gives
//!   a `u8` label map.
//! * Axis-aligned grids (any combination of scaling, flips, and axis
//!   permutations) use a separable kernel with precomputed per-axis indices;
//!   oblique grids use a general kernel.
//! * Target voxels whose centre falls outside the source image get `0`.
//! * Images with more than three axes are resampled volume by volume.

use super::geometry::{split_shape, with_spatial_shape, GridChange};
use crate::error::{Error, Result};
use crate::nifti::element::{dispatch_dtype, fortran_from_vec, ArrayData};
use crate::nifti::header::{invert_affine, matmul, Affine};
use crate::nifti::NiftiImage;
use rayon::prelude::*;

/// Interpolation method.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum Interpolation {
    /// Nearest neighbour: keeps stored values and the datatype (use for labels).
    Nearest,
    /// Trilinear: smooth, produces `f32`.
    #[default]
    Trilinear,
}

impl std::fmt::Display for Interpolation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Nearest => "nearest",
            Self::Trilinear => "trilinear",
        })
    }
}

impl std::str::FromStr for Interpolation {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self> {
        match s {
            "nearest" => Ok(Self::Nearest),
            "trilinear" => Ok(Self::Trilinear),
            _ => Err(Error::InvalidArgument(format!(
                "unknown interpolation '{s}' (expected 'nearest' or 'trilinear')"
            ))),
        }
    }
}

/// Largest number of voxels a resampling target may have (2^34, 64 GiB of f32).
const MAX_TARGET_VOXELS: usize = 1 << 34;

fn check_target_shape(shape: [usize; 3], volumes: usize) -> Result<()> {
    if shape.contains(&0) {
        return Err(Error::InvalidDimensions(format!(
            "target shape must be positive, got {shape:?}"
        )));
    }
    let total = shape
        .iter()
        .try_fold(volumes, |acc, &d| acc.checked_mul(d))
        .filter(|&n| n <= MAX_TARGET_VOXELS);
    if total.is_none() {
        return Err(Error::InvalidDimensions(format!(
            "target shape {shape:?} x {volumes} volume(s) is too large"
        )));
    }
    Ok(())
}

/// Resample onto the grid of `reference`: same shape (first three axes) and
/// affine. Use this to align images of different field of view, resolution,
/// or orientation (for example a segmentation onto its MRI).
pub fn resample_like(
    image: &NiftiImage,
    reference: &NiftiImage,
    interp: Interpolation,
) -> Result<NiftiImage> {
    let (spatial, _) = split_shape(reference.shape());
    resample_to_grid(image, &reference.affine(), spatial, interp)
}

/// Resample to new voxel sizes, keeping the field of view.
///
/// The output shape is `round(shape * spacing / target_spacing)` (at least 1).
/// When that rounding changes the extent, the voxel size actually used is
/// recorded in the affine, so world coordinates stay exact.
pub fn resample_to_spacing(
    image: &NiftiImage,
    target_spacing: [f64; 3],
    interp: Interpolation,
) -> Result<NiftiImage> {
    let (spatial, volumes) = split_shape(image.shape());
    let change = spacing_plan(spatial, &image.header().spacing(), target_spacing)?;
    check_target_shape(change.shape, volumes)?;
    apply_grid_change(image, &change, interp)
}

/// Resample to a new grid shape, keeping the field of view (voxel centres
/// are placed with the half-voxel convention used by ITK, SimpleITK, and
/// MONAI).
pub fn resample_to_shape(
    image: &NiftiImage,
    target_shape: [usize; 3],
    interp: Interpolation,
) -> Result<NiftiImage> {
    let (spatial, volumes) = split_shape(image.shape());
    check_target_shape(target_shape, volumes)?;
    apply_grid_change(image, &shape_plan(spatial, target_shape), interp)
}

/// Grid change for [`resample_to_shape`]: new voxel `i` maps to old voxel
/// `(i + 0.5) * ratio - 0.5`.
pub(crate) fn shape_plan(spatial: [usize; 3], target: [usize; 3]) -> GridChange {
    let mut map = identity();
    for axis in 0..3 {
        let ratio = spatial[axis] as f64 / target[axis] as f64;
        map[axis][axis] = ratio;
        map[axis][3] = 0.5 * (ratio - 1.0);
    }
    GridChange { map, shape: target }
}

/// Grid change for [`resample_to_spacing`] given the current voxel sizes.
pub(crate) fn spacing_plan(
    spatial: [usize; 3],
    spacing: &[f64],
    target_spacing: [f64; 3],
) -> Result<GridChange> {
    let mut target = [1usize; 3];
    for axis in 0..3 {
        let want = target_spacing[axis];
        if !(want.is_finite() && want > 0.0) {
            return Err(Error::InvalidArgument(format!(
                "target spacing must be finite and positive, got {want} for axis {axis}"
            )));
        }
        let current = spacing.get(axis).copied().unwrap_or(1.0);
        if !(current.is_finite() && current > 0.0) {
            return Err(Error::InvalidAffine(format!(
                "image spacing along axis {axis} is {current}; cannot resample by spacing"
            )));
        }
        let n = (spatial[axis] as f64 * current / want).round();
        if !(n.is_finite() && n <= MAX_TARGET_VOXELS as f64) {
            return Err(Error::InvalidDimensions(format!(
                "target spacing {target_spacing:?} gives an unrepresentable shape"
            )));
        }
        target[axis] = (n as usize).max(1);
    }
    Ok(shape_plan(spatial, target))
}

/// Resample onto an arbitrary grid given by its affine and shape.
///
/// Target voxels whose centres fall outside the source image are set to 0.
pub fn resample_to_grid(
    image: &NiftiImage,
    target_affine: &Affine,
    target_shape: [usize; 3],
    interp: Interpolation,
) -> Result<NiftiImage> {
    let (_, volumes) = split_shape(image.shape());
    check_target_shape(target_shape, volumes)?;
    let source = image.affine();
    let inverse = invert_affine(&source)
        .ok_or_else(|| Error::InvalidAffine(format!("image affine is singular: {source:?}")))?;
    let change = GridChange {
        map: matmul(&inverse, target_affine),
        shape: target_shape,
    };
    apply_grid_change(image, &change, interp)
}

pub(crate) const fn identity() -> Affine {
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

/// Resample `image` onto the grid described by `change` and update the
/// header geometry accordingly.
pub(crate) fn apply_grid_change(
    image: &NiftiImage,
    change: &GridChange,
    interp: Interpolation,
) -> Result<NiftiImage> {
    let (spatial, volumes) = split_shape(image.shape());
    let (map, target_shape) = (&change.map, change.shape);
    check_target_shape(target_shape, volumes)?;
    for row in map.iter().take(3) {
        if row.iter().any(|v| !v.is_finite()) {
            return Err(Error::InvalidAffine(format!(
                "voxel mapping is not finite: {map:?}"
            )));
        }
    }
    let plan = Plan::new(map, spatial, target_shape);
    let out_shape = with_spatial_shape(image.shape(), target_shape);
    let (data, keep_scaling) = match interp {
        Interpolation::Trilinear => {
            let src = image.f32_values()?;
            let out = crate::parallel::install(|| {
                plan.run_volumes(&src, volumes, |s, o| plan.trilinear(s, o))
            });
            (ArrayData::F32(fortran_from_vec(&out_shape, out)), false)
        }
        Interpolation::Nearest => {
            let data = dispatch_dtype!(image.dtype(), T => {
                let src = image.elements::<T>()?.as_cow();
                let out = crate::parallel::install(|| plan.run_volumes(&src, volumes, |s, o| plan.nearest(s, o)));
                ArrayData::new(fortran_from_vec(&out_shape, out))
            });
            (data, true)
        }
    };
    let mut header = image.header().clone();
    header.transform_voxels(map);
    if !keep_scaling {
        header.scl_slope = 1.0;
        header.scl_inter = 0.0;
    }
    Ok(NiftiImage::from_parts(header, data))
}

/// Precomputed sampling plan for one target grid.
struct Plan {
    src: [usize; 3],
    dst: [usize; 3],
    kind: PlanKind,
}

enum PlanKind {
    /// Each target axis reads exactly one source axis (scaling, flips,
    /// permutations): separable per-axis tables.
    Separable {
        /// Source axis read by each target axis.
        axis: [usize; 3],
        tables: [AxisTable; 3],
    },
    /// Oblique mapping: step through source coordinates voxel by voxel.
    General { map: Affine },
}

/// Per-target-index sampling data along one axis.
struct AxisTable {
    /// Lower source index, or `usize::MAX` when outside the source.
    lo: Vec<usize>,
    /// Upper source index (clamped).
    hi: Vec<usize>,
    /// Weight of `hi` for trilinear interpolation.
    frac: Vec<f32>,
    /// Nearest source index, or `usize::MAX` when outside the source.
    near: Vec<usize>,
}

const OUTSIDE: usize = usize::MAX;

impl AxisTable {
    fn new(n_dst: usize, n_src: usize, scale: f64, offset: f64) -> Self {
        let mut t = Self {
            lo: Vec::with_capacity(n_dst),
            hi: Vec::with_capacity(n_dst),
            frac: Vec::with_capacity(n_dst),
            near: Vec::with_capacity(n_dst),
        };
        let max = (n_src - 1) as f64;
        for i in 0..n_dst {
            let p = scale * i as f64 + offset;
            // Inside the source voxel extents [-0.5, n - 0.5].
            if p >= -0.5 - 1e-9 && p <= max + 0.5 + 1e-9 {
                let c = p.clamp(0.0, max);
                let lo = c.floor();
                let lo_i = lo as usize;
                t.lo.push(lo_i);
                t.hi.push((lo_i + 1).min(n_src - 1));
                t.frac.push((c - lo) as f32);
                t.near.push((((p + 0.5).floor()).clamp(0.0, max)) as usize);
            } else {
                t.lo.push(OUTSIDE);
                t.hi.push(OUTSIDE);
                t.frac.push(0.0);
                t.near.push(OUTSIDE);
            }
        }
        t
    }
}

impl Plan {
    fn new(map: &Affine, src: [usize; 3], dst: [usize; 3]) -> Self {
        // Separable when each target axis reads exactly one source axis and
        // no source axis is read twice.
        let mut axis = [0usize; 3];
        let mut used = [false; 3];
        let mut separable = true;
        for t in 0..3 {
            let mut nonzero = (0..3).filter(|&s| map[s][t].abs() > 1e-12);
            match (nonzero.next(), nonzero.next()) {
                (Some(s), None) if !used[s] => {
                    axis[t] = s;
                    used[s] = true;
                }
                _ => separable = false,
            }
        }
        let kind = if separable {
            let tables = std::array::from_fn(|t| {
                let s = axis[t];
                AxisTable::new(dst[t], src[s], map[s][t], map[s][3])
            });
            PlanKind::Separable { axis, tables }
        } else {
            PlanKind::General { map: *map }
        };
        Self { src, dst, kind }
    }

    fn src_len(&self) -> usize {
        self.src.iter().product()
    }

    fn dst_len(&self) -> usize {
        self.dst.iter().product()
    }

    /// Run `kernel(src_volume, dst_volume)` for each volume.
    fn run_volumes<T, U, F>(&self, src: &[T], volumes: usize, kernel: F) -> Vec<U>
    where
        T: Sync,
        U: Send + Default + Clone,
        F: Fn(&[T], &mut [U]) + Sync,
    {
        let (sn, dn) = (self.src_len(), self.dst_len());
        let mut out = vec![U::default(); dn * volumes];
        for v in 0..volumes {
            kernel(&src[v * sn..(v + 1) * sn], &mut out[v * dn..(v + 1) * dn]);
        }
        out
    }

    fn strides(&self) -> [usize; 3] {
        [1, self.src[0], self.src[0] * self.src[1]]
    }

    fn trilinear(&self, src: &[f32], out: &mut [f32]) {
        let [dx, dy, _] = self.dst;
        let stride = self.strides();
        match &self.kind {
            PlanKind::Separable { axis, tables } => {
                let [tx, ty, tz] = tables;
                let (sx, sy, sz) = (stride[axis[0]], stride[axis[1]], stride[axis[2]]);
                out.par_chunks_mut(dx * dy)
                    .enumerate()
                    .for_each(|(z, slab)| {
                        let (z0, z1) = (tz.lo[z], tz.hi[z]);
                        if z0 == OUTSIDE {
                            slab.fill(0.0);
                            return;
                        }
                        let fz = tz.frac[z];
                        for y in 0..dy {
                            let row = &mut slab[y * dx..(y + 1) * dx];
                            let (y0, y1) = (ty.lo[y], ty.hi[y]);
                            if y0 == OUTSIDE {
                                row.fill(0.0);
                                continue;
                            }
                            let fy = ty.frac[y];
                            let w = [
                                (1.0 - fy) * (1.0 - fz),
                                fy * (1.0 - fz),
                                (1.0 - fy) * fz,
                                fy * fz,
                            ];
                            let base = [
                                y0 * sy + z0 * sz,
                                y1 * sy + z0 * sz,
                                y0 * sy + z1 * sz,
                                y1 * sy + z1 * sz,
                            ];
                            for (x, dst) in row.iter_mut().enumerate() {
                                let (x0, x1) = (tx.lo[x], tx.hi[x]);
                                if x0 == OUTSIDE {
                                    *dst = 0.0;
                                    continue;
                                }
                                let fx = tx.frac[x];
                                let (a, b) = (x0 * sx, x1 * sx);
                                let mut acc = 0.0f32;
                                for c in 0..4 {
                                    let v0 = src[base[c] + a];
                                    let v1 = src[base[c] + b];
                                    acc += w[c] * (v0 + (v1 - v0) * fx);
                                }
                                *dst = acc;
                            }
                        }
                    });
            }
            PlanKind::General { map } => {
                let src_dims = self.src;
                out.par_chunks_mut(dx * dy)
                    .enumerate()
                    .for_each(|(z, slab)| {
                        for y in 0..dy {
                            for x in 0..dx {
                                let p = map_point(map, x, y, z);
                                slab[y * dx + x] = sample_trilinear(src, src_dims, stride, p);
                            }
                        }
                    });
            }
        }
    }

    fn nearest<T: Copy + Default + Send + Sync>(&self, src: &[T], out: &mut [T]) {
        let [dx, dy, _] = self.dst;
        let stride = self.strides();
        match &self.kind {
            PlanKind::Separable { axis, tables } => {
                let [tx, ty, tz] = tables;
                let (sx, sy, sz) = (stride[axis[0]], stride[axis[1]], stride[axis[2]]);
                out.par_chunks_mut(dx * dy)
                    .enumerate()
                    .for_each(|(z, slab)| {
                        let zi = tz.near[z];
                        for y in 0..dy {
                            let row = &mut slab[y * dx..(y + 1) * dx];
                            let yi = ty.near[y];
                            if zi == OUTSIDE || yi == OUTSIDE {
                                row.fill(T::default());
                                continue;
                            }
                            let base = yi * sy + zi * sz;
                            for (x, dst) in row.iter_mut().enumerate() {
                                let xi = tx.near[x];
                                *dst = if xi == OUTSIDE {
                                    T::default()
                                } else {
                                    src[base + xi * sx]
                                };
                            }
                        }
                    });
            }
            PlanKind::General { map } => {
                let src_dims = self.src;
                out.par_chunks_mut(dx * dy)
                    .enumerate()
                    .for_each(|(z, slab)| {
                        for y in 0..dy {
                            for x in 0..dx {
                                let p = map_point(map, x, y, z);
                                slab[y * dx + x] = sample_nearest(src, src_dims, stride, p);
                            }
                        }
                    });
            }
        }
    }
}

#[inline]
fn map_point(map: &Affine, x: usize, y: usize, z: usize) -> [f64; 3] {
    let (x, y, z) = (x as f64, y as f64, z as f64);
    std::array::from_fn(|r| map[r][0] * x + map[r][1] * y + map[r][2] * z + map[r][3])
}

#[inline]
fn inside(p: f64, n: usize) -> bool {
    p >= -0.5 - 1e-9 && p <= (n as f64) - 0.5 + 1e-9
}

#[inline]
fn sample_trilinear(src: &[f32], dims: [usize; 3], stride: [usize; 3], p: [f64; 3]) -> f32 {
    if !(0..3).all(|a| inside(p[a], dims[a])) {
        return 0.0;
    }
    let mut lo = [0usize; 3];
    let mut hi = [0usize; 3];
    let mut f = [0f32; 3];
    for a in 0..3 {
        let max = (dims[a] - 1) as f64;
        let c = p[a].clamp(0.0, max);
        let l = c.floor();
        lo[a] = l as usize;
        hi[a] = (lo[a] + 1).min(dims[a] - 1);
        f[a] = (c - l) as f32;
    }
    let at = |x: usize, y: usize, z: usize| src[x * stride[0] + y * stride[1] + z * stride[2]];
    let c00 = at(lo[0], lo[1], lo[2]) * (1.0 - f[0]) + at(hi[0], lo[1], lo[2]) * f[0];
    let c10 = at(lo[0], hi[1], lo[2]) * (1.0 - f[0]) + at(hi[0], hi[1], lo[2]) * f[0];
    let c01 = at(lo[0], lo[1], hi[2]) * (1.0 - f[0]) + at(hi[0], lo[1], hi[2]) * f[0];
    let c11 = at(lo[0], hi[1], hi[2]) * (1.0 - f[0]) + at(hi[0], hi[1], hi[2]) * f[0];
    let c0 = c00 * (1.0 - f[1]) + c10 * f[1];
    let c1 = c01 * (1.0 - f[1]) + c11 * f[1];
    c0 * (1.0 - f[2]) + c1 * f[2]
}

#[inline]
fn sample_nearest<T: Copy + Default>(
    src: &[T],
    dims: [usize; 3],
    stride: [usize; 3],
    p: [f64; 3],
) -> T {
    let mut offset = 0;
    for a in 0..3 {
        if !inside(p[a], dims[a]) {
            return T::default();
        }
        let i = ((p[a] + 0.5).floor()).clamp(0.0, (dims[a] - 1) as f64) as usize;
        offset += i * stride[a];
    }
    src[offset]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nifti::{DataType, NiftiElement};
    use ndarray::{ArrayD, IxDyn, ShapeBuilder};

    fn image<T: NiftiElement>(
        shape: &[usize],
        f: impl Fn(usize) -> T,
        affine: Affine,
    ) -> NiftiImage {
        let n: usize = shape.iter().product();
        let arr = ArrayD::from_shape_vec(IxDyn(shape).f(), (0..n).map(f).collect()).unwrap();
        NiftiImage::from_array(arr, affine).unwrap()
    }

    fn scaled(s: [f64; 3]) -> Affine {
        [
            [s[0], 0.0, 0.0, 10.0],
            [0.0, s[1], 0.0, -20.0],
            [0.0, 0.0, s[2], 30.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    }

    fn world(a: &Affine, v: [f64; 3]) -> [f64; 3] {
        std::array::from_fn(|r| (0..3).map(|c| a[r][c] * v[c]).sum::<f64>() + a[r][3])
    }

    #[test]
    fn same_shape_is_identity_for_any_layout() {
        let img = image(&[5, 6, 7], |v| v as f32, scaled([1.0, 2.0, 3.0]));
        for interp in [Interpolation::Trilinear, Interpolation::Nearest] {
            let r = resample_to_shape(&img, [5, 6, 7], interp).unwrap();
            assert_eq!(r.to_f32().unwrap(), img.to_f32().unwrap());
            assert_eq!(r.affine(), img.affine());
        }
    }

    #[test]
    fn half_voxel_convention_and_fov() {
        // A linear ramp along x resamples to the ramp evaluated at the new
        // voxel centres.
        let img = image(&[8, 1, 1], |v| v as f32, scaled([1.0, 1.0, 1.0]));
        let r = resample_to_shape(&img, [4, 1, 1], Interpolation::Trilinear).unwrap();
        let v = r.to_f32().unwrap();
        assert_eq!(v.as_slice_memory_order().unwrap(), &[0.5, 2.5, 4.5, 6.5]);
        // Voxel 0's centre sits at old voxel 0.5 in world space.
        let a = r.affine();
        assert!((world(&a, [0.0; 3])[0] - 10.5).abs() < 1e-12);
        assert!((r.spacing()[0] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn nearest_keeps_dtype_and_labels() {
        let img = image(&[4, 4, 4], |v| (v % 5) as u8, scaled([1.0; 3]));
        let r = resample_to_shape(&img, [7, 9, 3], Interpolation::Nearest).unwrap();
        assert_eq!(r.dtype(), DataType::UInt8);
        let values: std::collections::BTreeSet<u8> =
            r.as_array::<u8>().unwrap().iter().copied().collect();
        assert!(values.iter().all(|v| *v < 5));
    }

    #[test]
    fn spacing_resample_validates_and_rounds() {
        let img = image(&[10, 10, 10], |v| v as f32, scaled([1.0, 1.0, 1.0]));
        assert!(resample_to_spacing(&img, [f64::NAN, 1.0, 1.0], Interpolation::Nearest).is_err());
        assert!(resample_to_spacing(&img, [0.0, 1.0, 1.0], Interpolation::Nearest).is_err());
        assert!(resample_to_spacing(&img, [1e-12, 1.0, 1.0], Interpolation::Nearest).is_err());
        let r = resample_to_spacing(&img, [3.0, 1.0, 0.5], Interpolation::Trilinear).unwrap();
        assert_eq!(r.shape(), &[3, 10, 20]);
        // Achieved spacing along x is 10/3 because of rounding.
        assert!((r.spacing()[0] - 10.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn resample_like_aligns_in_world_space() {
        // Reference: 1 mm grid. Moving image: 2 mm grid shifted by 3 mm.
        let reference = image(&[10, 10, 10], |_| 0.0f32, scaled([1.0; 3]));
        let mut moving_affine = scaled([2.0; 3]);
        moving_affine[0][3] += 3.0;
        let moving = image(&[5, 5, 5], |v| v as f32, moving_affine);
        let r = resample_like(&moving, &reference, Interpolation::Nearest).unwrap();
        assert_eq!(r.shape(), reference.shape());
        assert_eq!(r.affine(), reference.affine());
        // Reference voxel x=0 is 3 mm left of the moving image's first voxel.
        let a = r.as_array::<f32>().unwrap();
        assert_eq!(a[[0, 0, 0]], 0.0);
        assert_eq!(a[[3, 0, 0]], 0.0); // moving voxel (0,0,0) has value 0
        assert_eq!(a[[5, 0, 0]], 1.0); // world 15 mm -> moving voxel x=1
    }

    #[test]
    fn oblique_and_permuted_grids() {
        let img = image(&[6, 5, 4], |v| v as f32, scaled([1.0; 3]));
        // Permutation: swap x and y in the target grid.
        let src = img.affine();
        let mut perm = src;
        for r in 0..3 {
            perm[r][0] = src[r][1];
            perm[r][1] = src[r][0];
        }
        let r = resample_to_grid(&img, &perm, [5, 6, 4], Interpolation::Nearest).unwrap();
        let (a, b) = (r.as_array::<f32>().unwrap(), img.as_array::<f32>().unwrap());
        assert_eq!(a[[2, 3, 1]], b[[3, 2, 1]]);
        // A 45-degree rotation still returns finite values and zero outside.
        let (c, s) = (
            std::f64::consts::FRAC_1_SQRT_2,
            std::f64::consts::FRAC_1_SQRT_2,
        );
        let mut rot = src;
        rot[0][0] = c;
        rot[0][1] = -s;
        rot[1][0] = s;
        rot[1][1] = c;
        let r = resample_to_grid(&img, &rot, [6, 6, 4], Interpolation::Trilinear).unwrap();
        assert!(r.to_f32().unwrap().iter().all(|v| v.is_finite()));
    }

    #[test]
    fn four_d_resamples_each_volume() {
        let img = image(&[4, 4, 4, 3], |v| (v / 64) as f32, scaled([1.0; 3]));
        let r = resample_to_shape(&img, [2, 2, 2], Interpolation::Trilinear).unwrap();
        assert_eq!(r.shape(), &[2, 2, 2, 3]);
        let a = r.as_array::<f32>().unwrap();
        for t in 0..3 {
            assert_eq!(a[[1, 1, 1, t]], t as f32);
        }
    }

    #[test]
    fn preserves_codes_and_scaling_rules() {
        let mut img = image(&[4, 4, 4], |v| v as i16, scaled([1.0; 3]));
        img.header_mut().sform_code = 4;
        img.header_mut().scl_slope = 2.0;
        let n = resample_to_shape(&img, [2, 2, 2], Interpolation::Nearest).unwrap();
        assert_eq!(n.header().sform_code, 4);
        assert_eq!(n.header().scl_slope, 2.0);
        let t = resample_to_shape(&img, [2, 2, 2], Interpolation::Trilinear).unwrap();
        assert_eq!(t.dtype(), DataType::Float32);
        assert_eq!(t.header().scl_slope, 1.0);
        assert_eq!(t.header().sform_code, 4);
    }

    #[test]
    fn two_d_images_resample() {
        let img = image(&[4, 6], |v| v as f32, scaled([1.0; 3]));
        let r = resample_to_shape(&img, [2, 3, 1], Interpolation::Trilinear).unwrap();
        assert_eq!(r.shape(), &[2, 3]);
    }
}
