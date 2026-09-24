//! Exact spatial transforms: crop, pad, flip, and 90-degree rotation.
//!
//! These rearrange voxels without interpolation, so they keep the datatype and
//! scaling of the input. All of them are *world-preserving*: the affine is
//! updated so every voxel keeps its position in scanner space, which is what
//! prevents left/right mix-ups when a transformed image is saved and viewed.
//! Axes beyond the third are carried through.

use super::geometry::{
    crop_or_pad_plan, crop_plan, flip_axes, flip_plan, rotate_plan, rotation_args, spatial_rank,
    split_shape, with_spatial_shape,
};
use crate::error::{Error, Result};
use crate::nifti::element::{dispatch_dtype, map_array, to_fortran, NiftiElement};
use crate::nifti::NiftiImage;
use ndarray::{ArrayD, Axis, IxDyn, ShapeBuilder, Slice};

/// Crop `shape` voxels starting at `offset` along the first three axes.
///
/// # Errors
/// [`Error::InvalidCropRegion`] if the region is empty or extends past the
/// image.
pub fn crop(image: &NiftiImage, offset: [usize; 3], shape: [usize; 3]) -> Result<NiftiImage> {
    let (spatial, _) = split_shape(image.shape());
    let change = crop_plan(spatial, offset, shape)?;
    let rank = spatial_rank(image.shape());
    let data = image.raw_data()?;
    let data = map_array!(data.as_ref(), |a| {
        let mut view = a.view();
        for axis in 0..rank {
            view.slice_axis_inplace(
                Axis(axis),
                Slice::from(offset[axis]..offset[axis] + shape[axis]),
            );
        }
        to_fortran(&view)
    });
    let mut header = image.header().clone();
    header.transform_voxels(&change.map);
    Ok(NiftiImage::from_parts(header, data))
}

/// Centre-crop or pad the first three axes to `target`.
///
/// Where an axis shrinks, the central region is kept (the extra voxel is
/// dropped from the far side when the difference is odd); where it grows, the
/// image is centred in the new grid. New voxels get `pad_value`, in scaled
/// units. If `pad_value` cannot be stored exactly in the image's datatype (for
/// example `-1` in a `u8` image), the output is converted to `f32`.
pub fn crop_or_pad(image: &NiftiImage, target: [usize; 3], pad_value: f64) -> Result<NiftiImage> {
    if !pad_value.is_finite() {
        return Err(Error::InvalidArgument(format!(
            "pad value must be finite, got {pad_value}"
        )));
    }
    let (spatial, _) = split_shape(image.shape());
    let (change, pads) = crop_or_pad_plan(spatial, target)?;
    let out_shape = with_spatial_shape(image.shape(), target);
    if out_shape
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .is_none()
    {
        return Err(Error::InvalidDimensions(format!(
            "target shape {target:?} is too large"
        )));
    }

    let mut src_start = [0usize; 3];
    let mut dst_start = [0usize; 3];
    let mut len = [0usize; 3];
    for axis in 0..3 {
        let (n, t) = (spatial[axis], target[axis]);
        if t <= n {
            src_start[axis] = (n - t) / 2;
            len[axis] = t;
        } else {
            dst_start[axis] = (t - n) / 2;
            len[axis] = n;
        }
    }

    let (slope, inter) = image.header().scaling();
    let raw_pad = (pad_value - inter) / slope;
    let fits =
        !pads || dispatch_dtype!(image.dtype(), T => T::from_f64(raw_pad).to_f64() == raw_pad);
    let (source, fill) = if fits {
        (image.clone(), raw_pad)
    } else {
        (image.with_data(image.to_f32()?)?, pad_value)
    };

    let data = source.raw_data()?;
    let data = map_array!(data.as_ref(), |a| {
        paste(a, &out_shape, src_start, dst_start, len, fill)
    });
    let mut header = source.header().clone();
    header.transform_voxels(&change.map);
    Ok(NiftiImage::from_parts(header, data))
}

/// Copy `len` voxels from `a` at `src_start` into a new array of `out_shape`
/// at `dst_start`, filling the rest with `fill`.
fn paste<T: NiftiElement>(
    a: &ArrayD<T>,
    out_shape: &[usize],
    src_start: [usize; 3],
    dst_start: [usize; 3],
    len: [usize; 3],
    fill: f64,
) -> ArrayD<T> {
    let rank = a.ndim().min(3);
    let mut out = ArrayD::from_elem(IxDyn(out_shape).f(), T::from_f64(fill));
    let mut src = a.view();
    let mut dst = out.view_mut();
    // A 1D/2D image that gained spatial axes: append them to the source view.
    while src.ndim() < dst.ndim() {
        let n = src.ndim();
        src = src.insert_axis(Axis(n));
    }
    for axis in 0..rank.max(dst.ndim().min(3)) {
        src.slice_axis_inplace(
            Axis(axis),
            Slice::from(src_start[axis]..src_start[axis] + len[axis]),
        );
        dst.slice_axis_inplace(
            Axis(axis),
            Slice::from(dst_start[axis]..dst_start[axis] + len[axis]),
        );
    }
    dst.assign(&src);
    out
}

/// Reverse the voxel order along the given spatial axes (0, 1, 2).
///
/// World-preserving: the affine is updated so the anatomy stays where it
/// was (like MONAI's `Flip` on a `MetaTensor`). Duplicate axes flip once.
pub fn flip(image: &NiftiImage, axes: &[usize]) -> Result<NiftiImage> {
    let selected = flip_axes(spatial_rank(image.shape()), axes)?;
    if !selected.contains(&true) {
        return Ok(image.clone());
    }
    let (spatial, _) = split_shape(image.shape());
    let change = flip_plan(spatial, selected);
    let data = image.raw_data()?;
    let data = map_array!(data.as_ref(), |a| {
        let mut view = a.view();
        for (axis, &s) in selected.iter().enumerate() {
            if s {
                view.invert_axis(Axis(axis));
            }
        }
        to_fortran(&view)
    });
    let mut header = image.header().clone();
    header.transform_voxels(&change.map);
    Ok(NiftiImage::from_parts(header, data))
}

/// Rotate by `k` × 90 degrees in the plane of two spatial axes.
///
/// Follows `numpy.rot90(a, k, axes)`: the rotation goes from the first axis
/// towards the second. World-preserving: the affine is updated so the anatomy
/// stays in place (anisotropic voxel sizes follow their axes).
pub fn rotate_90(image: &NiftiImage, axes: (usize, usize), k: i32) -> Result<NiftiImage> {
    let k = rotation_args(spatial_rank(image.shape()), axes, k)?;
    if k == 0 {
        return Ok(image.clone());
    }
    let (a, b) = axes;
    let (spatial, _) = split_shape(image.shape());
    let change = rotate_plan(spatial, axes, k);
    let data = image.raw_data()?;
    let data = map_array!(data.as_ref(), |arr| {
        let mut view = arr.view();
        match k {
            1 => {
                view.invert_axis(Axis(b));
                view.swap_axes(a, b);
            }
            2 => {
                view.invert_axis(Axis(a));
                view.invert_axis(Axis(b));
            }
            _ => {
                view.swap_axes(a, b);
                view.invert_axis(Axis(b));
            }
        }
        to_fortran(&view)
    });
    let mut header = image.header().clone();
    header.transform_voxels(&change.map);
    Ok(NiftiImage::from_parts(header, data))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nifti::header::Affine;
    use crate::nifti::DataType;
    use ndarray::{s, ArrayD, IxDyn, ShapeBuilder};

    fn affine() -> Affine {
        [
            [0.9, 0.1, 0.0, 10.0],
            [-0.1, 1.9, 0.0, -20.0],
            [0.0, 0.0, 3.0, 30.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    }

    fn image(shape: &[usize]) -> (ArrayD<i32>, NiftiImage) {
        let n: usize = shape.iter().product();
        let arr = ArrayD::from_shape_vec(IxDyn(shape).f(), (0..n as i32).collect()).unwrap();
        (arr.clone(), NiftiImage::from_array(arr, affine()).unwrap())
    }

    fn world(a: &Affine, v: [f64; 3]) -> [f64; 3] {
        crate::transforms::geometry::apply(a, v)
    }

    /// Assert that each output voxel sits at the same world position as the
    /// input voxel with the same value (values are unique voxel ids).
    fn assert_world_preserving(before: &NiftiImage, after: &NiftiImage) {
        let (a_in, a_out) = (before.affine(), after.affine());
        let src = before.to_scaled::<i32>().unwrap();
        let out = after.as_array::<i32>().unwrap();
        for (idx, &v) in out.indexed_iter() {
            let (old_idx, _) = src.indexed_iter().find(|&(_, &x)| x == v).unwrap();
            let old = [old_idx[0] as f64, old_idx[1] as f64, old_idx[2] as f64];
            let new = [idx[0] as f64, idx[1] as f64, idx[2] as f64];
            let (p, q) = (world(&a_in, old), world(&a_out, new));
            for r in 0..3 {
                assert!((p[r] - q[r]).abs() < 1e-9, "voxel {v}: {p:?} vs {q:?}");
            }
        }
    }

    #[test]
    fn crop_matches_slicing_and_world() {
        let (arr, img) = image(&[6, 7, 8]);
        let c = crop(&img, [1, 2, 3], [3, 4, 2]).unwrap();
        assert_eq!(
            c.as_array::<i32>().unwrap(),
            &arr.slice(s![1..4, 2..6, 3..5]).to_owned().into_dyn()
        );
        assert_world_preserving(&img, &c);
        assert!(crop(&img, [5, 0, 0], [2, 1, 1]).is_err());
        assert!(crop(&img, [usize::MAX, 0, 0], [2, 1, 1]).is_err());
    }

    #[test]
    fn flip_is_world_preserving_and_involutive() {
        let (_, img) = image(&[4, 5, 6]);
        for axes in [vec![0], vec![1, 2], vec![0, 1, 2], vec![2, 2]] {
            let f = flip(&img, &axes).unwrap();
            assert_world_preserving(&img, &f);
            let back = flip(&f, &axes).unwrap();
            assert_eq!(back.as_array::<i32>(), img.as_array::<i32>());
        }
        assert!(flip(&img, &[3]).is_err());
    }

    #[test]
    fn rotate_matches_numpy_rot90_and_world() {
        let (arr, img) = image(&[3, 4, 2]);
        let r = rotate_90(&img, (0, 1), 1).unwrap();
        assert_eq!(r.shape(), &[4, 3, 2]);
        // numpy.rot90(a, 1, (0, 1))[i, j] == a[j, n1 - 1 - i]
        let out = r.as_array::<i32>().unwrap();
        for i in 0..4 {
            for j in 0..3 {
                assert_eq!(out[[i, j, 1]], arr[[j, 3 - i, 1]]);
            }
        }
        for k in [-1, 1, 2, 3, 5] {
            let r = rotate_90(&img, (2, 0), k).unwrap();
            assert_world_preserving(&img, &r);
        }
        let four = rotate_90(&rotate_90(&img, (0, 1), 3).unwrap(), (0, 1), 1).unwrap();
        assert_eq!(four.as_array::<i32>(), img.as_array::<i32>());
        assert!(rotate_90(&img, (0, 0), 1).is_err());
    }

    #[test]
    fn crop_or_pad_centres_and_pads_in_scaled_units() {
        let (_, img) = image(&[4, 5, 6]);
        let p = crop_or_pad(&img, [6, 3, 6], -7.0).unwrap();
        assert_eq!(p.shape(), &[6, 3, 6]);
        assert_eq!(p.dtype(), DataType::Int32);
        assert_eq!(p.as_array::<i32>().unwrap()[[0, 0, 0]], -7);
        assert_world_preserving(&img, &crop(&p, [1, 0, 0], [4, 3, 6]).unwrap());

        // u8 cannot store -1: converted to f32.
        let u = img.with_dtype(DataType::UInt8).unwrap();
        let p = crop_or_pad(&u, [6, 5, 6], -1.0).unwrap();
        assert_eq!(p.dtype(), DataType::Float32);
        assert_eq!(p.to_f32().unwrap()[[0, 0, 0]], -1.0);

        // Scaled int16: pad value 0 in physical units is raw 1024.
        let mut ct = img.with_dtype(DataType::Int16).unwrap();
        ct.header_mut().scl_inter = -1024.0;
        let p = crop_or_pad(&ct, [5, 5, 6], 0.0).unwrap();
        assert_eq!(p.dtype(), DataType::Int16);
        assert_eq!(p.to_f32().unwrap()[[4, 0, 0]], 0.0);
    }

    #[test]
    fn four_d_and_two_d_images() {
        let (arr, img) = image(&[4, 5, 3, 2]);
        let c = crop(&img, [1, 1, 0], [2, 3, 3]).unwrap();
        assert_eq!(c.shape(), &[2, 3, 3, 2]);
        assert_eq!(
            c.as_array::<i32>().unwrap(),
            &arr.slice(s![1..3, 1..4, .., ..]).to_owned().into_dyn()
        );
        let f = flip(&img, &[0]).unwrap();
        assert_eq!(f.shape(), img.shape());
        let (_, flat) = image(&[4, 5]);
        let r = rotate_90(&flat, (0, 1), 1).unwrap();
        assert_eq!(r.shape(), &[5, 4]);
        let p = crop_or_pad(&flat, [2, 2, 3], 0.0).unwrap();
        assert_eq!(p.shape(), &[2, 2, 3]);
    }
}
