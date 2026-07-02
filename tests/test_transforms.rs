//! Integration tests for the transforms layer.
//!
//! These cover the correctness fixes to reorientation, resampling, cropping,
//! intensity normalization, and augmentation, with an emphasis on affine /
//! world-coordinate consistency and the half-pixel-center sampling convention.

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::suboptimal_flops,
    clippy::doc_markdown,
    clippy::missing_const_for_fn,
    clippy::uninlined_format_args,
    clippy::needless_range_loop,
    clippy::explicit_iter_loop,
    clippy::unreadable_literal
)]

use medrs::nifti::NiftiImage;
use medrs::transforms::{
    clamp, compute_center_crop_regions, compute_label_aware_crop_regions, crop, crop_or_pad,
    random_augment, random_gamma, random_gaussian_noise, reorient, resample_to_shape,
    resample_to_spacing, rotate_90, z_normalization, ForegroundDetector, Interpolation,
    Orientation, RandCropByPosNegLabelConfig,
};
use ndarray::{ArrayD, IxDyn, ShapeBuilder};

const EPS: f32 = 1e-4;

/// Build an F-order NIfTI image from C-order data.
fn make_image(data: Vec<f32>, shape: [usize; 3], affine: [[f32; 4]; 4]) -> NiftiImage {
    let c_order = ArrayD::from_shape_vec(shape.to_vec(), data).unwrap();
    let mut f_order = ArrayD::zeros(IxDyn(&shape).f());
    f_order.assign(&c_order);
    NiftiImage::from_array(f_order, affine)
}

/// A ramp along the X axis: value(x, y, z) = x.
fn x_ramp(shape: [usize; 3]) -> Vec<f32> {
    let [nx, ny, nz] = shape;
    let mut data = vec![0.0f32; nx * ny * nz];
    for x in 0..nx {
        for y in 0..ny {
            for z in 0..nz {
                data[x * ny * nz + y * nz + z] = x as f32;
            }
        }
    }
    data
}

fn identity_affine() -> [[f32; 4]; 4] {
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

/// Apply a 4x4 affine to a voxel index, returning the world coordinate.
fn world(affine: &[[f32; 4]; 4], voxel: [usize; 3]) -> [f32; 3] {
    world_f(affine, [voxel[0] as f32, voxel[1] as f32, voxel[2] as f32])
}

/// Apply a 4x4 affine to a continuous voxel coordinate.
fn world_f(affine: &[[f32; 4]; 4], v: [f32; 3]) -> [f32; 3] {
    let mut out = [0.0f32; 3];
    for (r, o) in out.iter_mut().enumerate() {
        *o = affine[r][0] * v[0] + affine[r][1] * v[1] + affine[r][2] * v[2] + affine[r][3];
    }
    out
}

/// Half-pixel-center source position for output index `i`: (i+0.5)*old/new - 0.5.
fn source_pos(i: usize, old: usize, new: usize) -> f32 {
    (i as f32 + 0.5) * old as f32 / new as f32 - 0.5
}

fn col_norm(affine: &[[f32; 4]; 4], col: usize) -> f32 {
    (affine[0][col].powi(2) + affine[1][col].powi(2) + affine[2][col].powi(2)).sqrt()
}

// ---------------------------------------------------------------------------
// C1: flipped-axis origin adjustment sign
// ---------------------------------------------------------------------------

#[test]
fn reorient_flip_axis_origin_concrete() {
    // 4-voxel X axis, identity affine (RAS), flip R -> L.
    let img = make_image(x_ramp([4, 2, 2]), [4, 2, 2], identity_affine());
    let out = reorient(&img, Orientation::LAS).unwrap();

    let a = out.affine();
    // Column 0 is negated.
    assert!(
        (a[0][0] - (-1.0)).abs() < EPS,
        "flipped column: {}",
        a[0][0]
    );
    // New origin along X is +3 (old voxel 3 becomes new voxel 0).
    assert!((a[0][3] - 3.0).abs() < EPS, "flipped origin: {}", a[0][3]);

    // Data is reversed along X: new (0, .,.) holds old x = 3.
    let d = out.to_f32().unwrap();
    assert!((d[[0, 0, 0]] - 3.0).abs() < EPS);
    assert!((d[[3, 0, 0]] - 0.0).abs() < EPS);
}

// ---------------------------------------------------------------------------
// Reorient round-trip identity on data AND affine (flip-involving affine)
// ---------------------------------------------------------------------------

#[test]
fn reorient_ras_lps_ras_roundtrip() {
    let affine = [
        [2.0, 0.0, 0.0, 10.0],
        [0.0, 3.0, 0.0, 20.0],
        [0.0, 0.0, 4.0, 30.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    let data: Vec<f32> = (0..(4 * 3 * 2)).map(|i| i as f32).collect();
    let img = make_image(data, [4, 3, 2], affine);

    let lps = reorient(&img, Orientation::LPS).unwrap();
    let back = reorient(&lps, Orientation::RAS).unwrap();

    // Affine restored.
    let a0 = img.affine();
    let a1 = back.affine();
    for r in 0..4 {
        for c in 0..4 {
            assert!(
                (a0[r][c] - a1[r][c]).abs() < EPS,
                "affine[{}][{}]: {} vs {}",
                r,
                c,
                a0[r][c],
                a1[r][c]
            );
        }
    }

    // Data restored.
    let d0 = img.to_f32().unwrap();
    let d1 = back.to_f32().unwrap();
    assert_eq!(d0, d1);
}

// ---------------------------------------------------------------------------
// Reorient preserves world coordinates of every voxel
// ---------------------------------------------------------------------------

#[test]
fn reorient_preserves_world_coordinates() {
    let affine = [
        [2.0, 0.0, 0.0, 10.0],
        [0.0, 3.0, 0.0, 20.0],
        [0.0, 0.0, 4.0, 30.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    let shape = [4, 3, 2];
    let data: Vec<f32> = (0..(4 * 3 * 2)).map(|i| i as f32).collect();
    let img = make_image(data, shape, affine);

    // RAS -> LPS flips X and Y; output voxel (Nx-1-i, Ny-1-j, k).
    let out = reorient(&img, Orientation::LPS).unwrap();
    let a_out = out.affine();
    let a_in = img.affine();

    for i in 0..shape[0] {
        for j in 0..shape[1] {
            for k in 0..shape[2] {
                let w_in = world(&a_in, [i, j, k]);
                let mapped = [shape[0] - 1 - i, shape[1] - 1 - j, k];
                let w_out = world(&a_out, mapped);
                for c in 0..3 {
                    assert!(
                        (w_in[c] - w_out[c]).abs() < 1e-3,
                        "world mismatch at ({},{},{}) axis {}: {} vs {}",
                        i,
                        j,
                        k,
                        c,
                        w_in[c],
                        w_out[c]
                    );
                }
            }
        }
    }
}

#[test]
fn reorient_rejects_non_3d() {
    let arr = ArrayD::<f32>::zeros(IxDyn(&[2, 2, 2, 2]).f());
    let img = NiftiImage::from_array(arr, identity_affine());
    assert!(reorient(&img, Orientation::LPS).is_err());
}

// M10: degenerate affine yielding duplicate axes must error.
#[test]
fn reorient_rejects_degenerate_affine() {
    let affine = [
        [1.0, 0.9, 0.0, 0.0],
        [0.0, 0.1, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    let img = make_image(x_ramp([3, 3, 3]), [3, 3, 3], affine);
    assert!(reorient(&img, Orientation::RAS).is_err());
}

// ---------------------------------------------------------------------------
// I1: oblique-affine resample column norms equal the ACHIEVED spacing
// (current_spacing * old_shape / new_shape), which matches the request only
// when the implied shape is integral.
// ---------------------------------------------------------------------------

#[test]
fn resample_oblique_affine_column_norms() {
    let theta: f32 = 30.0_f32.to_radians();
    let (c, s) = (theta.cos(), theta.sin());
    // Columns carry spacing (1.5, 2.0, 3.0) with an in-plane rotation on X/Y.
    let old = [8usize, 8, 8];
    let affine = [
        [1.5 * c, 2.0 * -s, 0.0, 0.0],
        [1.5 * s, 2.0 * c, 0.0, 0.0],
        [0.0, 0.0, 3.0, 5.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    let current = [1.5f32, 2.0, 3.0];
    let img = make_image(x_ramp(old), old, affine);

    // Awkward target that does not divide evenly, so rounding kicks in.
    let target = [1.1f32, 0.9, 1.3];
    let out = resample_to_spacing(&img, target, Interpolation::Trilinear).unwrap();
    let new = out.shape();
    let a = out.affine();
    for col in 0..3 {
        let achieved = current[col] * old[col] as f32 / new[col] as f32;
        assert!(
            (col_norm(&a, col) - achieved).abs() < 1e-3,
            "column {} norm = {} (achieved spacing {})",
            col,
            col_norm(&a, col),
            achieved
        );
    }
}

// Under half-pixel-center sampling, the world position of new voxel center i
// (via the new affine) equals the world position of its source position (via
// the old affine). Uses an oblique affine so all three columns cross-couple.
fn oblique_affine_with_origin() -> [[f32; 4]; 4] {
    let theta: f32 = 20.0_f32.to_radians();
    let (c, s) = (theta.cos(), theta.sin());
    [
        [1.5 * c, 2.0 * -s, 0.0, 4.0],
        [1.5 * s, 2.0 * c, 0.0, -3.0],
        [0.0, 0.0, 3.0, 5.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

#[test]
fn resample_to_shape_preserves_world_coordinates() {
    let old = [8usize, 8, 8];
    let affine = oblique_affine_with_origin();
    let img = make_image(x_ramp(old), old, affine);

    let new = [4usize, 6, 8];
    let out = resample_to_shape(&img, new, Interpolation::Trilinear).unwrap();
    let a_new = out.affine();

    for &i in &[1usize, 2] {
        for &j in &[1usize, 2, 4] {
            for &k in &[2usize, 5] {
                let p = [
                    source_pos(i, old[0], new[0]),
                    source_pos(j, old[1], new[1]),
                    source_pos(k, old[2], new[2]),
                ];
                let w_new = world(&a_new, [i, j, k]);
                let w_old = world_f(&affine, p);
                for d in 0..3 {
                    assert!(
                        (w_new[d] - w_old[d]).abs() < 1e-3,
                        "axis {} at ({},{},{}): {} vs {}",
                        d,
                        i,
                        j,
                        k,
                        w_new[d],
                        w_old[d]
                    );
                }
            }
        }
    }
}

#[test]
fn resample_to_spacing_preserves_world_coordinates() {
    // Awkward target spacing that forces shape rounding. Because the column is
    // scaled by the actual shape ratio, world preservation is exact regardless.
    let old = [8usize, 8, 8];
    let affine = oblique_affine_with_origin();
    let img = make_image(x_ramp(old), old, affine);

    let target = [1.1f32, 0.9, 1.3];
    let out = resample_to_spacing(&img, target, Interpolation::Trilinear).unwrap();
    let a_new = out.affine();
    let new = out.shape();

    for &i in &[1usize, 5, 8] {
        for &j in &[2usize, 9, 15] {
            for &k in &[3usize, 12] {
                let p = [
                    source_pos(i, old[0], new[0]),
                    source_pos(j, old[1], new[1]),
                    source_pos(k, old[2], new[2]),
                ];
                let w_new = world(&a_new, [i, j, k]);
                let w_old = world_f(&affine, p);
                for d in 0..3 {
                    assert!(
                        (w_new[d] - w_old[d]).abs() < 1e-3,
                        "axis {} at ({},{},{}): {} vs {}",
                        d,
                        i,
                        j,
                        k,
                        w_new[d],
                        w_old[d]
                    );
                }
            }
        }
    }
}

#[test]
fn resample_rejects_non_3d() {
    let arr = ArrayD::<f32>::zeros(IxDyn(&[4, 4]).f());
    let img = NiftiImage::from_array(arr, identity_affine());
    assert!(resample_to_shape(&img, [2, 2, 2], Interpolation::Trilinear).is_err());
    assert!(resample_to_spacing(&img, [1.0, 1.0, 1.0], Interpolation::Trilinear).is_err());
}

// ---------------------------------------------------------------------------
// I2/I3: half-pixel-center convention; ramp downsample values; nearest agrees
// ---------------------------------------------------------------------------

#[test]
fn resample_ramp_downsample_halfpixel_values() {
    // 8 -> 4 along X. Half-pixel source positions: (i + 0.5) * 2 - 0.5 = 2i + 0.5.
    let img = make_image(x_ramp([8, 2, 2]), [8, 2, 2], identity_affine());

    let tri = resample_to_shape(&img, [4, 2, 2], Interpolation::Trilinear).unwrap();
    let near = resample_to_shape(&img, [4, 2, 2], Interpolation::Nearest).unwrap();

    let td = tri.to_f32().unwrap();
    let nd = near.to_f32().unwrap();

    let expected_tri = [0.5f32, 2.5, 4.5, 6.5];
    let expected_near = [1.0f32, 3.0, 5.0, 7.0];
    for i in 0..4 {
        assert!(
            (td[[i, 0, 0]] - expected_tri[i]).abs() < 1e-3,
            "trilinear[{}] = {}",
            i,
            td[[i, 0, 0]]
        );
        assert!(
            (nd[[i, 0, 0]] - expected_near[i]).abs() < 1e-3,
            "nearest[{}] = {}",
            i,
            nd[[i, 0, 0]]
        );
        // Nearest samples the rounded trilinear position: they never differ by
        // more than half a voxel, confirming a shared sampling grid.
        assert!(
            (nd[[i, 0, 0]] - td[[i, 0, 0]]).abs() <= 0.5 + EPS,
            "nearest/trilinear disagree at {}",
            i
        );
    }
}

// Image/label boundary alignment: a step label resampled nearest lands within
// one voxel of where the ramp image crosses the same intensity boundary.
#[test]
fn resample_image_label_boundary_alignment() {
    let shape = [8, 2, 2];
    let ramp = make_image(x_ramp(shape), shape, identity_affine());

    // Label: 1 where x >= 4, else 0 (boundary between voxel 3 and 4, at x = 3.5).
    let [nx, ny, nz] = shape;
    let mut label_data = vec![0.0f32; nx * ny * nz];
    for x in 0..nx {
        for y in 0..ny {
            for z in 0..nz {
                label_data[x * ny * nz + y * nz + z] = if x >= 4 { 1.0 } else { 0.0 };
            }
        }
    }
    let label = make_image(label_data, shape, identity_affine());

    let target = [16, 2, 2];
    let ramp_up = resample_to_shape(&ramp, target, Interpolation::Trilinear).unwrap();
    let label_up = resample_to_shape(&label, target, Interpolation::Nearest).unwrap();

    let rd = ramp_up.to_f32().unwrap();
    let ld = label_up.to_f32().unwrap();

    // First output index where the ramp crosses 3.5 (the input boundary).
    let ramp_boundary = (0..target[0])
        .find(|&i| rd[[i, 0, 0]] >= 3.5)
        .expect("ramp crosses boundary");
    // First output index where the label switches to 1.
    let label_boundary = (0..target[0])
        .find(|&i| ld[[i, 0, 0]] >= 0.5)
        .expect("label switches");

    let diff = (ramp_boundary as i32 - label_boundary as i32).abs();
    assert!(
        diff <= 1,
        "boundary misaligned: {} vs {}",
        ramp_boundary,
        label_boundary
    );
}

// ---------------------------------------------------------------------------
// I4: crop_or_pad origin shift uses integer offsets on odd differences
// ---------------------------------------------------------------------------

#[test]
fn crop_or_pad_odd_crop_origin_integer() {
    // 5 -> 2 along X: diff 3 (odd), integer start = 1.
    let img = make_image(x_ramp([5, 4, 4]), [5, 4, 4], identity_affine());
    let out = crop_or_pad(&img, &[2, 4, 4]).unwrap();

    let a = out.affine();
    assert!((a[0][3] - 1.0).abs() < EPS, "crop origin = {}", a[0][3]);

    // New voxel 0 holds old x = 1.
    let d = out.to_f32().unwrap();
    assert!((d[[0, 0, 0]] - 1.0).abs() < EPS);
}

#[test]
fn crop_or_pad_odd_pad_origin_integer() {
    // 4 -> 7 along X: diff 3 (odd), integer before = 1.
    let img = make_image(x_ramp([4, 4, 4]), [4, 4, 4], identity_affine());
    let out = crop_or_pad(&img, &[7, 4, 4]).unwrap();

    let a = out.affine();
    assert!((a[0][3] - (-1.0)).abs() < EPS, "pad origin = {}", a[0][3]);
}

// ---------------------------------------------------------------------------
// I7: crop rejects non-3D images
// ---------------------------------------------------------------------------

#[test]
fn crop_rejects_non_3d() {
    let arr = ArrayD::<f32>::zeros(IxDyn(&[4, 4, 4, 2]).f());
    let img = NiftiImage::from_array(arr, identity_affine());
    assert!(crop(&img, [0, 0, 0], [2, 2, 2]).is_err());
}

// ---------------------------------------------------------------------------
// M1: rotate_90 updates affine, pixdim, and shape consistently
// ---------------------------------------------------------------------------

#[test]
fn rotate_90_affine_pixdim_consistency() {
    let affine = [
        [2.0, 0.0, 0.0, 5.0],
        [0.0, 3.0, 0.0, 6.0],
        [0.0, 0.0, 4.0, 7.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    let shape = [4, 2, 3];
    // Unique values per voxel so we can check the rotation is a permutation.
    let [nx, ny, nz] = shape;
    let mut data = vec![0.0f32; nx * ny * nz];
    for x in 0..nx {
        for y in 0..ny {
            for z in 0..nz {
                data[x * ny * nz + y * nz + z] = (x * 100 + y * 10 + z) as f32;
            }
        }
    }
    let img = make_image(data.clone(), shape, affine);

    let out = rotate_90(&img, (0, 1), 1).unwrap();

    // Shape swaps axes 0 and 1.
    assert_eq!(out.shape(), &[2, 4, 3]);

    // Expected affine after one 90-degree step in plane (0, 1), size_a = 4:
    //   col0 <- col1 = [0,3,0]; col1 <- -col0 = [-2,0,0];
    //   origin <- origin + col0 * (4 - 1) = [5,6,7] + [2,0,0]*3 = [11,6,7].
    let a = out.affine();
    let expected = [
        [0.0, -2.0, 0.0, 11.0],
        [3.0, 0.0, 0.0, 6.0],
        [0.0, 0.0, 4.0, 7.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    for r in 0..4 {
        for c in 0..4 {
            assert!(
                (a[r][c] - expected[r][c]).abs() < EPS,
                "affine[{}][{}] = {} (expected {})",
                r,
                c,
                a[r][c],
                expected[r][c]
            );
        }
    }

    // pixdim (spacing) follows the rotated column norms: (3, 2, 4).
    let sp = out.spacing();
    assert!((sp[0] - 3.0).abs() < EPS, "spacing[0] = {}", sp[0]);
    assert!((sp[1] - 2.0).abs() < EPS, "spacing[1] = {}", sp[1]);
    assert!((sp[2] - 4.0).abs() < EPS, "spacing[2] = {}", sp[2]);

    // Rotation is a permutation of voxel values.
    let mut before = data;
    let mut after: Vec<f32> = out.to_f32().unwrap().iter().copied().collect();
    before.sort_by(|x, y| x.partial_cmp(y).unwrap());
    after.sort_by(|x, y| x.partial_cmp(y).unwrap());
    assert_eq!(before, after);

    // World coordinates are preserved: input voxel (1,0,0) sits at the same
    // physical location after rotation.
    let a_in = img.affine();
    let w_in = world(&a_in, [1, 0, 0]);
    // Under one 90-degree step, input (i,j,k) maps to output (j, Sa-1-i, k).
    let mapped = [0usize, shape[0] - 1 - 1, 0];
    let w_out = world(&a, mapped);
    for c in 0..3 {
        assert!(
            (w_in[c] - w_out[c]).abs() < 1e-3,
            "world axis {}: {} vs {}",
            c,
            w_in[c],
            w_out[c]
        );
    }

    // The value at that output voxel equals the input value, tying the data
    // rotation to the affine update.
    let rotated = out.to_f32().unwrap();
    assert!((rotated[[mapped[0], mapped[1], mapped[2]]] - 100.0).abs() < EPS);
}

// ---------------------------------------------------------------------------
// M2: label-aware cropping honours min_pos_samples as a minimum
// ---------------------------------------------------------------------------

#[test]
fn label_aware_crop_respects_min_pos_samples() {
    // Single positive voxel; with min_pos_samples == num_samples every returned
    // region must be the positive-centred crop.
    let shape = [8, 8, 8];
    let mut label = vec![0.0f32; 8 * 8 * 8];
    label[6 * 64 + 6 * 8 + 6] = 1.0; // voxel (6,6,6)
    let label_img = make_image(label, shape, identity_affine());
    let image = make_image(vec![0.0f32; 8 * 8 * 8], shape, identity_affine());

    let config = RandCropByPosNegLabelConfig {
        patch_size: [2, 2, 2],
        pos_neg_ratio: 1.0,
        min_pos_samples: 4,
        seed: Some(7),
        background_label: 0.0,
    };
    let regions = compute_label_aware_crop_regions(&config, &image, &label_img, 4).unwrap();
    assert_eq!(regions.len(), 4);
    for r in &regions {
        // center (6,6,6) - half_size (1,1,1) = start (5,5,5)
        assert_eq!(r.start, [5, 5, 5], "expected positive-centred crop");
    }
}

// ---------------------------------------------------------------------------
// M3: center crop clamps end and size to the volume
// ---------------------------------------------------------------------------

#[test]
fn center_crop_clamps_to_volume() {
    let img = make_image(vec![0.0f32; 4 * 4 * 4], [4, 4, 4], identity_affine());
    let region = compute_center_crop_regions([8, 8, 8], &img);
    assert_eq!(region.end, [4, 4, 4]);
    assert_eq!(region.size, [4, 4, 4]);
    assert!(region.is_valid(&[4, 4, 4]));
}

// ---------------------------------------------------------------------------
// M4: foreground detector returns None (no underflow) when nothing is found
// ---------------------------------------------------------------------------

#[test]
fn foreground_detector_empty_volume() {
    let img = make_image(vec![0.0f32; 4 * 4 * 4], [4, 4, 4], identity_affine());
    let detector = ForegroundDetector::new(0.5, 0.0, 1);
    let bbox = detector.find_foreground_bbox(&img).unwrap();
    assert!(bbox.is_none());
}

// ---------------------------------------------------------------------------
// M5: clamp validates the bounds
// ---------------------------------------------------------------------------

#[test]
fn clamp_rejects_inverted_bounds() {
    let img = make_image(x_ramp([2, 2, 2]), [2, 2, 2], identity_affine());
    assert!(clamp(&img, 5.0, 1.0).is_err());
    assert!(clamp(&img, f64::NAN, 1.0).is_err());
    assert!(clamp(&img, 0.0, 10.0).is_ok());
}

// ---------------------------------------------------------------------------
// M7: z-normalization rejects non-finite statistics
// ---------------------------------------------------------------------------

#[test]
fn z_normalization_rejects_nan() {
    let mut data = x_ramp([2, 2, 2]);
    data[0] = f32::NAN;
    let img = make_image(data, [2, 2, 2], identity_affine());
    assert!(z_normalization(&img).is_err());
}

#[test]
fn z_normalization_constant_is_finite() {
    let img = make_image(vec![5.0f32; 8], [2, 2, 2], identity_affine());
    let out = z_normalization(&img).unwrap();
    for &v in out.to_f32().unwrap().iter() {
        assert!(v.is_finite());
    }
}

// ---------------------------------------------------------------------------
// M8: random_gamma validates the gamma range
// ---------------------------------------------------------------------------

#[test]
fn random_gamma_rejects_non_positive_range() {
    let img = make_image(x_ramp([2, 2, 2]), [2, 2, 2], identity_affine());
    assert!(random_gamma(&img, Some((0.0, 2.0)), Some(1)).is_err());
    assert!(random_gamma(&img, Some((-1.0, 1.0)), Some(1)).is_err());
    assert!(random_gamma(&img, Some((2.0, 1.0)), Some(1)).is_err()); // inverted
    assert!(random_gamma(&img, Some((0.5, 2.0)), Some(1)).is_ok());
}

// ---------------------------------------------------------------------------
// Seeded augmentation reproducibility
// ---------------------------------------------------------------------------

#[test]
fn augmentation_is_reproducible_with_seed() {
    let img = make_image(
        (0..64).map(|i| i as f32).collect(),
        [4, 4, 4],
        identity_affine(),
    );

    let a = random_augment(&img, Some(2024)).unwrap();
    let b = random_augment(&img, Some(2024)).unwrap();
    assert_eq!(a.to_f32().unwrap(), b.to_f32().unwrap());

    let n1 = random_gaussian_noise(&img, Some(0.1), Some(99)).unwrap();
    let n2 = random_gaussian_noise(&img, Some(0.1), Some(99)).unwrap();
    assert_eq!(n1.to_f32().unwrap(), n2.to_f32().unwrap());
}
