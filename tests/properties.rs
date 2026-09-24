//! Property tests over the public API.

use medrs::nifti::{self, Affine, NiftiImage};
use medrs::transforms::{self as t, Interpolation, Orientation};
use ndarray::{s, ArrayD, IxDyn, ShapeBuilder};
use proptest::prelude::*;

fn affine() -> Affine {
    [
        [-0.9, 0.1, 0.0, 90.0],
        [0.05, 1.1, -0.1, -126.0],
        [0.0, 0.0, 2.5, -72.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

/// An image whose values are their own Fortran-order index.
fn indexed(shape: &[usize]) -> NiftiImage {
    let n: usize = shape.iter().product();
    let values: Vec<i32> = (0..n as i32).collect();
    let array = ArrayD::from_shape_vec(IxDyn(shape).f(), values).unwrap();
    NiftiImage::from_array(array, affine()).unwrap()
}

fn shapes() -> impl Strategy<Value = Vec<usize>> {
    prop_oneof![
        prop::collection::vec(1usize..12, 3),
        prop::collection::vec(1usize..8, 4),
        prop::collection::vec(1usize..16, 2),
    ]
}

fn orientations() -> impl Strategy<Value = Orientation> {
    (0usize..6, 0usize..8).prop_map(|(perm, signs)| {
        let perms = [
            [0, 1, 2],
            [0, 2, 1],
            [1, 0, 2],
            [1, 2, 0],
            [2, 0, 1],
            [2, 1, 0],
        ];
        let letters = [['R', 'L'], ['A', 'P'], ['S', 'I']];
        let code: String = perms[perm]
            .iter()
            .enumerate()
            .map(|(i, &axis)| letters[axis][(signs >> i) & 1])
            .collect();
        code.parse().unwrap()
    })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn crop_paths_agree_with_slicing(
        shape in prop::collection::vec(2usize..14, 3),
        fractions in prop::collection::vec((0.0f64..1.0, 0.0f64..1.0), 3),
        name in prop::sample::select(vec!["x.nii", "x.nii.gz", "x.hdr", "x.jvol"]),
    ) {
        let img = indexed(&shape);
        let offset: [usize; 3] = std::array::from_fn(|i| (fractions[i].0 * (shape[i] - 1) as f64) as usize);
        let size: [usize; 3] = std::array::from_fn(|i| 1 + (fractions[i].1 * (shape[i] - offset[i] - 1) as f64) as usize);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(name);
        nifti::save(&img, &path).unwrap();
        let loaded = nifti::load_cropped(&path, offset, size).unwrap();
        let eager = t::crop(&img, offset, size).unwrap();
        let full = img.to_scaled::<i32>().unwrap();
        let expected = full
            .slice(s![offset[0]..offset[0] + size[0], offset[1]..offset[1] + size[1], offset[2]..offset[2] + size[2]])
            .to_owned()
            .into_dyn();
        prop_assert_eq!(loaded.to_scaled::<i32>().unwrap(), expected.clone());
        prop_assert_eq!(eager.to_scaled::<i32>().unwrap(), expected);
        // Files store the affine in single precision.
        let (a, b) = (loaded.affine(), eager.affine());
        for i in 0..3 {
            for j in 0..4 {
                prop_assert!((a[i][j] - b[i][j]).abs() < 1e-4);
            }
        }
    }

    // Images with fewer than three axes can gain axes when reoriented, so
    // this property is checked on 3D and 4D images.
    #[test]
    fn reorientation_round_trips_exactly(
        shape in prop_oneof![prop::collection::vec(1usize..12, 3), prop::collection::vec(1usize..8, 4)],
        target in orientations(),
    ) {
        let img = indexed(&shape);
        let there = t::reorient(&img, target).unwrap();
        prop_assert_eq!(there.orientation(), target);
        let back = t::reorient(&there, img.orientation()).unwrap();
        prop_assert_eq!(back.to_scaled::<i32>().unwrap(), img.to_scaled::<i32>().unwrap());
        let (a, b) = (back.affine(), img.affine());
        for i in 0..3 {
            for j in 0..4 {
                prop_assert!((a[i][j] - b[i][j]).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn flips_and_rotations_invert(shape in shapes(), axes in prop::collection::vec(0usize..2, 1..3), k in -4i32..8) {
        let img = indexed(&shape);
        let twice = t::flip(&t::flip(&img, &axes).unwrap(), &axes).unwrap();
        prop_assert_eq!(twice.to_scaled::<i32>().unwrap(), img.to_scaled::<i32>().unwrap());
        let turned = t::rotate_90(&t::rotate_90(&img, (0, 1), k).unwrap(), (0, 1), -k).unwrap();
        prop_assert_eq!(turned.to_scaled::<i32>().unwrap(), img.to_scaled::<i32>().unwrap());
        prop_assert_eq!(turned.affine(), img.affine());
    }

    #[test]
    fn resampling_to_the_same_grid_is_identity(shape in prop::collection::vec(1usize..10, 3)) {
        let img = indexed(&shape);
        let spatial = [shape[0], shape[1], shape[2]];
        for interp in [Interpolation::Nearest, Interpolation::Trilinear] {
            let out = t::resample_to_shape(&img, spatial, interp).unwrap();
            prop_assert_eq!(out.to_f64().unwrap(), img.to_f64().unwrap());
        }
    }

    #[test]
    fn serialization_round_trips(
        shape in shapes(),
        dtype in prop::sample::select(vec!["uint8", "int16", "uint32", "int64", "float16", "bfloat16", "float64"]),
        gzip in any::<bool>(),
    ) {
        let img = indexed(&shape).with_dtype(dtype.parse().unwrap()).unwrap();
        let bytes = nifti::to_bytes(&img, gzip.then_some(1)).unwrap();
        let back = nifti::from_bytes(bytes).unwrap();
        prop_assert_eq!(back.dtype(), img.dtype());
        prop_assert_eq!(back.to_f64().unwrap(), img.to_f64().unwrap());
    }

    #[test]
    fn jvol_lossless_round_trips_any_chunking(
        shape in shapes(),
        chunk in prop::collection::vec(1usize..9, 3),
        dtype in prop::sample::select(vec!["int8", "uint16", "int32", "float32", "float64"]),
    ) {
        let img = indexed(&shape).with_dtype(dtype.parse().unwrap()).unwrap();
        let options = medrs::jvol::JvolOptions::lossless()
            .with_chunk_shape([chunk[0], chunk[1], chunk[2]])
            .unwrap();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("p.jvol");
        medrs::jvol::save(&img, &path, &options).unwrap();
        let back = medrs::load(&path).unwrap();
        prop_assert_eq!(back.shape(), img.shape());
        prop_assert_eq!(back.to_f64().unwrap(), img.to_f64().unwrap());
    }
}
