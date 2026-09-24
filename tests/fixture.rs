//! End-to-end checks on a real MRI volume.

use medrs::transforms::{self as t, Interpolation, Orientation};
use medrs::{FastLoader, LoaderConfig, Pipeline};
use std::path::PathBuf;

fn fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/mprage_img.nii.gz")
}

#[test]
fn formats_agree_on_a_real_volume() {
    let image = medrs::load(fixture()).unwrap();
    let values = image.to_f32().unwrap();
    let dir = tempfile::tempdir().unwrap();
    for name in ["m.nii", "m.nii.gz", "m.hdr", "m.jvol"] {
        let path = dir.path().join(name);
        medrs::save(&image, &path).unwrap();
        let back = medrs::load(&path).unwrap();
        assert_eq!(back.to_f32().unwrap(), values, "{name}");
        assert_eq!(back.affine(), image.affine(), "{name}");
    }
    let mgzip = dir.path().join("m.mgz.nii.gz");
    medrs::nifti::save_mgzip(&image, &mgzip).unwrap();
    assert!(medrs::nifti::is_mgzip(&mgzip).unwrap());
    assert_eq!(medrs::load(&mgzip).unwrap().to_f32().unwrap(), values);
}

#[test]
fn preprocessing_pipeline_on_a_real_volume() {
    let image = medrs::load(fixture()).unwrap();
    let pipeline = Pipeline::new()
        .reorient(Orientation::LPS)
        .resample_to_spacing([2.0, 2.0, 2.0], Interpolation::Trilinear)
        .crop_or_pad([96, 96, 96], 0.0)
        .z_normalize();
    let out = pipeline.apply(&image).unwrap();
    assert_eq!(out.shape(), &[96, 96, 96]);
    assert_eq!(out.orientation(), Orientation::LPS);
    let eager = t::reorient(&image, Orientation::LPS).unwrap();
    let eager = t::resample_to_spacing(&eager, [2.0; 3], Interpolation::Trilinear).unwrap();
    let eager = t::crop_or_pad(&eager, [96, 96, 96], 0.0).unwrap();
    let eager = t::z_normalization(&eager).unwrap();
    let (a, b) = (out.to_f32().unwrap(), eager.to_f32().unwrap());
    let max = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max);
    assert!(max < 1e-4, "{max}");
}

#[test]
fn loader_reads_patches_from_a_real_volume() {
    let mut config = LoaderConfig::new([64, 64, 64]);
    config.patches_per_volume = 4;
    config.seed = Some(0);
    config.pipeline = Some(Pipeline::new().z_normalize().random_flip(&[0, 1, 2], 0.5));
    let loader = FastLoader::new(vec![fixture(); 3], None, config).unwrap();
    let patches: Vec<_> = loader.epoch(0).collect::<Result<_, _>>().unwrap();
    assert_eq!(patches.len(), 12);
    assert!(patches.iter().all(|p| p.image.shape() == [64, 64, 64]));
}
