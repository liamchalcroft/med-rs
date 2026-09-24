//! I/O benchmarks: `cargo bench --bench io`.

#![allow(missing_docs)]

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use medrs::jvol::{self, JvolOptions};
use medrs::nifti::{self, NiftiImage};
use ndarray::{ArrayD, IxDyn, ShapeBuilder};
use std::hint::black_box;
use std::path::PathBuf;
use tempfile::TempDir;

const SHAPE: [usize; 3] = [197, 233, 189];

/// A smooth head-like phantom with noise and a zero background, so that
/// compressed formats see realistic redundancy.
fn phantom() -> NiftiImage {
    let [nx, ny, nz] = SHAPE;
    let mut seed = 0x2545_f491_4f6c_dd1du64;
    let mut values = Vec::with_capacity(nx * ny * nz);
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                let (dx, dy, dz) = (
                    x as f32 / nx as f32 - 0.5,
                    y as f32 / ny as f32 - 0.5,
                    z as f32 / nz as f32 - 0.5,
                );
                let inside = dx * dx + dy * dy + dz * dz < 0.2;
                let v = if inside {
                    300.0
                        + 80.0 * (x as f32 * 0.07).sin() * (y as f32 * 0.05).cos()
                        + (seed % 16) as f32
                } else {
                    0.0
                };
                values.push(v);
            }
        }
    }
    let array = ArrayD::from_shape_vec(IxDyn(&SHAPE).f(), values).unwrap();
    NiftiImage::from_array(
        array,
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
    )
    .unwrap()
}

struct Files {
    _dir: TempDir,
    nii: PathBuf,
    gz: PathBuf,
    mgzip: PathBuf,
    jvol: PathBuf,
}

fn files(image: &NiftiImage) -> Files {
    let dir = TempDir::new().unwrap();
    let path = |name: &str| dir.path().join(name);
    let (nii, gz, mgzip, jvol) = (
        path("a.nii"),
        path("a.nii.gz"),
        path("a.mgz.nii.gz"),
        path("a.jvol"),
    );
    nifti::save(image, &nii).unwrap();
    nifti::save(image, &gz).unwrap();
    nifti::save_mgzip(image, &mgzip).unwrap();
    jvol::save(image, &jvol, &JvolOptions::lossless()).unwrap();
    Files {
        _dir: dir,
        nii,
        gz,
        mgzip,
        jvol,
    }
}

fn load(c: &mut Criterion) {
    let image = phantom();
    let f = files(&image);
    let mut group = c.benchmark_group("load_to_f32");
    group.sample_size(20);
    group.throughput(Throughput::Bytes((image.len() * 4) as u64));
    for (name, path) in [
        ("nii", &f.nii),
        ("nii.gz", &f.gz),
        ("mgzip", &f.mgzip),
        ("jvol", &f.jvol),
    ] {
        group.bench_with_input(BenchmarkId::from_parameter(name), path, |b, p| {
            b.iter(|| black_box(nifti::load(p).unwrap().to_f32().unwrap()));
        });
    }
    group.finish();

    let mut group = c.benchmark_group("load_cropped_64");
    group.sample_size(30);
    for (name, path) in [("nii", &f.nii), ("jvol", &f.jvol)] {
        group.bench_with_input(BenchmarkId::from_parameter(name), path, |b, p| {
            b.iter(|| black_box(nifti::load_cropped(p, [60, 70, 50], [64, 64, 64]).unwrap()));
        });
    }
    group.finish();
}

fn save(c: &mut Criterion) {
    let image = phantom();
    let dir = TempDir::new().unwrap();
    let mut group = c.benchmark_group("save");
    group.sample_size(10);
    group.throughput(Throughput::Bytes((image.len() * 4) as u64));
    group.bench_function("nii.gz", |b| {
        b.iter(|| nifti::save(&image, dir.path().join("s.nii.gz")).unwrap());
    });
    group.bench_function("mgzip", |b| {
        b.iter(|| nifti::save_mgzip(&image, dir.path().join("s.nii.gz")).unwrap());
    });
    group.bench_function("jvol", |b| {
        b.iter(|| jvol::save(&image, dir.path().join("s.jvol"), &JvolOptions::lossless()).unwrap());
    });
    group.finish();
}

criterion_group!(benches, load, save);
criterion_main!(benches);
