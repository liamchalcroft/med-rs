//! Transform benchmarks: `cargo bench --bench transforms`.

#![allow(missing_docs)]

use criterion::{criterion_group, criterion_main, Criterion};
use medrs::transforms::{self as t, Interpolation, Orientation};
use medrs::{NiftiImage, Pipeline};
use ndarray::{ArrayD, IxDyn, ShapeBuilder};
use std::hint::black_box;

fn volume() -> NiftiImage {
    let shape = [197, 233, 189];
    let n: usize = shape.iter().product();
    let values: Vec<i16> = (0..n).map(|i| ((i * 7919) % 4000) as i16 - 1000).collect();
    let array = ArrayD::from_shape_vec(IxDyn(&shape).f(), values).unwrap();
    let affine = [
        [-1.0, 0.0, 0.0, 90.0],
        [0.0, 1.0, 0.0, -126.0],
        [0.0, 0.0, 1.0, -72.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    NiftiImage::from_array(array, affine).unwrap()
}

fn transforms(c: &mut Criterion) {
    let image = volume();
    let mut group = c.benchmark_group("transforms");
    group.sample_size(20);
    group.bench_function("z_normalization", |b| {
        b.iter(|| black_box(t::z_normalization(&image).unwrap()));
    });
    group.bench_function("resample_trilinear_2mm", |b| {
        b.iter(|| {
            black_box(t::resample_to_spacing(&image, [2.0; 3], Interpolation::Trilinear).unwrap())
        });
    });
    group.bench_function("resample_nearest_0.8mm", |b| {
        b.iter(|| {
            black_box(t::resample_to_spacing(&image, [0.8; 3], Interpolation::Nearest).unwrap())
        });
    });
    group.bench_function("reorient_lps", |b| {
        b.iter(|| black_box(t::reorient(&image, Orientation::LPS).unwrap()));
    });
    group.finish();

    let pipeline = Pipeline::new()
        .reorient(Orientation::RAS)
        .resample_to_spacing([1.5; 3], Interpolation::Trilinear)
        .crop_or_pad([128, 128, 128], 0.0)
        .clamp(-500.0, 1500.0)
        .z_normalize();
    let mut group = c.benchmark_group("preprocessing");
    group.sample_size(20);
    group.bench_function("pipeline", |b| {
        b.iter(|| black_box(pipeline.apply(&image).unwrap()));
    });
    group.bench_function("eager", |b| {
        b.iter(|| {
            let x = t::reorient(&image, Orientation::RAS).unwrap();
            let x = t::resample_to_spacing(&x, [1.5; 3], Interpolation::Trilinear).unwrap();
            let x = t::crop_or_pad(&x, [128, 128, 128], 0.0).unwrap();
            let x = t::clamp(&x, -500.0, 1500.0).unwrap();
            black_box(t::z_normalization(&x).unwrap())
        });
    });
    group.finish();
}

criterion_group!(benches, transforms);
criterion_main!(benches);
