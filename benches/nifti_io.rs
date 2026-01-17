//! Criterion benchmarks for medrs NIfTI I/O operations.
//!
//! Run with: cargo bench --bench nifti_io
//!
//! These benchmarks track regression in core I/O performance:
//! - load() for uncompressed .nii files
//! - load() for compressed .nii.gz files  
//! - load_cropped() for byte-exact loading
//! - to_f32() materialization
//! - save() for uncompressed and compressed

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use medrs::nifti::{self, NiftiImage};
use ndarray::{ArrayD, IxDyn, ShapeBuilder};
use std::fs;
use tempfile::NamedTempFile;

/// Create a test NIfTI image with given shape
fn create_test_image(shape: &[usize]) -> NiftiImage {
    let numel: usize = shape.iter().product();
    let data: Vec<f32> = (0..numel).map(|i| (i % 256) as f32).collect();
    let c_order = ArrayD::from_shape_vec(shape.to_vec(), data).unwrap();
    let mut f_order = ArrayD::zeros(IxDyn(shape).f());
    f_order.assign(&c_order);
    let affine = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    NiftiImage::from_array(f_order, affine)
}

/// Benchmark load() for uncompressed .nii files
fn bench_load_uncompressed(c: &mut Criterion) {
    let mut group = c.benchmark_group("load_uncompressed");

    // Test different volume sizes
    for &shape in &[
        [64, 64, 64],
        [128, 128, 64],
        [197, 233, 189], // Typical MPRAGE
        [256, 256, 256],
    ] {
        let img = create_test_image(&shape);
        let file = NamedTempFile::new().unwrap();
        let path = file.path().to_str().unwrap().to_string() + ".nii";
        nifti::save(&img, &path).unwrap();

        let size_mb = shape.iter().product::<usize>() * 4 / (1024 * 1024);
        let label = format!("{}x{}x{} ({} MB)", shape[0], shape[1], shape[2], size_mb);

        group.throughput(Throughput::Bytes(
            (shape.iter().product::<usize>() * 4) as u64,
        ));
        group.bench_with_input(BenchmarkId::new("mmap", &label), &path, |b, path| {
            b.iter(|| {
                let img = nifti::load(black_box(path)).unwrap();
                black_box(img)
            })
        });

        // Also benchmark load + materialize
        group.bench_with_input(
            BenchmarkId::new("mmap+materialize", &label),
            &path,
            |b, path| {
                b.iter(|| {
                    let img = nifti::load(black_box(path)).unwrap();
                    let data = img.to_f32().unwrap();
                    black_box(data)
                })
            },
        );

        // Cleanup
        let _ = fs::remove_file(&path);
    }

    group.finish();
}

/// Benchmark load() for compressed .nii.gz files
fn bench_load_compressed(c: &mut Criterion) {
    let mut group = c.benchmark_group("load_compressed");

    for &shape in &[
        [64, 64, 64],
        [128, 128, 64],
        [197, 233, 189], // Typical MPRAGE
    ] {
        let img = create_test_image(&shape);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.nii.gz");
        let path_str = path.to_str().unwrap().to_string();
        nifti::save(&img, &path_str).unwrap();

        let size_mb = shape.iter().product::<usize>() * 4 / (1024 * 1024);
        let compressed_size = fs::metadata(&path).unwrap().len();
        let label = format!(
            "{}x{}x{} ({} MB -> {} KB)",
            shape[0],
            shape[1],
            shape[2],
            size_mb,
            compressed_size / 1024
        );

        group.throughput(Throughput::Bytes(
            (shape.iter().product::<usize>() * 4) as u64,
        ));
        group.bench_with_input(BenchmarkId::new("decompress", &label), &path_str, |b, p| {
            b.iter(|| {
                let img = nifti::load(black_box(p)).unwrap();
                black_box(img)
            })
        });
    }

    group.finish();
}

/// Benchmark load_cropped() for byte-exact loading
fn bench_load_cropped(c: &mut Criterion) {
    let mut group = c.benchmark_group("load_cropped");

    // Create a large volume to crop from
    let shape = [256, 256, 256];
    let img = create_test_image(&shape);
    let file = NamedTempFile::new().unwrap();
    let path = file.path().to_str().unwrap().to_string() + ".nii";
    nifti::save(&img, &path).unwrap();

    for &crop_size in &[32, 64, 96, 128] {
        let crop_shape = [crop_size, crop_size, crop_size];
        let offset = [64, 64, 64]; // Centered crop

        let size_mb = crop_shape.iter().product::<usize>() * 4 / (1024 * 1024);
        let label = format!("{}^3 ({} MB)", crop_size, size_mb);

        group.throughput(Throughput::Bytes(
            (crop_shape.iter().product::<usize>() * 4) as u64,
        ));
        group.bench_with_input(BenchmarkId::new("byte_exact", &label), &path, |b, path| {
            b.iter(|| {
                let img = nifti::load_cropped(black_box(path), offset, crop_shape).unwrap();
                black_box(img)
            })
        });
    }

    // Cleanup
    let _ = fs::remove_file(&path);
    group.finish();
}

/// Benchmark to_f32() materialization
fn bench_to_f32(c: &mut Criterion) {
    let mut group = c.benchmark_group("to_f32");

    for &shape in &[[64, 64, 64], [128, 128, 64], [197, 233, 189]] {
        let img = create_test_image(&shape);
        let file = NamedTempFile::new().unwrap();
        let path = file.path().to_str().unwrap().to_string() + ".nii";
        nifti::save(&img, &path).unwrap();

        // Load once (mmap'd)
        let loaded = nifti::load(&path).unwrap();

        let size_mb = shape.iter().product::<usize>() * 4 / (1024 * 1024);
        let label = format!("{}x{}x{} ({} MB)", shape[0], shape[1], shape[2], size_mb);

        group.throughput(Throughput::Bytes(
            (shape.iter().product::<usize>() * 4) as u64,
        ));
        group.bench_with_input(
            BenchmarkId::new("materialize", &label),
            &loaded,
            |b, img| {
                b.iter(|| {
                    let data = img.to_f32().unwrap();
                    black_box(data)
                })
            },
        );

        let _ = fs::remove_file(&path);
    }

    group.finish();
}

/// Benchmark save() operations
fn bench_save(c: &mut Criterion) {
    let mut group = c.benchmark_group("save");

    for &shape in &[[64, 64, 64], [128, 128, 64]] {
        let img = create_test_image(&shape);
        let size_mb = shape.iter().product::<usize>() * 4 / (1024 * 1024);
        let label = format!("{}x{}x{} ({} MB)", shape[0], shape[1], shape[2], size_mb);

        group.throughput(Throughput::Bytes(
            (shape.iter().product::<usize>() * 4) as u64,
        ));

        // Uncompressed save
        group.bench_with_input(BenchmarkId::new("uncompressed", &label), &img, |b, img| {
            b.iter(|| {
                let file = NamedTempFile::new().unwrap();
                let path = file.path().to_str().unwrap().to_string() + ".nii";
                nifti::save(black_box(img), &path).unwrap();
                black_box(path)
            })
        });

        // Compressed save
        group.bench_with_input(BenchmarkId::new("compressed", &label), &img, |b, img| {
            b.iter(|| {
                let dir = tempfile::tempdir().unwrap();
                let path = dir.path().join("test.nii.gz");
                let path_str = path.to_str().unwrap();
                nifti::save(black_box(img), path_str).unwrap();
                black_box(path_str.to_string())
            })
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_load_uncompressed,
    bench_load_compressed,
    bench_load_cropped,
    bench_to_f32,
    bench_save
);
criterion_main!(benches);
