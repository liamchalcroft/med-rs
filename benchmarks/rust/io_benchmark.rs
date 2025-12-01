//! Benchmarks for NIfTI I/O operations.
//!
//! Tests various volume sizes (32^3 to 512^3) and data types (u8, i16, i32, f16, bf16, f32, f64).

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use half::{bf16, f16};
use medrs::nifti::{self, NiftiImage};
use ndarray::ArrayD;
use tempfile::tempdir;

/// Volume sizes to benchmark (edge length of cubic volume)
const SIZES: &[usize] = &[32, 64, 128, 256];

/// Create a test image with u8 data
fn create_u8_image(size: usize) -> NiftiImage {
    let data: Vec<u8> = (0..size.pow(3)).map(|i| (i % 256) as u8).collect();
    let array = ArrayD::from_shape_vec(vec![size, size, size], data).unwrap();
    let affine = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    NiftiImage::from_array(array, affine)
}

/// Create a test image with i16 data
fn create_i16_image(size: usize) -> NiftiImage {
    let data: Vec<i16> = (0..size.pow(3)).map(|i| (i % 32768) as i16).collect();
    let array = ArrayD::from_shape_vec(vec![size, size, size], data).unwrap();
    let affine = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    NiftiImage::from_array(array, affine)
}

/// Create a test image with i32 data
fn create_i32_image(size: usize) -> NiftiImage {
    let data: Vec<i32> = (0..size.pow(3)).map(|i| i as i32).collect();
    let array = ArrayD::from_shape_vec(vec![size, size, size], data).unwrap();
    let affine = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    NiftiImage::from_array(array, affine)
}

/// Create a test image with f16 data
fn create_f16_image(size: usize) -> NiftiImage {
    let data: Vec<f16> = (0..size.pow(3))
        .map(|i| f16::from_f32((i % 256) as f32))
        .collect();
    let array = ArrayD::from_shape_vec(vec![size, size, size], data).unwrap();
    let affine = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    NiftiImage::from_array(array, affine)
}

/// Create a test image with bf16 data
fn create_bf16_image(size: usize) -> NiftiImage {
    let data: Vec<bf16> = (0..size.pow(3))
        .map(|i| bf16::from_f32((i % 256) as f32))
        .collect();
    let array = ArrayD::from_shape_vec(vec![size, size, size], data).unwrap();
    let affine = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    NiftiImage::from_array(array, affine)
}

/// Create a test image with f32 data
fn create_f32_image(size: usize) -> NiftiImage {
    let data: Vec<f32> = (0..size.pow(3)).map(|i| (i % 256) as f32).collect();
    let array = ArrayD::from_shape_vec(vec![size, size, size], data).unwrap();
    let affine = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    NiftiImage::from_array(array, affine)
}

/// Create a test image with f64 data
fn create_f64_image(size: usize) -> NiftiImage {
    let data: Vec<f64> = (0..size.pow(3)).map(|i| (i % 256) as f64).collect();
    let array = ArrayD::from_shape_vec(vec![size, size, size], data).unwrap();
    let affine = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    NiftiImage::from_array(array, affine)
}

// ============================================================================
// Save benchmarks by data type
// ============================================================================

fn bench_save_by_dtype(c: &mut Criterion) {
    let dir = tempdir().unwrap();
    let mut group = c.benchmark_group("save_by_dtype");
    let size = 128usize; // Use 128^3 for dtype comparison

    // u8
    let img = create_u8_image(size);
    let path = dir.path().join("test_u8.nii");
    group.throughput(Throughput::Bytes((size.pow(3)) as u64));
    group.bench_function("u8_128", |b| {
        b.iter(|| nifti::save(black_box(&img), black_box(&path)).unwrap());
    });

    // i16
    let img = create_i16_image(size);
    let path = dir.path().join("test_i16.nii");
    group.throughput(Throughput::Bytes((size.pow(3) * 2) as u64));
    group.bench_function("i16_128", |b| {
        b.iter(|| nifti::save(black_box(&img), black_box(&path)).unwrap());
    });

    // i32
    let img = create_i32_image(size);
    let path = dir.path().join("test_i32.nii");
    group.throughput(Throughput::Bytes((size.pow(3) * 4) as u64));
    group.bench_function("i32_128", |b| {
        b.iter(|| nifti::save(black_box(&img), black_box(&path)).unwrap());
    });

    // f16
    let img = create_f16_image(size);
    let path = dir.path().join("test_f16.nii");
    group.throughput(Throughput::Bytes((size.pow(3) * 2) as u64));
    group.bench_function("f16_128", |b| {
        b.iter(|| nifti::save(black_box(&img), black_box(&path)).unwrap());
    });

    // bf16
    let img = create_bf16_image(size);
    let path = dir.path().join("test_bf16.nii");
    group.throughput(Throughput::Bytes((size.pow(3) * 2) as u64));
    group.bench_function("bf16_128", |b| {
        b.iter(|| nifti::save(black_box(&img), black_box(&path)).unwrap());
    });

    // f32
    let img = create_f32_image(size);
    let path = dir.path().join("test_f32.nii");
    group.throughput(Throughput::Bytes((size.pow(3) * 4) as u64));
    group.bench_function("f32_128", |b| {
        b.iter(|| nifti::save(black_box(&img), black_box(&path)).unwrap());
    });

    // f64
    let img = create_f64_image(size);
    let path = dir.path().join("test_f64.nii");
    group.throughput(Throughput::Bytes((size.pow(3) * 8) as u64));
    group.bench_function("f64_128", |b| {
        b.iter(|| nifti::save(black_box(&img), black_box(&path)).unwrap());
    });

    group.finish();
}

fn bench_load_by_dtype(c: &mut Criterion) {
    let dir = tempdir().unwrap();
    let mut group = c.benchmark_group("load_by_dtype");
    let size = 128usize;

    // u8
    let img = create_u8_image(size);
    let path = dir.path().join("test_u8.nii");
    nifti::save(&img, &path).unwrap();
    group.throughput(Throughput::Bytes((size.pow(3)) as u64));
    group.bench_function("u8_128", |b| {
        b.iter(|| {
            let img = nifti::load(black_box(&path)).unwrap();
            black_box(img);
        });
    });

    // i16
    let img = create_i16_image(size);
    let path = dir.path().join("test_i16.nii");
    nifti::save(&img, &path).unwrap();
    group.throughput(Throughput::Bytes((size.pow(3) * 2) as u64));
    group.bench_function("i16_128", |b| {
        b.iter(|| {
            let img = nifti::load(black_box(&path)).unwrap();
            black_box(img);
        });
    });

    // i32
    let img = create_i32_image(size);
    let path = dir.path().join("test_i32.nii");
    nifti::save(&img, &path).unwrap();
    group.throughput(Throughput::Bytes((size.pow(3) * 4) as u64));
    group.bench_function("i32_128", |b| {
        b.iter(|| {
            let img = nifti::load(black_box(&path)).unwrap();
            black_box(img);
        });
    });

    // f16
    let img = create_f16_image(size);
    let path = dir.path().join("test_f16.nii");
    nifti::save(&img, &path).unwrap();
    group.throughput(Throughput::Bytes((size.pow(3) * 2) as u64));
    group.bench_function("f16_128", |b| {
        b.iter(|| {
            let img = nifti::load(black_box(&path)).unwrap();
            black_box(img);
        });
    });

    // bf16
    let img = create_bf16_image(size);
    let path = dir.path().join("test_bf16.nii");
    nifti::save(&img, &path).unwrap();
    group.throughput(Throughput::Bytes((size.pow(3) * 2) as u64));
    group.bench_function("bf16_128", |b| {
        b.iter(|| {
            let img = nifti::load(black_box(&path)).unwrap();
            black_box(img);
        });
    });

    // f32
    let img = create_f32_image(size);
    let path = dir.path().join("test_f32.nii");
    nifti::save(&img, &path).unwrap();
    group.throughput(Throughput::Bytes((size.pow(3) * 4) as u64));
    group.bench_function("f32_128", |b| {
        b.iter(|| {
            let img = nifti::load(black_box(&path)).unwrap();
            black_box(img);
        });
    });

    // f64
    let img = create_f64_image(size);
    let path = dir.path().join("test_f64.nii");
    nifti::save(&img, &path).unwrap();
    group.throughput(Throughput::Bytes((size.pow(3) * 8) as u64));
    group.bench_function("f64_128", |b| {
        b.iter(|| {
            let img = nifti::load(black_box(&path)).unwrap();
            black_box(img);
        });
    });

    group.finish();
}

// ============================================================================
// Save benchmarks by volume size (f32)
// ============================================================================

fn bench_save_by_size_uncompressed(c: &mut Criterion) {
    let dir = tempdir().unwrap();
    let mut group = c.benchmark_group("save_uncompressed");

    for &size in SIZES {
        let img = create_f32_image(size);
        let path = dir.path().join(format!("test_{}.nii", size));
        let bytes = (size.pow(3) * 4) as u64;

        group.throughput(Throughput::Bytes(bytes));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}x{}x{}", size, size, size)),
            &size,
            |b, _| {
                b.iter(|| nifti::save(black_box(&img), black_box(&path)).unwrap());
            },
        );
    }

    group.finish();
}

fn bench_save_by_size_gzipped(c: &mut Criterion) {
    let dir = tempdir().unwrap();
    let mut group = c.benchmark_group("save_gzipped");
    group.sample_size(10); // Fewer samples for slow gzip operations

    for &size in &[32, 64, 128] {
        let img = create_f32_image(size);
        let path = dir.path().join(format!("test_{}.nii.gz", size));
        let bytes = (size.pow(3) * 4) as u64;

        group.throughput(Throughput::Bytes(bytes));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}x{}x{}", size, size, size)),
            &size,
            |b, _| {
                b.iter(|| nifti::save(black_box(&img), black_box(&path)).unwrap());
            },
        );
    }

    group.finish();
}

// ============================================================================
// Load benchmarks by volume size (f32)
// ============================================================================

fn bench_load_by_size_uncompressed(c: &mut Criterion) {
    let dir = tempdir().unwrap();
    let mut group = c.benchmark_group("load_uncompressed");

    for &size in SIZES {
        let img = create_f32_image(size);
        let path = dir.path().join(format!("test_{}.nii", size));
        nifti::save(&img, &path).unwrap();
        let bytes = (size.pow(3) * 4) as u64;

        group.throughput(Throughput::Bytes(bytes));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}x{}x{}", size, size, size)),
            &path,
            |b, path| {
                b.iter(|| {
                    let img = nifti::load(black_box(path)).unwrap();
                    black_box(img);
                });
            },
        );
    }

    group.finish();
}

fn bench_load_by_size_gzipped(c: &mut Criterion) {
    let dir = tempdir().unwrap();
    let mut group = c.benchmark_group("load_gzipped");
    group.sample_size(10);

    for &size in &[32, 64, 128] {
        let img = create_f32_image(size);
        let path = dir.path().join(format!("test_{}.nii.gz", size));
        nifti::save(&img, &path).unwrap();
        let bytes = (size.pow(3) * 4) as u64;

        group.throughput(Throughput::Bytes(bytes));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}x{}x{}", size, size, size)),
            &path,
            |b, path| {
                b.iter(|| {
                    let img = nifti::load(black_box(path)).unwrap();
                    black_box(img);
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// Data conversion benchmarks
// ============================================================================

fn bench_to_f32_conversion(c: &mut Criterion) {
    let mut group = c.benchmark_group("to_f32");

    for &size in SIZES {
        // From u8
        let img = create_u8_image(size);
        group.bench_function(format!("from_u8_{}x{}x{}", size, size, size), |b| {
            b.iter(|| {
                let data = black_box(&img).to_f32();
                black_box(data);
            });
        });

        // From i16
        let img = create_i16_image(size);
        group.bench_function(format!("from_i16_{}x{}x{}", size, size, size), |b| {
            b.iter(|| {
                let data = black_box(&img).to_f32();
                black_box(data);
            });
        });

        // From f16
        let img = create_f16_image(size);
        group.bench_function(format!("from_f16_{}x{}x{}", size, size, size), |b| {
            b.iter(|| {
                let data = black_box(&img).to_f32();
                black_box(data);
            });
        });

        // From bf16
        let img = create_bf16_image(size);
        group.bench_function(format!("from_bf16_{}x{}x{}", size, size, size), |b| {
            b.iter(|| {
                let data = black_box(&img).to_f32();
                black_box(data);
            });
        });

        // From f32 (should be fastest - just clone)
        let img = create_f32_image(size);
        group.bench_function(format!("from_f32_{}x{}x{}", size, size, size), |b| {
            b.iter(|| {
                let data = black_box(&img).to_f32();
                black_box(data);
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_save_by_dtype,
    bench_load_by_dtype,
    bench_save_by_size_uncompressed,
    bench_save_by_size_gzipped,
    bench_load_by_size_uncompressed,
    bench_load_by_size_gzipped,
    bench_to_f32_conversion,
);

criterion_main!(benches);
