//! Integration tests for the `.jvol` volumetric compression bridge.

#![cfg(feature = "jvol")]

use medrs::jvol::{self, JvolOptions};
use medrs::nifti::{self, DataType, NiftiImage};
use ndarray::{Array4, ArrayD, IxDyn};
use tempfile::NamedTempFile;

// The jvol codec indexes arrays logically ([i, j, k]), not by physical memory
// order, so a plain standard-layout array is enough here; no F-order dance
// needed (unlike NIfTI binary I/O, which is not exercised by these tests).
fn make_array<T: Clone>(data: Vec<T>, shape: &[usize]) -> ArrayD<T> {
    ArrayD::from_shape_vec(IxDyn(shape), data).unwrap()
}

const AFFINE: [[f32; 4]; 4] = [
    [1.5, 0.0, 0.0, 10.0],
    [0.0, 1.5, 0.0, -20.0],
    [0.0, 0.0, 2.0, 5.0],
    [0.0, 0.0, 0.0, 1.0],
];

fn jvol_path() -> NamedTempFile {
    tempfile::Builder::new().suffix(".jvol").tempfile().unwrap()
}

#[test]
fn lossless_roundtrip_u8_exact() {
    let shape = vec![6, 7, 5];
    let data: Vec<u8> = (0..(6 * 7 * 5)).map(|i| (i % 256) as u8).collect();
    let arr = make_array(data, &shape);
    let img = NiftiImage::from_array(arr, AFFINE);

    let file = jvol_path();
    jvol::save(&img, file.path(), JvolOptions::lossless()).unwrap();
    let loaded = jvol::load(file.path()).unwrap();

    assert_eq!(loaded.shape(), img.shape());
    assert_eq!(loaded.dtype(), DataType::UInt8);
    let orig = img.as_array::<u8>().unwrap();
    let round = loaded.as_array::<u8>().unwrap();
    assert_eq!(orig, round);
}

#[test]
fn lossless_roundtrip_i16_exact() {
    let shape = vec![8, 6, 4];
    let data: Vec<i16> = (0..(8 * 6 * 4)).map(|i| (i as i16) * 37 - 1000).collect();
    let arr = make_array(data, &shape);
    let img = NiftiImage::from_array(arr, AFFINE);

    let file = jvol_path();
    jvol::save(&img, file.path(), JvolOptions::lossless()).unwrap();
    let loaded = jvol::load(file.path()).unwrap();

    assert_eq!(loaded.dtype(), DataType::Int16);
    let orig = img.as_array::<i16>().unwrap();
    let round = loaded.as_array::<i16>().unwrap();
    assert_eq!(orig, round);
}

#[test]
fn lossless_roundtrip_i32_exact() {
    let shape = vec![5, 5, 5];
    let data: Vec<i32> = (0..125).map(|i| i * 12345 - 500000).collect();
    let arr = make_array(data, &shape);
    let img = NiftiImage::from_array(arr, AFFINE);

    let file = jvol_path();
    jvol::save(&img, file.path(), JvolOptions::lossless()).unwrap();
    let loaded = jvol::load(file.path()).unwrap();

    assert_eq!(loaded.dtype(), DataType::Int32);
    let orig = img.as_array::<i32>().unwrap();
    let round = loaded.as_array::<i32>().unwrap();
    assert_eq!(orig, round);
}

#[test]
fn lossless_roundtrip_f32_exact() {
    let shape = vec![6, 6, 6];
    let data: Vec<f32> = (0..216).map(|i| (i as f32) * 0.125 - 13.5).collect();
    let arr = make_array(data, &shape);
    let img = NiftiImage::from_array(arr, AFFINE);

    let file = jvol_path();
    jvol::save(&img, file.path(), JvolOptions::lossless()).unwrap();
    let loaded = jvol::load(file.path()).unwrap();

    assert_eq!(loaded.dtype(), DataType::Float32);
    let orig = img.as_array::<f32>().unwrap();
    let round = loaded.as_array::<f32>().unwrap();
    assert_eq!(orig, round);
}

#[test]
fn lossless_roundtrip_f64_exact() {
    let shape = vec![4, 5, 6];
    let data: Vec<f64> = (0..120).map(|i| (i as f64) * 0.0001234567 - 5.5).collect();
    let arr = make_array(data, &shape);
    let img = NiftiImage::from_array(arr, AFFINE);

    let file = jvol_path();
    jvol::save(&img, file.path(), JvolOptions::lossless()).unwrap();
    let loaded = jvol::load(file.path()).unwrap();

    assert_eq!(loaded.dtype(), DataType::Float64);
    let orig = img.as_array::<f64>().unwrap();
    let round = loaded.as_array::<f64>().unwrap();
    assert_eq!(orig, round);
}

#[test]
fn affine_roundtrips_within_tolerance() {
    let shape = vec![4, 4, 4];
    let data: Vec<f32> = (0..64).map(|i| i as f32).collect();
    let arr = make_array(data, &shape);
    let img = NiftiImage::from_array(arr, AFFINE);

    let file = jvol_path();
    jvol::save(&img, file.path(), JvolOptions::lossless()).unwrap();
    let loaded = jvol::load(file.path()).unwrap();

    let expected = img.header().affine_f64();
    let got = loaded.header().affine_f64();
    for r in 0..4 {
        for c in 0..4 {
            assert!(
                (expected[r][c] - got[r][c]).abs() < 1e-6,
                "affine mismatch at [{r}][{c}]: {} != {}",
                expected[r][c],
                got[r][c]
            );
        }
    }
}

#[test]
fn four_dimensional_multi_channel_roundtrip() {
    let (ni, nj, nk, nc) = (5, 4, 3, 3);
    let mut arr4 = Array4::<f32>::zeros((ni, nj, nk, nc));
    for i in 0..ni {
        for j in 0..nj {
            for k in 0..nk {
                for c in 0..nc {
                    arr4[[i, j, k, c]] = (i * 1000 + j * 100 + k * 10 + c) as f32;
                }
            }
        }
    }
    let img = NiftiImage::from_array(arr4.into_dyn(), AFFINE);
    let expected_shape: [usize; 4] = (ni, nj, nk, nc).into();
    assert_eq!(img.shape(), &expected_shape);

    let file = jvol_path();
    jvol::save(&img, file.path(), JvolOptions::lossless()).unwrap();
    let loaded = jvol::load(file.path()).unwrap();

    assert_eq!(loaded.shape(), img.shape());
    let orig = img.as_array::<f32>().unwrap();
    let round = loaded.as_array::<f32>().unwrap();
    assert_eq!(orig, round);
}

#[test]
fn lossy_float_roundtrip_within_tolerance() {
    let shape = vec![32, 32, 32];
    let n = 32 * 32 * 32;
    let data: Vec<f32> = (0..n)
        .map(|i| ((i as f32) * 0.037).sin() * 500.0 + 500.0)
        .collect();
    let arr = make_array(data.clone(), &shape);
    let img = NiftiImage::from_array(arr, AFFINE);

    let file = jvol_path();
    jvol::save(&img, file.path(), JvolOptions::lossy(60)).unwrap();
    let loaded = jvol::load(file.path()).unwrap();

    let orig = img.to_f64().unwrap();
    let round = loaded.to_f64().unwrap();

    let range = data.iter().copied().fold(f32::MIN, f32::max)
        - data.iter().copied().fold(f32::MAX, f32::min);
    let mean_abs_err: f64 = orig
        .iter()
        .zip(round.iter())
        .map(|(a, b)| (a - b).abs())
        .sum::<f64>()
        / (n as f64);

    assert!(
        mean_abs_err < (range.abs() as f64) * 0.05,
        "lossy mean abs error too large: {mean_abs_err} (range {range})"
    );
}

#[test]
fn lossy_f32_decode_path_matches_reference() {
    // A lossy f32 volume decodes through the f32 fast path. Check the round trip
    // stays within a small fraction of the value range (the f32 path must not
    // degrade fidelity relative to the f64 path's tolerance).
    let shape = vec![48, 40, 36];
    let n = 48 * 40 * 36;
    let data: Vec<f32> = (0..n)
        .map(|i| {
            let x = (i % 48) as f32 / 48.0;
            let y = ((i / 48) % 40) as f32 / 40.0;
            let z = (i / (48 * 40)) as f32 / 36.0;
            ((x * 6.0).sin() + (y * 5.0).cos() + (z * 7.0).sin()) * 300.0 + 500.0
        })
        .collect();
    let arr = make_array(data.clone(), &shape);
    let img = NiftiImage::from_array(arr, AFFINE);

    let file = jvol_path();
    jvol::save(&img, file.path(), JvolOptions::lossy(70)).unwrap();
    let loaded = jvol::load(file.path()).unwrap();

    assert_eq!(loaded.dtype(), DataType::Float32);
    assert_eq!(loaded.shape(), img.shape());

    let orig = img.to_f64().unwrap();
    let round = loaded.to_f64().unwrap();
    let range = data.iter().copied().fold(f32::MIN, f32::max) as f64
        - data.iter().copied().fold(f32::MAX, f32::min) as f64;
    let mean_abs_err: f64 = orig
        .iter()
        .zip(round.iter())
        .map(|(a, b)| (a - b).abs())
        .sum::<f64>()
        / (n as f64);
    assert!(
        mean_abs_err < range * 0.02,
        "f32-path lossy mean abs error too large: {mean_abs_err} (range {range})"
    );
}

#[test]
fn load_downsampled_shape_spacing_and_fidelity() {
    let n = 64;
    let shape = vec![n, n, n];
    let data: Vec<f32> = (0..(n * n * n))
        .map(|i| {
            let x = (i % n) as f32 / n as f32;
            let y = ((i / n) % n) as f32 / n as f32;
            let z = (i / (n * n)) as f32 / n as f32;
            ((x * 6.0).sin() + (y * 5.0).cos() + (z * 7.0).sin()) * 300.0 + 500.0
        })
        .collect();
    let arr = make_array(data, &shape);
    let img = NiftiImage::from_array(arr, AFFINE);

    let file = jvol_path();
    jvol::save(&img, file.path(), JvolOptions::lossy(60)).unwrap();

    let full = jvol::load(file.path()).unwrap();
    let full_mean =
        full.to_f64().unwrap().iter().sum::<f64>() / (full.to_f64().unwrap().len() as f64);

    for factor in [2usize, 4] {
        let down = jvol::load_downsampled(file.path(), factor).unwrap();
        let expected = n / factor;
        assert_eq!(
            down.shape(),
            &[expected, expected, expected],
            "downsampled shape wrong for factor {factor}"
        );

        // Spacing scales by the factor (AFFINE spacing is 1.5, 1.5, 2.0).
        let sp = down.header().affine_f64();
        let sx = (sp[0][0] * sp[0][0] + sp[1][0] * sp[1][0] + sp[2][0] * sp[2][0]).sqrt();
        assert!(
            (sx - 1.5 * factor as f64).abs() < 1e-4,
            "factor {factor}: spacing {sx} != {}",
            1.5 * factor as f64
        );

        // Mean intensity preserved (downsampled is a low-pass approximation).
        let dvec = down.to_f64().unwrap();
        let down_mean = dvec.iter().sum::<f64>() / (dvec.len() as f64);
        assert!(
            (down_mean - full_mean).abs() < 0.05 * full_mean.abs().max(1.0),
            "factor {factor}: downsampled mean {down_mean} vs full {full_mean}"
        );
    }
}

#[test]
fn load_downsampled_factor_one_equals_full_load() {
    let shape = vec![16, 16, 16];
    let data: Vec<f32> = (0..(16 * 16 * 16))
        .map(|i| (i as f32 * 0.1).sin() * 50.0)
        .collect();
    let arr = make_array(data, &shape);
    let img = NiftiImage::from_array(arr, AFFINE);

    let file = jvol_path();
    jvol::save(&img, file.path(), JvolOptions::lossy(60)).unwrap();

    let full = jvol::load(file.path()).unwrap();
    let same = jvol::load_downsampled(file.path(), 1).unwrap();
    assert_eq!(full.shape(), same.shape());
}

#[test]
fn load_downsampled_rejects_lossless_and_bad_factor() {
    let shape = vec![16, 16, 16];
    let data: Vec<f32> = (0..(16 * 16 * 16)).map(|i| i as f32).collect();
    let arr = make_array(data, &shape);
    let img = NiftiImage::from_array(arr, AFFINE);

    let file = jvol_path();
    jvol::save(&img, file.path(), JvolOptions::lossless()).unwrap();
    assert!(
        jvol::load_downsampled(file.path(), 2).is_err(),
        "downsampled load of a lossless file must be rejected"
    );

    let lossy = jvol_path();
    jvol::save(&img, lossy.path(), JvolOptions::lossy(60)).unwrap();
    assert!(
        jvol::load_downsampled(lossy.path(), 3).is_err(),
        "non-power-of-two factor must be rejected"
    );
}

#[test]
fn lossy_on_integer_dtype_is_rejected() {
    let shape = vec![4, 4, 4];
    let data: Vec<i16> = (0..64)
        .collect::<Vec<_>>()
        .iter()
        .map(|&v| v as i16)
        .collect();
    let arr = make_array(data, &shape);
    let img = NiftiImage::from_array(arr, AFFINE);

    let file = jvol_path();
    let result = jvol::save(&img, file.path(), JvolOptions::lossy(60));
    assert!(
        result.is_err(),
        "lossy save of integer dtype must be rejected"
    );
}

#[test]
fn unsupported_dtype_falls_back_to_f64_lossless() {
    use half::bf16;

    let shape = vec![4, 4, 4];
    let data: Vec<bf16> = (0..64)
        .map(|i| bf16::from_f32((i as f32) * 0.5 - 10.0))
        .collect();
    let arr = make_array(data, &shape);
    let img = NiftiImage::from_array(arr, AFFINE);
    assert_eq!(img.dtype(), DataType::BFloat16);

    let file = jvol_path();
    jvol::save(&img, file.path(), JvolOptions::lossless()).unwrap();
    let loaded = jvol::load(file.path()).unwrap();

    assert_eq!(loaded.dtype(), DataType::BFloat16);
    let orig = img.as_array::<bf16>().unwrap();
    let round = loaded.as_array::<bf16>().unwrap();
    assert_eq!(orig, round);
}

// ============================================================================
// Enhancement 1: decode-as-dtype (output dtype override)
// ============================================================================

#[test]
fn decode_as_dtype_bf16_matches_f32_reference_within_rounding() {
    use half::bf16;

    let shape = vec![24, 20, 16];
    let n = 24 * 20 * 16;
    let data: Vec<f32> = (0..n)
        .map(|i| {
            let x = (i % 24) as f32 / 24.0;
            let y = ((i / 24) % 20) as f32 / 20.0;
            let z = (i / (24 * 20)) as f32 / 16.0;
            ((x * 6.0).sin() + (y * 5.0).cos() + (z * 7.0).sin()) * 300.0 + 500.0
        })
        .collect();
    let arr = make_array(data.clone(), &shape);
    let img = NiftiImage::from_array(arr, AFFINE);

    let file = jvol_path();
    jvol::save(&img, file.path(), JvolOptions::lossy(70)).unwrap();

    let reference = jvol::load(file.path()).unwrap();
    let bf16_decoded = jvol::load_as(file.path(), Some(DataType::BFloat16)).unwrap();
    assert_eq!(bf16_decoded.dtype(), DataType::BFloat16);
    assert_eq!(bf16_decoded.shape(), reference.shape());

    let ref_f64 = reference.to_f64().unwrap();
    let got_f64 = bf16_decoded.to_f64().unwrap();
    let range = data.iter().copied().fold(f32::MIN, f32::max) as f64
        - data.iter().copied().fold(f32::MAX, f32::min) as f64;

    // bf16 has ~8 bits of mantissa (relative error up to ~1/256 per value); the
    // wavelet-decoded f32 reference re-quantized to bf16 should stay within a
    // small multiple of that per-value tolerance in the mean.
    let mean_abs_err: f64 = ref_f64
        .iter()
        .zip(got_f64.iter())
        .map(|(a, b)| (a - b).abs())
        .sum::<f64>()
        / (n as f64);
    assert!(
        mean_abs_err < range * 0.01,
        "bf16 override mean abs error too large: {mean_abs_err} (range {range})"
    );

    // Sanity: the bf16 values are actually bf16-representable (round trip
    // through bf16 is a no-op).
    let arr = bf16_decoded.as_array::<bf16>().unwrap();
    for &v in arr.iter() {
        assert_eq!(bf16::from_f32(v.to_f32()), v);
    }
}

#[test]
fn decode_as_dtype_f32_override_of_f16_source() {
    use half::f16;

    let shape = vec![20, 18, 16];
    let n = 20 * 18 * 16;
    let data: Vec<f16> = (0..n)
        .map(|i| {
            let x = (i % 20) as f32 / 20.0;
            let y = ((i / 20) % 18) as f32 / 18.0;
            let z = (i / (20 * 18)) as f32 / 16.0;
            f16::from_f32(((x * 4.0).sin() + (y * 3.0).cos() + (z * 5.0).sin()) * 200.0 + 400.0)
        })
        .collect();
    let f64_reference: Vec<f64> = data.iter().map(|v| v.to_f64()).collect();
    let arr = make_array(data, &shape);
    let img = NiftiImage::from_array(arr, AFFINE);
    assert_eq!(img.dtype(), DataType::Float16);

    let file = jvol_path();
    jvol::save(&img, file.path(), JvolOptions::lossy(70)).unwrap();

    let f32_override = jvol::load_as(file.path(), Some(DataType::Float32)).unwrap();
    assert_eq!(f32_override.dtype(), DataType::Float32);
    assert_eq!(f32_override.shape(), img.shape());

    let got = f32_override.to_f64().unwrap();
    let range = f64_reference.iter().copied().fold(f64::MIN, f64::max)
        - f64_reference.iter().copied().fold(f64::MAX, f64::min);
    let mean_abs_err: f64 = f64_reference
        .iter()
        .zip(got.iter())
        .map(|(a, b)| (a - b).abs())
        .sum::<f64>()
        / (n as f64);
    assert!(
        mean_abs_err < range * 0.05,
        "f32 override of f16 source mean abs error too large: {mean_abs_err} (range {range})"
    );
}

#[test]
fn decode_as_dtype_integer_override_rounds() {
    // Lossless so the decode is exact; only the integer-cast rounding is under
    // test.
    let shape = vec![2, 2, 2];
    let data: Vec<f32> = vec![2.4, 2.6, -3.5, -3.4, 0.5, -0.5, 10.499, 10.501];
    let arr = make_array(data.clone(), &shape);
    let img = NiftiImage::from_array(arr, AFFINE);

    let file = jvol_path();
    jvol::save(&img, file.path(), JvolOptions::lossless()).unwrap();

    let rounded = jvol::load_as(file.path(), Some(DataType::Int16)).unwrap();
    assert_eq!(rounded.dtype(), DataType::Int16);
    let got = rounded.as_array::<i16>().unwrap();

    for (v, &expected_f32) in got.iter().zip(data.iter()) {
        let expected = expected_f32.round() as i16;
        assert_eq!(
            *v, expected,
            "rounding mismatch for source value {expected_f32}"
        );
    }
}

#[test]
fn decode_downsampled_as_dtype_overrides_output() {
    let n = 32;
    let shape = vec![n, n, n];
    let data: Vec<f32> = (0..(n * n * n))
        .map(|i| {
            let x = (i % n) as f32 / n as f32;
            let y = ((i / n) % n) as f32 / n as f32;
            let z = (i / (n * n)) as f32 / n as f32;
            ((x * 6.0).sin() + (y * 5.0).cos() + (z * 7.0).sin()) * 300.0 + 500.0
        })
        .collect();
    let arr = make_array(data, &shape);
    let img = NiftiImage::from_array(arr, AFFINE);

    let file = jvol_path();
    jvol::save(&img, file.path(), JvolOptions::lossy(60)).unwrap();

    let down = jvol::load_downsampled_as(file.path(), 2, Some(DataType::BFloat16)).unwrap();
    assert_eq!(down.dtype(), DataType::BFloat16);
    assert_eq!(down.shape(), &[n / 2, n / 2, n / 2]);
}

// ============================================================================
// Enhancement 2: decoded-image cache
// ============================================================================

// The jvol decoded-image cache is process-global (a static behind a RwLock),
// so tests that mutate its size must not run concurrently with each other:
// cargo test runs tests in parallel threads within one process by default.
static CACHE_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[test]
fn jvol_cache_two_loads_are_equal() {
    let _guard = CACHE_TEST_LOCK.lock().unwrap();
    jvol::set_jvol_cache_size(10);
    jvol::clear_jvol_cache();

    let shape = vec![10, 10, 10];
    let data: Vec<f32> = (0..1000).map(|i| i as f32 * 0.37).collect();
    let arr = make_array(data, &shape);
    let img = NiftiImage::from_array(arr, AFFINE);

    let file = jvol_path();
    jvol::save(&img, file.path(), JvolOptions::lossy(60)).unwrap();

    let first = jvol::load_cached(file.path(), None).unwrap();
    let second = jvol::load_cached(file.path(), None).unwrap();

    assert_eq!(first.shape(), second.shape());
    let a = first.to_f64().unwrap();
    let b = second.to_f64().unwrap();
    assert_eq!(a, b);

    jvol::clear_jvol_cache();
}

#[test]
fn jvol_cache_shared_matches_fresh_exactly() {
    // The cache re-backs the decoded volume with shared byte storage. On a
    // lossless volume with an asymmetric shape and distinct per-voxel values,
    // the cached (shared) result must equal a fresh decode exactly; a layout or
    // endianness mistake in the shared conversion would transpose or corrupt it.
    let _guard = CACHE_TEST_LOCK.lock().unwrap();
    jvol::set_jvol_cache_size(10);
    jvol::clear_jvol_cache();

    let shape = vec![7, 11, 13];
    let data: Vec<f32> = (0..7 * 11 * 13).map(|i| i as f32).collect();
    let img = NiftiImage::from_array(make_array(data, &shape), AFFINE);

    let file = jvol_path();
    jvol::save(&img, file.path(), JvolOptions::lossless()).unwrap();

    let fresh = jvol::load(file.path()).unwrap();
    let cached = jvol::load_cached(file.path(), None).unwrap();

    assert_eq!(fresh.shape(), cached.shape());
    assert_eq!(fresh.to_f64().unwrap(), cached.to_f64().unwrap());

    jvol::clear_jvol_cache();
}

#[test]
fn jvol_cache_size_zero_disables_caching() {
    let _guard = CACHE_TEST_LOCK.lock().unwrap();
    jvol::clear_jvol_cache();
    jvol::set_jvol_cache_size(0);

    let file = jvol_path();
    let shape_a = vec![6, 6, 6];
    let data_a: Vec<f32> = (0..216).map(|i| i as f32).collect();
    let img_a = NiftiImage::from_array(make_array(data_a, &shape_a), AFFINE);
    jvol::save(&img_a, file.path(), JvolOptions::lossless()).unwrap();
    let loaded_a = jvol::load_cached(file.path(), None).unwrap();
    assert_eq!(loaded_a.shape(), &[6, 6, 6]);

    // Overwrite the same path with a different shape. With caching disabled
    // (max_entries == 0, so insert() is a no-op), this must always be a fresh
    // decode regardless of file stamp granularity.
    let shape_b = vec![8, 8, 8];
    let data_b: Vec<f32> = (0..512).map(|i| i as f32 * 2.0).collect();
    let img_b = NiftiImage::from_array(make_array(data_b, &shape_b), AFFINE);
    jvol::save(&img_b, file.path(), JvolOptions::lossless()).unwrap();
    let loaded_b = jvol::load_cached(file.path(), None).unwrap();
    assert_eq!(
        loaded_b.shape(),
        &[8, 8, 8],
        "size-0 cache must not serve stale data from a disabled cache"
    );

    jvol::set_jvol_cache_size(10);
}

#[test]
fn jvol_cache_clear_empties_and_stays_functional() {
    let _guard = CACHE_TEST_LOCK.lock().unwrap();
    jvol::set_jvol_cache_size(10);
    jvol::clear_jvol_cache();

    let file = jvol_path();
    let shape = vec![6, 6, 6];
    let data: Vec<f32> = (0..216).map(|i| i as f32 * 1.5).collect();
    let img = NiftiImage::from_array(make_array(data, &shape), AFFINE);
    jvol::save(&img, file.path(), JvolOptions::lossless()).unwrap();

    let cached = jvol::load_cached(file.path(), None).unwrap();
    jvol::clear_jvol_cache();

    // After clear, a subsequent load must still succeed and reflect correct
    // (freshly decoded) content, i.e. clear() does not corrupt cache state.
    let after_clear = jvol::load_cached(file.path(), None).unwrap();
    assert_eq!(
        cached.to_f64().unwrap(),
        after_clear.to_f64().unwrap(),
        "content must still be correct after clear + reload"
    );

    jvol::clear_jvol_cache();
}

#[test]
fn jvol_cache_changed_file_invalidates() {
    let _guard = CACHE_TEST_LOCK.lock().unwrap();
    jvol::set_jvol_cache_size(10);
    jvol::clear_jvol_cache();

    let file = jvol_path();

    let shape_a = vec![5, 5, 5];
    let data_a: Vec<f32> = (0..125).map(|i| i as f32).collect();
    let img_a = NiftiImage::from_array(make_array(data_a, &shape_a), AFFINE);
    jvol::save(&img_a, file.path(), JvolOptions::lossless()).unwrap();
    let loaded_a = jvol::load_cached(file.path(), None).unwrap();
    assert_eq!(loaded_a.shape(), &[5, 5, 5]);

    // Different shape guarantees a different serialized file length, so the
    // cache's file-length stamp component catches the change even if the
    // filesystem's mtime resolution is too coarse to differ.
    let shape_b = vec![9, 9, 9];
    let data_b: Vec<f32> = (0..729).map(|i| i as f32 * 3.0).collect();
    let img_b = NiftiImage::from_array(make_array(data_b, &shape_b), AFFINE);
    jvol::save(&img_b, file.path(), JvolOptions::lossless()).unwrap();

    let loaded_b = jvol::load_cached(file.path(), None).unwrap();
    assert_eq!(
        loaded_b.shape(),
        &[9, 9, 9],
        "cache must invalidate on a changed file and return the new content"
    );

    jvol::clear_jvol_cache();
}

#[test]
fn jvol_cache_key_includes_output_dtype() {
    let _guard = CACHE_TEST_LOCK.lock().unwrap();
    jvol::set_jvol_cache_size(10);
    jvol::clear_jvol_cache();

    let shape = vec![6, 6, 6];
    let data: Vec<f32> = (0..216).map(|i| i as f32 * 0.5).collect();
    let img = NiftiImage::from_array(make_array(data, &shape), AFFINE);

    let file = jvol_path();
    jvol::save(&img, file.path(), JvolOptions::lossy(60)).unwrap();

    let as_f32 = jvol::load_cached(file.path(), None).unwrap();
    let as_bf16 = jvol::load_cached(file.path(), Some(DataType::BFloat16)).unwrap();
    assert_eq!(as_f32.dtype(), DataType::Float32);
    assert_eq!(as_bf16.dtype(), DataType::BFloat16);

    jvol::clear_jvol_cache();
}

// ============================================================================
// Enhancement 3: decode-once-to-mmap transcoding
// ============================================================================

#[test]
fn transcode_to_nii_roundtrips_and_is_zero_copy() {
    let shape = vec![24, 20, 16];
    let n = 24 * 20 * 16;
    let data: Vec<f32> = (0..n)
        .map(|i| {
            let x = (i % 24) as f32 / 24.0;
            let y = ((i / 24) % 20) as f32 / 20.0;
            let z = (i / (24 * 20)) as f32 / 16.0;
            ((x * 6.0).sin() + (y * 5.0).cos() + (z * 7.0).sin()) * 300.0 + 500.0
        })
        .collect();
    let arr = make_array(data, &shape);
    let img = NiftiImage::from_array(arr, AFFINE);

    let jvol_file = jvol_path();
    jvol::save(&img, jvol_file.path(), JvolOptions::lossy(70)).unwrap();

    let nii_dir = tempfile::tempdir().unwrap();
    let nii_path = nii_dir.path().join("transcoded.nii");
    jvol::transcode_to_nii(jvol_file.path(), &nii_path, Some(DataType::Float32)).unwrap();

    let reference = jvol::load_as(jvol_file.path(), Some(DataType::Float32)).unwrap();
    let loaded = nifti::load(&nii_path).unwrap();

    assert_eq!(loaded.dtype(), DataType::Float32);
    assert_eq!(loaded.shape(), reference.shape());
    assert!(
        loaded.can_zero_copy(),
        "a transcoded uncompressed f32 .nii should be zero-copy mmap-able"
    );

    let ref_data = reference.to_f64().unwrap();
    let got_data = loaded.to_f64().unwrap();
    assert_eq!(ref_data, got_data);
}

#[test]
fn transcode_to_nii_rejects_compressed_output_path() {
    let shape = vec![4, 4, 4];
    let data: Vec<f32> = (0..64).map(|i| i as f32).collect();
    let img = NiftiImage::from_array(make_array(data, &shape), AFFINE);

    let jvol_file = jvol_path();
    jvol::save(&img, jvol_file.path(), JvolOptions::lossless()).unwrap();

    let dir = tempfile::tempdir().unwrap();
    assert!(jvol::transcode_to_nii(jvol_file.path(), dir.path().join("out.nii.gz"), None).is_err());
    assert!(jvol::transcode_to_nii(jvol_file.path(), dir.path().join("out.jvol"), None).is_err());
}

#[test]
fn load_via_mmap_cache_transcodes_once_and_reloads() {
    let shape = vec![16, 14, 12];
    let n = 16 * 14 * 12;
    let data: Vec<f32> = (0..n).map(|i| (i as f32 * 0.13).sin() * 100.0).collect();
    let img = NiftiImage::from_array(make_array(data, &shape), AFFINE);

    let jvol_file = jvol_path();
    jvol::save(&img, jvol_file.path(), JvolOptions::lossy(60)).unwrap();

    let cache_dir = tempfile::tempdir().unwrap();
    let first = jvol::load_via_mmap_cache(jvol_file.path(), cache_dir.path(), None).unwrap();
    assert!(first.can_zero_copy());

    let second = jvol::load_via_mmap_cache(jvol_file.path(), cache_dir.path(), None).unwrap();
    assert_eq!(first.to_f64().unwrap(), second.to_f64().unwrap());

    // A dtype override produces a distinct cached .nii rather than colliding
    // with the default-dtype cache entry.
    let bf16 =
        jvol::load_via_mmap_cache(jvol_file.path(), cache_dir.path(), Some(DataType::BFloat16))
            .unwrap();
    assert_eq!(bf16.dtype(), DataType::BFloat16);
    assert_eq!(second.dtype(), DataType::Float32);
}

#[test]
fn dot_jvol_extension_dispatches_through_nifti_load_save() {
    let shape = vec![4, 4, 4];
    let data: Vec<f32> = (0..64).map(|i| i as f32).collect();
    let arr = make_array(data, &shape);
    let img = NiftiImage::from_array(arr, AFFINE);

    let file = jvol_path();
    nifti::save(&img, file.path()).unwrap();
    let loaded = nifti::load(file.path()).unwrap();

    assert_eq!(loaded.shape(), img.shape());
    let orig = img.as_array::<f32>().unwrap();
    let round = loaded.as_array::<f32>().unwrap();
    assert_eq!(orig, round);
}
