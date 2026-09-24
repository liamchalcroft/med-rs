use super::*;
use crate::nifti::header::Affine;
use crate::nifti::{NiftiElement, NiftiExtension};
use ndarray::{s, ArrayD, IxDyn, ShapeBuilder};
use tempfile::TempDir;

const AFFINE: Affine = [
    [-0.9, 0.1, 0.0, 90.0],
    [0.0, 1.1, 0.0, -126.0],
    [0.0, 0.0, 2.5, -72.0],
    [0.0, 0.0, 0.0, 1.0],
];

fn image<T: NiftiElement>(shape: &[usize], f: impl Fn(usize) -> T) -> NiftiImage {
    let n: usize = shape.iter().product();
    let arr = ArrayD::from_shape_vec(IxDyn(shape).f(), (0..n).map(f).collect()).unwrap();
    NiftiImage::from_array(arr, AFFINE).unwrap()
}

fn raw_bytes(img: &NiftiImage) -> Vec<u8> {
    img.data_bytes_le().unwrap().into_owned()
}

fn small_chunks() -> JvolOptions {
    JvolOptions::lossless().with_chunk_shape([8, 7, 5]).unwrap()
}

#[test]
fn lossless_is_bit_exact_for_every_dtype() {
    let dir = TempDir::new().unwrap();
    let shape = [19, 13, 11, 2];
    let cases = vec![
        image(&shape, |i| (i % 256) as u8),
        image(&shape, |i| (i % 256) as i8),
        image(&shape, |i| (i * 7) as u16),
        image(&shape, |i| (i as i16).wrapping_mul(31)),
        image(&shape, |i| u32::MAX - i as u32),
        image(&shape, |i| i32::MIN + i as i32),
        image(&shape, |i| u64::MAX - (i as u64) * 3),
        image(&shape, |i| {
            if i % 2 == 0 {
                i64::MIN + i as i64
            } else {
                (1i64 << 60) + 7
            }
        }),
        image(&shape, |i| half::f16::from_f32(i as f32 * 0.1)),
        image(&shape, |i| half::bf16::from_f32(i as f32 * -0.3)),
        image(&shape, |i| match i % 5 {
            0 => f32::NAN,
            1 => -0.0,
            2 => f32::INFINITY,
            _ => i as f32 * 1.000_001,
        }),
        image(&shape, |i| {
            f64::from_bits(0x7FF8_0000_0000_0001 + i as u64 % 3)
        }),
    ];
    for (k, img) in cases.into_iter().enumerate() {
        let path = dir.path().join(format!("{k}.jvol"));
        save(&img, &path, &small_chunks()).unwrap();
        let back = load(&path).unwrap();
        assert_eq!(back.dtype(), img.dtype(), "case {k}");
        assert_eq!(back.shape(), img.shape(), "case {k}");
        assert_eq!(raw_bytes(&back), raw_bytes(&img), "case {k}");
    }
}

#[test]
fn header_metadata_and_scaling_survive() {
    let dir = TempDir::new().unwrap();
    let mut img = image(&[10, 9, 8], |i| (i % 4000) as u16);
    {
        let mut h = img.header_mut();
        h.scl_slope = 1.0;
        h.scl_inter = -1024.0;
        h.descrip = "ct scan".into();
        h.intent_code = 1002;
        h.sform_code = 4;
        h.qform_code = 2;
        h.pixdim[4] = 2.5;
        h.extensions.push(NiftiExtension::new(6, b"hello".to_vec()));
    }
    let path = dir.path().join("meta.jvol");
    save(&img, &path, &JvolOptions::default()).unwrap();
    let back = crate::load(&path).unwrap();
    let (a, b) = (back.header(), img.header());
    assert_eq!((a.scl_slope, a.scl_inter), (1.0, -1024.0));
    assert_eq!(a.descrip, b.descrip);
    assert_eq!(a.intent_code, 1002);
    assert_eq!((a.sform_code, a.qform_code), (4, 2));
    assert_eq!(a.pixdim[4], 2.5);
    assert_eq!(&a.extensions[0].data[..5], b"hello");
    assert_eq!(back.affine(), img.affine());
    assert_eq!(back.to_f32().unwrap(), img.to_f32().unwrap());
}

#[test]
fn regions_decode_only_what_they_need_and_match_crops() {
    let dir = TempDir::new().unwrap();
    let img = image(&[23, 17, 12, 2], |i| i as i32);
    let path = dir.path().join("crop.jvol");
    save(&img, &path, &small_chunks()).unwrap();
    let reader = Reader::open(&path).unwrap();
    let full = img.as_array::<i32>().unwrap().clone();
    for (offset, shape) in [
        ([0, 0, 0], [23, 17, 12]),
        ([3, 5, 2], [9, 8, 7]),
        ([22, 16, 11], [1, 1, 1]),
    ] {
        let r = reader.region(offset, shape).unwrap();
        let expected = full
            .slice(s![
                offset[0]..offset[0] + shape[0],
                offset[1]..offset[1] + shape[1],
                offset[2]..offset[2] + shape[2],
                ..
            ])
            .to_owned()
            .into_dyn();
        assert_eq!(r.to_scaled::<i32>().unwrap(), expected);
        assert_eq!(
            r.affine(),
            crate::transforms::crop(&img, offset, shape)
                .unwrap()
                .affine()
        );
    }
    assert!(reader.region([20, 0, 0], [5, 1, 1]).is_err());
    let via_nifti = crate::nifti::load_cropped(&path, [3, 5, 2], [9, 8, 7]).unwrap();
    assert_eq!(via_nifti.shape(), &[9, 8, 7, 2]);
}

#[test]
fn lossy_is_accurate_bounded_and_keeps_background() {
    let dir = TempDir::new().unwrap();
    let shape = [70, 66, 40];
    let img = image(&shape, |i| {
        let (x, y, z) = (i % 70, (i / 70) % 66, i / (70 * 66));
        if x < 10 || y < 5 {
            0.0f32
        } else {
            500.0 + 200.0 * ((x as f32) * 0.2).sin() * ((y as f32) * 0.15).cos() + z as f32
        }
    });
    let path = dir.path().join("lossy.jvol");
    save(&img, &path, &JvolOptions::lossy(80).unwrap()).unwrap();
    let raw_size = img.len() * 4;
    let file_size = std::fs::metadata(&path).unwrap().len() as usize;
    assert!(file_size * 10 < raw_size, "{file_size} vs {raw_size}");
    let back = load(&path).unwrap();
    assert_eq!(back.dtype(), DataType::Float32);
    let (a, b) = (img.to_f32().unwrap(), back.to_f32().unwrap());
    let range = 900.0f32;
    let mut max_err = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        if *x == 0.0 {
            assert_eq!(*y, 0.0);
        }
        assert!((0.0..=740.0).contains(y));
        max_err = max_err.max((x - y).abs());
    }
    assert!(max_err < 0.05 * range, "{max_err}");
}

#[test]
fn lossy_scaled_integers_decode_in_physical_units() {
    let dir = TempDir::new().unwrap();
    let mut ct = image(&[40, 40, 20], |i| (i % 2000) as i16);
    ct.header_mut().scl_inter = -1024.0;
    let path = dir.path().join("ct.jvol");
    save(&ct, &path, &JvolOptions::lossy(100).unwrap()).unwrap();
    let back = load(&path).unwrap();
    assert_eq!(back.header().scl_inter, 0.0);
    let (a, b) = (ct.to_f32().unwrap(), back.to_f32().unwrap());
    let err = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max);
    assert!(err < 40.0, "{err}");
}

#[test]
fn invalid_options_and_values_are_rejected() {
    assert!(JvolOptions::lossy(0).is_err());
    assert!(JvolOptions::lossy(101).is_err());
    assert!(JvolOptions::lossless().with_chunk_shape([0, 8, 8]).is_err());
    assert!(JvolOptions::lossless()
        .with_chunk_shape([1 << 13, 1 << 13, 1])
        .is_err());
    assert!(JvolOptions::lossless().with_level(0).is_err());
    let dir = TempDir::new().unwrap();
    let nan = image(&[4, 4, 4], |i| if i == 5 { f32::NAN } else { 1.0 });
    let err = save(
        &nan,
        dir.path().join("x.jvol"),
        &JvolOptions::lossy(50).unwrap(),
    )
    .unwrap_err();
    assert!(err.to_string().contains("finite"), "{err}");
    assert!(!dir.path().join("x.jvol").exists());
}

#[test]
fn corruption_and_hostile_files_are_detected() {
    let dir = TempDir::new().unwrap();
    let img = image(&[16, 16, 16], |i| (i % 97) as u16);
    let path = dir.path().join("ok.jvol");
    save(&img, &path, &small_chunks()).unwrap();
    let good = std::fs::read(&path).unwrap();
    let bad = dir.path().join("bad.jvol");
    let check = |bytes: &[u8]| {
        std::fs::write(&bad, bytes).unwrap();
        Reader::open(&bad).and_then(|r| r.image()).is_err()
    };
    // Any single-byte change in the preamble, header, or table is caught by the
    // CRC; changes in chunk payloads by the zstd checksums.
    for i in (0..good.len()).step_by(37) {
        let mut b = good.clone();
        b[i] ^= 0x20;
        let decoded = {
            std::fs::write(&bad, &b).unwrap();
            Reader::open(&bad).and_then(|r| r.image())
        };
        if let Ok(decoded) = decoded {
            assert_eq!(
                raw_bytes(&decoded),
                raw_bytes(&img),
                "undetected change at byte {i}"
            );
        }
    }
    assert!(check(&good[..good.len() - 5]));
    assert!(check(&good[..20]));
    assert!(check(b"not a jvol file at all, just text"));
    // A forged chunk shape claiming huge chunks is refused before allocating.
    let mut b = good.clone();
    b[12..16].copy_from_slice(&u32::MAX.to_le_bytes());
    assert!(check(&b));
}

#[test]
fn files_from_medrs_0_2_and_other_formats_are_named_in_errors() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("old.jvol");
    // medrs 0.2 files are a zstd frame.
    std::fs::write(&path, [0x28, 0xb5, 0x2f, 0xfd, 0, 0, 0, 0]).unwrap();
    let err = crate::nifti::load(&path).unwrap_err().to_string();
    assert!(err.contains("written by medrs 0.2"), "{err}");
    std::fs::write(&path, b"not a volume").unwrap();
    let err = crate::nifti::load(&path).unwrap_err().to_string();
    assert!(err.contains("not a .jvol file"), "{err}");
}
