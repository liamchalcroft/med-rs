"""Reading and writing, checked against nibabel."""

import gzip
import os
import pickle
import shutil

import nibabel as nib
import numpy as np
import pytest

import medrs

from conftest import OBLIQUE, ramp

DTYPES = [
    np.uint8,
    np.int8,
    np.uint16,
    np.int16,
    np.uint32,
    np.int32,
    np.int64,
    np.float32,
    np.float64,
]


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("name", ["a.nii", "a.nii.gz"])
def test_reads_what_nibabel_writes(write_nib, dtype, name):
    data = ramp((7, 6, 5), dtype)
    path = write_nib(data, name=name)
    img = medrs.load(path)
    ref = nib.load(path)
    assert img.dtype == np.dtype(dtype).name
    np.testing.assert_array_equal(img.to_numpy(), np.asanyarray(ref.dataobj))
    np.testing.assert_allclose(img.affine, ref.affine, atol=1e-6)


def test_scaling_matches_get_fdata(write_nib):
    data = ramp((5, 4, 3), np.int16)
    path = write_nib(data, scl_slope=0.5, scl_inter=-10.0)
    img = medrs.load(path)
    ref = nib.load(path)
    np.testing.assert_allclose(img.to_numpy(), ref.get_fdata(), rtol=1e-6)
    assert img.to_numpy().dtype == np.float32
    np.testing.assert_array_equal(img.to_numpy("int16"), np.rint(ref.get_fdata()).astype(np.int16))


def test_nan_slope_means_no_scaling(write_nib, tmp_path):
    path = write_nib(ramp((4, 4, 4), np.int16), name="nan.nii")
    raw = bytearray(path.read_bytes())
    raw[112:116] = np.float32(np.nan).tobytes()
    path.write_bytes(bytes(raw))
    np.testing.assert_array_equal(medrs.load(path).to_numpy(), ramp((4, 4, 4), np.int16))


def test_big_endian_nifti2_with_extensions(tmp_path):
    data = ramp((6, 5, 4), np.float32)
    header = nib.Nifti2Header(endianness=">")
    img = nib.Nifti2Image(data, OBLIQUE, header=header)
    img.header.extensions.append(nib.nifti1.Nifti1Extension("comment", b"hello medrs"))
    img.header["descrip"] = b"big endian"
    path = tmp_path / "be.nii"
    nib.save(img, path)
    loaded = medrs.load(path)
    np.testing.assert_array_equal(loaded.to_numpy(), data)
    np.testing.assert_allclose(loaded.affine, OBLIQUE)
    h = loaded.header
    assert h["version"] == "nifti2"
    assert h["descrip"] == "big endian"
    assert h["extensions"][0][0] == 6
    assert h["extensions"][0][1].startswith(b"hello medrs")


@pytest.mark.parametrize("name", ["out.nii", "out.nii.gz", "out.hdr", "out.img.gz"])
def test_nibabel_reads_what_medrs_writes(tmp_path, name):
    data = ramp((9, 8, 7, 2), np.int16)
    img = medrs.NiftiImage(data, OBLIQUE).with_header(
        descrip="written by medrs", intent_code=1002, sform_code=4, qform_code=2, scl_slope=2.0
    )
    path = tmp_path / name
    img.save(path)
    ref = nib.load(path)
    np.testing.assert_array_equal(np.asanyarray(ref.dataobj), data * 2.0)
    np.testing.assert_allclose(ref.affine, OBLIQUE, atol=1e-5)
    # A sheared affine is stored in the qform as its closest rigid + zoom
    # approximation; nibabel computes the same one.
    expected = nib.Nifti1Header()
    expected.set_qform(OBLIQUE)
    np.testing.assert_allclose(ref.get_qform(), expected.get_qform(), atol=1e-5)
    assert ref.header["descrip"].item().decode() == "written by medrs"
    assert int(ref.header["intent_code"]) == 1002
    assert (int(ref.header["sform_code"]), int(ref.header["qform_code"])) == (4, 2)


def test_every_header_field_round_trips(tmp_path):
    img = medrs.NiftiImage(ramp((4, 3, 2, 5), np.uint8), OBLIQUE).with_header(
        descrip="d",
        aux_file="aux",
        intent_code=3,
        intent_name="ttest",
        intent_p=(12.0, 0.5, 0.0),
        cal_min=1.0,
        cal_max=9.0,
        dim_info=57,
        slice_code=1,
        slice_start=0,
        slice_end=1,
        slice_duration=0.25,
        toffset=1.5,
        temporal_units="s",
        extensions=[(6, b"note"), (4, b"afni")],
    )
    for name in ["a.nii", "a.nii.gz", "a.hdr", "a.jvol"]:
        img.save(tmp_path / name)
        back = medrs.load(tmp_path / name).header
        for key, value in img.header.items():
            if key == "extensions":
                assert [(c, d.rstrip(b"\0")) for c, d in back[key]] == [(6, b"note"), (4, b"afni")]
            elif isinstance(value, np.ndarray):
                # NIfTI-1 stores the qform quaternion in float32.
                np.testing.assert_allclose(back[key], value, atol=1e-4)
            elif isinstance(value, float):
                assert back[key] == pytest.approx(value, rel=1e-6), key
            elif isinstance(value, tuple) and value and isinstance(value[0], float):
                np.testing.assert_allclose(back[key], value, rtol=1e-6)
            else:
                assert back[key] == value, key


def test_load_header_matches_load(write_nib):
    path = write_nib(ramp((12, 10, 8), np.int16), name="h.nii.gz")
    h = medrs.load_header(path)
    assert h["shape"] == (12, 10, 8)
    assert h["dtype"] == "int16"
    np.testing.assert_allclose(h["affine"], medrs.load(path).affine)


@pytest.mark.parametrize("name", ["c.nii", "c.nii.gz", "c.jvol"])
def test_load_cropped_matches_nibabel_slicer(tmp_path, name):
    data = ramp((20, 18, 16, 2), np.float32)
    medrs.NiftiImage(data, OBLIQUE).save(tmp_path / name)
    crop = medrs.load_cropped(tmp_path / name, (3, 4, 5), (8, 7, 6))
    np.testing.assert_array_equal(crop.to_numpy(), data[3:11, 4:11, 5:11])
    expected = nib.Nifti1Image(data, OBLIQUE).slicer[3:11, 4:11, 5:11]
    np.testing.assert_allclose(crop.affine, expected.affine, atol=1e-5)
    with pytest.raises(ValueError, match="crop"):
        medrs.load_cropped(tmp_path / name, (15, 0, 0), (8, 1, 1))


def test_gzip_is_detected_by_content(tmp_path, write_nib):
    path = write_nib(ramp((5, 5, 5)), name="real.nii.gz")
    misnamed = tmp_path / "misnamed.nii"
    shutil.copy(path, misnamed)
    np.testing.assert_array_equal(medrs.load(misnamed).to_numpy(), ramp((5, 5, 5)))


def test_mgzip_is_standard_gzip(tmp_path):
    data = ramp((64, 64, 80), np.float32)
    path = tmp_path / "block.nii.gz"
    medrs.NiftiImage(data).save(path, mgzip=True)
    with gzip.open(path) as f:
        assert len(f.read()) > data.nbytes
    np.testing.assert_array_equal(np.asanyarray(nib.load(path).dataobj), data)
    np.testing.assert_array_equal(medrs.load(path).to_numpy(), data)


def test_saving_over_the_mapped_source(tmp_path):
    path = tmp_path / "same.nii"
    medrs.NiftiImage(ramp((32, 32, 32))).save(path)
    mapped = medrs.load(path)
    view = mapped.to_numpy()
    mapped.with_header(descrip="edited").save(path)
    np.testing.assert_array_equal(view, ramp((32, 32, 32)))
    assert medrs.load(path).header["descrip"] == "edited"
    assert sorted(os.listdir(tmp_path)) == ["same.nii"]


def test_bare_relative_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    medrs.NiftiImage(ramp((2, 2, 2))).save("bare.nii.gz")
    assert medrs.load("bare.nii.gz").shape == (2, 2, 2)


def test_load_multi_aligns_to_the_reference(tmp_path):
    fine = medrs.NiftiImage(ramp((20, 20, 20)), np.diag([1.0, 1.0, 1.0, 1.0]))
    coarse_affine = np.diag([2.0, 2.0, 2.0, 1.0])
    coarse_affine[:3, 3] = 0.5
    label = medrs.NiftiImage((ramp((10, 10, 10)) % 3).astype(np.uint8), coarse_affine)
    fine.save(tmp_path / "t1.nii")
    label.save(tmp_path / "seg.nii")
    same = medrs.NiftiImage(ramp((20, 20, 20), np.int16))
    same.save(tmp_path / "t2.nii")
    t1, seg, t2 = medrs.load_multi(
        [tmp_path / "t1.nii", tmp_path / "seg.nii", tmp_path / "t2.nii"],
        interpolation=["trilinear", "nearest", "trilinear"],
    )
    assert seg.shape == t1.shape == t2.shape
    np.testing.assert_allclose(seg.affine, t1.affine)
    assert seg.dtype == "uint8"
    assert t2.dtype == "int16"  # already on the grid: returned as loaded


def test_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        medrs.load(tmp_path / "missing.nii")
    (tmp_path / "junk.nii").write_bytes(b"x" * 400)
    with pytest.raises(medrs.FormatError):
        medrs.load(tmp_path / "junk.nii")
    medrs.NiftiImage(ramp((8, 8, 8))).save(tmp_path / "t.nii")
    raw = (tmp_path / "t.nii").read_bytes()
    (tmp_path / "t.nii").write_bytes(raw[:-10])
    with pytest.raises(medrs.FormatError, match="truncated"):
        medrs.load(tmp_path / "t.nii")
    with pytest.raises(ValueError, match="quality"):
        medrs.NiftiImage(ramp((2, 2, 2))).save(tmp_path / "x.nii", quality=50)


def test_pickle_round_trip():
    img = medrs.NiftiImage(ramp((3, 4, 5), np.uint16), OBLIQUE).with_header(descrip="pickled")
    back = pickle.loads(pickle.dumps(img))
    np.testing.assert_array_equal(back.to_numpy(), img.to_numpy())
    np.testing.assert_allclose(back.affine, img.affine)
    assert back.header["descrip"] == "pickled"


def test_real_volume_matches_nibabel(fixture_path):
    img = medrs.load(fixture_path)
    ref = nib.load(fixture_path)
    np.testing.assert_array_equal(img.to_numpy(), np.asanyarray(ref.dataobj))
    np.testing.assert_allclose(img.affine, ref.affine, atol=1e-6)
