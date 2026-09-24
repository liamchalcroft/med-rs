"""Conversions between images and NumPy, PyTorch, and JAX arrays."""

import nibabel as nib
import numpy as np
import pytest

import medrs

from conftest import OBLIQUE, ramp


@pytest.mark.parametrize(
    "dtype",
    [
        np.bool_,
        np.uint8,
        np.int8,
        np.uint16,
        np.int16,
        np.uint32,
        np.int32,
        np.uint64,
        np.int64,
        np.float16,
        np.float32,
        np.float64,
    ],
)
def test_constructor_keeps_values_and_dtype(dtype):
    data = (ramp((5, 4, 3)) % 2 if dtype is np.bool_ else ramp((5, 4, 3))).astype(dtype)
    img = medrs.NiftiImage(data, OBLIQUE)
    expected = "uint8" if dtype is np.bool_ else np.dtype(dtype).name
    assert img.dtype == expected
    np.testing.assert_array_equal(img.to_numpy(), data)
    np.testing.assert_allclose(img.affine, OBLIQUE)


@pytest.mark.parametrize(
    "make",
    [
        lambda a: a,
        lambda a: np.asfortranarray(a),
        lambda a: a[::-1, :, ::2],
        lambda a: a.astype(">f4"),
        lambda a: a.tolist(),
    ],
    ids=["c-order", "f-order", "strided", "big-endian", "list"],
)
def test_constructor_accepts_any_layout(make):
    data = ramp((6, 5, 8))
    expected = np.asarray(make(data), dtype=np.float32)
    np.testing.assert_array_equal(medrs.NiftiImage(make(data)).to_numpy(), expected)


def test_views_are_read_only_and_copies_are_writable(tmp_path):
    data = ramp((8, 7, 6))
    medrs.NiftiImage(data).save(tmp_path / "v.nii")
    img = medrs.load(tmp_path / "v.nii")
    view = img.to_numpy()
    assert not view.flags.writeable
    assert view.flags.f_contiguous
    copy = img.to_numpy(copy=True)
    copy[0, 0, 0] = -1
    assert img.to_numpy()[0, 0, 0] == 0


def test_numpy_array_protocol():
    img = medrs.NiftiImage(ramp((3, 3, 3), np.int16))
    np.testing.assert_array_equal(np.asarray(img), ramp((3, 3, 3), np.int16))
    assert np.asarray(img, dtype=np.float64).dtype == np.float64
    assert np.array(img).flags.writeable
    scaled = img.with_header(scl_slope=2.0)
    with pytest.raises(ValueError, match="copy"):
        np.asarray(scaled, copy=False)


@pytest.mark.parametrize("dtype", [np.int32, np.uint32, np.float32, np.float64])
@pytest.mark.parametrize("endian", ["<", ">"])
def test_file_values_for_every_byte_order(tmp_path, dtype, endian):
    data = ramp((6, 5, 4), dtype) * 3 + 1
    header = nib.Nifti1Header(endianness=endian)
    path = tmp_path / "e.nii"
    nib.save(nib.Nifti1Image(data, OBLIQUE, header=header, dtype=dtype), path)
    img = medrs.load(path)
    np.testing.assert_array_equal(img.to_numpy(), data)
    np.testing.assert_array_equal(img.to_numpy("float64"), data.astype(np.float64))


def test_float16_and_bfloat16():
    half = medrs.NiftiImage(ramp((4, 4, 4), np.float16))
    assert half.to_numpy().dtype == np.float16
    bf = half.with_dtype("bfloat16")
    assert bf.dtype == "bfloat16"
    assert bf.to_numpy().dtype == np.float32
    np.testing.assert_array_equal(bf.to_numpy(), ramp((4, 4, 4)))
    ml_dtypes = pytest.importorskip("ml_dtypes")
    arr = bf.to_numpy(ml_dtypes.bfloat16)
    assert arr.dtype == ml_dtypes.bfloat16
    back = medrs.NiftiImage(arr)
    assert back.dtype == "bfloat16"


def test_with_dtype_rounds_half_to_even_and_saturates():
    img = medrs.NiftiImage(np.array([-1.5, -0.5, 0.5, 1.5, 2.5, 300.0, np.nan], np.float32))
    np.testing.assert_array_equal(img.with_dtype("uint8").to_numpy(), [0, 0, 0, 2, 2, 255, 0])
    with pytest.raises(ValueError, match="unknown data type"):
        img.with_dtype("complex64")


def test_jax_conversion():
    jax = pytest.importorskip("jax")
    img = medrs.NiftiImage(ramp((4, 3, 2), np.int16)).with_header(scl_slope=0.5)
    arr = img.to_jax()
    assert arr.dtype == np.float32
    np.testing.assert_array_equal(np.asarray(arr), ramp((4, 3, 2)) * 0.5)
    on_cpu = img.to_jax(dtype="bfloat16", device="cpu")
    assert str(on_cpu.dtype) == "bfloat16"
    assert on_cpu.devices() == {jax.devices("cpu")[0]}


def test_torch_conversion(tmp_path):
    torch = pytest.importorskip("torch")
    data = ramp((8, 7, 6))
    medrs.NiftiImage(data).save(tmp_path / "t.nii")
    img = medrs.load(tmp_path / "t.nii")
    tensor = img.to_torch()
    tensor.add_(1)  # writable, not backed by the read-only file mapping
    np.testing.assert_array_equal(img.to_numpy(), data)
    assert medrs.NiftiImage(ramp((2, 2, 2), np.uint16)).to_torch().dtype == torch.int32
    assert img.to_torch(dtype=torch.bfloat16).dtype == torch.bfloat16
    assert img.to_torch(dtype="float16").dtype == torch.float16


def test_header_updates_are_validated():
    img = medrs.NiftiImage(ramp((2, 2, 2)))
    with pytest.raises(ValueError, match="unknown header field"):
        img.with_header(not_a_field=1)
    with pytest.raises(ValueError, match="with_affine"):
        img.with_header(affine=np.eye(4))
    moved = img.with_affine(OBLIQUE)
    np.testing.assert_allclose(moved.affine, OBLIQUE)
    with pytest.raises(ValueError, match="last row"):
        img.with_affine(np.ones((4, 4)))
    with pytest.raises(ValueError, match="shape"):
        img.with_data(np.zeros((3, 3, 3)))
