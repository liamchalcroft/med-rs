"""Transforms, checked against nibabel, scipy, and world-coordinate invariants."""

import nibabel as nib
import numpy as np
import pytest

import medrs

from conftest import OBLIQUE, ramp


def world(affine, index):
    return affine[:3, :3] @ np.asarray(index, float) + affine[:3, 3]


def assert_world_preserving(before, after):
    """Every output voxel sits where the input voxel with the same value was."""
    src, out = before.to_numpy(), after.to_numpy()
    positions = {v: np.unravel_index(i, src.shape) for i, v in enumerate(src.ravel())}
    for index in np.ndindex(*out.shape[:3]):
        value = out[index]
        np.testing.assert_allclose(
            world(after.affine, index), world(before.affine, positions[value][:3]), atol=1e-6
        )


@pytest.mark.parametrize("code", ["RAS", "LPS", "PIR", "SAL"])
def test_reorient_matches_nibabel(code):
    data = ramp((9, 8, 7), np.int32)
    img = medrs.NiftiImage(data, OBLIQUE)
    out = img.reorient(code)
    ornt = nib.orientations.ornt_transform(
        nib.orientations.io_orientation(OBLIQUE), nib.orientations.axcodes2ornt(code)
    )
    expected = nib.orientations.apply_orientation(data, ornt)
    np.testing.assert_array_equal(out.to_numpy(), expected)
    expected_affine = OBLIQUE @ nib.orientations.inv_ornt_aff(ornt, data.shape)
    np.testing.assert_allclose(out.affine, expected_affine, atol=1e-6)
    assert out.orientation == code
    assert out.dtype == "int32"


def test_invalid_orientation_is_rejected():
    with pytest.raises(ValueError, match="orientation"):
        medrs.NiftiImage(ramp((2, 2, 2))).reorient("RRA")


@pytest.mark.parametrize(
    "op",
    [
        lambda i: i.flip([0]),
        lambda i: i.flip([1, 2]),
        lambda i: i.rotate_90((0, 1), 1),
        lambda i: i.rotate_90((2, 0), 3),
        lambda i: i.crop((1, 2, 1), (4, 3, 2)),
        lambda i: i.crop_or_pad((4, 4, 4)),
    ],
    ids=["flip0", "flip12", "rot01", "rot20", "crop", "center-crop"],
)
def test_spatial_transforms_are_world_preserving(op):
    img = medrs.NiftiImage(ramp((6, 5, 4), np.int32), OBLIQUE)
    out = op(img)
    assert out.dtype == "int32"
    assert_world_preserving(img, out)


def test_flip_changes_the_data_and_saves_correctly(tmp_path):
    data = ramp((4, 3, 2))
    medrs.NiftiImage(data).flip([0]).save(tmp_path / "f.nii")
    np.testing.assert_array_equal(np.asanyarray(nib.load(tmp_path / "f.nii").dataobj), data[::-1])


def test_rotate_matches_numpy_rot90():
    data = ramp((5, 4, 3))
    out = medrs.NiftiImage(data).rotate_90((0, 1), 1).to_numpy()
    np.testing.assert_array_equal(out, np.rot90(data, 1, (0, 1)))


def test_chained_spatial_and_intensity_transforms_stay_consistent(tmp_path):
    data = ramp((10, 9, 8))
    out = medrs.NiftiImage(data).flip([0]).crop_or_pad((12, 6, 8)).z_normalize()
    expected = np.zeros((12, 6, 8), np.float32)
    expected[1:11] = data[::-1, 1:7]
    # Statistics include the zero padding: it is part of the image.
    np.testing.assert_allclose(
        out.to_numpy(), (expected - expected.mean()) / expected.std(), rtol=1e-5, atol=1e-6
    )
    out.save(tmp_path / "chain.nii.gz")
    np.testing.assert_allclose(medrs.load(tmp_path / "chain.nii.gz").to_numpy(), out.to_numpy())


def test_resample_matches_scipy_in_the_interior():
    ndimage = pytest.importorskip("scipy.ndimage")
    rng = np.random.default_rng(0)
    data = rng.normal(size=(20, 18, 16)).astype(np.float32)
    affine = np.diag([1.2, 1.0, 2.0, 1.0])
    img = medrs.NiftiImage(data, affine)
    out = img.resample((0.8, 1.5, 1.0))
    # Map every output voxel into the input grid and interpolate with scipy.
    to_input = np.linalg.inv(affine) @ out.affine
    grid = np.indices(out.shape).reshape(3, -1)
    coords = to_input[:3, :3] @ grid + to_input[:3, 3:4]
    expected = ndimage.map_coordinates(data, coords, order=1).reshape(out.shape)
    inside = np.all((coords >= 0) & (coords <= np.array(data.shape)[:, None] - 1), axis=0)
    np.testing.assert_allclose(out.to_numpy().ravel()[inside], expected.ravel()[inside], atol=1e-5)
    np.testing.assert_allclose(out.spacing, (0.8, 1.5, 1.0), rtol=0.05)


def test_nearest_resampling_keeps_labels():
    labels = (ramp((10, 10, 10)) % 4).astype(np.uint8)
    out = medrs.NiftiImage(labels).resample_to_shape((17, 7, 12), "nearest")
    assert out.dtype == "uint8"
    assert set(np.unique(out.to_numpy())) <= {0, 1, 2, 3}


def test_resample_like_aligns_in_world_space():
    fine = medrs.NiftiImage(np.zeros((10, 10, 10), np.float32))
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    affine[:3, 3] = 3.0
    moving = medrs.NiftiImage(ramp((5, 5, 5)), affine)
    out = moving.resample_like(fine, "nearest")
    assert out.shape == fine.shape
    np.testing.assert_allclose(out.affine, fine.affine)
    assert out.to_numpy()[5, 3, 3] == moving.to_numpy()[1, 0, 0]


def test_intensity_transforms():
    data = np.array([[[0.0, 2.0], [4.0, 6.0]]], np.float32)
    img = medrs.NiftiImage(data)
    z = img.z_normalize().to_numpy()
    np.testing.assert_allclose(z, (data - data.mean()) / data.std(), rtol=1e-6)
    nz = img.z_normalize(nonzero=True).to_numpy()
    fg = data[data != 0]
    np.testing.assert_allclose(nz[data != 0], (fg - fg.mean()) / fg.std(), rtol=1e-6)
    assert nz[0, 0, 0] == 0
    np.testing.assert_allclose(img.rescale(-1, 1).to_numpy(), data / 3 - 1, rtol=1e-6)
    np.testing.assert_array_equal(img.clamp(1, 5).to_numpy(), np.clip(data, 1, 5))
    np.testing.assert_allclose(img.adjust_gamma(2.0).to_numpy(), (data / 6) ** 2 * 6, rtol=1e-6)
    with pytest.raises(ValueError, match="NaN"):
        medrs.NiftiImage(np.array([1.0, np.nan], np.float32)).z_normalize()


def test_clamp_uses_scaled_units():
    ct = medrs.NiftiImage(np.array([0, 500, 1500, 4000], np.int16)).with_header(scl_inter=-1024.0)
    np.testing.assert_array_equal(ct.clamp(-500, 500).to_numpy(), [-500, -500, 476, 500])


def test_four_d_images_transform_per_volume():
    data = ramp((6, 5, 4, 3))
    img = medrs.NiftiImage(data, OBLIQUE)
    assert img.flip([0]).shape == (6, 5, 4, 3)
    np.testing.assert_array_equal(img.flip([0]).to_numpy(), data[::-1])
    assert img.resample_to_shape((3, 5, 2)).shape == (3, 5, 2, 3)
    np.testing.assert_array_equal(img.crop((1, 1, 1), (2, 2, 2)).to_numpy(), data[1:3, 1:3, 1:3])


@pytest.mark.parametrize("nonzero", [False, True])
def test_percentiles_match_numpy(nonzero):
    rng = np.random.default_rng(0)
    data = rng.gamma(2.0, 50.0, (31, 27, 19)).astype(np.float32)
    data[:5] = 0
    img = medrs.NiftiImage(data).with_header(scl_slope=0.5, scl_inter=3.0)
    scaled = data * 0.5 + 3.0
    # `nonzero` refers to scaled values: stored zeros are 3.0 here, so they count.
    reference = scaled[scaled != 0] if nonzero else scaled
    q = (0.0, 0.5, 25.0, 50.0, 99.5, 100.0)
    ours = img.percentiles(q, nonzero=nonzero)
    np.testing.assert_allclose(ours, np.percentile(reference, q), rtol=1e-6)

    lo, hi = np.percentile(reference, (0.5, 99.5))
    expected = np.clip((scaled - lo) / (hi - lo), 0, 1)
    out = img.rescale_percentiles(0.5, 99.5, nonzero=nonzero)
    assert out.dtype == "float32"
    np.testing.assert_allclose(out.to_numpy(), expected, atol=1e-5)
    out = img.rescale_percentiles(0.5, 99.5, nonzero=nonzero, out_min=-1, out_max=1)
    np.testing.assert_allclose(out.to_numpy(), expected * 2 - 1, atol=1e-5)


def test_percentiles_leave_out_zero_background():
    data = np.zeros((10, 10, 10), np.float32)
    data[2:8, 2:8, 2:8] = np.arange(216, dtype=np.float32).reshape(6, 6, 6) + 1
    img = medrs.NiftiImage(data)
    assert img.percentiles((0, 100), nonzero=True) == (1.0, 216.0)
    assert img.percentiles((0, 100)) == (0.0, 216.0)
    with pytest.raises(ValueError, match="between 0 and 100"):
        img.percentiles((101,))
    with pytest.raises(ValueError, match="no voxels"):
        medrs.NiftiImage(np.zeros((2, 2, 2), np.float32)).percentiles((50,), nonzero=True)
    with pytest.raises(ValueError, match="NaN"):
        medrs.NiftiImage(np.full((2, 2, 2), np.nan, np.float32)).percentiles((50,))
