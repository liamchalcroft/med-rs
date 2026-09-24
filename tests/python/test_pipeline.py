"""Pipelines: same results as eager transforms, reproducible randomness."""

import pickle

import numpy as np
import pytest

import medrs

from conftest import OBLIQUE, ramp


@pytest.fixture
def image():
    return medrs.NiftiImage(ramp((12, 10, 8), np.int16), OBLIQUE).with_header(scl_slope=2.0)


def test_matches_eager_transforms(image):
    pipeline = (
        medrs.Pipeline()
        .reorient("RAS")
        .crop((1, 1, 1), (10, 8, 6))
        .clamp(0, 800)
        .z_normalize()
        .clamp(-1, 1)
        .rescale(0, 10)
    )
    eager = image.reorient("RAS").crop((1, 1, 1), (10, 8, 6)).clamp(0, 800).z_normalize()
    eager = eager.clamp(-1, 1).rescale(0, 10)
    out = pipeline.apply(image)
    np.testing.assert_allclose(out.to_numpy(), eager.to_numpy(), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(out.affine, eager.affine, atol=1e-9)


def test_resampling_steps_interpolate_once(image):
    two_steps = medrs.Pipeline().resample_to_shape((24, 20, 16)).resample_to_shape((6, 5, 4))
    direct = image.resample_to_shape((6, 5, 4))
    np.testing.assert_allclose(two_steps.apply(image).to_numpy(), direct.to_numpy(), rtol=1e-5)


def test_random_steps_are_reproducible_and_joint(image):
    label = image.with_dtype("uint8")
    pipeline = (
        medrs.Pipeline()
        .random_flip((0, 1, 2), 0.5)
        .random_rotate_90((0, 1))
        .random_intensity_scale(0.2)
        .random_gaussian_noise(0.1)
    )
    a, la = pipeline.apply(image, label, seed=4)
    b, lb = pipeline.apply(image, label, seed=4)
    np.testing.assert_array_equal(a.to_numpy(), b.to_numpy())
    np.testing.assert_array_equal(la.to_numpy(), lb.to_numpy())
    assert la.dtype == "uint8"
    np.testing.assert_allclose(a.affine, la.affine)
    c = pipeline.apply(image, seed=5)
    assert not np.array_equal(a.to_numpy(), c.to_numpy())


def test_pipelines_are_immutable_picklable_and_validated():
    base = medrs.Pipeline().z_normalize()
    longer = base.clamp(-3, 3)
    assert (len(base), len(longer)) == (1, 2)
    restored = pickle.loads(pickle.dumps(longer.random_flip((0,), 1.0)))
    assert len(restored) == 3
    assert "random_flip" in repr(restored)
    with pytest.raises(ValueError, match="min <= max"):
        base.clamp(1, 0)
    with pytest.raises(ValueError, match="probability"):
        base.random_flip((0,), 1.5)
    with pytest.raises(ValueError, match="interpolation"):
        base.resample_to_shape((2, 2, 2), "cubic")


def test_cast_to_half_precision(image):
    out = medrs.Pipeline().z_normalize().cast("float16").apply(image)
    assert out.dtype == "float16"


def test_rescale_percentiles_step_matches_eager(image):
    pipeline = medrs.Pipeline().crop((1, 1, 1), (10, 8, 6)).rescale_percentiles(2, 98, True)
    eager = image.crop((1, 1, 1), (10, 8, 6)).rescale_percentiles(2, 98, nonzero=True)
    np.testing.assert_allclose(pipeline.apply(image).to_numpy(), eager.to_numpy(), atol=1e-6)
    restored = pickle.loads(pickle.dumps(pipeline))
    np.testing.assert_array_equal(
        restored.apply(image).to_numpy(), pipeline.apply(image).to_numpy()
    )
    with pytest.raises(ValueError, match="percentiles"):
        medrs.Pipeline().rescale_percentiles(90, 10)
