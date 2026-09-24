"""FastLoader: formats, determinism, labels, and shutdown."""

import threading
import time

import numpy as np
import pytest

import medrs

from conftest import ramp


@pytest.fixture
def dataset(tmp_path):
    images, labels = [], []
    for i, ext in enumerate(["nii", "nii.gz", "jvol", "nii", "hdr", "nii.gz"]):
        data = ramp((20, 18, 16)) + 100_000 * i
        path = tmp_path / f"img{i}.{ext}"
        medrs.NiftiImage(data).save(path)
        images.append(path)
        label = np.zeros((20, 18, 16), np.uint8)
        label[15:18, 2:5, 10:12] = 1
        label_path = tmp_path / f"seg{i}.nii.gz"
        medrs.NiftiImage(label).save(label_path)
        labels.append(label_path)
    return images, labels


def collect(epoch):
    return [(p.volume, p.offset, p.image.to_numpy().copy()) for p in epoch]


def test_patches_come_from_every_format_and_match_their_region(dataset):
    images, _ = dataset
    loader = medrs.FastLoader(images, (8, 8, 8), patches_per_volume=2, seed=1)
    assert len(loader) == 12
    seen = set()
    for patch in loader:
        seen.add(patch.volume)
        x, y, z = patch.offset
        full = ramp((20, 18, 16)) + 100_000 * patch.volume
        np.testing.assert_array_equal(patch.image.to_numpy(), full[x : x + 8, y : y + 8, z : z + 8])
    assert seen == set(range(6))


def test_order_does_not_depend_on_worker_count(dataset):
    images, _ = dataset
    runs = [
        collect(
            medrs.FastLoader(images, (6, 6, 6), patches_per_volume=3, workers=w, seed=7).epoch(2)
        )
        for w in (0, 1, 4)
    ]
    for other in runs[1:]:
        assert [(v, o) for v, o, _ in other] == [(v, o) for v, o, _ in runs[0]]
        for (_, _, a), (_, _, b) in zip(other, runs[0], strict=True):
            np.testing.assert_array_equal(a, b)
    loader = medrs.FastLoader(images, (6, 6, 6), seed=7)
    assert [p.offset for p in loader.epoch(0)] != [p.offset for p in loader.epoch(1)]


def test_labels_padding_and_pipeline(dataset):
    images, labels = dataset
    loader = medrs.FastLoader(
        images,
        (24, 24, 8),
        labels=labels,
        foreground_prob=1.0,
        pipeline=medrs.Pipeline().random_flip((0, 1, 2)).z_normalize(),
        seed=3,
    )
    for patch in loader:
        assert patch.image.shape == patch.label.shape == (24, 24, 8)
        assert patch.shape == (20, 18, 8)
        assert patch.image.dtype == "float32"
        assert patch.label.dtype == "uint8"
        assert patch.label.to_numpy().sum() > 0


def test_close_and_errors(dataset, tmp_path):
    images, _ = dataset
    loader = medrs.FastLoader(images * 20, (4, 4, 4), workers=4, prefetch=2)
    with loader.epoch(0) as epoch:
        next(epoch)
    start = time.monotonic()
    epoch = iter(loader)
    next(epoch)
    del epoch
    assert time.monotonic() - start < 10
    broken = medrs.FastLoader([images[0], tmp_path / "missing.nii"], (4, 4, 4), shuffle=False)
    it = iter(broken)
    next(it)
    with pytest.raises(FileNotFoundError):
        next(it)
    with pytest.raises(ValueError, match="foreground_prob requires label"):
        medrs.FastLoader(images, (4, 4, 4), foreground_prob=0.5)
    with pytest.raises(ValueError, match="labels"):
        medrs.FastLoader(images, (4, 4, 4), labels=images[:2])


def test_iteration_releases_the_gil(dataset):
    images, _ = dataset
    ticks = 0
    stop = threading.Event()

    def count():
        nonlocal ticks
        while not stop.is_set():
            ticks += 1

    thread = threading.Thread(target=count)
    thread.start()
    for _ in medrs.FastLoader(images * 4, (16, 16, 16), workers=0):
        pass
    stop.set()
    thread.join()
    assert ticks > 100


def test_patch_shape_choices_and_volume_weights(dataset):
    images, _ = dataset
    shapes = [(8, 8, 8), (12, 6, 4)]
    loader = medrs.FastLoader(
        images,
        shapes,
        patch_shape_weights=[1, 3],
        weights=[0, 0, 1, 0, 0, 1],
        volumes_per_epoch=40,
        seed=2,
    )
    assert len(loader) == 40
    patches = list(loader)
    assert {p.volume for p in patches} == {2, 5}
    counts = {s: sum(p.image.shape == s for p in patches) for s in shapes}
    assert sum(counts.values()) == 40
    assert counts[(12, 6, 4)] > counts[(8, 8, 8)]
    assert "patch_shape=[[8, 8, 8], [12, 6, 4]]" in repr(loader)
    with pytest.raises(ValueError, match="weights"):
        medrs.FastLoader(images, (8, 8, 8), weights=[1, 2])


def test_foreground_threshold_without_labels(tmp_path):
    data = np.zeros((30, 30, 30), np.float32)
    data[20:24, 3:7, 10:13] = 100
    path = tmp_path / "bright.nii"
    medrs.NiftiImage(data).save(path)
    loader = medrs.FastLoader(
        [path], (6, 6, 6), foreground_prob=1.0, foreground_threshold=50, patches_per_volume=10
    )
    for patch in loader:
        assert patch.label is None
        assert patch.image.to_numpy().max() == 100
    with pytest.raises(ValueError, match="foreground_threshold"):
        medrs.FastLoader([path], (6, 6, 6), foreground_prob=0.5)
