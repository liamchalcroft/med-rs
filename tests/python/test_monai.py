import numpy as np
import pytest

from conftest import OBLIQUE, ramp

monai = pytest.importorskip("monai")

import medrs  # noqa: E402
from medrs.monai import MedrsReader  # noqa: E402


def test_load_image_matches_nibabel_reader(tmp_path):
    from monai.data import NibabelReader
    from monai.transforms import Compose, EnsureChannelFirstd, LoadImaged

    image = tmp_path / "image.nii.gz"
    label = tmp_path / "label.nii.gz"
    medrs.NiftiImage(ramp((12, 10, 8)), OBLIQUE).save(image)
    medrs.NiftiImage((ramp((12, 10, 8)) % 3).astype(np.uint8), OBLIQUE).save(label)
    sample = {"image": str(image), "label": str(label)}

    def load(reader):
        keys = ["image", "label"]
        return Compose([LoadImaged(keys, reader=reader), EnsureChannelFirstd(keys)])(sample)

    ours, theirs = load(MedrsReader()), load(NibabelReader())
    for key in ("image", "label"):
        assert ours[key].shape == theirs[key].shape == (1, 12, 10, 8)
        np.testing.assert_allclose(ours[key].numpy(), theirs[key].numpy())
        np.testing.assert_allclose(ours[key].affine.numpy(), theirs[key].affine.numpy(), atol=1e-5)


def test_reader_suffixes():
    reader = MedrsReader()
    assert reader.verify_suffix(["a.nii.gz", "b.jvol", "c.HDR"])
    assert not reader.verify_suffix("scan.dcm")
