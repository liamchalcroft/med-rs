import pytest

import medrs
from tests.test_utils import create_small_test_file


def test_crop_helpers_reject_bad_patch_size_length():
    with create_small_test_file() as image_path, create_small_test_file() as label_path:
        with pytest.raises(ValueError, match="patch_size must be a 3-element sequence"):
            medrs.load_label_aware_cropped(image_path, label_path, [32, 32])

        with pytest.raises(ValueError, match="patch_size must be a 3-element sequence"):
            medrs.compute_crop_regions(image_path, label_path, [32, 32], 1)

        with pytest.raises(ValueError, match="patch_size must be a 3-element sequence"):
            medrs.compute_random_spatial_crops(image_path, [32, 32], 1)

        with pytest.raises(ValueError, match="patch_size must be a 3-element sequence"):
            medrs.compute_center_crop(image_path, [32, 32])


def test_crop_helpers_reject_zero_patch_dimension():
    with create_small_test_file() as image_path, create_small_test_file() as label_path:
        with pytest.raises(ValueError, match="patch_size dimension 0 must be positive"):
            medrs.load_label_aware_cropped(image_path, label_path, [0, 32, 32])
