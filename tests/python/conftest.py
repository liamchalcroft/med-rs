from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

FIXTURE = Path(__file__).parents[1] / "fixtures" / "mprage_img.nii.gz"

OBLIQUE = np.array(
    [
        [-0.9, 0.1, 0.05, 90.0],
        [0.08, 1.1, -0.1, -126.0],
        [0.0, 0.12, 2.5, -72.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


def ramp(shape, dtype=np.float32):
    """Values that encode their own position, so misplaced voxels show up."""
    return np.arange(np.prod(shape)).reshape(shape).astype(dtype)


@pytest.fixture
def write_nib(tmp_path):
    """Write an array with nibabel and return the path."""

    def write(data, affine=OBLIQUE, name="img.nii.gz", **header_fields):
        img = nib.Nifti1Image(data, affine, dtype=data.dtype)
        for key, value in header_fields.items():
            img.header[key] = value
        path = tmp_path / name
        nib.save(img, path)
        return path

    return write


@pytest.fixture
def fixture_path():
    if not FIXTURE.exists():
        pytest.skip("fixture volume not available")
    return FIXTURE
