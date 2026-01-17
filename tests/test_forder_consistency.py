"""Test F-order consistency across numpy, torch, and jax conversions.

This verifies that:
1. Values match between all conversion methods
2. Spatial indexing is consistent (accessing [x,y,z] gives same value)
3. Memory layout expectations are correct
"""

from pathlib import Path
import numpy as np
import pytest
import medrs

# Try importing optional dependencies
try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import jax.numpy as jnp

    HAS_JAX = True
except ImportError:
    HAS_JAX = False

try:
    import nibabel as nib

    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False


@pytest.fixture
def brain_image():
    """Load the brain test image fixture."""
    brain_path = Path(__file__).parent / "fixtures" / "mprage_img.nii"
    if not brain_path.exists():
        pytest.skip(f"Test image not found at {brain_path}")
    return medrs.load(str(brain_path)), brain_path


class TestFOrderConsistency:
    """Test F-order memory layout consistency across conversion methods."""

    def test_numpy_conversion_shape(self, brain_image):
        """Numpy conversion preserves shape."""
        img, _ = brain_image
        np_arr = img.to_numpy()
        assert np_arr.shape == tuple(img.shape)

    def test_numpy_f_contiguous(self, brain_image):
        """Numpy arrays from medrs are F-contiguous (column-major)."""
        img, _ = brain_image
        np_arr = img.to_numpy()
        # NIfTI uses F-order (Fortran/column-major)
        assert np_arr.flags["F_CONTIGUOUS"], "Array should be F-contiguous for NIfTI"

    @pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
    def test_torch_values_match_numpy(self, brain_image):
        """Torch tensor values match numpy array values at all test points."""
        img, _ = brain_image
        np_arr = img.to_numpy()
        torch_tensor = img.to_torch()

        # Test points at various locations
        test_points = [
            (0, 0, 0),
            (10, 20, 30),
            (50, 100, 80),
            (img.shape[0] - 1, img.shape[1] - 1, img.shape[2] - 1),
            (img.shape[0] // 2, img.shape[1] // 2, img.shape[2] // 2),
        ]

        for x, y, z in test_points:
            np_val = np_arr[x, y, z]
            torch_val = torch_tensor[x, y, z].item()
            assert abs(np_val - torch_val) < 1e-5, (
                f"Mismatch at ({x},{y},{z}): numpy={np_val}, torch={torch_val}"
            )

    @pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
    def test_torch_full_array_match(self, brain_image):
        """Torch tensor fully matches numpy array."""
        img, _ = brain_image
        np_arr = img.to_numpy()
        torch_tensor = img.to_torch()
        torch_np = torch_tensor.numpy()

        max_diff = np.abs(np_arr - torch_np).max()
        assert max_diff < 1e-5, f"Max difference too large: {max_diff}"

    @pytest.mark.skipif(not HAS_JAX, reason="jax not available")
    def test_jax_values_match_numpy(self, brain_image):
        """JAX array values match numpy array values at all test points."""
        img, _ = brain_image
        np_arr = img.to_numpy()
        jax_arr = img.to_jax()

        test_points = [
            (0, 0, 0),
            (10, 20, 30),
            (50, 100, 80),
            (img.shape[0] - 1, img.shape[1] - 1, img.shape[2] - 1),
            (img.shape[0] // 2, img.shape[1] // 2, img.shape[2] // 2),
        ]

        for x, y, z in test_points:
            np_val = np_arr[x, y, z]
            jax_val = float(jax_arr[x, y, z])
            assert abs(np_val - jax_val) < 1e-5, (
                f"Mismatch at ({x},{y},{z}): numpy={np_val}, jax={jax_val}"
            )

    @pytest.mark.skipif(not HAS_JAX, reason="jax not available")
    def test_jax_full_array_match(self, brain_image):
        """JAX array fully matches numpy array."""
        img, _ = brain_image
        np_arr = img.to_numpy()
        jax_arr = img.to_jax()
        jax_np = np.array(jax_arr)

        max_diff = np.abs(np_arr - jax_np).max()
        assert max_diff < 1e-5, f"Max difference too large: {max_diff}"

    @pytest.mark.skipif(not HAS_NIBABEL, reason="nibabel not available")
    def test_nibabel_values_match(self, brain_image):
        """medrs values match nibabel reference implementation."""
        img, brain_path = brain_image
        np_arr = img.to_numpy()

        nib_img = nib.load(str(brain_path))
        nib_data = nib_img.get_fdata().astype(np.float32)

        test_points = [
            (0, 0, 0),
            (10, 20, 30),
            (50, 100, 80),
            (img.shape[0] - 1, img.shape[1] - 1, img.shape[2] - 1),
            (img.shape[0] // 2, img.shape[1] // 2, img.shape[2] // 2),
        ]

        for x, y, z in test_points:
            np_val = np_arr[x, y, z]
            nib_val = nib_data[x, y, z]
            assert abs(np_val - nib_val) < 1e-5, (
                f"Mismatch vs nibabel at ({x},{y},{z}): medrs={np_val}, nibabel={nib_val}"
            )

    @pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
    def test_slice_consistency_torch(self, brain_image):
        """Slicing produces consistent results between numpy and torch."""
        img, _ = brain_image
        np_arr = img.to_numpy()
        torch_tensor = img.to_torch()

        mid_z = np_arr.shape[2] // 2
        np_slice = np_arr[:, :, mid_z]
        torch_slice = torch_tensor[:, :, mid_z].numpy()

        max_diff = np.abs(np_slice - torch_slice).max()
        assert max_diff < 1e-5, f"Slice mismatch: max_diff={max_diff}"

    @pytest.mark.skipif(not HAS_JAX, reason="jax not available")
    def test_slice_consistency_jax(self, brain_image):
        """Slicing produces consistent results between numpy and JAX."""
        img, _ = brain_image
        np_arr = img.to_numpy()
        jax_arr = img.to_jax()

        mid_z = np_arr.shape[2] // 2
        np_slice = np_arr[:, :, mid_z]
        jax_slice = np.array(jax_arr[:, :, mid_z])

        max_diff = np.abs(np_slice - jax_slice).max()
        assert max_diff < 1e-5, f"Slice mismatch: max_diff={max_diff}"


class TestSyntheticFOrder:
    """Test with synthetic data where exact values are known."""

    def test_coordinate_encoding_roundtrip(self):
        """Values encode coordinates correctly through roundtrip."""
        # Create array where value = x + 10*y + 100*z
        shape = (4, 5, 6)
        data = np.zeros(shape, dtype=np.float32)
        for x in range(shape[0]):
            for y in range(shape[1]):
                for z in range(shape[2]):
                    data[x, y, z] = x + 10 * y + 100 * z

        img = medrs.NiftiImage(data)
        np_back = img.to_numpy()

        test_coords = [(0, 0, 0), (1, 2, 3), (3, 4, 5), (2, 1, 4)]
        for x, y, z in test_coords:
            expected = x + 10 * y + 100 * z
            actual = np_back[x, y, z]
            assert abs(expected - actual) < 1e-5, (
                f"Coordinate mismatch at ({x},{y},{z}): expected={expected}, actual={actual}"
            )

    @pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
    def test_coordinate_encoding_torch(self):
        """Torch conversion preserves coordinate encoding."""
        shape = (4, 5, 6)
        data = np.zeros(shape, dtype=np.float32)
        for x in range(shape[0]):
            for y in range(shape[1]):
                for z in range(shape[2]):
                    data[x, y, z] = x + 10 * y + 100 * z

        img = medrs.NiftiImage(data)
        torch_tensor = img.to_torch()

        test_coords = [(0, 0, 0), (1, 2, 3), (3, 4, 5), (2, 1, 4)]
        for x, y, z in test_coords:
            expected = x + 10 * y + 100 * z
            actual = torch_tensor[x, y, z].item()
            assert abs(expected - actual) < 1e-5, (
                f"Torch coordinate mismatch at ({x},{y},{z}): expected={expected}, actual={actual}"
            )

    @pytest.mark.skipif(not HAS_JAX, reason="jax not available")
    def test_coordinate_encoding_jax(self):
        """JAX conversion preserves coordinate encoding."""
        shape = (4, 5, 6)
        data = np.zeros(shape, dtype=np.float32)
        for x in range(shape[0]):
            for y in range(shape[1]):
                for z in range(shape[2]):
                    data[x, y, z] = x + 10 * y + 100 * z

        img = medrs.NiftiImage(data)
        jax_arr = img.to_jax()

        test_coords = [(0, 0, 0), (1, 2, 3), (3, 4, 5), (2, 1, 4)]
        for x, y, z in test_coords:
            expected = x + 10 * y + 100 * z
            actual = float(jax_arr[x, y, z])
            assert abs(expected - actual) < 1e-5, (
                f"JAX coordinate mismatch at ({x},{y},{z}): expected={expected}, actual={actual}"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
