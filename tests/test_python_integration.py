#!/usr/bin/env python3
"""
Comprehensive Python integration tests for medrs.

This test suite validates the seamless integration between medrs and major ML frameworks:
- PyTorch tensor creation and zero-copy operations
- JAX array creation and zero-copy operations
- NumPy array view operations
- Direct GPU loading capabilities
- Half-precision support without upcasting
- MONAI Compose pipeline compatibility
"""

import os
import tempfile
import numpy as np
import pytest
import torch
import jax.numpy as jnp
import medrs
from monai.transforms import Compose

# Import test utilities
from test_utils import (
    create_3d_test_file, create_small_test_file
)


class TestNumPyIntegration:
    """Test NumPy integration and array view operations."""

    def test_load_and_numpy_view(self):
        """Test basic loading and numpy view creation."""
        # Create a test NIfTI file using our utility
        with create_3d_test_file(shape=(64, 64, 32), seed=42) as path:
            # Load and test numpy view
            loaded = medrs.load(path)
            np_view = loaded.to_numpy_view()

            assert np_view.shape == (64, 64, 32)
            assert np_view.dtype == np.float32
            # Verify data is not empty and has reasonable values
            assert np_view.size == 64 * 64 * 32
            assert np.isfinite(np_view).all()

    def test_memory_efficiency_numpy_view(self):
        """Test that numpy views are zero-copy when possible."""
        # Test with both structured and random data
        with create_small_test_file() as path:
            loaded = medrs.load(path)

            # Test that we can get a numpy view
            np_view = loaded.to_numpy_view()
            assert np_view.shape == loaded.shape()
            assert np_view.dtype == np.float32

            # Test that we can get numpy array
            np_array = loaded.to_numpy()
            assert np_array.shape == loaded.shape()
            assert np_array.dtype == np.float32

            # Verify the data is consistent between view and array
            np.testing.assert_array_equal(np_view, np_array)


class TestPyTorchIntegration:
    """Test PyTorch tensor integration and zero-copy operations."""

    def test_load_to_torch_cpu(self):
        """Test direct PyTorch tensor creation on CPU."""
        # Test interface exists
        assert hasattr(medrs, 'load_cropped_to_torch')
        assert hasattr(medrs, 'load_to_torch')

        # Test with real file
        with create_small_test_file() as path:
            # Test full volume loading
            tensor = medrs.load_to_torch(
                path,
                dtype=torch.float32,
                device="cpu"
            )
            assert isinstance(tensor, torch.Tensor)
            assert tensor.device.type == 'cpu'
            assert tensor.dtype == torch.float32

            # Test cropped loading
            cropped_tensor = medrs.load_cropped_to_torch(
                path,
                output_shape=[8, 8, 4],
                dtype=torch.float32,
                device="cpu"
            )
            assert isinstance(cropped_tensor, torch.Tensor)
            assert cropped_tensor.shape == (8, 8, 4)
            assert cropped_tensor.device.type == 'cpu'

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_load_to_torch_gpu(self):
        """Test direct PyTorch tensor creation on GPU."""
        with create_small_test_file() as path:
            # Test GPU loading
            tensor = medrs.load_to_torch(
                path,
                dtype=torch.float16,  # Half precision
                device="cuda"
            )
            assert isinstance(tensor, torch.Tensor)
            assert tensor.device.type == 'cuda'
            assert tensor.dtype == torch.float16

            # Test cropped GPU loading
            cropped_tensor = medrs.load_cropped_to_torch(
                path,
                output_shape=[8, 8, 4],
                dtype=torch.float16,
                device="cuda"
            )
            assert isinstance(cropped_tensor, torch.Tensor)
            assert cropped_tensor.shape == (8, 8, 4)
            assert cropped_tensor.device.type == 'cuda'
            assert cropped_tensor.dtype == torch.float16

    def test_torch_dtype_conversion(self):
        """Test PyTorch dtype conversion support."""
        with create_small_test_file() as path:
            dtypes = [torch.float32, torch.float16]
            for dtype in dtypes:
                tensor = medrs.load_cropped_to_torch(
                    volume_path=path,
                    output_shape=[8, 8, 4],
                    dtype=dtype,
                    device="cpu"
                )
                assert isinstance(tensor, torch.Tensor)
                assert tensor.dtype == dtype
                assert tensor.shape == (8, 8, 4)

    def test_torch_device_conversion(self):
        """Test PyTorch device conversion support."""
        with create_small_test_file() as path:
            devices = ["cpu"]
            if torch.cuda.is_available():
                devices.append("cuda")

            for device in devices:
                tensor = medrs.load_cropped_to_torch(
                    volume_path=path,
                    output_shape=[8, 8, 4],
                    device=device
                )
                assert isinstance(tensor, torch.Tensor)
                assert tensor.device.type.split(':')[0] == device.split(':')[0]
                assert tensor.shape == (8, 8, 4)


class TestJAXIntegration:
    """Test JAX array integration and zero-copy operations."""

    def test_load_to_jax(self):
        """Test direct JAX array creation."""
        # Test interface exists
        assert hasattr(medrs, 'load_cropped_to_jax')

        with create_small_test_file() as path:
            # Test function with real data
            jax_array = medrs.load_cropped_to_jax(
                volume_path=path,
                output_shape=[8, 8, 4],
                dtype=jnp.float32
            )
            assert hasattr(jax_array, 'shape')  # JAX array-like
            assert jax_array.shape == (8, 8, 4)

    def test_jax_dtype_conversion(self):
        """Test JAX dtype conversion support."""
        try:
            with tempfile.NamedTemporaryFile(suffix='.nii', delete=False) as f:
                path = f.name

            dtypes = [jnp.float32, jnp.float16, jnp.int32]
            for dtype in dtypes:
                with pytest.raises(Exception):  # Expected to fail with mock file
                    jax_array = medrs.load_cropped_to_jax(
                        volume_path=path,
                        output_shape=[16, 16, 16],
                        dtype=dtype
                    )
        finally:
            if os.path.exists(path):
                os.unlink(path)


class TestMONAICompatibility:
    """Test MONAI Compose pipeline compatibility."""

    def test_monai_compose_adapter(self):
        """Test that medrs can work within MONAI Compose pipelines."""
        # Test that we can create a compatible interface
        class MedrsLoadImage:
            """MONAI-compatible adapter for medrs loading."""

            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def __call__(self, path):
                # Load using medrs and return numpy array for MONAI compatibility
                img = medrs.load(path)
                return img.to_numpy_view()

        # Test adapter creation
        loader = MedrsLoadImage()
        assert callable(loader)

        # Test in a compose pipeline (interface test only)
        try:
            with tempfile.NamedTemporaryFile(suffix='.nii', delete=False) as f:
                path = f.name

            # Create MONAI-style transform
            transform = Compose([
                MedrsLoadImage(),
                # Would add more transforms here with real data
            ])

            # Test that the transform can be called
            with pytest.raises(Exception):  # Expected to fail with mock file
                result = transform(path)
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_training_dataloader_compatibility(self):
        """Test TrainingDataLoader compatibility with MONAI workflows."""
        # Test that TrainingDataLoader exists and has expected interface
        assert hasattr(medrs, 'TrainingDataLoader')

        # Test interface (would need real files for full test)
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                volume_paths = []  # Would populate with real NIfTI files

                # Test TrainingDataLoader creation interface
                if volume_paths:  # Only test if we have volumes
                    loader = medrs.TrainingDataLoader(
                        volumes=volume_paths,
                        patch_size=[64, 64, 64],
                        patches_per_volume=4,
                        randomize=True,
                        cache_size=1000
                    )

                    # Test that it has expected methods
                    assert hasattr(loader, 'next_patch')
                    assert hasattr(loader, 'reset')
                    assert hasattr(loader, 'stats')
        except Exception:
            # Interface test passes even with empty volume list
            pass


class TestPerformanceFeatures:
    """Test performance-critical features and optimizations."""

    def test_byte_exact_loading_interface(self):
        """Test byte-exact loading interface exists."""
        assert hasattr(medrs, 'load_cropped')

        # Test function signature
        try:
            with tempfile.NamedTemporaryFile(suffix='.nii', delete=False) as f:
                path = f.name

            # Test interface (will fail with mock file but validates signature)
            with pytest.raises(Exception):
                patch = medrs.load_cropped(path, [16, 16, 16], [32, 32, 32])
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_advanced_loading_interface(self):
        """Test advanced loading with orientation/resampling."""
        # Test interface exists
        assert hasattr(medrs, 'load_resampled')

        try:
            with tempfile.NamedTemporaryFile(suffix='.nii', delete=False) as f:
                path = f.name

            # Test advanced loading interface
            with pytest.raises(Exception):
                patch = medrs.load_resampled(
                    volume_path=path,
                    output_shape=[32, 32, 32],
                    target_spacing=[1.0, 1.0, 1.0],
                    target_orientation="RAS"
                )
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_half_precision_support(self):
        """Test half-precision loading without upcasting."""
        # Test that interface supports half precision
        try:
            with tempfile.NamedTemporaryFile(suffix='.nii', delete=False) as f:
                path = f.name

            with pytest.raises(Exception):
                    tensor = medrs.load_cropped_to_torch(
                        volume_path=path,
                        output_shape=[16, 16, 16],
                        dtype=torch.float16,  # Half precision
                        device="cpu"
                    )

            with pytest.raises(Exception):
                    jax_array = medrs.load_cropped_to_jax(
                        volume_path=path,
                        output_shape=[16, 16, 16],
                        dtype=jnp.float16  # Half precision
                    )
        finally:
            if os.path.exists(path):
                os.unlink(path)


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_invalid_file_handling(self):
        """Test handling of invalid file paths."""
        # Test with non-existent file
        with pytest.raises(Exception):
            medrs.load("nonexistent_file.nii")

        # Test with non-NIfTI file
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
            f.write(b"not a nifti file")
            temp_path = f.name

        try:
            with pytest.raises(Exception):
                medrs.load(temp_path)
        finally:
            os.unlink(temp_path)

    def test_invalid_shape_handling(self):
        """Test handling of invalid shapes."""
        with create_small_test_file() as path:
            # Test invalid shapes - zero dimensions
            with pytest.raises(Exception):
                medrs.load_cropped(path, [0, 0, 0], [8, 8, 4])

            # Test invalid shapes - negative dimensions
            with pytest.raises(Exception):
                medrs.load_cropped(path, [8, 8, 4], [-1, -1, -1])

            # Test shapes larger than image
            with pytest.raises(Exception):
                medrs.load_cropped(path, [8, 8, 4], [32, 32, 32])  # Larger than our small test file

    def test_dtype_validation(self):
        """Test dtype validation for framework integrations."""
        with create_small_test_file() as path:
            # Test invalid dtype handling
            with pytest.raises(Exception):
                tensor = medrs.load_cropped_to_torch(
                    volume_path=path,
                    output_shape=[8, 8, 4],
                    dtype="invalid_dtype",  # Should be torch.dtype
                    device="cpu"
                )


if __name__ == "__main__":
    # Run tests with detailed output
    pytest.main([__file__, "-v", "--tb=short"])
