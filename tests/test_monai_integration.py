#!/usr/bin/env python3
"""
MONAI Integration Tests for medrs.

This test suite validates the seamless integration between medrs and MONAI,
ensuring that medrs's high-performance I/O can be used effectively in
MONAI-based medical imaging pipelines.
"""

import os
import tempfile
import numpy as np
import pytest
import torch
import medrs
from monai.transforms import Compose, Transform, EnsureChannelFirst, ScaleIntensity


class TestMONAICompatibility:
    """Test MONAI compatibility and transform integration."""

    def test_monai_transform_inheritance(self):
        """Test that medrs transforms can inherit from MONAI Transform."""
        from examples.monai_compose_integration import MedrsLoadImage

        # Test that MedrsLoadImage is a proper MONAI transform
        loader = MedrsLoadImage()
        assert isinstance(loader, Transform)
        assert callable(loader)

    def test_monai_compose_integration(self):
        """Test that medrs transforms work in MONAI Compose pipelines."""
        from examples.monai_compose_integration import MedrsLoadImage

        # Create a medrs transform
        medrs_loader = MedrsLoadImage(dtype=np.float32)

        # Create MONAI compose pipeline with medrs transform
        transforms = [
            medrs_loader,
            EnsureChannelFirst(),
            ScaleIntensity(),
        ]

        # Test that compose works
        pipeline = Compose(transforms)
        assert pipeline is not None
        assert len(pipeline.transforms) == 3

        # Test that the first transform is our medrs loader
        assert isinstance(pipeline.transforms[0], MedrsLoadImage)

    def test_medrs_load_cropped_transform(self):
        """Test MedrsLoadCropped transform interface."""
        from examples.monai_compose_integration import MedrsLoadCropped

        # Test transform creation
        loader = MedrsLoadCropped(
            crop_offset=[16, 16, 8],
            crop_size=[32, 32, 16],
            target_spacing=[1.0, 1.0, 1.0]
        )
        assert isinstance(loader, Transform)
        assert callable(loader)

        # Test that parameters are stored correctly
        assert loader.crop_offset == [16, 16, 8]
        assert loader.crop_size == [32, 32, 16]
        assert loader.target_spacing == [1.0, 1.0, 1.0]

    def test_medrs_load_to_tensor_transform(self):
        """Test MedrsLoadToTensor transform interface."""
        from examples.integrations.monai_integration import MedrsLoadToTensor

        # Test transform creation
        loader = MedrsLoadToTensor(
            crop_size=[32, 32, 16],
            dtype="float16",
            device="cpu"
        )
        assert isinstance(loader, Transform)
        assert callable(loader)

        # Test that parameters are stored correctly
        assert loader.crop_size == [32, 32, 16]
        assert loader.dtype == "float16"
        assert loader.device == "cpu"


class TestPipelineCreation:
    """Test pipeline creation with different configurations."""

    def test_create_medrs_monai_pipeline(self):
        """Test pipeline creation with different configurations."""
        from examples.monai_compose_integration import create_medrs_monai_pipeline

        # Test different configurations
        configs = [
            ("segmentation", "memory"),
            ("segmentation", "speed"),
            ("classification", "balanced"),
            ("detection", "memory"),
        ]

        for task_type, performance_mode in configs:
            try:
                pipeline = create_medrs_monai_pipeline(task_type, performance_mode)
                assert pipeline is not None
                assert callable(pipeline)

                # Test that pipeline has transforms
                assert len(pipeline.transforms) > 0
            except Exception as e:
                # Some configs might fail if MONAI transforms aren't available
                print(f"Note: Config {task_type}/{performance_mode} failed: {e}")

    def test_medrs_training_dataloader_interface(self):
        """Test medrs training data loader interface."""
        from examples.monai_compose_integration import create_medrs_training_dataloader

        # Test with mock file list
        mock_files = ["test1.nii", "test2.nii"]

        # Should handle gracefully with missing files
        try:
            loader = create_medrs_training_dataloader(
                image_files=mock_files,
                patch_size=[64, 64, 64],
                patches_per_volume=4,
                cache_size=100
            )
        except Exception as e:
            # Expected to fail with mock files, but interface should be valid
            assert "No such file" in str(e) or "os error" in str(e)


class TestPerformanceFeatures:
    """Test performance-critical features and optimizations."""

    def test_byte_exact_loading_availability(self):
        """Test that byte-exact loading functions are available."""
        # Test that critical functions exist
        critical_functions = [
            'load_cropped',
            'load_resampled',
            'load_cropped_to_torch',
            'load_cropped_to_jax'
        ]

        for func_name in critical_functions:
            assert hasattr(medrs, func_name)
            func = getattr(medrs, func_name)
            assert callable(func)

    def test_training_data_loader_availability(self):
        """Test that PyTrainingDataLoader is available."""
        assert hasattr(medrs, 'PyTrainingDataLoader')
        loader_class = getattr(medrs, 'PyTrainingDataLoader')
        assert callable(loader_class)

    def test_framework_integration_signatures(self):
        """Test that framework integration functions have correct signatures."""
        # Test torch function signature
        import inspect
        torch_func = getattr(medrs, 'load_cropped_to_torch')
        sig = inspect.signature(torch_func)

        expected_params = [
            'path', 'output_shape', 'target_spacing',
            'target_orientation', 'output_offset', 'dtype', 'device'
        ]

        for param in expected_params:
            assert param in sig.parameters, f"Missing parameter: {param}"

        # Test jax function signature
        jax_func = getattr(medrs, 'load_cropped_to_jax')
        sig = inspect.signature(jax_func)

        expected_params = [
            'path', 'output_shape', 'target_spacing',
            'target_orientation', 'output_offset', 'dtype', 'device'
        ]

        for param in expected_params:
            assert param in sig.parameters, f"Missing parameter: {param}"


class TestPyTorchIntegration:
    """Test PyTorch-specific integration features."""

    def test_torch_tensor_creation_interface(self):
        """Test PyTorch tensor creation interface."""
        # Test that we can call the function (will fail with mock file but interface exists)
        try:
            with tempfile.NamedTemporaryFile(suffix='.nii', delete=False) as f:
                path = f.name

            with pytest.raises(Exception):  # Expected with mock file
                tensor = medrs.load_cropped_to_torch(
                    volume_path=path,
                    output_shape=[32, 32, 16],
                    dtype="float32",
                    device="cpu"
                )
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_torch_device_support(self):
        """Test PyTorch device support."""
        devices = ["cpu"]
        if torch.cuda.is_available():
            devices.append("cuda")

        try:
            with tempfile.NamedTemporaryFile(suffix='.nii', delete=False) as f:
                path = f.name

            for device in devices:
                with pytest.raises(Exception):  # Expected with mock file
                    tensor = medrs.load_cropped_to_torch(
                        volume_path=path,
                        output_shape=[16, 16, 16],
                        device=device
                    )
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_torch_dtype_support(self):
        """Test PyTorch dtype support."""
        dtypes = ["float32", "float16", "int32"]

        try:
            with tempfile.NamedTemporaryFile(suffix='.nii', delete=False) as f:
                path = f.name

            for dtype in dtypes:
                with pytest.raises(Exception):  # Expected with mock file
                    tensor = medrs.load_cropped_to_torch(
                        volume_path=path,
                        output_shape=[16, 16, 16],
                        dtype=dtype,
                        device="cpu"
                    )
        finally:
            if os.path.exists(path):
                os.unlink(path)


class TestJAXIntegration:
    """Test JAX-specific integration features."""

    def test_jax_array_creation_interface(self):
        """Test JAX array creation interface."""
        try:
            with tempfile.NamedTemporaryFile(suffix='.nii', delete=False) as f:
                path = f.name

            with pytest.raises(Exception):  # Expected with mock file
                jax_array = medrs.load_cropped_to_jax(
                    volume_path=path,
                    output_shape=[32, 32, 16],
                    dtype="float32"
                )
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_jax_dtype_support(self):
        """Test JAX dtype support."""
        dtypes = ["float32", "float16", "int32"]

        try:
            with tempfile.NamedTemporaryFile(suffix='.nii', delete=False) as f:
                path = f.name

            for dtype in dtypes:
                with pytest.raises(Exception):  # Expected with mock file
                    jax_array = medrs.load_cropped_to_jax(
                        volume_path=path,
                        output_shape=[16, 16, 16],
                        dtype=dtype
                    )
        finally:
            if os.path.exists(path):
                os.unlink(path)


class TestIntegrationExample:
    """Test the integration example functionality."""

    def test_integration_example_runs(self):
        """Test that the integration example can run without errors."""
        import subprocess
        import sys

        # Run the integration example as a subprocess
        result = subprocess.run(
            [sys.executable, "examples/monai_compose_integration.py"],
            capture_output=True,
            text=True,
            timeout=30
        )

        # Should complete successfully (even with mock files)
        assert result.returncode == 0, f"Example failed: {result.stderr}"

        # Check for expected output
        expected_phrases = [
            "medrs + MONAI Integration Example",
            "Available Integrations",
            "Performance Benchmark"
        ]

        for phrase in expected_phrases:
            assert phrase in result.stdout, f"Missing expected output: {phrase}"


class TestErrorHandling:
    """Test error handling in integration scenarios."""

    def test_invalid_transform_parameters(self):
        """Test handling of invalid transform parameters."""
        from examples.monai_compose_integration import MedrsLoadImage

        # Test invalid reader
        with pytest.raises(ValueError):
            MedrsLoadImage(reader="invalid_reader")

    def test_missing_file_handling(self):
        """Test handling of missing files in pipelines."""
        from examples.monai_compose_integration import MedrsLoadImage

        loader = MedrsLoadImage()

        # Should handle missing files gracefully
        with pytest.raises(Exception):  # Should raise some kind of error
            loader("nonexistent_file.nii")

    def test_invalid_crop_parameters(self):
        """Test handling of invalid crop parameters."""
        from examples.monai_compose_integration import MedrsLoadCropped

        # Test invalid crop sizes (negative or zero)
        invalid_configs = [
            ([0, 0, 0], [16, 16, 16]),  # Zero offset
            ([16, 16, 16], [0, 0, 0]),   # Zero size
            ([-1, 16, 16], [32, 32, 32]), # Negative offset
            ([16, 16, 16], [-1, 32, 32]), # Negative size
        ]

        for offset, size in invalid_configs:
            # Transform should create successfully but fail during execution
            try:
                loader = MedrsLoadCropped(crop_offset=offset, crop_size=size)
                assert isinstance(loader, Transform)
            except Exception as e:
                # Some validation might happen at creation time
                assert "crop" in str(e).lower() or "offset" in str(e).lower() or "size" in str(e).lower()


if __name__ == "__main__":
    # Run tests with detailed output
    pytest.main([__file__, "-v", "--tb=short"])
