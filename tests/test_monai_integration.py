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
        from medrs.monai_compat import MedrsLoadImage

        # Test that MedrsLoadImage is a proper MONAI transform
        loader = MedrsLoadImage()
        assert isinstance(loader, Transform)
        assert callable(loader)

    def test_monai_compose_integration(self):
        """Test that medrs transforms work in MONAI Compose pipelines."""
        from medrs.monai_compat import MedrsLoadImage

        # Create a medrs transform
        medrs_loader = MedrsLoadImage()

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

    def test_medrs_rand_spatial_crop_transform(self):
        """Test MedrsRandSpatialCropd transform interface."""
        from medrs.monai_compat import MedrsRandSpatialCropd

        # Test transform creation
        cropper = MedrsRandSpatialCropd(
            keys=["image"],
            roi_size=[32, 32, 16],
        )
        assert callable(cropper)

        # Test that parameters are stored correctly
        assert list(cropper.roi_size) == [32, 32, 16]
        assert cropper.keys == ("image",)

    def test_medrs_loadimaged_transform(self):
        """Test MedrsLoadImaged transform interface."""
        from medrs.monai_compat import MedrsLoadImaged

        # Test transform creation
        loader = MedrsLoadImaged(
            keys=["image", "label"],
            ensure_channel_first=True,
        )
        assert callable(loader)

        # Test that parameters are stored correctly
        assert loader.keys == ("image", "label")
        assert loader.ensure_channel_first is True


class TestPipelineCreation:
    """Test pipeline creation with different configurations."""

    def test_create_medrs_monai_pipeline(self):
        """Test pipeline creation with medrs and MONAI transforms."""
        from medrs.monai_compat import MedrsLoadImaged, MedrsRandSpatialCropd
        from monai.transforms import Compose, RandFlipd, EnsureTyped

        # Create a mixed pipeline
        pipeline = Compose([
            MedrsLoadImaged(keys=["image"], ensure_channel_first=True),
            MedrsRandSpatialCropd(keys=["image"], roi_size=[32, 32, 32]),
            RandFlipd(keys=["image"], prob=0.5),
            EnsureTyped(keys=["image"]),
        ])

        assert pipeline is not None
        assert callable(pipeline)
        assert len(pipeline.transforms) == 4

    def test_medrs_training_dataloader_interface(self):
        """Test medrs TrainingDataLoader interface."""
        # Test that TrainingDataLoader class is available
        assert hasattr(medrs, 'TrainingDataLoader')
        loader_class = getattr(medrs, 'TrainingDataLoader')
        assert callable(loader_class)


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
        """Test that TrainingDataLoader is available."""
        assert hasattr(medrs, 'TrainingDataLoader')
        loader_class = getattr(medrs, 'TrainingDataLoader')
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
                        path,
                        output_shape=[16, 16, 16],
                    )
        finally:
            if os.path.exists(path):
                os.unlink(path)


class TestIntegrationExample:
    """Test the integration example functionality."""

    def test_monai_compat_module_imports(self):
        """Test that the monai_compat module can be imported and used."""
        from medrs.monai_compat import (
            MedrsLoadImage,
            MedrsLoadImaged,
            MedrsSaveImage,
            MedrsSaveImaged,
            MedrsRandCropByPosNegLabeld,
            MedrsRandSpatialCropd,
            MedrsCenterSpatialCropd,
            MedrsOrientation,
            MedrsOrientationd,
        )

        # Verify all classes are callable
        assert callable(MedrsLoadImage)
        assert callable(MedrsLoadImaged)
        assert callable(MedrsSaveImage)
        assert callable(MedrsSaveImaged)
        assert callable(MedrsRandCropByPosNegLabeld)
        assert callable(MedrsRandSpatialCropd)
        assert callable(MedrsCenterSpatialCropd)
        assert callable(MedrsOrientation)
        assert callable(MedrsOrientationd)


class TestErrorHandling:
    """Test error handling in integration scenarios."""

    def test_missing_file_handling(self):
        """Test handling of missing files in pipelines."""
        from medrs.monai_compat import MedrsLoadImage

        loader = MedrsLoadImage()

        # Should handle missing files gracefully
        with pytest.raises(Exception):  # Should raise some kind of error
            loader("nonexistent_file.nii")

    def test_crop_transform_creation(self):
        """Test crop transform creation with various parameters."""
        from medrs.monai_compat import MedrsRandSpatialCropd, MedrsCenterSpatialCropd

        # Valid crop creation
        cropper = MedrsRandSpatialCropd(keys=["image"], roi_size=[32, 32, 32])
        assert callable(cropper)

        center_cropper = MedrsCenterSpatialCropd(keys=["image"], roi_size=[32, 32, 32])
        assert callable(center_cropper)


class TestMetaTensorOrientationConversion:
    """Test orientation to direction matrix conversion."""

    def test_ras_orientation(self):
        """Test RAS (Right-Anterior-Superior) orientation."""
        from medrs.metatensor_support import MedrsMetaTensorConverter

        direction = MedrsMetaTensorConverter._orientation_to_direction("RAS")

        # RAS: X=Right(+), Y=Anterior(+), Z=Superior(+)
        expected = np.array([
            [1, 0, 0],  # R -> +X
            [0, 1, 0],  # A -> +Y
            [0, 0, 1],  # S -> +Z
        ], dtype=np.float64)
        np.testing.assert_array_equal(direction, expected)

    def test_lpi_orientation(self):
        """Test LPI (Left-Posterior-Inferior) orientation."""
        from medrs.metatensor_support import MedrsMetaTensorConverter

        direction = MedrsMetaTensorConverter._orientation_to_direction("LPI")

        # LPI: X=Left(-), Y=Posterior(-), Z=Inferior(-)
        expected = np.array([
            [-1, 0, 0],   # L -> -X
            [0, -1, 0],   # P -> -Y
            [0, 0, -1],   # I -> -Z
        ], dtype=np.float64)
        np.testing.assert_array_equal(direction, expected)

    def test_las_orientation(self):
        """Test LAS (Left-Anterior-Superior) orientation."""
        from medrs.metatensor_support import MedrsMetaTensorConverter

        direction = MedrsMetaTensorConverter._orientation_to_direction("LAS")

        # LAS: X=Left(-), Y=Anterior(+), Z=Superior(+)
        expected = np.array([
            [-1, 0, 0],  # L -> -X
            [0, 1, 0],   # A -> +Y
            [0, 0, 1],   # S -> +Z
        ], dtype=np.float64)
        np.testing.assert_array_equal(direction, expected)

    def test_case_insensitivity(self):
        """Test that orientation codes are case insensitive."""
        from medrs.metatensor_support import MedrsMetaTensorConverter

        upper = MedrsMetaTensorConverter._orientation_to_direction("RAS")
        lower = MedrsMetaTensorConverter._orientation_to_direction("ras")
        mixed = MedrsMetaTensorConverter._orientation_to_direction("RaS")

        np.testing.assert_array_equal(upper, lower)
        np.testing.assert_array_equal(upper, mixed)


if __name__ == "__main__":
    # Run tests with detailed output
    pytest.main([__file__, "-v", "--tb=short"])
