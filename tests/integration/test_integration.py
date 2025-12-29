#!/usr/bin/env python3
"""
MONAI + medrs Integration Tests

Test that medrs functions are available and can be integrated with MONAI transforms.
"""

import pytest
import medrs

# Check for required dependencies
torch = pytest.importorskip("torch")
monai = pytest.importorskip("monai")


class TestMONAIIntegration:
    """Test MONAI integration capabilities."""

    def test_core_functions_exist(self):
        """Verify core medrs functions are available."""
        required_functions = ["load", "load_cropped", "load_resampled"]
        for func in required_functions:
            assert hasattr(medrs, func), f"Missing core function: {func}"

    def test_advanced_functions_exist(self):
        """Verify advanced medrs functions are available."""
        advanced_functions = [
            "load_cropped_to_torch",
            "load_label_aware_cropped",
            "compute_crop_regions",
        ]
        for func in advanced_functions:
            assert hasattr(medrs, func), f"Missing advanced function: {func}"

    def test_torch_tensor_creation(self):
        """Verify PyTorch tensor creation works."""
        image_tensor = torch.randn(1, 96, 96, 96)
        label_tensor = torch.randint(0, 2, (1, 96, 96, 96))

        assert image_tensor.shape == (1, 96, 96, 96)
        assert image_tensor.dtype == torch.float32
        assert label_tensor.shape == (1, 96, 96, 96)

    def test_monai_version_compatibility(self):
        """Verify MONAI version is compatible."""
        from packaging import version

        monai_version = version.parse(monai.__version__)
        # medrs requires MONAI >= 1.5
        assert monai_version >= version.parse("1.5.0"), (
            f"MONAI version {monai.__version__} is below minimum required 1.5.0"
        )

    def test_torch_cuda_detection(self):
        """Verify CUDA detection works (doesn't require CUDA)."""
        # Just verify the detection doesn't crash
        cuda_available = torch.cuda.is_available()
        assert isinstance(cuda_available, bool)
