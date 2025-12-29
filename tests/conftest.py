"""
pytest configuration and shared fixtures for medrs testing.

Provides comprehensive test infrastructure including:
- Reusable test data fixtures
- Device-specific testing
- Performance monitoring
- Property-based testing utilities
"""

from __future__ import annotations

import sys
import pytest
import tempfile
import shutil
from pathlib import Path
from typing import Generator, List, Tuple, Any
import numpy as np
import psutil

ROOT_DIR = Path(__file__).resolve().parents[1]

if sys.version_info < (3, 10):
    raise pytest.UsageError(
        "medrs requires Python >= 3.10; upgrade your interpreter to run the test suite."
    )

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import jax
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    HAS_JAX = False

try:
    import nibabel as nib
    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False

try:
    import medrs
    HAS_MEDRS = True
except ImportError:
    HAS_MEDRS = False

from tests.test_utils import (
    create_test_nifti_file
)


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Run slow tests (including large file tests)"
    )
    parser.addoption(
        "--test-device",
        choices=["cpu", "cuda", "all"],
        default="cpu",
        help="Device to test on (cpu, cuda, or all)"
    )


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "cuda: marks tests that require CUDA"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection based on command line options."""
    if not config.getoption("--run-slow"):
        skip_slow = pytest.mark.skip(reason="need --run-slow option to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)

    # Skip CUDA tests if not requested
    test_device = config.getoption("--test-device")
    if test_device == "cpu":
        skip_cuda = pytest.mark.skip(reason="need --test-device=cuda or --test-device=all to run CUDA tests")
        for item in items:
            if "cuda" in item.keywords:
                item.add_marker(skip_cuda)


@pytest.fixture(scope="session")
def test_data_dir() -> Generator[Path, None, None]:
    """Create temporary directory for test data."""
    temp_dir = Path(tempfile.mkdtemp(prefix="medrs_test_"))
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def temp_nifti_dir(test_data_dir: Path) -> Generator[Path, None, None]:
    """Create temporary directory with test NIfTI files."""
    nifti_dir = test_data_dir / "nifti_files"
    nifti_dir.mkdir()

    # Create test files with different properties
    test_files = {
        "small_3d.nii": (32, 32, 16),
        "medium_3d.nii": (64, 64, 32),
        "large_3d.nii": (128, 128, 64),
        "xlarge_3d.nii": (256, 256, 128),
        "small_4d.nii": (10, 32, 32, 16),
        "medium_4d.nii": (20, 64, 64, 32),
        "anisotropic.nii": (64, 64, 64),  # Will have non-isotropic spacing
    }

    for filename, shape in test_files.items():
        file_path = nifti_dir / filename
        create_test_nifti_file(
            file_path,
            shape=shape,
            add_structure=True,
            seed=hash(filename)  # Reproducible but different files
        )

        # Set anisotropic spacing for the anisotropic file
        if filename == "anisotropic.nii":
            img = nib.load(file_path)
            affine = img.affine.copy()
            affine[0, 0] = 2.0  # 2mm in x
            affine[1, 1] = 2.0  # 2mm in y
            affine[2, 2] = 5.0  # 5mm in z
            aniso_img = nib.Nifti1Image(img.get_fdata(), affine)
            nib.save(aniso_img, file_path)

    yield nifti_dir


@pytest.fixture
def test_dataset(temp_nifti_dir: Path) -> List[str]:
    """Provide list of test NIfTI file paths."""
    return [str(p) for p in sorted(temp_nifti_dir.glob("*.nii"))]


@pytest.fixture(params=["small_3d.nii", "medium_3d.nii", "large_3d.nii"])
def test_nifti_file(request, temp_nifti_dir: Path) -> str:
    """Parameterized fixture for different sized NIfTI files."""
    return str(temp_nifti_dir / request.param)


@pytest.fixture
def torch_device():
    """Provide appropriate torch device for testing."""
    if not HAS_TORCH:
        pytest.skip("PyTorch not available")
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(params=[
    pytest.param("float32", marks=pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")),
    pytest.param("float16", marks=pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")),
])
def torch_dtype(request):
    """Parameterized fixture for different torch dtypes."""
    if not HAS_TORCH:
        pytest.skip("PyTorch not available")
    import torch
    dtype_map = {"float32": torch.float32, "float16": torch.float16}
    return dtype_map[request.param]


@pytest.fixture
def torch_devices():
    """Provide list of available torch devices for testing."""
    if not HAS_TORCH:
        pytest.skip("PyTorch not available")

    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.extend(["cuda"] + [f"cuda:{i}" for i in range(torch.cuda.device_count())])

    return devices


@pytest.fixture
def memory_monitor():
    """Context manager for monitoring memory usage."""
    from contextlib import contextmanager
    import psutil
    import os

    @contextmanager
    def monitor():
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        yield lambda: process.memory_info().rss - initial_memory

    return monitor()


@pytest.fixture
def performance_monitor():
    """Context manager for monitoring performance."""
    from contextlib import contextmanager
    import time
    import gc

    @contextmanager
    def monitor():
        gc.collect()  # Clean up before measurement

        start_time = time.perf_counter()
        start_memory = psutil.Process().memory_info().rss

        yield

        end_time = time.perf_counter()
        end_memory = psutil.Process().memory_info().rss

        print(f"\nPerformance: {end_time - start_time:.3f}s")
        print(f"Memory delta: {(end_memory - start_memory) / 1024 / 1024:.1f}MB")

    return monitor()


class PropertyBasedTestHelper:
    """Helper for property-based testing."""

    @staticmethod
    def generate_valid_shapes() -> List[Tuple[int, int, int]]:
        """Generate valid 3D shapes for testing."""
        shapes = []
        for base in [8, 16, 32, 64]:
            shapes.extend([
                (base, base, base),
                (base * 2, base, base),
                (base, base * 2, base),
                (base, base, base * 2),
            ])
        return shapes

    @staticmethod
    def generate_invalid_shapes() -> List[Any]:
        """Generate invalid shapes for error testing."""
        return [
            [],  # Empty
            [64],  # 1D
            [64, 64],  # 2D
            [0, 64, 64],  # Zero dimension
            [64, 0, 64],  # Zero dimension
            [64, 64, 0],  # Zero dimension
            [-1, 64, 64],  # Negative dimension
            [64.5, 64, 64],  # Float dimension
            ["64", 64, 64],  # String dimension
        ]

    @staticmethod
    def generate_test_affines() -> List[np.ndarray]:
        """Generate test affine matrices."""
        affines = []

        # Identity matrix
        affines.append(np.eye(4))

        # With rotation
        for angle in [30, 45, 90, 180]:
            theta = np.radians(angle)
            rot = np.array([
                [np.cos(theta), -np.sin(theta), 0, 0],
                [np.sin(theta), np.cos(theta), 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])
            affines.append(rot)

        # With scaling
        for scale in [0.5, 1.0, 2.0, 5.0]:
            scale_matrix = np.diag([scale, scale, scale, 1])
            affines.append(scale_matrix)

        # Random valid affines
        for _ in range(3):
            # Generate random affine with valid properties
            affine = np.eye(4)
            affine[:3, :3] = np.random.randn(3, 3) * 2
            affine[:3, 3] = np.random.randn(3) * 100
            affines.append(affine)

        return affines


@pytest.fixture
def property_test_helper():
    """Provide property-based testing helper."""
    return PropertyBasedTestHelper()


@pytest.fixture
def sample_medical_volume():
    """Generate a sample medical volume with realistic properties."""
    shape = (64, 64, 32)

    # Create brain-like structure
    data = np.zeros(shape, dtype=np.float32)

    # Add brain-like elliptical structure
    center = np.array(shape) // 2
    radius = min(shape) // 3

    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                dist = np.sqrt(((np.array([i, j, k]) - center) ** 2).sum())
                if dist < radius:
                    # Brain tissue intensity
                    data[i, j, k] = np.random.normal(0.8, 0.1)
                elif dist < radius * 1.2:
                    # CSF intensity
                    data[i, j, k] = np.random.normal(0.3, 0.05)

    # Add some lesions (bright spots)
    for _ in range(5):
        lesion_center = [
            np.random.randint(radius, shape[0] - radius),
            np.random.randint(radius, shape[1] - radius),
            np.random.randint(radius, shape[2] - radius)
        ]
        lesion_radius = np.random.randint(2, 5)

        for i in range(max(0, lesion_center[0] - lesion_radius),
                      min(shape[0], lesion_center[0] + lesion_radius)):
            for j in range(max(0, lesion_center[1] - lesion_radius),
                          min(shape[1], lesion_center[1] + lesion_radius)):
                for k in range(max(0, lesion_center[2] - lesion_radius),
                              min(shape[2], lesion_center[2] + lesion_radius)):
                    dist = np.sqrt(((np.array([i, j, k]) - np.array(lesion_center)) ** 2).sum())
                    if dist < lesion_radius:
                        data[i, j, k] = np.random.normal(1.5, 0.2)

    return data


# Custom markers for better test organization
pytest.mark.unit = pytest.mark.unit
pytest.mark.integration = pytest.mark.integration
pytest.mark.performance = pytest.mark.performance
pytest.mark.cuda = pytest.mark.cuda
pytest.mark.slow = pytest.mark.slow


# Skip conditions
skip_if_no_torch = pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
skip_if_no_cuda = pytest.mark.skipif(not (HAS_TORCH and torch.cuda.is_available()), reason="CUDA not available")
skip_if_no_jax = pytest.mark.skipif(not HAS_JAX, reason="JAX not available")
skip_if_no_nibabel = pytest.mark.skipif(not HAS_NIBABEL, reason="nibabel not available")
skip_if_no_medrs = pytest.mark.skipif(not HAS_MEDRS, reason="medrs not available")
