#!/usr/bin/env python3
"""
Test data utilities for medrs.

This module provides utilities for generating synthetic NIfTI files for testing,
ensuring consistent test data across different test suites.
"""

import os
import tempfile
import numpy as np
import nibabel as nib
from pathlib import Path
from typing import Tuple, Optional, List, Union


def generate_random_nifti(
    shape: Tuple[int, ...] = (64, 64, 32),
    affine: Optional[np.ndarray] = None,
    dtype: np.dtype = np.float32,
    seed: Optional[int] = None,
    add_structure: bool = False,
) -> nib.Nifti1Image:
    """
    Generate a random NIfTI image for testing.

    Args:
        shape: Shape of the image (last 3 dimensions are spatial)
        affine: 4x4 affine transformation matrix (default: identity)
        dtype: Data type of the image
        seed: Random seed for reproducibility
        add_structure: Whether to add structured patterns (spheres, cubes) vs pure noise

    Returns:
        nibabel NIfTI image object
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate random data
    if add_structure:
        data = generate_structured_data(shape, seed=seed)
    else:
        data = np.random.randn(*shape).astype(dtype)

    # Create affine matrix if not provided
    if affine is None:
        affine = np.eye(4)
        # Set reasonable voxel spacing
        affine[0, 0] = 1.0  # 1mm in x
        affine[1, 1] = 1.0  # 1mm in y
        affine[2, 2] = 2.5  # 2.5mm in z (common for medical imaging)

    return nib.Nifti1Image(data, affine)


def generate_structured_data(shape: Tuple[int, ...], seed: Optional[int] = None) -> np.ndarray:
    """
    Generate structured medical imaging-like data with spheres and intensity variations.

    Args:
        shape: Shape of the data
        seed: Random seed for reproducibility

    Returns:
        Structured numpy array
    """
    if seed is not None:
        np.random.seed(seed)

    data = np.zeros(shape, dtype=np.float32)

    # Add background noise
    data += np.random.randn(*shape) * 0.1

    # Add some spherical structures (like organs or lesions)
    center = [s // 2 for s in shape[-3:]]

    # Large central structure
    radius = min(shape[-3:]) // 4
    add_sphere(data, center, radius, intensity=1.0)

    # Smaller structures
    for _ in range(3):
        pos = [np.random.randint(0, s) for s in shape[-3:]]
        r = np.random.randint(3, 8)
        intensity = np.random.uniform(0.5, 1.5)
        add_sphere(data, pos, r, intensity)

    # Add some cube-like structures
    for _ in range(2):
        pos = [np.random.randint(0, max(1, s - 10)) for s in shape[-3:]]
        size = np.random.randint(3, 8)
        intensity = np.random.uniform(0.3, 0.8)
        add_cube(data, pos, size, intensity)

    return data


def add_sphere(data: np.ndarray, center: List[int], radius: float, intensity: float) -> None:
    """Add a sphere to the data array."""
    shape = data.shape[-3:]
    coords = np.ogrid[[slice(0, s) for s in shape]]

    # Calculate distance from center for each voxel
    dist_sq = sum((coord - c) ** 2 for coord, c in zip(coords, center))
    mask = dist_sq <= radius**2

    # Apply sphere with smooth edges
    if data.ndim > 3:
        # Extend mask to match full data shape
        mask_shape = [1] * (data.ndim - 3) + list(mask.shape)
        mask = np.broadcast_to(mask, mask_shape)

    data[mask] += intensity


def add_cube(data: np.ndarray, corner: List[int], size: int, intensity: float) -> None:
    """Add a cube to the data array."""
    shape = data.shape

    # Calculate cube bounds
    start_idx = [corner[i] for i in range(3)]
    end_idx = [min(corner[i] + size, shape[-3 + i]) for i in range(3)]

    # Create slices for the cube
    slices = [slice(None)] * (data.ndim - 3)
    for i in range(3):
        slices.append(slice(start_idx[i], end_idx[i]))

    data[tuple(slices)] += intensity


def create_test_nifti_file(
    output_path: Union[str, Path],
    shape: Tuple[int, ...] = (64, 64, 32),
    affine: Optional[np.ndarray] = None,
    dtype: np.dtype = np.float32,
    seed: Optional[int] = None,
    add_structure: bool = False,
    compressed: bool = False,
) -> str:
    """
    Create a test NIfTI file and return the path.

    Args:
        output_path: Path where to save the file
        shape: Shape of the image
        affine: 4x4 affine transformation matrix
        dtype: Data type of the image
        seed: Random seed for reproducibility
        add_structure: Whether to add structured patterns
        compressed: Whether to compress the file (.nii.gz)

    Returns:
        Path to the created file
    """
    output_path = Path(output_path)

    # Add .nii extension if not present
    if not output_path.suffix:
        output_path = output_path.with_suffix(".nii")
    elif output_path.suffix == ".gz" and not output_path.stem.endswith(".nii"):
        output_path = output_path.with_name(output_path.stem + ".nii.gz")
    elif compressed and not output_path.suffix == ".gz":
        output_path = output_path.with_suffix(output_path.suffix + ".gz")

    # Generate the image
    img = generate_random_nifti(shape, affine, dtype, seed, add_structure)

    # Save the image
    nib.save(img, str(output_path))

    return str(output_path)


def create_test_dataset(
    output_dir: Union[str, Path],
    num_images: int = 5,
    shapes: Optional[List[Tuple[int, ...]]] = None,
    seeds: Optional[List[int]] = None,
    add_labels: bool = False,
) -> Tuple[List[str], List[str]]:
    """
    Create a test dataset of NIfTI files.

    Args:
        output_dir: Directory to save the files
        num_images: Number of images to create
        shapes: List of shapes for each image (default: varied shapes)
        seeds: List of random seeds for each image
        add_labels: Whether to create corresponding label files

    Returns:
        Tuple of (image_paths, label_paths)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if shapes is None:
        # Create varied shapes for realistic testing
        shapes = []
        for i in range(num_images):
            if i % 3 == 0:
                shapes.append((64, 64, 32))  # Standard
            elif i % 3 == 1:
                shapes.append((128, 128, 64))  # Large
            else:
                shapes.append((32, 32, 16))  # Small

    if seeds is None:
        seeds = list(range(num_images))

    image_paths = []
    label_paths = []

    for i in range(num_images):
        # Create image
        img_path = output_dir / f"image_{i:03d}.nii"
        img_path = create_test_nifti_file(
            img_path, shape=shapes[i], seed=seeds[i], add_structure=True
        )
        image_paths.append(img_path)

        # Create label if requested
        if add_labels:
            label_path = output_dir / f"label_{i:03d}.nii"
            # Generate binary label with some structures
            label_img = generate_random_nifti(
                shape=shapes[i],
                seed=seeds[i] + 1000,  # Different seed for labels
                dtype=np.uint8,
                add_structure=True,
            )
            # Convert to binary labels
            label_data = (label_img.get_fdata() > 0.5).astype(np.uint8)
            label_img = nib.Nifti1Image(label_data, label_img.affine)
            nib.save(label_img, str(label_path))
            label_paths.append(str(label_path))

    return image_paths, label_paths


def compare_nifti_images(img1_path: str, img2_path: str, tolerance: float = 1e-5) -> bool:
    """
    Compare two NIfTI images for equality within a tolerance.

    Args:
        img1_path: Path to first image
        img2_path: Path to second image
        tolerance: Numerical tolerance for comparison

    Returns:
        True if images are approximately equal
    """
    img1 = nib.load(img1_path)
    img2 = nib.load(img2_path)

    # Check shapes
    if img1.shape != img2.shape:
        return False

    # Check affine matrices
    if not np.allclose(img1.affine, img2.affine, atol=tolerance):
        return False

    # Check data
    data1 = img1.get_fdata()
    data2 = img2.get_fdata()

    return np.allclose(data1, data2, atol=tolerance)


class TempNiftiFile:
    """Context manager for temporary NIfTI files."""

    def __init__(
        self,
        shape: Tuple[int, ...] = (64, 64, 32),
        affine: Optional[np.ndarray] = None,
        dtype: np.dtype = np.float32,
        seed: Optional[int] = None,
        add_structure: bool = False,
        compressed: bool = False,
    ):
        """
        Initialize temporary NIfTI file context manager.

        Args:
            shape: Shape of the image
            affine: 4x4 affine transformation matrix
            dtype: Data type of the image
            seed: Random seed for reproducibility
            add_structure: Whether to add structured patterns
            compressed: Whether to compress the file
        """
        self.shape = shape
        self.affine = affine
        self.dtype = dtype
        self.seed = seed
        self.add_structure = add_structure
        self.compressed = compressed
        self.temp_file = None
        self.path = None

    def __enter__(self) -> str:
        """Create temporary file and return path."""
        self.temp_file = tempfile.NamedTemporaryFile(
            suffix=".nii.gz" if self.compressed else ".nii", delete=False
        )
        self.path = self.temp_file.name
        self.temp_file.close()

        create_test_nifti_file(
            self.path,
            shape=self.shape,
            affine=self.affine,
            dtype=self.dtype,
            seed=self.seed,
            add_structure=self.add_structure,
            compressed=self.compressed,
        )

        return self.path

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up temporary file."""
        if self.path and os.path.exists(self.path):
            os.unlink(self.path)


# Convenience functions for common test scenarios
def create_3d_test_file(
    shape: Tuple[int, int, int] = (64, 64, 32), seed: int = 42
) -> TempNiftiFile:
    """Create a 3D test file context manager."""
    return TempNiftiFile(shape=shape, seed=seed, add_structure=True)


def create_4d_test_file(
    shape: Tuple[int, int, int, int] = (10, 64, 64, 32), seed: int = 42
) -> TempNiftiFile:
    """Create a 4D test file context manager."""
    return TempNiftiFile(shape=shape, seed=seed, add_structure=True)


def create_small_test_file() -> TempNiftiFile:
    """Create a small test file for quick tests."""
    return TempNiftiFile(shape=(16, 16, 8), seed=123, add_structure=False)


def create_large_test_file() -> TempNiftiFile:
    """Create a large test file for performance tests."""
    return TempNiftiFile(shape=(256, 256, 128), seed=456, add_structure=True)
