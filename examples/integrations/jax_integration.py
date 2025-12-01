#!/usr/bin/env python3
"""
medrs + JAX Integration
======================

This example demonstrates how to use medrs with JAX for
high-performance medical image processing and deep learning.
"""

import jax
import jax.numpy as jnp
import time
import numpy as np
import medrs


def demonstrate_jax_loading():
    """Demonstrate medrs to JAX array conversion."""

    print(" medrs + JAX Integration Example")
    print("=" * 40)

    # Replace with your NIfTI file
    volume_path = "sample_volume.nii.gz"

    try:
        print("\n1. Direct JAX array loading:")
        start_time = time.time()

        # Load directly to JAX array
        jax_array = medrs.load_cropped_to_jax(
            volume_path,
            output_shape=[64, 64, 64],
            dtype=jnp.float32
        )

        load_time = time.time() - start_time

        print(f"    Shape: {jax_array.shape}")
        print(f"    Dtype: {jax_array.dtype}")
        print(f"    Device: {jax_array.devices()[0]}")
        print(f"     Load time: {load_time:.4f}s")

        # JAX operations
        print("\n2. JAX operations:")
        start_time = time.time()

        # Normalize
        normalized = (jax_array - jnp.mean(jax_array)) / (jnp.std(jax_array) + 1e-8)

        # Apply convolution-like operation
        kernel = jnp.ones((3, 3, 3)) / 27
        smoothed = jax.lax.conv_general_dilated(
            normalized[jnp.newaxis, jnp.newaxis, ...],
            kernel[jnp.newaxis, jnp.newaxis, ...],
            window_strides=[1, 1, 1],
            padding='SAME'
        ).squeeze()

        jax_time = time.time() - start_time

        print(f"     JAX processing time: {jax_time:.4f}s")
        print(f"    Normalized mean: {jnp.mean(normalized):.6f}")
        print(f"    Smoothed range: [{jnp.min(smoothed):.3f}, {jnp.max(smoothed):.3f}]")

        # JIT compilation
        print("\n3. JIT Compilation:")

        @jax.jit
        def process_jax_array(arr):
            """JIT-compiled processing function."""
            normalized = (arr - jnp.mean(arr)) / (jnp.std(arr) + 1e-8)
            # Simple edge detection
            sobel_x = jnp.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=jnp.float32)
            edges = jax.lax.conv_general_dilated(
                normalized[jnp.newaxis, jnp.newaxis, ...],
                sobel_x[jnp.newaxis, jnp.newaxis, jnp.newaxis, :],
                window_strides=[1, 1, 1],
                padding='SAME'
            ).squeeze()
            return jnp.abs(edges)

        # Warm up
        _ = process_jax_array(jax_array)

        # Time JIT version
        start_time = time.time()
        for _ in range(100):
            edges = process_jax_array(jax_array)
        jit_time = (time.time() - start_time) / 100

        print(f"    JIT time per call: {jit_time:.6f}s")
        print(f"    Edge range: [{jnp.min(edges):.3f}, {jnp.max(edges):.3f}]")

        # Batch processing
        print("\n4. Batch Processing:")

        def load_batch(volume_paths):
            """Load a batch of volumes using JAX."""
            arrays = []
            for path in volume_paths[:5]:  # Small batch for demo
                arr = medrs.load_cropped_to_jax(path, [32, 32, 32])
                arrays.append(arr)
            return jnp.stack(arrays)

        batch_paths = [volume_path] * 5  # Use same path for demo
        start_time = time.time()
        batch_array = load_batch(batch_paths)
        batch_time = time.time() - start_time

        print(f"    Batch shape: {batch_array.shape}")
        print(f"     Batch load time: {batch_time:.4f}s")
        print(f"    Average per volume: {batch_time/5:.4f}s")

        # Performance comparison
        print("\n5. Performance Comparison:")

        # Compare with NumPy
        numpy_array = medrs.load_cropped(volume_path, [0, 0, 0], [64, 64, 64]).to_numpy()
        numpy_time = time.time()
        numpy_normalized = (numpy_array - np.mean(numpy_array)) / (np.std(numpy_array) + 1e-8)
        numpy_time = time.time() - numpy_time

        print(f"    NumPy normalization: {numpy_time:.6f}s")
        print(f"    JAX normalization: {jax_time:.6f}s")
        print(f"    JAX speedup: {numpy_time/jax_time:.1f}x")

    except FileNotFoundError:
        print(f" File not found: {volume_path}")
        print("Replace with a valid NIfTI file path")
    except Exception as e:
        print(f" Error: {e}")


def demonstrate_jax_vmap():
    """Demonstrate JAX vmap for vectorized operations."""

    print("\n JAX vmap Demonstration:")
    print("-" * 30)

    # Create dummy batch of patches
    batch_size = 8
    patch_shape = (32, 32, 32)

    print("\n1. Vectorized Processing:")
    print(f"    Batch size: {batch_size}")
    print(f"    Patch shape: {patch_shape}")

    # Create mock data (would normally load with medrs)
    mock_batch = jnp.ones((batch_size,) + patch_shape)
    print(f"    Batch shape: {mock_batch.shape}")

    @jax.jit
    def normalize_patch(patch):
        """Normalize a single patch."""
        return (patch - jnp.mean(patch)) / (jnp.std(patch) + 1e-8)

    # Vectorized version
    normalize_batch = jax.vmap(normalize_patch, in_axes=0)

    # Time vectorized processing
    start_time = time.time()
    normalized_batch = normalize_batch(mock_batch)
    vectorized_time = time.time() - start_time

    # Non-vectorized version for comparison
    start_time = time.time()
    normalized_individual = jnp.stack([normalize_patch(patch) for patch in mock_batch])
    individual_time = time.time() - start_time

    print(f"    Vectorized time: {vectorized_time:.6f}s")
    print(f"    Individual time: {individual_time:.6f}s")
    print(f"    Speedup: {individual_time/vectorized_time:.1f}x")


def demonstrate_gpu_acceleration():
    """Demonstrate GPU acceleration with JAX."""

    print("\n GPU Acceleration:")
    print("-" * 20)

    devices = jax.devices()
    print(f"     Available devices: {[str(d) for d in devices]}")

    if any('gpu' in str(device).lower() for device in devices):
        print("    GPU acceleration available")

        # Transfer to GPU
        gpu_array = jnp.ones((64, 64, 64))
        print(f"    Array on device: {gpu_array.devices()[0]}")

        # GPU computation
        @jax.jit
        def gpu_computation(arr):
            return jnp.fft.fftn(arr)

        start_time = time.time()
        result = gpu_computation(gpu_array)
        gpu_time = time.time() - start_time

        print(f"    GPU FFT time: {gpu_time:.4f}s")
    else:
        print("     GPU not available, using CPU")


if __name__ == "__main__":
    demonstrate_jax_loading()
    demonstrate_jax_vmap()
    demonstrate_gpu_acceleration()

    print("\n JAX integration example completed!")
    print("\n JAX Benefits:")
    print("   - Just-in-time compilation for maximum speed")
    print("   - Automatic vectorization with vmap")
    print("   - XLA optimization across devices")
    print("   - Seamless GPU/TPU acceleration")
    print("   - Functional programming paradigm")
