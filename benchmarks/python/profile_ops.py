#!/usr/bin/env python3
"""
Profile medrs operations at small (32^3) and large (512^3) volumes.

Usage:
    uv run python profiling/profile_ops.py

For detailed profiling with py-spy:
    py-spy record -o profile.svg -- python profiling/profile_ops.py --size 256 --op z_norm

For cProfile:
    python -m cProfile -s cumtime profiling/profile_ops.py --size 256 --op z_norm
"""

import argparse
import time
import statistics
import numpy as np
import medrs


def create_image(size: int) -> medrs.NiftiImage:
    """Create a test image of given size."""
    data = np.random.rand(size, size, size).astype(np.float32) * 255
    affine = np.eye(4) * 2.0
    affine[3, 3] = 1.0
    return medrs.NiftiImage(data, affine.tolist())


def benchmark(name: str, func, iterations: int = 20) -> dict:
    """Run benchmark and return stats."""
    # Warmup
    func()

    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        func()
        times.append((time.perf_counter() - start) * 1000)  # ms

    return {
        "name": name,
        "mean": statistics.mean(times),
        "median": statistics.median(times),
        "stdev": statistics.stdev(times) if len(times) > 1 else 0,
        "min": min(times),
        "max": max(times),
    }


def print_result(result: dict, mb: float):
    """Print benchmark result."""
    throughput = mb / (result["mean"] / 1000) if result["mean"] > 0 else 0
    print(f"  {result['name']:25} "
          f"mean: {result['mean']:8.3f} ms  "
          f"median: {result['median']:8.3f} ms  "
          f"({throughput:.1f} MB/s)")


def profile_z_norm(img: medrs.NiftiImage, iterations: int) -> dict:
    return benchmark("z_normalization", lambda: medrs.z_normalization(img), iterations)


def profile_rescale(img: medrs.NiftiImage, iterations: int) -> dict:
    return benchmark("rescale_intensity", lambda: medrs.rescale_intensity(img, 0.0, 1.0), iterations)


def profile_crop(img: medrs.NiftiImage, size: int, iterations: int) -> dict:
    target = [size // 2] * 3
    return benchmark("crop_or_pad (half)", lambda: medrs.crop_or_pad(img, target), iterations)


def profile_crop_expand(img: medrs.NiftiImage, size: int, iterations: int) -> dict:
    target = [size * 2] * 3
    return benchmark("crop_or_pad (double)", lambda: medrs.crop_or_pad(img, target), iterations)


def profile_flip(img: medrs.NiftiImage, iterations: int) -> dict:
    return benchmark("flip (axis 0)", lambda: medrs.flip(img, [0]), iterations)


def profile_flip_all(img: medrs.NiftiImage, iterations: int) -> dict:
    return benchmark("flip (all axes)", lambda: medrs.flip(img, [0, 1, 2]), iterations)


def profile_resample(img: medrs.NiftiImage, iterations: int) -> dict:
    return benchmark("resample", lambda: medrs.resample(img, [1.0, 1.0, 1.0]), iterations)


def profile_all(size: int, iterations: int):
    """Profile all operations at given size."""
    mb = (size ** 3 * 4) / 1_000_000

    print(f"\n{'=' * 60}")
    print(f"  Volume: {size}^3 ({mb:.1f} MB)  |  {iterations} iterations")
    print(f"{'=' * 60}\n")

    img = create_image(size)

    print("Intensity transforms:")
    print_result(profile_z_norm(img, iterations), mb)
    print_result(profile_rescale(img, iterations), mb)

    print("\nSpatial transforms:")
    print_result(profile_crop(img, size, iterations), mb)
    print_result(profile_crop_expand(img, size, iterations), mb)
    print_result(profile_flip(img, iterations), mb)
    print_result(profile_flip_all(img, iterations), mb)

    # Resample is expensive
    resample_iters = max(1, iterations // 4) if size > 128 else iterations
    print(f"\nResample ({resample_iters} iterations):")
    print_result(profile_resample(img, resample_iters), mb)


def profile_single(op: str, size: int, iterations: int):
    """Profile a single operation (for detailed profiling)."""
    mb = (size ** 3 * 4) / 1_000_000
    print(f"Profiling {op} at {size}^3 ({mb:.1f} MB), {iterations} iterations\n")

    img = create_image(size)

    ops = {
        "z_norm": lambda: profile_z_norm(img, iterations),
        "rescale": lambda: profile_rescale(img, iterations),
        "crop": lambda: profile_crop(img, size, iterations),
        "crop_expand": lambda: profile_crop_expand(img, size, iterations),
        "flip": lambda: profile_flip(img, iterations),
        "flip_all": lambda: profile_flip_all(img, iterations),
        "resample": lambda: profile_resample(img, iterations),
    }

    if op not in ops:
        print(f"Unknown op: {op}. Available: {list(ops.keys())}")
        return

    result = ops[op]()
    print_result(result, mb)


def main():
    parser = argparse.ArgumentParser(description="Profile medrs operations")
    parser.add_argument("--size", type=int, help="Single size to profile")
    parser.add_argument("--op", type=str, help="Single operation to profile")
    parser.add_argument("--iterations", type=int, default=20, help="Number of iterations")
    parser.add_argument("--large", action="store_true", help="Include 512^3 volume")
    args = parser.parse_args()

    print("medrs Python profiling")
    print("======================\n")

    if args.size and args.op:
        # Single operation mode (for detailed profiling)
        profile_single(args.op, args.size, args.iterations)
    else:
        # Full profile
        profile_all(32, 100)
        profile_all(128, 20)
        profile_all(256, 10)

        if args.large:
            profile_all(512, 5)
        else:
            print("\n(Use --large flag to also profile 512^3 volumes)")


if __name__ == "__main__":
    main()
