#!/usr/bin/env python3
"""
Standalone benchmark for medrs library.

Run: python benchmarks/bench_medrs.py [--quick|--full]
Output: benchmarks/results/medrs_results.json
"""

import argparse
import os
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from common import (
    BenchmarkConfig,
    BenchmarkRunner,
    create_test_volume,
    create_label_volume,
    save_nifti,
)

import numpy as np

try:
    import medrs

    HAS_MEDRS = True
except ImportError:
    HAS_MEDRS = False
    print("ERROR: medrs not installed. Run: maturin develop --release")
    sys.exit(1)

HAS_MGZIP = hasattr(medrs, "convert_to_mgzip") and hasattr(medrs, "load_mgzip")


class MedrsBenchmarkRunner(BenchmarkRunner):
    """Benchmark runner for medrs library."""

    library_name = "medrs"

    def run_all(self):
        """Run all medrs benchmarks."""
        print("\nRunning medrs benchmarks...")
        print(f"  Sizes: {self.config.sizes}")
        print(
            f"  Warmup: {self.config.warmup_iterations}, Iterations: {self.config.benchmark_iterations}"
        )

        for dtype in self.config.dtypes:
            for size in self.config.sizes:
                self._run_size_benchmarks(size, dtype)

        return self.results

    def _run_size_benchmarks(self, size, dtype):
        print(f"\n  Size {size[0]}x{size[1]}x{size[2]} ({dtype}):")

        test_file = self.get_test_file(size, dtype)
        test_file_gz = self.get_test_file_gzipped(size, dtype)

        self._bench_load(test_file, size, dtype)
        self._bench_load_gzipped(test_file_gz, size, dtype)

        if HAS_MGZIP:
            test_file_mgz = self._get_or_create_mgzip(test_file_gz, size, dtype)
            self._bench_load_mgzip(test_file_mgz, size, dtype)

        self._bench_load_cropped(test_file, size, dtype)
        self._bench_load_cropped_gzipped(test_file_gz, size, dtype)
        self._bench_load_resampled(test_file, size, dtype)
        self._bench_load_resampled_gzipped(test_file_gz, size, dtype)
        self._bench_load_to_torch(test_file, size, dtype)
        self._bench_load_to_torch_gzipped(test_file_gz, size, dtype)
        self._bench_intensity_normalize(test_file, size, dtype)
        self._bench_intensity_normalize_gzipped(test_file_gz, size, dtype)

    def _get_or_create_mgzip(self, gzip_file, size, dtype):
        """Get or create mgzip version of test file."""
        mgzip_file = gzip_file.replace(".nii.gz", ".mgz.nii.gz")
        if not os.path.exists(mgzip_file):
            medrs.convert_to_mgzip(gzip_file, mgzip_file)
        return mgzip_file

    def _bench_load_mgzip(self, test_file, size, dtype):
        """Benchmark load_mgzip with parallel decompression."""
        for num_threads in [1, 4, 8]:

            def load_op(threads=num_threads):
                return medrs.load_mgzip(test_file, num_threads=threads)

            result = self.run_benchmark(
                f"load_mgzip_{num_threads}t",
                load_op,
                size,
                dtype,
                notes=f"mgzip,threads={num_threads}",
            )
            print(f"    load_mgzip ({num_threads} threads): {result.median_ms:.2f}ms")

    def _bench_load(self, test_file, size, dtype):
        """Benchmark basic load."""

        def load_op():
            return medrs.load(test_file)

        result = self.run_benchmark("load", load_op, size, dtype)
        print(f"    load: {result.median_ms:.2f}ms")

    def _bench_load_gzipped(self, test_file, size, dtype):
        """Benchmark load of gzipped .nii.gz file."""

        def load_op():
            return medrs.load(test_file)

        result = self.run_benchmark("load_gzipped", load_op, size, dtype, notes="compressed")
        print(f"    load_gzipped: {result.median_ms:.2f}ms")

    def _bench_load_cropped(self, test_file, size, dtype):
        """Benchmark load_cropped (exact crop, no resampling)."""
        crop_size = self.config.crop_size

        # Skip if crop is larger than volume
        if any(c > s for c, s in zip(crop_size, size)):
            return

        # Compute center offset for crop
        crop_offset = [(s - c) // 2 for s, c in zip(size, crop_size)]

        def load_cropped_op():
            return medrs.load_cropped(
                test_file,
                crop_offset=crop_offset,
                crop_shape=list(crop_size),
            )

        result = self.run_benchmark(
            "load_cropped", load_cropped_op, size, dtype, notes=f"crop={crop_size}"
        )
        print(f"    load_cropped: {result.median_ms:.2f}ms")

    def _bench_load_cropped_gzipped(self, test_file, size, dtype):
        crop_size = self.config.crop_size
        if any(c > s for c, s in zip(crop_size, size)):
            return

        crop_offset = [(s - c) // 2 for s, c in zip(size, crop_size)]

        def load_cropped_op():
            img = medrs.load(test_file)
            arr = img.to_numpy()
            return arr[
                crop_offset[0] : crop_offset[0] + crop_size[0],
                crop_offset[1] : crop_offset[1] + crop_size[1],
                crop_offset[2] : crop_offset[2] + crop_size[2],
            ]

        result = self.run_benchmark(
            "load_cropped_gzipped",
            load_cropped_op,
            size,
            dtype,
            notes=f"crop={crop_size},compressed",
        )
        print(f"    load_cropped_gzipped: {result.median_ms:.2f}ms")

    def _bench_load_resampled(self, test_file, size, dtype):
        # Resample to half resolution
        output_shape = [s // 2 for s in size]
        if any(s < 16 for s in output_shape):
            return

        def load_resampled_op():
            return medrs.load_resampled(
                test_file,
                output_shape=output_shape,
            )

        result = self.run_benchmark(
            "load_resampled", load_resampled_op, size, dtype, notes=f"target={output_shape}"
        )
        print(f"    load_resampled: {result.median_ms:.2f}ms")

    def _bench_load_resampled_gzipped(self, test_file, size, dtype):
        output_shape = [s // 2 for s in size]
        if any(s < 16 for s in output_shape):
            return

        def load_resampled_op():
            img = medrs.load(test_file)
            return img.resample_to_shape(output_shape)

        result = self.run_benchmark(
            "load_resampled_gzipped",
            load_resampled_op,
            size,
            dtype,
            notes=f"target={output_shape},compressed",
        )
        print(f"    load_resampled_gzipped: {result.median_ms:.2f}ms")

    def _bench_load_to_torch(self, test_file, size, dtype):
        try:
            import torch
        except ImportError:
            return

        crop_size = self.config.crop_size
        if any(c > s for c, s in zip(crop_size, size)):
            return

        def load_to_torch_op():
            return medrs.load_cropped_to_torch(
                path=test_file,
                output_shape=list(crop_size),
                dtype=torch.float32,
                device="cpu",
            )

        result = self.run_benchmark(
            "load_cropped_to_torch", load_to_torch_op, size, dtype, notes=f"crop={crop_size}"
        )
        print(f"    load_cropped_to_torch: {result.median_ms:.2f}ms")

    def _bench_load_to_torch_gzipped(self, test_file, size, dtype):
        try:
            import torch
        except ImportError:
            return

        crop_size = self.config.crop_size
        if any(c > s for c, s in zip(crop_size, size)):
            return

        def load_to_torch_op():
            img = medrs.load(test_file)
            cropped = img.crop_or_pad(list(crop_size))
            return cropped.to_torch()

        result = self.run_benchmark(
            "load_cropped_to_torch_gzipped",
            load_to_torch_op,
            size,
            dtype,
            notes=f"crop={crop_size},compressed",
        )
        print(f"    load_cropped_to_torch_gzipped: {result.median_ms:.2f}ms")

    def _bench_intensity_normalize(self, test_file, size, dtype):
        crop_size = self.config.crop_size
        if any(c > s for c, s in zip(crop_size, size)):
            return

        # Compute center offset for crop
        crop_offset = [(s - c) // 2 for s, c in zip(size, crop_size)]

        def load_normalized_op():
            img = medrs.load_cropped(
                test_file,
                crop_offset=crop_offset,
                crop_shape=list(crop_size),
            )
            # Apply z-normalization
            return medrs.z_normalization(img)

        result = self.run_benchmark(
            "load_cropped_normalized",
            load_normalized_op,
            size,
            dtype,
            notes=f"crop={crop_size},normalize=z_norm",
        )
        print(f"    load_cropped_normalized: {result.median_ms:.2f}ms")

    def _bench_intensity_normalize_gzipped(self, test_file, size, dtype):
        crop_size = self.config.crop_size
        if any(c > s for c, s in zip(crop_size, size)):
            return

        def load_normalized_op():
            img = medrs.load(test_file)
            cropped = img.crop_or_pad(list(crop_size))
            return medrs.z_normalization(cropped)

        result = self.run_benchmark(
            "load_cropped_normalized_gzipped",
            load_normalized_op,
            size,
            dtype,
            notes=f"crop={crop_size},normalize=z_norm,compressed",
        )
        print(f"    load_cropped_normalized_gzipped: {result.median_ms:.2f}ms")


def main():
    parser = argparse.ArgumentParser(description="Benchmark medrs library")
    parser.add_argument("--quick", action="store_true", help="Run quick benchmark")
    parser.add_argument("--full", action="store_true", help="Run full benchmark")
    parser.add_argument("--output", "-o", type=str, help="Output file path")
    args = parser.parse_args()

    # Select config
    if args.full:
        config = BenchmarkConfig.full()
    elif args.quick:
        config = BenchmarkConfig.quick()
    else:
        config = BenchmarkConfig()

    # Create output directory
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)

    # Run benchmarks
    runner = MedrsBenchmarkRunner(config)
    try:
        runner.run_all()
        runner.print_results()

        # Save results
        output_path = args.output or str(output_dir / "medrs_results.json")
        runner.save_results(output_path)
    finally:
        runner.cleanup()


if __name__ == "__main__":
    main()
