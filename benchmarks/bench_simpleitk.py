#!/usr/bin/env python3
"""
Standalone benchmark for SimpleITK library (C++ backend).

Run: python benchmarks/bench_simpleitk.py [--quick|--full]
Output: benchmarks/results/simpleitk_results.json
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from common import (
    BenchmarkConfig,
    BenchmarkRunner,
    create_test_volume,
    save_nifti,
)

import numpy as np

try:
    import SimpleITK as sitk

    HAS_SIMPLEITK = True
except ImportError:
    HAS_SIMPLEITK = False
    print("ERROR: SimpleITK not installed. Run: pip install SimpleITK")
    sys.exit(1)


class SimpleITKBenchmarkRunner(BenchmarkRunner):
    """Benchmark runner for SimpleITK library."""

    library_name = "simpleitk"

    def run_all(self):
        """Run all SimpleITK benchmarks."""
        print("\nRunning SimpleITK benchmarks...")
        print(f"  Sizes: {self.config.sizes}")
        print(
            f"  Warmup: {self.config.warmup_iterations}, Iterations: {self.config.benchmark_iterations}"
        )

        for dtype in self.config.dtypes:
            for size in self.config.sizes:
                self._run_size_benchmarks(size, dtype)

        return self.results

    def _run_size_benchmarks(self, size, dtype):
        """Run benchmarks for a specific size."""
        print(f"\n  Size {size[0]}x{size[1]}x{size[2]} ({dtype}):")

        test_file = self.get_test_file(size, dtype)
        test_file_gz = self.get_test_file_gzipped(size, dtype)

        self._bench_load(test_file, size, dtype)
        self._bench_load_gzipped(test_file_gz, size, dtype)
        self._bench_load_to_numpy(test_file, size, dtype)
        self._bench_load_to_numpy_gzipped(test_file_gz, size, dtype)
        self._bench_load_cropped(test_file, size, dtype)
        self._bench_load_cropped_gzipped(test_file_gz, size, dtype)
        self._bench_load_resampled(test_file, size, dtype)
        self._bench_load_resampled_gzipped(test_file_gz, size, dtype)
        self._bench_save(test_file, size, dtype)
        self._bench_save_gzipped(test_file, size, dtype)

    def _bench_load(self, test_file, size, dtype):
        """Benchmark basic load."""

        def load_op():
            return sitk.ReadImage(test_file)

        result = self.run_benchmark("load", load_op, size, dtype)
        print(f"    load: {result.median_ms:.2f}ms")

    def _bench_load_gzipped(self, test_file, size, dtype):
        """Benchmark load of gzipped file."""

        def load_op():
            return sitk.ReadImage(test_file)

        result = self.run_benchmark("load_gzipped", load_op, size, dtype, notes="compressed")
        print(f"    load_gzipped: {result.median_ms:.2f}ms")

    def _bench_load_to_numpy(self, test_file, size, dtype):
        """Benchmark full load to numpy array."""

        def load_op():
            img = sitk.ReadImage(test_file)
            return sitk.GetArrayFromImage(img)

        result = self.run_benchmark("load_to_numpy", load_op, size, dtype)
        print(f"    load_to_numpy: {result.median_ms:.2f}ms")

    def _bench_load_to_numpy_gzipped(self, test_file, size, dtype):
        """Benchmark full load to numpy for gzipped file."""

        def load_op():
            img = sitk.ReadImage(test_file)
            return sitk.GetArrayFromImage(img)

        result = self.run_benchmark(
            "load_to_numpy_gzipped", load_op, size, dtype, notes="compressed"
        )
        print(f"    load_to_numpy_gzipped: {result.median_ms:.2f}ms")

    def _bench_load_cropped(self, test_file, size, dtype):
        """Benchmark load then crop."""
        crop_size = self.config.crop_size
        if any(c > s for c, s in zip(crop_size, size)):
            return

        crop_offset = [(s - c) // 2 for s, c in zip(size, crop_size)]

        def load_cropped_op():
            img = sitk.ReadImage(test_file)
            return sitk.RegionOfInterest(
                img,
                size=list(crop_size),
                index=list(crop_offset),
            )

        result = self.run_benchmark(
            "load_cropped", load_cropped_op, size, dtype, notes=f"crop={crop_size}"
        )
        print(f"    load_cropped: {result.median_ms:.2f}ms")

    def _bench_load_cropped_gzipped(self, test_file, size, dtype):
        """Benchmark load then crop for gzipped file."""
        crop_size = self.config.crop_size
        if any(c > s for c, s in zip(crop_size, size)):
            return

        crop_offset = [(s - c) // 2 for s, c in zip(size, crop_size)]

        def load_cropped_op():
            img = sitk.ReadImage(test_file)
            return sitk.RegionOfInterest(
                img,
                size=list(crop_size),
                index=list(crop_offset),
            )

        result = self.run_benchmark(
            "load_cropped_gzipped",
            load_cropped_op,
            size,
            dtype,
            notes=f"crop={crop_size},compressed",
        )
        print(f"    load_cropped_gzipped: {result.median_ms:.2f}ms")

    def _bench_load_resampled(self, test_file, size, dtype):
        """Benchmark load and resample to half resolution."""
        target_size = [s // 2 for s in size]

        def load_resampled_op():
            img = sitk.ReadImage(test_file)
            resampler = sitk.ResampleImageFilter()
            resampler.SetSize(target_size)
            resampler.SetOutputSpacing([2.0, 2.0, 2.0])
            resampler.SetInterpolator(sitk.sitkLinear)
            return resampler.Execute(img)

        result = self.run_benchmark(
            "load_resampled", load_resampled_op, size, dtype, notes=f"target={target_size}"
        )
        print(f"    load_resampled: {result.median_ms:.2f}ms")

    def _bench_load_resampled_gzipped(self, test_file, size, dtype):
        """Benchmark load and resample for gzipped file."""
        target_size = [s // 2 for s in size]

        def load_resampled_op():
            img = sitk.ReadImage(test_file)
            resampler = sitk.ResampleImageFilter()
            resampler.SetSize(target_size)
            resampler.SetOutputSpacing([2.0, 2.0, 2.0])
            resampler.SetInterpolator(sitk.sitkLinear)
            return resampler.Execute(img)

        result = self.run_benchmark(
            "load_resampled_gzipped",
            load_resampled_op,
            size,
            dtype,
            notes=f"target={target_size},compressed",
        )
        print(f"    load_resampled_gzipped: {result.median_ms:.2f}ms")

    def _bench_save(self, test_file, size, dtype):
        """Benchmark save to uncompressed file."""
        img = sitk.ReadImage(test_file)
        output_path = os.path.join(self.tmpdir, f"output_{size[0]}.nii")

        def save_op():
            sitk.WriteImage(img, output_path)

        result = self.run_benchmark("save", save_op, size, dtype)
        print(f"    save: {result.median_ms:.2f}ms")

    def _bench_save_gzipped(self, test_file, size, dtype):
        """Benchmark save to gzipped file."""
        img = sitk.ReadImage(test_file)
        output_path = os.path.join(self.tmpdir, f"output_{size[0]}.nii.gz")

        def save_op():
            sitk.WriteImage(img, output_path)

        result = self.run_benchmark("save_gzipped", save_op, size, dtype, notes="compressed")
        print(f"    save_gzipped: {result.median_ms:.2f}ms")


def main():
    parser = argparse.ArgumentParser(description="Benchmark SimpleITK library")
    parser.add_argument("--quick", action="store_true", help="Quick benchmark (smaller sizes)")
    parser.add_argument("--full", action="store_true", help="Full benchmark (larger sizes)")
    args = parser.parse_args()

    if args.quick:
        config = BenchmarkConfig.quick()
    elif args.full:
        config = BenchmarkConfig.full()
    else:
        config = BenchmarkConfig()

    runner = SimpleITKBenchmarkRunner(config)
    try:
        runner.run_all()
        runner.print_results()

        results_dir = Path(__file__).parent / "results"
        results_dir.mkdir(exist_ok=True)
        runner.save_results(str(results_dir / "simpleitk_results.json"))
    finally:
        runner.cleanup()


if __name__ == "__main__":
    main()
