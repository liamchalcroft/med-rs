#!/usr/bin/env python3
"""
Standalone benchmark for nibabel library (baseline).

Run: python benchmarks/bench_nibabel.py [--quick|--full]
Output: benchmarks/results/nibabel_results.json
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
    import nibabel as nib

    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False
    print("ERROR: nibabel not installed. Run: pip install nibabel")
    sys.exit(1)


class NibabelBenchmarkRunner(BenchmarkRunner):
    """Benchmark runner for nibabel library."""

    library_name = "nibabel"

    def run_all(self):
        """Run all nibabel benchmarks."""
        print("\nRunning nibabel benchmarks...")
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
        self._bench_load_header_only(test_file, size, dtype)
        self._bench_load_header_only_gzipped(test_file_gz, size, dtype)
        self._bench_load_to_numpy(test_file, size, dtype)
        self._bench_load_to_numpy_gzipped(test_file_gz, size, dtype)
        self._bench_load_cropped(test_file, size, dtype)
        self._bench_load_cropped_gzipped(test_file_gz, size, dtype)
        self._bench_save(test_file, size, dtype)
        self._bench_save_gzipped(test_file, size, dtype)

    def _bench_load(self, test_file, size, dtype):
        """Benchmark basic load (returns proxy, no data loaded)."""

        def load_op():
            return nib.load(test_file)

        result = self.run_benchmark("load_proxy", load_op, size, dtype, notes="proxy only")
        print(f"    load_proxy: {result.median_ms:.2f}ms")

    def _bench_load_gzipped(self, test_file, size, dtype):
        """Benchmark load of gzipped file (proxy only)."""

        def load_op():
            return nib.load(test_file)

        result = self.run_benchmark(
            "load_proxy_gzipped", load_op, size, dtype, notes="proxy,compressed"
        )
        print(f"    load_proxy_gzipped: {result.median_ms:.2f}ms")

    def _bench_load_header_only(self, test_file, size, dtype):
        """Benchmark header-only load."""

        def load_op():
            img = nib.load(test_file)
            return img.header

        result = self.run_benchmark("load_header", load_op, size, dtype)
        print(f"    load_header: {result.median_ms:.2f}ms")

    def _bench_load_header_only_gzipped(self, test_file, size, dtype):
        """Benchmark header-only load for gzipped file."""

        def load_op():
            img = nib.load(test_file)
            return img.header

        result = self.run_benchmark("load_header_gzipped", load_op, size, dtype, notes="compressed")
        print(f"    load_header_gzipped: {result.median_ms:.2f}ms")

    def _bench_load_to_numpy(self, test_file, size, dtype):
        """Benchmark full load to numpy array."""

        def load_op():
            img = nib.load(test_file)
            return np.asarray(img.dataobj)

        result = self.run_benchmark("load", load_op, size, dtype, notes="full load")
        print(f"    load: {result.median_ms:.2f}ms")

    def _bench_load_to_numpy_gzipped(self, test_file, size, dtype):
        """Benchmark full load to numpy for gzipped file."""

        def load_op():
            img = nib.load(test_file)
            return np.asarray(img.dataobj)

        result = self.run_benchmark(
            "load_gzipped", load_op, size, dtype, notes="full load,compressed"
        )
        print(f"    load_gzipped: {result.median_ms:.2f}ms")

    def _bench_load_cropped(self, test_file, size, dtype):
        """Benchmark load then crop (nibabel must load full volume)."""
        crop_size = self.config.crop_size
        if any(c > s for c, s in zip(crop_size, size)):
            return

        crop_offset = [(s - c) // 2 for s, c in zip(size, crop_size)]

        def load_cropped_op():
            img = nib.load(test_file)
            data = np.asarray(img.dataobj)
            return data[
                crop_offset[0] : crop_offset[0] + crop_size[0],
                crop_offset[1] : crop_offset[1] + crop_size[1],
                crop_offset[2] : crop_offset[2] + crop_size[2],
            ]

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
            img = nib.load(test_file)
            data = np.asarray(img.dataobj)
            return data[
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

    def _bench_save(self, test_file, size, dtype):
        """Benchmark save to uncompressed file."""
        img = nib.load(test_file)
        data = np.asarray(img.dataobj)
        affine = img.affine
        output_path = os.path.join(self.tmpdir, f"output_{size[0]}.nii")

        def save_op():
            out_img = nib.Nifti1Image(data, affine)
            nib.save(out_img, output_path)

        result = self.run_benchmark("save", save_op, size, dtype)
        print(f"    save: {result.median_ms:.2f}ms")

    def _bench_save_gzipped(self, test_file, size, dtype):
        """Benchmark save to gzipped file."""
        img = nib.load(test_file)
        data = np.asarray(img.dataobj)
        affine = img.affine
        output_path = os.path.join(self.tmpdir, f"output_{size[0]}.nii.gz")

        def save_op():
            out_img = nib.Nifti1Image(data, affine)
            nib.save(out_img, output_path)

        result = self.run_benchmark("save_gzipped", save_op, size, dtype, notes="compressed")
        print(f"    save_gzipped: {result.median_ms:.2f}ms")


def main():
    parser = argparse.ArgumentParser(description="Benchmark nibabel library")
    parser.add_argument("--quick", action="store_true", help="Quick benchmark (smaller sizes)")
    parser.add_argument("--full", action="store_true", help="Full benchmark (larger sizes)")
    args = parser.parse_args()

    if args.quick:
        config = BenchmarkConfig.quick()
    elif args.full:
        config = BenchmarkConfig.full()
    else:
        config = BenchmarkConfig()

    runner = NibabelBenchmarkRunner(config)
    try:
        runner.run_all()
        runner.print_results()

        results_dir = Path(__file__).parent / "results"
        results_dir.mkdir(exist_ok=True)
        runner.save_results(str(results_dir / "nibabel_results.json"))
    finally:
        runner.cleanup()


if __name__ == "__main__":
    main()
