#!/usr/bin/env python3
"""
Mgzip thread scaling benchmark for medrs.

Analyzes parallel decompression performance across thread counts.

Run: python benchmarks/bench_mgzip.py [--quick|--full]
Output: benchmarks/results/mgzip_results.json
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from common import (
    BenchmarkConfig,
    BenchmarkRunner,
    BenchmarkResult,
    create_test_volume,
    save_nifti,
    time_function,
)

import numpy as np

try:
    import medrs

    HAS_MEDRS = True
except ImportError:
    HAS_MEDRS = False
    print("ERROR: medrs not installed. Run: maturin develop --release")
    sys.exit(1)

try:
    import nibabel as nib

    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False
    print("WARNING: nibabel not installed, skipping baseline comparison")


class MgzipBenchmarkRunner(BenchmarkRunner):
    """Benchmark runner for Mgzip thread scaling analysis."""

    library_name = "mgzip"

    def __init__(self, config, thread_counts=None):
        super().__init__(config)
        self.thread_counts = thread_counts or [1, 2, 4, 8, 16]
        self._mgzip_files = {}

    def run_all(self):
        """Run all Mgzip benchmarks."""
        print("\nRunning Mgzip thread scaling benchmarks...")
        print(f"  Sizes: {self.config.sizes}")
        print(f"  Thread counts: {self.thread_counts}")
        print(
            f"  Warmup: {self.config.warmup_iterations}, Iterations: {self.config.benchmark_iterations}"
        )

        for dtype in self.config.dtypes:
            for size in self.config.sizes:
                self._run_size_benchmarks(size, dtype)

        return self.results

    def _get_mgzip_file(self, size, dtype):
        """Get or create Mgzip version of test file."""
        key = (size, dtype)
        if key not in self._mgzip_files:
            gzip_file = self.get_test_file_gzipped(size, dtype)
            mgzip_file = gzip_file.replace(".nii.gz", ".mgz.nii.gz")
            if not os.path.exists(mgzip_file):
                print("    Converting to Mgzip format...")
                medrs.convert_to_mgzip(gzip_file, mgzip_file, num_threads=8)
            self._mgzip_files[key] = mgzip_file
        return self._mgzip_files[key]

    def _run_size_benchmarks(self, size, dtype):
        """Run benchmarks for a specific size."""
        print(f"\n  Size {size[0]}x{size[1]}x{size[2]} ({dtype}):")

        gzip_file = self.get_test_file_gzipped(size, dtype)
        mgzip_file = self._get_mgzip_file(size, dtype)
        nii_file = self.get_test_file(size, dtype)

        self._bench_baseline_nibabel(gzip_file, size, dtype)
        self._bench_medrs_gzip(gzip_file, size, dtype)
        self._bench_medrs_mmap(nii_file, size, dtype)
        self._bench_mgzip_thread_scaling(mgzip_file, size, dtype)
        self._bench_conversion_time(gzip_file, size, dtype)
        self._compare_file_sizes(gzip_file, mgzip_file, size, dtype)

    def _bench_baseline_nibabel(self, test_file, size, dtype):
        """Baseline: nibabel loading standard gzip."""
        if not HAS_NIBABEL:
            return

        def load_op():
            img = nib.load(test_file)
            return np.asarray(img.dataobj)

        result = self.run_benchmark("nibabel_gzip", load_op, size, dtype, notes="baseline")
        print(f"    nibabel (gzip): {result.median_ms:.2f}ms")

    def _bench_medrs_gzip(self, test_file, size, dtype):
        """medrs loading standard gzip (libdeflate)."""

        def load_op():
            return medrs.load(test_file)

        result = self.run_benchmark("medrs_gzip", load_op, size, dtype, notes="libdeflate")
        print(f"    medrs (gzip): {result.median_ms:.2f}ms")

    def _bench_medrs_mmap(self, test_file, size, dtype):
        """medrs loading uncompressed (mmap baseline)."""

        def load_op():
            return medrs.load(test_file)

        result = self.run_benchmark("medrs_mmap", load_op, size, dtype, notes="uncompressed,mmap")
        print(f"    medrs (mmap): {result.median_ms:.2f}ms")

    def _bench_mgzip_thread_scaling(self, test_file, size, dtype):
        """Benchmark Mgzip with different thread counts."""
        print("    Mgzip thread scaling:")

        for threads in self.thread_counts:

            def load_op(t=threads):
                return medrs.load_mgzip(test_file, num_threads=t)

            result = self.run_benchmark(
                f"mgzip_{threads}t", load_op, size, dtype, notes=f"threads={threads}"
            )
            print(f"      {threads} threads: {result.median_ms:.2f}ms")

    def _bench_conversion_time(self, gzip_file, size, dtype):
        """Benchmark conversion from gzip to Mgzip."""
        output_path = os.path.join(self.tmpdir, f"convert_test_{size[0]}.mgz.nii.gz")

        def convert_op():
            medrs.convert_to_mgzip(gzip_file, output_path, num_threads=8)

        result = self.run_benchmark("convert_to_mgzip", convert_op, size, dtype, notes="8 threads")
        print(f"    convert_to_mgzip: {result.median_ms:.2f}ms")

    def _compare_file_sizes(self, gzip_file, mgzip_file, size, dtype):
        """Compare file sizes between formats."""
        gzip_size = os.path.getsize(gzip_file)
        mgzip_size = os.path.getsize(mgzip_file)
        overhead = (mgzip_size - gzip_size) / gzip_size * 100

        print(
            f"    File sizes: gzip={gzip_size / 1024 / 1024:.2f}MB, mgzip={mgzip_size / 1024 / 1024:.2f}MB ({overhead:+.1f}%)"
        )

    def print_scaling_analysis(self):
        """Print thread scaling analysis."""
        print(f"\n{'=' * 70}")
        print("MGZIP THREAD SCALING ANALYSIS")
        print(f"{'=' * 70}")

        by_size = {}
        for r in self.results:
            if r.size not in by_size:
                by_size[r.size] = {}
            by_size[r.size][r.operation] = r

        for size, ops in sorted(by_size.items()):
            print(f"\nSize {size[0]}x{size[1]}x{size[2]}:")

            baseline = ops.get("nibabel_gzip")
            medrs_gzip = ops.get("medrs_gzip")
            medrs_mmap = ops.get("medrs_mmap")

            if baseline:
                print("  Baselines:")
                print(f"    nibabel (gzip):  {baseline.median_ms:8.2f}ms")
            if medrs_gzip:
                speedup = baseline.median_ms / medrs_gzip.median_ms if baseline else 0
                print(
                    f"    medrs (gzip):    {medrs_gzip.median_ms:8.2f}ms ({speedup:.1f}x vs nibabel)"
                )
            if medrs_mmap:
                print(f"    medrs (mmap):    {medrs_mmap.median_ms:8.2f}ms (uncompressed)")

            print("\n  Mgzip scaling:")
            print(
                f"    {'Threads':<10} {'Time (ms)':<12} {'Speedup vs 1t':<15} {'Speedup vs nibabel':<20}"
            )
            print(f"    {'-' * 55}")

            mgzip_1t = ops.get("mgzip_1t")
            for threads in self.thread_counts:
                op_name = f"mgzip_{threads}t"
                if op_name in ops:
                    r = ops[op_name]
                    speedup_1t = mgzip_1t.median_ms / r.median_ms if mgzip_1t else 0
                    speedup_nib = baseline.median_ms / r.median_ms if baseline else 0
                    print(
                        f"    {threads:<10} {r.median_ms:<12.2f} {speedup_1t:<15.2f}x {speedup_nib:<20.2f}x"
                    )


def main():
    parser = argparse.ArgumentParser(description="Benchmark Mgzip thread scaling")
    parser.add_argument("--quick", action="store_true", help="Quick benchmark")
    parser.add_argument("--full", action="store_true", help="Full benchmark")
    parser.add_argument(
        "--threads",
        type=str,
        default="1,2,4,8",
        help="Comma-separated thread counts (default: 1,2,4,8)",
    )
    args = parser.parse_args()

    thread_counts = [int(t) for t in args.threads.split(",")]

    if args.quick:
        config = BenchmarkConfig.quick()
    elif args.full:
        config = BenchmarkConfig.full()
    else:
        config = BenchmarkConfig()

    runner = MgzipBenchmarkRunner(config, thread_counts=thread_counts)
    try:
        runner.run_all()
        runner.print_results()
        runner.print_scaling_analysis()

        results_dir = Path(__file__).parent / "results"
        results_dir.mkdir(exist_ok=True)
        runner.save_results(str(results_dir / "mgzip_results.json"))
    finally:
        runner.cleanup()


if __name__ == "__main__":
    main()
