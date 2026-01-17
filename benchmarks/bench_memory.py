#!/usr/bin/env python3
"""
Memory profiling benchmark across medical imaging libraries.

Measures peak memory usage during various operations.

Run: python benchmarks/bench_memory.py [--quick|--full]
Output: benchmarks/results/memory_results.json
"""

import argparse
import gc
import json
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from common import (
    BenchmarkConfig,
    create_test_volume,
    save_nifti,
    get_rss_mb,
)

import numpy as np

try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("WARNING: psutil not installed, memory tracking limited")

try:
    import medrs

    HAS_MEDRS = True
except ImportError:
    HAS_MEDRS = False

try:
    import nibabel as nib

    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False

try:
    from monai.transforms import LoadImage

    HAS_MONAI = True
except ImportError:
    HAS_MONAI = False

try:
    import SimpleITK as sitk

    HAS_SIMPLEITK = True
except ImportError:
    HAS_SIMPLEITK = False


class MemoryResult:
    def __init__(self, library, operation, size, peak_mb, delta_mb, notes=""):
        self.library = library
        self.operation = operation
        self.size = size
        self.peak_mb = peak_mb
        self.delta_mb = delta_mb
        self.notes = notes

    def to_dict(self):
        return {
            "library": self.library,
            "operation": self.operation,
            "size": list(self.size),
            "peak_mb": round(self.peak_mb, 2) if self.peak_mb else None,
            "delta_mb": round(self.delta_mb, 2) if self.delta_mb else None,
            "notes": self.notes,
        }


class MemoryBenchmark:
    def __init__(self, config):
        self.config = config
        self.tmpdir = tempfile.mkdtemp(prefix="bench_memory_")
        self.results = []
        self._test_files = {}

    def cleanup(self):
        import shutil

        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _get_test_file(self, size, compressed=False):
        key = (size, compressed)
        if key not in self._test_files:
            data = create_test_volume(size, "float32")
            ext = ".nii.gz" if compressed else ".nii"
            path = os.path.join(self.tmpdir, f"test_{size[0]}{ext}")
            save_nifti(data, path)
            self._test_files[key] = path
        return self._test_files[key]

    def _measure_memory(self, func, warmup=1):
        """Measure memory usage of a function."""
        for _ in range(warmup):
            func()
            gc.collect()

        gc.collect()
        if HAS_PSUTIL:
            proc = psutil.Process()
            mem_before = proc.memory_info().rss / (1024 * 1024)
        else:
            mem_before = get_rss_mb() or 0

        result = func()

        if HAS_PSUTIL:
            mem_after = proc.memory_info().rss / (1024 * 1024)
        else:
            mem_after = get_rss_mb() or 0

        del result
        gc.collect()

        return mem_before, mem_after, mem_after - mem_before

    def _bench_load(self, library, load_func, size, compressed, notes=""):
        """Benchmark memory for a load operation."""
        test_file = self._get_test_file(size, compressed)

        def load_op():
            return load_func(test_file)

        mem_before, mem_after, delta = self._measure_memory(load_op)

        op_name = "load_gzipped" if compressed else "load"
        result = MemoryResult(
            library=library,
            operation=op_name,
            size=size,
            peak_mb=mem_after,
            delta_mb=delta,
            notes=notes,
        )
        self.results.append(result)
        return result

    def run_all(self):
        """Run all memory benchmarks."""
        print("\nRunning memory benchmarks...")
        print(f"  Sizes: {self.config.sizes}")

        for size in self.config.sizes:
            print(f"\n  Size {size[0]}x{size[1]}x{size[2]}:")
            volume_mb = np.prod(size) * 4 / (1024 * 1024)
            print(f"    Theoretical size: {volume_mb:.1f} MB (float32)")

            for compressed in [False, True]:
                comp_str = "gzipped" if compressed else "uncompressed"
                print(f"\n    {comp_str.upper()}:")

                if HAS_MEDRS:
                    r = self._bench_load(
                        "medrs",
                        lambda f: medrs.load(f),
                        size,
                        compressed,
                    )
                    print(f"      medrs: delta={r.delta_mb:.1f}MB")

                if HAS_NIBABEL:

                    def nibabel_load(f):
                        img = nib.load(f)
                        return np.asarray(img.dataobj)

                    r = self._bench_load("nibabel", nibabel_load, size, compressed)
                    print(f"      nibabel: delta={r.delta_mb:.1f}MB")

                if HAS_MONAI:
                    loader = LoadImage(image_only=True)
                    r = self._bench_load("monai", loader, size, compressed)
                    print(f"      monai: delta={r.delta_mb:.1f}MB")

                if HAS_SIMPLEITK:

                    def sitk_load(f):
                        img = sitk.ReadImage(f)
                        return sitk.GetArrayFromImage(img)

                    r = self._bench_load("simpleitk", sitk_load, size, compressed)
                    print(f"      simpleitk: delta={r.delta_mb:.1f}MB")

    def print_summary(self):
        """Print memory summary."""
        print(f"\n{'=' * 70}")
        print("MEMORY USAGE SUMMARY (Delta MB during load)")
        print(f"{'=' * 70}")

        by_size = {}
        for r in self.results:
            if r.size not in by_size:
                by_size[r.size] = {}
            key = (r.library, r.operation)
            by_size[r.size][key] = r

        libraries = sorted(set(r.library for r in self.results))

        print(f"\n{'Size':<15} {'Op':<15}", end="")
        for lib in libraries:
            print(f" {lib:>12}", end="")
        print()
        print("-" * (30 + 13 * len(libraries)))

        for size in sorted(by_size.keys()):
            for op in ["load", "load_gzipped"]:
                size_str = f"{size[0]}³"
                print(f"{size_str:<15} {op:<15}", end="")
                for lib in libraries:
                    key = (lib, op)
                    if key in by_size[size]:
                        delta = by_size[size][key].delta_mb
                        print(f" {delta:>10.1f}MB", end="")
                    else:
                        print(f" {'N/A':>12}", end="")
                print()

    def save_results(self, path):
        """Save results to JSON."""
        data = {
            "config": {
                "sizes": [list(s) for s in self.config.sizes],
            },
            "results": [r.to_dict() for r in self.results],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"\nResults saved to: {path}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark memory usage")
    parser.add_argument("--quick", action="store_true", help="Quick benchmark")
    parser.add_argument("--full", action="store_true", help="Full benchmark")
    args = parser.parse_args()

    if args.quick:
        config = BenchmarkConfig.quick()
    elif args.full:
        config = BenchmarkConfig.full()
    else:
        config = BenchmarkConfig()

    bench = MemoryBenchmark(config)
    try:
        bench.run_all()
        bench.print_summary()

        results_dir = Path(__file__).parent / "results"
        results_dir.mkdir(exist_ok=True)
        bench.save_results(str(results_dir / "memory_results.json"))
    finally:
        bench.cleanup()


if __name__ == "__main__":
    main()
