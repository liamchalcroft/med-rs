#!/usr/bin/env python3
"""
FastLoader throughput benchmark comparing medrs, MONAI, and TorchIO data loading.

Simulates training pipeline data loading with parallel prefetching.

Run: python benchmarks/bench_fastloader.py [--quick|--full]
Output: benchmarks/results/fastloader_results.json
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
    import medrs

    HAS_MEDRS = True
except ImportError:
    HAS_MEDRS = False
    print("WARNING: medrs not installed")

try:
    import torch
    from torch.utils.data import DataLoader, Dataset, IterableDataset

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("WARNING: torch not installed")

try:
    from monai.transforms import LoadImaged, RandSpatialCropd, Compose, EnsureChannelFirstd
    from monai.data import Dataset as MonaiDataset, DataLoader as MonaiDataLoader

    HAS_MONAI = True
except ImportError:
    HAS_MONAI = False
    print("WARNING: MONAI not installed")

try:
    import torchio as tio

    HAS_TORCHIO = True
except ImportError:
    HAS_TORCHIO = False
    print("WARNING: TorchIO not installed")


class ThroughputResult:
    def __init__(self, name, samples, total_time_s, samples_per_sec, memory_mb=None, notes=""):
        self.name = name
        self.samples = samples
        self.total_time_s = total_time_s
        self.samples_per_sec = samples_per_sec
        self.memory_mb = memory_mb
        self.notes = notes

    def to_dict(self):
        return {
            "name": self.name,
            "samples": self.samples,
            "total_time_s": round(self.total_time_s, 3),
            "samples_per_sec": round(self.samples_per_sec, 2),
            "memory_mb": round(self.memory_mb, 2) if self.memory_mb else None,
            "notes": self.notes,
        }


class FastLoaderBenchmark:
    def __init__(self, num_files=50, volume_size=(128, 128, 128), patch_size=(64, 64, 64)):
        self.num_files = num_files
        self.volume_size = volume_size
        self.patch_size = patch_size
        self.tmpdir = tempfile.mkdtemp(prefix="bench_fastloader_")
        self.files = []
        self.results = []

    def setup(self):
        """Create test files."""
        print(f"Creating {self.num_files} test files ({self.volume_size})...")
        for i in range(self.num_files):
            data = create_test_volume(self.volume_size, "float32")
            path = os.path.join(self.tmpdir, f"vol_{i:04d}.nii.gz")
            save_nifti(data, path)
            self.files.append(path)
        print(f"  Created {len(self.files)} files in {self.tmpdir}")

    def cleanup(self):
        import shutil

        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _measure_throughput(self, name, iterator_fn, num_samples, notes=""):
        """Measure throughput of an iterator."""
        gc.collect()
        memory_before = get_rss_mb()

        start = time.perf_counter()
        count = 0
        for _ in iterator_fn():
            count += 1
            if count >= num_samples:
                break
        end = time.perf_counter()

        memory_after = get_rss_mb()
        memory_delta = None
        if memory_before and memory_after:
            memory_delta = max(0, memory_after - memory_before)

        total_time = end - start
        samples_per_sec = count / total_time if total_time > 0 else 0

        result = ThroughputResult(
            name=name,
            samples=count,
            total_time_s=total_time,
            samples_per_sec=samples_per_sec,
            memory_mb=memory_delta,
            notes=notes,
        )
        self.results.append(result)
        return result

    def bench_medrs_fastloader(self, workers=4, prefetch=16):
        """Benchmark medrs FastLoader."""
        if not HAS_MEDRS:
            print("  Skipping medrs FastLoader (not installed)")
            return

        def iterator():
            loader = medrs.FastLoader(
                volumes=self.files,
                patch_shape=list(self.patch_size),
                prefetch=prefetch,
                workers=workers,
                shuffle=True,
                seed=42,
            )
            for patch in loader:
                yield patch

        result = self._measure_throughput(
            f"medrs_fastloader_w{workers}",
            iterator,
            self.num_files,
            notes=f"workers={workers},prefetch={prefetch}",
        )
        print(f"  medrs FastLoader (w={workers}): {result.samples_per_sec:.1f} samples/sec")

    def bench_medrs_fastloader_mgzip(self, workers=4, prefetch=16, mgzip_threads=4):
        """Benchmark medrs FastLoader with Mgzip."""
        if not HAS_MEDRS:
            print("  Skipping medrs FastLoader+Mgzip (not installed)")
            return

        mgzip_files = []
        print("  Converting files to Mgzip...")
        for f in self.files:
            mgz = f.replace(".nii.gz", ".mgz.nii.gz")
            if not os.path.exists(mgz):
                medrs.convert_to_mgzip(f, mgz, num_threads=4)
            mgzip_files.append(mgz)

        def iterator():
            loader = medrs.FastLoader(
                volumes=mgzip_files,
                patch_shape=list(self.patch_size),
                prefetch=prefetch,
                workers=workers,
                shuffle=True,
                seed=42,
                mgzip_threads=mgzip_threads,
            )
            for patch in loader:
                yield patch

        result = self._measure_throughput(
            f"medrs_fastloader_mgzip_w{workers}_t{mgzip_threads}",
            iterator,
            self.num_files,
            notes=f"workers={workers},mgzip_threads={mgzip_threads}",
        )
        print(
            f"  medrs FastLoader+Mgzip (w={workers},t={mgzip_threads}): {result.samples_per_sec:.1f} samples/sec"
        )

    def bench_medrs_sequential(self):
        """Benchmark sequential medrs loading."""
        if not HAS_MEDRS:
            return

        patch_size = self.patch_size

        def iterator():
            for f in self.files:
                img = medrs.load(f)
                arr = img.to_numpy()
                offset = [(s - p) // 2 for s, p in zip(arr.shape, patch_size)]
                patch = arr[
                    offset[0] : offset[0] + patch_size[0],
                    offset[1] : offset[1] + patch_size[1],
                    offset[2] : offset[2] + patch_size[2],
                ]
                yield patch

        result = self._measure_throughput(
            "medrs_sequential",
            iterator,
            self.num_files,
            notes="single-threaded",
        )
        print(f"  medrs sequential: {result.samples_per_sec:.1f} samples/sec")

    def bench_monai_dataloader(self, num_workers=4):
        """Benchmark MONAI DataLoader."""
        if not HAS_MONAI or not HAS_TORCH:
            print("  Skipping MONAI DataLoader (not installed)")
            return

        data_dicts = [{"image": f} for f in self.files]

        transform = Compose(
            [
                LoadImaged(keys=["image"], image_only=True),
                EnsureChannelFirstd(keys=["image"]),
                RandSpatialCropd(keys=["image"], roi_size=self.patch_size, random_size=False),
            ]
        )

        dataset = MonaiDataset(data=data_dicts, transform=transform)
        loader = MonaiDataLoader(dataset, batch_size=1, num_workers=num_workers, shuffle=True)

        def iterator():
            for batch in loader:
                yield batch["image"]

        result = self._measure_throughput(
            f"monai_dataloader_w{num_workers}",
            iterator,
            self.num_files,
            notes=f"num_workers={num_workers}",
        )
        print(f"  MONAI DataLoader (w={num_workers}): {result.samples_per_sec:.1f} samples/sec")

    def bench_torchio_queue(self, num_workers=4):
        """Benchmark TorchIO Queue."""
        if not HAS_TORCHIO or not HAS_TORCH:
            print("  Skipping TorchIO Queue (not installed)")
            return

        subjects = [tio.Subject(image=tio.ScalarImage(f)) for f in self.files]
        dataset = tio.SubjectsDataset(subjects)

        sampler = tio.UniformSampler(self.patch_size)
        queue = tio.Queue(
            dataset,
            max_length=32,
            samples_per_volume=1,
            sampler=sampler,
            num_workers=num_workers,
        )

        loader = DataLoader(queue, batch_size=1)

        def iterator():
            for batch in loader:
                yield batch["image"]["data"]

        result = self._measure_throughput(
            f"torchio_queue_w{num_workers}",
            iterator,
            self.num_files,
            notes=f"num_workers={num_workers}",
        )
        print(f"  TorchIO Queue (w={num_workers}): {result.samples_per_sec:.1f} samples/sec")

    def run_all(self, workers_list=None):
        """Run all benchmarks."""
        if workers_list is None:
            workers_list = [1, 4, 8]

        print(f"\n{'=' * 60}")
        print("FASTLOADER THROUGHPUT BENCHMARK")
        print(f"{'=' * 60}")
        print(f"Files: {self.num_files}")
        print(f"Volume size: {self.volume_size}")
        print(f"Patch size: {self.patch_size}")
        print()

        print("Sequential baselines:")
        self.bench_medrs_sequential()

        for workers in workers_list:
            print(f"\nParallel loading (workers={workers}):")
            self.bench_medrs_fastloader(workers=workers)
            self.bench_monai_dataloader(num_workers=workers)
            self.bench_torchio_queue(num_workers=workers)

        print("\nMgzip parallel decompression:")
        self.bench_medrs_fastloader_mgzip(workers=4, mgzip_threads=2)
        self.bench_medrs_fastloader_mgzip(workers=4, mgzip_threads=4)

    def print_summary(self):
        """Print summary comparison."""
        print(f"\n{'=' * 60}")
        print("THROUGHPUT SUMMARY")
        print(f"{'=' * 60}")
        print(f"{'Method':<45} {'Samples/sec':>12}")
        print("-" * 60)

        for r in sorted(self.results, key=lambda x: -x.samples_per_sec):
            print(f"{r.name:<45} {r.samples_per_sec:>12.1f}")

        if len(self.results) >= 2:
            best = max(self.results, key=lambda x: x.samples_per_sec)
            worst = min(self.results, key=lambda x: x.samples_per_sec)
            print(f"\nBest: {best.name} ({best.samples_per_sec:.1f} samples/sec)")
            print(f"Speedup vs slowest: {best.samples_per_sec / worst.samples_per_sec:.1f}x")

    def save_results(self, path):
        """Save results to JSON."""
        data = {
            "config": {
                "num_files": self.num_files,
                "volume_size": list(self.volume_size),
                "patch_size": list(self.patch_size),
            },
            "results": [r.to_dict() for r in self.results],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Results saved to: {path}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark FastLoader throughput")
    parser.add_argument("--quick", action="store_true", help="Quick benchmark (fewer files)")
    parser.add_argument("--full", action="store_true", help="Full benchmark (more files)")
    parser.add_argument("--files", type=int, default=50, help="Number of test files")
    parser.add_argument("--size", type=int, default=128, help="Volume size (cubic)")
    parser.add_argument("--patch", type=int, default=64, help="Patch size (cubic)")
    args = parser.parse_args()

    if args.quick:
        num_files = 20
        volume_size = (64, 64, 64)
        workers_list = [1, 4]
    elif args.full:
        num_files = 100
        volume_size = (128, 128, 128)
        workers_list = [1, 2, 4, 8]
    else:
        num_files = args.files
        volume_size = (args.size, args.size, args.size)
        workers_list = [1, 4, 8]

    patch_size = (args.patch, args.patch, args.patch)

    bench = FastLoaderBenchmark(
        num_files=num_files,
        volume_size=volume_size,
        patch_size=patch_size,
    )

    try:
        bench.setup()
        bench.run_all(workers_list=workers_list)
        bench.print_summary()

        results_dir = Path(__file__).parent / "results"
        results_dir.mkdir(exist_ok=True)
        bench.save_results(str(results_dir / "fastloader_results.json"))
    finally:
        bench.cleanup()


if __name__ == "__main__":
    main()
