#!/usr/bin/env python3
"""
Common utilities for medical imaging library benchmarks.

Provides:
- Test data generation
- Timing utilities
- Result storage format
- JSON output for comparison
"""

import gc
import json
import os
import sys
import tempfile
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import resource

    HAS_RESOURCE = True
except ImportError:
    HAS_RESOURCE = False

# Optional imports
try:
    import nibabel as nib

    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False
    nib = None


@dataclass
class BenchmarkResult:
    """Result of a single benchmark operation."""

    library: str
    operation: str
    size: Tuple[int, int, int]
    dtype: str
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    median_ms: float
    iterations: int
    memory_mb: Optional[float] = None  # Peak RSS delta during benchmark iterations.
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "library": self.library,
            "operation": self.operation,
            "size": list(self.size),
            "dtype": self.dtype,
            "mean_ms": round(self.mean_ms, 3),
            "std_ms": round(self.std_ms, 3),
            "min_ms": round(self.min_ms, 3),
            "max_ms": round(self.max_ms, 3),
            "median_ms": round(self.median_ms, 3),
            "iterations": self.iterations,
            "memory_mb": round(self.memory_mb, 2) if self.memory_mb else None,
            "notes": self.notes,
        }


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""

    sizes: List[Tuple[int, int, int]] = field(
        default_factory=lambda: [
            (64, 64, 64),
            (128, 128, 128),
            (256, 256, 256),
            (512, 512, 512),
        ]
    )
    dtypes: List[str] = field(default_factory=lambda: ["float32"])
    warmup_iterations: int = 3
    benchmark_iterations: int = 20
    crop_size: Tuple[int, int, int] = (64, 64, 64)

    @classmethod
    def quick(cls) -> "BenchmarkConfig":
        """Quick benchmark config for testing."""
        return cls(
            sizes=[(64, 64, 64), (128, 128, 128), (256, 256, 256)],
            warmup_iterations=1,
            benchmark_iterations=5,
        )

    @classmethod
    def full(cls) -> "BenchmarkConfig":
        """Full benchmark config with larger volumes."""
        return cls(
            sizes=[
                (64, 64, 64),
                (128, 128, 128),
                (256, 256, 256),
                (512, 512, 512),
            ],
            warmup_iterations=3,
            benchmark_iterations=30,
        )


def create_test_volume(shape: Tuple[int, int, int], dtype: str = "float32") -> np.ndarray:
    """Create synthetic medical imaging volume with realistic intensity distribution."""
    np_dtype = getattr(np, dtype)

    # Create gradient + spherical structure
    x = np.linspace(0, 1, shape[0])
    y = np.linspace(0, 1, shape[1])
    z = np.linspace(0, 1, shape[2])
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    # Spherical intensity pattern (brain-like)
    center = np.array([0.5, 0.5, 0.5])
    dist = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2 + (Z - center[2]) ** 2)
    data = (1.0 - dist) * 1000

    # Add noise
    rng = np.random.default_rng(42)
    data += rng.standard_normal(shape).astype(np.float32) * 50

    return data.astype(np_dtype)


def create_label_volume(shape: Tuple[int, int, int]) -> np.ndarray:
    """Create synthetic segmentation label volume."""
    label = np.zeros(shape, dtype=np.uint8)

    # Create spherical foreground region
    x = np.linspace(0, 1, shape[0])
    y = np.linspace(0, 1, shape[1])
    z = np.linspace(0, 1, shape[2])
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    center = np.array([0.5, 0.5, 0.5])
    dist = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2 + (Z - center[2]) ** 2)
    label[dist < 0.3] = 1

    return label


def save_nifti(data: np.ndarray, path: str, affine: Optional[np.ndarray] = None):
    """Save numpy array as NIfTI file using nibabel."""
    if not HAS_NIBABEL:
        raise RuntimeError("nibabel required for NIfTI I/O")
    if affine is None:
        affine = np.eye(4)
    img = nib.Nifti1Image(data, affine)
    nib.save(img, path)


def get_rss_mb() -> Optional[float]:
    """Return current/peak RSS in MB if available."""
    if HAS_PSUTIL:
        return psutil.Process().memory_info().rss / (1024 * 1024)
    if not HAS_RESOURCE:
        return None
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return rss / (1024 * 1024)
    return rss / 1024


def time_function(
    func: Callable, warmup: int = 3, iterations: int = 20
) -> Tuple[float, float, float, float, float, Optional[float]]:
    """Time a function with warmup and multiple iterations.

    Returns: (mean_ms, std_ms, min_ms, max_ms, median_ms, memory_mb)
    """
    # Warmup
    for _ in range(warmup):
        func()
        gc.collect()

    # Benchmark
    memory_before = get_rss_mb()
    max_rss = memory_before if memory_before is not None else None

    times = []
    for _ in range(iterations):
        gc.collect()
        start = time.perf_counter()
        func()
        end = time.perf_counter()
        times.append((end - start) * 1000)  # ms
        rss = get_rss_mb()
        if rss is not None:
            if max_rss is None or rss > max_rss:
                max_rss = rss

    memory_mb = None
    if memory_before is not None and max_rss is not None:
        memory_mb = max(0.0, max_rss - memory_before)

    return (
        float(np.mean(times)),
        float(np.std(times)),
        float(np.min(times)),
        float(np.max(times)),
        float(np.median(times)),
        memory_mb,
    )


class BenchmarkRunner:
    """Base class for library-specific benchmark runners."""

    library_name: str = "unknown"

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.tmpdir = tempfile.mkdtemp(prefix=f"bench_{self.library_name}_")
        self.results: List[BenchmarkResult] = []
        self._test_files: Dict[Tuple[int, int, int], str] = {}

    def cleanup(self):
        """Clean up temporary files."""
        import shutil

        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def get_test_file(self, shape: Tuple[int, int, int], dtype: str = "float32") -> str:
        """Get or create test file for given shape."""
        key = shape
        if key not in self._test_files:
            data = create_test_volume(shape, dtype)
            path = os.path.join(self.tmpdir, f"test_{shape[0]}x{shape[1]}x{shape[2]}.nii")
            save_nifti(data, path)
            self._test_files[key] = path
        return self._test_files[key]

    def get_test_file_gzipped(self, shape: Tuple[int, int, int], dtype: str = "float32") -> str:
        """Get or create gzipped test file for given shape."""
        key = ("gz", shape)
        if key not in self._test_files:
            data = create_test_volume(shape, dtype)
            path = os.path.join(self.tmpdir, f"test_{shape[0]}x{shape[1]}x{shape[2]}.nii.gz")
            save_nifti(data, path)
            self._test_files[key] = path
        return self._test_files[key]

    def run_benchmark(
        self,
        operation: str,
        func: Callable,
        shape: Tuple[int, int, int],
        dtype: str,
        notes: str = "",
    ) -> BenchmarkResult:
        """Run a single benchmark and record result."""
        mean_ms, std_ms, min_ms, max_ms, median_ms, memory_mb = time_function(
            func,
            self.config.warmup_iterations,
            self.config.benchmark_iterations,
        )

        result = BenchmarkResult(
            library=self.library_name,
            operation=operation,
            size=shape,
            dtype=dtype,
            mean_ms=mean_ms,
            std_ms=std_ms,
            min_ms=min_ms,
            max_ms=max_ms,
            median_ms=median_ms,
            iterations=self.config.benchmark_iterations,
            memory_mb=memory_mb,
            notes=notes,
        )
        self.results.append(result)
        return result

    def run_all(self) -> List[BenchmarkResult]:
        """Run all benchmarks. Override in subclass."""
        raise NotImplementedError

    def save_results(self, path: str):
        """Save results to JSON file."""
        data = {
            "library": self.library_name,
            "config": asdict(self.config),
            "results": [r.to_dict() for r in self.results],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Results saved to: {path}")

    def print_results(self):
        """Print results to console."""
        print(f"\n{'=' * 70}")
        print(f"{self.library_name.upper()} BENCHMARK RESULTS")
        print(f"{'=' * 70}")

        # Group by operation
        by_op: Dict[str, List[BenchmarkResult]] = {}
        for r in self.results:
            if r.operation not in by_op:
                by_op[r.operation] = []
            by_op[r.operation].append(r)

        for op, results in by_op.items():
            show_memory = any(r.memory_mb is not None for r in results)
            print(f"\n{op}:")
            if show_memory:
                print(
                    f"  {'Size':<15} {'Median':>10} {'Mean':>10} {'Std':>8} {'Min':>10} {'Max':>10} {'RSSΔ':>8}"
                )
                print(f"  {'-' * 75}")
            else:
                print(
                    f"  {'Size':<15} {'Median':>10} {'Mean':>10} {'Std':>8} {'Min':>10} {'Max':>10}"
                )
                print(f"  {'-' * 65}")
            for r in sorted(results, key=lambda x: x.size[0]):
                size_str = f"{r.size[0]}x{r.size[1]}x{r.size[2]}"
                if show_memory:
                    mem_str = f"{r.memory_mb:>8.2f}" if r.memory_mb is not None else f"{'-':>8}"
                    print(
                        f"  {size_str:<15} {r.median_ms:>10.2f} {r.mean_ms:>10.2f} {r.std_ms:>8.2f} {r.min_ms:>10.2f} {r.max_ms:>10.2f} {mem_str}"
                    )
                else:
                    print(
                        f"  {size_str:<15} {r.median_ms:>10.2f} {r.mean_ms:>10.2f} {r.std_ms:>8.2f} {r.min_ms:>10.2f} {r.max_ms:>10.2f}"
                    )


def load_results(path: str) -> Dict:
    """Load benchmark results from JSON file."""
    with open(path) as f:
        return json.load(f)


def compare_results(result_files: List[str]) -> None:
    """Compare results from multiple benchmark files."""
    all_data = {}
    for path in result_files:
        data = load_results(path)
        lib = data["library"]
        all_data[lib] = {r["operation"]: r for r in data["results"]}

    # Find common operations
    if not all_data:
        print("No results to compare")
        return

    libs = list(all_data.keys())
    first_lib = libs[0]
    operations = set(all_data[first_lib].keys())
    for lib in libs[1:]:
        operations &= set(all_data[lib].keys())

    print(f"\n{'=' * 80}")
    print("COMPARISON: " + " vs ".join(libs))
    print(f"{'=' * 80}")

    for op in sorted(operations):
        print(f"\n{op}:")
        print(f"  {'Library':<12} {'Median (ms)':>12} {'Speedup':>12}")
        print(f"  {'-' * 40}")

        # Get baseline (first library)
        results = [(lib, all_data[lib].get(op)) for lib in libs if op in all_data[lib]]
        if not results:
            continue

        baseline_median = results[0][1]["median_ms"]

        for lib, r in results:
            if r is None:
                continue
            median = r["median_ms"]
            if baseline_median > 0:
                speedup = baseline_median / median
                speedup_str = f"{speedup:.2f}x" if speedup != 1.0 else "baseline"
            else:
                speedup_str = "N/A"
            print(f"  {lib:<12} {median:>12.2f} {speedup_str:>12}")
