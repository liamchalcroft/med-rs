#!/usr/bin/env python3
"""
Standalone benchmark for MONAI library.

Run: python benchmarks/bench_monai.py [--quick|--full]
Output: benchmarks/results/monai_results.json
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
    import torch
    from monai.transforms import (
        LoadImage,
        RandSpatialCrop,
        Resize,
        ScaleIntensity,
        EnsureChannelFirst,
        Compose,
    )
    HAS_MONAI = True
except ImportError as e:
    HAS_MONAI = False
    print(f"ERROR: MONAI not installed. Run: pip install monai")
    print(f"  Import error: {e}")
    sys.exit(1)


class MonaiBenchmarkRunner(BenchmarkRunner):
    """Benchmark runner for MONAI library."""

    library_name = "monai"

    def __init__(self, config):
        super().__init__(config)
        # Pre-create transforms
        self.loader = LoadImage(image_only=True)

    def run_all(self):
        """Run all MONAI benchmarks."""
        print(f"\nRunning MONAI benchmarks...")
        print(f"  Sizes: {self.config.sizes}")
        print(f"  Warmup: {self.config.warmup_iterations}, Iterations: {self.config.benchmark_iterations}")

        for dtype in self.config.dtypes:
            for size in self.config.sizes:
                self._run_size_benchmarks(size, dtype)

        return self.results

    def _run_size_benchmarks(self, size, dtype):
        """Run benchmarks for a specific size."""
        print(f"\n  Size {size[0]}x{size[1]}x{size[2]} ({dtype}):")

        # Get test file
        test_file = self.get_test_file(size, dtype)

        # 1. Load benchmark
        self._bench_load(test_file, size, dtype)

        # 2. Load + crop benchmark
        self._bench_load_cropped(test_file, size, dtype)

        # 3. Load + resize benchmark
        self._bench_load_resampled(test_file, size, dtype)

        # 4. Load to tensor benchmark
        self._bench_load_to_torch(test_file, size, dtype)

        # 5. Intensity normalization benchmark
        self._bench_intensity_normalize(test_file, size, dtype)

    def _bench_load(self, test_file, size, dtype):
        """Benchmark basic load."""
        loader = LoadImage(image_only=True)

        def load_op():
            return loader(test_file)

        result = self.run_benchmark("load", load_op, size, dtype)
        print(f"    load: {result.median_ms:.2f}ms")

    def _bench_load_cropped(self, test_file, size, dtype):
        """Benchmark load + random crop."""
        crop_size = self.config.crop_size

        # Skip if crop is larger than volume
        if any(c > s for c, s in zip(crop_size, size)):
            return

        loader = LoadImage(image_only=True)
        channel_first = EnsureChannelFirst()
        cropper = RandSpatialCrop(roi_size=crop_size, random_size=False)

        def load_cropped_op():
            img = loader(test_file)
            img = channel_first(img)
            return cropper(img)

        result = self.run_benchmark("load_cropped", load_cropped_op, size, dtype,
                                    notes=f"crop={crop_size}")
        print(f"    load_cropped: {result.median_ms:.2f}ms")

    def _bench_load_resampled(self, test_file, size, dtype):
        """Benchmark load + resize."""
        target_shape = [s // 2 for s in size]
        if any(s < 16 for s in target_shape):
            return

        loader = LoadImage(image_only=True)
        channel_first = EnsureChannelFirst()
        resizer = Resize(spatial_size=target_shape)

        def load_resampled_op():
            img = loader(test_file)
            img = channel_first(img)
            return resizer(img)

        result = self.run_benchmark("load_resampled", load_resampled_op, size, dtype,
                                    notes=f"target={target_shape}")
        print(f"    load_resampled: {result.median_ms:.2f}ms")

    def _bench_load_to_torch(self, test_file, size, dtype):
        """Benchmark load + convert to torch tensor."""
        crop_size = self.config.crop_size
        if any(c > s for c, s in zip(crop_size, size)):
            return

        loader = LoadImage(image_only=True)
        channel_first = EnsureChannelFirst()
        cropper = RandSpatialCrop(roi_size=crop_size, random_size=False)

        def load_to_torch_op():
            img = loader(test_file)
            img = channel_first(img)
            img = cropper(img)
            # MONAI returns tensors by default, but ensure it's on CPU
            if isinstance(img, torch.Tensor):
                return img.cpu()
            return torch.from_numpy(np.ascontiguousarray(img))

        result = self.run_benchmark("load_cropped_to_torch", load_to_torch_op, size, dtype,
                                    notes=f"crop={crop_size}")
        print(f"    load_cropped_to_torch: {result.median_ms:.2f}ms")

    def _bench_intensity_normalize(self, test_file, size, dtype):
        """Benchmark intensity normalization."""
        crop_size = self.config.crop_size
        if any(c > s for c, s in zip(crop_size, size)):
            return

        loader = LoadImage(image_only=True)
        channel_first = EnsureChannelFirst()
        cropper = RandSpatialCrop(roi_size=crop_size, random_size=False)
        scaler = ScaleIntensity()

        def load_normalized_op():
            img = loader(test_file)
            img = channel_first(img)
            img = cropper(img)
            return scaler(img)

        result = self.run_benchmark("load_cropped_normalized", load_normalized_op, size, dtype,
                                    notes=f"crop={crop_size},normalize=True")
        print(f"    load_cropped_normalized: {result.median_ms:.2f}ms")


def main():
    parser = argparse.ArgumentParser(description="Benchmark MONAI library")
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
    runner = MonaiBenchmarkRunner(config)
    try:
        runner.run_all()
        runner.print_results()

        # Save results
        output_path = args.output or str(output_dir / "monai_results.json")
        runner.save_results(output_path)
    finally:
        runner.cleanup()


if __name__ == "__main__":
    main()
