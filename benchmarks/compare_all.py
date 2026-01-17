#!/usr/bin/env python3
"""
Compare benchmark results across all medical imaging libraries.

Run individual benchmarks first:
  python benchmarks/bench_medrs.py
  python benchmarks/bench_nibabel.py
  python benchmarks/bench_monai.py
  python benchmarks/bench_torchio.py
  python benchmarks/bench_simpleitk.py

Then compare:
  python benchmarks/compare_all.py
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def load_results(path: Path) -> Optional[Dict]:
    """Load benchmark results from JSON file."""
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def format_speedup(ratio: float) -> str:
    """Format speedup ratio."""
    if ratio >= 1.0:
        return f"\033[32m{ratio:.2f}x faster\033[0m"  # Green
    else:
        return f"\033[31m{1 / ratio:.2f}x slower\033[0m"  # Red


def compare_libraries(results: Dict[str, Dict]) -> None:
    """Compare results across libraries."""
    if not results:
        print("No results to compare")
        return

    libraries = list(results.keys())
    print(f"\n{'=' * 80}")
    print(f"BENCHMARK COMPARISON: {' vs '.join(libraries)}")
    print(f"{'=' * 80}")

    # Group results by operation and size
    by_op_size: Dict[str, Dict[Tuple, Dict[str, Dict]]] = defaultdict(lambda: defaultdict(dict))

    for lib, data in results.items():
        for result in data.get("results", []):
            op = result["operation"]
            size = tuple(result["size"])
            by_op_size[op][size][lib] = result

    # Print comparison for each operation
    for op in sorted(by_op_size.keys()):
        print(f"\n{op}:")
        print(f"  {'Size':<15}", end="")
        for lib in libraries:
            print(f" {lib:>12}", end="")
        if len(libraries) > 1:
            print(f" {'Speedup':>16}", end="")
        print()
        print(f"  {'-' * (15 + 13 * len(libraries) + (18 if len(libraries) > 1 else 0))}")

        for size in sorted(by_op_size[op].keys()):
            size_str = f"{size[0]}x{size[1]}x{size[2]}"
            print(f"  {size_str:<15}", end="")

            lib_results = by_op_size[op][size]
            medians = {}

            for lib in libraries:
                if lib in lib_results:
                    median = lib_results[lib]["median_ms"]
                    medians[lib] = median
                    print(f" {median:>10.2f}ms", end="")
                else:
                    print(f" {'N/A':>12}", end="")

            # Calculate speedup (medrs vs others)
            if len(libraries) > 1 and "medrs" in medians:
                baseline = medians["medrs"]
                # Compare against first non-medrs library
                for other_lib in libraries:
                    if other_lib != "medrs" and other_lib in medians:
                        ratio = medians[other_lib] / baseline
                        print(f" {format_speedup(ratio)}", end="")
                        break

            print()


def print_summary_table(results: Dict[str, Dict]) -> None:
    """Print summary table with averages."""
    if not results:
        return

    libraries = list(results.keys())
    print(f"\n{'=' * 80}")
    print("SUMMARY (Average median times in ms)")
    print(f"{'=' * 80}")

    # Calculate average times per operation
    op_averages: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

    for lib, data in results.items():
        for result in data.get("results", []):
            op = result["operation"]
            op_averages[op][lib].append(result["median_ms"])

    print(f"\n  {'Operation':<25}", end="")
    for lib in libraries:
        print(f" {lib:>12}", end="")
    if len(libraries) > 1 and "medrs" in libraries:
        print(f" {'vs medrs':>16}", end="")
    print()
    print(f"  {'-' * (25 + 13 * len(libraries) + (18 if len(libraries) > 1 else 0))}")

    for op in sorted(op_averages.keys()):
        print(f"  {op:<25}", end="")

        averages = {}
        for lib in libraries:
            if lib in op_averages[op]:
                avg = sum(op_averages[op][lib]) / len(op_averages[op][lib])
                averages[lib] = avg
                print(f" {avg:>10.2f}ms", end="")
            else:
                print(f" {'N/A':>12}", end="")

        # Speedup vs medrs
        if len(libraries) > 1 and "medrs" in averages:
            baseline = averages["medrs"]
            for other_lib in libraries:
                if other_lib != "medrs" and other_lib in averages:
                    ratio = averages[other_lib] / baseline
                    print(f" {format_speedup(ratio)}", end="")
                    break

        print()


def run_all_benchmarks(quick: bool = False, full: bool = False) -> None:
    """Run all benchmark scripts."""
    import subprocess

    scripts = [
        "bench_medrs.py",
        "bench_nibabel.py",
        "bench_monai.py",
        "bench_torchio.py",
        "bench_simpleitk.py",
        "bench_mgzip.py",
    ]

    args = []
    if quick:
        args.append("--quick")
    elif full:
        args.append("--full")

    bench_dir = Path(__file__).parent

    for script in scripts:
        script_path = bench_dir / script
        if not script_path.exists():
            print(f"Warning: {script} not found, skipping")
            continue

        print(f"\n{'=' * 60}")
        print(f"Running {script}...")
        print(f"{'=' * 60}")

        try:
            result = subprocess.run(
                [sys.executable, str(script_path)] + args,
                cwd=str(bench_dir),
                check=True,
            )
        except subprocess.CalledProcessError as e:
            print(f"Warning: {script} failed with exit code {e.returncode}")
        except FileNotFoundError:
            print(f"Warning: Could not run {script}")


def main():
    parser = argparse.ArgumentParser(description="Compare benchmark results")
    parser.add_argument("--run", action="store_true", help="Run all benchmarks before comparing")
    parser.add_argument(
        "--quick", action="store_true", help="Use quick benchmark config (with --run)"
    )
    parser.add_argument(
        "--full", action="store_true", help="Use full benchmark config (with --run)"
    )
    parser.add_argument("--results-dir", "-d", type=str, help="Directory containing result files")
    args = parser.parse_args()

    # Optionally run benchmarks first
    if args.run:
        run_all_benchmarks(quick=args.quick, full=args.full)

    # Load results
    results_dir = Path(args.results_dir) if args.results_dir else Path(__file__).parent / "results"

    result_files = {
        "medrs": results_dir / "medrs_results.json",
        "nibabel": results_dir / "nibabel_results.json",
        "monai": results_dir / "monai_results.json",
        "torchio": results_dir / "torchio_results.json",
        "simpleitk": results_dir / "simpleitk_results.json",
    }

    results = {}
    for lib, path in result_files.items():
        data = load_results(path)
        if data:
            results[lib] = data
            print(f"Loaded {lib} results from {path}")

    if not results:
        print("\nNo results files found. Run benchmarks first:")
        print("  python benchmarks/bench_medrs.py")
        print("  python benchmarks/bench_nibabel.py")
        print("  python benchmarks/bench_monai.py")
        print("  python benchmarks/bench_torchio.py")
        print("  python benchmarks/bench_simpleitk.py")
        print("\nOr use --run to run all benchmarks:")
        print("  python benchmarks/compare_all.py --run")
        return

    # Compare
    compare_libraries(results)
    print_summary_table(results)


if __name__ == "__main__":
    main()
