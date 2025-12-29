#!/usr/bin/env python3
"""
Generate plots from benchmark results.

Usage: python benchmarks/plot_results.py
Output: benchmarks/results/plots/
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


def load_results(results_dir: Path) -> dict:
    """Load all benchmark result JSONs."""
    results = {}
    for lib in ["medrs", "monai", "torchio"]:
        path = results_dir / f"{lib}_results.json"
        if path.exists():
            with open(path) as f:
                results[lib] = json.load(f)
        else:
            print(f"Warning: {path} not found")
    return results


def parse_size(size_tuple) -> int:
    """Parse size tuple to get voxel count."""
    # Assume cubic volumes for simplicity
    return size_tuple[0]  # Use first dimension as representative


def group_by_operation(results: dict) -> dict:
    """Group results by operation name."""
    by_op = defaultdict(lambda: {"medrs": [], "monai": [], "torchio": []})

    for lib, data in results.items():
        for result in data.get("results", []):
            op = result["operation"]
            size = parse_size(result["size"])
            median = result["median_ms"]
            std = result.get("std_ms", 0)

            by_op[op][lib].append({
                "size": size,
                "median": median,
                "std": std,
                "full_size": result["size"],
            })

    return dict(by_op)


def create_operation_plot(op_name: str, op_data: dict, output_dir: Path):
    """Create a plot for a single operation."""
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {
        "medrs": "#1f77b4",
        "monai": "#ff7f0e",
        "torchio": "#2ca02c",
    }

    markers = {
        "medrs": "o",
        "monai": "s",
        "torchio": "^",
    }

    # Sort by size
    sizes_seen = set()

    for lib in ["medrs", "monai", "torchio"]:
        data_points = op_data[lib]
        if not data_points:
            continue

        # Sort by size
        data_points = sorted(data_points, key=lambda x: x["size"])

        x = [p["size"] for p in data_points]
        y = [p["median"] for p in data_points]
        yerr = [p["std"] for p in data_points]

        ax.errorbar(x, y, yerr=yerr,
                   label=lib.upper(),
                   color=colors[lib],
                   marker=markers[lib],
                   markersize=8,
                   linewidth=2,
                   capsize=4,
                   alpha=0.8)

        sizes_seen.update(x)

    ax.set_xlabel("Volume Size (voxels per dimension)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Median Time (ms)", fontsize=12, fontweight="bold")
    ax.set_title(f"{op_name.replace('_', ' ').title()}", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, loc="upper left")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")

    # Set reasonable tick labels
    ax.tick_params(axis="both", which="major", labelsize=10)

    plt.tight_layout()

    output_path = output_dir / f"{op_name}.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  Saved: {output_path}")


def create_summary_plot(by_op: dict, output_dir: Path):
    """Create a summary plot with all operations as subplots."""
    ops = sorted(by_op.keys())
    n_ops = len(ops)

    if n_ops == 0:
        print("No operations to plot")
        return

    # Create subplot grid
    n_cols = 3
    n_rows = (n_ops + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_ops == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.reshape(1, -1)

    axes = axes.flatten() if n_ops > 1 else [axes[0]]

    colors = {
        "medrs": "#1f77b4",
        "monai": "#ff7f0e",
        "torchio": "#2ca02c",
    }

    markers = {
        "medrs": "o",
        "monai": "s",
        "torchio": "^",
    }

    for idx, op_name in enumerate(ops):
        ax = axes[idx]
        op_data = by_op[op_name]

        for lib in ["medrs", "monai", "torchio"]:
            data_points = op_data[lib]
            if not data_points:
                continue

            data_points = sorted(data_points, key=lambda x: x["size"])
            x = [p["size"] for p in data_points]
            y = [p["median"] for p in data_points]
            yerr = [p["std"] for p in data_points]

            ax.errorbar(x, y, yerr=yerr,
                       label=lib.upper(),
                       color=colors[lib],
                       marker=markers[lib],
                       markersize=6,
                       linewidth=1.5,
                       capsize=3,
                       alpha=0.8)

        ax.set_xlabel("Size", fontsize=10)
        ax.set_ylabel("Time (ms)", fontsize=10)
        ax.set_title(op_name.replace("_", " ").title(), fontsize=11, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")

    # Hide unused subplots
    for idx in range(n_ops, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    output_path = output_dir / "summary.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  Saved: {output_path}")


def create_speedup_heatmap(by_op: dict, output_dir: Path):
    """Create a heatmap showing speedup of medrs vs others."""
    ops = sorted(by_op.keys())

    # Get all unique sizes
    sizes = set()
    for op_data in by_op.values():
        for lib_data in op_data.values():
            for p in lib_data:
                sizes.add(p["size"])
    sizes = sorted(sizes)

    if not sizes:
        print("No size data for heatmap")
        return

    # Calculate speedups
    monai_speedups = []
    torchio_speedups = []

    for size in sizes:
        for op in ops:
            op_data = by_op[op]

            medrs_pt = next((p for p in op_data["medrs"] if p["size"] == size), None)
            monai_pt = next((p for p in op_data["monai"] if p["size"] == size), None)
            torchio_pt = next((p for p in op_data["torchio"] if p["size"] == size), None)

            if medrs_pt and monai_pt:
                speedup = monai_pt["median"] / medrs_pt["median"]
                monai_speedups.append(speedup)
            else:
                monai_speedups.append(np.nan)

            if medrs_pt and torchio_pt:
                speedup = torchio_pt["median"] / medrs_pt["median"]
                torchio_speedups.append(speedup)
            else:
                torchio_speedups.append(np.nan)

    # Create heatmap data
    n_sizes = len(sizes)
    n_ops = len(ops)

    monai_matrix = np.array(monai_speedups).reshape(n_sizes, n_ops).T
    torchio_matrix = np.array(torchio_speedups).reshape(n_sizes, n_ops).T

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # MONAI speedup
    im1 = ax1.imshow(monai_matrix, aspect="auto", cmap="RdYlGn", vmin=0.5, vmax=50)
    ax1.set_xticks(range(n_sizes))
    ax1.set_xticklabels([str(s) for s in sizes])
    ax1.set_yticks(range(n_ops))
    ax1.set_yticklabels([op.replace("_", "\n") for op in ops])
    ax1.set_title("medrs vs MONAI (Speedup)", fontweight="bold")
    ax1.set_xlabel("Volume Size")
    plt.colorbar(im1, ax=ax1, label="Speedup (x)")

    # TorchIO speedup
    im2 = ax2.imshow(torchio_matrix, aspect="auto", cmap="RdYlGn", vmin=0.5, vmax=100)
    ax2.set_xticks(range(n_sizes))
    ax2.set_xticklabels([str(s) for s in sizes])
    ax2.set_yticks(range(n_ops))
    ax2.set_yticklabels([op.replace("_", "\n") for op in ops])
    ax2.set_title("medrs vs TorchIO (Speedup)", fontweight="bold")
    ax2.set_xlabel("Volume Size")
    plt.colorbar(im2, ax=ax2, label="Speedup (x)")

    plt.tight_layout()
    output_path = output_dir / "speedup_heatmap.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  Saved: {output_path}")


def main():
    results_dir = Path(__file__).parent / "results"
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # Load results
    print("Loading benchmark results...")
    results = load_results(results_dir)

    if not results:
        print("No results found. Run benchmarks first:")
        print("  python benchmarks/bench_medrs.py")
        print("  python benchmarks/bench_monai.py")
        print("  python benchmarks/bench_torchio.py")
        return

    print(f"Loaded results for: {', '.join(results.keys())}")

    # Group by operation
    by_op = group_by_operation(results)

    # Create individual plots
    print("\nGenerating individual plots...")
    for op_name, op_data in sorted(by_op.items()):
        create_operation_plot(op_name, op_data, plots_dir)

    # Create summary plot
    print("\nGenerating summary plot...")
    create_summary_plot(by_op, plots_dir)

    # Create speedup heatmap
    print("\nGenerating speedup heatmap...")
    create_speedup_heatmap(by_op, plots_dir)

    print(f"\nAll plots saved to: {plots_dir}/")


if __name__ == "__main__":
    main()
