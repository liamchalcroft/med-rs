#!/usr/bin/env python3
"""
Generate publication-quality plots from benchmark results.

Usage: python benchmarks/plot_results.py
Output: benchmarks/results/plots/
"""

import json
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

STYLE_CONFIG = {
    "figure.facecolor": "#FAFAFA",
    "axes.facecolor": "#FFFFFF",
    "axes.edgecolor": "#CCCCCC",
    "axes.linewidth": 1.2,
    "axes.grid": True,
    "axes.axisbelow": True,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.labelweight": "medium",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.color": "#E5E5E5",
    "grid.linewidth": 0.8,
    "grid.alpha": 0.7,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "legend.frameon": True,
    "legend.framealpha": 0.95,
    "legend.edgecolor": "#CCCCCC",
    "legend.fontsize": 10,
    "legend.title_fontsize": 11,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
}

COLORS = {
    "medrs": "#2563EB",
    "monai": "#F97316",
    "torchio": "#10B981",
    "nibabel": "#EF4444",
    "simpleitk": "#8B5CF6",
}

COLORS_LIGHT = {
    "medrs": "#93C5FD",
    "monai": "#FDBA74",
    "torchio": "#6EE7B7",
    "nibabel": "#FCA5A5",
    "simpleitk": "#C4B5FD",
}

MARKERS = {
    "medrs": "o",
    "monai": "s",
    "torchio": "^",
    "nibabel": "D",
    "simpleitk": "p",
}

LIB_DISPLAY_NAMES = {
    "medrs": "medrs",
    "monai": "MONAI",
    "torchio": "TorchIO",
    "nibabel": "nibabel",
    "simpleitk": "SimpleITK",
}

SPEEDUP_CMAP = LinearSegmentedColormap.from_list(
    "speedup", ["#FEE2E2", "#FECACA", "#FCA5A5", "#FFFFFF", "#BBF7D0", "#86EFAC", "#4ADE80"], N=256
)

PERF_CMAP = LinearSegmentedColormap.from_list(
    "performance",
    ["#10B981", "#6EE7B7", "#A7F3D0", "#FEF3C7", "#FDE68A", "#FBBF24", "#F97316", "#EF4444"],
    N=256,
)


def apply_style():
    plt.rcParams.update(STYLE_CONFIG)


def get_lib_label(lib: str) -> str:
    return LIB_DISPLAY_NAMES.get(lib, lib)


def style_axis(ax, title=None, xlabel=None, ylabel=None):
    if title:
        ax.set_title(title, pad=12)
    if xlabel:
        ax.set_xlabel(xlabel, labelpad=8)
    if ylabel:
        ax.set_ylabel(ylabel, labelpad=8)
    ax.tick_params(axis="both", which="major", length=5, width=1)
    ax.tick_params(axis="both", which="minor", length=3, width=0.8)


def add_watermark(fig, text="medrs benchmarks"):
    fig.text(
        0.99,
        0.01,
        text,
        fontsize=8,
        color="#AAAAAA",
        ha="right",
        va="bottom",
        alpha=0.7,
        style="italic",
    )


def load_results(results_dir: Path) -> dict:
    results = {}

    for lib in ["medrs", "monai", "torchio", "nibabel", "simpleitk"]:
        path = results_dir / f"{lib}_results.json"
        if path.exists():
            with open(path) as f:
                results[lib] = json.load(f)
        else:
            print(f"Info: {path} not found (optional)")

    for name in ["mgzip", "fastloader", "memory"]:
        path = results_dir / f"{name}_results.json"
        if path.exists():
            with open(path) as f:
                results[name] = json.load(f)

    return results


def parse_size(size_tuple) -> int:
    return size_tuple[0]


def group_by_operation(results: dict) -> dict:
    all_libs = ["medrs", "monai", "torchio", "nibabel", "simpleitk"]
    by_op = defaultdict(lambda: {lib: [] for lib in all_libs})

    for lib in all_libs:
        if lib not in results:
            continue
        data = results[lib]
        for result in data.get("results", []):
            op = result["operation"]
            size = parse_size(result["size"])
            median = result["median_ms"]
            std = result.get("std_ms", 0)

            by_op[op][lib].append(
                {
                    "size": size,
                    "median": median,
                    "std": std,
                    "full_size": result["size"],
                }
            )

    return dict(by_op)


def create_operation_plot(op_name: str, op_data: dict, output_dir: Path):
    apply_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    plotted_any = False
    for lib in ["medrs", "monai", "torchio", "nibabel", "simpleitk"]:
        data_points = op_data.get(lib, [])
        if not data_points:
            continue

        plotted_any = True
        data_points = sorted(data_points, key=lambda x: x["size"])
        x = [p["size"] for p in data_points]
        y = [p["median"] for p in data_points]
        yerr = [p["std"] for p in data_points]

        ax.errorbar(
            x,
            y,
            yerr=yerr,
            label=get_lib_label(lib),
            color=COLORS[lib],
            marker=MARKERS[lib],
            markersize=9,
            markeredgecolor="white",
            markeredgewidth=1.5,
            linewidth=2.5,
            capsize=5,
            capthick=1.5,
            elinewidth=1.5,
            alpha=0.9,
            zorder=10 if lib == "medrs" else 5,
        )

    if not plotted_any:
        plt.close()
        return

    title = op_name.replace("_", " ").title()
    style_axis(ax, title=title, xlabel="Volume Size (voxels per dimension)", ylabel="Time (ms)")

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")

    ax.legend(loc="upper left", frameon=True, fancybox=True, shadow=True)

    ax.fill_between(
        [ax.get_xlim()[0], ax.get_xlim()[1]], 0, 1, alpha=0.03, color=COLORS["medrs"], zorder=0
    )

    add_watermark(fig)

    plt.tight_layout()
    output_path = output_dir / f"{op_name}.png"
    plt.savefig(output_path, facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {output_path}")


def create_summary_plot(by_op: dict, output_dir: Path):
    apply_style()
    ops = sorted(by_op.keys())
    n_ops = len(ops)

    if n_ops == 0:
        print("No operations to plot")
        return

    n_cols = 3
    n_rows = (n_ops + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 4.5 * n_rows))
    fig.patch.set_facecolor("#FAFAFA")

    if n_ops == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()

    for idx, op_name in enumerate(ops):
        ax = axes[idx]
        op_data = by_op[op_name]

        for lib in ["medrs", "monai", "torchio", "nibabel", "simpleitk"]:
            data_points = op_data.get(lib, [])
            if not data_points:
                continue

            data_points = sorted(data_points, key=lambda x: x["size"])
            x = [p["size"] for p in data_points]
            y = [p["median"] for p in data_points]

            ax.plot(
                x,
                y,
                label=get_lib_label(lib),
                color=COLORS[lib],
                marker=MARKERS[lib],
                markersize=7,
                markeredgecolor="white",
                markeredgewidth=1,
                linewidth=2,
                alpha=0.85,
                zorder=10 if lib == "medrs" else 5,
            )

        title = op_name.replace("_", " ").title()
        if len(title) > 25:
            title = title[:22] + "..."
        style_axis(ax, title=title, xlabel="Size", ylabel="Time (ms)")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.legend(fontsize=8, loc="upper left", framealpha=0.9)

    for idx in range(n_ops, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Benchmark Summary: All Operations", fontsize=16, fontweight="bold", y=1.02)
    add_watermark(fig)

    plt.tight_layout()
    output_path = output_dir / "summary.png"
    plt.savefig(output_path, facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {output_path}")


def create_speedup_heatmap(by_op: dict, output_dir: Path):
    apply_style()
    ops = sorted(by_op.keys())

    sizes = set()
    for op_data in by_op.values():
        for lib_data in op_data.values():
            for p in lib_data:
                sizes.add(p["size"])
    sizes = sorted(sizes)

    if not sizes or not ops:
        print("No data for heatmap")
        return

    compare_libs = []
    for lib in ["monai", "torchio", "nibabel", "simpleitk"]:
        has_data = any(any(p for p in by_op.get(op, {}).get(lib, [])) for op in ops)
        if has_data:
            compare_libs.append(lib)

    if not compare_libs:
        print("No comparison libraries found for heatmap")
        return

    n_libs = len(compare_libs)
    fig, axes = plt.subplots(1, n_libs, figsize=(7 * n_libs, max(6, len(ops) * 0.5)))
    fig.patch.set_facecolor("#FAFAFA")
    if n_libs == 1:
        axes = [axes]

    im = None
    for lib_idx, compare_lib in enumerate(compare_libs):
        speedups = []
        for size in sizes:
            for op in ops:
                op_data = by_op[op]
                medrs_pt = next((p for p in op_data.get("medrs", []) if p["size"] == size), None)
                other_pt = next(
                    (p for p in op_data.get(compare_lib, []) if p["size"] == size), None
                )

                if medrs_pt and other_pt and medrs_pt["median"] > 0:
                    speedup = other_pt["median"] / medrs_pt["median"]
                    speedups.append(speedup)
                else:
                    speedups.append(np.nan)

        n_sizes = len(sizes)
        n_ops_local = len(ops)
        matrix = np.array(speedups).reshape(n_sizes, n_ops_local).T

        ax = axes[lib_idx]

        im = ax.imshow(
            matrix, aspect="auto", cmap="RdYlGn", vmin=0.5, vmax=100, interpolation="nearest"
        )

        ax.set_xticks(range(n_sizes))
        ax.set_xticklabels([f"{s}³" for s in sizes], fontsize=10)
        ax.set_yticks(range(n_ops_local))
        ax.set_yticklabels([op.replace("_", " ").title()[:20] for op in ops], fontsize=9)

        for i in range(n_ops_local):
            for j in range(n_sizes):
                val = matrix[i, j]
                if not np.isnan(val):
                    text = f"{val:.0f}×" if val >= 10 else f"{val:.1f}×"
                    text_color = "white" if val > 5 or val < 0.8 else "#333333"
                    ax.text(
                        j,
                        i,
                        text,
                        ha="center",
                        va="center",
                        fontsize=8,
                        fontweight="bold",
                        color=text_color,
                    )

        lib_name = get_lib_label(compare_lib)
        ax.set_title(
            f"medrs vs {lib_name}\n(speedup factor)", fontsize=13, fontweight="bold", pad=10
        )
        ax.set_xlabel("Volume Size", fontsize=11, labelpad=8)

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("#CCCCCC")

    if im is not None:
        cbar = fig.colorbar(im, ax=axes, shrink=0.8, aspect=30, pad=0.02)
        cbar.set_label("Speedup (×)", fontsize=11, labelpad=10)
        cbar.ax.tick_params(labelsize=9)

    fig.suptitle(
        "Performance Speedup: medrs vs Other Libraries", fontsize=16, fontweight="bold", y=1.02
    )
    add_watermark(fig)

    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.15, right=0.92)
    output_path = output_dir / "speedup_heatmap.png"
    plt.savefig(output_path, facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {output_path}")


def create_mgzip_thread_scaling_plot(mgzip_data: dict, output_dir: Path):
    apply_style()
    if not mgzip_data:
        print("No Mgzip data available")
        return

    results = mgzip_data.get("results", [])
    if not results:
        print("No Mgzip results found")
        return

    by_size = defaultdict(dict)
    for r in results:
        size = tuple(r["size"])
        op = r["operation"]
        by_size[size][op] = r

    n_sizes = len(by_size)
    fig, axes = plt.subplots(1, n_sizes, figsize=(7 * n_sizes, 5.5))
    fig.patch.set_facecolor("#FAFAFA")
    if n_sizes == 1:
        axes = [axes]

    for idx, (size, ops) in enumerate(sorted(by_size.items())):
        ax = axes[idx]

        thread_data = []
        for op_name, data in ops.items():
            if op_name.startswith("mgzip_") and op_name.endswith("t"):
                try:
                    threads = int(op_name.replace("mgzip_", "").replace("t", ""))
                    thread_data.append((threads, data["median_ms"]))
                except (ValueError, KeyError):
                    pass

        if not thread_data:
            continue

        thread_data.sort(key=lambda x: x[0])
        threads = [d[0] for d in thread_data]
        times = [d[1] for d in thread_data]

        base_time = thread_data[0][1] if thread_data else 1
        speedups = [base_time / t for t in times]

        color_time = COLORS["medrs"]
        color_speedup = COLORS["torchio"]

        ax.fill_between(threads, times, alpha=0.15, color=color_time, zorder=1)
        (line1,) = ax.plot(
            threads,
            times,
            "o-",
            color=color_time,
            linewidth=2.5,
            markersize=10,
            markeredgecolor="white",
            markeredgewidth=2,
            label="Load Time",
            zorder=10,
        )

        ax.set_xlabel("Thread Count", fontsize=12, fontweight="medium", labelpad=8)
        ax.set_ylabel("Time (ms)", fontsize=12, fontweight="medium", color=color_time, labelpad=8)
        ax.tick_params(axis="y", labelcolor=color_time, colors=color_time)

        ax2 = ax.twinx()
        ax2.spines["right"].set_visible(True)
        ax2.spines["right"].set_color(color_speedup)

        (line2,) = ax2.plot(
            threads,
            speedups,
            "s--",
            color=color_speedup,
            linewidth=2.5,
            markersize=8,
            markeredgecolor="white",
            markeredgewidth=1.5,
            label="Actual Speedup",
            zorder=9,
        )

        (ideal_line,) = ax2.plot(
            threads,
            threads,
            ":",
            color="#888888",
            linewidth=2,
            alpha=0.7,
            label="Ideal (linear)",
            zorder=5,
        )

        ax2.set_ylabel(
            "Speedup (×)", fontsize=12, fontweight="medium", color=color_speedup, labelpad=8
        )
        ax2.tick_params(axis="y", labelcolor=color_speedup, colors=color_speedup)

        nibabel_time = ops.get("nibabel_gzip", {}).get("median_ms")
        medrs_gzip_time = ops.get("medrs_gzip", {}).get("median_ms")

        baseline_lines = []
        if nibabel_time:
            bl = ax.axhline(
                y=nibabel_time,
                color=COLORS["nibabel"],
                linestyle="--",
                linewidth=2,
                alpha=0.8,
                label=f"nibabel: {nibabel_time:.0f}ms",
            )
            baseline_lines.append(bl)
        if medrs_gzip_time:
            bl = ax.axhline(
                y=medrs_gzip_time,
                color=COLORS["monai"],
                linestyle="--",
                linewidth=2,
                alpha=0.8,
                label=f"medrs gzip: {medrs_gzip_time:.0f}ms",
            )
            baseline_lines.append(bl)

        ax.set_title(
            f"Mgzip Thread Scaling ({size[0]}³ volume)", fontsize=14, fontweight="bold", pad=12
        )
        ax.set_xticks(threads)
        ax.set_xticklabels([str(t) for t in threads])

        all_handles = [line1, line2, ideal_line] + baseline_lines
        all_labels = [h.get_label() for h in all_handles]
        ax.legend(
            all_handles,
            all_labels,
            loc="upper right",
            fontsize=9,
            frameon=True,
            fancybox=True,
            shadow=True,
        )

    fig.suptitle(
        "Parallel Decompression: Mgzip Thread Scaling Analysis",
        fontsize=16,
        fontweight="bold",
        y=1.02,
    )
    add_watermark(fig)

    plt.tight_layout()
    output_path = output_dir / "mgzip_thread_scaling.png"
    plt.savefig(output_path, facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {output_path}")


def create_fastloader_throughput_plot(fastloader_data: dict, output_dir: Path):
    apply_style()
    if not fastloader_data:
        print("No FastLoader data available")
        return

    results = fastloader_data.get("results", [])
    if not results:
        print("No FastLoader results found")
        return

    results = sorted(results, key=lambda x: x["samples_per_sec"])

    names = [r["name"] for r in results]
    throughputs = [r["samples_per_sec"] for r in results]

    colors = []
    edge_colors = []
    for name in names:
        if name.startswith("medrs"):
            colors.append(COLORS["medrs"])
            edge_colors.append(COLORS["medrs"])
        elif name.startswith("monai"):
            colors.append(COLORS["monai"])
            edge_colors.append(COLORS["monai"])
        elif name.startswith("torchio"):
            colors.append(COLORS["torchio"])
            edge_colors.append(COLORS["torchio"])
        else:
            colors.append("#9CA3AF")
            edge_colors.append("#6B7280")

    fig, ax = plt.subplots(figsize=(12, max(6, len(names) * 0.45)))
    fig.patch.set_facecolor("#FAFAFA")

    y_pos = np.arange(len(names))
    bars = ax.barh(
        y_pos,
        throughputs,
        height=0.7,
        color=colors,
        edgecolor=[c for c in edge_colors],
        linewidth=1.5,
        alpha=0.85,
    )

    ax.set_yticks(y_pos)
    display_names = []
    for n in names:
        dn = (
            n.replace("_", " ")
            .replace("medrs ", "medrs: ")
            .replace("monai ", "MONAI: ")
            .replace("torchio ", "TorchIO: ")
        )
        display_names.append(dn)
    ax.set_yticklabels(display_names, fontsize=10)

    style_axis(ax, xlabel="Throughput (samples/sec)", ylabel=None)
    ax.set_title("Data Loader Throughput Comparison", fontsize=14, fontweight="bold", pad=15)

    max_val = max(throughputs)
    for bar, val in zip(bars, throughputs):
        ax.text(
            val + max_val * 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}",
            va="center",
            ha="left",
            fontsize=10,
            fontweight="medium",
        )

    ax.set_xlim(0, max_val * 1.15)

    legend_patches = [
        mpatches.Patch(
            facecolor=COLORS["medrs"], edgecolor=COLORS["medrs"], linewidth=1.5, label="medrs"
        ),
        mpatches.Patch(
            facecolor=COLORS["monai"], edgecolor=COLORS["monai"], linewidth=1.5, label="MONAI"
        ),
        mpatches.Patch(
            facecolor=COLORS["torchio"], edgecolor=COLORS["torchio"], linewidth=1.5, label="TorchIO"
        ),
    ]
    ax.legend(
        handles=legend_patches,
        loc="lower right",
        fontsize=10,
        frameon=True,
        fancybox=True,
        shadow=True,
    )

    ax.axvline(
        x=throughputs[-1], color=COLORS["medrs"], linestyle="--", alpha=0.3, linewidth=2, zorder=0
    )

    add_watermark(fig)

    plt.tight_layout()
    output_path = output_dir / "fastloader_throughput.png"
    plt.savefig(output_path, facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {output_path}")


def create_memory_comparison_plot(memory_data: dict, output_dir: Path):
    apply_style()
    if not memory_data:
        print("No memory data available")
        return

    results = memory_data.get("results", [])
    if not results:
        print("No memory results found")
        return

    by_size_op = defaultdict(lambda: defaultdict(dict))
    for r in results:
        size = tuple(r["size"])
        op = r["operation"]
        lib = r["library"]
        by_size_op[size][op][lib] = r.get("delta_mb", 0)

    sizes = sorted(by_size_op.keys())
    ops = ["load", "load_gzipped"]

    n_sizes = len(sizes)
    fig, axes = plt.subplots(1, n_sizes, figsize=(7 * n_sizes, 5.5))
    fig.patch.set_facecolor("#FAFAFA")
    if n_sizes == 1:
        axes = [axes]

    for idx, size in enumerate(sizes):
        ax = axes[idx]
        size_data = by_size_op[size]

        all_libs = set()
        for op in ops:
            all_libs.update(size_data.get(op, {}).keys())
        libs = sorted(all_libs, key=lambda x: list(COLORS.keys()).index(x) if x in COLORS else 99)

        x = np.arange(len(ops))
        n_libs_local = len(libs)
        width = 0.8 / max(n_libs_local, 1)

        for lib_idx, lib in enumerate(libs):
            values = [size_data.get(op, {}).get(lib, 0) or 0 for op in ops]
            offset = (lib_idx - n_libs_local / 2 + 0.5) * width
            ax.bar(
                x + offset,
                values,
                width * 0.9,
                label=get_lib_label(lib),
                color=COLORS.get(lib, "#9CA3AF"),
                edgecolor="white",
                linewidth=1.5,
                alpha=0.85,
            )

        theoretical_mb = np.prod(size) * 4 / (1024 * 1024)
        ax.axhline(
            y=theoretical_mb,
            color="#EF4444",
            linestyle="--",
            linewidth=2,
            alpha=0.7,
            label=f"Theoretical: {theoretical_mb:.0f} MB",
        )

        style_axis(ax, xlabel=None, ylabel="Memory Delta (MB)")
        ax.set_title(f"Memory Usage ({size[0]}³ volume)", fontsize=14, fontweight="bold", pad=12)

        ax.set_xticks(x)
        ax.set_xticklabels(["Uncompressed\n(.nii)", "Compressed\n(.nii.gz)"], fontsize=11)

        ax.legend(fontsize=9, loc="upper right", frameon=True, fancybox=True, shadow=True)

    fig.suptitle("Memory Usage Comparison Across Libraries", fontsize=16, fontweight="bold", y=1.02)
    add_watermark(fig)

    plt.tight_layout()
    output_path = output_dir / "memory_comparison.png"
    plt.savefig(output_path, facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {output_path}")


def create_all_libraries_heatmap(by_op: dict, output_dir: Path):
    apply_style()
    ops = sorted(by_op.keys())

    all_sizes = set()
    for op_data in by_op.values():
        for lib_data in op_data.values():
            for p in lib_data:
                all_sizes.add(p["size"])

    if not all_sizes:
        print("No size data for all-libraries heatmap")
        return

    target_size = max(all_sizes)

    libs = ["medrs", "monai", "torchio", "nibabel", "simpleitk"]
    available_libs = []
    for lib in libs:
        has_data = any(
            any(p for p in by_op.get(op, {}).get(lib, []) if p["size"] == target_size) for op in ops
        )
        if has_data:
            available_libs.append(lib)

    if len(available_libs) < 2:
        print("Not enough libraries for comparison heatmap")
        return

    matrix = np.zeros((len(ops), len(available_libs)))
    for op_idx, op in enumerate(ops):
        for lib_idx, lib in enumerate(available_libs):
            data_points = by_op.get(op, {}).get(lib, [])
            pt = next((p for p in data_points if p["size"] == target_size), None)
            if pt:
                matrix[op_idx, lib_idx] = pt["median"]
            else:
                matrix[op_idx, lib_idx] = np.nan

    fig, ax = plt.subplots(figsize=(max(10, len(available_libs) * 2), max(8, len(ops) * 0.5)))
    fig.patch.set_facecolor("#FAFAFA")

    matrix_log = np.log10(matrix + 0.1)

    im = ax.imshow(matrix_log, aspect="auto", cmap=PERF_CMAP, interpolation="nearest")

    ax.set_xticks(range(len(available_libs)))
    ax.set_xticklabels(
        [get_lib_label(lib) for lib in available_libs], fontsize=11, fontweight="medium"
    )
    ax.set_yticks(range(len(ops)))
    ax.set_yticklabels([op.replace("_", " ").title()[:25] for op in ops], fontsize=10)

    for i in range(len(ops)):
        for j in range(len(available_libs)):
            val = matrix[i, j]
            if not np.isnan(val):
                if val < 1:
                    text = f"{val:.2f}"
                elif val < 100:
                    text = f"{val:.1f}"
                else:
                    text = f"{val:.0f}"
                text_color = "white" if matrix_log[i, j] > np.nanmedian(matrix_log) else "#333333"
                ax.text(
                    j,
                    i,
                    text,
                    ha="center",
                    va="center",
                    fontsize=9,
                    fontweight="bold",
                    color=text_color,
                )

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#CCCCCC")
        spine.set_linewidth(1.5)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8, aspect=30, pad=0.02)
    cbar.set_label("Time (ms, log scale)", fontsize=11, labelpad=10)
    cbar.ax.tick_params(labelsize=9)

    ax.set_title(
        f"Performance Matrix ({target_size}³ volume)\nLower values = faster (green is better)",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )

    add_watermark(fig)

    plt.tight_layout()
    output_path = output_dir / "all_libraries_heatmap.png"
    plt.savefig(output_path, facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {output_path}")


def main():
    results_dir = Path(__file__).parent / "results"
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    print("Loading benchmark results...")
    results = load_results(results_dir)

    if not results:
        print("No results found. Run benchmarks first:")
        print("  python benchmarks/bench_medrs.py")
        print("  python benchmarks/bench_monai.py")
        print("  python benchmarks/bench_torchio.py")
        print("  python benchmarks/bench_nibabel.py")
        print("  python benchmarks/bench_simpleitk.py")
        print("  python benchmarks/bench_mgzip.py")
        print("  python benchmarks/bench_fastloader.py")
        print("  python benchmarks/bench_memory.py")
        return

    core_libs = [
        lib for lib in ["medrs", "monai", "torchio", "nibabel", "simpleitk"] if lib in results
    ]
    special = [name for name in ["mgzip", "fastloader", "memory"] if name in results]
    print(f"Loaded results for: {', '.join(core_libs)}")
    if special:
        print(f"Special benchmarks: {', '.join(special)}")

    by_op = group_by_operation(results)

    if by_op:
        print("\nGenerating individual operation plots...")
        for op_name, op_data in sorted(by_op.items()):
            create_operation_plot(op_name, op_data, plots_dir)

        print("\nGenerating summary plot...")
        create_summary_plot(by_op, plots_dir)

        print("\nGenerating speedup heatmap...")
        create_speedup_heatmap(by_op, plots_dir)

        print("\nGenerating all-libraries heatmap...")
        create_all_libraries_heatmap(by_op, plots_dir)

    if "mgzip" in results:
        print("\nGenerating Mgzip thread scaling plot...")
        create_mgzip_thread_scaling_plot(results["mgzip"], plots_dir)

    if "fastloader" in results:
        print("\nGenerating FastLoader throughput plot...")
        create_fastloader_throughput_plot(results["fastloader"], plots_dir)

    if "memory" in results:
        print("\nGenerating memory comparison plot...")
        create_memory_comparison_plot(results["memory"], plots_dir)

    print(f"\nAll plots saved to: {plots_dir}/")


if __name__ == "__main__":
    main()
