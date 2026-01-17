#!/usr/bin/env python3
"""
Modal benchmark runner for medrs.

Runs comprehensive benchmarks on dedicated cloud hardware for reproducible results.

Usage:
    # Quick benchmarks (5-10 minutes)
    modal run modal_benchmark.py --mode quick

    # Full benchmarks (30-60 minutes)
    modal run modal_benchmark.py --mode full

    # Download results after run
    modal volume get medrs-results benchmarks/results/

Setup:
    1. Install Modal: pip install modal
    2. Authenticate: modal token new
    3. Run: modal run modal_benchmark.py
"""

import modal

app = modal.App("medrs-benchmark")

volume = modal.Volume.from_name("medrs-results", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "curl", "build-essential", "pkg-config", "libssl-dev", "cmake")
    .run_commands("curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y")
    .env({"PATH": "/root/.cargo/bin:$PATH"})
    .pip_install(
        "maturin>=1.4",
        "numpy>=1.25",
        "torch>=2.0",
        "monai>=1.3",
        "torchio>=0.19",
        "nibabel>=5.0",
        "simpleitk>=2.3",
        "psutil>=5.9",
        "matplotlib>=3.7",
    )
)


@app.function(
    image=image,
    cpu=8,
    memory=32768,
    timeout=7200,
    volumes={"/results": volume},
)
def run_benchmarks(
    mode: str = "quick",
    repo_url: str = "https://github.com/liamchalcroft/med-rs.git",
    branch: str = "main",
):
    import subprocess
    import shutil
    import os
    from datetime import datetime

    workdir = "/tmp/med-rs"

    print("=== medrs Benchmark Runner ===")
    print(f"Mode: {mode}")
    print(f"Repo: {repo_url}")
    print(f"Branch: {branch}")
    print(f"Started: {datetime.now().isoformat()}")
    print()

    print("Cloning repository...")
    subprocess.run(["git", "clone", "--depth", "1", "-b", branch, repo_url, workdir], check=True)
    os.chdir(workdir)

    print("Building medrs with maturin...")
    subprocess.run(
        ["maturin", "build", "--release"],
        check=True,
        env={**os.environ, "PATH": "/root/.cargo/bin:" + os.environ["PATH"]},
    )

    wheel = (
        list((workdir / "target/wheels").glob("*.whl"))[0]
        if (workdir := __import__("pathlib").Path(workdir)) and (workdir / "target/wheels").exists()
        else None
    )
    if wheel:
        subprocess.run(["pip", "install", str(wheel)], check=True)
    else:
        subprocess.run(
            ["maturin", "develop", "--release"],
            check=True,
            cwd=str(workdir),
            env={**os.environ, "PATH": "/root/.cargo/bin:" + os.environ["PATH"]},
        )

    print("Verifying installation...")
    subprocess.run(
        ["python", "-c", "import medrs; print('medrs imported successfully')"], check=True
    )

    flag = f"--{mode}"
    benchmarks = [
        ("medrs", "bench_medrs.py"),
        ("MONAI", "bench_monai.py"),
        ("TorchIO", "bench_torchio.py"),
        ("nibabel", "bench_nibabel.py"),
        ("SimpleITK", "bench_simpleitk.py"),
        ("Mgzip", "bench_mgzip.py"),
        ("FastLoader", "bench_fastloader.py"),
        ("Memory", "bench_memory.py"),
    ]

    results_dir = workdir / "benchmarks" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    for name, script in benchmarks:
        print(f"\n{'=' * 60}")
        print(f"Running {name} benchmarks...")
        print("=" * 60)
        try:
            subprocess.run(
                ["python", f"benchmarks/{script}", flag],
                check=True,
                cwd=str(workdir),
                timeout=1800,
            )
        except subprocess.TimeoutExpired:
            print(f"WARNING: {name} benchmark timed out after 30 minutes")
        except subprocess.CalledProcessError as e:
            print(f"WARNING: {name} benchmark failed: {e}")

    print("\n" + "=" * 60)
    print("Generating plots...")
    print("=" * 60)
    subprocess.run(["python", "benchmarks/plot_results.py"], cwd=str(workdir), check=False)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"/results/{timestamp}_{mode}"
    shutil.copytree(str(results_dir), output_dir)

    print(f"\n{'=' * 60}")
    print("BENCHMARKS COMPLETE")
    print(f"{'=' * 60}")
    print(f"Results saved to: {output_dir}")
    print(f"Finished: {datetime.now().isoformat()}")

    volume.commit()

    return output_dir


@app.function(
    image=image,
    cpu=2,
    memory=4096,
    timeout=300,
    volumes={"/results": volume},
)
def list_results():
    import os

    results = []
    for entry in sorted(os.listdir("/results")):
        path = f"/results/{entry}"
        if os.path.isdir(path):
            files = os.listdir(path)
            results.append({"name": entry, "files": len(files)})
    return results


@app.function(
    image=modal.Image.debian_slim().pip_install("matplotlib", "numpy"),
    cpu=2,
    memory=4096,
    timeout=300,
    volumes={"/results": volume},
)
def download_results(run_name: str, local_path: str = "./benchmark_results"):
    import shutil
    import os

    src = f"/results/{run_name}"
    if not os.path.exists(src):
        available = os.listdir("/results")
        raise ValueError(f"Run '{run_name}' not found. Available: {available}")

    shutil.copytree(src, local_path, dirs_exist_ok=True)
    return f"Downloaded {run_name} to {local_path}"


@app.local_entrypoint()
def main(mode: str = "quick", branch: str = "main"):
    print(f"Starting benchmark run (mode={mode}, branch={branch})...")
    result_path = run_benchmarks.remote(mode=mode, branch=branch)
    print(f"\nResults saved to Modal volume: {result_path}")
    print("\nTo download results:")
    print(f"  modal volume get medrs-results {result_path.split('/')[-1]} ./benchmark_results/")
