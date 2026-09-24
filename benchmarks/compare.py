"""Compare medrs with other Python medical imaging libraries on the same jobs.

Every library does the same work and produces materialized NumPy data, so
lazy loading cannot flatter anyone:

* load:   read a whole volume into memory
* patch:  read one 96^3 patch
* prepro: reorient to RAS, resample to 1.5 mm, and z-normalize

Libraries that are not installed are skipped. Run with
``python benchmarks/compare.py`` (add ``--size 256`` for a larger volume).
"""

from __future__ import annotations

import argparse
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable

import numpy as np

import medrs

PATCH = 96
FILES = {"nii": "v.nii", "nii.gz": "v.nii.gz", "mgzip": "v.mgz.nii.gz", "jvol": "v.jvol"}


def timed(fn: Callable[[], object], repeats: int) -> float:
    fn()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - start)
    return statistics.median(samples) * 1000


def volume(n: int) -> np.ndarray:
    """A smooth phantom with noise: compresses like a real MRI."""
    rng = np.random.default_rng(0)
    z, y, x = np.mgrid[0:n, 0:n, 0:n].astype(np.float32) / n - 0.5
    head = (x**2 + y**2 + z**2 < 0.2).astype(np.float32)
    data = head * (300 + 80 * np.sin(20 * x) * np.cos(15 * y))
    return (data + head * rng.normal(0, 5, data.shape)).astype(np.float32)


def jobs(files: dict[str, str], n: int) -> dict[str, dict[str, Callable[[], object]]]:
    lo = (n - PATCH) // 2
    window = (slice(lo, lo + PATCH),) * 3
    out: dict[str, dict[str, Callable[[], object]]] = {"medrs": {}}
    for fmt, path in files.items():
        out["medrs"][f"load {fmt}"] = lambda p=path: medrs.load(p).to_numpy(copy=True)
        out["medrs"][f"patch {fmt}"] = lambda p=path: medrs.load_cropped(
            p, (lo,) * 3, (PATCH,) * 3
        ).to_numpy(copy=True)
    pipeline = medrs.Pipeline().reorient("RAS").resample_to_spacing((1.5,) * 3).z_normalize()
    out["medrs"]["prepro"] = lambda: pipeline.apply(medrs.load(files["nii"])).to_numpy()

    try:
        import nibabel as nib
        from nibabel.processing import resample_to_output

        out["nibabel"] = {}
        for fmt in ("nii", "nii.gz"):
            path = files[fmt]
            # np.array forces the read; np.asarray would return an unread memmap.
            out["nibabel"][f"load {fmt}"] = lambda p=path: np.array(nib.load(p).dataobj)
            out["nibabel"][f"patch {fmt}"] = lambda p=path: np.asarray(nib.load(p).dataobj[window])

        def nib_prepro() -> np.ndarray:
            img = nib.as_closest_canonical(nib.load(files["nii"]))
            data = resample_to_output(img, (1.5,) * 3, order=1).get_fdata(dtype=np.float32)
            return (data - data.mean()) / data.std()

        out["nibabel"]["prepro"] = nib_prepro
    except ImportError:
        pass

    try:
        import SimpleITK as sitk

        out["simpleitk"] = {}
        for fmt in ("nii", "nii.gz"):
            path = files[fmt]
            out["simpleitk"][f"load {fmt}"] = lambda p=path: sitk.GetArrayFromImage(
                sitk.ReadImage(p)
            )
    except ImportError:
        pass

    try:
        from monai.transforms import (
            Compose,
            LoadImage,
            NormalizeIntensity,
            Orientation,
            Spacing,
        )

        load = LoadImage(image_only=True, ensure_channel_first=True)
        out["monai"] = {
            f"load {fmt}": lambda p=files[fmt]: load(p).numpy() for fmt in ("nii", "nii.gz")
        }
        transforms = Compose(
            [Orientation(axcodes="RAS"), Spacing(pixdim=(1.5,) * 3), NormalizeIntensity()]
        )
        out["monai"]["prepro"] = lambda: transforms(load(files["nii"])).numpy()
    except ImportError:
        pass
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=192, help="volume edge length")
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--run", nargs=2, metavar=("LIBRARY", "JOB"), help=argparse.SUPPRESS)
    parser.add_argument("--dir", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.run:
        # Child process: time one job so libraries never share process state.
        medrs.set_cache_limits(max_entries=0)  # measure real reads
        files = {fmt: os.path.join(args.dir, name) for fmt, name in FILES.items()}
        library, job = args.run
        print(timed(jobs(files, args.size)[library][job], args.repeats))
        return

    with tempfile.TemporaryDirectory() as tmp:
        image = medrs.NiftiImage(volume(args.size), np.diag([1.0, 1.0, 1.0, 1.0]))
        files = {fmt: os.path.join(tmp, name) for fmt, name in FILES.items()}
        image.save(files["nii"])
        image.save(files["nii.gz"])
        image.save(files["mgzip"], mgzip=True)
        image.save(files["jvol"])
        results: dict[str, dict[str, float]] = {}
        for library, library_jobs in jobs(files, args.size).items():
            for job in library_jobs:
                command = [sys.executable, __file__, "--size", str(args.size)]
                command += ["--repeats", str(args.repeats), "--dir", tmp, "--run", library, job]
                output = subprocess.run(command, capture_output=True, text=True, check=True)
                results.setdefault(library, {})[job] = float(output.stdout)

    machine = platform.processor() or platform.machine()
    print(f"{args.size}^3 float32 volume, {os.cpu_count()} CPUs, {machine}")
    print(f"median of {args.repeats} runs, milliseconds\n")
    names = sorted({job for lib in results.values() for job in lib})
    libraries = list(results)
    print(f"{'job':<16}" + "".join(f"{lib:>12}" for lib in libraries))
    for job in names:
        row = "".join(
            f"{results[lib][job]:>12.1f}" if job in results[lib] else f"{'-':>12}"
            for lib in libraries
        )
        print(f"{job:<16}{row}")


if __name__ == "__main__":
    main()
