"""Apples-to-apples load+materialize comparison across libraries.

Every library is timed doing the same job: open the file and produce the full
voxel array in memory (native dtype). This avoids the lazy-open-vs-materialize
mismatch that inflates naive "load" comparisons (medrs and nibabel both return
lazy proxies on open; the honest comparison forces the data to be read).

Run: python benchmarks/bench_load_comparison.py
Writes: benchmarks/results/load_comparison.json
"""

import json
import os
import statistics
import tempfile
import time
from pathlib import Path

import numpy as np

SIZES = [(128, 128, 128), (256, 256, 256)]
WARMUP = 2
ITERS = 7


def _time(fn, warmup=WARMUP, iters=ITERS):
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(iters):
        t = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t) * 1000.0)
    return min(samples), statistics.median(samples)


def _force(arr):
    # Touch every element so lazy proxies (nibabel ArrayProxy) fully materialize.
    return float(np.asarray(arr).sum())


def _loaders():
    loaders = {}
    import medrs

    loaders["medrs"] = lambda f: _force(medrs.load(f).to_numpy())
    try:
        import nibabel as nib

        loaders["nibabel"] = lambda f: _force(np.asarray(nib.load(f).dataobj))
    except ImportError:
        pass
    try:
        from monai.transforms import LoadImage

        _li = LoadImage(image_only=True)
        loaders["monai"] = lambda f: _force(np.asarray(_li(f)))
    except ImportError:
        pass
    try:
        import torchio as tio

        loaders["torchio"] = lambda f: _force(tio.ScalarImage(f).data.numpy())
    except ImportError:
        pass
    try:
        import SimpleITK as sitk

        loaders["simpleitk"] = lambda f: _force(sitk.GetArrayFromImage(sitk.ReadImage(f)))
    except ImportError:
        pass
    return loaders


def main():
    import medrs

    loaders = _loaders()
    results = {}
    with tempfile.TemporaryDirectory() as d:
        for size in SIZES:
            # Smooth structured volume that compresses like real MRI (~2-3x under
            # gzip), plus light noise. Random data is incompressible and an
            # unrepresentative worst case for the gzipped comparison.
            zz, yy, xx = np.mgrid[0 : size[0], 0 : size[1], 0 : size[2]]
            vol = (np.sin(xx / 20.0) * np.cos(yy / 25.0) + 0.5 * np.sin(zz / 15.0)).astype(
                np.float32
            )
            vol += np.random.randn(*size).astype(np.float32) * 0.05
            img = medrs.NiftiImage(vol)
            nii = os.path.join(d, "v.nii")
            gz = os.path.join(d, "v.nii.gz")
            img.save(nii)
            img.save(gz)
            for label, path in [("nii", nii), ("nii.gz", gz)]:
                key = f"{size[0]}^3 {label}"
                results[key] = {}
                for name, fn in loaders.items():
                    best, med = _time(lambda: fn(path))
                    results[key][name] = {"best_ms": round(best, 2), "median_ms": round(med, 2)}
                    print(f"{key:16} {name:10} best={best:8.2f}ms median={med:8.2f}ms")
                print()

    out = Path(__file__).parent / "results" / "load_comparison.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"wrote {out}")

    # Speedup table vs medrs
    print("\nSpeedup vs medrs (median, load+materialize):")
    for key, libs in results.items():
        base = libs.get("medrs", {}).get("median_ms")
        if not base:
            continue
        parts = [
            f"{n} {libs[n]['median_ms'] / base:.1f}x" for n in libs if n != "medrs"
        ]
        print(f"  {key:16} " + "  ".join(parts))


if __name__ == "__main__":
    main()
