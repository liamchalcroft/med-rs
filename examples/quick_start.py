"""Load, inspect, transform, and save an image.

Run: python examples/quick_start.py [image.nii.gz]
Without an argument, a synthetic image is used.
"""

import sys
import tempfile
from pathlib import Path

import numpy as np

import medrs


def main() -> None:
    out = Path(tempfile.mkdtemp())
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
    else:
        path = out / "synthetic.nii.gz"
        affine = np.diag([-1.2, 1.2, 2.0, 1.0])  # LAS, anisotropic voxels
        data = np.random.default_rng(0).normal(100, 20, (96, 96, 48)).astype(np.float32)
        medrs.NiftiImage(data, affine).save(path)

    img = medrs.load(path)
    print(img)
    print("header:", {k: img.header[k] for k in ("version", "sform_code", "qform_code")})

    processed = (
        img.reorient("RAS").resample((1.0, 1.0, 1.0)).crop_or_pad((128, 128, 96)).z_normalize()
    )
    print("processed:", processed)

    array = processed.to_numpy()
    print(f"mean {array.mean():.3f}, std {array.std():.3f}, orientation {processed.orientation}")

    target = out / "processed.nii.gz"
    processed.save(target)
    print("saved", target)


if __name__ == "__main__":
    main()
