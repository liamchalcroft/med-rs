"""How the `.jvol` chunk shape affects file size and random-crop reads.

Run: python benchmarks/jvol_chunks.py [image] [--quality Q ...] [--threads N]

Saves the image (as float32) with each chunk shape, then times `load_cropped`
at random positions and a full `load`, reporting medians in milliseconds.
By default the image is the MPRAGE test fixture and medrs uses one thread,
which approximates the cost per loader worker.
"""

import argparse
import statistics
import tempfile
import time
from pathlib import Path

import numpy as np

import medrs

FIXTURE = Path(__file__).parents[1] / "tests" / "fixtures" / "mprage_img.nii.gz"


def median_ms(fn, repeats):
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return statistics.median(times) * 1e3


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", nargs="?", type=Path, default=FIXTURE)
    parser.add_argument("--quality", type=int, nargs="+", default=[0, 60], help="0: lossless")
    parser.add_argument("--chunks", type=int, nargs="+", default=[16, 32, 48, 64, 96, 128])
    parser.add_argument("--crops", type=int, nargs="+", default=[64, 96, 128])
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=30)
    args = parser.parse_args()

    medrs.set_num_threads(args.threads)
    image = medrs.load(args.image).with_dtype("float32")
    spatial = image.shape[:3]
    print(f"{args.image.name}: {spatial}, {image.to_numpy().nbytes / 1e6:.1f} MB as float32")
    rng = np.random.default_rng(0)
    crops = [c for c in args.crops if all(c <= s for s in spatial)]
    with tempfile.TemporaryDirectory() as tmp:
        for quality in args.quality:
            label = "lossless" if quality == 0 else f"lossy, quality {quality}"
            print(f"\n{label}")
            columns = ["Chunk", "Size (MB)", *(f"{c}³ crop" for c in crops), "Full load"]
            print("| " + " | ".join(columns) + " |")
            print("|---" * (len(crops) + 3) + "|")
            for chunk in args.chunks:
                path = Path(tmp) / f"{quality}_{chunk}.jvol"
                image.save(path, quality=quality or None, chunk_shape=(chunk,) * 3)
                cells = [f"{chunk}³", f"{path.stat().st_size / 1e6:.1f}"]
                for crop in crops:
                    offsets = [
                        tuple(int(rng.integers(0, s - crop + 1)) for s in spatial)
                        for _ in range(args.repeats)
                    ]
                    positions = iter(offsets * 2)
                    medrs.load_cropped(path, offsets[0], (crop,) * 3)  # warm the page cache
                    ms = median_ms(
                        lambda: medrs.load_cropped(path, next(positions), (crop,) * 3),  # noqa: B023
                        args.repeats,
                    )
                    cells.append(f"{ms:.1f}")
                cells.append(f"{median_ms(lambda: medrs.load(path).to_numpy(), 5):.0f}")  # noqa: B023
                print("| " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
