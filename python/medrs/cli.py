"""The ``medrs`` command: inspect and convert image files."""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

import medrs

_FORMATS = {"nii": ".nii", "nii.gz": ".nii.gz", "mgzip": ".nii.gz", "jvol": ".jvol"}
_KNOWN_SUFFIXES = (".nii.gz", ".nii", ".hdr.gz", ".hdr", ".img.gz", ".img", ".jvol")


def _stem(path: Path) -> str:
    name = path.name
    for suffix in _KNOWN_SUFFIXES:
        if name.lower().endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def _info(args: argparse.Namespace) -> int:
    for path in args.paths:
        h = medrs.load_header(path)
        rank = min(len(h["shape"]), 3)
        spacing = ", ".join(f"{v:g}" for v in np.linalg.norm(h["affine"][:3, :rank], axis=0))
        print(f"{path}")
        print(f"  shape      {tuple(h['shape'])}  ({h['dtype']}, {h['version']})")
        print(f"  voxel size ({spacing}) {h['spatial_units']}")
        print(f"  xform      sform_code={h['sform_code']} qform_code={h['qform_code']}")
        if h["scl_slope"] not in (0.0, 1.0) or h["scl_inter"] != 0.0:
            print(f"  scaling    {h['scl_slope']:g} * x + {h['scl_inter']:g}")
        if h["descrip"]:
            print(f"  descrip    {h['descrip']}")
        for row in h["affine"]:
            print("  affine    " + " ".join(f"{v:10.4f}" for v in row))
    return 0


def _convert_one(src: Path, args: argparse.Namespace) -> Path:
    out_dir = Path(args.output) if args.output else src.parent
    dst: Path = out_dir / (_stem(src) + args.suffix + _FORMATS[args.to])
    if dst.resolve() == src.resolve():
        raise ValueError(f"{src}: output would overwrite the input; use -o or --suffix")
    medrs.load(src).save(
        dst,
        mgzip=args.to == "mgzip",
        compression_level=args.level if args.to in ("nii.gz", "mgzip") else None,
        quality=args.quality if args.to == "jvol" else None,
    )
    return dst


def _convert(args: argparse.Namespace) -> int:
    if args.output:
        os.makedirs(args.output, exist_ok=True)
    sources = [Path(p) for p in args.inputs]
    failed = 0
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for src, future in [(s, pool.submit(_convert_one, s, args)) for s in sources]:
            try:
                print(f"{src} -> {future.result()}")
            except Exception as e:  # report every failure, keep converting the rest
                failed += 1
                print(f"error: {src}: {e}", file=sys.stderr)
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="medrs", description=__doc__)
    parser.add_argument("--version", action="version", version=f"medrs {medrs.__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    info = commands.add_parser("info", help="print image headers")
    info.add_argument("paths", nargs="+", type=Path)
    info.set_defaults(run=_info)

    convert = commands.add_parser(
        "convert",
        help="convert images between formats",
        description="Convert images. 'mgzip' writes block-compressed .nii.gz, which is still "
        "standard gzip but decompresses in parallel with medrs.",
    )
    convert.add_argument("inputs", nargs="+", type=Path)
    convert.add_argument("--to", required=True, choices=sorted(_FORMATS))
    convert.add_argument("-o", "--output", help="output directory (default: next to each input)")
    convert.add_argument("--suffix", default="", help="text appended to output names")
    convert.add_argument("--level", type=int, help="gzip compression level (0-9)")
    convert.add_argument(
        "--quality", type=int, help="lossy .jvol quality (1-100); lossless if omitted"
    )
    convert.add_argument("-j", "--jobs", type=int, default=min(4, os.cpu_count() or 1))
    convert.set_defaults(run=_convert)

    args = parser.parse_args(argv)
    try:
        return int(args.run(args))
    except (OSError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
