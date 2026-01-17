#!/usr/bin/env python3
"""Command-line tools for medrs.

Usage:
    python -m medrs.cli convert-mgzip data/*.nii.gz
    python -m medrs.cli convert-mgzip data/ --recursive --workers 8
"""

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional


def find_nifti_files(paths: List[str], recursive: bool = False) -> List[Path]:
    """Find all .nii.gz files in given paths."""
    files = []
    for path_str in paths:
        path = Path(path_str)
        if path.is_file():
            if path.suffix == ".gz" or path.suffixes == [".nii", ".gz"]:
                files.append(path)
        elif path.is_dir():
            pattern = "**/*.nii.gz" if recursive else "*.nii.gz"
            files.extend(path.glob(pattern))
    return sorted(set(files))


def convert_single_file(
    input_path: Path,
    output_dir: Optional[Path],
    num_threads: int,
    overwrite: bool,
) -> tuple[Path, Optional[str]]:
    """Convert a single file to Mgzip format.

    Returns (output_path, error_message) where error_message is None on success.
    """
    from . import convert_to_mgzip

    if output_dir:
        stem = input_path.stem
        if stem.endswith(".nii"):
            stem = stem[:-4]
        output_path = output_dir / f"{stem}.nii.mgz"
    else:
        stem = input_path.stem
        if stem.endswith(".nii"):
            stem = stem[:-4]
        output_path = input_path.with_name(f"{stem}.nii.mgz")

    if output_path.exists() and not overwrite:
        return output_path, "skipped (exists)"

    try:
        convert_to_mgzip(str(input_path), str(output_path), num_threads)
        return output_path, None
    except Exception as e:
        return output_path, str(e)


def cmd_convert_mgzip(args: argparse.Namespace) -> int:
    """Convert gzip NIfTI files to Mgzip format."""
    files = find_nifti_files(args.paths, args.recursive)

    if not files:
        print("No .nii.gz files found", file=sys.stderr)
        return 1

    print(f"Found {len(files)} file(s) to convert")

    output_dir = Path(args.output) if args.output else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    success = 0
    skipped = 0
    failed = 0

    if args.workers == 1:
        for f in files:
            out, err = convert_single_file(f, output_dir, args.threads, args.overwrite)
            if err is None:
                success += 1
                if args.verbose:
                    print(f"  {f} -> {out}")
            elif err == "skipped (exists)":
                skipped += 1
                if args.verbose:
                    print(f"  {f} -> skipped (exists)")
            else:
                failed += 1
                print(f"  {f} -> FAILED: {err}", file=sys.stderr)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(convert_single_file, f, output_dir, args.threads, args.overwrite): f
                for f in files
            }
            for future in as_completed(futures):
                f = futures[future]
                out, err = future.result()
                if err is None:
                    success += 1
                    if args.verbose:
                        print(f"  {f} -> {out}")
                elif err == "skipped (exists)":
                    skipped += 1
                    if args.verbose:
                        print(f"  {f} -> skipped (exists)")
                else:
                    failed += 1
                    print(f"  {f} -> FAILED: {err}", file=sys.stderr)

    print(f"\nDone: {success} converted, {skipped} skipped, {failed} failed")
    return 0 if failed == 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="medrs",
        description="medrs command-line tools",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    convert_parser = subparsers.add_parser(
        "convert-mgzip",
        help="Convert .nii.gz files to Mgzip format for faster loading",
    )
    convert_parser.add_argument(
        "paths",
        nargs="+",
        help="Files or directories to convert",
    )
    convert_parser.add_argument(
        "-o",
        "--output",
        help="Output directory (default: same as input)",
    )
    convert_parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Search directories recursively",
    )
    convert_parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=1,
        help="Number of parallel file conversions (default: 1)",
    )
    convert_parser.add_argument(
        "-t",
        "--threads",
        type=int,
        default=0,
        help="Compression threads per file (0 = auto, default: 0)",
    )
    convert_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .nii.mgz files",
    )
    convert_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show progress for each file",
    )
    convert_parser.set_defaults(func=cmd_convert_mgzip)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
