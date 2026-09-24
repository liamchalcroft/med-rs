# Contributing

Thanks for helping improve medrs. Bug reports with a small reproducing script
are especially valuable. If a file loads differently in medrs than in nibabel,
please include the output of `medrs info` for it.

## Layout

| Path | Contents |
|---|---|
| `src/` | The `medrs` Rust crate: `nifti` (I/O), `transforms`, `pipeline`, `loader`, `jvol` |
| `python/src/` | PyO3 bindings (the `medrs-python` crate, built by maturin) |
| `python/medrs/` | The Python package: re-exports, type stubs, `monai.py`, `cli.py` |
| `tests/*.rs` | Rust integration and property tests (unit tests live next to the code) |
| `tests/python/` | Python tests, mostly checked against nibabel |
| `docs/` | The user guide. `mkdocs.yml` builds it, the README, and the changelog into medrs.readthedocs.io |
| `benches/` | Criterion benchmarks |
| `benchmarks/` | `compare.py` (the README comparison) and `jvol_chunks.py` (the guide's chunk-shape table) |
| `examples/` | Small runnable examples (run in CI) |

## Setup

You need a Rust toolchain (1.83 or newer) and Python 3.10 or newer.

```bash
python -m venv .venv && source .venv/bin/activate
pip install --group dev  # maturin, test and lint tools (pip 25.1 or newer)
maturin develop          # builds the extension into the virtualenv
```

## Checks

CI runs the following; please run them before opening a pull request:

```bash
cargo fmt --all --check
cargo clippy --workspace --all-features --all-targets -- -D warnings
cargo test --workspace --all-features --exclude medrs-python
maturin develop && pytest
ruff check . && ruff format --check . && mypy
python -m mypy.stubtest medrs._medrs --mypy-config-file pyproject.toml
mkdocs build  # strict: broken links and anchors fail
```

Benchmarks: `cargo bench --all-features` and `python benchmarks/compare.py`.
Preview the documentation site with `mkdocs serve`.

## Guidelines

- **Correctness first.** Every behaviour change needs a test. Where there is a
  reference implementation (nibabel, scipy, numpy), test against it.
- **Keep geometry consistent.** Spatial transforms must update the affine so
  that every voxel keeps its world position; derive header updates from the
  transform's `GridChange` (see `src/transforms/geometry.rs`).
- **No panics in library code.** Return `medrs::Error`; the crate denies
  `unwrap`, `expect`, and `panic!` outside tests.
- **Release the GIL** in bindings around any work that touches voxel data.
- Update `CHANGELOG.md` for user-visible changes.

## Releasing

1. Update the version in the root `Cargo.toml` (`[workspace.package]`) and add
   a dated `CHANGELOG.md` entry.
2. Merge to `main`, then publish a GitHub release tagged `vX.Y.Z`. The publish
   workflow checks that the tag matches the version, builds wheels and the
   sdist, and publishes to PyPI and crates.io.
