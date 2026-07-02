---
orphan: true
---

# Alternative loading strategies for mmap-style speed on compressed volumes

Status: design research, no code changes. Written 2026-07-02.

## The problem, precisely

medrs gets true zero-copy loading (`memmap2::Mmap::map`, see `src/nifti/io.rs`) only for
uncompressed `.nii`. Every compressed path (`.nii.gz` via libdeflate, `.nii.mgz` via `gzp`
parallel decode, `.jvol` via wavelet + Rice + zstd) requires decoding bytes before any voxel
is usable, because compressed bytes are not the voxel array. That is not a medrs limitation,
it is definitional: you cannot memory-map a value that has not been computed yet. `jvol`
already narrows the gap for one case, resolution: `load_downsampled` skips the fine wavelet
subbands and returns a factor-4 preview at roughly 30x less decode work than a full load
(`src/jvol/mod.rs:304-363`). It does nothing for the other case, space: a small spatial crop
still requires decoding the entire wavelet coefficient set, because the global lifting
transform in `codec/wavelet.rs` correlates coefficients across the whole volume. Cropped
reads on the gzip paths (`load_with_crop_gzipped`, `load_cropped_gzipped`,
`src/nifti/io.rs:1418-1463`) fully decompress first and slice afterward, for the same
structural reason (deflate is a stream cipher against sequential state, there is no random
access point without external indexing).

This document surveys seven candidate approaches to closing that gap, verified against
current (2025/2026) sources rather than training memory, and ends with a concrete roadmap.

## Executive summary, ranked by value for effort

1. **`madvise` on the existing mmap path.** A one-line gap: `load_uncompressed` calls
   `Mmap::map` with no hint at all, while the gzip path already applies
   `posix_fadvise(POSIX_FADV_SEQUENTIAL)`. Closing this inconsistency costs almost nothing
   and gives a modest, honestly-described win (fewer page-fault stalls on first sequential
   touch). Ship this first.
2. **Document bf16/int16 uncompressed-and-mmapped as a first-class strategy.** medrs
   already supports these dtypes end to end and `load_uncompressed` already mmaps any of
   them with zero interpretation. This is not a feature to build, it is a paragraph to write
   and, optionally, a conversion helper. It is the simplest real "smaller file, still zero
   decode" trade medrs has, complementary to (not a replacement for) `.jvol`/gzip.
3. **A chunked companion format for genuine crop-first-on-compressed**, if and when patch-based
   training against large volumes on compressed storage becomes a real, measured need. Blosc2's
   super-chunk/CFrame format and Zarr v3's sharding codec both do this natively and are
   specified, not hacked. Neither is a quick win: both require a new storage layer alongside
   `.jvol`, not a change to it. Recommend prototyping before committing.
4. **Everything else researched here is a "no" for now**: GPU decompression (nvCOMP/GDS/KvikIO),
   tiled JPEG2000/JP3D, a tiled rewrite of `jvol` itself, io_uring, O_DIRECT, and
   transparent filesystem compression. Reasons are specific to each and given below; none of
   them is a blanket dismissal of the technology, they are dismissals of the technology for
   medrs's actual workload (volumes of tens to low hundreds of MB, heterogeneous researcher
   hardware, repeat-epoch access patterns).
5. **OME-Zarr multiscale pyramids**: worth tracking for interop, not worth adopting for the
   core loading path. `jvol`'s single-coefficient-set resolution-progressive decode already
   beats explicit per-level pyramid storage on both disk footprint and decode cost for the
   "coarse preview" use case that OME-Zarr's multiscale groups target.

## Comparison table

| Approach | Closeness to mmap zero-copy | Compression ratio | Crop-first / progressive | CPU vs GPU | Rust + Python maturity | Integration effort | Headline downside |
|---|---|---|---|---|---|---|---|
| Quantized-uncompressed-mmap (bf16/int16, already supported) | Exact (true mmap, no decode) | 2-4x vs f32-uncompressed only | N/A, whole file always available | CPU (trivial upcast) | Already shipped in medrs | None, docs only | Not a compression strategy, doesn't touch already-compressed data |
| `madvise` on mmap path | Exact (same true mmap, fewer stalls) | Unaffected | N/A | CPU | `libc` crate, trivial | Few lines, Linux cfg-gated | Linux-only, minor magnitude |
| Blosc2 super-chunk/CFrame | Near, decode only touched chunks | ~2-5x on scientific float data (comparable to today's gzip path) | Yes, per-chunk random access, spec-documented | CPU (no GPU path found) | Rust binding is young/thin (`blosc2` crate v0.2.2, single maintainer); would likely need custom FFI atop `blosc2-src` | Moderate-high: new storage layer, own bindings | Best single-file fit (matches medrs's one-file-per-volume model), but the safe Rust API doesn't yet expose the parts you need |
| Zarr v3 sharding (`zarrs` crate) | Near, decode only touched chunks (sync path mature, async partial reads "experimental" per its own docs) | Similar to Blosc2 (same codec set: blosc/zstd/gzip) | Yes, first-class spec feature (ZEP0002) | CPU in Rust; GPU decode exists only Python-side via KvikIO+nvCOMP | `zarrs` is actively developed (v0.23.13, mid-2026), genuinely mature for a scientific Rust crate | High: directory/shard-based multi-file storage model, a real departure from a single `.nii`/`.jvol` file | Storage model change, not a drop-in feature |
| Google TensorStore | Same class as Zarr (chunked, async) | Same codec set | Yes | CPU, async I/O only | No Rust bindings exist at all | Not viable without writing and maintaining full C++ FFI | Zero Rust story |
| GPU decompression (nvCOMP + GDS/cuFile via KvikIO) | Good in principle for huge sequential reads, poor fit for medrs's per-volume sizes | Unaffected, same codecs, different decode location | N/A directly (orthogonal axis) | GPU | Python/C++ (KvikIO) mature; Rust essentially nonexistent (nvCOMP has no Rust bindings, `cufile-sys` is thin/unmaintained) | Multi-week FFI project minimum | Requires datacenter GPUs (GDS explicitly excludes consumer GeForce cards), specific NVMe/filesystem stack, silent no-op fallback when unsupported, nvCOMP EULA excludes "Critical Applications including medical devices" |
| Tiled JPEG2000 / JP3D | Would be near if it existed for 3D and float data | Unclear, likely worse than jvol for smooth scientific volumes | Yes in the 2D codestream (ROI decode is native), no for 3D | CPU | JP3D (the 3D component) was dropped from OpenJPEG after 2.4.0, build status "unknown"; no maintained Rust bindings target it | Not viable, would mean maintaining an abandoned codec branch | Dead end for medrs's exact need |
| Tiled rewrite of `jvol` itself | Would give crop-first at full resolution | Likely worse than global-wavelet jvol at tile boundaries, plus halo overhead | Yes, spatially, but loses today's uninterrupted progressive preview across tiles | CPU | N/A, internal | Large: touches `wavelet.rs`, `subbands.rs`, container format, and directly risks the lossless guarantee jvol relies on | High risk for a capability a companion chunked format already provides more cheaply |
| OME-Zarr / OME-NGFF multiscale | Chunk-level near (via underlying Zarr chunking), pyramid levels don't help resolution-progressive beyond what jvol already does | Chunked, independently-compressed levels cost real duplication (~14%+ geometric overhead, often worse in practice) vs jvol's zero-duplication single coefficient set | Yes for spatial crop-first (genuine gap jvol has); no advantage for progressive-resolution (jvol already wins here) | CPU (Rust via `zarrs`/`zarrs_ome`); GPU only Python-side | `zarrs_ome` (Rust) and `ome-zarr-py` (Python) both mature | Adopting means a parallel storage convention, not an extension of jvol | Microscopy-first spec; neuroimaging bridging (`nifti-zarr`) is still release-candidate, not adopted by BIDS/nibabel mainline for MRI |
| madvise / io_uring / O_DIRECT / transparent FS compression (grouped, OS-level) | Mixed, see per-item detail above and below | Unaffected (except FS compression, which is opaque) | N/A | CPU | N/A | Low for madvise, N/A (don't build) for the rest | io_uring solves a syscall-count problem medrs doesn't have; O_DIRECT actively defeats the page-cache reuse that makes repeat-epoch training fast; transparent FS compression works but is entirely outside the library's control and ZFS's ARC/page-cache interaction has a documented double-caching cliff |

## Per-approach detail

### 1. Quantized-uncompressed-mmap (the quick win that already exists)

medrs's `DataType` enum already includes `Int8`, `Int16`, `UInt16`, `Float16`, and
`BFloat16` alongside `Float32`/`Float64` (`src/nifti/header.rs`), and `load_uncompressed`
maps the file with `memmap2::Mmap::map` unconditionally, with zero dtype interpretation
(`src/nifti/io.rs:515-540`). That means storing an intensity volume as bf16 instead of f32
already gets a true zero-copy 2x-smaller file today, with no code change: the only cost is
a cheap elementwise upcast to f32 at patch-extraction time (microseconds for a 64^3 or 128^3
crop), a different order of magnitude from a gzip decompress over a multi-MB buffer.

We could not find a citeable domain-specific study on bf16-storage precision loss for MRI/CT
voxel intensities in a training context; that gap should be stated honestly rather than
papered over. The indirect evidence is strong, however: bf16 is the standard dtype for
*computing* gradients and activations during training (Kalamkar et al. 2019,
[arXiv:1905.12322](https://arxiv.org/abs/1905.12322), matches fp32 accuracy across a range of
training workloads), so storing an intensity volume in bf16, which is only ever *read* and
upcast before arithmetic, is a strictly smaller ask than computing in it.

Caveats worth stating in medrs's own docs: label/segmentation volumes already need lossless
integer dtypes and get no further benefit from narrowing (int16 is already exact for typical
label cardinalities). And this is orthogonal to compression, not competitive with it: bf16
buys 2x over f32-uncompressed, `.jvol`/gzip still win on raw disk footprint (5-10x+) when disk
space, not decode latency, is the binding constraint. The honest framing is "use both where
it matters": bf16 storage for the intensity channel when I/O latency dominates, `.jvol` when
disk footprint dominates, and combine them (a bf16-precision quantized jvol path) if both
matter, which the codec already supports as a lossy option.

### 2. `madvise` on the mmap path

`load_uncompressed` maps the file and returns immediately with no hint to the kernel about
access pattern, while the compressed-read path already calls `posix_fadvise(fd, ...,
POSIX_FADV_SEQUENTIAL)` before `read_to_end` (`read_file_with_readahead`,
`src/nifti/io.rs:200-223`). Applying the mmap-equivalent hint, `madvise(MADV_SEQUENTIAL)`
and/or `MADV_WILLNEED` right after `Mmap::map`, closes that inconsistency for large,
sequentially-accessed volumes: it prompts more aggressive kernel readahead and earlier
eviction of pages behind the read cursor, reducing first-touch page-fault stalls
([madvise(2)](https://man7.org/linux/man-pages/man2/madvise.2.html)). We found no benchmark
specific to this exact volume-loading workload, so the expected gain should be described as
modest (order 10-20% on first-touch latency for large sequential volumes), not
transformative; but it is essentially free, Linux-only cfg-gated like the existing
readahead code, and directly consistent with an optimization medrs already trusts on the
other path. `MADV_HUGEPAGE` is not recommended: its benefits are for long-lived, write-heavy
anonymous mappings, not one-shot read-only file-backed volumes, and it adds alignment
complexity for no clear gain here.

### 3. Chunked stores: Blosc2, Zarr v3 / `zarrs`, TensorStore

All three solve "decode only the chunks a crop touches" as a first-class, specified
capability, not a hack. They differ mainly in Rust maturity and storage-model fit.

**Blosc2** ([c-blosc2](https://github.com/Blosc/c-blosc2),
[python-blosc2](https://github.com/Blosc/python-blosc2),
[blosc.org](https://www.blosc.org/)) stores independently-compressed chunks in a
super-chunk with an index, supporting selective random-access reads without decompressing
the whole thing ([schunk docs](https://blosc.org/c-blosc2/reference/schunk.html)). Its
persistent "CFrame" format is a single file that Python's `blosc2` can open with
`mmap_mode="r"` ([save/load docs](https://www.blosc.org/python-blosc2/reference/save_load.html),
[cframe format](https://blosc.org/c-blosc2/format/cframe_format.html)), the header/index
gets mapped, only touched chunks decompress into RAM. This is the closest fit to medrs's
existing single-file-per-volume model. Codecs are blosclz/lz4/lz4hc/zstd/zlib with
shuffle/bitshuffle filters; typical ratios on real (non-synthetic) scientific float data run
2-5x with zstd+shuffle
([benchmark](https://aras-p.info/blog/2023/03/02/Float-Compression-8-Blosc/),
[SciPy proceedings](https://proceedings.scipy.org/articles/gerudo-f2bc6f59-000)), comparable
to, not obviously better than, what medrs's libdeflate path already gets on `.nii.gz`. CPU
only, no GPU path. The catch: the safe Rust `blosc2` crate (v0.2.2, docs.rs, single
maintainer as of mid-2026) wraps `Chunk -> SChunk -> Ndarray` but does not clearly surface
frame/mmap or partial-decompression access in its safe API; realistically medrs would build
its own thin FFI atop the lower-level `blosc2-src`/`blosc2-sys` crates rather than rely on
the current safe wrapper.

**Zarr v3** ([zarr-specs](https://zarr-specs.readthedocs.io/),
[ZEP0002 sharding](https://zarr.dev/zeps/accepted/ZEP0002.html)) groups many inner chunks
into shard files with an index; the sharding codec spec explicitly documents the
crop-first-on-compressed pattern (read the index, locate offset/length for the requested
chunk, decode only that slice). The Rust crate `zarrs`
([docs.rs](https://docs.rs/zarrs/latest/zarrs/), v0.23.13 as of mid-2026,
[releases](https://github.com/zarrs/zarrs/releases)) is genuinely production-grade for a
scientific-data crate: sharding is fully supported, `retrieve_subchunk_opt` gives synchronous
partial reads, though the async API's partial-read support for sharding is still
experimental per its own docs. Codecs: blosc, zstd, gzip, crc32c, transpose. GPU decode
exists only on the Python side, via `kvikio`
([docs](https://docs.rapids.ai/api/kvikio/stable/zarr/)) combined with nvCOMP for LZ4, not
something `zarrs` itself exposes. The real cost is the storage model: Zarr is a directory
hierarchy (or shard files) with JSON metadata, a genuine departure from a single portable
`.nii`/`.jvol` file, adopting it means a new storage layer and migration path, not a feature
added to an existing format.

**Google TensorStore** ([tensorstore.io](https://google.github.io/tensorstore/),
[github](https://github.com/google/tensorstore)) is technically the most complete of the
three (async chunked reads, zarr2/zarr3 drivers supporting blosc/zstd/bz2/gzip) but has no
Rust bindings anywhere in its repo, issues, or docs. For a Rust-first library this rules it
out unless medrs is willing to write and maintain a full C++ FFI layer for equivalent
functionality `zarrs` already provides natively in Rust.

**Verdict:** if crop-first-on-compressed becomes a real, measured requirement (not a
hypothetical), prototype both Blosc2 CFrame (single-file fit, but build your own FFI) and
`zarrs`-backed Zarr v3 sharding (mature Rust, but multi-file storage model) against a
representative crop-heavy workload before committing. Do not build either speculatively.

### 4. GPU decompression: nvCOMP, GPUDirect Storage / cuFile, RAPIDS KvikIO

**nvCOMP** ([github.com/NVIDIA/nvcomp](https://github.com/NVIDIA/nvcomp), v4.0) supports
GPU decompression of LZ4, Snappy, ANS, GDeflate, Cascaded, Bitcomp, Zstd, Deflate, and Gzip,
so it can in principle decode the exact gzip stream `.nii.gz` uses. Published throughput is
roughly 50 GB/s Deflate decode on an A100; NVIDIA's newer Blackwell-generation dedicated
Decompression Engine claims up to 600 GB/s, but that is datacenter-only hardware (B200 /
B300 / GB200 / GB300) and falls back to a slower SM-based path for buffers over 4 MB
([NVIDIA blog](https://developer.nvidia.com/blog/speeding-up-data-decompression-with-nvcomp-and-the-nvidia-blackwell-decompression-engine/)).
Its license is the standard NVIDIA SDK EULA, free to use but explicitly **excludes Critical
Applications including medical devices**
([LICENSE](https://github.com/NVIDIA/nvcomp/blob/main/LICENSE)); medrs is a research I/O
library rather than device software, but that clause is worth flagging given the domain.

**GPUDirect Storage (GDS) / cuFile**
([docs](https://docs.nvidia.com/gpudirect-storage/)) genuinely DMAs NVMe storage straight
to GPU memory, bypassing the CPU/host bounce buffer, but is gated to
Tesla/Quadro/datacenter-class GPUs with compute capability >= 6: consumer GeForce cards
(RTX 4090/5090 included) are explicitly unsupported, because GeForce drivers lack the
`nvidia-fs` kernel interfaces GDS needs. Filesystem support is ext4/XFS on NVMe or specific
RDMA filesystems, with strict O_DIRECT and 4 KB alignment requirements. Where hardware or
filesystem support is missing, cuFile silently falls back to plain POSIX read plus host
copy, i.e. no speedup and no error, a trap for anyone assuming GDS is active. It does
compose with nvCOMP (NVIDIA ships an example doing exactly this: read compressed bytes via
GDS, decode in place with nvCOMP).

**RAPIDS KvikIO** ([github.com/rapidsai/kvikio](https://github.com/rapidsai/kvikio)) is the
real, actively-maintained way to use this from Python (v26.06.00 as of mid-2026), wrapping
cuFile with automatic fallback. Its nvCOMP Python bindings are marked deprecated in current
releases. There are no Rust bindings for nvCOMP at all; `cufile-sys` exists as raw FFI
(v0.1.1, five functions, no visible maintenance activity).

**Verdict:** this is real, capable technology aimed at a different I/O shape than medrs's.
It pays off on large, sequential, storage/PCIe-bound reads, not medrs's pattern of many
volumes in the tens-to-low-hundreds-of-MB range. Compression ratio is unaffected (same
codecs, different decode location), CPU still owns file opens and cuFile orchestration, the
hardware requirement rules out laptops, Apple Silicon, and most researchers' consumer NVIDIA
workstations outright, and Rust tooling is essentially nonexistent, meaning medrs would need
a multi-week FFI project to reach a narrow subset of users. For volumes this size, CPU
decode (libdeflate, the Rice+zstd jvol codec) is unlikely to be the actual pipeline
bottleneck compared to disk IOPS, augmentation, or Python-side overhead. Recommend
profiling the real bottleneck in a representative training pipeline before investing here;
if medrs's typical volume size ever grows into genuinely huge (multi-GB) territory on
datacenter GPU users, KvikIO from Python is the sane entry point, not hand-rolled Rust FFI.

### 5. Tiled/region-decodable codecs: JPEG2000/JP3D and a tiled `jvol`

JPEG2000 natively supports tiling, resolution-progressive decode, and region-of-interest
decode within its codestream via `opj_set_decode_area()`, only the codeblocks intersecting
the requested window get inverse-wavelet-transformed and entropy-decoded. The piece medrs
would actually need, though, native 3D volumetric wavelet transform (JP3D, ISO/IEC
15444-10), was dropped from OpenJPEG after the 2.4.0 release; its build status is listed as
"unknown" in the project's own changelog, current OpenJPEG (2.5.4, September 2025) is
2D-only. No maintained Rust binding (`openjpeg-sys`, `openjpeg2-sys`, `jp2k`) targets JP3D.
JPEG2000's core transform also operates on integer samples; float32 medical data needs
either a lossy quantization step or a research-grade lossless-float extension
([Gonzalez Bosquet et al., 2007](https://jivp-eurasipjournals.springeropen.com/articles/10.1155/2007/85385))
that never reached mainstream tooling. This is a dead end for medrs's exact need: adopting
it means either giving up native 3D and going slice-wise (losing the cross-slice
compression jvol already exploits) or resurrecting an abandoned, previously buggy codec
branch.

The alternative, tiling `jvol` itself, was also assessed and is not recommended. The
lifting steps in `codec/wavelet.rs` read cross-boundary neighbors at every decomposition
level; at a tile boundary those neighbors do not exist in-tile, requiring either a halo/
overlap region that grows (and costs redundant storage/compute) with the number of levels,
or boundary clamping that risks the exact-round-trip guarantee jvol currently provides for
lossless/label data. The resolution-progressive preview path
(`decode_downsampled_f32`) would need reworking too, since there is no longer a single
global coarse subband, independent per-tile coarse levels would need seam-blending to avoid
visible tile boundaries in a downsampled view. This touches `wavelet.rs`, `subbands.rs`, and
the container format simultaneously, a substantial rewrite that risks the codec's two
headline properties (best-in-class ratio, uninterrupted progressive preview) for a
capability an existing chunked-store companion format already provides more cheaply and
with less risk.

For context, the field that actually faces this exact problem at greater scale
(connectomics/microscopy, TB-to-PB volumes) already solved it the boring way: N5, Zarr, and
chunked HDF5 compress fixed-size chunks independently with a generic codec, at a real but
bounded compression-ratio cost. A 2025 benchmark of OME-Zarr on 3D scientific
(holotomography) volumes found lossless ratios of only ~1.4-2.0x and lossy ratios of
~3.3-4.5x with byte-shuffle+zstd/blosc/gzip
([arXiv:2503.18037](https://arxiv.org/pdf/2503.18037)), because independent-chunk
compression cannot exploit the large-scale smoothness a global wavelet transform captures.
That is the trade: chunked stores buy crop-first cheaply, jvol's global wavelet buys ratio;
you cannot have jvol's exact ratio and cheap spatial crop-first in the same file without a
major, risky rewrite.

### 6. OME-Zarr / OME-NGFF multiscale pyramids

OME-NGFF ([spec v0.5](https://ngff.openmicroscopy.org/latest/), rebased onto Zarr v3) stores
each pyramid resolution level as a separate, independently-chunked/compressed Zarr array
under one group; the spec's axis/channel conventions and tooling (viewers, OMERO) are
microscopy-first, a pure-3D MRI/CT volume fits by omitting channel/time axes rather than
natively. Neuroimaging adoption is real but narrow: BIDS only accepts OME-Zarr/OME-TIFF for
its microscopy extension, not standard MRI (NIfTI remains BIDS's mandatory MRI format);
`nifti-zarr` ([github.com/neuroscales/nifti-zarr](https://github.com/neuroscales/nifti-zarr))
is a bridging spec precisely because plain OME-NGFF lacks the NIfTI affine/world-coordinate
convention, and it is still release-candidate (v1.0.rc1). DANDI uses OME-Zarr alongside
BIDS/NWB mostly for microscopy/connectomics/lightsheet data, not routine T1/T2/CT.

On storage efficiency, jvol wins decisively for the resolution-progressive use case: OME-Zarr
materializes each level as physically distinct, separately-compressed data, a classic
octave pyramid costs roughly 1 + 1/8 + 1/64 + ... ≈ 1.14x overhead for 3D (often worse in
practice, since each level pays its own chunk/compression-header cost), whereas jvol stores
one wavelet coefficient set and serves a factor-4 preview by partially decoding the same
stream, zero duplicated storage, plus the ~30x cheaper decode is a compute saving on top of,
not instead of, an avoided-read saving.

The one place OME-Zarr does something jvol genuinely lacks is spatial crop-first: Zarr's
per-chunk addressing means any bounding-box crop touches only overlapping chunks at any
single resolution level, true random-access spatial cropping on compressed data, which is
exactly the gap section 5 above discusses. Rust tooling (`zarrs` plus `zarrs_ome` for
reading/writing OME-Zarr multiscale hierarchies,
[zarrs book](https://book.zarrs.dev/zarrs_tools/docs/zarrs_ome.html)) and Python tooling
(`ome-zarr-py`, the reference implementation) are both mature.

**Verdict:** for medrs's stated case (fast full-or-coarse NIfTI loading for DL training),
jvol's single-file wavelet approach already matches or beats OME-Zarr on storage and decode
cost for progressive-resolution access. OME-Zarr's marginal value is interoperability with
the bioimaging/connectomics ecosystem (adjacent to, not native to, neuroimaging) and genuine
spatial crop-first, which is the same capability gap section 3's chunked-store investigation
targets. Track it; adopt only if spatial-patch streaming or explicit interop with that
ecosystem becomes a real requirement, not preemptively.

### 7. OS/IO-level techniques

Beyond the `madvise` recommendation in section 2, the remaining OS-level techniques
researched are not recommended:

- **io_uring** ([tokio-uring](https://github.com/tokio-rs/tokio-uring)) amortizes syscall
  overhead across many small/concurrent reads, a workload shape (thousands of tiny files,
  database-style access) medrs does not have; its per-file bottleneck is a single large
  sequential read plus CPU-bound decompression, not syscall count. `tokio-uring` is
  explicitly positioned for specialized Linux-only workloads (proxies, HTTP servers,
  databases) and requires Linux >= 5.10, a portability cost with no clear payoff here.
- **O_DIRECT** actively works against medrs's access pattern: training dataloaders re-read
  the same files across many epochs, and page-cache reuse is precisely what makes
  repeat-epoch access fast after the first pass. Evidence shows buffered I/O running up to
  2.3x faster than O_DIRECT for datasets that fit in cache
  ([P99 CONF](https://p99conf.io/2025/05/22/databases-linux-page-cache/)), only pulling
  ahead once per-process caching saturates past several GB, not medrs's typical regime.
- **mmap-the-compressed-file-then-decode vs read()-then-decode**: not worth it. For
  gzip, decompression throughput runs hundreds of MB/s to roughly 1-2 GB/s per core, while a
  single memcpy runs many GB/s per core on modern DDR4/5; skipping one memcpy by mmap-ing
  compressed bytes instead of reading them saves a cost an order of magnitude or more smaller
  than the decompression step itself
  ([Rapidgzip paper](https://arxiv.org/pdf/2308.08955), [lzbench](https://morotti.github.io/lzbench-web/)).
- **Transparent filesystem compression** (btrfs `zstd` mount option, ZFS
  `compression=zstd`, zram) genuinely gives "compressed on disk, decompressed transparently
  into the page cache on read," but it is entirely outside the library's control (a user
  opt-in at the filesystem/mount level), Linux/BSD-only, does not help over network mounts
  (NFS/S3), and ZFS in particular has a documented double-caching problem: its ARC is not
  integrated with the Linux page cache, so mmap-ing files on ZFS causes duplicated memory
  use and a real performance cliff once both caches are active
  ([OpenZFS ARC/page-cache issue](https://github.com/openzfs/zfs/issues/13178)). zram is a
  volatile RAM-only block device used for swap, not applicable to persistent `.nii.gz`
  storage at all. None of this is something medrs should build; at most it is worth a
  one-line doc note that users on a compressing filesystem get this for free, orthogonal to
  and not integrated with medrs's own decompression cache.

## Recommended roadmap

1. **Ship now, trivial effort:** add `madvise(MADV_SEQUENTIAL)` (and optionally
   `MADV_WILLNEED`) to `load_uncompressed`'s mmap path, cfg-gated to Linux like the existing
   `read_file_with_readahead`. Closes a real inconsistency with the compressed-read path's
   existing `posix_fadvise` hint, honest expected gain: modest, not transformative.
2. **Ship now, docs only:** add a section to medrs's docs recommending bf16 storage
   (uncompressed) for intensity volumes when zero decode-cost matters more than disk
   footprint, explicit that label volumes already get this for free via lossless int16, and
   explicit that this is complementary to, not competitive with, `.jvol`/gzip compression.
3. **Do not build:** GPU decompression (nvCOMP/GDS/KvikIO) near-term. Revisit only if
   profiling a real training pipeline shows CPU decode as the actual bottleneck, and the
   target users are confirmed to be on datacenter-class NVIDIA GPUs with a GDS-compatible
   storage stack.
4. **Do not build:** a tiled rewrite of `jvol`, or a JPEG2000/JP3D-based codec. Both are
   either abandoned upstream (JP3D) or too risky to jvol's lossless guarantee and progressive
   preview for a capability a companion format can provide more cheaply.
5. **Do not adopt (yet):** OME-Zarr as medrs's core storage convention. `jvol` already beats
   it on resolution-progressive storage efficiency and decode cost; OME-Zarr's genuine
   value-adds (spatial crop-first, bioimaging ecosystem interop) are only worth the storage
   layer change if a real user need for patch-based streaming or cross-tool interop emerges.
6. **Prototype, don't commit:** if and when crop-first-on-compressed becomes a measured
   requirement (e.g. patch-based training against volumes too large to fully decode
   per-epoch), build a small benchmark comparing Blosc2 CFrame partial-decode latency against
   `zarrs`-backed Zarr v3 sharded partial-decode latency, both against today's
   full-decode-then-crop baseline, before choosing which one to build as a companion storage
   format. Blosc2 keeps medrs's single-file-per-volume model but needs custom FFI atop
   `blosc2-src`; `zarrs` has more mature, spec-rigorous Rust support but means adopting a
   directory/shard-based multi-file storage layout.
7. **Skip:** io_uring, O_DIRECT, and transparent filesystem compression. Wrong workload
   shape, actively counterproductive, or entirely outside the library's control,
   respectively.
