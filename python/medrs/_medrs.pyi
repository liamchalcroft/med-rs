from collections.abc import Iterator, Sequence
from os import PathLike
from typing import Any, Literal, TypeAlias, TypedDict, final, overload

import numpy as np
import numpy.typing as npt

__all__ = [
    "Epoch",
    "FastLoader",
    "FormatError",
    "NiftiImage",
    "Patch",
    "Pipeline",
    "__version__",
    "_image_from_bytes",
    "_pipeline_from_steps",
    "clear_cache",
    "load",
    "load_cropped",
    "load_header",
    "load_multi",
    "num_threads",
    "set_cache_limits",
    "set_num_threads",
]

__version__: str

_StrPath: TypeAlias = str | PathLike[str]
_Interpolation: TypeAlias = Literal["nearest", "trilinear"]
_DType: TypeAlias = str | np.dtype[Any] | type | Any
_Shape3: TypeAlias = tuple[int, int, int] | Sequence[int]

class FormatError(ValueError):
    """A file is not a valid image of its format, or is corrupt."""

class _Header(TypedDict):
    version: Literal["nifti1", "nifti2"]
    shape: tuple[int, ...]
    dtype: str
    affine: npt.NDArray[np.float64]
    qform: npt.NDArray[np.float64] | None
    sform: npt.NDArray[np.float64] | None
    qform_code: int
    sform_code: int
    pixdim: tuple[float, float, float, float, float, float, float, float]
    spatial_units: Literal["unknown", "m", "mm", "um"]
    temporal_units: Literal["unknown", "s", "ms", "us", "hz", "ppm", "rads"]
    scl_slope: float
    scl_inter: float
    cal_min: float
    cal_max: float
    intent_code: int
    intent_name: str
    intent_p: tuple[float, float, float]
    dim_info: int
    slice_code: int
    slice_start: int
    slice_end: int
    slice_duration: float
    toffset: float
    descrip: str
    aux_file: str
    extensions: list[tuple[int, bytes]]

@final
class NiftiImage:
    """A medical image: voxel data plus its NIfTI header. Immutable."""

    def __new__(cls, data: npt.ArrayLike, affine: npt.ArrayLike | None = None) -> NiftiImage: ...
    @property
    def shape(self) -> tuple[int, ...]: ...
    @property
    def ndim(self) -> int: ...
    @property
    def dtype(self) -> str: ...
    @property
    def affine(self) -> npt.NDArray[np.float64]: ...
    @property
    def spacing(self) -> tuple[float, ...]: ...
    @property
    def orientation(self) -> str: ...
    @property
    def header(self) -> _Header: ...
    def to_numpy(self, dtype: _DType | None = None, *, copy: bool = False) -> npt.NDArray[Any]: ...
    def __array__(
        self, dtype: _DType | None = None, copy: bool | None = None
    ) -> npt.NDArray[Any]: ...
    def to_torch(self, dtype: _DType | None = None, device: Any = None) -> Any: ...
    def to_jax(self, dtype: _DType | None = None, device: Any = None) -> Any: ...
    def save(
        self,
        path: _StrPath,
        *,
        compression_level: int | None = None,
        mgzip: bool = False,
        quality: int | None = None,
        chunk_shape: _Shape3 | None = None,
    ) -> None: ...
    def resample(
        self, spacing: Sequence[float], interpolation: _Interpolation = "trilinear"
    ) -> NiftiImage: ...
    def resample_to_shape(
        self, shape: _Shape3, interpolation: _Interpolation = "trilinear"
    ) -> NiftiImage: ...
    def resample_like(
        self, reference: NiftiImage, interpolation: _Interpolation = "trilinear"
    ) -> NiftiImage: ...
    def reorient(self, orientation: str) -> NiftiImage: ...
    def crop(self, offset: _Shape3, shape: _Shape3) -> NiftiImage: ...
    def crop_or_pad(self, shape: _Shape3, pad_value: float = 0.0) -> NiftiImage: ...
    def flip(self, axes: Sequence[int]) -> NiftiImage: ...
    def rotate_90(self, axes: tuple[int, int] = (0, 1), k: int = 1) -> NiftiImage: ...
    def z_normalize(self, *, nonzero: bool = False) -> NiftiImage: ...
    def rescale(self, out_min: float = 0.0, out_max: float = 1.0) -> NiftiImage: ...
    def percentiles(self, q: Sequence[float], *, nonzero: bool = False) -> tuple[float, ...]: ...
    def rescale_percentiles(
        self,
        lower: float,
        upper: float,
        *,
        nonzero: bool = False,
        out_min: float = 0.0,
        out_max: float = 1.0,
    ) -> NiftiImage: ...
    def clamp(self, min: float, max: float) -> NiftiImage: ...
    def adjust_gamma(self, gamma: float) -> NiftiImage: ...
    def with_dtype(self, dtype: _DType) -> NiftiImage: ...
    def with_affine(self, affine: npt.ArrayLike) -> NiftiImage: ...
    def with_data(self, data: npt.ArrayLike) -> NiftiImage: ...
    def with_header(self, **fields: Any) -> NiftiImage: ...
    def __copy__(self) -> NiftiImage: ...
    def __deepcopy__(self, _memo: object, /) -> NiftiImage: ...

@final
class Pipeline:
    """An ordered list of transforms applied with fused passes. Immutable."""

    def __new__(cls) -> Pipeline: ...
    def reorient(self, orientation: str) -> Pipeline: ...
    def resample_to_spacing(
        self, spacing: Sequence[float], interpolation: _Interpolation = "trilinear"
    ) -> Pipeline: ...
    def resample_to_shape(
        self, shape: _Shape3, interpolation: _Interpolation = "trilinear"
    ) -> Pipeline: ...
    def crop(self, offset: _Shape3, shape: _Shape3) -> Pipeline: ...
    def crop_or_pad(self, shape: _Shape3, pad_value: float = 0.0) -> Pipeline: ...
    def flip(self, axes: Sequence[int]) -> Pipeline: ...
    def rotate_90(self, axes: tuple[int, int] = (0, 1), k: int = 1) -> Pipeline: ...
    def clamp(self, min: float, max: float) -> Pipeline: ...
    def z_normalize(self, nonzero: bool = False) -> Pipeline: ...
    def rescale(self, out_min: float = 0.0, out_max: float = 1.0) -> Pipeline: ...
    def rescale_percentiles(
        self,
        lower: float,
        upper: float,
        nonzero: bool = False,
        out_min: float = 0.0,
        out_max: float = 1.0,
    ) -> Pipeline: ...
    def adjust_gamma(self, gamma: float) -> Pipeline: ...
    def cast(self, dtype: _DType) -> Pipeline: ...
    def random_flip(self, axes: Sequence[int] = (0, 1, 2), prob: float = 0.5) -> Pipeline: ...
    def random_rotate_90(self, axes: tuple[int, int] = (0, 1)) -> Pipeline: ...
    def random_intensity_scale(self, range: float = 0.1) -> Pipeline: ...
    def random_intensity_shift(self, range: float = 0.1) -> Pipeline: ...
    def random_gaussian_noise(self, std: float = 0.1) -> Pipeline: ...
    def random_gamma(self, range: tuple[float, float] = (0.7, 1.5)) -> Pipeline: ...
    @overload
    def apply(
        self, image: NiftiImage, label: None = None, *, seed: int | None = None
    ) -> NiftiImage: ...
    @overload
    def apply(
        self, image: NiftiImage, label: NiftiImage, *, seed: int | None = None
    ) -> tuple[NiftiImage, NiftiImage]: ...
    def __len__(self) -> int: ...

@final
class Patch:
    """One training patch from a FastLoader."""

    @property
    def image(self) -> NiftiImage: ...
    @property
    def label(self) -> NiftiImage | None: ...
    @property
    def volume(self) -> int: ...
    @property
    def offset(self) -> tuple[int, int, int]: ...
    @property
    def shape(self) -> tuple[int, int, int]: ...

@final
class Epoch(Iterator[Patch]):
    """Iterator over the patches of one epoch."""

    def __iter__(self) -> Epoch: ...
    def __next__(self) -> Patch: ...
    def __length_hint__(self) -> int: ...
    def close(self) -> None: ...
    def __enter__(self) -> Epoch: ...
    def __exit__(self, *args: object) -> None: ...

@final
class FastLoader:
    """Streams training patches from volumes using worker threads."""

    def __new__(
        cls,
        images: Sequence[_StrPath],
        patch_shape: _Shape3 | Sequence[_Shape3],
        *,
        labels: Sequence[_StrPath] | None = None,
        patches_per_volume: int = 1,
        patch_shape_weights: Sequence[float] | None = None,
        foreground_prob: float | None = None,
        foreground_threshold: float | None = None,
        weights: Sequence[float] | None = None,
        volumes_per_epoch: int | None = None,
        pad_value: float = 0.0,
        pipeline: Pipeline | None = None,
        workers: int | None = None,
        prefetch: int | None = None,
        shuffle: bool = True,
        seed: int | None = None,
    ) -> FastLoader: ...
    def __len__(self) -> int: ...
    def __iter__(self) -> Epoch: ...
    def epoch(self, epoch: int) -> Epoch: ...
    @property
    def seed(self) -> int: ...

def load(path: _StrPath, *, cache: bool = False) -> NiftiImage:
    """Load an image (`.nii`, `.nii.gz`, `.hdr`/`.img`, or `.jvol`).

    With `cache=True`, decompressed data is kept in the cache (see
    `set_cache_limits`).
    """

def load_cropped(path: _StrPath, offset: _Shape3, shape: _Shape3) -> NiftiImage:
    """Load `shape` voxels starting at `offset` along the first three axes.

    Uncompressed files read only the region, and `.jvol` files decode only
    the chunks it overlaps. Gzipped files are decompressed once and cached.
    """

def load_header(path: _StrPath) -> _Header:
    """Read only the header of an image file.

    No voxel data is read: `.nii` files read the header and extensions,
    gzipped files decompress only those bytes, and `.jvol` files read only the
    header and chunk index, so scanning many files is cheap.
    """

def load_multi(
    paths: Sequence[_StrPath],
    *,
    reference: int = 0,
    interpolation: _Interpolation | Sequence[_Interpolation] | None = None,
) -> list[NiftiImage]: ...
def clear_cache() -> None: ...
def set_cache_limits(*, max_entries: int | None = None, max_bytes: int | None = None) -> None: ...
def set_num_threads(threads: int) -> None: ...
def num_threads() -> int: ...

# Pickle support.
def _image_from_bytes(data: bytes) -> NiftiImage: ...
def _pipeline_from_steps(steps: Sequence[tuple[str, tuple[Any, ...]]]) -> Pipeline: ...
