"""Use medrs to load images in MONAI.

``MedrsReader`` is a MONAI ``ImageReader``, so MONAI's own ``LoadImage`` and
``LoadImaged`` handle metadata, channels, and ``MetaTensor`` creation::

    from monai.transforms import Compose, EnsureChannelFirstd, LoadImaged
    from medrs.monai import MedrsReader

    transforms = Compose([
        LoadImaged(keys=["image", "label"], reader=MedrsReader()),
        EnsureChannelFirstd(keys=["image", "label"]),
    ])

Arrays are indexed ``[x, y, z, ...]`` like those of MONAI's ``NibabelReader``,
and the metadata has the keys MONAI's transforms use: ``affine``,
``original_affine``, ``spatial_shape``, ``space``, ``original_channel_dim``,
``pixdim``, and ``dim``.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
from monai.data import ImageReader
from monai.utils import SpaceKeys, ensure_tuple

import medrs

__all__ = ["MedrsReader"]

_SUFFIXES = (".nii", ".nii.gz", ".hdr", ".hdr.gz", ".img", ".img.gz", ".jvol")


class MedrsReader(ImageReader):  # type: ignore[misc]  # MONAI is untyped
    """MONAI image reader backed by :func:`medrs.load`.

    Args:
        channel_dim: Index of the channel dimension of the stored arrays, or
            ``"no_channel"``. By default 4D images are treated as channel-last
            and 3D images as having no channel, like ``NibabelReader``.
    """

    def __init__(self, channel_dim: int | Literal["no_channel"] | None = None) -> None:
        super().__init__()
        self.channel_dim: float | None = (
            float("nan") if isinstance(channel_dim, str) else channel_dim
        )

    def verify_suffix(
        self, filename: Sequence[os.PathLike[str] | str] | os.PathLike[str] | str
    ) -> bool:
        return all(str(f).lower().endswith(_SUFFIXES) for f in ensure_tuple(filename))

    def read(
        self, data: Sequence[os.PathLike[str] | str] | os.PathLike[str] | str, **kwargs: Any
    ) -> medrs.NiftiImage | list[medrs.NiftiImage]:
        images = [medrs.load(f, **kwargs) for f in ensure_tuple(data)]
        return images[0] if len(images) == 1 else images

    def get_data(
        self, img: medrs.NiftiImage | Sequence[medrs.NiftiImage]
    ) -> tuple[np.ndarray, dict[str, Any]]:
        images = ensure_tuple(img)
        meta = self._meta(images[0])
        for other in images[1:]:
            if not np.allclose(other.affine, images[0].affine):
                raise RuntimeError("all images must have the same affine to be stacked")
            if not np.array_equal(self._meta(other)["spatial_shape"], meta["spatial_shape"]):
                raise RuntimeError("all images must have the same spatial shape to be stacked")
        arrays = [_writable_c_array(image) for image in images]
        if len(arrays) == 1:
            return arrays[0], meta
        channel_dim = meta["original_channel_dim"]
        if not np.isnan(channel_dim):
            return np.concatenate(arrays, axis=int(channel_dim)), meta
        meta["original_channel_dim"] = 0
        return np.stack(arrays, axis=0), meta

    def _meta(self, image: medrs.NiftiImage) -> dict[str, Any]:
        spatial_rank = max(min(image.ndim, 3), 1)
        size = list(image.shape) + [1] * (7 - image.ndim)
        if self.channel_dim is None:
            channel_dim: float = float("nan") if image.ndim == spatial_rank else -1
        else:
            channel_dim = self.channel_dim
            if not np.isnan(channel_dim):
                size.pop(int(channel_dim))
        pixdim = np.asarray(image.header["pixdim"])
        return {
            "affine": image.affine,
            "original_affine": image.affine,
            "spatial_shape": np.asarray(size[:spatial_rank]),
            "space": SpaceKeys.RAS,
            "pixdim": pixdim,
            "original_pixdim": pixdim.copy(),
            "dim": np.asarray([image.ndim, *image.shape] + [1] * (7 - image.ndim)),
            "original_channel_dim": channel_dim,
        }


def _writable_c_array(image: medrs.NiftiImage) -> np.ndarray:
    """The image's voxels as a writable, C-contiguous array.

    MONAI wraps reader output in tensors that transforms may modify in place,
    and makes it C-contiguous; doing both here costs one copy instead of two.
    """
    array = image.to_numpy()
    if array.flags.writeable and array.flags.c_contiguous:
        return array
    return np.array(array, order="C")
