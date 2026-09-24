"""Use medrs to load images in MONAI.

``MedrsReader`` is a MONAI ``ImageReader``, so MONAI's own ``LoadImage`` and
``LoadImaged`` handle metadata, channels, and ``MetaTensor`` creation::

    from monai.transforms import Compose, EnsureChannelFirstd, LoadImaged
    from medrs.monai import MedrsReader

    transforms = Compose([
        LoadImaged(keys=["image", "label"], reader=MedrsReader()),
        EnsureChannelFirstd(keys=["image", "label"]),
    ])

The returned arrays and metadata match MONAI's ``NibabelReader``: C-contiguous
arrays indexed ``[x, y, z, ...]`` and a RAS affine.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

import numpy as np
from monai.data import ImageReader
from monai.utils import ensure_tuple

import medrs

__all__ = ["MedrsReader"]

_SUFFIXES = (".nii", ".nii.gz", ".hdr", ".hdr.gz", ".img", ".img.gz", ".jvol")


class MedrsReader(ImageReader):  # type: ignore[misc]  # MONAI is untyped
    """MONAI image reader backed by :func:`medrs.load`.

    Args:
        channel_dim: Index of the channel dimension of the stored arrays, or
            ``None`` if they have no channel dimension (then 4D images are
            treated as channel-last, like ``NibabelReader``).
    """

    def __init__(self, channel_dim: int | None = None) -> None:
        super().__init__()
        self.channel_dim = channel_dim

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
        arrays = []
        meta: dict[str, Any] = {}
        for image in ensure_tuple(img):
            array = np.ascontiguousarray(image.to_numpy())
            spatial_rank = min(image.ndim, 3)
            header = image.header
            if self.channel_dim is not None:
                channel_dim: float | int = self.channel_dim
            else:
                channel_dim = float("nan") if array.ndim == spatial_rank else -1
            meta = {
                "affine": image.affine,
                "original_affine": image.affine.copy(),
                "spatial_shape": np.asarray(image.shape[:spatial_rank]),
                "space": "RAS",
                "pixdim": np.asarray(header["pixdim"]),
                "dim": np.asarray([image.ndim, *image.shape] + [1] * (7 - image.ndim)),
                "original_channel_dim": channel_dim,
            }
            arrays.append(array)
        if len(arrays) == 1:
            return arrays[0], meta
        channel_dim = meta["original_channel_dim"]
        if isinstance(channel_dim, int):
            return np.concatenate(arrays, axis=channel_dim), meta
        meta["original_channel_dim"] = 0
        return np.stack(arrays, axis=0), meta
