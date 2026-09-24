"""Stream augmented training patches with FastLoader.

Run: python examples/training_loader.py
Creates a small synthetic dataset of images and label maps, then iterates two
epochs. With PyTorch installed, the patches are also batched through a
DataLoader.
"""

import tempfile
from pathlib import Path

import numpy as np

import medrs


def make_dataset(root: Path, count: int = 6) -> tuple[list[Path], list[Path]]:
    rng = np.random.default_rng(0)
    images, labels = [], []
    for i in range(count):
        shape = (80, 80, 60 + 4 * i)
        image = rng.normal(0, 1, shape).astype(np.float32)
        label = np.zeros(shape, np.uint8)
        label[30:40, 30:45, 20:30] = 1  # a small "lesion"
        image[label == 1] += 3
        images.append(root / f"image_{i}.nii.gz")
        labels.append(root / f"label_{i}.nii.gz")
        medrs.NiftiImage(image).save(images[-1])
        medrs.NiftiImage(label).save(labels[-1])
    return images, labels


def main() -> None:
    images, labels = make_dataset(Path(tempfile.mkdtemp()))
    pipeline = (
        medrs.Pipeline()
        .z_normalize()
        .random_flip(axes=(0, 1, 2), prob=0.5)
        .random_intensity_scale(0.1)
        .random_gaussian_noise(0.05)
    )
    loader = medrs.FastLoader(
        images,
        patch_shape=(48, 48, 48),
        labels=labels,
        patches_per_volume=4,
        foreground_prob=0.5,
        pipeline=pipeline,
        seed=0,
    )
    print(loader, f"{len(loader)} patches per epoch")
    for epoch in range(2):
        foreground = sum(p.label.to_numpy().any() for p in loader)
        print(f"epoch {epoch}: {foreground}/{len(loader)} patches contain foreground")

    try:
        import torch
    except ImportError:
        return

    class Patches(torch.utils.data.IterableDataset):
        def __iter__(self):
            for patch in loader:
                yield patch.image.to_torch()[None], patch.label.to_torch()[None]

    for x, y in torch.utils.data.DataLoader(Patches(), batch_size=4):
        print("batch", tuple(x.shape), x.dtype, tuple(y.shape), y.dtype)
        break


if __name__ == "__main__":
    main()
