"""
Visual comparison plots for medrs vs nibabel vs MONAI.
Generates side-by-side slice comparisons for verification.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import nibabel as nib
import medrs
from monai.transforms import (
    LoadImage,
    ScaleIntensity,
    ScaleIntensityRange,
    Flip,
    CenterSpatialCrop,
    EnsureChannelFirst,
)

# Output directory for plots
OUTPUT_DIR = Path("tests/output/comparison_plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TEST_IMAGE = "tests/fixtures/mprage_img.nii"


def get_center_slices(data):
    """Get center slices for each axis."""
    shape = data.shape
    return {
        "axial": data[:, :, shape[2] // 2],
        "coronal": data[:, shape[1] // 2, :],
        "sagittal": data[shape[0] // 2, :, :],
    }


def plot_comparison(data_dict, title, filename, diff=True):
    """
    Plot comparison of multiple arrays.

    Args:
        data_dict: Dict of {name: array} to compare
        title: Plot title
        filename: Output filename
        diff: Whether to show difference maps
    """
    names = list(data_dict.keys())
    n_sources = len(names)

    # Get center slices for each source
    slices_dict = {name: get_center_slices(data) for name, data in data_dict.items()}

    views = ["axial", "coronal", "sagittal"]

    if diff and n_sources >= 2:
        # Show sources + difference maps
        n_cols = n_sources + 1  # +1 for diff
        fig, axes = plt.subplots(3, n_cols, figsize=(4 * n_cols, 12))
    else:
        fig, axes = plt.subplots(3, n_sources, figsize=(4 * n_sources, 12))

    fig.suptitle(title, fontsize=14, fontweight="bold")

    for row, view in enumerate(views):
        for col, name in enumerate(names):
            ax = axes[row, col] if n_sources > 1 or diff else axes[row]
            slice_data = slices_dict[name][view]
            im = ax.imshow(slice_data.T, cmap="gray", origin="lower")
            ax.set_title(f"{name}\n{view}" if row == 0 else view)
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Add difference map if requested
        if diff and n_sources >= 2:
            ax = axes[row, n_sources]
            # Difference between first two sources
            diff_data = slices_dict[names[0]][view] - slices_dict[names[1]][view]
            max_diff = np.abs(diff_data).max()
            im = ax.imshow(
                diff_data.T,
                cmap="RdBu_r",
                origin="lower",
                vmin=-max_diff if max_diff > 0 else -1,
                vmax=max_diff if max_diff > 0 else 1,
            )
            ax.set_title(f"Diff ({names[0]}-{names[1]})\n{view}" if row == 0 else view)
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUTPUT_DIR / filename}")


def plot_basic_loading():
    """Compare basic loading between nibabel and medrs."""
    print("\n=== Basic Loading Comparison ===")

    # Load with nibabel
    nib_img = nib.load(TEST_IMAGE)
    nib_data = np.asarray(nib_img.dataobj, dtype=np.float32)

    # Load with medrs
    medrs_img = medrs.load(TEST_IMAGE)
    medrs_data = medrs_img.to_numpy()

    print(f"nibabel shape: {nib_data.shape}, dtype: {nib_data.dtype}")
    print(f"medrs shape: {medrs_data.shape}, dtype: {medrs_data.dtype}")
    print(f"Max diff: {np.abs(nib_data - medrs_data).max():.6f}")

    plot_comparison(
        {"nibabel": nib_data, "medrs": medrs_data},
        "Basic Loading: nibabel vs medrs",
        "01_basic_loading.png",
    )


def plot_intensity_rescale():
    """Compare intensity rescaling."""
    print("\n=== Intensity Rescale Comparison ===")

    # Load with nibabel
    nib_img = nib.load(TEST_IMAGE)
    nib_data = np.asarray(nib_img.dataobj, dtype=np.float32)

    # Manual rescale to [0, 1]
    nib_rescaled = (nib_data - nib_data.min()) / (nib_data.max() - nib_data.min())

    # medrs rescale
    medrs_img = medrs.load(TEST_IMAGE)
    medrs_rescaled = medrs_img.rescale(0.0, 1.0).to_numpy()

    # MONAI rescale
    loader = LoadImage(image_only=True)
    monai_data = loader(TEST_IMAGE)
    rescaler = ScaleIntensityRange(
        a_min=float(monai_data.min()), a_max=float(monai_data.max()), b_min=0.0, b_max=1.0
    )
    monai_rescaled = np.asarray(rescaler(monai_data))

    print(f"nibabel range: [{nib_rescaled.min():.4f}, {nib_rescaled.max():.4f}]")
    print(f"medrs range: [{medrs_rescaled.min():.4f}, {medrs_rescaled.max():.4f}]")
    print(f"monai range: [{monai_rescaled.min():.4f}, {monai_rescaled.max():.4f}]")

    plot_comparison(
        {"nibabel": nib_rescaled, "medrs": medrs_rescaled, "monai": monai_rescaled},
        "Intensity Rescale [0, 1]: nibabel vs medrs vs MONAI",
        "02_intensity_rescale.png",
        diff=False,
    )

    # Also plot medrs vs monai diff
    plot_comparison(
        {"medrs": medrs_rescaled, "monai": monai_rescaled},
        "Intensity Rescale [0, 1]: medrs vs MONAI (with diff)",
        "02b_intensity_rescale_diff.png",
    )


def plot_z_normalize():
    """Compare z-normalization."""
    print("\n=== Z-Normalization Comparison ===")

    # Load with nibabel
    nib_img = nib.load(TEST_IMAGE)
    nib_data = np.asarray(nib_img.dataobj, dtype=np.float32)

    # Manual z-normalize
    nib_znorm = (nib_data - nib_data.mean()) / nib_data.std()

    # medrs z-normalize
    medrs_img = medrs.load(TEST_IMAGE)
    medrs_znorm = medrs_img.z_normalize().to_numpy()

    print(f"nibabel mean/std: {nib_znorm.mean():.6f} / {nib_znorm.std():.6f}")
    print(f"medrs mean/std: {medrs_znorm.mean():.6f} / {medrs_znorm.std():.6f}")
    print(f"Max diff: {np.abs(nib_znorm - medrs_znorm).max():.6f}")

    plot_comparison(
        {"nibabel": nib_znorm, "medrs": medrs_znorm},
        "Z-Normalization: nibabel vs medrs",
        "03_z_normalize.png",
    )


def plot_clamp():
    """Compare clamping."""
    print("\n=== Clamp Comparison ===")

    # Load with nibabel
    nib_img = nib.load(TEST_IMAGE)
    nib_data = np.asarray(nib_img.dataobj, dtype=np.float32)

    # Clamp to [0, 100]
    nib_clamped = np.clip(nib_data, 0.0, 100.0)

    # medrs clamp
    medrs_img = medrs.load(TEST_IMAGE)
    medrs_clamped = medrs_img.clamp(0.0, 100.0).to_numpy()

    print(f"nibabel range: [{nib_clamped.min():.4f}, {nib_clamped.max():.4f}]")
    print(f"medrs range: [{medrs_clamped.min():.4f}, {medrs_clamped.max():.4f}]")

    plot_comparison(
        {"nibabel": nib_clamped, "medrs": medrs_clamped},
        "Clamp [0, 100]: nibabel vs medrs",
        "04_clamp.png",
    )


def plot_flip():
    """Compare flip operations."""
    print("\n=== Flip Comparison ===")

    medrs_img = medrs.load(TEST_IMAGE)

    # MONAI flip
    loader = LoadImage(image_only=True)
    monai_data = loader(TEST_IMAGE)
    monai_data = EnsureChannelFirst()(monai_data)

    for axis in [0, 1, 2]:
        # medrs flip
        medrs_flipped = medrs_img.flip([axis]).to_numpy()

        # MONAI flip
        flipper = Flip(spatial_axis=axis)
        monai_flipped = np.asarray(flipper(monai_data))[0]  # Remove channel dim

        print(f"Axis {axis} - Max diff: {np.abs(medrs_flipped - monai_flipped).max():.6f}")

        plot_comparison(
            {"medrs": medrs_flipped, "monai": monai_flipped},
            f"Flip Axis {axis}: medrs vs MONAI",
            f"05_flip_axis{axis}.png",
        )


def plot_resample():
    """Compare resampling."""
    print("\n=== Resample Comparison ===")

    medrs_img = medrs.load(TEST_IMAGE)
    original = medrs_img.to_numpy()

    # Resample to different spacing
    new_spacing = [2.0, 2.0, 2.0]
    medrs_resampled = medrs_img.resample(new_spacing).to_numpy()

    print(f"Original shape: {original.shape}")
    print(f"Resampled shape: {medrs_resampled.shape}")
    print(f"Original spacing: {medrs_img.spacing}")

    # For visualization, we'll just show the resampled result
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    fig.suptitle("Resample: Original vs 2mm spacing", fontsize=14, fontweight="bold")

    views = ["axial", "coronal", "sagittal"]
    orig_slices = get_center_slices(original)
    resamp_slices = get_center_slices(medrs_resampled)

    for col, view in enumerate(views):
        axes[0, col].imshow(orig_slices[view].T, cmap="gray", origin="lower")
        axes[0, col].set_title(f"Original {view}\n{original.shape}")
        axes[0, col].axis("off")

        axes[1, col].imshow(resamp_slices[view].T, cmap="gray", origin="lower")
        axes[1, col].set_title(f"Resampled {view}\n{medrs_resampled.shape}")
        axes[1, col].axis("off")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "06_resample.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUTPUT_DIR / '06_resample.png'}")


def plot_crop():
    """Compare center crop."""
    print("\n=== Center Crop Comparison ===")

    medrs_img = medrs.load(TEST_IMAGE)

    # MONAI crop
    loader = LoadImage(image_only=True)
    monai_data = loader(TEST_IMAGE)
    monai_data = EnsureChannelFirst()(monai_data)

    crop_size = [100, 100, 100]

    # medrs crop
    medrs_cropped = medrs_img.crop_or_pad(crop_size).to_numpy()

    # MONAI crop
    cropper = CenterSpatialCrop(roi_size=crop_size)
    monai_cropped = np.asarray(cropper(monai_data))[0]

    print(f"medrs cropped shape: {medrs_cropped.shape}")
    print(f"monai cropped shape: {monai_cropped.shape}")
    print(f"Max diff: {np.abs(medrs_cropped - monai_cropped).max():.6f}")

    plot_comparison(
        {"medrs": medrs_cropped, "monai": monai_cropped},
        f"Center Crop {crop_size}: medrs vs MONAI",
        "07_center_crop.png",
    )


def plot_crop_or_pad():
    """Compare crop_or_pad (padding case)."""
    print("\n=== Crop or Pad Comparison ===")

    medrs_img = medrs.load(TEST_IMAGE)
    original = medrs_img.to_numpy()

    # Pad to larger size
    pad_size = [256, 256, 256]
    medrs_padded = medrs_img.crop_or_pad(pad_size).to_numpy()

    print(f"Original shape: {original.shape}")
    print(f"Padded shape: {medrs_padded.shape}")

    # Crop to smaller size
    crop_size = [100, 120, 80]
    medrs_cropped = medrs_img.crop_or_pad(crop_size).to_numpy()

    print(f"Cropped shape: {medrs_cropped.shape}")

    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    fig.suptitle("Crop or Pad: Original vs Padded vs Cropped", fontsize=14, fontweight="bold")

    views = ["axial", "coronal", "sagittal"]
    orig_slices = get_center_slices(original)
    pad_slices = get_center_slices(medrs_padded)
    crop_slices = get_center_slices(medrs_cropped)

    for col, view in enumerate(views):
        axes[0, col].imshow(orig_slices[view].T, cmap="gray", origin="lower")
        axes[0, col].set_title(f"Original {view}\n{original.shape}")
        axes[0, col].axis("off")

        axes[1, col].imshow(pad_slices[view].T, cmap="gray", origin="lower")
        axes[1, col].set_title(f"Padded {view}\n{medrs_padded.shape}")
        axes[1, col].axis("off")

        axes[2, col].imshow(crop_slices[view].T, cmap="gray", origin="lower")
        axes[2, col].set_title(f"Cropped {view}\n{medrs_cropped.shape}")
        axes[2, col].axis("off")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "08_crop_or_pad.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUTPUT_DIR / '08_crop_or_pad.png'}")


def plot_pipeline():
    """Show a typical preprocessing pipeline."""
    print("\n=== Preprocessing Pipeline ===")

    medrs_img = medrs.load(TEST_IMAGE)

    # Typical preprocessing pipeline
    steps = {
        "1. Original": medrs_img.to_numpy(),
        "2. Resampled (1.5mm)": medrs_img.resample([1.5, 1.5, 1.5]).to_numpy(),
    }

    # Continue pipeline
    resampled = medrs_img.resample([1.5, 1.5, 1.5])
    steps["3. Z-Normalized"] = resampled.z_normalize().to_numpy()

    normalized = resampled.z_normalize()
    steps["4. Clamped [-3, 3]"] = normalized.clamp(-3.0, 3.0).to_numpy()

    clamped = normalized.clamp(-3.0, 3.0)
    steps["5. Cropped [128,128,128]"] = clamped.crop_or_pad([128, 128, 128]).to_numpy()

    n_steps = len(steps)
    fig, axes = plt.subplots(3, n_steps, figsize=(4 * n_steps, 12))
    fig.suptitle("Typical Preprocessing Pipeline", fontsize=14, fontweight="bold")

    views = ["axial", "coronal", "sagittal"]

    for col, (name, data) in enumerate(steps.items()):
        slices = get_center_slices(data)
        for row, view in enumerate(views):
            ax = axes[row, col]
            im = ax.imshow(slices[view].T, cmap="gray", origin="lower")
            if row == 0:
                ax.set_title(f"{name}\n{data.shape}\n{view}")
            else:
                ax.set_title(view)
            ax.axis("off")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "09_pipeline.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUTPUT_DIR / '09_pipeline.png'}")


def plot_load_cropped():
    """Compare load_cropped with manual crop."""
    print("\n=== Load Cropped Comparison ===")

    # Full load then crop
    medrs_full = medrs.load(TEST_IMAGE)

    # Define crop region
    offset = [50, 60, 40]
    size = [100, 100, 100]

    # Manual crop from full image
    full_data = medrs_full.to_numpy()
    manual_crop = full_data[
        offset[0] : offset[0] + size[0],
        offset[1] : offset[1] + size[1],
        offset[2] : offset[2] + size[2],
    ]

    # Load cropped directly
    medrs_cropped = medrs.load_cropped(TEST_IMAGE, offset, size)
    cropped_data = medrs_cropped.to_numpy()

    print(f"Manual crop shape: {manual_crop.shape}")
    print(f"load_cropped shape: {cropped_data.shape}")
    print(f"Max diff: {np.abs(manual_crop - cropped_data).max():.6f}")

    plot_comparison(
        {"manual_crop": manual_crop, "load_cropped": cropped_data},
        f"Load Cropped: Manual vs load_cropped (offset={offset}, size={size})",
        "10_load_cropped.png",
    )


def plot_save_load_roundtrip():
    """Test save/load roundtrip."""
    print("\n=== Save/Load Roundtrip ===")
    import tempfile
    import os

    # Load original
    medrs_img = medrs.load(TEST_IMAGE)
    original = medrs_img.to_numpy()

    # Apply some transforms
    transformed = medrs_img.z_normalize().clamp(-3.0, 3.0)
    transformed_data = transformed.to_numpy()

    # Save and reload with medrs
    with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as f:
        temp_path = f.name

    try:
        transformed.save(temp_path)
        reloaded = medrs.load(temp_path)
        reloaded_data = reloaded.to_numpy()

        # Also load with nibabel for comparison
        nib_reloaded = nib.load(temp_path)
        nib_data = np.asarray(nib_reloaded.dataobj, dtype=np.float32)

        print(f"Original shape: {original.shape}")
        print(f"Transformed shape: {transformed_data.shape}")
        print(f"Reloaded shape: {reloaded_data.shape}")
        print(f"medrs roundtrip diff: {np.abs(transformed_data - reloaded_data).max():.6f}")
        print(f"nibabel read diff: {np.abs(transformed_data - nib_data).max():.6f}")

        plot_comparison(
            {"original": transformed_data, "medrs_reload": reloaded_data, "nibabel_read": nib_data},
            "Save/Load Roundtrip: Original vs Reloaded",
            "11_roundtrip.png",
            diff=False,
        )
    finally:
        os.unlink(temp_path)


def main():
    """Generate all comparison plots."""
    print("=" * 60)
    print("Generating visual comparison plots")
    print("=" * 60)

    plot_basic_loading()
    plot_intensity_rescale()
    plot_z_normalize()
    plot_clamp()
    plot_flip()
    plot_resample()
    plot_crop()
    plot_crop_or_pad()
    plot_pipeline()
    plot_load_cropped()
    plot_save_load_roundtrip()

    print("\n" + "=" * 60)
    print(f"All plots saved to: {OUTPUT_DIR.absolute()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
