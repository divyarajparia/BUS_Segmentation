import os
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# ---------------------------------------------------
# Utility Functions
# ---------------------------------------------------

def compute_scale(mask_array: np.ndarray) -> float:
    """Return lesion scale as ratio of foreground pixels to total pixels."""
    binary = mask_array > 127  # threshold binary mask
    return binary.sum() / float(binary.size)

def compute_frequency(mask_array: np.ndarray, high_freq_ratio: float = 0.3) -> float:
    """Compute the high-frequency power ratio of a binary mask.

    Parameters
    ----------
    mask_array : np.ndarray
        Binary (0/255) mask array.
    high_freq_ratio : float, optional
        Radial threshold (0-1) for high frequencies expressed as a fraction of the
        maximum possible radius. Frequencies with radius > high_freq_ratio * r_max
        are considered high frequency. Default is 0.3.
    """
    # Convert to {0,1} floating mask
    binary = (mask_array > 127).astype(float)

    # 2-D FFT and power spectrum
    F = np.fft.fftshift(np.fft.fft2(binary))
    power = np.abs(F) ** 2

    # Build radial grid
    h, w = binary.shape
    cy, cx = h / 2.0, w / 2.0
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    r_max = r.max()

    high_mask = r > high_freq_ratio * r_max
    high_power = power[high_mask].sum()
    total_power = power.sum() if power.sum() > 0 else 1e-8

    return high_power / total_power


def detect_subdir(parent: str, keywords: tuple) -> str:
    """Return first child directory whose name contains any of the keywords."""
    for item in os.listdir(parent):
        if any(k in item.lower() for k in keywords):
            full = os.path.join(parent, item)
            if os.path.isdir(full):
                return item
    return None


def gather_dataset_metrics(dataset_name: str, dataset_root: str, limit: int = None):
    """Iterate over masks in a dataset and compute scale & frequency metrics.

    Parameters
    ----------
    dataset_name : str
        Label for dataset (e.g. "BUSI").
    dataset_root : str
        Root directory containing class subfolders (benign/malignant).
    limit : int, optional
        Max number of samples per class to process (for quick testing).
    """
    records = []

    if not os.path.isdir(dataset_root):
        print(f"[Warning] Dataset root not found: {dataset_root}")
        return pd.DataFrame(records)

    for cls in [d for d in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, d))]:
        cls_path = os.path.join(dataset_root, cls)
        mask_dir_name = detect_subdir(cls_path, ("mask", "masks"))
        if mask_dir_name is None:
            print(f"[Warning] Couldn't locate mask directory inside {cls_path}")
            continue

        mask_dir = os.path.join(cls_path, mask_dir_name)
        mask_files = [f for f in os.listdir(mask_dir) if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))]
        if limit:
            mask_files = mask_files[:limit]

        for fname in mask_files:
            fpath = os.path.join(mask_dir, fname)
            try:
                mask = Image.open(fpath).convert("L")
                mask_arr = np.array(mask)

                scale = compute_scale(mask_arr)
                freq = compute_frequency(mask_arr)

                records.append({
                    "dataset": dataset_name,
                    "class": cls,
                    "scale": scale,
                    "frequency": freq
                })
            except Exception as exc:
                print(f"[Error] Failed processing {fpath}: {exc}")

    return pd.DataFrame(records)

# ---------------------------------------------------
# Plotting
# ---------------------------------------------------

def plot_distribution(df: pd.DataFrame, output_path: str):
    """Create a scatter + marginal KDE plot similar to the provided example."""
    sns.set_style("white")

    # Joint plot with marginal distributions
    g = sns.JointGrid(data=df, x="scale", y="frequency", space=0, height=8)

    # Scatter layer
    g.plot_joint(sns.scatterplot, hue=df["dataset"], palette="Set2", s=20, alpha=0.7)

    # Marginals
    sns.kdeplot(data=df, x="scale", hue="dataset", fill=True, common_norm=False, alpha=0.2, ax=g.ax_marg_x, legend=False)
    sns.kdeplot(data=df, y="frequency", hue="dataset", fill=True, common_norm=False, alpha=0.2, ax=g.ax_marg_y, legend=False)

    # Aesthetic tweaks
    g.ax_joint.set_xlabel("Scale (Lesion Area / Image Area)", fontsize=12, fontweight="bold")
    g.ax_joint.set_ylabel("High-Frequency Power Ratio", fontsize=12, fontweight="bold")
    g.ax_joint.set_xlim(0, 1)
    g.ax_joint.set_ylim(0, 1)

    # Move legend outside
    handles, labels = g.ax_joint.get_legend_handles_labels()
    if handles:
        g.ax_joint.legend(handles=handles[1:], labels=labels[1:], title="Dataset", loc="upper right")

    plt.tight_layout()
    g.figure.savefig(output_path, dpi=300)
    print(f"\n✅ Saved plot to {output_path}\n")

# ---------------------------------------------------
# Main
# ---------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Scale vs Frequency analysis for BUSI and BUS-UCLM datasets")
    parser.add_argument("--busi", type=str, default="dataset/BioMedicalDataset/BUSI", help="Path to BUSI dataset root")
    parser.add_argument("--busuclm", type=str, default="dataset/BioMedicalDataset/BUS-UCLM", help="Path to BUS-UCLM dataset root")
    parser.add_argument("--limit", type=int, default=None, help="Max samples per class to process (optional)")
    parser.add_argument("--output", type=str, default="scale_frequency_distribution.png", help="Output image file")

    args = parser.parse_args()

    df_busi = gather_dataset_metrics("BUSI", args.busi, args.limit)
    df_bus_uclm = gather_dataset_metrics("BUS-UCLM", args.busuclm, args.limit)

    df_all = pd.concat([df_busi, df_bus_uclm], ignore_index=True)

    if df_all.empty:
        print("No data found – please check dataset paths.")
        return

    # Show basic stats
    for name, group in df_all.groupby("dataset"):
        print(f"{name}: {len(group)} samples | Mean scale = {group['scale'].mean():.4f} | Mean frequency = {group['frequency'].mean():.4f}")

    plot_distribution(df_all, args.output)


if __name__ == "__main__":
    main() 