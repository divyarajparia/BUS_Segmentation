import os
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from skimage import feature, filters, measure
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------
# Enhanced Metrics Computation
# ---------------------------------------------------

def compute_scale(mask_array: np.ndarray) -> float:
    """Return lesion scale as ratio of foreground pixels to total pixels."""
    binary = mask_array > 127  # threshold binary mask
    return binary.sum() / float(binary.size)

def compute_frequency(mask_array: np.ndarray, high_freq_ratio: float = 0.3) -> float:
    """Compute the high-frequency power ratio of a binary mask."""
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

def compute_texture_variance(image_array: np.ndarray) -> float:
    """Compute texture variance using Local Binary Pattern."""
    try:
        # Convert to grayscale if needed
        if len(image_array.shape) == 3:
            image_array = np.mean(image_array, axis=2)
        
        # Compute LBP
        lbp = feature.local_binary_pattern(image_array, P=8, R=1, method='uniform')
        
        # Compute variance of LBP histogram
        hist, _ = np.histogram(lbp.ravel(), bins=10, density=True)
        return np.var(hist)
    except:
        return 0.0

def compute_intensity_distribution(image_array: np.ndarray) -> dict:
    """Compute intensity distribution statistics."""
    try:
        # Convert to grayscale if needed
        if len(image_array.shape) == 3:
            image_array = np.mean(image_array, axis=2)
        
        # Flatten and normalize to [0,1]
        flat = image_array.flatten() / 255.0
        
        return {
            'mean_intensity': np.mean(flat),
            'std_intensity': np.std(flat),
            'skewness': stats.skew(flat),
            'kurtosis': stats.kurtosis(flat)
        }
    except:
        return {'mean_intensity': 0, 'std_intensity': 0, 'skewness': 0, 'kurtosis': 0}

def compute_anatomical_variance(mask_array: np.ndarray) -> dict:
    """Compute anatomical variance metrics."""
    try:
        binary = mask_array > 127
        
        if not np.any(binary):
            return {'solidity': 0, 'eccentricity': 0, 'extent': 0}
        
        # Find connected components
        labeled = measure.label(binary)
        props = measure.regionprops(labeled)
        
        if not props:
            return {'solidity': 0, 'eccentricity': 0, 'extent': 0}
        
        # Use largest component
        largest = max(props, key=lambda x: x.area)
        
        return {
            'solidity': largest.solidity,
            'eccentricity': largest.eccentricity,
            'extent': largest.extent
        }
    except:
        return {'solidity': 0, 'eccentricity': 0, 'extent': 0}

# ---------------------------------------------------
# Data Collection
# ---------------------------------------------------

def detect_subdir(parent: str, keywords: tuple) -> str:
    """Return first child directory whose name contains any of the keywords."""
    if not os.path.exists(parent):
        return None
    for item in os.listdir(parent):
        if any(k in item.lower() for k in keywords):
            full = os.path.join(parent, item)
            if os.path.isdir(full):
                return item
    return None

def gather_comprehensive_metrics(dataset_name: str, dataset_root: str, limit: int = None):
    """Gather comprehensive metrics from both images and masks."""
    records = []

    if not os.path.isdir(dataset_root):
        print(f"[Warning] Dataset root not found: {dataset_root}")
        return pd.DataFrame(records)

    for cls in [d for d in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, d))]:
        cls_path = os.path.join(dataset_root, cls)
        
        # Find mask and image directories
        mask_dir_name = detect_subdir(cls_path, ("mask", "masks"))
        image_dir_name = detect_subdir(cls_path, ("image", "images"))
        
        if mask_dir_name is None or image_dir_name is None:
            print(f"[Warning] Couldn't locate image/mask directories in {cls_path}")
            continue

        mask_dir = os.path.join(cls_path, mask_dir_name)
        image_dir = os.path.join(cls_path, image_dir_name)
        
        mask_files = [f for f in os.listdir(mask_dir) if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))]
        if limit:
            mask_files = mask_files[:limit]

        for mask_fname in mask_files:
            mask_path = os.path.join(mask_dir, mask_fname)
            
            # Find corresponding image
            image_fname = mask_fname.replace('_mask', '').replace('_segmentation', '')
            # Handle different naming conventions
            for possible_name in [image_fname, mask_fname.replace('_mask', ''), 
                                 mask_fname.split('_')[0] + '.png', mask_fname]:
                image_path = os.path.join(image_dir, possible_name)
                if os.path.exists(image_path):
                    break
            else:
                # If no exact match, try first available image
                image_files = [f for f in os.listdir(image_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
                if image_files:
                    image_path = os.path.join(image_dir, image_files[0])
                else:
                    continue

            try:
                # Load mask and image
                mask = Image.open(mask_path).convert("L")
                mask_arr = np.array(mask)
                
                image = Image.open(image_path).convert("RGB")
                image_arr = np.array(image)

                # Compute all metrics
                scale = compute_scale(mask_arr)
                frequency = compute_frequency(mask_arr)
                texture_var = compute_texture_variance(image_arr)
                intensity_stats = compute_intensity_distribution(image_arr)
                anatomical_stats = compute_anatomical_variance(mask_arr)

                record = {
                    "dataset": dataset_name,
                    "class": cls,
                    "scale": scale,
                    "frequency": frequency,
                    "texture_variance": texture_var,
                    **intensity_stats,
                    **anatomical_stats
                }
                
                records.append(record)

            except Exception as exc:
                print(f"[Error] Failed processing {mask_path}: {exc}")

    return pd.DataFrame(records)

# ---------------------------------------------------
# Enhanced Visualization
# ---------------------------------------------------

def plot_comprehensive_comparison(df: pd.DataFrame, output_path: str):
    """Create a comprehensive multi-panel comparison visualization."""
    
    # Set style similar to inspiration image
    plt.style.use('default')
    sns.set_palette("Set2")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    
    # Define colors for datasets
    colors = {'BUSI': '#1f77b4', 'BUS-UCLM': '#ff7f0e'}
    
    # Main scatter plot: Scale vs Frequency (like inspiration)
    ax_main = plt.subplot2grid((3, 4), (0, 0), colspan=2, rowspan=2)
    
    for dataset in df['dataset'].unique():
        data = df[df['dataset'] == dataset]
        ax_main.scatter(data['scale'], data['frequency'], 
                       label=dataset, alpha=0.7, s=50, c=colors[dataset])
    
    ax_main.set_xlabel('Scale (Lesion Area / Image Area)', fontweight='bold', fontsize=12)
    ax_main.set_ylabel('Frequency (High-Freq Power Ratio)', fontweight='bold', fontsize=12)
    ax_main.set_title('Scale vs Frequency Distribution', fontweight='bold', fontsize=14)
    ax_main.legend(title='Dataset', title_fontsize=12, fontsize=11)
    ax_main.grid(True, alpha=0.3)
    
    # Marginal distributions
    ax_top = plt.subplot2grid((3, 4), (0, 2), colspan=2)
    ax_right = plt.subplot2grid((3, 4), (1, 2), rowspan=1)
    
    # Scale distribution
    for dataset in df['dataset'].unique():
        data = df[df['dataset'] == dataset]
        ax_top.hist(data['scale'], alpha=0.6, label=dataset, bins=20, density=True, color=colors[dataset])
    ax_top.set_xlabel('Scale', fontweight='bold')
    ax_top.set_ylabel('Density', fontweight='bold')
    ax_top.set_title('Scale Distribution', fontweight='bold')
    ax_top.legend()
    
    # Frequency distribution
    for dataset in df['dataset'].unique():
        data = df[df['dataset'] == dataset]
        ax_right.hist(data['frequency'], alpha=0.6, label=dataset, bins=20, density=True, 
                     orientation='horizontal', color=colors[dataset])
    ax_right.set_ylabel('Frequency', fontweight='bold')
    ax_right.set_xlabel('Density', fontweight='bold')
    ax_right.set_title('Frequency Distribution', fontweight='bold', rotation=270, pad=20)
    
    # Texture variance comparison
    ax_texture = plt.subplot2grid((3, 4), (2, 0))
    texture_data = [df[df['dataset'] == dataset]['texture_variance'].values for dataset in df['dataset'].unique()]
    box = ax_texture.boxplot(texture_data, labels=df['dataset'].unique(), patch_artist=True)
    for patch, color in zip(box['boxes'], colors.values()):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax_texture.set_ylabel('Texture Variance', fontweight='bold')
    ax_texture.set_title('Texture Analysis', fontweight='bold')
    ax_texture.grid(True, alpha=0.3)
    
    # Intensity distribution comparison
    ax_intensity = plt.subplot2grid((3, 4), (2, 1))
    intensity_data = [df[df['dataset'] == dataset]['mean_intensity'].values for dataset in df['dataset'].unique()]
    box = ax_intensity.boxplot(intensity_data, labels=df['dataset'].unique(), patch_artist=True)
    for patch, color in zip(box['boxes'], colors.values()):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax_intensity.set_ylabel('Mean Intensity', fontweight='bold')
    ax_intensity.set_title('Intensity Distribution', fontweight='bold')
    ax_intensity.grid(True, alpha=0.3)
    
    # Anatomical variance (solidity)
    ax_anatomy = plt.subplot2grid((3, 4), (2, 2))
    anatomy_data = [df[df['dataset'] == dataset]['solidity'].values for dataset in df['dataset'].unique()]
    box = ax_anatomy.boxplot(anatomy_data, labels=df['dataset'].unique(), patch_artist=True)
    for patch, color in zip(box['boxes'], colors.values()):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax_anatomy.set_ylabel('Solidity', fontweight='bold')
    ax_anatomy.set_title('Shape Analysis', fontweight='bold')
    ax_anatomy.grid(True, alpha=0.3)
    
    # Summary statistics table
    ax_stats = plt.subplot2grid((3, 4), (2, 3))
    ax_stats.axis('off')
    
    # Create summary table
    summary_data = []
    for dataset in df['dataset'].unique():
        data = df[df['dataset'] == dataset]
        summary_data.append([
            dataset,
            f"{len(data)}",
            f"{data['scale'].mean():.3f}±{data['scale'].std():.3f}",
            f"{data['frequency'].mean():.3f}±{data['frequency'].std():.3f}",
            f"{data['texture_variance'].mean():.3f}±{data['texture_variance'].std():.3f}"
        ])
    
    table = ax_stats.table(cellText=summary_data,
                          colLabels=['Dataset', 'Samples', 'Scale', 'Frequency', 'Texture'],
                          cellLoc='center',
                          loc='center',
                          bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    ax_stats.set_title('Summary Statistics', fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Enhanced visualization saved to {output_path}")

# ---------------------------------------------------
# Main Function
# ---------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Comprehensive BUSI vs BUS-UCLM dataset comparison")
    parser.add_argument("--busi", type=str, default="debug_data/BUSI", help="Path to BUSI dataset root")
    parser.add_argument("--busuclm", type=str, default="dataset/BioMedicalDataset/BUS-UCLM", help="Path to BUS-UCLM dataset root")
    parser.add_argument("--limit", type=int, default=50, help="Max samples per class to process")
    parser.add_argument("--output", type=str, default="comprehensive_dataset_comparison.png", help="Output image file")

    args = parser.parse_args()

    print("🔍 Gathering comprehensive metrics for BUSI dataset...")
    df_busi = gather_comprehensive_metrics("BUSI", args.busi, args.limit)
    
    print("🔍 Gathering comprehensive metrics for BUS-UCLM dataset...")
    df_bus_uclm = gather_comprehensive_metrics("BUS-UCLM", args.busuclm, args.limit)

    df_all = pd.concat([df_busi, df_bus_uclm], ignore_index=True)

    if df_all.empty:
        print("❌ No data found – please check dataset paths.")
        return

    # Print detailed statistics
    print(f"\n📊 Dataset Comparison Summary:")
    print("="*60)
    
    for dataset in df_all['dataset'].unique():
        data = df_all[df_all['dataset'] == dataset]
        print(f"\n{dataset} Dataset:")
        print(f"  📈 Samples: {len(data)}")
        print(f"  📏 Scale: {data['scale'].mean():.4f} ± {data['scale'].std():.4f}")
        print(f"  🌊 Frequency: {data['frequency'].mean():.4f} ± {data['frequency'].std():.4f}")
        print(f"  🎨 Texture Variance: {data['texture_variance'].mean():.4f} ± {data['texture_variance'].std():.4f}")
        print(f"  💡 Mean Intensity: {data['mean_intensity'].mean():.4f} ± {data['mean_intensity'].std():.4f}")
        print(f"  🔷 Solidity: {data['solidity'].mean():.4f} ± {data['solidity'].std():.4f}")

    # Create comprehensive visualization
    print(f"\n🎨 Creating comprehensive visualization...")
    plot_comprehensive_comparison(df_all, args.output)
    
    # Key insights
    print(f"\n🔍 Key Insights:")
    print("="*60)
    
    busi_data = df_all[df_all['dataset'] == 'BUSI']
    busuclm_data = df_all[df_all['dataset'] == 'BUS-UCLM']
    
    if len(busi_data) > 0 and len(busuclm_data) > 0:
        scale_diff = busi_data['scale'].mean() - busuclm_data['scale'].mean()
        freq_diff = busi_data['frequency'].mean() - busuclm_data['frequency'].mean()
        texture_diff = busi_data['texture_variance'].mean() - busuclm_data['texture_variance'].mean()
        
        print(f"🔹 BUSI lesions are {abs(scale_diff):.3f} {'larger' if scale_diff > 0 else 'smaller'} in scale on average")
        print(f"🔹 BUSI images have {abs(freq_diff):.3f} {'higher' if freq_diff > 0 else 'lower'} frequency content")
        print(f"🔹 BUSI images show {abs(texture_diff):.3f} {'more' if texture_diff > 0 else 'less'} texture variance")
        
        if scale_diff > 0.05:
            print("⚠️  Significant scale difference detected - this could affect cross-domain performance")
        if freq_diff > 0.01:
            print("⚠️  Significant frequency difference detected - consider style transfer")
        if texture_diff > 0.01:
            print("⚠️  Significant texture difference detected - domain adaptation recommended")

if __name__ == "__main__":
    main() 