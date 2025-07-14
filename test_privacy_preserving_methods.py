"""
Test Privacy-Preserving Style Transfer Methods
============================================

This script tests and compares different privacy-preserving style transfer methods
for medical images. We'll start with a small batch to validate quality.
"""

import os
import numpy as np
import cv2
from privacy_preserving_style_transfer import PrivacyPreservingStyleTransfer, generate_styled_dataset
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd

def test_single_image_methods():
    """Test all methods on a single image to compare quality."""
    print("🔬 Testing Privacy-Preserving Style Transfer Methods")
    print("=" * 60)
    
    # Step 1: Extract BUSI style statistics
    print("\n📊 Step 1: Extracting BUSI Style Statistics")
    
    style_extractor = PrivacyPreservingStyleTransfer()
    busi_stats = style_extractor.extract_domain_statistics(
        dataset_path='dataset/BioMedicalDataset/BUSI',
        csv_file='train_frame.csv',
        save_path='privacy_style_stats/busi_privacy_stats.json'
    )
    
    print(f"   ✅ BUSI Statistics Summary:")
    print(f"      Mean intensity: {busi_stats['moments']['mean']:.2f}")
    print(f"      Std intensity: {busi_stats['moments']['std']:.2f}")
    print(f"      Skewness: {busi_stats['moments']['skewness']:.3f}")
    print(f"      Processed images: {busi_stats['processed_images']}")
    
    # Step 2: Test all methods on a single BUS-UCLM image
    print("\n🎨 Step 2: Testing All Methods on Single Image")
    
    # Get a test image
    test_image_path = "dataset/BioMedicalDataset/BUS-UCLM/benign/images/VITR_001.png"
    if not os.path.exists(test_image_path):
        print(f"❌ Test image not found: {test_image_path}")
        return
    
    methods = ['histogram_matching', 'fourier_domain', 'statistical_matching', 'gradient_based']
    
    # Create output directory for comparison
    comparison_dir = "privacy_method_comparison"
    os.makedirs(comparison_dir, exist_ok=True)
    
    # Load and save original image
    original_img = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)
    cv2.imwrite(os.path.join(comparison_dir, "00_original.png"), original_img)
    
    results = {}
    
    for method in methods:
        print(f"\n   🔧 Testing {method}...")
        
        try:
            # Initialize style transfer with method
            style_transfer = PrivacyPreservingStyleTransfer(method=method)
            style_transfer.load_style_statistics('privacy_style_stats/busi_privacy_stats.json')
            
            # Apply style transfer
            output_path = os.path.join(comparison_dir, f"{method}_result.png")
            styled_image = style_transfer.apply_style_transfer(test_image_path, output_path)
            
            # Calculate quality metrics
            original_resized = cv2.resize(original_img, (256, 256))
            
            # Basic quality metrics
            mse = np.mean((original_resized.astype(float) - styled_image.astype(float))**2)
            psnr = 20 * np.log10(255.0 / np.sqrt(mse)) if mse > 0 else float('inf')
            
            # Structural similarity (simplified)
            ssim = calculate_simple_ssim(original_resized, styled_image)
            
            results[method] = {
                'mse': mse,
                'psnr': psnr,
                'ssim': ssim,
                'mean_intensity': np.mean(styled_image),
                'std_intensity': np.std(styled_image)
            }
            
            print(f"      ✅ {method}: PSNR={psnr:.2f}, SSIM={ssim:.3f}")
            
        except Exception as e:
            print(f"      ❌ {method} failed: {e}")
            results[method] = None
    
    # Step 3: Visual comparison
    print("\n📊 Step 3: Quality Comparison")
    create_visual_comparison(comparison_dir, methods, results)
    
    # Step 4: Recommend best method
    print("\n🎯 Step 4: Method Recommendation")
    recommend_best_method(results)
    
    return results

def calculate_simple_ssim(img1, img2):
    """Calculate a simplified SSIM metric."""
    mu1 = np.mean(img1)
    mu2 = np.mean(img2)
    sigma1 = np.var(img1)
    sigma2 = np.var(img2)
    sigma12 = np.mean((img1 - mu1) * (img2 - mu2))
    
    c1 = (0.01 * 255)**2
    c2 = (0.03 * 255)**2
    
    ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / ((mu1**2 + mu2**2 + c1) * (sigma1 + sigma2 + c2))
    return ssim

def create_visual_comparison(comparison_dir, methods, results):
    """Create a visual comparison of all methods."""
    print("   📸 Creating visual comparison...")
    
    # Load images
    original = cv2.imread(os.path.join(comparison_dir, "00_original.png"), cv2.IMREAD_GRAYSCALE)
    
    # Create subplot figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # Original image
    axes[0].imshow(original, cmap='gray')
    axes[0].set_title('Original BUS-UCLM')
    axes[0].axis('off')
    
    # Method results
    for i, method in enumerate(methods, 1):
        result_path = os.path.join(comparison_dir, f"{method}_result.png")
        if os.path.exists(result_path):
            styled = cv2.imread(result_path, cv2.IMREAD_GRAYSCALE)
            axes[i].imshow(styled, cmap='gray')
            
            # Add quality metrics to title
            if results[method]:
                title = f"{method.replace('_', ' ').title()}\nPSNR: {results[method]['psnr']:.1f}"
                axes[i].set_title(title)
            else:
                axes[i].set_title(f"{method} (Failed)")
        else:
            axes[i].text(0.5, 0.5, f"{method}\n(Not Available)", 
                        transform=axes[i].transAxes, ha='center', va='center')
        
        axes[i].axis('off')
    
    # Hide last subplot if not needed
    axes[5].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(comparison_dir, "method_comparison.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"      ✅ Visual comparison saved to: {comparison_dir}/method_comparison.png")

def recommend_best_method(results):
    """Recommend the best method based on quality metrics."""
    valid_results = {k: v for k, v in results.items() if v is not None}
    
    if not valid_results:
        print("   ❌ No methods produced valid results")
        return
    
    # Score methods (higher is better)
    scores = {}
    for method, metrics in valid_results.items():
        # Normalize and combine metrics
        ssim_score = metrics['ssim'] * 100  # Higher is better
        psnr_score = min(metrics['psnr'], 50) / 50 * 100  # Higher is better, cap at 50
        
        # Check if intensity statistics are reasonable
        intensity_score = 100
        if metrics['mean_intensity'] < 20 or metrics['mean_intensity'] > 200:
            intensity_score -= 50
        if metrics['std_intensity'] < 10 or metrics['std_intensity'] > 80:
            intensity_score -= 30
        
        scores[method] = (ssim_score + psnr_score + intensity_score) / 3
    
    # Sort by score
    sorted_methods = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    print("   🏆 Method Rankings (Higher Score = Better):")
    for i, (method, score) in enumerate(sorted_methods, 1):
        metrics = valid_results[method]
        print(f"      {i}. {method}: Score={score:.1f}, SSIM={metrics['ssim']:.3f}, PSNR={metrics['psnr']:.1f}")
    
    best_method = sorted_methods[0][0]
    print(f"\n   🎯 Recommended Method: {best_method}")
    print(f"   💡 Reasons: Best balance of structure preservation and style transfer")
    
    return best_method

def test_small_batch_generation(method='histogram_matching'):
    """Test generating a small batch of styled images."""
    print(f"\n🔬 Testing Small Batch Generation with {method}")
    print("=" * 60)
    
    # Create a small subset for testing
    test_csv_path = "test_bus_uclm_subset.csv"
    create_test_subset(test_csv_path)
    
    # Generate styled dataset
    styled_samples = generate_styled_dataset(
        source_dataset_path='dataset/BioMedicalDataset/BUS-UCLM',
        source_csv=test_csv_path,
        style_stats_path='privacy_style_stats/busi_privacy_stats.json',
        output_dir=f'privacy_styled_test_{method}',
        method=method
    )
    
    print(f"   ✅ Generated {len(styled_samples)} styled images")
    
    # Visual inspection
    print("\n📊 Visual Inspection:")
    inspect_generated_images(f'privacy_styled_test_{method}', styled_samples[:3])
    
    return styled_samples

def create_test_subset(output_csv_path, num_samples=10):
    """Create a small subset of BUS-UCLM for testing."""
    full_csv_path = "dataset/BioMedicalDataset/BUS-UCLM/train_frame.csv"
    
    if not os.path.exists(full_csv_path):
        print(f"❌ Full CSV not found: {full_csv_path}")
        return
    
    df = pd.read_csv(full_csv_path)
    
    # Sample equally from both classes
    benign_samples = df[df['image_path'].str.contains('benign')].head(num_samples//2)
    malignant_samples = df[df['image_path'].str.contains('malignant')].head(num_samples//2)
    
    test_df = pd.concat([benign_samples, malignant_samples])
    test_df.to_csv(output_csv_path, index=False)
    
    print(f"   ✅ Created test subset: {output_csv_path} ({len(test_df)} samples)")

def inspect_generated_images(styled_dir, samples):
    """Inspect generated images for quality."""
    print("   🔍 Inspecting generated images...")
    
    inspection_dir = os.path.join(styled_dir, 'inspection')
    os.makedirs(inspection_dir, exist_ok=True)
    
    for i, sample in enumerate(samples):
        # Load original and styled images
        original_class = sample['image_path'].split()[0]
        original_name = sample['image_path'].split()[1] if ' ' in sample['image_path'] else sample['image_path']
        
        original_path = f"dataset/BioMedicalDataset/BUS-UCLM/{original_class}/images/{original_name}"
        styled_path = os.path.join(styled_dir, original_class, 'image', sample['image_path'])
        
        if os.path.exists(original_path) and os.path.exists(styled_path):
            original = cv2.imread(original_path, cv2.IMREAD_GRAYSCALE)
            styled = cv2.imread(styled_path, cv2.IMREAD_GRAYSCALE)
            
            # Create side-by-side comparison
            comparison = np.hstack([original, styled])
            
            # Save comparison
            comparison_path = os.path.join(inspection_dir, f"comparison_{i+1}.png")
            cv2.imwrite(comparison_path, comparison)
            
            # Calculate basic metrics
            mse = np.mean((original.astype(float) - styled.astype(float))**2)
            psnr = 20 * np.log10(255.0 / np.sqrt(mse)) if mse > 0 else float('inf')
            
            print(f"      Sample {i+1}: PSNR={psnr:.2f}, Mean={np.mean(styled):.1f}")
    
    print(f"      ✅ Inspection images saved to: {inspection_dir}")

def main():
    """Main test function."""
    print("🚀 Privacy-Preserving Style Transfer Testing")
    print("=" * 60)
    
    # Test 1: Single image method comparison
    results = test_single_image_methods()
    
    # Test 2: Small batch generation with best method
    if results:
        # Get best method from results
        valid_results = {k: v for k, v in results.items() if v is not None}
        if valid_results:
            # Simple scoring: prefer higher SSIM
            best_method = max(valid_results.items(), key=lambda x: x[1]['ssim'])[0]
            print(f"\n🎯 Proceeding with best method: {best_method}")
            
            # Generate small batch
            styled_samples = test_small_batch_generation(best_method)
            
            print(f"\n✅ Testing Complete!")
            print(f"   📊 Method comparison: privacy_method_comparison/")
            print(f"   🎨 Styled samples: privacy_styled_test_{best_method}/")
            print(f"   📈 Ready for server scaling if quality is good!")
        else:
            print("\n❌ No valid methods found. Please check the implementation.")
    
    # Provide next steps
    print("\n🎯 Next Steps:")
    print("   1. Check the visual comparisons in privacy_method_comparison/")
    print("   2. Inspect the generated samples in privacy_styled_test_*/")
    print("   3. If quality is good, scale up on server")
    print("   4. If not, we can tune parameters or try different approaches")

if __name__ == "__main__":
    main() 