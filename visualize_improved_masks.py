import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os

def visualize_improved_results():
    """Visualize the improved mask generation results"""
    
    print("🖼️  Visualizing Improved Mask Generation Results")
    print("=" * 60)
    
    # Find generated files
    test_dir = 'local_test_output'
    
    # Get the latest generated files (test_ prefix)
    generated_files = {
        'benign_img': f'{test_dir}/test_benign_01_img.png',
        'benign_mask': f'{test_dir}/test_benign_01_mask.png',
        'malignant_img': f'{test_dir}/test_malignant_01_img.png',
        'malignant_mask': f'{test_dir}/test_malignant_01_mask.png',
    }
    
    # Real BUSI files for comparison
    real_files = {
        'benign_img': 'debug_data/BUSI/benign/image/benign (1).png',
        'benign_mask': 'debug_data/BUSI/benign/mask/benign (1)_mask.png',
        'malignant_img': 'debug_data/BUSI/malignant/image/malignant (1).png',
        'malignant_mask': 'debug_data/BUSI/malignant/mask/malignant (1)_mask.png',
    }
    
    # Create comparison visualization
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('Improved Mask Generation: Real BUSI vs Generated', fontsize=16, fontweight='bold')
    
    # Row 1: Benign samples
    row_titles = ['Benign', 'Malignant']
    col_titles = ['Real Image', 'Real Mask', 'Generated Image', 'Generated Mask']
    
    for row, (class_name, row_title) in enumerate(zip(['benign', 'malignant'], row_titles)):
        # Real image and mask
        real_img = Image.open(real_files[f'{class_name}_img']).convert('L')
        real_mask = Image.open(real_files[f'{class_name}_mask']).convert('L')
        
        # Generated image and mask
        gen_img = Image.open(generated_files[f'{class_name}_img']).convert('L')
        gen_mask = Image.open(generated_files[f'{class_name}_mask']).convert('L')
        
        # Plot real image
        axes[row, 0].imshow(real_img, cmap='gray')
        axes[row, 0].set_title(f'Real {class_name.title()} Image')
        axes[row, 0].axis('off')
        
        # Plot real mask
        axes[row, 1].imshow(real_mask, cmap='gray')
        axes[row, 1].set_title(f'Real {class_name.title()} Mask')
        axes[row, 1].axis('off')
        
        # Plot generated image
        axes[row, 2].imshow(gen_img, cmap='gray')
        axes[row, 2].set_title(f'Generated {class_name.title()} Image')
        axes[row, 2].axis('off')
        
        # Plot generated mask
        axes[row, 3].imshow(gen_mask, cmap='gray')
        axes[row, 3].set_title(f'Generated {class_name.title()} Mask')
        axes[row, 3].axis('off')
        
        # Add row label
        axes[row, 0].text(-0.1, 0.5, row_title, transform=axes[row, 0].transAxes,
                         fontsize=14, fontweight='bold', va='center', rotation=90)
    
    plt.tight_layout()
    plt.savefig('improved_mask_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Analyze mask characteristics
    print("\n📊 Mask Analysis:")
    print("-" * 30)
    
    for class_name in ['benign', 'malignant']:
        print(f"\n{class_name.upper()} MASKS:")
        
        # Load masks
        real_mask = np.array(Image.open(real_files[f'{class_name}_mask']).convert('L'))
        gen_mask = np.array(Image.open(generated_files[f'{class_name}_mask']).convert('L'))
        
        # Convert to binary
        real_binary = (real_mask > 127).astype(int)
        gen_binary = (gen_mask > 127).astype(int)
        
        # Calculate statistics
        real_area = real_binary.sum()
        gen_area = gen_binary.sum()
        
        # Calculate shape characteristics
        real_y, real_x = np.where(real_binary)
        gen_y, gen_x = np.where(gen_binary)
        
        if len(real_y) > 0 and len(gen_y) > 0:
            # Aspect ratios
            real_height = real_y.max() - real_y.min() if len(real_y) > 1 else 1
            real_width = real_x.max() - real_x.min() if len(real_x) > 1 else 1
            real_aspect = real_width / max(real_height, 1)
            
            gen_height = gen_y.max() - gen_y.min() if len(gen_y) > 1 else 1
            gen_width = gen_x.max() - gen_x.min() if len(gen_x) > 1 else 1
            gen_aspect = gen_width / max(gen_height, 1)
            
            print(f"  Real:      Area={real_area:4d} pixels, Aspect={real_aspect:.2f}")
            print(f"  Generated: Area={gen_area:4d} pixels, Aspect={gen_aspect:.2f}")
            print(f"  Size Ratio: {gen_area/max(real_area, 1):.2f}x")
        else:
            print(f"  Real:      Area={real_area:4d} pixels")
            print(f"  Generated: Area={gen_area:4d} pixels")
    
    print(f"\n✅ Visualization saved as: improved_mask_comparison.png")
    print("\n🔍 Key Improvements:")
    print("  • Using generator's learned patterns instead of fixed ellipses")
    print("  • Adaptive thresholding based on tumor class")
    print("  • Morphological operations for realistic boundaries")
    print("  • Class-specific irregularity (malignant more irregular)")

if __name__ == "__main__":
    visualize_improved_results() 