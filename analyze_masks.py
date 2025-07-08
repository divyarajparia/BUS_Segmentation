import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def analyze_mask_shapes():
    """Analyze the shapes of generated masks"""
    
    print("🔍 Analyzing Generated Mask Shapes")
    print("=" * 40)
    
    # Files to analyze
    mask_files = {
        'Generated Benign': 'local_test_output/test_benign_01_mask.png',
        'Generated Malignant': 'local_test_output/test_malignant_01_mask.png',
        'Real Benign': 'debug_data/BUSI/benign/mask/benign (1)_mask.png',
        'Real Malignant': 'debug_data/BUSI/malignant/mask/malignant (1)_mask.png'
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle('Mask Shape Analysis: Generated vs Real', fontsize=14, fontweight='bold')
    
    for i, (name, file_path) in enumerate(mask_files.items()):
        try:
            # Load and convert mask
            mask = Image.open(file_path).convert('L')
            mask_array = np.array(mask)
            
            # Convert to binary
            binary_mask = (mask_array > 127).astype(int)
            
            # Plot
            row, col = i // 2, i % 2
            axes[row, col].imshow(mask_array, cmap='gray')
            axes[row, col].set_title(f'{name}')
            axes[row, col].axis('off')
            
            # Calculate shape metrics
            if binary_mask.sum() > 0:
                y_coords, x_coords = np.where(binary_mask)
                
                # Basic metrics
                area = binary_mask.sum()
                height = y_coords.max() - y_coords.min() + 1 if len(y_coords) > 1 else 1
                width = x_coords.max() - x_coords.min() + 1 if len(x_coords) > 1 else 1
                aspect_ratio = width / height
                
                # Circularity (4π*area/perimeter²) - lower means more irregular
                from scipy import ndimage
                try:
                    # Calculate perimeter using edge detection
                    edges = ndimage.sobel(binary_mask.astype(float))
                    perimeter = (edges > 0).sum()
                    circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
                except:
                    circularity = 0
                
                # Compactness
                bbox_area = height * width
                compactness = area / bbox_area if bbox_area > 0 else 0
                
                print(f"\n{name}:")
                print(f"  Area: {area} pixels")
                print(f"  Dimensions: {width}x{height}")
                print(f"  Aspect Ratio: {aspect_ratio:.2f}")
                print(f"  Circularity: {circularity:.3f} (lower = more irregular)")
                print(f"  Compactness: {compactness:.3f}")
                
                # Check if it's just a simple ellipse/circle
                if circularity > 0.7:
                    print("  ⚠️  Shape appears very regular (possibly elliptical)")
                elif circularity > 0.4:
                    print("  ✅ Shape has moderate irregularity")
                else:
                    print("  ✅ Shape is highly irregular (realistic)")
            else:
                print(f"\n{name}: Empty mask!")
                
        except Exception as e:
            print(f"❌ Error loading {name}: {e}")
    
    plt.tight_layout()
    plt.savefig('mask_shape_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\n✅ Analysis saved as: mask_shape_analysis.png")
    
    # Summary
    print(f"\n📝 Summary:")
    print(f"  • Circularity close to 1.0 = perfect circle")
    print(f"  • Circularity 0.4-0.7 = realistic tumor shape")
    print(f"  • Circularity < 0.4 = very irregular (good for malignant)")

if __name__ == "__main__":
    analyze_mask_shapes() 