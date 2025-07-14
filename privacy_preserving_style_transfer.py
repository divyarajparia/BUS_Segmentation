"""
Privacy-Preserving Style Transfer for Medical Images
==================================================

Multiple approaches for BUSI-style generation from BUS-UCLM without full dataset access.
Designed for federated learning scenarios with privacy constraints.

Methods implemented:
1. Statistical Histogram Matching (Simple & Effective)
2. Fourier Domain Style Transfer (Advanced)
3. Multi-Scale Statistical Matching (Robust)
4. Gradient-Based Texture Transfer (Sophisticated)
"""

import numpy as np
import cv2
from PIL import Image
import os
import pandas as pd
from scipy import ndimage
from scipy.stats import wasserstein_distance
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class PrivacyPreservingStyleTransfer:
    """
    Privacy-preserving style transfer for medical images.
    Only requires statistical information from target domain (BUSI).
    """
    
    def __init__(self, method='histogram_matching'):
        """
        Initialize style transfer with specified method.
        
        Args:
            method: 'histogram_matching', 'fourier_domain', 'statistical_matching', or 'gradient_based'
        """
        self.method = method
        self.style_stats = None
        
    def extract_domain_statistics(self, dataset_path, csv_file, save_path=None):
        """
        Extract privacy-preserving statistics from target domain (BUSI).
        Only statistical information is stored, not raw images.
        """
        print(f"📊 Extracting privacy-preserving statistics from {dataset_path}")
        
        # Load CSV
        csv_path = os.path.join(dataset_path, csv_file)
        df = pd.read_csv(csv_path)
        
        # Initialize statistics containers
        histograms = []
        fourier_stats = []
        moment_stats = []
        gradient_stats = []
        
        valid_count = 0
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing images"):
            try:
                # Handle different dataset formats
                image_filename = row['image_path']
                
                if ' ' in image_filename and '(' in image_filename:
                    # BUSI format
                    class_type = image_filename.split()[0]
                    img_path = os.path.join(dataset_path, class_type, 'image', image_filename)
                else:
                    # Fallback
                    img_path = os.path.join(dataset_path, image_filename)
                
                if not os.path.exists(img_path):
                    continue
                
                # Load image as grayscale
                image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if image is None:
                    continue
                
                # Resize to standard size
                image = cv2.resize(image, (256, 256))
                
                # Extract different types of statistics
                
                # 1. Histogram statistics
                hist = cv2.calcHist([image], [0], None, [256], [0, 256])
                histograms.append(hist.flatten())
                
                # 2. Fourier domain statistics
                f_transform = np.fft.fft2(image)
                magnitude = np.abs(f_transform)
                phase = np.angle(f_transform)
                
                fourier_stats.append({
                    'magnitude_mean': np.mean(magnitude),
                    'magnitude_std': np.std(magnitude),
                    'phase_mean': np.mean(phase),
                    'phase_std': np.std(phase)
                })
                
                # 3. Statistical moments
                moments = {
                    'mean': np.mean(image),
                    'std': np.std(image),
                    'skewness': self._calculate_skewness(image),
                    'kurtosis': self._calculate_kurtosis(image)
                }
                moment_stats.append(moments)
                
                # 4. Gradient statistics
                grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
                grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
                gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
                
                gradient_stats.append({
                    'grad_mean': np.mean(gradient_magnitude),
                    'grad_std': np.std(gradient_magnitude),
                    'grad_max': np.max(gradient_magnitude)
                })
                
                valid_count += 1
                
            except Exception as e:
                print(f"   ⚠️ Error processing {image_filename}: {e}")
                continue
        
        # Aggregate statistics
        self.style_stats = {
            'histogram': {
                'mean_histogram': np.mean(histograms, axis=0),
                'std_histogram': np.std(histograms, axis=0)
            },
            'fourier': {
                'magnitude_mean': np.mean([s['magnitude_mean'] for s in fourier_stats]),
                'magnitude_std': np.mean([s['magnitude_std'] for s in fourier_stats]),
                'phase_mean': np.mean([s['phase_mean'] for s in fourier_stats]),
                'phase_std': np.mean([s['phase_std'] for s in fourier_stats])
            },
            'moments': {
                'mean': np.mean([s['mean'] for s in moment_stats]),
                'std': np.mean([s['std'] for s in moment_stats]),
                'skewness': np.mean([s['skewness'] for s in moment_stats]),
                'kurtosis': np.mean([s['kurtosis'] for s in moment_stats])
            },
            'gradient': {
                'grad_mean': np.mean([s['grad_mean'] for s in gradient_stats]),
                'grad_std': np.mean([s['grad_std'] for s in gradient_stats]),
                'grad_max': np.mean([s['grad_max'] for s in gradient_stats])
            },
            'processed_images': valid_count
        }
        
        # Save statistics if requested
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'w') as f:
                # Convert numpy arrays to lists for JSON serialization
                stats_to_save = self._convert_for_json(self.style_stats)
                json.dump(stats_to_save, f, indent=2)
        
        print(f"   ✅ Extracted privacy-preserving statistics from {valid_count} images")
        return self.style_stats
    
    def load_style_statistics(self, stats_path):
        """Load pre-computed style statistics from file."""
        with open(stats_path, 'r') as f:
            self.style_stats = json.load(f)
        
        # Convert lists back to numpy arrays
        self.style_stats['histogram']['mean_histogram'] = np.array(self.style_stats['histogram']['mean_histogram'])
        self.style_stats['histogram']['std_histogram'] = np.array(self.style_stats['histogram']['std_histogram'])
        
        print(f"   ✅ Loaded privacy-preserving statistics from {stats_path}")
    
    def apply_style_transfer(self, source_image_path, output_path):
        """
        Apply privacy-preserving style transfer to source image.
        
        Args:
            source_image_path: Path to BUS-UCLM image
            output_path: Path to save styled image
        """
        if self.style_stats is None:
            raise ValueError("Style statistics not loaded. Call extract_domain_statistics() first.")
        
        # Load source image
        source_image = cv2.imread(source_image_path, cv2.IMREAD_GRAYSCALE)
        if source_image is None:
            raise ValueError(f"Could not load image: {source_image_path}")
        
        # Resize to standard size
        source_image = cv2.resize(source_image, (256, 256))
        
        # Apply style transfer based on method
        if self.method == 'histogram_matching':
            styled_image = self._histogram_matching(source_image)
        elif self.method == 'fourier_domain':
            styled_image = self._fourier_domain_transfer(source_image)
        elif self.method == 'statistical_matching':
            styled_image = self._statistical_matching(source_image)
        elif self.method == 'gradient_based':
            styled_image = self._gradient_based_transfer(source_image)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        # Save styled image as RGB to match BUSI format
        # Convert grayscale to RGB for consistency with BUSI
        if len(styled_image.shape) == 2:  # Grayscale
            styled_image_rgb = cv2.cvtColor(styled_image, cv2.COLOR_GRAY2RGB)
        else:  # Already RGB
            styled_image_rgb = styled_image
        
        cv2.imwrite(output_path, styled_image_rgb)
        return styled_image_rgb
    
    def _histogram_matching(self, source_image):
        """
        Method 1: Histogram Matching
        Simple but effective for medical images.
        """
        # Get source histogram
        source_hist = cv2.calcHist([source_image], [0], None, [256], [0, 256]).flatten()
        
        # Get target histogram from statistics
        target_hist = self.style_stats['histogram']['mean_histogram']
        
        # Compute cumulative distribution functions
        source_cdf = np.cumsum(source_hist)
        target_cdf = np.cumsum(target_hist)
        
        # Normalize CDFs
        source_cdf = source_cdf / source_cdf[-1]
        target_cdf = target_cdf / target_cdf[-1]
        
        # Create lookup table
        lookup_table = np.zeros(256, dtype=np.uint8)
        for i in range(256):
            # Find closest match in target CDF
            closest_idx = np.argmin(np.abs(target_cdf - source_cdf[i]))
            lookup_table[i] = closest_idx
        
        # Apply transformation
        styled_image = cv2.LUT(source_image, lookup_table)
        return styled_image
    
    def _fourier_domain_transfer(self, source_image):
        """
        Method 2: Fourier Domain Style Transfer
        Preserves content while changing frequency characteristics.
        """
        # Get Fourier transform of source
        f_source = np.fft.fft2(source_image)
        magnitude_source = np.abs(f_source)
        phase_source = np.angle(f_source)
        
        # Get target statistics
        target_stats = self.style_stats['fourier']
        
        # Modify magnitude while preserving phase (content)
        magnitude_styled = magnitude_source * (target_stats['magnitude_std'] / np.std(magnitude_source))
        magnitude_styled = magnitude_styled + (target_stats['magnitude_mean'] - np.mean(magnitude_styled))
        
        # Reconstruct with original phase
        f_styled = magnitude_styled * np.exp(1j * phase_source)
        
        # Inverse FFT
        styled_image = np.real(np.fft.ifft2(f_styled))
        
        # Normalize to [0, 255]
        styled_image = np.clip(styled_image, 0, 255).astype(np.uint8)
        
        return styled_image
    
    def _statistical_matching(self, source_image):
        """
        Method 3: Multi-Scale Statistical Matching
        Matches multiple statistical moments.
        """
        styled_image = source_image.astype(np.float64)
        target_stats = self.style_stats['moments']
        
        # Match mean and std
        styled_image = (styled_image - np.mean(styled_image)) / np.std(styled_image)
        styled_image = styled_image * target_stats['std'] + target_stats['mean']
        
        # Apply non-linear transformation to match higher-order moments
        # This is a simplified approach - more sophisticated methods exist
        current_skew = self._calculate_skewness(styled_image)
        target_skew = target_stats['skewness']
        
        if abs(current_skew - target_skew) > 0.1:
            # Apply gamma correction to adjust skewness
            gamma = 1.0 + (target_skew - current_skew) * 0.1
            styled_image = np.power(styled_image / 255.0, gamma) * 255.0
        
        # Clip to valid range
        styled_image = np.clip(styled_image, 0, 255).astype(np.uint8)
        
        return styled_image
    
    def _gradient_based_transfer(self, source_image):
        """
        Method 4: Gradient-Based Texture Transfer
        Preserves structure while changing texture characteristics.
        """
        # Calculate gradients
        grad_x = cv2.Sobel(source_image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(source_image, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Get target gradient statistics
        target_grad_stats = self.style_stats['gradient']
        
        # Normalize and rescale gradient
        current_grad_mean = np.mean(gradient_magnitude)
        current_grad_std = np.std(gradient_magnitude)
        
        if current_grad_std > 0:
            gradient_magnitude = (gradient_magnitude - current_grad_mean) / current_grad_std
            gradient_magnitude = gradient_magnitude * target_grad_stats['grad_std'] + target_grad_stats['grad_mean']
        
        # Reconstruct image with modified gradients (simplified approach)
        # This is a basic implementation - more sophisticated methods exist
        styled_image = source_image.astype(np.float64)
        
        # Apply intensity adjustment based on gradient statistics
        intensity_factor = target_grad_stats['grad_mean'] / max(current_grad_mean, 1e-6)
        styled_image = styled_image * intensity_factor
        
        # Match overall statistics
        target_stats = self.style_stats['moments']
        styled_image = (styled_image - np.mean(styled_image)) / max(np.std(styled_image), 1e-6)
        styled_image = styled_image * target_stats['std'] + target_stats['mean']
        
        # Clip to valid range
        styled_image = np.clip(styled_image, 0, 255).astype(np.uint8)
        
        return styled_image
    
    def _calculate_skewness(self, image):
        """Calculate skewness of image intensity distribution."""
        mean = np.mean(image)
        std = np.std(image)
        if std == 0:
            return 0
        return np.mean(((image - mean) / std) ** 3)
    
    def _calculate_kurtosis(self, image):
        """Calculate kurtosis of image intensity distribution."""
        mean = np.mean(image)
        std = np.std(image)
        if std == 0:
            return 0
        return np.mean(((image - mean) / std) ** 4) - 3
    
    def _convert_for_json(self, obj):
        """Convert numpy arrays to lists for JSON serialization."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_for_json(item) for item in obj]
        else:
            return obj

def generate_styled_dataset(source_dataset_path, source_csv, style_stats_path, 
                          output_dir, method='histogram_matching'):
    """
    Generate complete styled dataset using privacy-preserving style transfer.
    
    Args:
        source_dataset_path: Path to BUS-UCLM dataset
        source_csv: CSV file with BUS-UCLM samples
        style_stats_path: Path to BUSI style statistics
        output_dir: Output directory for styled images
        method: Style transfer method to use
    """
    print(f"🎨 Generating styled dataset using {method}")
    
    # Initialize style transfer
    style_transfer = PrivacyPreservingStyleTransfer(method=method)
    style_transfer.load_style_statistics(style_stats_path)
    
    # Load source CSV
    if os.path.exists(source_csv):
        # CSV is already full path
        csv_path = source_csv
    else:
        # CSV is relative to dataset path
        csv_path = os.path.join(source_dataset_path, source_csv)
    df = pd.read_csv(csv_path)
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    
    styled_samples = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Styling images"):
        try:
            # Get source image path
            image_filename = row['image_path']
            mask_filename = row['mask_path']
            
            # Handle different dataset formats
            if ' ' in image_filename:
                if '(' in image_filename:
                    # BUSI format
                    class_type = image_filename.split()[0]
                    source_path = os.path.join(source_dataset_path, class_type, 'image', image_filename)
                    mask_path = os.path.join(source_dataset_path, class_type, 'mask', mask_filename)
                else:
                    # BUS-UCLM format
                    class_type = image_filename.split()[0]
                    image_name = image_filename.split()[1]
                    mask_name = mask_filename.split()[1]
                    source_path = os.path.join(source_dataset_path, class_type, 'images', image_name)
                    mask_path = os.path.join(source_dataset_path, class_type, 'masks', mask_name)
            else:
                # Fallback
                source_path = os.path.join(source_dataset_path, image_filename)
                mask_path = os.path.join(source_dataset_path, mask_filename)
            
            if not os.path.exists(source_path):
                print(f"   ⚠️ Source image not found: {source_path}")
                continue
            
            # Create output paths
            styled_filename = f"styled_{image_filename}"
            styled_mask_filename = f"styled_{mask_filename}"
            
            # Create class directories
            output_image_dir = os.path.join(output_dir, class_type, 'image')
            output_mask_dir = os.path.join(output_dir, class_type, 'mask')
            os.makedirs(output_image_dir, exist_ok=True)
            os.makedirs(output_mask_dir, exist_ok=True)
            
            styled_image_path = os.path.join(output_image_dir, styled_filename)
            styled_mask_path = os.path.join(output_mask_dir, styled_mask_filename)
            
            # Apply style transfer to image
            style_transfer.apply_style_transfer(source_path, styled_image_path)
            
            # Process mask properly (resize and convert to match image format)
            if os.path.exists(mask_path):
                # Load original mask
                original_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if original_mask is None:
                    # Try as RGB and convert
                    original_mask = cv2.imread(mask_path)
                    if original_mask is not None:
                        original_mask = cv2.cvtColor(original_mask, cv2.COLOR_BGR2GRAY)
                
                if original_mask is not None:
                    # Resize mask to match styled image size (256x256)
                    resized_mask = cv2.resize(original_mask, (256, 256), interpolation=cv2.INTER_NEAREST)
                    
                    # Ensure binary mask (threshold if needed)
                    _, binary_mask = cv2.threshold(resized_mask, 127, 255, cv2.THRESH_BINARY)
                    
                    # Save processed mask
                    cv2.imwrite(styled_mask_path, binary_mask)
                else:
                    print(f"   ⚠️ Could not load mask: {mask_path}")
                    continue
            
            # Record styled sample with all required columns for CCSTDataset
            styled_samples.append({
                'image_path': styled_filename,
                'mask_path': styled_mask_filename,
                'class': class_type,
                'source_client': 'BUS-UCLM',  # Source dataset
                'style_client': 'BUSI',       # Style comes from BUSI
                'augmentation_type': 'styled', # Type of augmentation
                'original_image': image_filename
            })
            
        except Exception as e:
            print(f"   ⚠️ Error processing {image_filename}: {e}")
            continue
    
    # Save styled dataset CSV
    styled_df = pd.DataFrame(styled_samples)
    styled_csv_path = os.path.join(output_dir, 'styled_dataset.csv')
    styled_df.to_csv(styled_csv_path, index=False)
    
    print(f"   ✅ Generated {len(styled_samples)} styled images")
    print(f"   ✅ Styled dataset saved to: {output_dir}")
    print(f"   ✅ CSV saved to: {styled_csv_path}")
    
    return styled_samples

if __name__ == "__main__":
    # Example usage
    print("🎯 Privacy-Preserving Style Transfer for Medical Images")
    print("=" * 60)
    
    # Method comparison
    methods = ['histogram_matching', 'fourier_domain', 'statistical_matching', 'gradient_based']
    
    print("Available methods:")
    for i, method in enumerate(methods, 1):
        print(f"   {i}. {method}")
    
    print("\nRecommended order to try:")
    print("   1. histogram_matching (simplest, often most effective)")
    print("   2. fourier_domain (preserves structure well)")
    print("   3. statistical_matching (more sophisticated)")
    print("   4. gradient_based (most complex)") 