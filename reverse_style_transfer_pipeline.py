#!/usr/bin/env python3
"""
Reverse Style Transfer Pipeline: BUS-UCLM Style → BUSI Images

This pipeline:
1. Captures BUS-UCLM style statistics from the source dataset
2. Applies BUS-UCLM style to BUSI images
3. Creates a combined dataset with original BUS-UCLM + styled BUSI
4. Trains on combined dataset and tests on BUS-UCLM test set

This is the reverse of the previous approach - instead of making BUSI-style 
synthetic data, we make BUS-UCLM-style synthetic data.
"""

import os
import json
import pandas as pd
import numpy as np
from PIL import Image
import argparse
import cv2
from sklearn.feature_extraction import image as sk_image
from skimage import filters, morphology, measure
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.measure import shannon_entropy
from scipy import ndimage
from scipy.stats import entropy
import matplotlib.pyplot as plt
from tqdm import tqdm
import gc
import shutil
import warnings
warnings.filterwarnings('ignore')

class BUSUCLMStyleExtractor:
    """Extract comprehensive style statistics from BUS-UCLM dataset"""
    
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.style_stats = {}
        
    def load_image_safely(self, image_path):
        """Load image with error handling"""
        try:
            img = Image.open(image_path).convert('L')
            return np.array(img)
        except Exception as e:
            print(f"⚠️  Error loading {image_path}: {e}")
            return None
    
    def extract_histogram_stats(self, image):
        """Extract histogram statistics"""
        hist, bins = np.histogram(image, bins=256, range=(0, 255))
        hist = hist.astype(float) / hist.sum()  # Normalize
        
        return {
            'histogram': hist.tolist(),
            'bins': bins.tolist(),
            'mean': float(np.mean(image)),
            'std': float(np.std(image)),
            'min': float(np.min(image)),
            'max': float(np.max(image))
        }
    
    def extract_texture_stats(self, image):
        """Extract texture statistics using GLCM and LBP"""
        # GLCM features
        glcm = graycomatrix(image, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], 
                          levels=256, symmetric=True, normed=True)
        
        contrast = graycoprops(glcm, 'contrast')
        dissimilarity = graycoprops(glcm, 'dissimilarity')
        homogeneity = graycoprops(glcm, 'homogeneity')
        energy = graycoprops(glcm, 'energy')
        
        # LBP features
        lbp = local_binary_pattern(image, P=24, R=8, method='uniform')
        lbp_hist, _ = np.histogram(lbp, bins=26, range=(0, 25))
        lbp_hist = lbp_hist.astype(float) / lbp_hist.sum()
        
        return {
            'glcm_contrast': float(np.mean(contrast)),
            'glcm_dissimilarity': float(np.mean(dissimilarity)),
            'glcm_homogeneity': float(np.mean(homogeneity)),
            'glcm_energy': float(np.mean(energy)),
            'lbp_histogram': lbp_hist.tolist(),
            'texture_energy': float(np.sum(image**2) / (image.shape[0] * image.shape[1]))
        }
    
    def extract_frequency_stats(self, image):
        """Extract frequency domain statistics"""
        # FFT
        fft = np.fft.fft2(image)
        fft_magnitude = np.abs(fft)
        fft_phase = np.angle(fft)
        
        # Power spectrum
        power_spectrum = fft_magnitude**2
        
        # Frequency bands
        h, w = image.shape
        center_h, center_w = h // 2, w // 2
        
        # Low frequency (center region)
        low_freq_mask = np.zeros((h, w))
        low_freq_mask[center_h-h//8:center_h+h//8, center_w-w//8:center_w+w//8] = 1
        low_freq_power = np.sum(power_spectrum * low_freq_mask)
        
        # High frequency (outer region)
        high_freq_mask = 1 - low_freq_mask
        high_freq_power = np.sum(power_spectrum * high_freq_mask)
        
        return {
            'fft_mean_magnitude': float(np.mean(fft_magnitude)),
            'fft_std_magnitude': float(np.std(fft_magnitude)),
            'low_freq_power': float(low_freq_power),
            'high_freq_power': float(high_freq_power),
            'freq_ratio': float(low_freq_power / (high_freq_power + 1e-8))
        }
    
    def extract_morphological_stats(self, image):
        """Extract morphological statistics"""
        # Binary morphology
        binary = image > np.mean(image)
        
        # Morphological operations
        opened = morphology.opening(binary, morphology.disk(3))
        closed = morphology.closing(binary, morphology.disk(3))
        
        # Region properties
        labeled = measure.label(binary)
        regions = measure.regionprops(labeled)
        
        if regions:
            areas = [region.area for region in regions]
            eccentricities = [region.eccentricity for region in regions]
            
            return {
                'mean_region_area': float(np.mean(areas)),
                'std_region_area': float(np.std(areas)),
                'mean_eccentricity': float(np.mean(eccentricities)),
                'num_regions': len(regions),
                'opening_ratio': float(np.sum(opened) / np.sum(binary)),
                'closing_ratio': float(np.sum(closed) / np.sum(binary))
            }
        
        return {
            'mean_region_area': 0.0,
            'std_region_area': 0.0,
            'mean_eccentricity': 0.0,
            'num_regions': 0,
            'opening_ratio': 0.0,
            'closing_ratio': 0.0
        }
    
    def extract_comprehensive_stats(self, image):
        """Extract all style statistics from an image"""
        stats = {}
        
        # Basic histogram statistics
        stats['histogram'] = self.extract_histogram_stats(image)
        
        # Texture statistics
        stats['texture'] = self.extract_texture_stats(image)
        
        # Frequency statistics
        stats['frequency'] = self.extract_frequency_stats(image)
        
        # Morphological statistics
        stats['morphology'] = self.extract_morphological_stats(image)
        
        # Additional stats
        stats['entropy'] = float(shannon_entropy(image))
        stats['local_entropy'] = float(np.mean(filters.rank.entropy(image, morphology.disk(5))))
        
        return stats
    
    def extract_dataset_style(self, split='all'):
        """Extract style statistics from BUS-UCLM dataset"""
        print(f"🎯 Extracting BUS-UCLM style statistics...")
        
        # Load dataset splits
        if split == 'all':
            csv_files = ['train_frame.csv', 'val_frame.csv', 'test_frame.csv']
        else:
            csv_files = [f'{split}_frame.csv']
        
        all_stats = []
        total_images = 0
        
        for csv_file in csv_files:
            csv_path = os.path.join(self.dataset_path, csv_file)
            if not os.path.exists(csv_path):
                continue
                
            df = pd.read_csv(csv_path)
            print(f"  📊 Processing {csv_file}: {len(df)} images")
            
            for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {csv_file}"):
                # Parse image path
                image_entry = row['image_path']
                if ' ' in image_entry:
                    class_name, filename = image_entry.split(' ', 1)
                else:
                    class_name, filename = 'unknown', image_entry
                
                # Construct full path
                image_path = os.path.join(self.dataset_path, class_name, 'images', filename)
                
                # Load and process image
                image = self.load_image_safely(image_path)
                if image is not None:
                    stats = self.extract_comprehensive_stats(image)
                    stats['class'] = class_name
                    stats['filename'] = filename
                    all_stats.append(stats)
                    total_images += 1
        
        print(f"  ✅ Processed {total_images} images total")
        
        # Aggregate statistics
        self.style_stats = self.aggregate_stats(all_stats)
        return self.style_stats
    
    def aggregate_stats(self, all_stats):
        """Aggregate statistics across all images"""
        print("  🔄 Aggregating style statistics...")
        
        aggregated = {
            'histogram': {
                'mean_histogram': np.mean([s['histogram']['histogram'] for s in all_stats], axis=0).tolist(),
                'mean_intensity': np.mean([s['histogram']['mean'] for s in all_stats]),
                'std_intensity': np.mean([s['histogram']['std'] for s in all_stats]),
                'global_min': np.min([s['histogram']['min'] for s in all_stats]),
                'global_max': np.max([s['histogram']['max'] for s in all_stats])
            },
            'texture': {
                'mean_contrast': np.mean([s['texture']['glcm_contrast'] for s in all_stats]),
                'mean_dissimilarity': np.mean([s['texture']['glcm_dissimilarity'] for s in all_stats]),
                'mean_homogeneity': np.mean([s['texture']['glcm_homogeneity'] for s in all_stats]),
                'mean_energy': np.mean([s['texture']['glcm_energy'] for s in all_stats]),
                'mean_lbp_histogram': np.mean([s['texture']['lbp_histogram'] for s in all_stats], axis=0).tolist(),
                'mean_texture_energy': np.mean([s['texture']['texture_energy'] for s in all_stats])
            },
            'frequency': {
                'mean_fft_magnitude': np.mean([s['frequency']['fft_mean_magnitude'] for s in all_stats]),
                'mean_low_freq_power': np.mean([s['frequency']['low_freq_power'] for s in all_stats]),
                'mean_high_freq_power': np.mean([s['frequency']['high_freq_power'] for s in all_stats]),
                'mean_freq_ratio': np.mean([s['frequency']['freq_ratio'] for s in all_stats])
            },
            'morphology': {
                'mean_region_area': np.mean([s['morphology']['mean_region_area'] for s in all_stats]),
                'mean_eccentricity': np.mean([s['morphology']['mean_eccentricity'] for s in all_stats]),
                'mean_num_regions': np.mean([s['morphology']['num_regions'] for s in all_stats]),
                'mean_opening_ratio': np.mean([s['morphology']['opening_ratio'] for s in all_stats]),
                'mean_closing_ratio': np.mean([s['morphology']['closing_ratio'] for s in all_stats])
            },
            'entropy': {
                'mean_entropy': np.mean([s['entropy'] for s in all_stats]),
                'mean_local_entropy': np.mean([s['local_entropy'] for s in all_stats])
            }
        }
        
        return aggregated
    
    def save_style_stats(self, output_path):
        """Save style statistics to JSON file"""
        with open(output_path, 'w') as f:
            json.dump(self.style_stats, f, indent=2)
        print(f"  💾 Style statistics saved to {output_path}")

class BUSUCLMStyleTransfer:
    """Apply BUS-UCLM style to BUSI images"""
    
    def __init__(self, style_stats):
        self.style_stats = style_stats
        
    def apply_histogram_matching(self, image, reference_hist):
        """Apply histogram matching to match reference distribution"""
        # Calculate CDF of source image
        source_hist, _ = np.histogram(image, bins=256, range=(0, 255))
        source_hist = source_hist.astype(float) / source_hist.sum()
        source_cdf = np.cumsum(source_hist)
        
        # Reference CDF
        reference_hist = np.array(reference_hist)
        reference_cdf = np.cumsum(reference_hist)
        
        # Create mapping
        mapping = np.zeros(256, dtype=np.uint8)
        for i in range(256):
            # Find closest CDF value in reference
            diff = np.abs(reference_cdf - source_cdf[i])
            mapping[i] = np.argmin(diff)
        
        # Apply mapping
        matched = mapping[image]
        return matched
    
    def apply_texture_enhancement(self, image, texture_stats):
        """Apply texture enhancement based on BUS-UCLM texture statistics"""
        # Enhance contrast based on target statistics
        target_contrast = texture_stats['mean_contrast']
        
        # Apply CLAHE with parameters derived from target stats
        clahe = cv2.createCLAHE(clipLimit=target_contrast * 2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(image.astype(np.uint8))
        
        # Texture energy adjustment
        target_energy = texture_stats['mean_texture_energy']
        current_energy = np.sum(image**2) / (image.shape[0] * image.shape[1])
        
        if current_energy > 0:
            energy_ratio = target_energy / current_energy
            enhanced = enhanced * np.sqrt(energy_ratio)
        
        return np.clip(enhanced, 0, 255).astype(np.uint8)
    
    def apply_frequency_enhancement(self, image, frequency_stats):
        """Apply frequency domain enhancement"""
        # FFT
        fft = np.fft.fft2(image)
        fft_magnitude = np.abs(fft)
        fft_phase = np.angle(fft)
        
        # Target frequency characteristics
        target_freq_ratio = frequency_stats['mean_freq_ratio']
        
        # Enhance frequency components
        h, w = image.shape
        center_h, center_w = h // 2, w // 2
        
        # Create frequency masks
        y, x = np.ogrid[:h, :w]
        distance = np.sqrt((y - center_h)**2 + (x - center_w)**2)
        
        # Low frequency enhancement
        low_freq_mask = distance < min(h, w) // 8
        high_freq_mask = ~low_freq_mask
        
        # Adjust frequency components
        enhanced_magnitude = fft_magnitude.copy()
        enhanced_magnitude[low_freq_mask] *= target_freq_ratio
        enhanced_magnitude[high_freq_mask] *= (1.0 / target_freq_ratio)
        
        # Reconstruct
        enhanced_fft = enhanced_magnitude * np.exp(1j * fft_phase)
        enhanced_image = np.real(np.fft.ifft2(enhanced_fft))
        
        return np.clip(enhanced_image, 0, 255).astype(np.uint8)
    
    def apply_morphological_enhancement(self, image, morphology_stats):
        """Apply morphological enhancement"""
        # Binary morphology based on target statistics
        threshold = np.mean(image)
        binary = image > threshold
        
        # Apply opening and closing based on target ratios
        target_opening_ratio = morphology_stats['mean_opening_ratio']
        target_closing_ratio = morphology_stats['mean_closing_ratio']
        
        # Morphological operations
        opened = morphology.opening(binary, morphology.disk(2))
        closed = morphology.closing(binary, morphology.disk(2))
        
        # Blend based on target ratios
        enhanced = image.copy().astype(float)
        
        # Enhance regions based on morphological characteristics
        enhanced[opened] *= (1.0 + target_opening_ratio)
        enhanced[closed] *= (1.0 + target_closing_ratio)
        
        return np.clip(enhanced, 0, 255).astype(np.uint8)
    
    def transfer_style(self, image):
        """Apply complete BUS-UCLM style transfer to image"""
        # Start with original image
        styled = image.copy()
        
        # 1. Histogram matching
        reference_hist = self.style_stats['histogram']['mean_histogram']
        styled = self.apply_histogram_matching(styled, reference_hist)
        
        # 2. Texture enhancement
        styled = self.apply_texture_enhancement(styled, self.style_stats['texture'])
        
        # 3. Frequency domain enhancement
        styled = self.apply_frequency_enhancement(styled, self.style_stats['frequency'])
        
        # 4. Morphological enhancement
        styled = self.apply_morphological_enhancement(styled, self.style_stats['morphology'])
        
        # 5. Final intensity adjustment
        target_mean = self.style_stats['histogram']['mean_intensity']
        target_std = self.style_stats['histogram']['std_intensity']
        
        current_mean = np.mean(styled)
        current_std = np.std(styled)
        
        if current_std > 0:
            styled = (styled - current_mean) / current_std * target_std + target_mean
        
        return np.clip(styled, 0, 255).astype(np.uint8)

def create_reverse_combined_dataset(bus_uclm_path, busi_path, styled_busi_path, output_path):
    """Create combined dataset with original BUS-UCLM + styled BUSI"""
    print("📊 Creating reverse combined dataset...")
    
    combined_data = []
    
    # 1. Add all BUS-UCLM data (train + val + test)
    for split in ['train', 'val', 'test']:
        csv_path = os.path.join(bus_uclm_path, f'{split}_frame.csv')
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            for idx, row in df.iterrows():
                image_entry = row['image_path']
                mask_entry = row['mask_path']
                
                if ' ' in image_entry:
                    class_name, filename = image_entry.split(' ', 1)
                else:
                    class_name, filename = 'unknown', image_entry
                
                combined_data.append({
                    'image_path': f"{class_name} {filename}",
                    'mask_path': f"{class_name} {filename}",
                    'source': 'BUS-UCLM-original',
                    'augmentation_type': 'original',
                    'source_client': 'BUS-UCLM'
                })
    
    print(f"  ✅ Added {len(combined_data)} original BUS-UCLM samples")
    
    # 2. Add styled BUSI data
    styled_csv_path = os.path.join(styled_busi_path, 'styled_dataset.csv')
    if os.path.exists(styled_csv_path):
        try:
            styled_df = pd.read_csv(styled_csv_path)
            if len(styled_df) > 0:
                for idx, row in styled_df.iterrows():
                    combined_data.append({
                        'image_path': row['image_path'],
                        'mask_path': row['mask_path'],
                        'source': 'BUSI-styled-to-BUS-UCLM',
                        'augmentation_type': 'styled',
                        'source_client': 'BUSI'
                    })
                
                print(f"  ✅ Added {len(styled_df)} styled BUSI samples")
            else:
                print("  ⚠️  No styled BUSI samples found (empty dataset)")
        except Exception as e:
            print(f"  ⚠️  Error loading styled dataset: {e}")
            print("  ⚠️  Continuing with BUS-UCLM data only")
    
    # 3. Create combined DataFrame and shuffle
    combined_df = pd.DataFrame(combined_data)
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # 4. Save combined dataset
    os.makedirs(output_path, exist_ok=True)
    combined_csv_path = os.path.join(output_path, 'reverse_combined_train.csv')
    combined_df.to_csv(combined_csv_path, index=False)
    
    print(f"  💾 Combined dataset saved to {combined_csv_path}")
    print(f"  📊 Total samples: {len(combined_df)}")
    print(f"    - Original BUS-UCLM: {len(combined_df[combined_df['augmentation_type'] == 'original'])}")
    print(f"    - Styled BUSI: {len(combined_df[combined_df['augmentation_type'] == 'styled'])}")
    
    return combined_csv_path

def process_busi_with_bus_uclm_style(busi_path, style_stats, output_path, batch_size=10):
    """Process BUSI images with BUS-UCLM style transfer"""
    print("🎨 Applying BUS-UCLM style to BUSI images...")
    
    # Initialize style transfer
    style_transfer = BUSUCLMStyleTransfer(style_stats)
    
    # Create output directories
    os.makedirs(output_path, exist_ok=True)
    for class_name in ['benign', 'malignant']:
        os.makedirs(os.path.join(output_path, class_name, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_path, class_name, 'masks'), exist_ok=True)
    
    # Process BUSI training data
    train_csv = os.path.join(busi_path, 'train_frame.csv')
    df = pd.read_csv(train_csv)
    
    styled_data = []
    
    for i in tqdm(range(0, len(df), batch_size), desc="Processing BUSI batches"):
        batch = df.iloc[i:i+batch_size]
        
        for idx, row in batch.iterrows():
            try:
                # Parse BUSI format
                image_filename = row['image_path']
                mask_filename = row['mask_path']
                
                # Extract class name
                class_name = 'benign' if 'benign' in image_filename else 'malignant'
                
                # Load BUSI image - BUSI has different structure
                # BUSI structure: benign/image/filename and benign/mask/filename
                image_path = os.path.join(busi_path, class_name, 'image', image_filename)  
                mask_path = os.path.join(busi_path, class_name, 'mask', mask_filename)
                
                if os.path.exists(image_path) and os.path.exists(mask_path):
                    # Load and process image
                    image = Image.open(image_path).convert('L')
                    mask = Image.open(mask_path).convert('L')
                    
                    # Resize to match BUS-UCLM size (approximately 856x606)
                    image = image.resize((856, 606), Image.LANCZOS)
                    mask = mask.resize((856, 606), Image.LANCZOS)
                    
                    # Apply style transfer
                    image_array = np.array(image)
                    styled_array = style_transfer.transfer_style(image_array)
                    styled_image = Image.fromarray(styled_array)
                    
                    # Generate output filename
                    base_name = os.path.splitext(image_filename)[0]
                    styled_filename = f"styled_{base_name}.png"
                    styled_mask_filename = f"styled_{base_name}_mask.png"
                    
                    # Save styled image and mask
                    styled_image_path = os.path.join(output_path, class_name, 'images', styled_filename)
                    styled_mask_path = os.path.join(output_path, class_name, 'masks', styled_mask_filename)
                    
                    styled_image.save(styled_image_path)
                    mask.save(styled_mask_path)
                    
                    # Add to dataset
                    styled_data.append({
                        'image_path': f"{class_name} {styled_filename}",
                        'mask_path': f"{class_name} {styled_mask_filename}",
                        'original_source': 'BUSI',
                        'style_source': 'BUS-UCLM'
                    })
            
            except Exception as e:
                print(f"⚠️  Error processing {image_filename}: {e}")
                continue
        
        # Memory cleanup
        gc.collect()
    
    # Save styled dataset CSV
    styled_df = pd.DataFrame(styled_data)
    styled_csv_path = os.path.join(output_path, 'styled_dataset.csv')
    styled_df.to_csv(styled_csv_path, index=False)
    
    print(f"  ✅ Styled {len(styled_df)} BUSI images with BUS-UCLM style")
    print(f"  💾 Styled dataset saved to {styled_csv_path}")
    
    return styled_csv_path

def main():
    parser = argparse.ArgumentParser(description="Reverse Style Transfer: BUS-UCLM → BUSI")
    parser.add_argument('--bus-uclm-path', type=str, default='dataset/BioMedicalDataset/BUS-UCLM',
                        help='Path to BUS-UCLM dataset')
    parser.add_argument('--busi-path', type=str, default='dataset/BioMedicalDataset/BUSI',
                        help='Path to BUSI dataset')
    parser.add_argument('--output-dir', type=str, default='dataset/BioMedicalDataset/Reverse-BUS-UCLM-Style',
                        help='Output directory for results')
    parser.add_argument('--batch-size', type=int, default=10, help='Batch size for processing')
    
    args = parser.parse_args()
    
    print("🚀 Starting Reverse Style Transfer Pipeline")
    print("=" * 60)
    print(f"📁 BUS-UCLM Dataset: {args.bus_uclm_path}")
    print(f"📁 BUSI Dataset: {args.busi_path}")
    print(f"📁 Output Directory: {args.output_dir}")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Step 1: Extract BUS-UCLM style statistics
    print("\n🎯 Step 1: Extracting BUS-UCLM style statistics...")
    style_extractor = BUSUCLMStyleExtractor(args.bus_uclm_path)
    style_stats = style_extractor.extract_dataset_style('all')
    
    # Save style statistics
    style_stats_path = os.path.join(args.output_dir, 'bus_uclm_style_stats.json')
    style_extractor.save_style_stats(style_stats_path)
    
    # Step 2: Apply BUS-UCLM style to BUSI images
    print("\n🎨 Step 2: Applying BUS-UCLM style to BUSI images...")
    styled_busi_path = os.path.join(args.output_dir, 'styled_busi')
    styled_csv_path = process_busi_with_bus_uclm_style(
        args.busi_path, style_stats, styled_busi_path, args.batch_size
    )
    
    # Step 3: Create reverse combined dataset
    print("\n📊 Step 3: Creating reverse combined dataset...")
    combined_csv_path = create_reverse_combined_dataset(
        args.bus_uclm_path, args.busi_path, styled_busi_path, args.output_dir
    )
    
    # Step 4: Generate training command
    print("\n🏃 Step 4: Training command generated")
    print("=" * 60)
    print("To train the model with reverse style transfer, use:")
    print(f"python IS2D_main.py \\")
    print(f"  --train_data_type BUS-UCLM \\")
    print(f"  --test_data_type BUS-UCLM \\")
    print(f"  --ccst_augmented_path {args.output_dir} \\")
    print(f"  --train \\")
    print(f"  --final_epoch 100")
    print("=" * 60)
    
    print("\n✅ Reverse Style Transfer Pipeline Complete!")
    print(f"📈 Combined dataset: {combined_csv_path}")
    print(f"📊 Ready to train on BUS-UCLM-style augmented data and test on BUS-UCLM")

if __name__ == "__main__":
    main() 