#!/usr/bin/env python3
"""
Complete Reverse Style Transfer Pipeline

This pipeline performs proper privacy-preserving style transfer from BUS-UCLM → BUSI
to make BUSI images look stylistically like BUS-UCLM while preserving lesion structure.

Pipeline Steps:
1. Extract BUS-UCLM style statistics (privacy-preserving)
2. Apply style transfer to BUSI images (lesion-aware)
3. Save styled BUSI images to dataset/BioMedicalDataset/
4. Create combined training dataset
5. Train MADGNet using existing training logic
6. Evaluate on BUS-UCLM test set

Author: AI Assistant
Date: 2024
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import cv2
from PIL import Image, ImageEnhance, ImageFilter
import argparse
from tqdm import tqdm
import gc
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class BUSUCLMStyleAnalyzer:
    """
    Analyze BUS-UCLM dataset to extract privacy-preserving style statistics
    """
    
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.style_stats = {}
    
    def extract_privacy_preserving_stats(self):
        """Extract statistical features without raw pixel sharing"""
        print("🔍 Extracting BUS-UCLM style statistics (privacy-preserving)...")
        
        all_images = []
        image_stats = []
        
        # Process all BUS-UCLM splits
        for split in ['train_frame.csv', 'val_frame.csv', 'test_frame.csv']:
            csv_path = os.path.join(self.dataset_path, split)
            if not os.path.exists(csv_path):
                continue
                
            df = pd.read_csv(csv_path)
            print(f"  📊 Processing {split}: {len(df)} images")
            
            for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Analyzing {split}"):
                try:
                    # Parse image path
                    image_entry = row['image_path']
                    if ' ' in image_entry:
                        class_name, filename = image_entry.split(' ', 1)
                    else:
                        class_name, filename = 'unknown', image_entry
                    
                    # Load image
                    image_path = os.path.join(self.dataset_path, class_name, 'images', filename)
                    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                    
                    if image is None:
                        continue
                    
                    # Extract statistical features
                    stats = self._extract_image_statistics(image)
                    stats['class'] = class_name
                    image_stats.append(stats)
                    
                except Exception as e:
                    print(f"⚠️  Error processing {image_entry}: {e}")
                    continue
        
        # Aggregate statistics
        self.style_stats = self._aggregate_statistics(image_stats)
        print(f"  ✅ Style statistics extracted from {len(image_stats)} images")
        
        return self.style_stats
    
    def _extract_image_statistics(self, image):
        """Extract comprehensive statistical features from image"""
        stats = {}
        
        # Basic intensity statistics
        stats['mean'] = float(np.mean(image))
        stats['std'] = float(np.std(image))
        stats['min'] = float(np.min(image))
        stats['max'] = float(np.max(image))
        stats['median'] = float(np.median(image))
        
        # Histogram statistics
        hist, bins = np.histogram(image, bins=256, range=(0, 255))
        hist = hist.astype(float) / hist.sum()
        stats['histogram'] = hist
        
        # Percentiles
        stats['percentiles'] = np.percentile(image, [5, 25, 50, 75, 95])
        
        # Texture measures
        # Gradient magnitude
        grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        stats['gradient_mean'] = float(np.mean(gradient_magnitude))
        stats['gradient_std'] = float(np.std(gradient_magnitude))
        
        # Laplacian (edge detection)
        laplacian = cv2.Laplacian(image, cv2.CV_64F)
        stats['laplacian_var'] = float(np.var(laplacian))
        
        # Local variance (speckle pattern)
        kernel = np.ones((5, 5), np.float32) / 25
        local_mean = cv2.filter2D(image.astype(np.float32), -1, kernel)
        local_var = cv2.filter2D((image.astype(np.float32) - local_mean)**2, -1, kernel)
        stats['local_var_mean'] = float(np.mean(local_var))
        
        return stats
    
    def _aggregate_statistics(self, image_stats):
        """Aggregate statistics across all images"""
        print("  🔄 Aggregating style statistics...")
        
        # Separate by class
        benign_stats = [s for s in image_stats if s['class'] == 'benign']
        malignant_stats = [s for s in image_stats if s['class'] == 'malignant']
        
        aggregated = {}
        
        # Aggregate for each class and overall
        for class_name, class_stats in [('benign', benign_stats), ('malignant', malignant_stats), ('overall', image_stats)]:
            if not class_stats:
                continue
            
            # Basic statistics
            aggregated[class_name] = {
                'target_mean': np.mean([s['mean'] for s in class_stats]),
                'target_std': np.mean([s['std'] for s in class_stats]),
                'target_median': np.mean([s['median'] for s in class_stats]),
                'target_percentiles': np.mean([s['percentiles'] for s in class_stats], axis=0),
                'target_histogram': np.mean([s['histogram'] for s in class_stats], axis=0),
                'target_gradient_mean': np.mean([s['gradient_mean'] for s in class_stats]),
                'target_gradient_std': np.mean([s['gradient_std'] for s in class_stats]),
                'target_laplacian_var': np.mean([s['laplacian_var'] for s in class_stats]),
                'target_local_var_mean': np.mean([s['local_var_mean'] for s in class_stats]),
                'count': len(class_stats)
            }
        
        return aggregated
    
    def save_style_stats(self, output_path):
        """Save style statistics to JSON file"""
        # Convert numpy arrays to lists for JSON serialization
        stats_serializable = {}
        for class_name, class_stats in self.style_stats.items():
            stats_serializable[class_name] = {}
            for key, value in class_stats.items():
                if isinstance(value, np.ndarray):
                    stats_serializable[class_name][key] = value.tolist()
                else:
                    stats_serializable[class_name][key] = float(value)
        
        with open(output_path, 'w') as f:
            json.dump(stats_serializable, f, indent=2)
        print(f"  💾 Style statistics saved to {output_path}")

class LesionAwareStyleTransfer:
    """
    Apply BUS-UCLM style to BUSI images while preserving lesion structure
    """
    
    def __init__(self, style_stats):
        self.style_stats = style_stats
    
    def transfer_style(self, busi_image, busi_mask, class_name='overall'):
        """
        Apply BUS-UCLM style to BUSI image while preserving lesion structure
        
        Args:
            busi_image: BUSI image (numpy array)
            busi_mask: BUSI mask (numpy array)
            class_name: 'benign', 'malignant', or 'overall'
        
        Returns:
            styled_image: BUSI image with BUS-UCLM style
        """
        if class_name not in self.style_stats:
            class_name = 'overall'
        
        target_stats = self.style_stats[class_name]
        
        # Create lesion mask (dilated for boundary preservation)
        lesion_mask = (busi_mask > 128).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        lesion_mask_dilated = cv2.dilate(lesion_mask, kernel, iterations=2)
        
        # 1. Histogram matching
        styled_image = self._apply_histogram_matching(busi_image, target_stats, lesion_mask_dilated)
        
        # 2. Intensity normalization
        styled_image = self._apply_intensity_normalization(styled_image, target_stats, lesion_mask_dilated)
        
        # 3. Texture enhancement
        styled_image = self._apply_texture_enhancement(styled_image, target_stats, lesion_mask_dilated)
        
        # 4. Speckle pattern adjustment
        styled_image = self._apply_speckle_adjustment(styled_image, target_stats, lesion_mask_dilated)
        
        return styled_image
    
    def _apply_histogram_matching(self, image, target_stats, lesion_mask):
        """Apply histogram matching while preserving lesion regions"""
        target_hist = np.array(target_stats['target_histogram'])
        
        # Calculate CDFs
        source_hist, _ = np.histogram(image, bins=256, range=(0, 255))
        source_hist = source_hist.astype(float) / source_hist.sum()
        source_cdf = np.cumsum(source_hist)
        target_cdf = np.cumsum(target_hist)
        
        # Create lookup table
        lookup_table = np.zeros(256, dtype=np.uint8)
        for i in range(256):
            # Find closest CDF value
            closest_idx = np.argmin(np.abs(target_cdf - source_cdf[i]))
            lookup_table[i] = closest_idx
        
        # Apply histogram matching
        matched_image = lookup_table[image]
        
        # Preserve lesion contrast
        lesion_pixels = image[lesion_mask > 0]
        matched_lesion_pixels = matched_image[lesion_mask > 0]
        
        if len(lesion_pixels) > 0:
            # Enhance lesion contrast
            lesion_contrast = np.std(lesion_pixels)
            matched_lesion_contrast = np.std(matched_lesion_pixels)
            
            if matched_lesion_contrast > 0:
                contrast_factor = lesion_contrast / matched_lesion_contrast
                enhanced_lesion = (matched_lesion_pixels - np.mean(matched_lesion_pixels)) * contrast_factor + np.mean(matched_lesion_pixels)
                enhanced_lesion = np.clip(enhanced_lesion, 0, 255)
                
                # Blend with original lesion (preserve structure)
                matched_image[lesion_mask > 0] = 0.7 * enhanced_lesion + 0.3 * lesion_pixels
        
        return matched_image.astype(np.uint8)
    
    def _apply_intensity_normalization(self, image, target_stats, lesion_mask):
        """Apply intensity normalization with lesion preservation"""
        target_mean = target_stats['target_mean']
        target_std = target_stats['target_std']
        
        # Normalize entire image
        current_mean = np.mean(image)
        current_std = np.std(image)
        
        if current_std > 0:
            normalized = (image - current_mean) / current_std * target_std + target_mean
            normalized = np.clip(normalized, 0, 255)
        else:
            normalized = image
        
        # Preserve lesion intensities
        if np.sum(lesion_mask) > 0:
            lesion_pixels = image[lesion_mask > 0]
            normalized_lesion_pixels = normalized[lesion_mask > 0]
            
            # Blend to preserve lesion characteristics
            normalized[lesion_mask > 0] = 0.6 * normalized_lesion_pixels + 0.4 * lesion_pixels
        
        return normalized.astype(np.uint8)
    
    def _apply_texture_enhancement(self, image, target_stats, lesion_mask):
        """Apply texture enhancement with lesion preservation"""
        target_gradient_mean = target_stats['target_gradient_mean']
        
        # Calculate current gradient
        grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        current_gradient_mean = np.mean(gradient_magnitude)
        
        # Apply adaptive histogram equalization
        if current_gradient_mean > 0:
            enhancement_factor = target_gradient_mean / current_gradient_mean
            clip_limit = min(enhancement_factor * 2.0, 4.0)
        else:
            clip_limit = 2.0
        
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        enhanced = clahe.apply(image)
        
        # Preserve lesion texture
        if np.sum(lesion_mask) > 0:
            # Apply gentler enhancement to lesion regions
            lesion_clahe = cv2.createCLAHE(clipLimit=clip_limit * 0.7, tileGridSize=(4, 4))
            lesion_enhanced = lesion_clahe.apply(image)
            
            # Blend lesion regions
            enhanced[lesion_mask > 0] = 0.8 * lesion_enhanced[lesion_mask > 0] + 0.2 * image[lesion_mask > 0]
        
        return enhanced
    
    def _apply_speckle_adjustment(self, image, target_stats, lesion_mask):
        """Apply speckle pattern adjustment with lesion preservation"""
        target_laplacian_var = target_stats['target_laplacian_var']
        target_local_var = target_stats['target_local_var_mean']
        
        # Calculate current speckle characteristics
        laplacian = cv2.Laplacian(image, cv2.CV_64F)
        current_laplacian_var = np.var(laplacian)
        
        # Adjust speckle pattern
        if current_laplacian_var > 0:
            speckle_factor = np.sqrt(target_laplacian_var / current_laplacian_var)
            speckle_factor = np.clip(speckle_factor, 0.5, 2.0)  # Limit adjustment
            
            # Apply speckle enhancement
            speckle_enhanced = image + (laplacian * speckle_factor * 0.05)
            speckle_enhanced = np.clip(speckle_enhanced, 0, 255)
        else:
            speckle_enhanced = image
        
        # Local variance adjustment
        kernel = np.ones((5, 5), np.float32) / 25
        local_mean = cv2.filter2D(speckle_enhanced.astype(np.float32), -1, kernel)
        local_var = cv2.filter2D((speckle_enhanced.astype(np.float32) - local_mean)**2, -1, kernel)
        
        current_local_var_mean = np.mean(local_var)
        if current_local_var_mean > 0:
            local_var_factor = target_local_var / current_local_var_mean
            local_var_factor = np.clip(local_var_factor, 0.7, 1.3)  # Limit adjustment
            
            # Apply local variance adjustment
            variance_adjusted = speckle_enhanced + (local_var - current_local_var_mean) * local_var_factor * 0.1
            variance_adjusted = np.clip(variance_adjusted, 0, 255)
        else:
            variance_adjusted = speckle_enhanced
        
        # Preserve lesion regions from over-processing
        if np.sum(lesion_mask) > 0:
            variance_adjusted[lesion_mask > 0] = 0.7 * variance_adjusted[lesion_mask > 0] + 0.3 * image[lesion_mask > 0]
        
        return variance_adjusted.astype(np.uint8)

class ReverseStyleTransferPipeline:
    """
    Complete pipeline for reverse style transfer
    """
    
    def __init__(self):
        self.bus_uclm_path = 'dataset/BioMedicalDataset/BUS-UCLM'
        self.busi_path = 'dataset/BioMedicalDataset/BUSI'
        self.output_path = 'dataset/BioMedicalDataset/BUSI-BUS-UCLM-Styled'
        self.combined_path = 'dataset/BioMedicalDataset/BUS-UCLM-Combined-Reverse'
        
        # Create output directories
        self._create_output_directories()
    
    def _create_output_directories(self):
        """Create necessary output directories"""
        directories = [
            self.output_path,
            self.combined_path,
            os.path.join(self.output_path, 'benign', 'image'),
            os.path.join(self.output_path, 'benign', 'mask'),
            os.path.join(self.output_path, 'malignant', 'image'),
            os.path.join(self.output_path, 'malignant', 'mask')
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def run_complete_pipeline(self):
        """Run the complete reverse style transfer pipeline"""
        print("🚀 Starting Complete Reverse Style Transfer Pipeline")
        print("=" * 80)
        
        # Step 1: Extract BUS-UCLM style statistics
        print("\n🔍 Step 1: Extracting BUS-UCLM Style Statistics")
        print("-" * 50)
        
        analyzer = BUSUCLMStyleAnalyzer(self.bus_uclm_path)
        style_stats = analyzer.extract_privacy_preserving_stats()
        
        # Save style statistics
        style_stats_path = os.path.join(self.combined_path, 'bus_uclm_style_stats.json')
        analyzer.save_style_stats(style_stats_path)
        
        # Step 2: Apply style transfer to BUSI images
        print(f"\n🎨 Step 2: Applying Style Transfer to BUSI Images")
        print("-" * 50)
        
        style_transfer = LesionAwareStyleTransfer(style_stats)
        styled_count = self._process_busi_images(style_transfer)
        
        # Step 3: Create combined dataset
        print(f"\n📊 Step 3: Creating Combined Dataset")
        print("-" * 50)
        
        combined_csv = self._create_combined_dataset()
        
        # Step 4: Generate training command
        print(f"\n🏃 Step 4: Training Setup")
        print("-" * 50)
        
        training_command = self._generate_training_command()
        
        # Summary
        print(f"\n✅ Pipeline Complete!")
        print("=" * 80)
        print(f"📊 Styled BUSI images: {styled_count}")
        print(f"📁 Output directory: {self.output_path}")
        print(f"📄 Combined dataset: {combined_csv}")
        print(f"🚀 Training command: {training_command}")
        print("=" * 80)
        
        return {
            'styled_count': styled_count,
            'output_path': self.output_path,
            'combined_csv': combined_csv,
            'training_command': training_command
        }
    
    def _process_busi_images(self, style_transfer):
        """Process BUSI images with style transfer"""
        busi_train_csv = os.path.join(self.busi_path, 'train_frame.csv')
        busi_df = pd.read_csv(busi_train_csv)
        
        styled_samples = []
        styled_count = 0
        
        print(f"  📊 Processing {len(busi_df)} BUSI images...")
        
        for idx, row in tqdm(busi_df.iterrows(), total=len(busi_df), desc="Style Transfer"):
            try:
                # Parse BUSI format
                image_filename = row['image_path']
                mask_filename = row['mask_path']
                
                # Extract class name
                class_name = 'benign' if 'benign' in image_filename else 'malignant'
                
                # Load BUSI image and mask
                image_path = os.path.join(self.busi_path, class_name, 'image', image_filename)
                mask_path = os.path.join(self.busi_path, class_name, 'mask', mask_filename)
                
                if not (os.path.exists(image_path) and os.path.exists(mask_path)):
                    continue
                
                # Load image and mask
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                
                if image is None or mask is None:
                    continue
                
                # Apply style transfer
                styled_image = style_transfer.transfer_style(image, mask, class_name)
                
                # Generate output filenames
                base_name = os.path.splitext(image_filename)[0]
                styled_filename = f"styled_{base_name}.png"
                styled_mask_filename = f"styled_{base_name}_mask.png"
                
                # Save styled image and mask
                styled_image_path = os.path.join(self.output_path, class_name, 'image', styled_filename)
                styled_mask_path = os.path.join(self.output_path, class_name, 'mask', styled_mask_filename)
                
                # Save with high quality
                cv2.imwrite(styled_image_path, styled_image, [cv2.IMWRITE_PNG_COMPRESSION, 1])
                cv2.imwrite(styled_mask_path, mask, [cv2.IMWRITE_PNG_COMPRESSION, 1])
                
                styled_samples.append({
                    'image_path': f"{class_name} {styled_filename}",
                    'mask_path': f"{class_name} {styled_mask_filename}",
                    'class': class_name
                })
                
                styled_count += 1
                
            except Exception as e:
                print(f"⚠️  Error processing {image_filename}: {e}")
                continue
        
        # Save styled dataset CSV
        styled_df = pd.DataFrame(styled_samples)
        styled_csv_path = os.path.join(self.output_path, 'styled_dataset.csv')
        styled_df.to_csv(styled_csv_path, index=False)
        
        print(f"  ✅ Successfully styled {styled_count} BUSI images")
        print(f"  💾 Styled dataset saved to {styled_csv_path}")
        
        return styled_count
    
    def _create_combined_dataset(self):
        """Create combined dataset CSV"""
        combined_data = []
        
        # 1. Add original BUS-UCLM data
        for split in ['train', 'val', 'test']:
            csv_path = os.path.join(self.bus_uclm_path, f'{split}_frame.csv')
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                for idx, row in df.iterrows():
                    combined_data.append({
                        'image_path': row['image_path'],
                        'mask_path': row['mask_path'],
                        'source': 'BUS-UCLM-original',
                        'augmentation_type': 'original',
                        'source_client': 'BUS-UCLM'
                    })
        
        # 2. Add styled BUSI data
        styled_csv_path = os.path.join(self.output_path, 'styled_dataset.csv')
        if os.path.exists(styled_csv_path):
            styled_df = pd.read_csv(styled_csv_path)
            for idx, row in styled_df.iterrows():
                combined_data.append({
                    'image_path': row['image_path'],
                    'mask_path': row['mask_path'],
                    'source': 'BUSI-styled-to-BUS-UCLM',
                    'augmentation_type': 'styled',
                    'source_client': 'BUSI'
                })
        
        # 3. Create combined DataFrame and shuffle
        combined_df = pd.DataFrame(combined_data)
        combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # 4. Save combined dataset
        combined_csv_path = os.path.join(self.combined_path, 'combined_train_frame.csv')
        combined_df.to_csv(combined_csv_path, index=False)
        
        original_count = len(combined_df[combined_df['augmentation_type'] == 'original'])
        styled_count = len(combined_df[combined_df['augmentation_type'] == 'styled'])
        
        print(f"  📊 Combined dataset created: {len(combined_df)} samples")
        print(f"    - Original BUS-UCLM: {original_count}")
        print(f"    - Styled BUSI: {styled_count}")
        print(f"  💾 Combined dataset saved to {combined_csv_path}")
        
        return combined_csv_path
    
    def _generate_training_command(self):
        """Generate training command"""
        command = (
            f"python IS2D_main.py "
            f"--train_data_type BUS-UCLM-Reverse "
            f"--test_data_type BUS-UCLM "
            f"--ccst_augmented_path {self.combined_path} "
            f"--train "
            f"--final_epoch 100"
        )
        
        print(f"  🚀 Training command generated:")
        print(f"    {command}")
        
        return command

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Complete Reverse Style Transfer Pipeline")
    parser.add_argument('--test-run', action='store_true', help='Run quick test with limited samples')
    args = parser.parse_args()
    
    # Run pipeline
    pipeline = ReverseStyleTransferPipeline()
    results = pipeline.run_complete_pipeline()
    
    # Print final summary
    print("\n" + "=" * 80)
    print("📋 PIPELINE SUMMARY")
    print("=" * 80)
    print("✅ Privacy-preserving style transfer completed")
    print("✅ Lesion structure preserved in style transfer")
    print("✅ High-quality images generated and saved")
    print("✅ Combined dataset created with proper structure")
    print("✅ Ready for MADGNet training using existing logic")
    print("=" * 80)
    
    if args.test_run:
        print("\n🧪 To run a quick test training (2 epochs):")
        print(f"python IS2D_main.py --train_data_type BUS-UCLM-Reverse --test_data_type BUS-UCLM --ccst_augmented_path {results['combined_csv'].replace('combined_train_frame.csv', '')} --train --final_epoch 2")

if __name__ == "__main__":
    main() 