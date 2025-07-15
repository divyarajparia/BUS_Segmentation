#!/usr/bin/env python3
"""
Improved Style Transfer Pipeline
===============================
Enhanced version addressing the quality issues found in the analysis.
"""

import os
import cv2
import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import MinMaxScaler
from skimage import exposure, filters, feature, restoration
from skimage.morphology import dilation, disk
import matplotlib.pyplot as plt

class EnhancedStyleTransfer:
    """Improved style transfer with better quality and medical image preservation."""
    
    def __init__(self):
        self.style_stats = None
        self.device_map = {
            'histogram_levels': 256,
            'edge_threshold': 0.1,
            'lesion_protection_factor': 0.3,
            'intensity_strength': 0.8,
            'texture_strength': 0.6,
            'contrast_strength': 0.7
        }
    
    def extract_enhanced_style_stats(self, images: List[np.ndarray], masks: List[np.ndarray]) -> Dict:
        """Extract comprehensive style statistics with multi-scale analysis."""
        print("🔍 Extracting enhanced style statistics...")
        
        all_stats = {
            'global': {},
            'local': {},
            'texture': {},
            'edge': {},
            'multi_scale': {}
        }
        
        # Process images in batches to avoid memory issues
        batch_size = 50
        global_stats_accum = {'mean': [], 'std': [], 'percentiles': [], 'histograms': []}
        edge_stats_accum = {'mean': [], 'std': [], 'percentiles': [], 'histograms': []}
        texture_stats_accum = {'mean': [], 'std': [], 'histograms': []}
        
        for batch_start in range(0, len(images), batch_size):
            batch_end = min(batch_start + batch_size, len(images))
            batch_images = images[batch_start:batch_end]
            batch_masks = masks[batch_start:batch_end]
            
            print(f"   Processing batch {batch_start//batch_size + 1}/{(len(images)-1)//batch_size + 1}...")
            
            # Batch statistics
            batch_pixels = []
            batch_edge_pixels = []
            batch_texture_pixels = []
            
            for img, mask in zip(batch_images, batch_masks):
                batch_pixels.extend(img.flatten())
                
                # Edge statistics
                edges = filters.sobel(img)
                batch_edge_pixels.extend(edges.flatten())
                
                # Texture statistics using Local Binary Pattern
                lbp = feature.local_binary_pattern(img, P=8, R=1, method='uniform')
                batch_texture_pixels.extend(lbp.flatten())
            
            batch_pixels = np.array(batch_pixels)
            batch_edge_pixels = np.array(batch_edge_pixels)
            batch_texture_pixels = np.array(batch_texture_pixels)
            
            # Accumulate statistics
            global_stats_accum['mean'].append(np.mean(batch_pixels))
            global_stats_accum['std'].append(np.std(batch_pixels))
            global_stats_accum['percentiles'].append([np.percentile(batch_pixels, p) for p in [5, 25, 50, 75, 95]])
            hist, _ = np.histogram(batch_pixels, bins=256, range=(0, 255))
            global_stats_accum['histograms'].append(hist)
            
            edge_stats_accum['mean'].append(np.mean(batch_edge_pixels))
            edge_stats_accum['std'].append(np.std(batch_edge_pixels))
            edge_stats_accum['percentiles'].append([np.percentile(batch_edge_pixels, p) for p in [25, 50, 75, 95]])
            edge_hist, _ = np.histogram(batch_edge_pixels, bins=128, range=(0, np.max(batch_edge_pixels)))
            edge_stats_accum['histograms'].append(edge_hist)
            
            texture_stats_accum['mean'].append(np.mean(batch_texture_pixels))
            texture_stats_accum['std'].append(np.std(batch_texture_pixels))
            texture_hist, _ = np.histogram(batch_texture_pixels, bins=59, range=(0, 59))
            texture_stats_accum['histograms'].append(texture_hist)
        
        # Combine batch statistics
        all_pixels = np.array(global_stats_accum['mean'])  # Use means for memory efficiency
        edge_pixels = np.array(edge_stats_accum['mean'])
        texture_pixels = np.array(texture_stats_accum['mean'])
        
        # Combine accumulated statistics
        combined_histogram = np.sum(global_stats_accum['histograms'], axis=0)
        combined_edge_histogram = np.sum(edge_stats_accum['histograms'], axis=0)
        combined_texture_histogram = np.sum(texture_stats_accum['histograms'], axis=0)
        
        # Average percentiles across batches
        avg_percentiles = np.mean(global_stats_accum['percentiles'], axis=0)
        avg_edge_percentiles = np.mean(edge_stats_accum['percentiles'], axis=0)
        
        # Global intensity statistics
        all_stats['global'] = {
            'mean': float(np.mean(global_stats_accum['mean'])),
            'std': float(np.mean(global_stats_accum['std'])),
            'percentiles': [float(p) for p in avg_percentiles],
            'histogram': combined_histogram.tolist(),
            'skewness': float(self._skewness(all_pixels)),
            'kurtosis': float(self._kurtosis(all_pixels))
        }
        
        # Edge statistics
        all_stats['edge'] = {
            'mean': float(np.mean(edge_stats_accum['mean'])),
            'std': float(np.mean(edge_stats_accum['std'])),
            'percentiles': [float(p) for p in avg_edge_percentiles],
            'histogram': combined_edge_histogram.tolist()
        }
        
        # Texture statistics
        all_stats['texture'] = {
            'mean': float(np.mean(texture_stats_accum['mean'])),
            'std': float(np.mean(texture_stats_accum['std'])),
            'histogram': combined_texture_histogram.tolist()
        }
        
        # Multi-scale statistics (different image scales) - memory efficient
        scales = [0.5, 1.0, 2.0]
        scale_stats = {}
        
        for scale in scales:
            scale_means = []
            scale_stds = []
            
            # Process a subset of images for multi-scale analysis
            sample_indices = list(range(0, len(images), max(1, len(images) // 20)))  # Sample every 20th image
            
            for idx in sample_indices:
                img = images[idx]
                if scale != 1.0:
                    h, w = img.shape
                    new_h, new_w = int(h * scale), int(w * scale)
                    scaled_img = cv2.resize(img, (new_w, new_h))
                else:
                    scaled_img = img
                
                scale_means.append(np.mean(scaled_img))
                scale_stds.append(np.std(scaled_img))
            
            overall_mean = np.mean(scale_means)
            overall_std = np.mean(scale_stds)
            
            scale_stats[str(scale)] = {
                'mean': float(overall_mean),
                'std': float(overall_std),
                'contrast': float(overall_std) / (float(overall_mean) + 1e-8)
            }
        
        all_stats['multi_scale'] = scale_stats
        
        print(f"✅ Enhanced style statistics extracted from {len(images)} images")
        return all_stats
    
    def _skewness(self, data: np.ndarray) -> float:
        """Calculate skewness."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 3)
    
    def _kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 4) - 3
    
    def apply_enhanced_style_transfer(self, source_img: np.ndarray, source_mask: np.ndarray, 
                                    target_stats: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """Apply enhanced style transfer with better quality preservation."""
        
        # Step 1: Advanced histogram matching
        styled_img = self._advanced_histogram_matching(source_img, target_stats)
        
        # Step 2: Edge-preserving enhancement
        styled_img = self._edge_preserving_enhancement(source_img, styled_img, source_mask)
        
        # Step 3: Multi-scale texture transfer
        styled_img = self._multi_scale_texture_transfer(styled_img, target_stats)
        
        # Step 4: Lesion-aware intensity adjustment
        styled_img = self._lesion_aware_adjustment(source_img, styled_img, source_mask)
        
        # Step 5: Final quality enhancement
        styled_img = self._final_quality_enhancement(styled_img, target_stats)
        
        # Ensure proper range
        styled_img = np.clip(styled_img, 0, 255).astype(np.uint8)
        
        return styled_img, source_mask
    
    def _advanced_histogram_matching(self, source: np.ndarray, target_stats: Dict) -> np.ndarray:
        """Advanced histogram matching with multi-level processing."""
        
        # Get target histogram and percentiles
        target_hist = np.array(target_stats['global']['histogram'])
        target_percentiles = target_stats['global']['percentiles']
        
        # Normalize target histogram
        target_hist = target_hist / np.sum(target_hist)
        
        # Calculate cumulative distribution functions
        target_cdf = np.cumsum(target_hist)
        
        # Source histogram
        source_hist, _ = np.histogram(source.flatten(), bins=256, range=(0, 255))
        source_hist = source_hist / np.sum(source_hist)
        source_cdf = np.cumsum(source_hist)
        
        # Create mapping using linear interpolation
        mapping = np.zeros(256, dtype=np.uint8)
        for i in range(256):
            # Find closest CDF value in target
            diff = np.abs(target_cdf - source_cdf[i])
            mapping[i] = np.argmin(diff)
        
        # Apply mapping
        matched = mapping[source.astype(np.uint8)]
        
        # Blend with percentile-based matching for better results
        source_percentiles = [np.percentile(source, p) for p in [5, 25, 50, 75, 95]]
        
        # Create percentile-based enhancement
        enhanced = source.copy()
        for i, (src_p, tgt_p) in enumerate(zip(source_percentiles, target_percentiles)):
            if src_p > 0:
                factor = tgt_p / src_p
                mask = (source >= (src_p * 0.9)) & (source <= (src_p * 1.1))
                enhanced[mask] = enhanced[mask] * factor
        
        # Blend histogram matching with percentile matching
        result = 0.6 * matched + 0.4 * enhanced
        
        return result
    
    def _edge_preserving_enhancement(self, original: np.ndarray, styled: np.ndarray, 
                                   mask: np.ndarray) -> np.ndarray:
        """Preserve edges during style transfer."""
        
        # Calculate edge maps
        original_edges = filters.sobel(original)
        styled_edges = filters.sobel(styled)
        
        # Edge preservation factor
        edge_strength = np.maximum(original_edges, styled_edges)
        edge_mask = edge_strength > self.device_map['edge_threshold'] * np.max(edge_strength)
        
        # Preserve original edges in high-edge regions
        preserved = styled.copy()
        preserved[edge_mask] = 0.7 * original[edge_mask] + 0.3 * styled[edge_mask]
        
        # Extra protection for lesion edges
        if mask is not None:
            lesion_mask = mask > 128
            dilated_lesion = dilation(lesion_mask, disk(3))
            lesion_edges = dilated_lesion & edge_mask
            preserved[lesion_edges] = 0.8 * original[lesion_edges] + 0.2 * styled[lesion_edges]
        
        return preserved
    
    def _multi_scale_texture_transfer(self, styled: np.ndarray, target_stats: Dict) -> np.ndarray:
        """Apply multi-scale texture enhancement."""
        
        # Work at different scales
        scales = [0.5, 1.0, 2.0]
        enhanced_scales = []
        
        for scale in scales:
            scale_str = str(scale)
            if scale_str not in target_stats['multi_scale']:
                continue
                
            scale_stats = target_stats['multi_scale'][scale_str]
            
            if scale != 1.0:
                h, w = styled.shape
                new_h, new_w = int(h * scale), int(w * scale)
                scaled = cv2.resize(styled, (new_w, new_h))
            else:
                scaled = styled.copy()
            
            # Adjust contrast at this scale
            target_contrast = scale_stats['contrast']
            current_contrast = np.std(scaled) / (np.mean(scaled) + 1e-8)
            
            if current_contrast > 0:
                contrast_factor = target_contrast / current_contrast
                contrast_factor = np.clip(contrast_factor, 0.5, 2.0)  # Prevent extreme changes
                
                mean_val = np.mean(scaled)
                scaled = (scaled - mean_val) * contrast_factor + mean_val
            
            # Resize back if needed
            if scale != 1.0:
                h, w = styled.shape
                scaled = cv2.resize(scaled, (w, h))
            
            enhanced_scales.append(scaled)
        
        # Combine scales with weights
        if len(enhanced_scales) == 3:
            result = 0.2 * enhanced_scales[0] + 0.6 * enhanced_scales[1] + 0.2 * enhanced_scales[2]
        elif len(enhanced_scales) >= 1:
            result = enhanced_scales[1] if len(enhanced_scales) > 1 else enhanced_scales[0]
        else:
            result = styled
        
        return result
    
    def _lesion_aware_adjustment(self, original: np.ndarray, styled: np.ndarray, 
                               mask: np.ndarray) -> np.ndarray:
        """Apply lesion-aware intensity adjustment."""
        
        if mask is None:
            return styled
        
        lesion_mask = mask > 128
        if not np.any(lesion_mask):
            return styled
        
        # Dilate lesion mask for smoother transition
        dilated_mask = dilation(lesion_mask, disk(5))
        
        # Create transition weights
        transition_weights = np.zeros_like(mask, dtype=np.float32)
        transition_weights[lesion_mask] = self.device_map['lesion_protection_factor']
        transition_weights[dilated_mask & ~lesion_mask] = self.device_map['lesion_protection_factor'] * 0.5
        
        # Apply weighted blending
        result = styled.copy()
        blend_areas = transition_weights > 0
        result[blend_areas] = (1 - transition_weights[blend_areas]) * styled[blend_areas] + \
                              transition_weights[blend_areas] * original[blend_areas]
        
        return result
    
    def _final_quality_enhancement(self, styled: np.ndarray, target_stats: Dict) -> np.ndarray:
        """Final quality enhancement step."""
        
        # Global intensity adjustment
        target_mean = target_stats['global']['mean']
        current_mean = np.mean(styled)
        
        if current_mean > 0:
            intensity_factor = target_mean / current_mean
            intensity_factor = np.clip(intensity_factor, 0.7, 1.3)  # Prevent extreme changes
            styled = styled * intensity_factor
        
        # Gentle denoising while preserving structure
        styled = restoration.denoise_bilateral(styled, sigma_color=0.1, sigma_spatial=2)
        
        # Final contrast adjustment
        target_std = target_stats['global']['std']
        current_std = np.std(styled)
        
        if current_std > 0:
            contrast_factor = target_std / current_std
            contrast_factor = np.clip(contrast_factor, 0.8, 1.2)
            
            mean_val = np.mean(styled)
            styled = (styled - mean_val) * contrast_factor + mean_val
        
        return styled

class ImprovedStyleTransferPipeline:
    """Complete improved style transfer pipeline."""
    
    def __init__(self):
        self.style_transfer = EnhancedStyleTransfer()
        self.busi_path = "dataset/BioMedicalDataset/BUSI"
        self.bus_uclm_path = "dataset/BioMedicalDataset/BUS-UCLM"
        self.output_path = "dataset/BioMedicalDataset/BUSI-BUS-UCLM-Improved"
        
    def run_improved_pipeline(self):
        """Run the complete improved style transfer pipeline."""
        print("🚀 Starting improved reverse style transfer pipeline...")
        
        # Create output directory
        os.makedirs(self.output_path, exist_ok=True)
        os.makedirs(f"{self.output_path}/benign/image", exist_ok=True)
        os.makedirs(f"{self.output_path}/benign/mask", exist_ok=True)
        os.makedirs(f"{self.output_path}/malignant/image", exist_ok=True)
        os.makedirs(f"{self.output_path}/malignant/mask", exist_ok=True)
        
        # Step 1: Extract enhanced BUS-UCLM style
        bus_uclm_stats = self._extract_bus_uclm_style()
        
        # Step 2: Apply improved style transfer to BUSI
        styled_records = self._apply_improved_style_transfer(bus_uclm_stats)
        
        # Step 3: Create dataset CSV
        self._create_improved_dataset_csv(styled_records)
        
        print("✅ Improved style transfer pipeline completed!")
        return len(styled_records)
    
    def _extract_bus_uclm_style(self) -> Dict:
        """Extract comprehensive style statistics from BUS-UCLM."""
        print("📊 Extracting enhanced BUS-UCLM style statistics...")
        
        bus_uclm_train = pd.read_csv(f"{self.bus_uclm_path}/train_frame.csv")
        bus_uclm_val = pd.read_csv(f"{self.bus_uclm_path}/val_frame.csv")
        bus_uclm_test = pd.read_csv(f"{self.bus_uclm_path}/test_frame.csv")
        
        all_data = pd.concat([bus_uclm_train, bus_uclm_val, bus_uclm_test], ignore_index=True)
        
        images = []
        masks = []
        
        for _, row in all_data.iterrows():
            class_name = 'benign' if 'benign' in row['image_path'] else 'malignant'
            actual_img_filename = row['image_path'].replace('benign ', '').replace('malignant ', '')
            actual_mask_filename = row['mask_path'].replace('benign ', '').replace('malignant ', '')
            
            img_path = os.path.join(self.bus_uclm_path, class_name, 'images', actual_img_filename)
            mask_path = os.path.join(self.bus_uclm_path, class_name, 'masks', actual_mask_filename)
            
            if os.path.exists(img_path) and os.path.exists(mask_path):
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                
                if img is not None and mask is not None:
                    images.append(img)
                    masks.append(mask)
        
        print(f"📈 Loaded {len(images)} BUS-UCLM images for style analysis")
        
        # Extract enhanced style statistics
        style_stats = self.style_transfer.extract_enhanced_style_stats(images, masks)
        
        # Save style statistics
        with open(f"{self.output_path}/enhanced_bus_uclm_style_stats.json", 'w') as f:
            json.dump(style_stats, f, indent=2)
        
        return style_stats
    
    def _apply_improved_style_transfer(self, target_stats: Dict) -> List[Dict]:
        """Apply improved style transfer to BUSI images."""
        print("🎨 Applying improved style transfer to BUSI images...")
        
        busi_train = pd.read_csv(f"{self.busi_path}/train_frame.csv")
        styled_records = []
        
        for idx, (_, row) in enumerate(busi_train.iterrows()):
            class_name = 'benign' if 'benign' in row['image_path'] else 'malignant'
            
            img_path = os.path.join(self.busi_path, class_name, 'image', row['image_path'])
            mask_path = os.path.join(self.busi_path, class_name, 'mask', row['mask_path'])
            
            if os.path.exists(img_path) and os.path.exists(mask_path):
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                
                if img is not None and mask is not None:
                    # Apply enhanced style transfer
                    styled_img, styled_mask = self.style_transfer.apply_enhanced_style_transfer(
                        img, mask, target_stats
                    )
                    
                    # Generate output filenames
                    base_name = os.path.splitext(row['image_path'])[0]
                    styled_img_name = f"improved_{base_name}.png"
                    styled_mask_name = f"improved_{base_name}_mask.png"
                    
                    # Save styled images
                    img_output_path = os.path.join(self.output_path, class_name, 'image', styled_img_name)
                    mask_output_path = os.path.join(self.output_path, class_name, 'mask', styled_mask_name)
                    
                    cv2.imwrite(img_output_path, styled_img)
                    cv2.imwrite(mask_output_path, styled_mask)
                    
                    styled_records.append({
                        'image_path': f"{class_name} {styled_img_name}",
                        'mask_path': f"{class_name} {styled_mask_name}",
                        'class': class_name,
                        'original_image': row['image_path']
                    })
                    
                    if (idx + 1) % 50 == 0:
                        print(f"   ✅ Processed {idx + 1}/{len(busi_train)} images")
        
        print(f"🎯 Completed improved style transfer: {len(styled_records)} images")
        return styled_records
    
    def _create_improved_dataset_csv(self, styled_records: List[Dict]):
        """Create CSV for the improved styled dataset."""
        styled_df = pd.DataFrame(styled_records)
        csv_path = f"{self.output_path}/improved_styled_dataset.csv"
        styled_df.to_csv(csv_path, index=False)
        
        print(f"📁 Saved improved dataset CSV: {csv_path}")
        print(f"   📊 Total samples: {len(styled_records)}")
        print(f"   📊 Benign: {len(styled_df[styled_df['class'] == 'benign'])}")
        print(f"   📊 Malignant: {len(styled_df[styled_df['class'] == 'malignant'])}")

if __name__ == "__main__":
    pipeline = ImprovedStyleTransferPipeline()
    pipeline.run_improved_pipeline() 