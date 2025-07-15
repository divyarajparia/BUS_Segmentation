#!/usr/bin/env python3
"""
Focused Medical Style Transfer
==============================
Targets specific visual differences between BUSI and BUS-UCLM:
- Texture granularity and patterns
- Noise characteristics
- Local contrast variations
- Equipment-specific artifacts
"""

import os
import cv2
import numpy as np
import pandas as pd
import json
from typing import Dict, List, Tuple
from scipy import ndimage
from skimage import filters, restoration, feature, exposure
import matplotlib.pyplot as plt

class FocusedMedicalStyleTransfer:
    """Focused style transfer targeting key visual differences in medical ultrasound."""
    
    def __init__(self):
        self.patch_size = 64
        self.overlap = 32
    
    def extract_focused_characteristics(self, images: List[np.ndarray]) -> Dict:
        """Extract focused characteristics targeting key visual differences."""
        print("🎯 Extracting focused medical characteristics...")
        
        characteristics = {
            'texture_granularity': {},
            'noise_patterns': {},
            'local_contrast': {},
            'intensity_distributions': {}
        }
        
        # Sample subset for efficiency
        sample_images = images[:50] if len(images) > 50 else images
        
        all_texture_scores = []
        all_noise_levels = []
        all_local_contrasts = []
        all_intensities = []
        
        for i, img in enumerate(sample_images):
            if (i + 1) % 10 == 0:
                print(f"   Processing {i+1}/{len(sample_images)} images...")
            
            # 1. Texture Granularity Analysis
            texture_score = self._analyze_texture_granularity(img)
            all_texture_scores.append(texture_score)
            
            # 2. Noise Pattern Analysis
            noise_level = self._analyze_noise_patterns(img)
            all_noise_levels.append(noise_level)
            
            # 3. Local Contrast Analysis
            local_contrast = self._analyze_local_contrast(img)
            all_local_contrasts.append(local_contrast)
            
            # 4. Intensity Distribution
            all_intensities.extend(img.flatten()[::100])  # Sample pixels
        
        # Aggregate characteristics
        characteristics['texture_granularity'] = {
            'mean_granularity': float(np.mean(all_texture_scores)),
            'std_granularity': float(np.std(all_texture_scores)),
            'high_texture_ratio': float(np.sum(np.array(all_texture_scores) > np.mean(all_texture_scores)) / len(all_texture_scores))
        }
        
        characteristics['noise_patterns'] = {
            'mean_noise_level': float(np.mean(all_noise_levels)),
            'noise_consistency': float(1.0 / (np.std(all_noise_levels) + 1e-8)),
            'base_noise_floor': float(np.percentile(all_noise_levels, 10))
        }
        
        characteristics['local_contrast'] = {
            'mean_contrast': float(np.mean(all_local_contrasts)),
            'contrast_variation': float(np.std(all_local_contrasts)),
            'high_contrast_ratio': float(np.sum(np.array(all_local_contrasts) > np.mean(all_local_contrasts)) / len(all_local_contrasts))
        }
        
        all_intensities = np.array(all_intensities)
        characteristics['intensity_distributions'] = {
            'mean_intensity': float(np.mean(all_intensities)),
            'intensity_range': float(np.percentile(all_intensities, 95) - np.percentile(all_intensities, 5)),
            'intensity_skew': float(self._skewness(all_intensities)),
            'peak_intensity': float(np.percentile(all_intensities, 75))
        }
        
        print(f"✅ Extracted focused characteristics from {len(sample_images)} images")
        return characteristics
    
    def _analyze_texture_granularity(self, image: np.ndarray) -> float:
        """Analyze texture granularity - key difference between BUSI and BUS-UCLM."""
        
        # Use multiple Laplacian operators to detect fine texture
        laplacian_3 = ndimage.laplace(image.astype(np.float32))
        laplacian_5 = ndimage.laplace(ndimage.gaussian_filter(image, 1.0))
        
        # High-frequency texture energy
        texture_energy = np.var(laplacian_3) + 0.5 * np.var(laplacian_5)
        
        # Local texture variation
        local_std = ndimage.generic_filter(image.astype(np.float32), np.std, size=9)
        texture_variation = np.mean(local_std)
        
        # Combined granularity score
        granularity_score = texture_energy * 0.7 + texture_variation * 0.3
        
        return granularity_score
    
    def _analyze_noise_patterns(self, image: np.ndarray) -> float:
        """Analyze noise patterns specific to ultrasound equipment."""
        
        # Estimate noise using wavelet denoising
        try:
            denoised = restoration.denoise_wavelet(image, method='BayesShrink', mode='soft')
            noise_component = image.astype(np.float32) - denoised.astype(np.float32)
            
            # Noise level metric
            noise_level = np.std(noise_component)
            
            # Noise texture (how structured the noise is)
            noise_texture = np.var(ndimage.laplace(noise_component))
            
            # Combined noise pattern score
            pattern_score = noise_level + 0.3 * noise_texture
            
        except:
            # Fallback: simple high-frequency analysis
            high_freq = filters.gaussian(image, sigma=1.0) - filters.gaussian(image, sigma=2.0)
            pattern_score = np.std(high_freq)
        
        return pattern_score
    
    def _analyze_local_contrast(self, image: np.ndarray) -> float:
        """Analyze local contrast characteristics."""
        
        # Multi-scale local contrast
        contrasts = []
        for window_size in [5, 9, 15]:
            # Local mean and std
            local_mean = ndimage.uniform_filter(image.astype(np.float32), size=window_size)
            local_var = ndimage.uniform_filter(image.astype(np.float32)**2, size=window_size) - local_mean**2
            local_std = np.sqrt(np.maximum(local_var, 0))
            
            # Local contrast
            local_contrast = local_std / (local_mean + 1e-8)
            contrasts.append(np.mean(local_contrast[local_contrast < 10]))  # Cap extreme values
        
        return np.mean(contrasts)
    
    def _skewness(self, data: np.ndarray) -> float:
        """Calculate skewness."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 3)
    
    def apply_focused_style_transfer(self, source_img: np.ndarray, source_mask: np.ndarray,
                                   target_chars: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """Apply focused style transfer targeting key visual differences."""
        
        # Step 1: Enhance texture granularity
        enhanced_img = self._enhance_texture_granularity(source_img, target_chars['texture_granularity'])
        
        # Step 2: Adjust noise patterns
        enhanced_img = self._adjust_noise_patterns(enhanced_img, target_chars['noise_patterns'])
        
        # Step 3: Modify local contrast
        enhanced_img = self._modify_local_contrast(enhanced_img, target_chars['local_contrast'])
        
        # Step 4: Adjust intensity distribution
        enhanced_img = self._adjust_intensity_distribution(enhanced_img, target_chars['intensity_distributions'])
        
        # Step 5: Preserve lesion structures
        if source_mask is not None:
            enhanced_img = self._preserve_lesion_structures(source_img, enhanced_img, source_mask)
        
        # Ensure proper range
        enhanced_img = np.clip(enhanced_img, 0, 255).astype(np.uint8)
        
        return enhanced_img, source_mask
    
    def _enhance_texture_granularity(self, image: np.ndarray, texture_chars: Dict) -> np.ndarray:
        """Enhance texture granularity to match target characteristics."""
        
        target_granularity = texture_chars['mean_granularity']
        current_granularity = self._analyze_texture_granularity(image)
        
        if current_granularity <= 0:
            return image
        
        # Calculate enhancement factor
        enhancement_factor = target_granularity / current_granularity
        enhancement_factor = np.clip(enhancement_factor, 0.5, 2.5)
        
        # Apply unsharp masking for texture enhancement
        gaussian_blur = ndimage.gaussian_filter(image.astype(np.float32), sigma=0.8)
        sharpened = image + (enhancement_factor - 1.0) * (image - gaussian_blur)
        
        # Add fine-grained texture noise
        fine_noise = np.random.normal(0, 2.0, image.shape)
        fine_texture = ndimage.gaussian_filter(fine_noise, sigma=0.5)
        
        # Blend texture enhancement
        texture_strength = min(0.3, (enhancement_factor - 1.0) * 0.5)
        result = sharpened + texture_strength * fine_texture
        
        return result
    
    def _adjust_noise_patterns(self, image: np.ndarray, noise_chars: Dict) -> np.ndarray:
        """Adjust noise patterns to match target equipment characteristics."""
        
        target_noise_level = noise_chars['mean_noise_level']
        
        # Estimate current noise
        try:
            denoised = restoration.denoise_wavelet(image, method='BayesShrink', mode='soft')
            current_noise = image.astype(np.float32) - denoised.astype(np.float32)
            current_noise_level = np.std(current_noise)
        except:
            current_noise_level = np.std(image) * 0.1  # Fallback estimate
        
        if current_noise_level <= 0:
            current_noise_level = 1.0
        
        # Adjust noise level
        noise_factor = target_noise_level / current_noise_level
        noise_factor = np.clip(noise_factor, 0.3, 3.0)
        
        # Generate equipment-specific noise
        # Ultrasound has correlated speckle noise
        base_noise = np.random.rayleigh(2.0, image.shape)
        correlated_noise = ndimage.gaussian_filter(base_noise, sigma=1.2)
        
        # Normalize and scale noise
        correlated_noise = (correlated_noise - np.mean(correlated_noise)) / (np.std(correlated_noise) + 1e-8)
        scaled_noise = correlated_noise * target_noise_level * 0.3
        
        # Apply noise adjustment
        result = image.astype(np.float32) + scaled_noise
        
        return result
    
    def _modify_local_contrast(self, image: np.ndarray, contrast_chars: Dict) -> np.ndarray:
        """Modify local contrast characteristics."""
        
        target_contrast = contrast_chars['mean_contrast']
        current_contrast = self._analyze_local_contrast(image)
        
        if current_contrast <= 0:
            return image
        
        # Calculate contrast adjustment factor
        contrast_factor = target_contrast / current_contrast
        contrast_factor = np.clip(contrast_factor, 0.6, 1.8)
        
        # Apply adaptive local contrast enhancement
        # Use CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=contrast_factor*2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(image.astype(np.uint8))
        
        # Blend with original
        blend_factor = abs(contrast_factor - 1.0) * 0.7
        result = (1 - blend_factor) * image + blend_factor * enhanced.astype(np.float32)
        
        return result
    
    def _adjust_intensity_distribution(self, image: np.ndarray, intensity_chars: Dict) -> np.ndarray:
        """Adjust overall intensity distribution."""
        
        target_mean = intensity_chars['mean_intensity']
        target_range = intensity_chars['intensity_range']
        
        current_mean = np.mean(image)
        current_range = np.percentile(image, 95) - np.percentile(image, 5)
        
        # Intensity shift
        if current_mean > 0:
            intensity_shift = target_mean - current_mean
            intensity_shift = np.clip(intensity_shift, -30, 30)  # Limit extreme shifts
        else:
            intensity_shift = 0
        
        # Range adjustment
        if current_range > 0:
            range_factor = target_range / current_range
            range_factor = np.clip(range_factor, 0.7, 1.4)
        else:
            range_factor = 1.0
        
        # Apply adjustments
        adjusted = (image - current_mean) * range_factor + current_mean + intensity_shift
        
        return adjusted
    
    def _preserve_lesion_structures(self, original: np.ndarray, styled: np.ndarray, 
                                  mask: np.ndarray) -> np.ndarray:
        """Preserve lesion structures during style transfer."""
        
        lesion_mask = mask > 128
        if not np.any(lesion_mask):
            return styled
        
        # Create preservation weights
        preservation_strength = 0.4  # Moderate preservation
        
        # Dilate lesion mask for smooth transitions
        from skimage.morphology import dilation, disk
        dilated_mask = dilation(lesion_mask, disk(2))
        
        # Apply weighted blending
        result = styled.copy()
        
        # Strong preservation in lesion core
        result[lesion_mask] = (1 - preservation_strength) * styled[lesion_mask] + preservation_strength * original[lesion_mask]
        
        # Gradual transition around lesion
        transition_area = dilated_mask & ~lesion_mask
        if np.any(transition_area):
            transition_strength = preservation_strength * 0.5
            result[transition_area] = (1 - transition_strength) * styled[transition_area] + transition_strength * original[transition_area]
        
        return result


class FocusedStylePipeline:
    """Complete focused medical style transfer pipeline."""
    
    def __init__(self):
        self.style_transfer = FocusedMedicalStyleTransfer()
        self.busi_path = "dataset/BioMedicalDataset/BUSI"
        self.bus_uclm_path = "dataset/BioMedicalDataset/BUS-UCLM"
        self.output_path = "dataset/BioMedicalDataset/BUSI-BUS-UCLM-Focused"
        
    def run_focused_pipeline(self):
        """Run the focused medical style transfer pipeline."""
        print("🎯 Starting focused medical style transfer pipeline...")
        
        # Create output directory
        os.makedirs(self.output_path, exist_ok=True)
        os.makedirs(f"{self.output_path}/benign/image", exist_ok=True)
        os.makedirs(f"{self.output_path}/benign/mask", exist_ok=True)
        os.makedirs(f"{self.output_path}/malignant/image", exist_ok=True)
        os.makedirs(f"{self.output_path}/malignant/mask", exist_ok=True)
        
        # Step 1: Extract BUS-UCLM characteristics
        bus_uclm_characteristics = self._extract_bus_uclm_characteristics()
        
        # Step 2: Apply focused style transfer to BUSI
        styled_records = self._apply_focused_style_transfer(bus_uclm_characteristics)
        
        # Step 3: Create dataset CSV
        self._create_focused_dataset_csv(styled_records)
        
        print("✅ Focused medical style transfer completed!")
        return len(styled_records)
    
    def _extract_bus_uclm_characteristics(self) -> Dict:
        """Extract BUS-UCLM characteristics."""
        print("📊 Extracting BUS-UCLM characteristics...")
        
        # Load BUS-UCLM data
        train_df = pd.read_csv(f"{self.bus_uclm_path}/train_frame.csv")
        val_df = pd.read_csv(f"{self.bus_uclm_path}/val_frame.csv")
        
        all_data = pd.concat([train_df, val_df], ignore_index=True)
        
        images = []
        
        for _, row in all_data.iterrows():
            class_name = 'benign' if 'benign' in row['image_path'] else 'malignant'
            actual_img_filename = row['image_path'].replace('benign ', '').replace('malignant ', '')
            
            img_path = os.path.join(self.bus_uclm_path, class_name, 'images', actual_img_filename)
            
            if os.path.exists(img_path):
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    images.append(img)
        
        print(f"📈 Loaded {len(images)} BUS-UCLM images")
        
        # Extract characteristics
        characteristics = self.style_transfer.extract_focused_characteristics(images)
        
        # Save characteristics
        with open(f"{self.output_path}/focused_bus_uclm_characteristics.json", 'w') as f:
            json.dump(characteristics, f, indent=2)
        
        return characteristics
    
    def _apply_focused_style_transfer(self, target_characteristics: Dict) -> List[Dict]:
        """Apply focused style transfer to BUSI images."""
        print("🎨 Applying focused style transfer to BUSI...")
        
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
                    # Apply focused style transfer
                    styled_img, styled_mask = self.style_transfer.apply_focused_style_transfer(
                        img, mask, target_characteristics
                    )
                    
                    # Generate output filenames
                    base_name = os.path.splitext(row['image_path'])[0]
                    styled_img_name = f"focused_{base_name}.png"
                    styled_mask_name = f"focused_{base_name}_mask.png"
                    
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
        
        print(f"🎯 Completed focused style transfer: {len(styled_records)} images")
        return styled_records
    
    def _create_focused_dataset_csv(self, styled_records: List[Dict]):
        """Create CSV for the focused styled dataset."""
        styled_df = pd.DataFrame(styled_records)
        csv_path = f"{self.output_path}/focused_styled_dataset.csv"
        styled_df.to_csv(csv_path, index=False)
        
        print(f"📁 Saved focused dataset CSV: {csv_path}")
        print(f"   📊 Total samples: {len(styled_records)}")
        print(f"   📊 Benign: {len(styled_df[styled_df['class'] == 'benign'])}")
        print(f"   📊 Malignant: {len(styled_df[styled_df['class'] == 'malignant'])}")


if __name__ == "__main__":
    pipeline = FocusedStylePipeline()
    pipeline.run_focused_pipeline() 