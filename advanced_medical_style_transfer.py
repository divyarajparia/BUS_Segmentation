#!/usr/bin/env python3
"""
Advanced Medical Ultrasound Style Transfer
==========================================
Focuses on ultrasound-specific characteristics: speckle patterns, local textures, 
equipment artifacts, and medical imaging physics.
"""

import os
import cv2
import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy import ndimage, signal
from scipy.stats import rayleigh
from skimage import filters, restoration, feature, measure, morphology, segmentation
from skimage.util import random_noise
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

class MedicalUltrasoundStyleTransfer:
    """Advanced style transfer specifically designed for medical ultrasound imaging."""
    
    def __init__(self):
        self.target_characteristics = None
        self.patch_size = 32
        self.overlap = 16
        
    def extract_ultrasound_characteristics(self, images: List[np.ndarray], masks: List[np.ndarray]) -> Dict:
        """Extract comprehensive ultrasound-specific characteristics."""
        print("🔬 Extracting medical ultrasound characteristics...")
        
        characteristics = {
            'speckle_patterns': {},
            'local_textures': {},
            'equipment_signatures': {},
            'contrast_characteristics': {},
            'noise_models': {}
        }
        
        # Process in batches for memory efficiency
        batch_size = 20
        
        for batch_start in range(0, len(images), batch_size):
            batch_end = min(batch_start + batch_size, len(images))
            batch_images = images[batch_start:batch_end]
            
            print(f"   Processing batch {batch_start//batch_size + 1}/{(len(images)-1)//batch_size + 1}...")
            
            for img in batch_images:
                # Extract speckle patterns
                speckle_stats = self._analyze_speckle_patterns(img)
                self._accumulate_stats(characteristics['speckle_patterns'], speckle_stats)
                
                # Extract local texture characteristics
                texture_stats = self._analyze_local_textures(img)
                self._accumulate_stats(characteristics['local_textures'], texture_stats)
                
                # Analyze equipment-specific signatures
                equipment_stats = self._analyze_equipment_signatures(img)
                self._accumulate_stats(characteristics['equipment_signatures'], equipment_stats)
                
                # Contrast and intensity characteristics
                contrast_stats = self._analyze_contrast_characteristics(img)
                self._accumulate_stats(characteristics['contrast_characteristics'], contrast_stats)
                
                # Noise model characteristics
                noise_stats = self._analyze_noise_characteristics(img)
                self._accumulate_stats(characteristics['noise_models'], noise_stats)
        
        # Average accumulated statistics
        for category in characteristics:
            for key in characteristics[category]:
                if isinstance(characteristics[category][key], list):
                    characteristics[category][key] = np.mean(characteristics[category][key])
        
        print(f"✅ Extracted ultrasound characteristics from {len(images)} images")
        return characteristics
    
    def _accumulate_stats(self, category_dict: Dict, new_stats: Dict):
        """Accumulate statistics across images."""
        for key, value in new_stats.items():
            if key not in category_dict:
                category_dict[key] = []
            category_dict[key].append(value)
    
    def _analyze_speckle_patterns(self, image: np.ndarray) -> Dict:
        """Analyze ultrasound speckle patterns."""
        # Speckle noise follows Rayleigh distribution in ultrasound
        
        # Local variance analysis (speckle creates high local variance)
        local_var = ndimage.generic_filter(image.astype(np.float32), np.var, size=5)
        
        # Rayleigh distribution fitting for speckle characterization
        flat_img = image.flatten()
        rayleigh_param = rayleigh.fit(flat_img)[1]  # Scale parameter
        
        # Speckle contrast measure
        mean_intensity = np.mean(image)
        std_intensity = np.std(image)
        speckle_contrast = std_intensity / mean_intensity if mean_intensity > 0 else 0
        
        # Autocorrelation analysis for speckle size
        autocorr = signal.correlate2d(image, image, mode='same')
        autocorr_center = autocorr[autocorr.shape[0]//2, autocorr.shape[1]//2]
        normalized_autocorr = autocorr / autocorr_center
        
        # Find correlation length (speckle size indicator)
        center_x, center_y = normalized_autocorr.shape[0]//2, normalized_autocorr.shape[1]//2
        correlation_length = 0
        for r in range(1, min(center_x, center_y)):
            if normalized_autocorr[center_x + r, center_y] < 0.5:
                correlation_length = r
                break
        
        return {
            'local_variance_mean': float(np.mean(local_var)),
            'local_variance_std': float(np.std(local_var)),
            'rayleigh_scale': float(rayleigh_param),
            'speckle_contrast': float(speckle_contrast),
            'correlation_length': float(correlation_length),
            'speckle_intensity_ratio': float(np.sum(local_var > np.mean(local_var)) / local_var.size)
        }
    
    def _analyze_local_textures(self, image: np.ndarray) -> Dict:
        """Analyze local texture characteristics using multiple methods."""
        
        # Gray Level Co-occurrence Matrix (GLCM) features
        from skimage.feature import graycomatrix, graycoprops
        
        # Normalize image to 0-255 and convert to uint8
        normalized = ((image - np.min(image)) / (np.max(image) - np.min(image)) * 255).astype(np.uint8)
        
        # GLCM analysis
        distances = [1, 2, 3]
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        
        glcm = graycomatrix(normalized, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)
        
        # Calculate texture properties
        contrast = np.mean(graycoprops(glcm, 'contrast'))
        dissimilarity = np.mean(graycoprops(glcm, 'dissimilarity'))
        homogeneity = np.mean(graycoprops(glcm, 'homogeneity'))
        energy = np.mean(graycoprops(glcm, 'energy'))
        correlation = np.mean(graycoprops(glcm, 'correlation'))
        
        # Local Binary Pattern analysis
        from skimage.feature import local_binary_pattern
        lbp = local_binary_pattern(image, P=8, R=1, method='uniform')
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=10)
        lbp_hist = lbp_hist / np.sum(lbp_hist)  # Normalize
        
        # Gradient analysis
        grad_x = filters.sobel_h(image)
        grad_y = filters.sobel_v(image)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        return {
            'glcm_contrast': float(contrast),
            'glcm_dissimilarity': float(dissimilarity),
            'glcm_homogeneity': float(homogeneity),
            'glcm_energy': float(energy),
            'glcm_correlation': float(correlation),
            'lbp_uniformity': float(np.max(lbp_hist)),
            'lbp_entropy': float(-np.sum(lbp_hist * np.log2(lbp_hist + 1e-8))),
            'gradient_mean': float(np.mean(gradient_magnitude)),
            'gradient_std': float(np.std(gradient_magnitude)),
            'gradient_skewness': float(self._skewness(gradient_magnitude.flatten()))
        }
    
    def _analyze_equipment_signatures(self, image: np.ndarray) -> Dict:
        """Analyze equipment-specific imaging signatures."""
        
        # Frequency domain analysis for equipment characteristics
        f_transform = np.fft.fft2(image)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.log(np.abs(f_shift) + 1)
        
        # Power spectral density
        psd = np.abs(f_shift)**2
        
        # Analyze frequency characteristics
        center_x, center_y = magnitude_spectrum.shape[0]//2, magnitude_spectrum.shape[1]//2
        
        # Radial frequency analysis
        y, x = np.ogrid[:magnitude_spectrum.shape[0], :magnitude_spectrum.shape[1]]
        r = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        
        # Average power at different frequency rings
        max_radius = min(center_x, center_y)
        radii = np.linspace(0, max_radius, 10)
        radial_power = []
        
        for i in range(len(radii)-1):
            mask = (r >= radii[i]) & (r < radii[i+1])
            radial_power.append(np.mean(psd[mask]))
        
        # Equipment-specific artifact detection
        # Look for periodic patterns (scan line artifacts, etc.)
        autocorr = signal.correlate2d(image, image, mode='same')
        peak_prominence = self._find_periodic_artifacts(autocorr)
        
        return {
            'freq_center_power': float(magnitude_spectrum[center_x, center_y]),
            'freq_high_ratio': float(np.mean(radial_power[-3:])) / (float(np.mean(radial_power[:3])) + 1e-8),
            'freq_uniformity': float(np.std(radial_power)) / (float(np.mean(radial_power)) + 1e-8),
            'periodic_artifacts': float(peak_prominence),
            'psd_entropy': float(-np.sum((psd/np.sum(psd)) * np.log2((psd/np.sum(psd)) + 1e-8))),
            'spectral_centroid': float(np.sum(radii[:-1] * radial_power) / (np.sum(radial_power) + 1e-8))
        }
    
    def _analyze_contrast_characteristics(self, image: np.ndarray) -> Dict:
        """Analyze contrast and intensity characteristics."""
        
        # Multi-scale contrast analysis
        scales = [3, 5, 9, 15]
        contrast_scales = []
        
        for scale in scales:
            # Local contrast using different window sizes
            mean_filter = ndimage.uniform_filter(image.astype(np.float32), size=scale)
            var_filter = ndimage.uniform_filter(image.astype(np.float32)**2, size=scale) - mean_filter**2
            local_contrast = np.sqrt(var_filter) / (mean_filter + 1e-8)
            contrast_scales.append(np.mean(local_contrast))
        
        # Dynamic range analysis
        intensity_range = np.max(image) - np.min(image)
        effective_range = np.percentile(image, 95) - np.percentile(image, 5)
        
        # Contrast sensitivity analysis
        edges = filters.canny(image)
        edge_density = np.sum(edges) / edges.size
        
        return {
            'multi_scale_contrast': contrast_scales,
            'intensity_range': float(intensity_range),
            'effective_range': float(effective_range),
            'contrast_uniformity': float(np.std(contrast_scales)),
            'edge_density': float(edge_density),
            'intensity_skewness': float(self._skewness(image.flatten())),
            'intensity_kurtosis': float(self._kurtosis(image.flatten()))
        }
    
    def _analyze_noise_characteristics(self, image: np.ndarray) -> Dict:
        """Analyze noise characteristics specific to ultrasound."""
        
        # Estimate noise using wavelet denoising
        denoised = restoration.denoise_wavelet(image, method='BayesShrink', mode='soft')
        noise_estimate = image - denoised
        
        # Noise statistics
        noise_std = np.std(noise_estimate)
        noise_mean = np.mean(np.abs(noise_estimate))
        
        # Signal-to-noise ratio estimation
        signal_power = np.var(denoised)
        noise_power = np.var(noise_estimate)
        snr = signal_power / (noise_power + 1e-8)
        
        # Noise texture analysis
        noise_lbp = feature.local_binary_pattern(noise_estimate, P=8, R=1, method='uniform')
        noise_uniformity = len(np.unique(noise_lbp)) / (noise_lbp.size * 0.1)  # Normalized
        
        return {
            'noise_std': float(noise_std),
            'noise_mean': float(noise_mean),
            'snr_estimate': float(snr),
            'noise_uniformity': float(noise_uniformity),
            'noise_skewness': float(self._skewness(noise_estimate.flatten()))
        }
    
    def _find_periodic_artifacts(self, autocorr: np.ndarray) -> float:
        """Find periodic artifacts in autocorrelation."""
        center = autocorr.shape[0] // 2
        center_line = autocorr[center, :]
        
        # Find peaks (excluding center)
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(center_line[center+5:], height=np.max(center_line)*0.1)
        
        return float(len(peaks)) if len(peaks) > 0 else 0.0
    
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
    
    def apply_advanced_style_transfer(self, source_img: np.ndarray, source_mask: np.ndarray, 
                                   target_characteristics: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """Apply advanced medical ultrasound style transfer."""
        
        print("🎯 Applying advanced medical style transfer...")
        
        # Step 1: Speckle pattern transfer
        styled_img = self._transfer_speckle_patterns(source_img, target_characteristics['speckle_patterns'])
        
        # Step 2: Local texture enhancement
        styled_img = self._transfer_local_textures(styled_img, target_characteristics['local_textures'])
        
        # Step 3: Equipment signature simulation
        styled_img = self._simulate_equipment_signatures(styled_img, target_characteristics['equipment_signatures'])
        
        # Step 4: Contrast characteristic adjustment
        styled_img = self._adjust_contrast_characteristics(styled_img, target_characteristics['contrast_characteristics'])
        
        # Step 5: Noise model application
        styled_img = self._apply_noise_model(styled_img, target_characteristics['noise_models'])
        
        # Step 6: Lesion-aware refinement
        if source_mask is not None:
            styled_img = self._lesion_aware_refinement(source_img, styled_img, source_mask)
        
        # Ensure proper range
        styled_img = np.clip(styled_img, 0, 255).astype(np.uint8)
        
        return styled_img, source_mask
    
    def _transfer_speckle_patterns(self, image: np.ndarray, speckle_chars: Dict) -> np.ndarray:
        """Transfer speckle pattern characteristics."""
        
        # Generate synthetic speckle based on target characteristics
        target_contrast = speckle_chars['speckle_contrast']
        target_correlation_length = speckle_chars['correlation_length']
        
        # Current speckle characteristics
        current_contrast = np.std(image) / (np.mean(image) + 1e-8)
        
        # Adjust speckle contrast
        if current_contrast > 0:
            contrast_factor = target_contrast / current_contrast
            contrast_factor = np.clip(contrast_factor, 0.5, 2.0)  # Prevent extreme changes
            
            mean_val = np.mean(image)
            adjusted = (image - mean_val) * contrast_factor + mean_val
        else:
            adjusted = image.copy()
        
        # Add synthetic speckle with target characteristics
        speckle_noise = self._generate_correlated_speckle(
            image.shape, 
            target_correlation_length, 
            speckle_chars['rayleigh_scale']
        )
        
        # Blend original and synthetic speckle
        blending_factor = 0.3
        result = (1 - blending_factor) * adjusted + blending_factor * speckle_noise
        
        return result
    
    def _generate_correlated_speckle(self, shape: Tuple, correlation_length: float, scale: float) -> np.ndarray:
        """Generate correlated speckle noise."""
        
        # Generate white noise
        white_noise = np.random.rayleigh(scale, shape)
        
        # Apply spatial correlation
        if correlation_length > 0:
            # Create correlation kernel
            kernel_size = int(correlation_length * 2) + 1
            kernel = np.ones((kernel_size, kernel_size)) / (kernel_size * kernel_size)
            
            # Apply correlation
            correlated_speckle = ndimage.convolve(white_noise, kernel, mode='reflect')
        else:
            correlated_speckle = white_noise
        
        return correlated_speckle
    
    def _transfer_local_textures(self, image: np.ndarray, texture_chars: Dict) -> np.ndarray:
        """Transfer local texture characteristics."""
        
        # Patch-based texture transfer for more local control
        patch_size = self.patch_size
        overlap = self.overlap
        
        result = image.copy()
        
        for y in range(0, image.shape[0] - patch_size, patch_size - overlap):
            for x in range(0, image.shape[1] - patch_size, patch_size - overlap):
                # Extract patch
                patch = image[y:y+patch_size, x:x+patch_size]
                
                # Enhance texture in patch
                enhanced_patch = self._enhance_patch_texture(patch, texture_chars)
                
                # Blend back with overlap handling
                end_y, end_x = min(y+patch_size, image.shape[0]), min(x+patch_size, image.shape[1])
                result[y:end_y, x:end_x] = enhanced_patch[:end_y-y, :end_x-x]
        
        return result
    
    def _enhance_patch_texture(self, patch: np.ndarray, texture_chars: Dict) -> np.ndarray:
        """Enhance texture in a single patch."""
        
        # Target texture characteristics
        target_contrast = texture_chars['glcm_contrast']
        target_homogeneity = texture_chars['glcm_homogeneity']
        
        # Calculate current characteristics (simplified for efficiency)
        current_contrast = np.var(patch)
        
        # Adjust local contrast
        if current_contrast > 0:
            # Use unsharp masking for texture enhancement
            blurred = ndimage.gaussian_filter(patch, sigma=1.0)
            sharpened = patch + 0.5 * (patch - blurred)
            
            # Adjust based on target characteristics
            mean_val = np.mean(patch)
            enhanced = (sharpened - mean_val) * 1.2 + mean_val
            
            return np.clip(enhanced, 0, 255)
        
        return patch
    
    def _simulate_equipment_signatures(self, image: np.ndarray, equipment_chars: Dict) -> np.ndarray:
        """Simulate equipment-specific imaging signatures."""
        
        # Apply subtle frequency domain adjustments
        f_transform = np.fft.fft2(image)
        f_shift = np.fft.fftshift(f_transform)
        
        # Modify frequency response based on target characteristics
        target_high_ratio = equipment_chars['freq_high_ratio']
        
        # Create frequency mask for enhancement
        center_x, center_y = f_shift.shape[0]//2, f_shift.shape[1]//2
        y, x = np.ogrid[:f_shift.shape[0], :f_shift.shape[1]]
        r = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        
        # Enhance high frequencies based on target
        max_radius = min(center_x, center_y)
        high_freq_mask = r > (max_radius * 0.6)
        
        # Apply gentle frequency adjustment
        adjustment_factor = 1.0 + 0.1 * (target_high_ratio - 1.0)
        adjustment_factor = np.clip(adjustment_factor, 0.8, 1.2)
        
        f_shift[high_freq_mask] *= adjustment_factor
        
        # Transform back
        f_ishift = np.fft.ifftshift(f_shift)
        enhanced = np.fft.ifft2(f_ishift)
        enhanced = np.real(enhanced)
        
        return enhanced
    
    def _adjust_contrast_characteristics(self, image: np.ndarray, contrast_chars: Dict) -> np.ndarray:
        """Adjust contrast characteristics to match target."""
        
        # Multi-scale contrast adjustment
        target_contrasts = contrast_chars['multi_scale_contrast']
        
        result = image.copy()
        
        for i, scale in enumerate([3, 5, 9, 15]):
            if i < len(target_contrasts):
                # Calculate current contrast at this scale
                mean_filter = ndimage.uniform_filter(result.astype(np.float32), size=scale)
                var_filter = ndimage.uniform_filter(result.astype(np.float32)**2, size=scale) - mean_filter**2
                current_contrast = np.sqrt(var_filter) / (mean_filter + 1e-8)
                
                # Adjust
                target_contrast = target_contrasts[i]
                avg_current = np.mean(current_contrast)
                
                if avg_current > 0:
                    adjustment = target_contrast / avg_current
                    adjustment = np.clip(adjustment, 0.7, 1.3)
                    
                    # Apply local adjustment
                    result = (result - mean_filter) * adjustment + mean_filter
        
        return result
    
    def _apply_noise_model(self, image: np.ndarray, noise_chars: Dict) -> np.ndarray:
        """Apply target noise model characteristics."""
        
        target_snr = noise_chars['snr_estimate']
        target_noise_std = noise_chars['noise_std']
        
        # Estimate current noise level
        denoised = restoration.denoise_wavelet(image, method='BayesShrink', mode='soft')
        current_noise = image - denoised
        current_noise_std = np.std(current_noise)
        
        # Adjust noise level
        if current_noise_std > 0:
            noise_adjustment = target_noise_std / current_noise_std
            noise_adjustment = np.clip(noise_adjustment, 0.5, 2.0)
            
            # Apply noise adjustment
            adjusted_noise = current_noise * noise_adjustment
            result = denoised + adjusted_noise
        else:
            # Add synthetic noise if none present
            synthetic_noise = np.random.normal(0, target_noise_std, image.shape)
            result = image + synthetic_noise
        
        return result
    
    def _lesion_aware_refinement(self, original: np.ndarray, styled: np.ndarray, 
                               mask: np.ndarray) -> np.ndarray:
        """Apply lesion-aware refinement to preserve medical features."""
        
        lesion_mask = mask > 128
        if not np.any(lesion_mask):
            return styled
        
        # Stronger preservation of lesion areas
        dilated_mask = morphology.dilation(lesion_mask, morphology.disk(3))
        
        # Blend more conservatively in lesion areas
        preservation_weight = np.zeros_like(mask, dtype=np.float32)
        preservation_weight[lesion_mask] = 0.6  # Strong preservation
        preservation_weight[dilated_mask & ~lesion_mask] = 0.3  # Moderate preservation
        
        # Apply weighted blending
        result = styled.copy()
        blend_areas = preservation_weight > 0
        result[blend_areas] = ((1 - preservation_weight[blend_areas]) * styled[blend_areas] + 
                               preservation_weight[blend_areas] * original[blend_areas])
        
        return result


class AdvancedMedicalStylePipeline:
    """Complete advanced medical style transfer pipeline."""
    
    def __init__(self):
        self.style_transfer = MedicalUltrasoundStyleTransfer()
        self.busi_path = "dataset/BioMedicalDataset/BUSI"
        self.bus_uclm_path = "dataset/BioMedicalDataset/BUS-UCLM"
        self.output_path = "dataset/BioMedicalDataset/BUSI-BUS-UCLM-Advanced"
        
    def run_advanced_pipeline(self):
        """Run the complete advanced medical style transfer pipeline."""
        print("🚀 Starting advanced medical ultrasound style transfer...")
        
        # Create output directory
        os.makedirs(self.output_path, exist_ok=True)
        os.makedirs(f"{self.output_path}/benign/image", exist_ok=True)
        os.makedirs(f"{self.output_path}/benign/mask", exist_ok=True)
        os.makedirs(f"{self.output_path}/malignant/image", exist_ok=True)
        os.makedirs(f"{self.output_path}/malignant/mask", exist_ok=True)
        
        # Step 1: Extract advanced BUS-UCLM characteristics
        bus_uclm_characteristics = self._extract_bus_uclm_characteristics()
        
        # Step 2: Apply advanced style transfer to BUSI
        styled_records = self._apply_advanced_style_transfer(bus_uclm_characteristics)
        
        # Step 3: Create dataset CSV
        self._create_advanced_dataset_csv(styled_records)
        
        print("✅ Advanced medical style transfer completed!")
        return len(styled_records)
    
    def _extract_bus_uclm_characteristics(self) -> Dict:
        """Extract comprehensive BUS-UCLM characteristics."""
        print("🔬 Extracting BUS-UCLM medical imaging characteristics...")
        
        # Load all BUS-UCLM data
        train_df = pd.read_csv(f"{self.bus_uclm_path}/train_frame.csv")
        val_df = pd.read_csv(f"{self.bus_uclm_path}/val_frame.csv")
        test_df = pd.read_csv(f"{self.bus_uclm_path}/test_frame.csv")
        
        all_data = pd.concat([train_df, val_df, test_df], ignore_index=True)
        
        images = []
        masks = []
        
        # Load sample of images for characteristic extraction
        sample_size = min(100, len(all_data))  # Use subset for efficiency
        sampled_data = all_data.sample(sample_size, random_state=42)
        
        for _, row in sampled_data.iterrows():
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
        
        print(f"📈 Loaded {len(images)} BUS-UCLM images for characteristic analysis")
        
        # Extract advanced characteristics
        characteristics = self.style_transfer.extract_ultrasound_characteristics(images, masks)
        
        # Save characteristics
        with open(f"{self.output_path}/advanced_bus_uclm_characteristics.json", 'w') as f:
            json.dump(characteristics, f, indent=2)
        
        return characteristics
    
    def _apply_advanced_style_transfer(self, target_characteristics: Dict) -> List[Dict]:
        """Apply advanced style transfer to BUSI images."""
        print("🎨 Applying advanced medical style transfer to BUSI...")
        
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
                    # Apply advanced style transfer
                    styled_img, styled_mask = self.style_transfer.apply_advanced_style_transfer(
                        img, mask, target_characteristics
                    )
                    
                    # Generate output filenames
                    base_name = os.path.splitext(row['image_path'])[0]
                    styled_img_name = f"advanced_{base_name}.png"
                    styled_mask_name = f"advanced_{base_name}_mask.png"
                    
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
                        print(f"   🎯 Processed {idx + 1}/{len(busi_train)} images")
        
        print(f"✅ Completed advanced style transfer: {len(styled_records)} images")
        return styled_records
    
    def _create_advanced_dataset_csv(self, styled_records: List[Dict]):
        """Create CSV for the advanced styled dataset."""
        styled_df = pd.DataFrame(styled_records)
        csv_path = f"{self.output_path}/advanced_styled_dataset.csv"
        styled_df.to_csv(csv_path, index=False)
        
        print(f"📁 Saved advanced dataset CSV: {csv_path}")
        print(f"   📊 Total samples: {len(styled_records)}")
        print(f"   📊 Benign: {len(styled_df[styled_df['class'] == 'benign'])}")
        print(f"   📊 Malignant: {len(styled_df[styled_df['class'] == 'malignant'])}")


if __name__ == "__main__":
    pipeline = AdvancedMedicalStylePipeline()
    pipeline.run_advanced_pipeline() 