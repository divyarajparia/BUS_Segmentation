"""
Medical Image-Specific Privacy-Preserving Style Transfer
Designed specifically for grayscale medical ultrasound images.
No VGG dependency - uses medical image analysis techniques.
"""

import numpy as np
import cv2
from PIL import Image
import os
import json
import pandas as pd
from tqdm import tqdm
from scipy import ndimage, stats
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans
from skimage import feature, filters, measure, segmentation
from skimage.filters import gabor
import warnings
warnings.filterwarnings('ignore')

class MedicalPrivacyStyleTransfer:
    """
    Privacy-preserving style transfer specifically designed for medical ultrasound images.
    Uses medical image analysis techniques instead of natural image features.
    """
    
    def __init__(self):
        print("🏥 Medical Privacy-Preserving Style Transfer initialized")
        print("   Optimized for grayscale ultrasound images")
    
    def extract_medical_domain_statistics(self, dataset_path, csv_file, save_path=None):
        """
        Extract medical image-specific domain statistics for privacy-preserving style transfer.
        """
        print(f"🔬 Extracting medical domain statistics from {dataset_path}")
        
        # Load dataset
        csv_path = os.path.join(dataset_path, csv_file)
        df = pd.read_csv(csv_path)
        
        # Initialize medical statistics containers
        stats = {
            'intensity_distributions': [],
            'texture_features': [],
            'edge_characteristics': [],
            'ultrasound_speckle': [],
            'anatomical_patterns': [],
            'local_statistics': [],
            'frequency_analysis': [],
            'medical_gradients': []
        }
        
        valid_count = 0
        
        for idx in tqdm(range(len(df)), desc="Processing medical images"):
            try:
                # Load image
                image_path = self._get_image_path(df.iloc[idx], dataset_path)
                if not os.path.exists(image_path):
                    continue
                
                image = Image.open(image_path).convert('L')
                image_array = np.array(image).astype(np.float32)
                
                # Normalize to [0,1] for consistent processing
                image_norm = image_array / 255.0
                
                # Extract medical-specific features
                intensity_stats = self._extract_intensity_distribution(image_norm)
                stats['intensity_distributions'].append(intensity_stats)
                
                texture_stats = self._extract_medical_texture_features(image_norm)
                stats['texture_features'].append(texture_stats)
                
                edge_stats = self._extract_edge_characteristics(image_norm)
                stats['edge_characteristics'].append(edge_stats)
                
                speckle_stats = self._extract_ultrasound_speckle(image_norm)
                stats['ultrasound_speckle'].append(speckle_stats)
                
                anatomical_stats = self._extract_anatomical_patterns(image_norm)
                stats['anatomical_patterns'].append(anatomical_stats)
                
                local_stats = self._extract_local_statistics(image_norm)
                stats['local_statistics'].append(local_stats)
                
                freq_stats = self._extract_frequency_analysis(image_norm)
                stats['frequency_analysis'].append(freq_stats)
                
                gradient_stats = self._extract_medical_gradients(image_norm)
                stats['medical_gradients'].append(gradient_stats)
                
                valid_count += 1
                
            except Exception as e:
                print(f"Warning: Error processing image {idx}: {e}")
                continue
        
        if valid_count == 0:
            raise ValueError("No valid images found for style extraction")
        
        # Aggregate medical statistics
        aggregated_stats = self._aggregate_medical_statistics(stats, valid_count)
        
        if save_path:
            self._save_medical_statistics(aggregated_stats, save_path)
        
        print(f"✅ Medical statistics extracted from {valid_count} images")
        return aggregated_stats
    
    def _extract_intensity_distribution(self, image):
        """Extract intensity distribution characteristics specific to medical images"""
        flat = image.flatten()
        
        # Basic statistics
        mean_intensity = np.mean(flat)
        std_intensity = np.std(flat)
        skewness = stats.skew(flat)
        kurtosis = stats.kurtosis(flat)
        
        # Percentiles (important for medical image contrast)
        percentiles = np.percentile(flat, [5, 25, 50, 75, 95])
        
        # Histogram with medical-relevant bins
        hist, bins = np.histogram(flat, bins=64, density=True)
        
        # Dynamic range characteristics
        dynamic_range = np.max(flat) - np.min(flat)
        contrast = std_intensity / (mean_intensity + 1e-8)
        
        return {
            'mean': mean_intensity,
            'std': std_intensity,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'percentiles': percentiles.tolist(),
            'histogram': hist.tolist(),
            'dynamic_range': dynamic_range,
            'contrast': contrast
        }
    
    def _extract_medical_texture_features(self, image):
        """Extract texture features relevant to medical ultrasound images"""
        
        # 1. Gray Level Co-occurrence Matrix (GLCM) features
        # Multiple directions for ultrasound texture analysis
        glcm_features = {}
        directions = [0, 45, 90, 135]  # degrees
        
        for direction in directions:
            angle_rad = np.radians(direction)
            dx = int(np.round(np.cos(angle_rad)))
            dy = int(np.round(np.sin(angle_rad)))
            
            # Simple GLCM computation
            glcm = self._compute_simple_glcm(image, dx, dy)
            
            # GLCM features
            contrast = np.sum(glcm * np.square(np.arange(64)[:, None] - np.arange(64)[None, :]))
            energy = np.sum(glcm * glcm)
            homogeneity = np.sum(glcm / (1 + np.abs(np.arange(64)[:, None] - np.arange(64)[None, :])))
            
            glcm_features[f'{direction}deg'] = {
                'contrast': contrast,
                'energy': energy,
                'homogeneity': homogeneity
            }
        
        # 2. Local Binary Patterns for ultrasound speckle
        lbp = feature.local_binary_pattern(image, 8, 1, method='uniform')
        lbp_hist = np.histogram(lbp, bins=10, density=True)[0]
        
        # 3. Gabor filter responses at multiple scales/orientations
        gabor_responses = []
        for frequency in [0.1, 0.3, 0.5]:
            for theta in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
                real, _ = gabor(image, frequency=frequency, theta=theta)
                gabor_responses.append({
                    'mean': np.mean(real),
                    'std': np.std(real),
                    'energy': np.sum(real**2)
                })
        
        return {
            'glcm_features': glcm_features,
            'lbp_histogram': lbp_hist.tolist(),
            'gabor_responses': gabor_responses
        }
    
    def _extract_edge_characteristics(self, image):
        """Extract edge characteristics important for medical structure preservation"""
        
        # Multiple edge detection methods
        edges_canny = feature.canny(image, sigma=1.0)
        edges_sobel = filters.sobel(image)
        
        # Edge density and distribution
        edge_density = np.mean(edges_canny)
        edge_strength_mean = np.mean(edges_sobel)
        edge_strength_std = np.std(edges_sobel)
        
        # Edge orientation analysis
        gradient_x = filters.sobel_h(image)
        gradient_y = filters.sobel_v(image)
        edge_angles = np.arctan2(gradient_y, gradient_x)
        
        # Histogram of edge orientations
        angle_hist, _ = np.histogram(edge_angles[edges_canny], bins=18, range=(-np.pi, np.pi), density=True)
        
        # Edge continuity (important for anatomical structures)
        labeled_edges = measure.label(edges_canny)
        edge_props = measure.regionprops(labeled_edges)
        
        edge_lengths = [prop.area for prop in edge_props]
        avg_edge_length = np.mean(edge_lengths) if edge_lengths else 0
        max_edge_length = np.max(edge_lengths) if edge_lengths else 0
        
        return {
            'edge_density': edge_density,
            'edge_strength_mean': edge_strength_mean,
            'edge_strength_std': edge_strength_std,
            'orientation_histogram': angle_hist.tolist(),
            'avg_edge_length': avg_edge_length,
            'max_edge_length': max_edge_length
        }
    
    def _extract_ultrasound_speckle(self, image):
        """Extract ultrasound-specific speckle noise characteristics"""
        
        # Speckle noise analysis using local statistics
        kernel_size = 7
        kernel = np.ones((kernel_size, kernel_size)) / (kernel_size**2)
        
        # Local mean and variance
        local_mean = ndimage.convolve(image, kernel)
        local_var = ndimage.convolve(image**2, kernel) - local_mean**2
        
        # Speckle index (coefficient of variation)
        speckle_index = np.mean(np.sqrt(local_var) / (local_mean + 1e-8))
        
        # Speckle pattern analysis using autocorrelation
        image_centered = image - np.mean(image)
        autocorr = ndimage.correlate(image_centered, image_centered)
        
        # Central region of autocorrelation
        center = np.array(autocorr.shape) // 2
        autocorr_center = autocorr[center[0]-10:center[0]+11, center[1]-10:center[1]+11]
        
        # Speckle correlation length
        autocorr_profile = autocorr_center[10, :]  # Central horizontal line
        correlation_length = np.argmax(autocorr_profile < 0.1 * autocorr_profile[10])
        
        return {
            'speckle_index': speckle_index,
            'local_variance_mean': np.mean(local_var),
            'local_variance_std': np.std(local_var),
            'correlation_length': correlation_length
        }
    
    def _extract_anatomical_patterns(self, image):
        """Extract anatomical pattern characteristics from medical images"""
        
        # Multi-scale morphological operations
        patterns = {}
        
        for scale in [3, 7, 15]:  # Different structure sizes
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (scale, scale))
            
            # Convert to uint8 for OpenCV
            image_uint8 = (image * 255).astype(np.uint8)
            
            # Morphological operations
            opened = cv2.morphologyEx(image_uint8, cv2.MORPH_OPEN, kernel)
            closed = cv2.morphologyEx(image_uint8, cv2.MORPH_CLOSE, kernel)
            tophat = cv2.morphologyEx(image_uint8, cv2.MORPH_TOPHAT, kernel)
            blackhat = cv2.morphologyEx(image_uint8, cv2.MORPH_BLACKHAT, kernel)
            
            patterns[f'scale_{scale}'] = {
                'opening_mean': np.mean(opened) / 255.0,
                'closing_mean': np.mean(closed) / 255.0,
                'tophat_energy': np.sum(tophat**2) / (255.0**2),
                'blackhat_energy': np.sum(blackhat**2) / (255.0**2)
            }
        
        return patterns
    
    def _extract_local_statistics(self, image):
        """Extract local statistical characteristics"""
        
        # Divide image into regions for local analysis
        h, w = image.shape
        regions = []
        
        # 4x4 grid of regions
        for i in range(4):
            for j in range(4):
                r_start = i * h // 4
                r_end = (i + 1) * h // 4
                c_start = j * w // 4
                c_end = (j + 1) * w // 4
                
                region = image[r_start:r_end, c_start:c_end]
                regions.append({
                    'mean': np.mean(region),
                    'std': np.std(region),
                    'energy': np.sum(region**2)
                })
        
        # Global statistics from local variations
        region_means = [r['mean'] for r in regions]
        region_stds = [r['std'] for r in regions]
        
        return {
            'regions': regions,
            'spatial_mean_variation': np.std(region_means),
            'spatial_texture_variation': np.std(region_stds)
        }
    
    def _extract_frequency_analysis(self, image):
        """Extract frequency domain characteristics"""
        
        # 2D FFT
        fft = np.fft.fft2(image)
        fft_shift = np.fft.fftshift(fft)
        magnitude = np.abs(fft_shift)
        
        # Radial frequency analysis
        center = np.array(image.shape) // 2
        y, x = np.ogrid[:image.shape[0], :image.shape[1]]
        r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
        
        # Frequency bands
        max_r = min(center)
        low_freq = np.mean(magnitude[r < max_r//4])
        mid_freq = np.mean(magnitude[(r >= max_r//4) & (r < max_r//2)])
        high_freq = np.mean(magnitude[r >= max_r//2])
        
        # Dominant frequency detection
        magnitude_1d = np.mean(magnitude, axis=0)
        dominant_freq_idx = np.argmax(magnitude_1d)
        
        return {
            'dc_component': magnitude[center[0], center[1]],
            'low_freq_energy': low_freq,
            'mid_freq_energy': mid_freq,
            'high_freq_energy': high_freq,
            'dominant_frequency': dominant_freq_idx
        }
    
    def _extract_medical_gradients(self, image):
        """Extract gradient characteristics relevant to medical images"""
        
        # Multi-scale gradient analysis
        gradients = {}
        
        for sigma in [0.5, 1.0, 2.0]:  # Different smoothing scales
            smoothed = filters.gaussian(image, sigma=sigma)
            
            grad_x = filters.sobel_h(smoothed)
            grad_y = filters.sobel_v(smoothed)
            grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            gradients[f'sigma_{sigma}'] = {
                'mean_magnitude': np.mean(grad_magnitude),
                'std_magnitude': np.std(grad_magnitude),
                'max_magnitude': np.max(grad_magnitude)
            }
        
        return gradients
    
    def apply_medical_style_transfer(self, source_image_path, target_stats, output_path, method='medical_adaptive'):
        """
        Apply medical image-specific style transfer.
        """
        # Load source image
        source_image = Image.open(source_image_path).convert('L')
        source_array = np.array(source_image).astype(np.float32) / 255.0
        
        if method == 'medical_adaptive':
            styled_image = self._medical_adaptive_transfer(source_array, target_stats)
        elif method == 'ultrasound_optimized':
            styled_image = self._ultrasound_optimized_transfer(source_array, target_stats)
        elif method == 'structure_aware':
            styled_image = self._structure_aware_transfer(source_array, target_stats)
        else:
            # Default medical transfer
            styled_image = self._medical_adaptive_transfer(source_array, target_stats)
        
        # Convert back to [0,255] and save as RGB
        styled_uint8 = np.clip(styled_image * 255, 0, 255).astype(np.uint8)
        styled_rgb = cv2.cvtColor(styled_uint8, cv2.COLOR_GRAY2RGB)
        styled_pil = Image.fromarray(styled_rgb)
        styled_pil.save(output_path)
        
        return styled_uint8
    
    def _medical_adaptive_transfer(self, source_image, target_stats):
        """
        Medical image-specific adaptive style transfer that preserves anatomical structures.
        """
        # Extract target characteristics
        target_intensity = target_stats['intensity_distributions']
        target_mean = np.mean([stats['mean'] for stats in target_intensity])
        target_std = np.mean([stats['std'] for stats in target_intensity])
        target_percentiles = np.mean([stats['percentiles'] for stats in target_intensity], axis=0)
        
        # Step 1: Intensity distribution matching
        source_mean = np.mean(source_image)
        source_std = np.std(source_image)
        
        # Normalize source to have similar statistics
        normalized = (source_image - source_mean) / (source_std + 1e-8)
        scaled = normalized * target_std + target_mean
        
        # Step 2: Percentile-based refinement
        source_percentiles = np.percentile(scaled, [5, 25, 50, 75, 95])
        
        # Create mapping function
        mapping = np.interp(np.linspace(0, 1, 256), 
                           source_percentiles / np.max(source_percentiles),
                           target_percentiles / np.max(target_percentiles))
        
        # Apply mapping
        scaled_indices = np.clip(scaled * 255, 0, 255).astype(int)
        mapped_image = mapping[scaled_indices] / 255.0
        
        # Step 3: Edge-preserving smoothing
        mapped_uint8 = np.clip(mapped_image * 255, 0, 255).astype(np.uint8)
        smoothed = cv2.bilateralFilter(mapped_uint8, 9, 75, 75)
        
        # Step 4: Preserve critical edges from original
        source_edges = feature.canny(source_image, sigma=1.0)
        
        # Blend to preserve edges
        result = smoothed.astype(np.float32) / 255.0
        edge_weight = 0.2
        result[source_edges] = (
            (1 - edge_weight) * result[source_edges] +
            edge_weight * source_image[source_edges]
        )
        
        return np.clip(result, 0, 1)
    
    def _ultrasound_optimized_transfer(self, source_image, target_stats):
        """
        Ultrasound-optimized style transfer focusing on speckle and texture patterns.
        """
        # Focus on ultrasound-specific characteristics
        target_speckle = target_stats['ultrasound_speckle']
        target_speckle_index = np.mean([stats['speckle_index'] for stats in target_speckle])
        
        # Adjust speckle characteristics
        kernel_size = 7
        kernel = np.ones((kernel_size, kernel_size)) / (kernel_size**2)
        
        local_mean = ndimage.convolve(source_image, kernel)
        local_var = ndimage.convolve(source_image**2, kernel) - local_mean**2
        
        # Adjust local variance to match target speckle
        source_speckle_index = np.mean(np.sqrt(local_var) / (local_mean + 1e-8))
        speckle_adjustment = target_speckle_index / (source_speckle_index + 1e-8)
        
        # Apply speckle adjustment
        adjusted_var = local_var * speckle_adjustment
        noise = np.random.normal(0, np.sqrt(np.abs(adjusted_var)))
        
        styled_image = local_mean + 0.1 * noise  # Small noise contribution
        
        # Apply intensity matching
        styled_image = self._apply_intensity_matching(styled_image, target_stats)
        
        return np.clip(styled_image, 0, 1)
    
    def _structure_aware_transfer(self, source_image, target_stats):
        """
        Structure-aware transfer that preserves anatomical patterns.
        """
        # Use morphological operations to preserve structure
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        
        # Extract structural components
        source_uint8 = (source_image * 255).astype(np.uint8)
        opened = cv2.morphologyEx(source_uint8, cv2.MORPH_OPEN, kernel)
        structure = source_uint8.astype(np.float32) - opened.astype(np.float32)
        
        # Apply style to opened component only
        opened_norm = opened.astype(np.float32) / 255.0
        styled_structure = self._apply_intensity_matching(opened_norm, target_stats)
        
        # Recombine
        result = styled_structure + structure / 255.0
        
        return np.clip(result, 0, 1)
    
    def _apply_intensity_matching(self, image, target_stats):
        """Apply intensity distribution matching"""
        target_intensity = target_stats['intensity_distributions']
        target_mean = np.mean([stats['mean'] for stats in target_intensity])
        target_std = np.mean([stats['std'] for stats in target_intensity])
        
        # Simple statistical matching
        source_mean = np.mean(image)
        source_std = np.std(image)
        
        normalized = (image - source_mean) / (source_std + 1e-8)
        matched = normalized * target_std + target_mean
        
        return matched
    
    # Helper methods
    def _get_image_path(self, row, dataset_path):
        """Helper to get image path from dataframe row"""
        image_path_info = row['image_path']
        
        if '/' in image_path_info:
            return os.path.join(dataset_path, image_path_info)
        else:
            class_type = image_path_info.split()[0]
            
            # Try BUSI format first
            image_path_busi = os.path.join(dataset_path, class_type, 'image', image_path_info)
            if os.path.exists(image_path_busi):
                return image_path_busi
            
            # Try BUS-UCLM format
            actual_filename = ' '.join(image_path_info.split()[1:])
            image_path_busuclm = os.path.join(dataset_path, class_type, 'images', actual_filename)
            return image_path_busuclm if os.path.exists(image_path_busuclm) else image_path_busi
    
    def _compute_simple_glcm(self, image, dx, dy):
        """Simple GLCM computation"""
        # Quantize image to 64 levels for manageable GLCM size
        quantized = (image * 63).astype(int)
        h, w = quantized.shape
        
        glcm = np.zeros((64, 64))
        
        for i in range(max(0, -dy), min(h, h-dy)):
            for j in range(max(0, -dx), min(w, w-dx)):
                if 0 <= i+dy < h and 0 <= j+dx < w:
                    glcm[quantized[i, j], quantized[i+dy, j+dx]] += 1
        
        # Normalize
        if np.sum(glcm) > 0:
            glcm = glcm / np.sum(glcm)
        
        return glcm
    
    def _aggregate_medical_statistics(self, stats, valid_count):
        """Aggregate the collected medical statistics"""
        aggregated = {
            'processed_images': valid_count,
            'intensity_distributions': stats['intensity_distributions'],
            'texture_features': stats['texture_features'],
            'edge_characteristics': stats['edge_characteristics'],
            'ultrasound_speckle': stats['ultrasound_speckle'],
            'anatomical_patterns': stats['anatomical_patterns'],
            'local_statistics': stats['local_statistics'],
            'frequency_analysis': stats['frequency_analysis'],
            'medical_gradients': stats['medical_gradients']
        }
        return aggregated
    
    def _save_medical_statistics(self, stats, save_path):
        """Save medical statistics to file"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_for_json(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(item) for item in obj]
            else:
                return obj
        
        serializable_stats = convert_for_json(stats)
        
        with open(save_path, 'w') as f:
            json.dump(serializable_stats, f, indent=2)
        
        print(f"💾 Medical statistics saved to {save_path}")


def generate_medical_styled_dataset(source_dataset_path, source_csv, target_stats_path, 
                                   output_dir, method='medical_adaptive'):
    """
    Generate styled dataset using medical image-specific privacy-preserving methods.
    """
    print(f"🏥 Generating medical styled dataset using {method}")
    
    # Initialize medical style transfer
    style_transfer = MedicalPrivacyStyleTransfer()
    
    # Load target statistics
    with open(target_stats_path, 'r') as f:
        target_stats = json.load(f)
    
    # Load source CSV
    csv_path = os.path.join(source_dataset_path, source_csv)
    df = pd.read_csv(csv_path)
    
    # Create output directories
    os.makedirs(os.path.join(output_dir, 'benign', 'image'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'benign', 'mask'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'malignant', 'image'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'malignant', 'mask'), exist_ok=True)
    
    styled_samples = []
    
    for idx in tqdm(range(len(df)), desc="Generating medical styled images"):
        try:
            row = df.iloc[idx]
            image_path_info = row['image_path']
            mask_path_info = row['mask_path']
            
            # Determine class
            class_type = image_path_info.split()[0]
            
            # Get source paths
            source_image_path = style_transfer._get_image_path(row, source_dataset_path)
            
            # Generate output paths
            output_filename = f"medical_styled_{os.path.basename(image_path_info)}"
            output_mask_filename = f"medical_styled_{os.path.basename(mask_path_info)}"
            
            output_image_path = os.path.join(output_dir, class_type, 'image', output_filename)
            output_mask_path = os.path.join(output_dir, class_type, 'mask', output_mask_filename)
            
            # Apply medical style transfer
            style_transfer.apply_medical_style_transfer(
                source_image_path, target_stats, output_image_path, method=method
            )
            
            # Process mask (same as before)
            mask_source_path = source_image_path.replace('/image/', '/images/').replace('/images/', '/masks/')
            mask_filename = os.path.basename(mask_path_info)
            if ' ' in mask_filename:
                mask_filename = mask_filename.replace(' ', '_')
            mask_source_path = os.path.join(os.path.dirname(mask_source_path), mask_filename)
            
            if os.path.exists(mask_source_path):
                mask = cv2.imread(mask_source_path)
                if mask is not None:
                    mask_resized = cv2.resize(mask, (256, 256))
                    mask_gray = cv2.cvtColor(mask_resized, cv2.COLOR_BGR2GRAY) if len(mask_resized.shape) == 3 else mask_resized
                    _, mask_binary = cv2.threshold(mask_gray, 127, 255, cv2.THRESH_BINARY)
                    mask_rgb = cv2.cvtColor(mask_binary, cv2.COLOR_GRAY2BGR)
                    cv2.imwrite(output_mask_path, mask_rgb)
            
            styled_samples.append({
                'image_path': f"{class_type} {output_filename}",
                'mask_path': f"{class_type} {output_mask_filename}",
                'class': class_type,
                'source_client': 'BUS-UCLM',
                'style_client': 'BUSI', 
                'augmentation_type': 'medical_styled'
            })
            
        except Exception as e:
            print(f"Warning: Error processing {idx}: {e}")
            continue
    
    # Save dataset CSV
    styled_csv_path = os.path.join(output_dir, 'medical_styled_dataset.csv')
    styled_df = pd.DataFrame(styled_samples)
    styled_df.to_csv(styled_csv_path, index=False)
    
    print(f"✅ Generated {len(styled_samples)} medical styled images")
    print(f"📁 Output directory: {output_dir}")
    print(f"📄 CSV file: {styled_csv_path}")
    
    return styled_samples


if __name__ == "__main__":
    # Test the medical system
    print("🧪 Testing Medical Privacy-Preserving Style Transfer")
    
    # Extract medical BUSI statistics
    extractor = MedicalPrivacyStyleTransfer()
    stats = extractor.extract_medical_domain_statistics(
        'dataset/BioMedicalDataset/BUSI',
        'train_frame.csv',
        'privacy_style_stats/busi_medical_stats.json'
    )
    
    # Generate medical styled dataset
    generate_medical_styled_dataset(
        'dataset/BioMedicalDataset/BUS-UCLM',
        'train_frame.csv',
        'privacy_style_stats/busi_medical_stats.json',
        'dataset/BioMedicalDataset/BUS-UCLM-Medical-Styled',
        method='medical_adaptive'
    ) 