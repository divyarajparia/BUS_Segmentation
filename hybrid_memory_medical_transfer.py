"""
Hybrid Medical Privacy-Preserving Style Transfer
Offers both memory-efficient and performance-optimized versions.
"""

import numpy as np
import cv2
from PIL import Image
import os
import json
import pandas as pd
from tqdm import tqdm
from scipy import stats
from skimage import feature, filters
import gc
import warnings
warnings.filterwarnings('ignore')

class HybridMedicalStyleTransfer:
    """
    Hybrid medical style transfer with configurable complexity levels.
    """
    
    def __init__(self, complexity='medium', batch_size=50):
        """
        Initialize with complexity level:
        - 'light': Memory-efficient, basic features
        - 'medium': Balanced performance and memory
        - 'heavy': Full features (use only if sufficient memory)
        """
        self.complexity = complexity
        self.batch_size = batch_size
        print(f"🏥 Hybrid Medical Style Transfer initialized")
        print(f"   Complexity: {complexity}")
        print(f"   Batch size: {batch_size}")
    
    def extract_medical_domain_statistics(self, dataset_path, csv_file, save_path=None):
        """Extract medical statistics with configurable complexity."""
        print(f"🔬 Extracting medical statistics (complexity: {self.complexity})")
        
        csv_path = os.path.join(dataset_path, csv_file)
        df = pd.read_csv(csv_path)
        
        if self.complexity == 'light':
            return self._extract_light_statistics(df, dataset_path, save_path)
        elif self.complexity == 'medium':
            return self._extract_medium_statistics(df, dataset_path, save_path)
        else:  # heavy
            return self._extract_heavy_statistics(df, dataset_path, save_path)
    
    def _extract_light_statistics(self, df, dataset_path, save_path):
        """Light complexity - memory efficient, basic features."""
        print("   Using LIGHT complexity (memory-efficient)")
        
        streaming_stats = {
            'count': 0,
            'intensity_sum': 0.0,
            'intensity_sum_sq': 0.0,
            'percentile_samples': [],
            'edge_density_sum': 0.0,
            'contrast_sum': 0.0,
            'histogram_accumulator': np.zeros(64),
        }
        
        total_images = len(df)
        num_batches = (total_images + self.batch_size - 1) // self.batch_size
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, total_images)
            
            print(f"   Processing batch {batch_idx + 1}/{num_batches}")
            
            for idx in range(start_idx, end_idx):
                try:
                    row = df.iloc[idx]
                    image_path = self._get_image_path(row, dataset_path)
                    if not os.path.exists(image_path):
                        continue
                    
                    image = Image.open(image_path).convert('L')
                    image_array = np.array(image, dtype=np.float32) / 255.0
                    
                    # Basic statistics only
                    flat = image_array.flatten()
                    mean_val = np.mean(flat)
                    std_val = np.std(flat)
                    
                    # Sample for percentiles
                    sample_size = min(5000, len(flat))
                    sample_indices = np.random.choice(len(flat), sample_size, replace=False)
                    samples = flat[sample_indices]
                    
                    # Simple edge detection
                    edges = filters.sobel(image_array)
                    edge_density = np.mean(edges > 0.1)
                    
                    # Histogram
                    hist, _ = np.histogram(flat, bins=64, range=(0, 1), density=True)
                    
                    # Update streaming stats
                    streaming_stats['count'] += 1
                    streaming_stats['intensity_sum'] += mean_val
                    streaming_stats['intensity_sum_sq'] += mean_val**2
                    streaming_stats['percentile_samples'].extend(samples[:100])  # Limit samples
                    streaming_stats['edge_density_sum'] += edge_density
                    streaming_stats['contrast_sum'] += std_val / (mean_val + 1e-8)
                    streaming_stats['histogram_accumulator'] += hist
                    
                    # Limit samples to prevent memory growth
                    if len(streaming_stats['percentile_samples']) > 20000:
                        streaming_stats['percentile_samples'] = streaming_stats['percentile_samples'][::2]
                    
                    del image, image_array, flat, samples, edges
                    
                except Exception as e:
                    print(f"Warning: Error processing image {idx}: {e}")
                    continue
            
            gc.collect()
        
        # Finalize light statistics
        return self._finalize_light_stats(streaming_stats, save_path)
    
    def _extract_medium_statistics(self, df, dataset_path, save_path):
        """Medium complexity - balanced performance and memory."""
        print("   Using MEDIUM complexity (balanced)")
        
        streaming_stats = {
            'count': 0,
            'intensity_stats': [],
            'texture_stats': [],
            'edge_stats': [],
            'frequency_stats': []
        }
        
        # Smaller batches for medium complexity
        batch_size = max(20, self.batch_size // 2)
        total_images = len(df)
        num_batches = (total_images + batch_size - 1) // batch_size
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, total_images)
            
            print(f"   Processing batch {batch_idx + 1}/{num_batches}")
            
            batch_stats = []
            
            for idx in range(start_idx, end_idx):
                try:
                    row = df.iloc[idx]
                    image_path = self._get_image_path(row, dataset_path)
                    if not os.path.exists(image_path):
                        continue
                    
                    image = Image.open(image_path).convert('L')
                    image_array = np.array(image, dtype=np.float32) / 255.0
                    
                    # Medium complexity features
                    intensity_stats = self._extract_medium_intensity(image_array)
                    texture_stats = self._extract_medium_texture(image_array)
                    edge_stats = self._extract_medium_edges(image_array)
                    frequency_stats = self._extract_medium_frequency(image_array)
                    
                    batch_stats.append({
                        'intensity': intensity_stats,
                        'texture': texture_stats,
                        'edge': edge_stats,
                        'frequency': frequency_stats
                    })
                    
                    del image, image_array
                    
                except Exception as e:
                    print(f"Warning: Error processing image {idx}: {e}")
                    continue
            
            # Aggregate batch statistics
            if batch_stats:
                streaming_stats['count'] += len(batch_stats)
                
                # Aggregate intensity stats
                intensity_means = [s['intensity']['mean'] for s in batch_stats]
                intensity_stds = [s['intensity']['std'] for s in batch_stats]
                streaming_stats['intensity_stats'].extend(list(zip(intensity_means, intensity_stds)))
                
                # Keep only recent stats to limit memory
                if len(streaming_stats['intensity_stats']) > 1000:
                    streaming_stats['intensity_stats'] = streaming_stats['intensity_stats'][-1000:]
                
                # Similar for other stats...
                texture_energies = [s['texture']['energy'] for s in batch_stats]
                streaming_stats['texture_stats'].extend(texture_energies[-100:])  # Limit
                
                edge_densities = [s['edge']['density'] for s in batch_stats]
                streaming_stats['edge_stats'].extend(edge_densities[-100:])  # Limit
                
                freq_ratios = [s['frequency']['high_low_ratio'] for s in batch_stats]
                streaming_stats['frequency_stats'].extend(freq_ratios[-100:])  # Limit
            
            del batch_stats
            gc.collect()
        
        return self._finalize_medium_stats(streaming_stats, save_path)
    
    def _extract_heavy_statistics(self, df, dataset_path, save_path):
        """Heavy complexity - full features (use only if sufficient memory)."""
        print("   Using HEAVY complexity (full features)")
        print("   WARNING: This may cause OOM on systems with <16GB RAM")
        
        # Smaller batches for heavy processing to manage memory
        heavy_batch_size = max(5, self.batch_size // 10)  # Very small batches
        
        streaming_stats = {
            'count': 0,
            'intensity_sum': 0.0,
            'intensity_sum_sq': 0.0,
            'percentile_samples': [],
            'edge_density_sum': 0.0,
            'contrast_sum': 0.0,
            'histogram_accumulator': np.zeros(256),  # Higher resolution
            
            # Heavy complexity features
            'glcm_contrast_sum': 0.0,
            'glcm_dissimilarity_sum': 0.0,
            'glcm_homogeneity_sum': 0.0,
            'glcm_energy_sum': 0.0,
            'lbp_histogram_accumulator': np.zeros(256),
            'gabor_responses_accumulator': [],
            'morphological_features_sum': np.zeros(6),  # 6 morphological features
            'frequency_features_sum': np.zeros(4),  # 4 frequency domain features
            'texture_energy_sum': 0.0,
            'local_entropy_sum': 0.0,
            'fractal_dimension_sum': 0.0,
        }
        
        total_images = len(df)
        num_batches = (total_images + heavy_batch_size - 1) // heavy_batch_size
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * heavy_batch_size
            end_idx = min(start_idx + heavy_batch_size, total_images)
            
            print(f"   Processing heavy batch {batch_idx + 1}/{num_batches}")
            
            for idx in range(start_idx, end_idx):
                try:
                    row = df.iloc[idx]
                    image_path = self._get_image_path(row, dataset_path)
                    if not os.path.exists(image_path):
                        continue
                    
                    image = Image.open(image_path).convert('L')
                    image_array = np.array(image, dtype=np.float32) / 255.0
                    
                    # Resize to manageable size for heavy processing
                    if image_array.shape[0] * image_array.shape[1] > 512 * 512:
                        image_pil = Image.fromarray((image_array * 255).astype(np.uint8))
                        image_pil = image_pil.resize((512, 512), Image.LANCZOS)
                        image_array = np.array(image_pil, dtype=np.float32) / 255.0
                    
                    # Extract heavy features
                    heavy_features = self._extract_heavy_features(image_array)
                    
                    # Accumulate basic stats
                    flat = image_array.flatten()
                    streaming_stats['count'] += 1
                    streaming_stats['intensity_sum'] += np.mean(flat)
                    streaming_stats['intensity_sum_sq'] += np.mean(flat) ** 2
                    
                    # Sample for percentiles (limit memory)
                    sample_size = min(1000, len(flat))
                    sample_indices = np.random.choice(len(flat), sample_size, replace=False)
                    samples = flat[sample_indices]
                    streaming_stats['percentile_samples'].extend(samples[:100])  # Limit samples
                    
                    # Basic features
                    edges = filters.sobel(image_array)
                    streaming_stats['edge_density_sum'] += np.mean(edges > 0.1)
                    streaming_stats['contrast_sum'] += np.std(flat)
                    
                    # High-resolution histogram
                    hist, _ = np.histogram(flat, bins=256, range=(0, 1), density=True)
                    streaming_stats['histogram_accumulator'] += hist
                    
                    # Accumulate heavy features
                    streaming_stats['glcm_contrast_sum'] += heavy_features['glcm_contrast']
                    streaming_stats['glcm_dissimilarity_sum'] += heavy_features['glcm_dissimilarity']
                    streaming_stats['glcm_homogeneity_sum'] += heavy_features['glcm_homogeneity']
                    streaming_stats['glcm_energy_sum'] += heavy_features['glcm_energy']
                    
                    streaming_stats['lbp_histogram_accumulator'] += heavy_features['lbp_histogram']
                    streaming_stats['morphological_features_sum'] += heavy_features['morphological_features']
                    streaming_stats['frequency_features_sum'] += heavy_features['frequency_features']
                    streaming_stats['texture_energy_sum'] += heavy_features['texture_energy']
                    streaming_stats['local_entropy_sum'] += heavy_features['local_entropy']
                    streaming_stats['fractal_dimension_sum'] += heavy_features['fractal_dimension']
                    
                    # Gabor responses (limit to prevent memory explosion)
                    if len(streaming_stats['gabor_responses_accumulator']) < 1000:
                        streaming_stats['gabor_responses_accumulator'].extend(heavy_features['gabor_responses'][:10])
                    
                except Exception as e:
                    print(f"   Warning: Failed to process image {idx}: {e}")
                    continue
            
            # Force garbage collection every batch
            gc.collect()
            
            # Limit percentile samples to prevent memory explosion
            if len(streaming_stats['percentile_samples']) > 50000:
                streaming_stats['percentile_samples'] = streaming_stats['percentile_samples'][:50000]
        
        return self._finalize_heavy_stats(streaming_stats, save_path)
    
    def _extract_heavy_features(self, image_array):
        """Extract comprehensive heavy features from image."""
        features = {}
        
        # Convert to 8-bit for some feature extractors
        image_8bit = (image_array * 255).astype(np.uint8)
        
        # GLCM features (Gray-Level Co-occurrence Matrix)
        try:
            from skimage.feature import graycomatrix, graycoprops
            # Reduce levels for memory efficiency
            glcm = graycomatrix(image_8bit, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], 
                               levels=16, symmetric=True, normed=True)
            features['glcm_contrast'] = np.mean(graycoprops(glcm, 'contrast'))
            features['glcm_dissimilarity'] = np.mean(graycoprops(glcm, 'dissimilarity'))
            features['glcm_homogeneity'] = np.mean(graycoprops(glcm, 'homogeneity'))
            features['glcm_energy'] = np.mean(graycoprops(glcm, 'energy'))
        except:
            features['glcm_contrast'] = 0.0
            features['glcm_dissimilarity'] = 0.0
            features['glcm_homogeneity'] = 0.0
            features['glcm_energy'] = 0.0
        
        # LBP features (Local Binary Patterns)
        try:
            from skimage.feature import local_binary_pattern
            radius = 3
            n_points = 8 * radius
            lbp = local_binary_pattern(image_array, n_points, radius, method='uniform')
            # Create histogram
            lbp_hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256), density=True)
            features['lbp_histogram'] = lbp_hist
        except:
            features['lbp_histogram'] = np.zeros(256)
        
        # Gabor filter responses
        try:
            from skimage.filters import gabor
            gabor_responses = []
            # Multiple orientations and frequencies
            for theta in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
                for frequency in [0.1, 0.3]:
                    try:
                        filtered_real, _ = gabor(image_array, frequency=frequency, theta=theta)
                        gabor_responses.append(np.mean(np.abs(filtered_real)))
                    except:
                        gabor_responses.append(0.0)
            features['gabor_responses'] = gabor_responses[:8]  # Limit to 8 responses
        except:
            features['gabor_responses'] = [0.0] * 8
        
        # Morphological features
        try:
            from skimage.morphology import opening, closing, erosion, dilation, disk
            selem = disk(3)
            opened = opening(image_8bit, selem)
            closed = closing(image_8bit, selem)
            eroded = erosion(image_8bit, selem)
            dilated = dilation(image_8bit, selem)
            
            morph_features = [
                np.mean(opened),
                np.mean(closed),
                np.mean(eroded),
                np.mean(dilated),
                np.std(opened - image_8bit),
                np.std(closed - image_8bit)
            ]
            features['morphological_features'] = np.array(morph_features)
        except:
            features['morphological_features'] = np.zeros(6)
        
        # Frequency domain features
        try:
            # FFT analysis
            fft = np.fft.fft2(image_array)
            fft_shifted = np.fft.fftshift(fft)
            magnitude_spectrum = np.abs(fft_shifted)
            
            # Frequency features
            freq_features = [
                np.mean(magnitude_spectrum),
                np.std(magnitude_spectrum),
                np.mean(magnitude_spectrum[magnitude_spectrum.shape[0]//4:3*magnitude_spectrum.shape[0]//4,
                                       magnitude_spectrum.shape[1]//4:3*magnitude_spectrum.shape[1]//4]),  # Center energy
                np.sum(magnitude_spectrum > np.percentile(magnitude_spectrum, 95))  # High frequency components
            ]
            features['frequency_features'] = np.array(freq_features)
        except:
            features['frequency_features'] = np.zeros(4)
        
        # Texture energy
        try:
            # Laws' texture measures
            # L5 = [1, 4, 6, 4, 1]  # Level
            # E5 = [-1, -2, 0, 2, 1]  # Edge
            # Simple energy measure
            grad_x = np.gradient(image_array, axis=1)
            grad_y = np.gradient(image_array, axis=0)
            texture_energy = np.mean(grad_x**2 + grad_y**2)
            features['texture_energy'] = texture_energy
        except:
            features['texture_energy'] = 0.0
        
        # Local entropy
        try:
            from skimage.filters.rank import entropy
            from skimage.morphology import disk
            local_entropy = entropy(image_8bit, disk(5))
            features['local_entropy'] = np.mean(local_entropy)
        except:
            features['local_entropy'] = 0.0
        
        # Fractal dimension (box-counting method approximation)
        try:
            def fractal_dimension(image, max_box_size=64):
                # Simple box counting approximation
                sizes = []
                counts = []
                
                for size in [2, 4, 8, 16, 32]:
                    if size > min(image.shape) // 4:
                        break
                    
                    # Threshold image
                    thresh = image > np.mean(image)
                    
                    # Count boxes containing edges
                    h, w = thresh.shape
                    count = 0
                    for i in range(0, h - size, size):
                        for j in range(0, w - size, size):
                            box = thresh[i:i+size, j:j+size]
                            if np.any(box) and not np.all(box):
                                count += 1
                    
                    if count > 0:
                        sizes.append(size)
                        counts.append(count)
                
                if len(sizes) >= 2:
                    # Fit line to log-log plot
                    log_sizes = np.log(sizes)
                    log_counts = np.log(counts)
                    slope, _ = np.polyfit(log_sizes, log_counts, 1)
                    return -slope
                else:
                    return 1.5  # Default fractal dimension
            
            features['fractal_dimension'] = fractal_dimension(image_array)
        except:
            features['fractal_dimension'] = 1.5
        
        return features
    
    def _extract_medium_intensity(self, image):
        """Extract medium complexity intensity features."""
        flat = image.flatten()
        
        # Enhanced intensity statistics
        mean_val = np.mean(flat)
        std_val = np.std(flat)
        skewness = stats.skew(flat)
        percentiles = np.percentile(flat, [10, 25, 50, 75, 90])
        
        # Local contrast analysis
        kernel_size = 5
        kernel = np.ones((kernel_size, kernel_size)) / (kernel_size**2)
        
        # Convert to uint8 for OpenCV
        image_uint8 = (image * 255).astype(np.uint8)
        local_mean = cv2.filter2D(image_uint8, -1, kernel).astype(np.float32) / 255.0
        local_contrast = np.std(image - local_mean)
        
        return {
            'mean': mean_val,
            'std': std_val,
            'skewness': skewness,
            'percentiles': percentiles.tolist(),
            'local_contrast': local_contrast
        }
    
    def _extract_medium_texture(self, image):
        """Extract medium complexity texture features."""
        # Convert to uint8 for texture analysis
        image_uint8 = (image * 255).astype(np.uint8)
        
        # Local Binary Pattern (simplified)
        lbp = feature.local_binary_pattern(image, 8, 1, method='uniform')
        lbp_hist = np.histogram(lbp, bins=10, density=True)[0]
        lbp_uniformity = np.sum(lbp_hist**2)
        
        # Texture energy using Laplacian
        laplacian = cv2.Laplacian(image_uint8, cv2.CV_64F)
        texture_energy = np.var(laplacian)
        
        return {
            'energy': texture_energy,
            'lbp_uniformity': lbp_uniformity,
            'lbp_hist_std': np.std(lbp_hist)
        }
    
    def _extract_medium_edges(self, image):
        """Extract medium complexity edge features."""
        # Multiple edge detection methods
        edges_sobel = filters.sobel(image)
        edges_canny = feature.canny(image, sigma=1.0)
        
        # Edge statistics
        edge_density = np.mean(edges_canny)
        edge_strength = np.mean(edges_sobel)
        edge_strength_std = np.std(edges_sobel)
        
        # Edge orientation analysis (simplified)
        gradient_x = filters.sobel_h(image)
        gradient_y = filters.sobel_v(image)
        edge_angles = np.arctan2(gradient_y, gradient_x)
        
        # Dominant orientation
        angle_hist = np.histogram(edge_angles, bins=8, range=(-np.pi, np.pi))[0]
        dominant_orientation = np.argmax(angle_hist)
        
        return {
            'density': edge_density,
            'strength_mean': edge_strength,
            'strength_std': edge_strength_std,
            'dominant_orientation': dominant_orientation
        }
    
    def _extract_medium_frequency(self, image):
        """Extract medium complexity frequency features."""
        # FFT analysis
        fft = np.fft.fft2(image)
        fft_shift = np.fft.fftshift(fft)
        magnitude = np.abs(fft_shift)
        
        # Frequency analysis
        center = np.array(image.shape) // 2
        y, x = np.ogrid[:image.shape[0], :image.shape[1]]
        r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
        
        max_r = min(center)
        low_freq = np.mean(magnitude[r < max_r//4])
        high_freq = np.mean(magnitude[r > max_r//2])
        
        high_low_ratio = high_freq / (low_freq + 1e-8)
        
        return {
            'high_low_ratio': high_low_ratio,
            'dc_component': magnitude[center[0], center[1]]
        }
    
    def apply_medical_style_transfer(self, source_image_path, target_stats, output_path):
        """Apply style transfer based on complexity level."""
        source_image = Image.open(source_image_path).convert('L')
        # CRITICAL: Resize to 256x256 for consistency with BUSI dataset
        source_image = source_image.resize((256, 256), Image.LANCZOS)
        source_array = np.array(source_image, dtype=np.float32) / 255.0
        
        if self.complexity == 'light':
            styled_image = self._light_style_transfer(source_array, target_stats)
        elif self.complexity == 'medium':
            styled_image = self._medium_style_transfer(source_array, target_stats)
        else:  # heavy
            styled_image = self._heavy_style_transfer(source_array, target_stats)
        
        # Convert and save
        styled_uint8 = np.clip(styled_image * 255, 0, 255).astype(np.uint8)
        styled_rgb = cv2.cvtColor(styled_uint8, cv2.COLOR_GRAY2RGB)
        styled_pil = Image.fromarray(styled_rgb)
        styled_pil.save(output_path)
        
        # Cleanup
        del source_image, source_array, styled_image, styled_uint8, styled_rgb, styled_pil
        return True
    
    def _light_style_transfer(self, source_image, target_stats):
        """Light complexity style transfer."""
        # Simple intensity matching
        target_intensity = target_stats['intensity_distributions'][0]
        target_mean = target_intensity['mean']
        target_std = target_intensity['std']
        
        source_mean = np.mean(source_image)
        source_std = np.std(source_image)
        
        if source_std > 1e-8:
            normalized = (source_image - source_mean) / source_std
            styled = normalized * target_std + target_mean
        else:
            styled = source_image + (target_mean - source_mean)
        
        return np.clip(styled, 0, 1)
    
    def _medium_style_transfer(self, source_image, target_stats):
        """Medium complexity style transfer with better quality."""
        # Enhanced style transfer with texture and edge preservation
        
        # Step 1: Intensity matching
        target_intensity = target_stats['intensity_distributions'][0]
        target_mean = target_intensity['mean']
        target_std = target_intensity['std']
        
        # Step 2: Percentile-based matching
        if 'percentiles' in target_intensity:
            target_percentiles = np.array(target_intensity['percentiles'])
            source_percentiles = np.percentile(source_image, [10, 25, 50, 75, 90])
            
            # Create mapping
            styled = source_image.copy()
            for i, (src_pct, tgt_pct) in enumerate(zip(source_percentiles, target_percentiles)):
                if i == 0:
                    mask = source_image <= src_pct
                    styled[mask] = tgt_pct
                else:
                    prev_src = source_percentiles[i-1]
                    prev_tgt = target_percentiles[i-1]
                    mask = (source_image > prev_src) & (source_image <= src_pct)
                    if np.any(mask):
                        alpha = (source_image[mask] - prev_src) / (src_pct - prev_src + 1e-8)
                        styled[mask] = prev_tgt + alpha * (tgt_pct - prev_tgt)
        else:
            # Fallback to simple matching
            source_mean = np.mean(source_image)
            source_std = np.std(source_image)
            
            if source_std > 1e-8:
                normalized = (source_image - source_mean) / source_std
                styled = normalized * target_std + target_mean
            else:
                styled = source_image + (target_mean - source_mean)
        
        # Step 3: Edge-preserving smoothing
        styled_uint8 = np.clip(styled * 255, 0, 255).astype(np.uint8)
        smoothed = cv2.bilateralFilter(styled_uint8, 5, 50, 50)
        
        return smoothed.astype(np.float32) / 255.0
    
    def _heavy_style_transfer(self, source_image, target_stats):
        """Heavy complexity style transfer with comprehensive feature matching."""
        # Advanced style transfer using full feature set
        
        # Step 1: Enhanced intensity matching using high-resolution histogram
        target_intensity = target_stats['intensity_distributions'][0]
        target_mean = target_intensity['mean']
        target_std = target_intensity['std']
        target_histogram = np.array(target_intensity['histogram'])
        
        # High-resolution histogram matching
        source_hist, bins = np.histogram(source_image, bins=256, range=(0, 1), density=True)
        
        # Create CDF for histogram matching
        source_cdf = np.cumsum(source_hist)
        target_cdf = np.cumsum(target_histogram)
        
        # Normalize CDFs
        source_cdf = source_cdf / source_cdf[-1]
        target_cdf = target_cdf / target_cdf[-1]
        
        # Create mapping using inverse CDF
        mapping = np.interp(source_cdf, target_cdf, np.linspace(0, 1, 256))
        
        # Apply histogram matching
        source_indices = np.digitize(source_image, bins[:-1]) - 1
        source_indices = np.clip(source_indices, 0, 255)
        styled = mapping[source_indices]
        
        # Step 2: Texture feature matching using GLCM properties
        target_edge = target_stats['edge_characteristics'][0]
        target_glcm_contrast = target_edge['avg_glcm_contrast']
        target_glcm_homogeneity = target_edge['avg_glcm_homogeneity']
        
        # Extract current GLCM features
        current_features = self._extract_heavy_features(styled)
        current_contrast = current_features['glcm_contrast']
        current_homogeneity = current_features['glcm_homogeneity']
        
        # Adjust contrast to match target
        if current_contrast > 1e-8:
            contrast_ratio = target_glcm_contrast / (current_contrast + 1e-8)
            # Apply adaptive contrast enhancement
            mean_val = np.mean(styled)
            styled = mean_val + (styled - mean_val) * contrast_ratio
        
        # Step 3: Morphological feature matching
        target_freq = target_stats['frequency_features'][0]
        target_morph = np.array(target_freq['avg_morphological_features'])
        
        # Apply morphological operations to match target characteristics
        styled_uint8 = np.clip(styled * 255, 0, 255).astype(np.uint8)
        
        # Adaptive morphological operations based on target features
        from skimage.morphology import opening, closing, disk
        
        # Calculate optimal kernel size based on target morphological features
        kernel_size = max(1, int(np.mean(target_morph[:2]) / 50))  # Scale based on target
        kernel_size = min(kernel_size, 5)  # Limit to reasonable size
        
        selem = disk(kernel_size)
        
        # Balance opening and closing based on target features
        opening_weight = target_morph[0] / (target_morph[0] + target_morph[1] + 1e-8)
        closing_weight = 1 - opening_weight
        
        if opening_weight > 0.5:
            morphed = opening(styled_uint8, selem)
            styled_uint8 = (opening_weight * morphed + (1 - opening_weight) * styled_uint8).astype(np.uint8)
        else:
            morphed = closing(styled_uint8, selem)
            styled_uint8 = (closing_weight * morphed + (1 - closing_weight) * styled_uint8).astype(np.uint8)
        
        # Step 4: Frequency domain enhancement
        target_freq_features = np.array(target_freq['avg_frequency_features'])
        
        # FFT-based enhancement
        fft = np.fft.fft2(styled_uint8.astype(np.float32))
        fft_shifted = np.fft.fftshift(fft)
        magnitude = np.abs(fft_shifted)
        phase = np.angle(fft_shifted)
        
        # Enhance frequency components based on target characteristics
        h, w = magnitude.shape
        center_x, center_y = h // 2, w // 2
        
        # Create frequency mask based on target features
        y, x = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        
        # Adjust high frequencies based on target
        high_freq_enhancement = target_freq_features[3] / (np.mean(magnitude) + 1e-8)
        high_freq_enhancement = np.clip(high_freq_enhancement, 0.5, 2.0)
        
        freq_mask = np.ones_like(magnitude)
        freq_mask[dist_from_center > min(h, w) * 0.3] *= high_freq_enhancement
        
        enhanced_magnitude = magnitude * freq_mask
        enhanced_fft = enhanced_magnitude * np.exp(1j * phase)
        enhanced_fft_shifted = np.fft.ifftshift(enhanced_fft)
        enhanced_image = np.real(np.fft.ifft2(enhanced_fft_shifted))
        
        # Step 5: Texture energy preservation
        target_texture = target_stats['texture_features'][0]
        target_energy = target_texture['avg_energy']
        
        # Calculate current texture energy
        grad_x = np.gradient(enhanced_image, axis=1)
        grad_y = np.gradient(enhanced_image, axis=0)
        current_energy = np.mean(grad_x**2 + grad_y**2)
        
        # Adjust to match target texture energy
        if current_energy > 1e-8:
            energy_ratio = np.sqrt(target_energy / (current_energy + 1e-8))
            energy_ratio = np.clip(energy_ratio, 0.5, 2.0)
            
            # Apply edge-preserving enhancement
            enhanced_edges = cv2.bilateralFilter(
                enhanced_image.astype(np.uint8), 
                9, 
                int(75 * energy_ratio), 
                int(75 * energy_ratio)
            )
            enhanced_image = enhanced_edges
        
        # Step 6: Local entropy matching
        target_entropy = target_texture['avg_local_entropy']
        
        # Apply adaptive histogram equalization if needed
        if target_entropy > 50:  # High entropy target
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced_image = clahe.apply(enhanced_image.astype(np.uint8))
        
        # Step 7: Final normalization and clamping
        final_image = enhanced_image.astype(np.float32) / 255.0
        final_image = np.clip(final_image, 0, 1)
        
        # Ensure target mean and std are approximately maintained
        current_mean = np.mean(final_image)
        current_std = np.std(final_image)
        
        if current_std > 1e-8:
            final_image = (final_image - current_mean) / current_std
            final_image = final_image * target_std + target_mean
            final_image = np.clip(final_image, 0, 1)
        
        return final_image
    
    # Helper methods and finalization functions...
    def _finalize_light_stats(self, streaming_stats, save_path):
        """Finalize light complexity statistics."""
        count = streaming_stats['count']
        if count == 0:
            raise ValueError("No valid images processed")
        
        mean_intensity = streaming_stats['intensity_sum'] / count
        std_intensity = np.sqrt(streaming_stats['intensity_sum_sq'] / count - mean_intensity**2)
        
        if streaming_stats['percentile_samples']:
            percentiles = np.percentile(streaming_stats['percentile_samples'], [10, 25, 50, 75, 90])
        else:
            percentiles = [0.2, 0.3, 0.5, 0.7, 0.8]
        
        histogram = streaming_stats['histogram_accumulator'] / count
        
        final_stats = {
            'processed_images': count,
            'intensity_distributions': [{
                'mean': mean_intensity,
                'std': std_intensity,
                'percentiles': percentiles.tolist(),
                'histogram': histogram.tolist(),
                'contrast': streaming_stats['contrast_sum'] / count
            }]
        }
        
        if save_path:
            self._save_statistics(final_stats, save_path)
        
        return final_stats
    
    def _finalize_medium_stats(self, streaming_stats, save_path):
        """Finalize medium complexity statistics."""
        count = streaming_stats['count']
        if count == 0:
            raise ValueError("No valid images processed")
        
        # Aggregate medium stats
        if streaming_stats['intensity_stats']:
            means, stds = zip(*streaming_stats['intensity_stats'])
            avg_mean = np.mean(means)
            avg_std = np.mean(stds)
        else:
            avg_mean, avg_std = 0.5, 0.2
        
        final_stats = {
            'processed_images': count,
            'intensity_distributions': [{
                'mean': avg_mean,
                'std': avg_std,
                'percentiles': [0.2, 0.3, 0.5, 0.7, 0.8]  # Default
            }],
            'texture_features': [{
                'avg_energy': np.mean(streaming_stats['texture_stats']) if streaming_stats['texture_stats'] else 0.5
            }],
            'edge_characteristics': [{
                'avg_density': np.mean(streaming_stats['edge_stats']) if streaming_stats['edge_stats'] else 0.3
            }]
        }
        
        if save_path:
            self._save_statistics(final_stats, save_path)
        
        return final_stats
    
    def _finalize_heavy_stats(self, streaming_stats, save_path):
        """Finalize heavy complexity statistics."""
        count = streaming_stats['count']
        if count == 0:
            raise ValueError("No valid images processed")
        
        # Aggregate heavy stats
        mean_intensity = streaming_stats['intensity_sum'] / count
        std_intensity = np.sqrt(streaming_stats['intensity_sum_sq'] / count - mean_intensity**2)
        
        if streaming_stats['percentile_samples']:
            percentiles = np.percentile(streaming_stats['percentile_samples'], [10, 25, 50, 75, 90])
        else:
            percentiles = [0.2, 0.3, 0.5, 0.7, 0.8]
        
        histogram = streaming_stats['histogram_accumulator'] / count
        
        # Aggregate heavy features
        avg_glcm_contrast = streaming_stats['glcm_contrast_sum'] / count
        avg_glcm_dissimilarity = streaming_stats['glcm_dissimilarity_sum'] / count
        avg_glcm_homogeneity = streaming_stats['glcm_homogeneity_sum'] / count
        avg_glcm_energy = streaming_stats['glcm_energy_sum'] / count
        
        avg_lbp_histogram = streaming_stats['lbp_histogram_accumulator'] / count
        avg_morphological_features = streaming_stats['morphological_features_sum'] / count
        avg_frequency_features = streaming_stats['frequency_features_sum'] / count
        avg_texture_energy = streaming_stats['texture_energy_sum'] / count
        avg_local_entropy = streaming_stats['local_entropy_sum'] / count
        avg_fractal_dimension = streaming_stats['fractal_dimension_sum'] / count
        
        final_stats = {
            'processed_images': count,
            'intensity_distributions': [{
                'mean': mean_intensity,
                'std': std_intensity,
                'percentiles': percentiles.tolist(),
                'histogram': histogram.tolist(),
                'contrast': streaming_stats['contrast_sum'] / count
            }],
            'texture_features': [{
                'avg_energy': avg_texture_energy,
                'avg_local_entropy': avg_local_entropy,
                'avg_fractal_dimension': avg_fractal_dimension
            }],
            'edge_characteristics': [{
                'avg_density': streaming_stats['edge_density_sum'] / count,
                'avg_glcm_contrast': avg_glcm_contrast,
                'avg_glcm_dissimilarity': avg_glcm_dissimilarity,
                'avg_glcm_homogeneity': avg_glcm_homogeneity,
                'avg_glcm_energy': avg_glcm_energy
            }],
            'frequency_features': [{
                'avg_lbp_histogram_std': np.std(avg_lbp_histogram),
                'avg_morphological_features': avg_morphological_features.tolist(),
                'avg_frequency_features': avg_frequency_features.tolist()
            }]
        }
        
        if save_path:
            self._save_statistics(final_stats, save_path)
        
        return final_stats
    
    def _get_image_path(self, row, dataset_path):
        """Helper to get image path from dataframe row."""
        image_path_info = row['image_path']
        
        if '/' in image_path_info:
            return os.path.join(dataset_path, image_path_info)
        else:
            class_type = image_path_info.split()[0]
            
            image_path_busi = os.path.join(dataset_path, class_type, 'image', image_path_info)
            if os.path.exists(image_path_busi):
                return image_path_busi
            
            actual_filename = ' '.join(image_path_info.split()[1:])
            image_path_busuclm = os.path.join(dataset_path, class_type, 'images', actual_filename)
            return image_path_busuclm if os.path.exists(image_path_busuclm) else image_path_busi
    
    def _save_statistics(self, stats, save_path):
        """Save statistics to file."""
        def convert_numpy_types(obj):
            """Convert numpy types to native Python types for JSON serialization."""
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        # Convert all numpy types to JSON-serializable types
        serializable_stats = convert_numpy_types(stats)
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(serializable_stats, f, indent=2)
        print(f"💾 Statistics saved to {save_path}")


# Convenience functions for different complexity levels
def run_light_medical_pipeline():
    """Run light complexity (memory-efficient) pipeline."""
    extractor = HybridMedicalStyleTransfer(complexity='light', batch_size=50)
    stats = extractor.extract_medical_domain_statistics(
        'dataset/BioMedicalDataset/BUSI',
        'train_frame.csv',
        'privacy_style_stats/busi_light_stats.json'
    )
    return stats

def run_medium_medical_pipeline():
    """Run medium complexity (balanced) pipeline."""
    extractor = HybridMedicalStyleTransfer(complexity='medium', batch_size=25)
    stats = extractor.extract_medical_domain_statistics(
        'dataset/BioMedicalDataset/BUSI',
        'train_frame.csv',
        'privacy_style_stats/busi_medium_stats.json'
    )
    return stats

def run_heavy_medical_pipeline():
    """Run heavy complexity (full features) pipeline."""
    extractor = HybridMedicalStyleTransfer(complexity='heavy', batch_size=10)
    stats = extractor.extract_medical_domain_statistics(
        'dataset/BioMedicalDataset/BUSI',
        'train_frame.csv',
        'privacy_style_stats/busi_heavy_stats.json'
    )
    return stats 