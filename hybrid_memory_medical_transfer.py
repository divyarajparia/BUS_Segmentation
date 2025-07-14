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
        
        # Even smaller batches for heavy processing
        batch_size = max(10, self.batch_size // 5)
        
        # This would contain the full complex feature extraction
        # For now, fall back to medium to prevent OOM
        print("   Falling back to MEDIUM complexity for safety")
        return self._extract_medium_statistics(df, dataset_path, save_path)
    
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
        source_array = np.array(source_image, dtype=np.float32) / 255.0
        
        if self.complexity == 'light':
            styled_image = self._light_style_transfer(source_array, target_stats)
        elif self.complexity == 'medium':
            styled_image = self._medium_style_transfer(source_array, target_stats)
        else:  # heavy
            styled_image = self._medium_style_transfer(source_array, target_stats)  # Fallback
        
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
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(stats, f, indent=2)
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