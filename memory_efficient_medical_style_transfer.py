"""
Memory-Efficient Medical Privacy-Preserving Style Transfer
Optimized for large datasets with minimal memory footprint.
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

class MemoryEfficientMedicalStyleTransfer:
    """
    Memory-efficient medical style transfer using streaming statistics.
    Processes images in batches to avoid OOM errors.
    """
    
    def __init__(self, batch_size=50):
        self.batch_size = batch_size
        print(f"🏥 Memory-Efficient Medical Style Transfer initialized")
        print(f"   Batch size: {batch_size} images")
    
    def extract_medical_domain_statistics(self, dataset_path, csv_file, save_path=None):
        """
        Extract medical statistics using streaming computation to minimize memory usage.
        """
        print(f"🔬 Extracting medical statistics (memory-efficient) from {dataset_path}")
        
        # Load dataset
        csv_path = os.path.join(dataset_path, csv_file)
        df = pd.read_csv(csv_path)
        
        # Initialize streaming statistics accumulators
        streaming_stats = {
            'count': 0,
            'intensity_sum': 0.0,
            'intensity_sum_sq': 0.0,
            'percentile_samples': [],
            'edge_density_sum': 0.0,
            'texture_energy_sum': 0.0,
            'contrast_sum': 0.0,
            'histogram_accumulator': np.zeros(64),
            'gradient_stats': {'mean_sum': 0.0, 'std_sum': 0.0}
        }
        
        # Process in batches to manage memory
        total_images = len(df)
        num_batches = (total_images + self.batch_size - 1) // self.batch_size
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, total_images)
            
            print(f"   Processing batch {batch_idx + 1}/{num_batches} (images {start_idx}-{end_idx-1})")
            
            batch_stats = self._process_batch(df[start_idx:end_idx], dataset_path)
            self._update_streaming_stats(streaming_stats, batch_stats)
            
            # Force garbage collection after each batch
            gc.collect()
        
        # Finalize statistics
        final_stats = self._finalize_streaming_stats(streaming_stats)
        
        if save_path:
            self._save_statistics(final_stats, save_path)
        
        print(f"✅ Medical statistics extracted from {streaming_stats['count']} images")
        print(f"   Memory-efficient processing completed")
        
        return final_stats
    
    def _process_batch(self, batch_df, dataset_path):
        """Process a batch of images and return aggregated statistics"""
        batch_stats = {
            'intensity_values': [],
            'edge_densities': [],
            'texture_energies': [],
            'contrasts': [],
            'histograms': [],
            'gradient_means': [],
            'gradient_stds': []
        }
        
        for idx, row in batch_df.iterrows():
            try:
                # Load image
                image_path = self._get_image_path(row, dataset_path)
                if not os.path.exists(image_path):
                    continue
                
                image = Image.open(image_path).convert('L')
                image_array = np.array(image, dtype=np.float32) / 255.0
                
                # Extract lightweight statistics only
                intensity_stats = self._extract_lightweight_intensity(image_array)
                edge_stats = self._extract_lightweight_edges(image_array)
                texture_stats = self._extract_lightweight_texture(image_array)
                
                # Accumulate statistics
                batch_stats['intensity_values'].extend(intensity_stats['sample_values'])
                batch_stats['edge_densities'].append(edge_stats['density'])
                batch_stats['texture_energies'].append(texture_stats['energy'])
                batch_stats['contrasts'].append(intensity_stats['contrast'])
                batch_stats['histograms'].append(intensity_stats['histogram'])
                batch_stats['gradient_means'].append(edge_stats['gradient_mean'])
                batch_stats['gradient_stds'].append(edge_stats['gradient_std'])
                
                # Clear image from memory immediately
                del image, image_array
                
            except Exception as e:
                print(f"Warning: Error processing image {idx}: {e}")
                continue
        
        return batch_stats
    
    def _extract_lightweight_intensity(self, image):
        """Extract essential intensity statistics with minimal memory usage"""
        flat = image.flatten()
        
        # Sample subset for percentile calculation (memory efficient)
        sample_size = min(10000, len(flat))
        sample_indices = np.random.choice(len(flat), sample_size, replace=False)
        sample_values = flat[sample_indices]
        
        # Basic statistics
        mean_val = np.mean(flat)
        std_val = np.std(flat)
        contrast = std_val / (mean_val + 1e-8)
        
        # Lightweight histogram
        hist, _ = np.histogram(flat, bins=64, range=(0, 1), density=True)
        
        return {
            'sample_values': sample_values.tolist(),
            'contrast': contrast,
            'histogram': hist
        }
    
    def _extract_lightweight_edges(self, image):
        """Extract essential edge statistics with minimal memory usage"""
        # Simple edge detection
        edges = filters.sobel(image)
        
        # Basic edge statistics
        edge_density = np.mean(edges > 0.1)
        gradient_mean = np.mean(edges)
        gradient_std = np.std(edges)
        
        return {
            'density': edge_density,
            'gradient_mean': gradient_mean,
            'gradient_std': gradient_std
        }
    
    def _extract_lightweight_texture(self, image):
        """Extract essential texture statistics with minimal memory usage"""
        # Simple texture energy calculation
        # Using a smaller kernel for memory efficiency
        kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        texture_response = cv2.filter2D((image * 255).astype(np.uint8), -1, kernel)
        
        # Texture energy
        energy = np.mean(texture_response.astype(np.float32) ** 2)
        
        return {
            'energy': energy
        }
    
    def _update_streaming_stats(self, streaming_stats, batch_stats):
        """Update streaming statistics with batch results"""
        batch_count = len(batch_stats['edge_densities'])
        
        if batch_count > 0:
            streaming_stats['count'] += batch_count
            
            # Update intensity statistics
            if batch_stats['intensity_values']:
                intensity_values = np.array(batch_stats['intensity_values'])
                streaming_stats['intensity_sum'] += np.sum(intensity_values)
                streaming_stats['intensity_sum_sq'] += np.sum(intensity_values ** 2)
                
                # Keep a sample for percentile calculation (limit memory)
                new_samples = intensity_values[::max(1, len(intensity_values)//1000)]  # Subsample
                streaming_stats['percentile_samples'].extend(new_samples)
                
                # Limit percentile samples to prevent memory growth
                if len(streaming_stats['percentile_samples']) > 50000:
                    # Keep random subset
                    indices = np.random.choice(len(streaming_stats['percentile_samples']), 50000, replace=False)
                    streaming_stats['percentile_samples'] = [streaming_stats['percentile_samples'][i] for i in indices]
            
            # Update other statistics
            streaming_stats['edge_density_sum'] += np.sum(batch_stats['edge_densities'])
            streaming_stats['texture_energy_sum'] += np.sum(batch_stats['texture_energies'])
            streaming_stats['contrast_sum'] += np.sum(batch_stats['contrasts'])
            streaming_stats['gradient_stats']['mean_sum'] += np.sum(batch_stats['gradient_means'])
            streaming_stats['gradient_stats']['std_sum'] += np.sum(batch_stats['gradient_stds'])
            
            # Update histogram accumulator
            for hist in batch_stats['histograms']:
                streaming_stats['histogram_accumulator'] += hist
    
    def _finalize_streaming_stats(self, streaming_stats):
        """Convert streaming statistics to final domain statistics"""
        count = streaming_stats['count']
        
        if count == 0:
            raise ValueError("No valid images processed")
        
        # Compute final statistics
        intensity_mean = streaming_stats['intensity_sum'] / count
        intensity_variance = (streaming_stats['intensity_sum_sq'] / count) - (intensity_mean ** 2)
        intensity_std = np.sqrt(max(0, intensity_variance))
        
        # Compute percentiles from samples
        if streaming_stats['percentile_samples']:
            percentiles = np.percentile(streaming_stats['percentile_samples'], [5, 25, 50, 75, 95])
        else:
            percentiles = [0.2, 0.4, 0.5, 0.6, 0.8]  # Default values
        
        # Normalize histogram
        histogram = streaming_stats['histogram_accumulator'] / count
        
        # Finalize statistics
        final_stats = {
            'processed_images': count,
            'intensity_distributions': [{
                'mean': intensity_mean,
                'std': intensity_std,
                'percentiles': percentiles.tolist(),
                'histogram': histogram.tolist(),
                'contrast': streaming_stats['contrast_sum'] / count
            }],
            'edge_characteristics': [{
                'edge_density': streaming_stats['edge_density_sum'] / count,
                'gradient_mean': streaming_stats['gradient_stats']['mean_sum'] / count,
                'gradient_std': streaming_stats['gradient_stats']['std_sum'] / count
            }],
            'texture_features': [{
                'energy': streaming_stats['texture_energy_sum'] / count
            }]
        }
        
        return final_stats
    
    def apply_medical_style_transfer(self, source_image_path, target_stats, output_path, method='efficient_adaptive'):
        """
        Apply memory-efficient medical style transfer.
        """
        # Load source image
        source_image = Image.open(source_image_path).convert('L')
        source_array = np.array(source_image, dtype=np.float32) / 255.0
        
        # Apply efficient style transfer
        styled_image = self._efficient_adaptive_transfer(source_array, target_stats)
        
        # Convert and save
        styled_uint8 = np.clip(styled_image * 255, 0, 255).astype(np.uint8)
        styled_rgb = cv2.cvtColor(styled_uint8, cv2.COLOR_GRAY2RGB)
        styled_pil = Image.fromarray(styled_rgb)
        styled_pil.save(output_path)
        
        # Clean up memory
        del source_image, source_array, styled_image, styled_uint8, styled_rgb, styled_pil
        
        return True
    
    def _efficient_adaptive_transfer(self, source_image, target_stats):
        """
        Efficient adaptive style transfer with minimal memory usage.
        """
        # Get target characteristics
        target_intensity = target_stats['intensity_distributions'][0]
        target_mean = target_intensity['mean']
        target_std = target_intensity['std']
        target_percentiles = np.array(target_intensity['percentiles'])
        
        # Step 1: Basic intensity matching
        source_mean = np.mean(source_image)
        source_std = np.std(source_image)
        
        # Normalize and scale
        if source_std > 1e-8:
            normalized = (source_image - source_mean) / source_std
            scaled = normalized * target_std + target_mean
        else:
            scaled = source_image + (target_mean - source_mean)
        
        # Step 2: Percentile-based refinement (memory efficient)
        source_percentiles = np.percentile(scaled, [5, 25, 50, 75, 95])
        
        # Create simple mapping
        result = scaled.copy()
        for i, (src_pct, tgt_pct) in enumerate(zip(source_percentiles, target_percentiles)):
            if i == 0:  # 5th percentile
                mask = scaled <= src_pct
                result[mask] = tgt_pct
            elif i == len(source_percentiles) - 1:  # 95th percentile
                mask = scaled >= src_pct
                result[mask] = tgt_pct
            else:  # Interpolate between percentiles
                prev_src = source_percentiles[i-1]
                prev_tgt = target_percentiles[i-1]
                mask = (scaled > prev_src) & (scaled <= src_pct)
                if np.any(mask):
                    # Linear interpolation
                    alpha = (scaled[mask] - prev_src) / (src_pct - prev_src + 1e-8)
                    result[mask] = prev_tgt + alpha * (tgt_pct - prev_tgt)
        
        # Step 3: Light smoothing (memory efficient)
        result_uint8 = np.clip(result * 255, 0, 255).astype(np.uint8)
        smoothed = cv2.medianBlur(result_uint8, 3)  # Light smoothing
        
        return smoothed.astype(np.float32) / 255.0
    
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
    
    def _save_statistics(self, stats, save_path):
        """Save statistics to file"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"💾 Medical statistics saved to {save_path}")


def generate_memory_efficient_styled_dataset(source_dataset_path, source_csv, target_stats_path, output_dir):
    """
    Generate styled dataset using memory-efficient processing.
    """
    print(f"🏥 Generating memory-efficient styled dataset")
    
    # Initialize memory-efficient style transfer
    style_transfer = MemoryEfficientMedicalStyleTransfer(batch_size=50)
    
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
    
    # Process images one by one to minimize memory usage
    for idx in tqdm(range(len(df)), desc="Generating styled images"):
        try:
            row = df.iloc[idx]
            image_path_info = row['image_path']
            mask_path_info = row['mask_path']
            
            # Determine class
            class_type = image_path_info.split()[0]
            
            # Get source paths
            source_image_path = style_transfer._get_image_path(row, source_dataset_path)
            
            # Generate output paths
            output_filename = f"efficient_styled_{os.path.basename(image_path_info)}"
            output_mask_filename = f"efficient_styled_{os.path.basename(mask_path_info)}"
            
            output_image_path = os.path.join(output_dir, class_type, 'image', output_filename)
            output_mask_path = os.path.join(output_dir, class_type, 'mask', output_mask_filename)
            
            # Apply style transfer
            style_transfer.apply_medical_style_transfer(
                source_image_path, target_stats, output_image_path
            )
            
            # Process mask efficiently
            mask_source_path = source_image_path.replace('/image/', '/images/').replace('/images/', '/masks/')
            mask_filename = os.path.basename(mask_path_info)
            if ' ' in mask_filename:
                mask_filename = mask_filename.replace(' ', '_')
            mask_source_path = os.path.join(os.path.dirname(mask_source_path), mask_filename)
            
            if os.path.exists(mask_source_path):
                # Process mask with minimal memory
                mask = cv2.imread(mask_source_path)
                if mask is not None:
                    mask_resized = cv2.resize(mask, (256, 256))
                    mask_gray = cv2.cvtColor(mask_resized, cv2.COLOR_BGR2GRAY) if len(mask_resized.shape) == 3 else mask_resized
                    _, mask_binary = cv2.threshold(mask_gray, 127, 255, cv2.THRESH_BINARY)
                    mask_rgb = cv2.cvtColor(mask_binary, cv2.COLOR_GRAY2BGR)
                    cv2.imwrite(output_mask_path, mask_rgb)
                    
                    # Clean up mask memory
                    del mask, mask_resized, mask_gray, mask_binary, mask_rgb
            
            styled_samples.append({
                'image_path': f"{class_type} {output_filename}",
                'mask_path': f"{class_type} {output_mask_filename}",
                'class': class_type,
                'source_client': 'BUS-UCLM',
                'style_client': 'BUSI', 
                'augmentation_type': 'efficient_styled'
            })
            
            # Force garbage collection every 10 images
            if idx % 10 == 0:
                gc.collect()
            
        except Exception as e:
            print(f"Warning: Error processing {idx}: {e}")
            continue
    
    # Save dataset CSV
    styled_csv_path = os.path.join(output_dir, 'efficient_styled_dataset.csv')
    styled_df = pd.DataFrame(styled_samples)
    styled_df.to_csv(styled_csv_path, index=False)
    
    print(f"✅ Generated {len(styled_samples)} memory-efficient styled images")
    print(f"📁 Output directory: {output_dir}")
    print(f"📄 CSV file: {styled_csv_path}")
    
    return styled_samples


if __name__ == "__main__":
    # Test the memory-efficient system
    print("🧪 Testing Memory-Efficient Medical Style Transfer")
    
    # Extract medical BUSI statistics
    extractor = MemoryEfficientMedicalStyleTransfer(batch_size=50)
    stats = extractor.extract_medical_domain_statistics(
        'dataset/BioMedicalDataset/BUSI',
        'train_frame.csv',
        'privacy_style_stats/busi_efficient_stats.json'
    )
    
    # Generate styled dataset
    generate_memory_efficient_styled_dataset(
        'dataset/BioMedicalDataset/BUS-UCLM',
        'train_frame.csv',
        'privacy_style_stats/busi_efficient_stats.json',
        'dataset/BioMedicalDataset/BUS-UCLM-Efficient-Styled'
    ) 