#!/usr/bin/env python3
"""
Style Transfer Quality Analysis
==============================
Analyzes the quality and effectiveness of the reverse style transfer pipeline.
"""

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from skimage.metrics import structural_similarity as ssim
from skimage.feature import local_binary_pattern
from skimage.filters import sobel
import json
from pathlib import Path
import random
from typing import Dict, List, Tuple

class StyleTransferAnalyzer:
    """Analyzes style transfer quality and effectiveness."""
    
    def __init__(self):
        self.busi_path = "dataset/BioMedicalDataset/BUSI"
        self.bus_uclm_path = "dataset/BioMedicalDataset/BUS-UCLM"
        self.styled_path = "dataset/BioMedicalDataset/BUSI-BUS-UCLM-Styled"
        self.results = {}
        
    def load_random_samples(self, n_samples: int = 10) -> Dict:
        """Load random samples from each dataset."""
        samples = {
            'original_busi': [],
            'styled_busi': [],
            'real_bus_uclm': []
        }
        
        # Load original BUSI samples
        busi_train = pd.read_csv(f"{self.busi_path}/train_frame.csv")
        for _, row in busi_train.sample(min(n_samples, len(busi_train))).iterrows():
            # Extract class from filename
            class_name = 'benign' if 'benign' in row['image_path'] else 'malignant'
            
            # Construct proper paths with subdirectories
            img_path = os.path.join(self.busi_path, class_name, 'image', row['image_path'])
            mask_path = os.path.join(self.busi_path, class_name, 'mask', row['mask_path'])
            
            if os.path.exists(img_path) and os.path.exists(mask_path):
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if img is not None and mask is not None:
                    samples['original_busi'].append({
                        'image': img,
                        'mask': mask,
                        'path': img_path,
                        'class': class_name
                    })
        
        # Load styled BUSI samples
        styled_csv = f"{self.styled_path}/styled_dataset.csv"
        if os.path.exists(styled_csv):
            styled_df = pd.read_csv(styled_csv)
            for _, row in styled_df.sample(min(n_samples, len(styled_df))).iterrows():
                class_name = row['class']
                
                # Strip class prefix from filename (CSV has "benign styled_benign.png" but file is "styled_benign.png")
                # Be careful to only remove the prefix, not all instances
                actual_img_filename = row['image_path']
                actual_mask_filename = row['mask_path']
                
                if actual_img_filename.startswith('benign '):
                    actual_img_filename = actual_img_filename[7:]  # Remove "benign "
                elif actual_img_filename.startswith('malignant '):
                    actual_img_filename = actual_img_filename[10:]  # Remove "malignant "
                
                if actual_mask_filename.startswith('benign '):
                    actual_mask_filename = actual_mask_filename[7:]  # Remove "benign "
                elif actual_mask_filename.startswith('malignant '):
                    actual_mask_filename = actual_mask_filename[10:]  # Remove "malignant "
                
                # Construct proper paths with subdirectories
                img_path = os.path.join(self.styled_path, class_name, 'image', actual_img_filename)
                mask_path = os.path.join(self.styled_path, class_name, 'mask', actual_mask_filename)
                
                if os.path.exists(img_path) and os.path.exists(mask_path):
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    if img is not None and mask is not None:
                        samples['styled_busi'].append({
                            'image': img,
                            'mask': mask,
                            'path': img_path,
                            'class': class_name
                        })
        
        # Load real BUS-UCLM samples
        bus_uclm_train = pd.read_csv(f"{self.bus_uclm_path}/train_frame.csv")
        for _, row in bus_uclm_train.sample(min(n_samples, len(bus_uclm_train))).iterrows():
            # Extract class from filename and actual filename
            class_name = 'benign' if 'benign' in row['image_path'] else 'malignant'
            
            # BUS-UCLM CSV has "benign FILENAME.png" but actual file is just "FILENAME.png"
            actual_img_filename = row['image_path'].replace('benign ', '').replace('malignant ', '')
            actual_mask_filename = row['mask_path'].replace('benign ', '').replace('malignant ', '')
            
            # Construct proper paths with subdirectories (note: images/masks are plural)
            img_path = os.path.join(self.bus_uclm_path, class_name, 'images', actual_img_filename)
            mask_path = os.path.join(self.bus_uclm_path, class_name, 'masks', actual_mask_filename)
            
            if os.path.exists(img_path) and os.path.exists(mask_path):
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if img is not None and mask is not None:
                    samples['real_bus_uclm'].append({
                        'image': img,
                        'mask': mask,
                        'path': img_path,
                        'class': class_name
                    })
        
        return samples
    
    def compute_image_statistics(self, image: np.ndarray) -> Dict:
        """Compute comprehensive image statistics."""
        stats = {
            'mean': float(np.mean(image)),
            'std': float(np.std(image)),
            'min': float(np.min(image)),
            'max': float(np.max(image)),
            'median': float(np.median(image)),
            'q25': float(np.percentile(image, 25)),
            'q75': float(np.percentile(image, 75)),
            'skewness': float(self.skewness(image)),
            'kurtosis': float(self.kurtosis(image)),
            'contrast': float(np.std(image)) / (float(np.mean(image)) + 1e-8),
            'entropy': float(self.entropy(image))
        }
        
        # Gradient statistics
        grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        stats.update({
            'gradient_mean': float(np.mean(gradient_magnitude)),
            'gradient_std': float(np.std(gradient_magnitude)),
            'edge_density': float(np.sum(gradient_magnitude > np.mean(gradient_magnitude))) / image.size
        })
        
        return stats
    
    def skewness(self, image: np.ndarray) -> float:
        """Calculate skewness of image."""
        mean = np.mean(image)
        std = np.std(image)
        if std == 0:
            return 0
        return np.mean(((image - mean) / std) ** 3)
    
    def kurtosis(self, image: np.ndarray) -> float:
        """Calculate kurtosis of image."""
        mean = np.mean(image)
        std = np.std(image)
        if std == 0:
            return 0
        return np.mean(((image - mean) / std) ** 4) - 3
    
    def entropy(self, image: np.ndarray) -> float:
        """Calculate image entropy."""
        hist, _ = np.histogram(image, bins=256, range=(0, 256))
        hist = hist + 1e-8  # Avoid log(0)
        prob = hist / np.sum(hist)
        return -np.sum(prob * np.log2(prob))
    
    def compare_domains(self, samples: Dict) -> Dict:
        """Compare statistical properties across domains."""
        domain_stats = {}
        
        for domain, sample_list in samples.items():
            if not sample_list:
                continue
                
            stats_list = []
            for sample in sample_list:
                stats = self.compute_image_statistics(sample['image'])
                stats_list.append(stats)
            
            # Aggregate statistics
            aggregated = {}
            for key in stats_list[0].keys():
                values = [s[key] for s in stats_list]
                aggregated[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
            
            domain_stats[domain] = aggregated
        
        return domain_stats
    
    def create_comparison_visualization(self, samples: Dict, save_path: str = "style_transfer_comparison.png"):
        """Create comprehensive comparison visualization."""
        n_samples = min(3, min(len(samples[k]) for k in samples.keys() if samples[k]))
        
        fig, axes = plt.subplots(3, n_samples + 1, figsize=(20, 12))
        
        domains = ['original_busi', 'styled_busi', 'real_bus_uclm']
        domain_labels = ['Original BUSI', 'Styled BUSI', 'Real BUS-UCLM']
        
        for i, (domain, label) in enumerate(zip(domains, domain_labels)):
            if not samples[domain]:
                continue
                
            # Show sample images
            for j in range(n_samples):
                if j < len(samples[domain]):
                    img = samples[domain][j]['image']
                    axes[i, j].imshow(img, cmap='gray')
                    axes[i, j].set_title(f"{label} Sample {j+1}")
                    axes[i, j].axis('off')
                else:
                    axes[i, j].axis('off')
            
            # Show histogram comparison
            if samples[domain]:
                all_pixels = []
                for sample in samples[domain]:
                    all_pixels.extend(sample['image'].flatten())
                
                axes[i, -1].hist(all_pixels, bins=50, alpha=0.7, density=True, label=label)
                axes[i, -1].set_title(f"{label} Histogram")
                axes[i, -1].set_xlabel('Pixel Intensity')
                axes[i, -1].set_ylabel('Density')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Saved comparison visualization: {save_path}")
    
    def analyze_style_transfer_effectiveness(self, samples: Dict) -> Dict:
        """Analyze how well style transfer worked."""
        results = {}
        
        if not samples['original_busi'] or not samples['styled_busi'] or not samples['real_bus_uclm']:
            print("❌ Missing samples for analysis")
            return results
        
        # Compare styled BUSI to real BUS-UCLM
        styled_stats = []
        real_stats = []
        
        for sample in samples['styled_busi']:
            styled_stats.append(self.compute_image_statistics(sample['image']))
        
        for sample in samples['real_bus_uclm']:
            real_stats.append(self.compute_image_statistics(sample['image']))
        
        # Calculate distances between styled and real
        stat_keys = styled_stats[0].keys()
        distance_metrics = {}
        
        for key in stat_keys:
            styled_values = [s[key] for s in styled_stats]
            real_values = [s[key] for s in real_stats]
            
            # Calculate mean squared difference
            mse = mean_squared_error([np.mean(styled_values)], [np.mean(real_values)])
            distance_metrics[f'{key}_mse'] = mse
            
            # Calculate absolute difference in means
            abs_diff = abs(np.mean(styled_values) - np.mean(real_values))
            distance_metrics[f'{key}_abs_diff'] = abs_diff
        
        results['distance_metrics'] = distance_metrics
        
        # Style transfer quality score (lower is better)
        quality_score = np.mean([
            distance_metrics['mean_abs_diff'],
            distance_metrics['std_abs_diff'],
            distance_metrics['contrast_abs_diff'],
            distance_metrics['entropy_abs_diff']
        ])
        
        results['quality_score'] = quality_score
        
        return results
    
    def generate_recommendations(self, analysis_results: Dict, domain_stats: Dict) -> List[str]:
        """Generate recommendations for improving style transfer."""
        recommendations = []
        
        if 'distance_metrics' in analysis_results:
            metrics = analysis_results['distance_metrics']
            
            # Check mean intensity difference
            if metrics.get('mean_abs_diff', 0) > 10:
                recommendations.append("🔧 Large intensity difference detected. Consider adjusting histogram matching strength.")
            
            # Check contrast difference
            if metrics.get('contrast_abs_diff', 0) > 0.1:
                recommendations.append("🔧 Contrast mismatch detected. Enhance contrast normalization in style transfer.")
            
            # Check texture/entropy difference
            if metrics.get('entropy_abs_diff', 0) > 0.5:
                recommendations.append("🔧 Texture differences detected. Consider adding texture enhancement.")
            
            # Check edge preservation
            if metrics.get('gradient_mean_abs_diff', 0) > 5:
                recommendations.append("🔧 Edge structure differences detected. Improve edge-preserving techniques.")
        
        # Check domain statistics
        if domain_stats:
            styled_mean = domain_stats.get('styled_busi', {}).get('mean', {}).get('mean', 0)
            real_mean = domain_stats.get('real_bus_uclm', {}).get('mean', {}).get('mean', 0)
            
            if abs(styled_mean - real_mean) > 15:
                recommendations.append("🔧 Overall brightness mismatch. Adjust global intensity normalization.")
        
        if not recommendations:
            recommendations.append("✅ Style transfer appears to be working reasonably well.")
            recommendations.append("🔍 Consider fine-tuning parameters or trying different style transfer methods.")
        
        return recommendations
    
    def run_comprehensive_analysis(self):
        """Run complete style transfer analysis."""
        print("🔍 Starting comprehensive style transfer analysis...")
        
        # Load samples
        print("📁 Loading image samples...")
        samples = self.load_random_samples(n_samples=15)
        
        sample_counts = {k: len(v) for k, v in samples.items()}
        print(f"📊 Loaded samples: {sample_counts}")
        
        if not all(samples.values()):
            print("❌ Failed to load samples from all domains")
            return
        
        # Compare domain statistics
        print("📈 Computing domain statistics...")
        domain_stats = self.compare_domains(samples)
        
        # Analyze style transfer effectiveness
        print("🎯 Analyzing style transfer effectiveness...")
        effectiveness = self.analyze_style_transfer_effectiveness(samples)
        
        # Create visualizations
        print("🎨 Creating visualizations...")
        self.create_comparison_visualization(samples, "style_transfer_analysis.png")
        
        # Generate recommendations
        print("💡 Generating recommendations...")
        recommendations = self.generate_recommendations(effectiveness, domain_stats)
        
        # Save detailed results
        results = {
            'sample_counts': sample_counts,
            'domain_statistics': domain_stats,
            'effectiveness_analysis': effectiveness,
            'recommendations': recommendations
        }
        
        with open('style_transfer_analysis_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # Print summary
        print("\n" + "="*50)
        print("📋 STYLE TRANSFER ANALYSIS SUMMARY")
        print("="*50)
        
        print(f"\n📊 Sample Counts:")
        for domain, count in sample_counts.items():
            print(f"  {domain}: {count} images")
        
        if effectiveness.get('quality_score'):
            print(f"\n🎯 Style Transfer Quality Score: {effectiveness['quality_score']:.4f}")
            print("   (Lower scores indicate better style matching)")
        
        print(f"\n💡 Recommendations:")
        for rec in recommendations:
            print(f"  {rec}")
        
        print(f"\n📁 Detailed results saved to: style_transfer_analysis_results.json")
        print(f"🎨 Visualization saved to: style_transfer_analysis.png")
        print("="*50)

if __name__ == "__main__":
    analyzer = StyleTransferAnalyzer()
    analyzer.run_comprehensive_analysis() 