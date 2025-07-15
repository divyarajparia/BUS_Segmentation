#!/usr/bin/env python3
"""
Test Improved Style Transfer Quality
===================================
Compare the quality of improved style transfer vs original.
"""

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List

def load_samples_from_dataset(dataset_path: str, csv_filename: str, n_samples: int = 10) -> List[Dict]:
    """Load random samples from a dataset."""
    samples = []
    
    csv_path = os.path.join(dataset_path, csv_filename)
    if not os.path.exists(csv_path):
        print(f"❌ CSV not found: {csv_path}")
        return samples
    
    df = pd.read_csv(csv_path)
    
    for _, row in df.sample(min(n_samples, len(df))).iterrows():
        # Handle different CSV structures
        if 'class' in row:
            class_name = row['class']
            # Strip class prefix from filename if present
            img_filename = row['image_path']
            mask_filename = row['mask_path']
            if img_filename.startswith('benign ') or img_filename.startswith('malignant '):
                img_filename = img_filename.split(' ', 1)[1]
                mask_filename = mask_filename.split(' ', 1)[1]
        else:
            class_name = 'benign' if 'benign' in row['image_path'] else 'malignant'
            img_filename = row['image_path']
            mask_filename = row['mask_path']
        
        # Construct paths based on dataset structure
        if "Improved" in dataset_path:
            # Improved styled dataset structure
            img_path = os.path.join(dataset_path, class_name, 'image', img_filename)
            mask_path = os.path.join(dataset_path, class_name, 'mask', mask_filename)
        elif "Styled" in dataset_path:
            # Original styled dataset structure
            img_path = os.path.join(dataset_path, class_name, 'image', img_filename)
            mask_path = os.path.join(dataset_path, class_name, 'mask', mask_filename)
        else:
            # BUS-UCLM structure
            if img_filename.startswith('benign ') or img_filename.startswith('malignant '):
                actual_filename = img_filename.replace('benign ', '').replace('malignant ', '')
                actual_mask_filename = mask_filename.replace('benign ', '').replace('malignant ', '')
            else:
                actual_filename = img_filename
                actual_mask_filename = mask_filename
            img_path = os.path.join(dataset_path, class_name, 'images', actual_filename)
            mask_path = os.path.join(dataset_path, class_name, 'masks', actual_mask_filename)
        
        if os.path.exists(img_path) and os.path.exists(mask_path):
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            if img is not None and mask is not None:
                samples.append({
                    'image': img,
                    'mask': mask,
                    'path': img_path,
                    'class': class_name,
                    'filename': img_filename
                })
    
    return samples

def calculate_image_stats(images: List[np.ndarray]) -> Dict:
    """Calculate comprehensive statistics for a list of images."""
    all_pixels = []
    for img in images:
        all_pixels.extend(img.flatten())
    
    all_pixels = np.array(all_pixels)
    
    return {
        'mean': float(np.mean(all_pixels)),
        'std': float(np.std(all_pixels)),
        'min': float(np.min(all_pixels)),
        'max': float(np.max(all_pixels)),
        'q25': float(np.percentile(all_pixels, 25)),
        'q50': float(np.percentile(all_pixels, 50)),
        'q75': float(np.percentile(all_pixels, 75)),
        'contrast': float(np.std(all_pixels)) / (float(np.mean(all_pixels)) + 1e-8)
    }

def compare_style_transfer_quality():
    """Compare improved vs original style transfer quality."""
    print("🔍 Comparing style transfer quality...")
    
    # Dataset paths
    bus_uclm_path = "dataset/BioMedicalDataset/BUS-UCLM"
    original_styled_path = "dataset/BioMedicalDataset/BUSI-BUS-UCLM-Styled"
    improved_styled_path = "dataset/BioMedicalDataset/BUSI-BUS-UCLM-Improved"
    
    # Load samples
    print("📁 Loading samples...")
    bus_uclm_samples = load_samples_from_dataset(bus_uclm_path, "train_frame.csv", 15)
    original_samples = load_samples_from_dataset(original_styled_path, "styled_dataset.csv", 15)
    improved_samples = load_samples_from_dataset(improved_styled_path, "improved_styled_dataset.csv", 15)
    
    print(f"📊 Loaded samples:")
    print(f"   BUS-UCLM: {len(bus_uclm_samples)}")
    print(f"   Original styled: {len(original_samples)}")
    print(f"   Improved styled: {len(improved_samples)}")
    
    if not all([bus_uclm_samples, original_samples, improved_samples]):
        print("❌ Failed to load samples from all datasets")
        return
    
    # Calculate statistics
    print("📈 Calculating statistics...")
    
    bus_uclm_images = [s['image'] for s in bus_uclm_samples]
    original_images = [s['image'] for s in original_samples]
    improved_images = [s['image'] for s in improved_samples]
    
    bus_uclm_stats = calculate_image_stats(bus_uclm_images)
    original_stats = calculate_image_stats(original_images)
    improved_stats = calculate_image_stats(improved_images)
    
    # Calculate differences from target (BUS-UCLM)
    def calc_difference(target_stats, styled_stats):
        return {
            'mean_diff': abs(styled_stats['mean'] - target_stats['mean']),
            'std_diff': abs(styled_stats['std'] - target_stats['std']),
            'contrast_diff': abs(styled_stats['contrast'] - target_stats['contrast']),
            'q50_diff': abs(styled_stats['q50'] - target_stats['q50'])
        }
    
    original_diff = calc_difference(bus_uclm_stats, original_stats)
    improved_diff = calc_difference(bus_uclm_stats, improved_stats)
    
    # Calculate quality scores (lower is better)
    original_quality = np.mean([
        original_diff['mean_diff'],
        original_diff['std_diff'] * 10,  # Weight std more
        original_diff['contrast_diff'] * 100,  # Weight contrast heavily
        original_diff['q50_diff']
    ])
    
    improved_quality = np.mean([
        improved_diff['mean_diff'],
        improved_diff['std_diff'] * 10,
        improved_diff['contrast_diff'] * 100,
        improved_diff['q50_diff']
    ])
    
    # Create comparison visualization
    print("🎨 Creating comparison visualization...")
    
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    
    # Show sample images
    for i in range(3):
        if i < len(bus_uclm_samples):
            axes[0, i].imshow(bus_uclm_samples[i]['image'], cmap='gray')
            axes[0, i].set_title(f"BUS-UCLM {i+1}")
            axes[0, i].axis('off')
        
        if i < len(original_samples):
            axes[1, i].imshow(original_samples[i]['image'], cmap='gray')
            axes[1, i].set_title(f"Original Styled {i+1}")
            axes[1, i].axis('off')
        
        if i < len(improved_samples):
            axes[2, i].imshow(improved_samples[i]['image'], cmap='gray')
            axes[2, i].set_title(f"Improved Styled {i+1}")
            axes[2, i].axis('off')
    
    # Show histograms
    for i, (samples, label, color) in enumerate([
        (bus_uclm_samples, 'BUS-UCLM', 'blue'),
        (original_samples, 'Original', 'red'),
        (improved_samples, 'Improved', 'green')
    ]):
        all_pixels = []
        for sample in samples:
            all_pixels.extend(sample['image'].flatten())
        
        axes[i, 3].hist(all_pixels, bins=50, alpha=0.7, color=color, density=True)
        axes[i, 3].set_title(f"{label} Histogram")
        axes[i, 3].set_xlabel('Pixel Intensity')
        axes[i, 3].set_ylabel('Density')
    
    plt.tight_layout()
    plt.savefig('style_transfer_quality_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print results
    print("\n" + "="*60)
    print("📋 STYLE TRANSFER QUALITY COMPARISON")
    print("="*60)
    
    print(f"\n📊 Target (BUS-UCLM) Statistics:")
    for key, value in bus_uclm_stats.items():
        print(f"   {key}: {value:.3f}")
    
    print(f"\n📊 Original Styled Statistics:")
    for key, value in original_stats.items():
        print(f"   {key}: {value:.3f}")
    
    print(f"\n📊 Improved Styled Statistics:")
    for key, value in improved_stats.items():
        print(f"   {key}: {value:.3f}")
    
    print(f"\n🎯 Quality Scores (lower = better):")
    print(f"   Original styled quality: {original_quality:.3f}")
    print(f"   Improved styled quality: {improved_quality:.3f}")
    
    improvement = ((original_quality - improved_quality) / original_quality) * 100
    if improvement > 0:
        print(f"   ✅ Improvement: {improvement:.1f}% better")
    else:
        print(f"   ❌ Regression: {abs(improvement):.1f}% worse")
    
    print(f"\n📁 Visualization saved: style_transfer_quality_comparison.png")
    print("="*60)

if __name__ == "__main__":
    compare_style_transfer_quality() 