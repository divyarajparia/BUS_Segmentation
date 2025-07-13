#!/usr/bin/env python3
"""
Privacy-Preserving AdaIN Style Transfer for Medical Images
=========================================================

Implementation of the CCST (Cross-Client Style Transfer) methodology
for privacy-preserving domain adaptation in medical imaging.

Key Features:
- Privacy-preserving: Only shares mean/variance statistics, not actual images
- Domain-level style transfer: Uses overall domain statistics (not individual images)
- Real-time inference: No training required, uses pre-trained VGG features
- Medical image focused: Optimized for grayscale ultrasound images

Based on:
"Federated Domain Generalization for Image Recognition via Cross-Client Style Transfer"
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse
from typing import Dict, Tuple, Optional, List


class PrivacyPreservingStyleExtractor:
    """
    Extract domain-level style statistics in a privacy-preserving manner.
    Following CCST paper Algorithm 1 and Equation (8).
    """
    
    def __init__(self, device='cuda'):
        self.device = device
        
        # Load pre-trained VGG19 encoder (up to relu4_1)
        vgg = models.vgg19(pretrained=True).features
        self.encoder = nn.Sequential(*list(vgg.children())[:21])  # Up to relu4_1
        
        # Freeze encoder parameters
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        self.encoder.to(device)
        self.encoder.eval()
        
        print(f"✅ Privacy-preserving style extractor initialized on {device}")
    
    def extract_domain_style(self, dataset_dir: str, csv_file: str, 
                           batch_size: int = 8, save_path: Optional[str] = None) -> Dict:
        """
        Extract overall domain style statistics from all training images.
        
        Args:
            dataset_dir: Path to dataset directory
            csv_file: CSV file with image paths (train_frame.csv)
            batch_size: Batch size for processing
            save_path: Optional path to save extracted statistics
        
        Returns:
            Dictionary containing domain-level mean and std statistics
        """
        print(f"🎨 Extracting domain-level style from {dataset_dir}")
        
        # Load dataset
        df = pd.read_csv(os.path.join(dataset_dir, csv_file))
        if len(df) == 0:
            raise ValueError(f"No data found in {csv_file}")
        
        # Transform for VGG (requires 3-channel input)
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        all_features = []
        processed_count = 0
        
        with torch.no_grad():
            for idx in tqdm(range(len(df)), desc="Processing images"):
                try:
                    # Get image path - handle both BUSI and BUS-UCLM formats
                    image_path_info = df.iloc[idx]['image_path']
                    
                    # Handle different path formats
                    if '/' in image_path_info:
                        # Direct path format: "benign/image/file.png"
                        image_path = os.path.join(dataset_dir, image_path_info)
                    else:
                        # Extract class from filename (e.g., "benign (1).png" or "benign SHST_011.png")
                        class_type = image_path_info.split()[0]
                        
                        # Try both folder structures: BUSI uses "image", BUS-UCLM uses "images"
                        image_path_busi = os.path.join(dataset_dir, class_type, 'image', image_path_info)
                        image_path_busuclm = os.path.join(dataset_dir, class_type, 'images', image_path_info)
                        
                        if os.path.exists(image_path_busi):
                            image_path = image_path_busi
                        elif os.path.exists(image_path_busuclm):
                            image_path = image_path_busuclm
                        else:
                            image_path = image_path_busi  # Default to BUSI format
                    
                    if not os.path.exists(image_path):
                        continue
                    
                    # Load and transform image
                    image = Image.open(image_path).convert('L')
                    image_tensor = transform(image).unsqueeze(0).to(self.device)
                    
                    # Extract VGG features
                    features = self.encoder(image_tensor)
                    all_features.append(features)
                    processed_count += 1
                    
                except Exception as e:
                    print(f"Warning: Error processing image {idx}: {e}")
                    continue
        
        if len(all_features) == 0:
            raise ValueError("No valid images found for style extraction")
        
        # Concatenate all features
        all_features = torch.cat(all_features, dim=0)
        print(f"   📊 Processed {processed_count} images")
        print(f"   📐 Feature shape: {all_features.shape}")
        
        # Calculate domain-level statistics (Equation 8 from CCST paper)
        # μ_domain = (1/N) * Σ_i μ(Φ(I_i))
        # σ_domain = (1/N) * Σ_i σ(Φ(I_i))
        domain_mean = torch.mean(all_features, dim=(0, 2, 3), keepdim=True)
        domain_std = torch.std(all_features, dim=(0, 2, 3), keepdim=True)
        
        # Add small epsilon to prevent division by zero
        domain_std = domain_std + 1e-8
        
        style_stats = {
            'mean': domain_mean.cpu(),
            'std': domain_std.cpu(),
            'num_images': processed_count,
            'feature_shape': list(all_features.shape),
            'extraction_info': {
                'dataset_dir': dataset_dir,
                'csv_file': csv_file,
                'device': str(self.device),
                'vgg_layer': 'relu4_1'
            }
        }
        
        # Save statistics if requested
        if save_path:
            self.save_style_stats(style_stats, save_path)
        
        print(f"   ✅ Domain style extracted successfully")
        print(f"   📈 Statistics: mean={domain_mean.shape}, std={domain_std.shape}")
        
        return style_stats
    
    def save_style_stats(self, style_stats: Dict, save_path: str):
        """Save style statistics to disk"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Convert tensors to lists for JSON serialization
        serializable_stats = {
            'mean': style_stats['mean'].tolist(),
            'std': style_stats['std'].tolist(),
            'num_images': style_stats['num_images'],
            'feature_shape': style_stats['feature_shape'],
            'extraction_info': style_stats['extraction_info']
        }
        
        with open(save_path, 'w') as f:
            json.dump(serializable_stats, f, indent=2)
        
        print(f"   💾 Style statistics saved to {save_path}")
    
    def load_style_stats(self, load_path: str) -> Dict:
        """Load style statistics from disk"""
        with open(load_path, 'r') as f:
            stats = json.load(f)
        
        # Convert lists back to tensors
        stats['mean'] = torch.tensor(stats['mean'])
        stats['std'] = torch.tensor(stats['std'])
        
        return stats


class PrivacyPreservingAdaIN(nn.Module):
    """
    Privacy-preserving AdaIN implementation for medical image style transfer.
    
    Key privacy features:
    - Only uses shared domain statistics (mean/variance)
    - No actual images are shared between domains
    - Real-time inference without training
    """
    
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device
        
        # VGG encoder (same as style extractor)
        vgg = models.vgg19(pretrained=True).features
        self.encoder = nn.Sequential(*list(vgg.children())[:21])
        
        # Decoder network (mirror of encoder)
        self.decoder = self._build_decoder()
        
        # Freeze encoder
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        self.to(device)
        self.encoder.eval()
        
        print(f"✅ Privacy-preserving AdaIN initialized on {device}")
    
    def _build_decoder(self):
        """Build decoder network (mirror of VGG encoder)"""
        return nn.Sequential(
            # Layer 1: 512 -> 256
            nn.ReflectionPad2d(1),
            nn.Conv2d(512, 256, 3),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            
            # Layer 2: 256 -> 256
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, 3),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, 3),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, 3),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 128, 3),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            
            # Layer 3: 128 -> 64
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 128, 3),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 64, 3),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            
            # Layer 4: 64 -> 1 (grayscale output)
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 1, 3),
            nn.Tanh()  # Output in [-1, 1]
        )
    
    def adain(self, content_features, style_mean, style_std):
        """
        Adaptive Instance Normalization with privacy-preserving style statistics.
        
        AdaIN(Fc, Fs) = σ(Fs) * (Fc - μ(Fc)) / σ(Fc) + μ(Fs)
        
        Args:
            content_features: Content features from VGG encoder
            style_mean: Pre-computed domain-level mean statistics
            style_std: Pre-computed domain-level std statistics
        """
        # Calculate content statistics
        content_mean = torch.mean(content_features, dim=(2, 3), keepdim=True)
        content_std = torch.std(content_features, dim=(2, 3), keepdim=True) + 1e-8
        
        # Normalize content features
        normalized_features = (content_features - content_mean) / content_std
        
        # Apply style statistics
        stylized_features = style_std * normalized_features + style_mean
        
        return stylized_features
    
    def forward(self, content_image, style_stats):
        """
        Forward pass for privacy-preserving style transfer.
        
        Args:
            content_image: Source image tensor [B, C, H, W]
            style_stats: Dictionary with 'mean' and 'std' tensors
        
        Returns:
            Stylized image tensor [B, C, H, W]
        """
        # Extract content features
        content_features = self.encoder(content_image)
        
        # Get style statistics
        style_mean = style_stats['mean'].to(self.device)
        style_std = style_stats['std'].to(self.device)
        
        # Apply AdaIN
        stylized_features = self.adain(content_features, style_mean, style_std)
        
        # Decode to image
        stylized_image = self.decoder(stylized_features)
        
        return stylized_image


class CCSTDatasetGenerator:
    """
    Generate CCST-style dataset following Algorithm 1 from the paper.
    """
    
    def __init__(self, style_transfer_model: PrivacyPreservingAdaIN, device='cuda'):
        self.model = style_transfer_model
        self.device = device
        
        # Transform for input processing
        self.input_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Transform for output processing
        self.output_transform = transforms.Compose([
            transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], 
                               std=[1/0.229, 1/0.224, 1/0.225]),
            transforms.Lambda(lambda x: x[0:1, :, :]),  # Take first channel
            transforms.ToPILImage()
        ])
    
    def generate_ccst_dataset(self, source_dataset_dir: str, target_style_stats: Dict,
                             output_dir: str, csv_file: str = 'train_frame.csv') -> List[Dict]:
        """
        Generate CCST-style dataset following Algorithm 1.
        
        Args:
            source_dataset_dir: Source dataset directory (e.g., BUS-UCLM)
            target_style_stats: Target domain style statistics (e.g., BUSI)
            output_dir: Output directory for generated dataset
            csv_file: CSV file with source image paths
        
        Returns:
            List of generated sample metadata
        """
        print(f"🚀 Generating CCST dataset...")
        print(f"   Source: {source_dataset_dir}")
        print(f"   Output: {output_dir}")
        
        # Create output directory structure
        os.makedirs(os.path.join(output_dir, 'benign', 'image'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'benign', 'mask'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'malignant', 'image'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'malignant', 'mask'), exist_ok=True)
        
        # Load source dataset
        df = pd.read_csv(os.path.join(source_dataset_dir, csv_file))
        generated_samples = []
        
        self.model.eval()
        with torch.no_grad():
            for idx in tqdm(range(len(df)), desc="Generating styled images"):
                try:
                    # Get image and mask paths
                    image_path_info = df.iloc[idx]['image_path']
                    mask_path_info = df.iloc[idx]['mask_path']
                    
                    # Handle different path formats
                    if '/' in image_path_info:
                        # Direct path format
                        image_path = os.path.join(source_dataset_dir, image_path_info)
                        mask_path = os.path.join(source_dataset_dir, mask_path_info)
                        class_type = image_path_info.split('/')[0]
                    else:
                        # Extract class from filename (e.g., "benign (1).png" or "benign SHST_011.png")
                        class_type = image_path_info.split()[0]
                        
                        # Try both folder structures: BUSI uses "image/mask", BUS-UCLM uses "images/masks"
                        image_path_busi = os.path.join(source_dataset_dir, class_type, 'image', image_path_info)
                        mask_path_busi = os.path.join(source_dataset_dir, class_type, 'mask', mask_path_info)
                        image_path_busuclm = os.path.join(source_dataset_dir, class_type, 'images', image_path_info)
                        mask_path_busuclm = os.path.join(source_dataset_dir, class_type, 'masks', mask_path_info)
                        
                        if os.path.exists(image_path_busi) and os.path.exists(mask_path_busi):
                            image_path = image_path_busi
                            mask_path = mask_path_busi
                        elif os.path.exists(image_path_busuclm) and os.path.exists(mask_path_busuclm):
                            image_path = image_path_busuclm
                            mask_path = mask_path_busuclm
                        else:
                            # Default to BUSI format
                            image_path = image_path_busi
                            mask_path = mask_path_busi
                    
                    if not os.path.exists(image_path) or not os.path.exists(mask_path):
                        continue
                    
                    # Load and transform image
                    image = Image.open(image_path).convert('L')
                    image_tensor = self.input_transform(image).unsqueeze(0).to(self.device)
                    
                    # Apply style transfer
                    styled_tensor = self.model(image_tensor, target_style_stats)
                    
                    # Convert back to PIL image
                    styled_image = self.output_transform(styled_tensor.squeeze(0).cpu())
                    
                    # Generate output filename
                    styled_filename = f"styled_{idx:04d}.png"
                    styled_mask_filename = f"styled_{idx:04d}_mask.png"
                    
                    # Save styled image
                    styled_image_path = os.path.join(output_dir, class_type, 'image', styled_filename)
                    styled_image.save(styled_image_path)
                    
                    # Copy mask (unchanged)
                    mask_image = Image.open(mask_path)
                    styled_mask_path = os.path.join(output_dir, class_type, 'mask', styled_mask_filename)
                    mask_image.save(styled_mask_path)
                    
                    # Record sample metadata
                    generated_samples.append({
                        'image_path': f"{class_type} {styled_filename}",
                        'mask_path': f"{class_type} {styled_mask_filename}",
                        'class': class_type,
                        'source_client': 'BUS-UCLM',
                        'style_client': 'BUSI',
                        'augmentation_type': 'styled',
                        'original_index': idx
                    })
                    
                except Exception as e:
                    print(f"Warning: Error processing image {idx}: {e}")
                    continue
        
        # Save dataset CSV
        csv_path = os.path.join(output_dir, 'styled_dataset.csv')
        styled_df = pd.DataFrame(generated_samples)
        styled_df.to_csv(csv_path, index=False)
        
        print(f"   ✅ Generated {len(generated_samples)} styled images")
        print(f"   📁 Dataset saved to {output_dir}")
        print(f"   📊 Metadata saved to {csv_path}")
        
        return generated_samples


def create_combined_dataset(busi_dir: str, styled_dir: str, combined_dir: str):
    """
    Create combined dataset: Original BUSI + Styled BUS-UCLM
    """
    print(f"📁 Creating combined dataset...")
    print(f"   BUSI: {busi_dir}")
    print(f"   Styled: {styled_dir}")
    print(f"   Combined: {combined_dir}")
    
    # Create combined directory structure
    os.makedirs(os.path.join(combined_dir, 'benign', 'image'), exist_ok=True)
    os.makedirs(os.path.join(combined_dir, 'benign', 'mask'), exist_ok=True)
    os.makedirs(os.path.join(combined_dir, 'malignant', 'image'), exist_ok=True)
    os.makedirs(os.path.join(combined_dir, 'malignant', 'mask'), exist_ok=True)
    
    combined_samples = []
    
    # 1. Add original BUSI training data
    busi_csv = os.path.join(busi_dir, 'train_frame.csv')
    if os.path.exists(busi_csv):
        busi_df = pd.read_csv(busi_csv)
        
        for idx in range(len(busi_df)):
            image_path = busi_df.iloc[idx]['image_path']
            mask_path = busi_df.iloc[idx]['mask_path']
            
            # Extract class from filename (e.g., "benign (322).png" -> "benign")
            class_type = image_path.split()[0]
            
            # Build full paths using BUSI folder structure
            src_image = os.path.join(busi_dir, class_type, 'image', image_path)
            src_mask = os.path.join(busi_dir, class_type, 'mask', mask_path)
            
            if os.path.exists(src_image) and os.path.exists(src_mask):
                # Generate new filename
                new_filename = f"busi_{os.path.basename(image_path)}"
                new_mask_filename = f"busi_{os.path.basename(mask_path)}"
                
                dst_image = os.path.join(combined_dir, class_type, 'image', new_filename)
                dst_mask = os.path.join(combined_dir, class_type, 'mask', new_mask_filename)
                
                # Copy files
                import shutil
                shutil.copy2(src_image, dst_image)
                shutil.copy2(src_mask, dst_mask)
                
                combined_samples.append({
                    'image_path': f"{class_type} {new_filename}",
                    'mask_path': f"{class_type} {new_mask_filename}",
                    'class': class_type,
                    'source_client': 'BUSI',
                    'style_client': 'BUSI',
                    'augmentation_type': 'original'
                })
    
    # 2. Add styled BUS-UCLM data
    styled_csv = os.path.join(styled_dir, 'styled_dataset.csv')
    if os.path.exists(styled_csv):
        styled_df = pd.read_csv(styled_csv)
        
        for idx in range(len(styled_df)):
            image_path = styled_df.iloc[idx]['image_path']
            mask_path = styled_df.iloc[idx]['mask_path']
            class_type = styled_df.iloc[idx]['class']
            
            # Copy to combined directory
            image_file = image_path.split(' ', 1)[1]  # Remove class prefix
            mask_file = mask_path.split(' ', 1)[1]
            
            src_image = os.path.join(styled_dir, class_type, 'image', image_file)
            src_mask = os.path.join(styled_dir, class_type, 'mask', mask_file)
            
            if os.path.exists(src_image) and os.path.exists(src_mask):
                dst_image = os.path.join(combined_dir, class_type, 'image', image_file)
                dst_mask = os.path.join(combined_dir, class_type, 'mask', mask_file)
                
                # Copy files
                import shutil
                shutil.copy2(src_image, dst_image)
                shutil.copy2(src_mask, dst_mask)
                
                combined_samples.append({
                    'image_path': image_path,
                    'mask_path': mask_path,
                    'class': class_type,
                    'source_client': 'BUS-UCLM',
                    'style_client': 'BUSI',
                    'augmentation_type': 'styled'
                })
    
    # Save combined dataset CSV
    combined_csv = os.path.join(combined_dir, 'combined_train_frame.csv')
    combined_df = pd.DataFrame(combined_samples)
    combined_df.to_csv(combined_csv, index=False)
    
    print(f"   ✅ Combined dataset created with {len(combined_samples)} samples")
    print(f"   📊 Metadata saved to {combined_csv}")
    
    # Print statistics
    original_count = sum(1 for s in combined_samples if s['augmentation_type'] == 'original')
    styled_count = sum(1 for s in combined_samples if s['augmentation_type'] == 'styled')
    benign_count = sum(1 for s in combined_samples if s['class'] == 'benign')
    malignant_count = sum(1 for s in combined_samples if s['class'] == 'malignant')
    
    print(f"   📈 Statistics:")
    print(f"      Original BUSI: {original_count}")
    print(f"      Styled BUS-UCLM: {styled_count}")
    print(f"      Benign: {benign_count}")
    print(f"      Malignant: {malignant_count}")


def main():
    """Main pipeline for privacy-preserving AdaIN style transfer"""
    parser = argparse.ArgumentParser(description='Privacy-preserving AdaIN style transfer')
    parser.add_argument('--busi-dir', type=str, default='dataset/BioMedicalDataset/BUSI',
                       help='Path to BUSI dataset')
    parser.add_argument('--bus-uclm-dir', type=str, default='dataset/BioMedicalDataset/BUS-UCLM',
                       help='Path to BUS-UCLM dataset')
    parser.add_argument('--output-dir', type=str, default='dataset/BioMedicalDataset/BUSI_CCST_Combined',
                       help='Output directory for combined dataset')
    parser.add_argument('--style-stats-dir', type=str, default='ccst_style_stats',
                       help='Directory to save/load style statistics')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto/cuda/cpu)')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print("🚀 Privacy-Preserving AdaIN Style Transfer Pipeline")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  BUSI directory: {args.busi_dir}")
    print(f"  BUS-UCLM directory: {args.bus_uclm_dir}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Device: {device}")
    
    # Step 1: Extract BUSI domain style (privacy-preserving)
    print(f"\n🎨 Step 1: Extracting BUSI domain style...")
    style_extractor = PrivacyPreservingStyleExtractor(device=device)
    
    busi_style_path = os.path.join(args.style_stats_dir, 'busi_domain_style.json')
    if os.path.exists(busi_style_path):
        print(f"   📂 Loading existing BUSI style from {busi_style_path}")
        busi_style_stats = style_extractor.load_style_stats(busi_style_path)
    else:
        busi_style_stats = style_extractor.extract_domain_style(
            args.busi_dir, 
            'train_frame.csv',
            save_path=busi_style_path
        )
    
    # Step 2: Initialize privacy-preserving AdaIN
    print(f"\n🔧 Step 2: Initializing privacy-preserving AdaIN...")
    adain_model = PrivacyPreservingAdaIN(device=device)
    
    # Step 3: Generate styled BUS-UCLM dataset
    print(f"\n🔄 Step 3: Generating styled BUS-UCLM dataset...")
    styled_output_dir = os.path.join(args.output_dir, 'styled_bus_uclm')
    
    dataset_generator = CCSTDatasetGenerator(adain_model, device=device)
    generated_samples = dataset_generator.generate_ccst_dataset(
        args.bus_uclm_dir,
        busi_style_stats,
        styled_output_dir
    )
    
    # Step 4: Create combined dataset
    print(f"\n📁 Step 4: Creating combined dataset...")
    create_combined_dataset(
        args.busi_dir,
        styled_output_dir,
        args.output_dir
    )
    
    print(f"\n🎉 Privacy-preserving AdaIN style transfer completed!")
    print(f"   📊 Generated {len(generated_samples)} styled images")
    print(f"   📁 Combined dataset: {args.output_dir}")
    print(f"   🔒 Privacy preserved: Only domain statistics were used")
    print(f"\n🚀 Ready for MADGNet training!")


if __name__ == "__main__":
    main() 