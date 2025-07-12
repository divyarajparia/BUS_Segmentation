import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm
import shutil

class AdaINStyleTransfer(nn.Module):
    """
    AdaIN-based style transfer for medical imaging domain adaptation
    Simplified version of the CCST method for single-institution use
    """
    
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device
        
        # VGG encoder (up to relu4_1)
        vgg = models.vgg19(pretrained=True).features
        self.encoder = nn.Sequential(*list(vgg.children())[:21])  # Up to relu4_1
        
        # Decoder - mirror of encoder
        self.decoder = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(512, 256, 3),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            
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
            
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 128, 3),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 64, 3),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 1, 3),  # Single channel for medical images
        )
        
        # Freeze encoder
        for param in self.encoder.parameters():
            param.requires_grad = False
            
        self.to(device)
    
    def encode(self, x):
        """Extract VGG features"""
        return self.encoder(x)
    
    def decode(self, features):
        """Decode features back to image"""
        return self.decoder(features)
    
    def adain(self, content_features, style_features):
        """
        Adaptive Instance Normalization
        AdaIN(Fc, Fs) = σ(Fs) * (Fc - μ(Fc)) / σ(Fc) + μ(Fs)
        """
        # Calculate statistics
        content_mean = torch.mean(content_features, dim=(2, 3), keepdim=True)
        content_std = torch.std(content_features, dim=(2, 3), keepdim=True)
        
        style_mean = torch.mean(style_features, dim=(2, 3), keepdim=True)
        style_std = torch.std(style_features, dim=(2, 3), keepdim=True)
        
        # Normalize content with its own statistics
        normalized_content = (content_features - content_mean) / (content_std + 1e-8)
        
        # Apply style statistics
        stylized_features = style_std * normalized_content + style_mean
        
        return stylized_features
    
    def forward(self, content_image, style_features_stats):
        """
        Forward pass for style transfer
        content_image: BUS-UCLM image
        style_features_stats: Pre-computed BUSI domain style statistics
        """
        # Encode content
        content_features = self.encode(content_image)
        
        # Apply AdaIN with pre-computed style statistics
        style_mean, style_std = style_features_stats
        
        # Normalize content
        content_mean = torch.mean(content_features, dim=(2, 3), keepdim=True)
        content_std = torch.std(content_features, dim=(2, 3), keepdim=True)
        normalized_content = (content_features - content_mean) / (content_std + 1e-8)
        
        # Apply style
        stylized_features = style_std * normalized_content + style_mean
        
        # Decode
        stylized_image = self.decode(stylized_features)
        
        return stylized_image


class DomainStyleExtractor:
    """Extract overall domain style from a dataset (following CCST paper)"""
    
    def __init__(self, encoder, device='cuda'):
        self.encoder = encoder
        self.device = device
    
    def extract_domain_style(self, dataset_path, csv_file, image_folder='image', batch_size=8):
        """
        Extract overall domain style from all images in a dataset
        Following Equation 8 from the paper
        """
        print(f"🎨 Extracting domain style from {dataset_path}")
        
        # Load dataset
        df = pd.read_csv(os.path.join(dataset_path, csv_file))
        
        all_features = []
        
        # Transform for VGG (needs 3 channels)
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x),  # Convert to 3 channels
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
        ])
        
        with torch.no_grad():
            for idx in tqdm(range(len(df)), desc="Processing images"):
                # Get image path
                image_filename = df.image_path.iloc[idx]
                class_type = image_filename.split()[0]  # 'benign' or 'malignant'
                image_path = os.path.join(dataset_path, class_type, image_folder, image_filename)
                
                if not os.path.exists(image_path):
                    continue
                
                # Load and transform image
                image = Image.open(image_path).convert('L')  # Grayscale
                image_tensor = transform(image).unsqueeze(0).to(self.device)
                
                # Extract features
                features = self.encoder(image_tensor)
                all_features.append(features)
        
        # Concatenate all features
        all_features = torch.cat(all_features, dim=0)
        
        # Calculate domain-level statistics (Equation 8)
        domain_mean = torch.mean(all_features, dim=(0, 2, 3), keepdim=True)  # Shape: [1, 512, 1, 1]
        domain_std = torch.std(all_features, dim=(0, 2, 3), keepdim=True)    # Shape: [1, 512, 1, 1]
        
        print(f"   ✅ Domain style extracted: mean shape {domain_mean.shape}, std shape {domain_std.shape}")
        
        return (domain_mean, domain_std)


class AdaINDatasetGenerator:
    """Generate style-transferred dataset using AdaIN"""
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
    
    def generate_styled_dataset(self, source_dataset_path, target_style_stats, 
                              output_path, csv_file='train_frame.csv'):
        """
        Generate style-transferred dataset
        source_dataset_path: BUS-UCLM path
        target_style_stats: BUSI domain style statistics
        output_path: Where to save style-transferred images
        """
        print(f"🎨 Generating style-transferred dataset...")
        print(f"   Source: {source_dataset_path}")
        print(f"   Output: {output_path}")
        
        # Create output directories
        os.makedirs(os.path.join(output_path, 'benign', 'image'), exist_ok=True)
        os.makedirs(os.path.join(output_path, 'benign', 'mask'), exist_ok=True)
        os.makedirs(os.path.join(output_path, 'malignant', 'image'), exist_ok=True)
        os.makedirs(os.path.join(output_path, 'malignant', 'mask'), exist_ok=True)
        
        # Load source dataset
        df = pd.read_csv(os.path.join(source_dataset_path, csv_file))
        
        # Transform for processing
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Reverse transform for saving
        reverse_transform = transforms.Compose([
            transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], 
                               std=[1/0.229, 1/0.224, 1/0.225]),
            transforms.Lambda(lambda x: x[0:1, :, :]),  # Take first channel only
            transforms.ToPILImage()
        ])
        
        styled_samples = []
        
        self.model.eval()
        with torch.no_grad():
            for idx in tqdm(range(len(df)), desc="Generating styled images"):
                # Get paths
                image_filename = df.image_path.iloc[idx]
                mask_filename = df.mask_path.iloc[idx]
                
                # Parse class type - handle both formats
                if ' ' in image_filename:
                    class_type = image_filename.split()[0]  # BUSI format: "benign (1).png"
                else:
                    class_type = image_filename.split('/')[0]  # BUS-UCLM format: "benign/img.png"
                
                # Source paths
                if ' ' in image_filename:
                    # BUSI format
                    src_image_path = os.path.join(source_dataset_path, class_type, 'image', image_filename)
                    src_mask_path = os.path.join(source_dataset_path, class_type, 'mask', mask_filename)
                else:
                    # BUS-UCLM format
                    src_image_path = os.path.join(source_dataset_path, image_filename)
                    src_mask_path = os.path.join(source_dataset_path, mask_filename)
                
                if not os.path.exists(src_image_path) or not os.path.exists(src_mask_path):
                    continue
                
                # Load image
                image = Image.open(src_image_path).convert('L')
                image_tensor = transform(image).unsqueeze(0).to(self.device)
                
                # Apply style transfer
                styled_tensor = self.model(image_tensor, target_style_stats)
                
                # Convert back to PIL
                styled_image = reverse_transform(styled_tensor.squeeze(0).cpu())
                
                # Save styled image
                styled_filename = f"styled_{idx:04d}.png"
                styled_mask_filename = f"styled_{idx:04d}_mask.png"
                
                dst_image_path = os.path.join(output_path, class_type, 'image', styled_filename)
                dst_mask_path = os.path.join(output_path, class_type, 'mask', styled_mask_filename)
                
                # Save styled image
                styled_image.save(dst_image_path)
                
                # Copy mask (unchanged, following your current approach)
                shutil.copy2(src_mask_path, dst_mask_path)
                
                # Record for CSV
                styled_samples.append({
                    'image_path': f"{class_type} {styled_filename}",  # BUSI format
                    'mask_path': f"{class_type} {styled_mask_filename}",
                    'class': class_type,
                    'source': 'bus_uclm_styled_to_busi'
                })
        
        # Save CSV
        styled_df = pd.DataFrame(styled_samples)
        styled_df.to_csv(os.path.join(output_path, 'styled_dataset.csv'), index=False)
        
        print(f"   ✅ Generated {len(styled_samples)} styled images")
        return styled_samples


def main():
    """Main pipeline for AdaIN-based domain adaptation"""
    
    # Configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🚀 Starting AdaIN-based style transfer on {device}")
    
    # Paths
    busi_path = "dataset/BioMedicalDataset/BUSI"
    bus_uclm_path = "dataset/BioMedicalDataset/BUS-UCLM"
    output_path = "dataset/BioMedicalDataset/BUS-UCLM-AdaIN-styled"
    
    # Step 1: Initialize AdaIN model
    print("\n📊 Step 1: Initializing AdaIN model...")
    adain_model = AdaINStyleTransfer(device=device)
    
    # Step 2: Extract BUSI domain style
    print("\n🎨 Step 2: Extracting BUSI domain style...")
    style_extractor = DomainStyleExtractor(adain_model.encoder, device=device)
    busi_domain_style = style_extractor.extract_domain_style(
        busi_path, 
        'train_frame.csv',
        image_folder='image'
    )
    
    # Step 3: Generate styled BUS-UCLM dataset
    print("\n🔄 Step 3: Generating styled BUS-UCLM dataset...")
    dataset_generator = AdaINDatasetGenerator(adain_model, device=device)
    styled_samples = dataset_generator.generate_styled_dataset(
        bus_uclm_path,
        busi_domain_style,
        output_path,
        csv_file='train_frame.csv'
    )
    
    # Step 4: Create combined dataset (optional)
    print("\n📁 Step 4: Creating combined training dataset...")
    combined_path = "dataset/BioMedicalDataset/BUSI-AdaIN-Combined"
    create_combined_dataset(busi_path, output_path, combined_path)
    
    print("\n🎉 AdaIN-based style transfer completed!")
    print(f"   Styled BUS-UCLM images: {output_path}")
    print(f"   Combined dataset: {combined_path}")


def create_combined_dataset(busi_path, styled_path, combined_path):
    """Create combined dataset with original BUSI + styled BUS-UCLM"""
    
    # Create output structure
    os.makedirs(os.path.join(combined_path, 'benign', 'image'), exist_ok=True)
    os.makedirs(os.path.join(combined_path, 'benign', 'mask'), exist_ok=True)
    os.makedirs(os.path.join(combined_path, 'malignant', 'image'), exist_ok=True)
    os.makedirs(os.path.join(combined_path, 'malignant', 'mask'), exist_ok=True)
    
    combined_samples = []
    
    # Copy original BUSI training data
    print("   📋 Copying original BUSI training data...")
    busi_df = pd.read_csv(os.path.join(busi_path, 'train_frame.csv'))
    
    for idx in range(len(busi_df)):
        image_filename = busi_df.image_path.iloc[idx]
        mask_filename = busi_df.mask_path.iloc[idx]
        class_type = image_filename.split()[0]
        
        # Source paths
        src_img = os.path.join(busi_path, class_type, 'image', image_filename)
        src_mask = os.path.join(busi_path, class_type, 'mask', mask_filename)
        
        # Destination paths
        dst_img = os.path.join(combined_path, class_type, 'image', f"busi_{image_filename}")
        dst_mask = os.path.join(combined_path, class_type, 'mask', f"busi_{mask_filename}")
        
        if os.path.exists(src_img) and os.path.exists(src_mask):
            shutil.copy2(src_img, dst_img)
            shutil.copy2(src_mask, dst_mask)
            
            combined_samples.append({
                'image_path': f"{class_type} busi_{image_filename}",
                'mask_path': f"{class_type} busi_{mask_filename}",
                'class': class_type,
                'source': 'original_busi'
            })
    
    # Copy styled BUS-UCLM data
    print("   🎨 Copying styled BUS-UCLM data...")
    styled_df = pd.read_csv(os.path.join(styled_path, 'styled_dataset.csv'))
    
    for idx in range(len(styled_df)):
        image_filename = styled_df.image_path.iloc[idx]
        mask_filename = styled_df.mask_path.iloc[idx]
        class_type = styled_df['class'].iloc[idx]
        
        # Source paths
        src_img = os.path.join(styled_path, class_type, 'image', image_filename.split()[-1])
        src_mask = os.path.join(styled_path, class_type, 'mask', mask_filename.split()[-1])
        
        # Destination paths
        dst_img = os.path.join(combined_path, class_type, 'image', f"styled_{image_filename.split()[-1]}")
        dst_mask = os.path.join(combined_path, class_type, 'mask', f"styled_{mask_filename.split()[-1]}")
        
        if os.path.exists(src_img) and os.path.exists(src_mask):
            shutil.copy2(src_img, dst_img)
            shutil.copy2(src_mask, dst_mask)
            
            combined_samples.append({
                'image_path': f"{class_type} styled_{image_filename.split()[-1]}",
                'mask_path': f"{class_type} styled_{mask_filename.split()[-1]}",
                'class': class_type,
                'source': 'styled_bus_uclm'
            })
    
    # Save combined CSV
    combined_df = pd.DataFrame(combined_samples)
    combined_df.to_csv(os.path.join(combined_path, 'combined_train_frame.csv'), index=False)
    
    # Print statistics
    original_count = len([s for s in combined_samples if s['source'] == 'original_busi'])
    styled_count = len([s for s in combined_samples if s['source'] == 'styled_bus_uclm'])
    
    print(f"   ✅ Combined dataset created:")
    print(f"      Original BUSI: {original_count} samples")
    print(f"      Styled BUS-UCLM: {styled_count} samples")
    print(f"      Total: {len(combined_samples)} samples")


if __name__ == "__main__":
    main() 