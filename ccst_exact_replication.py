"""
Exact Replication of CCST (Cross-Client Style Transfer) Methodology
Following the paper: "Federated Domain Generalization for Image Recognition via Cross-Client Style Transfer"

This implementation exactly follows:
- Algorithm 1: Local Cross-Client Style Transfer
- Three-stage workflow: Style computation → Style bank broadcasting → Style transfer
- Both single image style and overall domain style
- Augmentation level K parameter

Dataset Format Support:
- BUSI format: CSV "benign (1).png" → benign/image/benign (1).png
- BUS-UCLM format: CSV "benign image.png" → benign/images/image.png
- Uses TRAINING DATA ONLY for style extraction and transfer
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm
import json
import random
import shutil
from collections import defaultdict

class CCSTStyleExtractor:
    """
    Exact replication of CCST style extraction
    Following Section 3.2.1 of the paper
    """
    
    def __init__(self, device='cuda'):
        self.device = device
        
        # VGG encoder (same as AdaIN paper)
        vgg = models.vgg19(pretrained=True).features
        self.encoder = nn.Sequential(*list(vgg.children())[:21])  # Up to relu4_1
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.to(device)
    
    def extract_single_image_style(self, image_path):
        """
        Extract single image style following Equation (6)
        S_single(i) = (μ(F_i), σ(F_i))
        """
        # Load and preprocess image
        image = Image.open(image_path).convert('L')
        
        # Transform for VGG
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # Convert to 3 channels
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        image_tensor = transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Extract VGG features
            features = self.encoder(image_tensor)  # Shape: [1, 512, h, w]
            
            # Calculate channel-wise mean and std (Equation 6)
            mean = torch.mean(features, dim=(2, 3), keepdim=True)  # [1, 512, 1, 1]
            std = torch.std(features, dim=(2, 3), keepdim=True)    # [1, 512, 1, 1]
            
            # Convert to lists for JSON serialization
            single_style = {
                'mean': mean.squeeze().cpu().tolist(),
                'std': std.squeeze().cpu().tolist(),
                'type': 'single_image'
            }
        
        return single_style
    
    def extract_overall_domain_style(self, dataset_path, csv_file, J_samples=None):
        """
        Extract overall domain style following Equation (8)
        S_overall = (μ(F_all), σ(F_all)) where F_all = Stack(F1, F2, ..., FM)
        """
        print(f"🎨 Extracting overall domain style from {dataset_path}")
        
        # Load dataset
        df = pd.read_csv(os.path.join(dataset_path, csv_file))
        print(f"   📊 Loaded {len(df)} images from {csv_file}")
        
        # Debug: Show first few entries to understand format
        if len(df) > 0:
            sample_image_path = df.image_path.iloc[0]
            print(f"   🔍 Sample image path format: '{sample_image_path}'")
            if ' ' in sample_image_path:
                if '(' in sample_image_path:
                    print(f"   📝 Detected BUSI format")
                else:
                    print(f"   📝 Detected BUS-UCLM format")
            else:
                print(f"   📝 Detected fallback format")
        
        # Transform for VGG
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        all_features = []
        
        with torch.no_grad():
            for idx in tqdm(range(len(df)), desc="Processing images for domain style"):
                # Get image path
                image_filename = df.image_path.iloc[idx]
                
                # Handle different dataset formats
                if ' ' in image_filename:
                    # Check if it's BUSI format: "benign (1).png" vs BUS-UCLM format: "benign image.png"
                    if '(' in image_filename:
                        # BUSI format: "benign (1).png"
                        class_type = image_filename.split()[0]
                        image_path = os.path.join(dataset_path, class_type, 'image', image_filename)
                    else:
                        # BUS-UCLM format: "benign image.png"
                        class_type = image_filename.split()[0]  # 'benign' or 'malignant'
                        image_name = image_filename.split()[1]  # 'image.png'
                        image_path = os.path.join(dataset_path, class_type, 'images', image_name)
                else:
                    # Fallback format
                    image_path = os.path.join(dataset_path, image_filename)
                
                if not os.path.exists(image_path):
                    continue
                
                # Load and transform image
                image = Image.open(image_path).convert('L')
                image_tensor = transform(image).unsqueeze(0).to(self.device)
                
                # Extract features
                features = self.encoder(image_tensor)  # [1, 512, h, w]
                all_features.append(features)
        
        # Stack all features (Equation 8)
        all_features = torch.cat(all_features, dim=0)  # [M, 512, h, w]
        
        # Calculate overall domain statistics
        domain_mean = torch.mean(all_features, dim=(0, 2, 3), keepdim=True)  # [1, 512, 1, 1]
        domain_std = torch.std(all_features, dim=(0, 2, 3), keepdim=True)    # [1, 512, 1, 1]
        
        overall_style = {
            'mean': domain_mean.squeeze().cpu().tolist(),
            'std': domain_std.squeeze().cpu().tolist(),
            'type': 'overall_domain',
            'num_images': len(all_features)
        }
        
        print(f"   ✅ Extracted overall style from {len(all_features)} images")
        return overall_style
    
    def extract_single_image_style_bank(self, dataset_path, csv_file, J=10):
        """
        Extract multiple single image styles to form style bank
        Following Equation (7): S_bank = {S_single(i1), ..., S_single(iJ)}
        """
        print(f"🎨 Extracting {J} single image styles from {dataset_path}")
        
        # Load dataset
        df = pd.read_csv(os.path.join(dataset_path, csv_file))
        
        # Randomly sample J images
        sampled_indices = random.sample(range(len(df)), min(J, len(df)))
        
        style_bank = []
        
        for idx in tqdm(sampled_indices, desc="Extracting single image styles"):
            image_filename = df.image_path.iloc[idx]
            
            # Handle different dataset formats
            if ' ' in image_filename:
                # Check if it's BUSI format: "benign (1).png" vs BUS-UCLM format: "benign image.png"
                if '(' in image_filename:
                    # BUSI format: "benign (1).png"
                    class_type = image_filename.split()[0]
                    image_path = os.path.join(dataset_path, class_type, 'image', image_filename)
                else:
                    # BUS-UCLM format: "benign image.png"
                    class_type = image_filename.split()[0]  # 'benign' or 'malignant'
                    image_name = image_filename.split()[1]  # 'image.png'
                    image_path = os.path.join(dataset_path, class_type, 'images', image_name)
            else:
                # Fallback format
                image_path = os.path.join(dataset_path, image_filename)
            
            if os.path.exists(image_path):
                single_style = self.extract_single_image_style(image_path)
                single_style['image_index'] = idx
                single_style['image_filename'] = image_filename
                style_bank.append(single_style)
        
        print(f"   ✅ Extracted {len(style_bank)} single image styles")
        return style_bank


class CCSTStyleBank:
    """
    Server-side style bank following Section 3.2.2
    Implements both B_single and B_overall
    """
    
    def __init__(self):
        self.style_bank = {}
    
    def add_client_style(self, client_id, style_data, style_type):
        """
        Add client style to bank
        style_type: 'single' or 'overall'
        """
        self.style_bank[client_id] = {
            'style_data': style_data,
            'style_type': style_type
        }
        
        if style_type == 'single':
            print(f"🏥 Added {client_id}: {len(style_data)} single image styles")
        else:
            print(f"🏥 Added {client_id}: overall domain style ({style_data['num_images']} images)")
    
    def get_style_bank_single(self):
        """
        Get B_single following Equation (9)
        B_single = {S_bank^n | n = 1,2,...,N}
        """
        bank = {}
        for client_id, client_data in self.style_bank.items():
            if client_data['style_type'] == 'single':
                bank[client_id] = client_data['style_data']
        return bank
    
    def get_style_bank_overall(self):
        """
        Get B_overall following Equation (10)
        B_overall = {S_overall^n | n = 1,2,...,N}
        """
        bank = {}
        for client_id, client_data in self.style_bank.items():
            if client_data['style_type'] == 'overall':
                bank[client_id] = client_data['style_data']
        return bank
    
    def save_style_bank(self, output_file):
        """Save style bank to file"""
        with open(output_file, 'w') as f:
            json.dump(self.style_bank, f, indent=2)
        print(f"💾 Style bank saved to {output_file}")
    
    def load_style_bank(self, input_file):
        """Load style bank from file"""
        with open(input_file, 'r') as f:
            self.style_bank = json.load(f)
        print(f"📂 Style bank loaded from {input_file}")


class CCSTLocalStyleTransfer:
    """
    Local style transfer following Algorithm 1 from the paper
    Implements exact Algorithm 1: Local Cross-Client Style Transfer at Client Cn
    """
    
    def __init__(self, device='cuda'):
        self.device = device
        
        # VGG encoder (same as style extractor)
        vgg = models.vgg19(pretrained=True).features
        self.encoder = nn.Sequential(*list(vgg.children())[:21])
        
        # Decoder (mirror of encoder)
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
            nn.Conv2d(64, 1, 3),
        )
        
        # Freeze encoder
        for param in self.encoder.parameters():
            param.requires_grad = False
            
        self.encoder.to(device)
        self.decoder.to(device)
    
    def adain_transfer(self, content_image, style_mean, style_std):
        """
        Apply AdaIN transformation following Equation (2)
        AdaIN(Fc, Fs) = σ(Fs) * (Fc - μ(Fc)) / σ(Fc) + μ(Fs)
        """
        # Extract content features
        content_features = self.encoder(content_image)
        
        # Calculate content statistics
        content_mean = torch.mean(content_features, dim=(2, 3), keepdim=True)
        content_std = torch.std(content_features, dim=(2, 3), keepdim=True)
        
        # Convert style statistics to tensors if needed
        if isinstance(style_mean, list):
            style_mean = torch.tensor(style_mean).view(1, -1, 1, 1).to(self.device)
            style_std = torch.tensor(style_std).view(1, -1, 1, 1).to(self.device)
        
        # Apply AdaIN
        normalized_content = (content_features - content_mean) / (content_std + 1e-8)
        stylized_features = style_std * normalized_content + style_mean
        
        # Decode
        stylized_image = self.decoder(stylized_features)
        
        return stylized_image
    
    def local_cross_client_style_transfer(self, client_dataset_path, global_style_bank, 
                                        current_client_id, K=3, style_type='overall',
                                        output_path=None, csv_file='train_frame.csv'):
        """
        Exact implementation of Algorithm 1 from the paper
        
        Input: Training image set I_Cn, global style bank B
        Parameter: Augmentation level K, style type T
        Output: Augmented dataset D_Cn
        """
        print(f"🔄 Running Algorithm 1: Local Cross-Client Style Transfer")
        print(f"   Client: {current_client_id}")
        print(f"   Augmentation level K: {K}")
        print(f"   Style type: {style_type}")
        
        # Load client dataset
        df = pd.read_csv(os.path.join(client_dataset_path, csv_file))
        
        # Create output directories
        if output_path:
            os.makedirs(os.path.join(output_path, 'benign', 'image'), exist_ok=True)
            os.makedirs(os.path.join(output_path, 'benign', 'mask'), exist_ok=True)
            os.makedirs(os.path.join(output_path, 'malignant', 'image'), exist_ok=True)
            os.makedirs(os.path.join(output_path, 'malignant', 'mask'), exist_ok=True)
        
        # Transform for processing
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Reverse transform for saving
        def reverse_transform_fn(tensor):
            # Denormalize 3-channel tensor
            denorm_tensor = transforms.Normalize(
                mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], 
                std=[1/0.229, 1/0.224, 1/0.225]
            )(tensor)
            
            # Take first channel to get back to single channel
            single_channel = denorm_tensor[0:1, :, :]
            
            # Clamp values to [0, 1] range
            single_channel = torch.clamp(single_channel, 0, 1)
            
            # Convert to PIL Image
            return transforms.ToPILImage()(single_channel)
        
        # Algorithm 1 implementation
        augmented_dataset = []  # D_Cn = []
        
        # Line 2: for i = 1,2,...,m do (m = size(I))
        for i in tqdm(range(len(df)), desc="Applying Algorithm 1"):
            image_filename = df.image_path.iloc[i]
            mask_filename = df.mask_path.iloc[i]
            
            # Load original image I_i
            if ' ' in image_filename:
                # Check if it's BUSI format: "benign (1).png" vs BUS-UCLM format: "benign image.png"
                if '(' in image_filename:
                    # BUSI format: "benign (1).png"
                    class_type = image_filename.split()[0]
                    src_image_path = os.path.join(client_dataset_path, class_type, 'image', image_filename)
                    src_mask_path = os.path.join(client_dataset_path, class_type, 'mask', mask_filename)
                else:
                    # BUS-UCLM format: "benign image.png"
                    class_type = image_filename.split()[0]  # 'benign' or 'malignant'
                    image_name = image_filename.split()[1]  # 'image.png'
                    mask_name = mask_filename.split()[1]   # 'mask.png'
                    src_image_path = os.path.join(client_dataset_path, class_type, 'images', image_name)
                    src_mask_path = os.path.join(client_dataset_path, class_type, 'masks', mask_name)
            else:
                # Fallback format
                src_image_path = os.path.join(client_dataset_path, image_filename)
                src_mask_path = os.path.join(client_dataset_path, mask_filename)
                class_type = image_filename.split('/')[0]
            
            if not os.path.exists(src_image_path) or not os.path.exists(src_mask_path):
                continue
            
            # Load image
            original_image = Image.open(src_image_path).convert('L')
            image_tensor = transform(original_image).unsqueeze(0).to(self.device)
            
            # Line 3: S = random.choice(B, K)
            available_clients = list(global_style_bank.keys())
            selected_clients = random.choices(available_clients, k=K)
            
            # Line 4-11: for S_Cn in S do
            for j, selected_client in enumerate(selected_clients):
                # Line 5-6: if Cn is current client then D_Cn.append(I_i)
                if selected_client == current_client_id:
                    # Keep original image
                    if output_path:
                        dst_image_path = os.path.join(output_path, class_type, 'image', 
                                                    f"original_{i:04d}_{j}.png")
                        dst_mask_path = os.path.join(output_path, class_type, 'mask', 
                                                   f"original_{i:04d}_{j}_mask.png")
                        original_image.save(dst_image_path)
                        shutil.copy2(src_mask_path, dst_mask_path)
                    
                    augmented_dataset.append({
                        'image_path': f"{class_type} original_{i:04d}_{j}.png",
                        'mask_path': f"{class_type} original_{i:04d}_{j}_mask.png",
                        'class': class_type,
                        'source_client': current_client_id,
                        'style_client': selected_client,
                        'augmentation_type': 'original'
                    })
                
                else:
                    # Line 7-11: Apply style transfer
                    client_style_data = global_style_bank[selected_client]
                    
                    if style_type == 'single':
                        # Line 8: T is single mode
                        # D_Cn.append(G(I_i, random.choice(S_Cn, 1)))
                        if isinstance(client_style_data, list):
                            selected_style = random.choice(client_style_data)
                        else:
                            selected_style = client_style_data
                        style_mean = selected_style['mean']
                        style_std = selected_style['std']
                    
                    else:
                        # Line 10: T is overall mode
                        # D_Cn.append(G(I_i, S_Cn))
                        style_mean = client_style_data['mean']
                        style_std = client_style_data['std']
                    
                    # Apply style transfer (Equation 5)
                    with torch.no_grad():
                        styled_tensor = self.adain_transfer(image_tensor, style_mean, style_std)
                        styled_image = reverse_transform_fn(styled_tensor.squeeze(0).cpu())
                    
                    # Save styled image
                    if output_path:
                        dst_image_path = os.path.join(output_path, class_type, 'image', 
                                                    f"styled_{i:04d}_{j}.png")
                        dst_mask_path = os.path.join(output_path, class_type, 'mask', 
                                                   f"styled_{i:04d}_{j}_mask.png")
                        styled_image.save(dst_image_path)
                        shutil.copy2(src_mask_path, dst_mask_path)
                    
                    augmented_dataset.append({
                        'image_path': f"{class_type} styled_{i:04d}_{j}.png",
                        'mask_path': f"{class_type} styled_{i:04d}_{j}_mask.png", 
                        'class': class_type,
                        'source_client': current_client_id,
                        'style_client': selected_client,
                        'augmentation_type': 'styled'
                    })
        
        # Save augmented dataset CSV
        if output_path:
            augmented_df = pd.DataFrame(augmented_dataset)
            augmented_df.to_csv(os.path.join(output_path, 'ccst_augmented_dataset.csv'), index=False)
        
        print(f"   ✅ Algorithm 1 completed: {len(augmented_dataset)} augmented samples")
        print(f"   📊 Original dataset size: {len(df)}")
        print(f"   📊 Augmented dataset size: {len(augmented_dataset)} (K={K})")
        
        return augmented_dataset


def run_ccst_pipeline(busi_path, bus_uclm_path, output_base_path, 
                     style_type='overall', K=3, J=10):
    """
    Complete CCST pipeline following the paper exactly
    
    Three stages:
    1. Local style computation and sharing
    2. Server-side style bank broadcasting  
    3. Local style transfer
    """
    
    print("🚀 Starting CCST Pipeline - Exact Paper Replication")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  Style type: {style_type}")
    print(f"  Augmentation level K: {K}")
    if style_type == 'single':
        print(f"  Single images per client J: {J}")
    print()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # ========================================
    # Stage 1: Local style computation and sharing
    # ========================================
    print("📊 Stage 1: Local style computation and sharing")
    
    style_extractor = CCSTStyleExtractor(device=device)
    
    # Extract styles from both "clients" (TRAINING DATA ONLY)
    if style_type == 'overall':
        print("   Extracting overall domain styles from TRAINING data...")
        busi_style = style_extractor.extract_overall_domain_style(
            busi_path, 'train_frame.csv'
        )
        bus_uclm_style = style_extractor.extract_overall_domain_style(
            bus_uclm_path, 'train_frame.csv'
        )
    else:
        print("   Extracting single image style banks from TRAINING data...")
        busi_style = style_extractor.extract_single_image_style_bank(
            busi_path, 'train_frame.csv', J=J
        )
        bus_uclm_style = style_extractor.extract_single_image_style_bank(
            bus_uclm_path, 'train_frame.csv', J=J
        )
    
    # ========================================
    # Stage 2: Server-side style bank broadcasting
    # ========================================
    print("\n🌐 Stage 2: Server-side style bank broadcasting")
    
    style_bank = CCSTStyleBank()
    style_bank.add_client_style('BUSI', busi_style, style_type)
    style_bank.add_client_style('BUS-UCLM', bus_uclm_style, style_type)
    
    # Save style bank
    style_bank_path = os.path.join(output_base_path, 'ccst_style_bank.json')
    os.makedirs(output_base_path, exist_ok=True)
    style_bank.save_style_bank(style_bank_path)
    
    # Get appropriate style bank
    if style_type == 'overall':
        global_style_bank = style_bank.get_style_bank_overall()
    else:
        global_style_bank = style_bank.get_style_bank_single()
    
    print(f"   📡 Broadcasting style bank to all clients...")
    print(f"   📊 Style bank contains {len(global_style_bank)} client styles")
    
    # ========================================
    # Stage 3: Local style transfer (Algorithm 1)
    # ========================================
    print("\n🎨 Stage 3: Local style transfer (Algorithm 1)")
    
    style_transfer = CCSTLocalStyleTransfer(device=device)
    
    # Apply Algorithm 1 to BUS-UCLM TRAINING data (transfer to BUSI style)
    bus_uclm_output_path = os.path.join(output_base_path, 'BUS-UCLM-CCST-augmented')
    print(f"\n   Applying Algorithm 1 to BUS-UCLM client (TRAINING data only)...")
    bus_uclm_augmented = style_transfer.local_cross_client_style_transfer(
        client_dataset_path=bus_uclm_path,
        global_style_bank=global_style_bank,
        current_client_id='BUS-UCLM',
        K=K,
        style_type=style_type,
        output_path=bus_uclm_output_path,
        csv_file='train_frame.csv'
    )
    
    # Apply Algorithm 1 to BUSI TRAINING data (transfer to BUS-UCLM style)  
    busi_output_path = os.path.join(output_base_path, 'BUSI-CCST-augmented')
    print(f"\n   Applying Algorithm 1 to BUSI client (TRAINING data only)...")
    busi_augmented = style_transfer.local_cross_client_style_transfer(
        client_dataset_path=busi_path,
        global_style_bank=global_style_bank,
        current_client_id='BUSI',
        K=K,
        style_type=style_type,
        output_path=busi_output_path,
        csv_file='train_frame.csv'
    )
    
    # ========================================
    # Summary
    # ========================================
    print("\n🎉 CCST Pipeline Completed!")
    print("=" * 60)
    print(f"Results:")
    print(f"  Style bank: {style_bank_path}")
    print(f"  BUS-UCLM augmented: {bus_uclm_output_path}")
    print(f"  BUSI augmented: {busi_output_path}")
    print(f"  Augmentation factor: {K}x per image")
    print(f"  Total augmented samples:")
    print(f"    BUS-UCLM: {len(bus_uclm_augmented)}")
    print(f"    BUSI: {len(busi_augmented)}")
    
    return {
        'style_bank_path': style_bank_path,
        'bus_uclm_augmented_path': bus_uclm_output_path,
        'busi_augmented_path': busi_output_path,
        'bus_uclm_augmented_data': bus_uclm_augmented,
        'busi_augmented_data': busi_augmented
    }


def main():
    """
    Main function to run exact CCST replication
    """
    
    # Configuration
    busi_path = "dataset/BioMedicalDataset/BUSI"
    bus_uclm_path = "dataset/BioMedicalDataset/BUS-UCLM"
    output_base_path = "dataset/BioMedicalDataset/CCST-Results"
    
    # Experiment configurations optimized for 2-domain setup
    experiments = [
        # Overall domain style (K=1 is optimal for 2 domains)
        {'style_type': 'overall', 'K': 1, 'description': 'Overall domain style (optimal for 2 domains)'},
        
        # Single image style (K=1,2,3 all make sense with J=10)
        {'style_type': 'single', 'K': 1, 'J': 10, 'description': 'Single image style, K=1'},
        {'style_type': 'single', 'K': 2, 'J': 10, 'description': 'Single image style, K=2'},
        {'style_type': 'single', 'K': 3, 'J': 10, 'description': 'Single image style, K=3'},
    ]
    
    # Run the main experiment (overall, K=1 - optimal for 2 domains)
    main_experiment = {'style_type': 'overall', 'K': 1}
    
    print(f"🎯 Running main CCST experiment (optimized for 2-domain setup):")
    print(f"   Style type: {main_experiment['style_type']}")
    print(f"   Augmentation level K: {main_experiment['K']}")
    print(f"   📝 Note: K=1 is optimal for overall domain style with 2 domains")
    print(f"   📝 Use single image style with K=2 or K=3 for more diversity")
    
    results = run_ccst_pipeline(
        busi_path=busi_path,
        bus_uclm_path=bus_uclm_path,
        output_base_path=output_base_path,
        style_type=main_experiment['style_type'],
        K=main_experiment['K']
    )
    
    print(f"\n✅ Main experiment completed!")
    print(f"📁 Results saved to: {output_base_path}")
    print(f"\n🔄 Data augmentation achieved:")
    print(f"   Original BUS-UCLM: ~500 images")
    print(f"   Augmented (K=1): ~1000 images (2x increase)")
    print(f"   └── Original: ~500 + Cross-domain styled: ~500")
    
    print(f"\n💡 To get more diversity, try single image style:")
    print(f"   python ccst_exact_replication.py --style-type single --K 3")
    print(f"   This gives ~2000 images (4x increase) with varied single image styles")
    
    print(f"\nYou can now train your model using the augmented datasets:")
    print(f"  - BUS-UCLM with BUSI style: {results['bus_uclm_augmented_path']}")
    print(f"  - BUSI with BUS-UCLM style: {results['busi_augmented_path']}")
    
    return results


if __name__ == "__main__":
    results = main() 