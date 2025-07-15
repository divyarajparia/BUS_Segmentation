import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm
from torchvision.transforms.functional import to_pil_image
import torch.nn.functional as F

# Import original classes (we'll override specific methods)
from ccst_exact_replication import *

class FixedCCSTStyleExtractor(CCSTStyleExtractor):
    """Fixed CCST Style Extractor with proper normalization"""
    
    def apply_style_transfer(self, content_image, style_dict, alpha=0.8):
        """Apply PROPER style transfer using trained AdaIN with FIXED normalization"""
        with torch.no_grad():
            device = content_image.device
            style_mean = style_dict['mean'].to(device)
            style_std = style_dict['std'].to(device)
            
            # Apply style transfer
            stylized_image = self.style_transfer_model(content_image, style_mean, style_std, alpha)
            
            # 🔧 FIX: Proper normalization instead of naive clamping
            # Min-max normalize to [0, 1] range to prevent black images
            min_val = stylized_image.min()
            max_val = stylized_image.max()
            if max_val > min_val:
                stylized_image = (stylized_image - min_val) / (max_val - min_val)
            else:
                stylized_image = torch.zeros_like(stylized_image) + 0.5  # Gray if constant
            
            return stylized_image

def fixed_tensor_to_pil_global(tensor):
    """Global fixed tensor to PIL conversion"""
    # Remove batch dimension if present
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    if tensor.dim() == 3 and tensor.size(0) == 1:
        tensor = tensor.squeeze(0)
    
    # Convert to numpy
    img_np = tensor.detach().cpu().numpy()
    
    # Use min-max normalization to [0, 255]
    if img_np.max() > img_np.min():
        normalized = (img_np - img_np.min()) / (img_np.max() - img_np.min()) * 255
    else:
        normalized = np.full_like(img_np, 128)
    
    # Ensure valid range
    normalized = np.clip(normalized, 0, 255)
    
    # Convert to PIL
    img_pil = Image.fromarray(normalized.astype(np.uint8), mode='L')
    
    return img_pil

def run_fixed_ccst_pipeline(source_dataset, source_csv, target_dataset, target_csv, 
                           output_dir, K=1, J_samples=None):
    """
    Run the FIXED CCST pipeline with proper normalization
    """
    print("🚀 Starting FIXED CCST Pipeline - No More Black Images!")
    print("🎨 Direction: Extract BUSI style → Apply to BUS-UCLM content")
    print("=" * 80)
    
    # Step 1: Extract target domain style (BUSI)
    print(f"\n📊 Step 1: Extracting BUSI style from {target_dataset}...")
    style_extractor = FixedCCSTStyleExtractor(device)  # Use fixed version
    
    # Train decoder first for high-quality reconstruction
    print("🎓 Training decoder for high-quality reconstruction...")
    style_extractor.train_decoder_with_data(target_dataset, target_csv, batch_size=4)
    
    target_style = style_extractor.extract_overall_domain_style(
        target_dataset, target_csv, J_samples
    )
    
    # Step 2: Apply style transfer to source domain (BUS-UCLM)
    print(f"\n🎨 Step 2: Applying BUSI style to BUS-UCLM content from {source_dataset}...")
    
    # Load source data
    source_csv_path = os.path.join(source_dataset, source_csv)
    df = pd.read_csv(source_csv_path)
    
    # Create output directories
    for class_type in ['benign', 'malignant']:
        os.makedirs(os.path.join(output_dir, class_type, 'image'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, class_type, 'mask'), exist_ok=True)
    
    # Process all images with proper normalization
    styled_data = []
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Applying FIXED BUSI style to BUS-UCLM"):
        try:
            # Get image and mask paths
            image_filename = row['image_path']
            mask_filename = row['mask_path']
            
            # Handle different formats
            if ' ' in image_filename:
                if '(' in image_filename:
                    class_type = image_filename.split()[0]
                    img_path = os.path.join(source_dataset, class_type, 'image', image_filename)
                    mask_path = os.path.join(source_dataset, class_type, 'mask', mask_filename)
                else:
                    class_type = image_filename.split()[0]
                    image_name = image_filename.split()[1]
                    mask_name = mask_filename.split()[1]
                    img_path = os.path.join(source_dataset, class_type, 'images', image_name)
                    mask_path = os.path.join(source_dataset, class_type, 'masks', mask_name)
            else:
                img_path = os.path.join(source_dataset, image_filename)
                mask_path = os.path.join(source_dataset, mask_filename)
                class_type = 'unknown'
            
            if not os.path.exists(img_path) or not os.path.exists(mask_path):
                continue
            
            # Load image
            image = Image.open(img_path).convert('L')
            image_tensor = transform(image).unsqueeze(0).to(device)
            
            # Apply FIXED style transfer
            stylized_tensor = style_extractor.apply_style_transfer(
                image_tensor, target_style, alpha=0.8
            )
            
            # Convert to PIL with FIXED normalization
            styled_image = fixed_tensor_to_pil_global(stylized_tensor)
            
            # Save styled image and copy mask
            styled_filename = f"styled_{idx:04d}.png"
            styled_mask_filename = f"styled_{idx:04d}_mask.png"
            
            styled_image_path = os.path.join(output_dir, class_type, 'image', styled_filename)
            styled_mask_path = os.path.join(output_dir, class_type, 'mask', styled_mask_filename)
            
            styled_image.save(styled_image_path)
            
            # Copy mask
            mask_image = Image.open(mask_path)
            mask_image.save(styled_mask_path)
            
            # Record metadata
            styled_data.append({
                'image_path': f"{class_type} {styled_filename}",
                'mask_path': f"{class_type} {styled_mask_filename}",
                'class': class_type,
                'source_client': 'BUS-UCLM',
                'style_client': 'BUSI',
                'augmentation_type': 'styled'
            })
            
        except Exception as e:
            print(f"   ⚠️ Error processing {idx}: {e}")
            continue
    
    # Save dataset CSV
    styled_csv_path = os.path.join(output_dir, 'styled_dataset.csv')
    styled_df = pd.DataFrame(styled_data)
    styled_df.to_csv(styled_csv_path, index=False)
    
    print(f"\n✅ FIXED CCST Pipeline Complete!")
    print(f"   Generated {len(styled_data)} styled images (should NOT be black!)")
    print(f"   Output directory: {output_dir}")
    print(f"   Dataset CSV: {styled_csv_path}")
    
    return styled_data
