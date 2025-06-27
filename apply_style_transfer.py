import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm
import shutil
from style_transfer import CycleGAN, Generator

class StyleTransferApplier:
    def __init__(self, model_path, device='cuda'):
        self.device = device
        self.cyclegan = CycleGAN(device)
        self.cyclegan.load_models(model_path)
        
    def apply_style_transfer(self, bus_uclm_dir, output_dir):
        """Apply style transfer to all BUS-UCLM images and save to output directory"""
        
        # Create output directory structure
        os.makedirs(os.path.join(output_dir, 'benign', 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'benign', 'masks'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'malignant', 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'malignant', 'masks'), exist_ok=True)
        
        # Transform for input images
        transform = transforms.Compose([
            transforms.Resize((352, 352)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        
        # Process each class
        for class_name in ['benign', 'malignant']:
            img_dir = os.path.join(bus_uclm_dir, class_name, 'images')
            mask_dir = os.path.join(bus_uclm_dir, class_name, 'masks')
            
            if not os.path.exists(img_dir):
                continue
                
            print(f"Processing {class_name} images...")
            
            for img_file in tqdm(os.listdir(img_dir)):
                if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                    
                # Load original image
                img_path = os.path.join(img_dir, img_file)
                mask_path = os.path.join(mask_dir, img_file)
                
                if not os.path.exists(mask_path):
                    continue
                
                # Load and transform image
                img = Image.open(img_path).convert('L')
                img_tensor = transform(img).unsqueeze(0).to(self.device)
                
                # Apply style transfer
                with torch.no_grad():
                    transferred_img = self.cyclegan.transfer_style(img_tensor)
                
                # Convert back to PIL image
                transferred_img = transferred_img.squeeze(0).cpu()
                transferred_img = (transferred_img + 1) / 2  # Denormalize from [-1,1] to [0,1]
                transferred_img = transforms.ToPILImage()(transferred_img)
                
                # Save transferred image
                output_img_path = os.path.join(output_dir, class_name, 'images', img_file)
                transferred_img.save(output_img_path)
                
                # Copy mask
                output_mask_path = os.path.join(output_dir, class_name, 'masks', img_file)
                shutil.copy2(mask_path, output_mask_path)
        
        print(f"Style transfer completed! Transferred images saved to {output_dir}")

def create_combined_dataset(busi_dir, style_transferred_dir, output_dir):
    """Create a combined dataset with BUSI and style-transferred BUS-UCLM"""
    
    # Create output directory structure
    os.makedirs(os.path.join(output_dir, 'benign', 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'benign', 'masks'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'malignant', 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'malignant', 'masks'), exist_ok=True)
    
    # Copy BUSI images
    print("Copying BUSI images...")
    for class_name in ['benign', 'malignant']:
        busi_img_dir = os.path.join(busi_dir, class_name, 'image')
        busi_mask_dir = os.path.join(busi_dir, class_name, 'mask')
        
        if os.path.exists(busi_img_dir):
            for img_file in os.listdir(busi_img_dir):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    # Copy image
                    src_img = os.path.join(busi_img_dir, img_file)
                    dst_img = os.path.join(output_dir, class_name, 'images', f"busi_{img_file}")
                    shutil.copy2(src_img, dst_img)
                    
                    # Copy mask
                    mask_file = img_file.replace('.jpg', '_mask.png').replace('.png', '_mask.png')
                    src_mask = os.path.join(busi_mask_dir, mask_file)
                    if os.path.exists(src_mask):
                        dst_mask = os.path.join(output_dir, class_name, 'masks', f"busi_{mask_file}")
                        shutil.copy2(src_mask, dst_mask)
    
    # Copy style-transferred images
    print("Copying style-transferred images...")
    for class_name in ['benign', 'malignant']:
        style_img_dir = os.path.join(style_transferred_dir, class_name, 'images')
        style_mask_dir = os.path.join(style_transferred_dir, class_name, 'masks')
        
        if os.path.exists(style_img_dir):
            for img_file in os.listdir(style_img_dir):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    # Copy image
                    src_img = os.path.join(style_img_dir, img_file)
                    dst_img = os.path.join(output_dir, class_name, 'images', f"style_{img_file}")
                    shutil.copy2(src_img, dst_img)
                    
                    # Copy mask
                    src_mask = os.path.join(style_mask_dir, img_file)
                    if os.path.exists(src_mask):
                        dst_mask = os.path.join(output_dir, class_name, 'masks', img_file)
                        shutil.copy2(src_mask, dst_mask)
    
    print(f"Combined dataset created at {output_dir}")

def create_combined_csv(output_dir):
    """Create CSV files for the combined dataset"""
    
    # Collect all image-mask pairs
    all_pairs = []
    
    for class_name in ['benign', 'malignant']:
        img_dir = os.path.join(output_dir, class_name, 'images')
        mask_dir = os.path.join(output_dir, class_name, 'masks')
        
        if not os.path.exists(img_dir):
            continue
            
        for img_file in os.listdir(img_dir):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                mask_file = img_file
                
                # Handle BUSI mask naming convention
                if img_file.startswith('busi_'):
                    mask_file = img_file.replace('busi_', '').replace('.jpg', '_mask.png').replace('.png', '_mask.png')
                
                mask_path = os.path.join(mask_dir, mask_file)
                if os.path.exists(mask_path):
                    all_pairs.append({
                        'image_path': f"{class_name} {img_file}",
                        'mask_path': f"{class_name} {mask_file}"
                    })
    
    # Shuffle and split
    np.random.shuffle(all_pairs)
    
    # Split into train/test/val (70/20/10)
    total = len(all_pairs)
    train_size = int(0.7 * total)
    test_size = int(0.2 * total)
    
    train_pairs = all_pairs[:train_size]
    test_pairs = all_pairs[train_size:train_size + test_size]
    val_pairs = all_pairs[train_size + test_size:]
    
    # Save CSV files
    for split_name, pairs in [('train', train_pairs), ('test', test_pairs), ('val', val_pairs)]:
        df = pd.DataFrame(pairs)
        csv_path = os.path.join(output_dir, f'{split_name}_frame.csv')
        df.to_csv(csv_path, index=False)
        print(f"{split_name}: {len(pairs)} samples")
    
    print(f"CSV files created for combined dataset")

def main():
    # Configuration
    busi_dir = "dataset/BioMedicalDataset/BUSI"
    bus_uclm_dir = "dataset/BioMedicalDataset/BUS-UCLM"
    style_transfer_model_path = "style_transfer_epoch_50.pth"  # Update with your model path
    style_transferred_dir = "dataset/BioMedicalDataset/BUS-UCLM-style-transferred"
    combined_dataset_dir = "dataset/BioMedicalDataset/BUSI-Combined"
    
    # Step 1: Apply style transfer to BUS-UCLM
    print("Step 1: Applying style transfer to BUS-UCLM images...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    applier = StyleTransferApplier(style_transfer_model_path, device)
    applier.apply_style_transfer(bus_uclm_dir, style_transferred_dir)
    
    # Step 2: Create combined dataset
    print("Step 2: Creating combined dataset...")
    create_combined_dataset(busi_dir, style_transferred_dir, combined_dataset_dir)
    
    # Step 3: Create CSV files
    print("Step 3: Creating CSV files...")
    create_combined_csv(combined_dataset_dir)
    
    print("All steps completed successfully!")

if __name__ == "__main__":
    main() 