import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from style_transfer import CycleGAN
import random

def visualize_style_transfer(busi_dir, bus_uclm_dir, model_path, num_samples=5, device='cuda'):
    """Visualize style transfer results"""
    
    # Load style transfer model
    cyclegan = CycleGAN(device)
    cyclegan.load_models(model_path)
    
    # Transform
    transform = transforms.Compose([
        transforms.Resize((352, 352)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    # Get sample images from each class
    classes = ['benign', 'malignant']
    
    fig, axes = plt.subplots(len(classes), num_samples, 3, figsize=(15, 6))
    if len(classes) == 1:
        axes = axes.reshape(1, -1, 3)
    
    for class_idx, class_name in enumerate(classes):
        bus_uclm_img_dir = os.path.join(bus_uclm_dir, class_name, 'images')
        busi_img_dir = os.path.join(busi_dir, class_name, 'image')
        
        if not os.path.exists(bus_uclm_img_dir):
            continue
            
        # Get random samples
        bus_uclm_files = [f for f in os.listdir(bus_uclm_img_dir) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        busi_files = [f for f in os.listdir(busi_img_dir) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not bus_uclm_files or not busi_files:
            continue
            
        # Randomly sample
        bus_uclm_samples = random.sample(bus_uclm_files, min(num_samples, len(bus_uclm_files)))
        busi_samples = random.sample(busi_files, min(num_samples, len(busi_files)))
        
        for sample_idx in range(num_samples):
            if sample_idx < len(bus_uclm_samples):
                # Load BUS-UCLM image
                bus_uclm_path = os.path.join(bus_uclm_img_dir, bus_uclm_samples[sample_idx])
                bus_uclm_img = Image.open(bus_uclm_path).convert('L')
                bus_uclm_tensor = transform(bus_uclm_img).unsqueeze(0).to(device)
                
                # Apply style transfer
                with torch.no_grad():
                    transferred_tensor = cyclegan.transfer_style(bus_uclm_tensor)
                
                # Convert back to PIL
                transferred_img = transferred_tensor.squeeze(0).cpu()
                transferred_img = (transferred_img + 1) / 2
                transferred_img = transforms.ToPILImage()(transferred_img)
                
                # Load corresponding BUSI image for comparison
                busi_path = os.path.join(busi_img_dir, busi_samples[sample_idx])
                busi_img = Image.open(busi_path).convert('L')
                
                # Display
                axes[class_idx, sample_idx, 0].imshow(bus_uclm_img, cmap='gray')
                axes[class_idx, sample_idx, 0].set_title(f'BUS-UCLM\n{class_name}')
                axes[class_idx, sample_idx, 0].axis('off')
                
                axes[class_idx, sample_idx, 1].imshow(transferred_img, cmap='gray')
                axes[class_idx, sample_idx, 1].set_title(f'Style Transferred\n{class_name}')
                axes[class_idx, sample_idx, 1].axis('off')
                
                axes[class_idx, sample_idx, 2].imshow(busi_img, cmap='gray')
                axes[class_idx, sample_idx, 2].set_title(f'BUSI Target\n{class_name}')
                axes[class_idx, sample_idx, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('style_transfer_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Visualization saved as 'style_transfer_visualization.png'")

def compare_datasets(busi_dir, bus_uclm_dir, combined_dir):
    """Compare dataset statistics"""
    
    def count_images(dataset_dir):
        count = 0
        for class_name in ['benign', 'malignant']:
            img_dir = os.path.join(dataset_dir, class_name, 'images')
            if os.path.exists(img_dir):
                count += len([f for f in os.listdir(img_dir) 
                            if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        return count
    
    busi_count = count_images(busi_dir)
    bus_uclm_count = count_images(bus_uclm_dir)
    combined_count = count_images(combined_dir)
    
    print("\n📊 Dataset Statistics:")
    print(f"  BUSI: {busi_count} images")
    print(f"  BUS-UCLM: {bus_uclm_count} images")
    print(f"  Combined: {combined_count} images")
    print(f"  Expected combined: {busi_count + bus_uclm_count} images")
    
    # Check class distribution
    print("\n📈 Class Distribution:")
    for class_name in ['benign', 'malignant']:
        busi_class_dir = os.path.join(busi_dir, class_name, 'image')
        bus_uclm_class_dir = os.path.join(bus_uclm_dir, class_name, 'images')
        combined_class_dir = os.path.join(combined_dir, class_name, 'images')
        
        busi_class_count = len([f for f in os.listdir(busi_class_dir) 
                              if f.lower().endswith(('.png', '.jpg', '.jpeg'))]) if os.path.exists(busi_class_dir) else 0
        bus_uclm_class_count = len([f for f in os.listdir(bus_uclm_class_dir) 
                                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]) if os.path.exists(bus_uclm_class_dir) else 0
        combined_class_count = len([f for f in os.listdir(combined_class_dir) 
                                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]) if os.path.exists(combined_class_dir) else 0
        
        print(f"  {class_name.capitalize()}:")
        print(f"    BUSI: {busi_class_count}")
        print(f"    BUS-UCLM: {bus_uclm_class_count}")
        print(f"    Combined: {combined_class_count}")

def main():
    # Configuration
    busi_dir = "dataset/BioMedicalDataset/BUSI"
    bus_uclm_dir = "dataset/BioMedicalDataset/BUS-UCLM"
    combined_dir = "dataset/BioMedicalDataset/BUSI-Combined"
    style_transfer_model_path = "style_transfer_epoch_50.pth"  # Update with your model path
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("🔍 Style Transfer Visualization and Analysis")
    
    # Check if model exists
    if not os.path.exists(style_transfer_model_path):
        print(f"❌ Style transfer model not found: {style_transfer_model_path}")
        print("Please train the style transfer model first or update the path")
        return
    
    # Visualize style transfer results
    print("\n🎨 Visualizing style transfer results...")
    visualize_style_transfer(busi_dir, bus_uclm_dir, style_transfer_model_path, num_samples=3, device=device)
    
    # Compare datasets
    print("\n📊 Comparing datasets...")
    compare_datasets(busi_dir, bus_uclm_dir, combined_dir)

if __name__ == "__main__":
    main() 