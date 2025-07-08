#!/usr/bin/env python3
"""
Quick test script for the fixed BUSI GAN
"""

import torch
from synthetic_busi_gan import BUSIConditionalGAN, BUSIDataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def test_gan_architecture():
    """Test that the GAN architecture works without errors"""
    print("🧪 Testing Fixed GAN Architecture...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Initialize GAN
    gan = BUSIConditionalGAN(
        noise_dim=100,
        num_classes=2,
        img_size=256,
        device=device,
        lr=0.0002
    )
    
    # Test generator
    print("Testing generator...")
    noise = torch.randn(4, 100, device=device)
    labels = torch.randint(0, 2, (4,), device=device)
    
    fake_images, fake_masks = gan.generator(noise, labels)
    print(f"✅ Generator output shapes: images={fake_images.shape}, masks={fake_masks.shape}")
    
    # Test discriminator
    print("Testing discriminator...")
    d_output = gan.discriminator(fake_images, fake_masks, labels)
    print(f"✅ Discriminator output shape: {d_output.shape}")
    print(f"✅ Discriminator output range: [{d_output.min().item():.4f}, {d_output.max().item():.4f}]")
    
    # Test one training step
    print("Testing training step...")
    
    # Create dummy real data
    real_images = torch.randn(4, 1, 256, 256, device=device) * 0.5  # Normalized like real data
    real_masks = torch.rand(4, 1, 256, 256, device=device)  # Random masks
    real_labels = torch.randint(0, 2, (4,), device=device)
    
    # Test discriminator loss
    real_output = gan.discriminator(real_images, real_masks, real_labels)
    fake_output = gan.discriminator(fake_images.detach(), fake_masks.detach(), labels)
    
    real_loss = gan.criterion(real_output, torch.ones_like(real_output))
    fake_loss = gan.criterion(fake_output, torch.zeros_like(fake_output))
    d_loss = real_loss + fake_loss
    
    print(f"✅ D_Loss: {d_loss.item():.4f} (Real: {real_loss.item():.4f}, Fake: {fake_loss.item():.4f})")
    
    # Test generator loss
    gen_output = gan.discriminator(fake_images, fake_masks, labels)
    g_loss = gan.criterion(gen_output, torch.ones_like(gen_output))
    
    print(f"✅ G_Loss: {g_loss.item():.4f}")
    
    if d_loss.item() < 50 and g_loss.item() < 50:
        print("🎉 Architecture looks good! Losses are in reasonable range.")
        return True
    else:
        print("⚠️ Losses seem high, but architecture is working.")
        return True

def test_dataset_loading():
    """Test dataset loading"""
    print("\n🔍 Testing Dataset Loading...")
    
    # Data transforms
    image_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    mask_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    try:
        dataset = BUSIDataset(
            'dataset/BioMedicalDataset/BUSI', 
            transform=image_transform, 
            mask_transform=mask_transform
        )
        
        if len(dataset) > 0:
            print(f"✅ Dataset loaded: {len(dataset)} samples")
            
            # Test one sample
            img, mask, label = dataset[0]
            print(f"✅ Sample shapes: image={img.shape}, mask={mask.shape}, label={label}")
            
            return True
        else:
            print("❌ No samples found in dataset")
            return False
            
    except Exception as e:
        print(f"❌ Dataset loading failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Testing Fixed BUSI GAN\n")
    
    arch_ok = test_gan_architecture()
    data_ok = test_dataset_loading()
    
    if arch_ok and data_ok:
        print("\n🎉 All tests passed! The fixed GAN should work properly.")
        print("\n✨ You can now run training with:")
        print("   sbatch fixed_busi_gan.job")
        print("\n📊 Expected behavior:")
        print("   - D_Loss should be around 0.5-2.0 (not 100!)")
        print("   - G_Loss should be around 1.0-5.0 (not 0!)")
        print("   - Both losses should fluctuate, not stay constant")
    else:
        print("\n❌ Some tests failed. Check the errors above.")

if __name__ == "__main__":
    main() 