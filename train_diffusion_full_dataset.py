"""
Train Diffusion Model on Full BUSI Dataset
==========================================
Train the diffusion model properly using the complete BUSI dataset.
"""

import torch
import torch.nn as nn
import os
from torch.utils.data import DataLoader
from simple_diffusion_busi import SimpleUNet, BUSIDataset, SimpleDiffusion
from tqdm import tqdm

def train_on_full_dataset():
    """Train diffusion model on complete BUSI dataset"""
    
    print("🏥 Training Diffusion Model on Full BUSI Dataset")
    print("=" * 60)
    
    # Check if full dataset exists
    full_dataset_path = "dataset/BioMedicalDataset/BUSI"
    if not os.path.exists(full_dataset_path):
        print(f"❌ Full dataset not found: {full_dataset_path}")
        print(f"   You need to access the complete BUSI dataset")
        return
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📱 Device: {device}")
    
    # Create dataset and dataloader
    train_csv = os.path.join(full_dataset_path, "train_frame.csv")
    if not os.path.exists(train_csv):
        print(f"❌ Training CSV not found: {train_csv}")
        print(f"   Make sure the full dataset structure is correct")
        return
    
    dataset = BUSIDataset(train_csv, full_dataset_path)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=2)
    
    print(f"✅ Dataset loaded: {len(dataset)} training samples")
    
    # Model and training setup
    model = SimpleUNet().to(device)
    diffusion = SimpleDiffusion()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    
    # Training parameters
    num_epochs = 50
    save_interval = 10
    
    print(f"🎯 Training Parameters:")
    print(f"   Epochs: {num_epochs}")
    print(f"   Batch size: {dataloader.batch_size}")
    print(f"   Learning rate: {optimizer.param_groups[0]['lr']}")
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training loop
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for batch_idx, (images, class_labels) in enumerate(pbar):
            images = images.to(device)
            class_labels = class_labels.to(device)
            
            # Random timesteps
            t = torch.randint(0, diffusion.num_timesteps, (images.shape[0],), device=device)
            
            # Forward pass
            loss = diffusion.p_losses(model, images, t, class_labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        avg_loss = epoch_loss / num_batches
        print(f'Epoch {epoch+1}, Average Loss: {avg_loss:.4f}')
        
        # Save checkpoint
        if (epoch + 1) % save_interval == 0 or epoch == num_epochs - 1:
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch + 1,
                'loss': avg_loss,
            }
            checkpoint_path = f'full_diffusion_model_epoch_{epoch+1}.pth'
            torch.save(checkpoint, checkpoint_path)
            print(f'✅ Saved checkpoint: {checkpoint_path}')
    
    print(f"\n🎉 Training Complete!")
    print(f"   Final checkpoint: full_diffusion_model_epoch_{num_epochs}.pth")
    print(f"   This model was trained on {len(dataset)} samples")
    print(f"   Quality should be MUCH better than the debug model!")

if __name__ == "__main__":
    train_on_full_dataset() 