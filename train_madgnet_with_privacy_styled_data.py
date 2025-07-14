"""
Train MADGNet with Privacy-Preserving Styled Dataset
==================================================

Train MADGNet on combined dataset:
- Original BUSI training images
- Privacy-styled BUS-UCLM images (generated from BUS-UCLM using BUSI style statistics)

Test on BUSI test set for fair comparison.
"""

import os
import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime

# Import your existing MADGNet model
try:
    from IS2D_models.mfmsnet import MFMSNet as MADGNet
except ImportError:
    print("Warning: Could not import MADGNet. Please ensure IS2D_models is available.")
    MADGNet = None

class CombinedPrivacyDataset(Dataset):
    """
    Dataset class for combined privacy-preserving training data.
    Handles both original BUSI and privacy-styled BUS-UCLM images.
    """
    
    def __init__(self, csv_file, transform=None, target_size=(256, 256)):
        """
        Args:
            csv_file: Path to combined CSV with both BUSI and styled data
            transform: Image transformations
            target_size: Target image size (width, height)
        """
        self.df = pd.read_csv(csv_file)
        self.transform = transform
        self.target_size = target_size
        
        print(f"📊 Combined Dataset Info:")
        print(f"   Total samples: {len(self.df)}")
        
        if 'source' in self.df.columns:
            source_counts = self.df['source'].value_counts()
            for source, count in source_counts.items():
                print(f"   {source}: {count} samples")
        
        if 'class' in self.df.columns:
            class_counts = self.df['class'].value_counts()
            for class_name, count in class_counts.items():
                print(f"   {class_name}: {count} samples")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Get paths based on dataset source
        if 'dataset_path' in row and pd.notna(row['dataset_path']):
            dataset_path = row['dataset_path']
        else:
            # Fallback logic
            if 'source' in row and row['source'] == 'BUS-UCLM-styled':
                dataset_path = 'dataset/BioMedicalDataset/BUS-UCLM-Privacy-Styled'
            else:
                dataset_path = 'dataset/BioMedicalDataset/BUSI'
        
        # Determine class
        if 'class' in row:
            class_name = row['class']
        else:
            # Extract from image filename
            if 'benign' in row['image_path'].lower():
                class_name = 'benign'
            else:
                class_name = 'malignant'
        
        # Build full paths
        image_path = os.path.join(dataset_path, class_name, 'image', row['image_path'])
        mask_path = os.path.join(dataset_path, class_name, 'mask', row['mask_path'])
        
        # Load image and mask
        try:
            image = Image.open(image_path).convert('RGB')
            mask = Image.open(mask_path).convert('L')
            
            # Resize
            image = image.resize(self.target_size, Image.BILINEAR)
            mask = mask.resize(self.target_size, Image.NEAREST)
            
            # Apply transforms
            if self.transform:
                image = self.transform(image)
            else:
                # Default transform
                image = transforms.ToTensor()(image)
            
            # Convert mask to tensor
            mask = torch.from_numpy(np.array(mask, dtype=np.float32) / 255.0)
            mask = mask.unsqueeze(0)  # Add channel dimension
            
            return image, mask, class_name
            
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            # Return dummy data
            dummy_image = torch.zeros(3, self.target_size[1], self.target_size[0])
            dummy_mask = torch.zeros(1, self.target_size[1], self.target_size[0])
            return dummy_image, dummy_mask, 'benign'

class PrivacyPreservingTrainer:
    """Trainer for MADGNet with privacy-preserving styled data."""
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.model.to(device)
        
        # Loss function
        self.criterion = nn.BCEWithLogitsLoss()
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.dice_scores = []
    
    def dice_coefficient(self, pred, target, smooth=1e-6):
        """Calculate Dice coefficient."""
        pred = torch.sigmoid(pred)
        pred = (pred > 0.5).float()
        
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()
        
        dice = (2.0 * intersection + smooth) / (union + smooth)
        return dice.item()
    
    def train_epoch(self, train_loader, optimizer):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        total_dice = 0
        
        pbar = tqdm(train_loader, desc="Training")
        for batch_idx, (images, masks, _) in enumerate(pbar):
            images, masks = images.to(self.device), masks.to(self.device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Calculate metrics
            dice = self.dice_coefficient(outputs, masks)
            total_loss += loss.item()
            total_dice += dice
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Dice': f'{dice:.4f}'
            })
        
        avg_loss = total_loss / len(train_loader)
        avg_dice = total_dice / len(train_loader)
        
        return avg_loss, avg_dice
    
    def validate(self, val_loader):
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        total_dice = 0
        
        with torch.no_grad():
            for images, masks, _ in tqdm(val_loader, desc="Validating"):
                images, masks = images.to(self.device), masks.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                dice = self.dice_coefficient(outputs, masks)
                
                total_loss += loss.item()
                total_dice += dice
        
        avg_loss = total_loss / len(val_loader)
        avg_dice = total_dice / len(val_loader)
        
        return avg_loss, avg_dice
    
    def train(self, train_loader, val_loader, optimizer, scheduler, epochs, output_dir):
        """Full training loop."""
        best_dice = 0
        best_model_path = os.path.join(output_dir, 'best_model.pth')
        
        # Training log
        log_file = os.path.join(output_dir, 'training_log.txt')
        
        with open(log_file, 'w') as f:
            f.write(f"Privacy-Preserving MADGNet Training Log\n")
            f.write(f"Started: {datetime.now()}\n")
            f.write(f"Total epochs: {epochs}\n\n")
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            print("-" * 50)
            
            # Training
            train_loss, train_dice = self.train_epoch(train_loader, optimizer)
            
            # Validation
            val_loss, val_dice = self.validate(val_loader)
            
            # Scheduler step
            if scheduler:
                scheduler.step(val_loss)
            
            # Save metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.dice_scores.append(val_dice)
            
            # Log results
            log_text = f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val Dice={val_dice:.4f}\n"
            print(log_text.strip())
            
            with open(log_file, 'a') as f:
                f.write(log_text)
            
            # Save best model
            if val_dice > best_dice:
                best_dice = val_dice
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_dice': best_dice,
                    'train_losses': self.train_losses,
                    'val_losses': self.val_losses,
                    'dice_scores': self.dice_scores
                }, best_model_path)
                print(f"💾 New best model saved: Dice = {best_dice:.4f}")
        
        return best_dice

def main():
    parser = argparse.ArgumentParser(description='Train MADGNet with Privacy-Preserving Styled Data')
    parser.add_argument('--combined_csv', type=str, required=True,
                        help='Path to combined training CSV')
    parser.add_argument('--test_csv', type=str, required=True,
                        help='Path to test CSV (BUSI test set)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--output_dir', type=str, default='privacy_preserving_results',
                        help='Output directory for results')
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='Validation split ratio')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("🚀 Privacy-Preserving MADGNet Training")
    print("=" * 60)
    print(f"📊 Training CSV: {args.combined_csv}")
    print(f"📊 Test CSV: {args.test_csv}")
    print(f"📁 Output: {args.output_dir}")
    
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Device: {device}")
    
    # Data transforms
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load combined dataset
    full_dataset = CombinedPrivacyDataset(args.combined_csv, transform=train_transform)
    
    # Split into train/val
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Update validation transform
    val_dataset.dataset.transform = val_transform
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    print(f"📊 Dataset split: {len(train_dataset)} train, {len(val_dataset)} val")
    
    # Initialize model
    if MADGNet is None:
        print("❌ MADGNet model not available. Please check imports.")
        return
    
    model = MADGNet(num_classes=1)  # Binary segmentation
    
    # Initialize trainer
    trainer = PrivacyPreservingTrainer(model, device)
    
    # Optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)
    
    # Train model
    print("\n🎯 Starting Training...")
    best_dice = trainer.train(train_loader, val_loader, optimizer, scheduler, args.epochs, args.output_dir)
    
    # Save final results
    results = {
        'best_dice': float(best_dice),
        'training_completed': str(datetime.now()),
        'total_epochs': args.epochs,
        'final_train_loss': float(trainer.train_losses[-1]),
        'final_val_loss': float(trainer.val_losses[-1]),
        'final_dice': float(trainer.dice_scores[-1])
    }
    
    with open(os.path.join(args.output_dir, 'training_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n🎉 Training Complete!")
    print(f"✅ Best Dice Score: {best_dice:.4f}")
    print(f"✅ Model saved to: {args.output_dir}/best_model.pth")
    print(f"✅ Results saved to: {args.output_dir}/training_results.json")
    
    print(f"\n🎯 Next: Run evaluation on BUSI test set")
    print(f"   python evaluate_privacy_preserving_model.py \\")
    print(f"     --model_path {args.output_dir}/best_model.pth \\")
    print(f"     --test_csv {args.test_csv} \\")
    print(f"     --output_dir {args.output_dir}/evaluation/")

if __name__ == "__main__":
    main() 