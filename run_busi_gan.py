#!/usr/bin/env python3
"""
Simple execution script for BUSI Conditional GAN
This script provides easy-to-use functions for training and generating synthetic data.
"""

import os
import subprocess
import sys

def train_gan(data_dir, epochs=200, batch_size=8, checkpoint_dir="checkpoints"):
    """Train the BUSI Conditional GAN"""
    print("🚀 Starting BUSI GAN Training...")
    print(f"   Data directory: {data_dir}")
    print(f"   Epochs: {epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Checkpoint directory: {checkpoint_dir}")
    
    cmd = [
        sys.executable, "synthetic_busi_gan.py",
        "--mode", "train",
        "--data_dir", data_dir,
        "--epochs", str(epochs),
        "--batch_size", str(batch_size),
        "--checkpoint_dir", checkpoint_dir
    ]
    
    subprocess.run(cmd)

def generate_synthetic_data(checkpoint_path, output_dir="synthetic_busi_dataset", 
                          num_benign=175, num_malignant=89):
    """Generate synthetic BUSI data using trained model"""
    print("🎯 Generating Synthetic BUSI Data...")
    print(f"   Checkpoint: {checkpoint_path}")
    print(f"   Output directory: {output_dir}")
    print(f"   Benign samples: {num_benign}")
    print(f"   Malignant samples: {num_malignant}")
    
    cmd = [
        sys.executable, "synthetic_busi_gan.py",
        "--mode", "generate",
        "--checkpoint", checkpoint_path,
        "--output_dir", output_dir,
        "--num_benign", str(num_benign),
        "--num_malignant", str(num_malignant)
    ]
    
    subprocess.run(cmd)

def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Easy BUSI GAN execution')
    parser.add_argument('action', choices=['train', 'generate'], 
                       help='Action to perform')
    parser.add_argument('--data_dir', default='dataset/BUSI',
                       help='BUSI dataset directory (for training)')
    parser.add_argument('--checkpoint', 
                       help='Checkpoint file (for generation)')
    parser.add_argument('--epochs', type=int, default=200,
                       help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--output_dir', default='synthetic_busi_dataset',
                       help='Output directory for synthetic data')
    parser.add_argument('--num_benign', type=int, default=175,
                       help='Number of benign samples to generate')
    parser.add_argument('--num_malignant', type=int, default=89,
                       help='Number of malignant samples to generate')
    
    args = parser.parse_args()
    
    if args.action == 'train':
        train_gan(args.data_dir, args.epochs, args.batch_size)
    elif args.action == 'generate':
        if not args.checkpoint:
            print("❌ Checkpoint file required for generation!")
            print("Use: python run_busi_gan.py generate --checkpoint checkpoints/busi_gan_final.pth")
            return
        generate_synthetic_data(args.checkpoint, args.output_dir, 
                              args.num_benign, args.num_malignant)

if __name__ == "__main__":
    main() 