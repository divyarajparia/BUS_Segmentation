#!/usr/bin/env python3
"""
Generate Final Synthetic BUSI Dataset for MADGNET Training
=========================================================
Generate 175 benign + 89 malignant synthetic samples using trained GAN.
"""

import os
import sys
import argparse
from synthetic_busi_gan import BUSIConditionalGAN

def main():
    parser = argparse.ArgumentParser(description='Generate final synthetic BUSI dataset')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained GAN checkpoint (e.g., checkpoints/busi_gan_final.pth)')
    parser.add_argument('--output_dir', type=str, default='synthetic_busi_madgnet',
                       help='Output directory for synthetic dataset')
    parser.add_argument('--num_benign', type=int, default=175,
                       help='Number of benign samples to generate')
    parser.add_argument('--num_malignant', type=int, default=89,
                       help='Number of malignant samples to generate')
    
    args = parser.parse_args()
    
    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"❌ Checkpoint not found: {args.checkpoint}")
        print("\n🔍 Available checkpoints:")
        checkpoint_dir = os.path.dirname(args.checkpoint) or 'checkpoints'
        if os.path.exists(checkpoint_dir):
            for f in os.listdir(checkpoint_dir):
                if f.endswith('.pth'):
                    print(f"   {os.path.join(checkpoint_dir, f)}")
        return
    
    print("🎨 Generating Final Synthetic BUSI Dataset for MADGNET Training")
    print("=" * 70)
    print(f"📁 Checkpoint: {args.checkpoint}")
    print(f"📁 Output: {args.output_dir}")
    print(f"🎯 Target: {args.num_benign} benign + {args.num_malignant} malignant = {args.num_benign + args.num_malignant} total")
    
    # Setup device
    device = 'cuda' if __import__('torch').cuda.is_available() else 'cpu'
    print(f"💻 Device: {device}")
    
    # Initialize GAN
    gan = BUSIConditionalGAN(
        noise_dim=100,
        num_classes=2,
        img_size=256,
        device=device
    )
    
    # Load checkpoint
    print("\n📥 Loading trained GAN...")
    try:
        gan.load_checkpoint(args.checkpoint)
        print("✅ GAN loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load GAN: {e}")
        return
    
    # Generate synthetic dataset
    print("\n🚀 Generating synthetic dataset...")
    try:
        gan.generate_synthetic_dataset(
            output_dir=args.output_dir,
            num_benign=args.num_benign,
            num_malignant=args.num_malignant
        )
        
        print(f"\n🎉 Success! Synthetic dataset generated at: {args.output_dir}")
        print(f"📊 Generated {args.num_benign + args.num_malignant} samples total")
        print(f"📁 Ready for MADGNET training!")
        
        # Print directory structure
        print(f"\n📂 Directory structure:")
        print(f"   {args.output_dir}/")
        print(f"   ├── benign/")
        print(f"   │   ├── image/ ({args.num_benign} images)")
        print(f"   │   └── mask/ ({args.num_benign} masks)")
        print(f"   ├── malignant/")
        print(f"   │   ├── image/ ({args.num_malignant} images)")
        print(f"   │   └── mask/ ({args.num_malignant} masks)")
        print(f"   └── synthetic_dataset.csv")
        
        return True
        
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 