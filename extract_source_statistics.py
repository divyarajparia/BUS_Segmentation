#!/usr/bin/env python3
"""
Extract Source Domain Statistics for Privacy-Preserving Adaptation
================================================================

This script extracts frequency domain statistics from the BUSI dataset
that will be used for privacy-preserving domain adaptation.

Usage:
    python extract_source_statistics.py --dataset_path dataset/BUSI
"""

import torch
from torch.utils.data import DataLoader
import os
import argparse
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Extract source domain statistics for privacy adaptation')
    parser.add_argument('--dataset_path', default='dataset/BUSI',
                       help='Path to BUSI dataset directory')
    parser.add_argument('--output_path', default='privacy_style_stats/busi_privacy_stats.json',
                       help='Output path for statistics file')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size for processing')
    
    args = parser.parse_args()
    
    # Check if dataset exists
    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        logger.error(f"❌ Dataset not found: {dataset_path}")
        logger.info("Please ensure the BUSI dataset is available at the specified path")
        return
    
    # Create output directory
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info("🔊 Starting source domain statistics extraction...")
    logger.info(f"   Dataset: {dataset_path}")
    logger.info(f"   Output:  {output_path}")
    
    try:
        # Import after path setup
        from dataset.BioMedicalDataset.BUSISegmentationDataset import BUSISegmentationDataset
        from advanced_privacy_methods import FrequencyDomainPrivacyAdapter
        from torchvision import transforms
        
        # Load BUSI dataset
        train_csv = dataset_path / 'train_frame.csv'
        if not train_csv.exists():
            logger.error(f"❌ Train CSV not found: {train_csv}")
            return
        
        # Simple transform to convert PIL to tensor
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])
        
        dataset = BUSISegmentationDataset(
            dataset_dir=str(dataset_path),
            mode='train',  # Use train mode to load train_frame.csv
            transform=transform,  # Convert PIL to tensor
            target_transform=transform  # Convert mask PIL to tensor
        )
        
        dataloader = DataLoader(
            dataset, 
            batch_size=args.batch_size, 
            shuffle=False, 
            num_workers=2
        )
        
        logger.info(f"✅ Dataset loaded: {len(dataset)} samples")
        
        # Extract statistics
        adapter = FrequencyDomainPrivacyAdapter()
        adapter.save_source_statistics(dataloader, str(output_path))
        
        logger.info("🎉 Statistics extraction completed successfully!")
        logger.info(f"📊 Statistics saved to: {output_path}")
        logger.info("")
        logger.info("Next steps:")
        logger.info("1. Train with: python train_madgnet_advanced_privacy.py")
        logger.info("2. Test with:  python test_advanced_privacy_methods.py")
        
    except ImportError as e:
        logger.error(f"❌ Import error: {e}")
        logger.info("Please ensure all required modules are available")
    except Exception as e:
        logger.error(f"❌ Error during extraction: {e}")
        logger.info("Please check the dataset format and try again")

if __name__ == '__main__':
    main() 