#!/usr/bin/env python3
"""
Standalone script to generate CCST-augmented data only
Use this if you want to generate data first and inspect it before training
"""

import os
import argparse
from ccst_exact_replication import run_ccst_pipeline

def main():
    parser = argparse.ArgumentParser(description='Generate CCST-augmented data for domain adaptation')
    parser.add_argument('--busi-path', type=str, 
                       default='dataset/BioMedicalDataset/BUSI',
                       help='Path to BUSI dataset')
    parser.add_argument('--bus-uclm-path', type=str, 
                       default='dataset/BioMedicalDataset/BUS-UCLM',
                       help='Path to BUS-UCLM dataset')
    parser.add_argument('--output-path', type=str, 
                       default='dataset/BioMedicalDataset/CCST-Results',
                       help='Output path for CCST results')
    parser.add_argument('--style-type', type=str, default='overall', 
                       choices=['overall', 'single'],
                       help='Style type: overall domain or single image')
    parser.add_argument('--K', type=int, default=1, 
                       help='Augmentation level (K=1 optimal for overall, K=2/3 for single)')
    parser.add_argument('--J', type=int, default=10, 
                       help='Number of single images per client (only for single style)')
    
    args = parser.parse_args()
    
    print("🚀 Generating CCST-Augmented Data")
    print("=" * 40)
    print(f"Configuration:")
    print(f"  BUSI path: {args.busi_path}")
    print(f"  BUS-UCLM path: {args.bus_uclm_path}")
    print(f"  Output path: {args.output_path}")
    print(f"  Style type: {args.style_type}")
    print(f"  Augmentation level K: {args.K}")
    if args.style_type == 'single':
        print(f"  Single images per client J: {args.J}")
    
    # Validate paths
    if not os.path.exists(args.busi_path):
        print(f"❌ BUSI dataset not found at {args.busi_path}")
        return
    
    if not os.path.exists(args.bus_uclm_path):
        print(f"❌ BUS-UCLM dataset not found at {args.bus_uclm_path}")
        return
    
    # Generate CCST data
    print(f"\n🎨 Starting CCST data generation...")
    results = run_ccst_pipeline(
        busi_path=args.busi_path,
        bus_uclm_path=args.bus_uclm_path,
        output_base_path=args.output_path,
        style_type=args.style_type,
        K=args.K,
        J=args.J
    )
    
    print(f"\n✅ CCST Data Generation Completed!")
    print(f"=" * 40)
    print(f"Generated datasets:")
    print(f"  📁 BUS-UCLM → BUSI style: {results['bus_uclm_augmented_path']}")
    print(f"     └── This is what you want for domain adaptation!")
    print(f"  📁 BUSI → BUS-UCLM style: {results['busi_augmented_path']}")
    print(f"     └── Optional, less useful for domain adaptation")
    print(f"  📁 Style bank: {results['style_bank_path']}")
    
    # Display data statistics
    print(f"\n📊 Data Statistics:")
    print(f"  BUS-UCLM augmented samples: {len(results['bus_uclm_augmented_data'])}")
    print(f"  BUSI augmented samples: {len(results['busi_augmented_data'])}")
    
    # Recommendations
    print(f"\n💡 Next Steps:")
    print(f"1. Inspect the generated data:")
    print(f"   ls -la {results['bus_uclm_augmented_path']}")
    print(f"   ls -la {results['bus_uclm_augmented_path']}/*/image/")
    
    print(f"\n2. Train with the augmented data:")
    print(f"   python train_with_ccst_data.py \\")
    print(f"       --ccst-augmented-path '{results['bus_uclm_augmented_path']}' \\")
    print(f"       --original-busi-path '{args.busi_path}' \\")
    print(f"       --num-epochs 100 --batch-size 8")
    
    print(f"\n3. Or use the complete job script:")
    print(f"   sbatch run_ccst_server.job")

if __name__ == "__main__":
    main() 