#!/usr/bin/env python3
"""Simple Checkpoint Diagnosis - Check for NaN values"""

import torch
import glob

print("CHECKPOINT DIAGNOSIS")
print("=" * 50)

# Find all checkpoints
checkpoint_files = glob.glob("*.pth")
checkpoint_files.sort()

print(f"Found {len(checkpoint_files)} checkpoints:")

for checkpoint_file in checkpoint_files:
    print(f"\n📂 {checkpoint_file}")
    
    try:
        # Load checkpoint
        checkpoint = torch.load(checkpoint_file, map_location='cpu')
        
        # Basic info
        if 'epoch' in checkpoint:
            print(f"   Epoch: {checkpoint['epoch']}")
        if 'loss' in checkpoint:
            print(f"   Loss: {checkpoint['loss']:.6f}")
        
        # Check weights for NaN/Inf
        state_dict = checkpoint['model_state_dict']
        
        nan_count = 0
        inf_count = 0
        total_params = 0
        
        for name, param in state_dict.items():
            total_params += param.numel()
            nan_count += torch.isnan(param).sum().item()
            inf_count += torch.isinf(param).sum().item()
        
        print(f"   Total params: {total_params:,}")
        print(f"   NaN values: {nan_count}")
        print(f"   Inf values: {inf_count}")
        
        if nan_count > 0 or inf_count > 0:
            print(f"   STATUS: ❌ CORRUPTED")
        else:
            print(f"   STATUS: ✅ HEALTHY")
            
    except Exception as e:
        print(f"   ERROR: {e}")

print(f"\n" + "=" * 50)
print("DIAGNOSIS COMPLETE") 