#!/usr/bin/env python3
"""
EMERGENCY DIAGNOSIS - Why is the trained model producing black images?
=====================================================================
"""

import torch
import numpy as np
import os

def emergency_diagnosis():
    print("🚨 EMERGENCY DIAGNOSIS: Trained Model Analysis")
    print("=" * 60)
    
    # Check checkpoint
    checkpoint_path = "joint_diffusion_epoch_50.pth"
    if not os.path.exists(checkpoint_path):
        checkpoint_path = "joint_diffusion_final.pth"
        if not os.path.exists(checkpoint_path):
            print("❌ No checkpoint found!")
            return
    
    print(f"📂 Analyzing: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    print(f"\n📊 Training Info:")
    print(f"   Epochs: {checkpoint['epoch']}")
    print(f"   Final Loss: {checkpoint['loss']:.8f}")
    print(f"   Training Samples: {checkpoint.get('num_samples', 'Unknown')}")
    
    # Critical check: Is the loss suspiciously low?
    if checkpoint['loss'] < 0.0001:
        print(f"🚨 CRITICAL: Loss is extremely low ({checkpoint['loss']:.8f})")
        print(f"   This suggests model collapse - it learned to output zeros!")
    elif checkpoint['loss'] > 10.0:
        print(f"🚨 CRITICAL: Loss is very high ({checkpoint['loss']:.8f})")
        print(f"   This suggests training failed completely!")
    else:
        print(f"✅ Loss seems reasonable ({checkpoint['loss']:.4f})")
    
    # Check model weights
    print(f"\n🔍 Model Weight Analysis:")
    state_dict = checkpoint['model_state_dict']
    
    total_params = 0
    zero_params = 0
    
    # Check critical layers
    critical_layers = ['final.weight', 'final.bias', 'enc1.0.weight', 'dec1.4.weight']
    
    for layer_name in critical_layers:
        if layer_name in state_dict:
            tensor = state_dict[layer_name]
            mean_val = tensor.mean().item()
            std_val = tensor.std().item()
            zero_count = (tensor.abs() < 1e-8).sum().item()
            total_count = tensor.numel()
            
            print(f"   {layer_name}:")
            print(f"     Mean: {mean_val:.8f}")
            print(f"     Std:  {std_val:.8f}")
            print(f"     Zeros: {zero_count}/{total_count} ({zero_count/total_count*100:.1f}%)")
            
            if abs(mean_val) < 1e-7 and std_val < 1e-7:
                print(f"     🚨 LAYER IS ESSENTIALLY ZERO!")
            elif zero_count / total_count > 0.9:
                print(f"     🚨 LAYER IS MOSTLY ZEROS!")
            else:
                print(f"     ✅ Layer looks normal")
    
    # Overall statistics
    for key, tensor in state_dict.items():
        if 'weight' in key:
            total_params += tensor.numel()
            zero_params += (tensor.abs() < 1e-8).sum().item()
    
    print(f"\n📈 Overall Statistics:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Near-zero parameters: {zero_params:,} ({zero_params/total_params*100:.2f}%)")
    
    if zero_params / total_params > 0.5:
        print(f"🚨 CRITICAL: Over 50% of weights are near zero!")
        print(f"   The model has likely collapsed during training.")
    
    # Check if this is actually a diffusion problem vs a training problem
    print(f"\n🎯 DIAGNOSIS:")
    
    if checkpoint['loss'] < 0.001:
        print(f"❌ DIAGNOSIS: MODEL COLLAPSE")
        print(f"   The loss is too low - model learned to always output zeros")
        print(f"   SOLUTION: Retrain with different learning rate/schedule")
    
    elif zero_params / total_params > 0.5:
        print(f"❌ DIAGNOSIS: WEIGHT DEATH") 
        print(f"   Too many weights died during training")
        print(f"   SOLUTION: Retrain with lower learning rate")
    
    else:
        print(f"⚠️  DIAGNOSIS: GENERATION ISSUE")
        print(f"   Model weights look OK, but generation produces black images")
        print(f"   LIKELY CAUSE: Denormalization or diffusion sampling bug")
        print(f"   SOLUTION: Fix the generation code, not the training")
    
    # Quick recommendation
    print(f"\n💡 RECOMMENDATION:")
    if checkpoint['loss'] < 0.001 or zero_params / total_params > 0.5:
        print(f"   🔄 RETRAIN the model with:")
        print(f"      - Lower learning rate (1e-5 instead of 1e-4)")
        print(f"      - Gradient clipping (0.5 instead of 1.0)")
        print(f"      - Different loss weighting")
    else:
        print(f"   🔧 FIX the generation code:")
        print(f"      - The model training was probably successful")
        print(f"      - The issue is in how we convert model output to images")

if __name__ == "__main__":
    emergency_diagnosis() 