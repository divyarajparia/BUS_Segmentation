#!/usr/bin/env python3
"""
Comprehensive Checkpoint Diagnosis
==================================
Systematically analyze all checkpoints to find the root cause of blank images.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from PIL import Image
import glob
from tqdm import tqdm

# Model classes (same as before)
class JointDiffusionUNet(nn.Module):
    def __init__(self, in_channels=2, out_channels=2, time_dim=256, num_classes=2):
        super().__init__()
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.ReLU(),
            nn.Linear(time_dim, time_dim)
        )
        
        # Class embedding
        self.class_emb = nn.Embedding(num_classes, time_dim)
        
        # Encoder
        self.enc1 = self._make_conv_block(in_channels, 64)
        self.enc2 = self._make_conv_block(64, 128)
        self.enc3 = self._make_conv_block(128, 256)
        self.enc4 = self._make_conv_block(256, 512)
        
        # Bottleneck with conditioning
        self.bottleneck = self._make_conv_block(512, 1024)
        self.cond_proj = nn.Linear(time_dim * 2, 1024)
        
        # Decoder
        self.dec4 = self._make_conv_block(1024 + 512, 512)
        self.dec3 = self._make_conv_block(512 + 256, 256)
        self.dec2 = self._make_conv_block(256 + 128, 128)
        self.dec1 = self._make_conv_block(128 + 64, 64)
        
        # Output heads
        self.final = nn.Conv2d(64, out_channels, 1)
        
    def _make_conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.ReLU()
        )
    
    def positional_encoding(self, timesteps, dim):
        """Sinusoidal positional encoding for timesteps"""
        half_dim = dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if dim % 2 == 1:  # Zero pad if odd dimension
            emb = F.pad(emb, (0, 1))
        return emb
    
    def forward(self, x, timesteps, class_labels=None):
        # Time embedding
        t_emb = self.positional_encoding(timesteps, 256)
        t_emb = self.time_mlp(t_emb)
        
        # Class embedding
        if class_labels is not None:
            c_emb = self.class_emb(class_labels)
            cond = torch.cat([t_emb, c_emb], dim=1)
        else:
            cond = t_emb
        
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        e4 = self.enc4(F.max_pool2d(e3, 2))
        
        # Bottleneck with conditioning
        b = self.bottleneck(F.max_pool2d(e4, 2))
        
        # Add conditioning
        cond_proj = self.cond_proj(cond).unsqueeze(-1).unsqueeze(-1)
        b = b + cond_proj
        
        # Decoder
        d4 = F.interpolate(b, scale_factor=2, mode='bilinear', align_corners=False)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)
        
        d3 = F.interpolate(d4, scale_factor=2, mode='bilinear', align_corners=False)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = F.interpolate(d3, scale_factor=2, mode='bilinear', align_corners=False)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = F.interpolate(d2, scale_factor=2, mode='bilinear', align_corners=False)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        return self.final(d1)


def analyze_checkpoint_weights(checkpoint_path):
    """Analyze the weights in a checkpoint for issues"""
    print(f"\n🔍 ANALYZING: {checkpoint_path}")
    print("-" * 60)
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Basic checkpoint info
        print(f"✅ Loaded successfully")
        if 'epoch' in checkpoint:
            print(f"   Epoch: {checkpoint['epoch']}")
        if 'loss' in checkpoint:
            print(f"   Loss: {checkpoint['loss']:.6f}")
        
        # Analyze model weights
        state_dict = checkpoint['model_state_dict']
        
        # Check for NaN/Inf values
        nan_params = []
        inf_params = []
        zero_params = []
        extreme_params = []
        
        total_params = 0
        
        for name, param in state_dict.items():
            total_params += param.numel()
            
            # Check for NaN
            if torch.isnan(param).any():
                nan_count = torch.isnan(param).sum().item()
                nan_params.append((name, nan_count, param.numel()))
            
            # Check for Inf
            if torch.isinf(param).any():
                inf_count = torch.isinf(param).sum().item()
                inf_params.append((name, inf_count, param.numel()))
            
            # Check for all zeros
            if torch.all(param == 0):
                zero_params.append(name)
            
            # Check for extreme values
            param_max = param.abs().max().item()
            if param_max > 100:
                extreme_params.append((name, param_max))
        
        print(f"   Total parameters: {total_params:,}")
        
        # Report issues
        if nan_params:
            print(f"   ❌ NaN VALUES FOUND in {len(nan_params)} layers:")
            for name, count, total in nan_params[:5]:  # Show first 5
                print(f"      {name}: {count}/{total} values are NaN")
            if len(nan_params) > 5:
                print(f"      ... and {len(nan_params) - 5} more layers")
        else:
            print(f"   ✅ No NaN values")
        
        if inf_params:
            print(f"   ❌ INFINITE VALUES FOUND in {len(inf_params)} layers:")
            for name, count, total in inf_params[:5]:
                print(f"      {name}: {count}/{total} values are Inf")
        else:
            print(f"   ✅ No Infinite values")
        
        if zero_params:
            print(f"   ⚠️  ZERO LAYERS: {len(zero_params)} layers are all zeros")
            for name in zero_params[:3]:
                print(f"      {name}")
        else:
            print(f"   ✅ No all-zero layers")
        
        if extreme_params:
            print(f"   ⚠️  EXTREME VALUES in {len(extreme_params)} layers:")
            for name, max_val in extreme_params[:3]:
                print(f"      {name}: max={max_val:.2e}")
        else:
            print(f"   ✅ No extreme values")
        
        # Overall health assessment
        is_healthy = len(nan_params) == 0 and len(inf_params) == 0
        print(f"   HEALTH: {'✅ HEALTHY' if is_healthy else '❌ CORRUPTED'}")
        
        return {
            'healthy': is_healthy,
            'nan_count': len(nan_params),
            'inf_count': len(inf_params),
            'zero_count': len(zero_params),
            'extreme_count': len(extreme_params),
            'checkpoint': checkpoint
        }
        
    except Exception as e:
        print(f"   ❌ ERROR loading checkpoint: {e}")
        return {'healthy': False, 'error': str(e)}


def test_model_forward_pass(model, device):
    """Test if the model can do a forward pass without NaN"""
    print(f"\n🧪 TESTING FORWARD PASS")
    print("-" * 30)
    
    try:
        model.eval()
        
        # Create test inputs
        x = torch.randn(1, 2, 64, 64, device=device)
        t = torch.tensor([500], device=device)
        class_labels = torch.tensor([0], device=device)
        
        print(f"   Input stats: min={x.min():.3f}, max={x.max():.3f}, mean={x.mean():.3f}")
        
        with torch.no_grad():
            output = model(x, t, class_labels)
            
        print(f"   Output shape: {output.shape}")
        print(f"   Output stats: min={output.min():.3f}, max={output.max():.3f}, mean={output.mean():.3f}")
        
        # Check for NaN/Inf in output
        has_nan = torch.isnan(output).any()
        has_inf = torch.isinf(output).any()
        
        if has_nan:
            print(f"   ❌ Output contains NaN values!")
            return False
        if has_inf:
            print(f"   ❌ Output contains Inf values!")
            return False
        
        print(f"   ✅ Forward pass successful!")
        return True
        
    except Exception as e:
        print(f"   ❌ Forward pass failed: {e}")
        return False


def test_simplified_generation(model, device, num_steps=10):
    """Test generation with fewer steps and debugging"""
    print(f"\n🎨 TESTING SIMPLIFIED GENERATION ({num_steps} steps)")
    print("-" * 50)
    
    try:
        model.eval()
        
        # Start with simple noise
        x = torch.randn(1, 2, 64, 64, device=device)
        class_labels = torch.tensor([0], device=device)
        
        print(f"   Initial noise: min={x.min():.3f}, max={x.max():.3f}")
        
        with torch.no_grad():
            for i in range(num_steps):
                t = torch.tensor([num_steps - i - 1], device=device)
                
                # Forward pass
                predicted_noise = model(x, t, class_labels)
                
                print(f"   Step {i}: pred_noise min={predicted_noise.min():.3f}, max={predicted_noise.max():.3f}")
                
                # Check for NaN
                if torch.isnan(predicted_noise).any():
                    print(f"   ❌ NaN detected at step {i}!")
                    return None
                
                # Simple denoising (without full DDPM math)
                x = x - 0.1 * predicted_noise
                
                print(f"   Step {i}: x min={x.min():.3f}, max={x.max():.3f}")
        
        print(f"   ✅ Generation completed!")
        return x[0, 0].cpu().numpy()  # Return first channel
        
    except Exception as e:
        print(f"   ❌ Generation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def comprehensive_diagnosis():
    """Run comprehensive diagnosis on all checkpoints"""
    
    print("🔬 COMPREHENSIVE CHECKPOINT DIAGNOSIS")
    print("=" * 80)
    
    # Find all checkpoints
    checkpoint_files = glob.glob("*.pth")
    if not checkpoint_files:
        print("❌ No .pth files found!")
        return
    
    checkpoint_files.sort()
    print(f"📂 Found {len(checkpoint_files)} checkpoint files:")
    for f in checkpoint_files:
        print(f"   {f}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📱 Using device: {device}")
    
    # Analyze each checkpoint
    results = {}
    for checkpoint_file in checkpoint_files:
        results[checkpoint_file] = analyze_checkpoint_weights(checkpoint_file)
    
    # Find healthy checkpoints
    healthy_checkpoints = [f for f, r in results.items() if r.get('healthy', False)]
    
    print(f"\n📊 SUMMARY")
    print("=" * 40)
    print(f"Total checkpoints: {len(checkpoint_files)}")
    print(f"Healthy checkpoints: {len(healthy_checkpoints)}")
    print(f"Corrupted checkpoints: {len(checkpoint_files) - len(healthy_checkpoints)}")
    
    if healthy_checkpoints:
        print(f"\n✅ HEALTHY CHECKPOINTS:")
        for f in healthy_checkpoints:
            print(f"   {f}")
        
        # Test the best healthy checkpoint
        best_checkpoint = healthy_checkpoints[0]
        print(f"\n🧪 TESTING BEST HEALTHY CHECKPOINT: {best_checkpoint}")
        
        # Load and test
        checkpoint = results[best_checkpoint]['checkpoint']
        model = JointDiffusionUNet(in_channels=2, out_channels=2).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Test forward pass
        forward_ok = test_model_forward_pass(model, device)
        
        if forward_ok:
            # Test simplified generation
            result = test_simplified_generation(model, device, num_steps=5)
            
            if result is not None:
                # Save test image
                if result.max() != result.min():
                    img_array = ((result - result.min()) / (result.max() - result.min()) * 255).astype(np.uint8)
                else:
                    img_array = np.full_like(result, 128, dtype=np.uint8)
                
                os.makedirs("diagnosis_output", exist_ok=True)
                Image.fromarray(img_array, mode='L').save("diagnosis_output/test_generation.png")
                print(f"   ✅ Test image saved: diagnosis_output/test_generation.png")
                print(f"   Image stats: min={img_array.min()}, max={img_array.max()}, unique={len(np.unique(img_array))}")
            
    else:
        print(f"\n❌ NO HEALTHY CHECKPOINTS FOUND!")
        print(f"   All checkpoints are corrupted with NaN/Inf values.")
        print(f"   This explains why you're getting blank images.")
        print(f"\n💡 RECOMMENDED ACTION:")
        print(f"   1. Check your training script for numerical instability")
        print(f"   2. Retrain with lower learning rate and gradient clipping")
        print(f"   3. Use mixed precision training carefully")
        print(f"   4. Check for division by zero in loss calculations")
    
    print(f"\n🎯 ROOT CAUSE ANALYSIS:")
    if not healthy_checkpoints:
        print(f"   The blank images are caused by NaN values in ALL model weights.")
        print(f"   This happened during training due to numerical instability.")
    else:
        print(f"   Some checkpoints are healthy. Test the best one for generation.")


if __name__ == "__main__":
    comprehensive_diagnosis() 