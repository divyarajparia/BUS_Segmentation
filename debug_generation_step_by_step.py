#!/usr/bin/env python3
"""
Step-by-Step Generation Debugging
=================================
Since checkpoints are healthy, debug the generation process itself.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Simplified model for testing
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
        print(f"      Model input: x range=[{x.min():.3f}, {x.max():.3f}], t={timesteps.item()}")
        
        # Time embedding
        t_emb = self.positional_encoding(timesteps, 256)
        print(f"      Time emb: range=[{t_emb.min():.3f}, {t_emb.max():.3f}]")
        
        t_emb = self.time_mlp(t_emb)
        print(f"      Time MLP: range=[{t_emb.min():.3f}, {t_emb.max():.3f}]")
        
        # Class embedding
        if class_labels is not None:
            c_emb = self.class_emb(class_labels)
            print(f"      Class emb: range=[{c_emb.min():.3f}, {c_emb.max():.3f}]")
            cond = torch.cat([t_emb, c_emb], dim=1)
        else:
            cond = t_emb
        
        print(f"      Cond: range=[{cond.min():.3f}, {cond.max():.3f}]")
        
        # Encoder
        e1 = self.enc1(x)
        print(f"      E1: range=[{e1.min():.3f}, {e1.max():.3f}], NaN={torch.isnan(e1).any()}")
        
        e2 = self.enc2(F.max_pool2d(e1, 2))
        print(f"      E2: range=[{e2.min():.3f}, {e2.max():.3f}], NaN={torch.isnan(e2).any()}")
        
        e3 = self.enc3(F.max_pool2d(e2, 2))
        print(f"      E3: range=[{e3.min():.3f}, {e3.max():.3f}], NaN={torch.isnan(e3).any()}")
        
        e4 = self.enc4(F.max_pool2d(e3, 2))
        print(f"      E4: range=[{e4.min():.3f}, {e4.max():.3f}], NaN={torch.isnan(e4).any()}")
        
        # Bottleneck with conditioning
        b = self.bottleneck(F.max_pool2d(e4, 2))
        print(f"      Bottleneck: range=[{b.min():.3f}, {b.max():.3f}], NaN={torch.isnan(b).any()}")
        
        # Add conditioning
        cond_proj = self.cond_proj(cond).unsqueeze(-1).unsqueeze(-1)
        print(f"      Cond proj: range=[{cond_proj.min():.3f}, {cond_proj.max():.3f}]")
        
        b = b + cond_proj
        print(f"      B + cond: range=[{b.min():.3f}, {b.max():.3f}], NaN={torch.isnan(b).any()}")
        
        # Decoder
        d4 = F.interpolate(b, scale_factor=2, mode='bilinear', align_corners=False)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)
        print(f"      D4: range=[{d4.min():.3f}, {d4.max():.3f}], NaN={torch.isnan(d4).any()}")
        
        d3 = F.interpolate(d4, scale_factor=2, mode='bilinear', align_corners=False)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        print(f"      D3: range=[{d3.min():.3f}, {d3.max():.3f}], NaN={torch.isnan(d3).any()}")
        
        d2 = F.interpolate(d3, scale_factor=2, mode='bilinear', align_corners=False)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        print(f"      D2: range=[{d2.min():.3f}, {d2.max():.3f}], NaN={torch.isnan(d2).any()}")
        
        d1 = F.interpolate(d2, scale_factor=2, mode='bilinear', align_corners=False)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        print(f"      D1: range=[{d1.min():.3f}, {d1.max():.3f}], NaN={torch.isnan(d1).any()}")
        
        output = self.final(d1)
        print(f"      Output: range=[{output.min():.3f}, {output.max():.3f}], NaN={torch.isnan(output).any()}")
        
        return output


def debug_generation_process():
    """Debug the generation process step by step"""
    
    print("🔬 DEBUGGING GENERATION PROCESS")
    print("=" * 60)
    
    device = torch.device('cpu')  # Use CPU for easier debugging
    
    # Load a healthy checkpoint
    checkpoint_path = "joint_diffusion_epoch_30.pth"
    print(f"📂 Loading: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model
    model = JointDiffusionUNet(in_channels=2, out_channels=2).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✅ Model loaded successfully")
    
    # Test a single forward pass first
    print(f"\n🧪 TESTING SINGLE FORWARD PASS")
    print("-" * 40)
    
    x = torch.randn(1, 2, 64, 64, device=device)
    t = torch.tensor([500], device=device)
    class_labels = torch.tensor([0], device=device)
    
    print(f"   Input: range=[{x.min():.3f}, {x.max():.3f}]")
    
    with torch.no_grad():
        output = model(x, t, class_labels)
    
    print(f"   Final output: range=[{output.min():.3f}, {output.max():.3f}]")
    print(f"   Contains NaN: {torch.isnan(output).any()}")
    
    if torch.isnan(output).any():
        print(f"   ❌ NaN detected in single forward pass!")
        return
    
    print(f"   ✅ Single forward pass OK")
    
    # Now test the diffusion sampling process
    print(f"\n🎨 TESTING DIFFUSION SAMPLING")
    print("-" * 40)
    
    # Simplified sampling (just 3 steps for debugging)
    num_timesteps = 1000
    betas = torch.linspace(0.0001, 0.02, num_timesteps).to(device)
    alphas = (1.0 - betas).to(device)
    alphas_cumprod = torch.cumprod(alphas, dim=0).to(device)
    
    x = torch.randn(1, 2, 64, 64, device=device)
    print(f"   Initial noise: range=[{x.min():.3f}, {x.max():.3f}]")
    
    # Test just a few steps
    test_steps = [999, 500, 100, 10, 0]
    
    for step_idx, i in enumerate(test_steps):
        print(f"\n   --- Step {step_idx + 1}: timestep {i} ---")
        
        t = torch.full((1,), i, device=device, dtype=torch.long)
        
        # Forward pass
        predicted_noise = model(x, t, class_labels)
        
        if torch.isnan(predicted_noise).any():
            print(f"   ❌ NaN detected in predicted noise at timestep {i}!")
            return
        
        # DDPM sampling step
        alpha_t = alphas[i]
        alpha_cumprod_t = alphas_cumprod[i]
        beta_t = betas[i]
        
        print(f"   Coefficients: alpha_t={alpha_t:.6f}, alpha_cumprod_t={alpha_cumprod_t:.6f}, beta_t={beta_t:.6f}")
        
        # Compute x_{t-1}
        sqrt_alpha_cumprod_t = torch.sqrt(alpha_cumprod_t)
        sqrt_one_minus_alpha_cumprod_t = torch.sqrt(1 - alpha_cumprod_t)
        
        print(f"   sqrt coeffs: sqrt_alpha={sqrt_alpha_cumprod_t:.6f}, sqrt_one_minus={sqrt_one_minus_alpha_cumprod_t:.6f}")
        
        # Check for potential division by zero
        if sqrt_alpha_cumprod_t == 0:
            print(f"   ❌ Division by zero detected! sqrt_alpha_cumprod_t = 0")
            return
        
        pred_x0 = (x - sqrt_one_minus_alpha_cumprod_t * predicted_noise) / sqrt_alpha_cumprod_t
        print(f"   pred_x0: range=[{pred_x0.min():.3f}, {pred_x0.max():.3f}], NaN={torch.isnan(pred_x0).any()}")
        
        if torch.isnan(pred_x0).any():
            print(f"   ❌ NaN detected in pred_x0 calculation!")
            return
        
        if i > 0:
            noise = torch.randn_like(x)
            x = torch.sqrt(alpha_t) * pred_x0 + torch.sqrt(beta_t) * noise
        else:
            x = pred_x0
        
        print(f"   x after step: range=[{x.min():.3f}, {x.max():.3f}], NaN={torch.isnan(x).any()}")
        
        if torch.isnan(x).any():
            print(f"   ❌ NaN detected in x after step!")
            return
    
    print(f"\n✅ Generation completed successfully!")
    print(f"   Final result: range=[{x.min():.3f}, {x.max():.3f}]")
    
    # Test image conversion
    sample = x[0, 0].cpu().numpy()
    print(f"   Sample stats: min={sample.min():.6f}, max={sample.max():.6f}, mean={sample.mean():.6f}")
    
    if np.isnan(sample).any():
        print(f"   ❌ NaN in final numpy array!")
    else:
        print(f"   ✅ Final result is clean!")


if __name__ == "__main__":
    debug_generation_process() 