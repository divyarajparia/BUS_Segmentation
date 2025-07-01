"""
Debug Diffusion Model Locally
============================

This script will help identify exactly why the diffusion model generates black images.
Run this after you have debug_data/ with real samples and checkpoints.
"""

import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def test_real_data_normalization():
    """Test how real BUSI data should be normalized"""
    print("🔍 STEP 1: Testing Real Data Normalization")
    print("=" * 50)
    
    debug_busi_dir = "debug_data/BUSI"
    
    if not os.path.exists(debug_busi_dir):
        print("❌ Debug BUSI data not found!")
        print("Please follow DEBUG_DATASET_INSTRUCTIONS.md first")
        return None
    
    real_samples = []
    
    # Load real samples
    for class_name in ['benign', 'malignant']:
        image_dir = f"{debug_busi_dir}/{class_name}/image"
        if os.path.exists(image_dir):
            images = [f for f in os.listdir(image_dir) if f.endswith('.png')]
            for img_file in images[:2]:  # Just test 2 samples per class
                img_path = os.path.join(image_dir, img_file)
                img = Image.open(img_path).convert('L')
                img_array = np.array(img)
                
                print(f"✅ {class_name} sample: {img_file}")
                print(f"   Original range: [{img_array.min()}, {img_array.max()}]")
                print(f"   Original mean: {img_array.mean():.2f}")
                
                # Test the normalization used in training
                # From simple_diffusion_busi.py: transforms.Normalize([0.5], [0.5])
                img_tensor = torch.tensor(img_array).float() / 255.0  # [0, 1]
                img_normalized = (img_tensor - 0.5) / 0.5  # [-1, 1]
                
                print(f"   Normalized range: [{img_normalized.min():.3f}, {img_normalized.max():.3f}]")
                print(f"   Normalized mean: {img_normalized.mean():.3f}")
                
                real_samples.append({
                    'class': class_name,
                    'original': img_array,
                    'normalized': img_normalized.numpy(),
                    'filename': img_file
                })
                
                # Test reverse conversion
                reconstructed = ((img_normalized + 1) * 127.5).clamp(0, 255).byte().numpy()
                print(f"   Reconstructed range: [{reconstructed.min()}, {reconstructed.max()}]")
                
                # Save test images
                Image.fromarray(reconstructed, mode='L').save(f"debug_reconstructed_{class_name}_{img_file}")
                print(f"   Saved: debug_reconstructed_{class_name}_{img_file}")
                print()
    
    return real_samples

def test_model_architecture():
    """Test if the model architecture is working"""
    print("🔍 STEP 2: Testing Model Architecture")
    print("=" * 50)
    
    try:
        from simple_diffusion_busi import SimpleUNet, SimpleDiffusion
        print("✅ Successfully imported model classes")
        
        # Test model creation
        model = SimpleUNet()
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✅ Model created with {total_params:,} parameters")
        
        # Test forward pass with random input
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        
        # Create test input in the expected range [-1, 1]
        test_input = torch.randn(2, 1, 64, 64, device=device)  # Small size for testing
        test_t = torch.randint(0, 1000, (2,), device=device)
        test_class = torch.tensor([0, 1], device=device)  # One benign, one malignant
        
        print(f"   Test input range: [{test_input.min():.3f}, {test_input.max():.3f}]")
        
        model.eval()
        with torch.no_grad():
            output = model(test_input, test_t, test_class)
        
        print(f"✅ Forward pass successful")
        print(f"   Output shape: {output.shape}")
        print(f"   Output range: [{output.min():.6f}, {output.max():.6f}]")
        print(f"   Output mean: {output.mean():.6f}")
        print(f"   Output std: {output.std():.6f}")
        
        # Check for common issues
        if output.std() < 0.001:
            print("⚠️  WARNING: Output has very low variance - model might be collapsed!")
        
        if torch.isnan(output).any():
            print("❌ ERROR: Output contains NaN values!")
            
        if torch.isinf(output).any():
            print("❌ ERROR: Output contains infinite values!")
        
        if torch.all(output == output[0]):
            print("❌ ERROR: All outputs are identical - model is definitely collapsed!")
        
        return model, device
        
    except Exception as e:
        print(f"❌ Error testing model: {e}")
        return None, None

def test_checkpoint_loading():
    """Test loading the actual checkpoint"""
    print("\n🔍 STEP 3: Testing Checkpoint Loading")
    print("=" * 50)
    
    checkpoint_path = "debug_data/checkpoints/diffusion_model_epoch_50.pth"
    
    if not os.path.exists(checkpoint_path):
        print("⚠️ No checkpoint found in debug_data/checkpoints/")
        print("Model might not have been trained properly or checkpoint wasn't copied")
        return None
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print(f"✅ Checkpoint loaded successfully")
        print(f"   Keys: {list(checkpoint.keys())}")
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            print(f"   Model parameters: {len(state_dict)}")
            
            # Check some parameter statistics
            param_stats = []
            for name, param in list(state_dict.items())[:10]:
                stats = {
                    'name': name,
                    'shape': param.shape,
                    'mean': param.mean().item(),
                    'std': param.std().item(),
                    'min': param.min().item(),
                    'max': param.max().item()
                }
                param_stats.append(stats)
                print(f"   {name}: shape={param.shape}, mean={stats['mean']:.6f}, std={stats['std']:.6f}")
            
            # Check for common training issues
            all_means = [s['mean'] for s in param_stats]
            all_stds = [s['std'] for s in param_stats]
            
            if all(abs(m) < 1e-6 for m in all_means):
                print("⚠️  WARNING: All parameters have near-zero means - might indicate training issues")
            
            if all(s < 1e-6 for s in all_stds):
                print("❌ ERROR: All parameters have near-zero std - model weights are collapsed!")
            
            return checkpoint
        else:
            print("❌ No model_state_dict found in checkpoint")
            return None
            
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        return None

def test_generation_step_by_step(model, device, checkpoint):
    """Test the generation process step by step"""
    print("\n🔍 STEP 4: Testing Generation Step by Step")
    print("=" * 50)
    
    if model is None or checkpoint is None:
        print("❌ Cannot test generation without model and checkpoint")
        return
    
    try:
        # Load checkpoint into model
        model.load_state_dict(checkpoint['model_state_dict'])
        print("✅ Loaded checkpoint weights into model")
        
        from simple_diffusion_busi import SimpleDiffusion
        diffusion = SimpleDiffusion(device=device)
        
        # Test with very few timesteps for debugging
        original_timesteps = diffusion.num_timesteps
        diffusion.num_timesteps = 5  # Just 5 steps for debugging
        
        print(f"🎨 Testing generation with {diffusion.num_timesteps} timesteps...")
        
        # Generate one sample
        shape = (1, 1, 64, 64)  # Small size
        class_labels = torch.tensor([0], device=device)  # Benign
        
        # Manual generation with detailed logging
        x = torch.randn(shape, device=device)
        print(f"   Initial noise: range=[{x.min():.3f}, {x.max():.3f}], mean={x.mean():.3f}")
        
        model.eval()
        with torch.no_grad():
            for i in range(diffusion.num_timesteps):
                t = torch.full((shape[0],), diffusion.num_timesteps - 1 - i, device=device, dtype=torch.long)
                
                # Predict noise
                predicted_noise = model(x, t, class_labels)
                print(f"   Step {i}: predicted_noise range=[{predicted_noise.min():.3f}, {predicted_noise.max():.3f}], std={predicted_noise.std():.3f}")
                
                # Check if model is predicting reasonable noise
                if predicted_noise.std() < 0.001:
                    print(f"   ⚠️  WARNING: Predicted noise has very low variance at step {i}")
                
                if torch.all(predicted_noise == predicted_noise.flatten()[0]):
                    print(f"   ❌ ERROR: All predicted noise values are identical at step {i}")
                
                # Apply denoising step
                alpha_t = diffusion.alphas[diffusion.num_timesteps - 1 - i]
                alpha_cumprod_t = diffusion.alphas_cumprod[diffusion.num_timesteps - 1 - i]
                beta_t = diffusion.betas[diffusion.num_timesteps - 1 - i]
                
                # Simplified denoising
                pred_x0 = (x - torch.sqrt(1 - alpha_cumprod_t) * predicted_noise) / torch.sqrt(alpha_cumprod_t)
                
                if i > 0:
                    noise = torch.randn_like(x)
                    x = torch.sqrt(alpha_t) * pred_x0 + torch.sqrt(beta_t) * noise
                else:
                    x = pred_x0
                
                print(f"   Step {i}: denoised x range=[{x.min():.3f}, {x.max():.3f}], mean={x.mean():.3f}")
        
        print(f"✅ Generation completed")
        print(f"   Final result: range=[{x.min():.6f}, {x.max():.6f}], mean={x.mean():.6f}, std={x.std():.6f}")
        
        # Test different conversion methods
        sample = x[0, 0].cpu().numpy()
        
        print(f"\n🔧 Testing conversion methods:")
        
        # Method 1: Standard normalization
        try:
            converted1 = ((sample + 1) * 127.5).astype(np.uint8)
            print(f"   Standard: range=[{converted1.min()}, {converted1.max()}], unique_values={len(np.unique(converted1))}")
            Image.fromarray(converted1, mode='L').save('debug_generation_standard.png')
        except:
            print("   Standard: Failed")
        
        # Method 2: Min-max normalization
        try:
            if sample.max() != sample.min():
                converted2 = ((sample - sample.min()) / (sample.max() - sample.min()) * 255).astype(np.uint8)
                print(f"   Min-max: range=[{converted2.min()}, {converted2.max()}], unique_values={len(np.unique(converted2))}")
                Image.fromarray(converted2, mode='L').save('debug_generation_minmax.png')
            else:
                print("   Min-max: Sample is constant!")
        except:
            print("   Min-max: Failed")
        
        # Method 3: Robust normalization  
        try:
            p1, p99 = np.percentile(sample, [1, 99])
            if p99 > p1:
                sample_robust = np.clip(sample, p1, p99)
                converted3 = ((sample_robust - p1) / (p99 - p1) * 255).astype(np.uint8)
                print(f"   Robust: range=[{converted3.min()}, {converted3.max()}], unique_values={len(np.unique(converted3))}")
                Image.fromarray(converted3, mode='L').save('debug_generation_robust.png')
            else:
                print("   Robust: No variance in percentiles!")
        except:
            print("   Robust: Failed")
        
        return sample
        
    except Exception as e:
        print(f"❌ Error in generation testing: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Run comprehensive local debugging"""
    print("🏥 LOCAL DIFFUSION MODEL DEBUGGING")
    print("=" * 60)
    print("This will identify exactly why your diffusion model generates black images")
    print()
    
    # Step 1: Test real data normalization
    real_samples = test_real_data_normalization()
    
    # Step 2: Test model architecture
    model, device = test_model_architecture()
    
    # Step 3: Test checkpoint loading
    checkpoint = test_checkpoint_loading()
    
    # Step 4: Test generation step by step
    generated_sample = test_generation_step_by_step(model, device, checkpoint)
    
    print("\n🎯 DIAGNOSIS SUMMARY")
    print("=" * 60)
    
    if real_samples is None:
        print("❌ ISSUE: No debug data found - follow DEBUG_DATASET_INSTRUCTIONS.md")
    else:
        print("✅ Real data loaded and normalized correctly")
    
    if model is None:
        print("❌ ISSUE: Model architecture problems")
    else:
        print("✅ Model architecture working")
    
    if checkpoint is None:
        print("❌ ISSUE: Checkpoint loading problems or model not trained")
    else:
        print("✅ Checkpoint loaded successfully")
    
    if generated_sample is None:
        print("❌ ISSUE: Generation process failed")
    else:
        print("✅ Generation process completed")
        print("   Check debug_generation_*.png files to see results")
    
    print(f"\n💡 NEXT STEPS:")
    if checkpoint is None:
        print("1. Make sure the diffusion model was actually trained successfully")
        print("2. Check if diffusion_model_epoch_50.pth exists on the server")
        print("3. Copy the checkpoint to debug_data/checkpoints/")
    elif generated_sample is not None and np.std(generated_sample) < 0.001:
        print("1. Model appears to be collapsed - may need retraining")
        print("2. Check training logs for convergence issues")
        print("3. Consider using a different architecture or learning rate")
    else:
        print("1. Check the saved debug_generation_*.png files")
        print("2. If they look reasonable, the issue might be in the batch generation")
        print("3. Try the robust normalization method in the actual generation script")

if __name__ == "__main__":
    main() 