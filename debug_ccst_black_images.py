#!/usr/bin/env python3
"""
Debug CCST Black Images Issue
============================
Investigate tensor values and fix normalization/denormalization issues
"""

import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from ccst_exact_replication import CCSTStyleExtractor

def debug_tensor_stats(tensor, name):
    """Print detailed tensor statistics"""
    print(f"\n📊 {name}:")
    print(f"   Shape: {tensor.shape}")
    print(f"   Min: {tensor.min().item():.6f}")
    print(f"   Max: {tensor.max().item():.6f}")
    print(f"   Mean: {tensor.mean().item():.6f}")
    print(f"   Std: {tensor.std().item():.6f}")
    print(f"   Zeros: {(tensor == 0).sum().item()}")
    print(f"   Ones: {(tensor == 1).sum().item()}")

def robust_tensor_to_image(tensor, save_path, debug=True):
    """Convert tensor to image with multiple normalization methods"""
    if debug:
        debug_tensor_stats(tensor, f"Input Tensor for {save_path}")
    
    # Ensure tensor is on CPU and remove batch dimension
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    if tensor.dim() == 3 and tensor.size(0) == 1:
        tensor = tensor.squeeze(0)
    
    tensor = tensor.detach().cpu()
    
    # Convert to numpy
    img_np = tensor.numpy()
    
    if debug:
        print(f"   Numpy stats: min={img_np.min():.6f}, max={img_np.max():.6f}")
    
    # Try multiple normalization methods
    methods = [
        ("method1_clamp01", lambda x: np.clip(x, 0, 1) * 255),
        ("method2_minmax", lambda x: (x - x.min()) / (x.max() - x.min() + 1e-8) * 255),
        ("method3_tanh_denorm", lambda x: np.clip((x + 1) * 127.5, 0, 255)),
        ("method4_sigmoid_like", lambda x: 255 / (1 + np.exp(-x))),
        ("method5_raw_scale", lambda x: np.abs(x) * 255)
    ]
    
    base_name = os.path.splitext(save_path)[0]
    saved_any = False
    
    for method_name, norm_func in methods:
        try:
            normalized = norm_func(img_np)
            
            if debug:
                print(f"   {method_name}: min={normalized.min():.1f}, max={normalized.max():.1f}")
            
            # Check if this method produces valid output
            if normalized.max() > normalized.min() and not np.isnan(normalized).any():
                img_pil = Image.fromarray(normalized.astype(np.uint8), mode='L')
                method_path = f"{base_name}_{method_name}.png"
                img_pil.save(method_path)
                
                if debug:
                    print(f"   ✅ Saved: {method_path}")
                saved_any = True
        except Exception as e:
            if debug:
                print(f"   ❌ {method_name} failed: {e}")
    
    return saved_any

def debug_ccst_step_by_step():
    """Debug CCST style transfer step by step"""
    print("🔍 Debugging CCST Style Transfer Step by Step")
    print("=" * 60)
    
    # Initialize CCST
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    style_extractor = CCSTStyleExtractor(device)
    
    # Step 1: Load a test image
    test_image_path = "debug_data/BUSI/benign/image/benign (1).png"
    if not os.path.exists(test_image_path):
        print(f"❌ Test image not found: {test_image_path}")
        return
    
    print(f"📷 Loading test image: {test_image_path}")
    
    # Load and transform image
    image = Image.open(test_image_path).convert('L')
    print(f"   Original PIL image size: {image.size}")
    
    # Apply transform
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image_tensor = transform(image).unsqueeze(0).to(device)
    debug_tensor_stats(image_tensor, "Input Image Tensor")
    
    # Step 2: Extract style from BUSI
    print(f"\n🎨 Extracting BUSI style...")
    style_dict = style_extractor.extract_overall_domain_style(
        "dataset/BioMedicalDataset/BUSI", 
        "train_frame.csv", 
        J_samples=10  # Small sample for debugging
    )
    
    print(f"   Style mean shape: {style_dict['mean'].shape}")
    print(f"   Style std shape: {style_dict['std'].shape}")
    
    # Step 3: Apply style transfer with detailed debugging
    print(f"\n🔧 Applying style transfer...")
    
    model = style_extractor.style_transfer_model
    
    # Forward pass step by step
    print("   Forward pass breakdown:")
    
    # 3a. Encoder
    content_feat = model.encoder(image_tensor)
    debug_tensor_stats(content_feat, "Content Features (VGG19)")
    
    # 3b. AdaIN
    style_mean = style_dict['mean'].to(device)
    style_std = style_dict['std'].to(device)
    
    debug_tensor_stats(style_mean, "Style Mean")
    debug_tensor_stats(style_std, "Style Std")
    
    stylized_feat = model.adain(content_feat, style_mean, style_std)
    debug_tensor_stats(stylized_feat, "Stylized Features (after AdaIN)")
    
    # 3c. Decoder
    decoded_output = model.decoder(stylized_feat)
    debug_tensor_stats(decoded_output, "Decoder Output (RAW)")
    
    # 3d. RGB to Grayscale conversion
    if decoded_output.size(1) == 3:
        grayscale_output = 0.299 * decoded_output[:, 0:1] + 0.587 * decoded_output[:, 1:2] + 0.114 * decoded_output[:, 2:3]
        debug_tensor_stats(grayscale_output, "After RGB→Grayscale")
        final_output = grayscale_output
    else:
        final_output = decoded_output
    
    # 3e. Clamping
    clamped_output = torch.clamp(final_output, 0, 1)
    debug_tensor_stats(clamped_output, "After Clamp [0,1]")
    
    # Step 4: Save debug images
    debug_dir = "debug_ccst_black_images"
    os.makedirs(debug_dir, exist_ok=True)
    
    print(f"\n💾 Saving debug images to {debug_dir}/")
    
    # Save at each step
    robust_tensor_to_image(image_tensor.squeeze(0)[0:1], f"{debug_dir}/01_original_input.png")
    robust_tensor_to_image(decoded_output, f"{debug_dir}/02_decoder_raw.png")
    robust_tensor_to_image(final_output, f"{debug_dir}/03_after_grayscale.png")
    robust_tensor_to_image(clamped_output, f"{debug_dir}/04_after_clamp.png")
    
    # Step 5: Test different decoder activations
    print(f"\n🧪 Testing decoder with different output activations...")
    
    # Check if decoder has final activation
    print(f"   Decoder architecture:")
    for i, layer in enumerate(model.decoder):
        print(f"     Layer {i}: {layer}")
    
    # Try without any final activation
    with torch.no_grad():
        # Get decoder output without any activation
        raw_decoder_output = model.decoder(stylized_feat)
        debug_tensor_stats(raw_decoder_output, "Raw Decoder (no activation)")
        
        # Try Tanh normalization
        tanh_output = torch.tanh(raw_decoder_output)
        debug_tensor_stats(tanh_output, "After Tanh")
        robust_tensor_to_image(tanh_output, f"{debug_dir}/05_tanh_output.png")
        
        # Try Sigmoid normalization
        sigmoid_output = torch.sigmoid(raw_decoder_output)
        debug_tensor_stats(sigmoid_output, "After Sigmoid")
        robust_tensor_to_image(sigmoid_output, f"{debug_dir}/06_sigmoid_output.png")
    
    print(f"\n✅ Debug complete! Check {debug_dir}/ for results")

def fix_ccst_output_normalization():
    """Fix the CCST output normalization issue"""
    print("🔧 Fixing CCST Output Normalization")
    print("=" * 50)
    
    # The issue is likely in the decoder output range and conversion
    # Let's create a proper denormalization function
    
    def fixed_denormalize_tensor(tensor):
        """Fixed denormalization for CCST output"""
        
        # Remove batch dimension if present
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)
        if tensor.dim() == 3 and tensor.size(0) == 1:
            tensor = tensor.squeeze(0)
        
        tensor = tensor.detach().cpu()
        img_np = tensor.numpy()
        
        print(f"   Input range: [{img_np.min():.4f}, {img_np.max():.4f}]")
        
        # Check what range the decoder is actually outputting
        if img_np.min() >= -2.0 and img_np.max() <= 2.0:
            # Decoder might be using Tanh activation [-1, 1]
            normalized = (img_np + 1.0) / 2.0 * 255
            print(f"   Using Tanh denormalization")
        elif img_np.min() >= 0.0 and img_np.max() <= 1.0:
            # Decoder might be using Sigmoid activation [0, 1]
            normalized = img_np * 255
            print(f"   Using Sigmoid denormalization")
        else:
            # Use min-max normalization as fallback
            if img_np.max() > img_np.min():
                normalized = (img_np - img_np.min()) / (img_np.max() - img_np.min()) * 255
                print(f"   Using Min-Max denormalization")
            else:
                normalized = np.full_like(img_np, 128)
                print(f"   Using Constant (image is flat)")
        
        return np.clip(normalized, 0, 255).astype(np.uint8)
    
    return fixed_denormalize_tensor

if __name__ == "__main__":
    # Run debug
    debug_ccst_step_by_step()
    
    # Test the fix
    fix_func = fix_ccst_output_normalization()
    print(f"\n✅ Fixed denormalization function created!")
    print(f"   Use this function in your CCST pipeline to fix black images") 