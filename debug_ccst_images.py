import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import os
from torchvision import transforms
from ccst_exact_replication import CCSTStyleExtractor, ProperAdaINStyleTransfer

def debug_tensor_stats(tensor, name):
    """Print detailed tensor statistics"""
    if isinstance(tensor, torch.Tensor):
        tensor_np = tensor.detach().cpu().numpy()
    else:
        tensor_np = tensor
    
    print(f"\n📊 {name} Statistics:")
    print(f"   Shape: {tensor_np.shape}")
    print(f"   Min: {tensor_np.min():.6f}")
    print(f"   Max: {tensor_np.max():.6f}")
    print(f"   Mean: {tensor_np.mean():.6f}")
    print(f"   Std: {tensor_np.std():.6f}")
    print(f"   Unique values: {len(np.unique(tensor_np))}")
    
    # Check for extreme values
    if tensor_np.min() < -10 or tensor_np.max() > 10:
        print(f"   ⚠️  EXTREME VALUES detected!")
    
    # Check for mostly zeros/ones
    zero_ratio = (tensor_np == 0).sum() / tensor_np.size
    one_ratio = (tensor_np == 1).sum() / tensor_np.size
    print(f"   Zero ratio: {zero_ratio:.3f}")
    print(f"   One ratio: {one_ratio:.3f}")
    
    if zero_ratio > 0.8 or one_ratio > 0.8:
        print(f"   ⚠️  MOSTLY FLAT IMAGE detected!")

def debug_style_transfer_step_by_step():
    """Debug the style transfer process step by step"""
    print("🔍 DEBUGGING CCST Style Transfer Step-by-Step")
    
    # Initialize components
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    style_extractor = CCSTStyleExtractor(device=device)
    
    # Load style statistics
    style_stats_path = "ccst_style_stats/busi_domain_style.json"
    if not os.path.exists(style_stats_path):
        print(f"❌ Style stats not found: {style_stats_path}")
        return
    
    import json
    with open(style_stats_path, 'r') as f:
        style_dict = json.load(f)
    
    # Convert to tensors
    style_mean = torch.tensor(style_dict['mean']).to(device)
    style_std = torch.tensor(style_dict['std']).to(device)
    
    debug_tensor_stats(style_mean, "Style Mean")
    debug_tensor_stats(style_std, "Style Std")
    
    # Check if style stats are reasonable
    if style_mean.abs().max() > 100 or style_std.max() > 100:
        print("❌ Style statistics are EXTREME - this will cause problems!")
        print("   Style mean should be around [-1, 1]")
        print("   Style std should be around [0, 2]")
        return
    
    # Load a test image
    test_image_path = "dataset/BioMedicalDataset/CCST-Results/BUS-UCLM-CCST-PROPER/benign/image/styled_VITR_001.png"
    if not os.path.exists(test_image_path):
        # Try original BUS-UCLM image
        test_image_path = "dataset/BioMedicalDataset/BUS-UCLM/benign/images/VITR_001.png"
        if not os.path.exists(test_image_path):
            print(f"❌ Test image not found in both styled and original locations")
            return
    
    # Process test image
    image = Image.open(test_image_path).convert('L')
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    
    image_tensor = transform(image).unsqueeze(0).to(device)
    debug_tensor_stats(image_tensor, "Input Image")
    
    # Step 1: Extract content features
    model = style_extractor.style_transfer_model
    content_feat = model.encoder(image_tensor)
    debug_tensor_stats(content_feat, "Content Features")
    
    # Step 2: Apply AdaIN
    stylized_feat = model.adain(content_feat, style_mean, style_std)
    debug_tensor_stats(stylized_feat, "Stylized Features (after AdaIN)")
    
    # Step 3: Decode
    decoded_output = model.decoder(stylized_feat)
    debug_tensor_stats(decoded_output, "Decoder Output (RAW)")
    
    # Step 4: RGB to Grayscale conversion (if applicable)
    if decoded_output.size(1) == 3:
        grayscale_output = 0.299 * decoded_output[:, 0:1] + 0.587 * decoded_output[:, 1:2] + 0.114 * decoded_output[:, 2:3]
        debug_tensor_stats(grayscale_output, "After RGB→Grayscale")
        final_output = grayscale_output
    else:
        final_output = decoded_output
    
    # Step 5: First clamp (in apply_style_transfer)
    clamped_once = torch.clamp(final_output, 0, 1)
    debug_tensor_stats(clamped_once, "After First Clamp [0,1]")
    
    # Step 6: Second clamp (in dataset processing)
    clamped_twice = torch.clamp(clamped_once, 0, 1)
    debug_tensor_stats(clamped_twice, "After Second Clamp [0,1]")
    
    # Save debug images at each step
    debug_output_dir = "debug_ccst_steps"
    os.makedirs(debug_output_dir, exist_ok=True)
    
    def save_tensor_as_image(tensor, filename):
        """Save tensor as image for visual inspection"""
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)
        if tensor.dim() == 3 and tensor.size(0) == 1:
            tensor = tensor.squeeze(0)
        
        # Convert to numpy
        img_np = tensor.detach().cpu().numpy()
        
        # Multiple normalization attempts
        methods = [
            ("raw", img_np),
            ("clamp_01", np.clip(img_np, 0, 1) * 255),
            ("minmax", (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8) * 255),
            ("std_norm", np.clip((img_np + 1) * 127.5, 0, 255))
        ]
        
        for method_name, normalized in methods:
            try:
                img_pil = Image.fromarray(normalized.astype(np.uint8), mode='L')
                save_path = f"{debug_output_dir}/{filename}_{method_name}.png"
                img_pil.save(save_path)
                print(f"   Saved: {save_path}")
            except Exception as e:
                print(f"   Failed to save {method_name}: {e}")
    
    # Save original
    save_tensor_as_image(image_tensor, "01_original")
    
    # Save decoder output
    save_tensor_as_image(decoded_output, "02_decoder_raw")
    
    # Save final output
    save_tensor_as_image(final_output, "03_final_output")
    
    # Save clamped versions
    save_tensor_as_image(clamped_once, "04_clamped_once")
    save_tensor_as_image(clamped_twice, "05_clamped_twice")

def check_decoder_weights():
    """Check if decoder was properly trained"""
    print("🔍 Checking Decoder Training State")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    style_extractor = CCSTStyleExtractor(device=device)
    
    # Get decoder weights
    decoder = style_extractor.style_transfer_model.decoder
    
    total_params = sum(p.numel() for p in decoder.parameters())
    trainable_params = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    
    print(f"   Total decoder parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    # Check if weights are initialized properly (not all zeros/random)
    first_conv = None
    for name, module in decoder.named_modules():
        if isinstance(module, nn.Conv2d):
            first_conv = module
            break
    
    if first_conv is not None:
        weight_stats = {
            'min': first_conv.weight.data.min().item(),
            'max': first_conv.weight.data.max().item(),
            'mean': first_conv.weight.data.mean().item(),
            'std': first_conv.weight.data.std().item()
        }
        print(f"   First conv layer weights: {weight_stats}")
        
        if abs(weight_stats['mean']) < 1e-6 and weight_stats['std'] < 1e-6:
            print("   ⚠️  Decoder weights appear to be all zeros - NOT TRAINED!")
        elif weight_stats['std'] > 10:
            print("   ⚠️  Decoder weights are very large - possible training issue!")

if __name__ == "__main__":
    print("🚨 CCST Image Quality Debug Session")
    print("=" * 60)
    
    # Check decoder training state
    check_decoder_weights()
    
    print("\n" + "=" * 60)
    
    # Debug step by step
    debug_style_transfer_step_by_step()
    
    print("\n" + "=" * 60)
    print("🔍 Check the debug_ccst_steps/ folder for visual inspection")
    print("📊 Look at the tensor statistics above to identify the problem") 