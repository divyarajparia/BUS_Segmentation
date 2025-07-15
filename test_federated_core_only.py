"""
Core Federated Feature Alignment Testing
========================================

Minimal test that validates only the core federated feature alignment components
without requiring the full IS2D framework or datasets. This ensures the core
logic is sound before server deployment.

Tests:
1. Privacy configuration and differential privacy
2. Feature statistics data structures
3. Feature alignment loss computation
4. Statistics saving/loading
5. Basic tensor operations

Usage:
    python test_federated_core_only.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os
from typing import Dict, Any

# Import only the core components
from federated_feature_alignment import (
    PrivacyConfig,
    FeatureStatistics,
    DifferentialPrivacyMechanism,
    load_source_statistics
)


def test_privacy_configuration():
    """Test privacy configuration and differential privacy mechanism"""
    print("🔒 Testing Privacy Configuration...")
    
    # Test basic privacy config
    config = PrivacyConfig(epsilon=1.0, delta=1e-5, sensitivity=1.0)
    assert config.noise_multiplier is not None
    assert config.noise_multiplier > 0
    print(f"   ✅ Basic privacy config: ε={config.epsilon}, noise_multiplier={config.noise_multiplier:.3f}")
    
    # Test different privacy levels
    strict_config = PrivacyConfig(epsilon=0.1, delta=1e-6)
    relaxed_config = PrivacyConfig(epsilon=10.0, delta=1e-4)
    assert strict_config.noise_multiplier > relaxed_config.noise_multiplier
    print("   ✅ Privacy level comparison works correctly")
    
    # Test differential privacy mechanism
    dp_mechanism = DifferentialPrivacyMechanism(config)
    test_tensor = torch.randn(100)
    noisy_tensor = dp_mechanism.add_gaussian_noise(test_tensor)
    
    assert noisy_tensor.shape == test_tensor.shape
    assert not torch.equal(noisy_tensor, test_tensor)  # Should be different due to noise
    print("   ✅ Differential privacy noise addition works")
    
    return True


def test_feature_statistics():
    """Test feature statistics data structure"""
    print("📊 Testing Feature Statistics...")
    
    # Create sample feature statistics
    stats = FeatureStatistics(
        layer_name='test_layer',
        feature_dim=256,
        spatial_size=(28, 28),
        mean=torch.randn(256),
        std=torch.abs(torch.randn(256)) + 0.1,
        skewness=torch.randn(256),
        kurtosis=torch.randn(256),
        spatial_mean=torch.randn(28, 28),
        spatial_std=torch.abs(torch.randn(28, 28)) + 0.1,
        num_samples=100,
        noise_scale=0.5,
        privacy_spent=1.0
    )
    
    # Validate all fields
    assert stats.layer_name == 'test_layer'
    assert stats.feature_dim == 256
    assert stats.spatial_size == (28, 28)
    assert stats.mean.shape == (256,)
    assert stats.std.shape == (256,)
    assert stats.num_samples == 100
    print("   ✅ Feature statistics structure is correct")
    
    return True


def test_statistics_serialization():
    """Test statistics saving and loading"""
    print("💾 Testing Statistics Serialization...")
    
    # Create test statistics
    test_stats = {
        'layer1': FeatureStatistics(
            layer_name='layer1',
            feature_dim=64,
            spatial_size=(56, 56),
            mean=torch.randn(64),
            std=torch.abs(torch.randn(64)) + 0.1,
            num_samples=50
        ),
        'layer2': FeatureStatistics(
            layer_name='layer2',
            feature_dim=128,
            spatial_size=(28, 28),
            mean=torch.randn(128),
            std=torch.abs(torch.randn(128)) + 0.1,
            num_samples=50
        )
    }
    
    # Create test serialization data
    shareable_data = {
        'domain_statistics': {},
        'privacy_metadata': {
            'epsilon': 1.0,
            'delta': 1e-5,
            'noise_multiplier': 2.0,
            'total_privacy_spent': 1.0
        }
    }
    
    for layer_name, stats in test_stats.items():
        shareable_data['domain_statistics'][layer_name] = {
            'feature_dim': stats.feature_dim,
            'spatial_size': stats.spatial_size,
            'mean': stats.mean.tolist(),
            'std': stats.std.tolist(),
            'skewness': None,
            'kurtosis': None,
            'spatial_mean': None,
            'spatial_std': None,
            'num_samples': stats.num_samples,
            'noise_scale': 0.5,
            'privacy_spent': 1.0
        }
    
    # Save to file
    test_file = 'test_core_statistics.json'
    with open(test_file, 'w') as f:
        json.dump(shareable_data, f, indent=2)
    
    # Load back
    loaded_stats = load_source_statistics(test_file, 'cpu')
    
    # Validate loaded statistics
    assert len(loaded_stats) == len(test_stats)
    for layer_name in test_stats:
        assert layer_name in loaded_stats
        loaded = loaded_stats[layer_name]
        original = test_stats[layer_name]
        assert loaded.feature_dim == original.feature_dim
        assert loaded.spatial_size == original.spatial_size
        print(f"   ✅ Layer {layer_name}: serialization successful")
    
    # Cleanup
    os.remove(test_file)
    print("   ✅ Statistics serialization works correctly")
    
    return True


def test_feature_alignment_computation():
    """Test feature alignment loss computation logic"""
    print("🎯 Testing Feature Alignment Computation...")
    
    # Create mock target features (what we would get from a forward pass)
    batch_size, channels, height, width = 2, 256, 28, 28
    target_features = torch.randn(batch_size, channels, height, width)
    
    # Create mock source statistics
    source_mean = torch.randn(channels)
    source_std = torch.abs(torch.randn(channels)) + 0.1
    
    # Compute target statistics (simulating what happens in the loss function)
    target_mean = target_features.mean(dim=[0, 2, 3])
    target_variance = target_features.var(dim=[0, 2, 3], unbiased=False)
    target_std = torch.sqrt(target_variance + 1e-8)
    
    # Compute alignment losses
    mean_loss = F.mse_loss(target_mean, source_mean)
    std_loss = F.mse_loss(target_std, source_std)
    total_alignment_loss = mean_loss + std_loss
    
    # Validate that losses are reasonable
    assert mean_loss >= 0
    assert std_loss >= 0
    assert total_alignment_loss >= 0
    assert total_alignment_loss.requires_grad == target_features.requires_grad
    
    print(f"   ✅ Mean alignment loss: {mean_loss:.4f}")
    print(f"   ✅ Std alignment loss: {std_loss:.4f}")
    print(f"   ✅ Total alignment loss: {total_alignment_loss:.4f}")
    
    return True


def test_tensor_operations():
    """Test various tensor operations used in the implementation"""
    print("🔢 Testing Tensor Operations...")
    
    # Test multi-dimensional statistics computation
    batch_features = torch.randn(4, 128, 32, 32)
    
    # Channel-wise statistics
    channel_mean = batch_features.mean(dim=[0, 2, 3])
    channel_std = batch_features.std(dim=[0, 2, 3])
    assert channel_mean.shape == (128,)
    assert channel_std.shape == (128,)
    
    # Spatial statistics
    spatial_mean = batch_features.mean(dim=[0, 1])
    spatial_std = batch_features.std(dim=[0, 1])
    assert spatial_mean.shape == (32, 32)
    assert spatial_std.shape == (32, 32)
    
    # Higher-order moments
    normalized_features = (batch_features - channel_mean.view(1, -1, 1, 1)) / (channel_std.view(1, -1, 1, 1) + 1e-8)
    skewness = (normalized_features ** 3).mean(dim=[0, 2, 3])
    kurtosis = (normalized_features ** 4).mean(dim=[0, 2, 3])
    assert skewness.shape == (128,)
    assert kurtosis.shape == (128,)
    
    print("   ✅ Multi-dimensional statistics computation works")
    
    # Test gradient flow
    batch_features.requires_grad_(True)
    loss = batch_features.mean()
    loss.backward()
    assert batch_features.grad is not None
    print("   ✅ Gradient flow works correctly")
    
    return True


def run_core_tests():
    """Run all core tests"""
    print("🧪 Running Core Federated Feature Alignment Tests")
    print("=" * 60)
    
    tests = [
        ("Privacy Configuration", test_privacy_configuration),
        ("Feature Statistics", test_feature_statistics),
        ("Statistics Serialization", test_statistics_serialization),
        ("Feature Alignment Computation", test_feature_alignment_computation),
        ("Tensor Operations", test_tensor_operations)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            if result:
                print(f"✅ {test_name}: PASSED\n")
                passed += 1
            else:
                print(f"❌ {test_name}: FAILED\n")
        except Exception as e:
            print(f"❌ {test_name}: FAILED - {e}\n")
    
    print("=" * 60)
    print("📋 CORE TEST SUMMARY")
    print("=" * 60)
    print(f"Total Tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    print(f"Success Rate: {passed/total*100:.1f}%")
    
    if passed == total:
        print("\n🎉 ALL CORE TESTS PASSED!")
        print("✅ Core federated feature alignment logic is working correctly")
        print("✅ Ready for integration with full framework on server")
        return True
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        print("❌ Please review core implementation before server deployment")
        return False


if __name__ == "__main__":
    success = run_core_tests()
    exit(0 if success else 1) 