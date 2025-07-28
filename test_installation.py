#!/usr/bin/env python3
# Copyright (c) 2021 Brno University of Technology
# Copyright (c) 2021 Nippon Telegraph and Telephone corporation (NTT).
# All rights reserved
# By Katerina Zmolikova, August 2021.

"""
Test script to verify the TD-SpeakerBeam installation works correctly.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import torch
import numpy as np

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from models.td_speakerbeam import TimeDomainSpeakerBeam
        from models.convolutional import TDConvNet
        from models.adapt_layers import make_adapt_layer
        from utils.filterbanks import make_enc_dec
        from utils.losses import singlesrc_neg_sisdr
        from utils.metrics import get_metrics
        from datasets.librimix_informed import LibriMixInformed
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

def test_model_creation():
    """Test that the model can be created."""
    print("Testing model creation...")
    
    try:
        model = TimeDomainSpeakerBeam(
            i_adapt_layer=7,
            adapt_layer_type='mul',
            adapt_enroll_dim=128,
            n_filters=512,
            kernel_size=16,
            stride=8,
            sample_rate=8000
        )
        print("✓ Model creation successful")
        return True, model
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        return False, None

def test_forward_pass():
    """Test that the model can perform a forward pass."""
    print("Testing forward pass...")
    
    success, model = test_model_creation()
    if not success:
        return False
    
    try:
        # Create dummy input
        batch_size = 2
        seq_len = 8000  # 1 second at 8kHz
        mixture = torch.randn(batch_size, seq_len)
        enrollment = torch.randn(batch_size, seq_len)
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            output = model(mixture, enrollment)
        
        print(f"✓ Forward pass successful. Output shape: {output.shape}")
        return True
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        return False

def test_loss_computation():
    """Test loss computation."""
    print("Testing loss computation...")
    
    try:
        from utils.losses import singlesrc_neg_sisdr
        
        # Create dummy signals
        est_target = torch.randn(2, 8000)
        target = torch.randn(2, 8000)
        
        loss = singlesrc_neg_sisdr(est_target, target)
        print(f"✓ Loss computation successful. Loss shape: {loss.shape}")
        return True
    except Exception as e:
        print(f"✗ Loss computation failed: {e}")
        return False

def test_metrics():
    """Test metrics computation."""
    print("Testing metrics computation...")
    
    try:
        from utils.metrics import get_metrics
        
        # Create dummy signals
        mix = np.random.randn(8000)
        target = np.random.randn(8000)
        est_target = np.random.randn(8000)
        
        metrics = get_metrics(mix, target, est_target, sample_rate=8000)
        print(f"✓ Metrics computation successful. Metrics: {list(metrics.keys())}")
        return True
    except Exception as e:
        print(f"✗ Metrics computation failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 50)
    print("TD-SpeakerBeam Installation Test")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_model_creation,
        test_forward_pass,
        test_loss_computation,
        test_metrics
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
            results.append(False)
        print()
    
    print("=" * 50)
    print("Test Summary:")
    print(f"Passed: {sum(results)}/{len(results)}")
    
    if all(results):
        print("🎉 All tests passed! Installation is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please check the installation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())