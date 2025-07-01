#!/usr/bin/env python3
"""
Test script to verify the CLIP dimension fix works correctly
"""

import torch
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.core.model_02 import PatchEncoder, get_clip_model
from src.core.config import DEVICE

def test_clip_dimensions():
    """Test that PatchEncoder works with different CLIP models"""
    
    print("🧪 Testing CLIP dimension detection and compatibility...")
    print("=" * 60)
    
    # Test with both CLIP models from the sweep
    clip_models = ['ViT-B/16', 'ViT-L/14']
    
    for model_name in clip_models:
        print(f"\n🔍 Testing {model_name}:")
        print("-" * 30)
        
        try:
            # Get CLIP model
            clip_model = get_clip_model(model_name)
            
            # Create PatchEncoder with this CLIP model
            encoder = PatchEncoder(
                hidden_dim=2048,
                output_dim=512, 
                clip_model=clip_model
            ).to(DEVICE)
            
            # Test with a dummy image batch
            batch_size = 4
            dummy_images = torch.randn(batch_size, 3, 224, 224, device=DEVICE)
            
            # Forward pass
            with torch.no_grad():
                patch_features = encoder(dummy_images)
            
            print(f"✅ Success! Output shape: {patch_features.shape}")
            print(f"   Expected: [batch_size={batch_size}, num_patches=256, output_dim=512]")
            
            # Verify output shape is correct
            expected_shape = (batch_size, 256, 512)
            if patch_features.shape == expected_shape:
                print(f"✅ Shape verification passed!")
            else:
                print(f"❌ Shape mismatch! Got {patch_features.shape}, expected {expected_shape}")
                
        except Exception as e:
            print(f"❌ Error with {model_name}: {e}")
            
    print(f"\n🎉 Testing complete!")

if __name__ == "__main__":
    test_clip_dimensions() 