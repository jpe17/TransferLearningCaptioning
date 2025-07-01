#!/usr/bin/env python3
"""
Test Model Fixes
================

This script tests the fixes for:
1. Apple Silicon MPS data type mismatch 
2. Model redownloading issue

Run this to verify everything works before running sweeps.
"""

import torch
import time
import os

def test_model_loading():
    """Test that models load without errors"""
    print("🧪 Testing Model Loading Fixes")
    print("=" * 50)
    
    try:
        from src.core.model_02 import get_clip_model, get_base_model, get_cache_info
        from src.core.config import DEVICE
        
        print(f"🖥️  Device: {DEVICE}")
        
        # Test 1: Load CLIP model
        print("\n📋 Test 1: Loading CLIP model...")
        start_time = time.time()
        clip_model = get_clip_model()
        load_time = time.time() - start_time
        print(f"✅ CLIP model loaded in {load_time:.2f}s")
        print(f"📊 CLIP dtype: {clip_model.dtype}")
        
        # Test 2: Load base model  
        print("\n📋 Test 2: Loading base model...")
        start_time = time.time()
        base_model = get_base_model()
        load_time = time.time() - start_time
        print(f"✅ Base model loaded in {load_time:.2f}s")
        
        # Test 3: Test data type compatibility
        print("\n📋 Test 3: Testing data type compatibility...")
        test_image = torch.randn(1, 3, 224, 224).to(DEVICE)
        print(f"Input image dtype: {test_image.dtype}")
        
        # Convert to model dtype (this should fix MPS error)
        test_image_converted = test_image.type(clip_model.dtype)
        print(f"Converted image dtype: {test_image_converted.dtype}")
        print("✅ Data type conversion works")
        
        # Test 4: Cache info
        print("\n📋 Test 4: Checking cache info...")
        cache_info = get_cache_info()
        print(f"📁 Cache directory: {cache_info['cache_dir']}")
        print(f"💾 Cache size: {cache_info['cache_size_gb']:.2f} GB")
        print(f"🧠 Models in memory: {cache_info['in_memory_models']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_encoder_forward():
    """Test that the encoder forward pass works without MPS errors"""
    print("\n🧪 Testing Encoder Forward Pass")
    print("=" * 30)
    
    try:
        from src.core.model_02 import PatchEncoder
        from src.core.config import DEVICE
        
        print("🔧 Creating PatchEncoder...")
        encoder = PatchEncoder().to(DEVICE)
        
        print("🖼️  Creating test image...")
        test_image = torch.randn(2, 3, 224, 224).to(DEVICE)  # Batch of 2 images
        print(f"Input shape: {test_image.shape}")
        print(f"Input dtype: {test_image.dtype}")
        
        print("⚡ Running forward pass...")
        with torch.no_grad():
            output = encoder(test_image)
        
        print(f"✅ Forward pass successful!")
        print(f"Output shape: {output.shape}")
        print(f"Output dtype: {output.dtype}")
        
        return True
        
    except Exception as e:
        print(f"❌ Encoder test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_full_model_creation():
    """Test creating full models (encoder + decoder)"""
    print("\n🧪 Testing Full Model Creation")
    print("=" * 30)
    
    try:
        from src.core.model_02 import create_models_efficiently
        
        print("🏗️  Creating models efficiently...")
        start_time = time.time()
        encoder, decoder = create_models_efficiently()
        creation_time = time.time() - start_time
        
        print(f"✅ Models created successfully in {creation_time:.2f}s")
        print(f"📊 Encoder parameters: {sum(p.numel() for p in encoder.parameters() if p.requires_grad):,}")
        print(f"📊 Decoder parameters: {sum(p.numel() for p in decoder.parameters() if p.requires_grad):,}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model creation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🚀 Testing Model Fixes")
    print("=" * 60)
    
    tests = [
        ("Model Loading", test_model_loading),
        ("Encoder Forward Pass", test_encoder_forward),
        ("Full Model Creation", test_full_model_creation)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Results Summary")
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}: {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 All fixes work! You can now run sweeps without errors.")
        print("💡 Next steps:")
        print("   1. Run: python -m src.deployment.preload_models")
        print("   2. Run: python -m src.mlops.run_complete_sweep")
        print("   3. Models will NOT redownload on future runs! 🚀")
    else:
        print("\n⚠️  Some tests failed. Check the error messages above.")

if __name__ == "__main__":
    main() 