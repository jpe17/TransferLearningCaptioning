#!/usr/bin/env python3
"""
Pre-load and cache models to speed up sweep initialization.
Run this once before starting sweeps.
"""

import torch
from ..core.model_02 import get_clip_model, get_base_model, get_cache_info, preload_models_with_cache
from ..core.config import *
from tqdm import tqdm
import time

def preload_all_models():
    """Pre-load and cache all models used in sweeps with PERSISTENT CACHING"""
    print("🚀 Pre-loading models with persistent cache (NO MORE REDOWNLOADING!)")
    print("=" * 50)
    
    start_time = time.time()
    
    # Use the new persistent caching preloader
    success = preload_models_with_cache()
    
    if success:
        # Also pre-load additional CLIP models (from sweep config)
        additional_clip_models = ['ViT-B/16']  # Add others if needed
        for model_name in additional_clip_models:
            if model_name != CLIP_MODEL:  # Skip if already loaded
                print(f"\n📥 Pre-loading additional CLIP: {model_name}")
                _ = get_clip_model(model_name)
        
        total_time = time.time() - start_time
        print("\n" + "=" * 50)
        print("✅ All models pre-loaded with persistent cache!")
        print("🚄 Your sweeps will now start INSTANTLY!")
        print(f"⏱️ Total time: {total_time:.2f}s")
        
        # Show cache information
        cache_info = get_cache_info()
        print(f"\n📦 Persistent cache info:")
        print(f"  📁 Directory: {cache_info['cache_dir']}")
        print(f"  💾 Size: {cache_info['cache_size_gb']:.2f} GB")
        print(f"  🧠 In-memory models: {len(cache_info['in_memory_models'])}")
        
        print(f"\n🎉 Cache created! Models will NOT be redownloaded on future runs.")
        
    else:
        print("❌ Failed to preload models")
        
def preload_models():
    """Alias for backward compatibility"""
    return preload_all_models()

if __name__ == "__main__":
    preload_all_models() 