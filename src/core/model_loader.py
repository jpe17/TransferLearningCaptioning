"""
Model Loading and Caching
=========================
"""

import torch
import clip
import os
import time
import shutil
from transformers import AutoModel
from .config import *

# Global model cache to avoid reloading
_MODEL_CACHE = {}

# Persistent cache directory for models (STOPS REDOWNLOADING!)
CACHE_DIR = os.path.join(os.getcwd(), ".model_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

def get_clip_model(model_name=None):
    """Get cached CLIP model to avoid duplicate loading AND redownloading"""
    model_name = model_name or CLIP_MODEL
    cache_key = f"clip_{model_name}"
    
    if cache_key not in _MODEL_CACHE:
        print(f"🔄 Loading CLIP model: {model_name}")
        
        # Use persistent cache directory to avoid redownloading
        model_cache_dir = os.path.join(CACHE_DIR, f"clip_{model_name.replace('/', '_')}")
        
        # Load CLIP with persistent cache
        try:
            _MODEL_CACHE[cache_key], _ = clip.load(
                model_name, 
                device=DEVICE,
                download_root=CACHE_DIR  # Persistent cache - NO MORE REDOWNLOADING!
            )
        except Exception as e:
            print(f"⚠️ Error loading CLIP from cache, retrying: {e}")
            _MODEL_CACHE[cache_key], _ = clip.load(model_name, device=DEVICE)
        
        # Always convert to float16
        _MODEL_CACHE[cache_key] = _MODEL_CACHE[cache_key].to(dtype=MODEL_DTYPE)
        
        if FREEZE_CLIP:
            for param in _MODEL_CACHE[cache_key].parameters():
                param.requires_grad = False
        print(f"✅ CLIP model {model_name} loaded and converted to {MODEL_DTYPE}")
    else:
        print(f"⚡ Using in-memory cached CLIP model: {model_name}")
    
    return _MODEL_CACHE[cache_key]

def get_base_model(model_name=None):
    """Get cached base language model to avoid reloading AND redownloading"""
    model_name = model_name or BASE_MODEL_NAME
    cache_key = f"base_{model_name}"
    
    if cache_key not in _MODEL_CACHE:
        print(f"🔄 Loading base model: {model_name}")
        
        # Use persistent cache directory to avoid redownloading
        base_cache_dir = os.path.join(CACHE_DIR, "transformers")
        
        _MODEL_CACHE[cache_key] = AutoModel.from_pretrained(
            model_name, 
            trust_remote_code=True,
            torch_dtype=MODEL_DTYPE,  # Use consistent dtype from config
            cache_dir=base_cache_dir  # Persistent cache - NO MORE REDOWNLOADING!
        )
        
        # Ensure model is actually in float16
        _MODEL_CACHE[cache_key] = _MODEL_CACHE[cache_key].to(dtype=MODEL_DTYPE)
        
        if FREEZE_BASE_MODEL:
            for param in _MODEL_CACHE[cache_key].parameters():
                param.requires_grad = False
        print(f"✅ Base model {model_name} loaded and converted to {MODEL_DTYPE}")
    else:
        print(f"⚡ Using in-memory cached base model: {model_name}")
    
    return _MODEL_CACHE[cache_key]

def clear_model_cache():
    """Clear both in-memory and persistent model cache"""
    global _MODEL_CACHE
    _MODEL_CACHE.clear()
    print("🧹 In-memory model cache cleared")
    
def get_cache_info():
    """Get information about the model cache"""
    cache_size = 0
    if os.path.exists(CACHE_DIR):
        cache_size = shutil.disk_usage(CACHE_DIR).used / (1024**3)  # GB
    
    return {
        "cache_dir": CACHE_DIR,
        "cache_size_gb": cache_size,
        "in_memory_models": list(_MODEL_CACHE.keys())
    }

def preload_models_with_cache():
    """Preload all models with persistent caching for faster startup"""
    print("🚀 Preloading models with persistent cache...")
    start_time = time.time()
    
    try:
        # Preload CLIP model
        clip_model = get_clip_model()
        print(f"✅ CLIP model preloaded")
        
        # Preload base model
        base_model = get_base_model()
        print(f"✅ Base model preloaded")
        
        load_time = time.time() - start_time
        print(f"🎉 All models preloaded in {load_time:.2f}s")
        
        # Show cache info
        cache_info = get_cache_info()
        print(f"📦 Cache directory: {cache_info['cache_dir']}")
        print(f"💾 Cache size: {cache_info['cache_size_gb']:.2f} GB")
        
        return True
        
    except Exception as e:
        print(f"❌ Error preloading models: {e}")
        return False 