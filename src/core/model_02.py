"""
Simple Patch-Level Encoder (ViT-L/14)
=====================================
"""

import torch
import torch.nn as nn
import clip
import random
import numpy as np
import os
from .config import *
from transformers import AutoModel
from tqdm import tqdm
import time
import shutil

# Global model cache to avoid reloading
_MODEL_CACHE = {}

# Persistent cache directory for models (STOPS REDOWNLOADING!)
CACHE_DIR = os.path.join(os.getcwd(), ".model_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(SEED)

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
        
        # CRITICAL: Force consistent dtype on MPS to avoid mismatch errors
        if DEVICE == "mps":
            _MODEL_CACHE[cache_key] = _MODEL_CACHE[cache_key].to(dtype=MODEL_DTYPE)
            print(f"🔧 Converted CLIP model to {MODEL_DTYPE} for MPS compatibility")
        
        if FREEZE_CLIP:
            for param in _MODEL_CACHE[cache_key].parameters():
                param.requires_grad = False
        print(f"✅ CLIP model {model_name} loaded and cached persistently")
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
        
        # CRITICAL: Ensure model is actually in the right dtype on MPS
        if DEVICE == "mps":
            _MODEL_CACHE[cache_key] = _MODEL_CACHE[cache_key].to(dtype=MODEL_DTYPE)
            print(f"🔧 Converted base model to {MODEL_DTYPE} for MPS compatibility")
        
        if FREEZE_BASE_MODEL:
            for param in _MODEL_CACHE[cache_key].parameters():
                param.requires_grad = False
        print(f"✅ Base model {model_name} loaded and cached persistently")
    else:
        print(f"⚡ Using in-memory cached base model: {model_name}")
    
    return _MODEL_CACHE[cache_key]

class PatchEncoder(nn.Module):
    """
    Simple patch encoder: extracts patch features from ViT-L/14
    """
    
    def __init__(self, hidden_dim=None, output_dim=None, clip_model=None):
        super().__init__()
        
        # Use config defaults if not provided
        hidden_dim = hidden_dim or ENCODER_HIDDEN_DIM
        output_dim = output_dim or ENCODER_OUTPUT_DIM
        
        # Use shared CLIP model or get cached one
        self.clip_model = clip_model or get_clip_model()
        
        # ViT specs from config
        self.num_patches = NUM_PATCHES
        
        # CRITICAL FIX: Dynamically determine CLIP dimension from the actual model
        # This fixes shape mismatch when different CLIP models (ViT-B/16 vs ViT-L/14) are used
        if hasattr(self.clip_model.visual, 'transformer'):
            # For Vision Transformer models
            self.clip_dim = self.clip_model.visual.transformer.width
        else:
            # Fallback to config value (should not happen with current CLIP models)
            self.clip_dim = CLIP_DIM
            print(f"⚠️ Warning: Could not detect CLIP dimension, using config default: {CLIP_DIM}")
        
        print(f"🔍 Detected CLIP dimension: {self.clip_dim}")
        
        # Simple trainable projection per patch
        self.projection = nn.Sequential(
            nn.Linear(self.clip_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, images):
        """
        Args:
            images: [batch_size, 3, 224, 224]
        Returns:
            patch_features: [batch_size, num_patches, output_dim]
        """
        batch_size = images.size(0)
        
        # Extract patch features (before final pooling)
        with torch.no_grad():
            # CRITICAL: Force ALL tensors to MODEL_DTYPE for MPS consistency
            images = images.to(dtype=MODEL_DTYPE)
            
            # Run through CLIP vision transformer but stop before final pooling
            x = self.clip_model.visual.conv1(images)  # Patch embedding
            x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
            x = x + self.clip_model.visual.positional_embedding[1:]  # Skip CLS token pos
            
            # Through transformer layers
            x = self.clip_model.visual.ln_pre(x)
            x = x.permute(1, 0, 2)
            x = self.clip_model.visual.transformer(x)
            x = x.permute(1, 0, 2)
            # x is now [batch_size, num_patches, clip_dim] - all patch features
            
            # Force to MODEL_DTYPE
            x = x.to(dtype=MODEL_DTYPE)
        
        # Project each patch - force consistent dtype
        patch_features = self.projection(x.to(dtype=MODEL_DTYPE))
        
        return patch_features


class PatchDecoder(nn.Module):
    """
    Autoregressive decoder: patch features + text sequence → next token prediction
    """
    
    def __init__(self, model_name=None, clip_model=None):
        super().__init__()
        
        # Use config default if not provided
        model_name = model_name or BASE_MODEL_NAME
        
        # Use shared/cached models
        self.base_model = get_base_model(model_name)
        self.clip_model = clip_model or get_clip_model()
        
        embed_dim = self.base_model.config.hidden_size
        
        # DYNAMIC CLIP TEXT DIM: Since we're using raw token embeddings, 
        # dimension matches the CLIP model's width (not fixed 512)
        if hasattr(self.clip_model, 'transformer'):
            self.clip_text_dim = self.clip_model.transformer.width
        else:
            self.clip_text_dim = CLIP_TEXT_DIM  # Fallback
            print(f"⚠️ Warning: Could not detect CLIP text dimension, using config default: {CLIP_TEXT_DIM}")
        
        print(f"🔍 Detected CLIP text dimension: {self.clip_text_dim}")
        
        # Trainable projection layers
        self.patch_projection = nn.Linear(PATCH_FEATURE_DIM, embed_dim)
        self.text_projection = nn.Linear(self.clip_text_dim, embed_dim)
        
        # Trainable output head for next-token prediction
        self.output_head = nn.Linear(embed_dim, CLIP_VOCAB_SIZE)
        
    def forward(self, patch_features, input_tokens):
        """
        Args:
            patch_features: [batch_size, num_patches, patch_feature_dim] 
            input_tokens: [batch_size, seq_len] - input sequence (for teacher forcing)
        Returns:
            logits: [batch_size, seq_len, vocab_size] - next token predictions
        """
        batch_size, seq_len = input_tokens.shape
        
        # ROOT FIX: Just use CLIP's token embeddings, skip the transformer entirely
        # Let the base language model handle contextual processing instead
        with torch.no_grad():
            # Get raw token embeddings from CLIP (no transformer, no sequence length constraints)
            text_features = self.clip_model.token_embedding(input_tokens).to(dtype=MODEL_DTYPE)
            # Add positional embeddings for the actual sequence length
            text_features = text_features + self.clip_model.positional_embedding[:seq_len].to(dtype=MODEL_DTYPE)
            # text_features: [batch_size, seq_len, clip_text_dim]
        
        # Project to base model's embedding space - force MODEL_DTYPE everywhere
        patch_embeds = self.patch_projection(patch_features.to(dtype=MODEL_DTYPE))
        text_embeds = self.text_projection(text_features.to(dtype=MODEL_DTYPE))
        
        # Combine: num_patches patch tokens + seq_len text tokens
        combined = torch.cat([patch_embeds, text_embeds], dim=1)  # [batch_size, num_patches + seq_len, embed_dim]
        
        # Process through base model with causal attention - force MODEL_DTYPE
        with torch.no_grad():
            outputs = self.base_model(inputs_embeds=combined.to(dtype=MODEL_DTYPE))
        
        # Extract only the text token outputs (ignore patch positions)
        text_outputs = outputs.last_hidden_state[:, NUM_PATCHES:, :]  # [batch_size, seq_len, embed_dim]
        
        # Predict next tokens - force MODEL_DTYPE
        logits = self.output_head(text_outputs.to(dtype=MODEL_DTYPE))
        
        return logits
    
    def generate(self, patch_features, max_length=None, start_token=None):
        """
        Autoregressive generation for inference
        """
        # Use config defaults if not provided
        max_length = max_length or MAX_GENERATION_LENGTH
        start_token = start_token or START_TOKEN_ID
        
        self.eval()
        batch_size = patch_features.size(0)
        device = patch_features.device
        
        # Start with start token
        generated = torch.full((batch_size, 1), start_token, device=device, dtype=torch.long)
        
        with torch.no_grad():
            for _ in range(max_length - 1):
                # Get logits for current sequence (forward method handles padding internally)
                logits = self.forward(patch_features, generated)  # [batch_size, current_len, vocab_size]
                
                # Get next token (take last position)
                next_token_logits = logits[:, -1, :]  # [batch_size, vocab_size]
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)  # [batch_size, 1]
                
                # Append to sequence
                generated = torch.cat([generated, next_token], dim=1)
                
                # Stop if end token
                if (next_token == END_TOKEN_ID).all():
                    break
        
        return generated
    
    def unfreeze_base_model(self):
        """Call this later when you want to fine-tune the base model"""
        for param in self.base_model.parameters():
            param.requires_grad = True
        print("Base model unfrozen - ready for fine-tuning!")

def create_models_efficiently(encoder_hidden_dim=None, encoder_output_dim=None, base_model_name=None):
    """Create encoder and decoder with shared CLIP model for maximum efficiency"""
    print("🚀 Creating models with shared components...")
    
    # Load CLIP once and share it
    clip_model = get_clip_model()
    
    # Create models with shared CLIP
    encoder = PatchEncoder(
        hidden_dim=encoder_hidden_dim,
        output_dim=encoder_output_dim,
        clip_model=clip_model
    ).to(DEVICE)
    
    decoder = PatchDecoder(
        model_name=base_model_name,
        clip_model=clip_model
    ).to(DEVICE)
    
    # CRITICAL: Ensure all parameters are in MODEL_DTYPE for MPS consistency
    if DEVICE == "mps":
        encoder = encoder.to(dtype=MODEL_DTYPE)
        decoder = decoder.to(dtype=MODEL_DTYPE)
        print(f"🔧 Converted all model parameters to {MODEL_DTYPE} for MPS compatibility")
    
    print(f"✅ Models created efficiently with shared components (device: {DEVICE}, dtype: {MODEL_DTYPE})")
    return encoder, decoder

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