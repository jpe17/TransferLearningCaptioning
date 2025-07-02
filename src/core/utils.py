"""
Core Utilities
==============
"""

import torch
import random
import numpy as np
import math
import os
import json
from datetime import datetime
from transformers import AutoTokenizer
from .config import BASE_MODEL_NAME

_TOKENIZER = None

def get_tokenizer():
    """Get the cached global tokenizer."""
    global _TOKENIZER
    if _TOKENIZER is None:
        print(f"🔄 Initializing tokenizer for model: {BASE_MODEL_NAME}")
        _TOKENIZER = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
        
        # Ensure all special tokens are properly set
        if _TOKENIZER.pad_token is None:
            _TOKENIZER.pad_token = _TOKENIZER.eos_token
            print(f"ℹ️ Set pad_token to eos_token: {_TOKENIZER.pad_token}")
        
        # Add any missing special tokens that might be needed
        special_tokens = {
            'pad_token': _TOKENIZER.pad_token,
            'eos_token': _TOKENIZER.eos_token,
            'bos_token': _TOKENIZER.bos_token if hasattr(_TOKENIZER, 'bos_token') else None
        }
        
        # Filter out None values
        special_tokens = {k: v for k, v in special_tokens.items() if v is not None}
        
        # Print tokenizer information for debugging
        print(f"ℹ️ Tokenizer vocabulary size: {len(_TOKENIZER)}")
        print(f"ℹ️ Tokenizer special tokens: {special_tokens}")
        
    return _TOKENIZER

def set_seed(seed):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def calculate_token_accuracy(logits, targets):
    """Calculate token-level accuracy"""
    predictions = torch.argmax(logits, dim=-1)
    mask = (targets != 0)
    correct = (predictions == targets) & mask
    return correct.sum().float() / mask.sum().float()

def calculate_perplexity(loss):
    """Calculate perplexity from loss"""
    return math.exp(loss)

def create_run_folder(base_dir="runs"):
    """Create timestamped run folder"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"run_{timestamp}"
    run_dir = os.path.join(base_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

def save_checkpoint(run_dir, step, encoder, decoder, optimizer, config=None):
    """Save model checkpoint and config"""
    checkpoint = {
        'step': step,
        'encoder_state_dict': encoder.state_dict(),
        'decoder_state_dict': decoder.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    
    checkpoint_path = os.path.join(run_dir, "model_checkpoint.pth")
    torch.save(checkpoint, checkpoint_path)
    
    # Save config if provided
    if config:
        config_path = os.path.join(run_dir, "config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
    
    print(f"💾 Saved checkpoint: {checkpoint_path}") 