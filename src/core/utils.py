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
from .config import BASE_MODEL_NAME
from .model_loader import get_model_tokenizer_pair

_TOKENIZER = None

def get_tokenizer():
    """Get the tokenizer directly from the model-tokenizer pair."""
    global _TOKENIZER
    if _TOKENIZER is None:
        print(f"🔄 Loading tokenizer from model-tokenizer pair: {BASE_MODEL_NAME}")
        # Get tokenizer from the paired loading function
        _, _TOKENIZER = get_model_tokenizer_pair(BASE_MODEL_NAME)
        
        # Print tokenizer information for debugging
        print(f"ℹ️ Tokenizer loaded directly from model")
        
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