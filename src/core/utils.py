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
        _TOKENIZER = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
        if _TOKENIZER.pad_token is None:
            _TOKENIZER.pad_token = _TOKENIZER.eos_token
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

def calculate_bleu_score(pred_tokens, target_tokens):
    """Simple BLEU-1 score"""
    try:
        tokenizer = get_tokenizer()
        pred_text = tokenizer.decode(pred_tokens.cpu(), skip_special_tokens=True)
        target_text = tokenizer.decode(target_tokens.cpu(), skip_special_tokens=True)
        
        pred_set = set(pred_text.split())
        target_set = set(target_text.split())
        
        if not pred_set:
            return 0.0
        
        matches = len(pred_set & target_set)
        return matches / len(pred_set)
    except:
        return 0.0

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