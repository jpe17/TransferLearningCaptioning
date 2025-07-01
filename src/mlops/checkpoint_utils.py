import os
import json
import torch
from datetime import datetime
from pathlib import Path
import glob
from ..core.config import CHECKPOINTS_BASE_DIR


def get_timestamp():
    """Get current timestamp in a filesystem-friendly format."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def create_run_directory(is_sweep=False):
    """Create a timestamped directory for the current run."""
    timestamp = get_timestamp()
    
    if is_sweep:
        run_dir = Path(CHECKPOINTS_BASE_DIR) / "sweeps" / f"sweep_{timestamp}"
    else:
        run_dir = Path(CHECKPOINTS_BASE_DIR) / "training" / f"run_{timestamp}"
    
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_config(run_dir, config_dict):
    """Save the full configuration used for this run."""
    config_path = run_dir / "config.json"
    
    # Convert any non-serializable values to strings
    serializable_config = {}
    for key, value in config_dict.items():
        if torch.is_tensor(value):
            serializable_config[key] = str(value)
        elif hasattr(value, '__dict__'):
            serializable_config[key] = str(value)
        else:
            serializable_config[key] = value
    
    with open(config_path, 'w') as f:
        json.dump(serializable_config, f, indent=2)
    
    print(f"Configuration saved to {config_path}")
    return config_path


def save_checkpoint(run_dir, epoch, encoder, decoder, optimizer, train_loss, val_loss=None, step_count=None):
    """Save model checkpoint in the run directory."""
    checkpoint_path = run_dir / "model_checkpoint.pth"
    
    checkpoint_data = {
        'epoch': epoch,
        'encoder_state_dict': encoder.state_dict(),
        'decoder_state_dict': decoder.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'timestamp': get_timestamp()
    }
    
    if val_loss is not None:
        checkpoint_data['val_loss'] = val_loss
    
    if step_count is not None:
        checkpoint_data['step_count'] = step_count
    
    torch.save(checkpoint_data, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")
    return checkpoint_path


def find_latest_checkpoint(is_sweep=False):
    """Find the most recent checkpoint directory."""
    if is_sweep:
        pattern = os.path.join(CHECKPOINTS_BASE_DIR, "sweeps", "sweep_*")
    else:
        pattern = os.path.join(CHECKPOINTS_BASE_DIR, "training", "run_*")
    
    checkpoint_dirs = glob.glob(pattern)
    if not checkpoint_dirs:
        return None
    
    # Sort by timestamp (newest first)
    checkpoint_dirs.sort(reverse=True)
    latest_dir = Path(checkpoint_dirs[0])
    checkpoint_path = latest_dir / "model_checkpoint.pth"
    
    if checkpoint_path.exists():
        return latest_dir, checkpoint_path
    
    return None


def load_checkpoint(checkpoint_path):
    """Load checkpoint from path."""
    if not os.path.exists(checkpoint_path):
        return None
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print(f"Checkpoint loaded from {checkpoint_path}")
        return checkpoint
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return None


def get_all_config_params():
    """Get all configuration parameters from config.py."""
    from ..core import config
    
    config_params = {}
    for attr_name in dir(config):
        if not attr_name.startswith('_') and attr_name.isupper():
            config_params[attr_name.lower()] = getattr(config, attr_name)
    
    return config_params 