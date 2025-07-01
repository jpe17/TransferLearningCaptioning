#!/usr/bin/env python3
"""
Complete Sweep Runner - One script to rule them all!
Pre-loads models, creates sweep, and runs experiments.
"""

import wandb
import torch
from ..core.model_02 import get_clip_model, get_base_model, _MODEL_CACHE
from ..core.config import *
from ..core.train import train
from tqdm import tqdm
import time

def check_models_cached():
    """Check if models are already cached"""
    clip_models = ['ViT-B/16', 'ViT-L/14']
    base_model = BASE_MODEL_NAME
    
    cached_clips = [f"clip_{model}" in _MODEL_CACHE for model in clip_models]
    cached_base = f"base_{base_model}" in _MODEL_CACHE
    
    return all(cached_clips) and cached_base

def preload_models_with_progress():
    """Pre-load models with progress indication"""
    print("🚀 Pre-loading models for faster sweep execution...")
    print("=" * 60)
    
    # Check what's already cached
    if check_models_cached():
        print("⚡ All models already cached! Skipping download.")
        return
    
    total_models = 3  # 2 CLIP + 1 base model
    completed = 0
    
    overall_progress = tqdm(total=total_models, desc="📦 Overall Progress", 
                          bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} models")
    
    # Pre-load CLIP models
    clip_models = ['ViT-B/16', 'ViT-L/14']
    for model_name in clip_models:
        print(f"\n📥 Loading CLIP: {model_name}")
        _ = get_clip_model(model_name)
        completed += 1
        overall_progress.update(1)
    
    # Pre-load base model
    print(f"\n📥 Loading base model: {BASE_MODEL_NAME}")
    _ = get_base_model(BASE_MODEL_NAME)
    completed += 1
    overall_progress.update(1)
    
    overall_progress.close()
    print("\n" + "=" * 60)
    print("✅ All models loaded and cached!")
    print("🚄 Sweep runs will now start instantly!")

def create_focused_sweep():
    """Create focused sweep configuration"""
    sweep_config = {
        'method': 'bayes',
        'metric': {
            'name': 'avg_train_loss',
            'goal': 'minimize'
        },
        'parameters': {
            'learning_rate': {
                'distribution': 'log_uniform_values',
                'min': 0.00005,
                'max': 0.0005      
            },
            'batch_size': {
                'values': [16, 32, 48]
            },
            'encoder_hidden_dim': {
                'values': [1024, 2048, 3072]
            },
            'encoder_output_dim': {
                'values': [256, 512, 768]
            },
            'freeze_base_model': {
                'values': [True, False]
            },
            'grad_clip_norm': {
                'values': [0.5, 1.0, 2.0]
            },
            'warmup_steps_ratio': {
                'values': [0.05, 0.1, 0.15]
            },
            'label_smoothing': {
                'values': [0.0, 0.1, 0.2]
            },
            'clip_model': {
                'values': ['ViT-B/16', 'ViT-L/14']
            },
            'epochs': {
                'values': [3, 5, 8]
            }
        },
        'early_terminate': {
            'type': 'hyperband',
            'min_iter': 1
        }
    }
    
    sweep_id = wandb.sweep(sweep_config, project="transformer-lenses-dev")
    return sweep_id

def create_quick_test():
    """Create quick test sweep (only 8 experiments)"""
    quick_config = {
        'method': 'grid',
        'metric': {
            'name': 'avg_train_loss',
            'goal': 'minimize'
        },
        'parameters': {
            'learning_rate': {
                'values': [0.0001, 0.0003]
            },
            'batch_size': {
                'values': [32]
            },
            'encoder_hidden_dim': {
                'values': [2048]
            },
            'encoder_output_dim': {
                'values': [512]
            },
            'freeze_base_model': {
                'values': [True, False]
            },
            'clip_model': {
                'values': ['ViT-B/16', 'ViT-L/14']
            }
        }
    }
    
    sweep_id = wandb.sweep(quick_config, project="transformer-lenses-dev-quick")
    return sweep_id

def main():
    """Main function - complete sweep workflow"""
    print("🚀 Transformer Lenses - Complete Sweep Runner")
    print("=" * 60)
    print("This script will:")
    print("  1. 📦 Pre-load and cache models (if needed)")
    print("  2. 🎯 Create wandb sweep configuration") 
    print("  3. 🤖 Run sweep experiments")
    print("=" * 60)
    
    # Step 1: Pre-load models
    print("\n🔥 STEP 1: Model Pre-loading")
    preload_models_with_progress()
    
    # Step 2: Choose sweep type
    print("\n🎯 STEP 2: Sweep Configuration")
    print("Choose sweep type:")
    print("  1. Focused sweep (10 parameters, ~50-100 experiments)")
    print("  2. Quick test (6 parameters, 8 experiments)")
    
    while True:
        choice = input("\nEnter choice (1 or 2): ").strip()
        if choice in ['1', '2']:
            break
        print("Please enter '1' or '2'")
    
    # Create sweep
    if choice == '1':
        print("\n📊 Creating focused sweep...")
        sweep_id = create_focused_sweep()
        experiment_count = "~50-100"
    else:
        print("\n⚡ Creating quick test sweep...")
        sweep_id = create_quick_test()
        experiment_count = "8"
    
    print(f"\n✅ Sweep created successfully!")
    print(f"🆔 Sweep ID: {sweep_id}")
    print(f"📈 Dashboard: https://wandb.ai/transformer-lenses-dev/sweeps/{sweep_id}")
    print(f"🧪 Expected experiments: {experiment_count}")
    
    # Step 3: Run sweep
    print(f"\n🤖 STEP 3: Running Experiments")
    while True:
        response = input("Start running sweep experiments now? (y/n): ").lower().strip()
        if response in ['y', 'yes']:
            print("\n🚀 Starting sweep agent...")
            print("📊 Watch progress in WandB dashboard (link above)")
            print("⏹️  Press Ctrl+C to stop at any time")
            print("-" * 60)
            
            try:
                max_runs = 8 if choice == '2' else 50
                wandb.agent(sweep_id, train, count=max_runs)
                print(f"\n🎉 Sweep completed! Check results in WandB dashboard.")
            except KeyboardInterrupt:
                print(f"\n⏹️  Sweep stopped by user")
                print(f"📋 To resume later: wandb agent {sweep_id}")
            break
        elif response in ['n', 'no']:
            print(f"\n📋 To run the sweep later:")
            print(f"   wandb agent {sweep_id}")
            print(f"\n🔗 Dashboard: https://wandb.ai/transformer-lenses-dev/sweeps/{sweep_id}")
            break
        else:
            print("Please enter 'y' or 'n'")
    
    print(f"\n✨ All done! Happy experimenting! ✨")

if __name__ == "__main__":
    main() 