"""
Test script to verify training works after tokenizer fixes
"""

import os
os.environ["WANDB_MODE"] = "disabled"  # Disable wandb for testing

from src.core.train_simple import train

if __name__ == "__main__":
    print("Starting test training run...")
    train()
    print("Test training completed successfully!") 