import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import os
from tqdm import tqdm

from .model import Encoder, Decoder
from .utils import set_seed, get_tokenizer, create_run_folder, save_checkpoint
from .model_loader import get_base_model, get_clip_model
from .data_processing import create_dataloaders
from .config import *

def train():
    set_seed(SEED)
    
    # Create run folder
    run_dir = create_run_folder()
    print(f"📁 Run directory: {run_dir}")
    
    # Config for saving
    config = {
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'epochs': EPOCHS,
        'seed': SEED,
        'device': str(DEVICE),
        'base_model': BASE_MODEL_NAME,
        'clip_model': CLIP_MODEL
    }
    
    # Data - reuse existing efficient dataloaders (with caching, workers, etc.)
    train_loader, _, _ = create_dataloaders(
        batch_size=BATCH_SIZE,
        max_samples=10000  # Limit for faster experimentation
    )
    
    # Models - reuse cached models (no reloading)
    clip_model = get_clip_model()
    base_model = get_base_model()
    qwen_dim = base_model.config.hidden_size
    
    encoder = Encoder(output_dim=qwen_dim, clip_model=clip_model).to(DEVICE)
    decoder = Decoder(encoder_output_dim=qwen_dim).to(DEVICE)
    
    # Training
    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=LEARNING_RATE)
    
    wandb.init(project="simple-captioning", name=os.path.basename(run_dir))
    
    step = 0
    for epoch in range(EPOCHS):
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            if not batch:  # Skip empty batches
                continue
                
            images = batch['image'].to(DEVICE)
            input_ids = batch['input_ids'].to(DEVICE)
            
            # Teacher forcing: shift tokens
            labels = input_ids[:, 1:].contiguous()
            input_ids = input_ids[:, :-1].contiguous()
            
            optimizer.zero_grad()
            
            # Core training: images -> encoder -> decoder -> loss
            patch_features = encoder(images)
            output = decoder(patch_features, input_ids, labels)
            loss = output['loss']
            
            loss.backward()
            optimizer.step()
            
            step += 1
            
            # Log and save every 50 steps
            if step % 50 == 0:
                print(f"Step {step}, Loss: {loss.item():.4f}")
                wandb.log({"loss": loss.item(), "step": step})
                save_checkpoint(run_dir, step, encoder, decoder, optimizer, config)
    
    # Final save
    save_checkpoint(run_dir, step, encoder, decoder, optimizer, config)
    
    wandb.finish()
    print(f"✅ Training complete! Results saved in: {run_dir}")

if __name__ == '__main__':
    train() 