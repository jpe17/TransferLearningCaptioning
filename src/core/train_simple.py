import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import os
from tqdm import tqdm

from .model import Encoder, Decoder
from .utils import set_seed, get_tokenizer, create_run_folder, save_checkpoint, calculate_perplexity, calculate_token_accuracy
from .model_loader import get_base_model, get_clip_model
from .data_processing import create_dataloaders
from .config import *

def validate(encoder, decoder, val_loader, tokenizer):
    """Run validation and return metrics"""
    encoder.eval()
    decoder.eval()
    
    total_val_loss = 0
    total_val_acc = 0
    total_val_perplexity = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation", leave=False):
            if not batch:
                continue

            images = batch['image'].to(DEVICE)
            input_ids = batch['input_ids'].to(DEVICE)
            
            labels = input_ids[:, 1:].contiguous()
            input_ids = input_ids[:, :-1].contiguous()
            labels[labels == tokenizer.pad_token_id] = -100
            
            patch_features = encoder(images)
            output = decoder(patch_features, input_ids, labels)
            loss = output['loss']
            
            total_val_loss += loss.item()
            total_val_acc += calculate_token_accuracy(output['logits'], labels)
            total_val_perplexity += calculate_perplexity(loss.item())
            
    encoder.train()
    decoder.train()
    
    num_batches = len(val_loader)
    return {
        "val_loss": total_val_loss / num_batches,
        "val_accuracy": total_val_acc / num_batches,
        "val_perplexity": total_val_perplexity / num_batches
    }

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
    train_loader, val_loader, _ = create_dataloaders(
        batch_size=BATCH_SIZE,
        max_samples=10000  # Limit for faster experimentation
    )
    
    # Models - reuse cached models (no reloading)
    clip_model = get_clip_model()
    base_model = get_base_model()
    qwen_dim = base_model.config.hidden_size
    
    encoder = Encoder(output_dim=qwen_dim, clip_model=clip_model).to(DEVICE)
    decoder = Decoder().to(DEVICE)
    
    # Training
    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=LEARNING_RATE)
    tokenizer = get_tokenizer()
    
    # --- Sanity Check ---
    # Verify that the tokenizer's vocabulary size matches the model's output layer size.
    # A mismatch here is a common cause of nonsensical output.
    model_vocab_size = decoder.base_model.config.vocab_size
    tokenizer_vocab_size = tokenizer.vocab_size
    
    assert model_vocab_size == tokenizer_vocab_size, \
        f"CRITICAL: Tokenizer vocab size ({tokenizer_vocab_size}) does not match model vocab size ({model_vocab_size})!"
    
    print("✅ Tokenizer and model vocabulary sizes are compatible.")
    # --- End Sanity Check ---
    
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
            
            # Mask padding tokens in labels to prevent them from contributing to the loss.
            # The loss function (CrossEntropyLoss) is configured to ignore index -100.
            tokenizer = get_tokenizer()
            labels[labels == tokenizer.pad_token_id] = -100
            
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
                train_perplexity = calculate_perplexity(loss.item())
                train_accuracy = calculate_token_accuracy(output['logits'], labels)
                
                print(f"Step {step}, Loss: {loss.item():.4f}, Acc: {train_accuracy:.4f}, PPL: {train_perplexity:.4f}")
                wandb.log({
                    "train_loss": loss.item(), 
                    "train_accuracy": train_accuracy,
                    "train_perplexity": train_perplexity,
                    "step": step
                })
                
                # Run validation
                val_metrics = validate(encoder, decoder, val_loader, tokenizer)
                print(f"Validation -> Loss: {val_metrics['val_loss']:.4f}, Acc: {val_metrics['val_accuracy']:.4f}, PPL: {val_metrics['val_perplexity']:.4f}")
                wandb.log(val_metrics)
                
                save_checkpoint(run_dir, step, encoder, decoder, optimizer, config)
    
    # Final save
    save_checkpoint(run_dir, step, encoder, decoder, optimizer, config)
    
    wandb.finish()
    print(f"✅ Training complete! Results saved in: {run_dir}")

if __name__ == '__main__':
    train() 