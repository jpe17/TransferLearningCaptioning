import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from .data_processing import create_dataloaders
from .model_02 import PatchEncoder, PatchDecoder, set_seed, create_models_efficiently
from .config import *
from ..mlops.checkpoint_utils import (
    create_run_directory, save_config, save_checkpoint, 
    find_latest_checkpoint, load_checkpoint, get_all_config_params
)
import os
import wandb
from tqdm import tqdm
import clip
import math
from collections import Counter

def calculate_bleu_score(predicted_tokens, target_tokens, tokenizer_decode_fn):
    """Calculate BLEU-4 score between predicted and target captions"""
    try:
        # Convert token tensors to text
        pred_text = tokenizer_decode_fn(predicted_tokens)
        target_text = tokenizer_decode_fn(target_tokens)
        
        # Simple BLEU-4 calculation (n-gram overlap)
        def get_ngrams(tokens, n):
            return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
        
        pred_tokens = pred_text.split()
        target_tokens = target_text.split()
        
        if not pred_tokens or not target_tokens:
            return 0.0
        
        # Calculate precision for n-grams 1 to 4
        bleu_scores = []
        for n in range(1, 5):
            pred_ngrams = Counter(get_ngrams(pred_tokens, n))
            target_ngrams = Counter(get_ngrams(target_tokens, n))
            
            if not pred_ngrams:
                bleu_scores.append(0.0)
                continue
                
            matches = sum((pred_ngrams & target_ngrams).values())
            total = sum(pred_ngrams.values())
            precision = matches / total if total > 0 else 0.0
            bleu_scores.append(precision)
        
        # Geometric mean of precisions
        if all(score > 0 for score in bleu_scores):
            bleu = math.exp(sum(math.log(score) for score in bleu_scores) / 4)
        else:
            bleu = 0.0
            
        return bleu
    except:
        return 0.0

def calculate_token_accuracy(logits, targets):
    """Calculate token-level accuracy"""
    predictions = torch.argmax(logits, dim=-1)
    # Mask out padding tokens (assuming 0 is padding)
    mask = (targets != 0)
    correct = (predictions == targets) & mask
    accuracy = correct.sum().float() / mask.sum().float()
    return accuracy.item()

def calculate_perplexity(loss):
    """Calculate perplexity from cross-entropy loss"""
    return math.exp(loss)

def decode_tokens(tokens, clip_model):
    """Simple token decoder using CLIP tokenizer"""
    try:
        # Remove special tokens and convert to text
        text_tokens = tokens[tokens != 0]  # Remove padding
        text_tokens = text_tokens[text_tokens != 49406]  # Remove start token
        text_tokens = text_tokens[text_tokens != 49407]  # Remove end token
        
        # This is a simplified decoder - in practice you'd want the full CLIP decode
        return " ".join([f"token_{t.item()}" for t in text_tokens[:10]])  # Simplified
    except:
        return "decode_error"

def train():
    """Main training loop for the Image Captioning Model."""
    
    # 1. Setup
    set_seed(SEED)
    
    # Get all config parameters
    base_config = get_all_config_params()
    
    # Initialize wandb - if run from sweep, config will be overridden
    wandb.init(project="transformer-lenses-dev", config=base_config)
    
    # Use wandb config (allows sweep to override)
    config = {}
    for key, default_value in base_config.items():
        config[key] = wandb.config.get(key, default_value)
    
    # Determine if this is a sweep run
    is_sweep = wandb.run.sweep_id is not None
    
    # Create timestamped run directory
    run_dir = create_run_directory(is_sweep=is_sweep)
    print(f"📁 Created run directory: {run_dir}")
    
    # Save configuration for this run
    save_config(run_dir, config)
    
    # Force limited steps for sweeps
    if is_sweep:
        config['epochs'] = 1
        print(f"🔄 Sweep detected - limiting to {config['max_steps_per_sweep']} steps per run")
    
    # 2. Dataloaders  
    print("📊 Creating dataloaders...")
    train_loader, val_loader, _ = create_dataloaders(
        images_dir=config.get('images_dir'),
        captions_file=config.get('captions_file'), 
        batch_size=config.get('batch_size'),
        val_split=config.get('val_split'),
        test_split=config.get('test_split'),
        seed=config.get('seed')
    )
    print(f"✅ Data loaded: {len(train_loader)} train batches, {len(val_loader)} val batches")

    # 3. Model, Optimizer, Loss
    print("Initializing model...")
    
    # Create encoder and decoder efficiently (shared CLIP model)
    encoder, decoder = create_models_efficiently(
        encoder_hidden_dim=config.get('encoder_hidden_dim'),
        encoder_output_dim=config.get('encoder_output_dim'),
        base_model_name=config.get('base_model_name')
    )
    
    # Combine all trainable parameters
    trainable_params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = optim.Adam(trainable_params, lr=config['learning_rate'])
    
    # Learning rate scheduling: warmup + cosine decay
    total_steps = len(train_loader) * config['epochs']
    warmup_steps = int(total_steps * config['warmup_steps_ratio'])
    
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_steps)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps)
    scheduler = SequentialLR(optimizer, [warmup_scheduler, cosine_scheduler], milestones=[warmup_steps])
    
    loss_fn = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=config['label_smoothing'])
    start_epoch = 0

    # Handle checkpoint resuming
    if config['resume_training'] and not is_sweep:
        latest_checkpoint_info = None
        
        if config['resume_from_latest']:
            # Try to find the latest checkpoint
            latest_checkpoint_info = find_latest_checkpoint(is_sweep=False)
        
        if latest_checkpoint_info:
            latest_dir, checkpoint_path = latest_checkpoint_info
            print(f"📂 Found latest checkpoint in {latest_dir}")
            checkpoint = load_checkpoint(checkpoint_path)
            
            if checkpoint:
                encoder.load_state_dict(checkpoint['encoder_state_dict'])
                decoder.load_state_dict(checkpoint['decoder_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_epoch = checkpoint['epoch'] + 1
                
                # Set new learning rate
                for param_group in optimizer.param_groups:
                    param_group['lr'] = config['learning_rate']
                
                print(f"🔄 Resuming from epoch {start_epoch}")
                print(f"📈 Learning rate set to {config['learning_rate']}")
        else:
            print("🆕 No checkpoint found, starting fresh")
    elif is_sweep:
        print("🔄 Sweep mode: Starting fresh (not resuming from checkpoint)")

    print("Model initialized.")
    wandb.watch(encoder, log="all")
    wandb.watch(decoder, log="all")

    # 4. Training & Validation Loop
    print("Starting training...")
    
    for epoch in range(start_epoch, config['epochs']):
        # --- Training Phase ---
        encoder.train()
        decoder.train()
        total_train_loss = 0
        step_count = 0
        
        # Adjust progress bar description based on sweep vs normal training
        if is_sweep:
            train_progress_bar = tqdm(train_loader, desc=f"Steps (max {config['max_steps_per_sweep']}) [Train]")
        else:
            train_progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']} [Train]")
        
        for images, decoder_inputs, decoder_targets, captions in train_progress_bar:
            if images is None: continue
            images, decoder_inputs, decoder_targets = images.to(DEVICE), decoder_inputs.to(DEVICE), decoder_targets.to(DEVICE).long()
            
            # Forward pass: encode images, then decode with input tokens
            patch_features = encoder(images)  # [batch_size, 256, 76]
            logits = decoder(patch_features, decoder_inputs)  # [batch_size, seq_len, vocab_size]
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), decoder_targets.reshape(-1))
            
            # Calculate training token accuracy
            train_token_acc = calculate_token_accuracy(logits, decoder_targets)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), config['grad_clip_norm'])
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), config['grad_clip_norm'])
            optimizer.step()
            scheduler.step()
            
            total_train_loss += loss.item()
            step_count += 1
            
            wandb.log({
                "train_batch_loss": loss.item(), 
                "train_batch_token_accuracy": train_token_acc,
                "train_batch_perplexity": calculate_perplexity(loss.item()),
                "learning_rate": scheduler.get_last_lr()[0], 
                "step": step_count
            })
            
            if is_sweep:
                train_progress_bar.set_postfix(
                    loss=loss.item(), 
                    acc=f"{train_token_acc:.3f}",
                    lr=scheduler.get_last_lr()[0], 
                    step=f"{step_count}/{config['max_steps_per_sweep']}"
                )
            else:
                train_progress_bar.set_postfix(
                    loss=loss.item(), 
                    acc=f"{train_token_acc:.3f}",
                    lr=scheduler.get_last_lr()[0]
                )
            
            # Stop early for sweeps after max_steps_per_sweep
            if is_sweep and step_count >= config['max_steps_per_sweep']:
                print(f"\n✅ Reached {config['max_steps_per_sweep']} steps, stopping training phase")
                break

        avg_train_loss = total_train_loss / step_count if step_count > 0 else 0
        
        # --- Validation Phase (skip for sweeps) ---
        avg_val_loss = None
        avg_val_token_accuracy = None
        avg_val_perplexity = None
        avg_val_bleu = None
        
        if not is_sweep:
            encoder.eval()
            decoder.eval()
            total_val_loss = 0
            total_token_accuracy = 0
            total_bleu_score = 0
            val_batches = 0
            
            val_progress_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{config['epochs']} [Validate]")
            with torch.no_grad():
                for images, decoder_inputs, decoder_targets, captions in val_progress_bar:
                    if images is None: continue
                    images, decoder_inputs, decoder_targets = images.to(DEVICE), decoder_inputs.to(DEVICE), decoder_targets.to(DEVICE).long()
                    
                    # Forward pass: encode images, then decode with input tokens  
                    patch_features = encoder(images)  # [batch_size, 256, 76]
                    logits = decoder(patch_features, decoder_inputs)  # [batch_size, seq_len, vocab_size]
                    loss = loss_fn(logits.reshape(-1, logits.size(-1)), decoder_targets.reshape(-1))
                    
                    # Calculate metrics
                    total_val_loss += loss.item()
                    
                    # Token accuracy
                    token_acc = calculate_token_accuracy(logits, decoder_targets)
                    total_token_accuracy += token_acc
                    
                    # BLEU score (simplified - just for first item in batch)
                    if len(decoder_targets) > 0:
                        pred_tokens = torch.argmax(logits[0], dim=-1)
                        target_tokens = decoder_targets[0]
                        bleu = calculate_bleu_score(
                            pred_tokens, target_tokens, 
                            lambda x: decode_tokens(x, decoder.clip_model)
                        )
                        total_bleu_score += bleu
                    
                    val_batches += 1
                    val_progress_bar.set_postfix(
                        loss=loss.item(), 
                        token_acc=f"{token_acc:.3f}",
                        bleu=f"{bleu:.3f}" if 'bleu' in locals() else "0.000"
                    )

            avg_val_loss = total_val_loss / len(val_loader)
            avg_val_token_accuracy = total_token_accuracy / val_batches
            avg_val_perplexity = calculate_perplexity(avg_val_loss)
            avg_val_bleu = total_bleu_score / val_batches
            
            # --- Logging & Checkpointing ---
            wandb.log({
                "epoch": epoch + 1,
                "avg_train_loss": avg_train_loss,
                "avg_val_loss": avg_val_loss,
                "val_token_accuracy": avg_val_token_accuracy,
                "val_perplexity": avg_val_perplexity,
                "val_bleu_score": avg_val_bleu,
                "train_perplexity": calculate_perplexity(avg_train_loss)
            })
            print(f"Epoch {epoch+1}/{config['epochs']} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Token Acc: {avg_val_token_accuracy:.3f} | BLEU: {avg_val_bleu:.3f}")
        else:
            # For sweeps, only log training metrics
            wandb.log({
                "epoch": epoch + 1,
                "avg_train_loss": avg_train_loss,
                "train_perplexity": calculate_perplexity(avg_train_loss),
                "steps_completed": step_count
            })
            print(f"Sweep | Steps: {step_count}/{config['max_steps_per_sweep']} | Train Loss: {avg_train_loss:.4f} | PPL: {calculate_perplexity(avg_train_loss):.2f}")

        # Save checkpoint (for both regular training and sweeps)
        save_checkpoint(
            run_dir=run_dir,
            epoch=epoch,
            encoder=encoder,
            decoder=decoder,
            optimizer=optimizer,
            train_loss=avg_train_loss,
            val_loss=avg_val_loss,
            step_count=step_count if is_sweep else None
        )

    wandb.finish()
    print(f"🎉 Training completed! Results saved in: {run_dir}")

if __name__ == '__main__':
    train() 