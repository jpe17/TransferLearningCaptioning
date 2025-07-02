import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import os
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
from rouge import Rouge
import nltk

from .model import Encoder, Decoder
from .utils import set_seed, create_run_folder, save_checkpoint, calculate_perplexity, calculate_token_accuracy
from .model_loader import get_clip_model, get_model_tokenizer_pair
from .data_processing import create_dataloaders
from .config import *
from .inference import CaptionGenerator

def validate(caption_generator, val_loader):
    """Run validation and compute advanced metrics."""
    caption_generator.encoder.eval()
    caption_generator.decoder.eval()

    all_generated_captions = []
    all_reference_captions = []

    example_table = wandb.Table(columns=["Image", "Generated Caption", "True Caption"])
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(val_loader, desc="Validation", leave=False)):
            if not batch:
                continue

            images = batch['image']
            true_captions = batch['caption']

            generated_captions = caption_generator.generate_caption(images)
            
            if not isinstance(generated_captions, list):
                generated_captions = [generated_captions]

            all_generated_captions.extend(generated_captions)
            all_reference_captions.extend(true_captions)

            if i == 0:
                for j in range(min(4, len(images))):
                    img = images[j]
                    gen_cap = generated_captions[j]
                    true_cap = true_captions[j]
                    example_table.add_data(wandb.Image(img), gen_cap, true_cap)

    # Prepare captions for metric calculation
    # NLTK BLEU expects list of lists of tokens for references
    # ROUGE expects list of strings
    tokenized_references = [[cap.split() for cap in all_reference_captions]]
    tokenized_generated = [cap.split() for cap in all_generated_captions]
    
    # BLEU Score
    bleu_score = corpus_bleu(tokenized_references, tokenized_generated)
    
    # ROUGE Score
    rouge = Rouge()
    # ROUGE expects a list of hypothesis strings and a list of reference strings
    rouge_scores = rouge.get_scores(all_generated_captions, all_reference_captions, avg=True)
    rouge_l_f = rouge_scores['rouge-l']['f']

    # METEOR Score
    # NLTK METEOR expects tokenized references and hypotheses
    meteor = sum(meteor_score([ref.split()], gen.split()) for ref, gen in zip(all_reference_captions, all_generated_captions)) / len(all_generated_captions)

    val_metrics = {
        "val_bleu": bleu_score,
        "val_rouge-L": rouge_l_f,
        "val_meteor": meteor,
        "validation_examples": example_table
    }

    caption_generator.encoder.train()
    caption_generator.decoder.train()

    return val_metrics

def train(test_mode=False):
    # Download NLTK data required for METEOR
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')
        
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
    base_model, _ = get_model_tokenizer_pair()
    qwen_dim = base_model.config.hidden_size
    
    encoder = Encoder(output_dim=qwen_dim, clip_model=clip_model).to(DEVICE)
    decoder = Decoder().to(DEVICE)
    caption_generator = CaptionGenerator(encoder, decoder)
    
    # Training
    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=LEARNING_RATE)
    
    # --- Sanity Check ---
    # Using the model and tokenizer directly as a pair
    print("✅ Using model and tokenizer directly as a pair.")
    # --- End Sanity Check ---
    
    wandb.init(project="simple-captioning", name=os.path.basename(run_dir))
    
    step = 0
    max_steps = 10 if test_mode else float('inf')  # Only do 10 steps in test mode
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
            tokenizer = decoder.tokenizer  # Use tokenizer directly from decoder
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
                
                # Run validation with advanced metrics
                val_metrics = validate(caption_generator, val_loader)
                print(f"Validation -> BLEU: {val_metrics['val_bleu']:.4f}, ROUGE-L: {val_metrics['val_rouge-L']:.4f}, METEOR: {val_metrics['val_meteor']:.4f}")
                
                # Log validation metrics to wandb, including the table of examples
                wandb.log(val_metrics)
                
                save_checkpoint(run_dir, step, encoder, decoder, optimizer, config)
            
            if step >= max_steps:
                break
    
    # Final save
    save_checkpoint(run_dir, step, encoder, decoder, optimizer, config)
    
    wandb.finish()
    print(f"✅ Training complete! Results saved in: {run_dir}")

if __name__ == '__main__':
    train() 