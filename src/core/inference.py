"""
Inference Module for Image Captioning
=====================================
"""

import torch
from transformers import AutoTokenizer
from .config import BASE_MODEL_NAME, DEVICE, IMAGE_SIZE, IMAGE_NORMALIZE_MEAN, IMAGE_NORMALIZE_STD
from .model import Encoder, Decoder
from .model_loader import get_clip_model, get_base_model
import os
import re
from torchvision import transforms

def find_latest_checkpoint_path(run_base_dir="runs"):
    """Find the 'model_checkpoint.pth' file in the latest run directory."""
    if not os.path.exists(run_base_dir):
        return None
        
    run_dirs = [d for d in os.listdir(run_base_dir) if os.path.isdir(os.path.join(run_base_dir, d))]
    if not run_dirs:
        return None

    latest_run_dir = sorted(run_dirs, reverse=True)[0]
    full_run_dir = os.path.join(run_base_dir, latest_run_dir)
    
    checkpoint_path = os.path.join(full_run_dir, "model_checkpoint.pth")
    if os.path.exists(checkpoint_path):
        return checkpoint_path
    
    return None

def load_models_from_checkpoint(checkpoint_path):
    """Load encoder and decoder from a checkpoint file."""
    print(f"🔄 Loading models from checkpoint: {checkpoint_path}")
    
    clip_model = get_clip_model()
    base_model = get_base_model()
    qwen_dim = base_model.config.hidden_size
    
    encoder = Encoder(output_dim=qwen_dim, clip_model=clip_model).to(DEVICE)
    decoder = Decoder(encoder_output_dim=qwen_dim, model_name=BASE_MODEL_NAME).to(DEVICE)
    
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    
    encoder.eval()
    decoder.eval()
    
    print("✅ Models loaded and in evaluation mode.")
    return encoder, decoder

def get_image_transform():
    """Get the image transformation pipeline"""
    return transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGE_NORMALIZE_MEAN, IMAGE_NORMALIZE_STD)
    ])

def initialize_inference_handler():
    """
    Initializes all components required for inference.
    
    Returns:
        A tuple containing the caption generator and the image transform function.
        Returns (None, None) if a checkpoint is not found.
    """
    checkpoint_path = find_latest_checkpoint_path()
    if checkpoint_path is None:
        print("⚠️ No checkpoint found. Inference handler not created.")
        return None, None
    
    encoder, decoder = load_models_from_checkpoint(checkpoint_path)
    caption_generator = CaptionGenerator(encoder, decoder)
    image_transform = get_image_transform()
    
    return caption_generator, image_transform

class CaptionGenerator:
    """
    Handles caption generation using the encoder-decoder model
    """
    
    def __init__(self, encoder, decoder, tokenizer_name=None):
        self.encoder = encoder
        self.decoder = decoder
        self.tokenizer_name = tokenizer_name or BASE_MODEL_NAME
        self.tokenizer = None
        self._load_tokenizer()
    
    def _load_tokenizer(self):
        """Load and configure tokenizer"""
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_name, 
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def generate_caption(self, images, max_length=50, temperature=0.7, do_sample=True):
        """
        Generate captions for images
        
        Args:
            images: Batch of preprocessed images
            max_length: Maximum caption length
            temperature: Sampling temperature (0 = greedy)
            do_sample: Whether to use sampling or greedy decoding
        """
        self.encoder.eval()
        self.decoder.eval()
        
        with torch.no_grad():
            # Extract patch features
            patch_features = self.encoder(images)
            batch_size = patch_features.size(0)
            
            # Project visual patches to QWEN dimension
            visual_embeds = self.decoder.visual_projection(patch_features)
            
            # Start with beginning token
            start_token_id = self._get_start_token()
            input_ids = torch.full((batch_size, 1), start_token_id, device=patch_features.device)
            
            # Generate tokens one by one
            for _ in range(max_length):
                # Get current text embeddings and project them
                text_embeds = self.decoder.base_model.get_input_embeddings()(input_ids)
                text_embeds = self.decoder.text_projection(text_embeds)
                
                # Concatenate visual and text embeddings
                combined_embeds = torch.cat([visual_embeds, text_embeds], dim=1)
                
                # Forward through QWEN
                outputs = self.decoder.base_model(inputs_embeds=combined_embeds, return_dict=True)
                
                # Get next token logits (last text position)
                next_token_logits = self.decoder.output_head(outputs.last_hidden_state[:, -1, :])
                
                # Sample next token
                next_token = self._sample_next_token(next_token_logits, temperature, do_sample)
                
                # Append to sequence
                input_ids = torch.cat([input_ids, next_token], dim=1)
                
                # Stop if EOS
                if self.tokenizer.eos_token_id and next_token.item() == self.tokenizer.eos_token_id:
                    break
            
            # Decode to text
            captions = []
            for i in range(batch_size):
                caption = self.tokenizer.decode(input_ids[i], skip_special_tokens=True)
                captions.append(caption)
            
            return captions if batch_size > 1 else captions[0]
    
    def _get_start_token(self):
        """Get appropriate start token"""
        return self.tokenizer.bos_token_id or self.tokenizer.eos_token_id
    
    def _sample_next_token(self, logits, temperature, do_sample):
        """Sample next token with temperature control"""
        if do_sample and temperature > 0:
            # Apply temperature and sample
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            # Greedy decoding
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
        
        return next_token

def generate_caption(encoder, decoder, images, **kwargs):
    """
    Convenience function for quick caption generation
    
    Args:
        encoder: Trained encoder model
        decoder: Trained decoder model  
        images: Batch of preprocessed images
        **kwargs: Additional arguments for generation (max_length, temperature, etc.)
    """
    generator = CaptionGenerator(encoder, decoder)
    return generator.generate_caption(images, **kwargs) 