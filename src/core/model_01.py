"""
Simple Encoder with Trainable MLP
=================================
"""

import torch
import torch.nn as nn
import clip
import random
import numpy as np
from .config import *
from transformers import AutoModel, AutoConfig


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(SEED)


class Encoder(nn.Module):
    """
    Simple encoder: frozen CLIP + trainable MLP projection
    """
    
    def __init__(self, output_dim=512, hidden_dim=2048):
        super().__init__()
        
        # Frozen CLIP encoder
        self.clip_model, _ = clip.load(CLIP_MODEL, device=DEVICE)
        for param in self.clip_model.parameters():
            param.requires_grad = False

        # Trainable FCNN projection: expand → activate → compress
        clip_dim = self.clip_model.visual.output_dim
        self.FCNN = nn.Sequential(
            nn.Linear(clip_dim, hidden_dim),    # Expand to higher dimension
            nn.GELU(),                          # Non-linear activation
            nn.Linear(hidden_dim, output_dim)   # Compress to target dimension
        )
        
    def forward(self, images):
        # Get CLIP features (frozen)
        with torch.no_grad():
            clip_features = self.clip_model.encode_image(images).float()

        # Project to desired dimensions (trainable)
        return self.FCNN(clip_features)


class Decoder(nn.Module):
    """
    Simple decoder: combines image features + text tokens → pre-trained language model
    """
    
    def __init__(self, model_name="gpt2", embed_dim=768, freeze_base=True):
        super().__init__()
        
        # Base language model (customizable)
        self.base_model = AutoModel.from_pretrained(model_name)
        
        # Use CLIP tokenizer (same as data processing)
        self.vocab_size = 49408  # CLIP vocab size
        
        # Freeze base model if requested
        if freeze_base:
            for param in self.base_model.parameters():
                param.requires_grad = False
        
        # Project image features to match text embedding dimension
        self.image_projection = nn.Linear(76, embed_dim)  # 76 from encoder → embed_dim
        
        # Project CLIP tokens to base model embedding dimension
        self.token_projection = nn.Linear(77, embed_dim)  # CLIP tokens are 77 max length
        
        # Final output head (trainable) - outputs CLIP vocab
        self.output_head = nn.Linear(embed_dim, self.vocab_size)
        
    def forward(self, image_features, clip_tokens):
        """
        Args:
            image_features: [batch_size, 76] from encoder
            clip_tokens: [batch_size, 77] from CLIP tokenizer (same as data processing)
        Returns:
            logits: [batch_size, 77, vocab_size] CLIP vocab predictions
        """
        batch_size = image_features.size(0)
        
        # Project image features to embedding space
        image_embeds = self.image_projection(image_features)  # [batch_size, embed_dim]
        image_embeds = image_embeds.unsqueeze(1)  # [batch_size, 1, embed_dim]
        
        # Project CLIP tokens to embedding space
        token_embeds = self.token_projection(clip_tokens.float())  # [batch_size, embed_dim]
        token_embeds = token_embeds.unsqueeze(1)  # [batch_size, 1, embed_dim]
        
        # Combine: [image_token] + [text_representation]
        combined_embeds = torch.cat([image_embeds, token_embeds], dim=1)  # [batch_size, 2, embed_dim]
        
        # Pass through base model
        outputs = self.base_model(inputs_embeds=combined_embeds)
        hidden_states = outputs.last_hidden_state  # [batch_size, 2, embed_dim]
        
        # Use the combined representation for prediction
        combined_repr = hidden_states.mean(dim=1)  # [batch_size, embed_dim]
        
        # Generate logits for CLIP vocabulary
        logits = self.output_head(combined_repr)  # [batch_size, vocab_size]
        
        return logits

    def generate(self, image_features, max_length=77):
        """Generate using CLIP tokenizer format"""
        self.eval()
        with torch.no_grad():
            # Start with empty CLIP tokens
            generated_tokens = torch.zeros(1, 77, device=image_features.device)
            
            for i in range(max_length):
                logits = self.forward(image_features, generated_tokens)
                next_token = logits.argmax(dim=-1)
                
                # This is simplified - you'd need proper CLIP decoding logic
                if i < 77:
                    generated_tokens[0, i] = next_token
                    
                # Check for end token (CLIP specific)
                if next_token.item() == 49407:  # CLIP EOS token
                    break
            
            return generated_tokens
