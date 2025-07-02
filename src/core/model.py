"""
Model Definitions
================
Simple models for image captioning with CLIP + QWEN.
"""

import torch
import torch.nn as nn
from .config import *
from .model_loader import get_clip_model, get_base_model, get_model_tokenizer_pair
from .utils import set_seed

# Disable compilation globally
torch._dynamo.config.suppress_errors = True
torch._dynamo.disable()

# Set random seed
set_seed(SEED)


class Encoder(nn.Module):
    """CLIP patch encoder"""
    
    def __init__(self, hidden_dim=None, output_dim=None, clip_model=None):
        super().__init__()
        self.hidden_dim = hidden_dim or ENCODER_HIDDEN_DIM
        self.output_dim = output_dim or ENCODER_OUTPUT_DIM
        
        self.clip_model = clip_model or get_clip_model()
        self.clip_dim = self.clip_model.visual.transformer.width
        
        self.projection = nn.Sequential(
            nn.Linear(self.clip_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.output_dim)
        )
        
    def forward(self, images):
        # Input: images [batch_size, 3, 224, 224]
        with torch.no_grad():
            # Match input dtype to what CLIP expects
            images = images.to(self.clip_model.visual.conv1.weight.dtype)  # [batch_size, 3, 224, 224]
            
            x = self.clip_model.visual.conv1(images)  # [batch_size, 768, 14, 14] (patch embedding)
            x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)  # [batch_size, 196, 768] (flatten patches)
            x = self.clip_model.visual.ln_pre(x)  # [batch_size, 196, 768] (layer norm)
            x = x.permute(1, 0, 2)  # [196, batch_size, 768] (seq_len first for transformer)
            x = self.clip_model.visual.transformer(x)  # [196, batch_size, 768] (transformer output)
            x = x.permute(1, 0, 2)  # [batch_size, 196, 768] (batch first again)
        
        # Match output dtype to projection layer
        x = x.to(self.projection[0].weight.dtype)
        # Output: x [batch_size, 196, 768] -> projection -> [batch_size, 196, output_dim]
        return self.projection(x)  # [batch_size, 196, ENCODER_OUTPUT_DIM]


class Decoder(nn.Module):
    """QWEN decoder with trainable attention"""
    
    def __init__(self, model_name=None):
        super().__init__()
        self.model_name = model_name or BASE_MODEL_NAME
        
        # Get model and tokenizer as a perfectly aligned pair
        self.base_model, self.tokenizer = get_model_tokenizer_pair(self.model_name)
        self.qwen_dim = self.base_model.config.hidden_size
        
        # Get vocabulary size directly from the model
        model_vocab_size = self.base_model.config.vocab_size
        print(f"🔍 Using model's vocabulary size: {model_vocab_size}")
        
        self.text_projection = nn.Linear(self.qwen_dim, self.qwen_dim)
        # Use the model's vocabulary size for the output head
        self.output_head = nn.Linear(self.qwen_dim, model_vocab_size)
        print(f"✅ Output head created with model's vocabulary size: {model_vocab_size}")

        # Make QWEN attention layers trainable - direct access for QWEN
        for layer in self.base_model.layers:
            attn = layer.self_attn
            
            # Replace q_proj and v_proj with trainable versions
            new_q = nn.Linear(attn.q_proj.in_features, attn.q_proj.out_features)
            new_v = nn.Linear(attn.v_proj.in_features, attn.v_proj.out_features)
            
            new_q.weight.data.copy_(attn.q_proj.weight.data)
            new_v.weight.data.copy_(attn.v_proj.weight.data)
            
            attn.q_proj = new_q
            attn.v_proj = new_v

        print(f"✅ Decoder ready (dim: {self.qwen_dim})")
    
    def forward(self, patch_features, input_tokens, labels=None):
        # Input shapes:
        # patch_features: [batch_size, num_patches, qwen_dim] (already projected by encoder)
        # input_tokens: [batch_size, seq_len] where seq_len varies per sample
        
        batch_size = patch_features.size(0)
        num_patches = patch_features.size(1)  # 196
        seq_len = input_tokens.size(1)
        
        # The encoder has already projected patch features to the correct dimension.
        visual_embeds = patch_features.to(self.text_projection.weight.dtype)
        
        with torch.no_grad():
            text_embeds = self.base_model.get_input_embeddings()(input_tokens)  # [batch_size, seq_len, qwen_dim]
            
        # Match text_embeds dtype to text_projection
        text_embeds = text_embeds.to(self.text_projection.weight.dtype)  # [batch_size, seq_len, qwen_dim]
        text_embeds = self.text_projection(text_embeds)  # [batch_size, seq_len, qwen_dim]
        
        # Concatenate visual and text embeddings
        combined_embeds = torch.cat([visual_embeds, text_embeds], dim=1)  # [batch_size, 196+seq_len, qwen_dim]
        total_len = num_patches + seq_len
        
        # Create a custom 4D attention mask to handle the visual prefix correctly.
        # This is the standard and correct format for Hugging Face models when mixing
        # bidirectional attention (for the image) and causal attention (for the text).
        
        # 1. Start with a causal mask for the entire sequence. This assumes no token
        #    can see a future token.
        #    Shape: [total_len, total_len]
        attention_mask = torch.tril(torch.ones((total_len, total_len), device=combined_embeds.device))
        
        # 2. Make the visual prefix part of the mask bidirectional. We modify the mask
        #    so that every token in the sequence (both visual and text) can attend to
        #    all visual tokens. This is done by setting the first `num_patches` columns to 1.
        attention_mask[:, :num_patches] = 1
        
        # 3. Reshape to the 4D format (batch_size, num_heads, seq_len, seq_len)
        #    by adding dimensions for batch and heads, which will be broadcasted by PyTorch.
        #    This is the final format expected by the Transformer model.
        attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)
        attention_mask = attention_mask.expand(batch_size, -1, -1, -1)

        # Forward through QWEN with the proper 4D attention mask.
        outputs = self.base_model(
            inputs_embeds=combined_embeds,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Get logits for text positions only
        text_hidden = outputs.last_hidden_state[:, num_patches:, :]  # [batch_size, seq_len, qwen_dim]
        logits = self.output_head(text_hidden)  # [batch_size, seq_len, vocab_size]
        
        # Calculate loss
        loss = None
        if labels is not None:
            # labels: [batch_size, seq_len]
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        return {"logits": logits, "loss": loss}


