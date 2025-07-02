"""
Model Definitions
================
Simple models for image captioning with CLIP + QWEN.
"""

import torch
import torch.nn as nn
from .config import *
from .model_loader import get_clip_model, get_base_model
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
        hidden_dim = hidden_dim or ENCODER_HIDDEN_DIM
        output_dim = output_dim or ENCODER_OUTPUT_DIM
        
        self.clip_model = clip_model or get_clip_model()
        self.clip_dim = self.clip_model.visual.transformer.width
        
        self.projection = nn.Sequential(
            nn.Linear(self.clip_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, images):
        with torch.no_grad():
            x = self.clip_model.visual.conv1(images.to(self.clip_model.visual.conv1.weight.dtype))
            x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
            x = self.clip_model.visual.ln_pre(x)
            x = x.permute(1, 0, 2)
            x = self.clip_model.visual.transformer(x)
            x = x.permute(1, 0, 2)
        
        # Convert output to the same dtype as projection layer weights
        x = x.to(self.projection[0].weight.dtype)
        return self.projection(x)


class Decoder(nn.Module):
    """QWEN decoder with trainable attention"""
    
    def __init__(self, model_name=None, encoder_output_dim=None):
        super().__init__()
        self.base_model = get_base_model(model_name)
        self.qwen_dim = self.base_model.config.hidden_size
        
        self.visual_projection = nn.Linear(encoder_output_dim, self.qwen_dim)
        self.text_projection = nn.Linear(self.qwen_dim, self.qwen_dim)
        self.output_head = nn.Linear(self.qwen_dim, self.base_model.config.vocab_size)

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
        # Project visual and text to same space
        visual_embeds = self.visual_projection(patch_features)
        
        with torch.no_grad():
            text_embeds = self.base_model.get_input_embeddings()(input_tokens)
        text_embeds = self.text_projection(text_embeds)
        
        # Concatenate and create attention mask
        combined_embeds = torch.cat([visual_embeds, text_embeds], dim=1)
        
        num_patches = visual_embeds.size(1)
        seq_len = text_embeds.size(1)
        total_len = num_patches + seq_len
        
        # Causal mask: text can't see future text, but can see all visual patches
        causal_mask = torch.triu(torch.ones(total_len, total_len), diagonal=1).bool()
        causal_mask[:, :num_patches] = False
        attention_mask = ~causal_mask.to(combined_embeds.device)
        
        # Forward through QWEN
        outputs = self.base_model(
            inputs_embeds=combined_embeds,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Get logits for text positions only
        text_hidden = outputs.last_hidden_state[:, num_patches:, :]
        logits = self.output_head(text_hidden)
        
        # Calculate loss
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        return {"logits": logits, "loss": loss}


