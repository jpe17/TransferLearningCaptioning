"""
Inference Module for Image Captioning
=====================================
"""

import torch
from .config import BASE_MODEL_NAME, DEVICE, IMAGE_SIZE, IMAGE_NORMALIZE_MEAN, IMAGE_NORMALIZE_STD
from .model import Encoder, Decoder
from .model_loader import get_clip_model, get_base_model
from .utils import get_tokenizer
import os
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
    decoder = Decoder(model_name=BASE_MODEL_NAME).to(DEVICE)
    
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
    Handles caption generation using a manual, step-by-step inference loop.
    """
    def __init__(self, encoder, decoder):
        self.encoder = encoder
        self.decoder = decoder
        self.tokenizer = get_tokenizer()

    @torch.no_grad()
    def generate_caption(self, images, max_length=50, temperature=0.7, do_sample=False):
        """
        Generates a caption for a batch of images using a handmade autoregressive loop.

        Args:
            images (torch.Tensor): The input images.
            max_length (int): The maximum length of the generated caption.
            temperature (float): Controls the randomness of sampling. Higher is more random.
            do_sample (bool): If True, uses sampling. If False, uses greedy decoding.
                              Defaults to False for deterministic output.
        """
        self.encoder.eval()
        self.decoder.eval()

        # Ensure images are on the correct device
        images = images.to(DEVICE)
        batch_size = images.size(0)

        # 1. Encode image to get visual features, already projected to the right dimension
        visual_embeds = self.encoder(images)

        # 2. Start generation with a proper start token
        # Use a neutral token that won't immediately terminate generation
        if hasattr(self.tokenizer, 'bos_token_id') and self.tokenizer.bos_token_id is not None:
            start_token_id = self.tokenizer.bos_token_id
        else:
            # Use a common word token instead of eos_token_id to avoid immediate termination
            start_token_id = self.tokenizer.encode("The", add_special_tokens=False)[0]
        
        input_ids = torch.full((batch_size, 1), start_token_id, dtype=torch.long, device=DEVICE)

        # 3. Generate tokens one by one
        for _ in range(max_length):
            # Embed the tokens generated so far
            text_embeds = self.decoder.base_model.get_input_embeddings()(input_ids)
            text_embeds = self.decoder.text_projection(text_embeds)

            # Combine visual and text embeddings
            combined_embeds = torch.cat([visual_embeds, text_embeds], dim=1)
            
            # Create the same attention mask as in training
            total_len = combined_embeds.size(1)
            num_patches = visual_embeds.size(1)
            
            # 1. Start with a causal mask for the entire sequence
            attention_mask = torch.tril(torch.ones((total_len, total_len), device=DEVICE))
            
            # 2. Make the visual prefix part of the mask bidirectional
            attention_mask[:, :num_patches] = 1
            
            # 3. Reshape to the 4D format (batch_size, num_heads, seq_len, seq_len)
            attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)
            attention_mask = attention_mask.expand(batch_size, -1, -1, -1)

            # Get model outputs
            outputs = self.decoder.base_model(
                inputs_embeds=combined_embeds,
                attention_mask=attention_mask,
                return_dict=True
            )

            # Get the logits for the very last token (the prediction for the next token)
            next_token_logits = outputs.last_hidden_state[:, -1, :]
            next_token_logits = self.decoder.output_head(next_token_logits)

            # 4. Sample the next token
            if do_sample and temperature > 0:
                probs = torch.softmax(next_token_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # 5. Append the new token
            input_ids = torch.cat([input_ids, next_token], dim=1)

            # 6. Stop if we generate the End-Of-Sentence token (but only if it's not the first token)
            if next_token.item() == self.tokenizer.eos_token_id and input_ids.size(1) > 2:
                break
        
        # 7. Decode the generated sequence into text
        captions = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        
        # Debug information
        print(f"🔍 Generated {input_ids.size(1)} tokens")
        print(f"🔍 Raw token IDs: {input_ids[0].tolist()}")
        print(f"🔍 Generated caption: '{captions[0] if batch_size == 1 else captions[0]}'")
        
        return captions[0] if batch_size == 1 else captions 