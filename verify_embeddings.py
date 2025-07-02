"""
Script to verify that our implementation uses the exact same embeddings as the original model.
"""

from transformers import AutoModel, AutoTokenizer
from src.core.model import Decoder
import torch

print("=== Embedding Verification ===")

# Load the original model and tokenizer
print("1. Loading original model and tokenizer...")
orig_model = AutoModel.from_pretrained('Qwen/Qwen3-0.6B-Base', trust_remote_code=True)
orig_tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B-Base', trust_remote_code=True)

# Create our decoder
print("\n2. Creating our decoder...")
decoder = Decoder()

# Get embeddings
print("\n3. Comparing embeddings...")
original_emb = orig_model.get_input_embeddings().weight
decoder_emb = decoder.base_model.get_input_embeddings().weight

# Check if they're the same
print(f"Original embedding shape: {original_emb.shape}")
print(f"Decoder embedding shape: {decoder_emb.shape}")
print(f"Are embeddings identical? {torch.allclose(original_emb, decoder_emb)}")

# Check sample tokens
print("\n4. Comparing sample tokens...")
for token_id in [0, 1000, 151643, 151935]:
    print(f"\nToken ID {token_id}:")
    print(f"Original: {original_emb[token_id, :5]}")
    print(f"Decoder:  {decoder_emb[token_id, :5]}")
    print(f"Match?    {torch.allclose(original_emb[token_id], decoder_emb[token_id])}")

# Check the extra embeddings
print("\n5. Analyzing extra embeddings (beyond tokenizer's vocabulary)...")
extra_start = len(orig_tokenizer)
extra_end = orig_model.config.vocab_size
print(f"Extra embeddings range: {extra_start} to {extra_end-1}")

extra_emb = original_emb[extra_start:extra_end]
print(f"Extra embeddings mean absolute value: {extra_emb.abs().mean().item():.6f}")
print(f"Regular embeddings mean absolute value: {original_emb[:extra_start].abs().mean().item():.6f}")
print(f"Ratio: {extra_emb.abs().mean().item() / original_emb[:extra_start].abs().mean().item():.6f}")

print("\n=== Conclusion ===")
if torch.allclose(original_emb, decoder_emb):
    print("✅ SUCCESS: Our implementation uses the exact same embeddings as the original model!")
else:
    print("❌ ERROR: Our implementation does NOT use the same embeddings as the original model!") 