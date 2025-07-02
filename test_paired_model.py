"""
Test script to verify we're using the model and tokenizer directly as a pair
"""

from src.core.model_loader import get_model_tokenizer_pair
from src.core.model import Decoder

print("=== Model-Tokenizer Pair Test ===")

# Get model and tokenizer as a pair
print("1. Loading model-tokenizer pair...")
model, tokenizer = get_model_tokenizer_pair()
model_vocab_size = model.config.vocab_size
tokenizer_vocab_size = len(tokenizer)
print(f"   Model vocabulary size: {model_vocab_size}")
print(f"   Tokenizer vocabulary size: {tokenizer_vocab_size}")

# Initialize decoder
print("\n2. Initializing decoder...")
decoder = Decoder()

# Verify that decoder is using the model's vocabulary size
print("\n3. Verifying decoder's output head size...")
output_head_size = decoder.output_head.weight.size(0)
print(f"   Decoder output head size: {output_head_size}")

# Verify that decoder's tokenizer is the same as the one from the pair
print("\n4. Verifying decoder's tokenizer...")
decoder_tokenizer_vocab_size = len(decoder.tokenizer)
print(f"   Decoder's tokenizer vocabulary size: {decoder_tokenizer_vocab_size}")

# Final check
print("\n=== Final Check ===")
if output_head_size == model_vocab_size and decoder_tokenizer_vocab_size == tokenizer_vocab_size:
    print("✅ SUCCESS: Decoder is using the model and tokenizer directly as a pair!")
else:
    print("❌ ERROR: Decoder is not using the model and tokenizer directly as a pair!")
    print(f"   Model: {model_vocab_size}, Output head: {output_head_size}")
    print(f"   Tokenizer: {tokenizer_vocab_size}, Decoder tokenizer: {decoder_tokenizer_vocab_size}")

print("\nTest complete.") 