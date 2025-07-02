"""
Simple test script to verify tokenizer and model vocabulary sizes match
"""

from src.core.utils import get_tokenizer
from src.core.model_loader import get_base_model
from src.core.model import Decoder

print("=== Tokenizer and Model Vocabulary Test ===")

# Get tokenizer
print("1. Loading tokenizer...")
tokenizer = get_tokenizer()
tokenizer_vocab_size = len(tokenizer)
print(f"   Tokenizer vocabulary size: {tokenizer_vocab_size}")

# Get base model
print("\n2. Loading base model...")
base_model = get_base_model()
model_vocab_size = base_model.config.vocab_size
print(f"   Base model vocabulary size: {model_vocab_size}")

# Initialize decoder (which should fix any mismatch)
print("\n3. Initializing decoder (should resize embeddings if needed)...")
decoder = Decoder()

# Verify output head size
print("\n4. Verifying output head size...")
output_head_size = decoder.output_head.weight.size(0)
print(f"   Output head size: {output_head_size}")

# Final check
print("\n=== Final Check ===")
if output_head_size == tokenizer_vocab_size:
    print("✅ SUCCESS: Tokenizer and model vocabulary sizes match!")
else:
    print("❌ ERROR: Tokenizer and model vocabulary sizes still don't match!")
    print(f"   Tokenizer: {tokenizer_vocab_size}, Output head: {output_head_size}")

print("\nTest complete.") 