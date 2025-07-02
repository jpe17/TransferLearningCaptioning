"""
Test script to verify we're using the model's vocabulary size directly
"""

from src.core.utils import get_tokenizer
from src.core.model_loader import get_base_model
from src.core.model import Decoder

print("=== Model Vocabulary Test ===")

# Get tokenizer directly from the model
print("1. Loading tokenizer...")
tokenizer = get_tokenizer()
print(f"   Base vocabulary size: {tokenizer.vocab_size}")
print(f"   Total vocabulary size (with special tokens): {len(tokenizer)}")

# Get base model
print("\n2. Loading base model...")
base_model = get_base_model()
model_vocab_size = base_model.config.vocab_size
print(f"   Model vocabulary size: {model_vocab_size}")
print(f"   Model embedding size: {base_model.get_input_embeddings().weight.shape}")

# Initialize decoder (which should use model's vocab size)
print("\n3. Initializing decoder...")
decoder = Decoder()

# Verify output head size
print("\n4. Verifying output head size...")
output_head_size = decoder.output_head.weight.size(0)
print(f"   Output head size: {output_head_size}")

# Final check
print("\n=== Final Check ===")
if output_head_size == model_vocab_size:
    print("✅ SUCCESS: Output head matches model's vocabulary size!")
else:
    print("❌ ERROR: Output head doesn't match model's vocabulary size!")
    print(f"   Model: {model_vocab_size}, Output head: {output_head_size}")

print("\nTest complete.") 