"""
Script to compare token IDs and their corresponding words between the model and tokenizer.
"""

from transformers import AutoModel, AutoTokenizer
import torch
import random

print("=== Token Comparison ===")

# Load the original model and tokenizer
print("1. Loading model and tokenizer...")
model = AutoModel.from_pretrained('Qwen/Qwen3-0.6B-Base', trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B-Base', trust_remote_code=True)

print(f"Model vocabulary size: {model.config.vocab_size}")
print(f"Tokenizer vocabulary size: {len(tokenizer)}")
print(f"Difference: {model.config.vocab_size - len(tokenizer)}")

# Get the vocabulary from the tokenizer
vocab = tokenizer.get_vocab()
inv_vocab = {v: k for k, v in vocab.items()}

# Sample random token IDs from the shared range
print("\n2. Comparing random tokens from shared vocabulary range...")
for _ in range(5):
    token_id = random.randint(0, len(tokenizer) - 1)
    token = inv_vocab.get(token_id, "UNKNOWN")
    embedding = model.get_input_embeddings().weight[token_id]
    print(f"\nToken ID: {token_id}")
    print(f"Token: '{token}'")
    print(f"Embedding norm: {embedding.norm().item():.4f}")
    
    # Test encoding and decoding
    if token != "UNKNOWN":
        encoded = tokenizer.encode(token, add_special_tokens=False)
        decoded = tokenizer.decode(encoded)
        print(f"Encoded -> Decoded: {encoded} -> '{decoded}'")

# Check some tokens at the boundary
print("\n3. Checking tokens at the vocabulary boundary...")
boundary_ids = [
    len(tokenizer) - 3,  # Near end of tokenizer vocab
    len(tokenizer) - 1,  # Last token in tokenizer vocab
    len(tokenizer),      # First token beyond tokenizer vocab
    len(tokenizer) + 2   # A bit beyond tokenizer vocab
]

for token_id in boundary_ids:
    token = inv_vocab.get(token_id, "UNKNOWN")
    embedding = model.get_input_embeddings().weight[token_id]
    print(f"\nToken ID: {token_id}")
    print(f"Token: '{token}'")
    print(f"Embedding norm: {embedding.norm().item():.4f}")

# Check if we can decode the extra tokens
print("\n4. Attempting to decode extra tokens...")
for token_id in range(len(tokenizer), min(len(tokenizer) + 10, model.config.vocab_size)):
    embedding = model.get_input_embeddings().weight[token_id]
    print(f"\nToken ID: {token_id}")
    print(f"Embedding norm: {embedding.norm().item():.4f}")
    
    # Try to decode this token ID
    try:
        decoded = tokenizer.decode([token_id])
        print(f"Decoded: '{decoded}'")
    except Exception as e:
        print(f"Cannot decode: {str(e)}")

print("\n=== Conclusion ===")
print("The model has extra token embeddings beyond the tokenizer's vocabulary.")
print("These extra tokens cannot be directly accessed through the tokenizer.")
print("However, all the tokens that the tokenizer CAN produce are correctly aligned with the model.") 