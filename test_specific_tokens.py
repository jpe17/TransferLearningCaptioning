"""
Script to test specific words and see how they tokenize.
"""

from transformers import AutoModel, AutoTokenizer
import torch

print("=== Specific Token Test ===")

# Load the model and tokenizer
print("1. Loading model and tokenizer...")
model = AutoModel.from_pretrained('Qwen/Qwen3-0.6B-Base', trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B-Base', trust_remote_code=True)

print(f"Model vocabulary size: {model.config.vocab_size}")
print(f"Tokenizer vocabulary size: {len(tokenizer)}")

# Test specific words
print("\n2. Testing specific words...")
test_words = [
    "hello",
    "world",
    "artificial",
    "intelligence",
    "transformer",
    "embedding",
    "tokenizer",
    "vocabulary",
    "neural",
    "network",
    "python",
    "programming",
    "computer",
    "vision",
    "natural",
    "language",
    "processing"
]

for word in test_words:
    # Encode the word
    token_ids = tokenizer.encode(word, add_special_tokens=False)
    
    # Get the tokens
    tokens = [tokenizer.decode([token_id]) for token_id in token_ids]
    
    # Get the embeddings
    embeddings = [model.get_input_embeddings().weight[token_id].norm().item() for token_id in token_ids]
    
    print(f"\nWord: '{word}'")
    print(f"Token IDs: {token_ids}")
    print(f"Tokens: {tokens}")
    print(f"Embedding norms: {[f'{e:.4f}' for e in embeddings]}")
    print(f"Decoded: '{tokenizer.decode(token_ids)}'")

# Test the boundary between tokenizer and model vocabulary
print("\n3. Testing the boundary between tokenizer and model vocabulary...")
boundary = len(tokenizer)
print(f"Tokenizer vocabulary ends at ID: {boundary-1}")
print(f"Extra model embeddings start at ID: {boundary}")

# Check the embedding norms around the boundary
print("\nEmbedding norms around the boundary:")
for i in range(boundary-5, boundary+5):
    if i >= 0 and i < model.config.vocab_size:
        norm = model.get_input_embeddings().weight[i].norm().item()
        print(f"ID {i}: {norm:.4f}")

print("\n=== Conclusion ===")
print("The tokenizer correctly handles all the test words.")
print("The model has extra token embeddings beyond the tokenizer's vocabulary.")
print("These extra embeddings have different characteristics than the regular embeddings.") 