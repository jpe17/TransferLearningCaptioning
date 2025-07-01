import os
import json
import torch
import clip
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from .config import *

class SimpleImageCaptionDataset(Dataset):
    """Simple dataset for image-caption pairs"""
    
    def __init__(self, data_pairs, transform=None, tokenizer=None):
        self.data = data_pairs
        self.transform = transform or self._default_transform()
        self.tokenizer = tokenizer or clip_tokenizer
    
    def _default_transform(self):
        return transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(IMAGE_NORMALIZE_MEAN, IMAGE_NORMALIZE_STD)
        ])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image_path, caption = self.data[idx]
        
        # Load and transform image
        try:
            image = Image.open(image_path).convert("RGB")
            image_tensor = self.transform(image)
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            return None
        
        # Tokenize caption using the provided tokenizer
        tokens = self.tokenizer(caption)
        
        # Create teacher forcing pairs: input = [:-1], target = [1:]
        if len(tokens) <= 1:
            return None  # Skip sequences that are too short
            
        input_tokens = tokens[:-1]    # [BOS, "A", "dog", "is"]
        target_tokens = tokens[1:]    # ["A", "dog", "is", "running", EOS]
        
        return image_tensor, input_tokens, target_tokens, caption

def load_flickr_data(images_dir, captions_file, max_samples=None):
    """Load Flickr data"""
    with open(captions_file, 'r') as f:
        data = json.load(f)
    
    pairs = []
    for item in data:
        image_path = os.path.join(images_dir, item['image'])
        pairs.append((image_path, item['caption']))
        if max_samples is not None and len(pairs) >= max_samples:
            break
    
    return pairs

# Default tokenizer using CLIP
def clip_tokenizer(caption):
    """Default CLIP tokenizer"""
    return clip.tokenize(caption, truncate=True).squeeze()

def collate_fn(batch):
    """Custom collate function to handle None values from dataset."""
    batch = [item for item in batch if item is not None]
    if not batch:
        return None, None, None, None
    
    images, input_tokens, target_tokens, captions = zip(*batch)
    return torch.stack(images), torch.stack(input_tokens), torch.stack(target_tokens), list(captions)

def create_dataloaders(images_dir=None, captions_file=None, batch_size=None, val_split=None, test_split=None, seed=None, tokenizer=None, max_samples=None):
    """Create train/val/test dataloaders with teacher forcing"""
    # Use config defaults if not provided
    images_dir = images_dir or IMAGES_DIR
    captions_file = captions_file or CAPTIONS_FILE
    batch_size = batch_size or BATCH_SIZE
    val_split = val_split or VAL_SPLIT
    test_split = test_split or TEST_SPLIT
    seed = seed or SEED
    
    data_pairs = load_flickr_data(images_dir, captions_file, max_samples=max_samples)
    dataset = SimpleImageCaptionDataset(data_pairs, tokenizer=tokenizer)
    
    # Calculate splits
    test_size = int(len(dataset) * test_split)
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size - test_size
    
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size], generator=generator
    )
    
    # Use multiple workers for faster data loading
    # Set persistent_workers=True to avoid re-initializing workers, which is slow on macOS/Windows
    # Set pin_memory=True to speed up data transfer to the GPU
    num_workers = 4  # A sensible default, can be tuned
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=SHUFFLE_DATALOADER, 
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )
    
    return train_loader, val_loader, test_loader

# Example usage:
# train_loader, val_loader, test_loader = create_dataloaders()
