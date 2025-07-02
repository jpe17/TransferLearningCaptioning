import os
import json
import torch
import clip
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from .config import *
from .model_loader import get_model_tokenizer_pair
import urllib.request
import zipfile
from tqdm import tqdm

class TqdmUpTo(tqdm):
    """Progress bar for file downloads"""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def prepare_flickr30k_data(images_dir, captions_file):
    """Download and prepare Flickr30k data if not found"""
    data_dir = os.path.dirname(images_dir)
    os.makedirs(data_dir, exist_ok=True)

    # Prepare Images
    if os.path.exists(images_dir):
        print(f"✅ Flickr30k images found at: {images_dir}")
    else:
        print("📂 Downloading Flickr30k images...")
        
        base_url = "https://github.com/awsaf49/flickr-dataset/releases/download/v1.0/"
        parts = ["flickr30k_part00", "flickr30k_part01", "flickr30k_part02"]
        part_paths = []

        for part in parts:
            part_path = os.path.join(data_dir, part)
            part_paths.append(part_path)
            download_url = base_url + part
            with TqdmUpTo(unit='B', unit_scale=True, unit_divisor=1024, miniters=1, desc=f"Downloading {part}") as t:
                urllib.request.urlretrieve(download_url, part_path, reporthook=t.update_to)

        zip_path = os.path.join(data_dir, "flickr30k.zip")
        print("Concatenating parts...")
        with open(zip_path, 'wb') as f_out:
            for part_path in part_paths:
                with open(part_path, 'rb') as f_in:
                    f_out.write(f_in.read())
                os.remove(part_path)

        print("Unzipping images...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        
        os.remove(zip_path)
        
        original_unzipped_dir = os.path.join(data_dir, 'flickr30k-images')
        if os.path.exists(original_unzipped_dir):
            os.rename(original_unzipped_dir, images_dir)
        print("✅ Images ready")

    # Prepare Captions
    if os.path.exists(captions_file):
        print(f"✅ Captions file found at: {captions_file}")
    else:
        print("📂 Downloading and processing captions...")
        
        captions_zip_path = os.path.join(data_dir, "caption_datasets.zip")
        download_url = "https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip"
        
        with TqdmUpTo(unit='B', unit_scale=True, unit_divisor=1024, miniters=1, desc="Downloading captions") as t:
            urllib.request.urlretrieve(download_url, captions_zip_path, reporthook=t.update_to)
            
        print("Processing captions...")
        with zipfile.ZipFile(captions_zip_path, 'r') as zip_ref:
            with zip_ref.open('dataset_flickr30k.json') as f:
                karpathy_data = json.load(f)

        output_data = []
        for item in karpathy_data['images']:
            for sentence in item['sentences']:
                output_data.append({
                    'image': item['filename'],
                    'caption': sentence['raw']
                })
        
        with open(captions_file, 'w') as f:
            json.dump(output_data, f)
            
        os.remove(captions_zip_path)
        print(f"✅ Captions file created at: {captions_file}")

class SimpleImageCaptionDataset(Dataset):
    """Simple dataset for image-caption pairs"""
    
    def __init__(self, data_pairs, transform=None):
        self.data = data_pairs
        self.transform = transform or self._default_transform()
        # Get tokenizer directly from model-tokenizer pair
        _, self.tokenizer = get_model_tokenizer_pair()
    
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
        
        # Tokenize caption using the tokenizer from model-tokenizer pair
        tokens = self.tokenizer.encode(caption, max_length=MAX_GENERATION_LENGTH, truncation=True, padding='max_length')
        tokens = torch.tensor(tokens, dtype=torch.long)
            
        return {
            'image': image_tensor,
            'input_ids': tokens,
            'caption': caption
        }

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

def collate_fn(batch):
    """Custom collate function to handle None values"""
    batch = [item for item in batch if item is not None]
    if not batch:
        return {}
    
    return {
        'image': torch.stack([item['image'] for item in batch]),
        'input_ids': torch.stack([item['input_ids'] for item in batch]),
        'caption': [item['caption'] for item in batch]
    }

def create_dataloaders(images_dir=None, captions_file=None, batch_size=None, val_split=None, test_split=None, seed=None, max_samples=None):
    """Create train/val/test dataloaders"""
    # Use absolute paths
    images_dir = "data/flickr30k/Images"
    captions_file = "data/flickr30k/captions.json"
    
    batch_size = batch_size or BATCH_SIZE
    val_split = val_split or VAL_SPLIT
    test_split = test_split or TEST_SPLIT
    seed = seed or SEED
    
    # Ensure data is present
    prepare_flickr30k_data(images_dir, captions_file)
    
    data_pairs = load_flickr_data(images_dir, captions_file, max_samples=max_samples)
    dataset = SimpleImageCaptionDataset(data_pairs)
    
    # Calculate splits
    test_size = int(len(dataset) * test_split)
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size - test_size
    
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size], generator=generator
    )
    
    # Create dataloaders with optimized settings
    num_workers = 4
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=SHUFFLE_DATALOADER, 
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True
    )
    
    return train_loader, val_loader, test_loader