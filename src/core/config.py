import torch
import os

# --- Device Configuration ---
# Force consistent dtypes on MPS to avoid dtype mismatch errors
if torch.cuda.is_available():
    DEVICE = "cuda"
    MODEL_DTYPE = torch.float16  # Use half precision on CUDA
elif torch.backends.mps.is_available():
    DEVICE = "mps" 
    MODEL_DTYPE = torch.float32  # Force float32 on MPS - ALL tensors must match!
else:
    DEVICE = "cpu"
    MODEL_DTYPE = torch.float32  # Use float32 on CPU

print(f"🔧 Using device: {DEVICE} with dtype: {MODEL_DTYPE}")

# --- Data Paths ---
# Adjust these paths to where your Flickr30k dataset is stored
IMAGES_DIR = "data/flickr30k/images"
CAPTIONS_FILE = "data/flickr30k/captions.json"

# --- Model & Training Configurations ---
CLIP_MODEL = 'ViT-L/14'  # Configurable - ViT-B/16 (faster) vs ViT-L/14 (better)
BATCH_SIZE = 32          # Configurable - major impact
EPOCHS = 5               # Configurable - different architectures need different training
LEARNING_RATE = 1e-4     # Configurable - major impact
SHUFFLE_DATALOADER = True
SEED = 42

# --- Decoder Specific Configurations (Legacy - not used in current model) ---
DECODER_LAYERS = 4
DECODER_HEADS = 4
DECODER_FFN_RATIO = 4
DECODER_DROPOUT = 0.1

# --- Training Hyperparameters ---
GRAD_CLIP_NORM = 1.0     # Configurable - affects training stability
LABEL_SMOOTHING = 0.1    # Configurable - affects overfitting and quality
WARMUP_STEPS_RATIO = 0.05  # Configurable - interacts with learning rate

# --- Sweep Configuration ---
MAX_STEPS_PER_SWEEP = 200

# --- Checkpoint & Output Configuration ---
CHECKPOINTS_BASE_DIR = "checkpoints"
RESUME_TRAINING = True
RESUME_FROM_LATEST = True

# --- Data Processing Configuration (Fixed - standard values) ---
IMAGE_SIZE = (224, 224)  # Fixed - standard for CLIP
IMAGE_NORMALIZE_MEAN = (0.485, 0.456, 0.406)  # Fixed - ImageNet standard
IMAGE_NORMALIZE_STD = (0.229, 0.224, 0.225)   # Fixed - ImageNet standard
VAL_SPLIT = 0.1          # Fixed - standard split
TEST_SPLIT = 0.1         # Fixed - standard split

# --- Model Architecture Configuration ---
# PatchEncoder parameters - CONFIGURABLE (major impact)
ENCODER_HIDDEN_DIM = 2048    # Configurable - affects model capacity
ENCODER_OUTPUT_DIM = 512     # Configurable - affects model capacity
NUM_PATCHES = 256           # Fixed - determined by ViT architecture
CLIP_DIM = 768             # Fixed - determined by ViT-L/14 (changes with model)

# PatchDecoder parameters - FIXED (proven values)
BASE_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"  # Fixed - fast and effective
CLIP_VOCAB_SIZE = 49408                        # Fixed - determined by CLIP
CLIP_TEXT_DIM = 512                           # Fixed - determined by CLIP
PATCH_FEATURE_DIM = 512                       # Fixed - matches encoder output

# Generation parameters - FIXED (standard values)
MAX_GENERATION_LENGTH = 77   # Fixed - standard for captions
START_TOKEN_ID = 49406      # Fixed - CLIP standard
END_TOKEN_ID = 49407        # Fixed - CLIP standard

# --- Model Training Configuration ---
FREEZE_BASE_MODEL = True    # Configurable - major training strategy decision
FREEZE_CLIP = True          # Fixed - always freeze for stability