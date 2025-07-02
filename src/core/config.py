import torch
import os

# --- Device Configuration ---
if torch.cuda.is_available():
    DEVICE = "cuda"
    MODEL_DTYPE = torch.float32
elif torch.backends.mps.is_available():
    DEVICE = "mps" 
    MODEL_DTYPE = torch.float32  # Force float32 on MPS for compatibility
else:
    DEVICE = "cpu"
    MODEL_DTYPE = torch.float32

# --- Data Paths ---
IMAGES_DIR = "/workspace/TransferLearningCaptioning/data/flickr30k/images"
CAPTIONS_FILE = "/workspace/TransferLearningCaptioning/data/flickr30k/captions.json"

# --- Core Training Configuration ---
CLIP_MODEL = 'ViT-B/16'
BATCH_SIZE = 16
EPOCHS = 1
LEARNING_RATE = 0.001
SHUFFLE_DATALOADER = True
SEED = 42

# --- Training Hyperparameters ---
GRAD_CLIP_NORM = 1.0
LABEL_SMOOTHING = 0.1
WARMUP_STEPS_RATIO = 0.1
MAX_STEPS_PER_SWEEP = 500

# --- Checkpoint Configuration ---
CHECKPOINTS_BASE_DIR = "checkpoints"
RESUME_TRAINING = True
RESUME_FROM_LATEST = True

# --- Data Processing ---
IMAGE_SIZE = (224, 224)
IMAGE_NORMALIZE_MEAN = (0.485, 0.456, 0.406)
IMAGE_NORMALIZE_STD = (0.229, 0.224, 0.225)
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1

# --- Model Architecture ---
ENCODER_HIDDEN_DIM = 1024
ENCODER_OUTPUT_DIM = 256
NUM_PATCHES = 196  # ViT-B/16: 14x14 patches
CLIP_DIM = 768     # ViT-B/16 dimension

BASE_MODEL_NAME = "Qwen/Qwen3-0.6B-Base"
PATCH_FEATURE_DIM = 256
MAX_GENERATION_LENGTH = 50

# --- Model Training Strategy ---
FREEZE_BASE_MODEL = True
FREEZE_CLIP = True