"""
Core ML Functionality
=====================

This package contains the essential machine learning components:
- Model architectures (model_01.py, model_02.py)
- Training logic (train.py)
- Data processing utilities (data_processing.py)
- Configuration management (config.py)

These are the critical components that define what your ML system does.
"""

from .config import *
from .data_processing import create_dataloaders, SimpleImageCaptionDataset
from .train import train

# Model imports - conditional to avoid import errors if dependencies aren't met
try:
    from .model_01 import Encoder as EncoderV1, Decoder as DecoderV1
    from .model_02 import PatchEncoder, PatchDecoder, create_models_efficiently
except ImportError as e:
    print(f"Warning: Could not import some models: {e}") 