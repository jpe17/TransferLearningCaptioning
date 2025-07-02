#!/usr/bin/env python3
"""
Test script to verify inference is working correctly
"""

import torch
from PIL import Image
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.core.inference import initialize_inference_handler

def test_inference():
    print("🧪 Testing inference...")
    
    # Initialize inference
    caption_generator, image_transform = initialize_inference_handler()
    
    if caption_generator is None:
        print("❌ Failed to initialize inference handler")
        return False
    
    # Create a dummy image (or use a real one if available)
    try:
        # Try to find an image in data directory
        test_image_path = "data/flickr30k/Images"
        if os.path.exists(test_image_path):
            images = os.listdir(test_image_path)
            if images:
                image_path = os.path.join(test_image_path, images[0])
                image = Image.open(image_path).convert("RGB")
                print(f"📷 Using test image: {image_path}")
            else:
                raise FileNotFoundError("No images found")
        else:
            raise FileNotFoundError("No data directory found")
    except Exception as e:
        print(f"⚠️ Could not load real image ({e}), creating dummy image")
        # Create a dummy RGB image
        import numpy as np
        dummy_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        image = Image.fromarray(dummy_array)
    
    # Transform image
    image_tensor = image_transform(image).unsqueeze(0)  # Add batch dimension
    
    print(f"🔍 Image tensor shape: {image_tensor.shape}")
    
    # Generate caption
    try:
        caption = caption_generator.generate_caption(image_tensor, max_length=20, temperature=0.7)
        print(f"✅ Generated caption: '{caption}'")
        
        if caption and len(caption.strip()) > 0:
            print("🎉 SUCCESS: Non-empty caption generated!")
            return True
        else:
            print("❌ FAILURE: Empty caption generated")
            return False
            
    except Exception as e:
        print(f"❌ Error during caption generation: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_inference()
    if success:
        print("\n🎉 All tests passed!")
    else:
        print("\n❌ Tests failed - check the issues above") 