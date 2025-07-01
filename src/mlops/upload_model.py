#!/usr/bin/env python3
"""
Upload Best Trained Model to Storage
Finds your best model and uploads it for production use
"""

import os
import sys
from .model_storage import ModelStorage, find_best_checkpoint, get_model_hash, STORAGE_OPTIONS

def main():
    print("🚀 Model Upload Tool")
    print("=" * 50)
    
    # Step 1: Find best checkpoint
    print("📊 Finding your best trained model...")
    best_checkpoint, best_loss = find_best_checkpoint()
    
    if not best_checkpoint:
        print("❌ No trained models found!")
        print("🔧 Train models first: python -m backend.run_complete_sweep")
        return
    
    print(f"✅ Best model found: {best_checkpoint}")
    print(f"📉 Validation loss: {best_loss:.4f}")
    
    # Get file size
    file_size_gb = os.path.getsize(best_checkpoint) / (1024**3)
    print(f"📦 Model size: {file_size_gb:.2f} GB")
    
    # Step 2: Choose storage option
    print("\n🗄️  Choose storage option:")
    for i, (key, info) in enumerate(STORAGE_OPTIONS.items(), 1):
        recommended = "⭐ RECOMMENDED" if info["recommended"] else ""
        print(f"  {i}. {key.upper()} {recommended}")
        print(f"     Cost: {info['cost']} | Limit: {info['size_limit']} | Setup: {info['setup']}")
    
    while True:
        try:
            choice = int(input(f"\nEnter choice (1-{len(STORAGE_OPTIONS)}): "))
            if 1 <= choice <= len(STORAGE_OPTIONS):
                storage_type = list(STORAGE_OPTIONS.keys())[choice-1]
                break
            else:
                print("Invalid choice!")
        except ValueError:
            print("Please enter a number!")
    
    # Step 3: Setup storage
    print(f"\n📤 Setting up {storage_type.upper()} storage...")
    
    if storage_type == "huggingface":
        setup_huggingface()
    elif storage_type == "aws_s3":
        setup_aws_s3()
    
    # Step 4: Upload model
    model_name = input("\nEnter model name (e.g., 'my-caption-model-v1'): ").strip()
    if not model_name:
        model_name = "caption-model-v1"
    
    storage = ModelStorage(storage_type)
    
    print(f"\n🚀 Uploading {file_size_gb:.2f}GB model to {storage_type}...")
    url = storage.upload_model(best_checkpoint, model_name)
    
    if url:
        print(f"\n✅ Upload successful!")
        print(f"🔗 Model URL: {url}")
        
        # Generate download script for production
        create_download_script(storage_type, model_name)
        
        # Update app.py configuration
        update_app_config(storage_type, model_name, url)
        
    else:
        print("❌ Upload failed!")

def setup_huggingface():
    """Setup Hugging Face Hub (RECOMMENDED)"""
    print("\n🤗 Hugging Face Setup:")
    print("1. Go to: https://huggingface.co/settings/tokens")
    print("2. Create a token with 'Write' permissions")
    print("3. Set environment variable: export HF_TOKEN=your_token")
    
    token = os.getenv('HF_TOKEN')
    if not token:
        token = input("Or enter your HF token now: ").strip()
        if token:
            os.environ['HF_TOKEN'] = token
    
    if token:
        print("✅ Hugging Face token configured")
    else:
        print("❌ No HF token provided - upload will fail")

def setup_aws_s3():
    """Setup AWS S3"""
    print("\n☁️  AWS S3 Setup:")
    print("1. Install: pip install boto3")
    print("2. Configure: aws configure")
    print("3. Set bucket: export AWS_S3_BUCKET=your-bucket-name")

def create_download_script(storage_type, model_name):
    """Create script to download model in production"""
    script_content = f'''#!/usr/bin/env python3
"""
Download production model
"""
from backend.model_storage import ModelStorage

storage = ModelStorage("{storage_type}")
storage.download_model("{model_name}", "production_model.pth")
print("✅ Production model downloaded!")
'''
    
    with open("download_production_model.py", "w") as f:
        f.write(script_content)
    
    print(f"📝 Created: download_production_model.py")

def update_app_config(storage_type, model_name, url):
    """Update app.py with model download configuration"""
    config_content = f'''
# Production Model Configuration
PRODUCTION_MODEL_STORAGE = "{storage_type}"
PRODUCTION_MODEL_NAME = "{model_name}"
PRODUCTION_MODEL_URL = "{url}"
PRODUCTION_MODEL_PATH = "production_model.pth"
'''
    
    with open("production_model_config.py", "w") as f:
        f.write(config_content)
    
    print(f"📝 Created: production_model_config.py")
    print(f"🔧 Add this to your deployment environment")

if __name__ == "__main__":
    main() 