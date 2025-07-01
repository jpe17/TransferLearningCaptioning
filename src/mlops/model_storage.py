#!/usr/bin/env python3
"""
Model Storage Solutions for Large Models (4GB+)
Handles uploading, downloading, and managing model checkpoints
"""

import os
import hashlib
import boto3
from google.cloud import storage as gcs
import requests
from pathlib import Path
import torch
from tqdm import tqdm

class ModelStorage:
    """Unified interface for storing large model files"""
    
    def __init__(self, storage_type="huggingface"):
        self.storage_type = storage_type
        self.setup_client()
    
    def setup_client(self):
        """Setup storage client based on type"""
        if self.storage_type == "aws_s3":
            self.s3_client = boto3.client('s3')
            self.bucket_name = os.getenv('AWS_S3_BUCKET', 'your-model-bucket')
        elif self.storage_type == "gcp_storage":
            self.gcs_client = gcs.Client()
            self.bucket_name = os.getenv('GCP_STORAGE_BUCKET', 'your-model-bucket')
    
    def upload_model(self, local_path, model_name):
        """Upload model to chosen storage"""
        print(f"📤 Uploading {local_path} as {model_name}...")
        
        if self.storage_type == "huggingface":
            return self._upload_to_huggingface(local_path, model_name)
        elif self.storage_type == "aws_s3":
            return self._upload_to_s3(local_path, model_name)
        elif self.storage_type == "gcp_storage":
            return self._upload_to_gcs(local_path, model_name)
        elif self.storage_type == "github_releases":
            return self._upload_to_github_releases(local_path, model_name)
    
    def download_model(self, model_name, local_path):
        """Download model from storage"""
        print(f"📥 Downloading {model_name} to {local_path}...")
        
        if self.storage_type == "huggingface":
            return self._download_from_huggingface(model_name, local_path)
        elif self.storage_type == "aws_s3":
            return self._download_from_s3(model_name, local_path)
        elif self.storage_type == "gcp_storage":
            return self._download_from_gcs(model_name, local_path)
    
    def _upload_to_huggingface(self, local_path, model_name):
        """Upload to Hugging Face Hub (RECOMMENDED - Free & Easy)"""
        try:
            from huggingface_hub import HfApi, upload_file
            
            api = HfApi()
            repo_id = f"your-username/{model_name}"  # Change this
            
            # Create repo if it doesn't exist
            try:
                api.create_repo(repo_id, exist_ok=True, private=True)
            except:
                pass
            
            # Upload file
            url = upload_file(
                path_or_fileobj=local_path,
                path_in_repo="model_checkpoint.pth",
                repo_id=repo_id,
                token=os.getenv('HF_TOKEN')  # Set your HF token
            )
            
            print(f"✅ Uploaded to: https://huggingface.co/{repo_id}")
            return url
            
        except ImportError:
            print("❌ Install: pip install huggingface_hub")
            return None
    
    def _download_from_huggingface(self, model_name, local_path):
        """Download from Hugging Face Hub"""
        try:
            from huggingface_hub import hf_hub_download
            
            repo_id = f"your-username/{model_name}"
            
            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename="model_checkpoint.pth",
                cache_dir=str(Path(local_path).parent),
                token=os.getenv('HF_TOKEN')
            )
            
            # Move to desired location
            os.rename(downloaded_path, local_path)
            print(f"✅ Downloaded from Hugging Face")
            return True
            
        except Exception as e:
            print(f"❌ Download failed: {e}")
            return False
    
    def _upload_to_s3(self, local_path, model_name):
        """Upload to AWS S3"""
        try:
            key = f"models/{model_name}/model_checkpoint.pth"
            
            # Upload with progress bar
            file_size = os.path.getsize(local_path)
            with tqdm(total=file_size, unit='B', unit_scale=True, desc="Uploading") as pbar:
                def upload_callback(bytes_transferred):
                    pbar.update(bytes_transferred)
                
                self.s3_client.upload_file(
                    local_path, self.bucket_name, key,
                    Callback=upload_callback
                )
            
            url = f"s3://{self.bucket_name}/{key}"
            print(f"✅ Uploaded to: {url}")
            return url
            
        except Exception as e:
            print(f"❌ S3 upload failed: {e}")
            return None

def get_model_hash(file_path):
    """Get SHA256 hash of model file for verification"""
    hash_sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()

def find_best_checkpoint(checkpoints_dir="checkpoints"):
    """Find the best checkpoint based on validation loss"""
    import glob
    import json
    
    best_loss = float('inf')
    best_checkpoint = None
    
    # Look through all training runs
    for run_dir in glob.glob(f"{checkpoints_dir}/training/run_*"):
        config_path = os.path.join(run_dir, "config.json")
        checkpoint_path = os.path.join(run_dir, "model_checkpoint.pth")
        
        if os.path.exists(checkpoint_path):
            try:
                # Load checkpoint to get loss
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                val_loss = checkpoint.get('val_loss', float('inf'))
                
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_checkpoint = checkpoint_path
                    
            except Exception as e:
                print(f"❌ Error reading {checkpoint_path}: {e}")
    
    return best_checkpoint, best_loss

# Storage options with pros/cons
STORAGE_OPTIONS = {
    "huggingface": {
        "cost": "Free",
        "size_limit": "50GB per repo",
        "speed": "Fast",
        "setup": "Easy (just HF token)",
        "recommended": True
    },
    "aws_s3": {
        "cost": "$0.023/GB/month",
        "size_limit": "Unlimited", 
        "speed": "Very Fast",
        "setup": "Medium (AWS account + credentials)",
        "recommended": False
    },
    "gcp_storage": {
        "cost": "$0.020/GB/month",
        "size_limit": "Unlimited",
        "speed": "Very Fast", 
        "setup": "Medium (GCP account + credentials)",
        "recommended": False
    },
    "github_releases": {
        "cost": "Free",
        "size_limit": "2GB per file",
        "speed": "Medium",
        "setup": "Easy",
        "recommended": False  # Too small for 4GB models
    }
} 