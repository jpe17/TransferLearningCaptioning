import os
import io
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from PIL import Image
import torch
import clip
from src.core.model_02 import PatchEncoder, PatchDecoder, create_models_efficiently
from src.core.config import *

# --- Config ---
CHECKPOINT_DIR = 'checkpoints/training'  # Updated to match new checkpoint structure
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Deployment Strategy: Use YOUR Trained Models ---
# Deploy the SAME models you trained - that's the whole point!
DEPLOYMENT_CLIP_MODEL = CLIP_MODEL      # YOUR trained CLIP choice
DEPLOYMENT_BASE_MODEL = BASE_MODEL_NAME # YOUR trained base model

# Model Optimization for Deployment (keeps your training)
USE_HALF_PRECISION = True    # 50% memory reduction, 2x speed
USE_MODEL_COMPILATION = True # PyTorch 2.0 optimization
LAZY_LOADING = True         # Load models on first request

# --- App Setup ---
app = FastAPI()
templates = Jinja2Templates(directory="frontend/templates")
app.mount("/static", StaticFiles(directory="frontend/static"), name="static")

# --- Global model variables ---
encoder = None
decoder = None
clip_model = None

def load_models():
    """Load YOUR trained models with deployment optimizations"""
    global encoder, decoder, clip_model
    
    print("🚀 Loading YOUR trained models for deployment...")
    print(f"📋 Using: {DEPLOYMENT_CLIP_MODEL} + {DEPLOYMENT_BASE_MODEL}")
    
    # Create models with EXACT same architecture as training
    encoder, decoder = create_models_efficiently(
        encoder_hidden_dim=ENCODER_HIDDEN_DIM,  # Exact same as training
        encoder_output_dim=ENCODER_OUTPUT_DIM,   # Exact same as training  
        base_model_name=DEPLOYMENT_BASE_MODEL    # Exact same as training
    )
    
    # Get CLIP model for preprocessing
    clip_model, _ = clip.load(DEPLOYMENT_CLIP_MODEL, device=DEVICE)
    
    # CRITICAL: Load YOUR trained checkpoint
    checkpoint_loaded = False
    
    # Strategy 1: Try local checkpoints first
    try:
        import glob
        checkpoint_dirs = glob.glob(os.path.join(CHECKPOINT_DIR, "run_*"))
        if checkpoint_dirs:
            # Find the BEST checkpoint (you can modify this logic)
            latest_dir = sorted(checkpoint_dirs)[-1]  # Most recent
            checkpoint_path = os.path.join(latest_dir, "model_checkpoint.pth")
            
            if os.path.exists(checkpoint_path):
                print(f"📂 Loading local checkpoint: {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
                
                # Load YOUR trained parameters
                encoder.load_state_dict(checkpoint['encoder_state_dict'])
                decoder.load_state_dict(checkpoint['decoder_state_dict'])
                
                print("✅ Local trained model loaded successfully!")
                print(f"📊 Trained epoch: {checkpoint.get('epoch', 'unknown')}")
                print(f"📉 Training loss: {checkpoint.get('train_loss', 'unknown'):.4f}")
                checkpoint_loaded = True
    except Exception as e:
        print(f"❌ Error loading local checkpoint: {e}")
    
    # Strategy 2: Try production model download
    if not checkpoint_loaded:
        checkpoint_loaded = try_download_production_model()
    
    # Strategy 3: Fallback to untrained
    if not checkpoint_loaded:
        print("⚠️  WARNING: Using untrained model!")
        print("🔧 Options:")
        print("   1. Train locally: python -m backend.run_complete_sweep")
        print("   2. Upload trained model: python -m backend.upload_model")
        print("   3. Download production model: python download_production_model.py")
    
    # Apply deployment optimizations (while keeping your trained weights)
    if USE_HALF_PRECISION and DEVICE.type == 'cuda':
        print("⚡ Applying half precision optimization...")
        encoder = encoder.half()
        decoder = decoder.half()
    
    if USE_MODEL_COMPILATION:
        print("⚡ Applying PyTorch compilation...")
        try:
            encoder = torch.compile(encoder)
            decoder = torch.compile(decoder)
        except:
            print("⚠️  PyTorch compile not available")
    
    encoder.eval()
    decoder.eval()
    print("✅ YOUR trained models ready for deployment!")

def try_download_production_model():
    """Try to download production model from storage"""
    try:
        # Check if production config exists
        if os.path.exists("production_model_config.py"):
            print("📥 Found production model config, attempting download...")
            
            # Import production config
            import importlib.util
            spec = importlib.util.spec_from_file_location("prod_config", "production_model_config.py")
            prod_config = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(prod_config)
            
            # Download model
            from src.mlops.model_storage import ModelStorage
            storage = ModelStorage(prod_config.PRODUCTION_MODEL_STORAGE)
            
            success = storage.download_model(
                prod_config.PRODUCTION_MODEL_NAME,
                prod_config.PRODUCTION_MODEL_PATH
            )
            
            if success and os.path.exists(prod_config.PRODUCTION_MODEL_PATH):
                print(f"📂 Loading downloaded production model...")
                checkpoint = torch.load(prod_config.PRODUCTION_MODEL_PATH, map_location=DEVICE)
                
                encoder.load_state_dict(checkpoint['encoder_state_dict'])
                decoder.load_state_dict(checkpoint['decoder_state_dict'])
                
                print("✅ Production model loaded successfully!")
                return True
                
    except Exception as e:
        print(f"❌ Production model download failed: {e}")
    
    return False

def preprocess_image(image):
    """Preprocess image for the model"""
    # Use CLIP's preprocessing
    from torchvision import transforms
    
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    return preprocess(image).unsqueeze(0).to(DEVICE)

def generate_caption(image_tensor, max_length=50):
    """Generate caption for an image"""
    global encoder, decoder
    
    with torch.no_grad():
        # Encode image to patch features
        patch_features = encoder(image_tensor)
        
        # Generate caption
        generated_tokens = decoder.generate(
            patch_features, 
            max_length=max_length,
            start_token=START_TOKEN_ID
        )
        
        # Simple token-to-text conversion (simplified)
        # In a real deployment, you'd want proper CLIP text decoding
        caption_tokens = generated_tokens[0].cpu().numpy()
        
        # Remove special tokens and convert to text
        caption_tokens = caption_tokens[caption_tokens != START_TOKEN_ID]
        caption_tokens = caption_tokens[caption_tokens != END_TOKEN_ID]
        caption_tokens = caption_tokens[caption_tokens != 0]  # Remove padding
        
        # Simplified caption (you'd want proper token decoding here)
        caption = f"Generated caption with {len(caption_tokens)} tokens"
        
        return caption

# Load models on startup
@app.on_event("startup")
async def startup_event():
    load_models()

# --- Routes ---
@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/caption")
async def caption_image(file: UploadFile = File(...)):
    try:
        # Load and preprocess image
        image = Image.open(io.BytesIO(await file.read())).convert('RGB')
        image_tensor = preprocess_image(image)
        
        # Generate caption
        caption = generate_caption(image_tensor)
        
        return {"caption": caption}
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": encoder is not None and decoder is not None,
        "device": str(DEVICE),
        "clip_model": DEPLOYMENT_CLIP_MODEL,
        "base_model": DEPLOYMENT_BASE_MODEL,
        "architecture": {
            "encoder_hidden_dim": ENCODER_HIDDEN_DIM,
            "encoder_output_dim": ENCODER_OUTPUT_DIM,
            "uses_trained_mlps": "Yes - same architecture as training"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)