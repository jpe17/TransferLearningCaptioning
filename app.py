import os
import io
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from PIL import Image
import torch
import clip

# Completely disable any compilation that might cause issues
import torch._dynamo
torch._dynamo.config.suppress_errors = True
torch._dynamo.disable()
os.environ["TORCH_COMPILE_DISABLE"] = "1"

from src.core.model_02 import PatchEncoder, PatchDecoder, create_models_efficiently
from src.core.config import *

# --- Simple Config ---
CHECKPOINT_DIR = 'checkpoints/training'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Using device: {DEVICE}")

# --- App Setup ---
app = FastAPI()
templates = Jinja2Templates(directory="frontend/templates")
app.mount("/static", StaticFiles(directory="frontend/static"), name="static")

# --- Global model variables ---
encoder = None
decoder = None
clip_model = None
clip_tokenizer = None

def load_models_simple():
    """Load models with simple, no-optimization approach"""
    global encoder, decoder, clip_model, clip_tokenizer
    
    print("🚀 Loading models (simple mode)...")
    
    try:
        # First try to find a valid checkpoint and load its config
        checkpoint_found = False
        checkpoint_config = None
        
        # First try the specific checkpoint path
        checkpoint_paths = [
            os.path.join("checkpoints", "sweeps", "sweep_20250701_163357", "model_checkpoint.pth"),
            os.path.join(CHECKPOINT_DIR, "latest_checkpoint.pth")
        ]
        
        # Try each path until we find a valid checkpoint
        for checkpoint_path in checkpoint_paths:
            print(f"🔍 Looking for checkpoint at: {checkpoint_path}")
            if os.path.exists(checkpoint_path):
                print(f"📂 Found checkpoint: {checkpoint_path}")
                
                # Try to load config from same directory
                config_path = os.path.join(os.path.dirname(checkpoint_path), "config.json")
                if os.path.exists(config_path):
                    import json
                    with open(config_path, 'r') as f:
                        checkpoint_config = json.load(f)
                    print(f"📋 Loaded config from: {config_path}")
                    print(f"Config encoder_output_dim: {checkpoint_config.get('encoder_output_dim', 'not found')}")
                
                checkpoint_found = True
                break
        
        # Create models with checkpoint-specific config if available
        if checkpoint_config:
            encoder_hidden_dim = checkpoint_config.get('encoder_hidden_dim', ENCODER_HIDDEN_DIM)
            encoder_output_dim = checkpoint_config.get('encoder_output_dim', ENCODER_OUTPUT_DIM)
            clip_model_name = checkpoint_config.get('clip_model', CLIP_MODEL)
            base_model_name = checkpoint_config.get('base_model_name', BASE_MODEL_NAME)
            
            print(f"Using checkpoint config: hidden_dim={encoder_hidden_dim}, output_dim={encoder_output_dim}")
            
            encoder, decoder = create_models_efficiently(
                encoder_hidden_dim=encoder_hidden_dim,
                encoder_output_dim=encoder_output_dim,
                base_model_name=base_model_name,
                clip_model_name=clip_model_name
            )
        else:
            # Fallback to default config
            print("No checkpoint config found, using default config")
            encoder, decoder = create_models_efficiently()
        
        print(f"Models created on device: {DEVICE}")
        
        # Get CLIP model for preprocessing and tokenizer
        clip_model, _ = clip.load(CLIP_MODEL, device=DEVICE)
        print(f"CLIP loaded on device: {DEVICE}")
        
        # Get the CLIP tokenizer for decoding
        clip_tokenizer = clip.simple_tokenizer.SimpleTokenizer()
        print("CLIP tokenizer initialized")
        
        # Now load the checkpoint if found
        if checkpoint_found:
            for checkpoint_path in checkpoint_paths:
                if os.path.exists(checkpoint_path):
                    print(f"📂 Loading checkpoint: {checkpoint_path}")
                    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
                    encoder.load_state_dict(checkpoint['encoder_state_dict'])
                    decoder.load_state_dict(checkpoint['decoder_state_dict'])
                    print("✅ Checkpoint loaded!")
                    break
        else:
            print("⚠️ No checkpoint found, using untrained model")
        
        # Set to eval mode
        encoder.eval()
        decoder.eval()
        
        print("✅ Models loaded successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        import traceback
        traceback.print_exc()
        return False

def preprocess_image_simple(image):
    """Simple image preprocessing"""
    try:
        from torchvision import transforms
        
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        
        tensor = preprocess(image).unsqueeze(0).to(DEVICE)
        print(f"Image tensor shape: {tensor.shape}, device: {tensor.device}")
        return tensor
        
    except Exception as e:
        print(f"Error in preprocessing: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

def decode_tokens_simple(tokens):
    """Convert tokens to text using CLIP's tokenizer"""
    global clip_tokenizer
    
    try:
        # Convert to CPU and numpy for processing
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.cpu().numpy()
        
        # Remove special tokens and padding
        filtered_tokens = []
        for token in tokens:
            if token not in [START_TOKEN_ID, END_TOKEN_ID, 0]:
                filtered_tokens.append(token)
        
        if not filtered_tokens:
            return "No valid tokens to decode"
        
        # Decode using CLIP tokenizer
        if clip_tokenizer:
            try:
                text = clip_tokenizer.decode(filtered_tokens)
                return text.strip()
            except Exception as e:
                print(f"Tokenizer decode error: {e}")
                return f"Decode error: {str(e)}"
        else:
            return "Tokenizer not available"
            
    except Exception as e:
        print(f"Error in decode_tokens_simple: {e}")
        return f"Token decoding failed: {str(e)}"

def debug_generate(decoder, patch_features, max_length=20):
    """Debug version of generate to see what's happening step by step"""
    decoder.eval()
    batch_size = patch_features.size(0)
    device = patch_features.device
    
    # Start with start token
    generated = torch.full((batch_size, 1), START_TOKEN_ID, device=device, dtype=torch.long)
    
    print(f"Starting generation with token: {START_TOKEN_ID}")
    
    with torch.no_grad():
        for step in range(max_length - 1):
            print(f"\n--- Step {step + 1} ---")
            print(f"Current sequence length: {generated.shape[1]}")
            print(f"Current sequence: {generated[0].tolist()}")
            
            # Get logits for current sequence
            try:
                logits = decoder.forward(patch_features, generated)
                print(f"Logits shape: {logits.shape}")
                
                # Get next token (take last position)
                next_token_logits = logits[:, -1, :]
                print(f"Next token logits shape: {next_token_logits.shape}")
                
                # Show top 5 predictions
                top5_logits, top5_indices = torch.topk(next_token_logits[0], 5)
                print(f"Top 5 predictions: {top5_indices.tolist()} with logits: {top5_logits.tolist()}")
                
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                print(f"Selected token: {next_token[0].item()}")
                
                # Check for repetition
                if generated.shape[1] >= 3:
                    last_3 = generated[0, -3:].tolist()
                    if len(set(last_3)) <= 1:  # All same or mostly same
                        print(f"⚠️ Detected repetition in last 3 tokens: {last_3}")
                
                # Append to sequence
                generated = torch.cat([generated, next_token], dim=1)
                
                # Stop if end token
                if (next_token == END_TOKEN_ID).all():
                    print(f"✅ Found end token, stopping")
                    break
                    
                # Stop if stuck in repetition
                if step > 3 and generated.shape[1] >= 5:
                    last_5 = generated[0, -5:].tolist()
                    if len(set(last_5)) <= 2:  # Too repetitive
                        print(f"🛑 Stopping due to repetition: {last_5}")
                        break
                        
            except Exception as e:
                print(f"❌ Error in step {step}: {e}")
                break
    
    print(f"\nFinal sequence: {generated[0].tolist()}")
    return generated

def generate_caption_simple(image_tensor):
    """Simple caption generation"""
    global encoder, decoder
    
    try:
        if encoder is None or decoder is None:
            return "Models not loaded"
        
        print("Starting caption generation...")
        
        # Ensure image tensor is on the same device as the model
        model_device = next(encoder.parameters()).device
        image_tensor = image_tensor.to(model_device)
        print(f"Moved image tensor to device: {model_device}")
        
        with torch.no_grad():
            # Encode image to patch features
            print("Encoding image...")
            patch_features = encoder(image_tensor)
            print(f"Patch features shape: {patch_features.shape}")
            
            # Generate caption tokens using debug version
            print("Generating tokens with debugging...")
            generated_tokens = debug_generate(decoder, patch_features, max_length=20)
            print(f"Generated tokens shape: {generated_tokens.shape}")
            print(f"Generated tokens: {generated_tokens}")
            
            # Convert tokens to text
            if isinstance(generated_tokens, torch.Tensor) and generated_tokens.numel() > 0:
                caption = decode_tokens_simple(generated_tokens[0])  # Take first batch item
                print(f"Decoded caption: '{caption}'")
                
                if not caption or caption.strip() == "":
                    caption = "Generated empty caption"
                
                return caption
            else:
                return "No tokens generated"
        
    except Exception as e:
        print(f"Error in caption generation: {str(e)}")
        import traceback
        traceback.print_exc()
        return f"Caption generation failed: {str(e)}"

# Load models on startup
@app.on_event("startup")
async def startup_event():
    success = load_models_simple()
    if not success:
        print("⚠️ Model loading failed - app will return error messages")

# --- Routes ---
@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/caption")
async def caption_image(file: UploadFile = File(...)):
    try:
        # Load and preprocess image
        image = Image.open(io.BytesIO(await file.read())).convert('RGB')
        image_tensor = preprocess_image_simple(image)
        
        # Generate caption
        caption = generate_caption_simple(image_tensor)
        
        return {"caption": caption, "status": "success"}
        
    except Exception as e:
        print(f"❌ Caption error: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=400, 
            content={"error": str(e), "status": "error"}
        )

@app.get("/health")
def health_check():
    """Simple health check"""
    global encoder, decoder, clip_model, clip_tokenizer
    return {
        "status": "healthy",
        "device": str(DEVICE),
        "models_loaded": encoder is not None and decoder is not None,
        "clip_model_loaded": clip_model is not None,
        "tokenizer_loaded": clip_tokenizer is not None,
        "torch_compile_disabled": True
    }

@app.get("/test")
def test_endpoint():
    """Simple test endpoint"""
    return {"message": "App is working!", "device": str(DEVICE)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)