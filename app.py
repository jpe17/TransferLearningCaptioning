"""
FastAPI App for Image Captioning
================================
"""

import torch
from PIL import Image
import os
import io
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

# Adjust sys.path to import from src
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from core.inference import initialize_inference_handler
from core.config import DEVICE

# --- App Setup ---
app = FastAPI()
templates = Jinja2Templates(directory="frontend/templates")
app.mount("/static", StaticFiles(directory="frontend/static"), name="static")

# --- Global instances ---
CAPTION_GENERATOR = None
IMAGE_TRANSFORM = None

@app.on_event("startup")
async def startup_event():
    """Load models and necessary components on startup."""
    global CAPTION_GENERATOR, IMAGE_TRANSFORM
    
    CAPTION_GENERATOR, IMAGE_TRANSFORM = initialize_inference_handler()
    if CAPTION_GENERATOR is None:
        print("⚠️ Model initialization failed. The app will run but captioning will fail.")

# --- API Endpoints ---

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/caption", response_class=JSONResponse)
async def caption_image(file: UploadFile = File(...)):
    """
    Takes an image file, preprocesses it, and returns a generated caption.
    """
    if CAPTION_GENERATOR is None or IMAGE_TRANSFORM is None:
        return JSONResponse(
            status_code=500,
            content={"error": "Model not loaded. Please check server logs."}
        )

    try:
        # Read and preprocess the image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_tensor = IMAGE_TRANSFORM(image).unsqueeze(0).to(DEVICE)
        
        # Generate caption
        caption = CAPTION_GENERATOR.generate_caption(
            images=image_tensor,
            max_length=50,
            temperature=0.7,
            do_sample=True
        )
        
        return {"caption": caption}

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"An error occurred: {str(e)}"}
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)