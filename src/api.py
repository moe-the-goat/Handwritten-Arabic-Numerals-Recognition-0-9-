#!/usr/bin/env python3
# api.py -- FastAPI backend serving the trained CNN model for live digit prediction.
#
# Endpoints:
#   POST /predict  - accept a canvas drawing, return top prediction + probabilities
#   GET  /health   - health check
#   GET  /samples  - one real training image per digit for reference
#   GET  /         - serve the frontend UI

import os
import sys
import io
import json
import base64
import numpy as np
from PIL import Image, ImageFilter

import torch
import torch.nn.functional as F
import torchvision.transforms as T

# Add src/ to path so we can import our project modules
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC_DIR)

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import OUTPUT_DIR, MODEL_DIR, NUM_CLASSES, CLASS_NAMES, IMAGE_SIZE
from models import ArabicDigitCNN

# Globals
app = FastAPI(title="Arabic Numeral Recognition API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained model and normalization stats at startup
DEVICE = torch.device("cpu")
MODEL = None
NORM_MEAN = None
NORM_STD = None
TRANSFORM = None


def load_model():
    global MODEL, NORM_MEAN, NORM_STD, TRANSFORM

    # Normalization stats
    norm_path = os.path.join(OUTPUT_DIR, "norm_stats.json")
    if not os.path.exists(norm_path):
        raise RuntimeError(f"Normalization stats not found: {norm_path}")
    with open(norm_path) as f:
        ns = json.load(f)
    NORM_MEAN = ns["mean"]
    NORM_STD = ns["std"]

    # Same transform used during evaluation (resize + normalize)
    TRANSFORM = T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[NORM_MEAN], std=[NORM_STD]),
    ])

    # Load trained CNN
    model_path = os.path.join(MODEL_DIR, "cnn_aug_best.pth")
    if not os.path.exists(model_path):
        raise RuntimeError(f"Trained model not found: {model_path}")

    MODEL = ArabicDigitCNN()
    state = torch.load(model_path, map_location=DEVICE, weights_only=True)
    MODEL.load_state_dict(state)
    MODEL.to(DEVICE)
    MODEL.eval()
    print(f"[API] Model loaded from {model_path}")
    print(f"[API] Normalization: mean={NORM_MEAN:.4f}, std={NORM_STD:.4f}")


@app.on_event("startup")
async def startup():
    load_model()


# Request / Response schemas

class PredictRequest(BaseModel):
    image: str  # base64-encoded PNG from the canvas


class PredictResponse(BaseModel):
    predicted_digit: int
    confidence: float
    probabilities: dict  # {"0": 0.01, "1": 0.02, ... "9": 0.95}
    processed_image: str  # base64 28x28 preview the model actually sees


# Image preprocessing

def preprocess_canvas_image(base64_str: str):
    """
    Turn a canvas drawing into a 28x28 tensor the model can consume.
    The canvas stroke is gray (not black) to match the training data's
    ink intensity range, and we apply a light blur for images from large
    canvases to mimic the antialiased pen strokes in the training set.
    """
    # Strip the data-URI prefix if present
    if "," in base64_str:
        base64_str = base64_str.split(",", 1)[1]
    img_bytes = base64.b64decode(base64_str)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGBA")

    # White background composite (canvas may have transparent pixels)
    bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
    bg.paste(img, mask=img)
    img_gray = bg.convert("L")  # grayscale

    # Locate the drawn digit's bounding box
    arr = np.array(img_gray, dtype=np.float32)
    ink_mask = arr < 250
    if not ink_mask.any():
        img_28 = Image.fromarray(np.full((28, 28), 255, dtype=np.uint8), mode="L")
        tensor = TRANSFORM(img_28).unsqueeze(0)
        return tensor, img_28

    # Crop to bounding box + 20% padding for whitespace
    rows = np.any(ink_mask, axis=1)
    cols = np.any(ink_mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    # 20% padding keeps whitespace similar to training images
    h, w = rmax - rmin, cmax - cmin
    pad = int(max(h, w) * 0.20)
    rmin = max(0, rmin - pad)
    rmax = min(arr.shape[0] - 1, rmax + pad)
    cmin = max(0, cmin - pad)
    cmax = min(arr.shape[1] - 1, cmax + pad)

    cropped = arr[rmin:rmax+1, cmin:cmax+1]

    # Make it square and center the digit
    h, w = cropped.shape
    size = max(h, w)
    square = np.full((size, size), 255.0, dtype=np.float32)
    y_offset = (size - h) // 2
    x_offset = (size - w) // 2
    square[y_offset:y_offset+h, x_offset:x_offset+w] = cropped

    # Resize to 28x28
    img_pil = Image.fromarray(square.astype(np.uint8), mode="L")
    img_28 = img_pil.resize((28, 28), Image.LANCZOS)

    # Light blur for large canvas images to match training antialiasing
    # (training data has antialiased pen strokes, not hard pixel edges)
    if max(img.size) > 100:
        img_28 = img_28.filter(ImageFilter.GaussianBlur(radius=0.5))

    # Apply the same normalization as during training
    tensor = TRANSFORM(img_28).unsqueeze(0)  # (1, 1, 28, 28)
    return tensor, img_28


# Endpoints

@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": MODEL is not None}


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        tensor, img_28 = preprocess_canvas_image(req.image)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image processing failed: {e}")

    # Inference
    with torch.no_grad():
        logits = MODEL(tensor.to(DEVICE))
        probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()

    predicted = int(probs.argmax())
    confidence = float(probs[predicted])

    # Probabilities for all 10 classes
    prob_dict = {CLASS_NAMES[i]: round(float(probs[i]), 4) for i in range(NUM_CLASSES)}

    # Encode the 28x28 preview so the UI can show what the model saw
    buf = io.BytesIO()
    img_28.save(buf, format="PNG")
    preview_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return PredictResponse(
        predicted_digit=predicted,
        confidence=round(confidence, 4),
        probabilities=prob_dict,
        processed_image=f"data:image/png;base64,{preview_b64}",
    )


# Reference training images (one per digit)

@app.get("/samples")
async def get_samples():
    """Return one actual training image per class so the user can see the style."""
    from data_pipeline import load_dataset, load_images_as_arrays
    paths, labels = load_dataset()
    samples = {}
    for d in range(NUM_CLASSES):
        idx = int(np.where(labels == d)[0][0])
        img = load_images_as_arrays(paths[idx:idx+1])[0]
        pil = Image.fromarray(img, mode="L")
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        samples[str(d)] = f"data:image/png;base64,{b64}"
    return JSONResponse(samples)


# Serve the frontend
STATIC_DIR = os.path.join(os.path.dirname(SRC_DIR), "ui")

@app.get("/")
async def serve_index():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return JSONResponse({"error": "Frontend not found. Place index.html in ui/ folder."})


# Mount static files (CSS, JS, etc.)
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=False)
