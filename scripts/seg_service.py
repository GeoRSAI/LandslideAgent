from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging

from src.models.seg_infer import run_segmentation

app = FastAPI(title="Landslide Segmentation Service")
logging.basicConfig(level=logging.INFO)

model = None

@app.on_event("startup")
def load_model():
    global model
    logging.info("Loading segmentation model...")
    model = True
    logging.info("Segmentation model loaded.")

class SegRequest(BaseModel):
    image_path: str

@app.post("/predict")
def predict(req: SegRequest):
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    logging.info(f"Processing segmentation for {req.image_path}")

    result = run_segmentation({"image_path": req.image_path})
    return result

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}
