from fastapi import FastAPI, File
import uvicorn
from PIL import Image
from data_loader import val_transform as transform
from predict import load_model, predict_image, get_disease_name_ru, MODEL_PATH, CLASS_NAMES
from model import get_device
from io import BytesIO

app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


_model = None
_device = None

@app.on_event("startup")
async def startup_event():
    global _device, _model

    _device = get_device()
    _model = load_model(MODEL_PATH, _device)

@app.get('/')
async def main(): 
    return {"success": True}

@app.post('/predict', tags=['Предсказание'])
async def predict(file_bytes: bytes = File()):
    image = Image.open(BytesIO(file_bytes)).convert("RGB")

    class_idx, confidence = predict_image(_model, image, _device, transform)
    disease = CLASS_NAMES[class_idx]
    disease_ru = get_disease_name_ru(disease)

    return {"success": True, "class_index": class_idx, "disease": disease, "disease_rus": disease_ru, "confidence": confidence}


if __name__ == '__main__':
    uvicorn.run('api:app', reload=True)
