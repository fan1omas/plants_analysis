import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.datasets import ImageFolder
from pathlib import Path

from model import create_model, get_device
from data_loader import val_transform as transform

MODEL_PATH = Path("models/model.pth")
IMAGE_PATH = Path("1.jpg") 

_model = None
_device = None

train_data = ImageFolder('../data/processed/train')
CLASS_NAMES = train_data.classes

import json

TRANSLATIONS = '../translations.json'
with open(TRANSLATIONS, "r", encoding="utf-8") as f:
    CLASS_NAMES_RU = json.load(f)

def get_disease_name_ru(name):
    global CLASS_NAMES_RU 
    return CLASS_NAMES_RU[name]

def load_model(model_path, device):
    global _model, _device
    if _model is None:
        _device = get_device()
        _model = create_model()
        _model.load_state_dict(torch.load(model_path, map_location=device))
        _model.to(_device)
        _model.eval()
    return _model

def predict_image(model, image_input, device, transform):
    if isinstance(image_input, (str, Path)):
        image = Image.open(image_input).convert("RGB")
    else:
        image = image_input.convert("RGB")
        
    image = transform(image)

    image = image.unsqueeze(0)
    image = image.to(device)

    with torch.no_grad():
        outputs = model(image)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    return predicted.item(), confidence.item()

if __name__ == "__main__":
    device = get_device()
    model = load_model(MODEL_PATH, device)
    
    class_idx, confidence = predict_image(model, IMAGE_PATH, device, transform)
    
    print(f"класс: {class_idx}, {CLASS_NAMES[class_idx]}")
    print(f"уверенность: {confidence*100:.2f}%")