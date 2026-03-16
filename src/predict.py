import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path

from model import create_model, get_device
from data_loader import val_transform as transform

MODEL_PATH = Path("models/old_model.pth")
IMAGE_PATH = Path("1.jpg") 

def load_model(model_path, device):
    model = create_model()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def predict_image(model, image_path, device, transform):
    image = Image.open(image_path).convert("RGB")
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
    
    print(f"класс: {class_idx}")
    print(f"уверенность: {confidence*100:.2f}%")