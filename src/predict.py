import torch
import torch.nn.functional as F
from PIL import Image
from pathlib import Path

from model import create_model, get_device
from data_loader import val_transform as transform

MODEL_PATH = Path("models/model.pth")
IMAGE_PATH = Path("image.jpg") 

_model = None
_device = None

CLASS_NAMES = [
    "apple_scab", "apple_black_rot", "apple_healthy",
    "blueberry_healthy", "cherry_powdery_mildew", "cherry_healthy",
    "corn_gray_leaf_spot", "corn_common_rust", "corn_northern_leaf_blight", "corn_healthy",
    "grape_black_rot", "grape_esca", "grape_leaf_blight", "grape_healthy",
    "orange_haunglongbing", "peach_bacterial_spot", "peach_healthy",
    "pepper_bacterial_spot", "pepper_healthy",
    "potato_early_blight", "potato_late_blight", "potato_healthy",
    "raspberry_healthy", "soybean_healthy", "squash_powdery_mildew",
    "strawberry_leaf_scorch", "strawberry_healthy",
    "tomato_bacterial_spot", "tomato_early_blight", "tomato_late_blight",
    "tomato_leaf_mold", "tomato_septoria_leaf_spot", "tomato_spider_mites",
    "tomato_target_spot", "tomato_yellow_leaf_curl", "tomato_mosaic_virus", "tomato_healthy"
]

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
    
    print(f"класс: {class_idx}")
    print(f"уверенность: {confidence*100:.2f}%")