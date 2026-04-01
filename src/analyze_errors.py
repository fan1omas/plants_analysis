from PIL import Image
from predict import load_model, predict_image, get_disease_name_ru, MODEL_PATH, CLASS_NAMES, BASE_DIR
from model import get_device
from data_loader import val_transform as transform
from pathlib import Path
from random import choice

error_classes = []

_device = get_device()
_model = load_model(MODEL_PATH, _device)

DATA_DIR = BASE_DIR / "data" / "processed" / "val"
for folder in DATA_DIR.iterdir():
    IMAGE_PATH = Path(choice(list(folder.iterdir())))
    image =  Image.open(IMAGE_PATH).convert("RGB")

    class_idx, _ = predict_image(_model, image, _device, transform)
    disease = CLASS_NAMES[class_idx]

    if folder.name != disease:
        print(f'Файл из {folder.name}, модель распознала как {disease}')
        error_classes.append(folder.name) 

print('=' * 40)
for i in error_classes:
    print(i)

