'''Для дообучения модели необходимо знать, какие классы хуже всего предсказываются, для этого
   был создан этот файл со случайным отбором фотографий из каждого класса и предугадыванием этих фотографий.
   Результат сохраняется в ../error_classes.json'''

from PIL import Image
from predict import load_model, predict_image, get_disease_name_ru, MODEL_PATH, CLASS_NAMES, BASE_DIR
from model import get_device
from data_loader import val_transform as transform
from pathlib import Path
from random import choice
import json
from time import time 

start_time = time()
NUMBER = 1000
error_classes = {}

_device = get_device()
_model = load_model(MODEL_PATH, _device)

DATA_DIR = BASE_DIR / "data" / "processed" / "val"

for _ in range(NUMBER):
    print(f'\r{_}', end='', flush=True)
    for folder in DATA_DIR.iterdir():
        IMAGE_PATH = Path(choice(list(folder.iterdir())))
        image =  Image.open(IMAGE_PATH).convert("RGB")

        class_idx, _ = predict_image(_model, image, _device, transform)
        disease = CLASS_NAMES[class_idx]

        if folder.name != disease:
            if folder.name in error_classes:
                error_classes[folder.name] += 1
            else:
                error_classes[folder.name] = 1

error_classes = dict(sorted(error_classes.items(), key=lambda x: x[1]))

with open(BASE_DIR / 'error_classes.json', 'w', encoding='utf-8') as f:
    json.dump(error_classes, f, indent=4)

end_time = time()
diff = end_time - start_time
print()
print(f'{diff // 60} минут.')

"""print('=' * 40)
for i in error_classes:
    print(f'{i}: {error_classes[i]}')"""

