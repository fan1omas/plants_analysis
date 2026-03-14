from pathlib import Path
from random import shuffle
import shutil   

PROJECT_ROOT = Path(__file__).parent.parent

SOURCE_DIR = PROJECT_ROOT / "data" / "raw" / "plantvillage" / "color"
TARGET_DIR = PROJECT_ROOT / "data" / "processed"
TRAIN_RATIO = 0.8                                                       #  80% train - 20% val

(TARGET_DIR / "train").mkdir(parents=True, exist_ok=True)
(TARGET_DIR / "val").mkdir(parents=True, exist_ok=True)

for class_folder in SOURCE_DIR.iterdir():
    if not class_folder.is_dir():
        continue

    train_class = TARGET_DIR / "train" / class_folder.name
    val_class = TARGET_DIR / "val" / class_folder.name
    
    train_class.mkdir(parents=True, exist_ok=True)
    val_class.mkdir(parents=True, exist_ok=True)
    
    images = list(class_folder.glob("*.jpg"))
    shuffle(images) 
    
    split_idx = int(len(images) * TRAIN_RATIO)          # 80%
    
    for img in images[:split_idx]:
        shutil.copy(img, train_class / img.name)
    
    for img in images[split_idx:]:
        shutil.copy(img, val_class / img.name)
