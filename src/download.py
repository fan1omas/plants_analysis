import kagglehub
from pathlib import Path
import shutil


PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data" / "raw"           # получаем путь к данным для обучения 
DATA_DIR.mkdir(exist_ok=True, parents=True)

path = kagglehub.dataset_download("abdallahalidev/plantvillage-dataset")

color_folder = None
for folder_path in Path(path).rglob("*"):   # рекурсивный поиск по всем папкам
    if folder_path.is_dir() and folder_path.name == "color":
        color_folder = folder_path
        break

target_path = DATA_DIR / "plantvillage" / "color"
shutil.copytree(color_folder, target_path)
shutil.rmtree(path)

print('done')