from torchvision.datasets import ImageFolder

train_data = ImageFolder("../data/processed/train")
val_data = ImageFolder("../data/processed/val")

print("порядок классов в train")
for i, name in enumerate(train_data.classes):
    print(f"{i}: {name}")

print("\nпорядок классов в val")
for i, name in enumerate(val_data.classes):
    print(f"{i}: {name}")

print("\nCLASS_NAMES в predict.py")
from predict import CLASS_NAMES
for i, name in enumerate(CLASS_NAMES):
    print(f"{i}: {name}")

print("\n=== Проверка ===")
if train_data.classes == CLASS_NAMES:
    print("порядок совпадает с train")
else:
    print("порядок не совпадает")