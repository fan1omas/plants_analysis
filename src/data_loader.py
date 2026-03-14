from PIL import Image
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from pathlib import Path

path = Path(__file__).parent.parent / "data" / "processed" 

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.RandomRotation(15),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])
 
d_train = ImageFolder(path / "train", transform=train_transform)
train_loader = DataLoader(d_train, batch_size=32, shuffle=True, num_workers=4)

d_val = ImageFolder(path / "val", transform=val_transform)
val_loader = DataLoader(d_val, batch_size=32, shuffle=False, num_workers=4)

print(f"📊 Train: {len(d_train)} фото, {len(train_loader)} батчей")
print(f"📊 Val:   {len(d_val)} фото, {len(val_loader)} батчей")
print(f"📋 Классов: {len(d_train.classes)}")