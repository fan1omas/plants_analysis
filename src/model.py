from torchvision import models
import torch
import torch.nn as nn

NUM_CLASSES = 38
 
def create_model(num_classes=NUM_CLASSES):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.fc.in_features, num_classes)
    )

    return model

def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")   # gpu
    else:
        device = torch.device("cpu")    # cpu
    
    return device 

if __name__ == "__main__":
    model = create_model()
    device = get_device() 
    model = model.to(device)

    print(f'Device: {device}')
    print(f'Параметров: {sum(p.numel() for p in model.parameters()):,}')
    print(f'Обучаемых: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}')

