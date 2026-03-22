import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

from model import create_model, get_device
from data_loader import train_loader, val_loader

NUM_EPOCHS = 10
LEARNING_RATE = 0.001
SAVE_PATH = Path("../models")

def train(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()                
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    return total_loss / len(loader), 100. * correct / total

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return total_loss / len(loader), 100. * correct / total


def main():
    device = get_device()
    model = create_model().to(device) 

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)

    best_accuracy = 0
    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        print(f'train loss: {train_loss}, train acc: {train_acc}\nval loss: {val_loss}, val acc: {val_acc}\n')

        if val_acc > best_accuracy: 
            best_accuracy = val_acc 
            torch.save(model.state_dict(), SAVE_PATH / f'model_v{epoch}')
    
    print('done') 



if __name__ == '__main__': 
    main()
    