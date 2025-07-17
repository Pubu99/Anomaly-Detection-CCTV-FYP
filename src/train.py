# src/train.py
import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from datetime import datetime

from dataset import SurveillanceDataset
from model import get_model

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logdir = f"runs/exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(logdir)

    train_ds = SurveillanceDataset("data/train")
    val_ds = SurveillanceDataset("data/val")

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)

    model = get_model(num_classes=len(os.listdir("data/train")))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_acc = 0.0

    for epoch in range(10):
        model.train()
        running_loss, correct = 0.0, 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/10")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()

        epoch_loss = running_loss / len(train_ds)
        epoch_acc = correct / len(train_ds)
        writer.add_scalar("Loss/Train", epoch_loss, epoch)
        writer.add_scalar("Accuracy/Train", epoch_acc, epoch)

        # Validation
        model.eval()
        val_correct = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                val_correct += (outputs.argmax(1) == labels).sum().item()
        val_acc = val_correct / len(val_ds)
        writer.add_scalar("Accuracy/Val", val_acc, epoch)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "models/best_model.pt")
            print(f"✅ Saved better model (Val Acc: {val_acc:.4f})")

    writer.close()

if __name__ == "__main__":
    train()
