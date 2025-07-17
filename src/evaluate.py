# src/evaluate.py
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os
from dataset import SurveillanceDataset
from model import get_model

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_ds = SurveillanceDataset("data/val")
    val_loader = DataLoader(val_ds, batch_size=32)

    model = get_model(num_classes=len(os.listdir("data/val")))
    model.load_state_dict(torch.load("models/best_model.pt"))
    model.to(device)
    model.eval()

    y_true, y_pred = [], []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(1).cpu().numpy()
            y_pred.extend(preds)
            y_true.extend(labels.numpy())

    print(classification_report(y_true, y_pred))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.savefig("outputs/confusion_matrix.png")
    plt.show()

if __name__ == "__main__":
    evaluate()
