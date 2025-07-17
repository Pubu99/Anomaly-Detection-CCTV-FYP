import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

class CNNClassifier(nn.Module):
    def __init__(self, num_classes=14):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)

class AnomalyClassifier:
    def __init__(self, model_path="models/best_model.pt", num_classes=14):
        self.model = CNNClassifier(num_classes)
        self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor()
        ])
        self.classes = [
            "Abuse", "Arrest", "Arson", "Assault", "Burglary", "Explosion", "Fighting",
            "Normal", "RoadAccidents", "Robbery", "Shooting", "Shoplifting", "Stealing", "Vandalism"
        ]

    def classify(self, image, top_k=3):
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        image_tensor = self.transform(image).unsqueeze(0)
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1).squeeze()
            topk_probs, topk_indices = torch.topk(probs, top_k)
            return {
                "predicted": topk_indices[0].item(),
                "confidence": topk_probs[0].item(),
                "topk_classes": [self.classes[i] for i in topk_indices],
                "topk_confidences": [round(p.item(), 2) for p in topk_probs],
                "all_probs": probs.tolist()
            }
