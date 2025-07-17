# src/realtime_demo.py
import cv2
import torch
from torchvision import transforms
from model import get_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_names = sorted(os.listdir("data/train"))

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

model = get_model(num_classes=len(class_names))
model.load_state_dict(torch.load("models/best_model.pt"))
model = model.to(device)
model.eval()

cap = cv2.VideoCapture(0)  # Webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = transform(Image.fromarray(img)).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(img).argmax(1).item()

    label = class_names[pred]
    cv2.putText(frame, f"Detected: {label}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Real-Time Anomaly Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
