import torch
import cv2

class ObjectDetector:
    def __init__(self):
        self.model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)
        self.model.conf = 0.4  # Confidence threshold

    def detect(self, frame):
        results = self.model(frame)
        detections = results.xyxy[0]  # x1, y1, x2, y2, conf, class
        return detections.cpu().numpy()
