import cv2
import numpy as np
from detector import ObjectDetector
from classifier import AnomalyClassifier

anomaly_classes = [
    "Abuse", "Arrest", "Arson", "Assault", "Burglary", "Explosion", "Fighting",
    "Normal", "RoadAccidents", "Robbery", "Shooting", "Shoplifting", "Stealing", "Vandalism"
]

detector = ObjectDetector()
classifier = AnomalyClassifier()

cap = cv2.VideoCapture("test_video.mp4")  # Replace with your video path

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    detections = detector.detect(frame)

    for det in detections:
        x1, y1, x2, y2, conf, cls = map(int, det[:6])
        cropped = frame[y1:y2, x1:x2]
        if cropped.size == 0:
            continue
        pred = classifier.predict(cropped)

        color = (0, 255, 0) if pred == 7 else (0, 0, 255)  # 7 = Normal
        label = anomaly_classes[pred]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow("Anomaly Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
