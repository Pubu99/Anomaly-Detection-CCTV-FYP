from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import cv2
from detector import ObjectDetector
from classifier import AnomalyClassifier
import threading

app = Flask(__name__)
UPLOAD_FOLDER = "static"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Globals to store processing results
processing_log = []
topk_confidences = []
topk_classes = []
processing = False
output_video = "output.mp4"

# Instantiate detector and classifier once to save load time
detector = ObjectDetector()
classifier = AnomalyClassifier()

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_video():
    global processing, processing_log, topk_confidences, topk_classes

    if processing:
        return jsonify({"status": "busy", "message": "Processing already in progress"}), 429

    f = request.files.get("file")
    if not f:
        return jsonify({"error": "No file provided"}), 400

    video_path = os.path.join(UPLOAD_FOLDER, f.filename)
    f.save(video_path)

    # Reset globals
    processing_log = []
    topk_confidences = []
    topk_classes = []
    processing = True

    # Process in background thread to avoid blocking
    thread = threading.Thread(target=process_video, args=(video_path,))
    thread.start()

    return jsonify({"status": "started", "filename": f.filename})

@app.route("/status", methods=["GET"])
def get_status():
    # Return current log and confidence info
    return jsonify({
        "processing": processing,
        "log": processing_log,
        "topk_confidences": topk_confidences,
        "topk_classes": topk_classes,
        "output_video": output_video if not processing else None
    })

@app.route("/video/<filename>")
def get_video(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

def process_video(path):
    global processing, processing_log, topk_confidences, topk_classes

    cap = cv2.VideoCapture(path)
    out_path = os.path.join(UPLOAD_FOLDER, output_video)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_path, fourcc, 20.0, (640, 480))

    log = []
    confidence_accum = {}

    anomaly_classes = [
        "Abuse", "Arrest", "Arson", "Assault", "Burglary", "Explosion", "Fighting",
        "Normal", "RoadAccidents", "Robbery", "Shooting", "Shoplifting", "Stealing", "Vandalism"
    ]

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (640, 480))
        detections = detector.detect(frame)

        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        frame_confidences = {}

        for det in detections:
            x1, y1, x2, y2, *_ = map(int, det[:6])
            cropped = frame[y1:y2, x1:x2]
            if cropped.size == 0:
                continue

            result = classifier.classify(cropped)
            pred = result["predicted"]
            label = anomaly_classes[pred]
            conf = result["confidence"]
            color = (0, 255, 0) if label == "Normal" else (0, 0, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            text = f"{label} ({int(conf*100)}%)"
            cv2.putText(frame, text, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            for i, (cls, conf_top) in enumerate(zip(result["topk_classes"], result["topk_confidences"])):
                bar_x = x1 + (i * 80)
                bar_y = y2 + 10
                cv2.rectangle(frame, (bar_x, bar_y),
                              (bar_x + 60, bar_y + 10), (180, 180, 180), -1)
                cv2.rectangle(frame, (bar_x, bar_y),
                              (bar_x + int(conf_top * 60), bar_y + 10), color, -1)
                cv2.putText(frame, cls, (bar_x, bar_y + 25),
                            cv2.FONT_HERSHEY_PLAIN, 0.7, (255, 255, 255), 1)

            if label != "Normal":
                log_entry = f"{timestamp:.2f}s - {label} ({int(conf*100)}%)"
                log.append(log_entry)

            for cls, conf_top in zip(result["topk_classes"], result["topk_confidences"]):
                frame_confidences[cls] = frame_confidences.get(cls, 0) + conf_top

        for cls, conf_sum in frame_confidences.items():
            confidence_accum[cls] = confidence_accum.get(cls, 0) + conf_sum

        out.write(frame)

    cap.release()
    out.release()

    sorted_conf = sorted(confidence_accum.items(), key=lambda x: x[1], reverse=True)[:3]
    topk_classes = [cls for cls, _ in sorted_conf]
    topk_confidences = [conf / frame_count for _, conf in sorted_conf]

    processing_log = log
    processing = False

if __name__ == "__main__":
    app.run(debug=True)
