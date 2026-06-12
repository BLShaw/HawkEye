import cv2
import torch
import os
from ultralytics import YOLO

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

MODEL_PATH = os.path.join(PROJECT_ROOT, "best.pt")
if not os.path.exists(MODEL_PATH):
    print("WARNING: 'best.pt' not found in project root. Downloading base yolov8n.pt instead.")
    MODEL_PATH = "yolov8n.pt"

model = YOLO(MODEL_PATH)

if torch.cuda.is_available():
    device = "cuda"
    print("Using GPU (CUDA) for accelerated inference.")
elif torch.backends.mps.is_available():
    device = "mps"
    print("Using Apple Silicon (MPS) for accelerated inference.")
else:
    device = "cpu"
    print("Using CPU for inference.")

model.to(device)

# Initialize laptop webcam (0 is built-in webcam)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    exit()

print("Camera started. Running drone detection... Press 'q' to quit.")

CONF_THRESHOLD = 0.45  # tweak: lower = more detections, higher = fewer false positives

while True:
    ret, frame = cap.read()
    if not ret:
        print("ERROR: Failed to grab frame.")
        break

    results = model(frame, conf=CONF_THRESHOLD, verbose=False)

    annotated_frame = results[0].plot()

    cv2.imshow("HawkEye: Drone Detection (Live)", annotated_frame)
    for result in results:
        for box in result.boxes:
            label_idx = int(box.cls)
            label = model.names[label_idx]
            conf = float(box.conf)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            print(f"{label.upper()} DETECTED | conf={conf:.2f} | bbox=[{x1},{y1},{x2},{y2}]")

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
