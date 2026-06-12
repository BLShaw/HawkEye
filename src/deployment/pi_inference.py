import subprocess
import cv2
import numpy as np
from ultralytics import YOLO
import time
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Use your fine-tuned model (copy best.pt from training to Pi)
# Fall back to base YOLOv8n if you haven't trained yet
MODEL_PATH = os.path.join(PROJECT_ROOT, "best.pt")  # replace with path to your fine-tuned weights

model = YOLO(MODEL_PATH)
model.to("cpu")  # Pi doesn't have a GPU

# Pi camera stream via rpicam-vid
cmd = [
    "rpicam-vid",
    "--width", "640",
    "--height", "480",
    "--framerate", "15",
    "--codec", "mjpeg",
    "--output", "-",
    "--timeout", "0",
    "--nopreview",
]

process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
print("Camera started. Running drone detection... CTRL+C to stop")

buffer = b""
CONF_THRESHOLD = 0.45  # tweak: lower = more detections, higher = fewer false positives

while True:
    buffer += process.stdout.read(4096)

    start = buffer.find(b"\xff\xd8")
    end = buffer.find(b"\xff\xd9")

    if start == -1 or end == -1:
        continue

    jpg = buffer[start : end + 2]
    buffer = buffer[end + 2 :]

    frame = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        continue

    results = model(frame, conf=CONF_THRESHOLD, verbose=False)

    for result in results:
        for box in result.boxes:
            label = model.names[int(box.cls)]
            conf = float(box.conf)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            print(f"DRONE DETECTED | conf={conf:.2f} | bbox=[{x1},{y1},{x2},{y2}]")
