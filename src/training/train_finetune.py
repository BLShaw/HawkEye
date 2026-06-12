"""
Fine-tune YOLOv11x drone detection model on your custom 3D-printed drone photos.

Before running:
1. pip install ultralytics
2. Label your My_Custom_Prints images on roboflow.com (class name: drone)
3. Export as YOLOv8 format and unzip into: Drone/dataset/
4. The dataset/ folder should have: data.yaml, train/, valid/ subfolders

Then run: python train_finetune.py
"""

from ultralytics import YOLO
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Start from the pre-trained YOLOv11x drone detection weights.
# Download from: https://github.com/doguilmak/Drone-Detection-YOLOv11x
# Place the .pt file in this folder and update the filename below.
PRETRAINED_WEIGHTS = os.path.join(PROJECT_ROOT, "drone_yolo11x.pt")  # rename to whatever you downloaded

# If you haven't downloaded that model yet, use the base YOLOv11x as fallback:
# PRETRAINED_WEIGHTS = "yolo11x.pt"

model = YOLO(PRETRAINED_WEIGHTS)

results = model.train(
    data=os.path.join(PROJECT_ROOT, "dataset", "data.yaml"),
    epochs=50,
    imgsz=640,
    batch=8,                    # lower to 4 if you get out-of-memory errors
    patience=15,                # stop early if no improvement
    lr0=0.001,                  # low LR for fine-tuning
    freeze=10,                  # freeze first 10 layers, only train the head
    name="drone_custom",
    project=os.path.join(PROJECT_ROOT, "runs", "train"),
    exist_ok=True,
    augment=True,               # random flips, brightness, etc.
    degrees=45,                 # rotate up to 45 degrees (drones seen from any angle)
    fliplr=0.5,
    flipud=0.2,
    hsv_h=0.02,
    hsv_s=0.5,
    hsv_v=0.3,
    scale=0.5,
)

print("\n=== TRAINING DONE ===")
print(f"Best model saved to: runs/train/drone_custom/weights/best.pt")
print("Copy best.pt to your Raspberry Pi and update Drone.py to use it.")
