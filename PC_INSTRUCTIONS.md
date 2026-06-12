# HawkEye: A Guide for Windows Users

This guide walks you through deploying the HawkEye drone detection system directly on a Windows laptop, using your built-in webcam for real-time inference, and Google Colab for heavy-duty AI training.

---

## Phase 1: Environment Setup

Before starting, ensure you have [Python 3.9+](https://www.python.org/downloads/) installed on your Windows machine.

1. **Open PowerShell** and navigate to the project directory.
2. **Create a Virtual Environment** (Recommended to prevent library conflicts):
   ```powershell
   python -m venv venv
   .\venv\Scripts\activate
   ```
3. **Install Dependencies**:
   ```powershell
   pip install ultralytics opencv-python numpy
   ```

---

## Phase 2: Capture Custom Drone Data (Optional)
*If you already have a trained model (`best.pt`), or just want to use the default YOLOv8 drone weights, you can skip to Phase 4.*

If you want the system to recognize your specific drone (e.g., a custom 3D printed model), you need to capture footage of it.

1. **Run the Data Capture Script**:
   ```powershell
   python src/data/capture_dataset.py
   ```
2. **Select Mode 1 (Drone Mode)**: The webcam will turn on. Hold your drone in front of the camera and rotate it slowly. The script automatically saves an image every 1 second into `My_Custom_Prints/`. Press `q` to quit.
3. **Select Mode 2 (Background Mode)**: Run the script again and press `2`. Walk around the room with **no drone** in the frame. This captures negative backgrounds so the AI learns what an empty room looks like.
4. **Auto-Label the Data**: 
   ```powershell
   python src/data/auto_labeler.py
   ```
   This script uses computer vision color-masking to automatically draw YOLO bounding boxes around your drone. The labeled dataset will be output into the `Annotated_Test/` folder.

5. **Upload to Roboflow**: Create a free account on [Roboflow](https://roboflow.com/), create a new project, and drag-and-drop the contents of your `Annotated_Test/` folder. Generate a new dataset version and copy your API key.

---

## Phase 3: Train the AI on Google Colab (Free GPU)

Since laptops usually don't have enough VRAM to train large YOLO models from scratch, we use Google Colab's free T4 GPUs.

1. **Upload Notebook**: Go to [Google Colab](https://colab.research.google.com/), click **Upload**, and select the `notebooks/colab_train.ipynb` file from this repository.
2. **Enable GPU**: In Colab, go to `Runtime` > `Change runtime type` > Select **T4 GPU** > Click Save.
3. **Run the Cells**: Start running the cells from top to bottom.
4. **Input API Key**: When prompted in the notebook, paste your Roboflow API Key. The notebook will automatically download your custom dataset, merge it with 12,000+ images of birds/airplanes/helicopters, and begin training.
5. **Download the Weights**: Once the 150 epochs of training are finished, download the resulting `best.pt` file to your Windows laptop.

---

## Phase 4: Live Inference on Windows

Now that you have your trained model, you can run real-time inference using your laptop's webcam.

1. **Place the Model**: Move the downloaded `best.pt` file directly into the `HawkEye/` root directory.
2. **Run Inference**:
   ```powershell
   python src/deployment/pc_inference.py
   ```
   
> [!NOTE]
> **Hardware Acceleration:** The script automatically checks if your Windows laptop has an NVIDIA GPU installed (CUDA). If it does, it will utilize it for maximum FPS. If not, it will run on the CPU.

3. **See the Results**: A window will pop up showing your live webcam feed. Hold up your drone (or show a picture of one on your phone) and watch the bounding boxes track it in real-time! Press `q` to close the window.

---

## Phase 5: The Web Dashboard (Ground Control Station)

To view the futuristic Ground Control UI:
1. Open the file `front/controller.html` directly in Chrome or Edge.
2. The UI features an artificial horizon, compass, telemetry dials, and simulated flight mode controls. (In the future, the `pc_inference.py` script can be tied to this UI via WebSockets to pass live telemetry data to the dashboard).
