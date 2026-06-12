import cv2
import os
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

print("========================================")
print("  DATASET CAPTURE TOOL")
print("========================================")
print("  1 = Capture DRONE images")
print("  2 = Capture BACKGROUND images (no drone)")
print("========================================")
mode = input("Choose mode (1 or 2): ").strip()

if mode == "1":
    SAVE_PATH = os.path.join(PROJECT_ROOT, "My_Custom_Prints")
    PREFIX    = "drone_img"
    WINDOW    = "DRONE CAPTURE — hold drone, show all angles"
    COLOR     = (0, 0, 255)
    LABEL1    = "MODE: DRONE — rotate & show all sides"
    LABEL2    = "Remove drone from frame for backgrounds (mode 2)"
else:
    SAVE_PATH = os.path.join(PROJECT_ROOT, "My_Backgrounds")
    PREFIX    = "bg_img"
    WINDOW    = "BACKGROUND CAPTURE — NO drone in frame!"
    COLOR     = (0, 200, 0)
    LABEL1    = "MODE: BACKGROUND — no drone in frame!"
    LABEL2    = "Move around room, different angles & lighting"

os.makedirs(SAVE_PATH, exist_ok=True)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

count = 0
last_capture = time.time()
INTERVAL = 1.0  # capture every 1 second

print(f"\nSaving to: {SAVE_PATH}/")
print("Auto-capturing 1 frame/sec — press Q to stop\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if time.time() - last_capture >= INTERVAL:
        img_name = os.path.join(SAVE_PATH, f"{PREFIX}_{int(time.time()*1000)}.jpg")
        cv2.imwrite(img_name, frame)
        count += 1
        last_capture = time.time()
        print(f"Saved: {img_name}  (total: {count})")

    disp = cv2.flip(frame, 1)
    cv2.rectangle(disp, (0, 0), (disp.shape[1], 80), (0, 0, 0), -1)
    cv2.putText(disp, LABEL1, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, COLOR, 2)
    cv2.putText(disp, LABEL2, (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(disp, f"Saved: {count} images  |  Q = quit",
                (10, disp.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.imshow(WINDOW, disp)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print(f"\nDone! {count} images saved to '{SAVE_PATH}/'")
if mode == "1":
    print("Next: upload My_Custom_Prints/ to Roboflow and annotate")
else:
    print("Next: upload My_Backgrounds/ to Roboflow — DON'T annotate anything (leave blank)")
    print("Then generate a new version (v3) and update the notebook to version(3)")
