"""
Capture background/negative images for drone training.
Point camera at your room WITHOUT any drone in frame.
Press SPACE to capture, Q to quit.
"""
import cv2, os, time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SAVE_PATH = os.path.join(PROJECT_ROOT, "My_Backgrounds")
os.makedirs(SAVE_PATH, exist_ok=True)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

count = 0
print("Point camera at your ROOM (no drone). SPACE=capture, Q=quit")
print(f"Target: 50+ images of different room angles/lighting")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    disp = frame.copy()
    cv2.putText(disp, f"BACKGROUND CAPTURE | Saved: {count}", (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.putText(disp, "SPACE=capture  Q=quit", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(disp, "NO DRONE in frame!", (10, 105),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imshow("Background Capture", disp)

    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):
        name = os.path.join(SAVE_PATH, f"bg_{int(time.time()*1000)}.jpg")
        cv2.imwrite(name, frame)
        count += 1
        print(f"Saved: {name}")
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"\nDone. {count} background images saved to '{SAVE_PATH}/'")
print("Upload this folder to Roboflow → DON'T annotate anything → re-export dataset")
