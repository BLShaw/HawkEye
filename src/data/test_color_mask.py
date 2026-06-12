import cv2
import numpy as np
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

def test_box(img_path):
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Check if red drone
    m1 = cv2.inRange(hsv, np.array([0, 100, 50]), np.array([10, 255, 255]))
    m2 = cv2.inRange(hsv, np.array([160, 100, 50]), np.array([180, 255, 255]))
    red_mask = m1 + m2
    
    if cv2.countNonZero(red_mask) > 1000:
        print("DETECTED: RED DRONE")
        # Find red center
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = max(contours, key=cv2.contourArea)
        rx, ry, rbw, rbh = cv2.boundingRect(best)
        
        # Expand based on user strategy
        expand_w = int(rbw * 1.5)
        expand_h = int(rbh * 1.5)
        nx = max(0, rx - int(rbw*0.75))
        ny = max(0, ry - int(rbh*0.75))
        nbw = min(w - nx, rbw + expand_w)
        nbh = min(h - ny, rbh + expand_h)
        
        preview = img.copy()
        cv2.rectangle(preview, (nx, ny), (nx+nbw, ny+nbh), (0, 255, 0), 4)
        cv2.imwrite("test_red_out.jpg", preview)
        return
        
    print("DETECTED: BLACK ROCKET")
    
    # We want really dark stuff that isn't skin
    # The rocket is very dark, hands are brighter and more saturated
    _, dark_mask = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY_INV)
    
    skin_mask = cv2.inRange(hsv, np.array([0, 10, 60]), np.array([25, 200, 255]))
    dark_no_skin = cv2.bitwise_and(dark_mask, cv2.bitwise_not(skin_mask))
    
    # Exclude top 25% (hair)
    dark_no_skin[0:int(h*0.25), :] = 0
    
    kernel = np.ones((7,7), np.uint8)
    clean = cv2.morphologyEx(dark_no_skin, cv2.MORPH_OPEN, kernel)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        best = max(contours, key=cv2.contourArea)
        bx, by, bbw, bbh = cv2.boundingRect(best)
        
        preview = img.copy()
        cv2.rectangle(preview, (bx, by), (bx+bbw, by+bbh), (0, 255, 0), 4)
        cv2.imwrite("test_black_out.jpg", preview)
        
test_box(os.path.join(PROJECT_ROOT, "My_Custom_Prints", "drone_img_1776927877.jpg"))
test_box(os.path.join(PROJECT_ROOT, "My_Custom_Prints", "drone_img_1776929963.jpg"))
test_box(os.path.join(PROJECT_ROOT, "My_Custom_Prints", "drone_img_1776929831.jpg"))
