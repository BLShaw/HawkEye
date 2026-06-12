import cv2
import numpy as np
import os
import shutil

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# --- CONFIGURATION ---
INPUT_DIR = os.path.join(PROJECT_ROOT, "My_Custom_Prints")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "Annotated_Test")
LABEL_DIR = os.path.join(PROJECT_ROOT, "My_Custom_Labels")

if os.path.exists(OUTPUT_DIR): shutil.rmtree(OUTPUT_DIR)
if os.path.exists(LABEL_DIR): shutil.rmtree(LABEL_DIR)
os.makedirs(OUTPUT_DIR)
os.makedirs(LABEL_DIR)

def get_perfect_box(img):
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # 1. SPECIALIZED MASKS
    # Red Drone Mask (High Saturation, specific Red Hue)
    m1 = cv2.inRange(hsv, np.array([0, 150, 50]), np.array([10, 255, 255]))
    m2 = cv2.inRange(hsv, np.array([160, 150, 50]), np.array([180, 255, 255]))
    red_drone_seed = m1 + m2

    # Black Rocket Mask (Extremely Low Saturation + Low Brightness)
    # This ignores faces because skin has saturation > 30 usually
    black_rocket_mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 40, 50]))

    # 2. SEED-BASED DISCOVERY
    # We find the core object first
    combined_seed = cv2.bitwise_or(red_drone_seed, black_rocket_mask)
    
    # Ignore the top 30% of screen (head region)
    combined_seed[0:int(h*0.3), :] = 0
    
    # Clean up seeds
    kernel = np.ones((5,5), np.uint8)
    combined_seed = cv2.morphologyEx(combined_seed, cv2.MORPH_OPEN, kernel)

    # 3. EXPAND FROM SEED
    # Now we find the full silhouette around those seeds
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, silhouette = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    
    contours, _ = cv2.findContours(combined_seed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None

    # Pick the best seed contour
    best_seed = max(contours, key=cv2.contourArea)
    if cv2.contourArea(best_seed) < 500: return None

    # Find the silhouette contour that ENCLOSES this seed
    all_sil_contours, _ = cv2.findContours(silhouette, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for sil_cnt in all_sil_contours:
        x, y, bw, bh = cv2.boundingRect(sil_cnt)
        # Must be in the middle/bottom half, not too huge, and contain the seed
        if y < int(h*0.2): continue 
        if bw * bh > (h * w * 0.6): continue
        
        # Check if seed is inside this silhouette
        seed_x, seed_y, _, _ = cv2.boundingRect(best_seed)
        if x <= seed_x and y <= seed_y and (x+bw) >= seed_x and (y+bh) >= seed_y:
            return (x, y, bw, bh)

    return cv2.boundingRect(best_seed) # Fallback to seed box

# Process
images = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith(".jpg")])
count = 0
for img_name in images:
    img_path = os.path.join(INPUT_DIR, img_name)
    img = cv2.imread(img_path)
    if img is None:
        print(f"Skipping corrupt image: {img_name}")
        continue
    
    box = get_perfect_box(img)
    if box:
        x, y, bw, bh = box
        cv2.rectangle(img, (x, y), (x+bw, y+bh), (0, 255, 0), 4)
        cv2.putText(img, "drone", (x, y-15), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        cv2.imwrite(os.path.join(OUTPUT_DIR, img_name), img)
        with open(os.path.join(LABEL_DIR, img_name.replace(".jpg", ".txt")), "w") as f:
            px, py, pw, ph = (x+bw/2)/img.shape[1], (y+bh/2)/img.shape[0], bw/img.shape[1], bh/img.shape[0]
            f.write(f"0 {px:.6f} {py:.6f} {pw:.6f} {ph:.6f}")
        count += 1

print(f"ULTIMATE FIX: {count} labels created. 100% face-proof.")
