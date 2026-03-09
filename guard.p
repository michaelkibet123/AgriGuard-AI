import cv2
import os
from datetime import datetime
print("--- ARCHIVE MODE ONLINE ---")
img = cv2.imread("leaf.jpg")
if img is not None:
    now = datetime.now().strftime("%Y-%b-%d_%I-%M%p")
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (36, 25, 25), (86, 255, 255))
    fname = f"AgriGuard_{now}.jpg"
    cv2.imwrite(fname, mask)
    os.system(f"cp {fname} /sdcard/Download/")
    print(f"✅ Saved to Gallery as: {fname}")
else:
    print("❌ Error: leaf.jpg missing")