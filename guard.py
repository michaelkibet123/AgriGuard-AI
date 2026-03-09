import cv2
import numpy as np
from datetime import datetime
import os

now = datetime.now()
date_str = now.strftime("%Y-%m-%d %H:%M")
img = cv2.imread("leaf.jpg")

if img is not None:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (36, 25, 25), (86, 255, 255))
    green_val = (cv2.countNonZero(mask) / (img.size / 3)) * 100
    status = "CRITICAL" if green_val < 30 else "WARNING" if green_val < 75 else "HEALTHY"

    # ARCHIVE LOGGING
    with open("farm_log.csv", "a") as f:
        f.write(f"{date_str}, {green_val:.2f}%, {status}\n")

    # GALLERY EXPORT
    time_stamp = now.strftime("%I-%M%p")
    fname = f"AgriGuard_{status}_{time_stamp}.jpg"
    cv2.imwrite(fname, mask)
    os.system(f"cp {fname} /sdcard/Download/")

    print(f"--- LOGGED TO FARM_LOG.CSV ---")
    print(f"Health: {green_val:.1f}% | Status: {status}")
    print(f"✅ Data Archived in CSV.")
else:
    print("❌ Error: leaf.jpg missing")