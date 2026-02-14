#!/usr/bin/env python3
from picamera2 import Picamera2
import time
import numpy as np
import cv2

# Initialize both cameras
picam0 = Picamera2(0)
picam1 = Picamera2(1)

# Configure cameras for stereo (matching resolution and format)
config0 = picam0.create_still_configuration(
    main={"size": (1280, 720), "format": "RGB888"}
)
config1 = picam1.create_still_configuration(
    main={"size": (1280, 720), "format": "RGB888"}
)

picam0.configure(config0)
picam1.configure(config1)

# Start cameras
picam0.start()
picam1.start()

time.sleep(2)  # Allow cameras to warm up

print("Cameras ready. Press Ctrl+C to stop.")
print("Capturing stereo images...")

try:
    frame_count = 0
    while True:
        # Capture from both cameras (approximately synchronized)
        img0 = picam0.capture_array()
        img1 = picam1.capture_array()
        
        # Save images
        cv2.imwrite(f'left_{frame_count:04d}.jpg', cv2.cvtColor(img0, cv2.COLOR_RGB2BGR))
        cv2.imwrite(f'right_{frame_count:04d}.jpg', cv2.cvtColor(img1, cv2.COLOR_RGB2BGR))
        
        print(f"Captured frame {frame_count}")
        frame_count += 1
        time.sleep(0.5)  # Capture every 0.5 seconds
        
except KeyboardInterrupt:
    print("\nStopping capture...")
    
finally:
    picam0.stop()
    picam1.stop()
    print(f"Captured {frame_count} stereo pairs")