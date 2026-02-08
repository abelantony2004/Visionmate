import cv2
from picamera2 import Picamera2
import os
import time
from PIL import Image
import numpy as np
import csv

# === Setup output ===
output_dir = "calib"
os.makedirs(output_dir, exist_ok=True)

distance = input("Enter distance (in meters): ").strip()

# === CSV Logging Setup ===
csv_path = os.path.join(output_dir, "capture_log.csv")
new_csv = not os.path.exists(csv_path)
csv_file = open(csv_path, mode='a', newline='')
csv_writer = csv.writer(csv_file)
if new_csv:
    csv_writer.writerow(["left_image", "right_image", "distance_m"])

# === Initialize cameras ===
cam_left = Picamera2(1)
cam_right = Picamera2(0)
config_left = cam_left.create_preview_configuration(main={"size": (1920, 1080)})
config_right = cam_right.create_preview_configuration(main={"size": (1920, 1080)})
cam_left.configure(config_left)
cam_right.configure(config_right)
cam_left.start()
cam_right.start()

print("\nLive preview started.")
print("Press 's' to START/RESUME saving images at 1 per second.")
print("Press 'p' to PAUSE saving images.")
print("Press 'q' to STOP and quit.\n")

# === Draw crosshair ===
def draw_crosshair(frame):
    h, w, _ = frame.shape
    cx, cy = w // 2, h // 2
    cv2.line(frame, (cx - 20, cy), (cx + 20, cy), (0, 255, 0), 1)
    cv2.line(frame, (cx, cy - 20), (cx, cy + 20), (0, 255, 0), 1)
    return frame

# === Get next index for file naming ===
def get_next_index(output_dir):
    indices = []
    for fname in os.listdir(output_dir):
        if fname.startswith("left") and fname.endswith(".png"):
            digits = ''.join(filter(str.isdigit, fname))
            if digits:
                indices.append(int(digits))
    return max(indices) + 1 if indices else 1

# === Main loop ===
capturing = False
paused = False
last_capture_time = 0
img_index = get_next_index(output_dir)

try:
    while True:
        frame_left = cam_left.capture_array()
        frame_right = cam_right.capture_array()

        preview_left = draw_crosshair(frame_left.copy())
        preview_right = draw_crosshair(frame_right.copy())

        combined_preview = np.hstack((preview_left, preview_right))
        cv2.imshow("Stereo Preview (Left | Right)", combined_preview)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            if not capturing or paused:
                capturing = True
                paused = False
                print("Started/Resumed capturing images every 1 second.")
        elif key == ord('p'):
            if capturing and not paused:
                paused = True
                print("Paused capturing images.")
        elif key == ord('q'):
            print("Quitting.")
            break

        if capturing and not paused:
            current_time = time.time()
            if current_time - last_capture_time >= 1.0:
                filename_left = f"left{img_index:03d}.png"
                filename_right = f"right{img_index:03d}.png"
                path_left = os.path.join(output_dir, filename_left)
                path_right = os.path.join(output_dir, filename_right)

                # Save raw images
                Image.fromarray(frame_left).save(path_left)
                Image.fromarray(frame_right).save(path_right)

                # Log to CSV
                csv_writer.writerow([filename_left, filename_right, distance])
                csv_file.flush()

                print(f"Saved: {filename_left}, {filename_right} at {distance}m")
                last_capture_time = current_time
                img_index += 1

except KeyboardInterrupt:
    print("\nInterrupted by user.")

finally:
    cam_left.stop()
    cam_right.stop()
    csv_file.close()
    cv2.destroyAllWindows()