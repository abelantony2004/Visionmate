import cv2
import numpy as np
import glob
import os

# === CONFIGURATION ===
chessboard_size = (8, 6)  # inner corners (width, height)
square_size = 0.036  # in meters (36mm)

# === Paths ===
calib_dir = "calib"
print(f"Looking for images in: {os.path.abspath(calib_dir)}")

# List all files in directory
print("\nAll files in calib directory:")
all_files = os.listdir(calib_dir)
for f in all_files:
    print(f"  - {f}")

# Try different patterns to find images
left_patterns = [
    os.path.join(calib_dir, "left*.*"),
    os.path.join(calib_dir, "*left*.*"),
    os.path.join(calib_dir, "*.png"),
    os.path.join(calib_dir, "*.jpg"),
    os.path.join(calib_dir, "*.jpeg")
]

right_patterns = [
    os.path.join(calib_dir, "right*.*"),
    os.path.join(calib_dir, "*right*.*"),
    os.path.join(calib_dir, "*.png"),
    os.path.join(calib_dir, "*.jpg"),
    os.path.join(calib_dir, "*.jpeg")
]

left_images = []
right_images = []

# Try each pattern
for pattern in left_patterns:
    files = glob.glob(pattern)
    if files:
        left_images = sorted(files)
        print(f"Found left images with pattern '{pattern}': {len(files)} files")
        break

for pattern in right_patterns:
    files = glob.glob(pattern)
    if files:
        right_images = sorted(files)
        print(f"Found right images with pattern '{pattern}': {len(files)} files")
        break

print(f"\nLeft images found: {len(left_images)}")
for img in left_images:
    print(f"  - {img}")

print(f"\nRight images found: {len(right_images)}")
for img in right_images:
    print(f"  - {img}")

if len(left_images) == 0 or len(right_images) == 0:
    print("\n❌ No calibration images found!")
    print("\nPossible solutions:")
    print("1. Make sure you have a folder named 'calib' in the same directory as this script")
    print("2. Put your images in that folder with names like:")
    print("   - left_001.png, left_002.png, etc.")
    print("   - right_001.png, right_002.png, etc.")
    print("3. Check if images are .png, .jpg, or .jpeg format")
    exit()

# Make sure we have matching number of images
if len(left_images) != len(right_images):
    print(f"\n⚠ Warning: Mismatched number of images (Left: {len(left_images)}, Right: {len(right_images)})")
    print("Using the minimum number of pairs...")
    min_pairs = min(len(left_images), len(right_images))
    left_images = left_images[:min_pairs]
    right_images = right_images[:min_pairs]

# === Prepare object points ===
objp = np.zeros((chessboard_size[0]*chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
objp *= square_size

objpoints = []  # 3D points in real world space
imgpoints_left = []  # 2D points in left images
imgpoints_right = []  # 2D points in right images

print("\n=== Detecting chessboard corners ===")
successful_pairs = 0

for i, (left_img_path, right_img_path) in enumerate(zip(left_images, right_images)):
    print(f"\nProcessing pair {i+1}/{len(left_images)}:")
    print(f"  Left: {os.path.basename(left_img_path)}")
    print(f"  Right: {os.path.basename(right_img_path)}")
    
    # Load images with error checking
    imgL = cv2.imread(left_img_path)
    imgR = cv2.imread(right_img_path)
    
    if imgL is None:
        print(f"  ❌ Failed to load left image: {left_img_path}")
        continue
    if imgR is None:
        print(f"  ❌ Failed to load right image: {right_img_path}")
        continue
    
    # Check image sizes
    if imgL.shape != imgR.shape:
        print(f"  ⚠ Image size mismatch: Left {imgL.shape}, Right {imgR.shape}")
    
    # Convert to grayscale
    grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
    
    # Detect chessboard corners
    retL, cornersL = cv2.findChessboardCorners(grayL, chessboard_size, 
                                              flags=cv2.CALIB_CB_ADAPTIVE_THRESH + 
                                                    cv2.CALIB_CB_NORMALIZE_IMAGE)
    
    retR, cornersR = cv2.findChessboardCorners(grayR, chessboard_size,
                                              flags=cv2.CALIB_CB_ADAPTIVE_THRESH + 
                                                    cv2.CALIB_CB_NORMALIZE_IMAGE)
    
    if retL and retR:
        # Refine corner positions
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        cornersL = cv2.cornerSubPix(grayL, cornersL, (11, 11), (-1, -1), criteria)
        cornersR = cv2.cornerSubPix(grayR, cornersR, (11, 11), (-1, -1), criteria)
        
        objpoints.append(objp)
        imgpoints_left.append(cornersL)
        imgpoints_right.append(cornersR)
        
        successful_pairs += 1
        print(f"  ✓ Successfully detected corners in both images")
        
        # Visualize detection
        imgL_detected = imgL.copy()
        imgR_detected = imgR.copy()
        cv2.drawChessboardCorners(imgL_detected, chessboard_size, cornersL, retL)
        cv2.drawChessboardCorners(imgR_detected, chessboard_size, cornersR, retR)
        
        # Display small preview
        preview = np.hstack((cv2.resize(imgL_detected, (320, 240)), 
                            cv2.resize(imgR_detected, (320, 240))))
        cv2.imshow(f"Pair {i+1} - Press any key to continue", preview)
        cv2.waitKey(500)  # Show for 0.5 seconds
        cv2.destroyWindow(f"Pair {i+1} - Press any key to continue")
    else:
        print(f"  ✗ Failed to detect corners: Left={retL}, Right={retR}")

cv2.destroyAllWindows()

# Check if we have enough successful pairs
if successful_pairs < 8:
    print(f"\n❌ Error: Only {successful_pairs} successful pairs. Need at least 8.")
    print("\nTips to improve detection:")
    print("1. Make sure chessboard is fully visible in all images")
    print("2. Ensure good lighting without shadows")
    print("3. Try different angles (flat, tilted)")
    print("4. Make sure chessboard fills most of the frame")
    exit()

print(f"\n✓ Successfully processed {successful_pairs}/{len(left_images)} image pairs")

# === Calibration ===
print("\n=== Calibrating cameras ===")

# Get image size from first successful image
img_size = grayL.shape[::-1]  # (width, height)

# Calibrate individual cameras
print("Calibrating left camera...")
retL, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(
    objpoints, imgpoints_left, img_size, None, None,
    flags=cv2.CALIB_FIX_K3 + cv2.CALIB_ZERO_TANGENT_DIST)

print("Calibrating right camera...")
retR, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(
    objpoints, imgpoints_right, img_size, None, None,
    flags=cv2.CALIB_FIX_K3 + cv2.CALIB_ZERO_TANGENT_DIST)

print(f"Left camera RMS error: {retL:.4f}")
print(f"Right camera RMS error: {retR:.4f}")

# Stereo calibration
print("\nPerforming stereo calibration...")
flags = cv2.CALIB_FIX_INTRINSIC
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)

retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = cv2.stereoCalibrate(
    objpoints, imgpoints_left, imgpoints_right,
    mtxL, distL, mtxR, distR,
    img_size,
    criteria=criteria,
    flags=flags
)

print(f"Stereo RMS error: {retval:.4f}")

# === Rectification ===
print("\n=== Computing rectification ===")
R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
    cameraMatrix1, distCoeffs1,
    cameraMatrix2, distCoeffs2,
    img_size, R, T, 
    alpha=0,  # Keep all pixels
    newImageSize=img_size
)

baseline = np.linalg.norm(T)
print(f"Baseline: {baseline*100:.2f} cm")

# === Save Calibration Data ===
print("\n=== Saving calibration data ===")

np.savez("stereo_calib.npz",
         cameraMatrix1=cameraMatrix1, distCoeffs1=distCoeffs1,
         cameraMatrix2=cameraMatrix2, distCoeffs2=distCoeffs2,
         R=R, T=T, E=E, F=F, 
         R1=R1, R2=R2, P1=P1, P2=P2, Q=Q,
         img_size=img_size,
         baseline_cm=baseline*100,
         rms_error=retval)

print("✅ Saved calibration data to: stereo_calib.npz")

# === Quick Verification ===
print("\n=== Quick verification ===")

# Load one pair for testing
test_imgL = cv2.imread(left_images[0])
test_imgR = cv2.imread(right_images[0])

# Create rectification maps
mapL1, mapL2 = cv2.initUndistortRectifyMap(cameraMatrix1, distCoeffs1, R1, P1, img_size, cv2.CV_32FC1)
mapR1, mapR2 = cv2.initUndistortRectifyMap(cameraMatrix2, distCoeffs2, R2, P2, img_size, cv2.CV_32FC1)

# Rectify images
rectL = cv2.remap(test_imgL, mapL1, mapL2, cv2.INTER_LINEAR)
rectR = cv2.remap(test_imgR, mapR1, mapR2, cv2.INTER_LINEAR)

# Display rectified images
display_rect = np.hstack((cv2.resize(rectL, (320, 240)), 
                         cv2.resize(rectR, (320, 240))))

# Draw horizontal lines to check alignment
for y in range(0, display_rect.shape[0], 40):
    cv2.line(display_rect, (0, y), (display_rect.shape[1], y), 
             (0, 255, 0), 1)

cv2.putText(display_rect, "Rectified - Check if lines are horizontal", 
           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

cv2.imshow("Rectification Check", display_rect)

# Create summary
summary = np.zeros((300, 800, 3), dtype=np.uint8)
summary_text = f"""
=== CALIBRATION SUMMARY ===
Successful image pairs: {successful_pairs}
Image size: {img_size[0]}x{img_size[1]}
Left RMS error: {retL:.4f}
Right RMS error: {retR:.4f}
Stereo RMS error: {retval:.4f}
Baseline: {baseline*100:.2f} cm

Files saved:
• stereo_calib.npz (OpenCV format)

Press any key to exit...
"""

y_pos = 40
for line in summary_text.strip().split('\n'):
    cv2.putText(summary, line, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 
               0.5, (255, 255, 255), 1)
    y_pos += 25

cv2.imshow("Calibration Summary", summary)

print("\n=== CALIBRATION COMPLETE ===")
print(f"• Used {successful_pairs} image pairs")
print(f"• Stereo RMS error: {retval:.4f}")
print(f"• Baseline: {baseline*100:.2f} cm")
print("\nCheck the rectified image - horizontal lines should align.")
print("Press any key to exit...")

cv2.waitKey(0)
cv2.destroyAllWindows()
print("\n✅ Ready for stereo vision!")