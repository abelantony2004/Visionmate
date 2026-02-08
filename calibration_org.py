import cv2
import numpy as np
from picamera2 import Picamera2
import os
import time
import json
from datetime import datetime

os.makedirs("calibration", exist_ok=True)

print("\n" + "="*50)
print("STEREO CALIBRATION - OPTIMIZED WITH INCREASED CAPTURE TIME")
print("="*50 + "\n")

# ---------------------------
# OPTIMIZED SETTINGS
# ---------------------------
chessboard_size = (8, 6)   # inner corners
square_size = 0.036         # meters (3.5 cm)
target_images = 70# Number of calibration images

# INCREASED CAPTURE TIME SETTINGS
min_image_interval = 1.5    # ⚡ INCREASED from 0.3 to 1.5 seconds
capture_delay = 0.5         # Delay after detection before capture
stability_check_frames = 3  # Check stability over multiple frames

# OPTIMIZED RESOLUTION
resolution = (640, 480)     # Optimal for Pi 5 with dual cameras
use_fast_detection = True   # Enable faster chessboard detection
downscale_display = True    # Display at lower resolution

# ---------------------------
# Initialize Cameras
# ---------------------------
print(f"Initializing cameras at {resolution[0]}x{resolution[1]}...")

left_cam = Picamera2(0)
right_cam = Picamera2(1)

config = left_cam.create_preview_configuration(
    main={"size": resolution, "format": "RGB888"},
    controls={"FrameRate": 30}
)

left_cam.configure(config)
right_cam.configure(config)

left_cam.start()
right_cam.start()
time.sleep(2)

print("Cameras ready!\n")

# ---------------------------
# Prepare object points
# ---------------------------
objp = np.zeros((np.prod(chessboard_size), 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
objp *= square_size

objpoints = []
imgpointsL = []
imgpointsR = []

count = 0
last_capture_time = 0
stability_buffer = []  # Store recent detections for stability check

print(f"Target: {target_images} stereo pairs at {resolution[0]}x{resolution[1]}")
print(f"Minimum capture interval: {min_image_interval} seconds")
print("\nControls:")
print("  S - Capture manually")
print("  A - Toggle auto capture (currently ON)")
print("  F - Toggle fast detection (currently ON)")
print("  Q - Quit and calibrate\n")

auto_capture = True
capture_countdown = 0
countdown_start_time = 0

def fast_find_chessboard(gray, chessboard_size):
    """Faster chessboard detection with optimization."""
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, 
                                            flags=cv2.CALIB_CB_FAST_CHECK)
    
    if ret and use_fast_detection:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 15, 0.01)
        corners = cv2.cornerSubPix(gray, corners, (7, 7), (-1, -1), criteria)
    elif ret:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    
    return ret, corners

def check_stereo_consistency(cornersL, cornersR):
    """Check if stereo corners are consistent with adaptive threshold."""
    if cornersL is None or cornersR is None:
        return False, "No corners"
    
    if len(cornersL) != len(cornersR):
        return False, "Corner count mismatch"
    
    y_diffs = np.abs(cornersL[:, 0, 1] - cornersR[:, 0, 1])
    avg_y_diff = np.mean(y_diffs)
    max_diff = np.max(y_diffs)
    
    # Adaptive threshold based on image height
    max_y_diff = resolution[1] * 0.08  # 8% of image height
    
    if avg_y_diff > max_y_diff:
        return False, f"Misaligned ({avg_y_diff:.0f}px)"
    
    return True, f"Aligned ({avg_y_diff:.0f}px)"

def check_stability(cornersL, cornersR, buffer_size=3):
    """Check if chessboard position is stable over multiple frames."""
    if len(stability_buffer) < buffer_size:
        stability_buffer.append((cornersL.copy() if cornersL is not None else None,
                               cornersR.copy() if cornersR is not None else None))
        return False
    
    # Remove oldest entry if buffer is full
    if len(stability_buffer) >= buffer_size:
        stability_buffer.pop(0)
    stability_buffer.append((cornersL.copy() if cornersL is not None else None,
                           cornersR.copy() if cornersR is not None else None))
    
    # Check if we have enough frames
    if len(stability_buffer) < buffer_size:
        return False
    
    # Calculate average position change
    total_change = 0
    valid_frames = 0
    
    for i in range(1, len(stability_buffer)):
        prevL, prevR = stability_buffer[i-1]
        currL, currR = stability_buffer[i]
        
        if prevL is not None and currL is not None:
            changeL = np.mean(np.abs(prevL - currL))
            changeR = np.mean(np.abs(prevR - currR))
            total_change += (changeL + changeR) / 2
            valid_frames += 1
    
    if valid_frames == 0:
        return False
    
    avg_change = total_change / valid_frames
    return avg_change < 2.0  # Less than 2 pixels average movement

def resize_for_display(img, scale=0.5):
    """Resize image for display to reduce processing."""
    if downscale_display:
        return cv2.resize(img, (0, 0), fx=scale, fy=scale)
    return img

try:
    while count < target_images:
        frame_start = time.time()
        
        # Capture frames
        frameL = left_cam.capture_array()
        frameR = right_cam.capture_array()
        
        # Convert to BGR
        frameL = cv2.cvtColor(frameL, cv2.COLOR_RGB2BGR)
        frameR = cv2.cvtColor(frameR, cv2.COLOR_RGB2BGR)
        
        # Convert to grayscale
        grayL = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)
        
        # Find chessboard corners
        retL, cornersL = fast_find_chessboard(grayL, chessboard_size)
        retR, cornersR = fast_find_chessboard(grayR, chessboard_size)
        
        # Prepare display images
        if downscale_display:
            displayL = resize_for_display(frameL, 0.7)
            displayR = resize_for_display(frameR, 0.7)
        else:
            displayL = frameL.copy()
            displayR = frameR.copy()
        
        # Draw corners if found
        if retL:
            cv2.drawChessboardCorners(displayL, chessboard_size, 
                                     cornersL * (0.7 if downscale_display else 1.0), 
                                     retL)
        if retR:
            cv2.drawChessboardCorners(displayR, chessboard_size, 
                                     cornersR * (0.7 if downscale_display else 1.0), 
                                     retR)
        
        # Combine display
        display = np.hstack((displayL, displayR))
        
        # Add status overlay
        status = f"Captured: {count}/{target_images} | Interval: {min_image_interval}s"
        cv2.putText(display, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6, (255, 255, 255), 2)
        
        # Check alignment and stability
        ready_for_capture = False
        alignment_status = ""
        status_color = (0, 0, 255)  # Default red
        
        if retL and retR:
            is_aligned, alignment_msg = check_stereo_consistency(cornersL, cornersR)
            is_stable = check_stability(cornersL, cornersR, stability_check_frames)
            
            if is_aligned:
                if is_stable:
                    alignment_status = f"READY ✓ {alignment_msg}"
                    status_color = (0, 255, 0)  # Green
                    ready_for_capture = True
                else:
                    alignment_status = f"MOVING ✗ {alignment_msg}"
                    status_color = (0, 165, 255)  # Orange
            else:
                alignment_status = f"MISALIGNED ✗ {alignment_msg}"
                status_color = (0, 165, 255)  # Orange
        else:
            what_missing = []
            if not retL: what_missing.append("LEFT")
            if not retR: what_missing.append("RIGHT")
            alignment_status = f"MISSING: {','.join(what_missing)}"
            status_color = (0, 0, 255)  # Red
        
        # Display alignment status
        cv2.putText(display, alignment_status, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        
        # Handle countdown for auto-capture
        current_time = time.time()
        
        if ready_for_capture and auto_capture and (current_time - last_capture_time > min_image_interval):
            if capture_countdown == 0:
                # Start countdown
                capture_countdown = int(capture_delay * 10)  # 10 ticks per second
                countdown_start_time = current_time
            
            # Update countdown
            elapsed = current_time - countdown_start_time
            remaining = max(0, capture_delay - elapsed)
            
            if remaining > 0:
                # Show countdown
                countdown_text = f"Capturing in: {remaining:.1f}s"
                cv2.putText(display, countdown_text, (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                # Draw countdown circle
                center_x = display.shape[1] - 40
                center_y = 40
                radius = 20
                angle = int(360 * (remaining / capture_delay))
                
                cv2.circle(display, (center_x, center_y), radius, (255, 255, 0), 2)
                cv2.ellipse(display, (center_x, center_y), (radius, radius), 
                           0, -90, angle-90, (255, 255, 0), 3)
            
            if remaining <= 0:
                # CAPTURE NOW!
                objpoints.append(objp.copy())
                imgpointsL.append(cornersL)
                imgpointsR.append(cornersR)
                
                # Save images
                cv2.imwrite(f"calibration/left_{count:03d}.png", frameL)
                cv2.imwrite(f"calibration/right_{count:03d}.png", frameR)
                
                print(f"✓ Auto-captured pair {count + 1} at {time.strftime('%H:%M:%S')}")
                count += 1
                last_capture_time = current_time
                capture_countdown = 0
                
                # Visual feedback
                cv2.rectangle(display, (0, 0), 
                            (display.shape[1], display.shape[0]), 
                            (0, 255, 0), 5)
                cv2.imshow("Stereo Calibration", display)
                cv2.waitKey(200)  # Show green flash
                
                # Clear stability buffer after capture
                stability_buffer.clear()
        else:
            capture_countdown = 0
        
        # Mode indicators
        mode_text = f"AUTO: {'ON' if auto_capture else 'OFF'} | FAST: {'ON' if use_fast_detection else 'OFF'}"
        cv2.putText(display, mode_text, (10, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 255), 1)
        
        # Show remaining time until next possible capture
        if auto_capture and last_capture_time > 0:
            time_since_last = current_time - last_capture_time
            if time_since_last < min_image_interval:
                time_remaining = min_image_interval - time_since_last
                wait_text = f"Next capture in: {time_remaining:.1f}s"
                cv2.putText(display, wait_text, (10, 150), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 255), 1)
        
        # Show display
        cv2.imshow("Stereo Calibration - Increased Capture Time", display)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('s') and ready_for_capture:
            # Manual capture
            objpoints.append(objp.copy())
            imgpointsL.append(cornersL)
            imgpointsR.append(cornersR)
            
            cv2.imwrite(f"calibration/left_{count:03d}.png", frameL)
            cv2.imwrite(f"calibration/right_{count:03d}.png", frameR)
            
            print(f"✓ Manual capture {count + 1} at {time.strftime('%H:%M:%S')}")
            count += 1
            last_capture_time = time.time()
            capture_countdown = 0
            stability_buffer.clear()
            
        elif key == ord('a'):
            auto_capture = not auto_capture
            capture_countdown = 0
            print(f"Auto-capture {'ON' if auto_capture else 'OFF'}")
            
        elif key == ord('f'):
            use_fast_detection = not use_fast_detection
            print(f"Fast detection {'ON' if use_fast_detection else 'OFF'}")
            
        elif key == ord('q'):
            print("\nEarly termination requested.")
            if len(objpoints) >= 8:
                print(f"Proceeding with {len(objpoints)} captured pairs.")
                break
            else:
                print(f"Need at least 8 pairs, currently {len(objpoints)}. Continuing...")
        
except KeyboardInterrupt:
    print("\nCalibration interrupted by user.")
except Exception as e:
    print(f"\nError: {e}")
    import traceback
    traceback.print_exc()

cv2.destroyAllWindows()
left_cam.stop()
right_cam.stop()

# Check if we have enough images
min_pairs = 8
if len(objpoints) < min_pairs:
    print(f"\n❌ Error: Only {len(objpoints)} pairs captured. Need at least {min_pairs}.")
    exit(1)

print(f"\n{'='*50}")
print(f"CALIBRATING WITH {len(objpoints)} IMAGE PAIRS")
print(f"Resolution: {resolution[0]}x{resolution[1]}")
print(f"Average interval: {min_image_interval} seconds")
print('='*50)

# ---------------------------
# Calibration
# ---------------------------
img_size = grayL.shape[::-1]

print("\n1. Calibrating cameras...")
print("   (This may take a minute...)")

calib_flags = (cv2.CALIB_FIX_K3 + cv2.CALIB_ZERO_TANGENT_DIST + 
               cv2.CALIB_USE_LU)

retL, K1, D1, rvecsL, tvecsL = cv2.calibrateCamera(
    objpoints, imgpointsL, img_size, None, None,
    flags=calib_flags)

retR, K2, D2, rvecsR, tvecsR = cv2.calibrateCamera(
    objpoints, imgpointsR, img_size, None, None,
    flags=calib_flags)

print("\n2. Stereo calibration...")
stereo_flags = cv2.CALIB_FIX_INTRINSIC
criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-6)

ret, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
    objpoints, imgpointsL, imgpointsR,
    K1, D1, K2, D2, img_size,
    criteria=criteria, flags=stereo_flags)

print(f"   Stereo RMS error: {ret:.4f}")

# ---------------------------
# Stereo Rectification
# ---------------------------
print("\n3. Computing rectification...")
R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
    K1, D1, K2, D2, img_size, R, T,
    flags=cv2.CALIB_ZERO_DISPARITY,
    alpha=0,
    newImageSize=img_size)

baseline = float(np.linalg.norm(T))
fx = P1[0, 0]

print(f"   Baseline: {baseline*100:.2f} cm")
print(f"   Focal length: {fx:.1f} px")
print(f"   Translation vector: [{T[0,0]:.3f}, {T[1,0]:.3f}, {T[2,0]:.3f}]")
print(f"   P1 principal x: {P1[0,2]:.3f}, P2 principal x: {P2[0,2]:.3f}, image center x: {img_size[0]/2:.1f}")

# ---------------------------
# Save Calibration Data
# ---------------------------
print("\n4. Saving calibration data...")

calibration_data = {
    'metadata': {
        'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'num_images': len(objpoints),
        'image_size': img_size,
        'chessboard_size': chessboard_size,
        'square_size': square_size,
        'stereo_rms_error': float(ret),
        'resolution': resolution,
        'baseline_cm': float(baseline*100),
        'focal_length_px': float(fx),
        'capture_interval': min_image_interval,
        'translation_vector': T.flatten().tolist(),
    },
    'K1': K1.tolist(), 'D1': D1.tolist(),
    'K2': K2.tolist(), 'D2': D2.tolist(),
    'R': R.tolist(), 'T': T.tolist(),
    'R1': R1.tolist(), 'R2': R2.tolist(),
    'P1': P1.tolist(), 'P2': P2.tolist(),
    'Q': Q.tolist(),
}

np.save("stereo_calibration4.npy", calibration_data)

with open("stereo_calibration4.json", "w") as f:
    json.dump(calibration_data, f, indent=2)

print("   ✅ stereo_calibration4.npy (OpenCV)")
print("   ✅ stereo_calibration4.json (readable)")

# ---------------------------
# Quick Verification
# ---------------------------
print("\n5. Quick verification...")

# Capture one test frame
left_cam.start()
right_cam.start()
time.sleep(0.5)

testL = cv2.cvtColor(left_cam.capture_array(), cv2.COLOR_RGB2BGR)
testR = cv2.cvtColor(right_cam.capture_array(), cv2.COLOR_RGB2BGR)

left_cam.stop()
right_cam.stop()

# Create maps
mapL1, mapL2 = cv2.initUndistortRectifyMap(K1, D1, R1, P1, img_size, cv2.CV_16SC2)
mapR1, mapR2 = cv2.initUndistortRectifyMap(K2, D2, R2, P2, img_size, cv2.CV_16SC2)

# Rectify
rectL = cv2.remap(testL, mapL1, mapL2, cv2.INTER_LINEAR)
rectR = cv2.remap(testR, mapR1, mapR2, cv2.INTER_LINEAR)

# Display results
display_orig = np.hstack((resize_for_display(testL), resize_for_display(testR)))
display_rect = np.hstack((resize_for_display(rectL), resize_for_display(rectR)))

# Add epipolar lines
for y in range(0, display_rect.shape[0], 30):
    cv2.line(display_rect, (0, y), (display_rect.shape[1], y), 
             (0, 255, 0) if (y//30)%2==0 else (255, 0, 0), 1)

cv2.imshow("Original (Resized)", display_orig)
cv2.putText(display_rect, "RECTIFIED - Lines should be horizontal", 
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
cv2.imshow("Rectified with Epipolar Lines", display_rect)

# Create summary image
summary = np.zeros((300, 800, 3), dtype=np.uint8)
summary_text = f"""
=== CALIBRATION COMPLETE ===
Images: {len(objpoints)} | Resolution: {resolution[0]}x{resolution[1]}
RMS Error: {ret:.4f} | Baseline: {baseline*100:.2f} cm
Focal Length: {fx:.1f} px

Files saved:
• stereo_calibration4.npy (OpenCV)
• stereo_calibration4.json (readable)

Press any key to exit...
"""

y_pos = 40
for line in summary_text.strip().split('\n'):
    cv2.putText(summary, line, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 
               0.5, (255, 255, 255), 1)
    y_pos += 25

cv2.imshow("Calibration Summary", summary)

print("\n" + "="*50)
print("CALIBRATION COMPLETE!")
print("="*50)
print(f"\nImages used: {len(objpoints)}")
print(f"Baseline: {baseline*100:.2f} cm")
print(f"RMS Error: {ret:.4f}")
print(f"\nCheck the rectified image - epipolar lines should be horizontal.")
print("Press any key to exit...")

cv2.waitKey(0)
cv2.destroyAllWindows()

print("\n✅ Ready for depth mapping!")
print("Use 'stereo_calibration4.npy' with OpenCV.\n")
