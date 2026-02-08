from picamera2 import Picamera2
import cv2
import numpy as np
from datetime import datetime
import time
import os
from ultralytics import YOLO
import json
import math

class DualCameraYOLO:
    def __init__(self, calibration_file="stereo_calibration4.json", resolution=(640, 480)):
        """
        Initialize stereo camera system with YOLO detection
        """
        self.resolution = resolution
        
        print("Loading calibration data...")
        # Load calibration parameters
        with open(calibration_file, 'r') as f:
            calib_data = json.load(f)
        
        # Extract calibration matrices
        self.K1 = np.array(calib_data['K1'])
        self.D1 = np.array(calib_data['D1'])
        self.K2 = np.array(calib_data['K2'])
        self.D2 = np.array(calib_data['D2'])
        self.R = np.array(calib_data['R'])
        self.T = np.array(calib_data['T'])
        self.R1 = np.array(calib_data['R1'])
        self.R2 = np.array(calib_data['R2'])
        self.P1 = np.array(calib_data['P1'])
        self.P2 = np.array(calib_data['P2'])
        self.Q = np.array(calib_data['Q'])
        
        # Get actual calibrated parameters
        self.baseline = abs(float(calib_data['metadata']['baseline_cm'])) / 100.0  # Convert to meters
        self.focal_length = float(calib_data['metadata']['focal_length_px'])
        
        print(f"✓ Calibration loaded: Baseline={self.baseline*100:.2f}cm, Focal={self.focal_length:.1f}px")
        
        print("Initializing cameras...")
        self.cam_left = Picamera2(0)
        self.cam_right = Picamera2(1)
        
        # Configure cameras
        config = {
            "size": self.resolution,
            "format": "RGB888",
        }
        
        self.cam_left.configure(self.cam_left.create_preview_configuration(main=config))
        self.cam_right.configure(self.cam_right.create_preview_configuration(main=config))
        
        print("Starting cameras...")
        self.cam_left.start()
        self.cam_right.start()
        
        # Load YOLO model
        print("Loading YOLO model...")
        try:
            self.model = YOLO("yolov8n.pt", task='detect')
            print("✓ Using YOLOv8n model")
        except Exception as e:
            print(f"⚠ Failed to load YOLO model: {e}")
            self.model = None
        
        # Detection parameters
        self.confidence_threshold = 0.5
        self.iou_threshold = 0.45
        
        # Generate rectification maps
        print("Creating rectification maps...")
        self.mapL1, self.mapL2 = cv2.initUndistortRectifyMap(
            self.K1, self.D1, self.R1, self.P1, self.resolution, cv2.CV_16SC2)
        self.mapR1, self.mapR2 = cv2.initUndistortRectifyMap(
            self.K2, self.D2, self.R2, self.P2, self.resolution, cv2.CV_16SC2)
        print("✓ Rectification maps created")
        
        # IMPROVED: Better stereo matcher for accurate depth
        print("Initializing stereo matcher...")
        self.stereo = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=96,      # Multiple of 16 for efficiency
            blockSize=5,
            P1=8 * 3 * 5**2,       # Adjusted for blockSize=5
            P2=32 * 3 * 5**2,
            disp12MaxDiff=5,
            uniquenessRatio=10,
            speckleWindowSize=50,
            speckleRange=16,
            preFilterCap=63,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )
        
        # For WLS filter (optional - improves disparity quality)
        try:
            import cv2.ximgproc
            self.wls_filter = cv2.ximgproc.createDisparityWLSFilter(self.stereo)
            self.right_matcher = cv2.ximgproc.createRightMatcher(self.stereo)
            self.use_wls = True
            print("✓ WLS filter enabled")
        except:
            self.use_wls = False
            print("⚠ WLS filter not available")
        
        # Depth parameters
        self.min_depth = 0.2  # meters (20cm minimum)
        self.max_depth = 15.0  # meters
        
        # Depth history for smoothing
        self.depth_history = {}
        self.history_size = 3
        
        print("System ready!")
    
    def rectify_frames(self, left_frame, right_frame):
        """Rectify stereo frames"""
        rect_left = cv2.remap(left_frame, self.mapL1, self.mapL2, cv2.INTER_LINEAR)
        rect_right = cv2.remap(right_frame, self.mapR1, self.mapR2, cv2.INTER_LINEAR)
        return rect_left, rect_right
    
    def capture_frames(self):
        """Capture frames from both cameras"""
        try:
            left_frame = self.cam_left.capture_array()
            right_frame = self.cam_right.capture_array()
            
            if left_frame is None or right_frame is None:
                return None, None
            
            # Convert to BGR for OpenCV
            left_frame = cv2.cvtColor(left_frame, cv2.COLOR_RGB2BGR)
            right_frame = cv2.cvtColor(right_frame, cv2.COLOR_RGB2BGR)
            
            return left_frame, right_frame
            
        except Exception as e:
            print(f"Capture error: {e}")
            return None, None
    
    def compute_disparity(self, rect_left, rect_right):
        """Compute disparity map between rectified frames"""
        # Convert to grayscale
        gray_left = cv2.cvtColor(rect_left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(rect_right, cv2.COLOR_BGR2GRAY)
        
        # Apply histogram equalization for better matching
        gray_left = cv2.equalizeHist(gray_left)
        gray_right = cv2.equalizeHist(gray_right)
        
        if self.use_wls:
            # Use WLS filter for better disparity
            left_disp = self.stereo.compute(gray_left, gray_right).astype(np.float32)
            right_disp = self.right_matcher.compute(gray_right, gray_left).astype(np.float32)
            filtered_disp = self.wls_filter.filter(left_disp, gray_left, None, right_disp)
            disparity = filtered_disp
        else:
            # Basic disparity calculation
            disparity = self.stereo.compute(gray_left, gray_right)
        
        # Convert to float and normalize
        disparity = disparity.astype(np.float32) / 16.0
        
        # Apply median filter to reduce noise
        disparity = cv2.medianBlur(disparity, 3)
        
        return disparity
    
    def detect_objects(self, image):
        """Detect objects using YOLO"""
        if self.model is None:
            return [], image
        
        try:
            results = self.model(image, conf=0.4, verbose=False)
            detections = []
            
            if results and results[0].boxes is not None:
                boxes = results[0].boxes.cpu().numpy()
                
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].astype(int)
                    confidence = box.conf[0]
                    class_id = int(box.cls[0])
                    class_name = self.model.names[class_id]
                    
                    # Calculate center
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    
                    # Only keep objects with reasonable size
                    width = x2 - x1
                    height = y2 - y1
                    if width < 20 or height < 20:
                        continue
                    
                    detections.append({
                        'bbox': (x1, y1, x2, y2),
                        'center': (center_x, center_y),
                        'class': class_name,
                        'confidence': confidence,
                        'width': width,
                        'height': height
                    })
            
            return detections, results[0].plot()
            
        except Exception as e:
            print(f"Detection error: {e}")
            return [], image
    
    def calculate_object_depth(self, disparity, bbox):
        """
        Calculate depth for a specific object using disparity within its bounding box
        """
        x1, y1, x2, y2 = bbox
        
        # Ensure bbox is within image bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(disparity.shape[1], x2)
        y2 = min(disparity.shape[0], y2)
        
        if x2 <= x1 or y2 <= y1:
            return None, 0.0
        
        # Extract disparity values in bounding box
        roi_disparity = disparity[y1:y2, x1:x2]
        
        # Filter out invalid disparities
        valid_mask = (roi_disparity > 0.5) & (roi_disparity < disparity.max() * 0.9)
        valid_disparities = roi_disparity[valid_mask]
        
        if len(valid_disparities) == 0:
            return None, 0.0
        
        # Use median for robustness against outliers
        median_disp = np.median(valid_disparities)
        
        # Calculate confidence
        confidence = len(valid_disparities) / (roi_disparity.size + 1e-6)
        
        # IMPORTANT: Calculate depth using formula: depth = (f * B) / disparity
        # where f is focal length, B is baseline, disparity is in pixels
        if median_disp > 0.5:  # Minimum disparity threshold
            # Depth in meters
            depth = (self.baseline * self.focal_length) / median_disp
            
            # Apply sanity checks
            if depth < self.min_depth or depth > self.max_depth:
                return None, confidence
            
            return depth, confidence
        
        return None, confidence
    
    def smooth_depth(self, object_id, new_depth):
        """Smooth depth measurements over time"""
        if object_id not in self.depth_history:
            self.depth_history[object_id] = []
        
        self.depth_history[object_id].append(new_depth)
        
        # Keep only recent measurements
        if len(self.depth_history[object_id]) > self.history_size:
            self.depth_history[object_id].pop(0)
        
        # Use weighted average (more weight to recent measurements)
        if len(self.depth_history[object_id]) > 0:
            weights = np.arange(1, len(self.depth_history[object_id]) + 1)
            smoothed = np.average(self.depth_history[object_id], weights=weights)
            return smoothed
        
        return new_depth
    
    def format_distance(self, depth):
        """Format distance for display"""
        if depth is None or depth < 0:
            return "N/A"
        elif depth < 1:
            return f"{depth*100:.0f}cm"
        else:
            return f"{depth:.1f}m"
    
    def visualize_disparity(self, disparity):
        """Create colored disparity map for visualization"""
        # Normalize for display
        disp_vis = np.clip(disparity, 0, self.stereo.getNumDisparities())
        disp_vis = cv2.normalize(disp_vis, None, 0, 255, cv2.NORM_MINMAX)
        disp_vis = disp_vis.astype(np.uint8)
        disp_colored = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)
        return disp_colored
    
    def run_detection_cycle(self):
        """Run one complete detection cycle"""
        # Capture frames
        left_frame, right_frame = self.capture_frames()
        if left_frame is None or right_frame is None:
            return None
        
        # Rectify frames
        rect_left, rect_right = self.rectify_frames(left_frame, right_frame)
        
        # Compute disparity
        disparity = self.compute_disparity(rect_left, rect_right)
        
        # Detect objects in left frame
        detections, annotated_left = self.detect_objects(rect_left)
        
        # Calculate depth for each object
        objects_with_depth = []
        for det in detections:
            depth, confidence = self.calculate_object_depth(disparity, det['bbox'])
            
            if depth is not None and confidence > 0.2:
                # Create unique object ID for smoothing
                obj_id = f"{det['class']}_{det['bbox'][0]}_{det['bbox'][1]}"
                
                # Smooth depth
                smoothed_depth = self.smooth_depth(obj_id, depth)
                
                # Add to results
                obj_data = det.copy()
                obj_data['depth'] = smoothed_depth
                obj_data['depth_confidence'] = confidence
                obj_data['raw_depth'] = depth
                objects_with_depth.append(obj_data)
                
                # Draw depth on image
                x1, y1, x2, y2 = det['bbox']
                depth_text = self.format_distance(smoothed_depth)
                cv2.putText(annotated_left, depth_text,
                           (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                           0.5, (0, 255, 0), 2)
        
        # Create disparity visualization
        disparity_viz = self.visualize_disparity(disparity)
        
        return {
            'original_left': left_frame,
            'original_right': right_frame,
            'rectified_left': rect_left,
            'rectified_right': rect_right,
            'annotated_left': annotated_left,
            'disparity': disparity,
            'disparity_viz': disparity_viz,
            'objects': objects_with_depth,
            'detections': detections
        }

def main():
    os.makedirs("stereo_output", exist_ok=True)
    
    print("=" * 60)
    print("STEREO VISION DEPTH ESTIMATION WITH YOLO")
    print("=" * 60)
    
    # Initialize system
    stereo_system = DualCameraYOLO(
        calibration_file="stereo_calibration4.json",
        resolution=(640, 480)
    )
    
    print("\nControls:")
    print("  [D] - Toggle disparity view")
    print("  [S] - Save current frame")
    print("  [R] - Reset depth history")
    print("  [1-5] - Adjust disparity range")
    print("  [Q] - Quit")
    print("=" * 60)
    
    # Display settings
    show_disparity = False
    disparity_scale = 96
    fps_history = []
    last_fps_time = time.time()
    
    try:
        while True:
            cycle_start = time.time()
            
            # Run detection cycle
            results = stereo_system.run_detection_cycle()
            if results is None:
                continue
            
            # Prepare display image
            if show_disparity:
                display_img = results['disparity_viz']
                cv2.putText(display_img, "DISPARITY MAP", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            else:
                display_img = results['annotated_left']
            
            # Calculate FPS
            cycle_time = time.time() - cycle_start
            fps = 1.0 / cycle_time if cycle_time > 0 else 0
            fps_history.append(fps)
            if len(fps_history) > 30:
                fps_history.pop(0)
            avg_fps = np.mean(fps_history) if fps_history else fps
            
            # Add overlay info
            overlay_text = []
            overlay_text.append(f"FPS: {avg_fps:.1f}")
            overlay_text.append(f"Mode: {'Disparity' if show_disparity else 'Detection'}")
            overlay_text.append(f"Baseline: {stereo_system.baseline*100:.1f}cm")
            overlay_text.append(f"Objects: {len(results['objects'])}")
            
            y_pos = 30
            for text in overlay_text:
                cv2.putText(display_img, text, (display_img.shape[1] - 200, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_pos += 20
            
            # Add depth measurements
            y_pos = 60
            for i, obj in enumerate(results['objects'][:5]):  # Show first 5
                depth_text = f"{obj['class']}: {stereo_system.format_distance(obj['depth'])}"
                cv2.putText(display_img, depth_text, (10, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                y_pos += 20
            
            # Display
            cv2.imshow("Stereo Vision Depth Estimation", display_img)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\nExiting...")
                break
                
            elif key == ord('d'):
                show_disparity = not show_disparity
                print(f"Switched to {'disparity' if show_disparity else 'detection'} view")
                
            elif key == ord('s'):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(f"stereo_output/capture_{timestamp}.jpg", display_img)
                print(f"Saved capture_{timestamp}.jpg")
                
            elif key == ord('r'):
                stereo_system.depth_history.clear()
                print("Depth history cleared")
                
            elif ord('1') <= key <= ord('5'):
                scale_idx = key - ord('0')
                disparity_scale = 16 * scale_idx * 3  # 48, 96, 144, 192, 240
                stereo_system.stereo.setNumDisparities(disparity_scale)
                print(f"Disparity scale: {disparity_scale}")
            
            # Print depth measurements periodically
            current_time = time.time()
            if current_time - last_fps_time > 0.5:
                if results['objects']:
                    for obj in results['objects']:
                        print(f"{obj['class']}: {stereo_system.format_distance(obj['depth'])} ", end="")
                    print()
                last_fps_time = current_time
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Cleaning up...")
        stereo_system.cam_left.stop()
        stereo_system.cam_right.stop()
        cv2.destroyAllWindows()
        print("Done!")

def test_calibration():
    """Test function to verify calibration accuracy"""
    print("\n=== CALIBRATION TEST ===")
    
    # Load calibration
    with open('stereo_calibration4.json', 'r') as f:
        calib_data = json.load(f)
    
    baseline = calib_data['metadata']['baseline_cm'] / 100.0
    focal_length = calib_data['metadata']['focal_length_px']
    
    print(f"Baseline: {baseline*100:.2f} cm")
    print(f"Focal length: {focal_length:.1f} px")
    print(f"Stereo RMS error: {calib_data['metadata']['stereo_rms_error']:.4f}")
    
    # Calculate expected disparities at different distances
    print("\nExpected disparities at different distances:")
    print("Distance | Expected Disparity")
    print("---------|-------------------")
    
    distances = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]  # meters
    for dist in distances:
        expected_disparity = (baseline * focal_length) / dist
        print(f"  {dist:.1f}m   |  {expected_disparity:.1f} px")
    
    print("\nIf close objects show high depth (small disparity), check:")
    print("1. Calibration accuracy (RMS error should be < 1.0)")
    print("2. Cameras are properly aligned")
    print("3. Baseline measurement is correct (7.13cm)")
    print("4. Focal length seems reasonable (~1000px for 640x480)")

if __name__ == "__main__":
    # Run calibration test first
    if os.path.exists('stereo_calibration4.json'):
        test_calibration()
    
    # Run main program
    main()