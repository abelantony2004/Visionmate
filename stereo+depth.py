import cv2
import numpy as np
from picamera2 import Picamera2
import time
import os
from datetime import datetime

# Try to import YOLO
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️ YOLO not available. Using OpenCV detection.")
    print("Install: pip install ultralytics")

class StereoYOLODepth:
    def __init__(self, calibration_file='stereo_calib3.npz'):
        """
        Complete Stereo Vision System with YOLO Object Detection and Depth Measurement
        Uses your calibration (8x6 chessboard, 36mm squares)
        """
        
        print("=" * 60)
        print("STEREO VISION + YOLO + DEPTH MEASUREMENT")
        print("Calibration: 8x6 chessboard | 36mm squares")  # Updated
        print("Baseline: 7.4 cm")
        print("=" * 60)
        
        # Optimize performance with frame skipping
        self.yolo_interval = 5        # Run YOLO every 5th frame
        self.depth_interval = 2       # Compute depth every 2nd frame
        
        # Initialize caches
        self.last_detections = []
        self.last_depth = None
        self.last_disp = None
        self.last_left_rect = None
        
        # 1. Load calibration data
        print("\n1. Loading calibration data...")
        if not os.path.exists(calibration_file):
            print(f"❌ ERROR: {calibration_file} not found!")
            print("Run calibration first: python stereo_calibrate.py")
            raise FileNotFoundError
        
        calib_data = np.load(calibration_file)
        
        # Extract calibration parameters
        self.K1 = calib_data['cameraMatrix1']
        self.D1 = calib_data['distCoeffs1']
        self.K2 = calib_data['cameraMatrix2']
        self.D2 = calib_data['distCoeffs2']
        self.R1 = calib_data['R1']
        self.R2 = calib_data['R2']
        self.P1 = calib_data['P1']
        self.P2 = calib_data['P2']
        self.Q = calib_data['Q']
        self.R = calib_data['R']
        self.T = calib_data['T']
        
        # Calculate baseline from translation vector
        self.baseline_m = np.linalg.norm(self.T)
        
        print(f"   ✓ Calibration loaded from {calibration_file}")
        print(f"   Baseline: {self.baseline_m*100:.1f} cm")
        
        # 2. Initialize cameras
        print("\n2. Initializing cameras...")
        self.cam_left = Picamera2(0)
        self.cam_right = Picamera2(1)
        
        # Use calibrated resolution
        self.img_size = (640, 480)
        
        config = self.cam_left.create_preview_configuration(
            main={"size": self.img_size, "format": "RGB888"})
        self.cam_left.configure(config)
        self.cam_right.configure(config)
        
        self.cam_left.start()
        self.cam_right.start()
        time.sleep(2)
        print(f"   ✓ Cameras ready at {self.img_size[0]}x{self.img_size[1]}")
        
        # 3. Create rectification maps
        print("\n3. Creating rectification maps...")
        self.setup_rectification()
        
        # 4. Setup stereo matcher (SGBM) - optimized for 7.4cm baseline
        print("\n4. Setting up stereo matcher...")
        self.setup_stereo_matcher()
        
        # 5. Load YOLO model
        print("\n5. Loading YOLO model...")
        self.setup_yolo()
        
        # 6. Color mapping for different object classes
        self.class_colors = {
            'person': (0, 255, 0),      # Green
            'car': (255, 0, 0),         # Blue
            'bicycle': (0, 255, 255),   # Yellow
            'motorbike': (255, 255, 0), # Cyan
            'bus': (0, 165, 255),       # Orange
            'truck': (128, 0, 128),     # Purple
            'chair': (255, 0, 255),     # Magenta
            'bottle': (255, 255, 255),  # White
            'cup': (0, 0, 255),         # Red
            'book': (139, 69, 19),      # Brown
            'cell phone': (255, 215, 0), # Gold
            'laptop': (70, 130, 180),   # Steel Blue
        }
        self.default_color = (200, 200, 200)
        
        # 7. Display configuration
        self.display_size = 320  # Size for each quadrant
        self.frame_count = 0
        self.fps_history = []
        self.last_print_time = time.time()   # for 1-second reporting

        
        # Initialize with placeholder data
        self.last_depth = np.zeros((self.img_size[1], self.img_size[0]), dtype=np.float32)
        self.last_disp = np.zeros((self.img_size[1], self.img_size[0]), dtype=np.float32)
        
        print("\n" + "=" * 60)
        print("✅ SYSTEM INITIALIZED")
        print("=" * 60)
    
    def setup_rectification(self):
        """Setup rectification maps from calibration"""
        try:
            # Create rectification maps
            self.map1_left, self.map2_left = cv2.initUndistortRectifyMap(
                self.K1, self.D1, self.R1, self.P1, 
                self.img_size, cv2.CV_16SC2)
            
            self.map1_right, self.map2_right = cv2.initUndistortRectifyMap(
                self.K2, self.D2, self.R2, self.P2, 
                self.img_size, cv2.CV_16SC2)
            
            print("   ✓ Rectification maps created")
            
        except Exception as e:
            print(f"❌ Error creating rectification maps: {e}")
            raise
    
    def setup_stereo_matcher(self):
        """Setup stereo matcher (SGBM) optimized for 7.4cm baseline"""
        self.stereo = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=160,      # ↑ more range for near objects
            blockSize=9,             # ↑ smoother matching
            P1=8 * 3 * 9**2,
            P2=32 * 3 * 9**2,
            disp12MaxDiff=1,
            uniquenessRatio=5,       # ↓ allow more matches
            speckleWindowSize=100,
            speckleRange=32,
            preFilterCap=63,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )

        print("   ✓ Stereo matcher ready (optimized for 7.4cm baseline)")
    
    def setup_yolo(self):
        """Setup YOLO model"""
        if YOLO_AVAILABLE:
            try:
                self.model = YOLO('yolov8n_ncnn_model')
                self.use_yolo = True
                print(f"   ✓ YOLOv8n loaded - {len(self.model.names)} classes")
                    
            except Exception as e:
                print(f"   ⚠️ YOLO failed: {e}")
                self.model = None
                self.use_yolo = False
                print("   Using OpenCV face detection as fallback")
        else:
            self.model = None
            self.use_yolo = False
    
    def capture_frames(self):
        """Capture frames from both cameras"""
        left_frame = self.cam_left.capture_array()
        right_frame = self.cam_right.capture_array()
        
        # Convert from RGB to BGR for OpenCV
        left_bgr = cv2.cvtColor(left_frame, cv2.COLOR_RGB2BGR)
        right_bgr = cv2.cvtColor(right_frame, cv2.COLOR_RGB2BGR)
        
        return left_bgr, right_bgr
    
    def rectify_frames(self, left_frame, right_frame):
        """Apply rectification to frames"""
        left_rect = cv2.remap(left_frame, self.map1_left, self.map2_left, cv2.INTER_LINEAR)
        right_rect = cv2.remap(right_frame, self.map1_right, self.map2_right, cv2.INTER_LINEAR)
        
        return left_rect, right_rect
    
    def compute_depth(self, left_rect, right_rect):
        """Compute depth map from rectified stereo pair"""
        # Convert to grayscale
        left_gray = cv2.cvtColor(left_rect, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_rect, cv2.COLOR_BGR2GRAY)
        
        # Compute disparity
        disparity = self.stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0
        
        # Filter invalid disparities
        disparity[disparity <= 0] = 0.1
        
        # Compute depth using Q matrix
        if disparity.max() > 0:
            points_3D = cv2.reprojectImageTo3D(disparity, self.Q)
            depth_map = points_3D[:, :, 2]
            
            # Filter invalid depths
            depth_map[depth_map <= 0] = 0
            depth_map[depth_map > 20] = 0
            
            # Apply median filter to reduce noise
            depth_map = cv2.medianBlur(depth_map.astype(np.float32), 3)
        else:
            depth_map = np.zeros_like(left_gray, dtype=np.float32)
        
        return depth_map, disparity
    
    def detect_objects(self, frame):
        """Detect objects using YOLO"""
        if not self.use_yolo or self.model is None:
            return self.detect_faces_fallback(frame)
        
        try:
            # Run YOLO inference
            results = self.model(frame, verbose=False, conf=0.4, iou=0.45)
            
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        class_id = int(box.cls[0])
                        class_name = self.model.names[class_id]
                        confidence = float(box.conf[0])
                        
                        # Skip very small detections
                        if (x2 - x1) < 20 or (y2 - y1) < 20:
                            continue
                        
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2
                        
                        # Calculate bounding box area
                        bbox_area = (x2 - x1) * (y2 - y1)
                        
                        detections.append({
                            'bbox': (x1, y1, x2, y2),
                            'center': (center_x, center_y),
                            'class': class_name,
                            'confidence': confidence,
                            'class_id': class_id,
                            'width': x2 - x1,
                            'height': y2 - y1,
                            'area': bbox_area
                        })
            
            # Sort by area (largest first)
            detections.sort(key=lambda x: x['area'], reverse=True)
            
            return detections
            
        except Exception as e:
            print(f"YOLO detection error: {e}")
            return self.detect_faces_fallback(frame)
    
    def detect_faces_fallback(self, frame):
        """Fallback face detection using OpenCV"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        detections = []
        for (x, y, w, h) in faces:
            if w > 50 and h > 50:
                detections.append({
                    'bbox': (x, y, x + w, y + h),
                    'center': (x + w//2, y + h//2),
                    'class': 'person',
                    'confidence': 0.8,
                    'class_id': 0,
                    'width': w,
                    'height': h,
                    'area': w * h
                })
        
        return detections
    
    def measure_object_distance(self, detections, depth_map):
        """Measure distance to each detected object"""
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            center_x = (x1 + x2) // 2
            center_y = int(y1 + 0.9 * (y2 - y1))  # 90% down the bbox


            
            # Ensure coordinates are within bounds
            center_y = min(max(center_y, 0), depth_map.shape[0] - 1)
            center_x = min(max(center_x, 0), depth_map.shape[1] - 1)
            
           # --- BETTER DEPTH REGION: bottom 30% of bbox ---
            h = y2 - y1
            w = x2 - x1

            y_start = int(y1 + 0.7 * h)
            y_end   = y2

            x_start = int(x1 + 0.2 * w)
            x_end   = int(x2 - 0.2 * w)

            # Clip bounds
            y_start = max(0, y_start)
            y_end   = min(depth_map.shape[0], y_end)
            x_start = max(0, x_start)
            x_end   = min(depth_map.shape[1], x_end)

            depth_region = depth_map[y_start:y_end, x_start:x_end]
            depth_region = depth_region[(depth_region > 0) & (depth_region < 10)]

            
            if len(depth_region) > 3:
                depth_value = np.median(depth_region)

                # ---- NON-LINEAR DEPTH CORRECTION ----
                # stronger correction when near, weaker when far
                if depth_value < 1.5:
                    depth_value *= 0.80      # near objects were too large
                elif depth_value < 3.0:
                    depth_value *= 0.90
                else:
                    depth_value *= 0.95
                
                # Reject impossible jumps
                if hasattr(self, "prev_depth"):
                    if abs(depth_value - self.prev_depth) > 2.0:   # >2m sudden jump → noise
                        depth_value = self.prev_depth	

                self.prev_depth = depth_value

                # -------------------------------------

                # ---------------------------------------------
                    
                # --- VALID DEPTH RANGE FOR 7.4 cm BASELINE ---
                if 0.3 < depth_value < 2.5:

                    # Temporal smoothing (remove jumps)
                    if hasattr(det, "prev_distance"):
                        depth_value = 0.7 * det["prev_distance"] + 0.3 * depth_value

                    det["prev_distance"] = depth_value
                    det['distance'] = float(depth_value)

                        
                    # Categorize distance
                    if depth_value < 1.0:
                        det['distance_category'] = 'Very Close'
                    elif depth_value < 2.0:
                        det['distance_category'] = 'Close'
                    elif depth_value < 5.0:
                        det['distance_category'] = 'Medium'
                    elif depth_value < 10.0:
                        det['distance_category'] = 'Far'
                    else:
                        det['distance_category'] = 'Very Far'
                else:
                    det['distance'] = 0.0
                    det['distance_category'] = 'Out of Range'
            else:
                det['distance'] = 0.0
                det['distance_category'] = 'No Depth Data'
        
        return detections
    
    def create_display(self, left_rect, detections, depth_map, disparity_map, fps):
        """Create 2x2 display with all information"""
        display_size = self.display_size
        
        # 1. Detection view
        detection_view = left_rect.copy()
        
        # Draw detections
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            class_name = det['class']
            distance = det.get('distance', 0)

            color = self.class_colors.get(class_name, self.default_color)

            # Draw bounding box
            cv2.rectangle(detection_view, (x1, y1), (x2, y2), color, 2)

            # --- FIX: compute bottom-center point ---
            center_x = (x1 + x2) // 2
            center_y = y2 - 5
            # ---------------------------------------

            # Draw center point
            cv2.circle(detection_view, (center_x, center_y), 4, color, -1)

            
            # Create label
            label = f"{class_name}"
            if distance > 0:
                label += f" {distance:.1f}m"
            
            # Put text with background
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            
            (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, 1)
            
            # Background rectangle
            cv2.rectangle(detection_view,
                         (x1, y1 - text_height - 10),
                         (x1 + text_width, y1),
                         color, -1)
            
            # Text
            cv2.putText(detection_view, label,
                       (x1, y1 - 5), font, font_scale,
                       (255, 255, 255), 1)
        
        # 2. Disparity map - FIXED: check if not None
        if disparity_map is not None and disparity_map.max() > 0:
            disp_norm = cv2.normalize(disparity_map, None, 0, 255, cv2.NORM_MINMAX)
            disp_norm = disp_norm.astype(np.uint8)
            disp_view = cv2.applyColorMap(disp_norm, cv2.COLORMAP_JET)
        else:
            disp_view = np.zeros((self.img_size[1], self.img_size[0], 3), dtype=np.uint8)
        
        # 3. Depth map - FIXED: check if not None
        if depth_map is not None and depth_map.max() > 0:
            depth_vis = depth_map.copy()
            depth_vis[depth_vis > 10] = 10
            depth_norm = cv2.normalize(depth_vis, None, 0, 255, cv2.NORM_MINMAX)
            depth_norm = depth_norm.astype(np.uint8)
            depth_view = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
        else:
            depth_view = np.zeros((self.img_size[1], self.img_size[0], 3), dtype=np.uint8)
        
        # 4. Original view
        original_view = left_rect.copy()
        
        # Resize all views
        detection_resized = cv2.resize(detection_view, (display_size, display_size))
        disp_resized = cv2.resize(disp_view, (display_size, display_size))
        depth_resized = cv2.resize(depth_view, (display_size, display_size))
        original_resized = cv2.resize(original_view, (display_size, display_size))
        
        # Create 2x2 grid
        top_row = np.hstack((detection_resized, disp_resized))
        bottom_row = np.hstack((original_resized, depth_resized))
        display = np.vstack((top_row, bottom_row))
        
        # Add borders
        border_color = (100, 100, 100)
        border_thickness = 2
        
        cv2.line(display, (display_size, 0),
                (display_size, display_size*2),
                border_color, border_thickness)
        
        cv2.line(display, (0, display_size),
                (display_size*2, display_size),
                border_color, border_thickness)
        
        # Add quadrant labels
        labels = [
            ("Detection", (10, 30)),
            ("Disparity", (display_size + 10, 30)),
            ("Original", (10, display_size + 30)),
            ("Depth", (display_size + 10, display_size + 30))
        ]
        
        for label_text, pos in labels:
            cv2.putText(display, label_text, pos,
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                       (255, 255, 255), 2)
        
        # Add FPS
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(display, fps_text, (display_size*2 - 100, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                   (0, 255, 255), 2)
        
        # Add object count
        obj_count = len(detections)
        obj_with_dist = len([d for d in detections if d.get('distance', 0) > 0])
        count_text = f"Objects: {obj_count}"
        
        cv2.putText(display, count_text, (10, display_size*2 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                   (0, 255, 255), 1)
        
        # Add baseline info
        baseline_text = f"Baseline: {self.baseline_m*100:.1f}cm"
        cv2.putText(display, baseline_text,
                   (display_size*2 - 150, display_size*2 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                   (200, 200, 200), 1)
        
        return display
    
    def print_report(self, detections):
        """Print detection report to console"""
        valid_detections = [d for d in detections if d.get('distance', 0) > 0]
        
        if valid_detections:
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Frame {self.frame_count}")
            for i, det in enumerate(valid_detections[:3]):
                print(f"  {i+1}. {det['class']}: {det['distance']:.2f}m "
                      f"(Conf: {det['confidence']:.2f})")
    
    def save_snapshot(self, display, detections):
        """Save current frame as snapshot"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"snapshot_{timestamp}.jpg"
        
        cv2.imwrite(filename, display)
        
        # Also save detection data
        if detections:
            data_filename = f"snapshot_{timestamp}_data.txt"
            with open(data_filename, 'w') as f:
                f.write(f"Snapshot: {timestamp}\n")
                f.write(f"Objects detected: {len(detections)}\n")
                
                for i, det in enumerate(detections):
                    f.write(f"{i+1}. {det['class']} - {det.get('distance', 0):.2f}m\n")
        
        print(f"✓ Snapshot saved: {filename}")
    
    def run(self):
        """Main stereo vision loop"""
        print("\n" + "=" * 60)
        print("LIVE STEREO VISION SYSTEM")
        print("=" * 60)
        print("Controls:")
        print("  Press 'r' - Print detection report")
        print("  Press 's' - Save snapshot")
        print("  Press 'p' - Pause/Resume")
        print("  Press 'q' - Quit")
        print("=" * 60)
        
        paused = False
        
        try:
            while True:
                if not paused:
                    start_time = time.time()
                    self.frame_count += 1
                    
                    # 1. Capture frames
                    left_raw, right_raw = self.capture_frames()
                    
                    # 2. Apply rectification
                    left_rect, right_rect = self.rectify_frames(left_raw, right_raw)
                    
                    # 3. Compute depth (with frame skipping)
                    if self.frame_count % self.depth_interval == 0:
                        self.last_depth, self.last_disp = self.compute_depth(left_rect, right_rect)
                    
                    depth_map = self.last_depth
                    disparity_map = self.last_disp
                    
                    # 4. Detect objects (with frame skipping)
                    if self.frame_count % self.yolo_interval == 0:
                        self.last_detections = self.detect_objects(left_rect)
                    
                    detections = self.last_detections
                    
                    # 5. Measure distances
                    detections_with_dist = self.measure_object_distance(detections, depth_map)
                    
                    # 6. Calculate FPS
                    proc_time = time.time() - start_time
                    fps = 1.0 / proc_time if proc_time > 0 else 0
                    self.fps_history.append(fps)
                    if len(self.fps_history) > 30:
                        self.fps_history.pop(0)
                    avg_fps = np.mean(self.fps_history) if self.fps_history else fps
                    
                    # 7. Create display (with proper None checks)
                    display = self.create_display(
                        left_rect, detections_with_dist, depth_map, disparity_map, avg_fps)
                    
                    # 8. Show display
                    cv2.imshow('Stereo Vision + YOLO + Depth', display)
                    
                    # 9. Handle keys
                    # ---------- PRINT EVERY 1 SECOND ----------
                    current_time = time.time()
                    if current_time - self.last_print_time >= 1.0:
                        self.print_report(detections_with_dist)
                        self.last_print_time = current_time
                    # ------------------------------------------
                    key = cv2.waitKey(1) & 0xFF
                    
                    if key == ord('r') and not paused:
                        self.print_report(detections_with_dist)
                    
                    elif key == ord('s') and not paused:
                        self.save_snapshot(display, detections_with_dist)
                    
                    elif key == ord('p'):
                        paused = not paused
                        status = "PAUSED" if paused else "RESUMED"
                        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] {status}")
                    
                    elif key == ord('q'):
                        print("\nStopping stereo vision system...")
                        break
                
                else:
                    # When paused, still handle key presses
                    key = cv2.waitKey(100) & 0xFF
                    
                    if key == ord('p'):
                        paused = False
                        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] RESUMED")
                    elif key == ord('q'):
                        print("\nStopping stereo vision system...")
                        break
        
        except KeyboardInterrupt:
            print("\n\n⚠️ Stopped by user")
        
        except Exception as e:
            print(f"\n❌ Error in main loop: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            cv2.destroyAllWindows()
            self.cam_left.stop()
            self.cam_right.stop()
            
            # Print performance summary
            if self.fps_history:
                print(f"\n" + "=" * 60)
                print("PERFORMANCE SUMMARY")
                print("=" * 60)
                print(f"Frames processed: {self.frame_count}")
                print(f"Average FPS: {np.mean(self.fps_history):.1f}")
                print(f"Max FPS: {np.max(self.fps_history):.1f}")
                print(f"Min FPS: {np.min(self.fps_history):.1f}")
                print(f"Baseline: {self.baseline_m*100:.1f} cm")
                print("=" * 60)
            
            print("\n✅ Stereo vision system stopped.")

def main():
    print("Starting Stereo Vision + YOLO + Depth System...")
    
    # Check for calibration file
    calib_files = ['stereo_calib3.npz', 'stereo_calibration3.npz']
    calib_file = None
    
    for file in calib_files:
        if os.path.exists(file):
            calib_file = file
            print(f"Found calibration file: {file}")
            break
    
    if calib_file is None:
        print("❌ No calibration file found!")
        print("Please run calibration first:")
        print("1. Place left/right images in 'calib' folder")
        print("2. Run: python stereo_calibrate.py")
        return
    
    try:
        system = StereoYOLODepth(calibration_file=calib_file)
        system.run()
    
    except Exception as e:
        print(f"\n❌ Error starting system: {e}")
        import traceback
        traceback.print_exc()
        
        print("\nTroubleshooting:")
        print("1. Check camera connections")
        print("2. Verify calibration file exists")
        print("3. Ensure camera permissions: sudo usermod -a -G video $USER")

if __name__ == "__main__":
    main()