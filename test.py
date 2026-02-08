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

class NormalStereoVisionWithYOLO:
    def __init__(self, calibration_file='stereo_calibration4.npy'):
        """
        NORMAL Stereo Vision (No Fish-Eye) with YOLO Detection and Distance Measurement
        Square display output
        """
        print("=" * 60)
        print("NORMAL STEREO VISION + YOLO + DISTANCE")
        print("No Fish-Eye | Square Display")
        print("=" * 60)
        
        # 1. Load calibration data with proper conversion
        print("\n1. Loading calibration (no distortion correction)...")
        if not os.path.exists(calibration_file):
            print(f"❌ ERROR: {calibration_file} not found!")
            print("Run calibration first!")
            raise FileNotFoundError
        
        self.calib = np.load(calibration_file, allow_pickle=True).item()
        
        # DEBUG: Print what's in the calibration file
        print(f"Calibration keys: {list(self.calib.keys())}")
        
        # Convert ALL matrices to numpy arrays with proper dtype
        # FIX: Convert from lists to numpy arrays
        self.R1 = np.array(self.calib['R1'], dtype=np.float32) if 'R1' in self.calib else np.eye(3, dtype=np.float32)
        self.R2 = np.array(self.calib['R2'], dtype=np.float32) if 'R2' in self.calib else np.eye(3, dtype=np.float32)
        self.P1 = np.array(self.calib['P1'], dtype=np.float32) if 'P1' in self.calib else np.eye(3, 4, dtype=np.float32)
        self.P2 = np.array(self.calib['P2'], dtype=np.float32) if 'P2' in self.calib else np.eye(3, 4, dtype=np.float32)
        self.Q = np.array(self.calib['Q'], dtype=np.float32) if 'Q' in self.calib else None
        
        # Verify matrix shapes
        print(f"R1 shape: {self.R1.shape}, dtype: {self.R1.dtype}")
        print(f"P1 shape: {self.P1.shape}, dtype: {self.P1.dtype}")
        
        self.image_size = (640, 480)  # Force square-friendly size
        self.baseline = self.calib.get('baseline', 0.1)
        
        print(f"   ✓ Calibration loaded (rectification only)")
        print(f"   Baseline: {self.baseline*100:.1f} cm")
        print(f"   Image size: {self.image_size}")
        
        # 2. Initialize cameras with SQUARE-FRIENDLY resolution
        print("\n2. Initializing cameras (square output)...")
        self.cam_left = Picamera2(0)
        self.cam_right = Picamera2(1)
        
        # Use resolution that's good for square display
        target_width, target_height = 640, 480  # 4:3 aspect ratio
        
        config = self.cam_left.create_preview_configuration(
            main={"size": (target_width, target_height)})
        self.cam_left.configure(config)
        self.cam_right.configure(config)
        
        self.cam_left.start()
        self.cam_right.start()
        time.sleep(2)
        print(f"   ✓ Cameras ready at {target_width}x{target_height}")
        
        # 3. Create SIMPLE rectification maps (NO distortion correction)
        print("\n3. Setting up NORMAL rectification (no fish-eye)...")
        # Identity matrix = NO distortion correction
        identity_matrix = np.eye(3, dtype=np.float32)
        zero_distortion = np.zeros((5, 1), dtype=np.float32)
        
        try:
            # FIX: Use only first 3 columns of P1 and P2 for initUndistortRectifyMap
            self.map1_left, self.map2_left = cv2.initUndistortRectifyMap(
                identity_matrix, zero_distortion, 
                self.R1, self.P1[:, :3],  # Use only first 3 columns
                self.image_size, cv2.CV_16SC2)
            
            self.map1_right, self.map2_right = cv2.initUndistortRectifyMap(
                identity_matrix, zero_distortion, 
                self.R2, self.P2[:, :3],  # Use only first 3 columns
                self.image_size, cv2.CV_16SC2)
            print("   ✓ Normal rectification ready (no fish-eye)")
            
        except Exception as e:
            print(f"❌ Error creating rectification maps: {e}")
            print("Creating identity maps as fallback...")
            self.create_identity_maps()
        
        # 4. Load YOLO model
        print("\n4. Loading YOLO model...")
        if YOLO_AVAILABLE:
            try:
                self.model = YOLO('yolov8n.pt')
                self.use_yolo = True
                print(f"   ✓ YOLOv8n loaded for {len(self.model.names)} classes")
            except Exception as e:
                print(f"   ⚠️ YOLO failed: {e}")
                self.model = None
                self.use_yolo = False
        else:
            self.model = None
            self.use_yolo = False
        
        # 5. Fast stereo matcher for depth
        print("\n5. Initializing stereo matcher...")
        self.stereo = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=64,
            blockSize=5,
            P1=8 * 3 * 5 ** 2,
            P2=32 * 3 * 5 ** 2,
            disp12MaxDiff=1,
            uniquenessRatio=15,
            speckleWindowSize=100,
            speckleRange=2,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )
        print("   ✓ Stereo matcher ready")
        
        # 6. Color mapping
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
            'book': (139, 69, 19),      # Brown,
            'cell phone': (255, 215, 0), # Gold
            'laptop': (70, 130, 180),   # Steel Blue
        }
        self.default_color = (200, 200, 200)
        
        # 7. For square display
        self.display_size = 320  # Size for each quadrant in the square display
        
        self.frame_count = 0
        print("\n" + "=" * 60)
        print("✅ SYSTEM READY - NORMAL VISION")
        print("=" * 60)
    
    def create_identity_maps(self):
        """Create identity rectification maps (no rectification)"""
        h, w = self.image_size[1], self.image_size[0]
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        
        self.map1_left = x.astype(np.float32)
        self.map2_left = y.astype(np.float32)
        self.map1_right = x.astype(np.float32)
        self.map2_right = y.astype(np.float32)
        print("   Using identity rectification (no actual rectification)")
    
    def capture_frames_normal(self):
        """Capture frames without fish-eye processing"""
        left_frame = self.cam_left.capture_array()
        right_frame = self.cam_right.capture_array()
        
        left_bgr = cv2.cvtColor(left_frame, cv2.COLOR_RGB2BGR)
        right_bgr = cv2.cvtColor(right_frame, cv2.COLOR_RGB2BGR)
        
        return left_bgr, right_bgr
    
    def rectify_frames_normal(self, left_frame, right_frame):
        """Apply NORMAL rectification (no distortion correction)"""
        # Resize to target size
        left_resized = cv2.resize(left_frame, self.image_size)
        right_resized = cv2.resize(right_frame, self.image_size)
        
        # Apply rectification ONLY (no undistortion)
        left_rect = cv2.remap(left_resized, self.map1_left, self.map2_left, cv2.INTER_LINEAR)
        right_rect = cv2.remap(right_resized, self.map1_right, self.map2_right, cv2.INTER_LINEAR)
        
        return left_rect, right_rect
    
    def compute_depth_normal(self, left_rect, right_rect):
        """Compute depth from NORMAL rectified images"""
        left_gray = cv2.cvtColor(left_rect, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_rect, cv2.COLOR_BGR2GRAY)
        
        disparity = self.stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0
        
        # Debug: Show disparity range
        # print(f"Disparity range: [{disparity.min():.1f}, {disparity.max():.1f}]")
        
        if disparity.max() > 0 and self.Q is not None:
            points_3D = cv2.reprojectImageTo3D(disparity, self.Q)
            depth_map = points_3D[:, :, 2]
            
            # Filter invalid depths
            depth_map[depth_map <= 0] = 0
            depth_map[depth_map > 50] = 0
            
            # Apply median filter to reduce noise
            depth_map = cv2.medianBlur(depth_map.astype(np.float32), 3)
        else:
            depth_map = np.zeros_like(left_gray, dtype=np.float32)
        
        # Debug: Show depth range
        # valid_depths = depth_map[depth_map > 0]
        # if len(valid_depths) > 0:
        #     print(f"Depth range: [{valid_depths.min():.2f}, {valid_depths.max():.2f}]m")
        
        return depth_map, disparity
    
    def detect_objects(self, frame):
        """Detect objects using YOLO"""
        if not self.use_yolo or self.model is None:
            return []
        
        try:
            results = self.model(frame, verbose=False, conf=0.5, iou=0.45)
            
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
                        
                        detections.append({
                            'bbox': (x1, y1, x2, y2),
                            'center': (center_x, center_y),
                            'class': class_name,
                            'confidence': confidence,
                            'class_id': class_id,
                            'width': x2 - x1,
                            'height': y2 - y1,
                        })
            
            return detections
            
        except Exception as e:
            print(f"YOLO detection error: {e}")
            return []
    
    def measure_object_distance(self, detections, depth_map):
        """Measure distance to each object"""
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            center_x, center_y = det['center']
            
            # Ensure coordinates are within bounds
            center_y = min(max(center_y, 0), depth_map.shape[0] - 1)
            center_x = min(max(center_x, 0), depth_map.shape[1] - 1)
            
            # Get depth in a small region around center
            y_start = max(0, center_y - 5)
            y_end = min(depth_map.shape[0], center_y + 5)
            x_start = max(0, center_x - 5)
            x_end = min(depth_map.shape[1], center_x + 5)
            
            depth_region = depth_map[y_start:y_end, x_start:x_end]
            depth_region = depth_region[depth_region > 0]  # Remove zeros
            
            if len(depth_region) > 3:
                # Use median to reduce noise
                depth_value = np.median(depth_region)
                
                if 0.1 < depth_value < 20:  # Valid depth range
                    det['distance'] = float(depth_value)
                    
                    # Simple categorization
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
    
    def create_square_display(self, left_rect, detections, depth_map, disparity_map, fps):
        """Create a SQUARE 2x2 display"""
        # Make all images the same size for square display
        display_size = self.display_size
        
        # 1. Detection view
        detection_view = left_rect.copy()
        
        # Draw detections
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            class_name = det['class']
            confidence = det['confidence']
            distance = det.get('distance', 0)
            
            color = self.class_colors.get(class_name, self.default_color)
            
            cv2.rectangle(detection_view, (x1, y1), (x2, y2), color, 2)
            
            label = f"{class_name}"
            if distance > 0:
                label += f" {distance:.1f}m"
            
            cv2.putText(detection_view, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            center_x, center_y = det['center']
            cv2.circle(detection_view, (center_x, center_y), 3, color, -1)
        
        # 2. Disparity map
        if disparity_map is not None and disparity_map.max() > 0:
            disp_norm = cv2.normalize(disparity_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            disp_view = cv2.applyColorMap(disp_norm, cv2.COLORMAP_JET)
        else:
            disp_view = np.zeros((self.image_size[1], self.image_size[0], 3), dtype=np.uint8)
        
        # 3. Depth map
        if depth_map is not None and depth_map.max() > 0:
            depth_norm = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            depth_view = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
            
            # Add depth scale
            cv2.putText(depth_view, "Close", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(depth_view, "Far", (depth_view.shape[1] - 40, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        else:
            depth_view = np.zeros((self.image_size[1], self.image_size[0], 3), dtype=np.uint8)
        
        # 4. Original view (for comparison)
        original_view = left_rect.copy()
        
        # Resize all to same square size
        detection_resized = cv2.resize(detection_view, (display_size, display_size))
        disp_resized = cv2.resize(disp_view, (display_size, display_size))
        depth_resized = cv2.resize(depth_view, (display_size, display_size))
        original_resized = cv2.resize(original_view, (display_size, display_size))
        
        # Create 2x2 grid (perfect square)
        top_row = np.hstack((detection_resized, disp_resized))
        bottom_row = np.hstack((original_resized, depth_resized))
        square_display = np.vstack((top_row, bottom_row))
        
        # Add borders and labels
        border_color = (100, 100, 100)
        border_thickness = 2
        
        # Add vertical border
        cv2.line(square_display, (display_size, 0), (display_size, display_size*2), 
                border_color, border_thickness)
        
        # Add horizontal border
        cv2.line(square_display, (0, display_size), (display_size*2, display_size), 
                border_color, border_thickness)
        
        # Add labels in corners
        labels = [
            ("Detection", (10, 30)),
            ("Disparity", (display_size + 10, 30)),
            ("Original", (10, display_size + 30)),
            ("Depth", (display_size + 10, display_size + 30))
        ]
        
        for label_text, pos in labels:
            cv2.putText(square_display, label_text, pos,
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add FPS
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(square_display, fps_text, (display_size*2 - 100, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Add object count
        obj_count = len(detections)
        obj_with_dist = len([d for d in detections if d.get('distance', 0) > 0])
        count_text = f"Objects: {obj_count}"
        
        cv2.putText(square_display, count_text, (10, display_size*2 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        
        # Add baseline info
        baseline_text = f"Baseline: {self.baseline*100:.1f}cm"
        cv2.putText(square_display, baseline_text,
                   (display_size*2 - 150, display_size*2 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return square_display
    
    def print_report(self, detections):
        """Print detection report"""
        valid_objects = [d for d in detections if d.get('distance', 0) > 0]
        
        if valid_objects:
            print(f"\nFrame {self.frame_count} - Objects with distance:")
            for i, det in enumerate(valid_objects[:3]):  # Show top 3
                print(f"  {i+1}. {det['class']}: {det.get('distance', 0):.2f}m ({det.get('distance_category', '')})")
    
    def save_frame(self, display_image, detections):
        """Save current frame"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"stereo_frame_{timestamp}.jpg"
        cv2.imwrite(filename, display_image)
        print(f"✓ Saved: {filename}")
        
        if detections:
            print(f"  Detected {len(detections)} objects")
    
    def run_normal_vision(self):
        """Main loop with NORMAL stereo vision"""
        print("\n" + "=" * 60)
        print("NORMAL STEREO VISION - SQUARE DISPLAY")
        print("=" * 60)
        print("Controls:")
        print("  Press 'r' - Print report")
        print("  Press 's' - Save frame")
        print("  Press 'd' - Toggle debug info")
        print("  Press 'q' - Quit")
        print("=" * 60)
        
        fps_history = []
        show_debug = False
        
        try:
            while True:
                start_time = time.time()
                self.frame_count += 1
                
                # 1. Capture frames (normal, no fish-eye)
                left_raw, right_raw = self.capture_frames_normal()
                
                # 2. Apply NORMAL rectification
                left_rect, right_rect = self.rectify_frames_normal(left_raw, right_raw)
                
                # 3. Compute depth
                depth_map, disparity_map = self.compute_depth_normal(left_rect, right_rect)
                
                # 4. Detect objects
                detections = self.detect_objects(left_rect)
                
                # 5. Measure distance
                detections_with_dist = self.measure_object_distance(detections, depth_map)
                
                # 6. Calculate FPS
                proc_time = time.time() - start_time
                fps = 1.0 / proc_time if proc_time > 0 else 0
                fps_history.append(fps)
                if len(fps_history) > 30:
                    fps_history.pop(0)
                avg_fps = np.mean(fps_history) if fps_history else fps
                
                # 7. Create SQUARE display
                display = self.create_square_display(
                    left_rect, detections_with_dist, depth_map, disparity_map, avg_fps)
                
                # 8. Show display
                cv2.imshow('Normal Stereo Vision - Square Display', display)
                
                # 9. Print report periodically
                if self.frame_count % 30 == 0 and detections_with_dist and show_debug:
                    self.print_report(detections_with_dist)
                
                # 10. Handle keys
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('r'):
                    self.print_report(detections_with_dist)
                
                elif key == ord('s'):
                    self.save_frame(display, detections_with_dist)
                
                elif key == ord('d'):
                    show_debug = not show_debug
                    print(f"Debug info: {'ON' if show_debug else 'OFF'}")
                
                elif key == ord('q'):
                    print("\nStopping...")
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
            
            if fps_history:
                print(f"\nPerformance:")
                print(f"  Average FPS: {np.mean(fps_history):.1f}")
                print(f"  Frames processed: {self.frame_count}")
            
            print("\n✅ Normal stereo vision stopped.")

def main():
    # Check for calibration file
    calib_files = ['stereo_calibration4.npy', 'stereo_calibration.npy', 'stereo_calib.npz']
    calib_file = None
    
    for file in calib_files:
        if os.path.exists(file):
            calib_file = file
            print(f"Found calibration file: {file}")
            break
    
    if calib_file is None:
        print("❌ ERROR: No calibration file found!")
        print("Available files in directory:")
        for f in os.listdir('.'):
            print(f"  {f}")
        print("\nPlease run calibration first!")
        return
    
    print("Starting NORMAL Stereo Vision...")
    
    try:
        system = NormalStereoVisionWithYOLO(calibration_file=calib_file)
        system.run_normal_vision()
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        print("\nMake sure both cameras are connected properly.")

if __name__ == "__main__":
    main()