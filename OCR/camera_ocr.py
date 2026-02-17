#!/usr/bin/env python3
"""
Camera OCR script for Raspberry Pi 5 with dual Camera Module 3.
Captures frames from specified camera and runs OCR analysis.
"""

import cv2
import numpy as np
from datetime import datetime
from pathlib import Path
import time

# Import OCR function from ocr2.py
import sys
sys.path.insert(0, str(Path(__file__).parent))
from ocr2 import get_better_ocr_system, _preprocess_for_ocr, _ocr_with_best_psm, _detect_misread_words, _autocorrect_text

try:
    from picamera2 import Picamera2
    PICAMERA2_AVAILABLE = True
except ImportError:
    PICAMERA2_AVAILABLE = False
    print("Warning: picamera2 not available. Install with: sudo apt install -y python3-picamera2")


class CameraOCR:
    
    def __init__(self, camera_num=0, resolution=(1280, 720)):

        self.camera_num = camera_num
        self.resolution = resolution
        self.picam = None
        self.output_dir = Path("ocr_captures")
        self.output_dir.mkdir(exist_ok=True)
        
    def initialize_camera(self):
        """Initialize Picamera2 for frame capture."""
        if not PICAMERA2_AVAILABLE:
            raise RuntimeError("picamera2 not available")
        
        print(f"Initializing Camera {self.camera_num}...")
        
        # Create Picamera2 instance for specific camera
        self.picam = Picamera2(self.camera_num)
        
        # Configure camera for still capture
        config = self.picam.create_still_configuration(
            main={"size": self.resolution, "format": "RGB888"},
            buffer_count=2
        )
        self.picam.configure(config)
        
        # Start camera
        self.picam.start()
        
        # Let camera warm up
        print("Warming up camera...")
        time.sleep(2)
        
        print(f"✓ Camera {self.camera_num} ready: {self.resolution}")
        
    def capture_frame(self, save=True, filename_prefix="capture"):
       
        if not self.picam:
            raise RuntimeError("Camera not initialized. Call initialize_camera() first.")
        
        # Capture frame (returns RGB)
        frame_rgb = self.picam.capture_array()
        
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        
        if save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = self.output_dir / f"{filename_prefix}_cam{self.camera_num}_{timestamp}.jpg"
            cv2.imwrite(str(filename), frame_bgr)
            print(f"Saved frame: {filename}")
        
        return frame_bgr
    
    def capture_and_ocr(self, autocorrect=True, confidence_threshold=60.0, save_frame=True):
        """
        Capture frame and run OCR analysis.
        
        Args:
            autocorrect: Enable spell-correction
            confidence_threshold: Confidence threshold for autocorrect
            save_frame: Save captured frame
            
        Returns:
            dict with 'text', 'frame', 'timestamp'
        """
        timestamp = datetime.now()
        
        print("\n" + "=" * 60)
        print(f"Capturing frame from Camera {self.camera_num}...")
        print("=" * 60)
        
        # Capture frame
        frame = self.capture_frame(save=save_frame, filename_prefix="ocr_input")
        
        print(f"Frame shape: {frame.shape}")
        print("Running OCR analysis...")
        
        # Save frame temporarily for OCR processing
        temp_path = self.output_dir / "temp_ocr_input.jpg"
        cv2.imwrite(str(temp_path), frame)
        
        # Run OCR
        extracted_text = get_better_ocr_system(
            str(temp_path),
            autocorrect=autocorrect,
            confidence_threshold=confidence_threshold
        )
        
        # Clean up temp file
        temp_path.unlink()
        
        print("\n" + "=" * 60)
        print("OCR RESULT")
        print("=" * 60)
        if extracted_text:
            print(extracted_text)
        else:
            print("No text detected")
        print("=" * 60)
        
        return {
            'text': extracted_text,
            'frame': frame,
            'timestamp': timestamp,
            'camera': self.camera_num
        }
    
    def close(self):
        """Release camera resources."""
        if self.picam:
            self.picam.stop()
            self.picam.close()
            print(f"Camera {self.camera_num} closed")


def main():
    """Main function to run camera OCR."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Capture frame and run OCR")
    parser.add_argument("--camera", type=int, default=0, choices=[0, 1],
                       help="Camera index (0 or 1)")
    parser.add_argument("--width", type=int, default=1920,
                       help="Capture width")
    parser.add_argument("--height", type=int, default=1080,
                       help="Capture height")
    parser.add_argument("--no-autocorrect", action="store_true",
                       help="Disable autocorrect")
    parser.add_argument("--confidence", type=float, default=70.0,
                       help="Confidence threshold for autocorrect (0-100)")
    parser.add_argument("--continuous", action="store_true",
                       help="Continuous capture mode (press Enter for each capture, 'q' to quit)")
    
    args = parser.parse_args()
    
    # Initialize camera OCR
    cam_ocr = CameraOCR(
        camera_num=args.camera,
        resolution=(args.width, args.height)
    )
    
    try:
        cam_ocr.initialize_camera()
        
        if args.continuous:
            print("\n" + "=" * 60)
            print("CONTINUOUS MODE")
            print("=" * 60)
            print("Press ENTER to capture and run OCR")
            print("Type 'q' and press ENTER to quit")
            print("=" * 60)
            
            while True:
                user_input = input("\nReady to capture (ENTER=capture, q=quit): ").strip().lower()
                
                if user_input == 'q':
                    print("Exiting...")
                    break
                
                # Capture and run OCR
                result = cam_ocr.capture_and_ocr(
                    autocorrect=not args.no_autocorrect,
                    confidence_threshold=args.confidence,
                    save_frame=True
                )
        else:
            # Single capture mode
            result = cam_ocr.capture_and_ocr(
                autocorrect=not args.no_autocorrect,
                confidence_threshold=args.confidence,
                save_frame=True
            )
            
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cam_ocr.close()


if __name__ == "__main__":
    main()
