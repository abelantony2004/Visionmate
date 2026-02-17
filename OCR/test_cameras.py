#!/usr/bin/env python3
"""
Camera detection and testing script for Raspberry Pi 5 with dual cameras.
Tests both libcamera and V4L2 interfaces.
"""

import cv2
import os
import numpy as np
from datetime import datetime


def detect_v4l2_cameras():
    """Detect cameras using V4L2 interface."""
    print("=" * 60)
    print("Detecting V4L2 Cameras...")
    print("=" * 60)
    
    available_cameras = []
    
    # Check /dev/video* devices
    video_devices = [f"/dev/video{i}" for i in range(10)]
    
    for idx, device in enumerate(video_devices):
        if os.path.exists(device):
            print(f"\nFound device: {device}")
            
            # Try to open with OpenCV
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                fps = cap.get(cv2.CAP_PROP_FPS)
                
                print(f"  Index: {idx}")
                print(f"  Resolution: {int(width)}x{int(height)}")
                print(f"  FPS: {fps}")
                
                # Try to capture a test frame
                ret, frame = cap.read()
                if ret:
                    print(f"  ✓ Frame capture successful")
                    available_cameras.append({
                        'index': idx,
                        'device': device,
                        'resolution': (int(width), int(height)),
                        'fps': fps,
                        'working': True
                    })
                else:
                    print(f"  ✗ Frame capture failed")
                    available_cameras.append({
                        'index': idx,
                        'device': device,
                        'working': False
                    })
                
                cap.release()
            else:
                print(f"  ✗ Cannot open device")
    
    return available_cameras


def test_camera_capture(camera_index, num_frames=5, save_test_frame=True):
    """Test capturing frames from a specific camera."""
    print("\n" + "=" * 60)
    print(f"Testing Camera Index {camera_index}")
    print("=" * 60)
    
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print(f"✗ Failed to open camera {camera_index}")
        return False
    
    print(f"✓ Camera {camera_index} opened successfully")
    
    # Get camera properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    
    # Capture test frames
    print(f"\nCapturing {num_frames} test frames...")
    successful_captures = 0
    
    for i in range(num_frames):
        ret, frame = cap.read()
        if ret:
            successful_captures += 1
            print(f"  Frame {i+1}: ✓ ({frame.shape})")
            
            # Save first frame as test
            if save_test_frame and i == 0:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"test_imgs/camera_{camera_index}_test_{timestamp}.jpg"
                os.makedirs("test_imgs", exist_ok=True)
                cv2.imwrite(filename, frame)
                print(f"  Saved test frame to: {filename}")
        else:
            print(f"  Frame {i+1}: ✗ Failed")
    
    cap.release()
    
    success_rate = (successful_captures / num_frames) * 100
    print(f"\nCapture Success Rate: {success_rate:.1f}% ({successful_captures}/{num_frames})")
    
    return successful_captures == num_frames


def test_picamera2():
    """Test if picamera2 is available (recommended for Pi Camera Module 3)."""
    print("\n" + "=" * 60)
    print("Testing Picamera2 Support")
    print("=" * 60)
    
    try:
        from picamera2 import Picamera2
        print("✓ picamera2 library is installed")
        
        try:
            # Try to detect cameras
            cameras = Picamera2.global_camera_info()
            print(f"✓ Found {len(cameras)} camera(s) via libcamera:")
            for idx, cam_info in enumerate(cameras):
                print(f"  Camera {idx}: {cam_info}")
            
            return True, len(cameras)
        except Exception as e:
            print(f"✗ Error detecting cameras: {e}")
            return True, 0
            
    except ImportError:
        print("✗ picamera2 not installed")
        print("  Install with: sudo apt install -y python3-picamera2")
        return False, 0


def list_available_cameras():
    """List all available camera interfaces."""
    print("\n" + "=" * 60)
    print("CAMERA DETECTION SUMMARY")
    print("=" * 60)
    
    # Test picamera2 first (recommended for Pi Camera Module 3)
    picam_available, picam_count = test_picamera2()
    
    # Test V4L2/OpenCV cameras
    v4l2_cameras = detect_v4l2_cameras()
    
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)
    
    if picam_available and picam_count > 0:
        print(f"✓ Use picamera2 for best performance with Pi Camera Module 3")
        print(f"  {picam_count} camera(s) detected via libcamera")
    
    if v4l2_cameras:
        working_cameras = [c for c in v4l2_cameras if c.get('working', False)]
        print(f"✓ {len(working_cameras)} working camera(s) via OpenCV/V4L2:")
        for cam in working_cameras:
            print(f"  - Camera index {cam['index']}: {cam['resolution']}")
    
    if not picam_available and not v4l2_cameras:
        print("✗ No cameras detected!")
        print("\nTroubleshooting:")
        print("  1. Check camera cable connections")
        print("  2. Enable camera in raspi-config")
        print("  3. Reboot the Raspberry Pi")
        print("  4. Check: vcgencmd get_camera")
    
    return v4l2_cameras


if __name__ == "__main__":
    print("\nRaspberry Pi 5 - Camera Detection & Testing")
    print("=" * 60)
    
    # List all available cameras
    cameras = list_available_cameras()
    
    # Test each working camera
    print("\n" + "=" * 60)
    print("RUNNING CAPTURE TESTS")
    print("=" * 60)
    
    working_cameras = [c for c in cameras if c.get('working', False)]
    
    if working_cameras:
        for cam in working_cameras:
            test_camera_capture(cam['index'], num_frames=3, save_test_frame=True)
    else:
        print("\nNo working cameras found to test.")
        print("Trying to test camera indices 0 and 1 anyway...")
        for idx in [0, 1]:
            try:
                test_camera_capture(idx, num_frames=3, save_test_frame=True)
            except Exception as e:
                print(f"Camera {idx} test failed: {e}")
    
    print("\n" + "=" * 60)
    print("Testing Complete!")
    print("=" * 60)
