#!/usr/bin/env python3
"""
Test script for running object detection on Raspberry Pi using Picamera2
without rendering frames.
"""
import os
import sys
import time

# Add parent directory of FollowObject to sys.path
THIS_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(THIS_DIR))

# Import detect function from the current directory
from detect import detect

try:
    from picamera2 import Picamera2
except ImportError:
    print("Error: picamera2 module not found. This script must be run on a Raspberry Pi.")
    sys.exit(1)

def main():
    model_path = os.path.join(THIS_DIR, "best.onnx")
    print(f"Loading ONNX model from: {model_path}")
    
    # Initialize and configure Picamera2
    picam2 = Picamera2()
    config = picam2.create_video_configuration(
        main={"format": "RGB888", "size": (640, 480)}
    )
    picam2.configure(config)
    picam2.start()
    
    # Warm up camera
    time.sleep(1.0)
    print("Starting Picamera2 object detection. Press Ctrl+C to quit...")
    
    try:
        while True:
            frame = picam2.capture_array()
            if frame is None:
                print("Error: Failed to grab frame from Picamera2.")
                time.sleep(0.1)
                continue
                
            # Run detection. BGR->RGB is performed by default since input_is_rgb=False.
            result = detect(frame, model_path=model_path, conf_threshold=0.15, iou_threshold=0.45, input_is_rgb=False)
            
            if result["success"]:
                detections = result.get("detections", [])
                if detections:
                    print(f"[{time.strftime('%X')}] Detections:")
                    for det in detections:
                        print(f"  - {det['class_name']} (conf={det['conf']:.2f}) at [{int(det['x1'])}, {int(det['y1'])}, {int(det['x2'])}, {int(det['y2'])}]")
            else:
                print(f"Error during detection: {result.get('error')}")
                
            time.sleep(0.05)  # Yield CPU execution
            
    except KeyboardInterrupt:
        print("\nStopping detection...")
    finally:
        picam2.stop()
        print("Camera stopped.")

if __name__ == "__main__":
    main()
