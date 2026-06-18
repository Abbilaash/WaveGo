#!/usr/bin/env python3
"""
Live green ball detection script using Picamera2 and OpenCV color/circularity detection.
Replaces ONNX YOLOv8 model inference.
"""
import os
import sys
import time
import cv2
import numpy as np

# Add parent directory to path so we can import detect.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from detect import detect

try:
    from picamera2 import Picamera2
except ImportError:
    print("Error: picamera2 module not found. This script must be run on a Raspberry Pi.")
    sys.exit(1)

def main():
    this_dir = os.path.dirname(os.path.realpath(__file__))
    
    # --- Fisheye Rectification Settings ---
    width, height = 640, 480
    
    # Empirical Camera Intrinsic Matrix (K)
    K = np.array([
        [320.0, 0.0, 320.0],
        [0.0, 320.0, 240.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)
    
    # Empirical Fisheye Distortion Coefficients [k1, k2, k3, k4]
    D = np.array([-0.06, 0.02, -0.01, 0.002], dtype=np.float32)
    
    # Precompute undistortion and rectification maps
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, (width, height), np.eye(3), balance=0.2)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), new_K, (width, height), cv2.CV_16SC2)
    
    # Initialize Picamera2
    picam2 = Picamera2()
    config = picam2.create_video_configuration(
        main={"format": "RGB888", "size": (width, height)}
    )
    picam2.configure(config)
    picam2.start()
    
    # Let auto-exposure and white balance settle
    time.sleep(1.5)
    print("\nStarting live green ball color/circularity detection.")
    print("Press Ctrl+C to stop.\n")
    
    saved_visuals = False
    
    try:
        while True:
            frame = picam2.capture_array()
            if frame is None:
                print("Error: Failed to grab frame.")
                time.sleep(0.1)
                continue
            
            # Rectify the frame to flatten the fisheye distortion
            flat_frame = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            
            # Save visual checkpoints on the first frame
            if not saved_visuals:
                # Convert to BGR for writing if they are RGB
                cv2.imwrite(os.path.join(this_dir, "step1_raw_distorted.jpg"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                cv2.imwrite(os.path.join(this_dir, "step2_rectified_flat.jpg"), cv2.cvtColor(flat_frame, cv2.COLOR_RGB2BGR))
                print("Saved visualization stages to disk:")
                print("  - step1_raw_distorted.jpg")
                print("  - step2_rectified_flat.jpg")
                saved_visuals = True
            
            # Run detection on the rectified frame
            # The picamera is configured for RGB888, so we pass input_is_rgb=True
            result = detect(flat_frame, input_is_rgb=True, conf_threshold=0.25)
            
            if result.get("success"):
                detections = result.get("detections", [])
                if detections:
                    print(f"[{time.strftime('%X')}] DETECTED {len(detections)} green ball(s):")
                    for det in detections:
                        print(f"     - conf={det['conf']:.2f} at [{int(det['x1'])}, {int(det['y1'])}, {int(det['x2'])}, {int(det['y2'])}]")
                    
                    # Convert the annotated frame (which is in RGB) to BGR for saving
                    annotated_bgr = cv2.cvtColor(result["annotated_frame"], cv2.COLOR_RGB2BGR)
                    cv2.imwrite(os.path.join(this_dir, "detection_result.jpg"), annotated_bgr)
            else:
                print(f"[{time.strftime('%X')}] Detection error: {result.get('error')}")
            
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\nStopping detection...")
    finally:
        picam2.stop()
        print("Camera stopped.")

if __name__ == "__main__":
    main()
