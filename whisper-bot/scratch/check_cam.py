import os
import sys
import cv2
import numpy as np

# Add parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from FollowObject.detect import detect
from picamera2 import Picamera2

def main():
    picam2 = Picamera2()
    picam2.configure(picam2.create_preview_configuration(main={"format": "RGB888", "size": (640, 480)}))
    picam2.start()
    
    print("Capturing frame...")
    frame = picam2.capture_array()
    picam2.stop()
    
    print("Frame shape:", frame.shape)
    print("Frame data type:", frame.dtype)
    print("First pixel (raw):", frame[0, 0])
    
    # Save raw frame (cv2 expects BGR, so if raw is RGB, it will swap R and B in jpeg)
    cv2.imwrite("raw_frame_raw.jpg", frame)
    # Save BGR-swapped frame (so if raw is RGB, cv2.imwrite gets BGR and color is correct)
    frame_swapped = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imwrite("raw_frame_swapped.jpg", frame_swapped)
    
    model_path = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "FollowObject", "best.onnx"))
    
    print("\n--- Running detect with input_is_rgb=True ---")
    res1 = detect(frame, model_path=model_path, input_is_rgb=True)
    print("Success:", res1.get("success"))
    print("Detections count:", len(res1.get("detections", [])))
    if res1.get("detections"):
        print("Detections:", res1["detections"])
        
    print("\n--- Running detect with input_is_rgb=False ---")
    res2 = detect(frame, model_path=model_path, input_is_rgb=False)
    print("Success:", res2.get("success"))
    print("Detections count:", len(res2.get("detections", [])))
    if res2.get("detections"):
        print("Detections:", res2["detections"])

if __name__ == "__main__":
    main()
