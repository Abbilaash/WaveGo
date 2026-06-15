#!/usr/bin/env python3
"""
Live camera test script that undistorts/rectifies wide-angle camera frames
using OpenCV's fisheye model and runs object detection on the flat frame.
"""
import os
import sys
import time
import cv2
import numpy as np
import onnxruntime as ort

# OpenCV VideoCapture will be used for live camera feed

def main():
    this_dir = os.path.dirname(os.path.realpath(__file__))
    model_path = os.path.join(this_dir, "best.onnx")
    print(f"Loading ONNX model from: {model_path}")
    
    # Initialize ONNX Session on CPU
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape  # Expecting [1, 3, 416, 416]
    input_h, input_w = input_shape[2], input_shape[3]
    
    class_names = {0: "1", 1: "ball"}
    
    # --- Fisheye Rectification Setup ---
    width, height = 640, 480
    
    # Empirical Camera Intrinsic Matrix (K)
    K = np.array([
        [320.0, 0.0, 320.0],
        [0.0, 320.0, 240.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)
    
    # Empirical Fisheye Distortion Coefficients [k1, k2, k3, k4]
    # Negative k1 corrects for barrel distortion
    D = np.array([-0.06, 0.02, -0.01, 0.002], dtype=np.float32)
    
    # Precompute undistortion and rectification maps for ultra-fast remapping
    # balance=0.0 crops the black borders out, giving a clean 640x480 flat frame
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, (width, height), np.eye(3), balance=0.0)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), new_K, (width, height), cv2.CV_16SC2)
    
    # Initialize OpenCV Camera (/dev/video1)
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        sys.exit(1)
        
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
    # Let auto-exposure settle
    time.sleep(1.5)
    print("\nStarting live fisheye rectification & detection using OpenCV.")
    print("This script flattens the wide-angle image and runs YOLO detection on the flat image.")
    print("Press Ctrl+C to stop.\n")
    
    saved_visuals = False
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("Error: Failed to grab frame.")
                time.sleep(0.1)
                continue
            
            # --- Rectify the frame ---
            # cv2.remap runs in ~1.5ms on the Pi CPU
            flat_frame = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            
            # Save visual validation files once
            if not saved_visuals:
                cv2.imwrite(os.path.join(this_dir, "raw_distorted.jpg"), frame)
                cv2.imwrite(os.path.join(this_dir, "rectified_flat.jpg"), flat_frame)
                print(f"Saved 'raw_distorted.jpg' and 'rectified_flat.jpg' to {this_dir} for visual verification.")
                saved_visuals = True
            
            # --- Preprocessing Rectified Frame ---
            resized = cv2.resize(flat_frame, (input_w, input_h))
            
            # Convert channels BGR -> RGB for YOLO
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            input_data = rgb.astype(np.float32) / 255.0
            input_data = np.transpose(input_data, (2, 0, 1))
            input_data = np.expand_dims(input_data, axis=0)
            input_data = np.ascontiguousarray(input_data, dtype=np.float32)
            
            # --- Run Inference ---
            outputs = session.run(None, {input_name: input_data})
            output = np.transpose(outputs[0][0])  # Shape [3549, 6]
            max_conf = float(np.max(output[:, 4:]))
            
            print(f"[{time.strftime('%X')}] Max Raw Conf: {max_conf:.4f}")
            
            # Parse and print detections
            boxes = []
            confidences = []
            class_ids = []
            
            for row in output:
                xc, yc, w, h = row[0:4]
                scores = row[4:]
                class_id = np.argmax(scores)
                conf = float(scores[class_id])
                
                if conf >= 0.15:
                    x_scale = width / input_w
                    y_scale = height / input_h
                    x1 = (xc - w / 2) * x_scale
                    y1 = (yc - h / 2) * y_scale
                    w_box = w * x_scale
                    h_box = h * y_scale
                    
                    boxes.append([int(x1), int(y1), int(w_box), int(h_box)])
                    confidences.append(conf)
                    class_ids.append(int(class_id))
            
            if len(boxes) > 0:
                indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.15, 0.45)
                if len(indices) > 0:
                    flat_indices = indices.flatten() if hasattr(indices, 'flatten') else indices
                    print("  >> DETECTED:")
                    for idx in flat_indices:
                        name = class_names.get(class_ids[idx], f"class_{class_ids[idx]}")
                        conf = confidences[idx]
                        print(f"     - {name} (conf={conf:.2f})")
            
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\nStopping detection...")
    finally:
        cap.release()
        print("Camera stopped.")

if __name__ == "__main__":
    main()
