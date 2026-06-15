#!/usr/bin/env python3
"""
Combined method live detection script:
1. Grabs a wide-angle frame from Picamera2.
2. Rectifies/flattens the frame using OpenCV's fisheye model.
3. Center-crops a square from the flat frame (digital zoom + aspect ratio correction).
4. Infers the YOLOv8 ONNX model and prints detected labels.
5. Saves visual outputs to disk for inspection.
"""
import os
import sys
import time
import cv2
import numpy as np
import onnxruntime as ort

try:
    from picamera2 import Picamera2
except ImportError:
    print("Error: picamera2 module not found. This script must be run on a Raspberry Pi.")
    sys.exit(1)

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
    # balance=0.2 retains more FOV while keeping lines relatively straight
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
    print("\nStarting live combined method (Rectify + Center Crop) detection.")
    print("Press Ctrl+C to stop.\n")
    
    saved_visuals = False
    
    try:
        while True:
            frame = picam2.capture_array()
            if frame is None:
                print("Error: Failed to grab frame.")
                time.sleep(0.1)
                continue
            
            # --- Step 1: Rectify the frame ---
            flat_frame = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            
            # --- Step 2: Center-Crop a 240x240 square from the flat frame ---
            h_flat, w_flat = flat_frame.shape[:2]
            crop_size = min(h_flat, w_flat, 240)
            start_x = (w_flat - crop_size) // 2
            start_y = (h_flat - crop_size) // 2
            cropped_flat = flat_frame[start_y:start_y + crop_size, start_x:start_x + crop_size]
            w_crop, h_crop = crop_size, crop_size
            
            # Save visual checkpoints on the first frame
            if not saved_visuals:
                cv2.imwrite(os.path.join(this_dir, "step1_raw_distorted.jpg"), frame)
                cv2.imwrite(os.path.join(this_dir, "step2_rectified_flat.jpg"), flat_frame)
                cv2.imwrite(os.path.join(this_dir, "step3_rectified_cropped.jpg"), cropped_flat)
                print("Saved visualization stages to disk:")
                print("  - step1_raw_distorted.jpg")
                print("  - step2_rectified_flat.jpg")
                print("  - step3_rectified_cropped.jpg")
                saved_visuals = True
            
            # --- Step 3: Resize cropped frame for YOLO ---
            resized = cv2.resize(cropped_flat, (input_w, input_h))
            
            # Convert BGR to RGB for model input
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            input_data = rgb.astype(np.float32) / 255.0
            input_data = np.transpose(input_data, (2, 0, 1))
            input_data = np.expand_dims(input_data, axis=0)
            input_data = np.ascontiguousarray(input_data, dtype=np.float32)
            
            # --- Step 4: Run ONNX Inference ---
            outputs = session.run(None, {input_name: input_data})
            output = np.transpose(outputs[0][0])  # Shape [3549, 6]
            max_conf = float(np.max(output[:, 4:]))
            
            print(f"[{time.strftime('%X')}] Max Raw Conf: {max_conf:.4f}")
            
            # Parse detections
            boxes = []
            confidences = []
            class_ids = []
            
            for row in output:
                xc, yc, w, h = row[0:4]
                scores = row[4:]
                class_id = np.argmax(scores)
                conf = float(scores[class_id])
                
                if conf >= 0.15:
                    x_scale = w_crop / input_w
                    y_scale = h_crop / input_h
                    # Calculate coordinates relative to the original 640x480 frame
                    x1 = (xc - w / 2) * x_scale + start_x
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
                    
                    annotated_flat = flat_frame.copy()
                    for idx in flat_indices:
                        name = class_names.get(class_ids[idx], f"class_{class_ids[idx]}")
                        conf = confidences[idx]
                        box = boxes[idx]
                        x, y, wb, hb = box
                        
                        # Print details
                        print(f"     - class_id={class_ids[idx]} (conf={conf:.2f}) at [{x}, {y}, {x+wb}, {y+hb}]")
                        
                        # Draw bounding box only (no label text)
                        cv2.rectangle(annotated_flat, (x, y), (x + wb, y + hb), (74, 222, 128), 2)
                    
                    # Save the detection result
                    cv2.imwrite(os.path.join(this_dir, "detection_result.jpg"), annotated_flat)
            
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\nStopping detection...")
    finally:
        picam2.stop()
        print("Camera stopped.")

if __name__ == "__main__":
    main()
