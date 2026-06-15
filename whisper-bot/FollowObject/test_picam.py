#!/usr/bin/env python3
"""
Live camera test script that captures frames from a normal webcam (non-wide angle)
using OpenCV and runs object detection on the cropped square frame.
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
    
    # --- Camera Resolution Setup ---
    width, height = 640, 480
    
    # Initialize OpenCV Camera (/dev/video1)
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        sys.exit(1)
        
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
    # Let auto-exposure settle
    time.sleep(1.5)
    print("\nStarting live object detection on normal webcam using OpenCV.")
    print("Press Ctrl+C to stop.\n")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("Error: Failed to grab frame.")
                time.sleep(0.1)
                continue
            
            # --- Center Crop the frame to square (preserves aspect ratio) ---
            h_orig, w_orig = frame.shape[:2]
            start_x, start_y = 0, 0
            w_crop, h_crop = w_orig, h_orig
            
            if w_orig > h_orig:
                start_x = (w_orig - h_orig) // 2
                w_crop = h_orig
                cropped = frame[:, start_x:start_x + w_crop]
            elif h_orig > w_orig:
                start_y = (h_orig - w_orig) // 2
                h_crop = w_orig
                cropped = frame[start_y:start_y + h_crop, :]
            else:
                cropped = frame
            
            # --- Preprocessing cropped frame ---
            resized = cv2.resize(cropped, (input_w, input_h))
            
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
                    x_scale = w_crop / input_w
                    y_scale = h_crop / input_h
                    x1 = (xc - w / 2) * x_scale + start_x
                    y1 = (yc - h / 2) * y_scale + start_y
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
                    annotated = frame.copy()
                    for idx in flat_indices:
                        name = class_names.get(class_ids[idx], f"class_{class_ids[idx]}")
                        conf = confidences[idx]
                        print(f"     - {name} (conf={conf:.2f})")
                        
                        # Draw bounding box
                        bx, by, bw, bh = boxes[idx]
                        cv2.rectangle(annotated, (bx, by), (bx + bw, by + bh), (74, 222, 128), 2)
                        cv2.putText(annotated, f"{name} {conf:.2f}", (bx, by - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (74, 222, 128), 1, cv2.LINE_AA)
                    
                    # Save annotated frame for diagnostics
                    cv2.imwrite(os.path.join(this_dir, "detected.jpg"), annotated)
            
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\nStopping detection...")
    finally:
        cap.release()
        print("Camera stopped.")

if __name__ == "__main__":
    main()
