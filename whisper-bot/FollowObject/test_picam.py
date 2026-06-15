#!/usr/bin/env python3
"""
Self-contained object detection script for Raspberry Pi using Picamera2 and ONNX runtime.
Runs headlessly, parses detections, and prints results without relying on detect.py.
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
    
    # Initialize and configure Picamera2
    picam2 = Picamera2()
    config = picam2.create_video_configuration(
        main={"format": "RGB888", "size": (640, 480)}
    )
    picam2.configure(config)
    picam2.start()
    
    # Let auto-exposure and white balance settle
    time.sleep(1.5)
    print("\nStarting independent live detection.")
    print("This script runs inference on every frame using BOTH channel interpretations:")
    print("  1. BGR-to-RGB (treats frame as BGR and converts to RGB)")
    print("  2. As-Is (treats frame as RGB directly, no conversion)")
    print("Press Ctrl+C to stop.\n")
    
    try:
        while True:
            frame = picam2.capture_array()
            if frame is None:
                print("Error: Failed to grab frame from Picamera2.")
                time.sleep(0.1)
                continue
            
            # --- Preprocessing with Center Cropping ---
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
                
            resized = cv2.resize(cropped, (input_w, input_h))
            
            # Path A: Treats frame as BGR and swaps channels to RGB
            rgb_a = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            input_a = rgb_a.astype(np.float32) / 255.0
            input_a = np.transpose(input_a, (2, 0, 1))
            input_a = np.expand_dims(input_a, axis=0)
            input_a = np.ascontiguousarray(input_a, dtype=np.float32)
            
            # Path B: Treats frame as RGB, no conversion
            input_b = resized.astype(np.float32) / 255.0
            input_b = np.transpose(input_b, (2, 0, 1))
            input_b = np.expand_dims(input_b, axis=0)
            input_b = np.ascontiguousarray(input_b, dtype=np.float32)
            
            # --- Run Inference Path A ---
            outputs_a = session.run(None, {input_name: input_a})
            output_a = np.transpose(outputs_a[0][0])  # Shape: [3549, 6]
            max_conf_a = float(np.max(output_a[:, 4:]))
            
            # --- Run Inference Path B ---
            outputs_b = session.run(None, {input_name: input_b})
            output_b = np.transpose(outputs_b[0][0])  # Shape: [3549, 6]
            max_conf_b = float(np.max(output_b[:, 4:]))
            
            # Output diagnostics on every frame
            print(f"[{time.strftime('%X')}] Max Raw Score -> BGR-to-RGB: {max_conf_a:.4f} | As-Is (RGB): {max_conf_b:.4f}")
            
            # Helper to parse and print boxes
            def parse_and_print(output, mode_name, threshold=0.15):
                boxes = []
                confidences = []
                class_ids = []
                
                for row in output:
                    xc, yc, w, h = row[0:4]
                    scores = row[4:]
                    class_id = np.argmax(scores)
                    conf = float(scores[class_id])
                    
                    if conf >= threshold:
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
                    indices = cv2.dnn.NMSBoxes(boxes, confidences, threshold, 0.45)
                    if len(indices) > 0:
                        flat_indices = indices.flatten() if hasattr(indices, 'flatten') else indices
                        print(f"  >> {mode_name} DETECTED:")
                        for idx in flat_indices:
                            name = class_names.get(class_ids[idx], f"class_{class_ids[idx]}")
                            conf = confidences[idx]
                            box = boxes[idx]
                            print(f"     * {name} (conf={conf:.2f}) at box=[{box[0]},{box[1]},{box[0]+box[2]},{box[1]+box[3]}]")
            
            parse_and_print(output_a, "BGR-to-RGB", threshold=0.15)
            parse_and_print(output_b, "As-Is (RGB)", threshold=0.15)
            
            time.sleep(0.1)  # Yield CPU and prevent terminal flooding
            
    except KeyboardInterrupt:
        print("\nStopping detection...")
    finally:
        picam2.stop()
        print("Camera stopped.")

if __name__ == "__main__":
    main()
