import cv2
import numpy as np

# Class names mapping: we define index 1 as 'ball' to match camera_opencv.py checks
CLASS_NAMES = ['0', 'ball']

def detect(frame, model_path=None, conf_threshold=0.25, input_is_rgb=False, crop_size=240):
    """
    Run green ball detection using color HSV filtering and contour circularity checking.
    Replaces the YOLOv8 ONNX-based model detection.
    """
    if frame is None or not hasattr(frame, 'shape') or len(frame.shape) < 3:
        return {"success": False, "detections": [], "annotated_frame": frame, "results": None, "error": "Invalid frame input"}
        
    try:
        # Determine color conversion based on input format
        if input_is_rgb:
            hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        else:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
        # Green HSV range
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        
        # Color mask
        mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Morphological operations to clean noise
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        mask = cv2.erode(mask, kernel, iterations=2)
        mask = cv2.dilate(mask, kernel, iterations=2)
        
        # Find contours
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        annotated_frame = frame.copy()
        
        for c in cnts:
            area = cv2.contourArea(c)
            if area < 150:  # Ignore small noise (bounding box must be of a reasonable size)
                continue
                
            perimeter = cv2.arcLength(c, True)
            if perimeter == 0:
                continue
                
            # Circularity metric: C = 4 * pi * Area / Perimeter^2
            circularity = 4 * np.pi * area / (perimeter ** 2)
            
            # Filter for circularity close to 1.0 (strict range 0.85 to 1.15)
            if 0.85 <= circularity <= 1.15:
                x, y, w, h = cv2.boundingRect(c)
                aspect_ratio = float(w) / h
                
                # Check aspect ratio to ensure it is not elongated (strict range 0.85 to 1.15)
                if 0.85 <= aspect_ratio <= 1.15:
                    # Calculate confidence: how close circularity is to 1.0
                    conf = max(0.0, min(1.0, 1.0 - abs(1.0 - circularity)))
                    
                    if conf < conf_threshold:
                        continue
                        
                    det = {
                        "x1": float(x),
                        "y1": float(y),
                        "x2": float(x + w),
                        "y2": float(y + h),
                        "conf": float(conf),
                        "class_id": 1,
                        "class_name": "ball"
                    }
                    detections.append(det)
                    
                    # Draw green bounding box and label
                    color = (74, 222, 128)  # Green color for drawing
                    cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), color, 2)
                    label = f"ball {conf:.2f}"
                    cv2.putText(annotated_frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    
        # Sort detections by bounding box area descending so the largest ball is first
        detections = sorted(detections, key=lambda d: (d["x2"] - d["x1"]) * (d["y2"] - d["y1"]), reverse=True)
        
        print(f"[Detect] Successfully detected {len(detections)} green ball(s).")
        return {"success": True, "detections": detections, "annotated_frame": annotated_frame, "results": None}
        
    except Exception as e:
        print(f"[Detect] Error during green ball detection: {e}")
        return {"success": False, "detections": [], "annotated_frame": frame, "results": None, "error": str(e)}



