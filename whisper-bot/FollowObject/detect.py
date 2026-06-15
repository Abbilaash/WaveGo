import cv2
import numpy as np
import onnxruntime as ort
import os

# Optional class names mapping – the current model detects a single class (ball)
CLASS_NAMES = ['ball']

# Global session cache
_session = None
_model_path_cached = None

def get_session(model_path=None):
    """Load and cache the ONNX runtime inference session."""
    global _session
    global _session, _model_path_cached
    if model_path is None:
        model_path = os.path.join(os.path.dirname(__file__), "best-wide-angle.onnx")
    if _session is not None and _model_path_cached == model_path:
        return _session
    try:
        _session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        _model_path_cached = model_path
    except Exception as e:
        print(f"[Detect] Failed to load model at {model_path}: {e}")
        fallback_path = os.path.join(os.path.dirname(__file__), "best-wide-angle.onnx")
        print(f"[Detect] Attempting fallback model at {fallback_path}")
        _session = ort.InferenceSession(fallback_path, providers=["CPUExecutionProvider"])
        _model_path_cached = fallback_path
    return _session

def detect(frame, model_path=None, conf_threshold=0.25, input_is_rgb=False, crop_size=240):
    """
    Run object detection using an End-to-End ONNX runtime model (exported with nms=True).
    Output shape is expected to be [1, 300, 6] in [x1, y1, x2, y2, conf, class_id] format.
    """
    try:
        session = get_session(model_path)
        input_name = session.get_inputs()[0].name
        input_shape = session.get_inputs()[0].shape  # [1, 3, 416, 416]
        input_h, input_w = input_shape[2], input_shape[3]
        print(f"[Detect] Model loaded from {model_path}, input shape {input_shape}")
        h_orig, w_orig = frame.shape[:2]
        print(f"[Detect] Processing frame size {h_orig}x{w_orig}, crop size {crop_size}")
        # Wide-angle cropping / digital zoom
        crop_size = min(h_orig, w_orig, crop_size)
        start_x = (w_orig - crop_size) // 2
        start_y = (h_orig - crop_size) // 2
        cropped = frame[start_y:start_y + crop_size, start_x:start_x + crop_size]
        # Resize and color formatting
        resized = cv2.resize(cropped, (input_w, input_h))
        rgb = resized if input_is_rgb else cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        # Prepare input tensor
        input_data = rgb.astype(np.float32) / 255.0
        input_data = np.transpose(input_data, (2, 0, 1))
        input_data = np.expand_dims(input_data, axis=0)
        input_data = np.ascontiguousarray(input_data, dtype=np.float32)
        # Inference
        outputs = session.run(None, {input_name: input_data})
        detections_raw = outputs[0][0]
        detections = []
        annotated_frame = frame.copy()
        x_scale = crop_size / input_w
        y_scale = crop_size / input_h
        for row in detections_raw:
            x1, y1, x2, y2, conf, class_id = row
            # Remove confidence threshold filter to display all detections
            # if conf < conf_threshold:
            #     continue
            real_x1 = int(x1 * x_scale + start_x)
            real_y1 = int(y1 * y_scale + start_y)
            real_x2 = int(x2 * x_scale + start_x)
            real_y2 = int(y2 * y_scale + start_y)
            det = {
                "x1": float(real_x1),
                "y1": float(real_y1),
                "x2": float(real_x2),
                "y2": float(real_y2),
                "conf": float(conf),
                "class_id": int(class_id),
                "class_name": CLASS_NAMES[int(class_id)] if CLASS_NAMES and int(class_id) < len(CLASS_NAMES) else f"class_{int(class_id)}"
            }
            detections.append(det)
            color = (74, 222, 128) if int(class_id) == 1 else (64, 128, 255)
            cv2.rectangle(annotated_frame, (real_x1, real_y1), (real_x2, real_y2), color, 2)
        print(f"[Detect] Successfully extracted {len(detections)} valid objects.")
        return {"success": True, "detections": detections, "annotated_frame": annotated_frame, "results": None}
    except Exception as e:
        return {"success": False, "detections": [], "annotated_frame": frame, "results": None, "error": str(e)}


