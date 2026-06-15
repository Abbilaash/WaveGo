import cv2
import numpy as np
import onnxruntime as ort
import os

# Global session cache
_session = None

def get_session(model_path=None):
	"""Load and cache the ONNX runtime inference session."""
	global _session
	if _session is not None:
		return _session
	
	if model_path is None:
		# Use the wide-angle optimized ONNX model
		model_path = os.path.join(os.path.dirname(__file__), "best-wide-angle.onnx")
	
	# Load with CPU execution provider for Raspberry Pi compatibility
	_session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
	return _session

def detect(frame, model_path=None, conf_threshold=0.10, iou_threshold=0.45, input_is_rgb=False, crop_size=240):
	"""
	Run object detection using pure onnxruntime.
	Avoids importing PyTorch/TensorFlow/Ultralytics.
	"""
	try:
		session = get_session(model_path)
		
		# Get input details
		input_name = session.get_inputs()[0].name
		input_shape = session.get_inputs()[0].shape  # Expecting [1, 3, 416, 416]
		input_h, input_w = input_shape[2], input_shape[3]
		h_orig, w_orig = frame.shape[:2]
		
		# Define crop size (e.g., 240 for zoom, or min(h, w) for no aspect ratio distortion)
		# A smaller crop size acts as a digital zoom, making the ball look larger and removing distortion
		crop_size = min(h_orig, w_orig, crop_size)
		start_x = (w_orig - crop_size) // 2
		start_y = (h_orig - crop_size) // 2
		cropped = frame[start_y:start_y + crop_size, start_x:start_x + crop_size]
		
		# Preprocess frame
		resized = cv2.resize(cropped, (input_w, input_h))
		
		# Model expects RGB format.
		if not input_is_rgb:
			rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
		else:
			rgb = resized
		input_data = rgb.astype(np.float32) / 255.0
		input_data = np.transpose(input_data, (2, 0, 1))  # HWC to CHW
		input_data = np.expand_dims(input_data, axis=0)   # Add batch dim
		
		# CRITICAL: Force contiguous C-order memory layout.
		input_data = np.ascontiguousarray(input_data, dtype=np.float32)
		
		# Run Inference
		outputs = session.run(None, {input_name: input_data})
		output = outputs[0][0]  # Shape: [6, 3549]
		
		# Transpose to [3549, 6] where columns are [xc, yc, w, h, class0_score, class1_score]
		output = np.transpose(output)
		
		max_conf = float(np.max(output[:, 4:]))
		print(f"[Detect] Max raw confidence score: {max_conf:.4f}")
		
		boxes = []
		confidences = []
		class_ids = []
		
		for row in output:
			xc, yc, w, h = row[0:4]
			scores = row[4:]
			class_id = np.argmax(scores)
			conf = float(scores[class_id])
			
			if conf >= conf_threshold:
				# Convert center coords [xc, yc, w, h] to top-left [x, y, w, h] of original frame
				x_scale = crop_size / input_w
				y_scale = crop_size / input_h
				
				x1 = (xc - w / 2) * x_scale + start_x
				y1 = (yc - h / 2) * y_scale + start_y
				w_box = w * x_scale
				h_box = h * y_scale
				
				boxes.append([int(x1), int(y1), int(w_box), int(h_box)])
				confidences.append(conf)
				class_ids.append(int(class_id))
				
		# Non-Maximum Suppression (NMS)
		indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, iou_threshold)
		
		detections = []
		annotated_frame = frame.copy()
		
		if len(indices) > 0:
			# Handle both OpenCV 4.x formats for NMS indexes
			flat_indices = indices.flatten() if hasattr(indices, 'flatten') else indices
			for idx in flat_indices:
				box = boxes[idx]
				x1, y1, w_box, h_box = box
				conf = confidences[idx]
				class_id = class_ids[idx]
				
				det = {
					"x1": float(x1),
					"y1": float(y1),
					"x2": float(x1 + w_box),
					"y2": float(y1 + h_box),
					"conf": conf,
					"class_id": class_id
				}
				detections.append(det)
				
				# Render bounding box only (no label text)
				color = (74, 222, 128) if class_id == 1 else (64, 128, 255)  # Green for ball, orange for class 0
				cv2.rectangle(annotated_frame, (x1, y1), (x1 + w_box, y1 + h_box), color, 2)
				
		return {
			"success": True,
			"detections": detections,
			"annotated_frame": annotated_frame,
			"results": None
		}
		
	except Exception as e:
		return {
			"success": False,
			"detections": [],
			"annotated_frame": frame,
			"results": None,
			"error": str(e)
		}
