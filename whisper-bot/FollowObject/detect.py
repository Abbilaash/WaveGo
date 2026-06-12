import cv2
from ultralytics import YOLO
import os

# Global model cache
_model = None

def get_model(model_path=None):
	"""Load and cache the YOLO ONNX model."""
	global _model
	if _model is not None:
		return _model
	
	if model_path is None:
		model_path = os.path.join(os.path.dirname(__file__), "best.onnx")
	
	_model = YOLO(model_path)
	return _model

def detect(frame, model_path=None):
	try:
		# Load model
		model = get_model(model_path)
		
		# Run inference
		results = model.predict(
			frame,
			conf=0.25,
			verbose=False
		)
		
		if not results or len(results) == 0:
			return {
				"success": False,
				"detections": [],
				"annotated_frame": frame,
				"results": None
			}
		
		result = results[0]
		
		# Extract detection data
		detections = []
		if result.boxes is not None:
			for box in result.boxes:
				det = {
					"x1": float(box.xyxy[0][0]),
					"y1": float(box.xyxy[0][1]),
					"x2": float(box.xyxy[0][2]),
					"y2": float(box.xyxy[0][3]),
					"conf": float(box.conf[0]),
					"class_id": int(box.cls[0]),
					"class_name": result.names[int(box.cls[0])] if int(box.cls[0]) in result.names else f"class_{int(box.cls[0])}"
				}
				detections.append(det)
		
		# Draw annotations on frame
		annotated = result.plot()
		
		return {
			"success": True,
			"detections": detections,
			"annotated_frame": annotated,
			"results": result
		}
		
	except Exception as e:
		return {
			"success": False,
			"detections": [],
			"annotated_frame": frame,
			"results": None,
			"error": str(e)
		}

# Standalone test mode
'''if __name__ == "__main__":
	cap = cv2.VideoCapture(0)
	while True:
		ret, frame = cap.read()
		if not ret:
			break
		
		result = detect(frame)
		
		cv2.imshow("ONNX Detection", result["annotated_frame"])
		if cv2.waitKey(1) & 0xFF == ord("q"):
			break
	
	cap.release()
	cv2.destroyAllWindows()'''
