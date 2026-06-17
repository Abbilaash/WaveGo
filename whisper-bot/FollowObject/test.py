import cv2
import os
import sys

# Add parent directory to path so we can import detect.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from detect import detect

def main():
	# Use model_path relative to test.py
	model_path = os.path.join(os.path.dirname(__file__), "best (4).onnx")
	print(f"Loading ONNX model from: {model_path}")
	
	cap = cv2.VideoCapture(0)
	if not cap.isOpened():
		print("Error: Could not open webcam.")
		return
		
	print("Starting object detection. Press 'q' to quit...")
	
	while True:
		ret, frame = cap.read()
		if not ret:
			print("Error: Failed to grab frame.")
			break
			
		# Run detection
		result = detect(frame, model_path=model_path, conf_threshold=0.25, iou_threshold=0.45)
		
		# Show the annotated frame
		cv2.imshow("ONNX Object Detection", result["annotated_frame"])
		
		# Log detections to console
		if result["success"] and result["detections"]:
			for det in result["detections"]:
				print(f"Detected: {det['class_name']} ({det['conf']:.2f}) at [{int(det['x1'])}, {int(det['y1'])}, {int(det['x2'])}, {int(det['y2'])}]")
		
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
			
	cap.release()
	cv2.destroyAllWindows()

if __name__ == "__main__":
	main()
