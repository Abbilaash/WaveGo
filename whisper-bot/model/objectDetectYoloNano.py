import os
import cv2
import numpy as np
import onnxruntime as ort

# Resolve the ONNX model path dynamically
MODEL_DIR = os.path.dirname(os.path.realpath(__file__))
CANDIDATE_PATHS = [
    os.path.join(MODEL_DIR, "yolov8n.onnx"),
    os.path.join(MODEL_DIR, "..", "yolov8n.onnx"),
    os.path.join(os.getcwd(), "yolov8n.onnx"),
    "yolov8n.onnx"
]

model_path = None
for path in CANDIDATE_PATHS:
    if os.path.exists(path):
        model_path = path
        break

if model_path is None:
    # Default fallback
    model_path = os.path.join(MODEL_DIR, "yolov8n.onnx")

# Load ONNX model once
try:
    session = ort.InferenceSession(
        model_path,
        providers=["CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name
except Exception as e:
    print(f"Error loading ONNX model at {model_path}: {e}")
    session = None
    input_name = None

# COCO labels
classes = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella",
    "handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite",
    "baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle",
    "wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich",
    "orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch",
    "potted plant","bed","dining table","toilet","tv","laptop","mouse","remote",
    "keyboard","cell phone","microwave","oven","toaster","sink","refrigerator","book",
    "clock","vase","scissors","teddy bear","hair drier","toothbrush"
]

def detect_objects(frame):
    """
    Detect objects in the given BGR image frame.
    Returns a list of dicts with keys 'box' (x, y, w, h), 'label', and 'confidence'.
    """
    if session is None:
        return []

    orig_h, orig_w = frame.shape[:2]

    # Preprocess
    img = cv2.resize(frame, (640, 640))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    img = np.ascontiguousarray(img)

    # Inference
    outputs = session.run(None, {input_name: img})
    predictions = outputs[0][0].T

    # Detection parsing
    boxes = []
    scores = []
    class_ids = []

    for pred in predictions:
        x_center, y_center, width, height = pred[:4]
        class_scores = pred[4:]
        class_id = np.argmax(class_scores)
        confidence = class_scores[class_id]

        if confidence < 0.4:
            continue

        # Convert to xyxy
        x1 = int((x_center - width / 2) * orig_w / 640)
        y1 = int((y_center - height / 2) * orig_h / 640)
        w_box = int(width * orig_w / 640)
        h_box = int(height * orig_h / 640)

        boxes.append([x1, y1, w_box, h_box])
        scores.append(float(confidence))
        class_ids.append(class_id)

    # Apply NMS
    indices = cv2.dnn.NMSBoxes(
        boxes,
        scores,
        score_threshold=0.4,
        nms_threshold=0.5
    )

    results = []
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w_box, h_box = boxes[i]
            label = classes[class_ids[i]]
            confidence = scores[i]

            print(f"Detected: {label}")

            results.append({
                "box": (x, y, w_box, h_box),
                "label": label,
                "confidence": confidence
            })

    return results

if __name__ == "__main__":
    from picamera2 import Picamera2
    import time

    print("Running YOLOv8 Nano object detection test...")
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"size": (640, 480)}
    )
    picam2.configure(config)
    picam2.start()

    try:
        time.sleep(2)
        frame = picam2.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        objects = detect_objects(frame)
        for obj in objects:
            x, y, w_box, h_box = obj["box"]
            cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"{obj['label']} {obj['confidence']:.2f}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        cv2.imwrite("detect_output.jpg", frame)
        print("Saved detect_output.jpg")
    finally:
        picam2.stop()
