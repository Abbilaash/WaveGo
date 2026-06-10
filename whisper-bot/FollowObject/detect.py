import cv2
import numpy as np
from inference_sdk import InferenceHTTPClient

# Initialize client
client = InferenceHTTPClient.init(
    api_url="https://serverless.roboflow.com",
    api_key="7nth6oqWo8gBn3lEJBvR"
)


def detect(frame):
    """
    Detection pipeline: takes a frame and returns the detected class label.
    
    Args:
        frame: numpy array representing the image frame or image file path
        
    Returns:
        str: The detected class label, or None if no detection found
    """
    try:
        # If frame is a numpy array, encode it to JPEG bytes
        if isinstance(frame, np.ndarray):
            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
        else:
            # Assume it's a file path
            with open(frame, 'rb') as f:
                frame_bytes = f.read()
        
        # Send frame to inference API
        result = client.infer(
            image_input=frame_bytes,
            workflow_id="detect-and-classify-2",
            workspace="infoverseericsson"
        )

        if not result or 'predictions' not in result:
            return None

        predictions = result['predictions']

        if 'detection_predictions' in predictions:
            detections = predictions['detection_predictions']
            if detections and len(detections) > 0:
                first = detections[0]
                label = first.get('class') or first.get('label') or first.get('name')
                confidence = first.get('confidence') or first.get('score') or first.get('probability') or 0.0
                box = None

                if 'box' in first and isinstance(first['box'], dict):
                    raw_box = first['box']
                    box = {
                        'x': int(raw_box.get('x', raw_box.get('left', 0))),
                        'y': int(raw_box.get('y', raw_box.get('top', 0))),
                        'w': int(raw_box.get('w', raw_box.get('width', raw_box.get('w', 0)))),
                        'h': int(raw_box.get('h', raw_box.get('height', raw_box.get('h', 0))))
                    }
                elif 'bbox' in first and isinstance(first['bbox'], (list, tuple)) and len(first['bbox']) >= 4:
                    x, y, w, h = first['bbox'][:4]
                    box = {'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h)}
                else:
                    x = first.get('x') or first.get('left') or first.get('xmin') or first.get('x1')
                    y = first.get('y') or first.get('top') or first.get('ymin') or first.get('y1')
                    w = first.get('w') or first.get('width') or first.get('x2') and int(first.get('x2', 0) - int(x or 0))
                    h = first.get('h') or first.get('height') or first.get('y2') and int(first.get('y2', 0) - int(y or 0))
                    if x is not None and y is not None and w is not None and h is not None:
                        try:
                            box = {'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h)}
                        except Exception:
                            box = None

                return {
                    'label': label,
                    'confidence': float(confidence),
                    'box': box,
                }

        if 'classification_predictions' in predictions:
            class_preds = predictions['classification_predictions']
            if isinstance(class_preds, dict) and class_preds:
                best_label, best_confidence = max(class_preds.items(), key=lambda item: item[1])
                return {
                    'label': best_label,
                    'confidence': float(best_confidence),
                    'box': None,
                }

        return None

    except Exception as e:
        print(f"Error in detection pipeline: {e}")
        return None
