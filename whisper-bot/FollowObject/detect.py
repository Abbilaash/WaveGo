import cv2
import numpy as np
import requests
import io

DETECT_SERVER = "http://127.0.0.1:5001/detect"


def detect(frame):
    """
    Send a frame to the external detection server and return the detection payload.

    Args:
        frame: numpy array (BGR) or path to an image file.

    Returns:
        dict: detection server response (contains 'detections' list and raw payload), or None on error.
    """
    try:
        # Prepare JPEG bytes
        if isinstance(frame, np.ndarray):
            _, buffer = cv2.imencode('.jpg', frame)
            img_bytes = buffer.tobytes()
        else:
            with open(frame, 'rb') as f:
                img_bytes = f.read()

        files = {'image': ('frame.jpg', io.BytesIO(img_bytes), 'image/jpeg')}
        resp = requests.post(DETECT_SERVER, files=files, timeout=30)
        if resp.status_code != 200:
            print(f"Detection server error: {resp.status_code} {resp.text}")
            return None
        return resp.json()
    except Exception as e:
        print(f"Error calling detection server: {e}")
        return None
    
