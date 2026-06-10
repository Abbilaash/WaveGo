import cv2
import numpy as np
import io
import cv2
import requests


def detect(frame_rgb):
    """Send an RGB frame (numpy array) to the local detection server and return JSON.

    The server exposes `/detect` (GET) for latest detections when using the WebRTC
    streaming mode. For single-shot detection via POST, this function will POST
    the JPEG-encoded frame as field 'image'.
    """
    try:
        # Convert RGB -> BGR for OpenCV encoding
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        ret, buf = cv2.imencode('.jpg', frame_bgr)
        if not ret:
            return None

        files = {'image': ('frame.jpg', io.BytesIO(buf.tobytes()), 'image/jpeg')}
        resp = requests.post('http://127.0.0.1:5001/detect', files=files, timeout=5)
        if resp.status_code == 200:
            return resp.json()
        else:
            # Try to return JSON body if present for debugging
            try:
                return resp.json()
            except Exception:
                return None
    except Exception as e:
        print('detect() error:', e)
        return None
        resp = requests.post(DETECT_SERVER, files=files, timeout=30)
        if resp.status_code != 200:
            print(f"Detection server error: {resp.status_code} {resp.text}")
            return None
        return resp.json()
    except Exception as e:
        print(f"Error calling detection server: {e}")
        return None
    
