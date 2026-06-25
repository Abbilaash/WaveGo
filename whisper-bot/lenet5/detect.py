import os
import cv2
import numpy as np
import onnxruntime as ort
import onnx

# ==========================================================
# LOAD AND PREPARE MODEL ONCE
# ==========================================================
THIS_DIR = os.path.dirname(os.path.realpath(__file__))
model_path = os.path.join(THIS_DIR, "mnist-12.onnx")

# Load and expose intermediate output tensors in memory
onnx_model = onnx.load(model_path)
for value_info in onnx_model.graph.value_info:
    onnx_model.graph.output.append(value_info)
model_bytes = onnx_model.SerializeToString()

session = ort.InferenceSession(
    model_bytes,
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape
channels = input_shape[1]
height = input_shape[2]
width = input_shape[3]

def detect(frame):
    """
    Detects a single digit in the given frame using the LeNet5 MNIST CNN model.
    
    Args:
        frame (np.ndarray): Input image array (BGR color or grayscale).
        
    Returns:
        dict: {"success": bool, "prediction": int, "confidence": float, "error": str}
    """
    try:
        if frame is None or frame.size == 0:
            return {"success": False, "error": "Empty camera frame"}

        h_f, w_f = frame.shape[:2]

        # Define a 200x200 Region of Interest (ROI) box in the center
        roi_size = min(200, h_f, w_f)
        x1 = (w_f - roi_size) // 2
        y1 = (h_f - roi_size) // 2
        x2 = x1 + roi_size
        y2 = y1 + roi_size

        # Crop to the localized ROI box
        roi = frame[y1:y2, x1:x2]

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # Otsu's thresholding automatically calculates the optimal threshold value
        _, thresh = cv2.threshold(
            blur,
            0,
            255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        # Find contours inside the ROI
        contours, _ = cv2.findContours(
            thresh,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if len(contours) == 0:
            return {"success": False, "error": "No digit contour found in the target area"}

        # Get the largest contour in the ROI
        largest = max(contours, key=cv2.contourArea)

        # Ignore if the detected contour is too tiny (noise)
        if cv2.contourArea(largest) < 80:
            return {"success": False, "error": "Digit too small or low contrast"}

        # Ignore if the contour boundary spans the entire ROI (detecting ROI box edges)
        x_c, y_c, w_c, h_c = cv2.boundingRect(largest)
        if w_c >= roi_size - 10 and h_c >= roi_size - 10:
            return {"success": False, "error": "ROI border detected instead of a digit"}

        # Mask other noise in the ROI to keep only the digit
        mask = np.zeros_like(thresh)
        cv2.drawContours(mask, [largest], -1, 255, thickness=cv2.FILLED)
        digit = cv2.bitwise_and(thresh, mask)

        # Crop tightly to the digit's bounding box
        digit_crop = digit[y_c:y_c+h_c, x_c:x_c+w_c]

        # Preserve aspect ratio (pad to square)
        size = max(w_c, h_c)
        square = np.zeros((size, size), dtype=np.uint8)
        x_offset = (size - w_c) // 2
        y_offset = (size - h_c) // 2
        square[y_offset:y_offset+h_c, x_offset:x_offset+w_c] = digit_crop

        # Resize dynamically to 70% of model input height/width
        target_digit_h = int(height * 0.7)
        target_digit_w = int(width * 0.7)
        digit_resized = cv2.resize(square, (target_digit_w, target_digit_h))

        # Place the resized digit in the center of the model-sized canvas
        canvas = np.zeros((height, width), dtype=np.uint8)
        off_y = (height - target_digit_h) // 2
        off_x = (width - target_digit_w) // 2
        canvas[off_y:off_y+target_digit_h, off_x:off_x+target_digit_w] = digit_resized

        # Center by Center of Mass (Weighted Centroid) to match MNIST training distribution
        moments = cv2.moments(canvas)
        if moments["m00"] > 0:
            cx = moments["m10"] / moments["m00"]
            cy = moments["m01"] / moments["m00"]
            shift_x = (width / 2.0) - cx
            shift_y = (height / 2.0) - cy
            M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
            canvas = cv2.warpAffine(canvas, M, (width, height))

        # Apply a light Gaussian blur to smooth the strokes and anti-alias the digit (like MNIST)
        canvas = cv2.GaussianBlur(canvas, (3, 3), 0)

        # Normalize & Reshape
        img = canvas.astype(np.float32) / 255.0

        if channels == 3:
            img = cv2.merge([img, img, img])
            img = np.transpose(img, (2, 0, 1))
        else:
            img = img.reshape(1, height, width)

        # Add batch dimension: [1, C, H, W]
        img = np.expand_dims(img, axis=0)

        # Run inference
        logits = session.run(
            None,
            {input_name: img}
        )[0][0]

        # Softmax
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        pred = int(np.argmax(probs))
        conf = float(probs[pred])

        return {
            "success": True,
            "prediction": pred,
            "confidence": conf * 100.0
        }
    except Exception as e:
        return {"success": False, "error": str(e)}
