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

# Extract weights from the ONNX model structure once
from onnx import numpy_helper
weights = {}
try:
    for initializer in onnx_model.graph.initializer:
        w_arr = numpy_helper.to_array(initializer)
        if initializer.name == "Parameter193":
            # Flattened fully connected weights of shape [16, 4, 4, 10] -> reshape to [256, 10]
            weights["fc"] = w_arr.reshape(256, 10).tolist()
        elif initializer.name == "Parameter194":
            # FC bias of shape [1, 10] -> flatten to [10]
            weights["bias3"] = w_arr.flatten().tolist()
        elif initializer.name == "Parameter5":
            # Conv1 weights of shape [8, 1, 5, 5]
            weights["conv1"] = w_arr.tolist()
        elif initializer.name == "Parameter6":
            # Conv1 bias of shape [8, 1, 1] -> flatten to [8]
            weights["bias1"] = w_arr.flatten().tolist()
        elif initializer.name == "Parameter87":
            # Conv2 weights of shape [16, 8, 5, 5]
            weights["conv2"] = w_arr.tolist()
        elif initializer.name == "Parameter88":
            # Conv2 bias of shape [16, 1, 1] -> flatten to [16]
            weights["bias2"] = w_arr.flatten().tolist()
except Exception as e:
    print(f"Error loading ONNX weights at startup: {e}")

def detect(frame):
    """
    Detects a single digit in the given frame using the LeNet5 MNIST CNN model.
    Assumes frame is a direct canvas drawing.
    """
    try:
        if frame is None or frame.size == 0:
            return {"success": False, "error": "Empty canvas drawing"}

        # Convert to Grayscale
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()

        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # Otsu's thresholding automatically calculates the optimal threshold value
        _, thresh = cv2.threshold(
            blur,
            0,
            255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        # Find contours inside the whole image
        contours, _ = cv2.findContours(
            thresh,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if len(contours) == 0:
            return {"success": False, "error": "No drawing marks detected on the canvas"}

        # Get the largest contour in the image
        largest = max(contours, key=cv2.contourArea)

        # Ignore if the detected contour is too tiny (noise)
        if cv2.contourArea(largest) < 2:
            return {"success": False, "error": "Drawing too faint or small"}

        # Get bounding box of largest contour
        x_c, y_c, w_c, h_c = cv2.boundingRect(largest)

        # Mask other noise to keep only the digit
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


def detect_explain(frame):
    """
    Runs LeNet5 inference, returns prediction, confidence, and ALL weights/activations.
    Assumes frame is a direct canvas drawing.
    """
    try:
        if frame is None or frame.size == 0:
            return {"success": False, "error": "Empty canvas drawing"}

        # Convert to Grayscale
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()

        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # Otsu's thresholding automatically calculates the optimal threshold value
        _, thresh = cv2.threshold(
            blur,
            0,
            255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        # Find contours inside the whole image
        contours, _ = cv2.findContours(
            thresh,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if len(contours) == 0:
            return {"success": False, "error": "No drawing marks detected on the canvas"}

        # Get the largest contour in the image
        largest = max(contours, key=cv2.contourArea)

        # Ignore if the detected contour is too tiny (noise)
        if cv2.contourArea(largest) < 2:
            return {"success": False, "error": "Drawing too faint or small"}

        # Get bounding box of largest contour
        x_c, y_c, w_c, h_c = cv2.boundingRect(largest)

        # Mask other noise to keep only the digit
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

        canvas = cv2.GaussianBlur(canvas, (3, 3), 0)

        # Normalize & Reshape
        img = canvas.astype(np.float32) / 255.0

        if channels == 3:
            img = cv2.merge([img, img, img])
            img = np.transpose(img, (2, 0, 1))
        else:
            img = img.reshape(1, height, width)

        img = np.expand_dims(img, axis=0)

        # Run inference capturing all intermediate activations
        output_names = [
            'Convolution28_Output_0',       # shape [1, 8, 28, 28]
            'ReLU32_Output_0',              # shape [1, 8, 28, 28] (Conv1 activations)
            'Pooling66_Output_0',           # shape [1, 8, 14, 14] (Pool1 activations)
            'Convolution110_Output_0',      # shape [1, 16, 14, 14]
            'ReLU114_Output_0',             # shape [1, 16, 14, 14] (Conv2 activations)
            'Pooling160_Output_0',          # shape [1, 16, 4, 4]   (Pool2 activations)
            'Pooling160_Output_0_reshape0', # shape [1, 256]        (FC inputs)
            'Plus214_Output_0'              # shape [1, 10]         (FC output logits)
        ]

        run_outputs = session.run(output_names, {input_name: img})
        activations = dict(zip(output_names, run_outputs))
        
        logits = activations['Plus214_Output_0'][0]
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        pred = int(np.argmax(probs))
        conf = float(probs[pred])

        # Prepare explanation response
        explanation = {
            "weights": weights,
            "activations": {
                "input": img[0][0].tolist(),  # shape [28, 28]
                "conv1": activations['ReLU32_Output_0'][0].tolist(),  # shape [8, 28, 28]
                "pool1": activations['Pooling66_Output_0'][0].tolist(),  # shape [8, 14, 14]
                "conv2": activations['ReLU114_Output_0'][0].tolist(),  # shape [16, 14, 14]
                "pool2": activations['Pooling160_Output_0'][0].tolist(),  # shape [16, 4, 4]
                "fc_input": activations['Pooling160_Output_0_reshape0'][0].tolist(),  # shape [256]
                "logits": logits.tolist(),  # shape [10]
                "probabilities": probs.tolist()  # shape [10]
            }
        }

        return {
            "success": True,
            "prediction": pred,
            "confidence": conf * 100.0,
            "explanation": explanation
        }
    except Exception as e:
        return {"success": False, "error": str(e)}
