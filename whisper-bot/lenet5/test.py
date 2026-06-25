import cv2
import numpy as np
import onnxruntime as ort

# =========================
# LOAD MODEL
# =========================

model_path = "mnist-12.onnx"

# =========================
# LOAD AND EXPOSE INTERMEDIATE TENSORS IN-MEMORY
# =========================
# ONNX Runtime by default only exposes graph outputs. We load the model with the 'onnx' library,
# register all intermediate value_info tensors as graph outputs, and load the modified bytes.
import onnx
onnx_model = onnx.load(model_path)

# Extract all intermediate value info tensors from graph and append them to graph outputs
for value_info in onnx_model.graph.value_info:
    onnx_model.graph.output.append(value_info)

# Serialize the modified graph to memory bytes
model_bytes = onnx_model.SerializeToString()

session = ort.InferenceSession(
    model_bytes,
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name
input_details = session.get_inputs()[0]
input_shape = input_details.shape  # e.g. [1, 1, 28, 28]

print("Model Loaded:", model_path)
print("Expected Input Shape:", input_shape)
print("Active Session Outputs:", [o.name for o in session.get_outputs()])

# Extract batch, channels, height, width dynamically from the model
# Handle dynamic batch size if specified as string/None
batch = input_shape[0] if isinstance(input_shape[0], int) else 1
channels = input_shape[1]
height = input_shape[2]
width = input_shape[3]

# =========================
# CAMERA
# =========================

cap = cv2.VideoCapture(0)

print("\nSPACE = Predict (Align digit in green box)")
print("Q = Quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    display = frame.copy()
    h_f, w_f = frame.shape[:2]

    # Define a 200x200 Region of Interest (ROI) box in the center
    roi_size = 200
    x1 = (w_f - roi_size) // 2
    y1 = (h_f - roi_size) // 2
    x2 = x1 + roi_size
    y2 = y1 + roi_size

    # Draw the green ROI box and guidelines
    cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(
        display,
        "SPACE = Predict (Align digit in box)",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.imshow("Camera", display)

    key = cv2.waitKey(1)
    if key == ord("q"):
        break

    if key == 32:  # SPACE
        # =========================
        # PREPROCESS (Within ROI)
        # =========================
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
            print("No digit found in the target area.")
            continue

        # Get the largest contour in the ROI
        largest = max(contours, key=cv2.contourArea)

        # Ignore if the detected contour is too tiny (noise)
        if cv2.contourArea(largest) < 80:
            print("Digit too small or low contrast.")
            continue

        # Ignore if the contour boundary spans the entire ROI (detecting ROI box edges)
        x_c, y_c, w_c, h_c = cv2.boundingRect(largest)
        if w_c >= roi_size - 10 and h_c >= roi_size - 10:
            print("Ignoring ROI border detection. Keep the digit inside the green box.")
            continue

        # Mask other noise in the ROI to keep only the digit
        mask = np.zeros_like(thresh)
        cv2.drawContours(mask, [largest], -1, 255, thickness=cv2.FILLED)
        digit = cv2.bitwise_and(thresh, mask)

        # Crop tightly to the digit's bounding box
        digit_crop = digit[y_c:y_c+h_c, x_c:x_c+w_c]

        # =========================
        # PRESERVE ASPECT RATIO (PAD TO SQUARE)
        # =========================
        size = max(w_c, h_c)
        square = np.zeros((size, size), dtype=np.uint8)
        x_offset = (size - w_c) // 2
        y_offset = (size - h_c) // 2
        square[y_offset:y_offset+h_c, x_offset:x_offset+w_c] = digit_crop

        # =========================
        # RESIZE & CENTER DYNAMICALLY
        # =========================
        # Leave a ~15% margin on each side (digit takes 70% of canvas)
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

        cv2.imshow("Network Input Preview", canvas)

        # =========================
        # NORMALIZE & RESHAPE
        # =========================
        img = canvas.astype(np.float32) / 255.0

        # Adjust dimensions based on the expected input shape dynamically
        # Models can expect NCHW [batch, channels, height, width]
        if channels == 3:
            # RGB/BGR expected
            img = cv2.merge([img, img, img])
            img = np.transpose(img, (2, 0, 1))  # to channels first (3, H, W)
        else:
            # Grayscale expected (1, H, W)
            img = img.reshape(1, height, width)

        # Add batch dimension: [1, C, H, W]
        img = np.expand_dims(img, axis=0)

        # =========================
        # INFERENCE & ACTIVATION VISUALIZATION
        # =========================
        # To inspect and output internal activations, we query every node's output tensor
        output_names = [
            'Convolution28_Output_0',
            'Plus30_Output_0',
            'ReLU32_Output_0',
            'Pooling66_Output_0',
            'Convolution110_Output_0',
            'Plus112_Output_0',
            'ReLU114_Output_0',
            'Pooling160_Output_0',
            'Times212_Output_0',
            'Plus214_Output_0'
        ]

        run_outputs = session.run(
            output_names,
            {input_name: img}
        )

        activations = dict(zip(output_names, run_outputs))
        logits = activations['Plus214_Output_0'][0]

        # Extract weights from the ONNX model structure to print in detail
        import onnx
        from onnx import numpy_helper
        onnx_model = onnx.load(model_path)
        weights = {}
        for initializer in onnx_model.graph.initializer:
            w_arr = numpy_helper.to_array(initializer)
            weights[initializer.name] = w_arr

        # =========================
        # SOFTMAX
        # =========================
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        pred = np.argmax(probs)

        # Print weights and node firing detail
        print("\n==================================================")
        print("    DETAILED NEURAL NET WEIGHTS & ACTIVATIONS     ")
        print("==================================================")
        
        print("\n--- LAYER 1: CONVOLUTION + RELU + MAXPOOL ---")
        print(f"Conv1 Weights (Parameter5) shape: {weights['Parameter5'].shape}")
        print(f"Conv1 Weights Mean: {np.mean(weights['Parameter5']):.6f}, Std: {np.std(weights['Parameter5']):.6f}")
        print(f"Bias1 Weights (Parameter6) values: {weights['Parameter6'].flatten()}")
        print(f"Conv1 Output ('Convolution28_Output_0') shape: {activations['Convolution28_Output_0'].shape}")
        print(f"ReLU1 Output ('ReLU32_Output_0') shape: {activations['ReLU32_Output_0'].shape}")
        print(f"ReLU1 Active Nodes (firing > 0): {np.sum(activations['ReLU32_Output_0'] > 0)} / {activations['ReLU32_Output_0'].size}")
        print(f"Maxpool1 Output ('Pooling66_Output_0') shape: {activations['Pooling66_Output_0'].shape}")
        print(f"Maxpool1 Output Mean: {np.mean(activations['Pooling66_Output_0']):.4f}")

        print("\n--- LAYER 2: CONVOLUTION + RELU + MAXPOOL ---")
        print(f"Conv2 Weights (Parameter87) shape: {weights['Parameter87'].shape}")
        print(f"Conv2 Weights Mean: {np.mean(weights['Parameter87']):.6f}, Std: {np.std(weights['Parameter87']):.6f}")
        print(f"Bias2 Weights (Parameter88) values: {weights['Parameter88'].flatten()}")
        print(f"Conv2 Output ('Convolution110_Output_0') shape: {activations['Convolution110_Output_0'].shape}")
        print(f"ReLU2 Output ('ReLU114_Output_0') shape: {activations['ReLU114_Output_0'].shape}")
        print(f"ReLU2 Active Nodes (firing > 0): {np.sum(activations['ReLU114_Output_0'] > 0)} / {activations['ReLU114_Output_0'].size}")
        print(f"Maxpool2 Output ('Pooling160_Output_0') shape: {activations['Pooling160_Output_0'].shape}")
        print(f"Maxpool2 Output Mean: {np.mean(activations['Pooling160_Output_0']):.4f}")

        print("\n--- LAYER 3: FULLY CONNECTED (DENSE) ---")
        print(f"FC Weights (Parameter193) shape: {weights['Parameter193'].shape}")
        print(f"FC Weights Mean: {np.mean(weights['Parameter193']):.6f}, Std: {np.std(weights['Parameter193']):.6f}")
        print(f"Bias3 Weights (Parameter194) values: {weights['Parameter194'].flatten()}")
        print(f"FC Output ('Plus214_Output_0') logits: {logits}")

        print("\n========================")
        for i in range(10):
            print(f"Digit {i}: {probs[i]*100:.2f}%")
        print("\nPrediction:", pred)
        print("Confidence:", f"{probs[pred]*100:.2f}%")

cap.release()
cv2.destroyAllWindows()