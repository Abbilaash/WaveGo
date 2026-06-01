import cv2
import numpy as np

# ---------------------------------
# Camera
# ---------------------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open webcam")
    exit()


def capture_average_frame(cap, n=3):
    frames = []

    for i in range(n):

        ret, frame = cap.read()

        if not ret:
            raise RuntimeError("Failed to capture frame")

        frames.append(frame.astype(np.float32))

        cv2.waitKey(200)

    avg = np.mean(frames, axis=0)

    return avg.astype(np.uint8)


# ---------------------------------
# Background
# ---------------------------------
input("Remove object and press ENTER...")

background = capture_average_frame(cap, 3)

cv2.imwrite("background.jpg", background)

print("Background captured")


# ---------------------------------
# Object Image
# ---------------------------------
input("Place object and press ENTER...")

frame = capture_average_frame(cap, 3)

cv2.imwrite("object.jpg", frame)

print("Object image captured")

cap.release()

# ---------------------------------
# Difference
# ---------------------------------
diff = cv2.absdiff(frame, background)

gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

gray = cv2.GaussianBlur(gray, (5, 5), 0)

_, thresh = cv2.threshold(
    gray,
    30,
    255,
    cv2.THRESH_BINARY
)

# ---------------------------------
# Morphological cleanup
# ---------------------------------
kernel = np.ones((7, 7), np.uint8)

thresh = cv2.morphologyEx(
    thresh,
    cv2.MORPH_CLOSE,
    kernel,
    iterations=2
)

thresh = cv2.dilate(
    thresh,
    kernel,
    iterations=2
)

# ---------------------------------
# Find contours
# ---------------------------------
contours, _ = cv2.findContours(
    thresh,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

if len(contours) == 0:
    print("No new object detected")
    exit()

# ---------------------------------
# Largest contour
# ---------------------------------
largest = max(contours, key=cv2.contourArea)

area = cv2.contourArea(largest)

print("Detected area:", area)

if area < 1500:
    print("Object too small")
    exit()

# ---------------------------------
# Bounding box
# ---------------------------------
x, y, w, h = cv2.boundingRect(largest)

result = frame.copy()

cv2.rectangle(
    result,
    (x, y),
    (x + w, y + h),
    (0, 255, 0),
    3
)

# ---------------------------------
# Save outputs
# ---------------------------------
cv2.imwrite("difference_mask.jpg", thresh)
cv2.imwrite("detected_object.jpg", result)

print(f"Bounding box: x={x}, y={y}, w={w}, h={h}")

cv2.imshow("Difference Mask", thresh)
cv2.imshow("Detected Object", result)

cv2.waitKey(0)
cv2.destroyAllWindows()