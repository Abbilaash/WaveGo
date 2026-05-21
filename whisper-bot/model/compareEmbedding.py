import face_recognition
import pickle
import numpy as np

# Load known face
with open("faces.pkl", "rb") as f:
    known = pickle.load(f)

known_name = known["name"]
known_encoding = known["encoding"]

# Load test image
image = face_recognition.load_image_file("test.jpg")

# Encode detected faces
encodings = face_recognition.face_encodings(image)

for encoding in encodings:

    distance = np.linalg.norm(
        known_encoding - encoding
    )

    print("Distance:", distance)

    if distance < 0.5:
        print("Recognized:", known_name)

    else:
        print("Unknown person")