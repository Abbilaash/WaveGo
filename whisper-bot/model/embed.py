import face_recognition
import pickle

# Load image
image = face_recognition.load_image_file("stella.jpg")

# Get face embedding
encoding = face_recognition.face_encodings(image)[0]

# Save
data = {
    "name": "Stella",
    "encoding": encoding
}

with open("faces.pkl", "wb") as f:
    pickle.dump(data, f)

print("Face saved")