import os
import cv2
import pickle
import numpy as np
import face_recognition

MODEL_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "model")
PKL_PATH = os.path.join(MODEL_DIR, "faces.pkl")

def get_rgb_image(img_input):
    if isinstance(img_input, bytes):
        img_input = cv2.imdecode(np.frombuffer(img_input, np.uint8), cv2.IMREAD_COLOR)
    return cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB)

def detect_faces(img_input):
    try:
        rgb = get_rgb_image(img_input)
        locations = face_recognition.face_locations(rgb)
        return [(left, top, right - left, bottom - top) for top, right, bottom, left in locations]
    except Exception:
        return []

def has_face(img_input):
    return len(detect_faces(img_input)) > 0

def get_embedding(img_input):
    try:
        rgb = get_rgb_image(img_input)
        encodings = face_recognition.face_encodings(rgb)
        return encodings[0] if encodings else None
    except Exception:
        return None

def save_face_data(name, images_bytes):
    os.makedirs(MODEL_DIR, exist_ok=True)
    embeddings = []
    
    for i, img_bytes in enumerate(images_bytes):
        img_path = os.path.join(MODEL_DIR, f"{name}_{i+1}.jpg")
        with open(img_path, "wb") as f:
            f.write(img_bytes)
        
        emb = get_embedding(img_bytes)
        if emb is not None:
            embeddings.append(emb)
            
    if not embeddings:
        return False
        
    avg_embedding = np.mean(embeddings, axis=0)
    
    faces = []
    if os.path.exists(PKL_PATH):
        try:
            with open(PKL_PATH, "rb") as f:
                data = pickle.load(f)
                if isinstance(data, list):
                    faces = data
                elif isinstance(data, dict):
                    faces = [data]
        except Exception:
            pass
            
    faces = [f for f in faces if f.get("name", "").lower() != name.lower()]
    faces.append({"name": name, "encoding": avg_embedding})
    
    with open(PKL_PATH, "wb") as f:
        pickle.dump(faces if len(faces) > 1 else faces[0], f)
    return True
