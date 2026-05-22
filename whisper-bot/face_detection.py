import os
import cv2
import pickle
import numpy as np
import face_recognition
from logger import log_action

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
        faces = [(left, top, right - left, bottom - top) for top, right, bottom, left in locations]
        log_action("BACKEND", "Face Detection Run", f"Detected {len(faces)} faces.")
        return faces
    except Exception as e:
        log_action("BACKEND", "Face Detection Error", str(e))
        return []

def recognize_faces(img_input):
    try:
        rgb = get_rgb_image(img_input)
        locations = face_recognition.face_locations(rgb)
        if not locations:
            return []
            
        encodings = face_recognition.face_encodings(rgb, locations)
        
        known_names = []
        known_encodings = []
        if os.path.exists(PKL_PATH):
            try:
                with open(PKL_PATH, "rb") as f:
                    data = pickle.load(f)
                    if isinstance(data, list):
                        db_faces = data
                    elif isinstance(data, dict):
                        db_faces = [data]
                    else:
                        db_faces = []
                    for face_info in db_faces:
                        known_names.append(face_info.get("name", "Unknown"))
                        known_encodings.append(face_info.get("encoding"))
            except Exception as e:
                log_action("BACKEND", "Load PKL inside recognize_faces Failed", str(e))
        
        results = []
        for (top, right, bottom, left), face_encoding in zip(locations, encodings):
            name = "Unknown"
            if known_encodings:
                matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.6)
                face_distances = face_recognition.face_distance(known_encodings, face_encoding)
                if True in matches:
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = known_names[best_match_index]
            
            # Print Hello <name> as requested
            if name != "Unknown":
                print(f"Hello {name}")
                log_action("BACKEND", f"Hello {name}", f"Recognized face at box ({left}, {top}, {right}, {bottom})")
            else:
                log_action("BACKEND", "Unknown Face Detected", f"Box ({left}, {top}, {right}, {bottom})")
            
            box = (left, top, right - left, bottom - top)
            results.append({"box": box, "name": name})
            
        return results
    except Exception as e:
        log_action("BACKEND", "recognize_faces Error", str(e))
        return []

def has_face(img_input):
    return len(detect_faces(img_input)) > 0

def get_embedding(img_input):
    try:
        rgb = get_rgb_image(img_input)
        encodings = face_recognition.face_encodings(rgb)
        emb = encodings[0] if encodings else None
        log_action("BACKEND", "Embedding Generated", "Success" if emb is not None else "No face found")
        return emb
    except Exception as e:
        log_action("BACKEND", "Embedding Generation Error", str(e))
        return None

def save_face_data(name, images_bytes):
    log_action("BACKEND", "Save Face Data Started", f"Name: {name}, Images: {len(images_bytes)}")
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
        log_action("BACKEND", "Save Face Data Failed", "No valid embeddings found in any images.")
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
        except Exception as e:
            log_action("BACKEND", "Load PKL Failed (Creating new list)", str(e))
            
    faces = [f for f in faces if f.get("name", "").lower() != name.lower()]
    faces.append({"name": name, "encoding": avg_embedding})
    
    with open(PKL_PATH, "wb") as f:
        pickle.dump(faces if len(faces) > 1 else faces[0], f)
    log_action("BACKEND", "Save Face Data Success", f"Saved face for {name} with averaged embedding.")
    return True
