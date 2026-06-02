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

db_faces_cache = None
db_cache_time = 0

def load_cached_faces():
    global db_faces_cache, db_cache_time
    if not os.path.exists(PKL_PATH):
        return [], []
        
    mtime = os.path.getmtime(PKL_PATH)
    if db_faces_cache is None or mtime > db_cache_time:
        try:
            with open(PKL_PATH, "rb") as f:
                data = pickle.load(f)
                if isinstance(data, list):
                    db_faces = data
                elif isinstance(data, dict):
                    db_faces = [data]
                else:
                    db_faces = []
                
                known_names = []
                known_encodings = []
                for face_info in db_faces:
                    known_names.append(face_info.get("name", "Unknown"))
                    known_encodings.append(face_info.get("encoding"))
                
                db_faces_cache = (known_names, known_encodings)
                db_cache_time = mtime
        except Exception as e:
            log_action("BACKEND", "Load PKL Failed", str(e))
            return [], []
            
    return db_faces_cache

def detect_faces(img_input):
    try:
        if isinstance(img_input, bytes):
            img_input = cv2.imdecode(np.frombuffer(img_input, np.uint8), cv2.IMREAD_COLOR)
            
        # Downscale for 16x speedup
        small_frame = cv2.resize(img_input, (0, 0), fx=0.25, fy=0.25)
        rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        locations = face_recognition.face_locations(rgb_small)
        faces = [(left * 4, top * 4, (right - left) * 4, (bottom - top) * 4) for top, right, bottom, left in locations]
        log_action("BACKEND", "Face Detection Run", f"Detected {len(faces)} faces.")
        return faces
    except Exception as e:
        log_action("BACKEND", "Face Detection Error", str(e))
        return []

def recognize_faces(img_input):
    try:
        if isinstance(img_input, bytes):
            img_input = cv2.imdecode(np.frombuffer(img_input, np.uint8), cv2.IMREAD_COLOR)
            
        # 1. Resize frame to 1/4 size for fast processing
        small_frame = cv2.resize(img_input, (0, 0), fx=0.25, fy=0.25)
        
        # Convert BGR to RGB
        rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        # 2. Detect face locations on small image
        locations = face_recognition.face_locations(rgb_small)
        if not locations:
            return []
            
        # 3. Calculate encodings on small image
        encodings = face_recognition.face_encodings(rgb_small, locations)
        
        # 4. Load cached database faces
        known_names, known_encodings = load_cached_faces()
        
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
            
            if name != "Unknown":
                print(f"Hello {name}")
                log_action("BACKEND", f"Hello {name}", f"Recognized face at box ({left*4}, {top*4}, {right*4}, {bottom*4})")
            else:
                log_action("BACKEND", "Unknown Face Detected", f"Box ({left*4}, {top*4}, {right*4}, {bottom*4})")
            
            # Scale coordinates back up by multiplying by 4
            box = (left * 4, top * 4, (right - left) * 4, (bottom - top) * 4)
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
        # Cleanup newly written files if we failed to save them as a face
        for i in range(len(images_bytes)):
            img_path = os.path.join(MODEL_DIR, f"{name}_{i+1}.jpg")
            if os.path.exists(img_path):
                try:
                    os.remove(img_path)
                except Exception:
                    pass
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
            
    # Filter out existing face with same name to avoid duplicates and preserve queue order
    faces = [f for f in faces if f.get("name", "").lower() != name.lower()]
    
    # If we already have 5 faces in the queue, we must remove the oldest one
    if len(faces) >= 5:
        oldest_face = faces[0]
        oldest_name = oldest_face.get("name", "")
        faces.pop(0)
        log_action("BACKEND", "Face Queue Limit Exceeded", f"Removing oldest face '{oldest_name}' from database.")
        
        if oldest_name:
            import glob
            oldest_images_pattern = os.path.join(MODEL_DIR, f"{oldest_name}_*.jpg")
            oldest_images = glob.glob(oldest_images_pattern)
            for img_file in oldest_images:
                try:
                    os.remove(img_file)
                    log_action("BACKEND", "Image Deleted", f"Deleted {os.path.basename(img_file)}")
                except Exception as e:
                    log_action("BACKEND", "Image Deletion Error", f"Failed to delete {os.path.basename(img_file)}: {str(e)}")
                    
    faces.append({"name": name, "encoding": avg_embedding})
    
    with open(PKL_PATH, "wb") as f:
        pickle.dump(faces if len(faces) > 1 else faces[0], f)
    log_action("BACKEND", "Save Face Data Success", f"Saved face for {name} with averaged embedding.")
    return True
