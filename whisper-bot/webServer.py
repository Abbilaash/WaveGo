#!/usr/bin/env python3
"""Robot-side web server for WAVEGO."""

from __future__ import annotations

import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import sys
# Load the local persistent Vosk package
THIS_DIR = os.path.dirname(os.path.realpath(__file__))
VOSK_PKG_DIR = os.path.join(THIS_DIR, "whisper", "vosk_package")
if os.path.exists(VOSK_PKG_DIR) and VOSK_PKG_DIR not in sys.path:
	sys.path.insert(0, VOSK_PKG_DIR)

import vosk  # Must be imported first to prevent OpenBLAS library conflicts
import socket
import subprocess
import threading
import time
import re
from typing import Optional, Tuple

# Monkeypatch importlib.metadata.version to prevent PackageNotFoundError for werkzeug/flask in environments with corrupt/missing package metadata
try:
	import importlib.metadata
	_orig_version = importlib.metadata.version
	def _hook_version(distribution_name):
		try:
			return _orig_version(distribution_name)
		except importlib.metadata.PackageNotFoundError:
			if distribution_name.lower() in ("werkzeug", "flask", "itsdangerous", "click", "jinja2", "markupsafe"):
				return "3.0.0"
			raise
	importlib.metadata.version = _hook_version
except Exception:
	pass


import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, send_from_directory, request


THIS_DIR = os.path.dirname(os.path.realpath(__file__))
if THIS_DIR in sys.path:
	sys.path.remove(THIS_DIR)
sys.path.insert(0, THIS_DIR)

# Clear whisper/__pycache__ on startup to prevent stale compiled file loads
try:
	import shutil
	pycache_path = os.path.join(THIS_DIR, "whisper", "__pycache__")
	if os.path.exists(pycache_path):
		shutil.rmtree(pycache_path)
except Exception:
	pass

CORE_DIR = os.path.join(THIS_DIR, "core")
if CORE_DIR not in sys.path:
	sys.path.insert(1, CORE_DIR)

RPi_DIR = os.path.normpath(os.path.join(THIS_DIR, "..", "RPi"))
if RPi_DIR not in sys.path:
	sys.path.insert(2, RPi_DIR)
else:
	sys.path.remove(RPi_DIR)
	sys.path.insert(2, RPi_DIR)

import camera_opencv
from FollowObject.detect import detect
# Optional import for Raspberry Pi camera
try:
    from picamera2 import Picamera2
except ImportError:
    Picamera2 = None  # Will be checked at runtime
import hardware_info
import camera_tilt
import robot
from logger import log_action

# Preload speech libraries at startup to prevent the Werkzeug developer reloader
# on Windows from detecting newly accessed files and restarting the server during requests.
try:
	import onnxruntime
	from tokenizers import Tokenizer
except Exception as e:
	log_action("BACKEND", "Preload Speech Libraries Error", str(e))


AP_DEFAULT_IP = "192.168.12.1"
NETWORK_CHECK_TARGET = ("1.1.1.1", 80)
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5000

app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.jinja_env.auto_reload = True


@app.before_request
def log_request_info():
	payload = request.get_json(silent=True) or request.form or request.args
	logged_payload = None
	if payload:
		logged_payload = dict(payload)
		if "images" in logged_payload:
			if isinstance(logged_payload["images"], list):
				logged_payload["images"] = [f"<base64_img_{idx+1}_data_length_{len(str(img))}>" for idx, img in enumerate(logged_payload["images"])]
			else:
				logged_payload["images"] = "<stripped_base64_data>"
	log_action("API_REQUEST", f"{request.method} {request.path}", f"IP: {request.remote_addr}, Payload: {logged_payload}")


@app.after_request
def log_response_info(response):
	log_action("API_RESPONSE", f"{request.method} {request.path}", f"Status: {response.status}")
	return response


state_lock = threading.Lock()
device_state = {
	"mode": "unknown",
	"ip": None,
	"ap_started": False,
}
camera = None
camera_error = None


def get_primary_ip() -> Optional[str]:
	"""Return the IP address used for the current outbound route."""
	probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
	try:
		probe.connect(NETWORK_CHECK_TARGET)
		return probe.getsockname()[0]
	except OSError:
		return None
	finally:
		probe.close()


def get_interface_ip(interface: str) -> Optional[str]:
	"""Return an IPv4 address for a local interface when available."""
	try:
		result = subprocess.run(
			["ip", "-4", "-o", "addr", "show", "dev", interface],
			check=False,
			capture_output=True,
			text=True,
		)
	except FileNotFoundError:
		return None

	for token in result.stdout.split():
		if token.count(".") == 3 and "/" in token:
			return token.split("/")[0]
	return None


def start_access_point() -> None:
	"""Start the fallback AP in a background process and configure static IP 192.168.4.1."""
	def run_ap_setup():
		try:
			# 1. Start the hotspot
			cmd_start = [
				"sudo", "nmcli", "device", "wifi", "hotspot",
				"ifname", "wlan0",
				"ssid", "WAVE_BOT",
				"password", "12345678"
			]
			subprocess.run(cmd_start, check=True)
			
			# 2. Modify connection profile to use 192.168.4.1
			cmd_modify_ip = [
				"sudo", "nmcli", "connection", "modify", "Hotspot",
				"ipv4.addresses", "192.168.4.1/24"
			]
			subprocess.run(cmd_modify_ip, check=True)
			
			cmd_modify_method = [
				"sudo", "nmcli", "connection", "modify", "Hotspot",
				"ipv4.method", "shared"
			]
			subprocess.run(cmd_modify_method, check=True)
			
			# 3. Bring the connection up to apply the changes
			cmd_up = ["sudo", "nmcli", "connection", "up", "Hotspot"]
			subprocess.run(cmd_up, check=True)
			print("[AccessPoint] Fallback AP started on static IP 192.168.4.1")
		except Exception as e:
			print(f"[AccessPoint] Error starting fallback AP: {e}")

	threading.Thread(target=run_ap_setup, daemon=True).start()


def get_camera():
	global camera, camera_error
	if camera is None and camera_error is None:
		try:
			camera = camera_opencv.Camera()
		except Exception as exc:
			camera_error = str(exc)
			print("Camera unavailable:", camera_error)
	return camera


def gen(frame_camera):
	"""Video streaming generator function."""
	while True:
		frame = frame_camera.get_frame()
		yield (
			b"--frame\r\n"
			b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
		)


def ensure_network() -> Tuple[str, str]:
	"""Return the active IP and network mode, starting AP mode if needed."""
	ip_address = get_primary_ip()
	if ip_address:
		return ip_address, "wifi"

	with state_lock:
		ap_started = device_state["ap_started"]
		if not ap_started:
			device_state["ap_started"] = True

	if not ap_started:
		start_access_point()

	deadline = time.time() + 30
	while time.time() < deadline:
		ip_address = get_interface_ip("wlan0")
		if ip_address:
			return ip_address, "ap"
		time.sleep(1)

	return AP_DEFAULT_IP, "ap"


def get_state() -> dict:
	with state_lock:
		if device_state["ip"] and device_state["mode"] != "unknown":
			state = dict(device_state)
			state["cpu_temp"] = hardware_info.get_cpu_tempfunc()
			state["cpu_use"] = hardware_info.get_cpu_use()
			state["ram_info"] = hardware_info.get_ram_info()
			try:
				import camera_opencv
				state["last_detected_face"] = camera_opencv.Camera.latest_face_name
			except Exception:
				state["last_detected_face"] = ""
			return state

	ip_address, mode = ensure_network()
	with state_lock:
		device_state["ip"] = ip_address
		device_state["mode"] = mode
		state = dict(device_state)
		state["cpu_temp"] = hardware_info.get_cpu_tempfunc()
		state["cpu_use"] = hardware_info.get_cpu_use()
		state["ram_info"] = hardware_info.get_ram_info()
		try:
			import camera_opencv
			state["last_detected_face"] = camera_opencv.Camera.latest_face_name
		except Exception:
			state["last_detected_face"] = ""
		return state


@app.route("/")
def index():
	state = get_state()
	return render_template("index.html", state=state)


@app.route("/lenet")
def lenet_page():
	return render_template("lenet.html")


@app.route("/video_feed")
def video_feed():
	camera_obj = get_camera()
	if camera_obj is None:
		return ("Camera unavailable: %s" % camera_error, 503)

	return Response(gen(camera_obj), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/status")
def api_status():
	return jsonify(get_state())


@app.route('/api/tilt/<direction>/<action>', methods=['POST'])
def api_tilt(direction, action):
	"""Control camera tilt. direction: up/down/left/right. action: start/stop."""
	direction = direction.lower()
	action = action.lower()
	if direction not in ("up", "down", "left", "right"):
		return jsonify({"success": False, "error": "invalid direction"}), 400
	if action not in ("start", "stop"):
		return jsonify({"success": False, "error": "invalid action"}), 400

	ok = False
	try:
		if action == 'start':
			ok = camera_tilt.start(direction)
		else:
			ok = camera_tilt.stop(direction)
		log_action("BACKEND", "Camera Tilt Command Executed", f"Direction: {direction}, Action: {action}, Success: {ok}")
	except Exception as exc:
		log_action("BACKEND", "Camera Tilt Command Error", str(exc))
		return jsonify({"success": False, "error": str(exc)}), 500

	if not ok:
		return jsonify({"success": False, "error": "robot unavailable"}), 503
	return jsonify({"success": True, "direction": direction, "action": action})


def stop_robot() -> None:
	"""Stop robot motion on both axes."""
	robot.stopLR()
	robot.stopFB()


@app.route('/api/default/<action>', methods=['POST'])
def api_default(action):
	"""Trigger preset robot behaviors: steady, jump, or handshake."""
	action = action.lower()
	if action not in ("steady", "jump", "handshake"):
		return jsonify({"success": False, "error": "invalid action"}), 400

	try:
		if action == 'steady':
			robot.steadyMode()
		elif action == 'jump':
			robot.jump()
		else:
			robot.handShake()
		log_action("BACKEND", "Default Action Command Executed", f"Action: {action}")
	except Exception as exc:
		log_action("BACKEND", "Default Action Command Error", str(exc))
		return jsonify({"success": False, "error": str(exc)}), 500

	return jsonify({"success": True, "action": action})


@app.route('/api/move/<action>', methods=['POST'])
def api_move(action):
	"""Control robot movement. action: forward/backward/left/right/stop. Optional speed param."""
	action = action.lower()
	speed = 100
	try:
		from flask import request as _flask_request
		if _flask_request.is_json and _flask_request.json and 'speed' in _flask_request.json:
			speed = int(_flask_request.json.get('speed', speed))
		else:
			speed = int(_flask_request.args.get('speed', speed))
	except Exception:
		speed = 100

	if action not in ("forward", "backward", "left", "right", "stop"):
		return jsonify({"success": False, "error": "invalid action"}), 400

	try:
		if action == 'forward':
			robot.forward(speed)
		elif action == 'backward':
			robot.backward(speed)
		elif action == 'left':
			robot.left(speed)
		elif action == 'right':
			robot.right(speed)
		else:
			stop_robot()
		log_action("BACKEND", "Movement Command Executed", f"Action: {action}, Speed: {speed}")
	except Exception as exc:
		log_action("BACKEND", "Movement Command Error", str(exc))
		return jsonify({"success": False, "error": str(exc)}), 500

	return jsonify({"success": True, "action": action, "speed": speed})


chatbot_predict = None
chatbot_lock = threading.Lock()

def get_chatbot_predict():
	global chatbot_predict
	with chatbot_lock:
		if chatbot_predict is None:
			import MiniLM
			chatbot_predict = MiniLM.predict_intent
	return chatbot_predict


pending_chatbot_actions = {}
PENDING_TIMEOUT = 30.0

# Timed movement calibrations (in seconds)
STEP_DURATION = 0.35      # 1 step = 0.35 seconds
DEGREE_DURATION = 0.0116    # (1 degree = 0.0116 seconds)

def parse_number(text: str) -> Optional[float]:
	# Try to find standard digits/floats
	matches = re.findall(r"[-+]?\d*\.\d+|\d+", text)
	if matches:
		return float(matches[0])
		
	# Try word equivalents
	word_to_num = {
		"one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
		"six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10
	}
	words = text.lower().split()
	for w in words:
		if w in word_to_num:
			return float(word_to_num[w])
	return None


import os
import numpy as np
import pickle
import re
from MiniLM.knowledge_inference import get_embedding as kb_get_embedding

# Load knowledge base once (global)
_KB_PATH = os.path.join(os.path.dirname(__file__), "knowledge", "knowledge_db.pkl")
try:
    with open(_KB_PATH, "rb") as f:
        _kb_data = pickle.load(f)
        _kb_chunks = _kb_data.get("chunks", [])
        _kb_embeddings = np.array(_kb_data.get("embeddings", []), dtype=np.float32) if _kb_data.get("embeddings") else np.empty((0, 0), dtype=np.float32)
except Exception:
    _kb_chunks = []
    _kb_embeddings = np.empty((0, 0), dtype=np.float32)

# --------------------------------
# Gemma3 Lazy Initialization
# --------------------------------
_gemma_session = None
_gemma_tokenizer = None
_gemma_lock = threading.Lock()
_gemma_load_error = None

def get_gemma3_session_and_tokenizer():
    global _gemma_session, _gemma_tokenizer, _gemma_load_error
    with _gemma_lock:
        if _gemma_session is None and _gemma_load_error is None:
            try:
                import onnxruntime as ort
                from tokenizers import Tokenizer
                
                script_dir = os.path.dirname(os.path.abspath(__file__))
                model_path = os.path.join(script_dir, "knowledge", "gemma3.onnx")
                tokenizer_path = os.path.join(script_dir, "knowledge", "tokenizer.json")
                
                if not os.path.exists(tokenizer_path):
                    raise FileNotFoundError(f"Gemma3 tokenizer.json not found at '{tokenizer_path}'.")
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"Gemma3 model not found at '{model_path}'.")
                
                _gemma_tokenizer = Tokenizer.from_file(tokenizer_path)
                _gemma_session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
            except Exception as e:
                _gemma_load_error = str(e)
                log_action("BACKEND", "Gemma3 Load Error", _gemma_load_error)
                
    if _gemma_load_error:
        raise RuntimeError(f"Gemma3 initialization failed: {_gemma_load_error}")
    return _gemma_session, _gemma_tokenizer

def _map_sentence_to_intent(sentence: str) -> str | None:
    """Simple mapping from a stored sentence to a movement intent."""
    s = sentence.lower()
    if "forward" in s:
        return "MOVE_FORWARD"
    if "backward" in s:
        return "MOVE_BACKWARD"
    if "left" in s:
        return "TURN_LEFT"
    if "right" in s:
        return "TURN_RIGHT"
    if "stop" in s:
        return "STOP"
    return None

def process_chatbot_text(command_text: str, client_ip: str) -> dict:
    global pending_chatbot_actions, _kb_chunks, _kb_embeddings
    
    try:
        predict_fn = get_chatbot_predict()
    except Exception as exc:
        log_action("BACKEND", "Chatbot Initialization Error", str(exc))
        return {
            "success": False,
            "error": f"Failed to initialize chatbot model: {str(exc)}"
        }
        
    # Clean up expired pending states
    now = time.time()
    expired_ips = [ip for ip, data in pending_chatbot_actions.items() if now - data.get("timestamp", 0) > PENDING_TIMEOUT]
    for ip in expired_ips:
        pending_chatbot_actions.pop(ip, None)
        
    pending = pending_chatbot_actions.get(client_ip)
    
    CONFIDENCE_THRESHOLD = 0.6

    # 1. Training command pattern: train on <full_sentence> (does not need "command" prefix)
    train_match = re.match(r'^train on (.+)$', command_text, re.IGNORECASE)
    if train_match:
        full_sentence = train_match.group(1).strip()
        try:
            new_emb = kb_get_embedding(full_sentence)
            try:
                with open(_KB_PATH, "rb") as f:
                    db = pickle.load(f)
            except FileNotFoundError:
                db = {"chunks": [], "embeddings": []}
            
            db_chunks = db.get("chunks", [])
            db_embeddings = np.array(db.get("embeddings", []), dtype=np.float32) if len(db.get("embeddings", [])) > 0 else np.empty((0, 0), dtype=np.float32)
            
            if db_embeddings.size == 0:
                db_embeddings = np.expand_dims(new_emb, axis=0)
            else:
                db_embeddings = np.vstack([db_embeddings, new_emb])
            
            if not isinstance(db_chunks, list):
                db_chunks = []
            db_chunks.append(full_sentence)
            
            db["chunks"] = db_chunks
            db["embeddings"] = db_embeddings.tolist()
            with open(_KB_PATH, "wb") as f:
                pickle.dump(db, f)
                
            _kb_chunks = db_chunks
            _kb_embeddings = db_embeddings
            
            log_action("BACKEND", "Chatbot Trained", f"Sentence: '{full_sentence}'")
            return {
                "success": True,
                "command": command_text,
                "intent": None,
                "score": 1.0,
                "action_taken": f"Trained on: '{full_sentence}'",
                "execution_success": True,
                "prompt_for_param": False,
                "threshold_passed": True
            }
        except Exception as exc:
            log_action("BACKEND", "Chatbot Training Error", str(exc))
            return {
                "success": False,
                "error": f"Failed to train on statement: {str(exc)}"
            }

    # 2. Scenario A: pending numeric param handling
    if pending is not None:
        num_val = parse_number(command_text)
        if num_val is not None and num_val > 0:
            pending_intent = pending["intent"]
            pending_chatbot_actions.pop(client_ip, None) # Clear pending state
            action_msg = ""
            if pending_intent in ("MOVE_FORWARD", "MOVE_BACKWARD"):
                duration = num_val * STEP_DURATION
                action_msg = f"Moving {pending_intent.split('_')[1].lower()} for {num_val} steps ({duration:.2f}s)."
                def run_move():
                    try:
                        if pending_intent == "MOVE_FORWARD":
                            robot.forward(100)
                        else:
                            robot.backward(100)
                        time.sleep(duration)
                        robot.stopFB()
                    except Exception as e:
                        log_action("BACKEND", "Timed Move Thread Error", str(e))
                threading.Thread(target=run_move, daemon=True).start()
            elif pending_intent in ("TURN_LEFT", "TURN_RIGHT"):
                duration = num_val * DEGREE_DURATION
                action_msg = f"Turning {pending_intent.split('_')[1].lower()} by {num_val} degrees ({duration:.2f}s)."
                def run_turn():
                    try:
                        if pending_intent == "TURN_LEFT":
                            robot.left(100)
                        else:
                            robot.right(100)
                        time.sleep(duration)
                        robot.stopLR()
                    except Exception as e:
                        log_action("BACKEND", "Timed Turn Thread Error", str(e))
                threading.Thread(target=run_turn, daemon=True).start()
                
            log_action("BACKEND", "Chatbot Param Command Executed", f"Intent: {pending_intent}, Param: {num_val}, Action: {action_msg}")
            return {
                "success": True,
                "command": command_text,
                "intent": pending_intent,
                "score": 1.0,
                "action_taken": action_msg,
                "execution_success": True,
                "prompt_for_param": False,
                "threshold_passed": True
            }
        
        # If no number, check if it starts with "command" to abort, otherwise clear pending and query KB.
        command_prefix_match = re.match(r'^command\b[:\s]*(.*)$', command_text, re.IGNORECASE)
        if command_prefix_match:
            clean_command = command_prefix_match.group(1).strip()
            try:
                best_intent, best_score = predict_fn(clean_command)
            except Exception as exc:
                log_action("BACKEND", "Chatbot Classification Error", str(exc))
                best_intent, best_score = None, 0.0
                
            if best_score >= CONFIDENCE_THRESHOLD and best_intent not in ("MOVE_FORWARD", "MOVE_BACKWARD", "TURN_LEFT", "TURN_RIGHT", pending["intent"]):
                # User wants to run another command: clear pending state and fall through to standard processing
                pending_chatbot_actions.pop(client_ip, None)
                pending = None
                # Replace command_text with clean_command for command execution below
                command_text = clean_command
            else:
                prompt_msg = ""
                if pending["intent"] in ("MOVE_FORWARD", "MOVE_BACKWARD"):
                    prompt_msg = "Please specify the number of steps to walk as a number (e.g. 3 or 5)."
                else:
                    prompt_msg = "Please specify the angle to turn in degrees as a number (e.g. 45 or 90)."
                return {
                    "success": True,
                    "command": command_text,
                    "intent": pending["intent"],
                    "score": best_score,
                    "action_taken": prompt_msg,
                    "execution_success": False,
                    "prompt_for_param": True,
                    "threshold_passed": True
                }
        else:
            # Not a number and didn't start with "command": abort pending state and fall back to Q&A
            pending_chatbot_actions.pop(client_ip, None)
            pending = None

    # Check if this input starts with "command" (explicit robot control request)
    command_prefix_match = re.match(r'^command\b[:\s]*(.*)$', command_text, re.IGNORECASE)
    
    if command_prefix_match:
        # Strip "command" prefix and process as standard command
        clean_command = command_prefix_match.group(1).strip()
        
        # Scenario B: No pending action, process standard command
        # First: Check KB for high similarity match of a command
        kb_matched_intent = None
        kb_best_score = 0.0
        kb_matched_sentence = None
        
        if _kb_embeddings.size > 0:
            try:
                kb_emb = kb_get_embedding(clean_command)
                scores = _kb_embeddings @ kb_emb
                best_idx = int(np.argmax(scores))
                kb_best_score = float(scores[best_idx])
                if kb_best_score > 0.65:
                    kb_matched_sentence = _kb_chunks[best_idx]
                    kb_matched_intent = _map_sentence_to_intent(kb_matched_sentence)
            except Exception as e:
                log_action("BACKEND", "KB search error", str(e))
                
        # Second: Run normal classification to compare with KB match
        try:
            classifier_intent, classifier_score = predict_fn(clean_command)
        except Exception as exc:
            log_action("BACKEND", "Chatbot Classification Error", str(exc))
            classifier_intent, classifier_score = None, 0.0

        # Decide on the best intent source (KB vs Classifier)
        best_intent = None
        best_score = 0.0
        
        if kb_matched_intent is not None:
            best_intent = kb_matched_intent
            best_score = kb_best_score
        elif classifier_score >= CONFIDENCE_THRESHOLD:
            best_intent = classifier_intent
            best_score = classifier_score

        # Execute command if detected
        if best_intent is not None:
            action_msg = ""
            execution_success = True
            prompt_for_param = False
            
            if best_intent in ("MOVE_FORWARD", "MOVE_BACKWARD", "TURN_LEFT", "TURN_RIGHT"):
                num_val = parse_number(clean_command)
                if num_val is not None and num_val > 0:
                    if best_intent in ("MOVE_FORWARD", "MOVE_BACKWARD"):
                        duration = num_val * STEP_DURATION
                        action_msg = f"Moving {best_intent.split('_')[1].lower()} for {num_val} steps ({duration:.2f}s)."
                        def run_move():
                            try:
                                if best_intent == "MOVE_FORWARD":
                                    robot.forward(100)
                                else:
                                    robot.backward(100)
                                time.sleep(duration)
                                robot.stopFB()
                            except Exception as e:
                                log_action("BACKEND", "Timed Move Thread Error", str(e))
                        threading.Thread(target=run_move, daemon=True).start()
                    else:
                        duration = num_val * DEGREE_DURATION
                        action_msg = f"Turning {best_intent.split('_')[1].lower()} by {num_val} degrees ({duration:.2f}s)."
                        def run_turn():
                            try:
                                if best_intent == "TURN_LEFT":
                                    robot.left(100)
                                else:
                                    robot.right(100)
                                time.sleep(duration)
                                robot.stopLR()
                            except Exception as e:
                                log_action("BACKEND", "Timed Turn Thread Error", str(e))
                        threading.Thread(target=run_turn, daemon=True).start()
                else:
                    pending_chatbot_actions[client_ip] = {"intent": best_intent, "timestamp": time.time()}
                    prompt_for_param = True
                    if best_intent in ("MOVE_FORWARD", "MOVE_BACKWARD"):
                        action_msg = "How many steps would you like to walk?"
                    else:
                        action_msg = "What angle would you like to turn (in degrees)?"
                    execution_success = False
                    
            elif best_intent == "DETECT_FACE":
                camera_opencv.Camera.modeSelect = 'faceDetection'
                camera_obj = get_camera()
                detected_name = None
                if camera_obj is not None:
                    frame_bytes = camera_obj.get_frame()
                    if frame_bytes:
                        import face_detection
                        faces = face_detection.recognize_faces(frame_bytes)
                        if faces:
                            known_faces = [f for f in faces if f.get("name", "Unknown") != "Unknown"]
                            if known_faces:
                                detected_name = known_faces[0]["name"]
                            else:
                                detected_name = faces[0]["name"]
                
                if detected_name and detected_name != "Unknown":
                    action_msg = f"Hello {detected_name}!"
                elif detected_name == "Unknown":
                    action_msg = "Face detection started. I see someone, but I don't recognize them."
                else:
                    action_msg = "Face detection started. I don't see any faces in front of me."
                    
            elif best_intent == "FACE_DETECT_OFF":
                camera_opencv.Camera.modeSelect = 'none'
                robot.lightCtrl('blue', 0)
                action_msg = "Face detection turned off."

            elif best_intent == "FOLLOW_FACE":
                # Look for "follow face <name>" or "follow <name>"
                name_match = re.search(r'(?:follow\s+face|follow|track\s+face|track)\s+(\w+)', clean_command, re.IGNORECASE)
                target_name = ""
                if name_match:
                    possible_name = name_match.group(1).strip()
                    # Ignore common stop words/pronouns
                    if possible_name.lower() not in ("me", "my", "face", "person", "the", "him", "her", "us"):
                        target_name = possible_name
                
                # Check latest detected face name as a fallback
                if not target_name:
                    target_name = camera_opencv.Camera.latest_face_name
                
                if target_name:
                    camera_opencv.Camera.followName = target_name
                    camera_opencv.Camera.modeSelect = 'faceFollowing'
                    action_msg = f"Started face following for: {target_name}."
                else:
                    action_msg = "Please specify who to follow. For example, say: 'follow face John'."
                    execution_success = False

            elif best_intent == "OBJECT_DETECT_START":
                camera_opencv.Camera.modeSelect = 'objectDetection'
                action_msg = "Object detection mode started."

            elif best_intent == "OBJECT_DETECT_STOP":
                camera_opencv.Camera.modeSelect = 'none'
                robot.buzzerCtrl(0, 0)
                robot.lightCtrl('blue', 0)
                action_msg = "Object detection mode stopped."

            elif best_intent == "STOP":
                global active_follow_color
                active_follow_color = None
                camera_opencv.Camera.modeSelect = 'none'
                camera_opencv.Camera.followColor = 'none'
                stop_robot()
                action_msg = "Stopped all robot motion and openCV modes."
                
            elif best_intent == "SIT":
                robot.steadyMode()
                action_msg = "Robot sitting down (stabilized steady mode)."
                
            elif best_intent == "STAND":
                robot.steadyMode()
                action_msg = "Robot standing up (stabilized steady mode)."
                
            elif best_intent == "JUMP":
                try:
                    robot.jump()
                    action_msg = "Performing jump action."
                except Exception as e:
                    action_msg = f"Failed to perform jump: {e}"
                    execution_success = False

            elif best_intent == "HANDSHAKE":
                try:
                    robot.handShake()
                    action_msg = "Performing handshake action."
                except Exception as e:
                    action_msg = f"Failed to perform handshake: {e}"
                    execution_success = False

            elif best_intent == "FOLLOW_RED":
                active_follow_color = "red"
                camera_opencv.Camera.modeSelect = 'followColor'
                camera_opencv.Camera.followColor = "red"
                action_msg = "Started color tracking mode following the 'red' ball."
                
            elif best_intent == "FOLLOW_GREEN":
                active_follow_color = "green"
                camera_opencv.Camera.modeSelect = 'followColor'
                camera_opencv.Camera.followColor = "green"
                action_msg = "Started color tracking mode following the 'green' ball."
                
            elif best_intent == "FOLLOW_BLUE":
                active_follow_color = "blue"
                camera_opencv.Camera.modeSelect = 'followColor'
                camera_opencv.Camera.followColor = "blue"
                action_msg = "Started color tracking mode following the 'blue' ball."
                
            elif best_intent == "FOLLOW_COLOR_STOP":
                active_follow_color = None
                camera_opencv.Camera.modeSelect = 'none'
                camera_opencv.Camera.followColor = 'none'
                stop_robot()
                action_msg = "Stopped target color tracking."

            elif best_intent == "BALL_SEARCH_START":
                camera_opencv.Camera.modeSelect = 'ballSearch'
                action_msg = "Ball search mode started. Rotating to search for the green ball."

            elif best_intent == "BALL_SEARCH_STOP":
                camera_opencv.Camera.modeSelect = 'none'
                stop_robot()
                action_msg = "Ball search mode stopped."

            elif best_intent == "DETECT_DIGIT":
                action_msg = "To detect a digit, please draw it on the handwriting canvas tab inside the mobile app."
                execution_success = False

            elif best_intent == "STATUS":
                state = get_state()
                cpu_temp = state.get("cpu_temp", "unknown")
                cpu_use = state.get("cpu_use", "unknown")
                action_msg = f"My hardware status is: CPU temperature is {cpu_temp} degrees Celsius, and CPU utilization is {cpu_use} percent."

            elif best_intent == "DIAGNOSTIC":
                camera_obj = get_camera()
                cam_status = "working" if camera_obj is not None else "unavailable"
                action_msg = f"Running system self-check. Camera status is {cam_status}. All local AI models are loaded and ready."

            elif best_intent in ("TILT_UP", "TILT_DOWN", "TILT_LEFT", "TILT_RIGHT"):
                direction = best_intent.split("_")[1].lower() # "up", "down", "left", "right"
                try:
                    camera_tilt.start(direction)
                    action_msg = f"Tilting camera {direction}."
                except Exception as e:
                    action_msg = f"Failed to tilt camera: {e}"
                    execution_success = False

            elif best_intent == "TILT_STOP":
                try:
                    camera_tilt.stop("up")
                    camera_tilt.stop("down")
                    camera_tilt.stop("left")
                    camera_tilt.stop("right")
                    action_msg = "Camera tilt stopped."
                except Exception as e:
                    action_msg = f"Failed to stop camera tilt: {e}"
                    execution_success = False
                
            else:
                execution_success = False
                action_msg = f"Intent '{best_intent}' recognized but no execution handler is mapped."
                
            log_action("BACKEND", "Chatbot Command Executed", f"Command: '{clean_command}', Intent: {best_intent}, Score: {best_score:.4f}, Action: {action_msg}")
            return {
                "success": True,
                "command": command_text,
                "intent": best_intent,
                "score": best_score,
                "threshold_passed": True,
                "action_taken": action_msg,
                "execution_success": execution_success,
                "prompt_for_param": prompt_for_param
            }
        else:
            return {
                "success": True,
                "command": command_text,
                "intent": None,
                "score": 0.0,
                "threshold_passed": False,
                "action_taken": "Command not understood (low match confidence). Make sure to say e.g., 'command forward 5' or 'command stop'.",
                "execution_success": False,
                "prompt_for_param": False
            }

    # 4. Fallback to normal chat system (Knowledge Base closest match + Gemma3 LLM generator)
    if _kb_embeddings.size > 0:
        try:
            kb_emb = kb_get_embedding(command_text)
            scores = _kb_embeddings @ kb_emb
            
            # Retrieve top 3 context chunks
            top_k = min(3, len(_kb_chunks))
            top_indices = np.argsort(scores)[::-1][:top_k]
            retrieved_chunks = [_kb_chunks[idx] for idx in top_indices]
            best_score = float(scores[top_indices[0]])
            context = "\n\n".join(retrieved_chunks)
            
            try:
                # Lazy load Gemma3 ONNX session and tokenizer
                session, tokenizer = get_gemma3_session_and_tokenizer()
                from knowledge.inference import generate_response
                
                # Construct the prompt
                prompt = f"<start_of_turn>user\nContext:\n{context}\n\nQuestion:\n{command_text}\n\nInstructions: Answer the question concisely in 1 sentence.<end_of_turn>\n<start_of_turn>model\n"
                
                log_action("BACKEND", "Chatbot Gemma3 Generation Started", f"Prompt length: {len(prompt)}")
                response_text = generate_response(
                    session=session,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    max_new_tokens=100,
                    temperature=0.0,
                    print_stream=False
                )
                log_action("BACKEND", "Chatbot Gemma3 Response", response_text)
                
                return {
                    "success": True,
                    "command": command_text,
                    "intent": None,
                    "score": best_score,
                    "threshold_passed": False,
                    "action_taken": response_text,
                    "execution_success": True,
                    "prompt_for_param": False
                }
            except Exception as exc:
                log_action("BACKEND", "Gemma3 Generation Error/Fallback", str(exc))
                # Fallback to returning just the closest matching sentence (old behavior)
                best_idx = top_indices[0]
                matched_sentence = _kb_chunks[best_idx]
                
                fallback_msg = matched_sentence
                if "model.onnx_data" in str(exc) or "external data" in str(exc).lower():
                    fallback_msg = f"[Fallback: Gemma3 weights missing. Place 'model.onnx_data' in knowledge/]\nMatched Context: {matched_sentence}"
                else:
                    fallback_msg = f"[Fallback: Gemma3 error - {str(exc)}]\nMatched Context: {matched_sentence}"
                    
                return {
                    "success": True,
                    "command": command_text,
                    "intent": None,
                    "score": best_score,
                    "threshold_passed": False,
                    "action_taken": fallback_msg,
                    "execution_success": True,
                    "prompt_for_param": False
                }
        except Exception as e:
            log_action("BACKEND", "Chatbot KB Fallback error", str(e))
            
    # Default fallback
    return {
        "success": True,
        "command": command_text,
        "intent": None,
        "score": 0.0,
        "threshold_passed": False,
        "action_taken": "I'm sorry, I don't understand that command. Please try again or teach me using 'train on <statement>'.",
        "execution_success": False,
        "prompt_for_param": False
    }


def speak_text_async(text: str):
	import threading
	import re
	import sys

	# Clean text to remove any markup, brackets, or system indicators
	clean_text = re.sub(r'\[.*?\]', '', text).strip()
	clean_text = re.sub(r'<.*?>', '', clean_text).strip()
	if not clean_text:
		return

	def run_tts():
		# Check if a Bluetooth device is connected (or assume Windows mock is connected)
		connected_macs = set()
		if sys.platform.startswith("linux"):
			try:
				import subprocess
				res = subprocess.run(["hcitool", "con"], capture_output=True, text=True, check=False)
				mac_pattern = re.compile(r"((?:[0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2})")
				for mac in mac_pattern.findall(res.stdout):
					connected_macs.add(mac)
			except Exception:
				pass
		else:
			connected_macs.add("MOCK_DEV_MAC")

		if len(connected_macs) == 0:
			return

		try:
			import pyttsx3
			engine = pyttsx3.init()
			engine.setProperty('rate', 150)
			engine.say(clean_text)
			engine.runAndWait()
			del engine
		except Exception as e:
			print(f"pyttsx3 failed: {e}. Falling back to espeak...")
			if sys.platform.startswith("linux"):
				try:
					import subprocess
					subprocess.run(["espeak", "-v", "en-us", "-s", "150", clean_text], capture_output=True)
				except Exception as es:
					print(f"espeak fallback failed: {es}")

	threading.Thread(target=run_tts, daemon=True).start()


@app.route('/api/chatbot/command', methods=['POST'])
def api_chatbot_command():
	data = request.json
	if not data or "command" not in data:
		return jsonify({"success": False, "error": "Missing command string"}), 400
		
	command_text = data["command"].strip()
	if not command_text:
		return jsonify({"success": False, "error": "Command string cannot be empty"}), 400
		
	res = process_chatbot_text(command_text, request.remote_addr)
	if not res.get("success", True):
		return jsonify(res), 500
		
	if "action_taken" in res:
		speak_text_async(res["action_taken"])
		
	return jsonify(res)


def convert_words_to_numbers(text: str) -> str:
    units = {
        "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
        "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
        "seventeen": 17, "eighteen": 18, "nineteen": 19
    }
    tens = {
        "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90
    }
    scales = {
        "hundred": 100, "thousand": 1000, "million": 1000000
    }
    
    words = text.split()
    new_words = []
    i = 0
    while i < len(words):
        num_block = []
        while i < len(words):
            word_clean = words[i].lower().strip(".,?!;:")
            if word_clean in units or word_clean in tens or word_clean in scales or word_clean == "and":
                if word_clean == "and" and not num_block:
                    break
                num_block.append(words[i])
                i += 1
            else:
                break
        
        if num_block:
            if len(num_block) == 1 and num_block[0].lower().strip(".,?!;:") == "and":
                new_words.append(num_block[0])
                continue
                
            last_word = num_block[-1]
            punctuation = ""
            for char in reversed(last_word):
                if char in ".,?!;:":
                    punctuation = char + punctuation
                else:
                    break
            
            clean_block = []
            for w in num_block:
                clean_w = w.lower().strip(".,?!;:")
                if clean_w:
                    clean_block.append(clean_w)
            
            try:
                val = 0
                current = 0
                for w in clean_block:
                    if w in units:
                        current += units[w]
                    elif w in tens:
                        current += tens[w]
                    elif w in scales:
                        scale = scales[w]
                        if current == 0:
                            current = 1
                        if scale == 100:
                            current *= 100
                        else:
                            val += current * scale
                            current = 0
                    elif w == "and":
                        continue
                val += current
                new_words.append(str(val) + punctuation)
            except Exception:
                new_words.extend(num_block)
        else:
            new_words.append(words[i])
            i += 1
            
    return " ".join(new_words)

_audio_transcriber = None

def get_audio_transcriber():
	global _audio_transcriber
	if _audio_transcriber is not None:
		return _audio_transcriber
	try:
		model_dir = os.path.join(THIS_DIR, "whisper")
		vosk_model_path = os.path.join(model_dir, "vosk-model-small-en-us-0.15")
		if os.path.exists(vosk_model_path):
			from whisper.AudioToText import AudioToTextTranscriber
			_audio_transcriber = AudioToTextTranscriber(model_dir)
			return _audio_transcriber
	except Exception as exc:
		log_action("BACKEND", "Audio Transcriber Init Error", str(exc))
	return None


@app.route('/api/chatbot/audio', methods=['POST'])
def api_chatbot_audio():
	if 'audio' not in request.files:
		return jsonify({"success": False, "error": "No audio file provided"}), 400
		
	audio_file = request.files['audio']
	temp_path = os.path.join(THIS_DIR, f"temp_voice_{int(time.time())}.wav")
	
	try:
		audio_file.save(temp_path)
	except Exception as exc:
		log_action("BACKEND", "Audio Upload Save Error", str(exc))
		return jsonify({"success": False, "error": f"Failed to save uploaded audio file: {str(exc)}"}), 500

	transcriber = get_audio_transcriber()
	if transcriber is None:
		if os.path.exists(temp_path):
			os.remove(temp_path)
		return jsonify({
			"success": False,
			"error": "Local audio transcriber is not initialized. Please ensure the Vosk model (vosk-model-small-en-us-0.15) is downloaded and extracted in the 'whisper' folder."
		}), 400

	try:
		transcribed_text = transcriber.transcribe(temp_path)
		log_action("BACKEND", "Speech-to-Text Success", f"Transcribed: '{transcribed_text}'")
	except Exception as exc:
		log_action("BACKEND", "Speech-to-Text Processing Error", str(exc))
		return jsonify({"success": False, "error": f"Failed to transcribe audio: {str(exc)}"}), 500
	finally:
		if os.path.exists(temp_path):
			os.remove(temp_path)

	if not transcribed_text:
		return jsonify({"success": False, "error": "Speech was not recognized or transcription is empty. Please speak clearly."}), 400

	# Convert spoken numbers to digit representations
	normalized_text = convert_words_to_numbers(transcribed_text)
	log_action("BACKEND", "Speech-to-Text Normalized", f"Original: '{transcribed_text}', Normalized: '{normalized_text}'")

	res = process_chatbot_text(normalized_text, request.remote_addr)
	if not res.get("success", True):
		return jsonify(res), 500
		
	if "action_taken" in res:
		speak_text_async(res["action_taken"])
		
	return jsonify(res)


@app.route("/api/img/<path:filename>")
def sendimg(filename):
	return send_from_directory(os.path.join(THIS_DIR, "dist", "img"), filename)


@app.route("/js/<path:filename>")
def sendjs(filename):
	return send_from_directory(os.path.join(THIS_DIR, "dist", "js"), filename)


@app.route("/css/<path:filename>")
def sendcss(filename):
	return send_from_directory(os.path.join(THIS_DIR, "dist", "css"), filename)


@app.route("/api/img/icon/<path:filename>")
def sendicon(filename):
	return send_from_directory(os.path.join(THIS_DIR, "dist", "img", "icon"), filename)


@app.route("/fonts/<path:filename>")
def sendfonts(filename):
	return send_from_directory(os.path.join(THIS_DIR, "dist", "fonts"), filename)


@app.route('/api/face/detect/<action>', methods=['POST'])
def api_face_detect(action):
	action = action.lower()
	if action not in ('start', 'stop'):
		return jsonify({"success": False, "error": "invalid action"}), 400
	try:
		import camera_opencv
		import robot
		if action == 'start':
			camera_opencv.Camera.modeSelect = 'faceDetection'
		else:
			camera_opencv.Camera.modeSelect = 'none'
			robot.buzzerCtrl(0, 0)
			robot.lightCtrl('blue', 0)
		log_action("BACKEND", "Face Detection Toggle Command Executed", f"Action: {action}, Mode: {camera_opencv.Camera.modeSelect}")
	except Exception as exc:
		log_action("BACKEND", "Face Detection Toggle Error", str(exc))
		return jsonify({"success": False, "error": str(exc)}), 500
	return jsonify({"success": True, "mode": camera_opencv.Camera.modeSelect})


@app.route('/api/search/ball', methods=['POST'])
def api_search_ball():
	data = request.get_json()
	if not data or 'action' not in data:
		return jsonify({"success": False, "error": "Missing action parameter"}), 400

	action = data.get('action', '').lower()
	try:
		import camera_opencv

		if action == 'start':
			# Enable ball search mode for UI feedback
			camera_opencv.Camera.modeSelect = 'ballSearch'
			# Capture the latest frame using the existing camera helper
			camera_obj = get_camera()
			if camera_obj is None:
				log_action("BACKEND", "Ball Search Error", "Camera unavailable")
				return jsonify({"success": False, "error": "Camera unavailable"}), 503
			frame_bytes = camera_obj.get_frame()
			if not frame_bytes:
				log_action("BACKEND", "Ball Search Error", "Could not capture frame")
				return jsonify({"success": False, "error": "Could not capture frame"}), 500
			# Decode JPEG bytes to a BGR NumPy array
			arr = np.frombuffer(frame_bytes, np.uint8)
			frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
			if frame is None:
				log_action("BACKEND", "Ball Search Error", "Could not decode frame")
				return jsonify({"success": False, "error": "Could not decode frame"}), 500
			# Run detection using the detect function
			from FollowObject.detect import detect
			result = detect(frame)
			log_action("BACKEND", "Ball Search Detection", f"Found {len(result.get('detections', []))} object(s)")
			return jsonify({
				"success": result.get('success', False),
				"detections": result.get('detections', []),
				"message": result.get('message', '')
			})

		elif action == 'stop':
			camera_opencv.Camera.modeSelect = 'none'
			log_action("BACKEND", "Ball Search Stopped", "Ball search mode disabled")
			return jsonify({"success": True, "message": "Ball search stopped"})

		else:
			return jsonify({"success": False, "error": "invalid action"}), 400

	except Exception as exc:
		log_action("BACKEND", "Ball Search Error", str(exc))
		return jsonify({"success": False, "error": str(exc)}), 500


# ----------------------------------------------------------
# Raspberry Pi specific ball search endpoint using Picamera2
# ----------------------------------------------------------
@app.route('/api/pi/ballsearch', methods=['POST'])
def api_pi_ballsearch():
    data = request.get_json()
    if not data or 'action' not in data:
        return jsonify({"success": False, "error": "Missing action parameter"}), 400

    action = data.get('action', '').lower()
    if Picamera2 is None:
        log_action("BACKEND", "Pi Ball Search Error", "Picamera2 module not available")
        return jsonify({"success": False, "error": "Picamera2 not available on this system"}), 500

    try:
        if action == 'start':
            picam = Picamera2()
            config = picam.create_video_configuration(main={"format": "RGB888", "size": (640, 480)})
            picam.configure(config)
            picam.start()
            time.sleep(1)  # allow camera to settle
            frame = picam.capture_array()
            picam.stop()
            if frame is None:
                log_action("BACKEND", "Pi Ball Search Error", "Failed to capture frame from Picamera2")
                return jsonify({"success": False, "error": "Failed to capture frame"}), 500
            result = detect(frame, input_is_rgb=True)
            log_action("BACKEND", "Pi Ball Search Detection", f"Found {len(result.get('detections', []))} object(s)")
            return jsonify({
                "success": result.get('success', False),
                "detections": result.get('detections', []),
                "message": result.get('message', '')
            })
        elif action == 'stop':
            log_action("BACKEND", "Pi Ball Search Stopped", "Ball search stop requested")
            return jsonify({"success": True, "message": "Pi ball search stopped"})
        else:
            return jsonify({"success": False, "error": "invalid action"}), 400
    except Exception as exc:
        log_action("BACKEND", "Pi Ball Search Error", str(exc))
        return jsonify({"success": False, "error": str(exc)}), 500

@app.route('/api/camera/stream/info', methods=['GET'])
def api_camera_stream_info():
	try:
		import picamera2
		return jsonify({"success": True, "available": True})
	except ImportError:
		return jsonify({"success": True, "available": False})


@app.route('/api/detect_digit', methods=['POST'])
def api_detect_digit():
	try:
		# Check JSON payload for base64 image
		data = request.get_json(silent=True) or {}
		img_b64 = data.get("image")
		explain = data.get("explain", False) or request.args.get("explain", "false").lower() == "true"
		
		if not img_b64:
			return jsonify({"success": False, "error": "Missing canvas drawing image payload (base64)"}), 400

		import base64
		if "," in img_b64:
			img_b64 = img_b64.split(",")[1]
		img_bytes = base64.b64decode(img_b64)
		arr = np.frombuffer(img_bytes, np.uint8)
		frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
		if frame is None:
			return jsonify({"success": False, "error": "Could not decode drawing image"}), 400
		
		# Run digit detection using LeNet5 MNIST CNN model
		if explain:
			from lenet5.detect import detect_explain
			res = detect_explain(frame)
		else:
			from lenet5.detect import detect as detect_digit
			res = detect_digit(frame)
		
		if not res.get("success", False):
			log_action("BACKEND", "Digit Detection Failed", res.get("error", "Unknown error"))
			return jsonify(res), 400
			
		log_action("BACKEND", "Digit Detected", f"Prediction: {res['prediction']}, Conf: {res['confidence']:.2f}%")
		
		chatbot_msg = f"CNN Digit Detection: I detected the digit '{res['prediction']}' with a confidence of {res['confidence']:.2f}%."
		
		response_data = {
			"success": True,
			"prediction": res["prediction"],
			"confidence": res["confidence"],
			"message": chatbot_msg
		}
		if explain and "explanation" in res:
			response_data["explanation"] = res["explanation"]
			
		speak_text_async(chatbot_msg)
		return jsonify(response_data)
	except Exception as exc:
		log_action("BACKEND", "Digit Detection Error", str(exc))
		return jsonify({"success": False, "error": str(exc)}), 500


def is_linux() -> bool:
	return sys.platform.startswith("linux")

def get_connected_macs() -> set[str]:
	import subprocess
	import re
	connected = set()
	if not is_linux():
		return connected
	try:
		res = subprocess.run(["hcitool", "con"], capture_output=True, text=True, check=False)
		mac_pattern = re.compile(r"((?:[0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2})")
		for mac in mac_pattern.findall(res.stdout):
			connected.add(mac.upper())
	except Exception:
		pass
	return connected

def is_device_connected(mac: str, connected_set: set[str]) -> bool:
	if mac.upper() in connected_set:
		return True
	if not is_linux():
		return False
	try:
		res = subprocess.run(["bluetoothctl", "info", mac], capture_output=True, text=True, check=False)
		return "Connected: yes" in res.stdout
	except Exception:
		return False

def scan_bluetooth_devices_helper() -> list[dict]:
	if not is_linux():
		return [
			{"mac": "00:11:22:33:44:55", "name": "JBL Flip 5 (Mock Speaker)", "connected": False},
			{"mac": "AA:BB:CC:DD:EE:FF", "name": "Sony WH-1000XM4 (Mock Headphones)", "connected": True},
			{"mac": "12:34:56:78:90:AB", "name": "Bose SoundLink (Mock Speaker)", "connected": False}
		]

	import subprocess
	import re

	subprocess.run(["bluetoothctl", "power", "on"], capture_output=True, text=True, check=False)
	try:
		subprocess.run(["timeout", "8", "bluetoothctl", "scan", "on"], capture_output=True, text=True, check=False)
	except Exception:
		pass

	devices = []
	seen_macs = set()
	connected_set = get_connected_macs()

	try:
		res = subprocess.run(["bluetoothctl", "devices"], capture_output=True, text=True, check=False)
		mac_pattern = re.compile(r"Device\s+((?:[0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2})\s+(.*)")
		for line in res.stdout.splitlines():
			match = mac_pattern.search(line)
			if match:
				mac = match.group(1)
				name = match.group(2).strip()
				if mac not in seen_macs:
					seen_macs.add(mac)
					devices.append({
						"mac": mac,
						"name": name,
						"connected": is_device_connected(mac, connected_set)
					})
	except Exception as e:
		log_action("BACKEND", "Bluetooth Scan Error", str(e))

	return devices

def connect_bluetooth_device_helper(mac: str) -> tuple[bool, str]:
	if not is_linux():
		time.sleep(1.5)
		return True, f"Connected successfully (Mock) to {mac}"

	import subprocess
	subprocess.run(["bluetoothctl", "power", "on"], capture_output=True, text=True, check=False)
	subprocess.run(["bluetoothctl", "agent", "on"], capture_output=True, text=True, check=False)
	subprocess.run(["bluetoothctl", "default-agent"], capture_output=True, text=True, check=False)

	subprocess.run(["bluetoothctl", "trust", mac], capture_output=True, text=True, check=False)
	subprocess.run(["bluetoothctl", "pair", mac], capture_output=True, text=True, check=False)
	res = subprocess.run(["bluetoothctl", "connect", mac], capture_output=True, text=True, check=False)

	stdout = res.stdout or ""
	stderr = res.stderr or ""
	combined = stdout + "\n" + stderr
	if "Connection successful" in combined or "successful" in combined.lower() or res.returncode == 0:
		return True, combined
	return False, combined

def disconnect_bluetooth_device_helper(mac: str) -> tuple[bool, str]:
	if not is_linux():
		time.sleep(0.5)
		return True, f"Disconnected successfully (Mock) from {mac}"

	import subprocess
	res = subprocess.run(["bluetoothctl", "disconnect", mac], capture_output=True, text=True, check=False)
	stdout = res.stdout or ""
	if res.returncode == 0 or "successful" in stdout.lower() or "disconnect" in stdout.lower():
		return True, stdout
	return False, stdout


@app.route('/api/bluetooth/scan', methods=['GET'])
def api_bluetooth_scan():
	try:
		devices = scan_bluetooth_devices_helper()
		return jsonify({"success": True, "devices": devices})
	except Exception as e:
		log_action("BACKEND", "API Bluetooth Scan Error", str(e))
		return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/bluetooth/connect', methods=['POST'])
def api_bluetooth_connect():
	data = request.json
	if not data or "mac" not in data:
		return jsonify({"success": False, "error": "Missing mac address"}), 400
	mac = data["mac"].strip()
	if not mac:
		return jsonify({"success": False, "error": "MAC address cannot be empty"}), 400
	try:
		success, output = connect_bluetooth_device_helper(mac)
		if success:
			log_action("BACKEND", "Bluetooth Connect Success", f"Connected to {mac}")
			return jsonify({"success": True, "message": f"Connected to {mac}", "details": output})
		else:
			log_action("BACKEND", "Bluetooth Connect Failed", f"Failed connecting to {mac}: {output}")
			return jsonify({"success": False, "error": "Connection failed", "details": output}), 500
	except Exception as e:
		log_action("BACKEND", "API Bluetooth Connect Error", str(e))
		return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/bluetooth/disconnect', methods=['POST'])
def api_bluetooth_disconnect():
	data = request.json
	if not data or "mac" not in data:
		return jsonify({"success": False, "error": "Missing mac address"}), 400
	mac = data["mac"].strip()
	if not mac:
		return jsonify({"success": False, "error": "MAC address cannot be empty"}), 400
	try:
		success, output = disconnect_bluetooth_device_helper(mac)
		if success:
			log_action("BACKEND", "Bluetooth Disconnect Success", f"Disconnected from {mac}")
			return jsonify({"success": True, "message": f"Disconnected from {mac}", "details": output})
		else:
			log_action("BACKEND", "Bluetooth Disconnect Failed", f"Failed disconnecting from {mac}: {output}")
			return jsonify({"success": False, "error": "Disconnection failed", "details": output}), 500
	except Exception as e:
		log_action("BACKEND", "API Bluetooth Disconnect Error", str(e))
		return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/object/detect/<action>', methods=['POST'])
def api_object_detect(action):
	action = action.lower()
	if action not in ('start', 'stop'):
		return jsonify({"success": False, "error": "invalid action"}), 400
	try:
		import camera_opencv
		import robot
		if action == 'start':
			camera_opencv.Camera.modeSelect = 'objectDetection'
		else:
			camera_opencv.Camera.modeSelect = 'none'
			robot.buzzerCtrl(0, 0)
			robot.lightCtrl('blue', 0)
		log_action("BACKEND", "Object Detection Toggle Command Executed", f"Action: {action}, Mode: {camera_opencv.Camera.modeSelect}")
	except Exception as exc:
		log_action("BACKEND", "Object Detection Toggle Error", str(exc))
		return jsonify({"success": False, "error": str(exc)}), 500
	return jsonify({"success": True, "mode": camera_opencv.Camera.modeSelect})


@app.route('/api/face/follow/<action>', methods=['POST'])
def api_face_follow(action):
	action = action.lower()
	if action not in ('start', 'stop'):
		return jsonify({"success": False, "error": "invalid action"}), 400
	try:
		import camera_opencv
		import robot
		if action == 'start':
			data = request.json
			if not data or "name" not in data:
				return jsonify({"success": False, "error": "Missing name"}), 400
			name = data["name"].strip()
			if not name:
				return jsonify({"success": False, "error": "Name cannot be empty"}), 400
			camera_opencv.Camera.followName = name
			camera_opencv.Camera.modeSelect = 'faceFollowing'
			log_action("BACKEND", "Face Follow Start", f"Started face following for: {name}")
			return jsonify({"success": True, "action": action, "name": name})
		else:
			camera_opencv.Camera.modeSelect = 'none'
			camera_opencv.Camera.followName = ''
			robot.buzzerCtrl(0, 0)
			robot.lightCtrl('blue', 0)
			robot.lookStopUD()
			robot.lookStopLR()
			log_action("BACKEND", "Face Follow Stop", "Stopped face following")
			return jsonify({"success": True, "action": action})
	except Exception as exc:
		log_action("BACKEND", "Face Follow Toggle Error", str(exc))
		return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/api/face/capture", methods=["POST", "GET"])
def api_face_capture():
	camera_obj = get_camera()
	if camera_obj is None:
		return jsonify({"success": False, "error": "Camera unavailable"}), 503
		
	import base64
	import face_detection
	import time
	
	images = []
	any_face = False
	
	for i in range(6):
		frame_bytes = camera_obj.get_frame()
		if frame_bytes:
			encoded_image = base64.b64encode(frame_bytes).decode("utf-8")
			images.append(encoded_image)
			if face_detection.has_face(frame_bytes):
				any_face = True
		if i < 5:  # sleep 0.5s between captures
			time.sleep(0.5)
			
	if not images:
		return jsonify({"success": False, "error": "Could not capture any frames"}), 500
		
	log_action("BACKEND", "Face Capture Command Executed", f"Captured 6 frames. Any human face: {any_face}")
	return jsonify({
		"success": True,
		"has_face": any_face,
		"images": images
	})


@app.route("/api/face/save", methods=["POST"])
def api_face_save():
	import base64
	import face_detection
	data = request.json
	if not data or "name" not in data or "images" not in data:
		return jsonify({"success": False, "error": "Missing name or images"}), 400
	name = data["name"].strip()
	if not name:
		return jsonify({"success": False, "error": "Name cannot be empty"}), 400
	images_base64 = data["images"]
	if not isinstance(images_base64, list) or len(images_base64) == 0:
		return jsonify({"success": False, "error": "Invalid images list"}), 400
	images_bytes = []
	for img_b64 in images_base64:
		try:
			if "," in img_b64:
				img_b64 = img_b64.split(",")[1]
			images_bytes.append(base64.b64decode(img_b64))
		except Exception:
			return jsonify({"success": False, "error": "Failed to decode base64 image"}), 400
	success = face_detection.save_face_data(name, images_bytes)
	if not success:
		return jsonify({"success": False, "error": "No faces could be encoded from the captured images"}), 400
	return jsonify({"success": True, "message": f"Successfully learned face for {name}"})


@app.route("/api/object/capture", methods=["POST", "GET"])
def api_object_capture():
	camera_obj = get_camera()
	if camera_obj is None:
		return jsonify({"success": False, "error": "Camera unavailable"}), 503
	frame_bytes = camera_obj.get_frame()
	if not frame_bytes:
		return jsonify({"success": False, "error": "Could not capture frame"}), 500
	import base64
	encoded_image = base64.b64encode(frame_bytes).decode("utf-8")
	return jsonify({
		"success": True,
		"image": encoded_image
	})


mobilenet_session = None
mobilenet_input_name = None

def get_crop_embedding(crop):
	global mobilenet_session, mobilenet_input_name
	if mobilenet_session is None:
		import onnxruntime as ort
		model_path = os.path.normpath(os.path.join(THIS_DIR, "mobilenetv3_embedding.onnx"))
		if not os.path.exists(model_path):
			raise FileNotFoundError(f"mobilenetv3_embedding.onnx not found at {model_path}")
		mobilenet_session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
		mobilenet_input_name = mobilenet_session.get_inputs()[0].name

	import cv2
	import numpy as np

	# Resize to 224x224
	resized = cv2.resize(crop, (224, 224))
	# Convert BGR to RGB
	rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
	# Normalize to [0, 1]
	normalized = rgb.astype(np.float32) / 255.0
	# ImageNet mean & std
	mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
	std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
	normalized = (normalized - mean) / std
	# HWC to CHW
	chw = np.transpose(normalized, (2, 0, 1))
	# Add batch dim NCHW
	nchw = np.expand_dims(chw, axis=0)

	# Inference
	outputs = mobilenet_session.run(None, {mobilenet_input_name: nchw})
	embedding = outputs[0][0]

	# L2 Normalize
	norm = np.linalg.norm(embedding)
	if norm > 1e-10:
		embedding = embedding / norm
	return embedding


@app.route("/api/object/submit", methods=["POST"])
def api_object_submit():
	data = request.json
	if not data or "images" not in data or "boxes" not in data:
		return jsonify({"success": False, "error": "Missing images or boxes"}), 400
	
	images = data["images"]
	boxes = data["boxes"]
	object_name = data.get("name", "unnamed_object").strip().replace(" ", "_")
	
	# Sanitize the object name to prevent directory traversal or invalid characters
	object_name = "".join(c for c in object_name if c.isalnum() or c in ("-", "_"))
	if not object_name:
		object_name = "unnamed_object"
	
	if not isinstance(images, list) or not isinstance(boxes, list) or len(images) != 3 or len(boxes) != 3:
		return jsonify({"success": False, "error": "Invalid images or boxes length"}), 400
	
	import base64
	import cv2
	import numpy as np
	import pickle
	import time
	import glob
	import shutil
	
	object_learning_dir = os.path.join(THIS_DIR, "ObjectLearning", object_name)
	os.makedirs(object_learning_dir, exist_ok=True)
	
	object_db_dir = os.path.join(THIS_DIR, "object_db")
	os.makedirs(object_db_dir, exist_ok=True)
	
	new_embeddings = []
	errors = []
	
	for i, (img_b64, box) in enumerate(zip(images, boxes)):
		if not img_b64:
			continue
		try:
			if "," in img_b64:
				img_b64 = img_b64.split(",")[1]
			img_bytes = base64.b64decode(img_b64)
			
			# Decode image with OpenCV
			nparr = np.frombuffer(img_bytes, np.uint8)
			img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
			if img is None:
				err_msg = f"Failed to decode image {i+1}"
				log_action("BACKEND", "Object Crop Error", err_msg)
				errors.append(err_msg)
				continue
			
			img_path = None
			# Crop if a bounding box was drawn
			if box and isinstance(box, dict) and all(k in box for k in ("x", "y", "w", "h")):
				x = int(box["x"])
				y = int(box["y"])
				w = int(box["w"])
				h = int(box["h"])
				
				# Ensure coordinates are within image boundaries
				img_h, img_w = img.shape[:2]
				x = max(0, min(x, img_w - 1))
				y = max(0, min(y, img_h - 1))
				w = max(1, min(w, img_w - x))
				h = max(1, min(h, img_h - y))
				
				# Crop
				cropped_img = img[y : y + h, x : x + w]
				
				# Save cropped image to ObjectLearning/crop_<timestamp>_<index>.jpg
				img_filename = f"crop_{int(time.time())}_{i+1}.jpg"
				img_path = os.path.join(object_learning_dir, img_filename)
				cv2.imwrite(img_path, cropped_img)
				log_action("BACKEND", "Object Crop Saved", f"Saved cropped image {i+1}")
			else:
				# If no bounding box was drawn, save the full image inside ObjectLearning/
				img_filename = f"crop_full_{int(time.time())}_{i+1}.jpg"
				img_path = os.path.join(object_learning_dir, img_filename)
				cv2.imwrite(img_path, img)
				cropped_img = img
				log_action("BACKEND", "Object Full Saved", f"No box drawn. Saved full image {i+1}")
				
			# Generate MobileNetV3 embedding for the crop
			if cropped_img is not None:
				try:
					emb = get_crop_embedding(cropped_img)
					new_embeddings.append(emb)
					log_action("BACKEND", "Object Embedding Success", f"Generated embedding for image {i+1}")
				except Exception as emb_err:
					err_msg = f"Failed to generate embedding for image {i+1}: {str(emb_err)}"
					log_action("BACKEND", "Object Embedding Error", err_msg)
					errors.append(err_msg)
		except Exception as e:
			err_msg = f"Failed to crop/save image {i+1}: {str(e)}"
			log_action("BACKEND", "Object Crop/Save Error", err_msg)
			errors.append(err_msg)
			
	# If we have successfully generated any embeddings, save to object_db/<object_name>.pkl
	if new_embeddings:
		pkl_path = os.path.join(object_db_dir, f"{object_name}.pkl")
		
		# Save new object data
		try:
			with open(pkl_path, "wb") as f:
				pickle.dump({
					"name": object_name,
					"embeddings": new_embeddings
				}, f)
			log_action("BACKEND", "Object PKL Save Success", f"Successfully saved object '{object_name}' with {len(new_embeddings)} embeddings to {pkl_path}")
		except Exception as pkl_save_err:
			log_action("BACKEND", "Object PKL Save Error", f"Failed to write individual pkl: {str(pkl_save_err)}")
			return jsonify({"success": False, "error": f"Failed to save pkl: {str(pkl_save_err)}"}), 500

		# Enforce the maximum limit of 10 objects inside object_db
		pkl_files = glob.glob(os.path.join(object_db_dir, "*.pkl"))
		if len(pkl_files) > 10:
			# Sort by modification time to find the oldest
			pkl_files.sort(key=os.path.getmtime)
			oldest_pkl = pkl_files[0]
			oldest_name = os.path.splitext(os.path.basename(oldest_pkl))[0]
			
			try:
				os.remove(oldest_pkl)
				log_action("BACKEND", "Object PKL Evicted", f"Evicted oldest pkl file: {oldest_pkl}")
			except Exception as del_err:
				log_action("BACKEND", "Object PKL Evict Error", f"Failed to delete {oldest_pkl}: {str(del_err)}")
				
			# Also clean up the corresponding crop folders in ObjectLearning
			oldest_dir = os.path.join(THIS_DIR, "ObjectLearning", oldest_name)
			if os.path.exists(oldest_dir):
				try:
					shutil.rmtree(oldest_dir)
					log_action("BACKEND", "Object Folder Deleted", f"Deleted folder for evicted object '{oldest_name}'")
				except Exception as clean_err:
					log_action("BACKEND", "Object Folder Delete Error", f"Failed to delete {oldest_name} directory: {str(clean_err)}")
	else:
		err_summary = ", ".join(errors) if errors else "No images or boxes were provided."
		log_action("BACKEND", "Object PKL Save Error", f"No embeddings were generated: {err_summary}")
		return jsonify({"success": False, "error": f"No embeddings were generated. Details: {err_summary}"}), 400
					
	return jsonify({"success": True, "message": "bounding success"})


active_follow_color = None

@app.route("/api/color/follow/<color>", methods=["POST"])
def api_color_follow(color):
	global active_follow_color
	color_lower = color.lower()
	if color_lower not in ("green", "blue", "red", "none"):
		return jsonify({"success": False, "error": "Invalid color option"}), 400
	
	try:
		if color_lower == "none":
			active_follow_color = None
			camera_opencv.Camera.modeSelect = 'none'
			camera_opencv.Camera.followColor = 'none'
			robot.stopFB()
			robot.stopLR()
			status = "deactivated"
		else:
			active_follow_color = color_lower
			camera_opencv.Camera.modeSelect = 'followColor'
			camera_opencv.Camera.followColor = color_lower
			status = "activated"
			
		log_action("BACKEND", "Follow Color", f"Color: {color_lower}, Status: {status}")
		return jsonify({
			"success": True,
			"color": color_lower,
			"status": status,
			"active_color": active_follow_color
		})
	except Exception as exc:
		log_action("BACKEND", "Follow Color Toggle Error", str(exc))
		return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/api/diagnostic")
def api_diagnostic():
	try:
		camera_obj = get_camera()
		if camera_obj is None:
			return jsonify({"success": False, "error": "Camera unavailable"})
		
		frame_bytes = camera_obj.get_frame()
		if not frame_bytes:
			return jsonify({"success": False, "error": "No frame bytes"})
		
		nparr = np.frombuffer(frame_bytes, np.uint8)
		decoded_frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
		
		latest_bgr = camera_opencv.Camera.latest_bgr_frame
		
		model_path = os.path.join(THIS_DIR, 'FollowObject', 'best4.onnx')
		
		res_decoded_f = detect(decoded_frame, model_path, input_is_rgb=False)
		res_decoded_t = detect(decoded_frame, model_path, input_is_rgb=True)
		
		res_latest_f = detect(latest_bgr, model_path, input_is_rgb=False) if latest_bgr is not None else None
		res_latest_t = detect(latest_bgr, model_path, input_is_rgb=True) if latest_bgr is not None else None
		
		return jsonify({
			"success": True,
			"latest_bgr_shape": latest_bgr.shape if latest_bgr is not None else None,
			"decoded_shape": decoded_frame.shape,
			"res_decoded_f_detections": res_decoded_f.get("detections"),
			"res_decoded_t_detections": res_decoded_t.get("detections"),
			"res_latest_f_detections": res_latest_f.get("detections") if res_latest_f else None,
			"res_latest_t_detections": res_latest_t.get("detections") if res_latest_t else None,
		})
	except Exception as e:
		return jsonify({"success": False, "error": str(e)})


def main() -> None:

	if not os.environ.get("WERKZEUG_RUN_MAIN"):
		# Execute the pycache cleanup commands requested by user
		if os.name != "nt":
			try:
				print("Cleaning pycache directories...")
				subprocess.run("find /home/rpi/WaveGo -type d -name \"__pycache__\" -exec rm -rf {} +", shell=True, check=False)
				subprocess.run("find ~/.local/lib/python3.13 -type d -name \"__pycache__\" -exec rm -rf {} +", shell=True, check=False)
				print("Cleaned pycache successfully.")
			except Exception as e:
				print("Failed to clean pycache:", e)

	if os.environ.get("WERKZEUG_RUN_MAIN") == "true" or not app.debug:
		state = get_state()
		print("network mode:", state["mode"])
		print("advertising ip:", state["ip"])

	# Trigger a non-blocking startup beep
	def startup_beep():
		time.sleep(1.5)
		try:
			robot.buzzerCtrl(1, 0)
			time.sleep(0.15)
			robot.buzzerCtrl(0, 0)
			print("Startup beep triggered.")
		except Exception as e:
			print("Failed to beep buzzer on startup:", e)

	threading.Thread(target=startup_beep, daemon=True).start()

	app.run(
		host=FLASK_HOST,
		port=FLASK_PORT,
		threaded=True,
		debug=False,
		use_reloader=False,
	)


if __name__ == "__main__":
	try:
		main()
	except KeyboardInterrupt:
		pass
