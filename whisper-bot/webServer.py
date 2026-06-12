#!/usr/bin/env python3
"""Robot-side web server for WAVEGO."""

from __future__ import annotations

import os
import sys
import socket
import subprocess
import threading
import time
import re
from typing import Optional, Tuple

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, send_from_directory, request


THIS_DIR = os.path.dirname(os.path.realpath(__file__))
if THIS_DIR in sys.path:
	sys.path.remove(THIS_DIR)
sys.path.insert(0, THIS_DIR)

RPi_DIR = os.path.normpath(os.path.join(THIS_DIR, "..", "RPi"))
if RPi_DIR not in sys.path:
	sys.path.insert(1, RPi_DIR)
else:
	sys.path.remove(RPi_DIR)
	sys.path.insert(1, RPi_DIR)

import camera_opencv
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
	"""Start the fallback AP in a background process."""
	command = ["sudo", "create_ap", "wlan0", "eth0", "WAVE_BOT", "12345678"]
	subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


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
			return state

	ip_address, mode = ensure_network()
	with state_lock:
		device_state["ip"] = ip_address
		device_state["mode"] = mode
		state = dict(device_state)
		state["cpu_temp"] = hardware_info.get_cpu_tempfunc()
		state["cpu_use"] = hardware_info.get_cpu_use()
		state["ram_info"] = hardware_info.get_ram_info()
		return state


@app.route("/")
def index():
	state = get_state()
	return render_template("index.html", state=state)


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


def process_chatbot_text(command_text: str, client_ip: str) -> dict:
	global pending_chatbot_actions
	
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
	
	CONFIDENCE_THRESHOLD = 0.4
	
	# Scenario A: We have a pending action waiting for a numeric parameter
	if pending is not None:
		# FIRST priority: Check if the user's input contains a valid number parameter.
		# If a number is present, we process it directly and NEVER classify it (avoiding "2" matching STOP).
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
		
		# SECOND priority: If no number is found, run classification to see if the user
		# entered an entirely different command (e.g. STOP or FOLLOW_BLUE) to abort the prompt.
		try:
			best_intent, best_score = predict_fn(command_text)
		except Exception as exc:
			log_action("BACKEND", "Chatbot Classification Error", str(exc))
			return {
				"success": False,
				"error": f"Failed to classify intent: {str(exc)}"
			}
			
		if best_score >= CONFIDENCE_THRESHOLD and best_intent not in ("MOVE_FORWARD", "MOVE_BACKWARD", "TURN_LEFT", "TURN_RIGHT", pending["intent"]):
			# User explicitly wants to do another command; clear pending state and fall through to Scenario B
			pending_chatbot_actions.pop(client_ip, None)
			pending = None
		else:
			# Prompt again for a valid number since no number or alternative command was recognized
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

	# Scenario B: No pending action, process standard command
	try:
		best_intent, best_score = predict_fn(command_text)
	except Exception as exc:
		log_action("BACKEND", "Chatbot Classification Error", str(exc))
		return {
			"success": False,
			"error": f"Failed to classify intent: {str(exc)}"
		}

	action_msg = ""
	execution_success = True
	prompt_for_param = False
	
	if best_score >= CONFIDENCE_THRESHOLD:
		if best_intent in ("MOVE_FORWARD", "MOVE_BACKWARD", "TURN_LEFT", "TURN_RIGHT"):
			# Check if user already provided a number in this command (e.g., "turn left 90")
			num_val = parse_number(command_text)
			if num_val is not None and num_val > 0:
				# Execute immediately
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
				# No number provided, set pending state and prompt
				pending_chatbot_actions[client_ip] = {
					"intent": best_intent,
					"timestamp": time.time()
				}
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
				# Capture frame and attempt face recognition immediately
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
		else:
			execution_success = False
			action_msg = f"Intent '{best_intent}' recognized but no execution handler is mapped."
	else:
		execution_success = False
		action_msg = "Command not understood (low match confidence)."
		
	log_action("BACKEND", "Chatbot Command Executed", f"Command: '{command_text}', Best Intent: {best_intent}, Score: {best_score:.4f}, Action: {action_msg}")
	
	return {
		"success": True,
		"command": command_text,
		"intent": best_intent if best_score >= CONFIDENCE_THRESHOLD else None,
		"score": best_score,
		"threshold_passed": bool(best_score >= CONFIDENCE_THRESHOLD),
		"action_taken": action_msg,
		"execution_success": execution_success,
		"prompt_for_param": prompt_for_param
	}


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
	return jsonify(res)


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

	res = process_chatbot_text(transcribed_text, request.remote_addr)
	if not res.get("success", True):
		return jsonify(res), 500
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
		from FollowObject.detect import detect

		if action == 'start':
			camera_opencv.Camera.modeSelect = 'ballSearch'
			
			# Get the latest frame from camera (same way other APIs do it)
			camera_obj = get_camera()
			if camera_obj is None:
				return jsonify({"success": False, "error": "Camera unavailable"}), 503
			
			frame_bytes = camera_obj.get_frame()
			if not frame_bytes:
				return jsonify({"success": False, "error": "Could not capture frame"}), 500
			
			# Convert frame bytes to numpy array
			nparr = np.frombuffer(frame_bytes, np.uint8)
			frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
			
			if frame is None:
				return jsonify({"success": False, "error": "Could not decode frame"}), 500
			
			# Run detection using the detect() function
			model_path = os.path.join(THIS_DIR, 'FollowObject', 'best.onnx')
			detection_result = detect(frame, model_path)
			
			log_action("BACKEND", "Ball Search Detection", f"Found {len(detection_result.get('detections', []))} object(s)")
			
			return jsonify({
				"success": detection_result.get('success', False),
				"detections": detection_result.get('detections', [])
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
	frame_bytes = camera_obj.get_frame()
	if not frame_bytes:
		return jsonify({"success": False, "error": "Could not capture frame"}), 500
	import base64
	import face_detection
	has_face = face_detection.has_face(frame_bytes)
	log_action("BACKEND", "Face Capture Command Executed", f"Has human face: {has_face}")
	encoded_image = base64.b64encode(frame_bytes).decode("utf-8")
	return jsonify({
		"success": True,
		"has_face": has_face,
		"image": encoded_image
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
