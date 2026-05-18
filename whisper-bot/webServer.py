#!/usr/bin/env python3
"""Robot-side web server for WAVEGO."""

from __future__ import annotations

import os
import sys
import socket
import subprocess
import threading
import time
from typing import Optional, Tuple

from flask import Flask, Response, jsonify, render_template, send_from_directory


THIS_DIR = os.path.dirname(os.path.realpath(__file__))
RPi_DIR = os.path.normpath(os.path.join(THIS_DIR, "..", "RPi"))
if RPi_DIR not in sys.path:
	sys.path.insert(0, RPi_DIR)

import camera_opencv
import hardware_info
import camera_tilt
import robot


AP_DEFAULT_IP = "192.168.12.1"
NETWORK_CHECK_TARGET = ("1.1.1.1", 80)
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5000

app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.jinja_env.auto_reload = True
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
	except Exception as exc:
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
	except Exception as exc:
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
	except Exception as exc:
		return jsonify({"success": False, "error": str(exc)}), 500

	return jsonify({"success": True, "action": action, "speed": speed})


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


def main() -> None:
	if os.environ.get("WERKZEUG_RUN_MAIN") == "true" or not app.debug:
		state = get_state()
		print("network mode:", state["mode"])
		print("advertising ip:", state["ip"])

	app.run(
		host=FLASK_HOST,
		port=FLASK_PORT,
		threaded=True,
		debug=True,
		use_reloader=True,
	)


if __name__ == "__main__":
	try:
		main()
	except KeyboardInterrupt:
		pass
