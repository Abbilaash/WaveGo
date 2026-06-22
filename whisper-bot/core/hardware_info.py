#!/usr/bin/python3

from __future__ import annotations

from pathlib import Path

import psutil


CPU_TEMP_PATH = Path("/sys/class/thermal/thermal_zone0/temp")


def get_cpu_tempfunc() -> str:
	"""Return CPU temperature as a string in Celsius."""
	try:
		result = CPU_TEMP_PATH.read_text().strip()
		return str(round(float(result) / 1000, 1))
	except Exception:
		return "unknown"


def get_cpu_use() -> str:
	"""Return CPU usage percentage as a string."""
	try:
		return str(round(psutil.cpu_percent(interval=None), 1))
	except Exception:
		return "unknown"


def get_ram_info() -> str:
	"""Return RAM usage percentage as a string."""
	try:
		return str(psutil.virtual_memory().percent)
	except Exception:
		return "unknown"
