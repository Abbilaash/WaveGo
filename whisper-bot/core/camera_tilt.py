#!/usr/bin/env python3
"""Camera tilt wrappers that call RPi.robot tilt functions safely."""
from __future__ import annotations

import os
import sys
import threading
from typing import Literal

THIS_DIR = os.path.dirname(os.path.realpath(__file__))
RPi_DIR = os.path.normpath(os.path.join(THIS_DIR, "..", "..", "RPi"))
if RPi_DIR not in sys.path:
    sys.path.insert(0, RPi_DIR)

try:
    import robot as robot_mod
except Exception:  # pragma: no cover - best effort import
    robot_mod = None

_tilt_lock = threading.Lock()

Direction = Literal["up", "down", "left", "right"]


def _has_robot() -> bool:
    return robot_mod is not None


def start(direction: Direction) -> bool:
    """Start tilting in a direction. Returns True on success."""
    if not _has_robot():
        return False
    with _tilt_lock:
        if direction == "up":
            robot_mod.lookUp()
        elif direction == "down":
            robot_mod.lookDown()
        elif direction == "left":
            robot_mod.lookLeft()
        elif direction == "right":
            robot_mod.lookRight()
        else:
            return False
    return True


def stop(direction: Direction) -> bool:
    """Stop tilt motion for the given axis."""
    if not _has_robot():
        return False
    with _tilt_lock:
        if direction in ("up", "down"):
            robot_mod.lookStopUD()
        elif direction in ("left", "right"):
            robot_mod.lookStopLR()
        else:
            return False
    return True


if __name__ == "__main__":
    print("camera_tilt: module for wrapping RPi.robot tilt functions")