"""Live camera face detection for WAVEGO."""

from __future__ import annotations

import threading
from typing import Optional

import cv2 as cv
import mediapipe as mp
import numpy as np


class FaceDetectionCamera:
    """Capture frames from Picamera2 and annotate detected faces."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._error: Optional[str] = None
        self._picam2 = None
        self._face_detector = None

        try:
            from picamera2 import Picamera2

            self._picam2 = Picamera2()
            preview_config = self._picam2.create_preview_configuration(main={"format": "RGB888", "size": (640, 480)})
            self._picam2.configure(preview_config)
            self._picam2.start()
            self._face_detector = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
        except Exception as exc:
            self._error = str(exc)

    @property
    def error(self) -> Optional[str]:
        return self._error

    def get_frame(self) -> bytes:
        if self._error:
            raise RuntimeError(self._error)

        if self._picam2 is None or self._face_detector is None:
            raise RuntimeError("camera is not initialized")

        with self._lock:
            frame = self._picam2.capture_array()
            if frame is None:
                raise RuntimeError("camera feed is not available")

            results = self._face_detector.process(frame)
            annotated = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
            frame_height, frame_width, _ = annotated.shape

            if results.detections:
                for face in results.detections:
                    bbox = face.location_data.relative_bounding_box
                    x1 = int(bbox.xmin * frame_width)
                    y1 = int(bbox.ymin * frame_height)
                    x2 = int((bbox.xmin + bbox.width) * frame_width)
                    y2 = int((bbox.ymin + bbox.height) * frame_height)

                    cv.rectangle(annotated, (x1, y1), (x2, y2), (255, 255, 255), 2)

                    key_points = np.array([(p.x, p.y) for p in face.location_data.relative_keypoints], dtype=np.float32)
                    if key_points.size:
                        key_points_coords = np.multiply(key_points, [frame_width, frame_height]).astype(int)
                        for px, py in key_points_coords:
                            cv.circle(annotated, (px, py), 4, (255, 255, 255), 2)
                            cv.circle(annotated, (px, py), 2, (0, 0, 0), -1)

            ok, buffer = cv.imencode(".jpg", annotated)
            if not ok:
                raise RuntimeError("failed to encode camera frame")
            return buffer.tobytes()

    def close(self) -> None:
        with self._lock:
            if self._face_detector is not None:
                self._face_detector.close()
                self._face_detector = None
            if self._picam2 is not None:
                self._picam2.stop()
                self._picam2.close()
                self._picam2 = None
