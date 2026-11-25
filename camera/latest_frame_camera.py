"""Threaded camera capture with a single latest-frame slot (no frame backlog)."""
from __future__ import annotations

import threading
import time
from typing import Optional

import cv2


class LatestFrameCamera:
    def __init__(self, index: int = 0, width: int = 1280, height: int = 720,
                 fps: int = 30, backend: int = cv2.CAP_MSMF):
        self.index = index
        self.requested_width = width
        self.requested_height = height
        self.fps = fps
        self.backend = backend

        self._cap = cv2.VideoCapture(index, backend)
        if not self._cap.isOpened():
            self._cap.release()
            raise RuntimeError(
                f"Could not open camera {index} with backend {backend}. "
                "Check the camera index or try another backend."
            )

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._cap.set(cv2.CAP_PROP_FPS, fps)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._lock = threading.Lock()
        self._frame = None
        self._sequence = 0
        self._captured_at = 0.0
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self.read_errors = 0

    def start(self):
        if self._running:
            return self
        self._running = True
        self._thread = threading.Thread(
            target=self._reader, name="GestureSense-Camera", daemon=True
        )
        self._thread.start()
        return self

    def _reader(self):
        while self._running:
            ok, frame = self._cap.read()
            captured_at = time.monotonic()
            if not ok or frame is None:
                self.read_errors += 1
                time.sleep(0.005)
                continue
            with self._lock:
                self._frame = frame
                self._sequence += 1
                self._captured_at = captured_at

    def latest(self):
        """Return (frame_copy, sequence, captured_at_monotonic)."""
        with self._lock:
            if self._frame is None:
                return None, self._sequence, None
            return self._frame.copy(), self._sequence, self._captured_at

    def stop(self):
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._cap.release()
        self._thread = None

    close = stop
