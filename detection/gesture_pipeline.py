"""Independent hand-detection/classification worker.

The detector is created, used, and closed inside this worker thread because
MediaPipe VIDEO-mode instances should not be called concurrently.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

import cv2

from detection.hand_detector import HandDetector
from detection.gesture_classifier import GestureClassifier


class GesturePipeline:
    def __init__(
        self,
        camera,
        model_path=None,
        processing_width=640,
        max_fps=30.0,
        detection_confidence=0.6,
        presence_confidence=0.6,
        tracking_confidence=0.6,
        mirror=False,
    ):
        self.camera = camera
        self.model_path = model_path
        self.processing_width = int(processing_width)
        self.max_fps = float(max_fps)
        self.detection_confidence = detection_confidence
        self.presence_confidence = presence_confidence
        self.tracking_confidence = tracking_confidence
        self.mirror = mirror

        self._lock = threading.Lock()
        self._latest = None
        self._running = False
        self._thread = None
        self.error = None
        self.processed_frames = 0
        self.last_processing_ms = 0.0

    def start(self):
        if self._running:
            return self
        self._running = True
        self._thread = threading.Thread(
            target=self._run, name="GestureSense-GesturePipeline", daemon=True
        )
        self._thread.start()
        return self

    def _run(self):
        detector = None
        try:
            detector = HandDetector(
                model_path=self.model_path,
                max_hands=2,
                detection_confidence=self.detection_confidence,
                presence_confidence=self.presence_confidence,
                tracking_confidence=self.tracking_confidence,
            )
            classifier = GestureClassifier()
            last_sequence = -1
            frame_interval = 1.0 / self.max_fps if self.max_fps > 0 else 0.0
            next_allowed = 0.0

            while self._running:
                frame, sequence, captured_at = self.camera.latest()
                if frame is None or sequence == last_sequence:
                    time.sleep(0.001)
                    continue
                now = time.monotonic()
                if now < next_allowed:
                    time.sleep(min(0.002, next_allowed - now))
                    continue

                last_sequence = sequence
                next_allowed = time.monotonic() + frame_interval
                if self.mirror:
                    frame = cv2.flip(frame, 1)

                started = time.perf_counter()
                if self.processing_width > 0 and frame.shape[1] > self.processing_width:
                    scale = self.processing_width / frame.shape[1]
                    target_height = max(1, round(frame.shape[0] * scale))
                    detection_frame = cv2.resize(
                        frame, (self.processing_width, target_height),
                        interpolation=cv2.INTER_AREA,
                    )
                else:
                    detection_frame = frame

                detection = detector.detect(detection_frame)
                classified = classifier.classify(detection)
                if detection["count"] == 0 or any(
                    g["gesture"] == "OPEN_PALM"
                    for g in classified["gestures"]
                ):
                    print(
                        f"[GESTURE DEBUG] "
                        f"Hands={detection['count']} | "
                        f"Classified={classified['gestures']}"
                    )
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                result = {
                    "classifier_result": classified,
                    "detection_result": detection,
                    "camera_sequence": sequence,
                    "frame_captured_at": captured_at,
                    "processed_at": time.monotonic(),
                    "processing_ms": elapsed_ms,
                    "processing_size": (
                        detection_frame.shape[1], detection_frame.shape[0]
                    ),
                }
                with self._lock:
                    self._latest = result
                    self.last_processing_ms = elapsed_ms
                    self.processed_frames += 1
        except Exception as exc:
            self.error = exc
            self._running = False
        finally:
            if detector is not None:
                detector.close()

    def latest(self):
        with self._lock:
            return dict(self._latest) if self._latest is not None else None

    def stop(self):
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        self._thread = None

    close = stop
