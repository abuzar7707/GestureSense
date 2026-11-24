
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class HandDetector:

    def __init__(
        self,
        model_path=None,
        max_hands=2,
        detection_confidence=0.6,
        presence_confidence=0.6,
        tracking_confidence=0.6,
    ):
        project_root = Path(__file__).resolve().parent.parent

        self.model_path = (
            Path(model_path)
            if model_path
            else project_root / "assets" / "models" / "hand_landmarker.task"
        )

        self.max_hands = max_hands
        self.is_initialized = False

        self.landmarker = None
        self.last_timestamp_ms = 0

        self._initialize(
            detection_confidence,
            presence_confidence,
            tracking_confidence,
        )

    def _initialize(
        self,
        detection_confidence,
        presence_confidence,
        tracking_confidence,
    ):
        """Initialize MediaPipe Hand Landmarker."""

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Hand Landmarker model not found: {self.model_path}"
            )

        base_options = python.BaseOptions(
            model_asset_path=str(self.model_path)
        )

        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=self.max_hands,
            min_hand_detection_confidence=detection_confidence,
            min_hand_presence_confidence=presence_confidence,
            min_tracking_confidence=tracking_confidence,
        )

        self.landmarker = vision.HandLandmarker.create_from_options(
            options
        )

        self.is_initialized = True

    def detect(self, frame):
        """
        Detect hands in an OpenCV BGR frame.

        Returns:
            {
                "count": int,
                "hands": [
                    {
                        "handedness": str,
                        "handedness_score": float,
                        "landmarks": [
                            {"x": float, "y": float, "z": float}
                        ]
                    }
                ],
                "timestamp_ms": int
            }
        """

        if not self.is_initialized:
            raise RuntimeError("Hand detector is not initialized.")

        if frame is None:
            raise ValueError("Input frame cannot be None.")

        if not isinstance(frame, np.ndarray):
            raise TypeError("Input frame must be a NumPy array.")

        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("Expected a 3-channel BGR image.")

        # OpenCV BGR -> MediaPipe RGB
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        rgb_frame = np.ascontiguousarray(rgb_frame)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # MediaPipe VIDEO mode requires increasing timestamps.
        timestamp_ms = time.monotonic_ns() // 1_000_000

        timestamp_ms = max(
            timestamp_ms,
            self.last_timestamp_ms + 1
        )

        self.last_timestamp_ms = timestamp_ms

        result = self.landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )

        detected_hands = []

        for hand_index, hand_landmarks in enumerate(
            result.hand_landmarks
        ):

            handedness = result.handedness[hand_index][0]

            landmarks = []

            for landmark in hand_landmarks:
                landmarks.append({
                    "x": float(landmark.x),
                    "y": float(landmark.y),
                    "z": float(landmark.z),
                })

            detected_hands.append({
                "handedness": handedness.category_name,
                "handedness_score": float(handedness.score),
                "landmarks": landmarks,
            })

        return {
            "count": len(detected_hands),
            "hands": detected_hands,
            "timestamp_ms": timestamp_ms,
        }

    def draw_landmarks(self, frame, detection_result):
        """Draw detected hand landmarks and connections."""

        output = frame.copy()

        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (5, 9), (9, 10), (10, 11), (11, 12),
            (9, 13), (13, 14), (14, 15), (15, 16),
            (13, 17), (17, 18), (18, 19), (19, 20),
            (0, 17),
        ]

        height, width = output.shape[:2]

        for hand in detection_result["hands"]:

            landmarks = hand["landmarks"]

            points = []

            for landmark in landmarks:
                x = int(landmark["x"] * width)
                y = int(landmark["y"] * height)

                points.append((x, y))

            for start, end in connections:
                cv2.line(
                    output,
                    points[start],
                    points[end],
                    (0, 255, 0),
                    2
                )

            for point in points:
                cv2.circle(
                    output,
                    point,
                    4,
                    (0, 140, 255),
                    -1
                )

            wrist_x, wrist_y = points[0]

            cv2.putText(
                output,
                hand["handedness"],
                (wrist_x, wrist_y - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        return output

    def close(self):
        """Release MediaPipe resources."""

        if self.landmarker is not None:
            self.landmarker.close()
            self.landmarker = None

        self.is_initialized = False
