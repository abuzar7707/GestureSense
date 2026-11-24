import math


class GestureClassifier:
    """Geometry-based classifier for MediaPipe 21-point hand landmarks.

    Confidence is a geometric quality score, not a calibrated probability.
    Peace and Rock intentionally ignore thumb position.
    """

    GESTURES = {
        "THUMBS_UP": (0, 0, 0, 0),       # index, middle, ring, pinky
        "PEACE": (1, 1, 0, 0),
        "OPEN_PALM": (1, 1, 1, 1, 1),    # thumb + four fingers
        "ROCK_SIGN": (1, 0, 0, 1),
    }

    FINGER_NAMES = ["thumb", "index", "middle", "ring", "pinky"]

    # MCP, PIP, DIP, TIP
    FINGER_INDICES = {
        "index": (5, 6, 7, 8),
        "middle": (9, 10, 11, 12),
        "ring": (13, 14, 15, 16),
        "pinky": (17, 18, 19, 20),
    }

    def __init__(
        self,
        extension_threshold=0.10,
        minimum_confidence=0.60,
        extended_angle=155.0,
        folded_angle=145.0,
    ):
        # Retained for compatibility with existing code that passes this arg.
        self.extension_threshold = extension_threshold
        self.minimum_confidence = float(minimum_confidence)
        self.extended_angle = float(extended_angle)
        self.folded_angle = float(folded_angle)

        if not 90.0 < self.folded_angle < self.extended_angle < 180.0:
            raise ValueError(
                "Angle thresholds must satisfy 90 < folded < extended < 180."
            )

    @staticmethod
    def _xyz(point):
        return (
            float(point["x"]),
            float(point["y"]),
            float(point.get("z", 0.0)),
        )

    @classmethod
    def _distance(cls, point_a, point_b):
        a, b = cls._xyz(point_a), cls._xyz(point_b)
        return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))

    @classmethod
    def _angle(cls, point_a, vertex, point_b):
        """Return the 3D angle A-vertex-B in degrees."""
        a, v, b = cls._xyz(point_a), cls._xyz(vertex), cls._xyz(point_b)
        va = tuple(a[i] - v[i] for i in range(3))
        vb = tuple(b[i] - v[i] for i in range(3))
        ma = math.sqrt(sum(x * x for x in va))
        mb = math.sqrt(sum(x * x for x in vb))
        if ma < 1e-8 or mb < 1e-8:
            return 0.0
        cosine = sum(va[i] * vb[i] for i in range(3)) / (ma * mb)
        return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))

    def _finger_geometry(self, landmarks):
        palm_scale = self._distance(landmarks[0], landmarks[9])
        if palm_scale < 1e-4:
            raise ValueError("Invalid or degenerate hand landmarks.")

        states, margins, angles, quality = {}, {}, {}, {}

        for finger, (mcp_i, pip_i, dip_i, tip_i) in self.FINGER_INDICES.items():
            pip_angle = self._angle(
                landmarks[mcp_i], landmarks[pip_i], landmarks[dip_i]
            )
            dip_angle = self._angle(
                landmarks[pip_i], landmarks[dip_i], landmarks[tip_i]
            )
            # Both joints should be relatively straight for an extended finger.
            joint_angle = min(pip_angle, dip_angle)
            angles[finger] = {
                "pip": round(pip_angle, 1),
                "dip": round(dip_angle, 1),
            }
            margins[finger] = round(
                (joint_angle - self.extended_angle) / 30.0, 3
            )

            if joint_angle >= self.extended_angle:
                states[finger] = 1
                quality[finger] = min(
                    1.0, 0.65 + (joint_angle - self.extended_angle) / 70.0
                )
            elif joint_angle <= self.folded_angle:
                states[finger] = 0
                quality[finger] = min(
                    1.0, 0.65 + (self.folded_angle - joint_angle) / 70.0
                )
            else:
                states[finger] = int(
                    joint_angle >= (self.folded_angle + self.extended_angle) / 2
                )
                quality[finger] = 0.45

        # Thumb uses its own joint and span checks; it is not judged by the
        # same plane-dependent rule as the four fingers.
        thumb_angle = self._angle(landmarks[2], landmarks[3], landmarks[4])
        thumb_span = self._distance(landmarks[4], landmarks[2]) / palm_scale
        thumb_abduction = self._distance(landmarks[4], landmarks[5]) / palm_scale
        thumb_straight = (
            thumb_angle >= self.extended_angle and thumb_span >= 0.28
        )
        states["thumb"] = int(thumb_straight)
        margins["thumb"] = round((thumb_angle - self.extended_angle) / 30.0, 3)
        angles["thumb"] = {
            "ip": round(thumb_angle, 1),
            "span": round(thumb_span, 3),
            "abduction": round(thumb_abduction, 3),
        }
        quality["thumb"] = (
            min(1.0, 0.65 + (thumb_angle - self.extended_angle) / 70.0)
            if thumb_straight else
            min(1.0, 0.65 + (self.folded_angle - thumb_angle) / 70.0)
        )

        return states, margins, angles, quality, thumb_straight, thumb_abduction

    @staticmethod
    def _pattern_score(states, quality, names, expected):
        scores = [
            quality[name] if states[name] == expected_state else 0.0
            for name, expected_state in zip(names, expected)
        ]
        return sum(scores) / len(scores) if scores else 0.0

    def classify_hand(self, hand):
        landmarks = hand.get("landmarks", [])
        if len(landmarks) != 21:
            raise ValueError("Expected exactly 21 hand landmarks.")

        states, margins, angles, quality, thumb_straight, thumb_abduction = (
            self._finger_geometry(landmarks)
        )
        four_names = ("index", "middle", "ring", "pinky")
        four_states = tuple(states[name] for name in four_names)
        candidates = []

        # Thumb is deliberately a wildcard for these two gestures.
        if four_states == self.GESTURES["PEACE"]:
            score = self._pattern_score(
                states, quality, four_names, self.GESTURES["PEACE"]
            )
            candidates.append(("PEACE", score))

        if four_states == self.GESTURES["ROCK_SIGN"]:
            score = self._pattern_score(
                states, quality, four_names, self.GESTURES["ROCK_SIGN"]
            )
            candidates.append(("ROCK_SIGN", score))

        if (
            four_states == (1, 1, 1, 1)
            and thumb_straight
            and thumb_abduction >= 0.30
        ):
            all_names = ("thumb",) + four_names
            score = self._pattern_score(
                states, quality, all_names, self.GESTURES["OPEN_PALM"]
            )
            candidates.append(("OPEN_PALM", score))

        # A thumbs-up requires folded fingers and a straight thumb pointing
        # upward in the image. If the hand is rotated sideways, it may return
        # UNKNOWN; this avoids incorrectly treating a sideways thumb as up.
        thumb_tip_above_mcp = (
            landmarks[4]["y"] < landmarks[2]["y"] - 0.04
        )
        if (
            four_states == self.GESTURES["THUMBS_UP"]
            and thumb_straight
            and thumb_tip_above_mcp
        ):
            finger_score = self._pattern_score(
                states, quality, four_names, self.GESTURES["THUMBS_UP"]
            )
            score = finger_score * 0.8 + quality["thumb"] * 0.2
            candidates.append(("THUMBS_UP", score))

        if candidates:
            gesture, confidence = max(candidates, key=lambda item: item[1])
        else:
            gesture, confidence = "UNKNOWN", 0.0

        if confidence < self.minimum_confidence:
            gesture = "UNKNOWN"

        return {
            "gesture": gesture,
            "confidence": round(confidence, 3),
            "handedness": hand.get("handedness", "Unknown"),
            "finger_states": states,
            "finger_margins": margins,
            "joint_angles": angles,
        }

    def classify(self, detection_result):
        """Classify every hand in a HandDetector result."""
        hands = detection_result.get("hands", [])
        classifications = [self.classify_hand(hand) for hand in hands]
        return {"count": len(classifications), "gestures": classifications}
