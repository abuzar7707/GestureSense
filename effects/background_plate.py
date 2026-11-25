"""In-memory clean background plate capture and validation.

New: capture_burst() builds the plate as a per-pixel MEDIAN over ~1 s of frames
and rejects the capture if anything moved. A single-frame capture can keep
motion-blurred remnants of a person (hair strands, shadows) that the
segmenter scores as background; the median removes anything transient.
"""
from __future__ import annotations

import time

import cv2
import numpy as np


class BackgroundPlate:
    def __init__(self, maximum_foreground_ratio=0.03):
        self.maximum_foreground_ratio = float(maximum_foreground_ratio)
        self.frame = None
        self.captured_foreground_ratio = None

    @staticmethod
    def foreground_ratio(person_mask, threshold=0.45):
        mask = np.asarray(person_mask, dtype=np.float32)
        if mask.ndim == 3:
            mask = np.squeeze(mask)
        return float(np.mean(mask >= threshold))

    def capture(self, frame, person_mask, force=False):
        if frame is None or not isinstance(frame, np.ndarray):
            raise ValueError("A valid camera frame is required.")
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("Expected a 3-channel BGR camera frame.")
        if person_mask is None:
            raise ValueError("A segmentation mask is required to validate the plate.")

        mask = np.asarray(person_mask, dtype=np.float32)
        if mask.ndim == 3:
            mask = np.squeeze(mask)
        if mask.shape != frame.shape[:2]:
            mask = cv2.resize(
                mask, (frame.shape[1], frame.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )

        ratio = self.foreground_ratio(mask)
        if not force and ratio > self.maximum_foreground_ratio:
            return {
                "success": False,
                "foreground_ratio": ratio,
                "message": (
                    "A person or foreground object is still detected. "
                    "Step fully out of frame and capture again."
                ),
            }

        self.frame = frame.copy()
        self.captured_foreground_ratio = ratio
        return {
            "success": True,
            "foreground_ratio": ratio,
            "message": "Clean background plate captured in memory.",
        }

    def capture_burst(
        self,
        get_frame_and_mask,
        duration=1.0,
        max_frames=15,
        motion_threshold=6.0,
        force=False,
    ):
        """Capture a plate as the per-pixel median of several frames.

        get_frame_and_mask: callable returning (bgr_frame, person_mask) or
            (None, None) if nothing is available yet. Blocks ~`duration` seconds,
            so call it from a worker thread or accept a short UI pause.
        motion_threshold: max allowed mean abs difference (0-255) between the
            median plate and any sampled frame, measured on a downscaled copy.
        """
        frames = []
        worst_ratio = 0.0
        deadline = time.monotonic() + float(duration)
        interval = float(duration) / max(1, int(max_frames))
        next_sample = 0.0

        while time.monotonic() < deadline and len(frames) < max_frames:
            if time.monotonic() < next_sample:
                time.sleep(0.005)
                continue
            frame, mask = get_frame_and_mask()
            if frame is None or mask is None:
                time.sleep(0.005)
                continue
            if frame.ndim != 3 or frame.shape[2] != 3:
                raise ValueError("Expected a 3-channel BGR camera frame.")
            worst_ratio = max(worst_ratio, self.foreground_ratio(mask))
            frames.append(frame.copy())
            next_sample = time.monotonic() + interval

        if len(frames) < 3:
            return {
                "success": False,
                "foreground_ratio": worst_ratio,
                "message": "Not enough frames received to build a background plate.",
            }
        if not force and worst_ratio > self.maximum_foreground_ratio:
            return {
                "success": False,
                "foreground_ratio": worst_ratio,
                "message": (
                    "A person or foreground object was detected during capture. "
                    "Step fully out of frame and capture again."
                ),
            }

        stack = np.stack(frames, axis=0)
        plate = np.median(stack, axis=0).astype(np.uint8)

        # Reject if the scene moved during the burst (person still leaving, etc.).
        small_plate = cv2.resize(plate, None, fx=0.25, fy=0.25).astype(np.float32)
        worst_motion = 0.0
        for f in frames:
            small = cv2.resize(f, None, fx=0.25, fy=0.25).astype(np.float32)
            worst_motion = max(worst_motion, float(np.mean(np.abs(small - small_plate))))
        if not force and worst_motion > motion_threshold:
            return {
                "success": False,
                "foreground_ratio": worst_ratio,
                "message": (
                    "The scene was still moving during capture. "
                    "Hold still out of frame and capture again."
                ),
            }

        self.frame = plate
        self.captured_foreground_ratio = worst_ratio
        return {
            "success": True,
            "foreground_ratio": worst_ratio,
            "message": f"Clean background plate captured from {len(frames)} frames.",
        }

    def get(self):
        return None if self.frame is None else self.frame.copy()

    def clear(self):
        self.frame = None
        self.captured_foreground_ratio = None

    @property
    def is_ready(self):
        return self.frame is not None