"""Independent MediaPipe Tasks person-segmentation worker.

Changes vs. previous version (public API unchanged, new args have defaults):
  * input_width: frame is downscaled before RGB conversion / MediaPipe. The
    model resizes to ~256px internally anyway, so full-res input was wasted work.
  * Blur and temporal smoothing run at the low mask resolution; only the final
    mask is upscaled to the camera resolution.
  * temporal_smoothing: EMA over masks to cut edge flicker (1.0 = off).
  * compute_composite: the full-res diagnostic composite is optional.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


class SegmentationPipeline:
    def __init__(
        self,
        camera,
        model_path,
        max_fps=30.0,
        blur_kernel=5,
        background_value=32,
        input_width=320,
        temporal_smoothing=0.8,
        compute_composite=True,
    ):
        self.camera = camera
        self.model_path = Path(model_path)
        self.max_fps = float(max_fps)
        self.blur_kernel = max(0, int(blur_kernel))
        self.background_value = int(background_value)
        self.input_width = max(0, int(input_width))  # 0 = full resolution
        self.temporal_smoothing = float(np.clip(temporal_smoothing, 0.05, 1.0))
        self.compute_composite = bool(compute_composite)

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
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Segmentation model not found: {self.model_path}")
        self._running = True
        self._thread = threading.Thread(
            target=self._run, name="GestureSense-SegmentationPipeline", daemon=True
        )
        self._thread.start()
        return self

    def _run(self):
        segmenter = None
        try:
            options = mp.tasks.vision.ImageSegmenterOptions(
                base_options=mp.tasks.BaseOptions(
                    model_asset_path=str(self.model_path)
                ),
                running_mode=mp.tasks.vision.RunningMode.VIDEO,
                output_category_mask=False,
                output_confidence_masks=True,
            )
            segmenter = mp.tasks.vision.ImageSegmenter.create_from_options(options)
            last_sequence = -1
            last_timestamp_ms = 0
            frame_interval = 1.0 / self.max_fps if self.max_fps > 0 else 0.0
            next_allowed = 0.0
            smoothed = None  # low-res EMA state

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
                started = time.perf_counter()

                height, width = frame.shape[:2]
                small = frame
                if self.input_width and width > self.input_width:
                    new_h = max(1, round(height * self.input_width / width))
                    small = cv2.resize(
                        frame, (self.input_width, new_h),
                        interpolation=cv2.INTER_AREA,
                    )

                rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=np.ascontiguousarray(rgb),
                )
                timestamp_ms = max(
                    last_timestamp_ms + 1, time.monotonic_ns() // 1_000_000
                )
                last_timestamp_ms = timestamp_ms
                result = segmenter.segment_for_video(mp_image, timestamp_ms)
                masks = result.confidence_masks or []
                if not masks:
                    raise RuntimeError("Segmentation returned no confidence masks.")

                mask = np.squeeze(masks[0].numpy_view()).astype(np.float32)
                mask = np.clip(mask, 0.0, 1.0)

                # Temporal smoothing + blur at low resolution (cheap).
                if self.temporal_smoothing < 1.0:
                    if smoothed is None or smoothed.shape != mask.shape:
                        smoothed = mask.copy()
                    else:
                        smoothed = (
                            self.temporal_smoothing * mask
                            + (1.0 - self.temporal_smoothing) * smoothed
                        )
                    mask = smoothed
                if self.blur_kernel > 1:
                    k = self.blur_kernel if self.blur_kernel % 2 else self.blur_kernel + 1
                    mask = cv2.GaussianBlur(mask, (k, k), 0)

                if mask.shape[:2] != (height, width):
                    mask = cv2.resize(
                        mask, (width, height), interpolation=cv2.INTER_LINEAR
                    )

                alpha = np.ascontiguousarray(np.clip(mask, 0.0, 1.0), dtype=np.float32)
                composite = None
                if self.compute_composite:
                    inverse_alpha = np.ascontiguousarray(1.0 - alpha, dtype=np.float32)
                    background = np.full_like(frame, self.background_value)
                    composite = cv2.blendLinear(
                        frame, background, alpha, inverse_alpha
                    )
                elapsed_ms = (time.perf_counter() - started) * 1000.0

                snapshot = {
                    "mask": alpha,
                    "composite": composite,
                    "camera_sequence": sequence,
                    "frame_captured_at": captured_at,
                    "processed_at": time.monotonic(),
                    "processing_ms": elapsed_ms,
                    "frame_size": (width, height),
                }
                with self._lock:
                    self._latest = snapshot
                    self.last_processing_ms = elapsed_ms
                    self.processed_frames += 1
        except Exception as exc:
            self.error = exc
            self._running = False
        finally:
            if segmenter is not None:
                segmenter.close()

    def latest(self, copy_arrays=True):
        with self._lock:
            if self._latest is None:
                return None
            result = dict(self._latest)
            if copy_arrays:
                result["mask"] = result["mask"].copy()
                if result["composite"] is not None:
                    result["composite"] = result["composite"].copy()
            return result

    def stop(self):
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        self._thread = None

    close = stop