"""
Gesture Sense - Optimized integrated pipeline benchmark

Run from project root:
    python benchmark_integrated_optimized.py

Optional:
    python benchmark_integrated_optimized.py --seconds 60 --camera 0

Uses:
  - MSMF 1280x720 threaded latest-frame capture
  - Existing hand detector at full camera resolution
  - Existing gesture classifier and decision engine
  - MediaPipe Tasks Selfie Segmenter
  - Gaussian-smoothed confidence mask + cv2.blendLinear compositing

Diagnostic only: does not modify existing project files.
Press Q or ESC to quit.
"""

from __future__ import annotations

import argparse
import threading
import time
from collections import defaultdict
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from detection.hand_detector import HandDetector
from detection.gesture_classifier import GestureClassifier
from decision.gesture_engine import GestureEngine

ROOT = Path(__file__).resolve().parent
HAND_MODEL = ROOT / "assets" / "models" / "hand_landmarker.task"
SEGMENT_MODEL = ROOT / "assets" / "models" / "selfie_segmenter_landscape.tflite"


class LatestFrameCamera:
    def __init__(self, index: int, width: int, height: int):
        self.cap = cv2.VideoCapture(index, cv2.CAP_MSMF)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open webcam with MSMF.")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.lock = threading.Lock()
        self.frame = None
        self.sequence = 0
        self.captured_at = 0.0
        self.running = True
        self.errors = 0
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def _reader(self):
        while self.running:
            ok, frame = self.cap.read()
            now = time.perf_counter()
            if not ok or frame is None:
                self.errors += 1
                time.sleep(0.005)
                continue
            with self.lock:
                self.frame = frame
                self.sequence += 1
                self.captured_at = now

    def latest(self):
        with self.lock:
            if self.frame is None:
                return None, self.sequence, None
            return self.frame.copy(), self.sequence, self.captured_at

    def close(self):
        self.running = False
        self.thread.join(timeout=2)
        self.cap.release()


def report(name, samples):
    if not samples:
        print(f"{name:<30} no samples")
        return
    a = np.asarray(samples, dtype=np.float64)
    print(
        f"{name:<30} mean {a.mean():7.2f} ms | "
        f"median {np.median(a):7.2f} ms | p95 {np.percentile(a, 95):7.2f} ms"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--threshold", type=float, default=0.0,
                        help="Optional mask cutoff before blur; 0 disables cutoff (default)")
    parser.add_argument("--blur", type=int, default=5)
    args = parser.parse_args()

    if not HAND_MODEL.exists():
        raise FileNotFoundError(HAND_MODEL)
    if not SEGMENT_MODEL.exists():
        raise FileNotFoundError(SEGMENT_MODEL)

    print("\nGESTURE SENSE - OPTIMIZED INTEGRATED PIPELINE")
    print("=" * 86)
    print("MSMF camera -> hand detection -> classifier -> decision -> segmentation -> blendLinear")
    print("Hand detection remains at full resolution; only compositing implementation changed.")
    print("Diagnostic only; existing application files are untouched.\n")

    camera = LatestFrameCamera(args.camera, args.width, args.height)
    print(f"Camera actual mode: {camera.width}x{camera.height}")

    detector = HandDetector(
        model_path=str(HAND_MODEL),
        max_hands=2,
        detection_confidence=0.6,
        presence_confidence=0.6,
        tracking_confidence=0.6,
    )
    classifier = GestureClassifier()
    engine = GestureEngine()

    options = mp.tasks.vision.ImageSegmenterOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(SEGMENT_MODEL)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        output_category_mask=False,
        output_confidence_masks=True,
    )
    segmenter = mp.tasks.vision.ImageSegmenter.create_from_options(options)

    timings = defaultdict(list)
    seen_sequence = -1
    processed = 0
    warmup = 30
    mp_timestamp = 0
    last_status = 0.0
    last_decision = None
    last_gestures = []
    start = time.perf_counter()
    deadline = start + args.seconds

    try:
        while time.perf_counter() < deadline:
            loop_start = time.perf_counter()
            frame, sequence, captured_at = camera.latest()
            if frame is None or sequence == seen_sequence:
                time.sleep(0.001)
                continue
            seen_sequence = sequence
            age_ms = (time.perf_counter() - captured_at) * 1000.0

            t = time.perf_counter()
            hand_result = detector.detect(frame)
            hand_ms = (time.perf_counter() - t) * 1000.0

            t = time.perf_counter()
            classified = classifier.classify(hand_result)
            classify_ms = (time.perf_counter() - t) * 1000.0
            last_gestures = classified.get("gestures", [])

            t = time.perf_counter()
            last_decision = engine.update(classified)
            decision_ms = (time.perf_counter() - t) * 1000.0

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            mp_timestamp = max(mp_timestamp + 1, int((time.perf_counter() - start) * 1000))

            t = time.perf_counter()
            seg_result = segmenter.segment_for_video(mp_image, mp_timestamp)
            seg_ms = (time.perf_counter() - t) * 1000.0

            t = time.perf_counter()
            if not seg_result.confidence_masks:
                raise RuntimeError("Segmentation returned no confidence masks.")
            mask = seg_result.confidence_masks[0].numpy_view()
            if mask.shape[:2] != frame.shape[:2]:
                mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]),
                                  interpolation=cv2.INTER_LINEAR)

            alpha = np.clip(mask, 0.0, 1.0)
            if args.threshold > 0:
                alpha = np.where(alpha >= args.threshold, alpha, 0.0).astype(np.float32)
            if args.blur > 0:
                k = max(1, args.blur)
                if k % 2 == 0:
                    k += 1
                alpha = cv2.GaussianBlur(alpha, (k, k), 0)

            background = np.full_like(frame, 32)
            alpha = np.ascontiguousarray(alpha, dtype=np.float32)
            inverse_alpha = np.ascontiguousarray(1.0 - alpha, dtype=np.float32)
            composite = cv2.blendLinear(frame, background, alpha, inverse_alpha)
            composite_ms = (time.perf_counter() - t) * 1000.0

            total_ms = (time.perf_counter() - loop_start) * 1000.0
            processed += 1

            if processed > warmup:
                timings["frame age"].append(age_ms)
                timings["hand detection"].append(hand_ms)
                timings["classification"].append(classify_ms)
                timings["decision engine"].append(decision_ms)
                timings["segmentation"].append(seg_ms)
                timings["blendLinear + mask"].append(composite_ms)
                timings["pipeline processing"].append(total_ms)

            preview = composite.copy()
            detector.draw_landmarks(preview, hand_result)
            cv2.imshow("Gesture Sense - Optimized Pipeline (Q to quit)", preview)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

            now = time.perf_counter()
            if now - last_status >= 2:
                elapsed = now - start
                print(
                    f"\rElapsed {elapsed:5.1f}s | camera {camera.sequence:4d} frames "
                    f"| processed {processed:4d} | capture {camera.sequence/elapsed:5.1f} FPS "
                    f"| pipeline {processed/elapsed:5.1f} FPS | age {age_ms:5.1f} ms",
                    end="", flush=True
                )
                last_status = now

    finally:
        camera.close()
        detector.close()
        segmenter.close()
        cv2.destroyAllWindows()

    elapsed = time.perf_counter() - start
    print("\n\n" + "=" * 86)
    print("FINAL RESULTS")
    print("=" * 86)
    print(f"Actual camera mode       : {camera.width}x{camera.height}")
    print(f"Camera unique-frame rate : {camera.sequence / elapsed:.2f} FPS")
    print(f"Pipeline processing rate : {processed / elapsed:.2f} FPS")
    print(f"Processed frames         : {processed}")
    print(f"Camera read errors       : {camera.errors}")
    print("\nStage timings (warmup excluded):")
    for name in (
        "frame age", "hand detection", "classification", "decision engine",
        "segmentation", "blendLinear + mask", "pipeline processing"
    ):
        report(name, timings[name])

    print("\nLast recognized gestures:")
    print(last_gestures)
    print("\nLast decision-engine output:")
    print(last_decision)
    print("\nThis diagnostic still uses a neutral background, not the final effect system.")


if __name__ == "__main__":
    main()
