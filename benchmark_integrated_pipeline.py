"""
Gesture Sense - Integrated pipeline benchmark (diagnostic only)

Run from the Gesture Sense project root:
    python benchmark_integrated_pipeline.py

This script does NOT modify the main application or project files.
It benchmarks:
  - MSMF webcam capture (latest-frame threaded capture)
  - Hand detection
  - Gesture classification
  - Decision engine update
  - MediaPipe Tasks Selfie Segmentation
  - Soft-alpha mask refinement and compositing

Expected project modules:
  detection/hand_detector.py
  detection/gesture_classifier.py
  decision/gesture_engine.py
  assets/models/hand_landmarker.task
  assets/models/selfie_segmenter_landscape.tflite

Press Q or ESC in the preview window to stop.
"""

from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path
from collections import defaultdict

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
    """Continuously reads the camera and exposes only the newest frame."""

    def __init__(self, camera_index: int, width: int, height: int):
        self.cap = cv2.VideoCapture(camera_index, cv2.CAP_MSMF)
        if not self.cap.isOpened():
            self.cap.release()
            raise RuntimeError("Could not open webcam with Media Foundation (MSMF).")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.reported_fps = self.cap.get(cv2.CAP_PROP_FPS)

        self.lock = threading.Lock()
        self.frame = None
        self.sequence = 0
        self.frame_time = 0.0
        self.running = True
        self.read_errors = 0
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def _reader(self):
        while self.running:
            ok, frame = self.cap.read()
            now = time.perf_counter()
            if not ok or frame is None:
                self.read_errors += 1
                time.sleep(0.005)
                continue
            with self.lock:
                self.frame = frame
                self.sequence += 1
                self.frame_time = now

    def get_latest(self):
        with self.lock:
            if self.frame is None:
                return None, self.sequence, None
            return self.frame.copy(), self.sequence, self.frame_time

    def close(self):
        self.running = False
        self.thread.join(timeout=2.0)
        self.cap.release()


def percentile(values, p):
    return float(np.percentile(np.asarray(values, dtype=np.float64), p)) if values else 0.0


def report_stage(name, values):
    if not values:
        print(f"{name:<28} no samples")
        return
    arr = np.asarray(values, dtype=np.float64)
    print(
        f"{name:<28} mean {np.mean(arr):7.2f} ms | "
        f"median {np.median(arr):7.2f} ms | p95 {np.percentile(arr, 95):7.2f} ms"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--blur", type=int, default=5)
    args = parser.parse_args()

    if not HAND_MODEL.exists():
        raise FileNotFoundError(f"Hand model not found: {HAND_MODEL}")
    if not SEGMENT_MODEL.exists():
        raise FileNotFoundError(f"Segmentation model not found: {SEGMENT_MODEL}")

    print("\nGESTURE SENSE - INTEGRATED PIPELINE BENCHMARK")
    print("=" * 86)
    print("Backend: MSMF | Target: 1280x720 @ 30 FPS")
    print("Pipeline: camera -> hands -> classifier -> decision -> segmentation -> soft alpha")
    print("This is a diagnostic script; it does not change the main application.\n")

    camera = LatestFrameCamera(args.camera, args.width, args.height)
    print(
        f"Camera actual mode: {camera.actual_width}x{camera.actual_height} | "
        f"driver-reported FPS: {camera.reported_fps:.1f}"
    )

    hand_detector = HandDetector(
        model_path=str(HAND_MODEL),
        max_hands=2,
        detection_confidence=0.6,
        presence_confidence=0.6,
        tracking_confidence=0.6,
    )
    classifier = GestureClassifier()
    gesture_engine = GestureEngine()

    segmenter_options = mp.tasks.vision.ImageSegmenterOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(SEGMENT_MODEL)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        output_category_mask=False,
        output_confidence_masks=True,
    )
    segmenter = mp.tasks.vision.ImageSegmenter.create_from_options(segmenter_options)

    timings = defaultdict(list)
    processed = 0
    last_sequence = -1
    start = time.perf_counter()
    end_at = start + args.seconds
    last_status = 0.0
    last_preview = None
    last_decision = None
    last_gestures = []
    warmup_frames = 30
    mp_timestamp = 0

    try:
        while time.perf_counter() < end_at:
            loop_start = time.perf_counter()
            frame, sequence, captured_at = camera.get_latest()

            if frame is None or sequence == last_sequence:
                time.sleep(0.001)
                continue

            last_sequence = sequence
            frame_age_ms = (time.perf_counter() - captured_at) * 1000.0
            timings["frame age"].append(frame_age_ms)

            # 1. Hand detection
            t = time.perf_counter()
            detection_result = hand_detector.detect(frame)
            timings["hand detection"].append((time.perf_counter() - t) * 1000.0)

            # 2. Gesture classification
            t = time.perf_counter()
            classification_result = classifier.classify(detection_result)
            timings["classification"].append((time.perf_counter() - t) * 1000.0)
            last_gestures = classification_result.get("gestures", [])

            # 3. Decision engine
            t = time.perf_counter()
            last_decision = gesture_engine.update(classification_result)
            timings["decision engine"].append((time.perf_counter() - t) * 1000.0)

            # 4. Person segmentation (VIDEO timestamps must increase)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            mp_timestamp = max(mp_timestamp + 1, int((time.perf_counter() - start) * 1000))

            t = time.perf_counter()
            segmentation_result = segmenter.segment_for_video(mp_image, mp_timestamp)
            timings["segmentation"].append((time.perf_counter() - t) * 1000.0)

            # 5. Soft alpha mask and composite.
            t = time.perf_counter()
            confidence_masks = segmentation_result.confidence_masks
            if not confidence_masks:
                raise RuntimeError("Segmentation returned no confidence masks.")

            person_mask = confidence_masks[0].numpy_view()
            if person_mask.shape[:2] != frame.shape[:2]:
                person_mask = cv2.resize(
                    person_mask,
                    (frame.shape[1], frame.shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                )

            alpha = np.clip(person_mask, 0.0, 1.0)
            if args.blur > 0:
                k = max(1, args.blur)
                if k % 2 == 0:
                    k += 1
                alpha = cv2.GaussianBlur(alpha, (k, k), 0)

            # Composite the person over a neutral background to measure the
            # soft-alpha path without requiring a clean background plate.
            background = np.full_like(frame, 32)
            alpha3 = alpha[:, :, None].astype(np.float32)
            composite = (
                frame.astype(np.float32) * alpha3
                + background.astype(np.float32) * (1.0 - alpha3)
            ).clip(0, 255).astype(np.uint8)
            timings["soft alpha + composite"].append((time.perf_counter() - t) * 1000.0)

            timings["pipeline processing"].append((time.perf_counter() - loop_start) * 1000.0)
            processed += 1

            # Draw landmarks only for preview; do not include in pipeline timing.
            preview = composite.copy()
            hand_detector.draw_landmarks(preview, detection_result)
            last_preview = preview

            now = time.perf_counter()
            if now - last_status >= 2.0:
                elapsed = now - start
                capture_fps = camera.sequence / elapsed if elapsed > 0 else 0.0
                processing_fps = processed / elapsed if elapsed > 0 else 0.0
                print(
                    f"\rElapsed {elapsed:5.1f}s | camera frames {camera.sequence:4d} "
                    f"| processed {processed:4d} | camera {capture_fps:5.1f} FPS "
                    f"| pipeline {processing_fps:5.1f} FPS "
                    f"| frame age {frame_age_ms:5.1f} ms",
                    end="",
                    flush=True,
                )
                last_status = now

            if last_preview is not None:
                cv2.imshow("Gesture Sense - Integrated Benchmark (Q to quit)", last_preview)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break

            if processed <= warmup_frames:
                # Keep warmup samples out of the final timing statistics.
                for key in list(timings.keys()):
                    if timings[key]:
                        timings[key].pop()

    finally:
        camera.close()
        hand_detector.close()
        segmenter.close()
        cv2.destroyAllWindows()

    elapsed = time.perf_counter() - start
    unique_camera_fps = camera.sequence / elapsed if elapsed > 0 else 0.0
    processing_fps = processed / elapsed if elapsed > 0 else 0.0

    print("\n\n" + "=" * 86)
    print("FINAL RESULTS")
    print("=" * 86)
    print(f"Actual camera mode       : {camera.actual_width}x{camera.actual_height}")
    print(f"Camera unique-frame rate : {unique_camera_fps:.2f} FPS")
    print(f"Pipeline processing rate : {processing_fps:.2f} FPS")
    print(f"Processed frames         : {processed}")
    print(f"Camera read errors       : {camera.read_errors}")
    print("\nStage timings (warmup excluded):")
    for stage in (
        "frame age",
        "hand detection",
        "classification",
        "decision engine",
        "segmentation",
        "soft alpha + composite",
        "pipeline processing",
    ):
        report_stage(stage, timings[stage])

    print("\nLast recognized gestures:")
    print(last_gestures)
    print("\nLast decision-engine output:")
    print(last_decision)
    print("\nInterpretation:")
    print("- Camera unique-frame rate is the actual webcam input rate.")
    print("- Pipeline processing rate is how quickly unique frames were processed.")
    print("- If processing FPS is below camera FPS, some newest frames will be skipped.")
    print("- If frame age stays low, the pipeline is not accumulating a stale frame queue.")
    print("- This test composites onto a neutral background; it does not test disappearance effects.")


if __name__ == "__main__":
    main()
