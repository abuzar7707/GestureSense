"""
Gesture Sense - Threaded Camera + Optimized Mask Benchmark

Run from the GestureSense project root:
    python benchmark_selfie_threaded.py

This is a diagnostic benchmark, not a replacement for the app camera module.
It keeps only the newest camera frame (no growing frame queue), so processing
does not wait directly on VideoCapture.read().

Controls:
    Q / ESC : quit and print summary
    S       : save current camera, mask and cutout

The mask/cutout stage uses OpenCV uint8 operations for a binary cutout.
This is intentionally a fast-path benchmark; soft alpha compositing will be
benchmarked separately after we establish the capture/processing baseline.
"""

import sys
import time
import threading
from pathlib import Path
from collections import deque

import cv2
import mediapipe as mp
import numpy as np


ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "assets" / "models" / "selfie_segmenter_landscape.tflite"


class LatestFrameCamera:
    """Continuously reads the camera and retains only the newest frame."""

    def __init__(self, camera_index=0, width=640, height=480):
        self.cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap.release()
            self.cap = cv2.VideoCapture(camera_index)

        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera {camera_index}")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.frame = None
        self.frame_time = 0.0
        self.sequence = 0
        self.running = False
        self.lock = threading.Lock()
        self.thread = None
        self.read_times = deque(maxlen=600)
        self.read_fps = 0.0

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()
        return self

    def _reader(self):
        fps_start = time.perf_counter()
        fps_count = 0

        while self.running:
            start = time.perf_counter()
            ok, frame = self.cap.read()
            elapsed_ms = (time.perf_counter() - start) * 1000.0

            if not ok:
                time.sleep(0.005)
                continue

            captured_at = time.perf_counter()
            with self.lock:
                self.frame = frame
                self.frame_time = captured_at
                self.sequence += 1
                self.read_times.append(elapsed_ms)

            fps_count += 1
            elapsed = captured_at - fps_start
            if elapsed >= 1.0:
                self.read_fps = fps_count / elapsed
                fps_count = 0
                fps_start = captured_at

    def get_latest(self):
        with self.lock:
            if self.frame is None:
                return None, 0.0, 0
            # Copy so the reader cannot overwrite the frame while processing.
            return self.frame.copy(), self.frame_time, self.sequence

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=2.0)
        self.cap.release()


def summarize(label, values):
    if not values:
        print(f"{label:30s}: no samples")
        return
    arr = np.asarray(values, dtype=np.float64)
    print(
        f"{label:30s}: mean={arr.mean():7.2f} ms | "
        f"median={np.median(arr):7.2f} ms | "
        f"p95={np.percentile(arr, 95):7.2f} ms"
    )


def main():
    if not MODEL.is_file():
        print(f"[ERROR] Model not found: {MODEL}")
        return 1

    camera = LatestFrameCamera().start()
    print("[INFO] MediaPipe:", mp.__version__)
    print("[INFO] Threaded latest-frame camera started.")
    print("[INFO] Warm-up: first 30 processed frames excluded.")
    print("[INFO] Q to quit; S to save sample.")

    options = mp.tasks.vision.ImageSegmenterOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        output_category_mask=False,
        output_confidence_masks=True,
    )

    stages = {
        "latest frame copy": [],
        "BGR to RGB + MP image": [],
        "model inference": [],
        "mask extraction": [],
        "binary mask + uint8 cutout": [],
        "display": [],
        "processing total (excl. display)": [],
        "frame age at processing": [],
    }

    processed = 0
    last_ts = 0
    last_sequence = -1
    processed_fps = 0.0
    fps_start = time.perf_counter()
    fps_count = 0
    last_frame = last_mask = last_cutout = None

    try:
        with mp.tasks.vision.ImageSegmenter.create_from_options(options) as segmenter:
            cv2.namedWindow("Gesture Sense - Threaded Camera", cv2.WINDOW_NORMAL)
            while True:
                loop_start = time.perf_counter()

                t = time.perf_counter()
                frame, captured_at, sequence = camera.get_latest()
                copy_ms = (time.perf_counter() - t) * 1000.0

                if frame is None:
                    time.sleep(0.002)
                    continue

                # Avoid processing the same camera frame repeatedly.
                if sequence == last_sequence:
                    time.sleep(0.001)
                    continue
                last_sequence = sequence

                age_ms = (time.perf_counter() - captured_at) * 1000.0

                t = time.perf_counter()
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=np.ascontiguousarray(rgb),
                )
                timestamp = time.perf_counter_ns() // 1_000_000
                timestamp = max(timestamp, last_ts + 1)
                last_ts = timestamp
                convert_ms = (time.perf_counter() - t) * 1000.0

                t = time.perf_counter()
                result = segmenter.segment_for_video(mp_image, timestamp)
                infer_ms = (time.perf_counter() - t) * 1000.0

                t = time.perf_counter()
                masks = result.confidence_masks or []
                if not masks:
                    raise RuntimeError("SelfieSegmenter returned no confidence mask.")
                person = np.squeeze(masks[0].numpy_view())
                if person.dtype != np.float32:
                    person = person.astype(np.float32)
                h, w = frame.shape[:2]
                if person.shape[:2] != (h, w):
                    person = cv2.resize(person, (w, h), interpolation=cv2.INTER_LINEAR)
                person_u8 = np.clip(person * 255.0, 0, 255).astype(np.uint8)
                mask_ms = (time.perf_counter() - t) * 1000.0

                # Fast binary cutout path: no full-frame float32 BGR conversion.
                t = time.perf_counter()
                _, binary = cv2.threshold(person_u8, 153, 255, cv2.THRESH_BINARY)
                cutout = cv2.bitwise_and(frame, frame, mask=binary)
                post_ms = (time.perf_counter() - t) * 1000.0

                processing_ms = (time.perf_counter() - loop_start) * 1000.0
                processed += 1
                fps_count += 1
                now = time.perf_counter()
                if now - fps_start >= 1.0:
                    processed_fps = fps_count / (now - fps_start)
                    fps_count = 0
                    fps_start = now

                preview = frame.copy()
                cv2.putText(
                    preview,
                    f"Threaded | {processed_fps:.1f} processed FPS | "
                    f"camera {camera.read_fps:.1f} FPS",
                    (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 255, 0), 2, cv2.LINE_AA
                )

                t = time.perf_counter()
                cv2.imshow("Gesture Sense - Threaded Camera", preview)
                cv2.imshow("Gesture Sense - Threaded Mask", binary)
                cv2.imshow("Gesture Sense - Threaded Cutout", cutout)
                key = cv2.waitKey(1) & 0xFF
                display_ms = (time.perf_counter() - t) * 1000.0

                if processed > 30:
                    stages["latest frame copy"].append(copy_ms)
                    stages["BGR to RGB + MP image"].append(convert_ms)
                    stages["model inference"].append(infer_ms)
                    stages["mask extraction"].append(mask_ms)
                    stages["binary mask + uint8 cutout"].append(post_ms)
                    stages["display"].append(display_ms)
                    stages["processing total (excl. display)"].append(processing_ms)
                    stages["frame age at processing"].append(age_ms)

                last_frame, last_mask, last_cutout = frame, binary, cutout

                if key in (ord("q"), 27):
                    break
                elif key == ord("s"):
                    out = ROOT / "threaded_benchmark_output"
                    out.mkdir(exist_ok=True)
                    cv2.imwrite(str(out / "camera.png"), last_frame)
                    cv2.imwrite(str(out / "mask.png"), last_mask)
                    cv2.imwrite(str(out / "cutout.png"), last_cutout)
                    print(f"[INFO] Saved sample to {out}")

    finally:
        camera.stop()
        cv2.destroyAllWindows()

    print("\n========== THREADED BENCHMARK ==========")
    print(f"Processed unique frames: {processed}")
    print(f"Camera latest-frame read FPS: {camera.read_fps:.2f}")
    summarize("Camera read call", list(camera.read_times))
    for name, values in stages.items():
        summarize(name, values)
    if stages["processing total (excl. display)"]:
        avg = np.mean(stages["processing total (excl. display)"])
        print(f"Processing throughput estimate: {1000.0 / avg:.2f} FPS")
    print("========================================\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
