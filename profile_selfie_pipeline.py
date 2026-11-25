"""
Gesture Sense - SelfieSegmenter Stage Profiler

Run from the GestureSense project root:
    python profile_selfie_pipeline.py

Optional:
    python profile_selfie_pipeline.py --camera 0 --seconds 45

This profiles each stage separately to locate the time outside model inference.
The first 30 frames are excluded from reported statistics.

Controls:
    Q / ESC : quit and print results
"""

import argparse
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "assets" / "models" / "selfie_segmenter_landscape.tflite"


def stats(values):
    if not values:
        return "n/a"
    a = np.asarray(values, dtype=np.float64)
    return f"mean {a.mean():7.2f} ms | median {np.median(a):7.2f} ms | p95 {np.percentile(a, 95):7.2f} ms"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--seconds", type=int, default=45)
    args = parser.parse_args()

    if not MODEL.is_file():
        raise FileNotFoundError(f"Model not found: {MODEL}")

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {args.camera}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    actual_size = (
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    print(f"[INFO] Actual camera resolution: {actual_size[0]}x{actual_size[1]}")
    print(f"[INFO] MediaPipe: {mp.__version__}")
    print("[INFO] Profiling for", args.seconds, "seconds; first 30 frames are warm-up.")

    options = mp.tasks.vision.ImageSegmenterOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        output_category_mask=False,
        output_confidence_masks=True,
    )

    names = [
        "camera read",
        "BGR -> RGB",
        "MediaPipe image creation",
        "model inference",
        "mask extraction + resize",
        "threshold + alpha + cutout",
        "preview + imshow + waitKey",
        "total frame time",
    ]
    samples = {name: [] for name in names}
    frame_no = 0
    last_ts = 0
    start = time.perf_counter()
    last_display = start
    display_frames = 0
    display_fps = 0.0

    try:
        with mp.tasks.vision.ImageSegmenter.create_from_options(options) as segmenter:
            cv2.namedWindow("Gesture Sense - Pipeline Profiler", cv2.WINDOW_NORMAL)
            while True:
                frame_start = time.perf_counter()

                t = time.perf_counter()
                ok, frame = cap.read()
                t_read = (time.perf_counter() - t) * 1000
                if not ok:
                    print("[WARN] Camera read failed.")
                    break

                t = time.perf_counter()
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                t_convert = (time.perf_counter() - t) * 1000

                t = time.perf_counter()
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=np.ascontiguousarray(rgb),
                )
                timestamp = time.perf_counter_ns() // 1_000_000
                timestamp = max(timestamp, last_ts + 1)
                last_ts = timestamp
                t_image = (time.perf_counter() - t) * 1000

                t = time.perf_counter()
                result = segmenter.segment_for_video(mp_image, timestamp)
                t_infer = (time.perf_counter() - t) * 1000

                t = time.perf_counter()
                masks = result.confidence_masks or []
                if not masks:
                    raise RuntimeError("SelfieSegmenter returned no confidence mask.")
                person = np.squeeze(masks[0].numpy_view()).astype(np.float32)
                h, w = frame.shape[:2]
                if person.shape[:2] != (h, w):
                    person = cv2.resize(person, (w, h), interpolation=cv2.INTER_LINEAR)
                person = np.clip(person, 0.0, 1.0)
                t_mask = (time.perf_counter() - t) * 1000

                t = time.perf_counter()
                binary = (person >= 0.60).astype(np.uint8) * 255
                alpha = np.clip((person - 0.40) / 0.40, 0.0, 1.0)
                alpha = cv2.GaussianBlur(alpha, (3, 3), 0)
                cutout = (
                    frame.astype(np.float32) * alpha[:, :, None]
                ).clip(0, 255).astype(np.uint8)
                t_post = (time.perf_counter() - t) * 1000

                frame_no += 1
                display_frames += 1
                now = time.perf_counter()
                if now - last_display >= 1.0:
                    display_fps = display_frames / (now - last_display)
                    display_frames = 0
                    last_display = now

                t = time.perf_counter()
                preview = frame.copy()
                cv2.putText(
                    preview,
                    f"SelfieSegmenter | {display_fps:.1f} FPS | Q to quit",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0), 2, cv2.LINE_AA
                )
                cv2.imshow("Gesture Sense - Pipeline Profiler", preview)
                cv2.imshow("Gesture Sense - Mask", binary)
                cv2.imshow("Gesture Sense - Cutout", cutout)
                key = cv2.waitKey(1) & 0xFF
                t_display = (time.perf_counter() - t) * 1000

                total = (time.perf_counter() - frame_start) * 1000

                if frame_no > 30:
                    values = [
                        t_read, t_convert, t_image, t_infer,
                        t_mask, t_post, t_display, total
                    ]
                    for name, value in zip(names, values):
                        samples[name].append(value)

                if key in (ord("q"), 27):
                    break
                if time.perf_counter() - start >= args.seconds:
                    break

    finally:
        cap.release()
        cv2.destroyAllWindows()

    print("\n========== STAGE PROFILE ==========")
    print(f"Model: SelfieSegmenter Landscape")
    print(f"Camera: {actual_size[0]}x{actual_size[1]}")
    print(f"Frames: {frame_no} (warm-up excluded from statistics)")
    for name in names:
        print(f"{name:34s}: {stats(samples[name])}")
    if samples["total frame time"]:
        mean_total = np.mean(samples["total frame time"])
        print(f"\nApprox. measured throughput: {1000.0 / mean_total:.2f} FPS")
    print("===================================\n")


if __name__ == "__main__":
    main()
