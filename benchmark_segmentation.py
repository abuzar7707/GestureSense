"""
Gesture Sense - Segmentation Performance Benchmark

Run from the GestureSense project root.

Run the fast landscape model:
    python benchmark_segmentation.py --model selfie

Run the multiclass model:
    python benchmark_segmentation.py --model multiclass

Optional camera size:
    python benchmark_segmentation.py --model selfie --width 640 --height 480

Controls:
    Q / ESC : quit and print benchmark summary
    S       : save current frame, mask and cutout

This benchmark runs ONE model at a time so the two models can be compared
without one model consuming CPU time while the other is being measured.
"""

import argparse
import sys
import time
from pathlib import Path
from collections import deque

import cv2
import mediapipe as mp
import numpy as np


ROOT = Path(__file__).resolve().parent
MODEL_PATHS = {
    "selfie": ROOT / "assets" / "models" / "selfie_segmenter_landscape.tflite",
    "multiclass": ROOT / "assets" / "models" / "selfie_multiclass_256x256.tflite",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("selfie", "multiclass"), required=True)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.60)
    return parser.parse_args()


def percentile(values, p):
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float32), p))


def main():
    args = parse_args()
    model_path = MODEL_PATHS[args.model]

    if not model_path.is_file():
        print(f"[ERROR] Model file not found: {model_path}")
        return 1

    print(f"[INFO] MediaPipe: {mp.__version__}")
    print(f"[INFO] Model: {args.model} ({model_path.name})")
    print(f"[INFO] Camera request: {args.width}x{args.height}")
    print("[INFO] Warm-up: first 30 frames are excluded from reported statistics.")

    vision = mp.tasks.vision
    options = vision.ImageSegmenterOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        output_category_mask=False,
        output_confidence_masks=True,
    )

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"[ERROR] Could not open camera {args.camera}.")
        return 1

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Actual camera resolution: {actual_w}x{actual_h}")

    inference_samples = []
    loop_samples = []
    fps_samples = deque(maxlen=30)
    last_timestamp = 0
    frame_number = 0
    measured_start = None
    fps_window_start = time.perf_counter()
    fps_window_frames = 0
    displayed_fps = 0.0
    latest_mask = None
    latest_cutout = None
    latest_frame = None

    try:
        with mp.tasks.vision.ImageSegmenter.create_from_options(options) as segmenter:
            print("[INFO] Model loaded. Keep your normal upper-body position.")
            print("[INFO] Press S to save a sample; Q to finish.")

            while True:
                loop_start = time.perf_counter()
                ok, frame = cap.read()
                if not ok:
                    print("[WARNING] Camera read failed.")
                    break

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=np.ascontiguousarray(rgb),
                )

                timestamp = time.perf_counter_ns() // 1_000_000
                timestamp = max(timestamp, last_timestamp + 1)
                last_timestamp = timestamp

                infer_start = time.perf_counter()
                result = segmenter.segment_for_video(mp_image, timestamp)
                infer_ms = (time.perf_counter() - infer_start) * 1000.0

                masks = result.confidence_masks or []
                if args.model == "selfie":
                    if len(masks) < 1:
                        raise RuntimeError("SelfieSegmenter returned no confidence mask.")
                    person = np.squeeze(masks[0].numpy_view()).astype(np.float32)
                else:
                    if len(masks) != 6:
                        raise RuntimeError(
                            f"Expected 6 multiclass masks; received {len(masks)}."
                        )
                    # Classes 1-5: hair, body skin, face skin, clothes, others.
                    class_arrays = [
                        np.squeeze(m.numpy_view()).astype(np.float32) for m in masks
                    ]
                    person = np.sum(np.stack(class_arrays[1:], axis=0), axis=0)

                h, w = frame.shape[:2]
                if person.shape[:2] != (h, w):
                    person = cv2.resize(person, (w, h), interpolation=cv2.INTER_LINEAR)
                person = np.clip(person, 0.0, 1.0)

                binary = (person >= args.threshold).astype(np.uint8) * 255
                alpha = np.clip(
                    (person - max(0.0, args.threshold - 0.20)) / 0.40,
                    0.0, 1.0
                )
                alpha = cv2.GaussianBlur(alpha, (3, 3), 0)
                cutout = (
                    frame.astype(np.float32) * alpha[:, :, None]
                ).clip(0, 255).astype(np.uint8)

                loop_ms = (time.perf_counter() - loop_start) * 1000.0
                frame_number += 1
                fps_window_frames += 1

                if frame_number == 31:
                    measured_start = time.perf_counter()
                    inference_samples.clear()
                    loop_samples.clear()

                if frame_number > 30:
                    inference_samples.append(infer_ms)
                    loop_samples.append(loop_ms)

                now = time.perf_counter()
                if now - fps_window_start >= 1.0:
                    displayed_fps = fps_window_frames / (now - fps_window_start)
                    fps_samples.append(displayed_fps)
                    fps_window_frames = 0
                    fps_window_start = now

                preview = frame.copy()
                cv2.putText(
                    preview,
                    f"{args.model.upper()} | {displayed_fps:.1f} FPS | "
                    f"Inference {infer_ms:.1f} ms",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                    (0, 255, 0), 2, cv2.LINE_AA
                )
                cv2.imshow("Gesture Sense - Benchmark Camera", preview)
                cv2.imshow("Gesture Sense - Benchmark Mask", binary)
                cv2.imshow("Gesture Sense - Benchmark Cutout", cutout)

                latest_frame, latest_mask, latest_cutout = frame, binary, cutout

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                elif key == ord("s") and latest_frame is not None:
                    out = ROOT / f"benchmark_{args.model}_output"
                    out.mkdir(exist_ok=True)
                    cv2.imwrite(str(out / "camera.png"), latest_frame)
                    cv2.imwrite(str(out / "mask.png"), latest_mask)
                    cv2.imwrite(str(out / "cutout.png"), latest_cutout)
                    print(f"[INFO] Saved sample to {out}")

    except Exception as exc:
        print(f"[ERROR] Benchmark failed: {exc}")
        raise
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print("\n========== BENCHMARK SUMMARY ==========")
    print(f"Model: {args.model}")
    print(f"Camera resolution: {actual_w}x{actual_h}")
    print(f"Frames captured: {frame_number}")
    print(f"Measured frames (after warm-up): {len(inference_samples)}")

    if inference_samples:
        print(
            "Inference latency ms: "
            f"mean={np.mean(inference_samples):.2f}, "
            f"median={np.median(inference_samples):.2f}, "
            f"p95={percentile(inference_samples, 95):.2f}"
        )
        print(
            "Capture + inference + mask processing loop ms: "
            f"mean={np.mean(loop_samples):.2f}, "
            f"median={np.median(loop_samples):.2f}, "
            f"p95={percentile(loop_samples, 95):.2f}"
        )
        if loop_samples:
            print(f"Approx. processing throughput: {1000.0 / np.mean(loop_samples):.2f} FPS")
        if fps_samples:
            print(f"Mean displayed FPS windows: {np.mean(fps_samples):.2f}")
    else:
        print("Not enough frames for measured statistics. Run longer and try again.")

    print("=======================================\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
