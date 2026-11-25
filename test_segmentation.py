"""
Gesture Sense - Milestone 3.1
Upper-body segmentation refinement diagnostic.

Run from the GestureSense project root:
    python test_segmentation_refined.py

Controls:
    Q / ESC : quit
    S       : save diagnostic images
    R       : reset temporal smoothing history

Trackbars (in the "Segmentation Controls" window):
    Threshold       : person confidence threshold (0.35 - 0.75)
    Blur             : Gaussian blur kernel size (1 = off)
    Morphology       : cleanup kernel size (1 = off)
    Temporal Smooth  : current-frame weight (100 = off, lower = smoother)

This script is standalone and does not modify the Gesture Sense app.
"""

import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "assets" / "models" / "selfie_segmenter_landscape.tflite"

CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

WINDOW_CAMERA = "Gesture Sense - Camera"
WINDOW_RAW = "Gesture Sense - Raw Mask"
WINDOW_REFINED = "Gesture Sense - Refined Mask"
WINDOW_CUTOUT = "Gesture Sense - Refined Cutout"
WINDOW_CONTROLS = "Gesture Sense - Segmentation Controls"


def odd_kernel(value: int) -> int:
    value = max(1, int(value))
    return value if value % 2 else value + 1


def create_controls():
    cv2.namedWindow(WINDOW_CONTROLS, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_CONTROLS, 520, 180)

    cv2.createTrackbar("Threshold (35-75)", WINDOW_CONTROLS, 50, 75, lambda _: None)
    cv2.setTrackbarMin("Threshold (35-75)", WINDOW_CONTROLS, 35)

    cv2.createTrackbar("Blur (1=off)", WINDOW_CONTROLS, 5, 21, lambda _: None)
    cv2.createTrackbar("Morphology (1=off)", WINDOW_CONTROLS, 3, 15, lambda _: None)

    # 100 means no temporal blending; 65 means 65% current + 35% previous.
    cv2.createTrackbar("Temporal current weight", WINDOW_CONTROLS, 65, 100, lambda _: None)


def refine_mask(raw_mask: np.ndarray, previous_mask: np.ndarray | None):
    threshold_value = cv2.getTrackbarPos("Threshold (35-75)", WINDOW_CONTROLS) / 100.0
    blur_size = odd_kernel(cv2.getTrackbarPos("Blur (1=off)", WINDOW_CONTROLS))
    morph_size = odd_kernel(cv2.getTrackbarPos("Morphology (1=off)", WINDOW_CONTROLS))
    current_weight = cv2.getTrackbarPos("Temporal current weight", WINDOW_CONTROLS) / 100.0

    # Temporal smoothing is applied to the confidence mask before thresholding.
    if previous_mask is not None and current_weight < 1.0:
        stable_mask = (
            current_weight * raw_mask
            + (1.0 - current_weight) * previous_mask
        )
    else:
        stable_mask = raw_mask.copy()

    # Smooth confidence transitions. A kernel size of 1 disables this step.
    if blur_size > 1:
        stable_mask = cv2.GaussianBlur(
            stable_mask, (blur_size, blur_size), sigmaX=0
        )

    binary = (stable_mask >= threshold_value).astype(np.uint8) * 255

    # Small opening removes isolated foreground specks; closing fills tiny holes.
    # Keep the kernel small to avoid erasing fingers and narrow body regions.
    if morph_size > 1:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (morph_size, morph_size)
        )
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # A soft alpha is useful for previewing edge quality.
    alpha = stable_mask.copy()
    alpha = np.clip((alpha - max(0.0, threshold_value - 0.20)) / 0.40, 0.0, 1.0)
    if blur_size > 1:
        alpha = cv2.GaussianBlur(alpha, (blur_size, blur_size), sigmaX=0)

    return binary, alpha, stable_mask, threshold_value, blur_size, morph_size, current_weight


def main():
    if not MODEL_PATH.is_file():
        print(f"[ERROR] Segmentation model not found:\n{MODEL_PATH}")
        print("Expected: assets/models/selfie_segmenter_landscape.tflite")
        return 1

    print(f"[INFO] MediaPipe: {mp.__version__}")
    print(f"[INFO] Model: {MODEL_PATH}")

    BaseOptions = mp.tasks.BaseOptions
    ImageSegmenter = mp.tasks.vision.ImageSegmenter
    ImageSegmenterOptions = mp.tasks.vision.ImageSegmenterOptions
    RunningMode = mp.tasks.vision.RunningMode

    options = ImageSegmenterOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=RunningMode.VIDEO,
        output_category_mask=False,
        output_confidence_masks=True,
    )

    camera = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not camera.isOpened():
        camera.release()
        camera = cv2.VideoCapture(CAMERA_INDEX)

    if not camera.isOpened():
        print(f"[ERROR] Could not open camera index {CAMERA_INDEX}.")
        return 1

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    create_controls()

    previous_mask = None
    last_timestamp_ms = 0

    fps_window_start = time.perf_counter()
    frame_counter = 0
    total_fps = 0.0
    inference_ms = 0.0
    processing_ms = 0.0

    try:
        with ImageSegmenter.create_from_options(options) as segmenter:
            print("[INFO] Segmentation model loaded.")
            print("[INFO] Use the trackbars to tune the mask. Press S to save.")

            while True:
                loop_start = time.perf_counter()
                ok, frame_bgr = camera.read()
                if not ok:
                    print("[WARNING] Could not read camera frame.")
                    break

                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=np.ascontiguousarray(frame_rgb),
                )

                timestamp_ms = time.perf_counter_ns() // 1_000_000
                timestamp_ms = max(timestamp_ms, last_timestamp_ms + 1)
                last_timestamp_ms = timestamp_ms

                inference_start = time.perf_counter()
                result = segmenter.segment_for_video(mp_image, timestamp_ms)
                inference_ms = (time.perf_counter() - inference_start) * 1000.0

                masks = result.confidence_masks or []
                if not masks:
                    raise RuntimeError("The segmenter returned no confidence masks.")

                # This landscape model returns one foreground/person confidence mask.
                raw_mask = np.squeeze(masks[0].numpy_view()).astype(np.float32)

                height, width = frame_bgr.shape[:2]
                if raw_mask.shape[:2] != (height, width):
                    raw_mask = cv2.resize(
                        raw_mask, (width, height), interpolation=cv2.INTER_LINEAR
                    )

                raw_mask = np.clip(raw_mask, 0.0, 1.0)

                (
                    refined_binary,
                    alpha,
                    stable_mask,
                    threshold_value,
                    blur_size,
                    morph_size,
                    current_weight,
                ) = refine_mask(raw_mask, previous_mask)

                previous_mask = stable_mask.copy()

                raw_binary = (raw_mask >= threshold_value).astype(np.uint8) * 255

                # Alpha cutout: softer edges than a hard binary mask.
                alpha_3 = alpha[:, :, None]
                cutout = (frame_bgr.astype(np.float32) * alpha_3).clip(0, 255).astype(np.uint8)

                processing_ms = (time.perf_counter() - loop_start) * 1000.0
                frame_counter += 1

                elapsed = time.perf_counter() - fps_window_start
                if elapsed >= 1.0:
                    total_fps = frame_counter / elapsed
                    frame_counter = 0
                    fps_window_start = time.perf_counter()

                camera_preview = frame_bgr.copy()
                cv2.putText(
                    camera_preview,
                    f"Pipeline: {total_fps:.1f} FPS | Inference: {inference_ms:.1f} ms",
                    (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                    (0, 255, 0), 2, cv2.LINE_AA,
                )

                raw_preview = raw_binary.copy()
                refined_preview = refined_binary.copy()
                cv2.putText(
                    raw_preview, f"Raw mask | threshold {threshold_value:.2f}",
                    (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, 255, 2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    refined_preview,
                    f"Refined | blur {blur_size} | morph {morph_size} | temporal {current_weight:.2f}",
                    (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.48, 255, 1,
                    cv2.LINE_AA,
                )

                cv2.imshow(WINDOW_CAMERA, camera_preview)
                cv2.imshow(WINDOW_RAW, raw_preview)
                cv2.imshow(WINDOW_REFINED, refined_preview)
                cv2.imshow(WINDOW_CUTOUT, cutout)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                elif key == ord("r"):
                    previous_mask = None
                    print("[INFO] Temporal smoothing history reset.")
                elif key == ord("s"):
                    output_dir = PROJECT_ROOT / "segmentation_refinement_output"
                    output_dir.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(output_dir / "camera.png"), frame_bgr)
                    cv2.imwrite(str(output_dir / "raw_mask.png"), raw_binary)
                    cv2.imwrite(str(output_dir / "refined_mask.png"), refined_binary)
                    cv2.imwrite(str(output_dir / "refined_cutout.png"), cutout)
                    cv2.imwrite(
                        str(output_dir / "confidence_mask.png"),
                        (raw_mask * 255).clip(0, 255).astype(np.uint8),
                    )
                    print(f"[INFO] Saved previews to: {output_dir}")

    except Exception as exc:
        print(f"\n[ERROR] Segmentation test failed: {exc}")
        raise
    finally:
        camera.release()
        cv2.destroyAllWindows()

    print("[INFO] Segmentation refinement test closed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
