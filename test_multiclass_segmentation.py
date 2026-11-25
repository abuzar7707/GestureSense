"""
Gesture Sense - Milestone 3.1
MediaPipe Selfie Multiclass segmentation benchmark.

Run from the GestureSense project root:
    python test_multiclass_segmentation.py

Expected model:
    assets/models/selfie_multiclass_256x256.tflite

Controls:
    Q / ESC : quit
    S       : save diagnostic images
    R       : reset temporal smoothing history

Trackbars:
    Threshold (35-75)       : threshold for combined person confidence
    Blur (1=off)            : Gaussian smoothing kernel
    Temporal current weight : 100 disables temporal blending

Class order from the model:
    0 background
    1 hair
    2 body-skin
    3 face-skin
    4 clothes
    5 others / accessories

The combined person mask uses classes 1-5. This intentionally includes the
model's "others/accessories" class for the first benchmark. If it leaks
non-person objects, we can compare a classes 1-4 mask afterward.
"""

import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = (
    PROJECT_ROOT / "assets" / "models" /
    "selfie_multiclass_256x256.tflite"
)

CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

WINDOW_CAMERA = "Gesture Sense - Multiclass Camera"
WINDOW_CATEGORY = "Gesture Sense - Multiclass Categories"
WINDOW_CONFIDENCE = "Gesture Sense - Combined Person Confidence"
WINDOW_MASK = "Gesture Sense - Multiclass Person Mask"
WINDOW_CUTOUT = "Gesture Sense - Multiclass Cutout"
WINDOW_CONTROLS = "Gesture Sense - Multiclass Controls"

CLASS_NAMES = [
    "background", "hair", "body-skin",
    "face-skin", "clothes", "others"
]


def odd_kernel(value: int) -> int:
    value = max(1, int(value))
    return value if value % 2 else value + 1


def create_controls():
    cv2.namedWindow(WINDOW_CONTROLS, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_CONTROLS, 520, 150)
    cv2.createTrackbar("Threshold (35-75)", WINDOW_CONTROLS, 50, 75, lambda _: None)
    cv2.setTrackbarMin("Threshold (35-75)", WINDOW_CONTROLS, 35)
    cv2.createTrackbar("Blur (1=off)", WINDOW_CONTROLS, 3, 21, lambda _: None)
    cv2.createTrackbar("Temporal current weight", WINDOW_CONTROLS, 85, 100, lambda _: None)


def main():
    if not MODEL_PATH.is_file():
        print(f"[ERROR] Multiclass model not found:\n{MODEL_PATH}")
        print("Expected: assets/models/selfie_multiclass_256x256.tflite")
        return 1

    print("[INFO] MediaPipe:", mp.__version__)
    print("[INFO] Model:", MODEL_PATH)

    BaseOptions = mp.tasks.BaseOptions
    ImageSegmenter = mp.tasks.vision.ImageSegmenter
    ImageSegmenterOptions = mp.tasks.vision.ImageSegmenterOptions
    RunningMode = mp.tasks.vision.RunningMode

    options = ImageSegmenterOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=RunningMode.VIDEO,
        output_category_mask=True,
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

    cv2.namedWindow(WINDOW_CAMERA, cv2.WINDOW_NORMAL)
    cv2.namedWindow(WINDOW_CATEGORY, cv2.WINDOW_NORMAL)
    cv2.namedWindow(WINDOW_CONFIDENCE, cv2.WINDOW_NORMAL)
    cv2.namedWindow(WINDOW_MASK, cv2.WINDOW_NORMAL)
    cv2.namedWindow(WINDOW_CUTOUT, cv2.WINDOW_NORMAL)
    create_controls()

    previous_mask = None
    last_timestamp_ms = 0
    fps_start = time.perf_counter()
    frame_count = 0
    pipeline_fps = 0.0
    inference_ms = 0.0

    try:
        with ImageSegmenter.create_from_options(options) as segmenter:
            print("[INFO] Multiclass model loaded.")
            print("[INFO] Press S to save previews, R to reset smoothing, Q to quit.")

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

                infer_start = time.perf_counter()
                result = segmenter.segment_for_video(mp_image, timestamp_ms)
                inference_ms = (time.perf_counter() - infer_start) * 1000.0

                masks = result.confidence_masks or []
                if len(masks) != 6:
                    raise RuntimeError(
                        "Expected six class confidence masks "
                        f"(background, hair, body-skin, face-skin, clothes, others); "
                        f"received {len(masks)}. Check model file and MediaPipe API."
                    )

                class_masks = [
                    np.squeeze(mask.numpy_view()).astype(np.float32)
                    for mask in masks
                ]

                h, w = frame_bgr.shape[:2]
                resized = []
                for mask in class_masks:
                    if mask.shape[:2] != (h, w):
                        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
                    resized.append(np.clip(mask, 0.0, 1.0))
                class_masks = resized

                # Sum foreground class probabilities (hair through accessories).
                person_confidence = np.clip(
                    np.sum(np.stack(class_masks[1:], axis=0), axis=0),
                    0.0, 1.0
                )

                threshold = cv2.getTrackbarPos("Threshold (35-75)", WINDOW_CONTROLS) / 100.0
                blur_size = odd_kernel(cv2.getTrackbarPos("Blur (1=off)", WINDOW_CONTROLS))
                current_weight = (
                    cv2.getTrackbarPos("Temporal current weight", WINDOW_CONTROLS) / 100.0
                )

                if previous_mask is not None and current_weight < 1.0:
                    stable = (
                        current_weight * person_confidence
                        + (1.0 - current_weight) * previous_mask
                    )
                else:
                    stable = person_confidence.copy()

                if blur_size > 1:
                    stable = cv2.GaussianBlur(stable, (blur_size, blur_size), 0)

                previous_mask = stable.copy()

                binary = (stable >= threshold).astype(np.uint8) * 255

                # Soft alpha derived from the stable combined confidence.
                alpha = np.clip((stable - max(0.0, threshold - 0.20)) / 0.40, 0.0, 1.0)
                if blur_size > 1:
                    alpha = cv2.GaussianBlur(alpha, (blur_size, blur_size), 0)
                cutout = (
                    frame_bgr.astype(np.float32) * alpha[:, :, None]
                ).clip(0, 255).astype(np.uint8)

                # Visualize winning category index. Colors are only diagnostic.
                category = np.argmax(np.stack(class_masks, axis=0), axis=0).astype(np.uint8)
                category_color = np.zeros((h, w, 3), dtype=np.uint8)
                palette = [
                    (0, 0, 0),       # background
                    (255, 80, 180),  # hair
                    (0, 220, 255),   # body skin
                    (255, 180, 0),   # face skin
                    (80, 220, 80),   # clothes
                    (180, 100, 255), # others/accessories
                ]
                for idx, color in enumerate(palette):
                    category_color[category == idx] = color

                frame_count += 1
                elapsed = time.perf_counter() - fps_start
                if elapsed >= 1.0:
                    pipeline_fps = frame_count / elapsed
                    frame_count = 0
                    fps_start = time.perf_counter()

                camera_preview = frame_bgr.copy()
                cv2.putText(
                    camera_preview,
                    f"Multiclass | {pipeline_fps:.1f} FPS | inference {inference_ms:.1f} ms",
                    (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                    (0, 255, 0), 2, cv2.LINE_AA
                )
                confidence_preview = (
                    stable * 255.0
                ).clip(0, 255).astype(np.uint8)

                cv2.putText(
                    category_color, "Categories: hair / skin / clothes / others",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                    (255, 255, 255), 1, cv2.LINE_AA
                )
                cv2.putText(
                    confidence_preview, f"Combined person confidence | threshold {threshold:.2f}",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.48, 255, 1, cv2.LINE_AA
                )
                cv2.putText(
                    binary, "Combined person binary mask",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, 255, 1, cv2.LINE_AA
                )

                cv2.imshow(WINDOW_CAMERA, camera_preview)
                cv2.imshow(WINDOW_CATEGORY, category_color)
                cv2.imshow(WINDOW_CONFIDENCE, confidence_preview)
                cv2.imshow(WINDOW_MASK, binary)
                cv2.imshow(WINDOW_CUTOUT, cutout)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                elif key == ord("r"):
                    previous_mask = None
                    print("[INFO] Temporal history reset.")
                elif key == ord("s"):
                    output_dir = PROJECT_ROOT / "multiclass_segmentation_output"
                    output_dir.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(output_dir / "camera.png"), frame_bgr)
                    cv2.imwrite(str(output_dir / "category_map.png"), category_color)
                    cv2.imwrite(str(output_dir / "confidence.png"), confidence_preview)
                    cv2.imwrite(str(output_dir / "binary_mask.png"), binary)
                    cv2.imwrite(str(output_dir / "cutout.png"), cutout)
                    for i, name in enumerate(CLASS_NAMES):
                        class_image = (class_masks[i] * 255).clip(0, 255).astype(np.uint8)
                        cv2.imwrite(str(output_dir / f"class_{i}_{name}.png"), class_image)
                    print(f"[INFO] Saved multiclass outputs to: {output_dir}")

    except Exception as exc:
        print(f"\n[ERROR] Multiclass segmentation test failed: {exc}")
        raise
    finally:
        camera.release()
        cv2.destroyAllWindows()

    print("[INFO] Multiclass segmentation test closed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
