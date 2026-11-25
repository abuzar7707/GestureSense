"""
Gesture Sense - Hand detection resolution / gesture accuracy comparison

Run from the project root:
    python test_gesture_resolution.py

Controls:
    1 = label current pose as OPEN_PALM
    2 = label current pose as THUMBS_UP
    3 = label current pose as PEACE
    4 = label current pose as ROCK_SIGN
    0 = label as NO_HAND / UNKNOWN
    SPACE = ignore current frame
    Q or ESC = finish and print results

The same camera frame is evaluated by two independent HandDetector instances:
  - Full resolution (normally 1280x720)
  - Reduced resolution (640x360)

The existing GestureClassifier is used for both. The test records a sample
when a label key is pressed. Hold one clear pose, press its label key, then
change pose and repeat. Collect at least 20-30 samples per gesture, using
both hands and varied distances/lighting if possible.

Diagnostic only; does not modify existing project files.
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from detection.hand_detector import HandDetector
from detection.gesture_classifier import GestureClassifier

ROOT = Path(__file__).resolve().parent
HAND_MODEL = ROOT / "assets" / "models" / "hand_landmarker.task"

LABELS = {
    ord("1"): "OPEN_PALM",
    ord("2"): "THUMBS_UP",
    ord("3"): "PEACE",
    ord("4"): "ROCK_SIGN",
    ord("0"): "NO_HAND",
}


def get_predictions(classifier, detector, frame):
    result = detector.detect(frame)
    classified = classifier.classify(result)
    predictions = [
        item.get("gesture", "UNKNOWN")
        for item in classified.get("gestures", [])
    ]
    return result, classified, predictions


def is_correct(expected, predictions):
    if expected == "NO_HAND":
        return len(predictions) == 0 or all(p == "UNKNOWN" for p in predictions)
    return expected in predictions


def draw_panel(frame, title, predictions, color):
    panel = frame.copy()
    cv2.rectangle(panel, (0, 0), (panel.shape[1], 42), (25, 25, 25), -1)
    cv2.putText(panel, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (245, 245, 245), 2, cv2.LINE_AA)
    text = ", ".join(predictions) if predictions else "NO HAND"
    cv2.putText(panel, text[:52], (12, 70), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, color, 2, cv2.LINE_AA)
    return panel


def print_summary(samples):
    print("\n\n" + "=" * 88)
    print("GESTURE RECOGNITION ACCURACY SUMMARY")
    print("=" * 88)
    print(f"{'Expected pose':<16} {'Samples':>8} {'1280x720':>12} {'640x360':>12}")
    print("-" * 56)

    all_full = []
    all_small = []

    for label in ("OPEN_PALM", "THUMBS_UP", "PEACE", "ROCK_SIGN", "NO_HAND"):
        rows = samples[label]
        if not rows:
            print(f"{label:<16} {0:>8} {'N/A':>12} {'N/A':>12}")
            continue
        full_rate = sum(r["full_ok"] for r in rows) / len(rows) * 100
        small_rate = sum(r["small_ok"] for r in rows) / len(rows) * 100
        print(f"{label:<16} {len(rows):>8} {full_rate:>11.1f}% {small_rate:>11.1f}%")
        all_full.extend(r["full_ok"] for r in rows)
        all_small.extend(r["small_ok"] for r in rows)

    print("-" * 56)
    if all_full:
        full_rate = sum(all_full) / len(all_full) * 100
        small_rate = sum(all_small) / len(all_small) * 100
        print(f"{'OVERALL':<16} {len(all_full):>8} {full_rate:>11.1f}% {small_rate:>11.1f}%")
        print(f"\nReduced-resolution accuracy difference: {small_rate - full_rate:+.1f} percentage points")
    else:
        print("No labeled samples were collected.")

    print("\nInterpretation:")
    print("- This is a manually labeled comparison, not an automated benchmark dataset.")
    print("- Collect balanced samples for all four gestures before drawing conclusions.")
    print("- If 640x360 is close in accuracy and meaningfully faster, it may be suitable.")
    print("- If it misses gestures or increases UNKNOWN results, retain full resolution.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--small-width", type=int, default=640)
    parser.add_argument("--small-height", type=int, default=360)
    args = parser.parse_args()

    if not HAND_MODEL.exists():
        raise FileNotFoundError(f"Hand model not found: {HAND_MODEL}")

    cap = cv2.VideoCapture(args.camera, cv2.CAP_MSMF)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam using MSMF.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, 30)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    full_detector = HandDetector(
        model_path=str(HAND_MODEL), max_hands=2,
        detection_confidence=0.6, presence_confidence=0.6,
        tracking_confidence=0.6,
    )
    small_detector = HandDetector(
        model_path=str(HAND_MODEL), max_hands=2,
        detection_confidence=0.6, presence_confidence=0.6,
        tracking_confidence=0.6,
    )
    full_classifier = GestureClassifier()
    small_classifier = GestureClassifier()

    samples = defaultdict(list)
    full_times = []
    small_times = []
    latest = {
        "full_predictions": [],
        "small_predictions": [],
        "full_result": None,
        "small_result": None,
        "full_frame": None,
        "small_frame": None,
    }

    print("\nGESTURE SENSE - GESTURE RESOLUTION ACCURACY TEST")
    print("=" * 88)
    print(f"Camera actual resolution: {actual_w}x{actual_h}")
    print(f"Comparison: {actual_w}x{actual_h} vs {args.small_width}x{args.small_height}")
    print("\nHold a pose steadily, then press its label key:")
    print("1 Open Palm | 2 Thumbs Up | 3 Peace | 4 Rock Sign | 0 No hand")
    print("SPACE ignores a frame | Q finishes the test\n")

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("Camera frame read failed.")
                break

            small_frame = cv2.resize(
                frame, (args.small_width, args.small_height),
                interpolation=cv2.INTER_AREA
            )

            t0 = time.perf_counter()
            full_result, full_classified, full_predictions = get_predictions(
                full_classifier, full_detector, frame
            )
            full_times.append((time.perf_counter() - t0) * 1000)

            t0 = time.perf_counter()
            small_result, small_classified, small_predictions = get_predictions(
                small_classifier, small_detector, small_frame
            )
            small_times.append((time.perf_counter() - t0) * 1000)

            full_detector.draw_landmarks(frame, full_result)
            small_detector.draw_landmarks(small_frame, small_result)

            full_panel = draw_panel(frame, f"FULL {actual_w}x{actual_h}", full_predictions, (80, 230, 120))
            small_panel = draw_panel(small_frame, f"REDUCED {args.small_width}x{args.small_height}",
                                     small_predictions, (80, 210, 255))
            small_panel = cv2.resize(
                small_panel, (full_panel.shape[1], full_panel.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )
            combined = np.hstack((full_panel, small_panel))

            cv2.putText(combined, "1 Open Palm | 2 Thumbs Up | 3 Peace | 4 Rock | 0 No Hand | SPACE Ignore | Q Quit",
                        (12, combined.shape[0] - 18), cv2.FONT_HERSHEY_SIMPLEX,
                        0.52, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow("Gesture Sense - Resolution Accuracy Comparison", combined)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key in LABELS:
                expected = LABELS[key]
                row = {
                    "full_predictions": list(full_predictions),
                    "small_predictions": list(small_predictions),
                    "full_ok": is_correct(expected, full_predictions),
                    "small_ok": is_correct(expected, small_predictions),
                }
                samples[expected].append(row)
                print(
                    f"{expected:<10} | full: {full_predictions or ['NO HAND']} "
                    f"({'OK' if row['full_ok'] else 'MISS'}) | "
                    f"reduced: {small_predictions or ['NO HAND']} "
                    f"({'OK' if row['small_ok'] else 'MISS'})"
                )
            elif key == ord(" "):
                pass

    finally:
        cap.release()
        full_detector.close()
        small_detector.close()
        cv2.destroyAllWindows()

    print_summary(samples)

    if full_times and small_times:
        print("\nDetector timing comparison (both detectors are run sequentially):")
        print(f"Full-resolution mean: {np.mean(full_times):.2f} ms")
        print(f"Reduced-resolution mean: {np.mean(small_times):.2f} ms")
        print("Note: timings include classification and can be affected by running two models together.")


if __name__ == "__main__":
    main()
