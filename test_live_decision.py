"""
Gesture Sense - Live Gesture Decision Engine Test

Run from the GestureSense project root:
    python test_live_decision.py

Optional:
    python test_live_decision.py --camera 1
    python test_live_decision.py --model assets/models/hand_landmarker.task
    python test_live_decision.py --transition-seconds 1.2

Controls:
    Q / ESC : quit
    C       : manually complete the simulated active transition
    R       : reset visibility state to VISIBLE and clear candidate tracking

This is a diagnostic harness only. It does not apply visual effects or
modify the main PySide6 application.
"""

import argparse
import sys
import time
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from detection.hand_detector import HandDetector
from detection.gesture_classifier import GestureClassifier
from decision.gesture_engine import GestureEngine
from decision.state_machine import VisibilityState


def parse_args():
    parser = argparse.ArgumentParser(
        description="Live webcam test for Gesture Sense decision engine."
    )
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index.")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Optional path to hand_landmarker.task. Defaults to detector's project path.",
    )
    parser.add_argument(
        "--transition-seconds",
        type=float,
        default=1.0,
        help="Duration used to simulate an effect transition.",
    )
    parser.add_argument(
        "--width", type=int, default=1280, help="Requested camera width."
    )
    parser.add_argument(
        "--height", type=int, default=720, help="Requested camera height."
    )
    return parser.parse_args()


def draw_text(frame, text, position, color=(235, 235, 235), scale=0.62, thickness=2):
    cv2.putText(
        frame,
        str(text),
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def main():
    args = parse_args()

    if args.transition_seconds <= 0:
        raise ValueError("--transition-seconds must be greater than zero.")

    camera = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not camera.isOpened():
        camera.release()
        camera = cv2.VideoCapture(args.camera)

    if not camera.isOpened():
        raise RuntimeError(
            f"Could not open camera index {args.camera}. "
            "Close other camera apps or try --camera 1."
        )

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    detector = None
    window_name = "Gesture Sense - Live Decision Test"
    transition_started_at = None
    last_decision = {"status": "WAITING_FOR_HAND", "visibility_state": "VISIBLE"}

    classifier = GestureClassifier()
    engine = GestureEngine(
        stability_seconds=0.6,
        minimum_confidence=0.60,
        cooldown_seconds=1.0,
    )

    try:
        detector = HandDetector(model_path=args.model)
        print("Gesture Sense live decision test started.")
        print("Press Q/ESC to quit, C to complete transition, R to reset.")

        while True:
            ok, frame = camera.read()
            if not ok or frame is None:
                print("Warning: camera frame could not be read.")
                break

            # Mirror the preview for natural hand movement. Classification
            # uses the mirrored frame consistently throughout this test.
            frame = cv2.flip(frame, 1)

            detection_result = detector.detect(frame)
            classifier_result = classifier.classify(detection_result)
            now = time.monotonic()

            # The decision engine is fed the classifier's native result shape.
            decision = engine.update(classifier_result, now=now)
            last_decision = decision

            # Simulate completion of the visual effect so the next state can
            # be tested. This harness does not render the effect itself.
            if engine.state_machine.is_transitioning:
                if transition_started_at is None:
                    transition_started_at = now
                elif now - transition_started_at >= args.transition_seconds:
                    engine.complete_transition(success=True)
                    transition_started_at = None
                    decision = {
                        "status": "SIMULATED_TRANSITION_COMPLETE",
                        "visibility_state": engine.state_machine.state.value,
                    }
                    last_decision = decision
            else:
                transition_started_at = None

            if detection_result.get("count", 0) > 0:
                display = detector.draw_landmarks(frame, detection_result)
            else:
                display = frame.copy()

            # Show each hand's classification, confidence and finger pattern.
            y = 28
            for index, item in enumerate(classifier_result.get("gestures", [])):
                gesture = item.get("gesture", "UNKNOWN")
                confidence = item.get("confidence", 0.0)
                handedness = item.get("handedness", "Hand")
                states = item.get("finger_states", {})
                state_text = (
                    f"T{states.get('thumb', '-')}"
                    f" I{states.get('index', '-')}"
                    f" M{states.get('middle', '-')}"
                    f" R{states.get('ring', '-')}"
                    f" P{states.get('pinky', '-')}"
                )
                draw_text(
                    display,
                    f"{handedness}: {gesture}  {confidence:.2f}  [{state_text}]",
                    (12, y),
                    color=(80, 240, 120) if gesture != "UNKNOWN" else (170, 170, 170),
                )
                y += 27

            if not classifier_result.get("gestures"):
                draw_text(display, "No hand detected", (12, y), color=(100, 190, 255))

            # Decision status panel.
            state = engine.state_machine.state.value
            status = last_decision.get("status", "WAITING")
            gesture = last_decision.get("gesture", "-")
            confidence = last_decision.get("confidence")
            action = last_decision.get("action", "-")
            effect = last_decision.get("effect", "-")

            panel_y = display.shape[0] - 125
            overlay = display.copy()
            cv2.rectangle(
                overlay,
                (0, panel_y),
                (display.shape[1], display.shape[0]),
                (20, 20, 20),
                -1,
            )
            cv2.addWeighted(overlay, 0.72, display, 0.28, 0, display)

            draw_text(display, f"VISIBILITY: {state}", (12, panel_y + 25))
            draw_text(display, f"DECISION: {status}", (12, panel_y + 52))
            draw_text(
                display,
                f"GESTURE: {gesture}    CONFIDENCE: "
                + (f"{confidence:.2f}" if confidence is not None else "-"),
                (12, panel_y + 79),
            )
            draw_text(display, f"ACTION: {action}    EFFECT: {effect}", (12, panel_y + 106))

            if engine.state_machine.is_transitioning:
                remaining = max(
                    0.0,
                    args.transition_seconds - (now - (transition_started_at or now)),
                )
                draw_text(
                    display,
                    f"SIMULATING EFFECT: {remaining:.1f}s",
                    (display.shape[1] - 280, 30),
                    color=(0, 220, 255),
                )

            draw_text(
                display,
                "Q/ESC: Quit | C: Complete transition | R: Reset",
                (12, display.shape[0] - 8),
                color=(190, 190, 190),
                scale=0.48,
                thickness=1,
            )

            cv2.imshow(window_name, display)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break
            elif key == ord("c"):
                if engine.complete_transition(success=True):
                    transition_started_at = None
                    last_decision = {
                        "status": "MANUALLY_COMPLETED",
                        "visibility_state": engine.state_machine.state.value,
                    }
            elif key == ord("r"):
                engine.state_machine.reset(VisibilityState.VISIBLE)
                engine.reset_candidate()
                engine.cooldown.reset()
                transition_started_at = None
                last_decision = {
                    "status": "RESET",
                    "visibility_state": VisibilityState.VISIBLE.value,
                }

    finally:
        camera.release()
        if detector is not None:
            detector.close()
        cv2.destroyAllWindows()
        print("Gesture Sense live decision test stopped.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise
