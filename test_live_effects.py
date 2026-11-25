"""Standalone live camera + Gesture Sense effects diagnostic.

Run from the GestureSense project root:
    python test_live_effects.py

Controls:
    B - capture a clean background plate (everyone must be out of frame)
    R - reset the background plate and visibility state
    Q / ESC - quit

Gestures (after a plate is captured):
    THUMBS_UP - instant disappear
    OPEN_PALM - fade out / fade in
    ROCK_SIGN - dissolve out / dissolve in
    PEACE - instant reappear
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from camera.latest_frame_camera import LatestFrameCamera
from detection.gesture_pipeline import GesturePipeline
from vision.segmentation_pipeline import SegmentationPipeline
from integration.pipeline_coordinator import PipelineCoordinator
from decision.gesture_engine import GestureEngine
from effects.background_plate import BackgroundPlate
from effects.effect_renderer import EffectsRenderer


WINDOW_NAME = "Gesture Sense - Live Effects Diagnostic"
CAMERA_INDEX = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 30


def draw_label(frame, text, y, color=(255, 255, 255), scale=0.65):
    cv2.putText(
        frame, text, (18, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
        (0, 0, 0), 4, cv2.LINE_AA,
    )
    cv2.putText(
        frame, text, (18, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
        color, 1, cv2.LINE_AA,
    )


def main():
    hand_model = PROJECT_ROOT / "assets" / "models" / "hand_landmarker.task"
    segmentation_model = (
        PROJECT_ROOT / "assets" / "models" / "selfie_segmenter_landscape.tflite"
    )

    missing = [str(path) for path in (hand_model, segmentation_model) if not path.is_file()]
    if missing:
        print("Required model file(s) not found:")
        for path in missing:
            print(f"  - {path}")
        print("Check the model paths and keep this script in the project root.")
        return 1

    camera = None
    coordinator = None
    plate_manager = BackgroundPlate()
    renderer = EffectsRenderer()
    last_completed_transition = None
    fps_started = time.perf_counter()
    fps_frames = 0
    displayed_fps = 0.0

    try:
        camera = LatestFrameCamera(
            index=CAMERA_INDEX,
            width=CAMERA_WIDTH,
            height=CAMERA_HEIGHT,
            fps=CAMERA_FPS,
            backend=cv2.CAP_MSMF,
        )

        gesture_pipeline = GesturePipeline(
            camera=camera,
            model_path=hand_model,
            processing_width=640,
            max_fps=30,
        )
        segmentation_pipeline = SegmentationPipeline(
            camera=camera,
            model_path=segmentation_model,
            max_fps=30,
        )
        coordinator = PipelineCoordinator(
            camera=camera,
            gesture_pipeline=gesture_pipeline,
            segmentation_pipeline=segmentation_pipeline,
            gesture_engine=GestureEngine(),
            gesture_result_ttl=0.40,
        )
        coordinator.start()

        print("Gesture Sense live effects diagnostic started.")
        print("Step out of frame, then press B to capture the clean background.")
        print("Gestures: THUMBS_UP=instant out, OPEN_PALM=fade, ROCK_SIGN=dissolve, PEACE=instant in.")
        print("Press R to reset; Q or ESC to quit.")

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

        while True:
            snapshot = coordinator.update()
            camera_frame = snapshot["camera"]["frame"]
            visual = snapshot["visual"]["snapshot"]
            decision = snapshot["decision"]
            gesture_snapshot = snapshot["gesture"]["snapshot"]

            if camera_frame is None or visual is None:
                waiting = camera_frame.copy() if camera_frame is not None else None
                if waiting is not None:
                    draw_label(waiting, "Waiting for segmentation pipeline...", 35, (0, 220, 255))
                    cv2.imshow(WINDOW_NAME, waiting)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                continue

            mask = visual["mask"]
            preview = camera_frame
            rendered = None

            if plate_manager.is_ready:
                rendered = renderer.render(
                    frame=camera_frame,
                    person_mask=mask,
                    background_plate=plate_manager.get(),
                    state_machine=coordinator.gesture_engine.state_machine,
                )
                preview = rendered.frame

                if rendered.transition_complete:
                    transition_key = (
                        coordinator.gesture_engine.state_machine.state.value,
                        rendered.action,
                        rendered.effect,
                    )
                    if transition_key != last_completed_transition:
                        coordinator.gesture_engine.complete_transition(success=True)
                        last_completed_transition = transition_key
                else:
                    last_completed_transition = None

            output = preview.copy()
            state = coordinator.gesture_engine.state_machine.state.value
            effect = coordinator.gesture_engine.state_machine.active_effect or "-"
            gesture_name = "NO HAND"
            gesture_confidence = 0.0
            if gesture_snapshot:
                items = gesture_snapshot["classifier_result"].get("gestures", [])
                if items:
                    strongest = max(items, key=lambda item: float(item.get("confidence", 0.0)))
                    gesture_name = strongest.get("gesture", "UNKNOWN")
                    gesture_confidence = float(strongest.get("confidence", 0.0))

            fps_frames += 1
            elapsed = time.perf_counter() - fps_started
            if elapsed >= 1.0:
                displayed_fps = fps_frames / elapsed
                fps_frames = 0
                fps_started = time.perf_counter()

            if plate_manager.is_ready:
                plate_text = f"BACKGROUND: READY ({plate_manager.captured_foreground_ratio:.1%} foreground at capture)"
                plate_color = (80, 230, 100)
            else:
                plate_text = "BACKGROUND: NOT CAPTURED - step fully out and press B"
                plate_color = (0, 190, 255)

            draw_label(output, plate_text, 30, plate_color, 0.58)
            draw_label(output, f"Gesture: {gesture_name}  Confidence: {gesture_confidence:.2f}", 62)
            draw_label(output, f"Decision: {decision.get('status', '-')}  State: {state}  Effect: {effect}", 94)
            draw_label(output, f"FPS: {displayed_fps:.1f}  Segmentation: {visual.get('processing_ms', 0):.1f} ms", 126)
            draw_label(output, "B: Capture background   R: Reset   Q: Quit", output.shape[0] - 20, (220, 220, 220), 0.55)

            errors = snapshot.get("errors", {})
            active_errors = [f"{name}: {value}" for name, value in errors.items() if value]
            if active_errors:
                draw_label(output, "PIPELINE ERROR - see terminal", 160, (0, 0, 255))
                for error in active_errors:
                    print(error)

            cv2.imshow(WINDOW_NAME, output)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break
            elif key in (ord("b"), ord("B")):
                capture = plate_manager.capture(camera_frame, mask)
                print(
                    f"Background capture: {'SUCCESS' if capture['success'] else 'REJECTED'} | "
                    f"foreground={capture['foreground_ratio']:.2%} | {capture['message']}"
                )
                if capture["success"]:
                    renderer.reset()
                    coordinator.gesture_engine.state_machine.reset("VISIBLE")
                    coordinator.gesture_engine.reset_candidate()
                    last_completed_transition = None
            elif key in (ord("r"), ord("R")):
                plate_manager.clear()
                renderer.reset()
                coordinator.gesture_engine.state_machine.reset("VISIBLE")
                coordinator.gesture_engine.reset_candidate()
                last_completed_transition = None
                print("Background plate and visibility state reset.")

        return 0

    except Exception as exc:
        print(f"Diagnostic failed: {type(exc).__name__}: {exc}")
        return 1
    finally:
        if coordinator is not None:
            coordinator.stop()
        elif camera is not None:
            camera.stop()
        cv2.destroyAllWindows()
        print("Gesture Sense diagnostic stopped.")


if __name__ == "__main__":
    raise SystemExit(main())
