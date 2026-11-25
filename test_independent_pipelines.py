"""Gesture Sense - Milestone 4 independent pipeline diagnostic.

Run from the project root:
    python test_independent_pipelines.py

Optional:
    python test_independent_pipelines.py --camera 1
    python test_independent_pipelines.py --seconds 60
    python test_independent_pipelines.py --hand-width 640

Controls:
    Q / ESC : quit
    R       : reset the decision engine to VISIBLE
    C       : manually complete an active simulated transition

This diagnostic does not modify the main app or existing project modules.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from camera.latest_frame_camera import LatestFrameCamera
from detection.gesture_pipeline import GesturePipeline
from vision.segmentation_pipeline import SegmentationPipeline
from integration.pipeline_coordinator import PipelineCoordinator
from decision.gesture_engine import GestureEngine
from decision.state_machine import VisibilityState


def parse_args():
    parser = argparse.ArgumentParser(
        description="Live test for Gesture Sense independent processing pipelines."
    )
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--hand-width", type=int, default=640)
    parser.add_argument("--gesture-max-fps", type=float, default=30.0)
    parser.add_argument("--segmentation-max-fps", type=float, default=30.0)
    parser.add_argument("--gesture-ttl", type=float, default=0.40)
    parser.add_argument("--seconds", type=float, default=0.0,
                        help="Auto-stop after N seconds; 0 means run until Q.")
    parser.add_argument(
        "--hand-model",
        type=Path,
        default=ROOT / "assets" / "models" / "hand_landmarker.task",
    )
    parser.add_argument(
        "--segmentation-model",
        type=Path,
        default=ROOT / "assets" / "models" / "selfie_segmenter_landscape.tflite",
    )
    return parser.parse_args()


def draw_line(frame, text, y, color=(235, 235, 235), scale=0.62):
    cv2.putText(
        frame, str(text), (12, y), cv2.FONT_HERSHEY_SIMPLEX,
        scale, color, 2, cv2.LINE_AA
    )


def main():
    args = parse_args()

    if not args.hand_model.is_file():
        raise FileNotFoundError(f"Hand model not found: {args.hand_model}")
    if not args.segmentation_model.is_file():
        raise FileNotFoundError(
            f"Segmentation model not found: {args.segmentation_model}"
        )

    camera = LatestFrameCamera(
        index=args.camera,
        width=args.width,
        height=args.height,
        fps=int(args.fps),
    )
    gesture_pipeline = GesturePipeline(
        camera=camera,
        model_path=args.hand_model,
        processing_width=args.hand_width,
        max_fps=args.gesture_max_fps,
        mirror=True,
    )
    segmentation_pipeline = SegmentationPipeline(
        camera=camera,
        model_path=args.segmentation_model,
        max_fps=args.segmentation_max_fps,
        blur_kernel=5,
        background_value=32,
    )
    engine = GestureEngine(
        stability_seconds=0.6,
        minimum_confidence=0.60,
        cooldown_seconds=1.0,
    )
    coordinator = PipelineCoordinator(
        camera=camera,
        gesture_pipeline=gesture_pipeline,
        segmentation_pipeline=segmentation_pipeline,
        gesture_engine=engine,
        gesture_result_ttl=args.gesture_ttl,
    )

    started = time.perf_counter()
    last_report = started
    last_transition_start = None
    simulated_transition_seconds = 1.0
    loops = 0
    decision_counts = {}
    last_gesture_sequence = -1
    gesture_updates = 0
    segmentation_updates = 0
    last_seg_sequence = -1
    last_snapshot = None

    print("\nGESTURE SENSE — MILESTONE 4 PIPELINE TEST")
    print("=" * 64)
    print(f"Camera requested       : {args.width}x{args.height} @ {args.fps:.0f} FPS")
    print(f"Hand detection width   : {args.hand_width}px")
    print(f"Gesture result TTL     : {args.gesture_ttl:.2f}s")
    print(f"Hand model             : {args.hand_model}")
    print(f"Segmentation model     : {args.segmentation_model}")
    print("\nStarting workers...")

    try:
        coordinator.start()
        print(f"Camera actual mode     : {camera.width}x{camera.height}")
        print("Workers started. Press Q/ESC to quit.")
        print("Keep your hand visible and try the configured gestures.")

        while True:
            snapshot = coordinator.update()
            last_snapshot = snapshot
            loops += 1
            now = time.perf_counter()

            gesture_snapshot = snapshot["gesture"]["snapshot"]
            visual_snapshot = snapshot["visual"]["snapshot"]
            decision = snapshot["decision"]
            status = decision.get("status", "UNKNOWN")
            decision_counts[status] = decision_counts.get(status, 0) + 1

            if gesture_snapshot is not None:
                seq = gesture_snapshot["camera_sequence"]
                if seq != last_gesture_sequence:
                    gesture_updates += 1
                    last_gesture_sequence = seq

            if visual_snapshot is not None:
                seq = visual_snapshot["camera_sequence"]
                if seq != last_seg_sequence:
                    segmentation_updates += 1
                    last_seg_sequence = seq

            # The diagnostic simulates the effects layer finishing a transition.
            if engine.state_machine.is_transitioning:
                if last_transition_start is None:
                    last_transition_start = now
                elif now - last_transition_start >= simulated_transition_seconds:
                    engine.complete_transition(success=True)
                    last_transition_start = None
            else:
                last_transition_start = None

            # Use the latest segmentation composite; fall back to camera frame
            # while the segmentation worker is warming up.
            if visual_snapshot is not None:
                display = visual_snapshot["composite"]
            else:
                display = snapshot["camera"]["frame"]

            if display is None:
                time.sleep(0.005)
                continue

            display = display.copy()
            h = display.shape[0]
            gesture_age = snapshot["gesture"]["age_seconds"]
            visual_age = snapshot["visual"]["age_seconds"]
            gesture_age_text = (
                f"{gesture_age * 1000:.0f} ms" if gesture_age is not None else "waiting"
            )
            visual_age_text = (
                f"{visual_age * 1000:.0f} ms" if visual_age is not None else "waiting"
            )

            draw_line(display, "GESTURE SENSE | INDEPENDENT PIPELINES", 28)
            draw_line(
                display,
                f"Visibility: {engine.state_machine.state.value} | Decision: {status}",
                58,
                color=(80, 230, 120) if status == "TRIGGERED" else (235, 235, 235),
            )

            if gesture_snapshot:
                classified = gesture_snapshot["classifier_result"]
                items = classified.get("gestures", [])
                if items:
                    descriptions = [
                        f"{x.get('handedness', 'Hand')} {x.get('gesture', 'UNKNOWN')} "
                        f"{float(x.get('confidence', 0.0)):.2f}"
                        for x in items
                    ]
                    gesture_text = " | ".join(descriptions)
                else:
                    gesture_text = "No recognized hand gesture"
            else:
                gesture_text = "Gesture pipeline warming up..."

            draw_line(display, gesture_text[:110], 88)
            draw_line(
                display,
                f"Gesture age: {gesture_age_text} | Fresh: {snapshot['gesture']['fresh']}",
                118,
            )
            draw_line(display, f"Segmentation age: {visual_age_text}", 148)
            draw_line(
                display,
                f"Gesture worker: {gesture_pipeline.processed_frames} frames | "
                f"{gesture_pipeline.last_processing_ms:.1f} ms",
                178,
            )
            draw_line(
                display,
                f"Segmentation worker: {segmentation_pipeline.processed_frames} frames | "
                f"{segmentation_pipeline.last_processing_ms:.1f} ms",
                208,
            )
            draw_line(
                display,
                f"Camera sequence: {snapshot['camera']['sequence']} | "
                f"Read errors: {snapshot['camera']['read_errors']}",
                238,
            )

            if snapshot["errors"]["gesture"]:
                draw_line(display, "GESTURE ERROR: " + snapshot["errors"]["gesture"][:90],
                          268, color=(0, 0, 255), scale=0.48)
            if snapshot["errors"]["segmentation"]:
                draw_line(display, "SEGMENTATION ERROR: " +
                          snapshot["errors"]["segmentation"][:90],
                          296, color=(0, 0, 255), scale=0.48)

            draw_line(
                display,
                "Q/ESC: Quit | R: Reset state | C: Complete transition",
                h - 14,
                color=(190, 190, 190),
                scale=0.48,
            )

            cv2.imshow("Gesture Sense - Independent Pipelines", display)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break
            elif key == ord("c"):
                if engine.complete_transition(success=True):
                    last_transition_start = None
            elif key == ord("r"):
                engine.state_machine.reset(VisibilityState.VISIBLE)
                engine.reset_candidate()
                engine.cooldown.reset()
                last_transition_start = None

            if args.seconds > 0 and now - started >= args.seconds:
                break

            if now - last_report >= 2.0:
                elapsed = now - started
                print(
                    f"\rElapsed {elapsed:5.1f}s | loops {loops:5d} | "
                    f"gesture frames {gesture_pipeline.processed_frames:5d} | "
                    f"seg frames {segmentation_pipeline.processed_frames:5d} | "
                    f"gesture {gesture_pipeline.last_processing_ms:5.1f}ms | "
                    f"seg {segmentation_pipeline.last_processing_ms:5.1f}ms",
                    end="", flush=True,
                )
                last_report = now

    finally:
        coordinator.stop()
        cv2.destroyAllWindows()

    elapsed = max(0.001, time.perf_counter() - started)
    print("\n\n" + "=" * 64)
    print("FINAL DIAGNOSTIC SUMMARY")
    print("=" * 64)
    print(f"Elapsed time                 : {elapsed:.2f}s")
    print(f"Camera mode                  : {camera.width}x{camera.height}")
    print(f"Camera frames captured       : {camera._sequence}")
    print(f"Camera read errors           : {camera.read_errors}")
    print(f"Gesture frames processed     : {gesture_pipeline.processed_frames}")
    print(f"Segmentation frames processed: {segmentation_pipeline.processed_frames}")
    print(f"Gesture processing rate      : {gesture_pipeline.processed_frames / elapsed:.2f} FPS")
    print(f"Segmentation processing rate : {segmentation_pipeline.processed_frames / elapsed:.2f} FPS")
    print(f"Coordinator update rate      : {loops / elapsed:.2f} FPS")
    print(f"Unique gesture results seen  : {gesture_updates}")
    print(f"Unique segmentation results  : {segmentation_updates}")
    print(f"Decision status counts      : {decision_counts}")
    print(f"Gesture worker error         : {gesture_pipeline.error}")
    print(f"Segmentation worker error    : {segmentation_pipeline.error}")

    if gesture_pipeline.processed_frames == 0:
        print("\nRESULT: FAIL — gesture pipeline processed no frames.")
    elif segmentation_pipeline.processed_frames == 0:
        print("\nRESULT: FAIL — segmentation pipeline processed no frames.")
    elif gesture_pipeline.error or segmentation_pipeline.error:
        print("\nRESULT: FAIL — one or more worker errors occurred.")
    else:
        print("\nRESULT: BOTH PIPELINES PROCESSED FRAMES.")
        print("Review the FPS, latency, and live preview before approving this milestone.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nTest interrupted by user.")
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise
