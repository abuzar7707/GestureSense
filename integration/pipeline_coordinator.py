"""Coordinates independent pipelines and feeds the existing GestureEngine.

Call update() from the application/UI thread. It does not own or complete visual
transitions; the effects layer must call engine.complete_transition(success=...).
"""
from __future__ import annotations

import time


class PipelineCoordinator:
    def __init__(
        self,
        camera,
        gesture_pipeline,
        segmentation_pipeline,
        gesture_engine,
        gesture_result_ttl=0.40,
    ):
        self.camera = camera
        self.gesture_pipeline = gesture_pipeline
        self.segmentation_pipeline = segmentation_pipeline
        self.gesture_engine = gesture_engine
        self.gesture_result_ttl = float(gesture_result_ttl)
        self._last_decision = {
            "status": "WAITING_FOR_GESTURE_PIPELINE",
            "visibility_state": self.gesture_engine.state_machine.state.value,
        }

    def start(self):
        self.camera.start()
        self.gesture_pipeline.start()
        self.segmentation_pipeline.start()
        return self

    def update(self, now=None, copy_visual=True):
        """Return the latest camera, gesture, segmentation and decision snapshots."""
        now = time.monotonic() if now is None else float(now)
        gesture_snapshot = self.gesture_pipeline.latest()

        if gesture_snapshot is None:
            classifier_result = {"count": 0, "gestures": []}
            gesture_age = None
            gesture_fresh = False
        else:
            gesture_age = max(0.0, now - gesture_snapshot["processed_at"])
            gesture_fresh = gesture_age <= self.gesture_result_ttl
            classifier_result = (
                gesture_snapshot["classifier_result"]
                if gesture_fresh else {"count": 0, "gestures": []}
            )

        # Re-reading the latest fresh classification is intentional: the
        # existing engine measures gesture stability across update() calls.
        self._last_decision = self.gesture_engine.update(
            classifier_result, now=now
        )

        visual = self.segmentation_pipeline.latest(copy_arrays=copy_visual)
        visual_age = (
            max(0.0, now - visual["processed_at"]) if visual is not None else None
        )
        camera_frame, camera_sequence, camera_captured_at = self.camera.latest()

        errors = {
            "gesture": repr(self.gesture_pipeline.error)
            if self.gesture_pipeline.error else None,
            "segmentation": repr(self.segmentation_pipeline.error)
            if self.segmentation_pipeline.error else None,
        }
        return {
            "camera": {
                "frame": camera_frame,
                "sequence": camera_sequence,
                "captured_at": camera_captured_at,
                "read_errors": self.camera.read_errors,
            },
            "gesture": {
                "snapshot": gesture_snapshot,
                "fresh": gesture_fresh,
                "age_seconds": gesture_age,
            },
            "visual": {
                "snapshot": visual,
                "age_seconds": visual_age,
            },
            "decision": dict(self._last_decision),
            "errors": errors,
        }

    def stop(self):
        # Stop consumers before releasing their shared camera source.
        self.gesture_pipeline.stop()
        self.segmentation_pipeline.stop()
        self.camera.stop()

    close = stop
