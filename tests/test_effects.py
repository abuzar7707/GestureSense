"""Unit tests for Gesture Sense effects renderer and background plate.

Run from project root:
    python -m unittest discover -s tests -p "test_effects.py" -v

Direct execution is also supported:
    python tests/test_effects.py
"""
import sys
from pathlib import Path

# Ensure the project root is importable even when this file is executed
# directly, in which case Python normally places only the tests directory
# at sys.path[0].
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import unittest
import numpy as np

from decision.state_machine import StateMachine, VisibilityState
from effects.background_plate import BackgroundPlate
from effects.effect_renderer import EffectsRenderer


class EffectsTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.full((80, 100, 3), 220, dtype=np.uint8)
        self.plate = np.full((80, 100, 3), 20, dtype=np.uint8)
        self.mask = np.ones((80, 100), dtype=np.float32)

    def test_visible_state_shows_camera_frame(self):
        sm = StateMachine(VisibilityState.VISIBLE)
        renderer = EffectsRenderer(edge_blur=0)
        result = renderer.render(self.frame, self.mask, self.plate, sm, now=1.0)
        self.assertTrue(np.all(result.frame == self.frame))

    def test_hidden_state_shows_background_plate(self):
        sm = StateMachine(VisibilityState.HIDDEN)
        renderer = EffectsRenderer(edge_blur=0)
        result = renderer.render(self.frame, self.mask, self.plate, sm, now=1.0)
        self.assertTrue(np.all(result.frame == self.plate))

    def test_fade_disappear_reaches_background(self):
        sm = StateMachine(VisibilityState.VISIBLE)
        sm.begin_transition("DISAPPEAR", "FADE")
        renderer = EffectsRenderer(fade_duration=1.0, edge_blur=0)
        # First render initializes the transition clock; the second render
        # occurs after the configured 1-second fade duration.
        renderer.render(self.frame, self.mask, self.plate, sm, now=1.0)
        result = renderer.render(self.frame, self.mask, self.plate, sm, now=2.0)
        self.assertTrue(result.transition_complete)
        self.assertTrue(np.all(result.frame == self.plate))

    def test_fade_reappear_reaches_camera(self):
        sm = StateMachine(VisibilityState.HIDDEN)
        sm.begin_transition("REAPPEAR", "FADE")
        renderer = EffectsRenderer(fade_duration=1.0, edge_blur=0)
        # First render initializes the transition clock; the second render
        # occurs after the configured 1-second fade duration.
        renderer.render(self.frame, self.mask, self.plate, sm, now=1.0)
        result = renderer.render(self.frame, self.mask, self.plate, sm, now=2.0)
        self.assertTrue(result.transition_complete)
        self.assertTrue(np.all(result.frame == self.frame))

    def test_background_plate_rejects_foreground(self):
        manager = BackgroundPlate(maximum_foreground_ratio=0.02)
        mask = np.ones((80, 100), dtype=np.float32)
        result = manager.capture(self.frame, mask)
        self.assertFalse(result["success"])

    def test_background_plate_accepts_empty_frame(self):
        manager = BackgroundPlate(maximum_foreground_ratio=0.02)
        mask = np.zeros((80, 100), dtype=np.float32)
        result = manager.capture(self.frame, mask)
        self.assertTrue(result["success"])
        self.assertTrue(manager.is_ready)


if __name__ == "__main__":
    unittest.main()
