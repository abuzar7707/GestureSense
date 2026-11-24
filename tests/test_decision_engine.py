import sys
from pathlib import Path
import unittest

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    
from decision.gesture_engine import GestureEngine
from decision.state_machine import StateMachine, VisibilityState


def classified(gesture, confidence=0.95):
    if gesture is None:
        return {"count": 0, "gestures": []}
    return {
        "count": 1,
        "gestures": [{"gesture": gesture, "confidence": confidence}],
    }


class GestureEngineTests(unittest.TestCase):
    def setUp(self):
        self.engine = GestureEngine(
            stability_seconds=0.6,
            minimum_confidence=0.6,
            cooldown_seconds=1.0,
        )

    def test_requires_stability_then_triggers(self):
        self.assertEqual(
            self.engine.update(classified("THUMBS_UP"), now=0)["status"],
            "STABILIZING",
        )
        result = self.engine.update(classified("THUMBS_UP"), now=0.61)
        self.assertEqual(result["status"], "TRIGGERED")
        self.assertEqual(result["action"], "DISAPPEAR")
        self.assertEqual(result["visibility_state"], "DISAPPEARING")

    def test_held_gesture_does_not_repeat(self):
        self.engine.update(classified("THUMBS_UP"), now=0)
        self.engine.update(classified("THUMBS_UP"), now=0.61)
        self.engine.complete_transition()
        result = self.engine.update(classified("THUMBS_UP"), now=2.0)
        self.assertEqual(result["status"], "HELD_GESTURE")

    def test_release_allows_new_gesture(self):
        self.engine.update(classified("THUMBS_UP"), now=0)
        self.engine.update(classified("THUMBS_UP"), now=0.61)
        self.engine.complete_transition()
        self.engine.update(classified(None), now=0.8)
        result = self.engine.update(classified("PEACE"), now=1.0)
        self.assertEqual(result["status"], "STABILIZING")

    def test_low_confidence_is_rejected(self):
        result = self.engine.update(classified("PEACE", 0.4), now=0)
        self.assertEqual(result["status"], "LOW_CONFIDENCE")

    def test_unknown_gesture_is_ignored(self):
        result = self.engine.update(classified(None), now=0)
        self.assertEqual(result["status"], "NO_GESTURE")

    def test_transition_blocks_new_input(self):
        self.engine.update(classified("THUMBS_UP"), now=0)
        self.engine.update(classified("THUMBS_UP"), now=0.61)
        result = self.engine.update(classified("PEACE"), now=0.7)
        self.assertEqual(result["status"], "TRANSITION_IN_PROGRESS")

    def test_peace_reappears_only_when_hidden(self):
        machine = StateMachine(VisibilityState.HIDDEN)
        engine = GestureEngine(state_machine=machine, stability_seconds=0)
        result = engine.update(classified("PEACE"), now=0)
        self.assertEqual(result["status"], "TRIGGERED")
        self.assertEqual(result["action"], "REAPPEAR")

    def test_held_gesture_does_not_reverse_after_transition(self):
        engine = GestureEngine(stability_seconds=0.0)
        first = engine.update(classified("OPEN_PALM"), now=0.0)
        self.assertEqual(first["status"], "TRIGGERED")
        engine.complete_transition(success=True)
        after = engine.update(classified("OPEN_PALM"), now=2.0)
        self.assertEqual(after["status"], "HELD_GESTURE")
        self.assertEqual(after["visibility_state"], "HIDDEN")


if __name__ == "__main__":
    unittest.main()
