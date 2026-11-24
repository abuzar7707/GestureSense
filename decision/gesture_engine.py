from time import monotonic

from decision.cooldown_manager import CooldownManager
from decision.state_machine import StateMachine


class GestureEngine:
    """
    Converts classifier output into validated actions.

    Expected classifier input:
        {
            "count": int,
            "gestures": [
                {"gesture": "PEACE", "confidence": 0.9, ...}
            ]
        }

    One stable hand/gesture must persist for the configured duration.
    A held gesture triggers once; it must be released or change before
    it can trigger again.
    """

    def __init__(
        self,
        stability_seconds=0.6,
        minimum_confidence=0.60,
        cooldown_seconds=1.0,
        state_machine=None,
        cooldown_manager=None,
    ):
        if stability_seconds < 0:
            raise ValueError("stability_seconds cannot be negative.")
        if not 0.0 <= minimum_confidence <= 1.0:
            raise ValueError("minimum_confidence must be between 0 and 1.")

        self.stability_seconds = float(stability_seconds)
        self.minimum_confidence = float(minimum_confidence)
        self.state_machine = state_machine or StateMachine()
        self.cooldown = cooldown_manager or CooldownManager(cooldown_seconds)

        self._candidate = None
        self._candidate_since = None
        self._latched = False

    def reset_candidate(self):
        self._candidate = None
        self._candidate_since = None
        self._latched = False

    @staticmethod
    def _select_gesture(classifier_result):
        """Select the strongest recognized hand result, if any."""
        gestures = classifier_result.get("gestures", [])
        valid = [
            item for item in gestures
            if item.get("gesture", "UNKNOWN") != "UNKNOWN"
        ]
        if not valid:
            return None
        return max(valid, key=lambda item: float(item.get("confidence", 0.0)))

    def update(self, classifier_result, now=None):
        """
        Process one classifier result and return a decision dictionary.

        `now` is injectable for deterministic tests; production callers
        should omit it.
        """
        now = monotonic() if now is None else float(now)

        if self.state_machine.is_transitioning:
            # Do not clear the latch during an effect. Otherwise a held
            # gesture could be treated as new immediately after completion.
            return self._result("TRANSITION_IN_PROGRESS")

        selected = self._select_gesture(classifier_result)
        if selected is None:
            self.reset_candidate()
            return self._result("NO_GESTURE")

        gesture = str(selected.get("gesture", "UNKNOWN")).upper()
        confidence = float(selected.get("confidence", 0.0))

        if confidence < self.minimum_confidence:
            self.reset_candidate()
            return self._result(
                "LOW_CONFIDENCE", gesture=gesture, confidence=confidence
            )

        if self._candidate != gesture:
            self._candidate = gesture
            self._candidate_since = now
            self._latched = False

        stable_for = max(0.0, now - self._candidate_since)

        if self._latched:
            return self._result(
                "HELD_GESTURE", gesture=gesture, confidence=confidence,
                stable_for=stable_for,
            )

        if stable_for < self.stability_seconds:
            return self._result(
                "STABILIZING", gesture=gesture, confidence=confidence,
                stable_for=stable_for,
            )

        action = self.state_machine.action_for_gesture(gesture)
        if action is None:
            self._latched = True
            return self._result(
                "ACTION_NOT_VALID_FOR_STATE",
                gesture=gesture,
                confidence=confidence,
                stable_for=stable_for,
            )

        cooldown_key = action["action"]
        if not self.cooldown.is_ready(cooldown_key, now=now):
            return self._result(
                "COOLDOWN", gesture=gesture, confidence=confidence,
                stable_for=stable_for,
                cooldown_remaining=self.cooldown.remaining(cooldown_key, now=now),
            )

        started = self.state_machine.begin_transition(
            action["action"], action["effect"]
        )
        if not started:
            return self._result(
                "ACTION_REJECTED", gesture=gesture, confidence=confidence
            )

        self.cooldown.trigger(cooldown_key, now=now)
        self._latched = True

        return self._result(
            "TRIGGERED",
            gesture=gesture,
            confidence=confidence,
            stable_for=stable_for,
            action=action["action"],
            effect=action["effect"],
            state=self.state_machine.state.value,
        )

    def complete_transition(self, success=True):
        """Call when the effects layer finishes its transition."""
        completed = self.state_machine.complete_transition(success=success)
        if completed:
            # Keep the current gesture latched until it is released/changed.
            pass
        return completed

    def _result(self, status, **extra):
        result = {
            "status": status,
            "visibility_state": self.state_machine.state.value,
        }
        result.update(extra)
        return result
