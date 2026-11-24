from enum import Enum


class VisibilityState(str, Enum):
    VISIBLE = "VISIBLE"
    DISAPPEARING = "DISAPPEARING"
    HIDDEN = "HIDDEN"
    APPEARING = "APPEARING"


class StateMachine:
    """Owns visibility state and validates effect transitions."""

    # Effect gestures map to the same effect in either direction.
    GESTURE_EFFECTS = {
        "THUMBS_UP": "DISAPPEAR",
        "PEACE": "REAPPEAR",
        "OPEN_PALM": "FADE",
        "ROCK_SIGN": "DISSOLVE",
    }

    def __init__(self, initial_state=VisibilityState.VISIBLE):
        self.state = VisibilityState(initial_state)
        self.active_effect = None

    @property
    def is_transitioning(self):
        return self.state in (
            VisibilityState.DISAPPEARING,
            VisibilityState.APPEARING,
        )

    def action_for_gesture(self, gesture):
        if self.is_transitioning:
            return None

        effect = self.GESTURE_EFFECTS.get(gesture)
        if effect is None:
            return None

        if self.state == VisibilityState.VISIBLE:
            if gesture == "PEACE":
                return None
            return {
                "action": "DISAPPEAR",
                "effect": effect if effect != "DISAPPEAR" else "INSTANT",
            }

        if self.state == VisibilityState.HIDDEN:
            if gesture == "THUMBS_UP":
                return None
            return {
                "action": "REAPPEAR",
                "effect": effect if effect != "REAPPEAR" else "INSTANT",
            }

        return None

    def begin_transition(self, action, effect):
        if self.is_transitioning:
            return False

        action = str(action).upper()
        if action == "DISAPPEAR" and self.state == VisibilityState.VISIBLE:
            self.state = VisibilityState.DISAPPEARING
        elif action == "REAPPEAR" and self.state == VisibilityState.HIDDEN:
            self.state = VisibilityState.APPEARING
        else:
            return False

        self.active_effect = str(effect).upper()
        return True

    def complete_transition(self, success=True):
        if not self.is_transitioning:
            return False

        if self.state == VisibilityState.DISAPPEARING:
            self.state = (
                VisibilityState.HIDDEN if success else VisibilityState.VISIBLE
            )
        else:
            self.state = (
                VisibilityState.VISIBLE if success else VisibilityState.HIDDEN
            )

        self.active_effect = None
        return True

    def reset(self, state=VisibilityState.VISIBLE):
        self.state = VisibilityState(state)
        self.active_effect = None
