from time import monotonic


class CooldownManager:
    """Tracks when a gesture action may be triggered again."""

    def __init__(self, cooldown_seconds=1.0):
        if cooldown_seconds < 0:
            raise ValueError("cooldown_seconds cannot be negative.")
        self.cooldown_seconds = float(cooldown_seconds)
        self._last_triggered_at = {}

    def is_ready(self, key="global", now=None):
        now = monotonic() if now is None else float(now)
        last = self._last_triggered_at.get(key)
        return last is None or now - last >= self.cooldown_seconds

    def trigger(self, key="global", now=None):
        now = monotonic() if now is None else float(now)
        self._last_triggered_at[key] = now

    def remaining(self, key="global", now=None):
        now = monotonic() if now is None else float(now)
        last = self._last_triggered_at.get(key)
        if last is None:
            return 0.0
        return max(0.0, self.cooldown_seconds - (now - last))

    def reset(self, key=None):
        if key is None:
            self._last_triggered_at.clear()
        else:
            self._last_triggered_at.pop(key, None)
