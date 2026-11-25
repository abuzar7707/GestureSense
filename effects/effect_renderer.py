"""Gesture Sense visual effects renderer.

Composites the live person mask over a captured clean background plate.
This module does not own the GestureEngine; the caller completes transitions
when RenderResult.transition_complete becomes True.

Changes vs. previous version:
  * Only the PERSON region is replaced by the plate. Everything outside the
    mask now stays live (before, the whole frame was swapped for the static
    plate whenever a transition/hidden state was active).
  * mask_expand: hidden region is dilated a few px so no body outline remains.
  * match_exposure: plate gain is matched to the live background (EMA-smoothed)
    so webcam auto-exposure drift doesn't leave a visible patch.
"""
from __future__ import annotations

from dataclasses import dataclass
import time

import cv2
import numpy as np

from decision.state_machine import VisibilityState


@dataclass
class RenderResult:
    frame: np.ndarray
    transition_complete: bool
    progress: float
    effect: str
    action: str
    status: str


class EffectsRenderer:
    def __init__(
        self,
        instant_duration=0.05,
        fade_duration=0.75,
        dissolve_duration=1.10,
        edge_blur=3,
        seed=17,
        mask_expand=10,
        match_exposure=True,
        exposure_smoothing=0.15,
        hold_decay=0.8,
        diff_threshold=28,
    ):
        self.durations = {
            "INSTANT": max(0.0, float(instant_duration)),
            "FADE": max(0.01, float(fade_duration)),
            "DISSOLVE": max(0.01, float(dissolve_duration)),
            "REAPPEAR": max(0.0, float(instant_duration)),
        }
        self.edge_blur = max(0, int(edge_blur))
        self.seed = int(seed)
        self.mask_expand = max(0, int(mask_expand))
        self.match_exposure = bool(match_exposure)
        self.exposure_smoothing = float(np.clip(exposure_smoothing, 0.01, 1.0))
        self.hold_decay = float(np.clip(hold_decay, 0.0, 0.98))
        self.diff_threshold = float(diff_threshold)
        self._hold = None
        self._transition_key = None
        self._transition_started = None
        self._noise = None
        self._noise_shape = None
        self._gain = np.ones(3, dtype=np.float32)

    def reset(self):
        self._transition_key = None
        self._transition_started = None
        self._noise = None
        self._noise_shape = None
        self._gain = np.ones(3, dtype=np.float32)
        self._hold = None

    def _get_noise(self, height, width):
        shape = (height, width)
        if self._noise is None or self._noise_shape != shape:
            rng = np.random.default_rng(self.seed)
            self._noise = rng.random(shape, dtype=np.float32)
            self._noise_shape = shape
        return self._noise

    @staticmethod
    def _validate_inputs(frame, person_mask, background_plate):
        if frame is None or not isinstance(frame, np.ndarray):
            raise ValueError("frame must be a valid NumPy image.")
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("frame must be a 3-channel BGR image.")
        if person_mask is None:
            raise ValueError("person_mask cannot be None.")
        if background_plate is None:
            raise ValueError("A clean background plate must be captured first.")
        if background_plate.shape != frame.shape:
            raise ValueError(
                "Background plate and camera frame must have identical dimensions."
            )

        mask = np.asarray(person_mask, dtype=np.float32)
        if mask.ndim == 3:
            mask = np.squeeze(mask)
        if mask.shape != frame.shape[:2]:
            mask = cv2.resize(
                mask, (frame.shape[1], frame.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
        return np.clip(mask, 0.0, 1.0)

    def _transition_progress(self, state_machine, now):
        state = state_machine.state
        effect = str(state_machine.active_effect or "").upper()

        if state not in (
            VisibilityState.DISAPPEARING,
            VisibilityState.APPEARING,
        ):
            self._transition_key = None
            self._transition_started = None
            return None

        action = (
            "DISAPPEAR"
            if state == VisibilityState.DISAPPEARING
            else "REAPPEAR"
        )
        key = (state.value, effect)
        if key != self._transition_key:
            self._transition_key = key
            self._transition_started = now

        duration = self.durations.get(effect, self.durations["FADE"])
        if duration <= 0:
            progress = 1.0
        else:
            progress = min(1.0, max(0.0, (now - self._transition_started) / duration))
        return action, effect, progress

    def _matched_plate(self, frame, mask, plate):
        """Scale the plate per channel to match the live background brightness."""
        if not self.match_exposure:
            return plate
        # Subsample for speed; use only pixels clearly outside the person.
        step = 8
        sub_mask = mask[::step, ::step] < 0.02
        if np.count_nonzero(sub_mask) >= 0.05 * sub_mask.size:
            live = frame[::step, ::step][sub_mask].mean(axis=0)
            ref = plate[::step, ::step][sub_mask].mean(axis=0)
            target = np.clip(live / np.maximum(ref, 1.0), 0.7, 1.4).astype(np.float32)
            a = self.exposure_smoothing
            self._gain = (1.0 - a) * self._gain + a * target
        g = self._gain
        if np.all(np.abs(g - 1.0) < 0.01):
            return plate
        return cv2.multiply(
            plate, (float(g[0]), float(g[1]), float(g[2]), 0.0), dtype=cv2.CV_8U
        )

    def _cover_mask(self, frame, mask, plate):
        """Region to replace with the plate, robust to a late/incomplete mask.

        = dilated mask, held briefly over time (covers motion while the mask
        lags the live frame), plus any pixels near the person that differ
        strongly from the plate (background subtraction catches limbs/hair the
        segmenter missed).
        """
        if self._hold is None or self._hold.shape != mask.shape:
            self._hold = mask.copy()
        else:
            self._hold = np.maximum(mask, self._hold * self.hold_decay)
        cover = self._hold
        if self.mask_expand > 0:
            k = 2 * self.mask_expand + 1
            cover = cv2.dilate(cover, cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (k, k)))

        if self.diff_threshold > 0:
            h, w = mask.shape
            sw, sh = max(1, w // 4), max(1, h // 4)
            f = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_AREA)
            p = cv2.resize(plate, (sw, sh), interpolation=cv2.INTER_AREA)
            diff = cv2.absdiff(f, p).max(axis=2)
            diff = cv2.GaussianBlur(diff, (5, 5), 0)
            changed = (diff > self.diff_threshold).astype(np.float32)
            near = cv2.resize(self._hold, (sw, sh), interpolation=cv2.INTER_AREA)
            near = cv2.dilate((near > 0.1).astype(np.float32), np.ones((25, 25), np.uint8))
            # Keep every changed blob that touches the person's neighbourhood
            # (whole arm / hair, however far it extends).
            changed_u8 = cv2.dilate(changed, np.ones((3, 3), np.uint8)).astype(np.uint8)
            _, labels = cv2.connectedComponents(changed_u8, connectivity=8)
            touching = np.unique(labels[(near > 0) & (changed_u8 > 0)])
            touching = touching[touching > 0]
            extra = np.isin(labels, touching).astype(np.float32)
            extra = cv2.dilate(extra, np.ones((5, 5), np.uint8))
            extra = cv2.resize(extra, (w, h), interpolation=cv2.INTER_LINEAR)
            cover = np.maximum(cover, extra)
        return np.clip(cover, 0.0, 1.0)

    def render(self, frame, person_mask, background_plate, state_machine, now=None):
        """Render current visibility state and active transition."""
        now = time.monotonic() if now is None else float(now)
        mask = self._validate_inputs(frame, person_mask, background_plate)

        state = state_machine.state
        transition = self._transition_progress(state_machine, now)

        # `shown` = how visible the person is (scalar or per-pixel array, 0..1).
        if transition is None:
            shown = 0.0 if state == VisibilityState.HIDDEN else 1.0
            action = "NONE"
            effect = "NONE"
            progress = 1.0
            complete = False
            status = state.value
        else:
            action, effect, progress = transition
            complete = progress >= 1.0
            status = "TRANSITION_COMPLETE" if complete else "TRANSITIONING"

            if effect in ("INSTANT", "REAPPEAR"):
                shown = 0.0 if action == "DISAPPEAR" else 1.0
            elif effect == "DISSOLVE":
                noise = self._get_noise(mask.shape[0], mask.shape[1])
                if action == "DISAPPEAR":
                    shown = (noise >= progress).astype(np.float32)
                else:
                    shown = (noise <= progress).astype(np.float32)
            else:  # FADE and unknown effects
                shown = 1.0 - progress if action == "DISAPPEAR" else progress

        # Fast path: fully visible -> the live frame is the output.
        if np.isscalar(shown) and shown >= 1.0:
            return RenderResult(
                frame=frame, transition_complete=complete, progress=float(progress),
                effect=effect, action=action, status=status,
            )

        # Region to cover with the plate: person mask (slightly expanded) * (1 - shown).
        plate = self._matched_plate(frame, mask, background_plate)
        cover_mask = self._cover_mask(frame, mask, plate)
        hidden = cover_mask * (1.0 - shown)

        if self.edge_blur > 1:
            k = self.edge_blur if self.edge_blur % 2 else self.edge_blur + 1
            hidden = cv2.GaussianBlur(hidden, (k, k), 0)

        # blendLinear weights: frame gets (1 - hidden), plate gets hidden.
        hidden = np.ascontiguousarray(np.clip(hidden, 0.0, 1.0), dtype=np.float32)
        keep = np.ascontiguousarray(1.0 - hidden, dtype=np.float32)
        output = cv2.blendLinear(frame, plate, keep, hidden)

        return RenderResult(
            frame=output,
            transition_complete=complete,
            progress=float(progress),
            effect=effect,
            action=action,
            status=status,
        )