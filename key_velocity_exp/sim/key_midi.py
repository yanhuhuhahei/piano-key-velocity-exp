"""MIDI event generation for the simulated keyboard (MuJoCo / dm_control), called every physics substep.

Two models can be selected:

* ``measured`` - the two-switch mechanism identified on the real piano (``TwoSwitchVelocityModel`` +
  ``TwoSwitchTracker``): a timer starts when the key passes depth A, note-on with
  ``velocity = f(dt)`` is sent when it passes depth B, note-off when the key rises back above A.
* ``legacy`` - the original RoboPianist rule, kept as the baseline: note-on when the key crosses the
  activation threshold (0.5 deg from the bottom), ``velocity = qvel * 127 / 3.5``, note-off when the
  key crosses the threshold on the way up.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from key_velocity_exp.common import LOWEST_MIDI
from key_velocity_exp.sim import piano_constants as pc
from key_velocity_exp.sim.velocity_model import TwoSwitchTracker, TwoSwitchVelocityModel

MODEL_NAMES = ("measured", "legacy")


class KeyMidi:
    """Turns key angles into MIDI events. ``after_substep`` returns ``(t, "NoteOn"|"NoteOff", midi_note, velocity)`` tuples."""

    def __init__(self, model_name: str, joint_range_max: np.ndarray, rest_angle: Optional[np.ndarray] = None, model_path: Optional[str] = None):
        if model_name not in MODEL_NAMES:
            raise ValueError(f"model_name must be one of {MODEL_NAMES}, got {model_name!r}")
        self.model_name = model_name
        self.q_max = np.asarray(joint_range_max, dtype=float)
        self._tracker: Optional[TwoSwitchTracker] = None
        if model_name == "measured":
            self._tracker = TwoSwitchTracker(TwoSwitchVelocityModel.load(model_path), self.q_max, rest_angle)
        self.reset()

    def reset(self) -> None:
        self._prev_state: Optional[np.ndarray] = None
        if self._tracker is not None:
            self._tracker.reset()

    @property
    def pressed(self) -> Optional[np.ndarray]:
        """Keys whose note is currently on (measured model only)."""
        return None if self._tracker is None else self._tracker.pressed

    def after_substep(self, t: float, qpos: np.ndarray, qvel: np.ndarray) -> list:
        qpos = np.asarray(qpos, dtype=float)
        qvel = np.asarray(qvel, dtype=float)
        if self._tracker is not None:
            on_keys, velocities, off_keys = self._tracker.update(t, qpos, qvel)
            events = [(t, "NoteOn", int(k) + LOWEST_MIDI, int(v)) for k, v in zip(on_keys, velocities)]
            events += [(t, "NoteOff", int(k) + LOWEST_MIDI, 0) for k in off_keys]
            return events
        # legacy: signed distance to the activation threshold, note events on sign changes
        clipped = np.clip(qpos, 0.0, self.q_max)
        state = (clipped - (self.q_max - pc.KEY_THRESHOLD)) / self.q_max
        if self._prev_state is None:
            self._prev_state = state.copy()
            return []
        change = state * self._prev_state <= 0
        events = []
        for k in np.flatnonzero(change & (qvel > 0)):
            events.append((t, "NoteOn", int(k) + LOWEST_MIDI, int(np.clip(qvel[k] * pc.QVEL_TO_VELOCITY_SCALING, 0, 127.1))))
        for k in np.flatnonzero(change & (qvel < 0)):
            events.append((t, "NoteOff", int(k) + LOWEST_MIDI, 0))
        self._prev_state = state.copy()
        return events
