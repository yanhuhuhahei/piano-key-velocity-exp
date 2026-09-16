"""Keyboard layout in the robot base frame, built from three taught poses.

Teach (with the UR freedrive / teach pendant) the fingertip *just touching*:
  1. a low white key  (e.g. C2)   -> pose_white_low
  2. a high white key (e.g. C6)   -> pose_white_high
  3. any black key    (e.g. C#4)  -> pose_black_ref
White keys are pressed at the same point along the key as the two taught white keys (use the
front third of the key), black keys at the same point as the taught black key. Standard 88-key
geometry (white pitch 23.5 mm) is used to interpolate every other key; the fitted pitch is printed
as a sanity check and per-key contact heights are refined by the contact search in run_experiment.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np

from key_velocity_exp.common import WHITE_KEY_PITCH_M, is_black, midi_to_name, white_key_index


@dataclass
class KeyLayout:
    note_white_low: int
    pose_white_low: list
    note_white_high: int
    pose_white_high: list
    note_black_ref: int
    pose_black_ref: list
    z_contact_measured: dict = field(default_factory=dict)  # str(midi_note) -> measured TCP z at contact
    notes: str = ""

    # ------------------------------------------------------------------ derived geometry
    def _fit(self):
        p0 = np.asarray(self.pose_white_low[:3])
        p1 = np.asarray(self.pose_white_high[:3])
        i0, i1 = white_key_index(self.note_white_low), white_key_index(self.note_white_high)
        axis = p1 - p0
        pitch = np.linalg.norm(axis[:2]) / (i1 - i0)
        u = np.array([axis[0], axis[1], 0.0]) / np.linalg.norm(axis[:2])
        z_slope = axis[2] / (i1 - i0)  # keyboard might not be perfectly level in the robot frame
        pb = np.asarray(self.pose_black_ref[:3])
        ib = white_key_index(self.note_black_ref)
        on_axis = p0 + u * (ib - i0) * pitch
        black_offset = pb - on_axis  # includes the "toward the fallboard" shift and the height step
        black_offset_along = float(np.dot(black_offset[:2], u[:2]))
        black_offset[:2] -= black_offset_along * u[:2]  # keep only the perpendicular xy part
        return p0, i0, u, pitch, z_slope, black_offset, black_offset_along

    def report(self) -> str:
        p0, i0, u, pitch, z_slope, boff, boff_along = self._fit()
        return (
            f"white-key pitch = {pitch * 1e3:.2f} mm (standard {WHITE_KEY_PITCH_M * 1e3:.1f}); "
            f"axis xy = ({u[0]:+.3f}, {u[1]:+.3f}); z slope = {z_slope * 1e3:+.3f} mm/key; "
            f"black key offset: back {np.linalg.norm(boff[:2]) * 1e3:.1f} mm, up {boff[2] * 1e3:.1f} mm, "
            f"along-axis residual {boff_along * 1e3:+.2f} mm"
        )

    def nominal_contact_pose(self, midi_note: int) -> np.ndarray:
        """UR pose with the fingertip just touching the key (before per-key refinement)."""
        p0, i0, u, pitch, z_slope, boff, _ = self._fit()
        idx = white_key_index(midi_note)
        p = p0 + u * (idx - i0) * pitch
        p[2] += z_slope * (idx - i0)
        if is_black(midi_note):
            p = p + boff
        rot = np.asarray(self.pose_white_low[3:6])
        return np.concatenate([p, rot])

    def contact_pose(self, midi_note: int) -> np.ndarray:
        pose = self.nominal_contact_pose(midi_note)
        z = self.z_contact_measured.get(str(midi_note))
        if z is not None:
            pose[2] = z
        return pose

    def hover_pose(self, midi_note: int, hover_m: float) -> np.ndarray:
        pose = self.contact_pose(midi_note)
        pose[2] += hover_m
        return pose

    def nearest_key(self, xy: np.ndarray) -> Optional[int]:
        """Key whose press point is nearest to xy (used by the simulator)."""
        if not hasattr(self, "_xy_cache"):
            self._xy_cache = np.array([self.nominal_contact_pose(n)[:2] for n in range(21, 109)])
        d = np.linalg.norm(self._xy_cache - np.asarray(xy), axis=1)
        i = int(np.argmin(d))
        return 21 + i if d[i] < 0.012 else None

    # ------------------------------------------------------------------ persistence
    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @staticmethod
    def load(path: str) -> "KeyLayout":
        with open(path) as f:
            return KeyLayout(**json.load(f))

    @staticmethod
    def synthetic() -> "KeyLayout":
        """A plausible layout for simulation: keyboard along +y in front of the robot, tool pointing down."""
        rot = [np.pi, 0.0, 0.0]
        z_white = 0.120
        p0 = np.array([0.45, -0.60, z_white])
        u = np.array([0.0, 1.0, 0.0])
        i_lo, i_hi = white_key_index(36), white_key_index(84)
        p1 = p0 + u * (i_hi - i_lo) * WHITE_KEY_PITCH_M
        ib = white_key_index(61)
        pb = p0 + u * (ib - i_lo) * WHITE_KEY_PITCH_M + np.array([-0.045, 0.0, 0.012])
        return KeyLayout(36, [*p0, *rot], 84, [*p1, *rot], 61, [*pb, *rot], notes="synthetic layout for --sim")


def describe_keys(layout: KeyLayout, notes: list[int]) -> str:
    lines = []
    for n in notes:
        p = layout.contact_pose(n)
        meas = "measured" if str(n) in layout.z_contact_measured else "nominal"
        lines.append(f"  {midi_to_name(n):4s} ({n:3d}) {'black' if is_black(n) else 'white'}  xyz = {p[0]:+.4f} {p[1]:+.4f} {p[2]:+.4f}  [{meas}]")
    return "\n".join(lines)
