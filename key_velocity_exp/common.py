"""Shared definitions for the key-velocity experiment.

Coordinate conventions
----------------------
* Robot poses are UR ``[x, y, z, rx, ry, rz]`` in the robot base frame (metres, axis-angle).
* ``z`` points up. Pressing a key means decreasing the TCP ``z`` while ``x, y`` and the orientation stay fixed.
* ``depth`` is measured downward from the *contact height* ``z_contact`` of a key
  (``depth = z_contact - z``), so ``depth = 0`` is "just touching", positive is pressed in.
* All timestamps are ``time.monotonic()`` seconds, shared by the robot sampler and the MIDI listener.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Iterator

import numpy as np

# ----------------------------------------------------------------------------- piano geometry

WHITE_KEY_PITCH_M = 0.0235  # centre-to-centre distance of white keys (standard 88-key, ~165 mm per octave)
NUM_KEYS = 88
LOWEST_MIDI = 21  # A0

BLACK_PITCH_CLASSES = {1, 3, 6, 8, 10}


def is_black(midi_note: int) -> bool:
    return (midi_note % 12) in BLACK_PITCH_CLASSES


def midi_to_name(midi_note: int) -> str:
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    return f"{names[midi_note % 12]}{midi_note // 12 - 1}"


def name_to_midi(name: str) -> int:
    name = name.strip().upper().replace("♯", "#")
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    flats = {"DB": "C#", "EB": "D#", "GB": "F#", "AB": "G#", "BB": "A#"}
    i = 1
    while i < len(name) and name[i] in "#B":
        i += 1
    pc, octave = name[:i], name[i:]
    pc = flats.get(pc, pc)
    return names.index(pc) + 12 * (int(octave) + 1)


def white_key_index(midi_note: int) -> float:
    """Position of a key along the keyboard, in units of white-key pitch, with A0 = 0.

    White keys get integer indices. A black key sits between its two white neighbours (index n + 0.5).
    """
    n_white = 0.0
    for m in range(LOWEST_MIDI, midi_note):
        if not is_black(m):
            n_white += 1
    if is_black(midi_note):
        return n_white - 0.5
    return n_white


# ----------------------------------------------------------------------------- press profiles


@dataclass
class PressProfile:
    """Desired TCP depth trajectory for one key press.

    kind:
      const      – constant descent speed ``v0`` through the key travel
      accel      – speed ramps linearly from ``v0`` (at start) to ``v1`` (at ``depth_total``)
      decel      – same as accel but normally with ``v1 < v0``
      two_stage  – slow ``v0`` until ``depth_mid``, then fast ``v1`` (or vice versa)
      sine       – half-sine speed bump peaking at ``v1`` with floor ``v0``
      kick       – touch the key at the low speed ``v0`` and accelerate *inside* the key at ``accel_max``
                   up to ``v1`` (speed = sqrt(v0^2 + 2 a d)); the fingertip keeps pushing the key so
                   the key cannot be launched ahead of it, which happens with a fast rigid impact
      throw      – kick, but *release* the key at ``depth_mid_m`` (speed -> 0 = brake): the light key
                   flies on alone through the switches while the heavy arm brakes early. This is the
                   only way to reach the top velocities without the arm hitting the key bed at speed.

    start_above_m: the descent starts this far *above* the contact height. Used to reach the
      target speed before the fingertip touches the key (a constant speed is otherwise impossible:
      the arm has to accelerate from rest inside the key travel). ``0`` = start from just touching,
      exactly as requested in the experiment plan; the acceleration then happens inside the key and
      is limited by ``accel_max``.
    depth_total_m: how deep below contact the press goes (P-45 key dip is about 10 mm; keep a margin
      and rely on the force abort in the robot driver).
    hold_s: dwell at the bottom before lifting.
    lift_speed: speed used to lift the key again (slow, so note-off is clean).
    """

    kind: str = "const"
    v0: float = 0.05
    v1: float = 0.05
    depth_mid_m: float = 0.004
    start_above_m: float = 0.0
    depth_total_m: float = 0.0095
    accel_max: float = 8.0
    hold_s: float = 0.3
    lift_speed: float = 0.03
    lift_height_m: float = 0.0  # extra height above contact reached after lifting (0 = back to start)
    stages: list = field(default_factory=list)  # kind == "staged": [[depth_m, speed, dwell_s], ...] executed in order (closed loop on depth)

    # ---- geometry helpers
    def speed_at_depth(self, d: float) -> float:
        """Nominal (pre-acceleration-limit) descent speed as a function of depth relative to contact."""
        D = self.depth_total_m
        if self.kind == "const":
            return self.v0
        if self.kind in ("accel", "decel"):
            a = np.clip(d / D, 0.0, 1.0)
            return self.v0 + (self.v1 - self.v0) * a
        if self.kind == "two_stage":
            return self.v0 if d < self.depth_mid_m else self.v1
        if self.kind == "sine":
            a = np.clip(d / D, 0.0, 1.0)
            return self.v0 + (self.v1 - self.v0) * np.sin(np.pi * a)
        if self.kind == "kick":
            return float(min(self.v1, np.sqrt(self.v0**2 + 2.0 * self.accel_max * max(d, 0.0))))
        if self.kind == "throw":
            if d >= self.depth_mid_m:
                return 0.0
            return float(min(self.v1, np.sqrt(self.v0**2 + 2.0 * self.accel_max * max(d, 0.0))))
        if self.kind == "staged":
            for depth, speed, _dwell in self.stages:
                if d < depth:
                    return float(speed)
            return float(self.stages[-1][1]) if self.stages else 0.0
        raise ValueError(f"unknown profile kind {self.kind}")

    def depth_trajectory(self, dt: float, descent_only: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """Integrate the profile into a time-sampled depth trajectory.

        Returns ``(t, depth)`` where depth is relative to the contact height (negative = above the key).
        The descent obeys ``accel_max`` starting from rest at ``-start_above_m``; after reaching
        ``depth_total_m`` it holds for ``hold_s`` and lifts at ``lift_speed``.
        """
        t_list, d_list = [0.0], [-self.start_above_m]
        d, v, t = -self.start_above_m, 0.0, 0.0
        max_steps = int(60.0 / dt)
        for _ in range(max_steps):
            v_target = self.speed_at_depth(max(d, 0.0))
            # above the key: aim for the speed wanted at contact (so we arrive at the right speed)
            if d < 0:
                v_target = self.speed_at_depth(0.0)
            dv = np.clip(v_target - v, -self.accel_max * dt, self.accel_max * dt)
            v = max(v + dv, 1e-4)
            d = d + v * dt
            t += dt
            t_list.append(t)
            d_list.append(min(d, self.depth_total_m))
            if d >= self.depth_total_m:
                break
        if descent_only:
            return np.asarray(t_list), np.asarray(d_list)
        # hold
        n_hold = int(round(self.hold_s / dt))
        for _ in range(n_hold):
            t += dt
            t_list.append(t)
            d_list.append(self.depth_total_m)
        # lift
        d_end = -(self.start_above_m + self.lift_height_m)
        d = self.depth_total_m
        while d > d_end:
            d -= self.lift_speed * dt
            t += dt
            t_list.append(t)
            d_list.append(max(d, d_end))
        return np.asarray(t_list), np.asarray(d_list)

    def reachable_speed(self) -> float:
        """Speed the profile can reach at ``depth_total_m`` when starting from rest at ``-start_above_m``."""
        return float(np.sqrt(2.0 * self.accel_max * (self.start_above_m + self.depth_total_m)))

    def label(self) -> str:
        if self.kind == "const":
            return f"const v={self.v0:.3f}"
        if self.kind == "two_stage":
            return f"two_stage {self.v0:.3f}->{self.v1:.3f}@{self.depth_mid_m * 1e3:.1f}mm"
        if self.kind == "kick":
            return f"kick {self.v0:.2f}->{self.v1:.2f} a={self.accel_max:.0f}"
        if self.kind == "throw":
            return f"throw {self.v0:.2f}->{self.v1:.2f} rel@{self.depth_mid_m * 1e3:.0f}mm"
        if self.kind == "staged":
            return "staged " + " | ".join(f"{d * 1e3:.1f}mm@{v:.2f}" + (f" hold{w:.1f}s" if w else "") for d, v, w in self.stages)
        return f"{self.kind} {self.v0:.3f}->{self.v1:.3f}"


# ----------------------------------------------------------------------------- data containers

ROBOT_SAMPLE_FIELDS = [
    "t",  # monotonic time
    "robot_t",  # controller timestamp (RTDE)
    "target_z",  # commanded TCP z
    "tcp_x",
    "tcp_y",
    "tcp_z",
    "tcp_rx",
    "tcp_ry",
    "tcp_rz",
    "vx",
    "vy",
    "vz",
    "wx",
    "wy",
    "wz",
    "fx",
    "fy",
    "fz",
    "tx",
    "ty",
    "tz",
    "q0",
    "q1",
    "q2",
    "q3",
    "q4",
    "q5",
    "qd0",
    "qd1",
    "qd2",
    "qd3",
    "qd4",
    "qd5",
]


@dataclass
class MidiEvent:
    t: float
    kind: str  # "note_on" | "note_off"
    note: int
    velocity: int

    def as_row(self) -> list:
        return [self.t, self.kind, self.note, self.velocity]


@dataclass
class TrialRecord:
    trial_id: int
    midi_note: int
    key_name: str
    key_is_black: bool
    profile: PressProfile
    key_pose_hover: list  # UR pose used above the key
    z_contact: float  # contact height (TCP z) for this key
    t_start: float  # monotonic time when the press command started
    t_end: float
    robot_samples: np.ndarray = field(default_factory=lambda: np.zeros((0, len(ROBOT_SAMPLE_FIELDS))))
    midi_events: list = field(default_factory=list)  # list[MidiEvent]
    aborted: bool = False
    abort_reason: str = ""
    extra: dict = field(default_factory=dict)

    def save(self, directory: str) -> str:
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, f"trial_{self.trial_id:04d}.npz")
        meta = {
            "trial_id": self.trial_id,
            "midi_note": self.midi_note,
            "key_name": self.key_name,
            "key_is_black": self.key_is_black,
            "profile": asdict(self.profile),
            "key_pose_hover": list(map(float, self.key_pose_hover)),
            "z_contact": float(self.z_contact),
            "t_start": self.t_start,
            "t_end": self.t_end,
            "aborted": self.aborted,
            "abort_reason": self.abort_reason,
            "extra": self.extra,
            "robot_sample_fields": ROBOT_SAMPLE_FIELDS,
        }
        midi = np.array([[e.t, 1.0 if e.kind == "note_on" else 0.0, e.note, e.velocity] for e in self.midi_events], dtype=float)
        np.savez_compressed(path, robot_samples=self.robot_samples, midi_events=midi.reshape(-1, 4), meta=json.dumps(meta, default=str))
        return path

    @staticmethod
    def load(path: str) -> "TrialRecord":
        z = np.load(path, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        events = [MidiEvent(t=float(r[0]), kind="note_on" if r[1] > 0.5 else "note_off", note=int(r[2]), velocity=int(r[3])) for r in z["midi_events"]]
        return TrialRecord(
            trial_id=meta["trial_id"],
            midi_note=meta["midi_note"],
            key_name=meta["key_name"],
            key_is_black=meta["key_is_black"],
            profile=PressProfile(**meta["profile"]),
            key_pose_hover=meta["key_pose_hover"],
            z_contact=meta["z_contact"],
            t_start=meta["t_start"],
            t_end=meta["t_end"],
            robot_samples=z["robot_samples"],
            midi_events=events,
            aborted=meta["aborted"],
            abort_reason=meta["abort_reason"],
            extra=meta.get("extra", {}),
        )

    # ---- convenience views
    def col(self, name: str) -> np.ndarray:
        return self.robot_samples[:, ROBOT_SAMPLE_FIELDS.index(name)]

    def depth(self) -> np.ndarray:
        return self.z_contact - self.col("tcp_z")

    def note_on_events(self) -> list:
        return [e for e in self.midi_events if e.kind == "note_on" and e.note == self.midi_note]


# ----------------------------------------------------------------------------- shared CLI / plotting helpers

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]  # categorical colours (light background)
TEXT, TEXT2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"

ROBOT_IP_ENV = "UR_ROBOT_IP"
MIDI_PORT_ENV = "KVE_MIDI_PORT"


def add_connection_args(p) -> None:
    """--robot-ip / --midi-port with environment-variable defaults (shared by every hardware script)."""
    p.add_argument("--robot-ip", default=os.environ.get(ROBOT_IP_ENV), help=f"UR controller IP (default: ${ROBOT_IP_ENV}); not needed with --sim")
    p.add_argument(
        "--midi-port", default=os.environ.get(MIDI_PORT_ENV), help=f"substring of the piano's MIDI input port name (default: ${MIDI_PORT_ENV}, else auto-detect; list ports with kve-midi --list)"
    )


def check_connection_args(p, args) -> None:
    if not getattr(args, "sim", False) and not args.robot_ip:
        p.error(f"--robot-ip is required for the real robot (or set {ROBOT_IP_ENV}); use --sim for a hardware-free dry run")


_clock = time.monotonic  # replaced by the simulator clock in --sim (see ur5_io.patch_sim_time)


def now() -> float:
    return _clock()


def load_session(directory: str) -> Iterator[TrialRecord]:
    for fn in sorted(os.listdir(directory)):
        if fn.startswith("trial_") and fn.endswith(".npz"):
            yield TrialRecord.load(os.path.join(directory, fn))
