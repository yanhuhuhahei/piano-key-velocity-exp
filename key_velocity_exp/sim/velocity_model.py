"""Key-press -> MIDI velocity model of a real digital piano, for use inside a simulator.

The model file (written by ``kve-export``) describes the mechanism identified with the robot:
  * the key passes depth A -> a timer starts; it passes depth B -> the timer stops and note-on is sent
    (``latency_s`` later); velocity = k * dt^b with dt in seconds; dt >= floor -> velocity 1;
  * note-off when the key rises back above A;
  * A, B, k, b, floor and the measured speed table are stored per measured key and interpolated linearly
    in key index for the other keys.

The depths are absolute (metres below the *resting* key surface at the fingertip) and become hinge
angles through the key lever: angle_A = rest_angle + A_m / lever (white keys: lever 0.15 m; black keys
are scaled by their shorter travel). The simulated white keys sag under gravity, so the resting angle
is measured with ``key_rest_angles()`` and passed to ``angle_thresholds`` / ``TwoSwitchTracker``.

``TwoSwitchTracker`` is a numpy implementation that runs once per physics substep; ``as_torch`` /
``velocity_torch`` give the same lookup table for vectorised (num_envs x 88) simulators.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np

from key_velocity_exp.sim import piano_constants

MODEL_PATH_ENV = "KVE_VELOCITY_MODEL"
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXAMPLE_MODEL_PATH = os.path.join(_REPO_ROOT, "examples", "yamaha_p45", "p45_velocity_model.json")


def default_model_path() -> str:
    """$KVE_VELOCITY_MODEL if set, else the Yamaha P-45 example model shipped in the repository."""
    return os.environ.get(MODEL_PATH_ENV) or EXAMPLE_MODEL_PATH


LUT_DT_MIN_S, LUT_DT_MAX_S, LUT_POINTS = 0.002, 1.0, 96  # common log-spaced dt grid of the per-key lookup table


@dataclass
class TwoSwitchVelocityModel:
    a_frac: np.ndarray  # (88,) timer-start depth as a fraction of the real key dip (informational)
    b_frac: np.ndarray  # (88,) note-on depth as a fraction of the real key dip (informational)
    a_off_rad: np.ndarray  # (88,) hinge angle of A above the resting angle: depth_A / lever (white 0.15 m; black scaled by its travel)
    b_off_rad: np.ndarray  # (88,)
    k: np.ndarray  # (88,) velocity = k * dt_s ** b (parametric summary; the lookup table below is what is used)
    b: np.ndarray  # (88,)
    floor_dt_s: np.ndarray  # (88,) dt at/above which velocity = 1
    latency_s: np.ndarray  # (88,)
    measured_keys: list
    lut: np.ndarray  # (88, LUT_POINTS) velocity on the common log-dt grid: measured table inside the covered range, power law outside
    path: str = ""

    @staticmethod
    def _log_grid() -> np.ndarray:
        return np.linspace(np.log(LUT_DT_MIN_S), np.log(LUT_DT_MAX_S), LUT_POINTS)

    @classmethod
    def load(cls, path: str | None = None) -> "TwoSwitchVelocityModel":
        path = path or default_model_path()
        with open(path) as f:
            m = json.load(f)
        keys = sorted(m["keys"], key=lambda k: k["key"])
        idx = np.array([k["key"] for k in keys], dtype=float)
        all_idx = np.arange(piano_constants.NUM_KEYS, dtype=float)

        def interp(field, default):
            vals = np.array([k[field] if k.get(field) is not None else default for k in keys], dtype=float)
            return np.interp(all_idx, idx, vals)  # constant extrapolation outside the measured range

        grid = cls._log_grid()
        per_key_lut = []
        for kk in keys:
            kp, bp, floor = kk["k"], kk["b"], kk.get("floor_dt_s") or 0.3
            pts = [(x["dt_s"], x["velocity_median"]) for x in (kk.get("speed_table") or []) if x.get("dt_s") and 1 < x["velocity_median"] < 127]
            dts = np.array([q[0] for q in pts]) if pts else np.zeros(0)
            vel = np.array([q[1] for q in pts]) if pts else np.zeros(0)
            dt_g = np.exp(grid)
            lut = kp * dt_g**bp  # power law everywhere ...
            if len(dts) >= 3:  # ... replaced by the measured (monotone) table inside the covered dt range
                order = np.argsort(dts)
                dts, vel = dts[order], vel[order]
                vel = np.minimum.accumulate(vel)  # velocity must not increase with dt
                inside = (dt_g >= dts[0]) & (dt_g <= dts[-1])
                lut[inside] = np.interp(np.log(dt_g[inside]), np.log(dts), vel)
                # keep the extrapolated parts continuous with the table ends
                lut[dt_g < dts[0]] = lut[dt_g < dts[0]] - (kp * dts[0] ** bp - vel[0])
                lut[dt_g > dts[-1]] = lut[dt_g > dts[-1]] - (kp * dts[-1] ** bp - vel[-1])
            lut = np.minimum.accumulate(lut)
            lut[dt_g >= floor] = 1.0
            per_key_lut.append(np.clip(lut, 1, 127))
        per_key_lut = np.array(per_key_lut)  # (n_meas, G)
        full = np.stack([np.interp(all_idx, idx, per_key_lut[:, g]) for g in range(LUT_POINTS)], axis=1)  # (88, G)

        # absolute switch depths (m, below the resting key surface at the fingertip) -> hinge angle offsets
        a_m, b_m = interp("A_m", 0.0048), interp("B_m", 0.0080)
        black = np.zeros(piano_constants.NUM_KEYS, dtype=bool)
        black[np.array(piano_constants.BLACK_KEY_INDICES)] = True
        lever = np.where(black, piano_constants.BLACK_KEY_LENGTH, piano_constants.WHITE_KEY_LENGTH)
        travel_scale = np.where(black, piano_constants.BLACK_KEY_TRAVEL_DISTANCE / piano_constants.WHITE_KEY_TRAVEL_DISTANCE, 1.0)
        return cls(
            a_off_rad=a_m * travel_scale / lever,
            b_off_rad=b_m * travel_scale / lever,
            a_frac=interp("A_frac", 0.48),
            b_frac=interp("B_frac", 0.80),
            k=interp("k", 4.5),
            b=interp("b", -0.6),
            floor_dt_s=interp("floor_dt_s", 0.3),
            latency_s=interp("latency_s", 0.004),
            measured_keys=[k["key"] for k in keys],
            lut=full,
            path=path,
        )

    # ------------------------------------------------------------------ numpy
    def angle_thresholds(self, joint_range_max: np.ndarray, rest_angle: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Hinge angles of A and B for every key: resting angle + measured depth / lever (kept below the joint limit)."""
        rest = np.zeros_like(joint_range_max) if rest_angle is None else rest_angle
        hi = joint_range_max - 1e-4
        return np.minimum(rest + self.a_off_rad, hi - 1e-3), np.minimum(rest + self.b_off_rad, hi)

    def velocity(self, dt_s: np.ndarray, key_ids: np.ndarray) -> np.ndarray:
        """Lookup-table velocity for dt (s) of the given keys (numpy)."""
        dt = np.clip(np.asarray(dt_s, dtype=float), LUT_DT_MIN_S, LUT_DT_MAX_S)
        grid = self._log_grid()
        pos = (np.log(dt) - grid[0]) / (grid[1] - grid[0])
        i0 = np.clip(np.floor(pos).astype(int), 0, LUT_POINTS - 2)
        w = pos - i0
        v = self.lut[key_ids, i0] * (1 - w) + self.lut[key_ids, i0 + 1] * w
        return np.clip(np.rint(v), 1, 127).astype(int)

    def velocity_power_law(self, dt_s: np.ndarray, key_ids: np.ndarray) -> np.ndarray:
        dt = np.maximum(np.asarray(dt_s, dtype=float), 1e-4)
        v = self.k[key_ids] * dt ** self.b[key_ids]
        v = np.where(dt >= self.floor_dt_s[key_ids], 1.0, v)
        return np.clip(np.rint(v), 1, 127).astype(int)

    # ------------------------------------------------------------------ torch (optional, for vectorised simulators)
    def as_torch(self, device):
        import torch

        t = lambda x: torch.as_tensor(np.asarray(x, dtype=np.float32), device=device)
        grid = self._log_grid()
        return {
            "a_frac": t(self.a_frac),
            "b_frac": t(self.b_frac),
            "a_off_rad": t(self.a_off_rad),
            "b_off_rad": t(self.b_off_rad),
            "k": t(self.k),
            "b": t(self.b),
            "floor_dt_s": t(self.floor_dt_s),
            "lut": t(self.lut),
            "log_dt0": float(grid[0]),
            "log_dt_step": float(grid[1] - grid[0]),
        }

    @staticmethod
    def velocity_torch(params: dict, dt_s, key_mask=None):
        """dt_s: (num_envs, 88) tensor of dt (seconds). Returns int32 velocities (1..127) from the per-key lookup table."""
        import torch

        dt = torch.clamp(dt_s, LUT_DT_MIN_S, LUT_DT_MAX_S)
        pos = (torch.log(dt) - params["log_dt0"]) / params["log_dt_step"]
        i0 = torch.clamp(torch.floor(pos).long(), 0, LUT_POINTS - 2)
        w = pos - i0.float()
        lut = params["lut"]  # (88, G)
        key_idx = torch.arange(lut.shape[0], device=lut.device).unsqueeze(0).expand_as(i0)
        v = lut[key_idx, i0] * (1 - w) + lut[key_idx, i0 + 1] * w
        return torch.clamp(torch.round(v), 1, 127).to(torch.int32)


# kept for readers of the original ExpressivePianist code
P45VelocityModel = TwoSwitchVelocityModel


def key_joints(root):
    """The 88 key hinge joints of the MJCF model, sorted by key index (0 = A0)."""
    joints = [j for j in root.find_all("joint") if j.name != "base_joint"]
    order = np.argsort([int(j.name.split("_")[-1]) for j in joints])
    return [joints[i] for i in order]


def key_rest_angles() -> np.ndarray:
    """Settled hinge angle of every key with nothing touching it.

    The MJCF key spring (1 Nm/rad, springref -1 deg) does not fully carry the white key's weight, so a
    free white key rests ~0.67 deg (about 1.7 mm at the tip, 17 % of the travel) below the geometric
    top; black keys rest at 0. The real piano's A/B depths are measured from the resting surface, so
    the switch angles are placed at rest + depth / lever."""
    from dm_control import mjcf

    from key_velocity_exp.sim.build_piano_mjcf import build

    root = build()
    physics = mjcf.Physics.from_mjcf_model(root)
    joints = key_joints(root)
    for _ in range(int(1.0 / physics.timestep())):
        physics.step()
    return np.array(physics.bind(joints).qpos, dtype=float)


class TwoSwitchTracker:
    """Per-key two-switch timer for a MuJoCo simulator (numpy, called every physics substep).

    update(t, qpos, qvel) -> (note_on_keys, note_on_velocities, note_off_keys)
    Crossing times are interpolated inside the substep using the joint velocity, so dt has
    sub-timestep resolution (with a 2 ms physics step, dt for velocity 60 is ~14 ms).
    """

    def __init__(self, model: TwoSwitchVelocityModel, joint_range_max: np.ndarray, rest_angle: np.ndarray | None = None):
        self.model = model
        q_max = np.asarray(joint_range_max, dtype=float)
        self.rest = np.zeros_like(q_max) if rest_angle is None else np.asarray(rest_angle, dtype=float)
        self.angle_a, self.angle_b = model.angle_thresholds(q_max, self.rest)
        self.reset()

    def reset(self) -> None:
        n = piano_constants.NUM_KEYS
        self.t_a = np.full(n, np.nan)  # time the key passed A on the way down
        self.pressed = np.zeros(n, dtype=bool)  # note currently on
        self.prev_q = np.zeros(n)

    @property
    def activation(self) -> np.ndarray:
        return self.pressed

    def update(self, t: float, qpos: np.ndarray, qvel: np.ndarray):
        q = np.asarray(qpos, dtype=float)
        v = np.asarray(qvel, dtype=float)
        safe_v = np.where(np.abs(v) > 1e-6, v, 1e-6)
        # arm the timer when crossing A downward
        cross_a = (q >= self.angle_a) & (self.prev_q < self.angle_a) & ~self.pressed
        t_cross_a = t - (q - self.angle_a) / safe_v
        self.t_a = np.where(cross_a, np.minimum(np.maximum(t_cross_a, t - 0.01), t), self.t_a)
        # note-on when crossing B downward with the timer armed
        cross_b = (q >= self.angle_b) & (self.prev_q < self.angle_b) & ~self.pressed & ~np.isnan(self.t_a)
        note_on_keys = np.flatnonzero(cross_b)
        velocities = np.zeros(0, dtype=int)
        if len(note_on_keys):
            t_cross_b = t - (q[note_on_keys] - self.angle_b[note_on_keys]) / safe_v[note_on_keys]
            t_cross_b = np.clip(t_cross_b, t - 0.01, t)
            dt = np.maximum(t_cross_b - self.t_a[note_on_keys], 1e-4)
            velocities = self.model.velocity(dt, note_on_keys)
            self.pressed[note_on_keys] = True
        # release: back above A -> note-off (if on) and disarm
        released = q < self.angle_a
        note_off_keys = np.flatnonzero(released & self.pressed)
        self.pressed[released] = False
        self.t_a[released] = np.nan
        self.prev_q = q.copy()
        return note_on_keys, velocities, note_off_keys
