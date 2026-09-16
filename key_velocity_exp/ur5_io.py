"""UR5 / UR5e driver for the key-press experiment (ur_rtde), plus a kinematic simulator for dry runs.

Real robot notes
----------------
* Requires the robot in *Remote Control* mode (e-series) with the ur_rtde control script allowed,
  reachable at ``--robot-ip``. Default RTDE rate is the controller maximum (500 Hz e-series,
  125 Hz CB3); pass ``--hz`` explicitly if you want a lower streaming rate.
* The TCP must be set to the fingertip (``--tcp``), otherwise the recorded "tcp_z" is not the fingertip.
* The tool must be somewhat compliant (rubber/silicone tip) and ``force_abort_n`` must be low enough
  to protect the key bed: the key dip is only ~10 mm and the arm is stiff.
* UR5 CB3 has no force/torque sensor; ``getActualTCPForce`` is estimated from joint currents (noise of
  several N). Use ``--contact-method midi`` or ``teach`` there; ``force`` works well on the UR5e.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Callable, Optional

import numpy as np

from key_velocity_exp.common import ROBOT_SAMPLE_FIELDS, now


class RobotAbort(RuntimeError):
    pass


@dataclass
class ServoParams:
    hz: float = 125.0
    lookahead: float = 0.03  # [0.03, 0.2]; smallest = least smoothing = best tracking of fast profiles
    gain: float = 1500.0  # [100, 2000]
    force_abort_n: float = 15.0  # abort press if |fz - baseline| exceeds this
    depth_abort_m: float = 0.0125  # abort press if the fingertip goes deeper than this below contact

    @property
    def dt(self) -> float:
        return 1.0 / self.hz


class RobotBase:
    """Interface used by the experiment. ``sample()`` returns one row of ROBOT_SAMPLE_FIELDS[1:]."""

    params: ServoParams
    stop_decel: float = 25.0  # deceleration used when a press is cut short (servoStop / speedStop)
    last_impact_n: float = 0.0  # peak |fz - baseline| while braking into the key bed (speedl)

    def tcp_pose(self) -> np.ndarray:
        raise NotImplementedError

    def move_l(self, pose, speed=0.1, accel=0.5) -> None:
        raise NotImplementedError

    def sample(self, target_z: float) -> list:
        raise NotImplementedError

    def servo_begin(self) -> None:
        pass

    def servo_step(self, pose: np.ndarray) -> None:
        """Command one servo cycle towards ``pose`` and block until the cycle period has elapsed."""
        raise NotImplementedError

    def servo_end(self) -> None:
        pass

    def speed_step(self, v_xyz: np.ndarray, accel: float) -> None:
        """Command a Cartesian tool velocity for one cycle (velocity control, blocks one period)."""
        raise NotImplementedError

    def speed_stop(self, decel: float) -> None:
        raise NotImplementedError

    def freedrive(self, enable: bool) -> None:
        pass

    def pause(self, seconds: float) -> None:
        """Wait while keeping the clock consistent (real: sleep; sim: advance the simulated clock)."""
        time.sleep(seconds)

    def ensure_running(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def close(self) -> None:
        pass

    # ---------------------------------------------------------------- shared helpers
    def stream_z(
        self,
        base_pose: np.ndarray,
        z_targets: np.ndarray,
        z_contact: float,
        on_sample: Optional[Callable[[list], None]] = None,
        stop_when: Optional[Callable[[list], bool]] = None,
        force_baseline: Optional[float] = None,
        settle: bool = True,
    ) -> tuple[np.ndarray, Optional[str]]:
        """Servo the TCP through ``z_targets`` (one per control cycle) at fixed xy/orientation.

        Returns (samples, abort_reason). Samples: rows [t, robot_t, target_z, ...].
        Aborts (and lifts a bit) if the force or depth limits are hit.
        """
        rows = []
        abort = None
        pose = np.array(base_pose, dtype=float)
        self.servo_begin()
        try:
            for z in z_targets:
                pose[2] = z
                self.servo_step(pose)
                row = [now()] + self.sample(z)
                rows.append(row)
                if on_sample is not None:
                    on_sample(row)
                fz = row[ROBOT_SAMPLE_FIELDS.index("fz")]
                depth = z_contact - row[ROBOT_SAMPLE_FIELDS.index("tcp_z")]
                if force_baseline is not None and abs(fz - force_baseline) > self.params.force_abort_n:
                    abort = f"force abort |fz-baseline|={abs(fz - force_baseline):.1f} N"
                    break
                if depth > self.params.depth_abort_m:
                    abort = f"depth abort depth={depth * 1e3:.1f} mm"
                    break
                if stop_when is not None and stop_when(row):
                    break
        finally:
            self.servo_end()
        if abort is None and settle:  # the arm lags the servo targets: keep sampling until it is at rest
            rows += self.settle_rows()
        if abort is not None:
            # back off 8 mm above contact
            safe = np.array(base_pose, dtype=float)
            safe[2] = z_contact + 0.008
            self.move_l(safe, speed=0.05, accel=0.5)
        return np.asarray(rows).reshape(-1, len(ROBOT_SAMPLE_FIELDS)), abort

    def press_speedl(
        self,
        base_pose: np.ndarray,
        z_contact: float,
        speed_at_depth: Callable[[float], float],
        depth_limit: float,
        accel: float,
        brake_accel: float,
        force_baseline: float,
        stop_force_n: float,
        timeout_s: float = 3.0,
        extra_stop: Optional[Callable[[], Optional[str]]] = None,
        stop_force_min_depth: float = 0.006,
        retreat: Optional[tuple] = None,  # (z_target, speed, accel): rise immediately when extra_stop fires
        brake_at_depth: Optional[float] = None,  # reverse as soon as the finger is this deep (e.g. just past switch B)
    ) -> tuple[np.ndarray, Optional[str], str]:
        """Velocity-controlled press: closed loop on the *measured* depth.

        Each cycle the descent speed ``speed_at_depth(depth)`` is commanded with ``speedL`` (tool
        acceleration ``accel``). The descent ends when the F/T sensor sees the key bed
        (|fz - baseline| > ``stop_force_n``) or, as a fallback, at ``depth_limit``; then speedStop is
        issued ONCE with ``brake_accel`` and the arm is sampled until it is at rest.
        Returns (samples, abort_reason, stop_reason).
        """
        rows: list = []
        abort = None
        reason = ""
        retreated = False
        z_i, fz_i, vz_i, tz_i = (ROBOT_SAMPLE_FIELDS.index(k) for k in ("tcp_z", "fz", "vz", "target_z"))
        t0 = now()
        try:
            while True:
                row = [now()] + self.sample(np.nan)
                depth = z_contact - row[z_i]
                dF = abs(row[fz_i] - force_baseline)
                if dF > self.params.force_abort_n:
                    abort = f"force abort |fz-baseline|={dF:.1f} N"
                    rows.append(row)
                    break
                if depth > self.params.depth_abort_m:
                    abort = f"depth abort depth={depth * 1e3:.1f} mm"
                    rows.append(row)
                    break
                if brake_at_depth is not None and depth >= brake_at_depth:
                    reason = f"passed B at {depth * 1e3:.1f} mm"
                elif depth >= stop_force_min_depth and dF > stop_force_n:
                    reason = f"force {dF:.1f} N at {depth * 1e3:.1f} mm"
                elif depth >= depth_limit:
                    reason = f"depth limit {depth * 1e3:.1f} mm"
                if reason:
                    rows.append(row)
                    if retreat is not None:
                        # reverse with the velocity controller: speedStop() blocks ~50 ms and only decelerates at
                        # ~15-20 m/s^2 in practice; speedL(+v_up, a_up) is sampled and reverses much harder
                        z_target, v_up, a_up = retreat
                        rows += self.retreat(z_target, v_up, a_up)
                        reason += f", lifted to {(z_contact - z_target) * 1e3:+.1f} mm"
                        retreated = True
                    break
                if extra_stop is not None:
                    r = extra_stop()
                    if r:
                        reason = f"{r} at {depth * 1e3:.1f} mm"
                        rows.append(row)
                        if retreat is not None:  # reverse right away: the velocity controller brakes and rises in one motion
                            z_target, v_up, a_up = retreat
                            rows += self.retreat(z_target, v_up, a_up)
                            reason += f", lifted to {(z_contact - z_target) * 1e3:+.1f} mm"
                            retreated = True
                        break
                if row[0] - t0 > timeout_s:
                    abort = "timeout (never reached the key bed - fingertip not on the key?)"
                    rows.append(row)
                    break
                v = speed_at_depth(max(depth, 0.0))
                if v <= 0.0:  # profile says "release": brake now, the key flies on by itself
                    reason = f"released at {depth * 1e3:.1f} mm"
                    rows.append(row)
                    break
                row[tz_i] = z_contact - depth - v * self.params.dt
                rows.append(row)
                self.speed_step(np.array([0.0, 0.0, -v]), accel)
        finally:
            if not retreated:
                self.speed_stop(brake_accel)  # exactly once
        if retreated:
            arr = np.asarray(rows).reshape(-1, len(ROBOT_SAMPLE_FIELDS))
            self.last_impact_n = float(np.max(np.abs(arr[:, fz_i] - force_baseline))) if len(arr) else 0.0
            reason += f", impact peak {self.last_impact_n:.1f} N"
            return arr, abort, reason
        # sample (without sending commands) until the arm is at rest; a note-on arriving now still lifts
        t_b = now()
        peak = 0.0
        while True:
            self.pause(self.params.dt)
            row = [now()] + self.sample(np.nan)
            rows.append(row)
            peak = max(peak, abs(row[fz_i] - force_baseline))
            if extra_stop is not None and retreat is not None and abort is None:
                r = extra_stop()
                if r:
                    z_target, v_up, a_up = retreat
                    rows += self.retreat(z_target, v_up, a_up)
                    reason += f"; {r} while braking, lifted to {(z_contact - z_target) * 1e3:+.1f} mm"
                    self.last_impact_n = 0.0
                    return np.asarray(rows).reshape(-1, len(ROBOT_SAMPLE_FIELDS)), abort, reason
            if abs(row[vz_i]) < 2e-3 and now() - t_b > 0.02:
                break
            if now() - t_b > 1.0:
                break
        if reason:
            reason += f", impact peak {peak:.1f} N"
        self.last_impact_n = peak
        if abort is not None:
            safe = np.array(base_pose, dtype=float)
            safe[2] = z_contact + 0.008
            self.move_l(safe, speed=0.05, accel=0.5)
        return np.asarray(rows).reshape(-1, len(ROBOT_SAMPLE_FIELDS)), abort, reason

    def settle_rows(self, max_s: float = 0.5) -> list:
        """Sample without commanding until the TCP is at rest (or ``max_s``)."""
        rows: list = []
        vz_i = ROBOT_SAMPLE_FIELDS.index("vz")
        t_b = now()
        while True:
            self.pause(self.params.dt)
            row = [now()] + self.sample(np.nan)
            rows.append(row)
            if (abs(row[vz_i]) < 2e-3 and now() - t_b > 0.02) or now() - t_b > max_s:
                break
        return rows

    def retreat(self, z_target: float, speed: float, accel: float, timeout_s: Optional[float] = None) -> list:
        """Rise with velocity control to ``z_target`` (reverses any downward motion at ``accel``), then stop.
        Samples the whole way, so note-off can be located on the lift."""
        rows: list = []
        z_i = ROBOT_SAMPLE_FIELDS.index("tcp_z")
        if timeout_s is None:
            timeout_s = abs(z_target - float(self.tcp_pose()[2])) / max(speed, 1e-3) + 1.0
        t0 = now()
        try:
            while True:
                row = [now()] + self.sample(np.nan)
                rows.append(row)
                if row[z_i] >= z_target - 0.0005 or now() - t0 > timeout_s:
                    break
                self.speed_step(np.array([0.0, 0.0, speed]), accel)
        finally:
            self.speed_stop(accel)
        for _ in range(10):  # let it settle
            self.pause(self.params.dt)
            rows.append([now()] + self.sample(np.nan))
        return rows

    def press_stages(
        self,
        base_pose: np.ndarray,
        z_contact: float,
        stages: list,
        accel: float,
        brake_accel: float,
        force_baseline: float,
        stop_force_n: float,
        stop_force_min_depth: float,
        extra_stop: Optional[Callable[[], Optional[str]]] = None,
        retreat: Optional[tuple] = None,
        timeout_s: float = 6.0,
    ) -> tuple[np.ndarray, Optional[str], str]:
        """Closed-loop staged press: for each (depth, speed, dwell) descend at ``speed`` until the
        measured depth reaches ``depth``, brake, dwell (sampling all the time), then the next stage.
        Used by the switch-identification experiment (slow approach - pause - fast finish)."""
        rows: list = []
        abort = None
        reason = ""
        z_i, fz_i, vz_i = (ROBOT_SAMPLE_FIELDS.index(k) for k in ("tcp_z", "fz", "vz"))
        t0 = now()
        done = False
        try:
            for d_target, v, dwell in stages:
                while True:
                    row = [now()] + self.sample(np.nan)
                    rows.append(row)
                    depth = z_contact - row[z_i]
                    dF = abs(row[fz_i] - force_baseline)
                    if dF > self.params.force_abort_n:
                        abort = f"force abort |fz-baseline|={dF:.1f} N"
                    elif depth > self.params.depth_abort_m:
                        abort = f"depth abort depth={depth * 1e3:.1f} mm"
                    elif now() - t0 > timeout_s:
                        abort = "timeout"
                    if abort:
                        done = True
                        break
                    if extra_stop is not None:
                        r = extra_stop()
                        if r:
                            reason = f"{r} at {depth * 1e3:.1f} mm"
                            if retreat is not None:
                                rows += self.retreat(retreat[0], retreat[1], retreat[2])
                                reason += f", lifted to {(z_contact - retreat[0]) * 1e3:+.1f} mm"
                            done = True
                            break
                    if depth >= stop_force_min_depth and dF > stop_force_n:
                        reason = f"force {dF:.1f} N at {depth * 1e3:.1f} mm"
                        done = True
                        break
                    if depth >= d_target:
                        break
                    self.speed_step(np.array([0.0, 0.0, -float(v)]), accel)
                if done:
                    break
                self.speed_stop(brake_accel)
                rows += self.settle_rows(0.3)
                t_d = now()
                while now() - t_d < dwell:
                    self.pause(self.params.dt)
                    rows.append([now()] + self.sample(np.nan))
                    if extra_stop is not None and extra_stop():
                        reason = f"note-on during dwell at {(z_contact - rows[-1][z_i]) * 1e3:.1f} mm"
                        done = True
                        break
                if done:
                    break
            if not reason and abort is None:
                reason = f"stages done at {(z_contact - rows[-1][z_i]) * 1e3:.1f} mm"
        finally:
            if "lifted" not in reason:
                self.speed_stop(brake_accel)
        if abort is not None:
            safe = np.array(base_pose, dtype=float)
            safe[2] = z_contact + 0.008
            self.move_l(safe, speed=0.05, accel=0.5)
        self.last_impact_n = 0.0
        return np.asarray(rows).reshape(-1, len(ROBOT_SAMPLE_FIELDS)), abort, reason

    def measure_force_baseline(self, n: int = 50) -> float:
        fz = []
        for _ in range(n):
            fz.append(self.sample(np.nan)[ROBOT_SAMPLE_FIELDS.index("fz") - 1])
            time.sleep(self.params.dt)
        return float(np.median(fz))


# ============================================================================ real robot


class URRobot(RobotBase):
    def __init__(self, ip: str, params: ServoParams, tcp: Optional[list] = None, rtde_frequency: float = -1.0, auto_hz: bool = True):
        import rtde_control
        import rtde_receive

        self.params = params
        self.ip = ip
        self.rtde_r = rtde_receive.RTDEReceiveInterface(ip, rtde_frequency)
        self.rtde_c = rtde_control.RTDEControlInterface(ip, rtde_frequency)
        if tcp is not None:
            self.rtde_c.setTcp(list(map(float, tcp)))
        # measure the controller cycle (500 Hz e-series, 125 Hz CB3) so the streamed trajectory is sampled correctly
        t0 = time.perf_counter()
        for _ in range(50):
            tp = self.rtde_c.initPeriod()
            self.rtde_c.waitPeriod(tp)
        measured_hz = 50.0 / (time.perf_counter() - t0)
        if auto_hz:
            params.hz = 500.0 if measured_hz > 300 else 125.0
        elif abs(measured_hz - params.hz) / params.hz > 0.2:
            print(f"[UR] WARNING: --hz {params.hz:.0f} but the controller cycles at ~{measured_hz:.0f} Hz; trajectories will run at the wrong speed")
        print(f"[UR] connected to {ip}; control rate {params.hz:.0f} Hz (measured {measured_hz:.0f}); TCP offset = {np.round(self.rtde_c.getTCPOffset(), 4).tolist()}")

    def tcp_pose(self) -> np.ndarray:
        return np.asarray(self.rtde_r.getActualTCPPose(), dtype=float)

    def move_l(self, pose, speed=0.1, accel=0.5) -> None:
        ok = self.rtde_c.moveL(list(map(float, pose)), float(speed), float(accel))
        if not ok:
            raise RobotAbort("moveL failed (protective stop / not in remote control?)")

    def sample(self, target_z: float) -> list:
        r = self.rtde_r
        return [r.getTimestamp(), target_z] + list(r.getActualTCPPose()) + list(r.getActualTCPSpeed()) + list(r.getActualTCPForce()) + list(r.getActualQ()) + list(r.getActualQd())

    def servo_begin(self) -> None:
        if self.rtde_r.isProtectiveStopped():
            raise RobotAbort("robot is in protective stop")

    def servo_step(self, pose: np.ndarray) -> None:
        t0 = self.rtde_c.initPeriod()
        self.rtde_c.servoL(list(map(float, pose)), 0.0, 0.0, self.params.dt, self.params.lookahead, self.params.gain)
        self.rtde_c.waitPeriod(t0)

    def servo_end(self) -> None:
        self.rtde_c.servoStop(float(self.stop_decel))

    def speed_step(self, v_xyz: np.ndarray, accel: float) -> None:
        # same pattern as the ur_rtde speed examples: speedL(..., time=dt) inside initPeriod/waitPeriod
        t0 = self.rtde_c.initPeriod()
        self.rtde_c.speedL([float(v_xyz[0]), float(v_xyz[1]), float(v_xyz[2]), 0.0, 0.0, 0.0], float(accel), self.params.dt)
        self.rtde_c.waitPeriod(t0)

    def speed_stop(self, decel: float) -> None:
        self.rtde_c.speedStop(float(decel))

    def ensure_running(self) -> None:
        """Re-upload the ur_rtde control script if it died (e.g. after a runtime error / protective stop)."""
        if not self.rtde_c.isConnected():
            print("[UR] reconnecting RTDE control interface")
            self.rtde_c.reconnect()
        if not self.rtde_c.isProgramRunning():
            print("[UR] control script not running - re-uploading it")
            self.rtde_c.reuploadScript()
            time.sleep(0.5)
        if self.rtde_r.isProtectiveStopped():
            raise RobotAbort("robot is in protective stop - clear it on the pendant")

    def zero_ft(self) -> None:
        try:
            self.rtde_c.zeroFtSensor()
        except Exception:
            pass  # CB3: no FT sensor

    def freedrive(self, enable: bool) -> None:
        if enable:
            self.rtde_c.teachMode()
        else:
            self.rtde_c.endTeachMode()

    def stop(self) -> None:
        try:
            self.rtde_c.servoStop(2.0)
            self.rtde_c.speedStop(2.0)
        except Exception:
            pass

    def close(self) -> None:
        self.stop()
        try:
            self.rtde_c.stopScript()
        finally:
            self.rtde_c.disconnect()
            self.rtde_r.disconnect()


# ============================================================================ simulator


class SimRobot(RobotBase):
    """Kinematic stand-in: the TCP follows the servo target with a first-order lag and drives a SimPiano.

    ``surface_z(xy) -> (note, z_surface)`` gives the true key surface under the fingertip; the sim
    piano gets ``depth = z_surface - tcp_z``. Force: ~0.6 N once the key moves, a stiff spring at the
    key bottom (10.5 mm), plus sensor noise.
    """

    def __init__(self, params: ServoParams, surface_fn: Callable[[np.ndarray], tuple], sim_piano, tau: float = 0.012, realtime: bool = False, noise_n: float = 0.15):
        self.params = params
        self.surface_fn = surface_fn
        self.piano = sim_piano
        self.tau = tau
        self.realtime = realtime
        self.noise_n = noise_n
        self.pose = np.array([0.45, -0.3, 0.3, np.pi, 0, 0], dtype=float)
        self.vel = np.zeros(6)
        self.t = now()
        self.rng = np.random.default_rng(0)

    def _physics(self, target: np.ndarray, dt: float) -> None:
        # first-order tracking with a velocity cap (UR5 max ~1 m/s)
        new = self.pose + (target - self.pose) * min(1.0, dt / self.tau)
        step = new[:3] - self.pose[:3]
        vmax = 1.0 * dt
        n = np.linalg.norm(step)
        if n > vmax:
            step *= vmax / n
        new[:3] = self.pose[:3] + step
        self.vel = np.concatenate([(new[:3] - self.pose[:3]) / dt, np.zeros(3)])
        self.pose = new
        note, zs = self.surface_fn(self.pose[:2])
        if note is not None:
            self.piano.update(note, zs - self.pose[2], self.t)

    def _force(self) -> np.ndarray:
        note, zs = self.surface_fn(self.pose[:2])
        fz = 0.0
        if note is not None:
            depth = zs - self.pose[2]
            if depth > 0:
                fz = 0.6 + 60.0 * depth  # key weight + light spring
            if depth > 0.0100:
                fz += 8000.0 * (depth - 0.0100)  # key bed (~10 mm dip) + rubber tip compliance
        return np.array([0, 0, fz, 0, 0, 0]) + self.rng.normal(0, self.noise_n, 6)

    # ---- interface
    def tcp_pose(self) -> np.ndarray:
        return self.pose.copy()

    def move_l(self, pose, speed=0.1, accel=0.5) -> None:
        pose = np.asarray(pose, dtype=float)
        dist = np.linalg.norm(pose[:3] - self.pose[:3])
        n = max(int(dist / speed / self.params.dt), 1)
        for i in range(1, n + 1):
            tgt = self.pose + (pose - self.pose) * (1.0 / (n - i + 1))
            self._advance(tgt)
        self.pose = pose.copy()
        self.vel[:] = 0

    def _advance(self, target: np.ndarray) -> None:
        dt = self.params.dt
        if self.realtime:
            time.sleep(dt)
            self.t = now()
        else:
            self.t += dt
        self._physics(target, dt)

    def sample(self, target_z: float) -> list:
        f = self._force()
        q = np.zeros(6)
        return [self.t, target_z] + list(self.pose) + list(self.vel) + list(f) + list(q) + list(q)

    def servo_step(self, pose: np.ndarray) -> None:
        self._advance(np.asarray(pose, dtype=float))

    def measure_force_baseline(self, n: int = 50) -> float:
        return float(np.median([self._force()[2] for _ in range(n)]))

    def pause(self, seconds: float) -> None:
        for _ in range(int(seconds / self.params.dt)):
            if np.linalg.norm(self.vel[:3]) > 1e-6:  # still braking / coasting under velocity control
                self._velocity_advance(np.zeros(3), self._last_decel)
            else:
                self._advance(self.pose.copy())

    _last_decel = 10.0

    def _velocity_advance(self, v_cmd: np.ndarray, accel: float) -> None:
        dt = self.params.dt
        dv = v_cmd - self.vel[:3]
        n = np.linalg.norm(dv)
        if n > accel * dt:
            dv *= accel * dt / n
        self.vel[:3] += dv
        target = self.pose.copy()
        target[:3] += self.vel[:3] * dt
        if self.realtime:
            time.sleep(dt)
            self.t = now()
        else:
            self.t += dt
        self.pose = target
        note, zs = self.surface_fn(self.pose[:2])
        if note is not None:
            self.piano.update(note, zs - self.pose[2], self.t)

    def speed_step(self, v_xyz: np.ndarray, accel: float) -> None:
        self._velocity_advance(np.asarray(v_xyz, dtype=float), accel)

    def speed_stop(self, decel: float) -> None:
        self._last_decel = decel
        if np.linalg.norm(self.vel[:3]) > 1e-6:
            self._velocity_advance(np.zeros(3), decel)


@dataclass
class PressMethod:
    """How a press is executed.
    ``servo``: stream the time-parametrised depth trajectory with servoL (smooth, good up to ~0.15 m/s).
    ``speedl``: closed-loop velocity control on the measured depth (tracks fast profiles).
    Both descend until the F/T sensor sees the key bed (|fz - baseline| > stop_force_n) or the
    profile's depth_total (fallback), then brake, dwell ``hold_s`` and lift slowly."""

    name: str = "servo"
    accel: float = 8.0  # tool acceleration (speedl) / profile ramp (servo)
    brake_accel: float = 25.0  # deceleration once the key bed is felt (braking distance v^2/2a: 0.3 m/s -> 1.8 mm)
    stop_force_n: float = 8.0  # force step above baseline that means "key bed reached" (a fast-moving GHS key alone pushes back 3-4 N)
    stop_force_min_depth_m: float = 0.009  # force stop is a fallback only: armed just above the key bed, after the note-on switch (~8 mm)
    after_note: str = "lift"  # what to do when the piano reports note-on: "lift" = reverse immediately and rise fast, "hold" = brake, dwell, lift slowly
    lift_fast_speed: float = 0.25  # m/s used by after_note="lift"
    brake_at_depth_m: Optional[float] = None  # speedl: reverse as soon as the finger passes this depth (just past B) instead of waiting for the MIDI note


def execute_press(robot: RobotBase, hover: np.ndarray, start: np.ndarray, z_contact: float, prof, method: PressMethod, force_baseline: float, extra_stop: Optional[Callable[[], Optional[str]]] = None):
    """Run one press from ``start`` (already there): descend, stop on force/depth (or when
    ``extra_stop()`` returns a reason, e.g. "note-on received"), hold, lift back to ``start``.

    Returns (samples, abort_reason, stop_reason).
    """
    fz_i, z_i = ROBOT_SAMPLE_FIELDS.index("fz"), ROBOT_SAMPLE_FIELDS.index("tcp_z")
    robot.stop_decel = method.brake_accel
    if prof.kind == "staged":
        method = PressMethod(**{**asdict(method), "name": "speedl"})
    if method.name == "servo":
        _, depth = prof.depth_trajectory(robot.params.dt, descent_only=True)
        hit = {}

        def stop_when(row):
            d = z_contact - row[z_i]
            dF = abs(row[fz_i] - force_baseline)
            if d >= method.stop_force_min_depth_m and dF > method.stop_force_n:
                hit["reason"] = f"force {dF:.1f} N at {d * 1e3:.1f} mm"
                return True
            if extra_stop is not None:
                r = extra_stop()
                if r:
                    hit["reason"] = f"{r} at {d * 1e3:.1f} mm"
                    return True
            return False

        samples, abort = robot.stream_z(hover, z_contact - depth, z_contact=z_contact, stop_when=stop_when, force_baseline=force_baseline, settle=False)
        reason = hit.get("reason", f"depth limit {prof.depth_total_m * 1e3:.1f} mm")
        if abort is None and reason.startswith("note-on") and method.after_note == "lift":
            rows = robot.retreat(float(start[2]), method.lift_fast_speed, method.brake_accel)
            samples = np.vstack([samples, np.asarray(rows).reshape(-1, len(ROBOT_SAMPLE_FIELDS))])
            reason += f", lifted fast to {(z_contact - start[2]) * 1e3:+.1f} mm"
            robot.last_impact_n = 0.0
            return samples, abort, reason
        if abort is None:
            rows = robot.settle_rows()
            samples = np.vstack([samples, np.asarray(rows).reshape(-1, len(ROBOT_SAMPLE_FIELDS))])
    elif prof.kind == "staged":
        retreat = (float(start[2]), method.lift_fast_speed, method.brake_accel) if method.after_note == "lift" else None
        samples, abort, reason = robot.press_stages(
            hover,
            z_contact,
            prof.stages,
            accel=method.accel,
            brake_accel=method.brake_accel,
            force_baseline=force_baseline,
            stop_force_n=method.stop_force_n,
            stop_force_min_depth=method.stop_force_min_depth_m,
            extra_stop=extra_stop,
            retreat=retreat,
        )
        if abort is None and "lifted" in reason:
            return samples, abort, reason
    elif method.name == "speedl":
        retreat = (float(start[2]), method.lift_fast_speed, method.brake_accel) if method.after_note == "lift" else None
        samples, abort, reason = robot.press_speedl(
            hover,
            z_contact,
            prof.speed_at_depth,
            depth_limit=prof.depth_total_m,
            accel=method.accel,
            brake_accel=method.brake_accel,
            force_baseline=force_baseline,
            stop_force_n=method.stop_force_n,
            extra_stop=extra_stop,
            stop_force_min_depth=method.stop_force_min_depth_m,
            retreat=retreat,
            brake_at_depth=method.brake_at_depth_m,
        )
        if abort is None and "lifted" in reason:
            if robot.last_impact_n > robot.params.force_abort_n * 0.6:
                print(f"        !! key-bed impact {robot.last_impact_n:.1f} N")
            robot.last_impact_n = 0.0
            return samples, abort, reason
    else:
        raise ValueError(method.name)
    if abort is None:
        if robot.last_impact_n > robot.params.force_abort_n:
            print(f"        !! key-bed impact {robot.last_impact_n:.1f} N exceeds --force-abort-n: lifting without hold (softer tip, higher --brake-accel or lower speed)")
        else:
            robot.pause(prof.hold_s)
        # slow, *sampled* lift (velocity control) so the depth at note-off is recorded too
        rows = robot.retreat(float(start[2]), prof.lift_speed, 2.0)
        samples = np.vstack([samples, np.asarray(rows).reshape(-1, len(ROBOT_SAMPLE_FIELDS))])
    robot.last_impact_n = 0.0
    return samples, abort, reason


def patch_sim_time(robot: "SimRobot"):
    """Make ``common.now()`` return the simulated clock so MIDI and robot timestamps agree in --sim."""
    from key_velocity_exp import common

    common._clock = lambda: robot.t  # noqa: E731
