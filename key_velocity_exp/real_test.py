#!/usr/bin/env python
"""Hands-on key-press test on the real UR5 + digital piano.

No automatic contact search: jog the fingertip (freedrive) until it *just touches* the key you want,
then start this script (or press ``h`` in interactive mode). That pose is the *home*: presses go
down from home by ``depth`` mm, and the home can be shifted left/right by whole keys or deeper.

  # identify the controller first
  kve-probe <robot-ip>

  # interactive session (recommended for the first tests)
  kve-test --robot-ip <ip> interactive

  # scripted speed sweep on the current key, recorded for analyze.py
  kve-test --robot-ip <ip> sweep \
      --speeds 0.03 0.05 0.1 0.2 0.3 --repeats 2 --start-above-mm 0 6 --out data/real_C4

Interactive commands
  h                 set home = current TCP pose (do this after jogging in freedrive)
  f                 toggle freedrive on/off
  w                 where am I: pose relative to home, TCP force, recent MIDI
  p V [DEPTH] [SA]  servoL press: lift, accelerate during the run-up (SA mm above home, default
                    v^2/2a + 1 mm), touch the key at constant speed V m/s, keep going until the F/T
                    sensor feels the key bed (--stop-force-n) or DEPTH mm (fallback), hold, lift slowly.
                    SA=0 forces a start from the touching position (accelerates inside the key).
  x V [DEPTH]       same, but with velocity control (speedL) - tracks V > 0.15 m/s properly
  k V1 [V0] [A]     "kick": touch the key at V0 (default min(0.05, V1/4) m/s) and accelerate INSIDE the key
                    at A m/s^2 (default --kick-accel) up to V1 - avoids launching the key with a hard impact
  th V1 [V0] [A] [R] "throw": like k, but release (brake hard) at depth R mm (default --release-depth-mm):
                    the light key flies on through the switches alone, the heavy arm never reaches the
                    key bed at speed - the way to reach the top velocities
  The descent also stops as soon as the piano sends note-on (--no-stop-on-note disables), which
  prevents a bounced key from being pressed a second time.
  a V0 V1 [DEPTH]   accelerating (V0<V1) / decelerating servoL press through the key travel
  xa V0 V1 [DEPTH]  same with velocity control
  s V1 V2 ...       sweep: one constant-speed press per listed speed
  l [N] / r [N]     shift home N keys (default 1) left / right along --key-axis (23.5 mm per white key)
  d MM / u MM       shift home down / up by MM (e.g. d 0.5 = a deeper contact reference)
  m DX DY DZ        shift home by mm in robot base x y z
  c N               calibrate key axis: jog to a key N keys to the RIGHT of home, then run this
  g                 go to home (via hover)
  t [SEC]           just listen to MIDI for SEC seconds (test the piano connection)
  q                 quit: returns to home (on the key) so the next run can start from it
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict

import numpy as np

from key_velocity_exp import common
from key_velocity_exp.common import WHITE_KEY_PITCH_M, PressProfile, TrialRecord, is_black, midi_to_name, now
from key_velocity_exp.ur5_io import PressMethod, RobotAbort, ServoParams, SimRobot, URRobot, execute_press

AXES = {"+x": np.array([1.0, 0, 0]), "-x": np.array([-1.0, 0, 0]), "+y": np.array([0, 1.0, 0]), "-y": np.array([0, -1.0, 0])}


class Session:
    def __init__(self, robot, midi, args):
        self.robot, self.midi, self.args = robot, midi, args
        self.home = np.asarray(robot.tcp_pose(), dtype=float)
        self.key_axis = AXES[args.key_axis].copy()
        self.key_pitch = WHITE_KEY_PITCH_M
        self.key_offset = 0  # keys moved from the original home (bookkeeping only)
        self.trial_id = 0
        self.out = args.out
        os.makedirs(self.out, exist_ok=True)
        self.params = robot.params
        self.method = PressMethod(
            name=args.press_method,
            accel=args.accel,
            brake_accel=args.brake_accel,
            stop_force_n=args.stop_force_n,
            stop_force_min_depth_m=args.stop_force_min_depth_mm * 1e-3,
            after_note=args.after_note,
            lift_fast_speed=args.lift_fast_speed,
        )
        print(f"[home] {np.round(self.home, 4).tolist()}  (fingertip should be just touching the key)")

    # ------------------------------------------------------------------ geometry helpers
    def hover(self) -> np.ndarray:
        p = self.home.copy()
        p[2] += self.args.hover_mm * 1e-3
        return p

    def set_home_here(self):
        self.home = np.asarray(self.robot.tcp_pose(), dtype=float)
        print(f"[home] set to {np.round(self.home, 4).tolist()}")

    def shift_home(self, delta_xyz_m: np.ndarray, note: str = ""):
        self.home[:3] += delta_xyz_m
        print(f"[home] shifted {np.round(delta_xyz_m * 1e3, 2).tolist()} mm {note} -> {np.round(self.home, 4).tolist()}")

    def shift_keys(self, n: int):
        self.shift_home(self.key_axis * self.key_pitch * n, note=f"({n:+d} keys)")
        self.key_offset += n
        self.goto_home()

    def calibrate_axis(self, n_keys: int):
        p = np.asarray(self.robot.tcp_pose(), dtype=float)
        d = p[:3] - self.home[:3]
        d[2] = 0.0
        if np.linalg.norm(d) < 0.005:
            print("[calib] move the fingertip to another key first (freedrive: f)")
            return
        self.key_axis = d / np.linalg.norm(d)
        self.key_pitch = float(np.linalg.norm(d) / n_keys)
        print(f"[calib] key axis = {np.round(self.key_axis, 3).tolist()}, pitch = {self.key_pitch * 1e3:.2f} mm/key (standard 23.5)")

    def goto_home(self):
        self.robot.move_l(self.hover(), speed=0.1, accel=0.5)
        self.robot.move_l(self.home, speed=0.03, accel=0.3)

    # ------------------------------------------------------------------ one press
    def press(self, prof: PressProfile, label: str = "", method: PressMethod | None = None) -> TrialRecord:
        method = method or self.method
        z_contact = float(self.home[2])
        hover = self.hover()
        start = self.home.copy()
        start[2] += prof.start_above_m
        self.robot.ensure_running()
        self.robot.move_l(hover, speed=0.1, accel=0.5)
        self.robot.move_l(start, speed=0.03, accel=0.3)
        self.robot.pause(self.args.settle_s)
        baseline = self.robot.measure_force_baseline()
        self.midi.drain()

        t_start = now()
        extra_stop = None
        if self.args.stop_on_note:

            def extra_stop():
                ev = [e for e in self.midi.events_between(t_start, now()) if e.kind == "note_on"]
                return f"note-on vel {ev[0].velocity}" if ev else None

        samples, abort, reason = execute_press(self.robot, hover, start, z_contact, prof, method, baseline, extra_stop=extra_stop)
        self.robot.pause(self.args.post_wait_s)
        t_end = now()
        events = self.midi.events_between(t_start - 0.05, t_end, clear=True)
        self.robot.move_l(hover, speed=0.1, accel=0.5)

        ons = [e for e in events if e.kind == "note_on"]
        note = ons[0].note if ons else (self.args.note if self.args.note is not None else -1)
        rec = TrialRecord(
            trial_id=self.trial_id,
            midi_note=note,
            key_name=midi_to_name(note) if note >= 0 else "?",
            key_is_black=is_black(note) if note >= 0 else False,
            profile=prof,
            key_pose_hover=list(map(float, hover)),
            z_contact=z_contact,
            t_start=t_start,
            t_end=t_end,
            robot_samples=samples,
            midi_events=events,
            aborted=abort is not None,
            abort_reason=abort or "",
            extra={
                "force_baseline": baseline,
                "home": self.home.tolist(),
                "key_offset": self.key_offset,
                "touch_setting": self.args.touch_setting,
                "hz": self.params.hz,
                "label": label,
                "method": asdict(method),
                "stop_reason": reason,
            },
        )
        rec.save(self.out)
        self.trial_id += 1
        self.report(rec)
        return rec

    def report(self, rec: TrialRecord):
        ons = [e for e in rec.midi_events if e.kind == "note_on"]
        if rec.aborted:
            print(f"  [{rec.trial_id:3d}] {rec.profile.label():26s} ABORT: {rec.abort_reason}")
            return
        if not ons:
            print(
                f"  [{rec.trial_id:3d}] {rec.profile.label():26s} no note-on  (max depth {rec.depth().max() * 1e3:.1f} mm, max |dF| {np.abs(rec.col('fz') - rec.extra['force_baseline']).max():.1f} N)"
            )
            return
        t, d = rec.col("t"), rec.depth()
        vz = -rec.col("vz")
        v_cmd = rec.profile.speed_at_depth(0.0)

        # mean speed between 3 and 7 mm (roughly the inter-switch region) from the measured depth
        def t_at(depth_m):
            idx = np.flatnonzero(d >= depth_m)  # first crossing
            if len(idx) == 0 or idx[0] == 0:
                return np.nan
            i = idx[0]
            w = (depth_m - d[i - 1]) / (d[i] - d[i - 1]) if d[i] != d[i - 1] else 0.0
            return t[i - 1] + w * (t[i] - t[i - 1])

        t3, t7 = t_at(0.003), t_at(0.007)
        v37 = 0.004 / (t7 - t3) if np.isfinite(t3) and np.isfinite(t7) and t7 > t3 else np.nan
        meth = rec.extra.get("method", {}).get("name", "servo")
        for e in ons:
            d_on = np.interp(e.t, t, d) * 1e3
            v_on = np.interp(e.t, t, vz)
            print(
                f"  [{rec.trial_id:3d}] {rec.profile.label():22s} {meth:6s} {midi_to_name(e.note):4s} vel={e.velocity:3d}  depth@on={d_on:5.2f} mm  v@on={v_on:.3f}  v(3-7mm)={v37:.3f}  peak={vz.max():.3f} m/s  t_on={(e.t - rec.t_start) * 1e3:6.1f} ms  stop: {rec.extra.get('stop_reason', '')}  |dF|max={np.abs(rec.col('fz') - rec.extra['force_baseline']).max():.1f} N"
            )
        d_first = np.interp(ons[0].t, t, d) * 1e3
        if d_first < 6.0:
            print(
                f"        !! note-on while the fingertip was only {d_first:.1f} mm deep: the key was launched ahead of the finger by the impact - touch slower and accelerate inside the key (k V1 V0)"
            )
        if rec.profile.kind == "const" and np.isfinite(v37) and v37 < 0.8 * v_cmd:
            print(f"        !! tracking: commanded {v_cmd:.3f} m/s but only {v37:.3f} m/s through 3-7 mm - increase run-up (SA), use x (speedL) or a higher --accel")
        offs = [e for e in rec.midi_events if e.kind == "note_off"]
        if len(ons) > 1:
            same = all(e.note == ons[0].note for e in ons)
            print(
                f"        !! {len(ons)} note-ons: {[(midi_to_name(e.note), e.velocity) for e in ons]} " + ("- key bounced off the key bed and was pressed again" if same else "(hit a neighbour key?)")
            )
        if not offs:
            print("        !! no note-off received (key still down or lift too short)")

    def const(self, v: float, depth_mm=None, start_above_mm=None) -> PressProfile:
        p = PressProfile("const", v0=v, v1=v)
        p.depth_total_m = (depth_mm if depth_mm is not None else self.args.depth_mm) * 1e-3
        p.accel_max = self.args.accel
        need = 1.5 * v**2 / (2 * p.accel_max) + 0.002  # run-up to be at speed v when touching the key (1.5x margin: the velocity loop lags)
        if start_above_mm is None:
            p.start_above_m = need  # always at speed before contact (SA=0 to start from the touching position)
        elif start_above_mm == 0:
            p.start_above_m = 0.0
        else:
            p.start_above_m = max(start_above_mm * 1e-3, need)
        return p

    # ------------------------------------------------------------------ status
    def where(self):
        p = np.asarray(self.robot.tcp_pose(), dtype=float)
        rel = (p[:3] - self.home[:3]) * 1e3
        row = self.robot.sample(np.nan)
        i = common.ROBOT_SAMPLE_FIELDS.index("fx") - 1  # sample() rows have no leading monotonic time
        f = row[i : i + 3]
        print(
            f"[where] TCP {np.round(p, 4).tolist()}\n        rel. to home dx={rel[0]:+.2f} dy={rel[1]:+.2f} dz={rel[2]:+.2f} mm   key offset {self.key_offset:+d}\n        TCP force xyz = {np.round(f, 2).tolist()} N"
        )
        ev = self.midi.drain()
        if ev:
            print("        recent MIDI:", [(e.kind, midi_to_name(e.note), e.velocity) for e in ev[-8:]])

    def listen(self, sec: float):
        print(f"[midi] listening {sec:.0f} s on '{getattr(self.midi, 'port_name', 'sim')}' - play some keys")
        self.midi.drain()
        t0 = time.time()
        while time.time() - t0 < sec:
            time.sleep(0.1)
            for e in self.midi.drain():
                print(f"   {e.kind:8s} {midi_to_name(e.note):4s} vel={e.velocity}")


# ---------------------------------------------------------------------------- modes


def interactive(s: Session):
    freedrive = False
    print(__doc__.split("Interactive commands")[1])
    while True:
        try:
            line = input("kv> ").strip()
        except (EOFError, KeyboardInterrupt):
            line = "q"
        if not line:
            continue
        cmd, *a = line.split()
        try:
            if cmd == "q":
                break
            elif cmd == "h":
                if freedrive:
                    s.robot.freedrive(False)
                    freedrive = False
                s.set_home_here()
            elif cmd == "f":
                freedrive = not freedrive
                s.robot.freedrive(freedrive)
                print(f"[freedrive] {'ON - move the arm by hand' if freedrive else 'OFF'}")
            elif cmd == "w":
                s.where()
            elif cmd == "p":
                v = float(a[0])
                s.press(s.const(v, float(a[1]) if len(a) > 1 else None, float(a[2]) if len(a) > 2 else None))
            elif cmd == "x":
                v = float(a[0])
                fast = PressMethod("speedl", s.args.accel, s.args.brake_accel, s.args.stop_force_n, s.args.stop_force_min_depth_mm * 1e-3, s.args.after_note, s.args.lift_fast_speed)
                s.press(s.const(v, float(a[1]) if len(a) > 1 else None, None), method=fast)
            elif cmd in ("k", "th"):
                v1 = float(a[0])
                v0 = float(a[1]) if len(a) > 1 else min(0.05, v1 / 4)
                acc = float(a[2]) if len(a) > 2 else s.args.kick_accel
                p = PressProfile("kick" if cmd == "k" else "throw", v0=v0, v1=v1, accel_max=acc)
                if cmd == "th":
                    p.depth_mid_m = (float(a[3]) if len(a) > 3 else s.args.release_depth_mm) * 1e-3
                p.depth_total_m = s.args.depth_mm * 1e-3
                p.start_above_m = 1.5 * v0**2 / (2 * s.args.accel) + 0.002
                fast = PressMethod("speedl", max(acc, s.args.accel), s.args.brake_accel, s.args.stop_force_n, s.args.stop_force_min_depth_mm * 1e-3, s.args.after_note, s.args.lift_fast_speed)
                s.press(p, method=fast)
            elif cmd in ("a", "xa"):
                v0, v1 = float(a[0]), float(a[1])
                p = PressProfile("accel" if v1 > v0 else "decel", v0=v0, v1=v1, accel_max=s.args.accel)
                p.depth_total_m = (float(a[2]) if len(a) > 2 else s.args.depth_mm) * 1e-3
                p.start_above_m = v0**2 / (2 * p.accel_max) + 0.001
                fast = PressMethod("speedl", s.args.accel, s.args.brake_accel, s.args.stop_force_n, s.args.stop_force_min_depth_mm * 1e-3, s.args.after_note, s.args.lift_fast_speed)
                s.press(p, method=fast if cmd == "xa" else None)
            elif cmd == "s":
                for v in map(float, a):
                    s.press(s.const(v))
            elif cmd in ("l", "r"):
                n = int(a[0]) if a else 1
                s.shift_keys(-n if cmd == "l" else n)
            elif cmd in ("d", "u"):
                mm = float(a[0])
                s.shift_home(np.array([0, 0, (-mm if cmd == "d" else mm) * 1e-3]))
            elif cmd == "m":
                s.shift_home(np.array(list(map(float, a[:3]))) * 1e-3)
            elif cmd == "c":
                s.calibrate_axis(int(a[0]) if a else 1)
            elif cmd == "g":
                s.goto_home()
            elif cmd == "t":
                s.listen(float(a[0]) if a else 5.0)
            else:
                print("unknown command; see the list above")
        except RobotAbort as e:
            print(f"[robot] {e}")
        except (ValueError, IndexError) as e:
            print(f"bad arguments: {e}")


def sweep(s: Session):
    args = s.args
    plan = [(v, sa, r) for r in range(args.repeats) for sa in args.start_above_mm for v in args.speeds]
    print(f"[sweep] {len(plan)} presses on the current key: speeds {args.speeds} m/s, start above {args.start_above_mm} mm, {args.repeats} repeats")
    for v, sa, r in plan:
        s.press(s.const(v, start_above_mm=sa), label=f"rep{r}")


def build(args):
    params = ServoParams(hz=args.hz or 500.0, lookahead=args.lookahead, gain=args.gain, force_abort_n=args.force_abort_n)
    params.depth_abort_m = (args.depth_mm + 3.0) * 1e-3
    if args.sim:
        from key_velocity_exp.layout import KeyLayout
        from key_velocity_exp.midi_io import SimPiano
        from key_velocity_exp.ur5_io import patch_sim_time

        layout = KeyLayout.synthetic()

        def surface_fn(xy):
            n = layout.nearest_key(xy)
            return (None, 0.0) if n is None else (n, layout.nominal_contact_pose(n)[2])

        piano = SimPiano()
        robot = SimRobot(params, surface_fn, piano, realtime=False)
        patch_sim_time(robot)
        robot.pose = layout.nominal_contact_pose(60)  # as if the user had jogged the tip onto C4
        return robot, piano
    from key_velocity_exp.midi_io import MidiListener

    midi = MidiListener(args.midi_port)
    print(f"[midi] listening on '{midi.port_name}'")
    robot = URRobot(args.robot_ip, params, tcp=args.tcp, rtde_frequency=args.hz or -1.0, auto_hz=args.hz is None)
    return robot, midi


def build_parser(doc: str = __doc__):
    p = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", nargs="?", default="interactive", choices=["interactive", "sweep", "press"])
    p.add_argument("--sim", action="store_true", help="dry run with the simulator instead of hardware")
    common.add_connection_args(p)
    p.add_argument("--hz", type=float, default=None, help="control rate; default: measured from the controller")
    p.add_argument("--tcp", type=float, nargs=6, default=None, help="fingertip TCP offset x y z rx ry rz (else keep the robot's current TCP)")
    p.add_argument("--lookahead", type=float, default=0.03)
    p.add_argument("--gain", type=float, default=1500)
    p.add_argument("--force-abort-n", type=float, default=15.0)
    p.add_argument("--press-method", default="servo", choices=["servo", "speedl"], help="default method for p/a/sweep (x/xa always use speedl)")
    p.add_argument("--accel", type=float, default=8.0, help="tool acceleration m/s^2 (profile ramp for servo, speedL accel for speedl)")
    p.add_argument("--brake-accel", type=float, default=25.0, help="deceleration once the key bed is felt (m/s^2)")
    p.add_argument("--stop-force-n", type=float, default=8.0, help="fallback: force step above baseline that ends the descent (key bed); a fast GHS key alone pushes back 3-4 N")
    p.add_argument("--kick-accel", type=float, default=20.0, help="k command: in-key acceleration m/s^2")
    p.add_argument("--release-depth-mm", type=float, default=4.0, help="th command: depth at which the arm lets go of the key")
    p.add_argument(
        "--stop-force-min-depth-mm", type=float, default=9.0, help="force stop only counts below this depth (after the ~8 mm note-on switch; tool inertia and key reaction look like contact force)"
    )
    p.add_argument(
        "--after-note", default="lift", choices=["lift", "hold"], help="lift = rise immediately when note-on arrives (prevents re-triggering by the bouncing key); hold = brake, dwell, lift slowly"
    )
    p.add_argument("--lift-fast-speed", type=float, default=0.25, help="rise speed for --after-note lift")
    p.add_argument("--no-stop-on-note", dest="stop_on_note", action="store_false", help="keep descending after note-on (default: stop immediately)")
    p.add_argument("--key-axis", default="+y", choices=list(AXES), help="robot-base direction of 'one key to the right' (or use interactive 'c')")
    p.add_argument("--hover-mm", type=float, default=12.0)
    p.add_argument("--depth-mm", type=float, default=10.5, help="fallback depth limit below home if the key bed is not felt (P-45 key dip ~10 mm)")
    p.add_argument("--settle-s", type=float, default=0.4)
    p.add_argument("--post-wait-s", type=float, default=0.4)
    p.add_argument("--note", type=int, default=None, help="MIDI note under the fingertip (only used if the piano stays silent)")
    p.add_argument("--touch-setting", default="medium")
    p.add_argument("--out", default=None)
    # sweep / press
    p.add_argument("--speeds", type=float, nargs="*", default=[0.03, 0.05, 0.08, 0.12, 0.2, 0.3])
    p.add_argument("--start-above-mm", type=float, nargs="*", default=[0.0])
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--speed", type=float, default=0.05, help="press mode: constant speed")
    return p


def parse_args():
    p = build_parser()
    a = p.parse_args()
    common.check_connection_args(p, a)
    a.out = a.out or os.path.join("data", "real_test", time.strftime("%Y%m%d_%H%M%S"))
    return a


def main():
    args = parse_args()
    robot, midi = build(args)
    s = Session(robot, midi, args)
    with open(os.path.join(s.out, "session.json"), "w") as f:
        json.dump({"args": vars(args), "params": asdict(robot.params), "home": s.home.tolist()}, f, indent=2, default=str)
    normal_exit = False
    try:
        if args.mode == "interactive":
            interactive(s)
        elif args.mode == "sweep":
            sweep(s)
        elif args.mode == "press":
            s.press(s.const(args.speed, start_above_mm=args.start_above_mm[0]))
        normal_exit = True
    except KeyboardInterrupt:
        print("\n[abort] Ctrl-C")
    except RobotAbort as e:
        print(f"[abort] {e}")
    finally:
        try:
            robot.stop()
            if normal_exit:
                # finish *on* the key at home, so the next run can take the current pose as home directly
                s.goto_home()
                print(f"[exit] robot left at home {np.round(s.home, 4).tolist()}")
            else:
                robot.move_l(s.hover(), speed=0.1)  # after an abort stay clear of the key
        except Exception:
            pass
        robot.close()
        midi.close()
    if s.trial_id:
        from key_velocity_exp.analyze import summarize_session

        summarize_session(s.out, quiet=True)
        print(f"[done] {s.trial_id} presses saved in {s.out}  (summary.csv written; analyze.py {s.out} for fits/plots)")


if __name__ == "__main__":
    main()
