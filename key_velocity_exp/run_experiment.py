#!/usr/bin/env python
"""Fully automatic key-velocity experiment: UR5 presses piano keys with prescribed speed profiles
while the robot state (500/125 Hz) and the piano's MIDI output are recorded.

Typical use
-----------
  # 0) dry run of the whole pipeline without hardware
  kve-run --sim --preset quick --out data/kv_sim

  # 1) teach the keyboard layout once (robot in freedrive, fingertip just touching the key)
  kve-run --robot-ip <ip> --teach-layout key_layout.json

  # 2) run the experiment (contact height of each key is found automatically with the F/T sensor)
  kve-run --robot-ip <ip> --layout key_layout.json \
      --keys C2,E3,C4,C#4,G5,C6 --preset full --repeats 3 --touch-setting medium --out data/kv_medium

  # 3) analyse
  kve-analyze data/kv_medium
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import asdict

import numpy as np

from key_velocity_exp import common
from key_velocity_exp.common import PressProfile, TrialRecord, is_black, midi_to_name, name_to_midi, now
from key_velocity_exp.layout import KeyLayout, describe_keys
from key_velocity_exp.ur5_io import PressMethod, RobotAbort, ServoParams, SimRobot, URRobot, execute_press

# ----------------------------------------------------------------------------- experiment presets

CONST_SPEEDS = [0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 0.30, 0.40, 0.55]  # m/s descent speed


def build_profiles(preset: str, speeds: list[float], start_above: list[float], kick_v0: float = 0.05, kick_accel: float = 20.0) -> list[PressProfile]:
    """Every profile is instantiated for each ``start_above`` height (in metres)."""
    base: list[PressProfile] = []
    if preset in ("const", "quick", "full"):
        sp = speeds if preset != "quick" else [0.03, 0.08, 0.2, 0.45]
        base += [PressProfile("const", v0=v, v1=v) for v in sp]
    if preset in ("varied", "full", "quick"):
        pairs = [(0.03, 0.30), (0.05, 0.45), (0.30, 0.03), (0.45, 0.05)] if preset != "quick" else [(0.03, 0.30), (0.30, 0.03)]
        for v0, v1 in pairs:
            base.append(PressProfile("accel" if v1 > v0 else "decel", v0=v0, v1=v1))
        if preset != "quick":
            for v0, v1, dm in [(0.03, 0.30, 0.002), (0.03, 0.30, 0.004), (0.03, 0.30, 0.006), (0.30, 0.03, 0.004)]:
                base.append(PressProfile("two_stage", v0=v0, v1=v1, depth_mid_m=dm))
            base.append(PressProfile("sine", v0=0.03, v1=0.40))
    if preset == "kick":
        base += [PressProfile("kick", v0=min(kick_v0, v / 4), v1=v, accel_max=kick_accel) for v in speeds]
    if preset == "throw":
        base += [PressProfile("throw", v0=min(kick_v0, v / 4), v1=v, accel_max=kick_accel, depth_mid_m=0.004) for v in speeds]
    if not base:
        raise ValueError(f"unknown preset {preset}")
    out = []
    for sa in start_above:
        for p in base:
            q = PressProfile(**asdict(p))
            # a non-zero start height is meant to reach the contact speed *before* touching the key:
            # make sure the run-up is long enough (v^2 / 2a plus 1 mm of margin)
            q.start_above_m = sa if sa == 0 else max(sa, 1.5 * q.speed_at_depth(0.0) ** 2 / (2 * min(q.accel_max, 8.0)) + 0.002)
            out.append(q)
    return out


REGISTER_PRESETS = {
    # low / mid / high registers, white and black keys
    "registers": ["C2", "F#2", "C3", "C4", "C#4", "G4", "C5", "A#5", "C6"],
    "quick": ["C4", "C#4", "C2", "C6"],
    "middle": ["C4"],
}


# ----------------------------------------------------------------------------- contact search


def find_contact(robot, layout: KeyLayout, note: int, method: str, args, midi) -> float:
    hover = layout.hover_pose(note, args.hover_mm * 1e-3)
    nominal = layout.nominal_contact_pose(note)[2]
    robot.move_l(hover, speed=0.1, accel=0.5)
    robot.pause(0.2)
    if method == "nominal":
        return float(nominal)
    if method == "teach":
        print(f"  [teach] freedrive ON – place the fingertip just touching {midi_to_name(note)}, then press Enter")
        robot.freedrive(True)
        input()
        robot.freedrive(False)
        z = float(robot.tcp_pose()[2])
        robot.move_l(hover, speed=0.05)
        return z

    dt = robot.params.dt
    if method == "force":
        v_search, floor = 0.004, nominal - 0.006
        baseline = robot.measure_force_baseline()
        thr = args.contact_force_n
        z_targets = np.arange(hover[2], floor, -v_search * dt)
        fz_i, z_i = common.ROBOT_SAMPLE_FIELDS.index("fz"), common.ROBOT_SAMPLE_FIELDS.index("tcp_z")
        hit = {}
        recent: list = []  # median of the last 5 samples rejects F/T noise spikes

        def stop_when(row):
            recent.append(row[fz_i])
            if len(recent) > 5:
                recent.pop(0)
            if len(recent) == 5 and abs(float(np.median(recent)) - baseline) > thr:
                hit["z"] = row[z_i]
                return True
            return False

        robot.stream_z(hover, z_targets, z_contact=nominal, stop_when=stop_when, force_baseline=None)
        robot.move_l(hover, speed=0.05)
        if "z" not in hit:
            raise RobotAbort(f"no contact found above {floor:.4f} for {midi_to_name(note)}")
        return float(hit["z"])
    if method == "midi":
        v_search, floor = 0.02, nominal - 0.012
        z_targets = np.arange(hover[2], floor, -v_search * dt)
        t0 = now()
        z_i = common.ROBOT_SAMPLE_FIELDS.index("tcp_z")
        hit = {}

        def stop_when(row):
            ev = [e for e in midi.events_between(t0, now()) if e.kind == "note_on" and e.note == note]
            if ev:
                hit["z"] = row[z_i]
                return True
            return False

        robot.stream_z(hover, z_targets, z_contact=nominal, stop_when=stop_when, force_baseline=None)
        robot.move_l(hover, speed=0.05)
        if "z" not in hit:
            raise RobotAbort(f"no note-on while searching {midi_to_name(note)} (is the piano on / MIDI port right?)")
        # note-on fires at the 2nd switch; reference the contact height ``assumed_d2`` above it
        # (a constant offset per key – the analysis re-estimates the true switch depths).
        return float(hit["z"] + args.assumed_d2_mm * 1e-3 + v_search * 0.006)
    raise ValueError(method)


# ----------------------------------------------------------------------------- one trial


def run_trial(robot, layout: KeyLayout, midi, trial_id: int, note: int, prof: PressProfile, args) -> TrialRecord:
    z_contact = layout.contact_pose(note)[2]
    hover = layout.hover_pose(note, args.hover_mm * 1e-3)
    start = hover.copy()
    start[2] = z_contact + prof.start_above_m
    robot.ensure_running()
    robot.move_l(hover, speed=0.15, accel=0.8)
    robot.move_l(start, speed=0.03, accel=0.3)
    robot.pause(args.settle_s)
    baseline = robot.measure_force_baseline()
    midi.drain()

    method = PressMethod(
        name=args.press_method,
        accel=max(args.accel, prof.accel_max),
        brake_accel=args.brake_accel,
        stop_force_n=args.stop_force_n,
        stop_force_min_depth_m=args.stop_force_min_depth_mm * 1e-3,
        after_note=args.after_note,
        lift_fast_speed=args.lift_fast_speed,
    )
    t_start = now()
    extra_stop = None
    if args.stop_on_note:

        def extra_stop():
            ev = [e for e in midi.events_between(t_start, now()) if e.kind == "note_on" and e.note == note]
            return f"note-on vel {ev[0].velocity}" if ev else None

    samples, abort, reason = execute_press(robot, hover, start, z_contact, prof, method, baseline, extra_stop=extra_stop)
    robot.pause(args.post_wait_s)  # let note-off / late MIDI arrive
    t_end = now()
    events = midi.events_between(t_start - 0.05, t_end, clear=True)
    robot.move_l(hover, speed=0.1, accel=0.5)

    rec = TrialRecord(
        trial_id=trial_id,
        midi_note=note,
        key_name=midi_to_name(note),
        key_is_black=is_black(note),
        profile=prof,
        key_pose_hover=list(map(float, hover)),
        z_contact=z_contact,
        t_start=t_start,
        t_end=t_end,
        robot_samples=samples,
        midi_events=events,
        aborted=abort is not None,
        abort_reason=abort or "",
        extra={"force_baseline": baseline, "touch_setting": args.touch_setting, "sim": args.sim, "hz": robot.params.hz, "method": asdict(method), "stop_reason": reason},
    )
    return rec


# ----------------------------------------------------------------------------- main


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sim", action="store_true", help="no hardware: kinematic robot + virtual 2-switch piano")
    p.add_argument("--sim-realtime", action="store_true", help="simulate in real time instead of as fast as possible")
    common.add_connection_args(p)
    p.add_argument("--hz", type=float, default=None, help="servo/record rate (default: controller max, 500 e-series / 125 CB3)")
    p.add_argument("--tcp", type=float, nargs=6, default=None, help="TCP offset x y z rx ry rz of the fingertip")
    p.add_argument("--lookahead", type=float, default=0.03)
    p.add_argument("--gain", type=float, default=1500)
    p.add_argument("--force-abort-n", type=float, default=15.0)
    p.add_argument("--press-method", default="speedl", choices=["servo", "speedl"], help="servo = servoL trajectory (<=0.15 m/s), speedl = velocity control (fast presses)")
    p.add_argument("--accel", type=float, default=8.0, help="tool acceleration m/s^2")
    p.add_argument("--brake-accel", type=float, default=25.0)
    p.add_argument("--stop-force-n", type=float, default=8.0, help="fallback force step that ends the descent (key bed)")
    p.add_argument("--layout", default="key_layout.json", help="taught keyboard layout (see --teach-layout)")
    p.add_argument("--teach-layout", metavar="PATH", help="interactively teach the 3 reference keys, save layout and exit")
    p.add_argument("--teach-notes", default="C2,C6,C#4", help="notes used for --teach-layout: white_low,white_high,black_ref")
    p.add_argument("--keys", default="registers", help="comma list of note names or a preset: " + ",".join(REGISTER_PRESETS))
    p.add_argument(
        "--preset",
        default="full",
        choices=["const", "varied", "full", "quick", "kick", "throw"],
        help="kick = touch at --kick-v0 and accelerate inside the key to each speed; throw = kick then release at 4 mm",
    )
    p.add_argument("--kick-v0", type=float, default=0.05)
    p.add_argument("--kick-accel", type=float, default=20.0)
    p.add_argument("--stop-force-min-depth-mm", type=float, default=9.0, help="force stop only counts below this depth (after the note-on switch)")
    p.add_argument(
        "--after-note", default="lift", choices=["lift", "hold"], help="lift = rise immediately when note-on arrives (prevents re-triggering by the bouncing key); hold = brake, dwell, lift slowly"
    )
    p.add_argument("--lift-fast-speed", type=float, default=0.25, help="rise speed for --after-note lift")
    p.add_argument("--no-stop-on-note", dest="stop_on_note", action="store_false")
    p.add_argument("--speeds", type=float, nargs="*", default=CONST_SPEEDS, help="constant descent speeds m/s")
    p.add_argument("--start-above-mm", type=float, nargs="*", default=[0.0, 6.0], help="start heights above contact (0 = from touching)")
    p.add_argument("--depth-mm", type=float, default=10.5, help="fallback depth limit below contact (descent normally ends on the key-bed force)")
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--shuffle", action="store_true", default=True)
    p.add_argument("--no-shuffle", dest="shuffle", action="store_false")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--hover-mm", type=float, default=15.0)
    p.add_argument("--settle-s", type=float, default=0.4)
    p.add_argument("--post-wait-s", type=float, default=0.4)
    p.add_argument("--contact-method", default="force", choices=["force", "midi", "teach", "nominal"])
    p.add_argument("--contact-force-n", type=float, default=0.5, help="force step above baseline that counts as contact (force method; P-45 key weight is ~0.6 N)")
    p.add_argument("--assumed-d2-mm", type=float, default=7.5, help="midi contact method: assumed depth of the note-on switch")
    p.add_argument("--touch-setting", default="medium", help="P-45 touch sensitivity used (fixed/soft/medium/hard), stored in metadata")
    p.add_argument("--out", default=None, help="session directory (default data/key_velocity/<timestamp>)")
    p.add_argument("--plan-only", action="store_true", help="print the trial plan and exit")
    a = p.parse_args()
    if not a.plan_only:
        common.check_connection_args(p, a)
    return a


def teach_layout(robot, path: str, notes: list[str]):
    poses = []
    for role, n in zip(["white_low", "white_high", "black_ref"], notes):
        print(f"[teach] freedrive ON – put the fingertip just touching {n} ({role}) at the normal press point, then press Enter")
        robot.freedrive(True)
        input()
        robot.freedrive(False)
        poses.append(list(map(float, robot.tcp_pose())))
        print(f"   recorded {np.round(poses[-1], 4).tolist()}")
    lay = KeyLayout(name_to_midi(notes[0]), poses[0], name_to_midi(notes[1]), poses[1], name_to_midi(notes[2]), poses[2])
    print("[teach]", lay.report())
    lay.save(path)
    print(f"[teach] saved {path}")


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    params = ServoParams(hz=args.hz or 500.0, lookahead=args.lookahead, gain=args.gain, force_abort_n=args.force_abort_n)

    # ---------------- keys
    key_names = REGISTER_PRESETS.get(args.keys, args.keys.split(","))
    notes = [name_to_midi(k) for k in key_names]
    profiles = build_profiles(args.preset, args.speeds, [s * 1e-3 for s in args.start_above_mm], args.kick_v0, args.kick_accel)
    for pr in profiles:
        pr.depth_total_m = args.depth_mm * 1e-3
        if pr.kind != "kick":
            pr.accel_max = args.accel
    plan = [(n, pr, r) for r in range(args.repeats) for n in notes for pr in profiles]
    if args.shuffle:
        rng.shuffle(plan)
    print(f"[plan] {len(notes)} keys x {len(profiles)} profiles x {args.repeats} repeats = {len(plan)} trials")
    if args.plan_only:
        for i, (n, pr, r) in enumerate(plan):
            print(f"  {i:4d} {midi_to_name(n):4s} start_above={pr.start_above_m * 1e3:.0f}mm {pr.label()} rep={r}")
        return

    out = args.out or os.path.join("data", "key_velocity", time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out, exist_ok=True)

    # ---------------- hardware
    if args.sim:
        from key_velocity_exp.midi_io import SimPiano
        from key_velocity_exp.ur5_io import patch_sim_time

        layout = KeyLayout.synthetic()
        true_offsets = {n: np.random.default_rng(1).normal(0, 0.0007) for n in range(21, 109)}

        def surface_fn(xy):
            n = layout.nearest_key(xy)
            if n is None:
                return None, 0.0
            return n, layout.nominal_contact_pose(n)[2] + true_offsets[n]

        piano = SimPiano()
        robot = SimRobot(params, surface_fn, piano, realtime=args.sim_realtime)
        patch_sim_time(robot)
        midi = piano
        print("[sim] virtual piano: switches at 3.0 / 8.0 mm, latency 6 ms")
    else:
        from key_velocity_exp.midi_io import MidiListener

        robot = URRobot(args.robot_ip, params, tcp=args.tcp, rtde_frequency=args.hz or -1.0, auto_hz=args.hz is None)
        if args.teach_layout:
            try:
                teach_layout(robot, args.teach_layout, args.teach_notes.split(","))
            finally:
                robot.close()
            return
        layout = KeyLayout.load(args.layout)
        print("[layout]", layout.report())
        midi = MidiListener(args.midi_port)
        print(f"[midi] listening on '{midi.port_name}'")
        if args.contact_method == "force":
            robot.zero_ft()

    with open(os.path.join(out, "session.json"), "w") as f:
        json.dump({"args": vars(args), "params": asdict(params), "notes": notes, "profiles": [asdict(p) for p in profiles], "layout": asdict(layout)}, f, indent=2, default=str)

    done = 0
    try:
        # ---------------- contact height of every key
        print(f"[contact] method={args.contact_method}")
        for n in notes:
            z = find_contact(robot, layout, n, args.contact_method, args, midi)
            layout.z_contact_measured[str(n)] = z
            print(f"  {midi_to_name(n):4s} z_contact = {z:.4f}  (nominal {layout.nominal_contact_pose(n)[2]:.4f}, diff {(z - layout.nominal_contact_pose(n)[2]) * 1e3:+.2f} mm)")
        layout.save(os.path.join(out, "key_layout_measured.json"))
        print(describe_keys(layout, notes))

        # ---------------- trials
        for i, (n, pr, rep) in enumerate(plan):
            rec = run_trial(robot, layout, midi, i, n, pr, args)
            rec.extra["repeat"] = rep
            rec.save(out)
            ons = rec.note_on_events()
            vel = ons[0].velocity if ons else None
            d_on = ""
            if ons and len(rec.robot_samples):
                t = rec.col("t")
                d_on = f" depth@on={np.interp(ons[0].t, t, rec.depth()) * 1e3:5.2f}mm"
            status = f"ABORT {rec.abort_reason}" if rec.aborted else ("no note-on" if vel is None else f"vel={vel:3d}{d_on}")
            others = [e for e in rec.midi_events if e.kind == "note_on" and e.note != n]
            if others:
                status += f"  (!! also hit {[midi_to_name(e.note) for e in others]})"
            print(f"[{i + 1:4d}/{len(plan)}] {midi_to_name(n):4s} sa={pr.start_above_m * 1e3:.0f}mm {pr.label():28s} {status}")
            done += 1
    except KeyboardInterrupt:
        print("\n[abort] Ctrl-C – stopping robot")
    except RobotAbort as e:
        print(f"\n[abort] {e}")
    finally:
        try:
            robot.stop()
            if notes:
                robot.move_l(layout.hover_pose(notes[0], args.hover_mm * 1e-3), speed=0.1)
        except Exception:
            pass
        robot.close()
        midi.close()
    print(f"[done] {done} trials saved in {out}")
    if done:
        from key_velocity_exp.analyze import summarize_session

        summarize_session(out, quiet=True)
        print(f"[done] summary written: {os.path.join(out, 'summary.csv')} – run analyze.py {out} for the full analysis")


if __name__ == "__main__":
    main()
