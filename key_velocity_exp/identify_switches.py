#!/usr/bin/env python
"""One-shot identification of the piano's velocity mechanism on one key:
home confirmation -> switch depths A and B (+ MIDI latency) -> velocity map by speed sweep.

  kve-identify --out data/piano_id_C4

Jog the fingertip onto the key (roughly touching) and run. The script then:
  0. confirms the key surface with the F/T sensor (2 mm/s approach, 0.4 N step) and takes it as home
     (--no-refine-home keeps the taught pose);
  b. measures B (note-on switch) and the MIDI latency with slow constant-speed presses;
  a. measures A (timer-start switch) with the pause probe (model-free, bisection to 0.1 mm);
  s. sweeps the mean key speed A->B from ~0.005 to ~0.45 m/s (key driven, speed reached before A,
     constant through A->B), adding speeds until MIDI velocity 1..127 is covered or the arm limit
     is reached; ends with slow presses to check reference drift.
Outputs (SI units) in --out: results_si.json, fig_summary_si.png, plus the per-phase figures and
every press as trial_XXXX.npz. Re-analyse any time with analyze_switches.py <dir>.

Details of the model:

Model of the P-45 (two-contact GHS action):  the timer starts when the key passes depth A, stops
when it passes depth B (note-on is sent then, ``latency`` later);  velocity = f(dt).

Jog the fingertip so it just touches the key (as for real_test.py), then:

  kve-identify --phase all --out data/switch_id_C4

Phases (``--phase b|a|m|all``):
  b  B and latency: slow constant-speed presses (--slow-speeds x --repeats-b); fit
     t_on - t_contact = B / v + L.  The slow sampled lift also gives the note-off depth.
  a  A by "pause probe": descend slowly to depth d, pause --probe-pause s, then finish fast.
     d < A -> the timer starts after the pause -> normal (high) velocity;  d > A -> the pause is
     inside dt -> velocity below --probe-low-vel or no note at all.  Coarse sweep + bisection to
     --probe-resolution-mm.  Model-free.
  m  dt -> velocity mapping: presses at many speeds (constant for slow, in-key acceleration "kick"
     for fast so the key stays in contact) plus shape tests (slow->fast vs fast->slow between A and B
     with the same mean speed = same dt).  dt is computed from the measured fingertip trajectory
     (time A is crossed) and the MIDI time of note-on minus latency.  Ends with three slow presses
     (phase "b2") to detect drift of the contact reference during the session.

IMPORTANT: use a stiff fingertip.  The analysis uses the TCP position as the key position; a soft
tip compresses several mm under the 5-8 N seen at 0.2-0.35 m/s and the depths/dt become wrong
(such presses are flagged "compressed" and excluded).

  s  velocity map by speed sweep (assumes velocity = f(mean key speed A->B)): every press reaches its
     target speed *before* A (in-key acceleration chosen per speed) and holds it through A->B; the
     descent ends at note-on.  After each round the covered velocity range is checked: speeds are
     added above/below and in gaps larger than --sweep-gap until 1..127 is covered or the arm limits
     (--sweep-max-speed) are reached.  Needs A/B: run phases b+a first or pass --ab-from <switch_id.json>.

Results: trial_XXXX.npz (all raw data), switch_id.json (A, B, latency, note-off depth, dt->velocity
fit + table) and figures, written by analyze_switches.py (re-run it on the directory any time).
"""

from __future__ import annotations

import json
import os
import random
import time

import numpy as np

from key_velocity_exp import common
from key_velocity_exp import real_test as rt
from key_velocity_exp.common import PressProfile, now
from key_velocity_exp.ur5_io import PressMethod, RobotAbort


def parse():
    p = rt.build_parser(__doc__)
    p.add_argument("--phase", default="all", choices=["all", "b", "a", "m", "s", "bas", "bam"], help="all/bas = home + B + A + speed sweep; bam = home + B + A + shape tests; or a single phase")
    p.add_argument("--ab-from", metavar="JSON", help="take A, B and latency from an earlier switch_id.json (for --phase s/m without re-running b and a)")
    # phase s
    p.add_argument("--sweep-speeds", type=float, nargs="*", default=[0.005, 0.01, 0.02, 0.04, 0.08, 0.15, 0.25, 0.35, 0.45], help="initial target mean speeds A->B [m/s]")
    p.add_argument("--sweep-repeats", type=int, default=2)
    p.add_argument("--sweep-fill-rounds", type=int, default=3, help="rounds of adding speeds (gaps / extension)")
    p.add_argument("--sweep-gap", type=float, default=8.0, help="add a speed between neighbours whose median velocities differ by more than this")
    p.add_argument("--sweep-min-speed", type=float, default=0.002)
    p.add_argument("--sweep-max-speed", type=float, default=0.55, help="arm limit: braking from here after B already hits the key bed")
    p.add_argument("--sweep-max-kick-speed", type=float, default=0.55, help="above this the run-up is in the air (impact at contact) instead of an in-key kick")
    p.add_argument(
        "--brake-after-b-mm",
        type=float,
        default=None,
        help="reverse as soon as the finger is this far past B (position-triggered, saves the ~4 ms MIDI latency travel before the key bed); default: wait for note-on",
    )
    p.add_argument("--impact-stop-n", type=float, default=None, help="phase s: stop adding faster speeds once a press hit the key bed harder than this")
    p.add_argument(
        "--hard",
        action="store_true",
        help="push for velocity 127: speeds 0.45..0.8 m/s ascending, kick accel up to 45 m/s^2, brake 40 m/s^2, brake 0.6 mm after B, force abort 80 N, impact guard 60 N",
    )
    p.add_argument("--refine-home", dest="refine_home", action="store_true", default=True, help="find the key surface with the F/T sensor (0.4 N step) before starting (default)")
    p.add_argument("--no-refine-home", dest="refine_home", action="store_false", help="keep the taught pose as home")
    # phase b
    p.add_argument("--slow-speeds", type=float, nargs="*", default=[0.02, 0.03, 0.05, 0.08, 0.12])
    p.add_argument("--repeats-b", type=int, default=3)
    p.add_argument("--lift-slow", type=float, default=0.02, help="lift speed in phase b (note-off depth)")
    # phase a
    p.add_argument("--probe-slow", type=float, default=0.02)
    p.add_argument("--probe-fast", type=float, default=0.20)
    p.add_argument("--probe-pause", type=float, default=0.4)
    p.add_argument("--probe-from-mm", type=float, default=0.5)
    p.add_argument("--probe-step-mm", type=float, default=1.0)
    p.add_argument("--probe-resolution-mm", type=float, default=0.1)
    p.add_argument("--probe-repeats", type=int, default=2, help="confirmations of the final boundary")
    p.add_argument("--probe-low-vel", type=int, default=20, help="velocity below this (or no note) = timer already running before the pause")
    p.add_argument("--b-guess-mm", type=float, default=8.2, help="used for the probe's fast-phase target until phase b has measured B")
    # phase m
    p.add_argument("--map-speeds", type=float, nargs="*", default=[0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45])
    p.add_argument("--repeats-m", type=int, default=3)
    p.add_argument("--kick-end-at-a", dest="kick_end_at_a", action="store_true", default=True, help="phase m kicks: pick the acceleration so the target speed is reached exactly at A (default)")
    p.add_argument("--kick-fixed-accel", dest="kick_end_at_a", action="store_false", help="phase m kicks: use --kick-accel for every speed instead")
    p.add_argument("--kick-accel-max", type=float, default=30.0, help="cap for the per-speed kick acceleration")
    p.add_argument(
        "--const-max-speed",
        type=float,
        default=0.12,
        help="phase m: speeds up to this use a constant-speed press with an in-air run-up (impact at contact = v); faster speeds touch softly and accelerate inside the key (kick)",
    )
    p.add_argument("--no-shape-tests", dest="shape_tests", action="store_false")
    p.add_argument("--a-guess-mm", type=float, default=3.0, help="used for the shape tests if phase a was not run")
    # gentler in-key acceleration than real_test's default: 20 m/s^2 (actually ~45 with the velocity loop)
    # throws the hammer ahead of the key and trips the 15 N force abort with a stiff tip
    p.set_defaults(kick_accel=8.0, force_abort_n=25.0)
    a = p.parse_args()
    common.check_connection_args(p, a)
    if a.hard:
        a.phase = "s" if a.phase in ("all", "bas") else a.phase
        a.sweep_speeds = [0.45, 0.55, 0.65, 0.75]
        a.sweep_repeats = 1
        a.sweep_fill_rounds = 0
        a.sweep_max_speed = 0.8
        a.sweep_max_kick_speed = 0.8
        a.kick_accel_max = 45.0
        a.brake_accel = 40.0
        a.force_abort_n = 80.0
        a.brake_after_b_mm = 0.6 if a.brake_after_b_mm is None else a.brake_after_b_mm
        a.impact_stop_n = 60.0 if a.impact_stop_n is None else a.impact_stop_n
    a.out = a.out or os.path.join("data", "switch_id", time.strftime("%Y%m%d_%H%M%S"))
    return a


# --------------------------------------------------------------------------- helpers


def refine_home(s: rt.Session, step_n: float = 0.4):
    """Descend at 2 mm/s from 3 mm above home until the F/T sensor sees the key (median of 5 > step_n)."""
    robot = s.robot
    above = s.home.copy()
    above[2] += 0.003
    robot.move_l(s.hover(), speed=0.1)
    robot.move_l(above, speed=0.02)
    robot.pause(0.5)
    base = robot.measure_force_baseline()
    fz_i, z_i = common.ROBOT_SAMPLE_FIELDS.index("fz") - 1, common.ROBOT_SAMPLE_FIELDS.index("tcp_z") - 1
    recent = []
    z_hit = None
    t0 = now()
    try:
        while now() - t0 < 4.0:
            row = robot.sample(np.nan)
            recent.append(row[fz_i])
            recent = recent[-5:]
            if len(recent) == 5 and abs(float(np.median(recent)) - base) > step_n:
                z_hit = row[z_i]
                break
            robot.speed_step(np.array([0, 0, -0.002]), 1.0)
    finally:
        robot.speed_stop(2.0)
    if z_hit is None:
        print("[refine] no contact within 6 mm - keeping the taught home")
    else:
        print(f"[refine] key surface at z={z_hit:.5f} ({(z_hit - s.home[2]) * 1e3:+.2f} mm vs taught home)")
        s.home[2] = z_hit
    robot.move_l(s.hover(), speed=0.05)


def method(args, name="speedl", after_note="lift", accel=None):
    brake_at = getattr(args, "_brake_at_depth_m", None)
    return PressMethod(name, accel or args.accel, args.brake_accel, args.stop_force_n, args.stop_force_min_depth_mm * 1e-3, after_note, args.lift_fast_speed, brake_at)


def set_limits_from_b(args, b_m: float):
    """Once B is known, the fallbacks must sit *beyond* B: force stop from B+0.5 mm, depth limit B+2.5 mm.
    (In the first real session B was 9.55 mm while the force fallback was armed from 9 mm -> the shape
    tests were braked before the second switch.)"""
    args.stop_force_min_depth_mm = b_m * 1e3 + 0.5
    args.depth_mm = b_m * 1e3 + 2.5
    args._brake_at_depth_m = None  # enabled inside phase s only (phases b/a need the slow, sampled lift)
    print(
        f"[limits] B = {b_m * 1e3:.2f} mm -> force fallback from {args.stop_force_min_depth_mm:.1f} mm, depth limit {args.depth_mm:.1f} mm"
        + (f", position brake at {args._brake_at_depth_m * 1e3:.2f} mm" if args._brake_at_depth_m else "")
    )


def press(s: rt.Session, prof: PressProfile, m: PressMethod, phase: str, **extra):
    rec = s.press(prof, label=phase, method=m)
    rec.extra.update({"phase": phase, **extra})
    rec.save(s.out)  # re-save with the extra fields
    return rec


def first_velocity(rec):
    ons = rec.note_on_events()
    return ons[0].velocity if ons else None


# --------------------------------------------------------------------------- phases


def phase_b(s: rt.Session, args) -> tuple[float, float]:
    print(f"\n[phase b] B and latency: speeds {args.slow_speeds} x {args.repeats_b}")
    plan = [v for _ in range(args.repeats_b) for v in args.slow_speeds]
    random.Random(1).shuffle(plan)
    m = method(args, after_note="hold")
    for v in plan:
        prof = s.const(v)
        prof.lift_speed = args.lift_slow
        prof.hold_s = 0.3
        press(s, prof, m, "b")
    from key_velocity_exp.analyze_switches import fit_b

    res = fit_b(s.out)
    if res.get("ok"):
        print(
            f"[phase b] B = {res['B_mm']:.2f} mm, latency = {res['latency_ms']:.1f} ms (rms {res['rms_ms']:.1f} ms, n={res['n']}), note-off depth = {res.get('noteoff_depth_mm', float('nan')):.2f} mm"
        )
        return res["B_mm"] * 1e-3, res["latency_ms"] * 1e-3
    print("[phase b] fit failed:", res.get("reason"))
    return args.b_guess_mm * 1e-3, 0.008


def probe(s: rt.Session, args, d_mm: float, b_m: float) -> str:
    """One pause-probe at depth d_mm. Returns 'before' (timer not started before the pause) or 'after'."""
    stages = [[d_mm * 1e-3, args.probe_slow, args.probe_pause], [b_m + 0.0015, args.probe_fast, 0.0]]
    prof = PressProfile("staged", v0=args.probe_slow, v1=args.probe_fast, stages=stages, accel_max=args.kick_accel)
    prof.depth_total_m = args.depth_mm * 1e-3
    prof.start_above_m = 0.002
    rec = press(s, prof, method(args, after_note="lift", accel=args.kick_accel), "a", probe_depth_mm=d_mm)
    vel = first_velocity(rec)
    cls = "after" if (vel is None or vel < args.probe_low_vel) else "before"
    print(f"        probe d={d_mm:5.2f} mm -> velocity {vel} -> A is {'ABOVE' if cls == 'after' else 'below'} this depth")
    rec.extra["probe_class"] = cls
    rec.save(s.out)
    return cls


def phase_a(s: rt.Session, args, b_m: float) -> float:
    print(f"\n[phase a] pause probe for A: slow {args.probe_slow} m/s, pause {args.probe_pause} s, fast {args.probe_fast} m/s, B={b_m * 1e3:.2f} mm")
    lo, hi = None, None  # lo: deepest 'before', hi: shallowest 'after'
    d = args.probe_from_mm
    while d < b_m * 1e3 - 0.8:
        cls = probe(s, args, d, b_m)
        if cls == "before":
            lo = d
        else:
            hi = d
            break
        d += args.probe_step_mm
    if hi is None:
        print("[phase a] never crossed A before B-0.8 mm?! (check --probe-low-vel / MIDI)")
        return float("nan")
    if lo is None:
        lo = 0.0
    while hi - lo > args.probe_resolution_mm:
        mid = 0.5 * (lo + hi)
        if probe(s, args, mid, b_m) == "before":
            lo = mid
        else:
            hi = mid
    for _ in range(args.probe_repeats - 1):  # confirm both sides of the boundary
        probe(s, args, lo, b_m)
        probe(s, args, hi, b_m)
    a_mm = 0.5 * (lo + hi)
    print(f"[phase a] A = {a_mm:.2f} mm  (between {lo:.2f} and {hi:.2f})")
    return a_mm * 1e-3


def phase_m(s: rt.Session, args, a_m: float, b_m: float):
    print(f"\n[phase m] dt -> velocity: speeds {args.map_speeds} x {args.repeats_m}, shape tests {'on' if args.shape_tests else 'off'}")
    plan = []
    for r in range(args.repeats_m):
        for v in args.map_speeds:
            plan.append(("speed", v))
        if args.shape_tests:
            mid = 0.5 * (a_m + b_m)
            for va, vb in [(0.05, 0.30), (0.30, 0.05), (0.08, 0.25), (0.25, 0.08)]:
                plan.append(("shape", (mid, va, vb)))
    random.Random(2).shuffle(plan)
    plan += [("drift", v) for v in (0.02, 0.05, 0.02)]  # re-measure B at the end: reference drift check
    for kind, val in plan:
        if kind == "drift":
            prof = s.const(val)
            prof.lift_speed = args.lift_slow
            press(s, prof, method(args, after_note="hold"), "b2", target_speed=val)
            continue
        if kind == "speed":
            v = val
            if v <= args.const_max_speed:
                prof = s.const(v)
                m = method(args, after_note="lift")
            else:
                v0 = min(0.05, v / 4)
                if args.kick_end_at_a:
                    # choose the in-key acceleration so that v is reached exactly at A: the hammer lead that
                    # builds up after the acceleration ends then has no time to grow before B
                    acc = float(np.clip((v**2 - v0**2) / (2 * max(a_m, 0.001)), 2.0, args.kick_accel_max))
                else:
                    acc = args.kick_accel
                prof = PressProfile("kick", v0=v0, v1=v, accel_max=acc)
                prof.depth_total_m = args.depth_mm * 1e-3
                prof.start_above_m = 1.5 * v0**2 / (2 * args.accel) + 0.002
                m = method(args, after_note="lift", accel=max(acc, args.accel))
            press(s, prof, m, "m", target_speed=v)
        else:
            mid, va, vb = val
            stages = [[max(a_m - 0.0007, 0.0005), 0.03, 0.0], [mid, va, 0.0], [b_m + 0.0015, vb, 0.0]]
            prof = PressProfile("staged", v0=va, v1=vb, stages=stages, accel_max=args.kick_accel)
            prof.depth_total_m = args.depth_mm * 1e-3
            prof.start_above_m = 0.002
            press(s, prof, method(args, after_note="lift", accel=args.kick_accel), "m", shape=f"{va:.2f}->{vb:.2f}@{mid * 1e3:.1f}mm")


# --------------------------------------------------------------------------- phase s: speed sweep


def press_at_speed(s: rt.Session, args, v: float, a_m: float, phase: str = "s", **extra):
    """One press with the key driven at constant speed v through A->B (speed reached before A)."""
    if v <= 0.05:
        prof = s.const(v)  # touching at v itself is a negligible impact
        m = method(args, after_note="lift")
        how = "const"
    elif v <= args.sweep_max_kick_speed:
        v0 = min(0.05, v / 4)
        acc = float(np.clip((v**2 - v0**2) / (2 * max(a_m - 0.0003, 0.001)), 2.0, args.kick_accel_max))
        prof = PressProfile("kick", v0=v0, v1=v, accel_max=acc)
        prof.depth_total_m = args.depth_mm * 1e-3
        prof.start_above_m = 1.5 * v0**2 / (2 * args.accel) + 0.002
        m = method(args, after_note="lift", accel=max(acc, args.accel))
        how = f"kick a={acc:.0f}"
    else:
        prof = s.const(v)  # run-up in the air: impact at contact (hammer may be thrown - flagged by the analysis)
        m = method(args, after_note="lift")
        how = "const(impact)"
    return press(s, prof, m, phase, target_speed=v, how=how, **extra)


def phase_s(s: rt.Session, args, a_m: float, b_m: float):
    from key_velocity_exp.analyze_switches import depth_at, descent_crossing, first_on

    print(f"\n[phase s] velocity map by speed sweep, A={a_m * 1e3:.2f} B={b_m * 1e3:.2f} mm, start speeds {args.sweep_speeds}, x{args.sweep_repeats}")
    if args.brake_after_b_mm is not None:
        args._brake_at_depth_m = b_m + args.brake_after_b_mm * 1e-3
        print(f"[phase s] position-triggered brake {args.brake_after_b_mm:.1f} mm past B (at {args._brake_at_depth_m * 1e3:.2f} mm)")
    results: dict = {}  # target speed -> list of (velocity, v_mean_AB, lead_mm)
    guard = {"tripped": False}

    def run(speeds):
        plan = [v for _ in range(args.sweep_repeats) for v in speeds]
        if args.impact_stop_n is not None:
            plan.sort()  # ascending: stop before the impacts get worse
        else:
            random.Random(len(results)).shuffle(plan)
        for v in plan:
            if guard["tripped"] and v > max(results):
                print(f"[phase s] impact guard: skipping {v:.3f} m/s")
                continue
            rec = press_at_speed(s, args, v, a_m)
            if args.impact_stop_n is not None and len(rec.robot_samples):
                imp = float(np.max(np.abs(rec.col("fz") - rec.extra["force_baseline"])))
                if imp > args.impact_stop_n:
                    guard["tripped"] = True
                    print(f"[phase s] impact guard: {imp:.0f} N > {args.impact_stop_n:.0f} N at {v:.3f} m/s - no faster presses")
            on = first_on(rec)
            if on is None:
                results.setdefault(v, []).append((None, np.nan, np.nan))
                continue
            tA, tB = descent_crossing(rec, a_m), descent_crossing(rec, b_m)
            v_ab = (b_m - a_m) / (tB - tA) if np.isfinite(tA) and np.isfinite(tB) and tB > tA else np.nan
            lead = (b_m - depth_at(rec, on.t)) * 1e3  # >0: note-on before the finger reached B (hammer ahead)
            results.setdefault(v, []).append((on.velocity, v_ab, lead))

    def summary():
        rows = []
        for v in sorted(results):
            vel = [r[0] for r in results[v] if r[0] is not None]
            vab = [r[1] for r in results[v] if np.isfinite(r[1])]
            lead = [r[2] for r in results[v] if np.isfinite(r[2])]
            rows.append((v, float(np.median(vel)) if vel else np.nan, float(np.median(vab)) if vab else np.nan, float(np.median(lead)) if lead else np.nan, len(results[v]) - len(vel)))
        return rows

    speeds = sorted(set(args.sweep_speeds))
    for rnd in range(args.sweep_fill_rounds + 1):
        todo = [v for v in speeds if v not in results]
        if todo:
            print(f"[phase s] round {rnd}: {len(todo)} speeds x {args.sweep_repeats}: {[round(v, 4) for v in sorted(todo)]}")
            run(todo)
        rows = summary()
        print("[phase s] target v -> median velocity (measured mean speed A->B, hammer lead at B, no-note count):")
        for v, mv, vab, lead, nn in rows:
            print(f"     {v:6.3f} m/s -> vel {mv:5.1f}   (v_AB {vab:.3f} m/s, lead {lead:+.2f} mm, no-note {nn})")
        have = [(v, mv) for v, mv, *_ in rows if np.isfinite(mv)]
        if not have:
            print("[phase s] no note-ons at all - check MIDI / depth")
            return
        new = []
        vmin, velmin = have[0]
        vmax, velmax = have[-1]
        if velmax < 127 and vmax * 1.3 <= args.sweep_max_speed and not guard["tripped"]:
            new.append(round(vmax * 1.3, 4))
        elif velmax < 127:
            print(f"[phase s] velocity max so far {velmax:.0f} at {vmax:.3f} m/s: arm limit --sweep-max-speed {args.sweep_max_speed} reached")
        if velmin > 1 and vmin / 1.6 >= args.sweep_min_speed:
            new.append(round(vmin / 1.6, 4))
        for (v1, m1), (v2, m2) in zip(have[:-1], have[1:]):
            if abs(m2 - m1) > args.sweep_gap and v2 / v1 > 1.08:
                new.append(round(float(np.sqrt(v1 * v2)), 4))
        new = [v for v in new if v not in results]
        if not new or rnd == args.sweep_fill_rounds:
            if new:
                print(f"[phase s] stopping after {rnd} fill rounds; would still add {new}")
            break
        speeds = sorted(set(speeds) | set(new))
    rows = summary()
    have = [(v, mv) for v, mv, *_ in rows if np.isfinite(mv)]
    print(f"[phase s] done: velocity {have[0][1]:.0f} .. {have[-1][1]:.0f} covered with speeds {have[0][0]:.3f} .. {have[-1][0]:.3f} m/s")


# --------------------------------------------------------------------------- main


def main():
    args = parse()
    robot, midi = rt.build(args)
    s = rt.Session(robot, midi, args)
    os.makedirs(s.out, exist_ok=True)
    normal = False
    try:
        if args.refine_home and not args.sim:
            refine_home(s)
        b_m, lat = args.b_guess_mm * 1e-3, 0.008
        a_m = args.a_guess_mm * 1e-3
        if args.ab_from:
            with open(args.ab_from) as f:
                prev = json.load(f)["model_for_sim"]
            a_m, b_m, lat = prev["A_mm"] * 1e-3, prev["B_mm"] * 1e-3, prev["latency_ms"] * 1e-3
            print(f"[ab-from] A={a_m * 1e3:.2f} mm B={b_m * 1e3:.2f} mm latency={lat * 1e3:.1f} ms from {args.ab_from}")
        do_b = args.phase in ("all", "bas", "bam", "b")
        do_a = args.phase in ("all", "bas", "bam", "a")
        do_m = args.phase in ("bam", "m")
        do_s = args.phase in ("all", "bas", "s")
        if do_b:
            b_m, lat = phase_b(s, args)
        set_limits_from_b(args, b_m)
        if do_a:
            a_m = phase_a(s, args, b_m)
            if not np.isfinite(a_m):
                a_m = args.a_guess_mm * 1e-3
        if do_m:
            phase_m(s, args, a_m, b_m)
        if do_s:
            phase_s(s, args, a_m, b_m)
            args._brake_at_depth_m = None
            for v in (0.02, 0.05, 0.02):  # drift check of the contact reference at the very end
                prof = s.const(v)
                prof.lift_speed = args.lift_slow
                press(s, prof, method(args, after_note="hold"), "b2", target_speed=v)
        normal = True
    except KeyboardInterrupt:
        print("\n[abort] Ctrl-C")
    except RobotAbort as e:
        print(f"[abort] {e}")
    finally:
        try:
            robot.stop()
            if normal:
                s.goto_home()
            else:
                robot.move_l(s.hover(), speed=0.1)
        except Exception:
            pass
        robot.close()
        midi.close()
    if s.trial_id:
        with open(os.path.join(s.out, "session.json"), "w") as f:
            json.dump({"args": vars(args), "A_mm": a_m * 1e3, "B_mm": b_m * 1e3, "latency_ms": lat * 1e3, "home": s.home.tolist()}, f, indent=2, default=str)
        from key_velocity_exp.analyze_switches import analyze, print_report

        print_report(analyze(s.out))


if __name__ == "__main__":
    main()
