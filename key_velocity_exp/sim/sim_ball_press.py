#!/usr/bin/env python
"""Press the simulated piano keys with a rigid ball at constant speed and compare the simulator's MIDI
velocity with the real measurements (one kve-identify session directory per key).

  kve-sim-ball --real-dirs data/piano_id_C1 ... data/piano_id_C7 --velocity-model my_piano_velocity_model.json --out data/sim_vs_real
  kve-sim-ball --real-dirs data/piano_id_C1 ... data/piano_id_C7 --model legacy --out data/sim_vs_real     # baseline

The ball (mocap sphere, radius 8 mm) descends onto the front of the key at the same target speeds as
the real sweeps (run-up in the air, constant speed through the key, stop 9.5 mm below the surface,
hold, lift). For every press the key's own angle is recorded, the mean key speed between A and B at
the press point is computed exactly like on the real robot, and the simulator's note-on velocity is
read from ``KeyMidi``. Output in --out: sim_vs_real_<model>.json, sim_traces_<model>.npz,
fig_sim_vs_real_<model>.png (and a GIF of --render-note, which needs an OpenGL backend, e.g. MUJOCO_GL=egl).
"""

from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np

from key_velocity_exp.common import GRID, PALETTE, TEXT, TEXT2, name_to_midi
from key_velocity_exp.sim import piano_constants as pc


def build_scene(ball_radius: float, timestep: float):
    from dm_control import mjcf

    from key_velocity_exp.sim.build_piano_mjcf import build

    root = build()
    root.option.timestep = timestep
    ball = root.worldbody.add("body", name="ball", mocap=True, pos=[0, 0, 0.2])
    ball.add("geom", name="ball_geom", type="sphere", size=[ball_radius], rgba=[0.9, 0.3, 0.2, 1], contype=1, conaffinity=1, solref=[0.002, 1])
    root.worldbody.add("light", pos=[0, 0, 1], dir=[0, 0, -1], diffuse=[0.8, 0.8, 0.8])
    root.worldbody.add("camera", name="press_cam", mode="fixed", pos=[0.16, -0.12, 0.08], xyaxes=[0.6, 0.8, 0, -0.35, 0.26, 0.9])
    physics = mjcf.Physics.from_mjcf_model(root)
    return root, physics


def key_press_geometry(physics, root, midi_note: int, from_tip_m: float):
    """World position above the press point of a white key and the lever arm from the hinge."""
    key = midi_note - 21
    body = root.find("body", f"white_key_{key}")
    joint = root.find("joint", f"white_joint_{key}")
    if body is None:
        raise ValueError(f"key {midi_note} is not a white key")
    bpos = np.array(physics.bind(body).xpos)
    tip_x = bpos[0] + pc.WHITE_KEY_LENGTH / 2
    press_x = tip_x - from_tip_m
    lever = press_x - (bpos[0] - pc.WHITE_KEY_LENGTH / 2)  # hinge is at the back of the key
    surface_z = bpos[2] + pc.WHITE_KEY_HEIGHT / 2
    return np.array([press_x, bpos[1], surface_z]), lever, physics.bind(joint)


def run_press(physics, root, key_midi, joints, midi_note, v, ball_radius, from_tip_m, depth_total, run_up, hold_s, accel, dt, render_every: int = 0, frames: list | None = None):
    pos0, lever, jnt = key_press_geometry(physics, root, midi_note, from_tip_m)
    ball_id = physics.model.name2id("ball", "body")
    mocap_id = physics.model.body_mocapid[ball_id]
    # reset and let the keys settle under gravity with the ball far away (white keys sag ~1.7 mm)
    with physics.reset_context():
        pass
    physics.data.mocap_pos[mocap_id] = [pos0[0], pos0[1], pos0[2] + 0.1]
    for _ in range(int(0.5 / dt)):
        physics.step()
    q_rest = float(jnt.qpos[0])
    key_midi.reset()
    z_contact = pos0[2] - q_rest * lever + ball_radius  # ball centre when just touching the *resting* key surface

    # trajectory: accelerate from rest during the run-up, constant v through the key, stop at depth_total, hold, lift
    z_list = []
    z, vel = z_contact + run_up, 0.0
    while z > z_contact - depth_total:
        vel = min(v, vel + accel * dt)
        z -= vel * dt
        z_list.append(z)
    z_list += [z_contact - depth_total] * int(hold_s / dt)
    z_end = z_list[-1]
    while z_end < z_contact + run_up:
        z_end += 0.03 * dt
        z_list.append(z_end)

    physics.data.mocap_pos[mocap_id] = [pos0[0], pos0[1], z_list[0]]
    physics.forward()
    if render_every and frames is not None:
        # point the camera at this key
        cam = physics.model.name2id("press_cam", "camera")
        physics.model.cam_pos[cam] = [pos0[0] + 0.09, pos0[1] - 0.12, pos0[2] + 0.07]
    jb = physics.bind(joints)
    t_log, q_log, ball_log = [], [], []
    events = []
    for i, z in enumerate(z_list):
        physics.data.mocap_pos[mocap_id] = [pos0[0], pos0[1], z]
        physics.step()
        if render_every and frames is not None and i % render_every == 0:
            frames.append(physics.render(height=360, width=480, camera_id="press_cam"))
        events += key_midi.after_substep(float(physics.data.time), np.array(jb.qpos), np.array(jb.qvel))
        t_log.append(float(physics.data.time))
        q_log.append(float(jnt.qpos[0]))
        ball_log.append(z_contact - z)
    return np.array(t_log), (np.array(q_log) - q_rest) * lever, np.array(ball_log), events, lever, q_rest


def _label_frames(frames, text):
    from PIL import Image, ImageDraw

    out = []
    for fr in frames:
        im = Image.fromarray(fr)
        ImageDraw.Draw(im).text((8, 8), text, fill=(20, 20, 20))
        out.append(np.asarray(im))
    return out


def _save_gif(frames, path, fps=25):
    from PIL import Image

    ims = [Image.fromarray(f) for f in frames]
    ims[0].save(path, save_all=True, append_images=ims[1:], duration=int(1000 / fps), loop=0)


def crossing_time(t, d, depth):
    idx = np.flatnonzero(d >= depth)
    if len(idx) == 0 or idx[0] == 0:
        return np.nan
    i = idx[0]
    w = (depth - d[i - 1]) / (d[i] - d[i - 1]) if d[i] != d[i - 1] else 0.0
    return t[i - 1] + w * (t[i] - t[i - 1])


def note_from_dir(d: str) -> str:
    return os.path.basename(os.path.normpath(d)).split("_")[-1]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--real-dirs", nargs="+", required=True, help="kve-identify session directories (results_si.json), one per measured white key; the note is read from the directory name (..._C4)")
    p.add_argument("--model", default="measured", choices=["measured", "legacy"], help="measured = two-switch model from --velocity-model; legacy = RoboPianist threshold rule (baseline)")
    p.add_argument("--velocity-model", default=None, help="model JSON written by kve-export (default: $KVE_VELOCITY_MODEL or the P-45 example)")
    p.add_argument("--speeds", type=float, nargs="*", default=None, help="key speeds m/s (default: the real sweep's target speeds per key)")
    p.add_argument("--ball-radius", type=float, default=0.008)
    p.add_argument("--from-tip-m", type=float, default=0.005, help="press point distance from the key tip (5 mm -> lever 0.145 m -> key dip 9.7 mm, close to the real ~10 mm at the fingertip)")
    p.add_argument("--depth-mm", type=float, default=9.5)
    p.add_argument("--run-up-mm", type=float, default=6.0)
    p.add_argument("--accel", type=float, default=30.0)
    p.add_argument("--timestep", type=float, default=0.0005)
    p.add_argument("--out", default="data/sim_vs_real")
    p.add_argument("--render-note", default="", help="key whose presses are rendered to a GIF, e.g. C4 (needs MUJOCO_GL=egl or a display; default: no rendering)")
    p.add_argument("--render-speeds", type=float, nargs="*", default=[0.05, 0.25, 0.45], help="speeds (m/s) rendered for --render-note")
    a = p.parse_args()
    # dm_control picks its OpenGL backend at import time: no rendering needed -> disable it (works on headless
    # machines without EGL); rendering requested -> normalise the variable (dm_control wants lower case)
    os.environ["MUJOCO_GL"] = os.environ.get("MUJOCO_GL", "egl").lower() if a.render_note else "off"

    from key_velocity_exp.sim.key_midi import KeyMidi
    from key_velocity_exp.sim.velocity_model import TwoSwitchVelocityModel, key_joints, key_rest_angles

    root, physics = build_scene(a.ball_radius, a.timestep)
    joints = key_joints(root)
    q_max = np.array(physics.bind(joints).range)[:, 1]
    q_rest = key_rest_angles()
    vmodel = TwoSwitchVelocityModel.load(a.velocity_model)
    key_midi = KeyMidi(a.model, q_max, rest_angle=q_rest, model_path=a.velocity_model)
    sw_a, sw_b = vmodel.angle_thresholds(q_max, q_rest)  # switch angles used to measure the key's A->B speed (also in legacy mode)
    print(
        f"[sim] velocity model: {a.model} ({vmodel.path}); physics dt {a.timestep * 1e3:.2f} ms; press point {a.from_tip_m * 1e3:.0f} mm from the key tip; white key rests {np.degrees(q_rest[39]):.2f} deg below the geometric top"
    )

    os.makedirs(a.out, exist_ok=True)
    results = []
    traces = {}  # f"{note}_{speed}" -> dict of arrays, saved for kve-compare
    gif_frames = []
    for d in a.real_dirs:
        with open(os.path.join(d, "results_si.json")) as f:
            r = json.load(f)
        note = note_from_dir(d)
        midi = name_to_midi(note)
        speeds = a.speeds or sorted({round(x["target_speed_m_s"], 4) for x in r.get("speed_table", [])})
        real_tab = {round(x["target_speed_m_s"], 4): x for x in r.get("speed_table", [])}
        real_map = np.array(r["speed_to_velocity"]) if r.get("speed_to_velocity") else None  # isotonic real map (m/s -> velocity)

        def real_at(v_ab):
            if real_map is None or not np.isfinite(v_ab) or v_ab < real_map[0, 0] * 0.95 or v_ab > real_map[-1, 0] * 1.05:
                return None
            return float(np.interp(np.log(v_ab), np.log(real_map[:, 0]), real_map[:, 1]))

        print(f"\n== {note} (midi {midi}) real A={r['A_m'] * 1e3:.2f} B={r['B_m'] * 1e3:.2f} mm; {len(speeds)} speeds")
        for v in speeds:
            render = bool(a.render_note) and note == a.render_note and any(abs(v - rv) < 1e-3 for rv in a.render_speeds)
            frames = [] if render else None
            t, key_depth, ball_depth, events, lever, q_rest_k = run_press(
                physics,
                root,
                key_midi,
                joints,
                midi,
                v,
                a.ball_radius,
                a.from_tip_m,
                a.depth_mm * 1e-3,
                a.run_up_mm * 1e-3,
                0.2,
                a.accel,
                a.timestep,
                render_every=int(0.01 / a.timestep) if render else 0,
                frames=frames,
            )
            if render and frames:
                gif_frames += _label_frames(frames, f"{note}  target {v:.2f} m/s  ({a.model})")
            ons = [e for e in events if e[1] == "NoteOn" and e[2] == midi and e[3] > 0]
            offs = [e for e in events if e[1] == "NoteOff" and e[2] == midi]
            traces[f"{note}_{v:.4f}"] = dict(
                t=t, key_depth=key_depth, ball_depth=ball_depth, note_on_t=np.array([e[0] for e in ons]), note_on_vel=np.array([e[3] for e in ons]), note_off_t=np.array([e[0] for e in offs])
            )
            # depths of the simulated switches at the press point (the model places them at rest + depth/lever_tip)
            dip_here = (q_max[midi - 21] - q_rest_k) * lever  # travel from the resting surface to the bottom at the press point
            dA, dB = (sw_a[midi - 21] - q_rest_k) * lever, (sw_b[midi - 21] - q_rest_k) * lever
            tA, tB = crossing_time(t, key_depth, dA), crossing_time(t, key_depth, dB)
            v_ab = (dB - dA) / (tB - tA) if np.isfinite(tA) and np.isfinite(tB) and tB > tA else np.nan
            sim_vel = ons[0][3] if ons else None
            rt = real_tab.get(round(v, 4))
            real_vel = rt["velocity_median"] if rt else None
            real_vab = rt["mean_speed_AB_m_s"] if rt else None
            real_same_speed = real_at(v_ab)
            results.append(
                dict(
                    note=note,
                    midi=midi,
                    target_speed=v,
                    sim_velocity=sim_vel,
                    sim_v_mean_AB=float(v_ab),
                    sim_n_note_on=len(ons),
                    sim_n_note_off=len(offs),
                    real_velocity=real_vel,
                    real_v_mean_AB=real_vab,
                    real_velocity_at_sim_speed=real_same_speed,
                    sim_key_dip_at_press_m=float(dip_here),
                    sim_max_key_depth_m=float(key_depth.max()),
                )
            )
            rs = "-" if real_same_speed is None else f"{real_same_speed:.0f}"
            print(
                f"   v={v:6.3f}  sim: v_AB={v_ab:.3f} vel={sim_vel} (on {len(ons)}, off {len(offs)})   real @same target: v_AB={real_vab if real_vab is None else round(real_vab, 3)} vel={real_vel}   real @sim speed: {rs}"
            )

    with open(os.path.join(a.out, f"sim_vs_real_{a.model}.json"), "w") as f:
        json.dump({"model": a.model, "velocity_model_path": vmodel.path, "results": results}, f, indent=2)
    np.savez_compressed(os.path.join(a.out, f"sim_traces_{a.model}.npz"), **{f"{k}__{fld}": arr for k, tr in traces.items() for fld, arr in tr.items()})
    if gif_frames:
        _save_gif(gif_frames, os.path.join(a.out, f"sim_ball_press_{a.render_note}_{a.model}.gif"))
        print(f"[render] {len(gif_frames)} frames -> sim_ball_press_{a.render_note}_{a.model}.gif")

    # ---- summary + figure
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    notes = list(dict.fromkeys(r["note"] for r in results))
    ncols = min(4, len(notes) + 1)
    nrows = math.ceil((len(notes) + 1) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.8 * ncols, 3.5 * nrows), facecolor="white", squeeze=False)
    axes = axes.ravel()
    errs_all = []
    for i, n in enumerate(notes):
        ax = axes[i]
        rs = [r for r in results if r["note"] == n and r["real_velocity"] is not None]
        xr = [r["real_v_mean_AB"] for r in rs]
        yr = [r["real_velocity"] for r in rs]
        rs2 = [r for r in results if r["note"] == n and r["sim_velocity"] is not None and np.isfinite(r["sim_v_mean_AB"])]
        xs = [r["sim_v_mean_AB"] for r in rs2]
        ys = [r["sim_velocity"] for r in rs2]
        ax.plot(xr, yr, "o-", color=PALETTE[0], markersize=4, linewidth=1.5, label="real piano (robot)")
        ax.plot(xs, ys, "s--", color=PALETTE[1], markersize=4, linewidth=1.5, label=f"sim ({a.model})")
        errs = [
            r["sim_velocity"] - r["real_velocity_at_sim_speed"]
            for r in results
            if r["note"] == n and r["sim_velocity"] is not None and r.get("real_velocity_at_sim_speed") is not None and 1 < r["real_velocity_at_sim_speed"] < 127
        ]
        errs_all += errs
        mae = np.mean(np.abs(errs)) if errs else np.nan
        ax.set_xscale("log")
        ax.set_title(f"{n}: MAE {mae:.1f}", color=TEXT, loc="left", fontsize=10)
        ax.set_xlabel("mean key speed A→B [m/s]", color=TEXT2, fontsize=9)
        ax.set_ylabel("MIDI velocity", color=TEXT2, fontsize=9)
        ax.grid(True, color=GRID, linewidth=0.8)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        if i == 0:
            ax.legend(frameon=False, fontsize=8)
    ax = axes[len(notes)]
    ax.hist(errs_all, bins=np.arange(-30.5, 31.5, 2), color=PALETTE[2])
    ax.set_title(f"sim − real at the same key speed, all keys: MAE {np.mean(np.abs(errs_all)):.1f}, bias {np.mean(errs_all):+.1f}", color=TEXT, loc="left", fontsize=9)
    ax.set_xlabel("velocity error", color=TEXT2, fontsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    for ax in axes[len(notes) + 1 :]:
        ax.axis("off")
    fig.suptitle(f"Rigid ball pressing the simulated piano vs the real piano (velocity model: {a.model})", color=TEXT)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, f"fig_sim_vs_real_{a.model}.png"), dpi=140)
    print(f"\n[done] all keys, compared at the same measured key speed: MAE {np.mean(np.abs(errs_all)):.1f}, bias {np.mean(errs_all):+.1f} velocity units (n={len(errs_all)}) -> {a.out}")


if __name__ == "__main__":
    main()
