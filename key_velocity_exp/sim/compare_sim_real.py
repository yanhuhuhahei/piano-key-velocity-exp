#!/usr/bin/env python
"""Side-by-side comparison of the simulated ball presses (kve-sim-ball) with the real robot presses.

  kve-compare --sim-dir data/sim_vs_real --real-dirs data/piano_id_C1 ... data/piano_id_C7

Produces in --sim-dir:
  fig_compare_overview.png       velocity vs mean key speed A->B per key: real (median, min-max, isotonic map) vs each simulated model
  fig_compare_traces_<note>.png  key depth vs time for three speeds: real fingertip vs simulated key, note-on markers
  compare_table.md               per-key MAE / bias for each model and a per-speed table for the trace key
"""

from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np

from key_velocity_exp.common import GRID, PALETTE, TEXT, TEXT2, load_session

MODEL_STYLE = {"measured": ("--", PALETTE[1]), "legacy": (":", PALETTE[6])}


def _style(ax, title, xl, yl):
    ax.set_title(title, color=TEXT, loc="left", fontsize=10)
    ax.set_xlabel(xl, color=TEXT2, fontsize=9)
    ax.set_ylabel(yl, color=TEXT2, fontsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(colors=TEXT2, labelsize=8)


def load_sim(sim_dir, model):
    path = os.path.join(sim_dir, f"sim_vs_real_{model}.json")
    if not os.path.exists(path):
        return None, None
    with open(path) as f:
        res = json.load(f)["results"]
    tr_path = os.path.join(sim_dir, f"sim_traces_{model}.npz")
    traces = None
    if os.path.exists(tr_path):
        z = np.load(tr_path)
        traces = {}
        for k in z.files:
            key, fld = k.split("__")
            traces.setdefault(key, {})[fld] = z[k]
    return res, traces


def real_trace(real_dir, target_speed):
    """Real press (phase s/m, kick or const) closest to the target speed: fingertip depth vs time and note-on."""
    best = None
    for rec in load_session(real_dir):
        ts = rec.extra.get("target_speed")
        if ts is None or rec.aborted or rec.extra.get("phase") not in ("s", "m"):
            continue
        d = abs(ts - target_speed)
        if best is None or d < best[0]:
            best = (d, rec)
    if best is None or best[0] > 0.02:
        return None
    rec = best[1]
    on = [e for e in rec.midi_events if e.kind == "note_on"]
    return dict(
        t=rec.col("t") - rec.t_start,
        depth=rec.depth(),
        note_on_t=np.array([e.t - rec.t_start for e in on]),
        note_on_vel=np.array([e.velocity for e in on]),
        label=rec.profile.label(),
        target=rec.extra.get("target_speed"),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sim-dir", default="data/sim_vs_real", help="output directory of kve-sim-ball")
    p.add_argument("--real-dirs", nargs="+", required=True, help="the kve-identify session directories given to kve-sim-ball")
    p.add_argument("--models", nargs="*", default=["measured", "legacy"], help="sim_vs_real_<model>.json files to include (missing ones are skipped)")
    p.add_argument("--trace-note", default="C4")
    p.add_argument("--trace-speeds", type=float, nargs="*", default=[0.05, 0.25, 0.45])
    a = p.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sims = {m: load_sim(a.sim_dir, m) for m in a.models}
    sims = {m: v for m, v in sims.items() if v[0] is not None}
    if not sims:
        raise SystemExit(f"no sim_vs_real_<model>.json in {a.sim_dir} - run kve-sim-ball first")
    reals = {}
    for d in a.real_dirs:
        note = os.path.basename(os.path.normpath(d)).split("_")[-1]
        with open(os.path.join(d, "results_si.json")) as f:
            reals[note] = (d, json.load(f))
    notes = list(reals)

    # ------------------------------------------------------------------ overview figure
    ncols = min(4, len(notes) + 1)
    nrows = math.ceil((len(notes) + 1) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.7 * nrows), facecolor="white", squeeze=False)
    axes = axes.ravel()
    table_rows = []
    for i, note in enumerate(notes):
        ax = axes[i]
        d, r = reals[note]
        st = r.get("speed_table", [])
        ax.errorbar(
            [x["mean_speed_AB_m_s"] for x in st],
            [x["velocity_median"] for x in st],
            yerr=[[x["velocity_median"] - x["velocity_min"] for x in st], [x["velocity_max"] - x["velocity_median"] for x in st]],
            fmt="o",
            color=PALETTE[0],
            markersize=4,
            capsize=2,
            linewidth=1,
            label="real piano (median, min–max)",
        )
        if r.get("speed_to_velocity"):
            arr = np.array(r["speed_to_velocity"])
            ax.step(arr[:, 0], arr[:, 1], where="mid", color=PALETTE[0], linewidth=1, alpha=0.5)
        row = {"note": note}
        for j, (m, (res, _)) in enumerate(sims.items()):
            ls, col = MODEL_STYLE.get(m, ("-.", PALETTE[(j + 2) % len(PALETTE)]))
            pts = [x for x in res if x["note"] == note and x["sim_velocity"] is not None and np.isfinite(x["sim_v_mean_AB"])]
            pts.sort(key=lambda x: x["sim_v_mean_AB"])
            ax.plot([x["sim_v_mean_AB"] for x in pts], [x["sim_velocity"] for x in pts], "s" + ls, color=col, markersize=3.5, linewidth=1.4, label=f"sim ({m})")
            errs = [x["sim_velocity"] - x["real_velocity_at_sim_speed"] for x in pts if x.get("real_velocity_at_sim_speed") is not None and 1 < x["real_velocity_at_sim_speed"] < 127]
            row[f"{m}_mae"] = float(np.mean(np.abs(errs))) if errs else np.nan
            row[f"{m}_bias"] = float(np.mean(errs)) if errs else np.nan
            row[f"{m}_n"] = len(errs)
        table_rows.append(row)
        ax.set_xscale("log")
        _style(ax, f"{note}   MAE " + " / ".join(f"{m} {row.get(f'{m}_mae', np.nan):.1f}" for m in sims), "mean key speed A→B [m/s]", "MIDI velocity")
        if i == 0:
            ax.legend(frameon=False, fontsize=8)
    ax = axes[len(notes)]
    x = np.arange(len(notes))
    width = 0.8 / max(len(sims), 1)
    for j, m in enumerate(sims):
        _, col = MODEL_STYLE.get(m, ("-.", PALETTE[(j + 2) % len(PALETTE)]))
        ax.bar(x + (j - (len(sims) - 1) / 2) * width, [r.get(f"{m}_mae", np.nan) for r in table_rows], width, color=col, label=f"sim {m}")
    ax.set_xticks(x)
    ax.set_xticklabels(notes)
    _style(ax, "MAE vs real at the same key speed", "", "MAE [velocity]")
    ax.legend(frameon=False, fontsize=8)
    for ax in axes[len(notes) + 1 :]:
        ax.axis("off")
    fig.suptitle("Simulated rigid-ball presses vs real robot presses", color=TEXT)
    fig.tight_layout()
    fig.savefig(os.path.join(a.sim_dir, "fig_compare_overview.png"), dpi=140)
    plt.close(fig)

    # ------------------------------------------------------------------ trace comparison for one key
    note = a.trace_note
    first_model = next(iter(sims))
    if note in reals:
        d, r = reals[note]
        _, tr_sim = sims[first_model]
        fig, axes = plt.subplots(1, len(a.trace_speeds), figsize=(5.2 * len(a.trace_speeds), 4.4), facecolor="white", sharey=True)
        axes = np.atleast_1d(axes)
        A_mm, B_mm = r["A_m"] * 1e3, r["B_m"] * 1e3
        for ax, v in zip(axes, a.trace_speeds):
            rt = real_trace(d, v)
            if rt is not None:
                t0 = rt["t"][np.argmax(rt["depth"] > 0)] if np.any(rt["depth"] > 0) else 0.0
                ax.plot((rt["t"] - t0) * 1e3, rt["depth"] * 1e3, color=PALETTE[0], linewidth=2, label=f"real fingertip ({rt['label']})")
                for tt, vel in zip(rt["note_on_t"], rt["note_on_vel"]):
                    ax.axvline((tt - t0) * 1e3, color=PALETTE[0], linestyle="--", linewidth=1)
                    ax.annotate(f"real vel {vel}", ((tt - t0) * 1e3, B_mm), color=PALETTE[0], fontsize=8, xytext=(4, 6), textcoords="offset points")
            if tr_sim:
                cands = [k for k in tr_sim if k.startswith(note + "_")]
                if cands:
                    key = min(cands, key=lambda k: abs(float(k.split("_")[1]) - v))
                    if abs(float(key.split("_")[1]) - v) < 0.02:
                        tr = tr_sim[key]
                        kd = tr["key_depth"] * 1e3
                        t0s = tr["t"][np.argmax(kd > 0.05)] if np.any(kd > 0.05) else tr["t"][0]
                        ax.plot((tr["t"] - t0s) * 1e3, kd, color=PALETTE[1], linewidth=2, label=f"sim key ({first_model})")
                        ax.plot((tr["t"] - t0s) * 1e3, tr["ball_depth"] * 1e3, color=PALETTE[1], linewidth=1, alpha=0.5, linestyle=":", label="sim ball")
                        for tt, vel in zip(tr["note_on_t"], tr["note_on_vel"]):
                            ax.axvline((tt - t0s) * 1e3, color=PALETTE[1], linestyle="--", linewidth=1)
                            ax.annotate(f"sim vel {vel}", ((tt - t0s) * 1e3, B_mm - 1.5), color=PALETTE[1], fontsize=8, xytext=(4, -10), textcoords="offset points")
            for y, lab in ((A_mm, "A"), (B_mm, "B")):
                ax.axhline(y, color=GRID, linewidth=1)
                ax.annotate(lab, (ax.get_xlim()[0], y), color=TEXT2, fontsize=8, xytext=(3, 2), textcoords="offset points")
            ax.set_ylim(-3, 12)
            _style(ax, f"{note}  target {v:.2f} m/s", "time since key surface [ms]", "depth below key surface [mm]")
            ax.legend(frameon=False, fontsize=7, loc="lower right")
        fig.suptitle(f"Press trajectories: real fingertip vs simulated key ({note})", color=TEXT)
        fig.tight_layout()
        fig.savefig(os.path.join(a.sim_dir, f"fig_compare_traces_{note}.png"), dpi=140)
        plt.close(fig)

    # ------------------------------------------------------------------ table
    head = "| key | " + " | ".join(f"sim {m} MAE | bias" for m in sims) + " | n |"
    lines = [head, "|" + "---|" * (2 * len(sims) + 2)]
    for r in table_rows:
        cells = " | ".join(f"{r.get(f'{m}_mae', np.nan):.1f} | {r.get(f'{m}_bias', np.nan):+.1f}" for m in sims)
        lines.append(f"| {r['note']} | {cells} | {r.get(f'{first_model}_n', 0)} |")
    for m, (res, _) in sims.items():
        e = [
            x["sim_velocity"] - x["real_velocity_at_sim_speed"]
            for x in res
            if x["sim_velocity"] is not None and x.get("real_velocity_at_sim_speed") is not None and 1 < x["real_velocity_at_sim_speed"] < 127
        ]
        if e:
            lines.append(f"| **all ({m})** | {np.mean(np.abs(e)):.1f} | {np.mean(e):+.1f} |" + " | |" * (len(sims) - 1) + f" {len(e)} |")
    if note in reals:
        res, _ = sims[first_model]
        others = [m for m in sims if m != first_model]
        lines += [
            "",
            f"### {note}: per target speed (real median vs sim)",
            "| target m/s | real v_AB | real velocity | sim v_AB | sim " + first_model + " | " + " | ".join(f"sim {m}" for m in others) + " |",
            "|" + "---|" * (5 + len(others)),
        ]
        d, r = reals[note]
        st = {round(x["target_speed_m_s"], 4): x for x in r.get("speed_table", [])}
        other_maps = {m: {round(x["target_speed"], 4): x for x in sims[m][0] if x["note"] == note} for m in others}
        for x in sorted([x for x in res if x["note"] == note], key=lambda x: x["target_speed"]):
            s0 = st.get(round(x["target_speed"], 4))
            oth = " | ".join(str(other_maps[m].get(round(x["target_speed"], 4), {}).get("sim_velocity", "")) for m in others)
            real_cells = f"{s0['mean_speed_AB_m_s']:.3f} | {s0['velocity_median']:.0f}" if s0 else " | "
            lines.append(f"| {x['target_speed']:.3f} | {real_cells} | {x['sim_v_mean_AB']:.3f} | {x['sim_velocity']} | {oth} |")
    md = "\n".join(lines)
    with open(os.path.join(a.sim_dir, "compare_table.md"), "w") as f:
        f.write(md + "\n")
    print(md)
    print(f"\nwritten: {a.sim_dir}/fig_compare_overview.png, fig_compare_traces_{note}.png, compare_table.md")


if __name__ == "__main__":
    main()
