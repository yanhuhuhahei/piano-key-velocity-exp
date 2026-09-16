#!/usr/bin/env python
"""Analysis of a kve-identify session -> switch_id.json, results_si.json + figures.

kve-analyze-switches data/piano_id_C4
kve-analyze-switches data/piano_id_C4 --pool data/velmap_C4 data/hard_C4   # add more sessions of the same key
"""

from __future__ import annotations

import json
import os

import numpy as np

from key_velocity_exp.common import GRID, PALETTE, TEXT, TEXT2, TrialRecord, load_session

# --------------------------------------------------------------------------- kinematics helpers


def descent_crossing(rec: TrialRecord, depth_m: float):
    """Time the fingertip first reaches ``depth_m`` on the way down (linear interpolation)."""
    t, d = rec.col("t"), rec.depth()
    idx = np.flatnonzero(d >= depth_m)
    if len(idx) == 0 or idx[0] == 0:
        return np.nan
    i = idx[0]
    w = (depth_m - d[i - 1]) / (d[i] - d[i - 1]) if d[i] != d[i - 1] else 0.0
    return float(t[i - 1] + w * (t[i] - t[i - 1]))


def depth_at(rec: TrialRecord, t_q: float) -> float:
    return float(np.interp(t_q, rec.col("t"), rec.depth()))


def lift_crossing_depth(rec: TrialRecord, t_q: float):
    """Depth at time t_q if it lies inside the recorded lift (after max depth)."""
    t, d = rec.col("t"), rec.depth()
    i_b = int(np.argmax(d))
    if t_q < t[i_b] or t_q > t[-1]:
        return np.nan
    return float(np.interp(t_q, t[i_b:], d[i_b:]))


def mean_speed(rec: TrialRecord, d0: float, d1: float):
    t0, t1 = descent_crossing(rec, d0), descent_crossing(rec, d1)
    return (d1 - d0) / (t1 - t0) if np.isfinite(t0) and np.isfinite(t1) and t1 > t0 else np.nan


def first_on(rec):
    ons = [e for e in rec.midi_events if e.kind == "note_on"]
    return ons[0] if ons else None


def first_off_after(rec, t_on):
    offs = [e for e in rec.midi_events if e.kind == "note_off" and e.t > t_on]
    return offs[0] if offs else None


# --------------------------------------------------------------------------- phase b


def fit_b(directory: str, phase: str = "b") -> dict:
    xs, ys, offs, d_on_slow = [], [], [], []
    for rec in load_session(directory):
        if rec.extra.get("phase") != phase or rec.aborted:
            continue
        on = first_on(rec)
        if on is None:
            continue
        t_c = descent_crossing(rec, 0.0)
        v = mean_speed(rec, 0.001, 0.006)
        if not (np.isfinite(t_c) and np.isfinite(v)):
            continue
        xs.append(1.0 / v)
        ys.append(on.t - t_c)
        if v <= 0.035:
            d_on_slow.append(depth_at(rec, on.t))
        off = first_off_after(rec, on.t)
        if off is not None:
            d_off = lift_crossing_depth(rec, off.t)
            if np.isfinite(d_off):
                offs.append(d_off)
    if len(xs) < 3:
        out = {"ok": False, "reason": f"only {len(xs)} usable slow presses", "n": len(xs)}
        if d_on_slow:
            out["B_mm_slow"] = float(np.median(d_on_slow) * 1e3)
        return out
    xs, ys = np.array(xs), np.array(ys)
    A = np.vstack([xs, np.ones(len(xs))]).T
    (B, L), *_ = np.linalg.lstsq(A, ys, rcond=None)
    L_free = L
    if L < 0:  # a negative latency is unphysical: refit with L = 0 (B = weighted mean of v * delay)
        L = 0.0
        B = float(np.sum(xs * ys) / np.sum(xs * xs))
    pred = B * xs + L
    out = {
        "ok": True,
        "B_mm": float(B * 1e3),
        "latency_ms": float(L * 1e3),
        "latency_unconstrained_ms": float(L_free * 1e3),
        "rms_ms": float(np.sqrt(np.mean((pred - ys) ** 2)) * 1e3),
        "n": int(len(xs)),
        "points": [(float(x), float(y)) for x, y in zip(xs, ys)],
    }
    if d_on_slow:
        out["B_mm_slow"] = float(np.median(d_on_slow) * 1e3)  # depth at note-on of the <=0.03 m/s presses (latency-free check)
    if offs:
        out["noteoff_depth_mm"] = float(np.median(offs) * 1e3)
        out["noteoff_depth_std_mm"] = float(np.std(offs) * 1e3)
    return out


# --------------------------------------------------------------------------- phase a


def fit_a(directory: str) -> dict:
    before, after = [], []
    rows = []
    for rec in load_session(directory):
        if rec.extra.get("phase") != "a":
            continue
        d = rec.extra.get("probe_depth_mm")
        cls = rec.extra.get("probe_class")
        if d is None or cls is None:
            continue
        on = first_on(rec)
        rows.append((d, cls, on.velocity if on else None))
        (before if cls == "before" else after).append(d)
    if not before or not after:
        return {"ok": False, "reason": "need probes on both sides of A", "probes": rows}
    lo, hi = max(before), min(after)
    consistent = lo < hi
    return {"ok": True, "A_mm": float(0.5 * (lo + hi)), "A_lo_mm": float(lo), "A_hi_mm": float(hi), "consistent": bool(consistent), "probes": sorted(rows, key=lambda r: r[0])}


# --------------------------------------------------------------------------- phase m


def _refit(pts: list, out: dict) -> dict:
    """Recompute fits/tables of a mapping dict from (possibly pooled) points."""
    clean = [p for p in pts if not p["launched"] and not p["compressed"] and not p["braked_before_B"] and p["dt_ms"] > 0]
    good = [p for p in clean if p["kind"] in ("const", "kick") and 1 < p["velocity"] < 127]
    out = dict(out)
    out.update(
        {
            "n_points": len(pts),
            "n_used": len(good),
            "n_launched": sum(p["launched"] for p in pts),
            "n_compressed": sum(p["compressed"] for p in pts),
            "n_braked": sum(p["braked_before_B"] for p in pts),
            "points": pts,
        }
    )
    if len(good) < 4:
        return out
    x = np.log(np.array([p["dt_ms"] for p in good]))
    y = np.array([p["velocity"] for p in good], dtype=float)
    A = np.vstack([x, np.ones_like(x)]).T
    (c1, c0), *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = c0 + c1 * x
    r2 = 1 - np.sum((y - pred) ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-9)
    out["log_fit"] = {"formula": "velocity = c0 + c1 * ln(dt_ms)", "c0": float(c0), "c1": float(c1), "r2": float(r2), "rms": float(np.sqrt(np.mean((y - pred) ** 2)))}
    (b, a), *_ = np.linalg.lstsq(A, np.log(y), rcond=None)
    pred_p = np.exp(a + b * x)
    r2p = 1 - np.sum((y - pred_p) ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-9)
    out["power_fit"] = {
        "formula": "velocity = k * dt_ms^b",
        "k": float(np.exp(a)),
        "b": float(b),
        "r2": float(r2p),
        "rms": float(np.sqrt(np.mean((y - pred_p) ** 2))),
        "dt_range_ms": [float(np.exp(x.min())), float(np.exp(x.max()))],
    }
    out["best_fit"] = "power_fit" if r2p > r2 else "log_fit"
    try:
        from sklearn.isotonic import IsotonicRegression

        iso = IsotonicRegression(increasing=False, out_of_bounds="clip").fit(x, y)
        grid = np.geomspace(min(np.exp(x)), max(np.exp(x)), 30)
        out["table"] = [[float(g), float(v)] for g, v in zip(grid, iso.predict(np.log(grid)))]
    except Exception:  # pragma: no cover
        pass
    groups = {}
    for p in clean:
        key = p.get("session", "primary")
        groups.setdefault(key, []).append(p["velocity"] - (float(np.exp(a + b * np.log(p["dt_ms"]))) if out["best_fit"] == "power_fit" else float(c0 + c1 * np.log(p["dt_ms"]))))
    out["residual_by_shape"] = {k: {"mean": float(np.mean(v)), "std": float(np.std(v)), "n": len(v)} for k, v in groups.items()}
    return out


def _stop_depth_mm(reason: str):
    import re

    m = re.search(r"at ([0-9.]+) mm", reason or "")
    return float(m.group(1)) if m else np.nan


def mapping(directory: str, A_m: float, B_m: float, L_s: float, B_by_phase: dict | None = None, tol_m: float = 0.0008) -> dict:
    """Per press: dt of the FINGER between A and B (crossing times of the measured depth), mean of the
    measured TCP speed inside A..B, and the MIDI note-on. The contact reference can drift between
    phases, so B (and A, shifted by the same amount) are taken per phase when ``B_by_phase`` is given.
    'launched': note-on arrived while the finger was still > tol above B (hammer/key ahead).
    'braked_before_B': a force/depth fallback stopped the arm above B (speed through A->B disturbed).
    The MIDI latency is re-estimated from the fast presses (how far past B the finger was at note-on)."""
    raw = []
    for rec in load_session(directory):
        ph = rec.extra.get("phase")
        if ph not in ("m", "b", "b2", "s") or rec.aborted:
            continue
        on = first_on(rec)
        if on is None:
            continue
        B_ph = (B_by_phase or {}).get(ph, B_m)
        A_ph = A_m + (B_ph - B_m)
        t, d, vz = rec.col("t"), rec.depth(), -rec.col("vz")
        t_A, t_Bc = descent_crossing(rec, A_ph), descent_crossing(rec, B_ph)
        if not np.isfinite(t_A):
            continue
        d_on = depth_at(rec, on.t)
        v_on = float(np.interp(on.t, t, vz))
        lead = B_ph - d_on  # > 0: note-on before the finger reached B
        i_b = int(np.argmax(d))
        inside = (d[: i_b + 1] >= A_ph) & (d[: i_b + 1] <= B_ph)
        if inside.sum() >= 3:
            v_ab = float(np.mean(vz[: i_b + 1][inside]))
        elif np.isfinite(t_Bc) and t_Bc > t_A:
            v_ab = float((B_ph - A_ph) / (t_Bc - t_A))
        else:
            v_ab = np.nan
        dt_finger = (t_Bc - t_A) if np.isfinite(t_Bc) and t_Bc > t_A else np.nan
        dt_midi = on.t - L_s - t_A
        reason = str(rec.extra.get("stop_reason", ""))
        stop_d = _stop_depth_mm(reason) if (reason.startswith("force") or reason.startswith("depth")) else np.nan
        braked = bool(np.isfinite(stop_d) and stop_d < B_ph * 1e3 - 0.1)
        raw.append(
            dict(
                trial_id=rec.trial_id,
                phase=ph,
                kind=rec.profile.kind,
                label=rec.profile.label(),
                shape=rec.extra.get("shape"),
                target_speed=rec.extra.get("target_speed"),
                how=rec.extra.get("how"),
                velocity=int(on.velocity),
                v_mean_AB=v_ab,
                v_at_B=float(np.interp(t_Bc, t, vz)) if np.isfinite(t_Bc) else v_on,
                v_at_on=v_on,
                dt_finger_ms=float(dt_finger * 1e3) if np.isfinite(dt_finger) else np.nan,
                dt_midi_ms=float(dt_midi * 1e3),
                depth_at_on_mm=float(d_on * 1e3),
                lead_mm=float(lead * 1e3),
                stop_depth_mm=stop_d,
                braked_before_B=braked,
                launched=bool(lead > tol_m),
            )
        )
    # MIDI latency from the fast presses: the finger overshoots B by v * L before the note arrives
    fast = [p for p in raw if not p["launched"] and p["v_at_on"] > 0.15 and p["lead_mm"] < 0]
    L_fast = np.nan
    if len(fast) >= 3:
        x = np.array([p["v_at_on"] for p in fast])
        y = np.array([-p["lead_mm"] * 1e-3 for p in fast])
        L_fast = float(np.sum(x * y) / np.sum(x * x))  # through the origin: lag = v * L
    L_use = max(L_s, L_fast if np.isfinite(L_fast) else 0.0)
    pts = []
    for p in raw:
        expected_overshoot = max(p["v_at_on"], 0.0) * L_use
        p["compressed"] = bool(-p["lead_mm"] * 1e-3 > tol_m + expected_overshoot)
        p["dt_ms"] = p["dt_finger_ms"] if np.isfinite(p["dt_finger_ms"]) else p["dt_midi_ms"]
        p["dt_source"] = "finger crossings" if np.isfinite(p["dt_finger_ms"]) else "midi - latency"
        pts.append(p)
    clean = [p for p in pts if not p["launched"] and not p["compressed"] and not p["braked_before_B"] and p["dt_ms"] > 0]
    good = [p for p in clean if p["kind"] in ("const", "kick") and 1 < p["velocity"] < 127]
    accel = [p for p in clean if p["kind"] not in ("const", "kick")]
    out = {
        "n_points": len(pts),
        "n_used": len(good),
        "n_accelerating": len(accel),
        "n_launched": sum(p["launched"] for p in pts),
        "n_compressed": sum(p["compressed"] for p in pts),
        "n_braked": sum(p["braked_before_B"] for p in pts),
        "latency_from_fast_ms": float(L_fast * 1e3) if np.isfinite(L_fast) else None,
        "n_velocity_floor": sum(p["velocity"] <= 1 for p in pts),
        "points": pts,
    }
    if len(good) >= 4:
        x = np.log(np.array([p["dt_ms"] for p in good]))
        y = np.array([p["velocity"] for p in good], dtype=float)
        A = np.vstack([x, np.ones_like(x)]).T
        (c1, c0), *_ = np.linalg.lstsq(A, y, rcond=None)
        pred = c0 + c1 * x
        r2 = 1 - np.sum((y - pred) ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-9)
        out["log_fit"] = {"formula": "velocity = c0 + c1 * ln(dt_ms)", "c0": float(c0), "c1": float(c1), "r2": float(r2), "rms": float(np.sqrt(np.mean((y - pred) ** 2)))}
        (b, a), *_ = np.linalg.lstsq(A, np.log(y), rcond=None)
        pred_p = np.exp(a + b * x)
        r2p = 1 - np.sum((y - pred_p) ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-9)
        out["power_fit"] = {
            "formula": "velocity = k * dt_ms^b",
            "k": float(np.exp(a)),
            "b": float(b),
            "r2": float(r2p),
            "rms": float(np.sqrt(np.mean((y - pred_p) ** 2))),
            "dt_range_ms": [float(np.exp(x.min())), float(np.exp(x.max()))],
        }
        if r2p > r2:
            c_pred = lambda dd: float(np.exp(a + b * np.log(dd)))  # noqa: E731
            c_inv = lambda v: float(np.exp((np.log(v) - a) / b)) if v > 0 else np.nan  # noqa: E731
            out["best_fit"] = "power_fit"
        else:
            c_pred = lambda dd: float(c0 + c1 * np.log(dd))  # noqa: E731
            c_inv = lambda v: float(np.exp((v - c0) / c1)) if c1 != 0 else np.nan  # noqa: E731
            out["best_fit"] = "log_fit"
        try:
            from sklearn.isotonic import IsotonicRegression

            iso = IsotonicRegression(increasing=False, out_of_bounds="clip").fit(x, y)
            grid = np.geomspace(min(np.exp(x)), max(np.exp(x)), 30)
            out["table"] = [[float(g), float(v)] for g, v in zip(grid, iso.predict(np.log(grid)))]
            out["table_rms"] = float(np.sqrt(np.mean((y - iso.predict(x)) ** 2)))
        except Exception as e:  # pragma: no cover
            out["table_error"] = str(e)
        groups = {}
        for p in clean:
            key = p["shape"] or (p["label"] if p["kind"] != "const" else ("const(m/s)" if p["phase"] in ("m", "s") else "const(b)"))
            groups.setdefault(key, []).append(p["velocity"] - c_pred(p["dt_ms"]))
        out["residual_by_shape"] = {k: {"mean": float(np.mean(v)), "std": float(np.std(v)), "n": len(v)} for k, v in groups.items()}
        for p in pts:
            p["dt_implied_ms"] = c_inv(p["velocity"])
            p["finger_dt_over_implied"] = float(p["dt_ms"] / p["dt_implied_ms"]) if p["dt_implied_ms"] and p["dt_implied_ms"] > 0 else np.nan
    return out


# --------------------------------------------------------------------------- figures


def _style(ax, title, xl, yl):
    ax.set_title(title, color=TEXT, loc="left", fontsize=11)
    ax.set_xlabel(xl, color=TEXT2)
    ax.set_ylabel(yl, color=TEXT2)
    ax.grid(True, color=GRID, linewidth=0.8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=TEXT2)


def figures(directory: str, res: dict):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    b = res.get("B")
    if b and b.get("ok"):
        fig, ax = plt.subplots(figsize=(6.5, 4), facecolor="white")
        x, y = np.array(b["points"]).T
        ax.scatter(x, y * 1e3, s=28, color=PALETTE[0], edgecolors="white", linewidths=1, label="slow presses")
        xx = np.linspace(0, x.max() * 1.05, 20)
        ax.plot(xx, (b["B_mm"] * 1e-3 * xx + b["latency_ms"] * 1e-3) * 1e3, color=PALETTE[1], linewidth=2, label=f"B = {b['B_mm']:.2f} mm, latency = {b['latency_ms']:.1f} ms")
        _style(ax, "Switch B: contact → note-on delay vs 1/speed", "1 / speed [s/m]", "delay [ms]")
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(directory, "fig_B_fit.png"), dpi=150)
        plt.close(fig)
    a = res.get("A")
    if a and a.get("probes"):
        fig, ax = plt.subplots(figsize=(6.5, 3.6), facecolor="white")
        for d, cls, vel in a["probes"]:
            ax.scatter(d, vel if vel is not None else 0, s=40, color=PALETTE[0] if cls == "before" else PALETTE[1], edgecolors="white", linewidths=1)
        if a.get("ok"):
            ax.axvline(a["A_mm"], color=TEXT2, linestyle="--", linewidth=1.2)
            ax.annotate(f"A = {a['A_mm']:.2f} mm", (a["A_mm"], ax.get_ylim()[1] * 0.9), color=TEXT2, fontsize=9, xytext=(5, 0), textcoords="offset points")
        ax.scatter([], [], color=PALETTE[0], label="timer started after the pause (d < A)")
        ax.scatter([], [], color=PALETTE[1], label="pause inside dt (d > A)")
        _style(ax, "Switch A: velocity of pause-probe presses vs pause depth", "pause depth [mm]", "MIDI velocity (0 = no note)")
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(directory, "fig_A_probe.png"), dpi=150)
        plt.close(fig)
    m = res.get("mapping")
    if m and m.get("log_fit"):
        fig, ax = plt.subplots(figsize=(6.5, 4.2), facecolor="white")
        ok = [p for p in m["points"] if not p["launched"] and not p["compressed"] and not p["braked_before_B"]]
        shapes = list(dict.fromkeys(("constant speed" if p["kind"] == "const" else (p["shape"] or p["kind"])) for p in ok))
        for i, sh in enumerate(shapes):
            pts = [p for p in ok if ("constant speed" if p["kind"] == "const" else (p["shape"] or p["kind"])) == sh]
            ax.scatter([p["dt_ms"] for p in pts], [p["velocity"] for p in pts], s=28, color=PALETTE[i % len(PALETTE)], edgecolors="white", linewidths=1, label=sh)
        la = [p for p in m["points"] if p["launched"]]
        if la:
            ax.scatter([p["dt_ms"] for p in la], [p["velocity"] for p in la], s=28, facecolors="none", edgecolors=TEXT2, label="key launched ahead of finger (excluded)")
        br = [p for p in m["points"] if p["braked_before_B"] and not p["compressed"] and not p["launched"]]
        if br:
            ax.scatter([p["dt_ms"] for p in br], [p["velocity"] for p in br], s=28, marker="s", facecolors="none", edgecolors=PALETTE[3], label="arm braked before note-on (excluded)")
        co = [p for p in m["points"] if p["compressed"]]
        if co:
            ax.scatter([p["dt_ms"] for p in co], [p["velocity"] for p in co], s=34, marker="x", color=TEXT2, label="tip compressed, TCP ≠ key (excluded)")
        xx = np.geomspace(min(p["dt_ms"] for p in m["points"]), max(p["dt_ms"] for p in m["points"]), 40)
        if m.get("best_fit") == "power_fit":
            f = m["power_fit"]
            ax.plot(xx, f["k"] * xx ** f["b"], color=TEXT2, linewidth=1.5, label=f"{f['k']:.0f}·dt^{f['b']:.2f}  R²={f['r2']:.3f}")
        else:
            f = m["log_fit"]
            ax.plot(xx, f["c0"] + f["c1"] * np.log(xx), color=TEXT2, linewidth=1.5, label=f"{f['c0']:.1f} {f['c1']:+.1f}·ln(dt)  R²={f['r2']:.3f}")
        if m.get("table"):
            tb = np.array(m["table"])
            ax.step(tb[:, 0], tb[:, 1], where="mid", color=PALETTE[7], linewidth=1, alpha=0.7, label="isotonic table")
        ax.set_xscale("log")
        _style(ax, "finger dt (A→B) → MIDI velocity  (fit: presses that kept driving the key)", "finger dt between A and B [ms]", "MIDI velocity")
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(directory, "fig_dt_velocity.png"), dpi=150)
        plt.close(fig)


def fig_speed(directory: str, res: dict):
    tb = res.get("speed_table")
    if not tb:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.8, 4.2), facecolor="white")
    pts = [p for p in res["mapping"]["points"] if p.get("target_speed") is not None and not p["launched"] and not p["compressed"] and not p["braked_before_B"] and p["kind"] in ("const", "kick")]
    ax.scatter([p["v_mean_AB"] for p in pts], [p["velocity"] for p in pts], s=22, color=PALETTE[0], alpha=0.6, edgecolors="white", linewidths=0.8, label="presses (key driven through A→B)")
    ex = [p for p in res["mapping"]["points"] if p.get("target_speed") is not None and (p["launched"] or p["braked_before_B"])]
    if ex:
        ax.scatter([p["v_mean_AB"] for p in ex], [p["velocity"] for p in ex], s=26, facecolors="none", edgecolors=TEXT2, label="hammer ahead / braked before B (excluded)")
    vm = res.get("velocity_map")
    if vm:
        arr = np.array(vm["speed_to_velocity"])
        ax.step(arr[:, 0], arr[:, 1], where="mid", color=PALETTE[1], linewidth=2, label="isotonic map")
    for y in (1, 127):
        ax.axhline(y, color=GRID, linewidth=1)
    ax.set_xscale("log")
    _style(ax, "MIDI velocity vs mean key speed A→B", "mean speed A→B [m/s]", "MIDI velocity")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(directory, "fig_velocity_vs_speed.png"), dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- SI outputs


def write_si(directory: str, res: dict) -> dict:
    """results_si.json (metres, seconds, m/s) and a 2x2 summary figure in SI units."""
    b, a, m = res["B"], res["A"], res["mapping"]
    model = res["model_for_sim"]
    A_m, B_m = model["A_mm"] * 1e-3, model["B_mm"] * 1e-3
    L_s = model["latency_ms"] * 1e-3
    L_fast = m.get("latency_from_fast_ms")
    si = {
        "units": {"depth": "m (below the key surface = home)", "time": "s", "speed": "m/s", "velocity": "MIDI 1..127"},
        "A_m": A_m,
        "A_source": res.get("A_source"),
        "A_bracket_m": [a["A_lo_mm"] * 1e-3, a["A_hi_mm"] * 1e-3] if a.get("ok") else None,
        "B_m": B_m,
        "B_fit_rms_s": b["rms_ms"] * 1e-3 if b.get("ok") else None,
        "B_check_slow_noteon_depth_m": b.get("B_mm_slow", np.nan) * 1e-3 if b.get("ok") else None,
        "noteoff_depth_m": (b.get("noteoff_depth_mm") or np.nan) * 1e-3 if b.get("ok") else None,
        "latency_s": L_s,
        "latency_from_fast_presses_s": L_fast * 1e-3 if L_fast is not None else None,
        "AB_distance_m": B_m - A_m,
        "drift_m": {k: v * 1e-3 for k, v in res.get("drift_mm_vs_phase_b", {}).items()},
    }
    pf = m.get("power_fit")
    si["best_fit"] = m.get("best_fit")
    if pf:
        # velocity = k * dt_ms^b = k * 1000^b * dt_s^b
        si["velocity_from_dt"] = {
            "formula": "velocity = k * dt_s^b",
            "k": pf["k"] * 1000.0 ** pf["b"],
            "b": pf["b"],
            "r2": pf["r2"],
            "rms": pf["rms"],
            "dt_range_s": [d * 1e-3 for d in pf["dt_range_ms"]],
        }
    lf = m.get("log_fit")
    if lf:
        si["velocity_from_dt_loglinear"] = {"formula": "velocity = c0 + c1 * ln(dt_s)", "c0": lf["c0"] + lf["c1"] * np.log(1000.0), "c1": lf["c1"], "r2": lf["r2"], "rms": lf["rms"]}
    if m.get("table"):
        si["dt_to_velocity_table"] = [[d * 1e-3, v] for d, v in m["table"]]
    pts = [p for p in m["points"] if p["kind"] in ("const", "kick") and not p["launched"] and not p["braked_before_B"]]
    floor = [p["dt_ms"] for p in pts if p["velocity"] <= 1]
    above = [p["dt_ms"] for p in pts if p["velocity"] > 1]
    if floor and above:
        si["velocity_floor_dt_s"] = {"longest_dt_with_velocity_gt_1_s": max(above) * 1e-3, "shortest_dt_with_velocity_1_s": min(floor) * 1e-3}
    vm = res.get("velocity_map")
    if vm:
        si["speed_to_velocity"] = vm["speed_to_velocity"]
        si["velocity_to_speed_m_s"] = vm["velocity_to_speed"]
        si["covered_velocity"] = vm["covered_velocity"]
        si["covered_speed_m_s"] = vm["covered_speed_m_s"]
    if res.get("speed_table"):
        si["speed_table"] = [
            {
                "target_speed_m_s": r["target_speed"],
                "mean_speed_AB_m_s": r["v_mean_AB"],
                "dt_s": r["dt_ms"] * 1e-3,
                "velocity_median": r["velocity_median"],
                "velocity_min": r["velocity_min"],
                "velocity_max": r["velocity_max"],
                "n": r["n"],
            }
            for r in res["speed_table"]
        ]
    # ---- full 1..127 model: measured range from the isotonic map, beyond it the fitted curve (flagged)
    if pf and vm:
        k_si, bexp = si["velocity_from_dt"]["k"], pf["b"]
        use_log = m.get("best_fit") == "log_fit" and lf is not None
        c0s = si["velocity_from_dt_loglinear"]["c0"] if use_log else None
        c1s = lf["c1"] if use_log else None

        def vel_of_dt(dt_s):
            return float(c0s + c1s * np.log(dt_s)) if use_log else float(k_si * dt_s**bexp)

        def dt_of_vel(vel):
            return float(np.exp((vel - c0s) / c1s)) if use_log else float((vel / k_si) ** (1.0 / bexp))

        fit_label = f"velocity = {c0s:.1f} {c1s:+.1f}·ln(dt)" if use_log else f"velocity = {k_si:.3g}·dt^{bexp:.2f}"
        dAB = B_m - A_m
        vmin_meas, vmax_meas = vm["covered_speed_m_s"]
        velmin_meas, velmax_meas = vm["covered_velocity"]
        rows = []
        for vel in range(1, 128):
            dt_fit = dt_of_vel(vel)  # s
            v_fit = dAB / dt_fit
            v_meas = vm["velocity_to_speed"].get(str(vel)) if isinstance(next(iter(vm["velocity_to_speed"])), str) else vm["velocity_to_speed"].get(vel)
            if v_meas is not None and velmin_meas < vel <= velmax_meas:
                rows.append({"velocity": vel, "mean_speed_AB_m_s": v_meas, "dt_s": dAB / v_meas, "source": "measured (isotonic map)"})
            elif vel <= 1:
                rows.append({"velocity": vel, "mean_speed_AB_m_s": None, "dt_s": None, "source": "timeout: any dt above the floor"})
            else:
                rows.append({"velocity": vel, "mean_speed_AB_m_s": v_fit, "dt_s": dt_fit, "source": "extrapolated (" + ("log-linear" if use_log else "power law") + ")"})
        si["velocity_to_speed_full_1_127"] = rows
        si["speed_for_velocity_127_m_s"] = rows[-1]["mean_speed_AB_m_s"]
        si["dt_for_velocity_127_s"] = rows[-1]["dt_s"]
        si["extrapolation_note"] = f"measured up to velocity {velmax_meas} ({vmax_meas:.3f} m/s); values above come from the better fit ({fit_label}) and carry its uncertainty"
    with open(os.path.join(directory, "results_si.json"), "w") as f:
        json.dump(si, f, indent=2, default=float)

    # ---- model figure: measured + extrapolated to 127
    if pf and vm:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7.2, 4.6), facecolor="white")
        good = [p for p in m["points"] if p["kind"] in ("const", "kick") and not p["launched"] and not p["braked_before_B"] and not p["compressed"]]
        ax.scatter([p["v_mean_AB"] for p in good], [p["velocity"] for p in good], s=24, color=PALETTE[0], alpha=0.7, edgecolors="white", linewidths=0.8, label="measured presses")
        v_meas_grid = np.geomspace(vmin_meas, vmax_meas, 60)
        ax.plot(v_meas_grid, np.clip([vel_of_dt(dAB / v) for v in v_meas_grid], 1, 127), color=PALETTE[1], linewidth=2, label="fit  " + fit_label)
        v127 = si["speed_for_velocity_127_m_s"]
        v_ext = np.geomspace(vmax_meas, max(v127 * 1.05, vmax_meas * 1.01), 40)
        ax.plot(v_ext, np.clip([vel_of_dt(dAB / v) for v in v_ext], 1, 127), color=PALETTE[1], linewidth=2, linestyle="--", label="extrapolated")
        ax.axhline(127, color=GRID, linewidth=1)
        ax.axvline(v127, color=TEXT2, linestyle=":", linewidth=1.2)
        ax.annotate(f"127 @ {v127:.2f} m/s (dt {si['dt_for_velocity_127_s'] * 1e3:.1f} ms)", (v127, 127), color=TEXT2, fontsize=9, xytext=(-8, -14), textcoords="offset points", ha="right")
        ax.axvspan(vmin_meas, vmax_meas, color=PALETTE[0], alpha=0.05, label="measured speed range")
        ax.set_xscale("log")
        ax.set_ylim(0, 135)
        _style(ax, "Velocity model: MIDI velocity vs mean key speed A→B (SI)", "mean speed A→B [m/s]", "MIDI velocity")
        ax.legend(frameon=False, fontsize=8, loc="upper left")
        fig.tight_layout()
        fig.savefig(os.path.join(directory, "fig_velocity_model_si.png"), dpi=150)
        plt.close(fig)

    # ---- summary figure
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), facecolor="white")
    ax = axes[0, 0]
    if b.get("ok"):
        x, y = np.array(b["points"]).T
        ax.scatter(x, y, s=28, color=PALETTE[0], edgecolors="white", linewidths=1, label="slow constant-speed presses")
        xx = np.linspace(0, x.max() * 1.05, 20)
        ax.plot(xx, B_m * xx + L_s, color=PALETTE[1], linewidth=2, label=f"fit: B = {B_m * 1e3:.2f} mm ({B_m:.5f} m), L = {L_s * 1e3:.1f} ms")
        ax.legend(frameon=False, fontsize=8)
    _style(ax, "Switch B: (t_note-on − t_contact) = B / v + L", "1 / key speed [s/m]", "delay [s]")
    ax = axes[0, 1]
    if a.get("probes"):
        for d, cls, vel in a["probes"]:
            ax.scatter(d * 1e-3, vel if vel is not None else 0, s=40, color=PALETTE[0] if cls == "before" else PALETTE[1], edgecolors="white", linewidths=1)
        ax.scatter([], [], color=PALETTE[0], label="pause before the timer starts (d < A)")
        ax.scatter([], [], color=PALETTE[1], label="pause inside dt (d > A): velocity collapses")
        if a.get("ok"):
            ax.axvspan(a["A_lo_mm"] * 1e-3, a["A_hi_mm"] * 1e-3, color=PALETTE[3], alpha=0.25, label=f"A = {A_m * 1e3:.2f} mm ({A_m:.5f} m)")
        if si.get("noteoff_depth_m"):
            ax.axvline(si["noteoff_depth_m"], color=TEXT2, linestyle=":", linewidth=1.2, label=f"note-off depth {si['noteoff_depth_m'] * 1e3:.2f} mm")
        ax.legend(frameon=False, fontsize=8)
    _style(ax, "Switch A: pause-probe classification", "pause depth [m]", "MIDI velocity (0 = no note)")
    ax = axes[1, 0]
    good = [p for p in m["points"] if p["kind"] in ("const", "kick") and not p["launched"] and not p["braked_before_B"] and not p["compressed"]]
    if good:
        ax.scatter([p["dt_ms"] * 1e-3 for p in good], [p["velocity"] for p in good], s=24, color=PALETTE[0], alpha=0.7, edgecolors="white", linewidths=0.8, label="presses (key driven A→B)")
        if pf:
            dd = np.geomspace(min(p["dt_ms"] for p in good), max(p["dt_ms"] for p in good), 60) * 1e-3
            if m.get("best_fit") == "log_fit" and lf:
                c0s, c1s = si["velocity_from_dt_loglinear"]["c0"], lf["c1"]
                ax.plot(dd, np.clip(c0s + c1s * np.log(dd), 0, 127), color=PALETTE[1], linewidth=2, label=f"velocity = {c0s:.1f} {c1s:+.1f}·ln(dt)  (R²={lf['r2']:.3f})")
            else:
                k_si = si["velocity_from_dt"]["k"]
                ax.plot(dd, np.clip(k_si * dd ** pf["b"], 0, 127), color=PALETTE[1], linewidth=2, label=f"velocity = {k_si:.3g}·dt^{pf['b']:.2f}  (R²={pf['r2']:.3f})")
        if m.get("table"):
            tb = np.array(m["table"])
            ax.step(tb[:, 0] * 1e-3, tb[:, 1], where="mid", color=TEXT2, linewidth=1, alpha=0.7, label="isotonic table")
        if "velocity_floor_dt_s" in si:
            ax.axvline(si["velocity_floor_dt_s"]["shortest_dt_with_velocity_1_s"], color=PALETTE[7], linestyle="--", linewidth=1, label="timeout → velocity 1")
        ax.set_xscale("log")
        ax.legend(frameon=False, fontsize=8)
    _style(ax, "MIDI velocity vs dt between A and B", "dt (A→B) [s]", "MIDI velocity")
    ax = axes[1, 1]
    if good:
        ax.scatter([p["v_mean_AB"] for p in good], [p["velocity"] for p in good], s=24, color=PALETTE[0], alpha=0.7, edgecolors="white", linewidths=0.8, label="presses")
        if vm:
            arr = np.array(vm["speed_to_velocity"])
            ax.step(arr[:, 0], arr[:, 1], where="mid", color=PALETTE[1], linewidth=2, label="isotonic map")
        ax.set_xscale("log")
        for yv in (1, 127):
            ax.axhline(yv, color=GRID, linewidth=1)
        ax.legend(frameon=False, fontsize=8)
    _style(ax, "MIDI velocity vs mean key speed A→B", "mean speed A→B [m/s]", "MIDI velocity")
    fig.suptitle(
        f"A = {A_m:.5f} m, B = {B_m:.5f} m, A→B = {(B_m - A_m):.5f} m, latency ≈ {L_s * 1e3:.1f} ms" + (f" (fast presses: {L_fast:.1f} ms)" if L_fast is not None else ""), color=TEXT, fontsize=11
    )
    fig.tight_layout()
    fig.savefig(os.path.join(directory, "fig_summary_si.png"), dpi=150)
    plt.close(fig)
    return si


# --------------------------------------------------------------------------- main


def analyze(directory: str, make_plots: bool = True, pool: list | None = None) -> dict:
    """``pool``: extra session directories of the same key whose presses are added to the dt/velocity
    mapping (A/B/latency from ``directory``; each pooled session is re-referenced by its own slow
    presses so a small home difference does not bias dt)."""
    res = {"B": fit_b(directory), "A": fit_a(directory)}
    prior = {}
    sess = os.path.join(directory, "session.json")
    if os.path.exists(sess):
        with open(sess) as f:
            prior = json.load(f)
    B_m = res["B"]["B_mm"] * 1e-3 if res["B"].get("ok") else (prior.get("B_mm", 8.2) * 1e-3)
    L_s = res["B"]["latency_ms"] * 1e-3 if res["B"].get("ok") else (prior.get("latency_ms", 1.0) * 1e-3)
    if res["A"].get("ok") and res["A"].get("consistent"):
        A_m = res["A"]["A_mm"] * 1e-3
        res["A_source"] = "pause probe"
    elif res["B"].get("noteoff_depth_mm") is not None:
        # the key releases switch A on the way up: note-off depth is an independent estimate of A
        A_m = res["B"]["noteoff_depth_mm"] * 1e-3
        res["A_source"] = "note-off depth (probe missing or inconsistent)"
    elif prior.get("A_mm"):
        A_m = prior["A_mm"] * 1e-3
        res["A_source"] = "session.json (--ab-from / earlier phases)"
    else:
        A_m = 0.003
        res["A_source"] = "default"
    # reference drift: depth at note-on of the slowest presses, per phase
    B_by_phase = {"b": B_m}
    drift = {}
    for ph in ("m", "b2"):
        fb = fit_b(directory, phase=ph)
        if "B_mm_slow" in fb and res["B"].get("B_mm_slow") is not None:
            B_by_phase[ph] = fb["B_mm_slow"] * 1e-3 - 0.025 * L_s  # slow presses (~0.025 m/s) overshoot B by v * latency before the note arrives
            drift[ph] = float((B_by_phase[ph] - B_m) * 1e3)
    res["drift_mm_vs_phase_b"] = drift
    res["mapping"] = mapping(directory, A_m, B_m, L_s, B_by_phase)
    if pool:
        for extra in pool:
            # re-reference: B of the pooled session from its own slow presses (fit_b over phases b/s/m)
            Bx = None
            for ph in ("b", "s", "m"):
                fb = fit_b(extra, phase=ph)
                if fb.get("B_mm_slow") is not None:
                    Bx = fb["B_mm_slow"] * 1e-3 - 0.025 * L_s
                    break
            shift = (Bx - B_m) if Bx is not None else 0.0
            mx = mapping(extra, A_m + shift, B_m + shift, L_s)
            for p in mx["points"]:
                p["session"] = os.path.basename(os.path.normpath(extra))
            res["mapping"]["points"] += mx["points"]
            res.setdefault("pooled_sessions", []).append({"dir": extra, "n_points": mx["n_points"], "B_shift_mm": shift * 1e3})
        # refit on the pooled points
        pooled = res["mapping"]["points"]
        res["mapping"] = _refit(pooled, res["mapping"])
    res["model_for_sim"] = {
        "A_mm": float(A_m * 1e3),
        "B_mm": float(B_m * 1e3),
        "A_source": res.get("A_source", "default"),
        "B_source": "measured" if res["B"].get("ok") else "default",
        "latency_ms": float(L_s * 1e3),
        "noteoff_depth_mm": res["B"].get("noteoff_depth_mm"),
        "velocity_from_dt": res["mapping"].get(res["mapping"].get("best_fit", "log_fit")),
        "velocity_from_dt_alternatives": {"log_fit": res["mapping"].get("log_fit"), "power_fit": res["mapping"].get("power_fit")},
        "table_dt_ms_velocity": res["mapping"].get("table"),
    }
    # velocity map by target speed (phase s / m constant & kick presses that drove the key)
    m = res["mapping"]
    by_speed: dict = {}
    for p in m["points"]:
        if p.get("target_speed") is None or p["launched"] or p["compressed"] or p["braked_before_B"] or p["kind"] not in ("const", "kick"):
            continue
        by_speed.setdefault(round(p["target_speed"], 4), []).append(p)
    table = []
    for v in sorted(by_speed):
        ps = by_speed[v]
        table.append(
            {
                "target_speed": v,
                "n": len(ps),
                "v_mean_AB": float(np.median([q["v_mean_AB"] for q in ps])),
                "velocity_median": float(np.median([q["velocity"] for q in ps])),
                "velocity_min": int(min(q["velocity"] for q in ps)),
                "velocity_max": int(max(q["velocity"] for q in ps)),
                "hammer_lead_mm": float(np.median([q["lead_mm"] for q in ps])),
                "dt_ms": float(np.median([q["dt_ms"] for q in ps])),
            }
        )
    res["speed_table"] = table
    if len(table) >= 3:
        vv = np.array([r["v_mean_AB"] for r in table])
        yy = np.array([r["velocity_median"] for r in table])
        try:
            from sklearn.isotonic import IsotonicRegression

            iso = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(np.log(vv), yy)
            grid_v = np.geomspace(vv.min(), vv.max(), 40)
            fwd = iso.predict(np.log(grid_v))
            # inverse: for each MIDI velocity the (lowest) mean speed A->B that reaches it, within the covered range
            inv = {}
            for vel in range(1, 128):
                idx = np.flatnonzero(fwd >= vel)
                inv[vel] = float(grid_v[idx[0]]) if len(idx) else None
            res["velocity_map"] = {
                "speed_to_velocity": [[float(a), float(b)] for a, b in zip(grid_v, fwd)],
                "velocity_to_speed": inv,
                "covered_velocity": [int(yy.min()), int(yy.max())],
                "covered_speed_m_s": [float(vv.min()), float(vv.max())],
            }
        except Exception as e:  # pragma: no cover
            res["velocity_map_error"] = str(e)
    with open(os.path.join(directory, "switch_id.json"), "w") as f:
        json.dump(res, f, indent=2, default=float)
    if make_plots:
        figures(directory, res)
        fig_speed(directory, res)
    res["si"] = write_si(directory, res)
    return res


def print_report(res: dict):
    b, a, m = res["B"], res["A"], res["mapping"]
    print("\n=== switch identification ===")
    if b.get("ok"):
        print(f"B (note-on switch) = {b['B_mm']:.2f} mm below contact, MIDI latency = {b['latency_ms']:.1f} ms (rms {b['rms_ms']:.1f} ms, n={b['n']})")
        if "noteoff_depth_mm" in b:
            print(f"note-off on the way up at {b['noteoff_depth_mm']:.2f} ± {b['noteoff_depth_std_mm']:.2f} mm")
    else:
        print("B: not fitted -", b.get("reason"))
    if a.get("ok"):
        print(f"A (timer-start switch) = {a['A_mm']:.2f} mm  (bracketed {a['A_lo_mm']:.2f} .. {a['A_hi_mm']:.2f}{'' if a['consistent'] else ', INCONSISTENT probes'})")
    else:
        print("A: not found -", a.get("reason"))
    print(f"A used for dt: {res['model_for_sim']['A_mm']:.2f} mm  [{res.get('A_source')}]")
    if res.get("drift_mm_vs_phase_b"):
        print("contact reference drift vs phase b (from the slowest presses): " + ", ".join(f"{k}: {v:+.2f} mm" for k, v in res["drift_mm_vs_phase_b"].items()))
    if b.get("ok") and b.get("B_mm_slow") is not None:
        print(f"check: depth at note-on of the <=0.03 m/s presses = {b['B_mm_slow']:.2f} mm (unconstrained latency fit was {b.get('latency_unconstrained_ms', float('nan')):.1f} ms)")
    if m.get("latency_from_fast_ms") is not None:
        print(f"MIDI latency from the fast presses (finger overshoot past B at note-on): {m['latency_from_fast_ms']:.1f} ms")
    print(
        f"mapping points: {m['n_points']}, fit uses {m['n_used']} driven presses (constant speed + kicks with note-on at B); {m.get('n_accelerating', 0)} other shapes for comparison  (excluded: {m['n_launched']} note-on before the finger reached B = key/hammer ran ahead, {m['n_compressed']} finger deeper than B at note-on = tip compressed, {m['n_braked']} arm braked before note-on)"
    )
    comp = [-p["lead_mm"] for p in m["points"] if p["compressed"]]
    if comp:
        print(f"   finger beyond B at note-on on the excluded presses (more than latency explains): median {np.median(comp):.1f} mm, max {max(comp):.1f} mm")
    if m.get("log_fit"):
        f = m["log_fit"]
        print(f"log-linear : velocity = {f['c0']:.1f} {f['c1']:+.1f} * ln(dt_ms)   R^2={f['r2']:.3f}  rms={f['rms']:.1f}")
        fp = m["power_fit"]
        print(
            f"power law  : velocity = {fp['k']:.0f} * dt_ms^{fp['b']:.3f}          R^2={fp['r2']:.3f}  rms={fp['rms']:.1f}   (dt {fp['dt_range_ms'][0]:.0f}-{fp['dt_range_ms'][1]:.0f} ms)   -> using {m['best_fit']}"
        )
        print("residual vs that fit per profile (>> 0 means the piano saw a shorter dt than the finger's: hammer/key thrown ahead):")
        for k, v in m.get("residual_by_shape", {}).items():
            print(f"   {k:32s} {v['mean']:+6.1f} ± {v['std']:4.1f}  (n={v['n']})")
        if m.get("table"):
            tb = m["table"]
            print("dt->velocity table (ms: vel): " + ", ".join(f"{d:.0f}:{v:.0f}" for d, v in tb[::5]))
    tb = res.get("speed_table")
    if tb:
        print("speed table (target -> measured v_AB, median velocity [min-max], hammer lead):")
        for r in tb:
            print(
                f"   {r['target_speed']:6.3f} -> {r['v_mean_AB']:.3f} m/s  dt {r['dt_ms']:6.1f} ms  vel {r['velocity_median']:5.1f} [{r['velocity_min']}-{r['velocity_max']}]  lead {r['hammer_lead_mm']:+.2f} mm  n={r['n']}"
            )
        vm = res.get("velocity_map")
        if vm:
            print(
                f"velocity covered {vm['covered_velocity'][0]}..{vm['covered_velocity'][1]} with mean speeds {vm['covered_speed_m_s'][0]:.3f}..{vm['covered_speed_m_s'][1]:.3f} m/s (velocity_map in switch_id.json)"
            )
    si = res.get("si", {})
    if si:
        print("\n--- SI summary ---")
        print(
            f"A = {si['A_m']:.5f} m   B = {si['B_m']:.5f} m   A->B = {si['AB_distance_m']:.5f} m   latency = {si['latency_s'] * 1e3:.1f} ms"
            + (f" (fast presses {si['latency_from_fast_presses_s'] * 1e3:.1f} ms)" if si.get("latency_from_fast_presses_s") else "")
        )
        if si.get("velocity_from_dt"):
            f = si["velocity_from_dt"]
            print(f"power law : velocity = {f['k']:.4g} * dt_s^{f['b']:.3f}   (R^2 {f['r2']:.3f}, dt {f['dt_range_s'][0]:.4f}-{f['dt_range_s'][1]:.3f} s)")
        if si.get("velocity_from_dt_loglinear"):
            f = si["velocity_from_dt_loglinear"]
            print(f"log-linear: velocity = {f['c0']:.2f} + {f['c1']:.2f} * ln(dt_s)   (R^2 {f['r2']:.3f})   -> better: {res['mapping'].get('best_fit')}")
        if si.get("velocity_floor_dt_s"):
            fl = si["velocity_floor_dt_s"]
            print(f"velocity 1 (timeout) for dt >= ~{fl['shortest_dt_with_velocity_1_s']:.3f} s  (last dt with velocity > 1: {fl['longest_dt_with_velocity_gt_1_s']:.3f} s)")
        if si.get("covered_velocity"):
            print(f"covered velocity {si['covered_velocity'][0]}..{si['covered_velocity'][1]} with mean speeds {si['covered_speed_m_s'][0]:.4f}..{si['covered_speed_m_s'][1]:.3f} m/s")
        if si.get("velocity_to_speed_full_1_127"):
            rows = si["velocity_to_speed_full_1_127"]
            print("velocity -> mean speed A->B [m/s] (m = measured, e = extrapolated):")
            line = []
            for r in rows:
                if r["velocity"] in (1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 127):
                    sp = r["mean_speed_AB_m_s"]
                    line.append(f"{r['velocity']}:{'-' if sp is None else f'{sp:.3f}'}{'e' if r['source'].startswith('extra') else ('m' if r['source'].startswith('meas') else '')}")
            print("   " + "  ".join(line))
            print(f"   127 needs {si['speed_for_velocity_127_m_s']:.3f} m/s (dt {si['dt_for_velocity_127_s'] * 1e3:.1f} ms) - {si['extrapolation_note']}")
    print("written: results_si.json, fig_summary_si.png, switch_id.json, fig_*.png")


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("session_dir")
    p.add_argument("--pool", nargs="*", default=None, help="other session directories of the same key to add to the velocity fit")
    p.add_argument("--no-plots", action="store_true")
    a = p.parse_args()
    print_report(analyze(a.session_dir, make_plots=not a.no_plots, pool=a.pool))


if __name__ == "__main__":
    main()
