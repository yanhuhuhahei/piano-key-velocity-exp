#!/usr/bin/env python
"""Analysis of a key-velocity session recorded by kve-run (run_experiment.py) or kve-test.

What it estimates
-----------------
1. Per trial: contact time, note-on time, depth at note-on, mean/peak descent speed, speed at note-on.
2. Switch model (from constant-speed trials): ``t_on - t_contact = d2 / v + L`` gives the depth ``d2``
   of the note-on switch and the MIDI latency ``L`` (linear fit in 1/v).
3. First-switch depth ``d1``: the depth for which "mean speed between d1 and d2" predicts MIDI velocity
   with the smallest monotone-fit residual *across all profiles* (constant, accelerating, decelerating,
   two-stage). If velocity really is measured from the time between two switches, a single ``d1``
   collapses every profile onto one curve.
4. The mapping ``MIDI velocity = f(v_sensor)`` (log-linear fit + isotonic table), pooled and per key,
   plus black/white and register comparisons.

Outputs (in the session directory): summary.csv, analysis.json, fig_*.png
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from key_velocity_exp.common import GRID, PALETTE, TEXT, TEXT2, TrialRecord, load_session

# ----------------------------------------------------------------------------- per-trial kinematics


class Kinematics:
    """Time <-> depth on the descent (monotone part) of one trial, using the *measured* TCP z."""

    def __init__(self, rec: TrialRecord):
        t = rec.col("t")
        d = rec.depth()
        i_bottom = int(np.argmax(d))
        self.t, self.d = t[: i_bottom + 1], d[: i_bottom + 1]
        # measured vertical speed (RTDE actual TCP speed) or derivative of z
        vz = -rec.col("vz")[: i_bottom + 1]
        if not np.any(np.abs(vz) > 1e-6):
            vz = np.gradient(self.d, self.t)
        self.v = vz
        self.d_max = float(self.d[-1]) if len(self.d) else 0.0
        self.dt = float(np.median(np.diff(t))) if len(t) > 1 else np.nan

    def time_at_depth(self, depth: float) -> float:
        if depth > self.d_max or depth < self.d[0]:
            return np.nan
        i = int(np.searchsorted(self.d, depth))
        if i == 0:
            return float(self.t[0])
        d0, d1 = self.d[i - 1], self.d[i]
        w = 0.0 if d1 == d0 else (depth - d0) / (d1 - d0)
        return float(self.t[i - 1] + w * (self.t[i] - self.t[i - 1]))

    def depth_at_time(self, t: float) -> float:
        return float(np.interp(t, self.t, self.d))

    def speed_at_time(self, t: float) -> float:
        return float(np.interp(t, self.t, self.v))

    def mean_speed(self, d_a: float, d_b: float) -> float:
        ta, tb = self.time_at_depth(d_a), self.time_at_depth(d_b)
        if np.isnan(ta) or np.isnan(tb) or tb <= ta:
            return np.nan
        return (d_b - d_a) / (tb - ta)


def trial_row(rec: TrialRecord) -> dict:
    k = Kinematics(rec)
    ons = rec.note_on_events()
    row = dict(
        trial_id=rec.trial_id,
        key=rec.key_name,
        midi_note=rec.midi_note,
        black=rec.key_is_black,
        kind=rec.profile.kind,
        v0=rec.profile.v0,
        v1=rec.profile.v1,
        depth_mid_mm=rec.profile.depth_mid_m * 1e3,
        start_above_mm=rec.profile.start_above_m * 1e3,
        profile=rec.profile.label(),
        aborted=rec.aborted,
        n_note_on=len(ons),
        velocity=ons[0].velocity if ons else np.nan,
        t_contact=k.time_at_depth(0.0),
        t_on=ons[0].t if ons else np.nan,
        depth_max_mm=k.d_max * 1e3,
        v_peak=float(np.nanmax(k.v)) if len(k.v) else np.nan,
        v_mean_1_7mm=k.mean_speed(0.001, 0.007),
        wrong_keys=len([e for e in rec.midi_events if e.kind == "note_on" and e.note != rec.midi_note]),
    )
    if ons:
        row["depth_on_mm"] = k.depth_at_time(ons[0].t) * 1e3
        row["v_at_on"] = k.speed_at_time(ons[0].t)
        row["v_mean_to_on"] = k.mean_speed(0.0, k.depth_at_time(ons[0].t)) if k.depth_at_time(ons[0].t) > 0 else np.nan
        row["dt_contact_to_on"] = ons[0].t - row["t_contact"]
    return row


def summarize_session(directory: str, quiet: bool = False) -> pd.DataFrame:
    rows = [trial_row(r) for r in load_session(directory)]
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(directory, "summary.csv"), index=False)
    if not quiet and len(df):
        print(df[["trial_id", "key", "profile", "start_above_mm", "velocity", "depth_on_mm", "v_mean_1_7mm", "aborted"]].to_string(index=False))
    return df


# ----------------------------------------------------------------------------- model fits


def fit_switch_depth(df: pd.DataFrame) -> dict:
    """t_on - t_contact = d2/v + L using constant-speed trials that were at speed before contact."""
    sel = df[(df.kind == "const") & df.velocity.notna() & (df.start_above_mm > 0) & ~df.aborted].copy()
    if len(sel) < 3:
        sel = df[(df.kind == "const") & df.velocity.notna() & ~df.aborted].copy()
    sel = sel[sel.v_mean_1_7mm.notna() & sel.dt_contact_to_on.notna()]
    if len(sel) < 3:
        return {"ok": False, "reason": "not enough constant-speed trials"}
    x, y = 1.0 / sel.v_mean_1_7mm.values, sel.dt_contact_to_on.values
    A = np.vstack([x, np.ones_like(x)]).T
    (d2, L), res, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ np.array([d2, L])
    per_key = {}
    for key, g in sel.groupby("key"):
        if len(g) >= 3:
            xk, yk = 1.0 / g.v_mean_1_7mm.values, g.dt_contact_to_on.values
            Ak = np.vstack([xk, np.ones_like(xk)]).T
            (d2k, Lk), *_ = np.linalg.lstsq(Ak, yk, rcond=None)
            per_key[key] = {"d2_mm": d2k * 1e3, "latency_ms": Lk * 1e3, "n": int(len(g))}
    return {"ok": True, "d2_mm": float(d2 * 1e3), "latency_ms": float(L * 1e3), "rms_ms": float(np.sqrt(np.mean((pred - y) ** 2)) * 1e3), "n": int(len(sel)), "per_key": per_key}


def isotonic_residual(v_sens: np.ndarray, vel: np.ndarray) -> tuple[float, object]:
    from sklearn.isotonic import IsotonicRegression

    x = np.log(v_sens)
    iso = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(x, vel)
    r = vel - iso.predict(x)
    return float(np.sqrt(np.mean(r**2))), iso


def scan_first_switch(recs: list[TrialRecord], df: pd.DataFrame, d2_m: float, latency_s: float) -> dict:
    """Find d1 minimizing the monotone-fit residual of velocity vs mean speed between d1 and d2."""
    kin = {r.trial_id: Kinematics(r) for r in recs}
    good = df[df.velocity.notna() & ~df.aborted]
    grid = np.arange(0.0002, d2_m - 0.0008, 0.0002)
    results = []
    for d1 in grid:
        vs, vel = [], []
        for _, row in good.iterrows():
            k = kin[row.trial_id]
            v = k.mean_speed(d1, d2_m)
            if np.isfinite(v) and v > 0:
                vs.append(v)
                vel.append(row.velocity)
        if len(vs) < 6:
            continue
        rms, _ = isotonic_residual(np.array(vs), np.array(vel))
        results.append((d1, rms, len(vs)))
    if not results:
        return {"ok": False}
    d1_best, rms_best, n = min(results, key=lambda r: r[1])
    return {"ok": True, "d1_mm": float(d1_best * 1e3), "rms_velocity": rms_best, "n": int(n), "scan": [(float(a * 1e3), float(b)) for a, b, _ in results]}


def fit_mapping(v_sens: np.ndarray, vel: np.ndarray) -> dict:
    """MIDI velocity ~ a + b ln(v_sens); returns coefficients and R^2. Also a power-law-of-dt view."""
    x = np.log(v_sens)
    A = np.vstack([x, np.ones_like(x)]).T
    (b, a), *_ = np.linalg.lstsq(A, vel, rcond=None)
    pred = a + b * x
    ss_res, ss_tot = np.sum((vel - pred) ** 2), np.sum((vel - vel.mean()) ** 2)
    return {"a": float(a), "b": float(b), "r2": float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan, "rms": float(np.sqrt(ss_res / len(vel))), "n": int(len(vel))}


# ----------------------------------------------------------------------------- figures


def _style(ax, title, xlabel, ylabel):
    ax.set_title(title, color=TEXT, loc="left", fontsize=11)
    ax.set_xlabel(xlabel, color=TEXT2)
    ax.set_ylabel(ylabel, color=TEXT2)
    ax.grid(True, color=GRID, linewidth=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=TEXT2)


def make_figures(directory: str, recs: list[TrialRecord], df: pd.DataFrame, res: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    good = df[df.velocity.notna() & ~df.aborted]
    keys = list(dict.fromkeys(good.key))
    key_color = {k: PALETTE[i % len(PALETTE)] for i, k in enumerate(keys)}

    # 1. velocity vs mean descent speed, constant-speed trials, one series per key
    fig, ax = plt.subplots(figsize=(7, 4.5), facecolor="white")
    c = good[good.kind == "const"]
    for k in keys:
        g = c[c.key == k]
        if len(g):
            ax.scatter(g.v_mean_1_7mm, g.velocity, s=28, color=key_color[k], label=k, edgecolors="white", linewidths=1)
    ax.set_xscale("log")
    _style(ax, "MIDI velocity vs descent speed (constant-speed presses)", "mean speed 1–7 mm below contact [m/s]", "MIDI velocity")
    if len(keys) > 1:
        ax.legend(frameon=False, fontsize=8, title="key", title_fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(directory, "fig_velocity_vs_speed_const.png"), dpi=150)
    plt.close(fig)

    # 2. switch depth fit
    sw = res.get("switch_fit", {})
    if sw.get("ok"):
        fig, ax = plt.subplots(figsize=(7, 4.5), facecolor="white")
        sel = c[c.dt_contact_to_on.notna()]
        ax.scatter(1 / sel.v_mean_1_7mm, sel.dt_contact_to_on * 1e3, s=28, color=PALETTE[0], edgecolors="white", linewidths=1, label="trials")
        xx = np.linspace(0, (1 / sel.v_mean_1_7mm).max() * 1.05, 50)
        ax.plot(xx, (sw["d2_mm"] * 1e-3 * xx + sw["latency_ms"] * 1e-3) * 1e3, color=PALETTE[1], linewidth=2, label=f"fit: d2 = {sw['d2_mm']:.2f} mm, latency = {sw['latency_ms']:.1f} ms")
        _style(ax, "Contact → note-on delay vs 1/speed", "1 / descent speed [s/m]", "t(note-on) − t(contact) [ms]")
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(directory, "fig_switch_depth_fit.png"), dpi=150)
        plt.close(fig)

    # 3. d1 scan + 4. all profiles collapsed onto v_sensor
    sc = res.get("first_switch_scan", {})
    if sc.get("ok"):
        fig, ax = plt.subplots(figsize=(7, 4), facecolor="white")
        s = np.array(sc["scan"])
        ax.plot(s[:, 0], s[:, 1], color=PALETTE[0], linewidth=2)
        ax.axvline(sc["d1_mm"], color=PALETTE[1], linewidth=1.5, linestyle="--")
        ax.annotate(f"best d1 = {sc['d1_mm']:.1f} mm", (sc["d1_mm"], s[:, 1].max()), color=TEXT2, fontsize=9, xytext=(6, -4), textcoords="offset points")
        _style(ax, "Residual of monotone fit vs assumed first-switch depth", "assumed d1 [mm]", "RMS residual [MIDI velocity]")
        fig.tight_layout()
        fig.savefig(os.path.join(directory, "fig_first_switch_scan.png"), dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 4.5), facecolor="white")
        kinds = list(dict.fromkeys(good.kind))
        for i, kd in enumerate(kinds):
            g = good[good.kind == kd]
            ax.scatter(g.v_sensor, g.velocity, s=28, color=PALETTE[i % len(PALETTE)], label=kd, edgecolors="white", linewidths=1)
        m = res.get("mapping_pooled", {})
        if m:
            xx = np.geomspace(good.v_sensor.min(), good.v_sensor.max(), 50)
            ax.plot(xx, m["a"] + m["b"] * np.log(xx), color=TEXT2, linewidth=1.5, label=f"{m['a']:.1f} + {m['b']:.1f}·ln v  (R²={m['r2']:.3f})")
        ax.set_xscale("log")
        _style(ax, f"All profiles on one curve: velocity vs mean speed between d1={sc['d1_mm']:.1f} and d2={sw.get('d2_mm', 0):.1f} mm", "mean speed between switches [m/s]", "MIDI velocity")
        ax.legend(frameon=False, fontsize=8, title="profile", title_fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(directory, "fig_velocity_vs_sensor_speed_all.png"), dpi=150)
        plt.close(fig)

    # 5. example time series: the fastest and slowest constant press that produced a note
    ex = c.sort_values("v_mean_1_7mm")
    if len(ex):
        picks = [ex.iloc[0].trial_id, ex.iloc[-1].trial_id]
        fig, axes = plt.subplots(3, len(picks), figsize=(5 * len(picks), 7.5), facecolor="white", sharex="col")
        axes = np.atleast_2d(axes.T).T if len(picks) > 1 else axes.reshape(3, 1)
        by_id = {r.trial_id: r for r in recs}
        for j, tid in enumerate(picks):
            r = by_id[tid]
            t0 = r.t_start
            t = r.col("t") - t0
            axes[0, j].plot(t * 1e3, (r.z_contact - r.col("target_z")) * 1e3, color=GRID, linewidth=2, label="target")
            axes[0, j].plot(t * 1e3, r.depth() * 1e3, color=PALETTE[0], linewidth=2, label="measured")
            axes[1, j].plot(t * 1e3, -r.col("vz"), color=PALETTE[0], linewidth=2)
            axes[2, j].plot(t * 1e3, r.col("fz"), color=PALETTE[0], linewidth=2)
            for e in r.note_on_events():
                for a in axes[:, j]:
                    a.axvline((e.t - t0) * 1e3, color=PALETTE[1], linestyle="--", linewidth=1.2)
                axes[0, j].annotate(f"note-on vel {e.velocity}", ((e.t - t0) * 1e3, 0.5), color=PALETTE[1], fontsize=9, xytext=(4, 0), textcoords="offset points")
            _style(axes[0, j], f"{r.key_name}  {r.profile.label()}  start {r.profile.start_above_m * 1e3:.0f} mm above", "", "depth [mm]")
            _style(axes[1, j], "", "", "descent speed [m/s]")
            _style(axes[2, j], "", "time since press start [ms]", "TCP force z [N]")
            axes[0, j].legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(directory, "fig_example_trials.png"), dpi=150)
        plt.close(fig)


# ----------------------------------------------------------------------------- main


def analyze(directory: str, make_plots: bool = True) -> dict:
    recs = list(load_session(directory))
    if not recs:
        raise SystemExit(f"no trials in {directory}")
    df = summarize_session(directory, quiet=True)
    res: dict = {"n_trials": len(df), "n_aborted": int(df.aborted.sum()), "n_no_note": int(df.velocity.isna().sum() - df.aborted.sum()), "n_wrong_key": int((df.wrong_keys > 0).sum())}

    sw = fit_switch_depth(df)
    res["switch_fit"] = sw
    if sw.get("ok"):
        d2, L = sw["d2_mm"] * 1e-3, sw["latency_ms"] * 1e-3
        sc = scan_first_switch(recs, df, d2, L)
        res["first_switch_scan"] = sc
        if sc.get("ok"):
            d1 = sc["d1_mm"] * 1e-3
            kin = {r.trial_id: Kinematics(r) for r in recs}
            df["v_sensor"] = [kin[t].mean_speed(d1, d2) for t in df.trial_id]
            good = df[df.velocity.notna() & ~df.aborted & df.v_sensor.notna() & (df.v_sensor > 0)]
            res["mapping_pooled"] = fit_mapping(good.v_sensor.values, good.velocity.values)
            per_key = {}
            for key, g in good.groupby("key"):
                if len(g) >= 4:
                    per_key[key] = fit_mapping(g.v_sensor.values, g.velocity.values)
            res["mapping_per_key"] = per_key
            # black vs white / register residuals relative to the pooled fit
            m = res["mapping_pooled"]
            good = good.assign(resid=good.velocity - (m["a"] + m["b"] * np.log(good.v_sensor)))
            res["residual_by_key"] = {k: {"mean": float(g.resid.mean()), "std": float(g.resid.std()), "n": int(len(g))} for k, g in good.groupby("key")}
            res["residual_black_vs_white"] = {("black" if b else "white"): float(g.resid.mean()) for b, g in good.groupby("black")}
            res["residual_by_profile_kind"] = {k: {"mean": float(g.resid.mean()), "std": float(g.resid.std()), "n": int(len(g))} for k, g in good.groupby("kind")}
            # isotonic lookup table (monotone, non-parametric)
            rms, iso = isotonic_residual(good.v_sensor.values, good.velocity.values)
            vv = np.geomspace(good.v_sensor.min(), good.v_sensor.max(), 25)
            res["isotonic_table"] = {"v_sensor_m_s": vv.tolist(), "velocity": iso.predict(np.log(vv)).tolist(), "rms": rms}
            df.to_csv(os.path.join(directory, "summary.csv"), index=False)

    with open(os.path.join(directory, "analysis.json"), "w") as f:
        json.dump(res, f, indent=2, default=float)
    if make_plots:
        make_figures(directory, recs, df, res)
    return res


def print_report(res: dict) -> None:
    print(f"trials: {res['n_trials']}  aborted: {res['n_aborted']}  no note-on: {res['n_no_note']}  hit wrong key: {res['n_wrong_key']}")
    sw = res.get("switch_fit", {})
    if sw.get("ok"):
        print(f"note-on switch depth d2 = {sw['d2_mm']:.2f} mm, MIDI latency = {sw['latency_ms']:.1f} ms (fit rms {sw['rms_ms']:.1f} ms, n={sw['n']})")
        for k, v in sw.get("per_key", {}).items():
            print(f"   {k:4s} d2 = {v['d2_mm']:.2f} mm  latency = {v['latency_ms']:.1f} ms (n={v['n']})")
    else:
        print("switch fit failed:", sw.get("reason"))
    sc = res.get("first_switch_scan", {})
    if sc.get("ok"):
        print(f"first switch depth d1 = {sc['d1_mm']:.1f} mm (monotone-fit rms {sc['rms_velocity']:.2f} velocity units)")
    m = res.get("mapping_pooled")
    if m:
        print(f"pooled mapping: velocity = {m['a']:.1f} + {m['b']:.1f} * ln(v_sensor [m/s])   R^2 = {m['r2']:.3f}, rms = {m['rms']:.2f}")
        print("residual by profile kind (should be ~0 if only the inter-switch speed matters):")
        for k, v in res["residual_by_profile_kind"].items():
            print(f"   {k:10s} mean {v['mean']:+.2f}  std {v['std']:.2f}  n={v['n']}")
        print("residual by key:")
        for k, v in res["residual_by_key"].items():
            print(f"   {k:4s} mean {v['mean']:+.2f}  std {v['std']:.2f}  n={v['n']}")
        print("black vs white:", {k: round(v, 2) for k, v in res["residual_black_vs_white"].items()})


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("session_dir")
    p.add_argument("--no-plots", action="store_true")
    a = p.parse_args()
    print_report(analyze(a.session_dir, make_plots=not a.no_plots))


if __name__ == "__main__":
    main()
