#!/usr/bin/env python
"""Export the identified velocity mechanism of one piano (one session directory per measured key) into a
single model JSON that the simulator side (``key_velocity_exp.sim.velocity_model``) reads.

  kve-export data/piano_id_C1 data/piano_id_C2 ... data/piano_id_C7 --piano "Yamaha P-45" --out my_piano_velocity_model.json

Each session directory must contain results_si.json (written by kve-identify / kve-analyze-switches);
the key is read from the directory name (…_C4 -> C4) unless --notes is given. Per measured key the
model stores:
  A_m, B_m          switch depths in metres below the resting key surface (fingertip press point)
  A_frac, B_frac    the same as fractions of the key dip (--key-dip-m, default 0.010 m)
  k, b              velocity = k * dt_s^b   (dt in seconds, key driven through A->B)
  floor_dt_s        dt at/above which the piano reports velocity 1
  latency_s         MIDI latency
  speed_table       the measured mean-speed -> velocity table (used as a lookup table inside the covered range)
Keys that were not measured are interpolated linearly in key index by the simulator.
"""

from __future__ import annotations

import argparse
import json
import os
import re

from key_velocity_exp.common import name_to_midi

DEFAULT_OUT = "piano_velocity_model.json"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dirs", nargs="+", help="kve-identify session directories, one per measured key")
    p.add_argument("--notes", nargs="*", help="note names for the dirs (default: parsed from the directory name, e.g. piano_id_C4 -> C4)")
    p.add_argument("--key-dip-m", type=float, default=0.010, help="full key travel at the press point on the real piano")
    p.add_argument("--piano", default="digital piano", help="name of the instrument, stored in the description")
    p.add_argument("--touch-setting", default="", help="touch-sensitivity setting used, stored in the description")
    p.add_argument("--out", default=DEFAULT_OUT)
    a = p.parse_args()
    keys = []
    for i, d in enumerate(a.dirs):
        with open(os.path.join(d, "results_si.json")) as f:
            r = json.load(f)
        if a.notes:
            name = a.notes[i]
        else:
            m = re.search(r"([A-G]#?\d)", os.path.basename(os.path.normpath(d)))
            if m is None:
                raise SystemExit(f"cannot read the note name from '{d}'; pass --notes")
            name = m.group(1)
        fit = r.get("velocity_from_dt")
        if not fit:
            print(f"skip {d}: no velocity fit in results_si.json")
            continue
        fl = r.get("velocity_floor_dt_s", {})
        keys.append(
            {
                "note": name,
                "midi": name_to_midi(name),
                "key": name_to_midi(name) - 21,
                "A_m": r["A_m"],
                "B_m": r["B_m"],
                "A_frac": r["A_m"] / a.key_dip_m,
                "B_frac": r["B_m"] / a.key_dip_m,
                "k": fit["k"],
                "b": fit["b"],
                "fit_r2": fit["r2"],
                "fit_dt_range_s": fit["dt_range_s"],
                "floor_dt_s": fl.get("shortest_dt_with_velocity_1_s"),
                "latency_s": r.get("latency_from_fast_presses_s") or r.get("latency_s") or 0.0,
                "covered_velocity": r.get("covered_velocity"),
                "covered_speed_m_s": r.get("covered_speed_m_s"),
                "speed_table": r.get("speed_table"),
                "source": os.path.normpath(d),
            }
        )
    if not keys:
        raise SystemExit("nothing to export")
    keys.sort(key=lambda k: k["midi"])
    model = {
        "description": (
            f"{a.piano}: velocity mechanism measured with a UR robot (rigid tip, key driven through A->B). "
            "Timer starts when the key passes depth A, stops at depth B (note-on); velocity = k * dt_s^b, "
            "velocity 1 when dt >= floor_dt_s; note-off when the key rises back above A. Depth fractions are relative to the key dip at the press point."
        ),
        "piano": a.piano,
        "key_dip_m": a.key_dip_m,
        "touch_setting": a.touch_setting,
        "keys": keys,
    }
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(model, f, indent=2)
    print(f"wrote {a.out}")
    for k in keys:
        floor = "-" if k["floor_dt_s"] is None else f"{k['floor_dt_s']:.3f} s"
        print(
            f"  {k['note']:3s} midi {k['midi']:3d}  A {k['A_m'] * 1e3:.2f} mm  B {k['B_m'] * 1e3:.2f} mm  velocity = {k['k']:.3f} * dt^{k['b']:.3f} (R2 {k['fit_r2']:.3f})  floor {floor}  latency {k['latency_s'] * 1e3:.1f} ms"
        )


if __name__ == "__main__":
    main()
