# Piano key-velocity identification with a UR robot

*中文版本见 [README.zh-CN.md](README.zh-CN.md).*

Tools to measure **how a digital piano turns key motion into MIDI velocity**, by pressing keys with a
UR5e under controlled speed profiles while logging the robot state (500 Hz) and the piano's MIDI
output on one clock, and to **validate the identified model in a MuJoCo simulator**.

The output for one piano is a small JSON model (per measured key: two switch depths, the MIDI
latency and a `dt → velocity` curve) that a simulator can use so that simulated key presses produce
the same MIDI velocities as the real instrument. The repository was used to identify a Yamaha P-45
(results in [`examples/yamaha_p45/`](examples/yamaha_p45/)); this README is the procedure to repeat
the measurement on another piano.

---

## Contents

1. [What is measured (the model)](#1-what-is-measured-the-model)
2. [Requirements](#2-requirements)
3. [Installation](#3-installation)
4. [Step 0 - dry run without hardware](#4-step-0---dry-run-without-hardware)
5. [Real experiment, step by step](#5-real-experiment-step-by-step)
6. [Simulation validation, step by step](#6-simulation-validation-step-by-step)
7. [Using the model in your own simulator](#7-using-the-model-in-your-own-simulator)
8. [Files and data formats](#8-files-and-data-formats)
9. [Troubleshooting and pitfalls](#9-troubleshooting-and-pitfalls)
10. [Reference results: Yamaha P-45](#10-reference-results-yamaha-p-45)

---

## 1. What is measured (the model)

Digital-piano actions have **two contacts per key** at different depths. The controller measures the
time `dt` the key needs to travel from the first contact (depth **A**) to the second (depth **B**),
sends *note-on* when B closes (plus a small MIDI latency `L`) and maps `dt` to velocity 1–127.
Velocity therefore depends only on the key's mean speed between A and B, not on the force and not
on what happens after B. *Note-off* is sent when the key rises back above A.

```
depth below the resting key surface
   0 ──── key surface (fingertip just touching)
   A ──── timer starts                                 ┐
   B ──── timer stops, note-on sent L later            ┘ dt = (B - A) / mean speed
 bed ──── key bottom (≈ 10 mm on the P-45)
                       velocity = f(dt)  (≈ k · dt^b, velocity 1 for dt ≥ timeout)
```

The scripts identify, per key: **A, B, L**, the **`dt → velocity` curve** (power law + isotonic
table), the **timeout** (`dt` above which velocity is 1) and the **speed → velocity map** the arm
actually reached. Everything is done on *one key at a time* with the fingertip taught by hand; no
keyboard calibration is needed for the main procedure.

## 2. Requirements

**Hardware**

| item | notes |
|---|---|
| UR robot | Tested on a **UR5e** (PolyScope 5.x, built-in F/T sensor, 500 Hz RTDE). A CB3 UR5 works for the slow presses only: no F/T sensor means no key-bed detection and no home refinement (`--no-refine-home`, `--contact-method midi`). |
| Fingertip | **Stiff** tip with a rounded end (radius ≈ 5–10 mm), e.g. a 3D-printed or metal probe. A soft tip compresses several mm under the 5–20 N seen on fast presses and corrupts the depth measurement (such presses are flagged and excluded, but you lose them). Mount it so it points straight down. |
| Piano | Any digital piano with USB-MIDI. Note the key dip (≈ 10 mm on GHS actions) and the touch-sensitivity setting you use. |
| PC | Linux (ALSA MIDI) on the robot's network. Windows/macOS should work with `mido`/`python-rtmidi` but were not tested. |
| Safety | Keep a hand on the e-stop the first time each profile runs. The scripts abort a press on a force step (default 25 N) and on a depth limit, and lift immediately when the note arrives, but the arm is stiff and the key bed is 2 mm past switch B. |

**Software**: Python 3.10–3.12, [uv](https://docs.astral.sh/uv/) (or pip), the `ur_rtde` Python
wheel (installed automatically), and for the simulation part MuJoCo + dm_control (`sim` extra).

## 3. Installation

```bash
git clone <this repository> && cd piano-key-velocity-exp
uv sync --extra sim            # creates .venv with everything (drop --extra sim on the robot PC if you do not need MuJoCo there)
uv run kve-midi --list         # should list your MIDI ports (empty list = no piano connected yet)
```

Every command below is a console script installed in the environment; run them with `uv run <cmd>`
or activate the environment (`source .venv/bin/activate`) and call them directly.

| command | what it does |
|---|---|
| `kve-probe <ip>` | identify the UR controller (model, PolyScope, F/T sensor, RTDE rate) |
| `kve-jog` | move the TCP in small slow steps, print the pose |
| `kve-midi` | list MIDI ports / print every note event of the piano |
| `kve-test` | hands-on presses on one key (interactive prompt or scripted sweep) |
| `kve-identify` | **the main experiment**: A, B, latency and the velocity curve of one key |
| `kve-analyze-switches` | re-analyse / pool `kve-identify` sessions |
| `kve-export` | merge the per-key results into one model JSON |
| `kve-sim-ball` | MuJoCo: rigid ball presses the simulated keys with that model |
| `kve-compare` | figures and tables: real vs simulated velocities |
| `kve-run` / `kve-analyze` | optional multi-key automatic experiment with a taught keyboard layout |

Connection settings can be given once as environment variables instead of on every command line:

```bash
export UR_ROBOT_IP=192.168.1.10          # --robot-ip
export KVE_MIDI_PORT="Digital Piano"     # --midi-port (substring of the port name; omit to auto-detect)
```

## 4. Step 0 - dry run without hardware

Every robot script accepts `--sim`: a kinematic robot model presses a *virtual two-switch key*
(truth: A = 3.0 mm, B = 8.0 mm, latency 6 ms, velocity = 177.9 − 29.5·ln dt_ms). Run this first to
check the installation and to see what the outputs look like:

```bash
uv run kve-identify --sim --out data/sim_id_C4          # ≈ 1 min, prints the full report
uv run kve-analyze-switches data/sim_id_C4              # re-analyse any time
```

Expected in the report: `B = 8.0x mm`, `latency ≈ 7 ms` (6 ms + half a control period), `A = 2.9x
mm (bracketed 2.9 .. 3.0)`, `log-linear: velocity = 177.x −29.x·ln(dt_ms)  R^2 = 1.000`, drift
`< 0.05 mm`, velocity covered `1..105`. If you see this, the pipeline works; the same commands
without `--sim` run on the robot.

Also useful: `uv run kve-test --sim sweep --speeds 0.03 0.1 0.3 --out data/sim_test` (hands-on
mode) and `uv run kve-run --sim --preset quick --keys quick --out data/sim_run && uv run kve-analyze
data/sim_run` (automatic multi-key experiment).

## 5. Real experiment, step by step

Budget: **≈ 25–30 min per key** for `kve-identify` (about 60–90 presses), plus ≈ 30 min of setup
the first day. Measuring one key per octave (C1…C7) takes an afternoon.

### 5.1 Piano

1. Connect the piano by USB, power it on, and check the MIDI stream:
   ```bash
   uv run kve-midi --list        # note the port name, e.g. "Digital Piano:Digital Piano MIDI 1 20:0"
   uv run kve-midi               # play a few keys by hand: note_on/note_off with velocities must appear
   ```
   If several ports are listed, pass `--midi-port <substring>` (or set `KVE_MIDI_PORT`) to every
   script.
2. Set the touch sensitivity you want to characterise (on the P-45: Medium was the default; Medium
   and Hard gave the same MIDI curve). Record it with `--touch-setting <name>`; it is stored in every
   session's metadata. Only the `dt → velocity` curve depends on it, the switch depths do not.
3. Measure the key dip once (press a white key fully at the point where the fingertip will press,
   ≈ 10 mm on most actions). It is only used by `kve-export --key-dip-m` to express A/B as
   fractions.

### 5.2 Robot

1. Network: put the PC on the robot's subnet and check `ping <robot-ip>`. On the pendant enable
   **Remote Control** (e-Series: *Settings → System → Remote Control*, then the icon top right).
2. Identify the controller:
   ```bash
   uv run kve-probe 192.168.1.10
   ```
   It prints the model, PolyScope version, whether the F/T raw wrench is available (all zero =
   CB3 without sensor) and the RTDE rate. `kve-identify` needs the F/T sensor for the key-bed stop
   and the home refinement.
3. Mount the fingertip pointing down and set **TCP** and **payload** on the pendant to the tip
   (*Installation → General → TCP / Payload*). The recorded `tcp_z` must be the tip, otherwise the
   depths are wrong. Alternatively pass `--tcp x y z rx ry rz` (metres / axis-angle) to the
   scripts; check with `kve-jog p` that the printed position moves as expected.
4. Place the piano so that the keyboard axis is parallel to the robot's base X or Y axis (the
   interactive `l`/`r` commands shift the home by whole keys along `--key-axis`, default `+y`; or
   calibrate the axis on the spot with `c`).
5. Safety settings: normal mode is fine. The scripts limit themselves to |ΔF| ≤ 25 N (`--force-abort-n`,
   80 N with `--hard`), the UR's own tool-force limit (150 N default) is the last resort.

### 5.3 First contact and hands-on presses (`kve-test`)

Jog the fingertip so it is **just touching** the key you want to measure, at the point where a
finger would play it (front third of a white key), then start the interactive session:

```bash
uv run kve-test interactive --robot-ip 192.168.1.10 --touch-setting medium --out data/first_touch
```

Inside the `kv>` prompt (the full command list is printed at start-up):

| command | purpose |
|---|---|
| `t` | listen to the piano for 5 s: play the key by hand, the note must appear |
| `f` | toggle freedrive: move the arm by hand, then `f` again |
| `h` | take the current pose as **home** (= contact reference, depth 0) |
| `w` | where am I: pose relative to home, TCP force, recent MIDI |
| `p 0.05` | one constant-speed press at 0.05 m/s (run-up above the key is automatic), prints velocity, depth and speed at note-on, stop reason, peak force |
| `p 0.1` … `x 0.2` | faster presses (`x` = velocity control, use it above ≈ 0.15 m/s) |
| `k 0.3` | *kick*: touch slowly and accelerate inside the key up to 0.3 m/s (how fast presses are done in the identification) |
| `r` / `l` | move home one white key right/left |
| `d 0.3` / `u 0.3` | move the contact reference 0.3 mm deeper/higher |
| `q` | quit; the arm is left at home so the next script can start from it |

What to look for on the first key (numbers from the P-45, yours will differ somewhat):

* `p 0.03` → velocity ≈ 20–30, `depth@on` ≈ 7–9 mm, `stop: note-on … lifted` – the note arrives
  before the key bed and the arm lifts. If `depth@on` is < 5 mm the home is too deep (the tip was
  already pressing the key): `u 0.5` and repeat. If there is no note at 10 mm, the home is too high
  or the wrong key: `w`, `d 0.5`.
* `k 0.25` → velocity ≈ 60–70, no `!!` warnings. A `!! note-on while the fingertip was only x mm
  deep` warning means the key was launched ahead of the finger (impact too hard): use `k` instead
  of `p`/`x` for speeds above 0.1 m/s. `!! N note-ons … key bounced` means the key re-triggered.
* `|dF|max` stays below ≈ 15 N; it rises with speed because the arm reaches the key bed 2 mm after B.

Every press is saved as `trial_XXXX.npz`, so these sessions can also be analysed later.

### 5.4 Identify one key (`kve-identify`)

With the fingertip just touching the key (as left by `kve-test`, or jogged by hand):

```bash
uv run kve-identify --robot-ip 192.168.1.10 --touch-setting medium --out data/<piano>_id_C4
```

The script runs four phases on that key and prints a report at the end (about 25 min):

| phase | what the robot does | result |
|---|---|---|
| **home** | descends at 2 mm/s until the F/T sensor sees 0.4 N, takes that as depth 0 (`--no-refine-home` keeps the jogged pose) | contact reference |
| **b** | 15 slow constant-speed presses (0.02–0.12 m/s), slow sampled lift | **B** and **latency** from `t_on − t_contact = B/v + L`; note-off depth on the way up |
| **a** | *pause probe*: descend slowly to depth d, pause 0.4 s, finish fast; coarse 1 mm sweep then bisection to 0.1 mm | **A**: for d < A the pause happens before the timer starts (normal velocity), for d > A the pause is inside dt (velocity < 20 or no note) |
| **s** | speed sweep: the key is driven at constant speed through A→B (kick acceleration chosen per speed so the target is reached at A), 2 presses per speed, speeds added until velocity 1…127 is covered or the arm limit is reached; ends with 3 slow presses (drift check) | **`dt → velocity` fit**, speed → velocity table, timeout |

Outputs in `--out`: `results_si.json` (everything in SI units: `A_m`, `B_m`, `latency_s`,
`velocity_from_dt` = `k·dt_s^b`, the log-linear alternative, `dt_to_velocity_table`,
`speed_to_velocity`, `velocity_to_speed_full_1_127` with measured/extrapolated flags),
`switch_id.json` (full analysis), `session.json`, `trial_XXXX.npz` (raw data) and the figures
`fig_summary_si.png`, `fig_B_fit.png`, `fig_A_probe.png`, `fig_dt_velocity.png`,
`fig_velocity_vs_speed.png`, `fig_velocity_model_si.png`.

**Acceptance checks** (compare with the P-45 report in
[`examples/yamaha_p45/piano_id_C4`](examples/yamaha_p45/piano_id_C4)):

* B fit: rms ≲ 2 ms, latency 0–8 ms, and the check line *depth at note-on of the ≤ 0.03 m/s
  presses* within ≈ 0.2 mm of B.
* A: bracket width ≤ 0.1 mm and `consistent`; the note-off depth (printed with the B fit) should
  agree with A within ≈ 0.1 mm – two independent measurements of the same switch.
* Drift check `b2` ≤ 0.25 mm; larger drift means the home moved (loose tip, piano moved, arm warmed
  up): re-run.
* Velocity fit R² ≥ 0.95 over the driven presses; excluded counts (`launched`, `compressed`,
  `braked before B`) small compared to the total. Many `compressed` presses = the tip is too soft.
* Covered velocity range: the UR5e reaches ≈ 0.45 m/s mean key speed, i.e. velocity ≈ 90–100 on the
  P-45; above that the model is the fitted curve (flagged *extrapolated*). Optional: `--hard`
  (0.45–0.75 m/s, 40–80 N key-bed impacts) pushes towards 127 – rigid tip, hand on the e-stop.

Useful options: `--phase b|a|s` to run one phase, `--ab-from data/<piano>_id_C4/switch_id.json`
to reuse A/B for another sweep, `--sweep-repeats 3` for more repeats, `--b-guess-mm` /
`--a-guess-mm` if your piano's switches are far from 8 / 3 mm (used only before they are measured),
`--probe-low-vel` if the pause inside dt gives velocities above 20 on your instrument.

Re-analyse or pool sessions of the same key (each pooled session is re-referenced by its own slow
presses):

```bash
uv run kve-analyze-switches data/<piano>_id_C4 --pool data/<piano>_velmap_C4 data/<piano>_hard_C4
```

### 5.5 Repeat across the keyboard

Repeat 5.3/5.4 for one key per octave, e.g. C1…C7 (`r 7` in `kve-test` moves 7 white keys =
one octave, or jog by hand), and optionally a black key and a bass/treble key of the same octave to
check that A and B are the same across the keyboard. Name the directories `<anything>_<note>`
(`piano_id_C4`): the note is read from the name by `kve-export` and `kve-sim-ball`. If you want the
curve for another touch setting, repeat only phase `s` with `--ab-from` (A/B do not change).

### 5.6 Export the model

```bash
uv run kve-export data/<piano>_id_C1 data/<piano>_id_C2 ... data/<piano>_id_C7 \
    --piano "Kawai ES120" --touch-setting normal --key-dip-m 0.010 --out kawai_es120_velocity_model.json
```

The JSON holds, per measured key, `A_m`, `B_m`, `k`, `b`, `floor_dt_s`, `latency_s` and the
measured `speed_table`; unmeasured keys are interpolated by the simulator. Compare with
[`examples/yamaha_p45/p45_velocity_model.json`](examples/yamaha_p45/p45_velocity_model.json).

### 5.7 Optional: automatic multi-key experiment (`kve-run`)

`kve-run` drives many keys and speed *profiles* (constant, accelerating, decelerating, two-stage)
from a keyboard layout taught on three keys, finds the contact height of every key with the F/T
sensor and records everything in the same format; `kve-analyze` then fits the switch depths from
all profiles at once. It was used for the first exploratory sessions; `kve-identify` is the
recommended procedure.

```bash
uv run kve-run --robot-ip <ip> --teach-layout key_layout.json                      # freedrive onto C2, C6, C#4
uv run kve-run --robot-ip <ip> --layout key_layout.json --keys C4 --preset quick --repeats 1 --out data/kv_check
uv run kve-run --robot-ip <ip> --layout key_layout.json --keys registers --preset full --repeats 3 --out data/kv_full
uv run kve-analyze data/kv_full
```

## 6. Simulation validation, step by step

Two levels of validation are built in.

### 6.1 Does the analysis recover a known mechanism? (`--sim`)

Step 0 above: the virtual piano has known A/B/latency/curve and the whole pipeline recovers them
(A 2.97 vs 3.00 mm, B 8.01 vs 8.00 mm, latency 7.0 vs 6 ms + ½ control period, curve within 0.5
velocity units). Run it whenever you change the analysis.

### 6.2 Does the exported model reproduce the real piano in MuJoCo? (`kve-sim-ball`, `kve-compare`)

A rigid ball (mocap sphere, r = 8 mm) presses the front of each measured white key in a MuJoCo model
of an 88-key keyboard (RoboPianist geometry: hinged box keys, 10 mm dip, spring) at the same target
speeds as the real sweeps. The simulated key's own angle gives the mean key speed A→B exactly like
on the robot, and the two-switch tracker with your model gives the simulated velocity. Errors are
evaluated at the same *measured* key speed (the arm did not always reach its target).

```bash
# needs: uv sync --extra sim
uv run kve-sim-ball --real-dirs data/<piano>_id_C1 ... data/<piano>_id_C7 \
    --velocity-model kawai_es120_velocity_model.json --model measured --out data/sim_vs_real
uv run kve-sim-ball --real-dirs data/<piano>_id_C1 ... data/<piano>_id_C7 --model legacy --out data/sim_vs_real   # baseline
uv run kve-compare --sim-dir data/sim_vs_real --real-dirs data/<piano>_id_C1 ... data/<piano>_id_C7
```

`--model legacy` is the original RoboPianist rule (note-on 0.5° from the bottom, velocity =
qvel·127/3.5) as a baseline. Outputs: `sim_vs_real_<model>.json`, `fig_sim_vs_real_<model>.png`
(per-key curves + error histogram), `fig_compare_overview.png`, `fig_compare_traces_C4.png` (real
fingertip vs simulated key depth over time with the note-on instants and the A/B levels) and
`compare_table.md` (MAE / bias per key and model). On a machine without EGL the scripts run
headless (`MUJOCO_GL` is handled automatically); add `--render-note C4` with `MUJOCO_GL=egl` to get
a GIF of the presses.

**What to expect**: with the P-45 model the simulator reproduces the real velocities with an MAE of
2.1 velocity units over 115 presses (legacy rule: 9.0). Deviations concentrate at the velocity-1
timeout edge (the real piano jumps from 1 to ≈ 10 within 0.001 m/s) and at keys whose A/B differ
from their neighbours. To validate the model on your own simulator instead, see section 7.

## 7. Using the model in your own simulator

`key_velocity_exp.sim.velocity_model` is self-contained (numpy; optional torch helpers):

```python
import numpy as np
from key_velocity_exp.sim.velocity_model import TwoSwitchVelocityModel, TwoSwitchTracker

model = TwoSwitchVelocityModel.load("kawai_es120_velocity_model.json")   # 88 keys, interpolated
# hinge-angle thresholds for your key joints: resting angle + measured depth / lever length
angle_a, angle_b = model.angle_thresholds(joint_range_max, rest_angle)
tracker = TwoSwitchTracker(model, joint_range_max, rest_angle)          # call once per physics substep:
on_keys, velocities, off_keys = tracker.update(t, qpos, qvel)          # crossing times interpolated inside the step
# or, if you track A/B crossings yourself:
vel = model.velocity(dt_seconds, key_ids)                              # lookup table (measured range) + power law outside
```

The depths are absolute (metres below the *resting* key surface at the press point), so measure the
resting sag of your simulated keys (`key_rest_angles()` does it for the bundled MJCF) and keep the
A→B distance equal to the real one; on the bundled keys B ends up only 0.2 mm above the bottom
because the white keys sag 1.7 mm under gravity. For a vectorised simulator use `model.as_torch()`
and `TwoSwitchVelocityModel.velocity_torch()`.

## 8. Files and data formats

```
key_velocity_exp/
  common.py             key geometry, PressProfile (speed profile -> depth trajectory), TrialRecord I/O, CLI helpers
  midi_io.py            timestamped MIDI listener (mido/rtmidi) and the simulated two-switch piano
  ur5_io.py             ur_rtde driver (servoL / speedL presses, key-bed stop, force/depth abort) and a kinematic simulator
  real_test.py          hands-on presses (kve-test); Session class reused by kve-identify
  identify_switches.py  the identification experiment (kve-identify)
  analyze_switches.py   its analysis: B/latency fit, A probe, dt->velocity fits, SI outputs, figures
  export_piano_model.py per-key results -> model JSON (kve-export)
  run_experiment.py / analyze.py / layout.py   automatic multi-key experiment
  probe_robot.py / jog_robot.py                controller identification, slow TCP moves
  sim/velocity_model.py two-switch velocity model + tracker for simulators (numpy / torch)
  sim/key_midi.py       MIDI generation for the simulated keyboard (measured model or legacy rule)
  sim/build_piano_mjcf.py, sim/piano_constants.py   the 88-key MuJoCo keyboard
  sim/sim_ball_press.py, sim/compare_sim_real.py    rigid-ball validation and comparison
examples/yamaha_p45/    P-45 results: piano_id_C1..C7 (results_si.json, switch_id.json, figures), p45_velocity_model.json, sim_vs_real/
docs/p45_case_study.md  working log of the P-45 campaign: robot limits, key launch, hammer flight, final numbers
```

**Per press** (`trial_XXXX.npz`): `robot_samples` [N × 33] with the columns of
`common.ROBOT_SAMPLE_FIELDS` (monotonic time, RTDE time, commanded z, TCP pose, TCP speed, TCP
force, q, qd), `midi_events` [M × 4] (time, on/off, note, velocity) and a JSON `meta` (profile,
home, phase, target speed, stop reason …). `TrialRecord.load()` reads it; `rec.depth()` is the
fingertip depth below home.

**Per session**: `session.json`, `summary.csv`, `switch_id.json`, `results_si.json`, figures.

## 9. Troubleshooting and pitfalls

Things that cost time on the P-45 (details and numbers in
[docs/p45_case_study.md](docs/p45_case_study.md)):

* **The robot, not the piano, limits the velocity range.** `servoL` cannot follow more than ≈ 0.15
  m/s inside 10 mm of travel; velocity control (`speedL`) follows the command but the arm still needs
  a run-up. Trust only the *measured* speed (printed as `v(3-7mm)` / `v_AB`); the analysis never uses
  the commanded speed.
* **Key launch.** A rigid tip hitting a light key at ≥ 0.25 m/s launches the key ahead of the finger:
  note-on arrives with the tip only 3–4 mm deep, followed by bounces and re-triggers. Fast presses
  therefore touch slowly and accelerate *inside* the key (`kick`); the descent stops at note-on.
* **Hammer flight.** Accelerations above ≈ 10 m/s² inside the key throw the hammer ahead of the key:
  the piano sees a shorter dt than the finger. Such presses are flagged `launched` and excluded; the
  sweep chooses the acceleration so the target speed is reached *at* A.
* **Premature braking.** A GHS key moving at 0.4 m/s pushes back 5–6 N by itself; the force stop must
  be armed only below B (`set_limits_from_b` does this once B is known) or the arm brakes before the
  second switch.
* **Soft tips** compress under load: TCP ≠ key. Presses where the tip is deeper than B at note-on by
  more than latency·speed are flagged `compressed`.
* **Home drift**: the reference is re-measured with slow presses at the end (`b2`); > 0.25 mm means
  the tip or the piano moved.
* **No note / wrong key**: `t` in `kve-test`; check `--midi-port`, the key under the tip (`--note`
  only labels the session), and that the tip is not between two keys.
* **`moveL failed` / protective stop**: clear it on the pendant; the scripts re-upload the control
  script automatically (`ensure_running`).
* **MuJoCo on a headless machine**: `kve-sim-ball` disables rendering unless `--render-note` is
  given; for GIFs install EGL and set `MUJOCO_GL=egl`.

## 10. Reference results: Yamaha P-45

Medium touch (Hard gave the same curve), stiff tip, UR5e, one key per octave. Full reports and
figures in [`examples/yamaha_p45/`](examples/yamaha_p45/).

| key | A [mm] | B [mm] | velocity = k·dt_s^b | timeout dt | latency |
|---|---|---|---|---|---|
| C1 | 5.02 | 8.07 | 6.57·dt^−0.50 | 0.19 s | 3.9 ms |
| C2 | 4.91 | 8.43 | 6.09·dt^−0.53 | 0.24 s | 3.4 ms |
| C3 | 4.72 | 8.16 | 5.44·dt^−0.57 | 0.25 s | 4.2 ms |
| C4 | 4.84 | 7.95 | 3.82·dt^−0.65 | 0.31 s | 3.4 ms |
| C5 | 4.91 | 8.20 | 5.74·dt^−0.55 | 0.27 s | 5.1 ms |
| C6 | 4.91 | 8.23 | 5.27·dt^−0.57 | 0.28 s | 3.7 ms |
| C7 | 3.71 | 6.90 | 4.26·dt^−0.64 | 0.23 s | 4.0 ms |

Simulated rigid-ball presses vs the real piano at the same key speed (115 presses):

| | C1 | C2 | C3 | C4 | C5 | C6 | C7 | all |
|---|---|---|---|---|---|---|---|---|
| measured two-switch model, MAE | 2.8 | 1.4 | 2.1 | 1.4 | 1.4 | 2.8 | 3.8 | **2.1** (bias −1.1) |
| legacy threshold rule, MAE | 7.9 | 8.6 | 7.3 | 8.2 | 10.4 | 8.8 | 12.5 | **9.0** (bias −5.5) |

Velocity 127 needs ≈ 0.7 m/s mean key speed through A→B on the P-45 (extrapolated; the arm reached
0.45 m/s = velocity 98).
