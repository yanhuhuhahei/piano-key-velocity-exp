# Case study: identifying the Yamaha P-45 with a UR5e (September 2026)

This is the working log of the original measurement campaign, kept as a reference for what to
expect on a new piano: the robot-side limits that were hit, the key/hammer effects that were
discovered, and the final numbers. Commands were updated to the console-script names of this
repository; the raw data of these sessions live in the `ExpressivePianist` repository
(`data/piano_id_C1..C7`, `data/switch_id_C4*`, `data/velmap_C4`, `data/hard_C4`), the derived
results are in [`../examples/yamaha_p45/`](../examples/yamaha_p45/). For the procedure to follow on
a new piano read the top-level [README](../README.md).

---

Goal: measure how the MIDI velocity (loudness) reported by the P-45 depends on how the key is
pressed, by pressing keys with a UR5 under prescribed speed profiles while logging the robot state
(TCP pose / speed / force at the RTDE rate) and the piano's MIDI output on one clock.

## 1. How the P-45 measures velocity

* The GHS action has **two switches per key** on a rubber contact strip, closing at two different
  key depths. The controller measures the **time Δt between the first and the second contact** and
  maps it to velocity 1–127; note-on is sent when the *second* switch closes. Velocity therefore
  depends only on the key's mean speed between the two switch depths, not on the force, not on
  the speed at the key surface, and not on what happens after the second switch.
* Touch-sensitivity setting (hold GRAND PIANO/FUNCTION and press A2–C3): Fixed (A2, constant
  velocity), Soft (A#2), Medium (B2, default), Hard (C3). It only changes the Δt→velocity curve, so
  record it in `--touch-setting`; the switch depths do not change.
* Community measurements report a usable range of roughly 15–110 on Medium; 127 needs extremely
  fast strikes. Expect saturation at both ends.

So the model to identify is

    velocity = f( (d2 - d1) / (t(d2) - t(d1)) )         with unknown switch depths d1 < d2

and `f` is a monotone (roughly log-linear) curve per touch setting.

## 2. Experiment design

**Setup**
* Compliant fingertip (silicone/rubber tip) on the UR flange; TCP set to the tip (`--tcp`).
* Piano USB→PC (`amidi -l` / `python midi_io.py` must show the P-45), robot in Remote Control.
* Teach three reference keys once (`--teach-layout`); the script interpolates all other keys with
  the standard 23.5 mm white-key pitch and refines each key's contact height automatically
  (`--contact-method force` on UR5e; `midi` or `teach` on a CB3 UR5 without F/T sensor).

**Factors**
| factor | levels | why |
|---|---|---|
| key (register / colour) | C2, F#2, C3, C4, C#4, G4, C5, A#5, C6 (`--keys registers`) | graded hammers and black-key geometry might change the Δt for the same fingertip motion |
| profile | `const` v ∈ {0.02 … 0.55} m/s; `accel`/`decel` ramps; `two_stage` slow→fast switching at 2/4/6 mm; `sine` | constant speed gives the curve; the others test that only the inter-switch speed matters and locate the switches |
| start height | 0 mm (start from just touching, accelerates inside the key) and 6 mm above | at 0 mm the first mm are an acceleration phase, so the "constant" speed is only reached late; starting above makes the speed truly constant through the switches |
| repeats | ≥ 3 | velocity quantisation and timing jitter |

Trials are interleaved randomly (`--seed`) so drift does not alias with a factor.

**Per trial** the script: moves to hover → to the start height → settles → zeroes the force
baseline → streams the depth trajectory with `servoL` at the RTDE rate (recording t, target z, TCP
pose, TCP speed, TCP force, q, qd every cycle) → waits for the note-off → saves `trial_XXXX.npz`
with all MIDI events in that window. A force or depth limit aborts the press and lifts.

**Analysis** (`analyze.py`)
1. Constant-speed trials: `t_on − t_contact = d2 / v + L` → depth of the note-on switch `d2` and
   the MIDI latency `L` (linear fit in 1/v, pooled and per key).
2. Scan `d1`: for each candidate compute the mean speed between `d1` and `d2` for *every* trial of
   every profile, fit a monotone curve velocity ~ log(speed) and take the `d1` with the smallest
   residual. If the two-switch model is right, one `d1` collapses all profiles onto one curve;
   the "residual by profile kind" table in the report is the test.
3. Fit `velocity = a + b·ln(v_sensor)` pooled and per key; report residuals per key, black vs white,
   and an isotonic lookup table (non-parametric). These are what a controller needs: to hit a
   target velocity, drive the fingertip so that its mean speed between `d1` and `d2` equals
   `f⁻¹(velocity)`.

## 3. Hands-on test on the real setup (`real_test.py`, no automatic contact search)

Hardware as of 2026-09-04: the P-45 is ALSA client `24:0` (`Digital Piano:Digital Piano MIDI 1`),
auto-selected by `--midi-port "Digital Piano"`. The UR5 hangs off `enp0s31f6` (host
192.168.113.111/24); nothing on that /24 answered ARP/ping when this was written, so check the
robot's IP on the pendant (Settings → System → Network) and pass it as `--robot-ip`. If the robot
is still on 192.168.1.x (gs-core used 192.168.1.189) either give the robot a 192.168.113.x address
or add a 192.168.1.x address to `enp0s31f6`.

```bash
# 1) what is it? model, PolyScope version (5.x = UR5e with F/T sensor, 3.x = CB3 UR5), RTDE rate
kve-probe <ip>

# 2) jog the fingertip (freedrive) until it just touches a key, then
kve-test interactive --robot-ip <ip> [--tcp x y z rx ry rz]
#    kv> t          listen to the piano for 5 s (play a key by hand)
#    kv> h          take the current pose as home (= contact reference)
#    kv> p 0.05     press 9.5 mm at 0.05 m/s; prints velocity, depth and speed at note-on
#    kv> p 0.2 9.5 6   same at 0.2 m/s starting 6 mm above (at speed before contact)
#    kv> a 0.03 0.3 accelerating press; a 0.3 0.03 decelerating
#    kv> r / l      move home one white key right / left (--key-axis, or calibrate with c)
#    kv> d 0.3      move the contact reference 0.3 mm deeper
#    kv> s 0.03 0.05 0.1 0.2   sweep

# 3) scripted sweep on the current key, then analyse
kve-test sweep --robot-ip <ip> \
    --speeds 0.03 0.05 0.08 0.12 0.2 0.3 --start-above-mm 0 6 --repeats 3 --out data/real_C4
kve-analyze data/real_C4
```

Every press is saved in the same `trial_XXXX.npz` format as the automatic experiment, so
`analyze.py` works on these sessions too (the switch-depth fit needs ≥ 3 constant speeds with a
start height > 0). `--sim` dry-runs the same script on the simulator.

### What the first real presses showed (2026-09-04, C4, from the touching position)

Commanded 0.03 … 6 m/s gave velocity 22 → 51 and then a flat 51 from 0.3 m/s on, while hand
presses reached 75–89. The recordings show why: the *actual* fingertip speed saturated at
0.158 m/s and note-on always came at 97 ms. Two limits stacked up:

1. the trajectory generator capped acceleration at 5 m/s², so inside 9.5 mm of travel no profile
   could exceed ~0.29 m/s - every command ≥ 0.3 produced the same target trajectory;
2. `servoL` (lookahead 0.03 s) only followed that ramp at ~4–6 m/s², peaking at 0.158 m/s.

So the plateau is the robot, not the piano. Rules for a rigorous run:

* **Be at speed before touching the key.** Use a run-up (`p V DEPTH SA`, default now automatic:
  v²/2a + 1 mm) so the speed is constant through both switches; from-contact starts (`SA=0`) are a
  separate, deliberately "accelerating" condition.
* **Use velocity control for fast presses** (`x V`, `--press-method speedl`): speedL has no
  lookahead smoothing and follows the commanded speed; it brakes after `--brake-depth-mm` (8.8 mm,
  just past the note-on switch) with `--brake-accel`. Braking from 0.3 m/s at 15 m/s² needs 3 mm,
  so the key bed takes the rest - keep the rubber tip, keep `--force-abort-n` at 15 N, and raise
  the speed gradually (0.15 → 0.2 → 0.3 → 0.4) watching `|dF|max` in the printout.
* **Trust only measured speed.** Every printout now shows `v(3-7mm)` (mean speed between 3 and
  7 mm depth, the inter-switch region), the peak speed and a `!! tracking` warning when the
  commanded speed was not reached; the analysis uses the measured speed, never the command.
* Same speed several times, randomised order, both start conditions, several keys; then
  `analyze.py` fits d1/d2/latency and the velocity curve, and the "residual by profile kind" table
  tells whether the inter-switch speed alone explains the velocity.
* Note-on depth drifting shallower at faster presses (8.4 → 7.8 mm) together with the force dipping
  ~1 N below baseline suggests the key may run ahead of the fingertip (or the tool flexes) during
  hard accelerations - another reason to accelerate *before* contact.

### Key launch and bounce (second real session)

At 0.25–0.3 m/s the note-on sometimes arrived when the fingertip was only 2.8–4.5 mm deep and
was followed by 2–3 note-ons of the same key. A rigid fingertip hitting a light key at speed is a
nearly elastic impact: the key is launched ahead of the finger (up to ~2× the finger speed),
closes the switches on its own, bounces off the key bed, hits the descending finger and gets
pressed again. The reported velocity then belongs to the flying key, not to the finger, and the
same command gives different velocities between `p` and `x`. Countermeasures now built in:

* **Kick profile** (`k V1 [V0] [A]`, `--preset kick`): touch the key slowly (0.05 m/s) and
  accelerate *inside* the key at A m/s² up to V1. The finger pushes the key the whole way, so the
  key cannot separate; the inter-switch speed is still V1 if A is large enough to reach it before
  ~3 mm (`v(3-7mm)` in the printout tells).
* **Stop on note-on** (default; `--no-stop-on-note` disables): the descent ends as soon as the
  piano reports the note, so a bounced key meets a stationary fingertip and is not re-triggered.
* A softer (foam/silicone) fingertip makes the contact inelastic and reduces the key-bed impact.
* Printout flags: `!! note-on while the fingertip was only x mm deep` (launch) and `!! N note-ons
  ... key bounced` (re-trigger).

### The three press modes of `real_test.py`

| | `p V` (servo) | `x V` (speedl) | `k V1 V0 A` (kick) |
|---|---|---|---|
| approach | run-up above the key, at speed V before contact | same | slow contact at V0 (0.05 m/s) |
| inside the key | follows a pre-computed time/depth trajectory with `servoL` (lookahead 0.03 s → lags/attenuates above ~0.15 m/s) | closed-loop `speedL`: each 2 ms the commanded speed is V | closed-loop `speedL`: speed = √(V0² + 2·A·depth) up to V1, i.e. the finger *pushes and accelerates* the key |
| impact at contact | hard (rigid tip at V) → key can be launched | hard | soft |
| end of descent | note-on → immediate fast rise (`--after-note lift`); otherwise key-bed force (≥ 6 mm) or depth limit | same | same |

The force stop is only armed below `--stop-force-min-depth-mm` (6 mm): during in-key acceleration
the tool's inertia (m·a, several N at 20–30 m/s²) looks like a contact force and would otherwise
stop the press early - which itself launches the key.

### Why velocity plateaued at ~90–98 (third real session) and how far the arm can go

`k 0.30 … x 0.50` all ended with `stop: force … at 6.1–6.7 mm` and the recordings show the speed
at 8 mm depth (the note-on switch) had already collapsed to ≈0 while the speed at 6 mm was the
commanded one. The force fallback (5 N, armed from 6 mm) fired the moment it was armed, because
a GHS key moving at 0.4 m/s pushes back 5–6 N by itself and the tool's inertia during in-key
acceleration adds several N more (the velocity loop accelerates at 30–45 m/s², not the 20
commanded). So the arm braked *before* the second switch: premature braking, not a piano limit.
Defaults are now 8 N armed from 9 mm (the note-on lift is the primary stop), the run-up carries a
1.5× margin, and `k` touches at min(0.05, V1/4) so a slow kick does not launch the key.

Log-linear fit of the un-braked real points (v = mean speed 3–7 mm): velocity ≈ 141 + 38.8·ln v,
i.e. 89 @ 0.26, 100 @ 0.35, 110 @ 0.45, 120 @ 0.58, **127 @ ≈ 0.7 m/s**. Braking distance
v²/2a at 25 m/s²: 0.3 → 1.8 mm, 0.4 → 3.2 mm, 0.5 → 5 mm, 0.7 → 9.8 mm; the key bed is at
~10 mm and the arm's effective mass is 10–50× a finger's, so pushing the key at ≥0.45 m/s all the
way to the switch means hitting the bed hard. Two ways out:

* `th V1 [V0] [A] [R]` (**throw**): accelerate the key to V1 inside the first R mm (default 4 mm)
  and *release* (brake at 25 m/s²): the light key flies on alone through the switches - this is
  what a real action does - while the arm stops before the bed. Expect more scatter than `k`.
* A long-travel compliant fingertip (≥ 5 mm of foam/spring) so the arm can decelerate after the
  switch while the tip compresses.

The simulator's virtual key does not fly, so `th` shows "no note-on" in `--sim`.

## 3b. One-shot identification: home → A, B, latency → velocity map (`identify_switches.py`)

```bash
# jog the fingertip roughly onto the key, then (about 25 min on the real setup):
kve-identify --out data/piano_id_C4
kve-analyze-switches data/piano_id_C4     # re-analyse
```

Default `--phase all` = home confirmation with the F/T sensor (`--no-refine-home` to skip) → phase b
(B, latency) → phase a (A by pause probe) → phase s (speed sweep until velocity 1..127 is covered or
the arm limit is reached) → 3 slow presses for a drift check. Everything below is written in SI
units to `results_si.json` (A_m, B_m, latency_s, velocity = k·dt_s^b and the log-linear alternative,
dt→velocity table, speed→velocity map, velocity→speed inverse, timeout dt for velocity 1) and drawn
in `fig_summary_si.png`: (1) B fit, delay vs 1/v; (2) A probe classification with the A bracket and
the note-off depth; (3) velocity vs dt with the fit and the timeout line; (4) velocity vs mean speed.

### Phase details

For the simulator we need the piano's *mechanism*, not just velocity per press: timer starts at
depth **A**, stops at depth **B** (note-on, sent `latency` later), velocity = f(dt). The script runs
three phases on one key (home = fingertip just touching, as in real_test.py):

| phase | what the robot does | what it yields |
|---|---|---|
| **b** | slow constant-speed presses (0.02–0.12 m/s, ×3), slow sampled lift | `t_on − t_contact = B/v + L` → **B**, **latency**; depth at note-off on the way up |
| **a** | *pause probe*: descend at 0.02 m/s to depth d, pause 0.4 s, finish at 0.2 m/s; coarse 1 mm sweep then bisection to 0.1 mm | d < A → pause is before the timer → normal velocity; d > A → pause inside dt → velocity < 20 or no note. The boundary is **A**, model-free |
| **m** | presses at 10 speeds (constant when slow, in-key acceleration when fast) + shape tests (slow→fast vs fast→slow between A and B with the same mean speed) | dt = (t_on − L) − t(A crossed, from the fingertip trajectory) vs velocity → log fit and isotonic table; shape residuals test "velocity depends on dt only"; presses where the fingertip was not at B when B closed (launched key) are excluded |

```bash
kve-identify --phase all --refine-home --out data/switch_id_C4
kve-analyze-switches data/switch_id_C4     # re-analyse any time
```

`switch_id.json` → `model_for_sim` = A_mm, B_mm, latency_ms, noteoff_depth_mm, `velocity_from_dt`
(c0 + c1·ln dt_ms) and a 30-row dt→velocity table. Figures: `fig_B_fit.png`, `fig_A_probe.png`,
`fig_dt_velocity.png`. Simulator check (truth A 3.0 / B 8.0 mm, 6 ms, 177.9 − 29.5·ln dt):
recovered A 2.97, B 8.01, 7.0 ms, 177.9 − 29.4·ln dt, R² 0.999, shape residuals < 0.5.

Repeat per touch-sensitivity setting (only f changes) and on a bass / treble / black key to check
that A and B are the same across the keyboard.

### Result of the first stiff-tip identification (C4, Medium touch, 2026-09-05)

* **B = 7.81 mm**, latency ≈ 1 ms, **A = 4.83 mm** (pause-probe boundary 4.75/4.81–4.88 mm; note-off
  on the way up at 4.83 ± 0.03 mm - two independent measurements agree). Reference drift < 0.05 mm.
* Constant-speed presses (0.02–0.11 m/s, finger dt 25–160 ms): **velocity = 67.4 − 9.7·ln(dt_ms)**,
  R² 0.96, rms 1.3 - this is the "key driven by the finger" regime the two-depth model describes.
* Every accelerating press (kick, speed steps, probes restarting from rest) produced velocities
  37–50 above that curve, and note-on arrived while the finger was still 1.5–2.3 mm above B (kick
  0.15: note-on at 5.5–5.9 mm, velocity 65; speed step at 6.3 mm → note-on at 6.3–6.5 mm).
  The piano therefore measured a much shorter dt than the key's: the **hammer is thrown ahead of the
  key** by the acceleration (the P-45's sensors follow the hammer/action, not the key surface).
  A simulator that triggers on key depth alone reproduces slow, smooth playing only; fast strokes
  need a hammer with its own inertia that separates from the key when the key stops accelerating.
* Practical: with a stiff tip keep `--kick-accel` ≤ 8–10 m/s² (identify_switches now defaults to 8,
  force abort 25 N); an acceleration finished before A keeps the hammer on the key.

### Second stiff-tip session (`data/switch_id_C4_fast`, speeds to 0.45, kick accel 8 m/s²)

* Reproduced: B 7.86 mm, A 4.78 mm (probe) / 4.82 mm (note-off), latency ≈ 0–2 ms, drift < 0.25 mm.
* All kicks with V1 ≥ 0.25 gave the *same* velocity 65–66. Not a piano limit: with a = 8 m/s² the
  key can only reach 0.36 m/s by B, so every V1 ≥ 0.3 runs the identical acceleration profile through
  A→B and the finger dt is pinned at ≈ 12 ms; kick 0.25 also has dt ≈ 12 ms → 66. The piano is
  consistent with a pure dt dependence.
* With those short-dt points the curve is clearly not log-linear: **velocity ≈ 232 · dt_ms^−0.51**
  (R² 0.96, rms 3.8, dt 13–170 ms), i.e. velocity ∝ √(key speed): 0.02 m/s → 19, 0.1 → 40,
  0.25 → 66, extrapolated 0.5 → 94, 0.9 → 126. To fill 70–127 the key must be *driven* through
  A→B at 0.3–0.9 m/s: a higher `--kick-accel` (20–40) that finishes accelerating before A, or a
  faster constant run-up - both risk throwing the hammer, which the analysis flags.
* Hammer flight confirmed once more: fast→slow steps at 6.3 mm gave velocity 66 with a finger dt of
  19 ms (hammer kept the fast phase's speed after the key slowed); slow→fast steps triggered note-on
  at 6.3–6.5 mm, before the key reached B.

### Phase s: velocity map by speed sweep (assumes velocity = f(mean key speed A→B))

```bash
kve-identify --phase s \
    --ab-from data/switch_id_C4_fast/switch_id.json --sweep-repeats 2 --out data/velmap_C4
```

Every press drives the key at a constant target speed through A→B: ≤ 0.05 m/s as a plain constant
press, above that as a kick whose acceleration is chosen per speed so the target is reached exactly
at A (cap `--kick-accel-max`), above `--sweep-max-kick-speed` with an in-air run-up (impact, flagged).
The descent ends at note-on. After each round the median velocity per speed is inspected: a speed
×1.3 is added while the top velocity is below 127 (until `--sweep-max-speed`, default 0.55 m/s -
braking from there already reaches the key bed), a speed ÷1.6 while the bottom is above 1, and a
geometric-mean speed wherever neighbours differ by more than `--sweep-gap` velocity units; up to
`--sweep-fill-rounds` rounds. The analysis prints a speed table (target → measured mean speed A→B,
median/min/max velocity, hammer lead at B) and writes `velocity_map` into `switch_id.json`:
`speed_to_velocity` (isotonic, 40 points) and `velocity_to_speed` (1…127 → lowest mean speed that
reaches it, `null` outside the covered range). Values of 1 and 127 are clipped by the piano and are
kept in the table but not in the parametric fits. Simulator check: 1…100 covered with 0.005…0.36 m/s
in 2 fill rounds.

### Speed-sweep result (`data/velmap_C4`, C4, Medium, stiff tip, 2026-09-05)

Velocity 1–93 covered with mean key speeds 0.005–0.43 m/s, two presses per speed, repeatability
±1–2 velocity units. Floor: dt > ~250 ms (≤ 0.012 m/s) → velocity 1 (clip); 0.013 → 9,
0.014 → 13, 0.02 → 19. Then 0.04 → 25, 0.08 → 31, 0.12 → 41, 0.18 → 55, 0.22 → 62, 0.31 → 75,
0.38 → 87, 0.43 → 91. MIDI latency from the fast presses ≈ 3–4 ms (the finger is 1–1.7 mm past
B when the note arrives at 0.3–0.45 m/s). The arm limit is ~0.45 m/s: note-on then arrives after
the finger has reached the key bed and the force fallback (8 N) stops the press; impact peaks
20–40 N. Reaching 127 would need ≈ 0.7 m/s driven through A→B.

### Pushing for velocity 127 (`--hard`)

Medium and Hard touch settings gave identical MIDI curves (within ±2 at every speed), so the only
route to 110–127 is a faster key: ≈ 0.65 m/s through A→B by the fitted curve. `--hard` runs
0.45/0.55/0.65/0.75 m/s once each, ascending, with kick acceleration up to 45 m/s² (0.65 m/s needs
44 m/s² to be at speed by A = 4.8 mm), braking 40 m/s², force abort 80 N, **position-triggered
braking 0.6 mm past B** (the piano's timer has stopped at B; waiting for the MIDI note costs another
v·4 ms ≈ 2.6 mm of key-bed travel) and an impact guard that stops adding speed once a press exceeded
60 N at the F/T sensor. Expect key-bed impacts of 40–80 N; the UR's own force limit (150 N) is the
last resort. Use a rigid tip, keep a hand on the e-stop.

```bash
kve-identify --hard --ab-from data/piano_id_C4/switch_id.json --out data/hard_C4
```

### Final velocity model for C4 (pooled, extrapolated to 127)

`analyze_switches.py data/piano_id_C4 --pool data/velmap_C4 data/hard_C4` fits all 91 driven presses
of the three sessions (each pooled session re-referenced by its own slow presses):

* A = 4.84 mm, B = 7.95 mm (A→B = 3.11 mm), MIDI latency ≈ 3.4 ms, velocity 1 for dt ≥ 0.31 s.
* **velocity = 3.82 · dt_s^−0.645** (R² 0.980, rms 3.8; dt 6–262 ms), equivalently
  velocity ≈ 3.82 · (0.00311 / v)^−0.645 with v the mean key speed A→B in m/s.
* Measured up to velocity 98 at 0.444 m/s; the arm cannot drive the key faster from a soft contact
  (speed-loop rise time ≈ 20 ms, key bed 2 mm past B). Above that the model is the fitted curve:
  100 → 0.49 m/s, 110 → 0.57, 120 → 0.65, **127 → 0.71 m/s (dt 4.4 ms)**. The single-session
  fits bracket this at 0.65–0.77 m/s, which is the uncertainty of the extrapolation.
* `results_si.json` → `velocity_to_speed_full_1_127` lists every velocity with its mean speed and
  whether it is measured or extrapolated; `fig_velocity_model_si.png` shows both parts.

## 3c. Applying the measurements to the simulators

`export_piano_model.py data/piano_id_C1 … data/piano_id_C7` writes
`ExpressivePianist/asset_zoo/piano/p45_velocity_model.json` (per measured key: A/B as fractions of
the key dip, power-law k/b, timeout, latency, and the measured speed table). Measured on C1..C7:

| key | A/dip | B/dip | velocity = k·dt^b | timeout dt | latency |
|---|---|---|---|---|---|
| C1 | 0.50 | 0.81 | 6.57·dt^−0.50 | 0.19 s | 3.9 ms |
| C2 | 0.49 | 0.84 | 6.09·dt^−0.53 | 0.24 s | 3.4 ms |
| C3 | 0.47 | 0.82 | 5.44·dt^−0.57 | 0.25 s | 4.2 ms |
| C4 | 0.48 | 0.80 | 3.82·dt^−0.65 | 0.31 s | 3.4 ms |
| C5 | 0.49 | 0.82 | 5.74·dt^−0.55 | 0.27 s | 5.1 ms |
| C6 | 0.49 | 0.82 | 5.27·dt^−0.57 | 0.28 s | 3.7 ms |
| C7 | 0.37 | 0.69 | 4.26·dt^−0.64 | 0.23 s | 4.0 ms |

`ExpressivePianist/asset_zoo/piano/velocity_model.py` (`P45VelocityModel`) interpolates these over
the 88 keys and evaluates velocity from a per-key lookup table on a log-dt grid: inside the measured
range the table reproduces the real speed→velocity curve, outside it follows the power law, and
dt ≥ timeout gives 1. The switch depths are placed as *absolute* depths below the resting key surface
(angle = rest + depth / 0.15 m for white keys; black keys scaled by their 0.8× travel), because the
simulated white keys sag 0.69° (1.7 mm at the tip, 17 % of the travel) under gravity - the MJCF
spring (1 Nm/rad, ref −1°) does not carry the key weight - and A→B must stay 3.1 mm as on the real
key. `key_rest_angles()` measures the sag from the MJCF. Consequence: on a simulated white key B sits
only 0.2 mm above the bottom (3.72° of 3.81°); making the spring hold the key (springref ≈ −1.8°
or stiffness 1.7 Nm/rad) would restore the real 10 mm dip but changes the key dynamics the
policies were trained with, so it is left as a decision for the training side. Both simulators use it:

* dm_control (`sim_robopianist/midi_module.py`, `piano.py`): `TwoSwitchTracker` arms the timer when a
  key passes angle_A, sends note-on with velocity = f(dt) when it passes angle_B, note-off when it
  rises back above angle_A; crossing times are interpolated inside the 2 ms substep.
  `piano.activation` now follows the piano's own note-on/off, and `piano_state` crosses zero at
  angle_B instead of the old 0.5°-from-bottom threshold, so the state observation, the activation
  used by the rewards (f1, key_pressing) and the MIDI velocity share one definition of "pressed".
* mjlab (`custom_mjlab/env/robopianist_rl_env.py`, `_p45_update`): the same tracker vectorised in
  torch over envs × keys, called every physics substep (2.5 ms) with sub-step interpolation.

`EP_VELOCITY_MODEL=legacy` restores the old behaviour (note-on at the 0.5° activation threshold,
velocity = qvel·127/3.5) for comparisons with old checkpoints.

### Sim vs real: rigid ball pressing the simulated keys (`sim_ball_press.py`)

```bash
MUJOCO_GL=egl kve-sim-ball --out data/sim_vs_real
MUJOCO_GL=egl kve-sim-ball --model legacy --out data/sim_vs_real
```

A mocap sphere (r = 8 mm) presses the front of C1..C7 at the same target speeds as the real sweeps
(6 mm run-up, constant speed through the key, stop 9.5 mm below the surface, hold, lift). The key's
own angle gives the mean key speed A→B at the press point exactly as on the robot, the MidiModule
gives the simulated velocity; the script prints a per-octave table and writes
`sim_vs_real_<model>.json` and `fig_sim_vs_real_<model>.png` (real curve vs sim curve per octave,
error histogram, MAE/bias). Errors are evaluated at the same *measured* key speed (the real arm did
not always reach its target speed).

Result (press point 5 mm from the tip; switches at the measured absolute depths below the resting
key surface; 115 presses):

| | C1 | C2 | C3 | C4 | C5 | C6 | C7 | all |
|---|---|---|---|---|---|---|---|---|
| sim P-45 model MAE | 2.8 | 1.4 | 2.1 | 1.4 | 1.4 | 2.8 | 3.8 | **2.1** (bias −1.1) |
| sim legacy MAE | 7.9 | 8.6 | 7.3 | 8.2 | 10.4 | 8.8 | 12.5 | **9.0** (bias −5.5) |

The remaining deviations sit mostly at the velocity-1 timeout edge, where the real piano jumps from
1 to ~10 within 0.001 m/s, and at C7 (its A/B are 1 mm shallower than the other keys, so the
interpolation over neighbouring keys is coarser there). The legacy model (note-on at the 0.5°
activation threshold, velocity = qvel·127/3.5) is too soft below 0.2 m/s (C4: 0.05 m/s → 12 vs 27
real, 0.11 → 27 vs 37), too loud above (0.45 → 109 vs 98), and has no timeout floor.

`sim_ball_press.py` also saves every press trajectory (`sim_traces_<model>.npz`) and renders the
C4 presses at 0.05 / 0.25 / 0.45 m/s to `sim_ball_press_C4_<model>.gif` (`--render-note`,
`--render-speeds`). `compare_sim_real.py` then puts everything side by side:
`fig_compare_overview.png` (real median with min–max bars and isotonic map, sim p45, sim legacy,
per-octave MAE bars), `fig_compare_traces_C4.png` (real fingertip depth vs simulated key depth
over time with the note-on instants and the A/B levels) and `compare_table.md`.

## 4. Running the full automatic experiment

```bash
# dry run, no hardware (virtual 2-switch key at 3 / 8 mm, 6 ms latency)
kve-run --sim --preset quick --keys quick --out data/kv_sim
kve-analyze data/kv_sim

# check MIDI
kve-midi

# teach layout (freedrive; fingertip just touching C2, C6, C#4 at the normal press point)
kve-run --robot-ip <ip> --tcp 0 0 0.12 0 0 0 \
    --teach-layout key_layout.json

# short check run, then the full grid
kve-run --robot-ip <ip> --tcp 0 0 0.12 0 0 0 \
    --keys C4 --preset quick --repeats 1 --out data/kv_check
kve-run --robot-ip <ip> --tcp 0 0 0.12 0 0 0 \
    --keys registers --preset full --repeats 3 --touch-setting medium --out data/kv_medium
kve-analyze data/kv_medium
```

Useful options: `--plan-only` prints the trial list; `--speeds`, `--start-above-mm`, `--depth-mm`
change the grid; `--hz`, `--lookahead`, `--gain` tune tracking (keep lookahead at 0.03 for fast
profiles); `--force-abort-n` / `--contact-force-n` for the F/T thresholds; `--contact-method midi`
when there is no F/T sensor (contact height is then referenced `--assumed-d2-mm` above the
note-on height; the analysis re-estimates the true depths relative to that reference).

Safety: keep `--depth-mm` ≤ 9.5 (P-45 key dip ≈ 10 mm), start with slow speeds and one key, keep a
hand on the e-stop the first time each profile runs. `Ctrl-C` stops the servo and lifts.

## 5. Files

| file | role |
|---|---|
| `common.py` | key geometry, `PressProfile` (speed profile → depth trajectory), `TrialRecord` I/O |
| `layout.py` | keyboard layout from three taught poses, per-key contact heights |
| `midi_io.py` | timestamped MIDI listener (mido/rtmidi) and the simulated two-switch piano |
| `ur5_io.py` | ur_rtde driver (`servoL` streaming, contact search, force/depth abort) and a kinematic simulator |
| `run_experiment.py` | plan → contact search → trials → `trial_XXXX.npz` + `summary.csv` |
| `real_test.py` | manual-home hands-on test: interactive press / shift / sweep on the real robot |
| `probe_robot.py` | identify the controller (UR5 CB3 vs UR5e, F/T sensor, RTDE rate) |
| `identify_switches.py` / `analyze_switches.py` | switch depths A/B, latency, dt→velocity table for the simulator |
| `analyze.py` | switch-depth / latency fit, first-switch scan, velocity mapping, figures |

Output per session: `session.json`, `key_layout_measured.json`, `trial_XXXX.npz`
(`robot_samples` [N × 33], `midi_events` [M × 4], `meta`), `summary.csv`, `analysis.json`, `fig_*.png`.
