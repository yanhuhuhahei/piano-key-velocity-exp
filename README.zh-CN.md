# 用 UR 机械臂辨识数字钢琴的键速–力度（MIDI velocity）模型

*English version: [README.md](README.md).*

这个仓库用来测量 **数字钢琴如何把琴键运动转换成 MIDI 力度（velocity）**：用 UR5e 以受控的速度曲线按键，
在同一时钟下记录机器人状态（500 Hz）和钢琴的 MIDI 输出；然后把辨识出的模型放进 MuJoCo 仿真里 **做验证**。

一台钢琴的最终产物是一个小的 JSON 模型（每个测过的键：两个触点深度、MIDI 延迟、`dt → velocity` 曲线），
仿真器读入它以后，仿真中的按键就能产生与真琴一致的 MIDI 力度。仓库已经用来辨识过一台 Yamaha P-45
（结果见 [`examples/yamaha_p45/`](examples/yamaha_p45/)）；本文是在 **另一台钢琴上重复整个实验的步骤**。

---

## 目录

1. [测的是什么（模型）](#1-测的是什么模型)
2. [软硬件要求](#2-软硬件要求)
3. [安装](#3-安装)
4. [第 0 步：无硬件空跑](#4-第-0-步无硬件空跑)
5. [实物实验步骤](#5-实物实验步骤)
6. [仿真验证步骤](#6-仿真验证步骤)
7. [在自己的仿真器里使用模型](#7-在自己的仿真器里使用模型)
8. [文件与数据格式](#8-文件与数据格式)
9. [排错与坑](#9-排错与坑)
10. [参考结果：Yamaha P-45](#10-参考结果yamaha-p-45)

---

## 1. 测的是什么（模型）

数字钢琴的键盘机构每个键有 **两个触点**，在不同深度闭合。控制器测量琴键从第一个触点（深度 **A**）
走到第二个触点（深度 **B**）所用的时间 `dt`，在 B 闭合时发出 *note-on*（再加一个很小的 MIDI 延迟 `L`），
并把 `dt` 映射为力度 1–127。因此力度只取决于琴键在 A→B 之间的平均速度，与按压力无关，也与过了 B
以后发生的事无关。琴键回升越过 A 时发出 *note-off*。

```
以静止键面为零点的深度
   0 ──── 键面（指尖刚好接触）
   A ──── 计时开始                                   ┐
   B ──── 计时结束，L 之后发出 note-on                ┘ dt = (B - A) / 平均速度
 键底 ──── 键行程末端（P-45 约 10 mm）
                       velocity = f(dt)  (≈ k · dt^b，dt ≥ 超时值时 velocity = 1)
```

脚本对每个键辨识：**A、B、L**、**`dt → velocity` 曲线**（幂律 + 单调查找表）、**超时值**（dt 大于它
力度为 1）以及机械臂实际达到的 **速度 → 力度表**。整个流程 *一次只测一个键*，指尖位置手动示教，
主流程不需要键盘标定。

## 2. 软硬件要求

**硬件**

| 项目 | 说明 |
|---|---|
| UR 机械臂 | 在 **UR5e**（PolyScope 5.x，自带六维力传感器，RTDE 500 Hz）上测试过。CB3 的 UR5 只能做慢速按压：没有力传感器就没有键底检测和 home 精修（`--no-refine-home`、`--contact-method midi`）。 |
| 指尖 | **刚性** 指尖，末端圆头（半径约 5–10 mm），例如 3D 打印或金属探头。软指尖在快速按压时（5–20 N）会被压缩几毫米，破坏深度测量（这类按压会被标记并剔除，但数据就浪费了）。安装时让指尖竖直向下。 |
| 钢琴 | 任何带 USB-MIDI 的数字钢琴。记录键行程（GHS 键盘约 10 mm）和你使用的触感设置。 |
| 电脑 | 与机器人同网段的 Linux（ALSA MIDI）。Windows/macOS 理论上可以（`mido`/`python-rtmidi`），未测试。 |
| 安全 | 每种按压曲线第一次跑时手放在急停上。脚本会在力阶跃（默认 25 N）和深度限制时中止按压，并在收到 note 后立即抬起，但机械臂是刚性的，键底只在触点 B 下方 2 mm。 |

**软件**：Python 3.10–3.12、[uv](https://docs.astral.sh/uv/)（或 pip）、`ur_rtde` 的 Python 轮子（自动安装），
仿真部分需要 MuJoCo + dm_control（`sim` extra）。

## 3. 安装

```bash
git clone <本仓库> && cd piano-key-velocity-exp
uv sync --extra sim            # 创建 .venv 并安装全部依赖（机器人电脑上不需要 MuJoCo 可去掉 --extra sim）
uv run kve-midi --list         # 应列出 MIDI 端口（空列表 = 钢琴还没接上）
```

下面所有命令都是安装到环境里的控制台脚本，用 `uv run <cmd>` 运行，或先 `source .venv/bin/activate` 再直接调用。

| 命令 | 作用 |
|---|---|
| `kve-probe <ip>` | 识别 UR 控制器（型号、PolyScope、力传感器、RTDE 频率） |
| `kve-jog` | 小步慢速移动 TCP、打印位姿 |
| `kve-midi` | 列出 MIDI 端口 / 打印钢琴的每个音符事件 |
| `kve-test` | 单键手动按压（交互式提示符或脚本化扫描） |
| `kve-identify` | **主实验**：一个键的 A、B、延迟和力度曲线 |
| `kve-analyze-switches` | 重新分析 / 合并 `kve-identify` 的会话 |
| `kve-export` | 把各键结果合并成一个模型 JSON |
| `kve-sim-ball` | MuJoCo：刚性小球用该模型按压仿真琴键 |
| `kve-compare` | 真实 vs 仿真力度的图表 |
| `kve-run` / `kve-analyze` | 可选：基于示教键盘布局的多键自动实验 |

连接参数可以用环境变量设置一次，不用每条命令都写：

```bash
export UR_ROBOT_IP=192.168.1.10          # 对应 --robot-ip
export KVE_MIDI_PORT="Digital Piano"     # 对应 --midi-port（端口名子串；不设则自动检测）
```

## 4. 第 0 步：无硬件空跑

所有机器人脚本都支持 `--sim`：一个运动学机器人模型按压 *虚拟双触点琴键*（真值：A = 3.0 mm，
B = 8.0 mm，延迟 6 ms，velocity = 177.9 − 29.5·ln dt_ms）。先跑它来检查安装并熟悉输出：

```bash
uv run kve-identify --sim --out data/sim_id_C4          # 约 1 分钟，打印完整报告
uv run kve-analyze-switches data/sim_id_C4              # 随时可重新分析
```

报告中应看到：`B = 8.0x mm`、`latency ≈ 7 ms`（6 ms + 半个控制周期）、`A = 2.9x mm (bracketed 2.9 .. 3.0)`、
`log-linear: velocity = 177.x −29.x·ln(dt_ms)  R^2 = 1.000`、漂移 `< 0.05 mm`、覆盖力度 `1..105`。
看到这些就说明流程正常；去掉 `--sim` 的同一条命令就在真机上跑。

另外可试：`uv run kve-test --sim sweep --speeds 0.03 0.1 0.3 --out data/sim_test`（手动模式）和
`uv run kve-run --sim --preset quick --keys quick --out data/sim_run && uv run kve-analyze data/sim_run`（多键自动实验）。

## 5. 实物实验步骤

时间预算：`kve-identify` **每个键约 25–30 分钟**（60–90 次按压），第一天另加约 30 分钟搭建。
每个八度测一个键（C1…C7）需要一个下午。

### 5.1 钢琴

1. USB 连接钢琴并开机，检查 MIDI 数据流：
   ```bash
   uv run kve-midi --list        # 记下端口名，例如 "Digital Piano:Digital Piano MIDI 1 20:0"
   uv run kve-midi               # 手弹几个键：必须出现带力度的 note_on/note_off
   ```
   如果列出多个端口，给每个脚本传 `--midi-port <子串>`（或设置 `KVE_MIDI_PORT`）。
2. 设置要表征的触感灵敏度（P-45 上 Medium 是默认值；Medium 和 Hard 给出相同的 MIDI 曲线）。
   用 `--touch-setting <名称>` 记录，它会存进每个会话的元数据。只有 `dt → velocity` 曲线依赖触感设置，
   触点深度不变。
3. 测一次键行程（在指尖将要按压的位置把白键按到底，多数键盘约 10 mm）。它只在
   `kve-export --key-dip-m` 里用来把 A/B 表示成行程比例。

### 5.2 机器人

1. 网络：电脑接入机器人子网，`ping <robot-ip>`。在示教器上开启 **Remote Control**
   （e-Series：*Settings → System → Remote Control*，然后点右上角图标）。
2. 识别控制器：
   ```bash
   uv run kve-probe 192.168.1.10
   ```
   会打印型号、PolyScope 版本、力传感器原始值是否可用（全零 = 没有传感器的 CB3）和 RTDE 频率。
   `kve-identify` 的键底停止和 home 精修需要力传感器。
3. 安装指尖竖直向下，并在示教器上把 **TCP** 和 **负载** 设为指尖（*Installation → General → TCP / Payload*）。
   记录的 `tcp_z` 必须是指尖，否则深度全错。也可以给脚本传 `--tcp x y z rx ry rz`（米 / 轴角）；
   用 `kve-jog p` 检查打印的位置随移动变化是否符合预期。
4. 摆放钢琴使键盘方向平行于机器人基座的 X 或 Y 轴（交互命令 `l`/`r` 沿 `--key-axis`（默认 `+y`）
   整键平移 home；也可以现场用 `c` 标定键轴）。
5. 安全设置：普通模式即可。脚本自身限制 |ΔF| ≤ 25 N（`--force-abort-n`，`--hard` 时 80 N），
   UR 自带的工具力限制（默认 150 N）是最后一道保护。

### 5.3 第一次接触与手动按压（`kve-test`）

把指尖点动到 **刚好接触** 要测的琴键，位置在手指弹奏的地方（白键前三分之一），然后启动交互会话：

```bash
uv run kve-test interactive --robot-ip 192.168.1.10 --touch-setting medium --out data/first_touch
```

在 `kv>` 提示符里（启动时会打印完整命令列表）：

| 命令 | 用途 |
|---|---|
| `t` | 监听钢琴 5 秒：手弹这个键，必须出现音符 |
| `f` | 切换自由驱动：手动拖动机械臂，再按 `f` 关闭 |
| `h` | 把当前位姿设为 **home**（= 接触参考，深度 0） |
| `w` | 我在哪：相对 home 的位姿、TCP 力、最近的 MIDI |
| `p 0.05` | 一次 0.05 m/s 匀速按压（键上方的助跑自动计算），打印 note-on 时的力度、深度和速度、停止原因、峰值力 |
| `p 0.1` … `x 0.2` | 更快的按压（`x` = 速度控制，≈ 0.15 m/s 以上用它） |
| `k 0.3` | *kick*：慢速触键后在键内加速到 0.3 m/s（辨识实验中快速按压的方式） |
| `r` / `l` | home 右移/左移一个白键 |
| `d 0.3` / `u 0.3` | 接触参考下移/上移 0.3 mm |
| `q` | 退出；机械臂停在 home，下一个脚本可以直接从这里开始 |

第一个键上要观察的现象（数字来自 P-45，你的琴会有差别）：

* `p 0.03` → 力度约 20–30，`depth@on` 约 7–9 mm，`stop: note-on … lifted`——音符在到键底之前到达，
  机械臂随即抬起。如果 `depth@on` < 5 mm，说明 home 太深（指尖已经把键压下去了）：`u 0.5` 后重试。
  如果 10 mm 都没有音符，home 太高或按错了键：`w`，`d 0.5`。
* `k 0.25` → 力度约 60–70，没有 `!!` 警告。出现 `!! note-on while the fingertip was only x mm deep`
  说明琴键被撞飞、跑到了手指前面（冲击太硬）：0.1 m/s 以上请用 `k` 而不是 `p`/`x`。
  `!! N note-ons … key bounced` 说明琴键回弹后被再次触发。
* `|dF|max` 保持在约 15 N 以下；它随速度上升，因为机械臂在 B 之后 2 mm 就到键底了。

每次按压都存成 `trial_XXXX.npz`，这些会话之后也可以分析。

### 5.4 辨识一个键（`kve-identify`）

指尖刚好接触琴键（`kve-test` 退出时的状态，或手动点动到位）：

```bash
uv run kve-identify --robot-ip 192.168.1.10 --touch-setting medium --out data/<piano>_id_C4
```

脚本在这个键上跑四个阶段，最后打印报告（约 25 分钟）：

| 阶段 | 机器人动作 | 结果 |
|---|---|---|
| **home** | 以 2 mm/s 下降直到力传感器看到 0.4 N，以此为深度 0（`--no-refine-home` 保留点动的位姿） | 接触参考 |
| **b** | 15 次慢速匀速按压（0.02–0.12 m/s），慢速采样抬起 | 由 `t_on − t_contact = B/v + L` 得到 **B** 和 **延迟**；抬起途中的 note-off 深度 |
| **a** | *暂停探测*：慢速下到深度 d，停 0.4 s，再快速压到底；先 1 mm 粗扫，再二分到 0.1 mm | **A**：d < A 时暂停发生在计时开始之前（力度正常），d > A 时暂停落在 dt 之内（力度 < 20 或无音符） |
| **s** | 速度扫描：琴键以恒定速度被驱动通过 A→B（每个速度单独选择 kick 加速度，使目标速度恰好在 A 处达到），每个速度 2 次，不断加速度点直到覆盖力度 1…127 或到达机械臂极限；最后 3 次慢压做漂移检查 | **`dt → velocity` 拟合**、速度 → 力度表、超时值 |

`--out` 里的输出：`results_si.json`（全部 SI 单位：`A_m`、`B_m`、`latency_s`、`velocity_from_dt` = `k·dt_s^b`、
对数线性备选、`dt_to_velocity_table`、`speed_to_velocity`、带实测/外推标记的 `velocity_to_speed_full_1_127`）、
`switch_id.json`（完整分析）、`session.json`、`trial_XXXX.npz`（原始数据）以及图 `fig_summary_si.png`、
`fig_B_fit.png`、`fig_A_probe.png`、`fig_dt_velocity.png`、`fig_velocity_vs_speed.png`、`fig_velocity_model_si.png`。

**验收检查**（对照 [`examples/yamaha_p45/piano_id_C4`](examples/yamaha_p45/piano_id_C4) 里 P-45 的报告）：

* B 拟合：rms ≲ 2 ms，延迟 0–8 ms，检查行 *depth at note-on of the ≤ 0.03 m/s presses* 与 B 相差 ≈ 0.2 mm 以内。
* A：区间宽度 ≤ 0.1 mm 且 `consistent`；note-off 深度（随 B 拟合一起打印）应与 A 相差 ≈ 0.1 mm 以内——
  这是同一个触点的两个独立测量。
* 漂移检查 `b2` ≤ 0.25 mm；更大说明 home 动了（指尖松动、钢琴移位、机械臂温升）：重跑。
* 驱动按压的力度拟合 R² ≥ 0.95；剔除计数（`launched`、`compressed`、`braked before B`）相对总数很少。
  很多 `compressed` = 指尖太软。
* 覆盖的力度范围：UR5e 能达到约 0.45 m/s 的琴键平均速度，即 P-45 上力度约 90–100；之上的模型是拟合曲线
  （标记为 *extrapolated*）。可选：`--hard`（0.45–0.75 m/s，键底冲击 40–80 N）向 127 推进——刚性指尖，手放急停上。

常用选项：`--phase b|a|s` 只跑一个阶段，`--ab-from data/<piano>_id_C4/switch_id.json` 复用 A/B 再做一次扫描，
`--sweep-repeats 3` 增加重复，`--b-guess-mm` / `--a-guess-mm` 用于触点深度与 8 / 3 mm 差很多的琴
（只在测出之前使用），`--probe-low-vel` 用于暂停落在 dt 内时力度仍高于 20 的琴。

重新分析或合并同一个键的多个会话（每个合并进来的会话都用它自己的慢压重新对齐参考）：

```bash
uv run kve-analyze-switches data/<piano>_id_C4 --pool data/<piano>_velmap_C4 data/<piano>_hard_C4
```

### 5.5 沿键盘重复

对每个八度一个键重复 5.3/5.4，例如 C1…C7（`kve-test` 里 `r 7` 移动 7 个白键 = 一个八度，或手动点动），
可选再测一个黑键和同一八度的低音/高音键，检查 A、B 在整个键盘上是否一致。目录命名为 `<任意>_<音名>`
（如 `piano_id_C4`）：`kve-export` 和 `kve-sim-ball` 从目录名读音名。要测另一个触感设置的曲线，
只需用 `--ab-from` 重跑阶段 `s`（A/B 不变）。

### 5.6 导出模型

```bash
uv run kve-export data/<piano>_id_C1 data/<piano>_id_C2 ... data/<piano>_id_C7 \
    --piano "Kawai ES120" --touch-setting normal --key-dip-m 0.010 --out kawai_es120_velocity_model.json
```

JSON 里每个测过的键有 `A_m`、`B_m`、`k`、`b`、`floor_dt_s`、`latency_s` 和实测的 `speed_table`；
没测的键由仿真器插值。可对照 [`examples/yamaha_p45/p45_velocity_model.json`](examples/yamaha_p45/p45_velocity_model.json)。

### 5.7 可选：多键自动实验（`kve-run`）

`kve-run` 基于三个键示教出的键盘布局，自动在多个键上跑多种速度 *曲线*（匀速、加速、减速、两段式），
用力传感器找每个键的接触高度，以同样格式记录；`kve-analyze` 再从所有曲线一起拟合触点深度。
它用于最初的探索性实验；推荐流程是 `kve-identify`。

```bash
uv run kve-run --robot-ip <ip> --teach-layout key_layout.json                      # 自由驱动到 C2、C6、C#4
uv run kve-run --robot-ip <ip> --layout key_layout.json --keys C4 --preset quick --repeats 1 --out data/kv_check
uv run kve-run --robot-ip <ip> --layout key_layout.json --keys registers --preset full --repeats 3 --out data/kv_full
uv run kve-analyze data/kv_full
```

## 6. 仿真验证步骤

内置两个层次的验证。

### 6.1 分析流程能否还原一个已知机构？（`--sim`）

即上面的第 0 步：虚拟钢琴的 A/B/延迟/曲线已知，整个流程能把它们还原出来（A 2.97 vs 3.00 mm，
B 8.01 vs 8.00 mm，延迟 7.0 vs 6 ms + ½ 控制周期，曲线误差 0.5 个力度单位以内）。每次改动分析代码后都跑一遍。

### 6.2 导出的模型能否在 MuJoCo 里复现真琴？（`kve-sim-ball`、`kve-compare`）

一个刚性小球（mocap 球，r = 8 mm）在 88 键 MuJoCo 键盘模型（RoboPianist 几何：铰接盒状琴键、10 mm 行程、弹簧）
上，以与真实扫描相同的目标速度按压每个测过的白键前端。用仿真琴键自身的角度计算 A→B 平均速度
（与机器人上的算法完全相同），用双触点跟踪器 + 你的模型得到仿真力度。误差在相同的 *实测* 琴键速度下比较
（机械臂并不总能达到目标速度）。

```bash
# 需要：uv sync --extra sim
uv run kve-sim-ball --real-dirs data/<piano>_id_C1 ... data/<piano>_id_C7 \
    --velocity-model kawai_es120_velocity_model.json --model measured --out data/sim_vs_real
uv run kve-sim-ball --real-dirs data/<piano>_id_C1 ... data/<piano>_id_C7 --model legacy --out data/sim_vs_real   # 基线
uv run kve-compare --sim-dir data/sim_vs_real --real-dirs data/<piano>_id_C1 ... data/<piano>_id_C7
```

`--model legacy` 是原 RoboPianist 规则（距键底 0.5° 触发 note-on，velocity = qvel·127/3.5），作为基线。
输出：`sim_vs_real_<model>.json`、`fig_sim_vs_real_<model>.png`（每键曲线 + 误差直方图）、
`fig_compare_overview.png`、`fig_compare_traces_C4.png`（真实指尖 vs 仿真琴键深度随时间变化，
标出 note-on 时刻和 A/B 深度）以及 `compare_table.md`（每键每模型的 MAE / 偏差）。
没有 EGL 的机器上脚本会无头运行（`MUJOCO_GL` 自动处理）；加 `--render-note C4` 并设置 `MUJOCO_GL=egl` 可以得到按压过程的 GIF。

**预期结果**：用 P-45 模型，仿真在 115 次按压上以 MAE 2.1 个力度单位复现真实力度（legacy 规则：9.0）。
偏差集中在力度 1 的超时边缘（真琴在 0.001 m/s 之内从 1 跳到约 10）和 A/B 与相邻键不同的键上。
要在自己的仿真器上验证模型，见第 7 节。

## 7. 在自己的仿真器里使用模型

`key_velocity_exp.sim.velocity_model` 是自包含的（numpy；可选 torch 辅助函数）：

```python
import numpy as np
from key_velocity_exp.sim.velocity_model import TwoSwitchVelocityModel, TwoSwitchTracker

model = TwoSwitchVelocityModel.load("kawai_es120_velocity_model.json")   # 88 键，已插值
# 你的琴键关节的铰链角阈值：静止角 + 实测深度 / 杠杆长度
angle_a, angle_b = model.angle_thresholds(joint_range_max, rest_angle)
tracker = TwoSwitchTracker(model, joint_range_max, rest_angle)          # 每个物理子步调用一次：
on_keys, velocities, off_keys = tracker.update(t, qpos, qvel)          # 越界时刻在子步内插值
# 或者如果你自己跟踪 A/B 穿越：
vel = model.velocity(dt_seconds, key_ids)                              # 查找表（实测范围）+ 范围外幂律
```

深度是绝对值（按压点处 *静止* 键面以下的米数），所以要先量出你仿真琴键的静止下垂
（自带 MJCF 可用 `key_rest_angles()`），并保持 A→B 距离与真琴一致；在自带的键盘上 B 最终只在键底上方 0.2 mm，
因为白键在重力下下垂 1.7 mm。向量化仿真器用 `model.as_torch()` 和 `TwoSwitchVelocityModel.velocity_torch()`。

## 8. 文件与数据格式

```
key_velocity_exp/
  common.py             琴键几何、PressProfile（速度曲线 -> 深度轨迹）、TrialRecord 读写、CLI 辅助
  midi_io.py            带时间戳的 MIDI 监听器（mido/rtmidi）和仿真双触点钢琴
  ur5_io.py             ur_rtde 驱动（servoL / speedL 按压、键底停止、力/深度中止）和运动学仿真器
  real_test.py          手动按压（kve-test）；Session 类被 kve-identify 复用
  identify_switches.py  辨识实验（kve-identify）
  analyze_switches.py   其分析：B/延迟拟合、A 探测、dt->velocity 拟合、SI 输出、图
  export_piano_model.py 各键结果 -> 模型 JSON（kve-export）
  run_experiment.py / analyze.py / layout.py   多键自动实验
  probe_robot.py / jog_robot.py                控制器识别、慢速 TCP 移动
  sim/velocity_model.py 供仿真器使用的双触点力度模型 + 跟踪器（numpy / torch）
  sim/key_midi.py       仿真键盘的 MIDI 生成（实测模型或 legacy 规则）
  sim/build_piano_mjcf.py, sim/piano_constants.py   88 键 MuJoCo 键盘
  sim/sim_ball_press.py, sim/compare_sim_real.py    刚性小球验证与对比
examples/yamaha_p45/    P-45 结果：piano_id_C1..C7（results_si.json、switch_id.json、图）、p45_velocity_model.json、sim_vs_real/
docs/p45_case_study.md  P-45 实验的工作记录：机械臂极限、琴键被撞飞、榔头飞行、最终数字
```

**每次按压**（`trial_XXXX.npz`）：`robot_samples` [N × 33]，列名见 `common.ROBOT_SAMPLE_FIELDS`
（单调时钟、RTDE 时间、指令 z、TCP 位姿、TCP 速度、TCP 力、q、qd）；`midi_events` [M × 4]
（时间、on/off、音符、力度）；JSON `meta`（曲线、home、阶段、目标速度、停止原因……）。
`TrialRecord.load()` 读取；`rec.depth()` 是指尖相对 home 的深度。

**每个会话**：`session.json`、`summary.csv`、`switch_id.json`、`results_si.json`、图。

## 9. 排错与坑

在 P-45 上花掉时间的问题（细节和数字见 [docs/p45_case_study.md](docs/p45_case_study.md)）：

* **限制力度范围的是机械臂，不是钢琴。** `servoL` 在 10 mm 行程内跟不上 ≈ 0.15 m/s 以上的速度；
  速度控制（`speedL`）能跟上指令但仍需要助跑。只相信 *实测* 速度（打印为 `v(3-7mm)` / `v_AB`）；分析从不使用指令速度。
* **琴键被撞飞。** 刚性指尖以 ≥ 0.25 m/s 撞击轻琴键会把键打到手指前面：指尖只有 3–4 mm 深时 note-on 就到了，
  随后回弹、重复触发。因此快速按压都是慢速触键后在键内加速（`kick`），并在 note-on 时停止下降。
* **榔头飞行。** 键内加速度超过 ≈ 10 m/s² 会把榔头甩到琴键前面：钢琴看到的 dt 比手指的短。
  这类按压被标记为 `launched` 并剔除；扫描时选择加速度使目标速度恰好 *在 A 处* 达到。
* **提前刹车。** 0.4 m/s 运动的 GHS 琴键自身就有 5–6 N 反力；力停止必须只在 B 以下启用
  （测出 B 后 `set_limits_from_b` 自动设置），否则机械臂在第二个触点之前就刹车了。
* **软指尖** 受载压缩：TCP ≠ 琴键。note-on 时指尖比 B 深出超过 延迟·速度 的按压被标记为 `compressed`。
* **home 漂移**：结束时用慢压重测参考（`b2`）；> 0.25 mm 说明指尖或钢琴动了。
* **没有音符 / 按错键**：在 `kve-test` 里 `t`；检查 `--midi-port`、指尖下的键（`--note` 只用于标注会话）、指尖没有卡在两键之间。
* **`moveL failed` / 保护性停止**：在示教器上清除；脚本会自动重新上传控制脚本（`ensure_running`）。
* **无头机器上的 MuJoCo**：不给 `--render-note` 时 `kve-sim-ball` 关闭渲染；要 GIF 需安装 EGL 并设置 `MUJOCO_GL=egl`。

## 10. 参考结果：Yamaha P-45

Medium 触感（Hard 曲线相同），刚性指尖，UR5e，每个八度一个键。完整报告和图见 [`examples/yamaha_p45/`](examples/yamaha_p45/)。

| 键 | A [mm] | B [mm] | velocity = k·dt_s^b | 超时 dt | 延迟 |
|---|---|---|---|---|---|
| C1 | 5.02 | 8.07 | 6.57·dt^−0.50 | 0.19 s | 3.9 ms |
| C2 | 4.91 | 8.43 | 6.09·dt^−0.53 | 0.24 s | 3.4 ms |
| C3 | 4.72 | 8.16 | 5.44·dt^−0.57 | 0.25 s | 4.2 ms |
| C4 | 4.84 | 7.95 | 3.82·dt^−0.65 | 0.31 s | 3.4 ms |
| C5 | 4.91 | 8.20 | 5.74·dt^−0.55 | 0.27 s | 5.1 ms |
| C6 | 4.91 | 8.23 | 5.27·dt^−0.57 | 0.28 s | 3.7 ms |
| C7 | 3.71 | 6.90 | 4.26·dt^−0.64 | 0.23 s | 4.0 ms |

仿真刚性小球按压 vs 真琴（相同琴键速度下，115 次按压）：

| | C1 | C2 | C3 | C4 | C5 | C6 | C7 | 全部 |
|---|---|---|---|---|---|---|---|---|
| 实测双触点模型 MAE | 2.8 | 1.4 | 2.1 | 1.4 | 1.4 | 2.8 | 3.8 | **2.1**（偏差 −1.1） |
| legacy 阈值规则 MAE | 7.9 | 8.6 | 7.3 | 8.2 | 10.4 | 8.8 | 12.5 | **9.0**（偏差 −5.5） |

在 P-45 上达到力度 127 需要 A→B 平均键速约 0.7 m/s（外推值；机械臂达到了 0.45 m/s = 力度 98）。
