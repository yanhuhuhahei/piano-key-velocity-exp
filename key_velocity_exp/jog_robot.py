#!/usr/bin/env python
"""Read the UR TCP pose and move it by small, slow steps (moveL). Handy for positioning the fingertip
over a key when freedrive is too coarse, and for checking the TCP offset.

  kve-jog --robot-ip 192.168.1.10                 # interactive
  kve-jog --robot-ip 192.168.1.10 dz -0.01        # one relative move: z down 1 cm, then exit

Commands (units: metres; orientation is kept unless you give all 6 values):
  dx / dy / dz <m>          relative move along one base axis, e.g.  dz -0.01
  d <dx> <dy> <dz>          relative move in x y z
  xyz <x> <y> <z>           absolute TCP position, keep current orientation
  pose <x> <y> <z> <rx> <ry> <rz>   absolute 6-D TCP pose (axis-angle)
  home                      go back to the pose read at start-up
  p                         print current pose
  q                         quit
Every move is clamped to --max-step metres of translation and runs at --speed m/s.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np


class Jogger:
    def __init__(self, ip: str, speed: float, accel: float, max_step: float):
        from rtde_control import RTDEControlInterface
        from rtde_receive import RTDEReceiveInterface

        self.rtde_r = RTDEReceiveInterface(ip)
        self.rtde_c = RTDEControlInterface(ip)
        self.speed, self.accel, self.max_step = speed, accel, max_step

    def current_pose(self) -> np.ndarray:
        return np.asarray(self.rtde_r.getActualTCPPose(), dtype=float)

    def print_pose(self, tag: str = "TCP") -> np.ndarray:
        q = self.rtde_r.getActualQ()
        pose = self.current_pose()
        print(f"{tag} joints (rad):        ", np.round(q, 4).tolist())
        print(f"{tag} position (m):        ", np.round(pose[:3], 5).tolist())
        print(f"{tag} orientation (axang): ", np.round(pose[3:], 5).tolist())
        return pose

    def move_to(self, target: np.ndarray) -> None:
        cur = self.current_pose()
        step = np.linalg.norm(target[:3] - cur[:3])
        if step > self.max_step:
            print(f"refused: translation {step * 1e3:.1f} mm > --max-step {self.max_step * 1e3:.0f} mm")
            return
        print(f"moving {np.round((target[:3] - cur[:3]) * 1e3, 2).tolist()} mm -> {np.round(target[:3], 5).tolist()} at {self.speed} m/s")
        ok = self.rtde_c.moveL(target.tolist(), self.speed, self.accel)
        if not ok:
            print("moveL failed (protective stop? not in Remote Control?)")
        self.print_pose("  now")

    def handle(self, cmd: str, args: list[str], home: np.ndarray) -> bool:
        cur = self.current_pose()
        if cmd == "q":
            return False
        if cmd == "p":
            self.print_pose()
        elif cmd in ("dx", "dy", "dz"):
            t = cur.copy()
            t["xyz".index(cmd[1])] += float(args[0])
            self.move_to(t)
        elif cmd == "d":
            t = cur.copy()
            t[:3] += np.array(list(map(float, args[:3])))
            self.move_to(t)
        elif cmd == "xyz":
            t = cur.copy()
            t[:3] = list(map(float, args[:3]))
            self.move_to(t)
        elif cmd == "pose":
            self.move_to(np.array(list(map(float, args[:6]))))
        elif cmd == "home":
            self.move_to(home)
        else:
            print("unknown command; see kve-jog --help")
        return True

    def close(self) -> None:
        self.rtde_c.stopScript()
        self.rtde_c.disconnect()
        self.rtde_r.disconnect()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", nargs="*", help="one-shot command, e.g. dz -0.01 (omit for the interactive prompt)")
    p.add_argument("--robot-ip", default=os.environ.get("UR_ROBOT_IP"), help="UR controller IP (or set UR_ROBOT_IP)")
    p.add_argument("--speed", type=float, default=0.02, help="moveL speed m/s (slow: the fingertip may be on the keyboard)")
    p.add_argument("--accel", type=float, default=0.2, help="moveL acceleration m/s^2")
    p.add_argument("--max-step", type=float, default=0.05, help="refuse translations larger than this in one command [m]")
    a = p.parse_args()
    if not a.robot_ip:
        p.error("--robot-ip is required (or set UR_ROBOT_IP)")
    jog = Jogger(a.robot_ip, a.speed, a.accel, a.max_step)
    home = jog.print_pose("start")
    try:
        if a.command:
            jog.handle(a.command[0], a.command[1:], home)
            return
        while True:
            try:
                line = input("ur> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not line:
                continue
            cmd, *args = line.split()
            try:
                if not jog.handle(cmd, args, home):
                    break
            except (ValueError, IndexError) as e:
                print(f"bad arguments: {e}")
    finally:
        jog.close()


if __name__ == "__main__":
    sys.exit(main())
