#!/usr/bin/env python
"""Identify the UR controller: model (UR5 CB3 vs UR5e), PolyScope version, F/T sensor, mode, RTDE rate.

kve-probe 192.168.1.10          # or: export UR_ROBOT_IP=192.168.1.10; kve-probe
"""

from __future__ import annotations

import argparse
import os
import socket
import time


def tcp_open(ip: str, port: int, timeout: float = 1.0) -> bool:
    s = socket.socket()
    s.settimeout(timeout)
    try:
        s.connect((ip, port))
        return True
    except OSError:
        return False
    finally:
        s.close()


def probe(ip: str) -> None:
    ports = {29999: "dashboard", 30001: "primary", 30002: "secondary", 30003: "realtime", 30004: "RTDE"}
    print(f"[probe] {ip}")
    for p, name in ports.items():
        print(f"   port {p:5d} ({name:9s}): {'open' if tcp_open(ip, p) else 'closed'}")
    if not tcp_open(ip, 29999):
        print("[probe] dashboard not reachable - robot off, wrong IP, or no route (check `ip route` / the pendant's network settings)")
        return

    import dashboard_client

    db = dashboard_client.DashboardClient(ip)
    db.connect()
    model = db.getRobotModel()
    version = db.polyscopeVersion()
    print(f"   robot model     : {model}")
    print(f"   PolyScope       : {version}")
    try:
        print(f"   serial          : {db.getSerialNumber()}")
    except Exception:
        pass
    print(f"   robot mode      : {db.robotmode()}")
    print(f"   safety status   : {db.safetystatus()}")
    try:
        print(f"   remote control  : {db.isInRemoteControl()}")
    except Exception as e:  # CB3 has no remote-control concept
        print(f"   remote control  : n/a ({e})")
    major = None
    for tok in version.replace("URSoftware", "").split():
        if tok[0].isdigit():
            major = int(tok.split(".")[0])
            break
    if major is not None:
        family = "e-Series (UR5e-class: built-in 6-axis F/T sensor, 500 Hz RTDE)" if major >= 5 else "CB3 (UR5: no F/T sensor, TCP force estimated from joint currents, 125 Hz RTDE)"
        print(f"   family          : {family}")
    db.disconnect()

    import rtde_receive

    r = rtde_receive.RTDEReceiveInterface(ip)
    time.sleep(0.2)
    pose = r.getActualTCPPose()
    print(f"   TCP pose        : {[round(v, 4) for v in pose]}")
    print(f"   TCP force (est.): {[round(v, 2) for v in r.getActualTCPForce()]}")
    try:
        raw = r.getFtRawWrench()
        print(f"   F/T raw wrench  : {[round(v, 2) for v in raw]}  (all zero => no F/T sensor / CB3)")
    except Exception as e:
        print(f"   F/T raw wrench  : unavailable ({e})")
    t0 = time.time()
    n = 0
    t_prev = r.getTimestamp()
    while time.time() - t0 < 1.0:
        t = r.getTimestamp()
        if t != t_prev:
            n += 1
            t_prev = t
        time.sleep(0.0005)
    print(f"   RTDE rate       : ~{n} Hz")
    r.disconnect()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("ip", nargs="?", default=os.environ.get("UR_ROBOT_IP"), help="UR controller IP (default: $UR_ROBOT_IP)")
    a = p.parse_args()
    if not a.ip:
        p.error("give the robot IP as an argument or set UR_ROBOT_IP")
    probe(a.ip)


if __name__ == "__main__":
    main()
