#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class Proc:
    name: str
    p: subprocess.Popen
    log: Path


def popen_pg(name: str, cmd: list[str], cwd: Optional[Path], log: Path, env: Optional[dict] = None) -> Proc:
    log.parent.mkdir(parents=True, exist_ok=True)
    f = open(log, "w", buffering=1, encoding="utf-8")
    p = subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd else None,
        stdout=f,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        preexec_fn=os.setsid if os.name != "nt" else None,  # process group
    )
    return Proc(name, p, log)


def kill_pg(proc: Proc, grace_s: float = 5.0) -> None:
    if proc.p.poll() is not None:
        return
    try:
        if os.name != "nt":
            os.killpg(os.getpgid(proc.p.pid), signal.SIGTERM)
        else:
            proc.p.terminate()
    except Exception:
        pass

    t0 = time.time()
    while time.time() - t0 < grace_s:
        if proc.p.poll() is not None:
            return
        time.sleep(0.1)

    try:
        if os.name != "nt":
            os.killpg(os.getpgid(proc.p.pid), signal.SIGKILL)
        else:
            proc.p.kill()
    except Exception:
        pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--px4-dir", type=Path, required=True, help="PX4-Autopilot repo path")
    ap.add_argument("--world", default="default", help="Gazebo world name (e.g., default, windy, baylands, ...)")
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--outdir", type=Path, default=Path("./px4_runs"))
    ap.add_argument("--max-run-s", type=float, default=60.0)

    # Gazebo launcher script (simulation-gazebo)
    ap.add_argument("--simulation-gazebo", type=Path, required=True, help="Path to simulation-gazebo script")

    # PX4 airframe autostart id for gz_x500
    ap.add_argument("--autostart", type=int, default=4001, help="PX4_SYS_AUTOSTART for x500 (default 4001)")
    ap.add_argument("--model", default="gz_x500", help="PX4_SIM_MODEL (default gz_x500)")
    ap.add_argument("--headless", action="store_true")
    args = ap.parse_args()

    px4_bin = args.px4_dir / "build/px4_sitl_default/bin/px4"
    if not px4_bin.exists():
        raise SystemExit(f"PX4 binary not found: {px4_bin}\nBuild first: `cd {args.px4_dir} && make px4_sitl`")

    args.outdir.mkdir(parents=True, exist_ok=True)

    stop = {"flag": False}

    def _sig(_s, _f):
        stop["flag"] = True

    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)

    for i in range(args.runs):
        if stop["flag"]:
            break

        run_dir = args.outdir / f"run_{i:03d}"
        run_dir.mkdir(parents=True, exist_ok=True)

        gz_log = run_dir / "gz.log"
        px4_log = run_dir / "px4.log"

        # 1) Start gz-server via simulation-gazebo (standalone mode helper)
        # docs: python3 simulation-gazebo --world <name> :contentReference[oaicite:4]{index=4}
        gz_cmd = ["python3", str(args.simulation_gazebo), "--world", args.world]
        if args.headless:
            # simulation-gazebo 자체의 headless 옵션은 버전에 따라 다를 수 있어
            # (GUI를 안 띄우고 싶으면 일반적으로 gz sim 쪽에서 -s/--server-only를 쓰는데,
            #  여기서는 문서 기반으로 최소만 유지)
            pass

        gz = popen_pg("gz", gz_cmd, cwd=run_dir, log=gz_log)

        # 2) Start PX4 binary and let it attach to detected gz-server
        # docs example: PX4_GZ_STANDALONE=1 PX4_SYS_AUTOSTART=4001 PX4_SIM_MODEL=gz_x500 PX4_GZ_WORLD=windy ./build/.../px4 :contentReference[oaicite:5]{index=5}
        env = os.environ.copy()
        env["PX4_GZ_STANDALONE"] = "1"
        env["PX4_SYS_AUTOSTART"] = str(args.autostart)
        env["PX4_SIM_MODEL"] = args.model
        env["PX4_GZ_WORLD"] = args.world

        px4 = popen_pg("px4", [str(px4_bin)], cwd=args.px4_dir, log=px4_log, env=env)

        # run loop
        t0 = time.time()
        rc = 0
        while True:
            if stop["flag"]:
                rc = 130
                break
            if px4.p.poll() is not None:
                rc = px4.p.returncode or 0
                break
            if gz.p.poll() is not None:
                rc = gz.p.returncode or 0
                break
            if time.time() - t0 > args.max_run_s:
                rc = 0
                break
            time.sleep(0.2)

        kill_pg(px4)
        kill_pg(gz)

        print(f"[run {i}] done rc={rc} logs: {run_dir}")
        time.sleep(2.0)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())