#!/usr/bin/env python3
"""
Multi-run launcher for PX4 SITL + Gazebo GZ.

What this script does
---------------------
- Scans run_xxx directories under a run-root
- For each scenario:
    1) Loads scenario.yaml
    2) Starts Gazebo separately
    3) Starts PX4 SITL using an already-built PX4 binary
    4) Starts a pymavlink commander thread
    5) Monitors timeout / failures / completion
    6) Cleans up process trees

Design notes
------------
- Gazebo is launched in standalone mode.
- PX4 binary is assumed to be already built.
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, TextIO, Tuple
import yaml
import math
import shutil
import re

_THIS_FILE = Path(__file__).resolve()
_TOOLS_DIR = _THIS_FILE.parents[1]
_COMMANDER_DIR = _TOOLS_DIR / "commander"
_SCENARIO_DIR = _TOOLS_DIR / "scenario"
_QGC_DIR = _TOOLS_DIR / "QGC" / "squashfs-root"
_FAKE_GCS = _COMMANDER_DIR / "fake_gcs_heartbeat.py"

if str(_COMMANDER_DIR) not in sys.path:
    sys.path.insert(0, str(_COMMANDER_DIR))

if str(_SCENARIO_DIR) not in sys.path:
    sys.path.insert(0, str(_SCENARIO_DIR))

from vehicle_model_builder import VehicleConfig, generate_vehicle_model
from pymavlink_px4_commander import PX4MissionRunner
from px4_gz_standalone import (
    prepare_simulation, gazebo_command, gazebo_gui_command, spawn_model,
    px4_command, px4_environment, stop_process,
)


@dataclass
class PX4ScenarioConfig:
    # sim instance parameters
    px4_dir: str = "${FLIGHTSTACK_SIM_ROOT}/ap/px4"
    instance: int = 0
    mavlink_url: str = "udp://127.0.0.1:14540"
    startup_delay_s: float = 0.0
    max_run_s: float = 60.0
    max_retries: int = 3

    # sim setup
    vehicle: int = 4001
    frame: str = "gz_x500"
    world: str = "default"
    location: str = "Purdue"
    scenario_name: str = "unnamed"
    gcs_outport: int = 14550
    vehicle_config: VehicleConfig | None = None


@dataclass
class ProcHandle:
    """Container holding a running process and its log file info."""
    name: str
    proc: subprocess.Popen
    log_path: Path
    log_file: TextIO


def _popen(
    name: str,
    cmd: list[str],
    cwd: Optional[Path],
    log_path: Path,
    env: Optional[dict[str, str]] = None,
    verbose: bool = False,
) -> ProcHandle:
    """
    Start a subprocess and redirect stdout/stderr into a log file.

    - The process is started in a new process group (POSIX) so we can terminate
      the whole tree (parent + children) later.
    - Logs are line-buffered for real-time tailing (tail -f).
    """
    print(f"[LAUNCH] {name}: {' '.join(cmd)}")
    print(f"[CWD]    {name}: {cwd if cwd else os.getcwd()}")
    print(f"[LOG]    {name}: {log_path}")

    # Print environmental variables
    if verbose and env is not None:
        print(f"[ENV]    {name}:")
        for k in sorted(env.keys()):
            if k.startswith("PX4") or k.startswith("GZ") or k in ("PATH", "HOME"):
                print(f"         {k}={env[k]}")

    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Line-buffered text logs (buffering=1 works with text=True).
    f = open(log_path, "w", buffering=1, encoding="utf-8")

    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd) if cwd else None,
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            start_new_session=True,
        )
    except Exception:
        f.close()
        raise

    return ProcHandle(name=name, proc=proc, log_path=log_path, log_file=f)


def _finalize_proc(ph: Optional[ProcHandle], grace_s: float = 5.0) -> None:
    """Best-effort: ensure process is dead and log file is closed."""

    if ph is None:
        return

    try:
        stop_process(ph.proc)
    except Exception:
        pass

    try:
        ph.proc.wait(timeout=grace_s)
    except Exception:
        pass

    try:
        ph.log_file.flush()
    except Exception:
        pass

    try:
        ph.log_file.close()
    except Exception:
        pass


def _read_location_from_txt(txt_path: Path, name: str) -> Tuple[float, float, float, float]:
    """
    Parse a line like:
      Purdue=40.41176161953683,-86.93352081596879,0,0

    Returns:
      lat, lon, alt, heading_deg
    """
    if not txt_path.is_file():
        raise FileNotFoundError(f"locations file not found: {txt_path}")

    for line in txt_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        key, sep, value = line.partition("=")
        if sep and key.strip() == name:
            parts = [p.strip() for p in value.split(",")]
            if len(parts) < 4:
                raise ValueError(f"Invalid location line for {name}: {line}")

            lat = float(parts[0])
            lon = float(parts[1])
            alt = float(parts[2])
            heading_deg = float(parts[3])
            return lat, lon, alt, heading_deg

    raise KeyError(f"Location '{name}' not found in {txt_path}")


def _run_qgc_cmd(out_port: int) -> list[str]:
    """
    Launch QGroundControl (GCS).
    QGC automatically listens on UDP port 14550.
    """
    qgc_bin = _QGC_DIR / "AppRun"
    return [str(qgc_bin)]


def _run_fake_gcs_cmd(connect_url: str = "udpout:127.0.0.1:14550", rate_hz: float = 10.0) -> list[str]:
    """
    Launch a lightweight fake GCS heartbeat sender.

    Parameters
    ----------
    connect_url : str
        pymavlink connection URL used to send GCS heartbeat to PX4.
        Usually udpout:127.0.0.1:14550 for local PX4 SITL.

    rate_hz : float
        Heartbeat rate in Hz.
    """
    return [
        sys.executable,
        str(_FAKE_GCS),
        "--connect", connect_url,
        "--rate-hz", str(rate_hz),
        "--source-system", "255",
        "--source-component", "190",
    ]


def _finalize_runner(runner, timeout: float = 2.0) -> None:
    """
    Best-effort shutdown for a mission runner thread.
    """
    if runner is None:
        return

    try:
        runner.request_stop()
    except Exception:
        pass


def run_once(
    px4_dir: Path,
    instance: int,
    scenario_path: Path,
    logs_dir: Path,
    gcs_outport: int,
    mavlink_url: str,
    startup_delay_s: float,
    max_run_s: float,
    stop: dict,
    current: dict,
    vehicle: str,
    frame: str,
    world: str,
    location: str,
    headless: bool,
    verbose: bool,
    vehicle_config: VehicleConfig | None = None,
) -> int:
    """Start Gazebo, spawn the selected SDF, and attach an independent PX4.

    Each attempt has a private Gazebo partition. PX4 parameters and raw ULogs
    live in logs_dir/rootfs, separate from the firmware build's runtime state.
    """
    runner = None
    try:
        if not frame.startswith("gz_"):
            raise ValueError(f"Expected a gz_<model> frame, got {frame!r}")
        sim = prepare_simulation(
            px4_dir, world, frame[3:], instance,
            partition=f"fire-px4-{os.getpid()}-{time.monotonic_ns()}",
        )
        if vehicle_config is not None:
            if sim.model_name != vehicle_config.model_name:
                raise ValueError("PX4 frame model must match common.vehicle.model_name")
            sim.model_path = generate_vehicle_model(vehicle_config, sim.model_path, scenario_path.parent)
        cmd = px4_command(px4_dir, instance, logs_dir / "rootfs", vehicle)
        lat, lon, alt, heading_deg = _read_location_from_txt(
            _THIS_FILE.parent / "locations.txt", location,
        )
        print(f"[SIM] world={sim.world_path} model={sim.model_path}")
        print(f"[SIM] entity={sim.entity_name} partition={sim.env['GZ_PARTITION']}")
        current["gz"] = _popen(
            "gazebo", gazebo_command(sim, headless, verbose), cwd=logs_dir,
            log_path=logs_dir / "gazebo.log", env=sim.env, verbose=verbose,
        )
        spawn_model(sim, (0, 0, 0.25, 0, 0, math.pi / 2 - math.radians(heading_deg)),
                    process=current["gz"].proc, cancelled=lambda: stop["flag"])
        if stop["flag"]:
            return 130

        if not headless:
            current["gz_gui"] = _popen(
                "gazebo_gui", gazebo_gui_command(verbose), cwd=logs_dir,
                log_path=logs_dir / "gazebo_gui.log", env=sim.env, verbose=verbose,
            )

        if headless:
            gcs_cmd = _run_fake_gcs_cmd(f"udp:127.0.0.1:{gcs_outport}")
        else:
            gcs_cmd = _run_qgc_cmd(gcs_outport)
        current["gcs"] = _popen(
            "gcs_fake" if headless else "gcs_qgc", gcs_cmd, cwd=logs_dir,
            log_path=logs_dir / "gcs.log",
        )
        env = px4_environment(sim, vehicle)
        env.update(PX4_HOME_LAT=str(lat), PX4_HOME_LON=str(lon), PX4_HOME_ALT=str(alt))
        current["sitl"] = _popen(
            "sitl", cmd, cwd=logs_dir, log_path=logs_dir / "sitl.log", env=env, verbose=verbose,
        )
        time.sleep(startup_delay_s)
        if stop["flag"]:
            return 130
        runner = PX4MissionRunner(scenario_path=scenario_path)
        runner.start()
        started = time.monotonic()
        while not stop["flag"]:
            for name in ("gz", "gz_gui", "sitl", "gcs"):
                handle = current.get(name)
                if handle is not None and handle.proc.poll() is not None:
                    raise RuntimeError(f"{name} exited unexpectedly; see {handle.log_path}")
            if time.monotonic() - started > max_run_s:
                print(f"[TIMEOUT] exceeded {max_run_s:.1f}s")
                return 124
            status = runner.get_status()
            print(f"[RUNNER] state={status.state.name} done={status.done} "
                  f"success={status.success} msg={status.message}")
            if runner.is_done():
                if not status.success:
                    print(f"[RUNNER] error: {status.error}")
                return 0 if status.success else 1
            time.sleep(1)
        return 130
    except InterruptedError:
        return 130
    except Exception as exc:
        print(f"[ERROR] Standalone simulation failed: {exc}")
        return 1
    finally:
        _finalize_runner(runner)
        for name in ("sitl", "gz_gui", "gz", "gcs"):
            _finalize_proc(current.get(name))
            current[name] = None


def _collect_px4_logs(run_dir: Path):
    """
    Extract PX4 ULog file path from sitl.log and move it to run_dir.

    This method parses the PX4 SITL stdout/stderr log (sitl.log)
    to find the exact `.ulg` file generated during the run,
    ensuring deterministic and race-free log collection.

    Parameters
    ----------
    run_dir : Path
        Run-specific log directory containing px4_logs folder and corresponding sitl.log
        (for example, `data/run_000`).

    Returns
    -------
    bool
        True if a .ulg file was successfully collected, False otherwise.
    """
    dst_root = run_dir
    dst_root.mkdir(parents=True, exist_ok=True)

    sitl_log = dst_root / "sitl.log"

    if not sitl_log.exists():
        print(f"[WARN] sitl.log not found: {sitl_log}")
        return False
    
    text = sitl_log.read_text(errors="ignore")
    matches = re.findall(r"\./log/[^\s]+\.ulg", text)

    if not matches:
        print(f"[WARN] No .ulg path found in {sitl_log}")
        return False

    src_root = run_dir / "rootfs"

    if not src_root.exists():
        print("[WARN] No PX4 log directory found")
        return False

    rel_path = Path(matches[-1].replace("./", ""))
    src_ulg_file = src_root / rel_path

    if not src_ulg_file.exists():
        print(f"[WARN] .ulg file not found: {src_ulg_file}")
        return False

    dst_ulg_file = dst_root / rel_path.name
    shutil.move(str(src_ulg_file), dst_ulg_file)

    print(f"[INFO] PX4 log moved: {dst_ulg_file}")
    return True


def _cleanup_failed_ulg(run_dir: Path):
    raw_dir = run_dir
    if not raw_dir.exists():
        return

    ulg_files = list(raw_dir.glob("*.ulg"))

    if not ulg_files:
        return

    for f in ulg_files:
        try:
            f.unlink()
            print(f"[CLEANUP] removed {f}")
        except Exception as e:
            print(f"[WARN] failed to remove {f}: {e}")


def _load_scenario_yaml_px4(run_dir: Path) -> PX4ScenarioConfig:
    """
    Load scenario.yaml from run_dir and extract PX4-specific configuration.

    Supported keys in scenario.yaml:
      px4_dir: ${FLIGHTSTACK_SIM_ROOT}/ap/px4
      instance: 0
      vehicle: 4001
      frame: gz_x500
      world: default
      location: Purdue
      gcs_outport: 14550
      connect_url (mavlink): udp:127.0.0.1:14540
    """
    scenario_path = run_dir / "scenario.yaml"
    if not scenario_path.exists():
        raise FileNotFoundError(f"Missing scenario.yaml: {scenario_path}")

    data: dict[str, Any] = yaml.safe_load(scenario_path.read_text(encoding="utf-8")) or {}

    # The scenario generator stores simulator settings under autopilots.px4.
    sim = data.get("autopilots", {}).get("px4", {}).get("sim", {})
    scenario = data.get("common", {}).get("scenario", {})
    mavlink = data.get("autopilots",{}).get("px4",{}).get("mavlink",{})

    cfg = PX4ScenarioConfig(
        px4_dir=Path(os.path.expandvars(str(sim.get("px4_dir", PX4ScenarioConfig.px4_dir)))).resolve(),
        instance=int(sim.get("instance", PX4ScenarioConfig.instance)),
        vehicle=str(sim.get("vehicle", PX4ScenarioConfig.vehicle)),
        frame=str(sim.get("frame", PX4ScenarioConfig.frame)),
        world=str(sim.get("world", PX4ScenarioConfig.world)),
        vehicle_config=VehicleConfig.from_common(data.get("common", {})),
        location=str(sim.get("location", PX4ScenarioConfig.location)),
        scenario_name=str(scenario.get("name", PX4ScenarioConfig.scenario_name)),
        gcs_outport=str(sim.get("qgc_outport", PX4ScenarioConfig.gcs_outport)),
        mavlink_url=str(mavlink.get("connect_url", PX4ScenarioConfig.mavlink_url)),
    )
    return cfg


def _iter_run_dirs(data_root: Path) -> list[Path]:
    """
    Find ./data/run_* directories, sorted by name.
    """
    if not data_root.exists():
        return []
    return sorted([p for p in data_root.iterdir() if p.is_dir() and p.name.startswith("run_")])


def _prepare_run_dir(run_dir: Path, force: bool) -> bool:
    """
    Decide whether to skip or run.
    
    Returns:
        True  -> skip
        False -> run
    """
    logs_dir = run_dir / "px4_logs"
    ulg_files = list(logs_dir.glob("*.ulg")) if logs_dir.exists() else []

    # case 1: force → always clean up and run
    if force:
        if (run_dir / "px4_logs").exists():
            print(f"[CLEAN] Removing existing logs in {run_dir}")
            shutil.rmtree(run_dir / "px4_logs")
        return False

    # case 2: valid log exists → skip
    if logs_dir.exists() and len(ulg_files) > 0:
        return True

    # case 3: no log → run
    return False


def _apply_cli_overrides(cfg: PX4ScenarioConfig, args: argparse.Namespace) -> PX4ScenarioConfig:
    cfg = copy.deepcopy(cfg)
    cfg.startup_delay_s = args.startup_delay_s
    cfg.max_run_s = args.max_run_s
    cfg.max_retries = args.max_retries

    # Ensure that fake GCS heartbeat will use UDP 14550 separately.
    if args.headless:
        cfg.gcs_outport = 14550

    return cfg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", type=Path, default=Path("./data/sitl_logs"), help="Root folder containing run_xxx/scenario.yaml")
    ap.add_argument("--force", action="store_true", help="Re-run even if px4_logs already exist")
    ap.add_argument("--startup-delay-s", type=float, default=5.0)
    ap.add_argument("--max-run-s", type=float, default=60.0)
    ap.add_argument("--max-retries", type=int, default=3, help="Number of retries for failed runs (0 for no retries)")
    ap.add_argument("--headless", action="store_true", help="Run Gazebo/QGC in headless mode (for batch SITL in CLI modes)")
    ap.add_argument("--verbose", action="store_true", help="Verbose logging of commands and environment variables")
    args = ap.parse_args()
    if args.max_retries < 0:
        ap.error("--max-retries must be non-negative")

    # Access to the resolved run root path
    data_root = args.run_root.resolve()

    stop = {"flag": False}
    current: dict[str, Optional[ProcHandle]] = {
        "gz": None,
        "gz_gui": None,
        "gcs": None,
        "sitl": None,
    }

    def _sig(_signum, _frame):
        stop["flag"] = True

    # Terminate on Ctrl+C or SIGTERM
    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)

    # Iterations for run directories
    run_dirs = _iter_run_dirs(data_root)
    if not run_dirs:
        print(f"No run_xxx directories found under: {data_root}")
        return 2

    overall_rc = 0

    for run_dir in run_dirs:
        if stop["flag"]:
            print("Interrupted. Exiting.")
            return 130

        if _prepare_run_dir(run_dir, force=args.force):
            print(f"[SKIP] {run_dir} (px4_logs exists and contains .ulg)")
            continue

        # Load scenario.yaml
        try:
            cfg = _load_scenario_yaml_px4(run_dir)
        except Exception as e:
            print(f"[ERROR] {run_dir}: failed to load scenario.yaml: {e}")
            overall_rc = 2
            continue

        # Create output directory for this run
        logs_root = run_dir / "px4_logs"
        logs_root.mkdir(parents=True, exist_ok=True)

        # Get scenario path
        scenario_path = run_dir / "scenario.yaml"

        # Set other configurations and overrides if headless mode is enabled
        print(f"[DEBUG] cfg.px4_dir={cfg.px4_dir}")
        cfg = _apply_cli_overrides(cfg, args)

        # Print configuration info
        print(f"\n---- {run_dir.name}: RUN {cfg.scenario_name} Scenario ----")
        print(f"  px4_dir={cfg.px4_dir} vehicle={cfg.vehicle}")
        print(f"  world={cfg.world} location={cfg.location}")
        print(f"  instance={cfg.instance} gcs_outport={cfg.gcs_outport}")
        print(f"  startup_delay={cfg.startup_delay_s} max_run_s={cfg.max_run_s} max_retries={cfg.max_retries}")
        print(f"  logs_root={logs_root} scenario_path={scenario_path} mavlink_url={cfg.mavlink_url}")

        # Execute the scenario
        rc = 1
        attempts = cfg.max_retries + 1
        for attempt in range(1, attempts + 1):

            if stop["flag"]:
                print("Interrupted. Exiting.")
                return 130
            
            print(f"[RUN] Attempt {attempt}/{attempts} for {run_dir.name}")
            rc = run_once(
                px4_dir=cfg.px4_dir,
                instance=cfg.instance,
                scenario_path=scenario_path,
                logs_dir=logs_root,
                gcs_outport=cfg.gcs_outport,
                mavlink_url=cfg.mavlink_url,
                startup_delay_s=cfg.startup_delay_s,
                max_run_s=cfg.max_run_s,
                stop=stop,
                current=current,
                vehicle=cfg.vehicle,
                frame=cfg.frame,
                world=cfg.world,
                location=cfg.location,
                headless=args.headless,
                verbose=args.verbose,
                vehicle_config=cfg.vehicle_config,
            )

            # Collect the ULog from this run's private rootfs after shutdown.
            success_log = _collect_px4_logs(logs_root)
            if rc == 130 or stop["flag"]:
                return 130
            
            if rc == 0 and success_log:
                break

            # delete failed logs to avoid confusion in the next attempt
            _cleanup_failed_ulg(logs_root)

            # timeout retry
            if rc == 124:
                print(f"[RETRY] Timeout detected → retrying ({attempt}/{attempts})")

            # log failure retry
            elif not success_log:
                print(f"[RETRY] No .ulg collected → retrying ({attempt}/{attempts})")

            # other errors (e.g., mission fail)
            else:
                print(f"[RETRY] rc={rc} → retrying ({attempt}/{attempts})")

            # last attempt
            if attempt == attempts:
                print(f"[ERROR] Failed to run and collect logsafter {attempt} attempts")
                rc = max(rc, 1)

        overall_rc = max(overall_rc, 1 if rc != 0 else 0)

    return overall_rc


if __name__ == "__main__":
    raise SystemExit(main())
