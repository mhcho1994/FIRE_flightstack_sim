#!/usr/bin/env python3
"""
Multi-run launcher for ArduPilot ArduCopter SITL + Gazebo (gz sim).

What this script does:
---------------------
- Scans run_xxx directories under a run-root
- For each scenario:
      1) Launch ArduPilot SITL via sim_vehicle.py (gazebo-iris, JSON model)
      2) Wait a bit for SITL to initialize
      3) Launch Gazebo (gz sim) with the provided SDF world
      4) Keep both processes alive until:
           - one of them exits, OR
           - max runtime is reached
      5) Terminate both process trees cleanly (SIGTERM then SIGKILL)

Design notes
------------
  - Each launched command runs in its own *process group* (Linux/macOS).
    This allows us to kill the whole subtree (sim_vehicle.py typically spawns
    multiple children: mavproxy, arducopter SITL, etc.).
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
import shlex
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from tracemalloc import stop
from typing import Any, Optional, TextIO
import yaml
import math
import shutil
import re
import platform

_THIS_FILE = Path(__file__).resolve()
_TOOLS_DIR = _THIS_FILE.parents[1]
_COMMANDER_DIR = _TOOLS_DIR / "commander"
_SCENARIO_DIR = _TOOLS_DIR / "scenario"

if str(_COMMANDER_DIR) not in sys.path:
    sys.path.insert(0, str(_COMMANDER_DIR))

if str(_SCENARIO_DIR) not in sys.path:
    sys.path.insert(0, str(_SCENARIO_DIR))

from vehicle_model_builder import VehicleConfig, prepare_ardupilot_world
from pymavlink_ardupilot_commander import ArduPilotMissionRunner, MissionState, MissionStatus
print("[LOADING] Import ArduPilotMissionRunner")


@dataclass
class ArdupilotScenarioConfig:
    # sim instance parameters
    ardupilot_dir: str = "${FLIGHTSTACK_SIM_ROOT}/ap/ardupilot"
    instance: int = 0
    mavlink_url: str = "udp:127.0.0.1:14550"
    startup_delay_s: float = 0.0
    max_run_s: float = 60.0
    max_retries: int = 3

    # sim setup
    vehicle: str = "ArduCopter"
    frame: str = "gazebo-iris"
    model: str = "JSON"
    world: str = "iris_runway"
    location: str = "Purdue"
    scenario_name: str = "unnamed"
    gcs_outport: int = 14551
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
    verbose: bool = False
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

    if verbose and env is not None:
        print(f"[ENV]    {name}:")
        for k in sorted(env.keys()):
            if k.startswith("PX4") or k.startswith("GZ") or k in ("PATH", "HOME"):
                print(f"         {k}={env[k]}")

    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Line-buffered text logs (buffering=1 works with text=True).
    f = open(log_path, "w", buffering=1, encoding="utf-8")

    # preexec_fn=os.setsid: create a new process group/session (POSIX).
    # On Windows, os.setsid is not available; we fall back to default behavior.
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd else None,
        stdout=f,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        preexec_fn=os.setsid if os.name != "nt" else None, # new process group
    )

    return ProcHandle(name=name, proc=proc, log_path=log_path, log_file=f)


def _run_quiet(cmd: str) -> None:
    """Run a shell command quietly, ignoring all errors."""
    try:
        subprocess.run(
            cmd,
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except Exception:
        pass


def _pkill_patterns(patterns: list[str], sig: str) -> None:
    """
    Send a signal to processes matched by full command line.

    Parameters
    ----------
    patterns : list[str]
        Patterns passed to `pkill -f`.
    sig : str
        Signal name such as 'TERM' or 'KILL'.
    """
    if os.name == "nt":
        return

    for pattern in patterns:
        _run_quiet(f"pkill -{sig} -f {shlex.quote(pattern)}")


def _cleanup_gz_processes() -> None:
    """
    Best-effort cleanup for Gazebo / gz sim related processes that may survive
    after killing the parent shell/process group.

    Why needed:
    - Killing the bash PID or even its process group does not always terminate
      the actual gz server process.
    - Depending on the setup, Gazebo may appear as:
        * gz sim
        * gzserver / gzclient
    """
    if os.name == "nt":
        return

    patterns = [
        r"gz sim",
        r"gz sim server",
        r"gz sim gui",
    ]

    # Graceful termination first
    _pkill_patterns(patterns, "INT")


def _cleanup_sitl_processes() -> None:
    """Best-effort cleanup for common ArduPilot SITL related processes."""
    if os.name == "nt":
        return

    patterns = [
        r"sim_vehicle.py",
        r"arducopter",
        r"ArduCopter",
        r"arduplane",
        r"ArduPlane",
        r"ardurover",
        r"ArduRover",
        r"ardusub",
        r"ArduSub",
        r"antennatracker",
        r"AntennaTracker",
    ]

    _pkill_patterns(patterns, "INT")


def _cleanup_gcs_processes() -> None:
    """
    Best-effort cleanup for MAVProxy related processes that may survive
    after killing the parent shell/process group.

    Why needed:
    - Killing the bash PID or even its process group does not always terminate
      the actual mavproxy process.
    - Depending on the setup, MAVProxy may appear as:
        * mavproxy
    """
    if os.name == "nt":
        return

    patterns = [
        r"mavproxy",
    ]

    _pkill_patterns(patterns, "INT")


def _kill_tree(ph: ProcHandle, grace_s: float = 5.0) -> None:
    """
    Terminate a process and its descendants.

    For interactive tools such as sim_vehicle.py, we try SIGINT first
    because it is closer to Ctrl-C and may allow cleaner shutdown.
    Then we escalate to SIGTERM and finally SIGKILL.
    """
    if ph.proc.poll() is not None:
        return

    if os.name != "nt":
        try:
            pgid = os.getpgid(ph.proc.pid)
        except Exception:
            pgid = None
    else:
        pgid = None

    # 1) Try SIGINT first (best for sim_vehicle.py / SITL style processes)
    try:
        if os.name != "nt" and pgid is not None:
            os.killpg(pgid, signal.SIGINT)
        else:
            ph.proc.send_signal(signal.CTRL_BREAK_EVENT if os.name == "nt" else signal.SIGINT)
    except Exception:
        pass

    t0 = time.time()
    while time.time() - t0 < min(grace_s, 2.0):
        if ph.proc.poll() is not None:
            return
        time.sleep(0.1)

    # 2) Escalate to SIGTERM
    try:
        if os.name != "nt" and pgid is not None:
            os.killpg(pgid, signal.SIGTERM)
        else:
            ph.proc.terminate()
    except Exception:
        pass

    t1 = time.time()
    while time.time() - t1 < max(grace_s - 2.0, 1.0):
        if ph.proc.poll() is not None:
            return
        time.sleep(0.1)

    # 3) Final hard kill
    try:
        if os.name != "nt" and pgid is not None:
            os.killpg(pgid, signal.SIGKILL)
        else:
            ph.proc.kill()
    except Exception:
        pass

    # Final fallback cleanup for leftovers
    if "gz" in ph.name:
        _cleanup_gz_processes()

    if "sitl" in ph.name:
        _cleanup_sitl_processes()

    if "gcs" in ph.name:
        _cleanup_gcs_processes()


def _finalize_proc(ph: Optional[ProcHandle], grace_s: float = 5.0) -> None:
    """Best-effort: ensure process is dead and log file is closed."""
    if ph is None:
        return
    try:
        _kill_tree(ph)
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


def _ensure_waf_ready(ap_dir: Path) -> None:
    """Ensure ArduPilot waf submodule/bootstrap is present."""
    waf = ap_dir / "modules" / "waf"
    waf_light = ap_dir / "modules" / "waf" / "waf-light"

    if waf.exists() and waf_light.exists():
        return

    print("[INFO] Missing waf or waf-light. Initializing submodules...")
    subprocess.run(["git", "submodule", "update", "--init", "--recursive"], cwd=ap_dir)

    if not waf.exists() or not waf_light.exists():
        raise RuntimeError(
            "waf bootstrap failed: './waf' or 'modules/waf/waf-light' still missing."
        )


def _configure_ardupilot(ap_dir: Path) -> None:
    """
    Run waf configure robustly.

    ArduPilot may fetch waf submodule on the first call and ask to run again,
    so we allow one retry.
    """
    _ensure_waf_ready(ap_dir)

    for attempt in (1, 2):
        try:
            print(f"[INFO] Configuring ArduPilot SITL (attempt {attempt})...")
            subprocess.run(["./waf", "configure", "--board", "sitl"], cwd=ap_dir, check=True)
            return
        except subprocess.CalledProcessError as e:
            if attempt == 1:
                print("[WARN] Initial configure failed. Retrying after submodule/bootstrap check...")
                _ensure_waf_ready(ap_dir)
                continue
            raise RuntimeError("ArduPilot waf configure failed after retry.") from e


def _ensure_ardupilot_built(ap_dir: Path, vehicle: str = "copter") -> Path:
    """
    Ensure ArduPilot SITL binary exists. If not, build it.

    Returns
    -------
    Path to SITL binary
    """

    bin_map = {
        "copter": "arducopter",
        "plane": "arduplane",
        "rover": "ardurover",
    }

    if vehicle not in bin_map:
        raise ValueError(f"Unknown vehicle type: {vehicle}")

    binary = ap_dir / "build" / "sitl" / "bin" / bin_map[vehicle]

    # already built
    if binary.exists():
        print(f"[INFO] ArduPilot already built: {binary}")
        return binary

    print("[INFO] ArduPilot not built. Building SITL...")

    # Important: configure may need a retry after waf submodule bootstrap
    _configure_ardupilot(ap_dir)

    print(f"[INFO] Building ArduPilot target: {vehicle}")
    subprocess.run(["./waf", vehicle], cwd=ap_dir)

    if not binary.exists():
        raise RuntimeError(f"Build finished but binary not found: {binary}")

    print(f"[INFO] Build complete: {binary}")

    return binary


def _run_sitl_cmd(vehicle: str, frame: str, model: str, instance: int, gcs_outport: int, mavlink_url: str, location: str) -> list[str]:
    """
    Run the sim_vehicle.py command for ArduCopter SITL.

    Notes:
      - `-I <instance>` helps separate multiple runs (some paths/ports are derived).
      - `--out=udp:127.0.0.1:<port>` is useful if you want to connect QGC/MAVSDK.
      - `--no-rebuild` makes repeated runs faster.
      Use custom locations.txt file for reading spawning locations
    """
    if 'microsoft-standard-WSL2' in platform.release():
        return [
            "sim_vehicle.py",
            "-v", f"{vehicle}",
            "-f", f"{frame}",
            "--model", f"{model}",
            "--no-rebuild",
            "-I", str(instance),
            f"--location={location}",
            f"--out={mavlink_url}",
            f"--out={_get_gcs_url(gcs_outport)}",
        ]

    else:
        return [
            "sim_vehicle.py",
            "-v", f"{vehicle}",
            "-f", f"{frame}",
            "--model", f"{model}",
            "--no-rebuild",
            "-I", str(instance),
            f"--location={location}",
            f"--out={_get_gcs_url(gcs_outport)}",
        ]


def _get_gcs_url(gcs_outport: int) -> str:
    """
    Return the default MAVProxy output URL for forwarding MAVLink traffic
    to a local GCS listener.

    The GCS is expected to listen on ``127.0.0.1:<gcs_outport>``, and
    MAVProxy/sim_vehicle.py sends traffic using the ``udp:<host>:<port>``
    URL format.
    """
    return f"udp:127.0.0.1:{gcs_outport}"


def _apply_headless_terminal_env(env: dict[str, str]) -> dict[str, str]:
    """
    Prevent ArduPilot's run_in_terminal_window.sh from opening UI terminals.

    An empty DISPLAY makes the script skip xterm/GUI terminals and fall back to
    logging the SITL process to /tmp/<name>.log. The tmux/screen/zellij vars are
    removed for this child process so headless runs do not create terminal panes.
    """
    env = env.copy()
    env["DISPLAY"] = ""
    env.pop("WAYLAND_DISPLAY", None)
    env.pop("SITL_RITW_TERMINAL", None)
    env.pop("TMUX", None)
    env.pop("STY", None)
    env.pop("ZELLIJ", None)
    return env


def _run_mavproxy_cmd(out_port: int, headless: bool = False) -> list[str]:
    """
    Run the mavproxy command to connect to the SITL instance.
    Open the mavconsole and map to monitor MAVLink messages.

    Example:
      mavproxy.py --master=udp:127.0.0.1:14550
    """

    if headless:
        return [
            "mavproxy.py", 
            f"--master=udp:127.0.0.1:{out_port}",
        ]
    else:
        return [
            "mavproxy.py", 
            f"--master=udp:127.0.0.1:{out_port}",
            "--map",
            "--console",
        ]


def _run_gz_cmd(world_sdf: str, verbose: str = "-v4", headless: bool = False) -> list[str]:
    """
    Run the Gazebo Sim command.

    Example:
      gz sim -v4 -r iris_runway.sdf "--headless-rendering"
    """
    if headless:
        return ["gz", "sim", verbose, "-s", "-r", world_sdf]
    else:
        return ["gz", "sim", verbose, "-r", world_sdf]


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
    ardupilot_dir: Path,
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
    model: str, 
    world: str,
    location: str,
    headless: bool,
    verbose: bool,
) -> int:
    """
    Run a single ArduPilot SITL + Gazebo + MAVProxy simulation episode.

    This function orchestrates the full lifecycle of a single simulation run:
    launching required processes (Gazebo, MAVProxy, SITL), executing a mission
    via `ArduPilotMissionRunner`, monitoring execution status, and handling
    termination conditions.

    Execution Flow
    --------------
    1. Launch Gazebo (if not already running)
    2. Launch MAVProxy (if not already running)
    3. Launch ArduPilot SITL (if not already running)
    4. Start mission runner with the provided scenario
    5. Periodically monitor:
        - user interrupt (Ctrl+C)
        - timeout condition
        - mission completion status
    6. Finalize runner and processes
    7. Clean up SITL and Gazebo processes

    Parameters
    ----------
    ardupilot_dir : Path
        Root directory of the ArduPilot repository.

    instance : int
        SITL instance index (used for multi-instance simulation).

    scenario_path : Path
        Path to scenario YAML defining mission/trajectory.

    logs_dir : Path
        Directory where log files (SITL, Gazebo, MAVProxy) are stored.

    mavproxy_outport : int
        UDP port used by MAVProxy for forwarding MAVLink messages.

    mavlink_url : str
        MAVLink connection string (e.g., "udp://127.0.0.1:14550").

    startup_delay_s : float
        Delay (seconds) inserted between launching each component to allow
        proper initialization.

    max_run_s : float
        Maximum allowed runtime (seconds) before timeout termination.

    stop : dict
        Shared flag dictionary for external interrupt handling.
        Expected format: {"flag": bool}

    current : dict
        Dictionary storing running subprocess handles:
        {"gz": Popen, "gcs": Popen, "sitl": Popen}

    vehicle : str
        ArduPilot vehicle type (e.g., "ArduCopter").

    frame : str
        Frame configuration (e.g., "gazebo-iris").

    model : str
        Simulation model type (e.g., "JSON").

    world : str
        Gazebo world name or an explicit generated SDF path.

    location : str
        Predefined ArduPilot location (e.g., "Purdue").

    headless : bool
        Whether to run in headless mode (no GUI). If True, MAVProxy console/map and Gazebo GUI will be disabled.

    verbose : bool
        Whether to print detailed logs and environment variables for debugging.

    Returns
    -------
    int
        Exit code representing run outcome:
        - 0 : successful completion
        - 1 : mission runner error
        - 124 : timeout
        - 130 : user interrupt (Ctrl+C)

    Notes
    -----
    - Processes are launched only if not already present in `current`.
    - SITL and Gazebo are explicitly terminated at the end of the run.
    - MAVProxy may persist across runs depending on external management.
    - This function is designed for iterative batch execution of scenarios.
    """

    # set logs dir
    sitl_log = logs_dir / "sitl.log"
    gz_log = logs_dir / "gazebo.log"
    gcs_log = logs_dir / "gcs.log"

    # Gazebo first
    if current.get("gz") is None:
        time.sleep(startup_delay_s)
        print(f"[INFO] Waiting {startup_delay_s:.1f}s for Gazebo to initialize...")

        if verbose:
            gz_verbosity = "-v4"
        else:
            gz_verbosity = "-v1"

        world_sdf = world if world.endswith(".sdf") else f"{world}.sdf"
        gz = _popen("gazebo", _run_gz_cmd(world_sdf, gz_verbosity, headless=headless), cwd=logs_dir, log_path=gz_log)
        current["gz"] = gz

    # GCS (MAVProxy Console and Map) second
    if current.get("gcs") is None:
        time.sleep(startup_delay_s)
        print(f"[INFO] Waiting {startup_delay_s:.1f}s for GCS to initialize...")

        gcs_env = os.environ.copy()
        gcs_env["PATH"] = f"{Path.home() / '.local/bin'}:{gcs_env.get('PATH', '')}"

        gcs = _popen("gcs", _run_mavproxy_cmd(gcs_outport, headless=headless), cwd=logs_dir, log_path=gcs_log, env=gcs_env)
        current["gcs"] = gcs

    # Ardupilot SITL third
    if current.get("sitl") is None:
        time.sleep(startup_delay_s)
        print(f"[INFO] Waiting {startup_delay_s:.1f}s for SITL to initialize...")

        # pass environmental variables for SITL location info
        locations_path = Path(_THIS_FILE.parent / "locations.txt").resolve()

        if not locations_path.exists():
            raise FileNotFoundError(
                f"ArduPilot locations.txt not found: {locations_path}"
            )

        gz_env = os.environ.copy()
        gz_env["ARDUPILOT_LOCATIONS"] = str(locations_path)

        if headless:
            gz_env = _apply_headless_terminal_env(gz_env)

        sitl = _popen("sitl", _run_sitl_cmd(vehicle, frame, model, instance, gcs_outport, mavlink_url, location), cwd=logs_dir, log_path=sitl_log, env=gz_env)
        current["sitl"] = sitl

    time.sleep(startup_delay_s)
    runner = ArduPilotMissionRunner(scenario_path=scenario_path)
    runner.start()

    t0 = time.time()

    while True:
        # 1) user interrupt
        if stop["flag"]:
            print("[STOP] user interrupt")
            _finalize_runner(runner)
            rc = 130
            break
        
        # 2) max runtime exceeded
        if time.time() - t0 > max_run_s:
            print(f"[TIMEOUT] exceeded {max_run_s:.1f}s")
            _finalize_runner(runner)
            rc = 124
            break

        # 3) commander finished
        status = runner.get_status()

        print(
            f"[RUNNER] state={status.state.name} "
            f"done={status.done} success={status.success} "
            f"msg={status.message}"
        )
            
        if runner.is_done():
            if status.success:
                rc = 0
                _finalize_runner(runner)
                print(f"[RUNNER] normally terminated")
            else:
                rc = 1
                _finalize_runner(runner)
                print(f"[RUNNER] error: {status.error}")
            break

        time.sleep(1.0)

    _finalize_proc(current.get("sitl"))
    current["sitl"] = None
    _finalize_proc(current.get("gz"))
    current["gz"] = None
    _finalize_proc(current.get("gcs"))
    current["gcs"] = None

    time.sleep(startup_delay_s)
    print('[HIT] move to next iteration')
    
    return rc


def _check_ardupilot_logs(run_dir: Path) -> bool:
    """
    Check whether ArduPilot SITL generated a valid binary log.

    A successful ArduPilot SITL run is expected to create:
      run_dir/ardu_logs/raw/logs/

    and at least one `.BIN` file inside that directory.

    Parameters
    ----------
    run_dir : Path
        Run-specific log directory.
        Example: data/sitl_logs/run_0000

    Returns
    -------
    bool
        True if the logs directory exists and contains at least one `.BIN` file,
        False otherwise.
    """
    raw_root = run_dir
    logs_dir = raw_root / "logs"

    if not raw_root.exists():
        print(f"[WARN] ArduPilot raw log directory not found: {raw_root}")
        return False

    if not logs_dir.exists():
        print(f"[WARN] ArduPilot logs directory not found: {logs_dir}")
        return False

    if not logs_dir.is_dir():
        print(f"[WARN] ArduPilot logs path is not a directory: {logs_dir}")
        return False

    bin_files = sorted(logs_dir.glob("*.BIN"))

    if not bin_files:
        print(f"[WARN] No ArduPilot BIN log found in: {logs_dir}")
        return False

    print(f"[INFO] ArduPilot BIN log found: {bin_files[-1]}")
    return True


def _cleanup_failed_bin(run_dir: Path):
    raw_dir = run_dir
    if not raw_dir.exists():
        return

    bin_files = list(raw_dir.glob("*.BIN"))

    if not bin_files:
        return

    for f in bin_files:
        try:
            f.unlink()
            print(f"[CLEANUP] removed {f}")
        except Exception as e:
            print(f"[WARN] failed to remove {f}: {e}")


def _load_scenario_yaml_ardupilot(run_dir: Path) -> ArdupilotScenarioConfig:
    """
    Load scenario.yaml from run_dir and extract ArduPilot-specific configuration.

    Supported keys in scenario.yaml:
      ardupilot_dir: ${FLIGHTSTACK_SIM_ROOT}/ap/ardupilot
      instance: 0
      vehicle: ArduCopter
      frame: gazebo-iris
      model: JSON
      world: iris_runway
      location: Purdue
      gcs_outport: 14551
      connect_url (mavlink): udp:127.0.0.1:14550
    """
    scenario_path = run_dir / "scenario.yaml"
    if not scenario_path.exists():
        raise FileNotFoundError(f"Missing scenario.yaml: {scenario_path}")

    data: dict[str, Any] = yaml.safe_load(scenario_path.read_text(encoding="utf-8")) or {}

    # Accept both top-level keys and nested (e.g., {"sim": {...}})
    # You can extend this mapping if your scenario.yaml has a different schema.
    sim = data.get("autopilots",{}).get("ardupilot",{}).get("sim",{})
    scenario = data.get("common",{}).get("scenario",{})
    mavlink = data.get("autopilots",{}).get("ardupilot",{}).get("mavlink",{})
    cfg = ArdupilotScenarioConfig(
        ardupilot_dir=Path(os.path.expandvars(str(sim.get("ardupilot_dir", ArdupilotScenarioConfig.ardupilot_dir)))).resolve(),
        instance=int(sim.get("instance", ArdupilotScenarioConfig.instance)),
        vehicle=str(sim.get("vehicle", ArdupilotScenarioConfig.vehicle)),
        frame=str(sim.get("frame", ArdupilotScenarioConfig.frame)),
        model=str(sim.get("model", ArdupilotScenarioConfig.model)),
        world=str(sim.get("world", ArdupilotScenarioConfig.world)),
        vehicle_config=VehicleConfig.from_common(data.get("common", {})),
        location=str(sim.get("location", ArdupilotScenarioConfig.location)),
        scenario_name=str(scenario.get("name", ArdupilotScenarioConfig.scenario_name)),
        gcs_outport=str(sim.get("mavproxy_outport", ArdupilotScenarioConfig.gcs_outport)),
        mavlink_url=str(mavlink.get("connect_url", ArdupilotScenarioConfig.mavlink_url)),
    )
    return cfg


def _iter_run_dirs(data_root: Path) -> list[Path]:
    """
    Find ./data/run_* directories, sorted by name.
    """
    if not data_root.exists():
        return []
    return sorted([p for p in data_root.iterdir() if p.is_dir() and p.name.startswith("run_")])


def _prepare_run_dir(run_dir: Path, force: bool) -> tuple[bool, Path]:
    """
    Decide whether to skip or run.
    
    Returns:
        True  -> skip
        False -> run
    """
    logs_dir = run_dir / "ardu_logs"
    bin_files = list(logs_dir.rglob("*.BIN")) if logs_dir.exists() else []

    # case 1: force → always clean up and run
    if force:
        if (run_dir / "ardu_logs").exists():
            print(f"[CLEAN] Removing existing logs in {run_dir}")
            shutil.rmtree(run_dir / "ardu_logs")
        return False, logs_dir

    # case 2: valid log exists → skip
    if logs_dir.exists() and len(bin_files) > 0:
        return True, logs_dir

    # case 3: no log → run
    return False, logs_dir


def _apply_cli_overrides(cfg: ArdupilotScenarioConfig, args: argparse.Namespace) -> ArdupilotScenarioConfig:
    cfg = copy.deepcopy(cfg)
    cfg.startup_delay_s = args.startup_delay_s
    cfg.max_run_s = args.max_run_s
    cfg.max_retries = args.max_retries

    return cfg


def main() -> int:
    # Parse command-line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", type=Path, default=Path("./data/sitl_logs"), help="Root folder containing run_xxx/scenario.yaml and run_xxx/ardu_logs")
    ap.add_argument("--prepare-only", action="store_true", help="Generate and validate per-run SDFs without launching SITL or changing logs")
    ap.add_argument("--force", action="store_true", help="Re-run even if ardu_logs exists")
    ap.add_argument("--startup-delay-s", type=float, default=5.0)
    ap.add_argument("--max-run-s", type=float, default=700.0)
    ap.add_argument("--max-retries", type=int, default=3, help="Number of retries for failed runs (0 for no retries)")
    ap.add_argument("--headless", action="store_true", help="Run Gazebo/QGC in headless mode (for batch SITL in CLI modes)")
    ap.add_argument("--verbose", action="store_true", help="Verbose logging of commands and environment variables")
    args = ap.parse_args()

    # Access to the resolved run root path
    data_root = args.run_root.resolve()

    stop = {"flag": False}
    current: dict[str, Optional["ProcHandle"]] = {
        "gz": None, 
        "gcs": None, 
        "sitl": None,
        }  

    # Mark stop flag and terminate running processes on SIGINT/SIGTERM
    def _sig(_signum, _frame):
        stop["flag"] = True
        # immediately kill running processes (so ArduPilot doesn't linger)
        _finalize_proc(current.get("gz"))
        _finalize_proc(current.get("gcs"))
        _finalize_proc(current.get("sitl"))
        current["gz"] = None
        current["gcs"] = None
        current["sitl"] = None

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

        # Check/Create output directory for this run
        skip, logs_dir = ((False, run_dir / "ardu_logs") if args.prepare_only else
                          _prepare_run_dir(run_dir, force=args.force))
        if skip:
            print(f"[SKIP] {run_dir} (ardu_logs exists and contains .BIN)")
            continue

        # Load scenario.yaml
        try:
            cfg = _load_scenario_yaml_ardupilot(run_dir)
        except Exception as e:
            print(f"[ERROR] {run_dir}: failed to load scenario.yaml: {e}")
            overall_rc = 2
            continue

        # Get scenario path
        scenario_path = run_dir / "scenario.yaml"

        # Set other configurations
        cfg = _apply_cli_overrides(cfg, args)

        # Materialize once, before starting any process. Retries reuse exactly
        # the same SDF and never resample or compound the scale factors.
        try:
            if cfg.vehicle_config is not None:
                cfg.world = str(prepare_ardupilot_world(cfg.vehicle_config, cfg.world, run_dir))
                print(f"[SDF] {cfg.world}")
        except (OSError, ValueError) as exc:
            print(f"[ERROR] {run_dir}: failed to prepare vehicle SDF: {exc}")
            overall_rc = 2
            continue
        if args.prepare_only:
            print(f"[PREPARED] {run_dir.name}: world={cfg.world}")
            continue

        # Confirm that Ardupilot is built
        _ensure_ardupilot_built(ap_dir=cfg.ardupilot_dir, vehicle=cfg.vehicle.replace("Ardu", "").lower())

        # Print configuration info
        print(f"\n---- {run_dir.name}: RUN {cfg.scenario_name} Scenario ----")
        print(f"  ardupilot_dir={cfg.ardupilot_dir} vehicle={cfg.vehicle}")
        print(f"  world={cfg.world} location={cfg.location}")
        print(f"  instance={cfg.instance} gcs_outport={cfg.gcs_outport}")
        print(f"  startup_delay={cfg.startup_delay_s} max_run_s={cfg.max_run_s} max_retries={cfg.max_retries}")
        print(f"  logs_root={logs_dir} scenario_path={scenario_path} mavlink_url={cfg.mavlink_url}")

        # Execute the scenario
        rc = 1
        for attempt in range(1, cfg.max_retries + 1):

            if stop["flag"]:
                print("Interrupted. Exiting.")
                return 130

            print(f"[RUN] Attempt {attempt}/{cfg.max_retries} for {run_dir.name}")
            rc = run_once(
                ardupilot_dir=cfg.ardupilot_dir,
                instance=cfg.instance,
                scenario_path=scenario_path,
                logs_dir=logs_dir,
                gcs_outport=cfg.gcs_outport,
                mavlink_url=cfg.mavlink_url,
                startup_delay_s=cfg.startup_delay_s,
                max_run_s=cfg.max_run_s,
                stop=stop,
                current=current,
                vehicle=cfg.vehicle,
                frame=cfg.frame,
                model=cfg.model,
                world=cfg.world,
                location=cfg.location,
                headless=args.headless,
                verbose=args.verbose,
            )

            # Attempt to check logs from Ardupilot SITL
            success_log = _check_ardupilot_logs(logs_dir)

            if rc == 0 and success_log:
                break

            # delete failed logs to avoid confusion in the next attempt
            _cleanup_failed_bin(logs_dir)

            # timeout retry
            if rc == 124:
                print(f"[RETRY] Timeout detected → retrying ({attempt}/{cfg.max_retries})")

            # log failure retry
            elif not success_log:
                print(f"[RETRY] No .BIN collected → retrying ({attempt}/{cfg.max_retries})")

            # other errors (e.g., mission fail)
            else:
                print(f"[RETRY] rc={rc} → retrying ({attempt}/{cfg.max_retries})")

            # last attempt
            if attempt == cfg.max_retries:
                print(f"[ERROR] Failed to run and collect logs after {attempt} attempts")
                rc = max(rc, 1)

        overall_rc = max(overall_rc, 1 if rc != 0 else 0)

    return overall_rc


if __name__ == "__main__":
    raise SystemExit(main())
