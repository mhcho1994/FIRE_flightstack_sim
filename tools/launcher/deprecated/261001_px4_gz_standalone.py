#!/usr/bin/env python3
"""Run Gazebo and PX4 independently using the shared FIRE vehicle.

Run from the repository root. Missing PX4 SITL build outputs are built
automatically before either component starts (make -C ap/px4 px4_sitl).
After changing firmware, run that make command explicitly to rebuild.

Terminal 1 (omit --headless to open the Gazebo GUI):
    python3 tools/launcher/px4_gz_standalone.py gazebo --headless
Terminal 2 (waits for Terminal 1 to create fire_px4vision_0):
    python3 tools/launcher/px4_gz_standalone.py px4

Both terminals default to world=default_fire, model=fire_px4vision, instance=0,
vehicle=4006 and the same Gazebo partition. No environment snippet is required.
For other models, pass matching --world, --model, --instance and --partition
arguments to both commands, and the correct --vehicle to the PX4 command.
Connect QGroundControl separately, or use tools/commander/fake_gcs_heartbeat.py
for a headless GCS. PX4 parameters and ULogs live in logs/px4_standalone/0;
--work-dir selects another directory. Stop PX4 and then Gazebo with Ctrl+C.
For automated scenario missions use run_px4_gz_sitl.py, which starts and stops
both processes itself. Do not start these manual commands for a batch run.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import time
from dataclasses import dataclass
from typing import Callable
import xml.etree.ElementTree as ET

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PARTITION = "fire_px4_standalone"


@dataclass
class Simulation:
    world_path: Path
    world_name: str
    model_path: Path
    model_name: str
    entity_name: str
    env: dict[str, str]


def ensure_px4_built(px4_dir: Path) -> None:
    """Build missing SITL outputs before preparing Gazebo's plugin paths."""
    build = px4_dir / "build/px4_sitl_default"
    required = (build / "bin/px4", build / "etc/init.d-posix/rcS")
    if all(path.is_file() for path in required):
        return
    if not (px4_dir / "Makefile").is_file():
        raise FileNotFoundError(f"PX4 source Makefile not found: {px4_dir / 'Makefile'}")

    print(f"[BUILD] PX4 SITL outputs missing. Building in {px4_dir}...", flush=True)
    try:
        subprocess.run(["make", "px4_sitl"], cwd=px4_dir, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"PX4 SITL build failed (exit {exc.returncode}) in {px4_dir}; see build output above"
        ) from exc
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"PX4 SITL build finished but outputs are missing: {', '.join(missing)}")
    print("[BUILD] PX4 SITL ready.", flush=True)


def _prepend_paths(env: dict[str, str], key: str, paths: list[Path]) -> None:
    entries = [str(p) for p in paths if p.is_dir()]
    entries.extend(p for p in env.get(key, "").split(":") if p)
    env[key] = ":".join(dict.fromkeys(entries))


def simulation_environment(px4_dir: Path, partition: str) -> dict[str, str]:
    env = os.environ.copy()
    fire = PROJECT_ROOT / "gz/FIRE_moonshot_gazebo"
    upstream = px4_dir / "Tools/simulation/gz"
    external = PROJECT_ROOT / "gz/PX4_gazebo_models"
    _prepend_paths(env, "GZ_SIM_RESOURCE_PATH", [
        fire / "models", fire / "worlds", upstream / "models", upstream / "worlds",
        external / "models", external / "worlds",
    ])
    _prepend_paths(env, "GZ_SIM_SYSTEM_PLUGIN_PATH", [
        px4_dir / "build/px4_sitl_default/src/modules/simulation/gz_plugins",
        PROJECT_ROOT / "build/fire_gz_plugins/lib",
    ])
    # Standalone PX4 does not source gz_env.sh. Gazebo needs the PX4 sensor
    # systems and custom plugin path before its server starts.
    config = px4_dir / "src/modules/simulation/gz_bridge/server.config"
    if not config.is_file():
        raise FileNotFoundError(f"PX4 Gazebo server configuration missing: {config}")
    env["GZ_SIM_SERVER_CONFIG_PATH"] = str(config)
    env["GZ_PARTITION"] = partition
    env.setdefault("GZ_IP", "127.0.0.1")
    return env


def _resource_file(value: str, relative: Path, env: dict[str, str]) -> Path:
    direct = Path(os.path.expandvars(value)).expanduser()
    if direct.is_file():
        return direct.resolve()
    for root in env["GZ_SIM_RESOURCE_PATH"].split(":"):
        candidate = Path(root) / relative
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"Gazebo resource not found: {value} (searched GZ_SIM_RESOURCE_PATH)")


def prepare_simulation(px4_dir: Path, world: str, model: str, instance: int,
                       partition: str) -> Simulation:
    if instance < 0:
        raise ValueError("PX4 instance must be non-negative")
    env = simulation_environment(px4_dir, partition)
    world_file = world if world.endswith(".sdf") else f"{world}.sdf"
    world_path = _resource_file(world_file, Path(world_file), env)
    model_path = _resource_file(model, Path(model) / "model.sdf", env)
    world_xml = ET.parse(world_path).getroot()
    model_xml = ET.parse(model_path).getroot()
    world_element = world_xml.find("world")
    model_element = model_xml.find("model")
    if world_element is None or model_element is None:
        raise ValueError("Expected an SDF world file and an SDF model file")
    for root in (world_xml, model_xml):
        for plugin in root.iter("plugin"):
            if any(name in plugin.get("filename", "")
                   for name in ("ArduPilotPlugin", "MotorCommandMux")):
                raise ValueError("Use a PX4 world without ArduPilotPlugin/motor mux (e.g. default_fire)")
    world_name = world_element.attrib["name"]
    model_name = model_element.attrib["name"]
    if not all(re.fullmatch(r"[A-Za-z0-9_-]+", n) for n in (world_name, model_name)):
        raise ValueError("World/model names must contain only letters, numbers, underscores or hyphens")
    return Simulation(world_path, world_name, model_path, model_name,
                      f"{model_name}_{instance}", env)


def gazebo_command(sim: Simulation, headless: bool, verbose: bool = False) -> list[str]:
    """Start only the server; attach the GUI after spawn_model completes.

    Gazebo 8's GUI can discard entity creation updates during initial-state
    loading. Starting the GUI after spawn includes the vehicle in its snapshot.
    """
    cmd = ["gz", "sim", "-s", "-v", "4" if verbose else "1", "-r"]
    if headless:
        cmd.append("--headless-rendering")
    return cmd + [str(sim.world_path)]


def gazebo_gui_command(verbose: bool = False) -> list[str]:
    """Connect to the existing server using the same simulation environment."""
    return ["gz", "sim", "-g", "-v", "4" if verbose else "1"]


def _service(sim: Simulation, service: str, request_type: str, response_type: str,
             request: str, timeout_ms: int = 1000) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["gz", "service", "-s", f"/world/{sim.world_name}/{service}",
         "--reqtype", request_type, "--reptype", response_type,
         "--timeout", str(timeout_ms), "--req", request],
        env=sim.env, text=True, capture_output=True, timeout=timeout_ms / 1000 + 5,
    )


def wait_for_scene(sim: Simulation, *, entity: bool = False, timeout: float = 45,
                   process: subprocess.Popen | None = None,
                   cancelled: Callable[[], bool] = lambda: False) -> str:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cancelled():
            raise InterruptedError("Simulation startup interrupted")
        if process is not None and process.poll() is not None:
            raise RuntimeError(f"Gazebo exited during startup (status {process.returncode})")
        try:
            result = _service(sim, "scene/info", "gz.msgs.Empty", "gz.msgs.Scene", "")
            # Scene messages do not contain the world name. The service path
            # selects the world; check the model name only when attaching.
            ready = (f'name: "{sim.entity_name}"' in result.stdout if entity else
                     re.search(r"^(?:ambient|background|model|light|name|id)\s*[:{]", result.stdout, re.MULTILINE))
            if result.returncode == 0 and ready:
                return result.stdout
        except subprocess.TimeoutExpired:
            pass
        time.sleep(0.2)
    raise TimeoutError(f"Gazebo {'model ' + sim.entity_name if entity else 'world ' + sim.world_name} not ready")


def spawn_model(sim: Simulation, pose: tuple[float, ...], **wait_options) -> None:
    scene = wait_for_scene(sim, **wait_options)
    if f'name: "{sim.entity_name}"' in scene:
        raise RuntimeError(f"Gazebo model already exists: {sim.entity_name}")
    x, y, z, roll, pitch, yaw = pose
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    qx, qy = sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy
    qz, qw = cr * cp * sy - sr * sp * cy, cr * cp * cy + sr * sp * sy
    request = (
        f"sdf_filename: {json.dumps(str(sim.model_path))}, "
        f"name: {json.dumps(sim.entity_name)}, allow_renaming: false, "
        f"pose: {{ position: {{ x: {x}, y: {y}, z: {z} }}, "
        f"orientation: {{ x: {qx}, y: {qy}, z: {qz}, w: {qw} }} }}"
    )
    result = _service(sim, "create", "gz.msgs.EntityFactory", "gz.msgs.Boolean", request, 5000)
    if result.returncode != 0 or not re.search(r"data:\s*true", result.stdout):
        raise RuntimeError(f"Gazebo model creation failed: {result.stdout} {result.stderr}")
    wait_for_scene(sim, entity=True, **wait_options)
    print(f"[READY] {sim.world_name}/{sim.entity_name} from {sim.model_path}", flush=True)


def px4_environment(sim: Simulation, vehicle: str) -> dict[str, str]:
    env = sim.env.copy()
    for key in ("PX4_SIM_MODEL", "PX4_GZ_MODEL", "PX4_GZ_MODEL_POSE"):
        env.pop(key, None)
    env.update(PX4_GZ_STANDALONE="1", PX4_GZ_MODEL_NAME=sim.entity_name,
               PX4_GZ_WORLD=sim.world_name, PX4_SYS_AUTOSTART=str(vehicle), PX4_SIMULATOR="gz")
    return env


def px4_command(px4_dir: Path, instance: int, work_dir: Path, vehicle: str,
                *, daemon: bool = True) -> list[str]:
    build = px4_dir / "build/px4_sitl_default"
    binary = build / "bin/px4"
    etc = build / "etc"
    if not binary.is_file() or not (etc / "init.d-posix/rcS").is_file():
        raise FileNotFoundError(f"Build PX4 first: make -C {px4_dir} px4_sitl")
    if not list((etc / "init.d-posix/airframes").glob(f"{vehicle}_*")):
        raise ValueError(f"PX4 airframe {vehicle} is not present in {etc}")
    work_dir.mkdir(parents=True, exist_ok=True)
    return [str(binary), *(["-d"] if daemon else []), "-i", str(instance),
            "-w", str(work_dir.resolve()), "-s", "etc/init.d-posix/rcS", str(etc)]


def stop_process(proc: subprocess.Popen) -> None:
    """Stop only the session created for this subprocess, including its children."""
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            break
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            continue
    proc.wait()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("component", choices=("gazebo", "px4"))
    parser.add_argument("--px4-dir", type=Path, default=PROJECT_ROOT / "ap/px4")
    parser.add_argument("--world", default="default_fire", help="World name or SDF path")
    parser.add_argument("--model", default="fire_px4vision", help="Model name or model.sdf path")
    parser.add_argument("--instance", type=int, default=0)
    parser.add_argument("--vehicle", default="4006", help="PX4 airframe autostart ID")
    parser.add_argument("--partition", default=os.environ.get("GZ_PARTITION", DEFAULT_PARTITION))
    parser.add_argument("--pose", type=float, nargs=6, default=(0, 0, 0.25, 0, 0, math.pi / 2),
                        metavar=("X", "Y", "Z", "ROLL", "PITCH", "YAW"))
    parser.add_argument("--work-dir", type=Path, help="PX4 parameters and logs directory")
    parser.add_argument("--headless", action="store_true", help="Run Gazebo without its GUI")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    px4_dir = args.px4_dir.expanduser().resolve()
    ensure_px4_built(px4_dir)
    sim = prepare_simulation(px4_dir, args.world, args.model, args.instance, args.partition)
    print(f"[SIM] partition={args.partition} world={sim.world_path} model={sim.model_path}", flush=True)
    if args.component == "px4":
        work_dir = args.work_dir or PROJECT_ROOT / "logs/px4_standalone" / str(args.instance)
        cmd = px4_command(px4_dir, args.instance, work_dir, args.vehicle, daemon=False)
        wait_for_scene(sim, entity=True)
        os.execvpe(cmd[0], cmd, px4_environment(sim, args.vehicle))
    proc = subprocess.Popen(gazebo_command(sim, args.headless, args.verbose),
                            env=sim.env, start_new_session=True)
    def interrupted(_signum, _frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    gui = None
    try:
        spawn_model(sim, tuple(args.pose), process=proc)
        if args.headless:
            return proc.wait()
        gui = subprocess.Popen(gazebo_gui_command(args.verbose),
                               env=sim.env, start_new_session=True)
        while proc.poll() is None and gui.poll() is None:
            time.sleep(0.2)
        return proc.returncode if proc.returncode is not None else gui.returncode
    finally:
        if gui is not None:
            stop_process(gui)
        stop_process(proc)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except (OSError, ValueError, RuntimeError, TimeoutError) as exc:
        raise SystemExit(f"ERROR: {exc}")
