"""Capture controller parameters from a BIN/ULog file or a fresh SITL boot."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import runpy
import signal
import socket
import struct
import subprocess
import sys
import tempfile
import time

import yaml

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.commander.fc_parameter_profiles import GAIN_GROUPS, normalize_profiles, parameter_names
from tools.scenario.create_params_sets import create_params_sets


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SITL_OUTPUT = PROJECT_ROOT / "data/parameters/sitl_controller_gains.yaml"
DEFAULT_FILE_OUTPUT = PROJECT_ROOT / "data/parameters/flight_test_controller_gains.yaml"
CONTEXT_PARAMETERS = {"ardupilot": (), "px4": (
    "MC_ROLLRATE_K", "MC_PITCHRATE_K", "MC_YAWRATE_K", "MC_YAW_WEIGHT",
)}


from tools.commander.fc_parameter_registry import SCHEMAS, schema_for_firmware, schema_groups
from tools.commander.firmware_identity import read_firmware_identity


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_controller_parameters(connection, autopilot: str, *, timeout=2.0, attempts=3,
                               check_stop=lambda: None, firmware=None, parameter_schema=None) -> tuple[dict, dict]:
    """Read every supported group twice; never send PARAM_SET or arm commands.

    This low-level API reads current values. The fresh-boot CLI workflow captures
    startup values for the selected firmware and airframe configuration.
    """
    try:
        from tools.commander.mavlink_parameters import ParameterClient
    except ImportError as exc:
        raise RuntimeError("Reading SITL parameters requires pymavlink") from exc
    firmware = firmware or read_firmware_identity(connection, autopilot, timeout=timeout * attempts, check_stop=check_stop)
    schema = schema_for_firmware(firmware, parameter_schema)
    groups = list(schema_groups(autopilot, schema))
    names = parameter_names(autopilot, groups, schema)
    client = ParameterClient(connection, timeout=timeout, attempts=attempts,
                             check_stop=check_stop)
    client.require_disarmed(autopilot)
    first = {name: client.get_param(name) for name in (*names, *CONTEXT_PARAMETERS[autopilot])}
    second = {name: client.get_param(name) for name in first}
    for name, value in first.items():
        if second[name] != value:
            raise RuntimeError(f"{name}: parameter changed during capture ({value} -> {second[name]})")
    client.require_disarmed(autopilot)
    profile = create_params_sets({autopilot: dict(groups=groups,
        baseline={name: second[name] for name in names}, parameters={},
        parameter_schema=schema, firmware=firmware)})[autopilot]
    return profile, {name: second[name] for name in CONTEXT_PARAMETERS[autopilot]}


def validate_scenario(path: Path, autopilots: list[str]) -> dict:
    if path.name != "scenario.yaml":
        raise ValueError("capture requires a generated file named scenario.yaml")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("autopilots"), dict):
        raise ValueError("scenario must contain an autopilots mapping")
    if not autopilots or len(set(autopilots)) != len(autopilots):
        raise ValueError("select distinct autopilots")
    for autopilot in autopilots:
        if autopilot not in GAIN_GROUPS:
            raise ValueError(f"Unsupported autopilot: {autopilot}")
        entry = data["autopilots"].get(autopilot)
        if not isinstance(entry, dict) or not isinstance(entry.get("sim"), dict):
            raise ValueError(f"scenario requires autopilots.{autopilot}.sim")
        if entry.get("controller_gains") is not None:
            raise ValueError(f"{autopilot}: generate the capture scenario without --controller-gains")
        sim = entry["sim"]
        for name in (f"{autopilot}_dir", "vehicle", "frame", "world", "location"):
            if name not in sim:
                raise ValueError(f"{autopilot}.sim.{name} is required for SITL capture")
        instance = sim.get("instance", 0)
        if isinstance(instance, bool) or not isinstance(instance, int) or not 0 <= instance <= 9:
            raise ValueError("capture supports SITL instances 0 through 9")
    return data


def _check_ports(ports):
    for kind, port in ports:
        with socket.socket(socket.AF_INET, kind) as sock:
            if kind == socket.SOCK_STREAM:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("0.0.0.0", port))
            except OSError as exc:
                raise RuntimeError(f"Port {port} is occupied; stop the SITL using this instance first") from exc


def _firmware_identity(root: Path, binary: Path, startup_files: list[Path]) -> dict:
    def git(*args):
        result = subprocess.run(["git", "-C", str(root), *args], text=True,
                                capture_output=True, timeout=10)
        return result.stdout.strip() if result.returncode == 0 else None
    dirty = git("status", "--porcelain", "--untracked-files=no")
    return dict(directory=str(root), commit=git("rev-parse", "HEAD"),
                tracked_source_dirty=None if dirty is None else bool(dirty),
                binary=str(binary), binary_sha256=sha256_file(binary),
                startup_files={str(path): sha256_file(path) for path in startup_files})


def _connect_and_read(url: str, autopilot: str, current: dict, timeout: float, *,
                      parameter_schema=None, firmware_root=None):
    from pymavlink import mavutil

    deadline = time.monotonic() + timeout
    def check_stop():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"{autopilot}: capture exceeded {timeout:g} seconds")
        for name, handle in current.items():
            if handle.proc.poll() is not None:
                raise RuntimeError(f"{autopilot}: {name} exited; see {handle.log_path}")

    connection = None
    try:
        while connection is None:
            check_stop()
            try:
                connection = mavutil.mavlink_connection(url, source_system=245,
                    source_component=mavutil.mavlink.MAV_COMP_ID_MISSIONPLANNER,
                    autoreconnect=False, retries=1)
            except OSError:
                time.sleep(0.2)
        expected = {"ardupilot": mavutil.mavlink.MAV_AUTOPILOT_ARDUPILOTMEGA,
                    "px4": mavutil.mavlink.MAV_AUTOPILOT_PX4}[autopilot]
        while True:
            check_stop()
            heartbeat = connection.recv_match(type="HEARTBEAT", blocking=True, timeout=0.2)
            if heartbeat is None or heartbeat.get_srcComponent() != mavutil.mavlink.MAV_COMP_ID_AUTOPILOT1:
                continue
            if heartbeat.autopilot != expected:
                raise RuntimeError(f"{url}: expected {autopilot}, got autopilot={heartbeat.autopilot}")
            if heartbeat.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED:
                raise RuntimeError("SITL capture requires a disarmed vehicle")
            connection.target_system = heartbeat.get_srcSystem()
            connection.target_component = heartbeat.get_srcComponent()
            break
        firmware = read_firmware_identity(connection, autopilot, check_stop=check_stop)
        if firmware_root is not None:
            from tools.launcher.firmware_probe import add_version_hint
            firmware = add_version_hint(firmware, firmware_root)
        profile, context = read_controller_parameters(connection, autopilot, check_stop=check_stop,
            firmware=firmware, parameter_schema=parameter_schema)
        return profile, dict(context=context, firmware_identity=firmware, parameter_schema=profile["parameter_schema"], system_id=connection.target_system,
                             component_id=connection.target_component, connection=url, read_passes=2)
    finally:
        if connection is not None:
            connection.close()


def _capture_stack(autopilot: str, scenario_path: Path, work_dir: Path, timeout: float):
    # Reuse the launchers' model/environment and process helpers, but never
    # instantiate a mission runner or attach a ground station.
    from tools.launcher import run_px4_gz_sitl as px4
    from tools.launcher import run_ardupilot_gz_sitl as ardu
    from tools.scenario.vehicle_model_builder import prepare_ardupilot_world, resource_file

    loader = ardu._load_scenario_yaml_ardupilot if autopilot == "ardupilot" else px4._load_scenario_yaml_px4
    cfg = loader(scenario_path.parent)
    logs = work_dir / autopilot
    logs.mkdir()
    current = {}
    startup_files = []
    try:
        root = cfg.ardupilot_dir if autopilot == "ardupilot" else cfg.px4_dir
        lat, lon, alt, heading = px4._read_location_from_txt(PROJECT_ROOT / "tools/launcher/locations.txt", cfg.location)
        if autopilot == "ardupilot":
            if cfg.vehicle != "ArduCopter" or cfg.model != "JSON":
                raise ValueError("ArduPilot controller capture requires ArduCopter with model JSON")
            _check_ports([(socket.SOCK_STREAM, 5760 + cfg.instance * 10 + offset) for offset in (0, 2, 3)]
                         + [(socket.SOCK_DGRAM, 9002 + cfg.instance * 10)])
            binary = ardu._ensure_ardupilot_built(root, "copter")
            vehicle_info = root / "Tools/autotest/pysim/vehicleinfo.py"
            registry = runpy.run_path(str(vehicle_info))["VehicleInfo"]().options
            frame = registry[cfg.vehicle]["frames"].get(cfg.frame)
            if frame is None or "default_params_filename" not in frame:
                raise ValueError(f"No explicit default parameter files for ArduPilot frame {cfg.frame}")
            defaults = frame["default_params_filename"]
            defaults = [defaults] if isinstance(defaults, str) else defaults
            startup_files = [root / "Tools/autotest" / name for name in defaults]
            if not all(path.is_file() for path in startup_files):
                raise FileNotFoundError("ArduPilot frame defaults are incomplete")
            world = (prepare_ardupilot_world(cfg.vehicle_config, cfg.world, logs)
                     if cfg.vehicle_config is not None else resource_file(cfg.world, "worlds"))
            env = os.environ.copy()
            env["GZ_PARTITION"] = f"fire-defaults-{os.getpid()}-{time.monotonic_ns()}"
            env.pop("GZ_SIM_SERVER_CONFIG_PATH", None)
            px4._prepend_paths(env, "GZ_SIM_RESOURCE_PATH", [
                PROJECT_ROOT / "gz/FIRE_moonshot_gazebo/models", PROJECT_ROOT / "gz/FIRE_moonshot_gazebo/worlds",
                PROJECT_ROOT / "gz/ardupilot_gazebo/models", PROJECT_ROOT / "gz/ardupilot_gazebo/worlds"])
            px4._prepend_paths(env, "GZ_SIM_SYSTEM_PLUGIN_PATH", [
                PROJECT_ROOT / "gz/ardupilot_gazebo/build", PROJECT_ROOT / "build/fire_gz_plugins/lib"])
            current["gazebo"] = px4._popen("gazebo", ardu._run_gz_cmd(str(world), "-v1", headless=True),
                cwd=logs, log_path=logs / "gazebo.log", env=env)
            command = [str(binary), "-S", "-w", "--model", cfg.model, "--speedup", "1",
                "--defaults", ",".join(map(str, startup_files)), "--sim-address=127.0.0.1",
                f"-I{cfg.instance}", "--home", f"{lat},{lon},{alt},{heading}"]
            url = f"tcp:127.0.0.1:{5760 + cfg.instance * 10}"
        else:
            _check_ports([(socket.SOCK_DGRAM, port + cfg.instance) for port in (14540, 14580, 18570)])
            px4.ensure_px4_built(root)
            if not cfg.frame.startswith("gz_"):
                raise ValueError("PX4 capture requires a gz_ frame")
            sim = px4.prepare_simulation(root, cfg.world, cfg.frame[3:], cfg.instance,
                partition=f"fire-defaults-{os.getpid()}-{time.monotonic_ns()}")
            if cfg.vehicle_config is not None:
                if sim.model_name != cfg.vehicle_config.model_name:
                    raise ValueError("PX4 frame and common.vehicle.model_name do not match")
                sim.model_path = px4.generate_vehicle_model(cfg.vehicle_config, sim.model_path, logs)
            current["gazebo"] = px4._popen("gazebo", px4.gazebo_command(sim, True),
                cwd=logs, log_path=logs / "gazebo.log", env=sim.env)
            px4.spawn_model(sim, (0, 0, 0.25, 0, 0, math.pi / 2 - math.radians(heading)),
                            process=current["gazebo"].proc, timeout=min(timeout, 45))
            command = px4.px4_command(root, cfg.instance, logs / "rootfs", cfg.vehicle)
            binary = Path(command[0])
            etc = root / "build/px4_sitl_default/etc"
            startup_files = [etc / "init.d-posix/rcS", etc / "init.d/rc.mc_defaults"]
            startup_files.extend(sorted((etc / "init.d-posix/airframes").glob(f"{cfg.vehicle}_*")))
            env = px4.px4_environment(sim, cfg.vehicle)
            # Inherited overrides must not alter the selected SITL startup configuration.
            for key in list(env):
                if key.startswith("PX4_PARAM_") or key in ("PX4_SIM_SPEED_FACTOR", "PX4_ESTIMATOR"):
                    del env[key]
            env.update(PX4_HOME_LAT=str(lat), PX4_HOME_LON=str(lon), PX4_HOME_ALT=str(alt))
            world = sim.world_path
            url = f"udpin:127.0.0.1:{14540 + cfg.instance}"
        current["sitl"] = px4._popen("sitl", command, cwd=logs, log_path=logs / "sitl.log", env=env)
        profile, report = _connect_and_read(url, autopilot, current, timeout,
            parameter_schema=getattr(cfg, "parameter_schema", None), firmware_root=root)
        report.update(firmware=_firmware_identity(root, binary, startup_files),
            vehicle=cfg.vehicle, frame=cfg.frame, instance=cfg.instance, world=str(world),
            world_sha256=sha256_file(Path(world)), location=cfg.location,
            command=command, command_cwd=str(logs), clean_storage=True, mission_started=False,
            captured_at=datetime.now(timezone.utc).isoformat())
        (logs / "controller_parameters.parm").write_text(
            "".join(f"{name} {value:.9g}\n" for name, value in profile["baseline"].items()), encoding="utf-8")
        return profile, report
    finally:
        for handle in reversed(list(current.values())):
            px4._finalize_proc(handle)



def relative_metadata(report: dict, destination: Path) -> dict:
    """Serialize filesystem references relative to the containing JSON directory.

    Keep the in-memory report and execution argv intact. Command paths in the
    serialized report are normalized for inspection, including PX4's script
    argument, which PX4 normally resolves against its runtime work directory.
    """
    base = Path(destination).absolute().parent

    def relative(value):
        return Path(os.path.relpath(value, base)).as_posix() if Path(value).is_absolute() else value

    result = copy.deepcopy(report)
    result["paths_relative_to"] = "metadata_directory"
    for name in ("scenario", "session_dir"):
        result[name] = relative(result[name])
    for autopilot, entry in result["autopilots"].items():
        if "world" in entry:
            entry["world"] = relative(entry["world"])
        firmware = entry.get("firmware", {})
        for name in ("directory", "binary"):
            if name in firmware:
                firmware[name] = relative(firmware[name])
        if "startup_files" in firmware:
            firmware["startup_files"] = {relative(path): digest
                                         for path, digest in firmware["startup_files"].items()}
        if "command" in entry:
            command = entry["command"]
            cwd = entry.get("command_cwd", str(Path(report["session_dir"]) / autopilot))
            entry["command_cwd"] = relative(cwd)
            work_dir = Path(command[command.index("-w") + 1]) if autopilot == "px4" and "-w" in command else None
            converted = []
            for index, argument in enumerate(command):
                previous = command[index - 1] if index else None
                if previous == "--defaults":
                    converted.append(",".join(relative(path) for path in argument.split(",")))
                elif previous == "-s" and work_dir is not None and work_dir.is_absolute():
                    converted.append(relative(str(work_dir / argument)))
                else:
                    converted.append(relative(argument))
            entry["command"] = converted
    return result


def _capture_from_sitl(scenario: Path, output: Path = DEFAULT_SITL_OUTPUT, *,
                            autopilots=None, work_dir: Path | None = None,
                            timeout=60.0, overwrite=False) -> dict:
    """Capture the scenario's SITL startup parameters into a reusable profile.

    Values depend on the selected firmware, vehicle/frame and startup parameters.
    Each invocation creates a new session directory; no existing EEPROM/rootfs
    is reused or deleted. Failed captures retain logs and never publish a profile.
    """
    scenario, output = Path(scenario).resolve(), Path(output).resolve()
    autopilots = list(GAIN_GROUPS) if autopilots is None else list(autopilots)
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be finite and positive")
    if output == scenario:
        raise ValueError("output must differ from the input scenario")
    validate_scenario(scenario, autopilots)
    metadata_path = output.with_suffix(".meta.json")
    if not overwrite and (output.exists() or metadata_path.exists()):
        raise FileExistsError(f"{output} or its metadata exists; choose a new output or use --overwrite")
    parent = Path(work_dir).resolve() if work_dir is not None else scenario.parent / "parameter_captures"
    parent.mkdir(parents=True, exist_ok=True)
    session = Path(tempfile.mkdtemp(prefix="capture_", dir=parent))
    print(f"[CAPTURE] Logs and fresh runtime storage: {session}", flush=True)
    profiles = {}
    report = dict(status="capturing", scenario=str(scenario), scenario_sha256=sha256_file(scenario),
                  session_dir=str(session), autopilots={}, started_at=datetime.now(timezone.utc).isoformat())
    try:
        for autopilot in autopilots:
            print(f"[CAPTURE] Starting {autopilot}", flush=True)
            profiles[autopilot], report["autopilots"][autopilot] = _capture_stack(autopilot, scenario, session, timeout)
        normalized = normalize_profiles(profiles)
        for autopilot in profiles:
            report["autopilots"][autopilot]["baseline_sha256"] = normalized[autopilot]["baseline_sha256"]
        content = "# Captured from fresh SITL boots; see the adjacent .meta.json for provenance.\n"
        content += yaml.safe_dump(profiles, sort_keys=False)
        report.update(status="complete", profile_sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
                      finished_at=datetime.now(timezone.utc).isoformat())
        _publish_capture(output, content, relative_metadata(report, metadata_path))
        return profiles
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}",
                      finished_at=datetime.now(timezone.utc).isoformat())
        raise
    finally:
        session_report = session / "capture.json"
        session_report.write_text(json.dumps(relative_metadata(report, session_report), indent=2) + "\n", encoding="utf-8")


def _detect_log_format(path):
    """Choose a decoder from the header, independently of firmware and suffix."""
    with path.open("rb") as stream:
        header = stream.read(16)
    if len(header) >= 16:
        for kind, magic in (("bin", b"\xa3\x95\x80"), ("ulog", b"ULog\x01\x12\x35")):
            if header.startswith(magic):
                return kind
    raise ValueError(f"{path.name}: invalid BIN/ULog log header; "
                     "trajectory CSV/NPZ files do not preserve controller parameters")


_FIRMWARE_LABEL = re.compile(
    r"^(PX4|ArduPilot|ArduCopter|ArduPlane|ArduSub|ArduRover|Rover|AntennaTracker|Blimp)(?=\s|$)",
    re.IGNORECASE)
_FIRMWARE_VERSION = re.compile(r"^\s+[vV]?(\d+)\.(\d+)\.(\d+)(?=[\s()+-]|$)")


def _identify_log_firmware(history, log_format, requested):
    """Require an explicit firmware label; format and gain names are not identity."""
    evidence, stacks, versions, vehicles = [], set(), set(), set()
    version_type, git_hash = None, None

    def add_label(value, field):
        nonlocal git_hash
        value = str(value).strip("\x00 \t\r\n")
        match = _FIRMWARE_LABEL.match(value)
        if match is None:
            return False
        stacks.add("px4" if match[1].upper() == "PX4" else "ardupilot")
        if match[1].lower() != "px4":
            vehicles.add(match[1])
        evidence.append(dict(field=field, value=value))
        version = _FIRMWARE_VERSION.match(value[match.end():])
        if version:
            versions.add(tuple(int(part) for part in version.groups()))
            commit = re.search(r"\(([0-9a-fA-F]{7,40})\)", value)
            if commit:
                git_hash = commit[1]
        return True

    if log_format == "bin":
        for value in history["firmware"]:
            add_label(value, "MSG.Message")
        for record in history["firmware_versions"]:
            if add_label(record.get("FWS", ""), "VER.FWS"):
                evidence.append(dict(field="VER", value=record))
                if all(key in record for key in ("Maj", "Min", "Pat")):
                    versions.add(tuple(int(record[key]) for key in ("Maj", "Min", "Pat")))
                version_type = record.get("FWT", version_type)
                if "GH" in record:
                    git_hash = f"{record['GH']:08x}"
    else:
        firmware = history["firmware"]
        add_label(firmware.get("sys_name", ""), "ULog.sys_name")
        if "ver_sw_release" in firmware:
            release = int(firmware["ver_sw_release"])
            if not 0 <= release <= 0xffffffff:
                raise ValueError("Invalid ULog firmware ver_sw_release")
            versions.add(((release >> 24) & 255, (release >> 16) & 255, (release >> 8) & 255))
            version_type = release & 255
            evidence.append(dict(field="ULog.ver_sw_release", value=release))
        git_hash = firmware.get("ver_sw")

    if not stacks:
        raise ValueError("Cannot identify autopilot from logged firmware information "
                         "(BIN MSG/VER or ULog sys_name); no filename or parameter-name fallback")
    if len(stacks) != 1:
        raise ValueError(f"Conflicting firmware autopilot identities: {sorted(stacks)}")
    autopilot = next(iter(stacks))
    if requested != "auto" and requested != autopilot:
        raise ValueError(f"Logged firmware is {autopilot}, not {requested}")
    if len(versions) > 1:
        raise ValueError(f"Conflicting firmware versions: {sorted(versions)}")
    version = ".".join(map(str, next(iter(versions)))) if versions else None
    vehicle = "ArduCopter" if any(v.lower() == "arducopter" for v in vehicles) else next(iter(vehicles), None)
    return dict(autopilot=autopilot, version=version, version_type=version_type,
                git_hash=git_hash, evidence=evidence, **({"vehicle": vehicle} if vehicle else {}))


def _record(history, name, value, timestamp, *, initial):
    value = float(value)
    previous = history["final"].get(name)
    if initial:
        history["initial"].setdefault(name, value)
    if name not in history["first_seen_us"]:
        history["first_seen_us"][name] = timestamp
    if previous != value and (previous is not None or not initial):
        history["changes"].append(dict(timestamp_us=timestamp, parameter=name,
                                       before=previous, after=value))
    history["final"][name] = value


def _read_bin(path, tracked):
    try:
        from pymavlink import DFReader
    except ImportError as exc:
        raise RuntimeError("Reading BIN logs requires pymavlink") from exc
    history = dict(initial={}, final={}, first_seen_us={}, changes=[])
    firmware, firmware_versions = [], []
    log = DFReader.DFReader_binary(str(path))
    try:
        while True:
            message = log.recv_match(type=["PARM", "MSG", "VER"], strict=True)
            if message is None:
                break
            if message.get_type() == "MSG":
                text = str(message.Message)
                if _FIRMWARE_LABEL.match(text) and text not in firmware:
                    firmware.append(text)
                continue
            if message.get_type() == "VER":
                record = {key: getattr(message, key) for key in ("FWS", "Maj", "Min", "Pat", "FWT", "GH")
                          if hasattr(message, key)}
                if record not in firmware_versions:
                    firmware_versions.append(record)
                continue
            name = message.Name
            if isinstance(name, bytes):
                name = name.decode("ascii")
            name = name.rstrip("\x00")
            if name in tracked:
                timestamp = getattr(message, "TimeUS", None)
                if timestamp is None and hasattr(message, "TimeMS"):
                    timestamp = int(message.TimeMS) * 1000
                # Value is the actual logged setting; Default is not the flight setting.
                _record(history, name, message.Value, timestamp, initial=True)
    finally:
        log.close()
    history.update(firmware=firmware, firmware_versions=firmware_versions,
                   timestamp_basis="PARM.TimeUS (or TimeMS * 1000); null if absent",
                   initial_basis="first observed PARM.Value for each parameter; not an atomic boot snapshot")
    return history


def _read_ulog(path, tracked):
    try:
        from pyulog import ULog
    except ImportError as exc:
        raise RuntimeError("Reading ULog files requires pyulog") from exc
    # Filtering out all data topics makes pyulog's change timestamps stale.
    # Read all topics so timestamps retain the preceding sample time.
    with path.open("rb") as stream:
        log = ULog(stream)
    if log.file_corruption:
        raise ValueError(f"{path.name}: pyulog detected file corruption")
    history = dict(initial={}, final={}, first_seen_us={}, changes=[])
    for name, value in log.initial_parameters.items():
        if name in tracked:
            _record(history, name, value, log.start_timestamp, initial=True)
    for timestamp, name, value in log.changed_parameters:
        if name in tracked:
            _record(history, name, value, timestamp, initial=False)
    firmware = {key: str(log.msg_info_dict[key]) for key in (
        "ver_sw", "ver_sw_release", "ver_hw", "sys_name") if key in log.msg_info_dict}
    history.update(firmware=firmware, dropout_count=len(log.dropouts),
                   timestamp_basis="pyulog preceding data sample timestamp (microseconds), not exact change time",
                   initial_basis="ULog initial_parameters section; excludes parameters first seen later")
    return history


def read_flight_parameters(log_file, *, autopilot="auto", snapshot="initial", groups=None, parameter_schema=None):
    """Return (profile, provenance) without writing files or starting SITL.

    Default scope is every supported controller group. Missing values fail;
    callers can explicitly select complete groups. Provenance uses original log
    names and includes both endpoints and all observed changes for selected gains
    and context parameters. Autopilot identity comes from logged firmware, while
    the binary header selects the decoder. Names and values remain native to the
    recorded firmware; missing gains never fall back to defaults.
    """
    path = Path(log_file).resolve(strict=True)
    if autopilot not in ("auto", *GAIN_GROUPS):
        raise ValueError(f"Unsupported autopilot: {autopilot}")
    if snapshot not in ("initial", "final"):
        raise ValueError("snapshot must be initial or final")
    source_hash = sha256_file(path)
    log_format = _detect_log_format(path)
    # Read once, keeping supported candidates until the firmware has been identified.
    candidates = {name for _, registry in SCHEMAS.values()
                  for names in registry.values() for name in names}
    for names in CONTEXT_PARAMETERS.values():
        candidates.update(names)
    try:
        history = (_read_bin if log_format == "bin" else _read_ulog)(path, candidates)
    except (TypeError, IndexError, struct.error) as exc:
        raise ValueError(f"{path.name}: cannot decode flight log: {exc}") from exc
    if sha256_file(path) != source_hash:
        raise ValueError("Input log changed during capture; use a completed flight log")
    identity = _identify_log_firmware(history, log_format, autopilot)
    autopilot = identity["autopilot"]
    schema = schema_for_firmware(identity, parameter_schema)
    registry = schema_groups(autopilot, schema)
    groups = list(registry) if groups is None else list(groups)
    if (not groups or len(set(groups)) != len(groups)
            or any(group not in registry for group in groups)):
        raise ValueError(f"{autopilot}: groups must be distinct supported names: {list(registry)}")
    names = parameter_names(autopilot, groups, schema)
    tracked = set(names) | set(CONTEXT_PARAMETERS[autopilot])
    for key in ("initial", "final", "first_seen_us"):
        history[key] = {name: value for name, value in history[key].items() if name in tracked}
    history["changes"] = [event for event in history["changes"] if event["parameter"] in tracked]
    # Validate only the selected stack/groups, including their observed history.
    observed = list(history["initial"].items()) + list(history["final"].items())
    observed.extend((event["parameter"], event["after"]) for event in history["changes"])
    for name, value in observed:
        if not math.isfinite(value):
            raise ValueError(f"{name}: non-finite logged parameter")
    selected = history[snapshot]
    baseline = {name: selected[name] for name in names if name in selected}
    missing = [name for name in names if name not in selected]
    if missing:
        raise ValueError(f"{autopilot}: missing {snapshot} controller parameters: {', '.join(missing)}. "
                         "No defaults were substituted; select complete --groups or supply a full log")
    profiles = create_params_sets({autopilot: dict(groups=groups, baseline=baseline, parameters={},
        parameter_schema=schema, firmware=identity)})
    context_names = CONTEXT_PARAMETERS[autopilot]
    metadata = dict(autopilot=autopilot, log_format=log_format, firmware_identity=identity,
                    snapshot=snapshot, source_log=str(path),
                    source_sha256=source_hash, groups=profiles[autopilot]["groups"],
                    baseline_sha256=normalize_profiles(profiles)[autopilot]["baseline_sha256"],
                    history=history, parameter_schema=schema, conversions={},
                    context_parameters={name: selected[name] for name in context_names if name in selected},
                    missing_context_parameters=[name for name in context_names if name not in selected])
    return profiles, metadata


def _load_merge_capture(output, metadata_path):
    """Validate an existing capture pair, including legacy single-stack metadata."""
    if not output.exists() and not metadata_path.exists():
        return {}, {}
    if not output.is_file() or not metadata_path.is_file():
        raise ValueError("Cannot merge: both the existing YAML and its .meta.json are required")
    content = output.read_bytes()
    normalized = normalize_profiles(yaml.safe_load(content), base_dir=output.parent, source=str(output))
    if not normalized:
        raise ValueError("Cannot merge: existing profile has no autopilots")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError("Cannot merge: metadata must be a mapping")
    if metadata.get("profile_sha256") != hashlib.sha256(content).hexdigest():
        raise ValueError("Cannot merge: existing YAML does not match metadata profile_sha256")
    if metadata.get("paths_relative_to") != "metadata_directory":
        raise ValueError("Cannot merge: unsupported metadata path base")
    if "autopilots" in metadata:
        if metadata.get("schema_version") != 2 or not isinstance(metadata["autopilots"], dict):
            raise ValueError("Cannot merge: unsupported capture metadata schema")
        entries = metadata["autopilots"]
    else:
        # The original capture tool stored one autopilot's provenance at the root.
        autopilot = metadata.get("autopilot")
        if not isinstance(autopilot, str) or metadata.get("schema_version", 1) != 1:
            raise ValueError("Cannot merge: unsupported legacy capture metadata")
        entries = {autopilot: {key: value for key, value in metadata.items()
                              if key not in ("profile_sha256", "paths_relative_to", "schema_version")}}
    if set(entries) != set(normalized):
        raise ValueError("Cannot merge: metadata autopilots do not match the existing profile")
    for autopilot, entry in entries.items():
        profile = normalized[autopilot]
        if (not isinstance(entry, dict) or entry.get("autopilot") != autopilot
                or entry.get("baseline_sha256") != profile["baseline_sha256"]
                or entry.get("groups") != profile["groups"]):
            raise ValueError(f"Cannot merge: {autopilot} metadata does not match its baseline/groups")
        if (entry.get("snapshot") not in ("initial", "final")
                or not isinstance(entry.get("history"), dict)
                or not isinstance(entry["history"].get("changes"), list)
                or not isinstance(entry.get("conversions"), dict)
                or not isinstance(entry.get("missing_context_parameters"), list)
                or not isinstance(entry.get("source_log"), str) or not entry["source_log"]):
            raise ValueError(f"Cannot merge: incomplete {autopilot} capture metadata")
        # Legacy absolute paths can be retained even when the log is no longer
        # available. No previous source log needs to be opened during a merge.
        if Path(entry["source_log"]).is_absolute():
            entry["source_log"] = Path(os.path.relpath(entry["source_log"], metadata_path.parent)).as_posix()
    profiles = {ap: {key: entry[key] for key in ("groups", "baseline", "parameters", "parameter_schema", "firmware") if key in entry}
                for ap, entry in normalized.items()}
    return profiles, entries


def _capture_from_file(log_file, output, *, autopilot="auto", snapshot="initial",
                                   groups=None, overwrite=False, merge=False, parameter_schema=None):
    """Save a profile and .meta.json; return all saved autopilot profiles.

    merge=True adds/replaces only the captured autopilot and preserves the other
    stack's complete profile and provenance. overwrite=True replaces both files.
    Existing captures must have matching metadata to merge; absent output is OK.
    """
    if merge and overwrite:
        raise ValueError("merge and overwrite are mutually exclusive")
    source, output = Path(log_file).resolve(strict=True), Path(output).resolve()
    if output.suffix.lower() not in (".yaml", ".yml"):
        raise ValueError("Output must be a .yaml or .yml profile")
    metadata_path = output.with_suffix(".meta.json")
    if source in (output, metadata_path):
        raise ValueError("Output must not replace the source log")
    if not (overwrite or merge) and (output.exists() or metadata_path.exists()):
        raise FileExistsError(f"{output} or its metadata exists; choose another path or use --merge/--overwrite")
    profiles, entries = _load_merge_capture(output, metadata_path) if merge else ({}, {})
    captured, entry = read_flight_parameters(source, autopilot=autopilot, snapshot=snapshot, groups=groups, parameter_schema=parameter_schema)
    captured_at = datetime.now(timezone.utc).isoformat()
    entry.update(source_log=Path(os.path.relpath(source, metadata_path.parent)).as_posix(),
                 captured_at=captured_at)
    profiles.update(captured)
    entries[entry["autopilot"]] = entry
    content = "# Captured from a flight log; see adjacent .meta.json for original names and changes.\n"
    content += yaml.safe_dump(profiles, sort_keys=False)
    metadata = dict(schema_version=2, paths_relative_to="metadata_directory",
                    profile_sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
                    updated_at=captured_at, autopilots=entries)
    _publish_capture(output, content, metadata)
    return profiles


def _publish_capture(output: Path, content: str, metadata: dict):
    """Publish metadata first and YAML last; the profile hash detects an interrupted pair."""
    report = json.dumps(metadata, indent=2, allow_nan=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    for path, value in ((output.with_suffix(".meta.json"), report), (output, content)):
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(value)
        try:
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def _validate_capture_options(*, from_file, from_sitl, output, autopilot, snapshot,
                              groups, autopilots, work_dir, timeout, merge, overwrite, parameter_schema=None):
    if (from_file is None) == (from_sitl is None):
        raise ValueError("Choose exactly one of --from-file LOG or --from-sitl SCENARIO")
    if merge and overwrite:
        raise ValueError("--merge and --overwrite are mutually exclusive")
    if from_file is not None:
        unsupported = {"--autopilots": autopilots, "--work-dir": work_dir, "--timeout": timeout}
    else:
        unsupported = {"--autopilot": autopilot, "--snapshot": snapshot, "--groups": groups,
                       "--merge": True if merge else None, "--parameter-schema": parameter_schema}
    invalid = [name for name, value in unsupported.items() if value is not None]
    if invalid:
        mode = "--from-file" if from_file is not None else "--from-sitl"
        raise ValueError(f"{', '.join(invalid)} cannot be used with {mode}")
    if output is not None and Path(output).suffix.lower() not in (".yaml", ".yml"):
        raise ValueError("Output must be a .yaml or .yml profile")


def _capture_output_path(output, *, from_file: bool) -> Path:
    if output is None:
        output = DEFAULT_FILE_OUTPUT if from_file else DEFAULT_SITL_OUTPUT
    return Path(output).resolve()


def capture_flight_parameters(*, from_file=None, from_sitl=None, output=None,
                              autopilot=None, snapshot=None, groups=None,
                              autopilots=None, work_dir=None, timeout=None,
                              merge=False, overwrite=False, parameter_schema=None) -> dict:
    """Capture using exactly one source; return all saved autopilot profiles.

    from_file accepts real-flight or SITL BIN/ULog logs and supports autopilot,
    snapshot, groups and merge. from_sitl boots the given scenario with fresh
    storage and supports autopilots, work_dir and timeout. Default outputs are
    data/parameters/flight_test_controller_gains.yaml and sitl_controller_gains.yaml
    under PROJECT_ROOT, respectively. output overrides either default.
    overwrite applies to both modes.
    """
    _validate_capture_options(from_file=from_file, from_sitl=from_sitl, output=output,
        autopilot=autopilot, snapshot=snapshot, groups=groups, autopilots=autopilots,
        work_dir=work_dir, timeout=timeout, merge=merge, overwrite=overwrite, parameter_schema=parameter_schema)
    output = _capture_output_path(output, from_file=from_file is not None)
    if from_file is not None:
        return _capture_from_file(from_file, output, autopilot="auto" if autopilot is None else autopilot,
            snapshot="initial" if snapshot is None else snapshot, groups=groups,
            merge=merge, overwrite=overwrite, parameter_schema=parameter_schema)
    return _capture_from_sitl(from_sitl, output,
        autopilots=autopilots, work_dir=work_dir, timeout=60.0 if timeout is None else timeout,
        overwrite=overwrite)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--from-file", type=Path, metavar="LOG",
                        help="Read a BIN/ULog file from real flight or SITL; identify firmware from its contents")
    source.add_argument("--from-sitl", type=Path, metavar="SCENARIO",
                        help="Boot a generated scenario.yaml without controller_gains and read parameters")
    parser.add_argument("--output", type=Path,
                        help="Output YAML; defaults under repository root: "
                             "--from-file: data/parameters/flight_test_controller_gains.yaml; "
                             "--from-sitl: data/parameters/sitl_controller_gains.yaml")
    file_options = parser.add_argument_group("File options (--from-file)")
    file_options.add_argument("--autopilot", choices=("auto", *GAIN_GROUPS),
                              help="Default: identify logged firmware; an explicit stack must match")
    file_options.add_argument("--parameter-schema", choices=tuple(SCHEMAS),
                              help="Explicit native schema when the logged firmware version is unavailable")
    file_options.add_argument("--snapshot", choices=("initial", "final"), help="Default: initial logged values")
    file_options.add_argument("--groups", nargs="+", help="Default: all supported groups for the detected stack")
    sitl_options = parser.add_argument_group("SITL options (--from-sitl)")
    sitl_options.add_argument("--autopilots", nargs="+", choices=tuple(GAIN_GROUPS),
                             help="Default: both stacks, sequentially")
    sitl_options.add_argument("--work-dir", type=Path, help="Capture sessions parent; default: beside scenario.yaml")
    sitl_options.add_argument("--timeout", type=float, help="Connection/readback timeout per stack (default: 60 seconds)")
    output_mode = parser.add_mutually_exclusive_group()
    output_mode.add_argument("--merge", action="store_true",
                             help="File mode only: add/update one autopilot in an existing file capture")
    output_mode.add_argument("--overwrite", action="store_true", help="Replace the entire existing capture")
    args = parser.parse_args(argv)
    options = vars(args)
    try:
        _validate_capture_options(**options)
    except ValueError as exc:
        parser.error(str(exc))

    previous = signal.getsignal(signal.SIGTERM)
    def interrupted(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    try:
        profiles = capture_flight_parameters(**options)
        output = _capture_output_path(args.output, from_file=args.from_file is not None)
        metadata_path = output.with_suffix(".meta.json")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        for autopilot, entry in profiles.items():
            if args.from_file is None:
                print(f"[CAPTURE] {autopilot}: {len(entry['baseline'])} controller parameters from SITL")
                continue
            capture_info = metadata["autopilots"][autopilot]
            identity = capture_info.get("firmware_identity")
            if identity is not None:
                version = identity["version"] or "not recorded"
                print(f"[CAPTURE] {autopilot}: firmware version={version}, format={capture_info['log_format']}")
            print(f"[CAPTURE] {autopilot}: {len(entry['baseline'])} gains, snapshot={capture_info['snapshot']}")
            print(f"[CAPTURE] {autopilot}: {len(capture_info['history']['changes'])} gain/context changes; "
                  f"{len(capture_info['conversions'])} parameter name/unit conversions (see metadata)")
            if capture_info["missing_context_parameters"]:
                print(f"[CAPTURE] {autopilot}: missing context parameters: {capture_info['missing_context_parameters']}")
        print(output)
        print(metadata_path)
        return 0
    except KeyboardInterrupt:
        print("[CAPTURE] Interrupted; capture stopped", file=sys.stderr)
        return 130
    except (OSError, ValueError, RuntimeError, ImportError, yaml.YAMLError) as exc:
        print(f"[CAPTURE] {exc}", file=sys.stderr)
        return 1
    finally:
        signal.signal(signal.SIGTERM, previous)


if __name__ == "__main__":
    raise SystemExit(main())
