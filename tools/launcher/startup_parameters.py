"""Prepare and audit per-attempt parameter files without MAVLink writes."""

import hashlib
import json
import os
from pathlib import Path
import shlex

import yaml

from tools.commander.fc_parameter_profiles import parse_controller_parameters
from tools.commander.flight_parameters import parse_flight_parameters, validate_flight_parameter_file
from tools.scenario.create_params_sets import PARAMETER_FILENAMES, parameter_file_bytes, encode_parameter_values
from tools.launcher.parameter_bridge import BRIDGE_VERSION, bridge_gains, bridge_flight
from tools.launcher.firmware_probe import identify_firmware, binary_sha256


def validate_parameter_file(config, autopilot, run_dir):
    """Reject missing, modified, or stale files before starting simulator processes."""
    if config is None:
        return None
    content = parameter_file_bytes(config, autopilot)
    if "parameter_file" in config:
        path = Path(run_dir) / config["parameter_file"]
        actual = path.read_bytes()
        if "parameter_file_sha256" in config and hashlib.sha256(actual).hexdigest() != config["parameter_file_sha256"]:
            raise ValueError(f"{path}: parameter file SHA256 mismatch")
        if actual != content:
            raise ValueError(f"{path}: parameter file does not match controller_gains.values; regenerate the file")
    return content


def collect_boot_parameters(data, autopilot, run_dir):
    """Validate both configuration blocks and return one native boot payload."""
    settings = data.get("autopilots", {}).get(autopilot, {})
    gains = parse_controller_parameters(settings.get("controller_gains"), autopilot)
    content = validate_parameter_file(gains, autopilot, run_dir)
    flight = parse_flight_parameters(settings.get("flight_parameters"), autopilot)
    flight_content = validate_flight_parameter_file(flight, autopilot, run_dir)
    values = {} if gains is None else dict(gains["values"])
    sources = [] if gains is None else [gains.get("parameter_file", "scenario.controller_gains.values")]
    if flight is not None:
        if values.keys() & flight["values"].keys():
            raise ValueError("Controller gains and flight parameters overlap")
        values.update(flight["values"])
        sources.append(flight.get("parameter_file", "scenario.flight_parameters.values"))
        content = (flight_content if gains is None else
                   encode_parameter_values(values, autopilot, label="controller gains and flight parameters"))
    return content, values, sources


def has_boot_parameters(run_dir, autopilot):
    data = yaml.safe_load((Path(run_dir) / "scenario.yaml").read_text()) or {}
    settings = data.get("autopilots", {}).get(autopilot, {})
    return any(settings.get(key) is not None for key in ("controller_gains", "flight_parameters"))


def identify_scenario_firmware(data, autopilot, launch_config=None):
    sim = data.get("autopilots", {}).get(autopilot, {}).get("sim", {})
    config = launch_config or {}
    root = config.get(f"{autopilot}_dir", sim.get(f"{autopilot}_dir"))
    if root is None:
        root = Path(__file__).resolve().parents[2] / "ap" / autopilot
    root = Path(os.path.expandvars(str(root))).resolve()
    return identify_firmware(root, autopilot, config.get("vehicle", sim.get("vehicle", "ArduCopter")),
        parameter_schema=config.get("parameter_schema") or sim.get("parameter_schema"))


def resolve_boot_parameters(data, autopilot, run_dir, target):
    content, original_values, sources = collect_boot_parameters(data, autopilot, run_dir)
    if content is None:
        return None, None
    if target.get("autopilot") != autopilot:
        raise ValueError("Boot firmware does not match requested autopilot")
    if binary_sha256(target["binary"]) != target["binary_sha256"]:
        raise ValueError("Firmware binary changed after identification")
    settings = data["autopilots"][autopilot]
    gains, gain_changes = bridge_gains(settings.get("controller_gains"), autopilot, target)
    flight, flight_changes = bridge_flight(settings.get("flight_parameters"), autopilot, target)
    values = {} if gains is None else dict(gains["values"])
    if flight is not None:
        if values.keys() & flight["values"].keys():
            raise ValueError("Bridged controller and flight parameters overlap")
        values.update(flight["values"])
    if values != original_values:
        label = ("controller gains and flight parameters" if gains is not None and flight is not None
                 else "controller gains" if gains is not None else "flight parameters")
        content = encode_parameter_values(values, autopilot, label=label)
    report = dict(autopilot=autopilot, method="startup_file", bridge_version=BRIDGE_VERSION,
        target_firmware=target, controller_gains=gains, flight_parameters=flight,
        source_controller_gains=settings.get("controller_gains"),
        source_flight_parameters=settings.get("flight_parameters"),
        conversions=gain_changes + flight_changes, values=values,
        source=sources[0] if len(sources) == 1 else sources,
        sha256=hashlib.sha256(content).hexdigest(),
        storage_policy="wipe_run_eeprom" if autopilot == "ardupilot" else "import_after_airframe")
    return content, report


def prepare_boot_parameters(scenario_path: Path, autopilot: str, logs_dir: Path, *,
                            target_firmware=None, launch_config=None) -> Path | None:
    """Validate source artifacts, bridge once, and freeze the effective boot/readback inputs."""
    scenario_path, logs_dir = Path(scenario_path), Path(logs_dir)
    scenario = scenario_path.read_bytes()
    data = yaml.safe_load(scenario) or {}
    if collect_boot_parameters(data, autopilot, scenario_path.parent)[0] is None:
        return None
    target = target_firmware or identify_scenario_firmware(data, autopilot, launch_config)
    content, report = resolve_boot_parameters(data, autopilot, scenario_path.parent, target)
    logs_dir.mkdir(parents=True, exist_ok=True)
    path = logs_dir / ("boot_parameters" + Path(PARAMETER_FILENAMES[autopilot]).suffix)
    path.write_bytes(content)
    report.update(file=path.name, scenario_sha256=hashlib.sha256(scenario).hexdigest())
    (logs_dir / "boot_parameters.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return path.resolve()


def read_boot_report(parameter_file, scenario_path):
    if parameter_file is None:
        return None
    path = Path(parameter_file)
    report = json.loads(path.with_suffix('.json').read_text())
    if (hashlib.sha256(path.read_bytes()).hexdigest() != report['sha256'] or
            hashlib.sha256(Path(scenario_path).read_bytes()).hexdigest() != report['scenario_sha256']):
        raise ValueError('Boot payload or scenario changed after preparation')
    if binary_sha256(report['target_firmware']['binary']) != report['target_firmware']['binary_sha256']:
        raise ValueError('Firmware binary changed after boot preparation')
    return report


def prepare_px4_startup_script(original: Path, work_dir: Path, parameter_file: Path) -> str:
    """Insert a native BSON import after airframe defaults, before simulator/modules.

    The firmware build remains untouched. Refuse unknown rcS layouts instead of
    silently importing before a parameter reset or after controllers have started.
    Return a fixed relative -s path: PX4 passes that argument through /bin/sh.
    """
    original, work_dir = Path(original), Path(work_dir)
    script = original.read_text(encoding="utf-8")
    marker = "\ndataman start\n"
    if script.count(marker) != 1 or '. "$autostart_file"' not in script.split(marker)[0]:
        raise ValueError(f"{original}: unsupported PX4 startup layout for parameter import")
    injection = ("\n# FIRE: import this attempt's parameters before module startup.\n"
                 f"param import {shlex.quote(str(Path(parameter_file).resolve()))} || exit 1\n")
    path = work_dir.parent / "boot_parameters.rcS"
    path.write_text(script.replace(marker, injection + marker), encoding="utf-8")
    return "../boot_parameters.rcS"
