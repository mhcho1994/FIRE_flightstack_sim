"""Prepare and audit per-attempt parameter files without MAVLink writes."""

import hashlib
import json
from pathlib import Path
import shlex

import yaml

from tools.commander.fc_parameter_profiles import parse_controller_parameters
from tools.scenario.create_params_sets import PARAMETER_FILENAMES, parameter_file_bytes


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


def prepare_boot_parameters(scenario_path: Path, autopilot: str, logs_dir: Path) -> Path | None:
    """Freeze the validated native file in this attempt's logs for boot and archival.

    Older resolved scenarios without a file reference are rendered here, using
    their frozen values. They are never sampled again and never use PARAM_SET.
    """
    scenario_path, logs_dir = Path(scenario_path), Path(logs_dir)
    data = yaml.safe_load(scenario_path.read_text(encoding="utf-8")) or {}
    config = parse_controller_parameters(data.get("autopilots", {}).get(autopilot, {}).get("controller_gains"), autopilot)
    content = validate_parameter_file(config, autopilot, scenario_path.parent)
    if content is None:
        return None
    logs_dir.mkdir(parents=True, exist_ok=True)
    path = logs_dir / ("boot_parameters" + Path(PARAMETER_FILENAMES[autopilot]).suffix)
    path.write_bytes(content)
    report = dict(autopilot=autopilot, method="startup_file", file=path.name,
                  sha256=hashlib.sha256(content).hexdigest(),
                  source=config.get("parameter_file", "scenario.controller_gains.values"),
                  values=config["values"],
                  storage_policy="wipe_run_eeprom" if autopilot == "ardupilot" else "import_after_airframe")
    (logs_dir / "boot_parameters.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return path.resolve()


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
    injection = ("\n# FIRE: import this attempt's controller gains before module startup.\n"
                 f"param import {shlex.quote(str(Path(parameter_file).resolve()))} || exit 1\n")
    path = work_dir.parent / "boot_parameters.rcS"
    path.write_text(script.replace(marker, injection + marker), encoding="utf-8")
    return "../boot_parameters.rcS"
