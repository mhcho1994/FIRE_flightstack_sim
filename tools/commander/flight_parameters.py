"""Explicit, unsampled flight defaults applied through the SITL boot files.

Only the REAL32 horizontal mission-speed parameters in the selected schema
are supported. Keep their native units separate from mission speeds in m/s.
"""

import hashlib
import math
from pathlib import Path
import struct

from tools.scenario.create_params_sets import encode_parameter_values
from tools.commander.fc_parameter_registry import schema_groups


SPEED_PARAMETERS = {"ardupilot": ("WPNAV_SPEED", 100.0), "px4": ("MPC_XY_CRUISE", 1.0)}
FILENAMES = {"ardupilot": "ardupilot_flight.parm", "px4": "px4_flight.bson"}


def parse_flight_parameters(raw, autopilot):
    if raw is None:
        return None
    if autopilot not in SPEED_PARAMETERS:
        raise ValueError(f"Unsupported autopilot: {autopilot}")
    allowed = {"values", "source", "parameter_file", "parameter_file_sha256", "parameter_schema"}
    if not isinstance(raw, dict) or set(raw) - allowed:
        raise ValueError("flight_parameters must be a mapping with known fields")
    schema = raw.get("parameter_schema")
    schema_groups(autopilot, schema)
    name, multiplier = (("WP_SPD", 1.0) if schema == "arducopter-4.7" else SPEED_PARAMETERS[autopilot])
    values = raw.get("values")
    if not isinstance(values, dict) or set(values) != {name}:
        raise ValueError(f"{autopilot}.flight_parameters.values must contain only {name}")
    value = values[name]
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or not 0.1 <= value / multiplier <= 20):
        raise ValueError(f"{name}: horizontal speed must be in [0.1, 20] m/s in native units")
    value = struct.unpack("<f", struct.pack("<f", value))[0]
    if "source" in raw and not isinstance(raw["source"], str):
        raise ValueError("flight_parameters.source must be a string")
    if "parameter_file" in raw:
        path = raw["parameter_file"]
        if (not isinstance(path, str) or not path.strip() or Path(path).is_absolute()
                or ".." in Path(path).parts):
            raise ValueError("flight_parameters.parameter_file must be relative to the run directory")
    if "parameter_file_sha256" in raw:
        digest = raw["parameter_file_sha256"]
        if ("parameter_file" not in raw or not isinstance(digest, str) or len(digest) != 64
                or any(c not in "0123456789abcdef" for c in digest)):
            raise ValueError("flight_parameters.parameter_file_sha256 requires a file and SHA256 digest")
    return dict(raw, values={name: value})


def speed_defaults(speed_m_s, autopilot, *, source):
    name, multiplier = SPEED_PARAMETERS[autopilot]
    return parse_flight_parameters(dict(values={name: speed_m_s * multiplier}, source=source), autopilot)


def validate_flight_parameter_file(config, autopilot, run_dir):
    config = parse_flight_parameters(config, autopilot)
    if config is None:
        return None
    content = encode_parameter_values(config["values"], autopilot)
    if "parameter_file" in config:
        path = Path(run_dir) / config["parameter_file"]
        actual = path.read_bytes()
        if "parameter_file_sha256" in config and hashlib.sha256(actual).hexdigest() != config["parameter_file_sha256"]:
            raise ValueError(f"{path}: flight parameter file SHA256 mismatch")
        if actual != content:
            raise ValueError(f"{path}: file does not match flight_parameters.values; regenerate the file")
    return content


def write_flight_parameter_file(config, autopilot, run_dir):
    config = parse_flight_parameters(config, autopilot)
    content = encode_parameter_values(config["values"], autopilot)
    relative = Path("parameters") / FILENAMES[autopilot]
    path = Path(run_dir) / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return dict(config, parameter_file=relative.as_posix(),
                parameter_file_sha256=hashlib.sha256(content).hexdigest())
