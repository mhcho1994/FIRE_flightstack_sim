"""Explicit, unsampled flight defaults applied through the SITL boot files.

Horizontal speed and explicitly supported REAL32 PX4 navigation settings.
Hardware, calibration and failsafe enable/disable parameters are never copied.
"""

import hashlib
import math
from pathlib import Path
import struct

from tools.scenario.create_params_sets import encode_parameter_values
from tools.commander.fc_parameter_registry import schema_groups


SPEED_PARAMETERS = {"ardupilot": ("WPNAV_SPEED", 100.0), "px4": ("MPC_XY_CRUISE", 1.0)}
# All entries are REAL32 in PX4 1.17. Keep this allowlist separate from gains.
REPLAY_FLOAT_PARAMETERS = {
    'MPC_XY_VEL_MAX': (0.1, 30), 'MPC_ACC_HOR': (0.1, 30),
    'MPC_ACC_HOR_MAX': (0.1, 30), 'MPC_JERK_AUTO': (0.1, 100),
    'MPC_Z_V_AUTO_UP': (0.1, 20), 'MPC_Z_V_AUTO_DN': (0.1, 20),
    'MPC_TKO_SPEED': (0.1, 10), 'MPC_LAND_SPEED': (0.1, 10),
    'MPC_LAND_ALT1': (0., 1000), 'MPC_LAND_ALT2': (0., 1000),
    'RTL_RETURN_ALT': (0., 1000), 'RTL_DESCEND_ALT': (0., 1000),
    'RTL_LAND_DELAY': (-1., 3600), 'RTL_MIN_DIST': (0., 1000),
    'NAV_ACC_RAD': (0.05, 100), 'NAV_MC_ALT_RAD': (0.05, 100),
    'COM_DISARM_LAND': (0.1, 3600),
}
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
    extras = {**REPLAY_FLOAT_PARAMETERS, 'MIS_TAKEOFF_ALT': (0., 1000)} if autopilot == 'px4' and schema == 'px4-mc-v1' else {}
    if not isinstance(values, dict) or name not in values or set(values) - {name} - extras.keys():
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
    clean = {name: value}
    for key in values.keys() - {name}:
        v = values[key]
        lo, hi = extras[key]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not lo <= v <= hi:
            raise ValueError(f'{key}: value outside [{lo}, {hi}]')
        clean[key] = struct.unpack('<f', struct.pack('<f', v))[0]
    return dict(raw, values=clean)


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
