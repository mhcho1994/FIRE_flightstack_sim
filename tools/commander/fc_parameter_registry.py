"""Native controller parameter schemas. Legacy profiles keep their original units."""

import re
import json
import math
import struct
from pathlib import Path

ARDU46 = {
    "rate": tuple(f"ATC_RAT_{axis}_{term}" for axis in ("RLL", "PIT", "YAW") for term in "PID"),
    "attitude": tuple(f"ATC_ANG_{axis}_P" for axis in ("RLL", "PIT", "YAW")),
    "position": ("PSC_POSXY_P", "PSC_POSZ_P"),
    "velocity": tuple(f"PSC_VEL{axis}_{term}" for axis in ("XY", "Z") for term in "PID"),
    "acceleration": tuple(f"PSC_ACCZ_{term}" for term in "PID"),
}
ARDU47 = dict(ARDU46,
    position=("PSC_NE_POS_P", "PSC_D_POS_P"),
    velocity=tuple(f"PSC_{axis}_VEL_{term}" for axis in ("NE", "D") for term in "PID"),
    acceleration=tuple(f"PSC_D_ACC_{term}" for term in "PID"))
PX4_MC = {
    "rate": tuple(f"MC_{axis}RATE_{term}" for axis in ("ROLL", "PITCH", "YAW") for term in "PID"),
    "attitude": tuple(f"MC_{axis}_P" for axis in ("ROLL", "PITCH", "YAW")),
    "position": ("MPC_XY_P", "MPC_Z_P"),
    "velocity": tuple(f"MPC_{axis}_VEL_{term}_ACC" for axis in ("XY", "Z") for term in "PID"),
}
DEFAULT_GROUPS = {"px4": tuple(PX4_MC), "ardupilot": tuple(ARDU46)}
LEGACY_PX4_MC = dict(PX4_MC)
PX4_CATALOG = json.loads(Path(__file__).with_name('px4_parameter_catalog.json').read_text())
ARDUPILOT_CATALOG = json.loads(Path(__file__).with_name('ardupilot_parameter_catalog.json').read_text())
ARDU47_EXTENDED = dict(ARDU47, **{group: tuple(names) for group, names in ARDUPILOT_CATALOG['groups'].items()})
PX4_MC.update({group: tuple(names) for group, names in PX4_CATALOG['groups'].items()})


def parameter_spec(autopilot, name):
    catalog = PX4_CATALOG if autopilot == 'px4' else ARDUPILOT_CATALOG if autopilot == 'ardupilot' else {}
    return catalog.get('parameters', {}).get(name, {})


def validate_parameter(autopilot, name, value):
    spec = parameter_spec(autopilot, name)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{name}: expected finite numeric value')
    if spec.get('type', '').startswith('Int'):
        bits = int(spec['type'][3:])
        if value != int(value) or not -(2**(bits-1)) <= value < 2**(bits-1):
            raise ValueError(f"{name}: expected {spec['type']}")
        value = int(value)
    if spec.get('values') and value not in [float(v['value']) for v in spec['values']]:
        raise ValueError(f'{name}: unsupported enum value {value}')
    if spec.get('bitmask'):
        mask = sum(1 << int(b['index']) for b in spec['bitmask'])
        if int(value) < 0 or int(value) & ~mask:
            raise ValueError(f'{name}: unsupported bitmask bits')
    for bound, op in [('min', lambda a,b: a < b), ('max', lambda a,b: a > b)]:
        if not spec.get('values') and bound in spec and op(value, struct.unpack('<f', struct.pack('<f', float(spec[bound])))[0]):
            raise ValueError(f'{name}: outside {bound}={spec[bound]}')
    return value


SCHEMAS = {"arducopter-4.6": ("ardupilot", ARDU46),
           "arducopter-4.7": ("ardupilot", ARDU47_EXTENDED), "px4-mc-v1": ("px4", PX4_MC)}
LEGACY_SCHEMAS = {"ardupilot": "arducopter-4.6", "px4": "px4-mc-v1"}
# Compatibility for callers selecting groups, not for firmware detection.
GAIN_GROUPS = {"ardupilot": ARDU46, "px4": LEGACY_PX4_MC}
PROFILE_METADATA = ("parameter_schema", "firmware")


def schema_groups(autopilot, schema=None):
    schema = LEGACY_SCHEMAS.get(autopilot) if schema is None else schema
    if schema not in SCHEMAS or SCHEMAS[schema][0] != autopilot:
        raise ValueError(f"Unsupported {autopilot} parameter schema: {schema!r}")
    return SCHEMAS[schema][1]


def parameter_names(autopilot, groups, schema=None):
    registry = schema_groups(autopilot, schema)
    return tuple(dict.fromkeys(name for group in groups for name in registry[group]))


def schema_for_firmware(identity, explicit=None):
    """Unknown releases require an explicit schema; never infer units from names."""
    ap = identity["autopilot"]
    version = identity.get("schema_version_hint") or identity.get("version") or ""
    match = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)", version)
    family = tuple(map(int, match.groups()[:2])) if match else None
    inferred = None
    if ap == "ardupilot" and identity.get("vehicle", "ArduCopter") != "ArduCopter":
        raise ValueError("Controller schemas support ArduCopter only")
    if ap == "ardupilot" and family in ((4, 6), (4, 7)):
        inferred = f"arducopter-{family[0]}.{family[1]}"
    elif ap == "px4" and family in ((1, 15), (1, 16), (1, 17)):
        inferred = "px4-mc-v1"
    if explicit is not None:
        schema_groups(ap, explicit)
        if inferred is not None and explicit != inferred:
            raise ValueError(f"Explicit schema {explicit} conflicts with firmware {version}")
        return explicit
    if inferred is None:
        raise ValueError(f"Cannot select {ap} parameter schema for firmware {version or 'unknown'} "
                         f"({identity.get('git_hash')}); supply an explicit parameter_schema")
    return inferred


def profile_metadata(entry, autopilot):
    schema_groups(autopilot, entry.get("parameter_schema"))
    firmware = entry.get("firmware")
    if firmware is not None:
        if not isinstance(firmware, dict) or firmware.get("autopilot", autopilot) != autopilot:
            raise ValueError("firmware must be a matching firmware identity mapping")
        # Firmware describes provenance; parameter_schema describes stored units.
        # Old converted captures can legitimately have different source firmware.
    return {key: entry[key] for key in PROFILE_METADATA if key in entry}
