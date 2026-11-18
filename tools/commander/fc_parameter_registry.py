"""Native controller parameter schemas. Legacy profiles keep their original units."""

import re

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
SCHEMAS = {"arducopter-4.6": ("ardupilot", ARDU46),
           "arducopter-4.7": ("ardupilot", ARDU47), "px4-mc-v1": ("px4", PX4_MC)}
LEGACY_SCHEMAS = {"ardupilot": "arducopter-4.6", "px4": "px4-mc-v1"}
# Compatibility for callers selecting groups, not for firmware detection.
GAIN_GROUPS = {"ardupilot": ARDU46, "px4": PX4_MC}
PROFILE_METADATA = ("parameter_schema", "firmware")


def schema_groups(autopilot, schema=None):
    schema = LEGACY_SCHEMAS.get(autopilot) if schema is None else schema
    if schema not in SCHEMAS or SCHEMAS[schema][0] != autopilot:
        raise ValueError(f"Unsupported {autopilot} parameter schema: {schema!r}")
    return SCHEMAS[schema][1]


def parameter_names(autopilot, groups, schema=None):
    registry = schema_groups(autopilot, schema)
    return tuple(name for group in groups for name in registry[group])


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
    elif ap == "px4" and family in ((1, 15), (1, 16)):
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
