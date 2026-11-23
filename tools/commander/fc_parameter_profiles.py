"""Controller gain profiles, reproducible sampling, and resolved scenario validation.

Profiles are resolved by the scenario generator. Mission runners only accept
absolute values and never sample or multiply the vehicle's persisted values.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import random
import struct

import yaml


from tools.commander.fc_parameter_registry import (
    GAIN_GROUPS, parameter_names, schema_groups, profile_metadata, parameter_spec, validate_parameter, PROFILE_METADATA,
)
# Version the sampling recipe; all controller groups use the same namespace.
SAMPLING_NAMESPACE = "controller-gains-v1"
SAMPLING = (f"{SAMPLING_NAMESPACE}: SHA256(namespace, seed, run_id, autopilot, parameter); "
            "independent Python Random.uniform")


def _selected_groups(value, autopilot, schema=None):
    # An omitted groups field is the original rate-only schema, including
    # snapshots that also contain other controller parameters.
    if not isinstance(value, (list, tuple)) or not value or any(not isinstance(g, str) for g in value):
        raise ValueError("groups must be a non-empty list of controller group names")
    if len(set(value)) != len(value):
        raise ValueError("groups must not contain duplicates")
    _keys(dict.fromkeys(value), schema_groups(autopilot, schema), "groups")
    return [group for group in schema_groups(autopilot, schema) if group in value]


def _mapping(value, label):
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")
    return value


def _keys(value, allowed, label):
    unknown = set(value) - set(allowed)
    if unknown:
        raise ValueError(f"{label}: unknown keys {sorted(map(str, unknown))}")


def _number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError(f"{label} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number")
    return result


def _gain(value, label):
    value = _number(value, label)
    try:
        encoded = struct.unpack("<f", struct.pack("<f", value))[0]
    except OverflowError as exc:
        raise ValueError(f"{label} exceeds float32 range") from exc
    if value < 0 or (value != 0 and encoded == 0):
        raise ValueError(f"{label} must be non-negative and representable as float32")
    return value


def _parameter(value, name, autopilot):
    if autopilot == 'ardupilot':
        from tools.commander.fc_parameter_registry import ARDU46, ARDU47
        if any(name in names for registry in (ARDU46, ARDU47) for names in registry.values()):
            return _gain(value, name)
    if parameter_spec(autopilot, name):
        value = validate_parameter(autopilot, name, value)
        return value if isinstance(value, int) else struct.unpack('<f', struct.pack('<f', value))[0]
    return _gain(value, name)


def _parameters(value, autopilot, groups, label, *, complete=False, schema=None):
    value = _mapping(value, label)
    names = parameter_names(autopilot, groups, schema)
    _keys(value, names, label)
    if complete and set(value) != set(names):
        raise ValueError(f"{label}: all {len(names)} gain values for groups {groups} are required; missing {sorted(set(names) - set(value))}")
    return {name: _parameter(number, name, autopilot) for name, number in value.items()}


def _digest(baseline):
    return hashlib.sha256(json.dumps(baseline, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _pair(value, label):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{label} must be [minimum, maximum]")
    low, high = (_number(v, label) for v in value)
    if not math.isfinite(high - low):
        raise ValueError(f"{label}: range width must be finite")
    if low > high:
        raise ValueError(f"{label}: minimum exceeds maximum")
    return [low, high]


def _transform(baseline, operation, value):
    if operation == "set":
        return value
    if operation == "scale":
        return baseline * value
    return baseline + value


def _read_baseline(path, names):
    """Read a MAVProxy NAME VALUE snapshot, selecting only the requested controller groups."""
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split("#", 1)[0].replace(",", " ").split()
        if not fields or fields[0] not in names:
            continue
        if len(fields) != 2 or fields[0] in result:
            raise ValueError(f"{path}: invalid or duplicate controller gain line: {line}")
        result[fields[0]] = float(fields[1])
    return result


def load_profiles(path: Path | None) -> dict:
    """Load and validate every rule before creating any dataset output."""
    if path is None:
        return {}
    path = Path(path).resolve()
    return normalize_profiles(yaml.safe_load(path.read_text(encoding="utf-8")),
                              base_dir=path.parent, source=str(path))


def normalize_profiles(raw: dict, *, base_dir: Path | None = None, source="<mapping>") -> dict:
    """Validate a profile supplied by Python or YAML using the same rules."""
    base_dir = Path.cwd() if base_dir is None else Path(base_dir)
    raw = _mapping(raw, "controller gain profile")
    _keys(raw, GAIN_GROUPS, "controller gain profile")
    profiles = {}
    for autopilot, entry in raw.items():
        entry = _mapping(entry, autopilot)
        _keys(entry, ("groups", "baseline", "parameters", "group_parameters", *PROFILE_METADATA), autopilot)
        metadata = profile_metadata(entry, autopilot)
        schema = entry.get("parameter_schema")
        groups = _selected_groups(entry.get("groups", ["rate"]), autopilot, schema)
        baseline_source = entry.get("baseline")
        if isinstance(baseline_source, str):
            baseline_path = (base_dir / baseline_source).resolve()
            baseline = _read_baseline(baseline_path, parameter_names(autopilot, groups, schema))
            baseline_source = str(baseline_path)
        else:
            baseline = baseline_source
            baseline_source = f"{source}#{autopilot}.baseline"
        baseline = _parameters(baseline, autopilot, groups, "baseline", complete=True, schema=schema)
        rules = {}
        group_rules = _mapping(entry.get('group_parameters', {}), 'group_parameters')
        _keys(group_rules, groups, 'group_parameters')
        for group, rule in group_rules.items():
            if rule == 'fixed':
                continue
            for name in parameter_names(autopilot, [group], schema):
                if name in rules and rules[name] != rule:
                    raise ValueError(f"{name}: conflicting rules in overlapping groups")
                rules[name] = rule
        rules.update(_mapping(entry.get("parameters", {}), "parameters"))
        _keys(rules, parameter_names(autopilot, groups, schema), "parameters")
        normalized = {}
        for name, rule in rules.items():
            rule = _mapping(rule, name)
            _keys(rule, ("operation", "value", "uniform", "choices", "bounds"), name)
            operation = rule.get("operation")
            if operation not in ("set", "scale", "offset"):
                raise ValueError(f"{name}.operation must be set, scale, or offset")
            if sum(k in rule for k in ("value", "uniform", "choices")) != 1:
                raise ValueError(f"{name}: specify exactly one of value, uniform or choices")
            normalized[name] = {"operation": operation}
            if 'choices' in rule:
                if operation != 'set' or not isinstance(rule['choices'], list) or not rule['choices']:
                    raise ValueError(f'{name}: choices requires set and a nonempty list')
                normalized[name]['choices'] = [_parameter(v, name, autopilot) for v in rule['choices']]
                limits = normalized[name]['choices']
            elif "uniform" in rule:
                if parameter_spec(autopilot, name).get('type', '').startswith('Int'):
                    raise ValueError(f'{name}: use discrete choices for INT32 variation')
                limits = _pair(rule["uniform"], f"{name}.uniform")
                normalized[name]["uniform"] = limits
            else:
                value = _number(rule["value"], f"{name}.value")
                normalized[name]["value"] = value
                limits = [value, value]
            if operation == "scale" and limits[0] < 0:
                raise ValueError(f"{name}: scale must be non-negative")
            bounds = _pair(rule["bounds"], f"{name}.bounds") if "bounds" in rule else None
            if bounds is not None:
                normalized[name]["bounds"] = bounds
            for value in limits:
                final = _parameter(_transform(baseline[name], operation, value), name, autopilot)
                if bounds is not None and not bounds[0] <= final <= bounds[1]:
                    raise ValueError(f"{name}: transformed range exceeds final gain bounds {bounds}")
        profiles[autopilot] = dict(groups=groups, baseline=baseline, baseline_sha256=_digest(baseline),
                                   baseline_source=baseline_source, parameters=normalized, **metadata)
    return profiles


def resolve_profiles(profiles: dict, seed: int, run_id: int) -> dict:
    resolved = {}
    for autopilot, profile in profiles.items():
        values = dict(profile["baseline"])
        draws = {}
        for name, rule in profile["parameters"].items():
            if "uniform" in rule or "choices" in rule:
                key = json.dumps([SAMPLING_NAMESPACE, seed, run_id, autopilot, name]).encode()
                rng = random.Random(int.from_bytes(hashlib.sha256(key).digest(), "big"))
                draw = rng.choice(rule["choices"]) if "choices" in rule else rng.uniform(*rule["uniform"])
            else:
                draw = rule["value"]
            draws[name] = draw
            values[name] = _parameter(_transform(profile["baseline"][name], rule["operation"], draw), name, autopilot)
        resolved[autopilot] = dict(
            groups=list(profile["groups"]), baseline=dict(profile["baseline"]), baseline_sha256=profile["baseline_sha256"],
            baseline_source=profile["baseline_source"], values=values,
            sampling=dict(method=SAMPLING, seed=seed, run_id=run_id,
                          rules=profile["parameters"], draws=draws), **profile_metadata(profile, autopilot))
    return resolved


def parse_controller_parameters(raw, autopilot: str) -> dict | None:
    """Validate absolute scenario values; omitted values restore the baseline."""
    if raw is None:
        return None
    raw = _mapping(raw, "parameters")
    _keys(raw, ("groups", "baseline", "values", "baseline_sha256", "baseline_source", "sampling",
                "parameter_file", "parameter_file_sha256", *PROFILE_METADATA), "parameters")
    profile_metadata(raw, autopilot)
    schema = raw.get("parameter_schema")
    if "parameter_file" in raw:
        if not isinstance(raw["parameter_file"], str) or not raw["parameter_file"].strip():
            raise ValueError("parameters.parameter_file must be a non-empty path")
    if "parameter_file_sha256" in raw:
        digest = raw["parameter_file_sha256"]
        if "parameter_file" not in raw or not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("parameters.parameter_file_sha256 requires a file and a SHA256 hex digest")
    groups = _selected_groups(raw.get("groups", ["rate"]), autopilot, schema)
    baseline = _parameters(raw.get("baseline"), autopilot, groups, "parameters.baseline", complete=True, schema=schema)
    digest = _digest(baseline)
    if raw.get("baseline_sha256", digest) != digest:
        raise ValueError("parameters.baseline_sha256 does not match the baseline")
    values = dict(baseline)
    values.update(_parameters(raw.get("values", {}), autopilot, groups, "parameters.values", schema=schema))
    return dict(raw, groups=groups, baseline=baseline, baseline_sha256=digest, values=values)
