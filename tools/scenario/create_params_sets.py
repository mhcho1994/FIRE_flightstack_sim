"""Create reusable gain profiles and deterministic native SITL parameter files."""

from __future__ import annotations

import argparse
import copy
import hashlib
from pathlib import Path
import struct
import sys

import yaml

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.commander.fc_parameter_profiles import (
    GAIN_GROUPS, load_profiles, normalize_profiles, parameter_names,
    parse_controller_parameters, resolve_profiles,
)


from tools.commander.fc_parameter_registry import profile_metadata, PROFILE_METADATA


DEFAULT_PROFILE = Path(__file__).parent / "fixtures" / "default_controller_gain.yaml"
PARAMETER_FILENAMES = {"ardupilot": "ardupilot.parm", "px4": "px4.bson"}


def create_params_sets(profile=None, *, autopilots=None, groups=None, parameters=None) -> dict:
    """Return a YAML-compatible profile with frozen inline baselines.

    ``profile`` is a path or a mapping; None uses the checkout's captured default profile.
    ``groups`` is a list shared by both stacks or a mapping of stack to list.
    ``parameters`` replaces the rules for each specified stack, allowing P/I/D
    to be configured independently. An empty rule mapping restores its baseline.
    This function never samples or writes files.
    """
    profiles = (normalize_profiles(profile) if isinstance(profile, dict)
                else load_profiles(DEFAULT_PROFILE if profile is None else Path(profile)))
    selected = list(profiles) if autopilots is None else list(autopilots)
    if not selected or len(set(selected)) != len(selected) or set(selected) - set(profiles):
        raise ValueError("autopilots must select distinct stacks present in the profile")
    if parameters is not None and (not isinstance(parameters, dict) or set(parameters) - set(selected)):
        raise ValueError("parameters must map selected autopilots to parameter rules")
    if isinstance(groups, dict) and set(groups) - set(selected):
        raise ValueError("groups contains an unselected autopilot")
    result = {}
    for autopilot in selected:
        entry = profiles[autopilot]
        chosen = groups.get(autopilot, entry["groups"]) if isinstance(groups, dict) else groups
        chosen = entry["groups"] if chosen is None else chosen
        if not isinstance(chosen, (list, tuple)) or not chosen or any(g not in entry["groups"] for g in chosen):
            raise ValueError(f"{autopilot}: groups must select groups present in the profile")
        names = parameter_names(autopilot, chosen, entry.get("parameter_schema"))
        rules = {name: rule for name, rule in entry["parameters"].items() if name in names}
        if parameters is not None and autopilot in parameters:
            rules = parameters[autopilot]
        result[autopilot] = dict(groups=list(chosen),
                                 baseline={name: entry["baseline"][name] for name in names},
                                 parameters=copy.deepcopy(rules), **profile_metadata(entry, autopilot))
    normalized = normalize_profiles(result)
    return {ap: {key: entry[key] for key in ("groups", "baseline", "parameters", *PROFILE_METADATA) if key in entry}
            for ap, entry in normalized.items()}


def save_params_sets(profiles: dict, path: Path, *, provenance: dict | None = None) -> Path:
    """Save a validated profile in the same schema as default_controller_gain.yaml."""
    profiles = create_params_sets(profiles)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = yaml.safe_dump(profiles, sort_keys=False)
    if provenance is None:
        path.write_text(content, encoding="utf-8")
    else:
        from tools.scenario.gain_provenance import publish_derived_profile
        publish_derived_profile(path, content, profiles, provenance)
    return path


def parameter_file_bytes(config: dict, autopilot: str) -> bytes:
    """Encode absolute REAL32 gains as ArduPilot defaults or PX4 native BSON.

    PX4 exports float parameters as BSON doubles containing float32-rounded
    values. Match that format without requiring a MongoDB/BSON dependency.
    """
    config = parse_controller_parameters(config, autopilot)
    values = {name: config["values"][name] for name in parameter_names(autopilot, config["groups"], config.get("parameter_schema"))}
    return encode_parameter_values(values, autopilot, label="controller gains")


def encode_parameter_values(values: dict, autopilot: str, *, label="flight parameters") -> bytes:
    """Encode already validated REAL32 values, shared by gains and flight settings."""
    if autopilot not in PARAMETER_FILENAMES:
        raise ValueError(f"Unsupported autopilot: {autopilot}")
    values = {name: struct.unpack("<f", struct.pack("<f", value))[0] for name, value in values.items()}
    if autopilot == "ardupilot":
        lines = [f"# Generated {label}; applied at SITL boot."]
        lines.extend(f"{name} {value:.9g}" for name, value in values.items())
        return ("\n".join(lines) + "\n").encode("ascii")
    body = b"".join(b"\x01" + name.encode("ascii") + b"\x00" + struct.pack("<d", value)
                    for name, value in values.items())
    return struct.pack("<i", len(body) + 5) + body + b"\x00"


def write_parameter_files(resolved: dict, run_dir: Path) -> dict:
    """Write native files and return scenario gain blocks with relative paths/hashes."""
    prepared = {}
    for autopilot, config in resolved.items():
        if autopilot not in PARAMETER_FILENAMES:
            raise ValueError(f"Unsupported autopilot: {autopilot}")
        clean = {key: value for key, value in config.items()
                 if key not in ("parameter_file", "parameter_file_sha256")}
        clean = parse_controller_parameters(clean, autopilot)
        prepared[autopilot] = (clean, parameter_file_bytes(clean, autopilot))
    result = {}
    for autopilot, (config, content) in prepared.items():
        relative = Path("parameters") / PARAMETER_FILENAMES[autopilot]
        path = Path(run_dir) / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        result[autopilot] = dict(config, parameter_file=relative.as_posix(),
                                parameter_file_sha256=hashlib.sha256(content).hexdigest())
    return result


def resolve_params_sets(profiles: dict, *, seed: int, run_id: int, run_dir: Path | None = None) -> dict:
    """Sample a profile once; optionally export native files for the resolved set."""
    for name, value in (("seed", seed), ("run_id", run_id)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    resolved = resolve_profiles(normalize_profiles(profiles), seed, run_id)
    return write_parameter_files(resolved, run_dir) if run_dir is not None else resolved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, help="Input profile; defaults to fixtures/default_controller_gain.yaml beside this script")
    parser.add_argument("--preserve-gain-provenance", action="store_true",
                        help="Validate capture metadata and preserve it with --output; "
                             "does not compare SITL environments (default: disabled)")
    parser.add_argument("--output", type=Path, help="Write a reusable YAML profile")
    parser.add_argument("--params-dir", type=Path, help="Write parameters/ and controller_gains.resolved.yaml here")
    parser.add_argument("--autopilots", nargs="+", choices=tuple(GAIN_GROUPS))
    parser.add_argument("--groups", nargs="+", choices=tuple(dict.fromkeys(g for groups in GAIN_GROUPS.values() for g in groups)))
    parser.add_argument("--baseline-only", action="store_true", help="Clear input rules before adding overrides")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-id", type=int, default=0)
    for operation in ("set", "scale", "offset"):
        parser.add_argument(f"--{operation}", nargs=2, action="append", default=[], metavar=("PARAM", "VALUE"))
        parser.add_argument(f"--random-{operation}", nargs=3, action="append", default=[], metavar=("PARAM", "LOW", "HIGH"))
    args = parser.parse_args()
    if args.output is None and args.params_dir is None:
        parser.error("Provide --output and/or --params-dir")
    try:
        profiles = create_params_sets(args.profile, autopilots=args.autopilots, groups=args.groups)
        provenance = None
        if args.preserve_gain_provenance:
            from tools.scenario.gain_provenance import load_capture_provenance
            provenance = load_capture_provenance(args.profile or DEFAULT_PROFILE, profiles)
        if args.baseline_only:
            for entry in profiles.values():
                entry["parameters"] = {}
        seen = set()
        for operation in ("set", "scale", "offset"):
            for random in (False, True):
                for name, *numbers in getattr(args, f"random_{operation}" if random else operation):
                    stacks = [ap for ap, entry in profiles.items() if name in entry["baseline"]]
                    if len(stacks) != 1 or name in seen:
                        raise ValueError(f"{name}: select a supported parameter and specify only one override")
                    seen.add(name)
                    rule = dict(operation=operation)
                    rule["uniform" if random else "value"] = ([float(n) for n in numbers] if random else float(numbers[0]))
                    profiles[stacks[0]]["parameters"][name] = rule
        profiles = create_params_sets(profiles)
        resolved = resolve_params_sets(profiles, seed=args.seed, run_id=args.run_id)
        if args.output is not None:
            print(save_params_sets(profiles, args.output, provenance=provenance))
        if args.params_dir is not None:
            resolved = write_parameter_files(resolved, args.params_dir)
            path = args.params_dir / "controller_gains.resolved.yaml"
            path.write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")
            print(path)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
