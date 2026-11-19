"""Optional capture/target environment checks, without booting or changing gains."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import hashlib
import json
import os
from pathlib import Path
import runpy
import subprocess

import yaml

from tools.scenario.gain_provenance import check_baseline, load_capture_provenance

POLICIES = ("off", "record", "require-match")


def validate_policy(policy):
    if policy not in POLICIES:
        raise ValueError(f"gain environment policy must be one of {POLICIES}")
    return policy


def scenario_policy(data, autopilot):
    block = data.get("autopilots", {}).get(autopilot, {}).get("controller_gain_environment")
    if block is None:
        return "off"
    if not isinstance(block, dict):
        raise ValueError("controller_gain_environment must be a mapping")
    return validate_policy(block.get("policy", "off"))


def enabled_for_run(run_dir, autopilot):
    data = yaml.safe_load((Path(run_dir) / "scenario.yaml").read_text()) or {}
    return scenario_policy(data, autopilot) != "off"


def _hash_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _commit(root):
    try:
        result = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                                capture_output=True, text=True, timeout=10)
        return result.stdout.strip() if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def collect_environment(autopilot, settings):
    """Fingerprint the binary and the startup files also recorded by SITL capture.

    Paths and timestamps do not form the identity. No build or simulator is run.
    Missing artifacts are explicit, never treated as a matching environment.
    """
    settings = asdict(settings) if is_dataclass(settings) else dict(settings)
    required = (f"{autopilot}_dir", "vehicle", "frame")
    if any(settings.get(key) in (None, "") for key in required):
        raise ValueError(f"{autopilot}: environment check requires {', '.join(required)}")
    root = Path(os.path.expandvars(str(settings[f"{autopilot}_dir"]))).expanduser().resolve()
    vehicle, frame = str(settings["vehicle"]), str(settings["frame"])
    missing, files = [], []
    if autopilot == "px4":
        binary = root / "build/px4_sitl_default/bin/px4"
        etc = root / "build/px4_sitl_default/etc"
        files = [etc / "init.d-posix/rcS", etc / "init.d/rc.mc_defaults"]
        airframes = sorted((etc / "init.d-posix/airframes").glob(f"{vehicle}_*"))
        if len(airframes) != 1:
            missing.append("exactly one selected PX4 airframe startup file")
        files.extend(airframes)
    elif autopilot == "ardupilot":
        binary = root / "build/sitl/bin" / vehicle.replace("Ardu", "ardu").lower()
        try:
            registry = runpy.run_path(str(root / "Tools/autotest/pysim/vehicleinfo.py"))["VehicleInfo"]().options
            defaults = registry[vehicle]["frames"][frame]["default_params_filename"]
            defaults = [defaults] if isinstance(defaults, str) else defaults
            files = [root / "Tools/autotest" / name for name in defaults]
            if not files:
                missing.append("ArduPilot frame defaults")
        except (OSError, KeyError, TypeError, ValueError, ImportError) as exc:
            missing.append(f"ArduPilot frame defaults: {type(exc).__name__}")
    else:
        raise ValueError(f"Unsupported autopilot: {autopilot}")

    def read(path):
        try:
            return _hash_file(path)
        except OSError:
            missing.append(str(path.relative_to(root)))
            return None

    # Fresh PX4 capture deliberately clears these inherited overrides.
    overrides = ({key: value for key, value in os.environ.items()
                  if key.startswith("PX4_PARAM_") or key in ("PX4_ESTIMATOR", "PX4_SIM_SPEED_FACTOR")}
                 if autopilot == "px4" else {})
    return dict(autopilot=autopilot, vehicle=vehicle, frame=frame, commit=_commit(root),
                binary_sha256=read(binary),
                startup_files={str(p.relative_to(root)): read(p) for p in files},
                startup_overrides=overrides, missing=sorted(missing))


def _captured_environment(provenance, autopilot):
    entry = provenance.get("capture", {})
    firmware = entry.get("firmware", {})
    identity = entry.get("firmware_identity", {})
    if not isinstance(firmware, dict) or not isinstance(identity, dict):
        raise ValueError(f"{autopilot}: invalid firmware provenance")
    startup_files = firmware.get("startup_files", {})
    if not isinstance(startup_files, dict):
        raise ValueError(f"{autopilot}: invalid startup file provenance")
    files = {}
    for path, value in startup_files.items():
        if not isinstance(path, str) or (value is not None and not isinstance(value, str)):
            raise ValueError(f"{autopilot}: invalid startup file hash")
        try:
            relative = Path(path).relative_to(firmware["directory"])
        except (KeyError, TypeError, ValueError):
            continue
        files[str(relative)] = value
    return dict(autopilot=identity.get("autopilot", autopilot),
                vehicle=str(entry["vehicle"]) if "vehicle" in entry else None,
                frame=entry.get("frame"), commit=firmware.get("commit", identity.get("git_hash")),
                binary_sha256=firmware.get("binary_sha256"), startup_files=files,
                startup_overrides={} if provenance.get("source_kind") == "sitl" else None)


def compare_environments(provenance, current, autopilot, *, final=False):
    captured = _captured_environment(provenance, autopilot)
    differences, source_missing, target_missing = [], [], list(current.get("missing", []))
    if provenance.get("status") != "verified":
        source_missing.append("capture metadata")
    for key in ("autopilot", "vehicle", "frame", "binary_sha256", "startup_overrides"):
        old, new = captured.get(key), current.get(key)
        if old is None:
            source_missing.append(key)
        elif new is None:
            target_missing.append(key)
        elif old != new:
            differences.append(dict(field=key, captured=old, target=new))
    old_files, new_files = captured["startup_files"], current.get("startup_files", {})
    if not old_files:
        source_missing.append("startup_files")
    if not new_files:
        target_missing.append("startup_files")
    if old_files and new_files:
        if set(old_files) != set(new_files):
            # A missing build may not yet contain the selected airframe script.
            # Known, different selections remain mismatches even before a build.
            if set(new_files) - set(old_files) or not current.get("missing"):
                differences.append(dict(field="startup_files.selection", captured=sorted(old_files), target=sorted(new_files)))
            else:
                target_missing.extend(f"startup_files.{name}" for name in set(old_files) - set(new_files))
        for name in sorted(set(old_files) & set(new_files)):
            if old_files[name] is None:
                source_missing.append(f"startup_files.{name}")
            elif new_files[name] is None:
                target_missing.append(f"startup_files.{name}")
            elif old_files[name] != new_files[name]:
                differences.append(dict(field=f"startup_files.{name}", captured=old_files[name], target=new_files[name]))
    old_commit, new_commit = captured.get("commit"), current.get("commit")
    if old_commit is not None and not isinstance(old_commit, str):
        raise ValueError(f"{autopilot}: invalid firmware commit provenance")
    if old_commit and new_commit and not (old_commit.startswith(new_commit) or new_commit.startswith(old_commit)):
        differences.append(dict(field="firmware.commit", captured=old_commit, target=new_commit))
    status = ("mismatch" if differences else "unknown" if source_missing else
              ("unknown" if final else "pending") if target_missing else "match")
    return dict(status=status, differences=differences, source_missing=sorted(set(source_missing)),
                target_missing=sorted(set(target_missing)))


def enforce_policy(policy, comparison, *, final=False):
    validate_policy(policy)
    allowed = ("match",) if final else ("match", "pending")
    if policy == "require-match" and comparison["status"] not in allowed:
        raise ValueError("gain environment require-match: " + json.dumps(comparison, sort_keys=True))


def inspect_gain_environments(profile_path, profiles, targets, policy):
    validate_policy(policy)
    if policy == "off":
        return {}
    origins = load_capture_provenance(profile_path, profiles)
    result = {}
    for autopilot in profiles:
        current = collect_environment(autopilot, targets[autopilot])
        comparison = compare_environments(origins[autopilot], current, autopilot)
        enforce_policy(policy, comparison)
        result[autopilot] = dict(policy=policy, provenance=origins[autopilot],
                                planned_environment=current, comparison=comparison)
        print(f"[GAIN ENV] {autopilot}: {comparison['status']} ({policy})")
    return result


def check_execution_environment(data, autopilot, gain_config, launch_config):
    policy = scenario_policy(data, autopilot)
    if policy == "off":
        return None
    if gain_config is None:
        raise ValueError("gain environment policy requires controller_gains")
    entry = data["autopilots"][autopilot]
    origin = entry["controller_gain_environment"].get("provenance")
    check_baseline(origin, gain_config["baseline"], autopilot)
    settings = dict(entry.get("sim", {}))
    settings.update(launch_config)
    current = collect_environment(autopilot, settings)
    comparison = compare_environments(origin, current, autopilot, final=True)
    enforce_policy(policy, comparison, final=True)
    return dict(policy=policy, execution_environment=current, comparison=comparison)
