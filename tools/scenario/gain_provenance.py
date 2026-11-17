"""Validate capture provenance only when an environment policy is enabled."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile

import yaml

from tools.commander.fc_parameter_profiles import normalize_profiles


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def check_baseline(provenance: dict, baseline: dict, autopilot: str) -> None:
    if not isinstance(provenance, dict) or provenance.get("status") not in ("verified", "missing"):
        raise ValueError(f"{autopilot}: invalid gain provenance")
    captured = provenance.get("baseline")
    if not isinstance(captured, dict) or any(captured.get(name) != value for name, value in baseline.items()):
        raise ValueError(f"{autopilot}: baseline does not match its capture provenance")
    if provenance["status"] == "verified" and not isinstance(provenance.get("capture"), dict):
        raise ValueError(f"{autopilot}: missing capture provenance snapshot")


def load_capture_provenance(path: Path, selected_profiles: dict) -> dict:
    """Read SITL, flight-log (legacy/current), or derived-profile sidecars.

    Snapshots are self-contained: subsequent launches do not depend on the source
    log, the original firmware checkout, or the original profile still existing.
    """
    path = Path(path).resolve()
    content = path.read_bytes()
    profiles = normalize_profiles(yaml.safe_load(content), base_dir=path.parent, source=str(path))
    meta_path = path.with_suffix(".meta.json")
    if not meta_path.exists():
        origins = {ap: dict(status="missing", source_kind="unknown", source_profile=str(path),
                           source_profile_sha256=digest(content), baseline=profile["baseline"])
                   for ap, profile in profiles.items()}
    else:
        raw_meta = meta_path.read_bytes()
        meta = json.loads(raw_meta)
        if not isinstance(meta, dict) or meta.get("profile_sha256") != digest(content):
            raise ValueError(f"{meta_path}: profile_sha256 does not match the profile")
        derived = meta.get("kind") == "derived_gain_profile"
        if derived:
            if meta.get("schema_version") != 1:
                raise ValueError(f"{meta_path}: unsupported derived profile metadata")
            entries = meta.get("autopilots")
        elif meta.get("status") == "complete":
            entries = meta.get("autopilots")
        elif meta.get("schema_version", 1) == 2:
            entries = meta.get("autopilots")
        elif meta.get("schema_version", 1) == 1 and meta.get("autopilot") in profiles:
            entries = {meta["autopilot"]: meta}
        else:
            raise ValueError(f"{meta_path}: unsupported or incomplete capture metadata")
        if not isinstance(entries, dict) or set(entries) != set(profiles):
            raise ValueError(f"{meta_path}: metadata autopilots do not match the profile")
        origins = {}
        for ap, profile in profiles.items():
            entry = entries[ap]
            if not isinstance(entry, dict) or entry.get("baseline_sha256") != profile["baseline_sha256"]:
                raise ValueError(f"{ap}: metadata baseline_sha256 does not match the profile")
            if "groups" in entry and entry["groups"] != profile["groups"]:
                raise ValueError(f"{ap}: metadata groups do not match the profile")
            if derived:
                origin = copy.deepcopy(entry.get("provenance"))
            else:
                kind = "sitl" if meta.get("status") == "complete" else "flight_log"
                origin = dict(status="verified", source_kind=kind, source_profile=str(path),
                              source_profile_sha256=digest(content), source_metadata_sha256=digest(raw_meta),
                              baseline=profile["baseline"], capture=copy.deepcopy(entry))
            check_baseline(origin, profile["baseline"], ap)
            origins[ap] = origin
    result = {}
    for ap, profile in selected_profiles.items():
        if ap not in origins:
            raise ValueError(f"{ap}: capture provenance is missing")
        check_baseline(origins[ap], profile["baseline"], ap)
        result[ap] = copy.deepcopy(origins[ap])
    return result


def publish_derived_profile(path: Path, content: str, profiles: dict, provenance: dict) -> None:
    """Publish metadata first; the YAML hash detects interrupted pair updates."""
    normalized = normalize_profiles(profiles)
    entries = {}
    for ap, profile in normalized.items():
        check_baseline(provenance[ap], profile["baseline"], ap)
        entries[ap] = dict(groups=profile["groups"], baseline_sha256=profile["baseline_sha256"],
                           provenance=provenance[ap])
    meta = dict(schema_version=1, kind="derived_gain_profile", profile_sha256=digest(content.encode()),
                autopilots=entries)
    for target, text in ((path.with_suffix(".meta.json"), json.dumps(meta, indent=2, allow_nan=False) + "\n"),
                         (path, content)):
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            try:
                stream.write(text)
                stream.close()
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
