"""Reuse matching successful runs and preserve superseded or failed attempts."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import time

import yaml

from tools.commander.fc_parameter_profiles import parse_controller_parameters
from tools.launcher.startup_parameters import validate_parameter_file
from tools.launcher.gain_environment import check_execution_environment


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_result(logs_dir: Path, result: dict) -> None:
    logs_dir.mkdir(parents=True, exist_ok=True)
    temporary = logs_dir / ".result.json.tmp"
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(logs_dir / "result.json")


def _read_result(logs_dir: Path) -> dict | None:
    try:
        result = json.loads((logs_dir / "result.json").read_text(encoding="utf-8"))
        return result if isinstance(result, dict) else {"status": "incomplete"}
    except FileNotFoundError:
        return None
    except (ValueError, UnicodeError):
        return {"status": "incomplete"}


def flight_logs(logs_dir: Path, *, collected_only: bool = False) -> list[Path]:
    """Find flight records without mistaking ArduPilot's eeprom.bin for a log."""
    if not logs_dir.exists():
        return []
    if logs_dir.name == "px4_logs":
        files = list(logs_dir.glob("*.ulg"))
        roots = [] if collected_only else [logs_dir / "rootfs/log"]
        suffix = ".ulg"
    else:
        files = list(logs_dir.glob("*.BIN"))
        roots = [logs_dir / "logs", logs_dir / "raw/logs"]
        suffix = ".bin"
    for root in roots:
        if root.is_dir() and not root.is_symlink():
            files.extend(p for p in root.rglob("*") if p.suffix.lower() == suffix)
    return sorted({p for p in files if p.is_file() and not p.is_symlink()})


def capture_run_inputs(run_dir: Path, autopilot: str, launch_config=None, *, headless: bool | None = None) -> dict:
    """Identify the frozen scenario, validated boot gains, and effective CLI settings."""
    scenario = (run_dir / "scenario.yaml").read_bytes()
    data = yaml.safe_load(scenario) or {}
    if not isinstance(data, dict) or not isinstance(data.get("autopilots", {}), dict):
        raise ValueError("scenario.yaml and autopilots must be mappings")
    settings = data.get("autopilots", {}).get(autopilot, {})
    if not isinstance(settings, dict):
        raise ValueError(f"autopilots.{autopilot} must be a mapping")
    gains = parse_controller_parameters(settings.get("controller_gains"), autopilot)
    parameters = validate_parameter_file(gains, autopilot, run_dir)
    config = asdict(launch_config) if is_dataclass(launch_config) else dict(launch_config or {})
    # Retry count and time between attempts do not change an individual flight.
    config.pop("max_retries", None)
    if headless is not None:
        config["headless"] = headless
    config = json.loads(json.dumps(config, sort_keys=True, default=str, allow_nan=False))
    inputs = dict(version=1, autopilot=autopilot,
                  scenario_sha256=hashlib.sha256(scenario).hexdigest(),
                  parameters_sha256=None if parameters is None else hashlib.sha256(parameters).hexdigest(),
                  launch_config=config)
    environment_check = check_execution_environment(data, autopilot, gains, config)
    if environment_check is not None:
        inputs["gain_environment"] = environment_check
    encoded = json.dumps(inputs, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return dict(inputs, sha256=hashlib.sha256(encoded).hexdigest())


def _flight_log_manifest(logs_dir: Path) -> list[dict]:
    """Require nonempty collected logs and detect later replacement/truncation."""
    manifest = []
    for path in flight_logs(logs_dir, collected_only=True):
        size = path.stat().st_size
        if not size:
            return []
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        manifest.append(dict(path=str(path.relative_to(logs_dir)), size=size, sha256=digest.hexdigest()))
    return manifest


def _copy_inputs(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("scenario.yaml", "inputs.json"):
        path = source / name
        if path.is_file():
            shutil.copy2(path, destination / name)
    for name in ("parameters", "generated"):
        path = source / name
        if path.is_dir():
            shutil.copytree(path, destination / name, symlinks=True)


class AttemptLogs:
    """Create a session lazily, only when an attempt needs to be archived."""

    def __init__(self, logs_dir: Path, *, inputs: dict | None = None):
        self.logs_dir = logs_dir
        self.run_dir = logs_dir.parent
        self.autopilot = logs_dir.name.removesuffix("_logs")
        self.inputs = inputs
        self.snapshot = logs_dir / "run_inputs"
        self.session_dir: Path | None = None
        self.result: dict = {}
        self.started_monotonic = 0.0

    def begin(self, attempt: int) -> None:
        autopilot = "ardupilot" if self.autopilot == "ardu" else self.autopilot
        current = capture_run_inputs(self.run_dir, autopilot,
                                     self.inputs["launch_config"] if self.inputs else None)
        if self.inputs is not None and current != self.inputs:
            raise ValueError("Run inputs changed after validation; reload the scenario before starting")
        self.inputs = current
        if self.snapshot.exists():
            shutil.rmtree(self.snapshot)
        _copy_inputs(self.run_dir, self.snapshot)
        (self.snapshot / "inputs.json").write_text(json.dumps(self.inputs, indent=2) + "\n", encoding="utf-8")
        self.started_monotonic = time.monotonic()
        self.result = {
            "autopilot": self.autopilot,
            "attempt": attempt,
            "status": "running",
            "started_at": _utc_now(),
            "inputs": self.inputs,
        }
        _write_result(self.logs_dir, self.result)
        if "gain_environment" in self.inputs:
            report = self.inputs["gain_environment"]
            (self.logs_dir / "gain_environment.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            print(f"[GAIN ENV] {self.autopilot}: {report['comparison']['status']} ({report['policy']})")

    def finish(self, exit_code: int, log_collected: bool, *, interrupted: bool = False,
               collection_error: str | None = None) -> bool:
        if interrupted or exit_code == 130:
            status = "interrupted"
        elif exit_code == 124:
            status = "timeout"
        elif exit_code != 0:
            status = "failed"
        elif not log_collected:
            status = "missing_log"
        else:
            manifest = _flight_log_manifest(self.logs_dir)
            status = "success" if manifest else "missing_log"
            self.result["flight_log_manifest"] = manifest
            try:
                current = capture_run_inputs(self.run_dir, self.inputs["autopilot"], self.inputs["launch_config"])
                if current != self.inputs:
                    raise ValueError("Run inputs changed during the attempt")
            except (OSError, ValueError, yaml.YAMLError) as exc:
                status = "inputs_changed"
                self.result["input_error"] = str(exc)
        # PX4 can materialize vehicle files during run_once, after begin().
        generated = self.run_dir / "generated"
        if generated.is_dir():
            saved = self.snapshot / "generated"
            if saved.exists():
                shutil.rmtree(saved)
            shutil.copytree(generated, saved, symlinks=True)
        self.result.update(
            status=status, exit_code=exit_code, log_collected=log_collected,
            ended_at=_utc_now(), elapsed_s=round(time.monotonic() - self.started_monotonic, 3),
            flight_logs=[str(p.relative_to(self.logs_dir)) for p in flight_logs(self.logs_dir)],
        )
        if collection_error is not None:
            self.result["collection_error"] = collection_error
        _write_result(self.logs_dir, self.result)
        if status == "success":
            return True
        self.archive(self.result)
        return False

    def archive(self, result: dict) -> None:
        """Copy first; only remove active flight/log outputs after a complete copy."""
        if self.session_dir is None:
            archive_root = self.run_dir / f"{self.autopilot}_attempts"
            archive_root.mkdir(parents=True, exist_ok=True)
            prefix = datetime.now(timezone.utc).strftime("session_%Y%m%dT%H%M%SZ_")
            self.session_dir = Path(tempfile.mkdtemp(prefix=prefix, dir=archive_root))
            if (self.snapshot / "scenario.yaml").is_file():
                _copy_inputs(self.snapshot, self.session_dir)
            elif "recovered_at" not in result:
                _copy_inputs(self.run_dir, self.session_dir)
            # Legacy recovered output has no trustworthy input snapshot. Do not
            # label old flight logs with the newly requested scenario/gains.

        destination = self.session_dir / f"attempt_{result.get('attempt', 0):03d}"
        pending = destination.with_name(destination.name + ".partial")
        # Preserve runtime state as evidence, but never follow firmware symlinks.
        shutil.copytree(self.logs_dir, pending, symlinks=True)
        pending.rename(destination)
        result["archived_to"] = str(destination.relative_to(self.run_dir))
        _write_result(self.logs_dir, result)
        print(f"[ARCHIVE] {result['status']}: {destination}")

        _clear_attempt_outputs(self.logs_dir)


def _clear_attempt_outputs(logs_dir: Path) -> None:
    # Keep runtime storage; launchers reapply configured boot parameters on each attempt.
    outputs = flight_logs(logs_dir)
    outputs.extend(p for p in logs_dir.iterdir()
                   if p.is_file() and not p.is_symlink()
                   and (p.name.startswith("boot_parameters.") or p.suffix.lower() in (".log", ".tlog") or p.name.endswith(".tlog.raw")
                        or p.name in ("wind.json", "wind_events.jsonl", "wind_samples.jsonl", "mission_sequence_map.json",
                                      "controller_gains.json", "controller_gains.json.tmp", "gain_environment.json")))
    for path in set(outputs):
        path.unlink()


def prepare_run_logs(logs_dir: Path, *, force: bool, inputs: dict | None = None) -> bool:
    """Reuse only matching successful runs; archive stale or unfinished output."""
    result = _read_result(logs_dir)
    if not force and result and result.get("status") == "success":
        if inputs is None and (logs_dir.parent / "scenario.yaml").is_file():
            autopilot = "ardupilot" if logs_dir.name == "ardu_logs" else "px4"
            inputs = capture_run_inputs(logs_dir.parent, autopilot)
        if (inputs is not None and result.get("inputs") == inputs
                and result.get("exit_code") == 0 and result.get("log_collected") is True):
            manifest = _flight_log_manifest(logs_dir)
            if manifest and result.get("flight_log_manifest") == manifest:
                return True
        print(f"[RERUN] {logs_dir.parent}: inputs changed or successful result/log evidence is incomplete")

    if logs_dir.exists():
        unfinished = result is not None and result.get("status") != "success"
        stale_success = result is not None and result.get("status") == "success" and not force
        legacy_output = result is None and (flight_logs(logs_dir) or list(logs_dir.glob("*.log")))
        if unfinished or stale_success or legacy_output:
            archived = (logs_dir.parent / result["archived_to"]
                        if result and result.get("archived_to") else None)
            if archived is None or not (archived / "result.json").is_file():
                recovered = dict(result or {"autopilot": logs_dir.name.removesuffix("_logs"), "attempt": 0})
                if recovered.get("status") in (None, "running"):
                    recovered["status"] = "incomplete"
                recovered["recovered_at"] = _utc_now()
                recovered["flight_logs"] = [str(p.relative_to(logs_dir)) for p in flight_logs(logs_dir)]
                if not (logs_dir / "run_inputs/scenario.yaml").is_file():
                    recovered["input_snapshot_missing"] = True
                _write_result(logs_dir, recovered)
                AttemptLogs(logs_dir).archive(recovered)
            else:
                # A previous cleanup may have been interrupted after archiving.
                _clear_attempt_outputs(logs_dir)
        if force:
            print(f"[CLEAN] Removing active logs in {logs_dir}")
            shutil.rmtree(logs_dir)
    return False
