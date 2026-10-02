"""Keep unsuccessful SITL attempts outside the normal plotting directories."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import tempfile
import time


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


class AttemptLogs:
    """Create a session lazily, only when an attempt needs to be archived."""

    def __init__(self, logs_dir: Path):
        self.logs_dir = logs_dir
        self.run_dir = logs_dir.parent
        self.autopilot = logs_dir.name.removesuffix("_logs")
        self.session_dir: Path | None = None
        self.result: dict = {}
        self.started_monotonic = 0.0

    def begin(self, attempt: int) -> None:
        self.started_monotonic = time.monotonic()
        self.result = {
            "autopilot": self.autopilot,
            "attempt": attempt,
            "status": "running",
            "started_at": _utc_now(),
        }
        _write_result(self.logs_dir, self.result)

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
            status = "success"
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
            scenario = self.run_dir / "scenario.yaml"
            if scenario.is_file():
                shutil.copy2(scenario, self.session_dir / scenario.name)
            generated = self.run_dir / "generated"
            if generated.is_dir():
                shutil.copytree(generated, self.session_dir / "generated", symlinks=True)

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
    # Keep parameters/dataman/EEPROM for the next attempt, as before.
    outputs = flight_logs(logs_dir)
    outputs.extend(p for p in logs_dir.iterdir()
                   if p.is_file() and not p.is_symlink()
                   and (p.suffix.lower() in (".log", ".tlog") or p.name.endswith(".tlog.raw")))
    for path in set(outputs):
        path.unlink()


def prepare_run_logs(logs_dir: Path, *, force: bool) -> bool:
    """Skip completed runs; recover unfinished output before it can be overwritten."""
    result = _read_result(logs_dir)
    collected = flight_logs(logs_dir, collected_only=True)
    # Keep compatibility with historical logs that predate result.json.
    if not force and collected and (result is None or result.get("status") == "success"):
        return True

    if logs_dir.exists():
        unfinished = result is not None and result.get("status") != "success"
        legacy_partial = result is None and not collected and (flight_logs(logs_dir) or list(logs_dir.glob("*.log")))
        if unfinished or legacy_partial:
            archived = (logs_dir.parent / result["archived_to"]
                        if result and result.get("archived_to") else None)
            if archived is None or not (archived / "result.json").is_file():
                recovered = dict(result or {"autopilot": logs_dir.name.removesuffix("_logs"), "attempt": 0})
                if recovered.get("status") in (None, "running"):
                    recovered["status"] = "incomplete"
                recovered["recovered_at"] = _utc_now()
                recovered["flight_logs"] = [str(p.relative_to(logs_dir)) for p in flight_logs(logs_dir)]
                _write_result(logs_dir, recovered)
                AttemptLogs(logs_dir).archive(recovered)
            else:
                # A previous cleanup may have been interrupted after archiving.
                _clear_attempt_outputs(logs_dir)
        if force:
            print(f"[CLEAN] Removing active logs in {logs_dir}")
            shutil.rmtree(logs_dir)
    return False
