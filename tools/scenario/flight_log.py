"""Format-safe entry point for native real-flight mission analysis."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from types import ModuleType
from typing import Any


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def detect_log_format(path):
    """Select the decoder by magic bytes, never by the filename extension."""
    with Path(path).open('rb') as stream:
        header = stream.read(16)
    if len(header) >= 16:
        for kind, magic in (('bin', b'\xa3\x95\x80'), ('ulog', b'ULog\x01\x12\x35')):
            if header.startswith(magic):
                return kind
    raise ValueError(f'{Path(path).name}: invalid BIN/ULog log header; '
                     'trajectory CSV/NPZ files do not preserve controller parameters')


@dataclass
class FlightLog:
    path: Path
    sha256: str
    autopilot: str
    log_format: str
    data: Any
    adapter: ModuleType

    def analyze(self, plans):
        from tools.scenario.mission_timeline import analyze_log
        return analyze_log(self, plans)

    def parameter_at(self, name, timestamp):
        return self.adapter.parameter_at(self.data, name, timestamp)

    def positions(self, flight):
        """Yield boot microseconds, latitude, longitude, height above flight home."""
        return self.adapter.positions(self.data, flight)


def read_flight_log(path):
    path = Path(path).expanduser().resolve(strict=True)
    kind = detect_log_format(path)
    if kind == 'bin':
        from tools.scenario import ardupilot_mission as adapter
        autopilot = 'ardupilot'
    else:
        from tools.scenario import px4_mission as adapter
        autopilot = 'px4'
    data, digest = adapter.read_log(path)
    adapter.validate_identity(data)
    return FlightLog(path, digest, autopilot, kind, data, adapter)
