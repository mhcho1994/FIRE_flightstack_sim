#!/usr/bin/env python3
"""
Load SITL autopilot response logs for plotting.

The loader normalizes PX4, ArduPilot, and CogniPilot logs into a common
structure with four time series:

    position: x, y, z
    velocity: vx, vy, vz
    attitude: roll, pitch, yaw
    rate: p, q, r

All angles and rates are returned in radians and radians/second.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

try:
    from pyulog import ULog
except Exception:
    ULog = None

try:
    from pymavlink import mavutil
except Exception:
    mavutil = None


AUTOPILOTS = ("ardu", "px4", "cogni")
AUTOPILOT_ALIASES = {
    "ardupilot": "ardu",
    "ardu": "ardu",
    "ap": "ardu",
    "px4": "px4",
    "cognipilot": "cogni",
    "cogni": "cogni",
    "cp": "cogni",
}


@dataclass(frozen=True)
class TimeSeries:
    """A named vector-valued time series."""

    t: np.ndarray
    values: np.ndarray
    labels: tuple[str, ...]
    boot_t: Optional[np.ndarray] = None

    @classmethod
    def from_boot_time(cls, t, values, labels):
        raw = np.asarray(t, dtype=float)
        return cls(normalize_time(raw), values, labels, raw)


@dataclass(frozen=True)
class AutopilotResponse:
    """Loaded response for one autopilot."""

    autopilot: str
    source_file: Path
    position: Optional[TimeSeries] = None
    velocity: Optional[TimeSeries] = None
    attitude: Optional[TimeSeries] = None
    rate: Optional[TimeSeries] = None


def canonical_autopilot(name: str) -> str:
    key = name.strip().lower()
    if key not in AUTOPILOT_ALIASES:
        raise ValueError(f"Unknown autopilot {name!r}; expected one of {AUTOPILOTS}")
    return AUTOPILOT_ALIASES[key]


def normalize_time(t: np.ndarray) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    finite = np.flatnonzero(np.isfinite(t))
    if finite.size == 0:
        return t
    return t - t[finite[0]]


def align_to_vertical_motion_start(response: AutopilotResponse, threshold_m_s: float = 0.2) -> AutopilotResponse:
    """Shift all available time axes so t=0 is the first vertical motion sample."""
    if response.velocity is None:
        return response

    vz = response.velocity.values[:, 2]
    moving = np.flatnonzero(np.isfinite(vz) & (np.abs(vz) >= threshold_m_s))
    if moving.size == 0:
        return response

    t0 = response.velocity.t[moving[0]]

    def shift(ts: Optional[TimeSeries]) -> Optional[TimeSeries]:
        if ts is None:
            return None
        return TimeSeries(ts.t - t0, ts.values, ts.labels, ts.boot_t)

    return AutopilotResponse(
        autopilot=response.autopilot,
        source_file=response.source_file,
        position=shift(response.position),
        velocity=shift(response.velocity),
        attitude=shift(response.attitude),
        rate=shift(response.rate),
    )


def resolve_log_directory(run_dir: Path, autopilot: str) -> Path:
    """Resolve an active run or an explicitly selected attempt/log directory.

    Never search an entire run recursively: that could select another session.
    """
    run_dir = Path(run_dir)
    autopilot = canonical_autopilot(autopilot)
    nested = run_dir / f"{autopilot}_logs"
    if nested.is_dir():
        return nested

    result_path = run_dir / "result.json"
    if result_path.is_file():
        declared = json.loads(result_path.read_text()).get("autopilot")
        if declared is not None:
            return run_dir if canonical_autopilot(declared) == autopilot else nested

    if run_dir.name == f"{autopilot}_logs" or (
        run_dir.name.startswith("attempt_")
        and run_dir.parent.parent.name == f"{autopilot}_attempts"
    ):
        return run_dir
    return nested


def find_source_file(run_dir: Path, autopilot: str) -> Optional[Path]:
    """Find a raw/processed log in an active run or an explicit attempt."""
    logs_dir = resolve_log_directory(run_dir, autopilot)
    autopilot = canonical_autopilot(autopilot)

    if autopilot == "px4":
        return _find_first_file(
            [
                logs_dir / "raw",
                logs_dir,
                logs_dir / "processed",
            ],
            (".ulg", ".csv"),
        )

    if autopilot == "ardu":
        return _find_first_file(
            [
                logs_dir / "raw" / "logs",
                logs_dir / "logs",
                logs_dir / "raw",
                logs_dir,
                logs_dir / "processed",
            ],
            (".bin", ".BIN", ".csv"),
        )

    if autopilot == "cogni":
        return _find_first_file(
            [
                logs_dir / "raw",
                logs_dir,
                logs_dir / "processed",
            ],
            (".csv", ".ulg", ".bin", ".BIN"),
        )

    return None


def load_run(run_dir: Path, autopilots: Iterable[str]) -> dict[str, AutopilotResponse]:
    """Load all requested autopilots found under a run directory."""
    responses: dict[str, AutopilotResponse] = {}
    for name in autopilots:
        autopilot = canonical_autopilot(name)
        source = find_source_file(run_dir, autopilot)
        if source is None:
            continue
        responses[autopilot] = load_response(source, autopilot)
    return responses


def load_response(path: Path, autopilot: str) -> AutopilotResponse:
    """Load one explicit file for one autopilot."""
    path = Path(path)
    autopilot = canonical_autopilot(autopilot)
    suffix = path.suffix.lower()

    if autopilot == "px4":
        if suffix == ".ulg":
            return read_px4_ulg(path)
        if suffix == ".csv":
            return read_generic_csv(path, autopilot)

    if autopilot == "ardu":
        if suffix == ".bin":
            return read_ardu_bin(path)
        if suffix == ".csv":
            return read_generic_csv(path, autopilot)

    if autopilot == "cogni":
        if suffix == ".csv":
            return read_generic_csv(path, autopilot)
        if suffix == ".ulg":
            loaded = read_px4_ulg(path)
            return AutopilotResponse("cogni", path, loaded.position, loaded.velocity, loaded.attitude, loaded.rate)
        if suffix == ".bin":
            loaded = read_ardu_bin(path)
            return AutopilotResponse("cogni", path, loaded.position, loaded.velocity, loaded.attitude, loaded.rate)

    raise ValueError(f"Unsupported {autopilot} source file: {path}")


def read_px4_ulg(path: Path) -> AutopilotResponse:
    """Read PX4 ULog response topics."""
    if ULog is None:
        raise RuntimeError("pyulog is not installed. Install with: pip install pyulog")

    ulog = ULog(str(path))

    position = velocity = attitude = rate = None

    loc = _get_ulog_data(ulog, "vehicle_local_position")
    if loc is not None:
        t = np.asarray(loc["timestamp"], dtype=float) / 1e6
        position = TimeSeries.from_boot_time(t, _columns(loc, ("x", "y", "z")), ("x", "y", "z"))
        velocity = TimeSeries.from_boot_time(t, _columns(loc, ("vx", "vy", "vz")), ("vx", "vy", "vz"))

    att = _get_ulog_data(ulog, "vehicle_attitude")
    if att is not None:
        t = np.asarray(att["timestamp"], dtype=float) / 1e6
        q = _columns(att, ("q[0]", "q[1]", "q[2]", "q[3]"))
        attitude = TimeSeries.from_boot_time(t, quat_wxyz_to_euler(q), ("roll", "pitch", "yaw"))

    angular = _get_ulog_data(ulog, "vehicle_angular_velocity")
    if angular is not None:
        t = np.asarray(angular["timestamp"], dtype=float) / 1e6
        if all(k in angular for k in ("xyz[0]", "xyz[1]", "xyz[2]")):
            values = _columns(angular, ("xyz[0]", "xyz[1]", "xyz[2]"))
        else:
            values = _columns(angular, ("rollspeed", "pitchspeed", "yawspeed"))
        rate = TimeSeries.from_boot_time(t, values, ("p", "q", "r"))

    return AutopilotResponse("px4", path, position, velocity, attitude, rate)


def read_ardu_bin(path: Path) -> AutopilotResponse:
    """Read ArduPilot BIN response topics."""
    if mavutil is None:
        raise RuntimeError("pymavlink is not installed. Install with: pip install pymavlink")

    position = velocity = attitude = rate = None

    sim_rows = _collect_ardu_rows(
        path,
        ["SIM2"],
        lambda msg: [msg.TimeUS / 1e6, msg.PN, msg.PE, msg.PD, msg.VN, msg.VE, msg.VD],
    )
    if len(sim_rows) < 2:
        sim_rows = _collect_ardu_rows(
            path,
            ["XKF1", "NKF1"],
            lambda msg: [msg.TimeUS / 1e6, msg.PN, msg.PE, msg.PD, msg.VN, msg.VE, msg.VD],
        )
    if len(sim_rows) >= 2:
        arr = np.asarray(sim_rows, dtype=float)
        t = arr[:, 0]
        position = TimeSeries.from_boot_time(t, arr[:, 1:4], ("x", "y", "z"))
        velocity = TimeSeries.from_boot_time(t, arr[:, 4:7], ("vx", "vy", "vz"))

    att_rows = _collect_ardu_rows(path, ["ATT"], _ardu_attitude_row)
    if len(att_rows) >= 2:
        arr = np.asarray(att_rows, dtype=float)
        attitude = TimeSeries.from_boot_time(arr[:, 0], arr[:, 1:4], ("roll", "pitch", "yaw"))

    rate_rows = _collect_ardu_rows(path, ["RATE", "IMU"], _ardu_rate_row)
    if len(rate_rows) >= 2:
        arr = np.asarray(rate_rows, dtype=float)
        rate = TimeSeries.from_boot_time(arr[:, 0], arr[:, 1:4], ("p", "q", "r"))

    return AutopilotResponse("ardu", path, position, velocity, attitude, rate)


def read_generic_csv(path: Path, autopilot: str) -> AutopilotResponse:
    """
    Read a CSV using flexible column aliases.

    Required for plotting:
      - time column: timestamp, TimeUS, time_s, t, or time
      - any subset of position/velocity/attitude/rate columns
    """
    df = pd.read_csv(path)
    time_col = _first_existing(df, ("timestamp", "TimeUS", "time_us", "time_s", "TimeS", "t", "time"))
    if time_col is None:
        raise ValueError(f"{path} does not contain a recognizable time column")

    t = _csv_time_seconds(df[time_col].to_numpy(dtype=float), time_col)

    position = _csv_series(df, t, (("x", "PN", "pos_x", "position_x", "gtx", "x_smooth"),
                                  ("y", "PE", "pos_y", "position_y", "gty", "y_smooth"),
                                  ("z", "PD", "pos_z", "position_z", "gtz", "z_smooth")), ("x", "y", "z"))
    velocity = _csv_series(df, t, (("vx", "VN", "vel_x", "velocity_x", "vx_smooth"),
                                  ("vy", "VE", "vel_y", "velocity_y", "vy_smooth"),
                                  ("vz", "VD", "vel_z", "velocity_z", "vz_smooth")), ("vx", "vy", "vz"))
    attitude = _csv_series(df, t, (("roll", "Roll", "phi"),
                                  ("pitch", "Pitch", "theta"),
                                  ("yaw", "Yaw", "psi")), ("roll", "pitch", "yaw"))
    rate = _csv_series(df, t, (("p", "R", "GyrX", "rollspeed", "angular_velocity_x"),
                              ("q", "P", "GyrY", "pitchspeed", "angular_velocity_y"),
                              ("r", "Y", "GyrZ", "yawspeed", "angular_velocity_z")), ("p", "q", "r"))

    if attitude is not None and _looks_like_degrees(attitude.values):
        attitude = TimeSeries(attitude.t, np.deg2rad(attitude.values), attitude.labels)
    if rate is not None and _looks_like_degrees(rate.values):
        rate = TimeSeries(rate.t, np.deg2rad(rate.values), rate.labels)

    if time_col in ("timestamp", "TimeUS", "time_us"):
        raw_t = df[time_col].to_numpy(dtype=float) / 1e6
        def with_boot(series):
            return TimeSeries(series.t, series.values, series.labels, raw_t) if series else None
        position, velocity, attitude, rate = map(with_boot, (position, velocity, attitude, rate))
    return AutopilotResponse(canonical_autopilot(autopilot), path, position, velocity, attitude, rate)


def quat_wxyz_to_euler(q: np.ndarray) -> np.ndarray:
    """Convert quaternion [w, x, y, z] rows to roll, pitch, yaw."""
    q = np.asarray(q, dtype=float)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sin_pitch = 2.0 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sin_pitch, -1.0, 1.0))
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.column_stack([roll, pitch, yaw])


def _find_first_file(search_dirs: Iterable[Path], suffixes: tuple[str, ...]) -> Optional[Path]:
    suffixes_lower = tuple(s.lower() for s in suffixes)
    for base in search_dirs:
        if not base.exists():
            continue
        files = sorted(
            p for p in base.rglob("*")
            if p.is_file() and p.suffix.lower() in suffixes_lower
            and p.name.lower() != "eeprom.bin"
        )
        if files:
            return files[0]
    return None


def _get_ulog_data(ulog: ULog, name: str) -> Optional[dict]:
    try:
        return ulog.get_dataset(name).data
    except Exception:
        return None


def _columns(data: dict, names: tuple[str, ...]) -> np.ndarray:
    missing = [name for name in names if name not in data]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    return np.column_stack([np.asarray(data[name], dtype=float) for name in names])


def _collect_ardu_rows(path: Path, msg_types: list[str], row_fn) -> list[list[float]]:
    mlog = mavutil.mavlink_connection(str(path))
    rows: list[list[float]] = []
    while True:
        msg = mlog.recv_match(type=msg_types, blocking=False)
        if msg is None:
            break
        row = row_fn(msg)
        if row is not None and all(np.isfinite(row)):
            rows.append(row)
    return rows


def _ardu_attitude_row(msg) -> Optional[list[float]]:
    try:
        return [msg.TimeUS / 1e6, np.deg2rad(msg.Roll), np.deg2rad(msg.Pitch), np.deg2rad(msg.Yaw)]
    except Exception:
        return None


def _ardu_rate_row(msg) -> Optional[list[float]]:
    try:
        if msg.get_type() == "RATE":
            return [msg.TimeUS / 1e6, np.deg2rad(msg.R), np.deg2rad(msg.P), np.deg2rad(msg.Y)]
        return [msg.TimeUS / 1e6, np.deg2rad(msg.GyrX), np.deg2rad(msg.GyrY), np.deg2rad(msg.GyrZ)]
    except Exception:
        return None


def _first_existing(df: pd.DataFrame, candidates: tuple[str, ...]) -> Optional[str]:
    for name in candidates:
        if name in df.columns:
            return name
    return None


def _csv_time_seconds(raw: np.ndarray, column_name: str) -> np.ndarray:
    if column_name.lower() in ("timestamp", "timeus", "time_us") or np.nanmax(raw) > 1e5:
        raw = raw / 1e6
    return normalize_time(raw)


def _csv_series(
    df: pd.DataFrame,
    t: np.ndarray,
    aliases: tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]],
    labels: tuple[str, str, str],
) -> Optional[TimeSeries]:
    cols = [_first_existing(df, group) for group in aliases]
    if any(col is None for col in cols):
        return None
    values = df[[cols[0], cols[1], cols[2]]].to_numpy(dtype=float)
    return TimeSeries(t, values, labels)


def _looks_like_degrees(values: np.ndarray) -> bool:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return False
    return np.nanmax(np.abs(finite)) > (2.0 * np.pi + 0.1)


def crop_to_mission_segment(response: AutopilotResponse, events_path: Path) -> AutopilotResponse:
    """Crop every signal in boot time, then align to the same mission-start event."""
    if not events_path.is_file():
        raise ValueError(f"Missing mission wind events: {events_path}")
    events = [json.loads(line) for line in events_path.read_text().splitlines() if line.strip()]
    boundaries = []
    for name in ("mission_segment_start", "mission_segment_end"):
        matches = [e for e in events if e.get("event") == name]
        if len(matches) != 1 or matches[0].get("boot_s") is None:
            raise ValueError(f"Expected one timestamped {name} in {events_path}")
        boundaries.append(float(matches[0]["boot_s"]))
    start, end = boundaries
    if not np.isfinite([start, end]).all() or end <= start:
        raise ValueError(f"Invalid mission boundaries in {events_path}")
    def crop(series):
        if series is None:
            return None
        if series.boot_t is None:
            raise ValueError("Mission cropping requires boot timestamps (ULog/BIN or timestamp/TimeUS/time_us CSV)")
        mask = (series.boot_t >= start) & (series.boot_t < end)
        if np.count_nonzero(mask) < 2:
            raise ValueError("Mission interval has fewer than two samples; check log/event clock alignment")
        return TimeSeries(series.boot_t[mask] - start, series.values[mask], series.labels, series.boot_t[mask])
    return AutopilotResponse(response.autopilot, response.source_file,
                             *(crop(getattr(response, key)) for key in ("position", "velocity", "attitude", "rate")))
