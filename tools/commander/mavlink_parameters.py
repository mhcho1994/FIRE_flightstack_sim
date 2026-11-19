"""Verified float parameter transactions on a commander's existing connection."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
import struct
import time

from pymavlink import mavutil


def values_match(actual: float, requested: float) -> bool:
    return math.isfinite(actual) and math.isclose(actual, requested, rel_tol=2e-6, abs_tol=1e-9)


class ParameterClient:
    """One receiver, bounded retries, and source/name/type checked responses.

    This client handles REAL32 controller gains only. In particular, it must
    not be used to encode PX4 integer parameters as numeric floats.
    """

    def __init__(self, connection, *, check_stop=lambda: None, on_message=None,
                 timeout=2.0, attempts=3):
        self.m = connection
        self.check_stop = check_stop
        self.on_message = on_message
        self.timeout = timeout
        self.attempts = attempts

    def _receive(self, deadline):
        self.check_stop()
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None
        # Consume on this connection's mission thread only. Preserve startup
        # STATUSTEXT via the observer, e.g. ArduPilot's one-shot EKF messages.
        msg = self.m.recv_match(blocking=True, timeout=min(0.2, remaining))
        if msg is None:
            return None
        if (msg.get_srcSystem(), msg.get_srcComponent()) != (self.m.target_system, self.m.target_component):
            return None
        if self.on_message is not None:
            self.on_message(msg)
        if msg.get_type() == "HEARTBEAT" and msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED:
            raise RuntimeError("Refusing to configure controller gains on an armed vehicle")
        return msg

    def require_disarmed(self, autopilot):
        expected = {"ardupilot": mavutil.mavlink.MAV_AUTOPILOT_ARDUPILOTMEGA,
                    "px4": mavutil.mavlink.MAV_AUTOPILOT_PX4}[autopilot]
        deadline = time.monotonic() + self.timeout * self.attempts
        while time.monotonic() < deadline:
            msg = self._receive(deadline)
            if msg is not None and msg.get_type() == "HEARTBEAT":
                if msg.autopilot != expected:
                    raise RuntimeError(f"Controller gain target is not {autopilot}")
                return
        raise TimeoutError("No target heartbeat while checking controller gain configuration state")

    def _wait_value(self, name, expected=None):
        deadline = time.monotonic() + self.timeout
        last = None
        while time.monotonic() < deadline:
            msg = self._receive(deadline)
            if msg is None or msg.get_type() != "PARAM_VALUE":
                continue
            param_id = msg.param_id
            if isinstance(param_id, bytes):
                param_id = param_id.decode("ascii", errors="replace")
            if param_id.rstrip("\x00") != name:
                continue
            if msg.param_type != mavutil.mavlink.MAV_PARAM_TYPE_REAL32:
                raise ValueError(f"{name}: expected REAL32 parameter, received type {msg.param_type}")
            last = float(msg.param_value)
            if not math.isfinite(last):
                raise ValueError(f"{name}: vehicle returned a nonfinite value")
            if expected is None or values_match(last, expected):
                return last
        detail = f"; last value={last}, requested={expected}" if last is not None else ""
        raise TimeoutError(f"{name}: no matching PARAM_VALUE{detail}")

    def get_param(self, name):
        for attempt in range(self.attempts):
            self.check_stop()
            self.m.mav.param_request_read_send(self.m.target_system, self.m.target_component,
                                             name.encode("ascii"), -1)
            try:
                return self._wait_value(name)
            except TimeoutError:
                if attempt + 1 == self.attempts:
                    raise

    def set_param(self, name, value):
        value = struct.unpack("<f", struct.pack("<f", value))[0]
        if not math.isfinite(value):
            raise ValueError(f"{name}: requested value must be finite")
        for attempt in range(self.attempts):
            self.check_stop()
            self.m.mav.param_set_send(self.m.target_system, self.m.target_component,
                                    name.encode("ascii"), value, mavutil.mavlink.MAV_PARAM_TYPE_REAL32)
            try:
                return self._wait_value(name, expected=value)
            except TimeoutError:
                if attempt + 1 == self.attempts:
                    raise


def verify_controller_parameters(connection, config, autopilot: str, log_path: Path, *,
                     check_stop=lambda: None, on_message=None, parameter_label="controller gains"):
    """Verify boot-applied gains before ARM, without sending any PARAM_SET."""
    if config is None:
        return
    client = ParameterClient(connection, check_stop=check_stop, on_message=on_message)
    report = dict(autopilot=autopilot, status="verifying", method="startup_file", configuration=config,
                  started_at=datetime.now(timezone.utc).isoformat(),
                  verified={}, context={})
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def save():
        temporary = log_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(log_path)

    save()
    try:
        client.require_disarmed(autopilot)
        if autopilot == "px4" and "rate" in config.get("groups", ["rate"]):
            for axis in ("ROLL", "PITCH", "YAW"):
                name = f"MC_{axis}RATE_K"
                report["context"][name] = client.get_param(name)
        if autopilot == "px4" and "attitude" in config.get("groups", []):
            report["context"]["MC_YAW_WEIGHT"] = client.get_param("MC_YAW_WEIGHT")
        # Firmware has already imported the native file during startup.
        for name, value in config["values"].items():
            actual = client.get_param(name)
            report["verified"][name] = actual
            if not values_match(actual, value):
                raise RuntimeError(f"{name}: readback {actual} does not match requested {value}")
        report["status"] = "verified"
        print(f"[PARAMS] Verified {len(report['verified'])} {autopilot} {parameter_label}")
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        save()


def verify_flight_parameters(connection, config, autopilot, log_path, **kwargs):
    """Read back boot flight defaults through the same REAL32 gate as gains."""
    from tools.commander.flight_parameters import parse_flight_parameters
    config = parse_flight_parameters(config, autopilot)
    if config is not None:
        verify_controller_parameters(connection, dict(config, groups=[]), autopilot, log_path,
                                     parameter_label="flight parameters", **kwargs)
