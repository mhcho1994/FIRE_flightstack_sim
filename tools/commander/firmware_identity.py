"""Read firmware identity from the connected, disarmed autopilot."""

import time



def read_firmware_identity(connection, autopilot, *, timeout=8.0, check_stop=lambda: None,
                           on_message=None):
    from pymavlink import mavutil
    from tools.commander.mavlink_parameters import ParameterClient

    client = ParameterClient(connection, timeout=timeout, attempts=1,
                             check_stop=check_stop, on_message=on_message)
    client.require_disarmed(autopilot)
    deadline = time.monotonic() + timeout
    request_at = 0.0
    while time.monotonic() < deadline:
        if time.monotonic() >= request_at:
            connection.mav.command_long_send(connection.target_system, connection.target_component,
                mavutil.mavlink.MAV_CMD_REQUEST_MESSAGE, 0,
                mavutil.mavlink.MAVLINK_MSG_ID_AUTOPILOT_VERSION, 0, 0, 0, 0, 0, 0)
            request_at = time.monotonic() + 1.0
        msg = client._receive(deadline)
        if msg is None or msg.get_type() != "AUTOPILOT_VERSION":
            continue
        release = int(msg.flight_sw_version)
        raw = bytes(msg.flight_custom_version)
        if autopilot == "ardupilot":
            commit = raw.rstrip(b"\0").decode("ascii", errors="strict").lower()
        else:
            # PX4 puts vendor version in the low 24 bits, commit in the upper 40.
            commit = f"{int.from_bytes(raw, 'little'):016x}"[:10]
        return dict(autopilot=autopilot,
                    version=".".join(str((release >> shift) & 255) for shift in (24, 16, 8)),
                    version_type=release & 255, git_hash=commit,
                    identity_source="AUTOPILOT_VERSION")
    raise TimeoutError(f"{autopilot}: no AUTOPILOT_VERSION response")


def verify_firmware_identity(connection, expected, *, check_stop=lambda: None, on_message=None,
                             log_path=None):
    import json
    from pathlib import Path

    report = dict(status="verifying", expected=expected)
    try:
        actual = read_firmware_identity(connection, expected["autopilot"],
                                       check_stop=check_stop, on_message=on_message)
        report["actual"] = actual
        for key in ("autopilot", "version", "version_type"):
            if key in expected and actual[key] != expected[key]:
                raise RuntimeError(f"Firmware {key} mismatch: expected {expected[key]}, got {actual[key]}")
        wanted, observed = expected.get("git_hash", ""), actual["git_hash"]
        if not wanted or not observed or not (wanted.startswith(observed) or observed.startswith(wanted)):
            raise RuntimeError(f"Firmware commit mismatch: expected {wanted}, got {observed}")
        report["status"] = "verified"
        return actual
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if log_path is not None:
            path = Path(log_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
