#!/usr/bin/env python3
"""PX4 MAVLink adapter for the shared MissionSessionRunner.

Retains the threaded start/stop/status interface used by the Gazebo launcher.
Firmware-specific readiness, upload, mode, arm and takeoff commands live here;
mission phases, parameter verification, wind monitoring and completion are shared.
"""

from __future__ import annotations

import sys
import time
import threading
from enum import Enum, auto
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, List

from pymavlink import mavutil

# Support direct file execution as well as python -m tools.<package>.<module>.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.commander.mission_wind_adapter import WindPolicy, add_wind_checkpoints


class MissionState(Enum):
    """
    Finite state machine describing the mission runner lifecycle.
    """

    IDLE = auto()               # Runner created but not started
    CONNECTING = auto()         # MAVLink connection in progress
    HEARTBEAT_OK = auto()       # Vehicle heartbeat received
    VERIFYING_PARAMETERS = auto()
    CLEARING_MISSION = auto()   # Clear previous mission data
    UPLOADING_MISSION = auto()  # Upload mission data
    MISSION_UPLOADED = auto()   # Mission uploaded
    ARMING = auto()             # Sending arm command
    ARMED = auto()              # Vehicle successfully armed
    SETTING_AUTO = auto()       # Setting auto mode
    STARTING_MISSION = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()             # Error occurred


@dataclass
class MissionStatus:
    """
    Snapshot of mission runner state.

    This object is returned to external callers so they can
    monitor progress without accessing internal variables.
    """

    state: MissionState             # Current FSM state
    message: str = ""               # Human-readable status message
    done: bool = False              # True when runner finished execution
    success: bool = False           # True if mission completed successfully
    error: Optional[str] = None     # Error message if FAILED


def _get(d: Dict[str, Any], path: str, default=None):
    """
    Utility function to retrieve nested dictionary values.

    Example
    -------
    _get(config, "autopilots.px4.mavlink.connect_url")

    Equivalent to:
        config["autopilots"]["px4"]["mavlink"]["connect_url"]

    but safe against missing keys.
    """

    cur: Any = d
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _parse_connect_url(scn: Dict[str, Any]) -> str:
    """
    Convert scenario MAVLink connection URLs into pymavlink format.

    Supported input examples
    ------------------------
    Scenario format:
        udp:127.0.0.1:14550

    Also accepts:
        udp://127.0.0.1:14550
        udpin:127.0.0.1:14550
        tcp:127.0.0.1:5760

    Output examples
    ---------------
    pymavlink format:
        udpin:127.0.0.1:14550

    If the scenario file does not explicitly define a connection URL,
    this function falls back to the default SITL UDP output port.
    """

    url = _get(scn, "autopilots.px4.mavlink.connect_url")

    if isinstance(url, str) and url.strip():
        url = url.strip()

        # Scenario style: udp:127.0.0.1:14550
        if url.startswith("udp:"):
            hostport = url[len("udp:"):]
            return f"udpin:{hostport}"

        # Legacy style: udp://127.0.0.1:14550
        if url.startswith("udp://"):
            hostport = url[len("udp://"):]
            return f"udpin:{hostport}"

        # Already pymavlink-compatible
        if url.startswith(("udpin:", "udpout:", "tcp:", "tcpin:", "tcpout:", "serial:")):
            return url
        
        raise ValueError(
            f"Unsupported MAVLink connect_url format: {url!r}. "
            "Expected forms like 'udp:127.0.0.1:14550' or 'udpin:127.0.0.1:14550'."
        )

    # Default connection port used by SITL
    return f"udpin:127.0.0.1:14550"


def _build_items_from_scenario(scn):
    """Compatibility helper for callers needing logical items without home."""
    from tools.commander.mission_session import legacy_items
    items = legacy_items(scn)
    for item in items:
        if item['p4'] is None:
            item['p4'] = float('nan')
    if WindPolicy.parse(_get(scn, 'common.scenario.wind_policy')).mode == 'mission_only':
        items, _ = add_wind_checkpoints(items)
    return (items, float(_get(scn, 'common.scenario.takeoff_alt_m', 10.)),
            any(it['command'] == 22 for it in items), any(it['command'] == 21 for it in items))


class PX4MissionRunner:
    """Threaded firmware adapter; MissionSessionRunner owns phase lifecycle."""

    def __init__(self, scenario_path: Path, *, wind=None, boot_parameters=None, target_firmware=None):
        """
        Parameters
        ----------
        scenario_path : Path
            Path to scenario.yaml configuration file.
        """

        self.scenario_path = Path(scenario_path)
        self.boot_parameters = boot_parameters
        self.target_firmware = target_firmware
        self.autopilot = "px4"
        self.wind = wind

        # Current mission status
        self.status = MissionStatus(MissionState.IDLE, "initialized")

        # Lock protecting shared status object
        self._lock = threading.Lock()

        # Prevent re-running of mission runner
        self._has_started_run = False

        # Worker thread executing the mission
        self._thread: Optional[threading.Thread] = None

        # External stop request flag
        self._stop_requested = False

        # MAVLink connection handle
        self._m = None

    def _set_status(
        self,
        state: MissionState,
        message: str = "",
        done: bool = False,
        success: bool = False,
        error: Optional[str] = None,
    ) -> None:
        """
        Atomically update mission status.

        A lock is used because status may be read from other threads.
        """

        with self._lock:
            self.status = MissionStatus(
                state=state,
                message=message,
                done=done,
                success=success,
                error=error,
            )

    def get_status(self) -> MissionStatus:
        """
        Return a thread-safe snapshot of the current mission status.
        """

        with self._lock:
            return MissionStatus(
                state=self.status.state,
                message=self.status.message,
                done=self.status.done,
                success=self.status.success,
                error=self.status.error,
            )

    def request_stop(self) -> None:
        """
        Request graceful termination of the mission runner.
        """
        self._stop_requested = True

    def is_done(self) -> bool:
        """
        Check whether the runner has completed execution.
        """
        return self.get_status().done

    def start(self) -> None:
        """
        Start mission execution in a background thread.
        """

        if self._thread is not None:
            raise RuntimeError("Mission runner already started")

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _check_stop(self) -> None:
        """
        Raise exception if an external stop was requested.
        """

        if self._stop_requested:
            raise RuntimeError("Mission runner stop requested")

    def _wait_heartbeat(self, m) -> None:
        """
        Wait until the autopilot sends a MAVLink heartbeat.

        Heartbeat confirms that:
        - MAVLink connection is alive
        - autopilot is running
        """
        self._set_status(MissionState.CONNECTING, "waiting for heartbeat")

        msg = m.recv_match(type="HEARTBEAT", blocking=True, timeout=60)

        if msg is None:
            raise RuntimeError("Heartbeat timeout (60s)")
        
        print(f"[HEARTBEAT_CHECKING] Heartbeat OK (sys={msg.get_srcSystem()}, comp={msg.get_srcComponent()})")

        self._set_status(MissionState.HEARTBEAT_OK, "heartbeat received")

    def _wait_command_ack(self, m, command_id: int, timeout: float = 5.0) -> bool:
        """
        Wait for COMMAND_ACK of a specific command.

        Returns
        -------
        True  : if accepted
        False : if denied / failed / timeout
        """
        t0 = time.time()
        while time.time() - t0 < timeout:
            msg = m.recv_match(type="COMMAND_ACK", blocking=True, timeout=0.5)
            if msg is None:
                continue

            if msg.command != command_id:
                continue

            result = msg.result
            if result == mavutil.mavlink.MAV_RESULT_ACCEPTED:
                print(f"[PX4] COMMAND_ACK OK: cmd={command_id}")
                return True

            print(f"[PX4] COMMAND_ACK FAILED: cmd={command_id}, result={result}")
            return False

        print(f"[PX4] COMMAND_ACK TIMEOUT: cmd={command_id}")
        return False

    def _clear_mission(self, m, timeout: float = 5.0) -> bool:
        """
        Clear all mission items stored in the vehicle.
        """
        self._set_status(MissionState.CLEARING_MISSION, "clearing mission")

        m.mav.mission_clear_all_send(
            m.target_system,
            m.target_component,
            mavutil.mavlink.MAV_MISSION_TYPE_MISSION,
        )
        t0 = time.time()
        while time.time() - t0 < timeout:
            msg = m.recv_match(type=["MISSION_ACK", "STATUSTEXT"], blocking=True, timeout=0.5)
            if msg is None:
                continue

            if msg.get_type() == "MISSION_ACK":
                return True

            if msg.get_type() == "STATUSTEXT":
                print(f"[MISSION] AP: {msg.text}")

        print("[MISSION] Mission clear timeout")
        return False
    
    def _upload_mission_items(self, m, items: list[dict], timeout: float = 30.0) -> bool:
        """
        Upload mission items using MAVLink mission protocol.

        Each item dict should contain:
            frame, command, autocontinue, p1..p7
        """
        self._set_status(MissionState.UPLOADING_MISSION, f"uploading {len(items)} items")

        m.mav.mission_count_send(
            m.target_system,
            m.target_component,
            len(items),
            mavutil.mavlink.MAV_MISSION_TYPE_MISSION,
        )

        sent = set()
        t0 = time.time()

        while time.time() - t0 < timeout:
            for _ in range(len(items)):

                req = m.recv_match(
                    type=["MISSION_REQUEST", "MISSION_REQUEST_INT", "MISSION_ACK", "STATUSTEXT"],
                    blocking=True,
                    timeout=0.5,
                )

                if not req:
                    raise RuntimeError("[MISSION] Mission upload timeout")

                if req.get_type() == "STATUSTEXT":
                    print(f"[MISSION] AP: {req.text}")
                    continue

                if req.get_type() == "MISSION_ACK":
                    ack_type = req.type
                    self._set_status(MissionState.MISSION_UPLOADED, "mission uploaded")
                    return ack_type == mavutil.mavlink.MAV_MISSION_ACCEPTED

                if req.get_type() in ("MISSION_REQUEST", "MISSION_REQUEST_INT"):
                    seq = req.seq
                    if seq < 0 or seq >= len(items):
                        print(f"[MISSION] Invalid mission request seq={seq}")
                        return False

                    it = items[seq]

                    m.mav.mission_item_int_send(
                        m.target_system,
                        m.target_component,
                        seq,
                        it["frame"],
                        it["command"],
                        1 if seq == 0 else 0,
                        it["autocontinue"],
                        it["p1"], it["p2"], it["p3"], it["p4"],
                        int(it["p5"] * 1e7),
                        int(it["p6"] * 1e7),
                        float(it["p7"]),
                        mavutil.mavlink.MAV_MISSION_TYPE_MISSION,
                    )

                    sent.add(seq)
                    print(f"[MISSION] Sent mission item seq={seq}, cmd={it['command']}")

        print("[MISSION] Mission upload timeout")
        return False
        
    def _read_mission_item(self, m, seq: int, timeout: float = 5.0):
        """
        Read back one mission item from the vehicle.
        Returns the MAVLink message or None.
        """
        m.mav.mission_request_int_send(
            m.target_system,
            m.target_component,
            seq,
        )

        t0 = time.time()
        while time.time() - t0 < timeout:
            msg = m.recv_match(
                type=["MISSION_ITEM_INT", "MISSION_ITEM", "STATUSTEXT"],
                blocking=True,
                timeout=0.5,
            )
            if msg is None:
                continue

            if msg.get_type() == "STATUSTEXT":
                print(f"AP: {msg.text}")
                continue

            if getattr(msg, "seq", None) == seq:
                return msg

        return None
    
    def _decode_statustext(self, msg) -> str:
        text = getattr(msg, "text", "")
        if isinstance(text, bytes):
            text = text.decode(errors="ignore")
        return str(text).strip()

    def _is_blocking_preflight_text(self, text: str) -> bool:
        t = text.lower()
        bad_keywords = [
            "preflight fail",
            "arming denied",
            "resolve system health failures first",
            "ekf2 missing data",
            "no connection",
            "accel #0 fail",
            "gyro #0 fail",
            "gps failure",
            "health failures",
        ]
        return any(k in t for k in bad_keywords)

    def _wait_ready_to_arm(
        self,
        m,
        timeout: float = 60.0,
        stable_ready_s: float = 2.0,
        no_error_hold_s: float = 2.0,
        require_global_position: bool = True,
    ) -> bool:
        """
        Wait until PX4 is genuinely ready to arm.

        Conditions:
        - PX4 HEARTBEAT observed
        - SYS_STATUS reports enabled sensors are healthy
        - optional GLOBAL_POSITION_INT observed
        - no recent blocking STATUSTEXT preflight errors
        - conditions remain good for stable_ready_s seconds
        """
        self._set_status(MissionState.HEARTBEAT_OK, "waiting for readiness before arming")

        t0 = time.time()
        ready_since = None

        px4_hb_seen = False
        sys_status_seen = False
        global_pos_seen = False

        sensors_enabled = 0
        sensors_health = 0

        last_error_text = None
        last_error_time = 0.0

        last_print_t = 0.0

        while time.time() - t0 < timeout:
            self._check_stop()

            msg = m.recv_match(
                type=["HEARTBEAT", "SYS_STATUS", "GLOBAL_POSITION_INT", "STATUSTEXT"],
                blocking=True,
                timeout=0.5,
            )
            now = time.time()

            if msg is not None:
                mtype = msg.get_type()

                if mtype == "HEARTBEAT":
                    px4_hb_seen = True

                elif mtype == "SYS_STATUS":
                    sys_status_seen = True
                    sensors_enabled = int(msg.onboard_control_sensors_enabled)
                    sensors_health = int(msg.onboard_control_sensors_health)

                elif mtype == "GLOBAL_POSITION_INT":
                    global_pos_seen = True

                elif mtype == "STATUSTEXT":
                    text = self._decode_statustext(msg)
                    print(f"[PX4] {text}")

                    if self._is_blocking_preflight_text(text):
                        last_error_text = text
                        last_error_time = now

            sensor_ok = False
            if sys_status_seen:
                missing_health = sensors_enabled & ~sensors_health
                sensor_ok = (missing_health == 0) or (missing_health == 65536)

            error_ok = (last_error_text is None) or ((now - last_error_time) >= no_error_hold_s)
            pos_ok = (global_pos_seen if require_global_position else True)

            ready = px4_hb_seen and sys_status_seen and sensor_ok and pos_ok and error_ok

            if ready:
                if ready_since is None:
                    ready_since = now

                if now - ready_since >= stable_ready_s:
                    print("[MONITOR] PX4 ready to arm")
                    return True
            else:
                ready_since = None

            if now - last_print_t > 1.0:
                last_print_t = now
                print("[MONITOR] " f"hb={px4_hb_seen}, " f"sys={sys_status_seen}, " f"sensor_ok={sensor_ok}, " f"global_pos={global_pos_seen}, " f"recent_error={last_error_text}")
                if sys_status_seen and missing_health == 0:
                    print("[MONITOR] sensor health check passed")
                elif sys_status_seen and missing_health == 65536:
                    print("[MONITOR] sensor health check passed (no radio control)")

        print("[MONITOR] timeout waiting for PX4 readiness")
        print("[MONITOR] final bits: " f"enabled=0x{sensors_enabled:08x}, " f"health=0x{sensors_health:08x}")

        if last_error_text:
            print(f"[MONITOR] last blocking error: {last_error_text}")
        return False
    
    def _arm(self, m) -> bool:
        self._set_status(MissionState.ARMING, "arming vehicle")

        m.mav.command_long_send(
            m.target_system,
            m.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0,
            1, 0, 0, 0, 0, 0, 0,
        )

        if not self._wait_command_ack(m, mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, timeout=10.0):
            return False

        t0 = time.time()
        while time.time() - t0 < 10.0:
            hb = m.recv_match(type="HEARTBEAT", blocking=True, timeout=0.5)
            if hb is None:
                continue

            armed = bool(hb.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
            if armed:
                self._set_status(MissionState.ARMED, "vehicle armed")
                print("[ARM] vehicle armed")
                return True

        print("[ARM] timeout waiting for armed state")
        return False

    def _set_mode_auto_mission(self, m) -> bool:
        """
        Force PX4 into AUTO.MISSION using MAVLink SET_MODE with
        base_mode + PX4 custom_mode.

        PX4 packing:
            custom_mode = (sub_mode << 24) | (main_mode << 16)

        AUTO.MISSION:
            main_mode = 4   # PX4_CUSTOM_MAIN_MODE_AUTO
            sub_mode  = 4   # PX4_CUSTOM_SUB_MODE_AUTO_MISSION
        """
        self._set_status(MissionState.SETTING_AUTO, "switching to AUTO.MISSION")

        # PX4 custom mode values
        PX4_CUSTOM_MAIN_MODE_AUTO = 4
        PX4_CUSTOM_SUB_MODE_AUTO_MISSION = 4

        # Enable custom mode interpretation in base_mode
        base_mode = mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED

        # PX4 packs main/sub mode into custom_mode like this
        custom_mode = (
            (PX4_CUSTOM_SUB_MODE_AUTO_MISSION << 24)
            | (PX4_CUSTOM_MAIN_MODE_AUTO << 16)
        )

        try:
            m.mav.set_mode_send(
                m.target_system,
                base_mode,
                custom_mode,
            )
        except Exception as e:
            print(f"[MODE CHANGE] SET_MODE AUTO.MISSION failed: {e}")
            return False

        # Wait briefly for heartbeat update
        deadline = time.time() + 2.0
        while time.time() < deadline:
            hb = m.recv_match(type="HEARTBEAT", blocking=True, timeout=0.5)
            if hb is None:
                continue

            # base_mode must indicate custom mode is enabled
            custom_enabled = bool(
                hb.base_mode & mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED
            )

            # Decode PX4 custom_mode from heartbeat
            hb_custom = int(hb.custom_mode)
            main_mode = (hb_custom >> 16) & 0xFF
            sub_mode = (hb_custom >> 24) & 0xFF

            if custom_enabled and main_mode == PX4_CUSTOM_MAIN_MODE_AUTO \
                    and sub_mode == PX4_CUSTOM_SUB_MODE_AUTO_MISSION:
                print("[MODE CHANGE] PX4 is now in AUTO.MISSION")
                return True

        print("[MODE CHANGE] AUTO.MISSION mode change not confirmed from HEARTBEAT")
        return False

    def _mission_start(self, m, first_item: int = 0, last_item: int = 0xFFFF) -> bool:
        self._set_status(MissionState.STARTING_MISSION, "sending mission start")

        m.mav.command_long_send(
            m.target_system,
            m.target_component,
            mavutil.mavlink.MAV_CMD_MISSION_START,
            0,
            float(first_item),
            float(last_item),
            0, 0, 0, 0, 0,
        )

        return self._wait_command_ack(m, mavutil.mavlink.MAV_CMD_MISSION_START, timeout=8.0)


    def _run(self):
        from tools.commander.mission_session_runner import MissionSessionRunner
        if self._has_started_run:
            raise RuntimeError('Mission runner _run already entered')
        self._has_started_run = True
        try:
            MissionSessionRunner(self).run()
        except Exception as exc:
            self._set_status(MissionState.FAILED, 'mission session failed', done=True,
                             success=False, error=str(exc))

    def session_connect_url(self, scenario):
        return _parse_connect_url(scenario)

    def session_ready(self, connection):
        if not self._wait_ready_to_arm(connection):
            raise TimeoutError('Vehicle did not become ready to arm')

    def session_observe(self, message):
        if message.get_type() == 'STATUSTEXT':
            print(f'[PX4] {self._decode_statustext(message)}')

    def session_prepare(self, connection, phase):
        if phase.get('initial_state', {}).get('kind') == 'in_air':
            from tools.commander.air_start import prepare_air_start
            phase['_initialization_result'] = prepare_air_start(self, connection, phase)
            return
        if phase.get('replay_policy') == 'recorded_time':
            self.session_set_mode(connection, 'HOLD')
            deadline = time.monotonic() + 8
            while time.monotonic() < deadline:
                hb = connection.recv_match(type='HEARTBEAT', blocking=True, timeout=.2)
                if hb and hb.get_srcSystem() == connection.target_system and self.session_mode_name(hb) == 'HOLD':
                    break
            else:
                raise RuntimeError('Initial HOLD not confirmed')
            connection.mav.mission_set_current_send(connection.target_system, connection.target_component, 0)
            if not self._arm(connection):
                raise RuntimeError('Arming failed')
            return
        connection.mav.mission_set_current_send(connection.target_system, connection.target_component, 0)
        if not self._set_mode_auto_mission(connection):
            raise RuntimeError('AUTO.MISSION not confirmed')
        if not self._arm(connection):
            raise RuntimeError('Arming failed')

    def session_start(self, connection, phase):
        if phase.get('replay_policy') == 'recorded_time':
            self.session_set_mode(connection, 'MISSION')
            return
        if not self._mission_start(connection, first_item=0):
            raise RuntimeError('Mission start failed')

    def session_mode_name(self, heartbeat):
        main, sub = (heartbeat.custom_mode >> 16) & 0xff, (heartbeat.custom_mode >> 24) & 0xff
        return {3:'HOLD',4:'MISSION',5:'RTL',6:'LAND'}.get(sub, f'PX4:{main}:{sub}') if main==4 else f'PX4:{main}:{sub}'

    def session_set_mode(self, connection, mode):
        sub = {'HOLD':3,'MISSION':4,'RTL':5,'LAND':6}[mode]
        connection.mav.set_mode_send(connection.target_system,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, (sub << 24) | (4 << 16))

    def session_mode_ok(self, heartbeat, final_land):
        main = (heartbeat.custom_mode >> 16) & 0xff
        sub = (heartbeat.custom_mode >> 24) & 0xff
        return main == 4 and (sub == 4 or (final_land and sub == 6))


def main():
    """
    Standalone execution entry point for testing.
    """

    import argparse

    parser = argparse.ArgumentParser(description="PX4 Mission Runner")
    parser.add_argument(
        "--scenario",
        type=Path,
        help="Path to scenario.yaml configuration file",
    )
    args = parser.parse_args()

    runner = PX4MissionRunner(args.scenario)
    runner._run()

    status = runner.get_status()
    print(f"[RUNNER] Mission finished: {status.state.name}, success={status.success}")

if __name__ == "__main__":
    main()
