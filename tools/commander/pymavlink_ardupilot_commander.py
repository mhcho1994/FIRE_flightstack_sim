#!/usr/bin/env python3
"""ArduPilot MAVLink adapter for the shared MissionSessionRunner.

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

    IDLE = auto()              # Runner created but not started
    CONNECTING = auto()        # MAVLink connection in progress
    HEARTBEAT_OK = auto()      # Vehicle heartbeat received
    VERIFYING_PARAMETERS = auto()
    CLEARING_MISSION = auto()
    UPLOADING_MISSION = auto()
    MISSION_UPLOADED = auto()
    SETTING_GUIDED = auto()    # Requesting GUIDED mode
    WAITING_EKF = auto()        # Waiting for a stable position estimate
    ARMING = auto()            # Sending arm command
    ARMED = auto()             # Vehicle successfully armed
    TAKING_OFF = auto()
    AUTO = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()            # Error occurred


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
    _get(config, "autopilots.ardupilot.mavlink.connect_url")

    Equivalent to:
        config["autopilots"]["ardupilot"]["mavlink"]["connect_url"]

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

    url = _get(scn, "autopilots.ardupilot.mavlink.connect_url")

    if isinstance(url, str) and url.strip():
        url = url.strip()

        # Scenario style: udp:127.0.0.1:14550
        if url.startswith("udp:"):
            hostport = url[len("udp:"):]
            return f"udp:{hostport}"

        # Legacy style: udp://127.0.0.1:14550
        if url.startswith("udp://"):
            hostport = url[len("udp://"):]
            return f"udp:{hostport}"

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


class ArduPilotMissionRunner:
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
        self.autopilot = "ardupilot"
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

        # IMUs that have reported EKF GPS fusion via one-shot STATUSTEXT.
        self._ekf_gps_fusion_imus: set[int] = set()

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

        # For debugging
        # print(
        #     f"HIT: _wait_heartbeat "
        #     f"self_id={id(self)} "
        #     f"thread={threading.get_ident()}"
        # )

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
                print(f"[ARDUPILOT] COMMAND_ACK OK: cmd={command_id}")
                return True

            print(f"[ARDUPILOT] COMMAND_ACK FAILED: cmd={command_id}, result={result}")
            return False

        print(f"[ARDUPILOT] COMMAND_ACK TIMEOUT: cmd={command_id}")
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
            msg = m.recv_match(type=["MISSION_ACK", "STATUSTEXT"], blocking=True, timeout=5.0)
            if msg is None:
                continue

            if msg.get_type() == "MISSION_ACK":
                return True

            if msg.get_type() == "STATUSTEXT":
                print(f"[MISSION] AP: {msg.text}")

        print("[MISSION] Mission clear timeout")
        return False

    def _drain_mavlink(self, m, duration: float = 1.0) -> None:
        """
        Drain stale MAVLink messages before starting a new transaction.
        """
        t0 = time.time()
        while time.time() - t0 < duration:
            msg = m.recv_match(blocking=False)
            if msg is None:
                time.sleep(0.02)

    def _upload_mission_items(self, m, items: list[dict], timeout: float = 30.0) -> bool:
        """
        Upload mission items using MAVLink mission protocol.

        Each item dict should contain:
            frame, command, autocontinue, p1..p7
        """
        self._set_status(MissionState.UPLOADING_MISSION, f"uploading {len(items)} items")

        # self._drain_mavlink(m, duration=0.5)

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
                    type=["MISSION_REQUEST_INT", "MISSION_REQUEST", "MISSION_ACK", "STATUSTEXT"],
                    blocking=True,
                    timeout=10.0,
                )

                if req is None:
                    continue

                mtype = req.get_type()

                # if not req:
                #     raise RuntimeError("[MISSION] missionupload timeout")

                if mtype == "STATUSTEXT":
                    print(f"[MISSION] AP: {req.text}")
                    continue

                if mtype in ("MISSION_REQUEST", "MISSION_REQUEST_INT"):
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
                    print(f"[MISSION_UPLOAD] Sent mission item seq={seq}, cmd={it['command']}")
                    continue

                if mtype == "MISSION_ACK":
                    ack_type = req.type
                    if ack_type == mavutil.mavlink.MAV_MISSION_ACCEPTED:
                        if len(sent) < len(items):
                            # stale ACK from a previous operation (e.g. mission clear); ignore
                            continue
                        self._set_status(MissionState.MISSION_UPLOADED, "mission uploaded")
                        return True
                    print(f"[MISSION_UPLOAD] Failed MISSION_ACK: type={ack_type}, sent={len(sent)}/{len(items)}")
                    return False

                    # ack_type = req.type

                    # ack_name = mavutil.mavlink.enums["MAV_MISSION_RESULT"][ack_type].name
                    # print(f"[DEBUG] ACK TYPE {ack_type}: {ack_name}, sent={len(sent)}/{len(items)}")

                    # if ack_type == mavutil.mavlink.MAV_MISSION_ACCEPTED:
                    #     print("[MISSION_UPLOAD] Mission accepted")
                    #     self._set_status(MissionState.MISSION_UPLOADED, "mission uploaded")

                    # print(
                    #     f"[MISSION_UPLOAD] Early or failed MISSION_ACK: "
                    #     f"type={ack_type}, sent={len(sent)}/{len(items)}"
                    # )
                    # if ack_type != mavutil.mavlink.MAV_MISSION_ACCEPTED:
                    #     return False

                    # receiving ACCEPTED too early -> ignore because it could be a stale ACK
                    # continue

        print("[MISSION_UPLOAD] Mission upload timeout")
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

    def _record_ekf_gps_fusion_text(self, text: str) -> bool:
        """
        Record one-shot EKF GPS fusion STATUSTEXT messages when they appear.

        ArduPilot emits messages such as "EKF3 IMU0 is using GPS" early during
        startup. Later phases should reuse this remembered fact instead of
        waiting for the same STATUSTEXT to be emitted again.
        """
        t = text.lower()
        if "is using gps" not in t:
            return False

        matched = False
        for imu_idx in (0, 1):
            if f"imu{imu_idx}" in t:
                matched = True
                if imu_idx not in self._ekf_gps_fusion_imus:
                    self._ekf_gps_fusion_imus.add(imu_idx)
                    print(f"[EKF_WAIT] remembered GPS fusion on IMU{imu_idx}")

        return matched

    def _format_ekf_gps_fusion_imus(self) -> str:
        if not self._ekf_gps_fusion_imus:
            return "none"
        return ",".join(f"IMU{i}" for i in sorted(self._ekf_gps_fusion_imus))

    def _is_blocking_preflight_text(self, text: str) -> bool:
        t = text.lower()

        if "is using gps" in t:
            return False

        blocking_keywords = [
            "prearm:",
            "arm:",
            "arming denied",
            "ekf",
            "gps",
            "check",
            "calibration",
            "gyro",
            "accel",
            "compass",
            "baro",
            "not healthy",
            "need position estimate",
            "waiting for home",
        ]

        ignore_keywords = [
            "armed",
            "disarmed",
            "flying",
            "mode",
            "throttle",
            "mission",
        ]

        if any(k in t for k in ignore_keywords):
            return False

        return any(k in t for k in blocking_keywords)

    def _wait_ready_to_arm(
        self,
        m,
        timeout: float = 60.0,
        stable_ready_s: float = 2.0,
        no_error_hold_s: float = 2.0,
    ) -> bool:
        """
        Wait until Ardupilot is stably ready to arm.

        Readiness conditions
        --------------------
        1. Pre-arm check passed from SYS_STATUS
        2. EKF attitude is healthy
        3. EKF horizontal velocity is healthy
        4. EKF horizontal absolute position is healthy
        (optional if require_global_position=False)
        5. No recent blocking pre-arm / EKF related STATUSTEXT error for
        at least `no_error_hold_s`
        6. All required readiness conditions remain continuously satisfied for
        at least `stable_ready_s`

        Parameters
        ----------
        m :
            pymavlink connection
        timeout : float
            Overall timeout in seconds.
        stable_ready_s : float
            Required continuous ready duration before success.
        no_error_hold_s : float
            Minimum time since the last blocking error text.

        Returns
        -------
        bool
            True if ready to arm, False on timeout.
        """
        t0 = time.time()

        prearm_ok = False
        ekf_attitude_ok = False
        ekf_velocity_ok = False
        ekf_position_ok = False
        ekf_ready = False

        last_error_text = None
        last_error_time = 0.0
        ready_since = None

        while time.time() - t0 < timeout:
            self._check_stop()

            msg = m.recv_match(
                type=["SYS_STATUS", "EKF_STATUS_REPORT", "STATUSTEXT"],
                blocking=True,
                timeout=1.0,
            )

            now = time.time()

            if msg is None:
                continue

            mtype = msg.get_type()

            if mtype == "SYS_STATUS":
                health = msg.onboard_control_sensors_health

                if health & mavutil.mavlink.MAV_SYS_STATUS_PREARM_CHECK:
                    if not prearm_ok:
                        print("[MONITOR] pre-arm check passed")
                    prearm_ok = True
                else:
                    prearm_ok = False
                    ready_since = None

            elif mtype == "EKF_STATUS_REPORT":
                flags = msg.flags

                ekf_attitude_ok = bool(flags & mavutil.mavlink.EKF_ATTITUDE)
                ekf_velocity_ok = bool(flags & mavutil.mavlink.EKF_VELOCITY_HORIZ)
                ekf_position_ok = bool(flags & mavutil.mavlink.EKF_POS_HORIZ_ABS)

                # If any required EKF condition drops, reset stable timer
                ekf_ready = ekf_attitude_ok and ekf_velocity_ok and ekf_position_ok

                if not ekf_ready:
                    ready_since = None

            elif mtype == "STATUSTEXT":
                text = self._decode_statustext(msg)
                if text:
                    print(f"[AP] {text}")
                    self._record_ekf_gps_fusion_text(text)

                if text and self._is_blocking_preflight_text(text):
                    last_error_text = text
                    last_error_time = now
                    ready_since = None

            # Evaluate current readiness
            no_recent_error = (now - last_error_time) >= no_error_hold_s
            all_ready = prearm_ok and ekf_ready and no_recent_error

            if all_ready:
                if ready_since is None:
                    ready_since = now

                if now - ready_since >= stable_ready_s:
                    print("[MONITOR] vehicle ready to arm")
                    print(
                        "[MONITOR] "
                        f"prearm_ok={prearm_ok}, "
                        f"ekf_ok={ekf_ready}, "
                        f"gps_fusion_seen={self._format_ekf_gps_fusion_imus()}, "
                        f"last_error={last_error_text!r}"
                    )
                    return True
            else:
                ready_since = None

        print("[MONITOR] ready-to-arm timeout")
        print(
            "[MONITOR] final state: "
            f"prearm_ok={prearm_ok}, "
            f"ekf_ok={ekf_ready}, "
            f"gps_fusion_seen={self._format_ekf_gps_fusion_imus()}, "
            f"last_error={last_error_text!r}"
        )
        return False

    # def _wait_prearm_ok(self, m, timeout=60):
    #     """
    #     Wait until vehicle passes pre-arm checks.
    #     """

    #     t0 = time.time()

    #     while time.time() - t0 < timeout:

    #         msg = m.recv_match(type="SYS_STATUS", blocking=True, timeout=1)

    #         if msg is None:
    #             continue

    #         health = msg.onboard_control_sensors_health

    #         if health & mavutil.mavlink.MAV_SYS_STATUS_PREARM_CHECK:
    #             print("[MONITOR] pre-arm check passed")
    #             return True

    #     print("[MONITOR] pre-arm check timeout")
    #     return False
    
    # def _wait_ekf_ready(self, m, timeout=60.0) -> bool:
    #     """
    #     Wait until EKF reports healthy state.
    #     """
    #     t0 = time.time()

    #     while time.time() - t0 < timeout:

    #         msg = m.recv_match(type="EKF_STATUS_REPORT", blocking=True, timeout=1.0)
    #         if msg is None:
    #             continue

    #         flags = msg.flags

    #         attitude_ok = flags & mavutil.mavlink.EKF_ATTITUDE
    #         pos_ok = flags & mavutil.mavlink.EKF_POS_HORIZ_ABS
    #         vel_ok = flags & mavutil.mavlink.EKF_VELOCITY_HORIZ

    #         if attitude_ok and pos_ok and vel_ok:
    #             print("[MONITOR] EKF check passed")
    #             return True

    #     print("[MONITOR] EKF not ready, timeout")
    #     return False

    def _arm(self, m) -> None:
        """
        Send arm command and wait until motors are armed.

        Arming enables the vehicle's motors and allows flight commands.
        """

        self._set_status(MissionState.ARMING, "arming vehicle")

        m.arducopter_arm()

        # Block until autopilot reports motors armed
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            self._check_stop()
            heartbeat = m.recv_match(type='HEARTBEAT', blocking=True, timeout=.2)
            if (heartbeat is not None and heartbeat.get_srcSystem() == m.target_system
                    and heartbeat.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED):
                break
        else:
            raise TimeoutError('Arming timeout')

        self._set_status(
            MissionState.ARMED,
            "vehicle armed",
            done=False,
            success=False,
        )

    def _set_mode(self, m, mode_name: str) -> None:
        """
        Request a flight mode change.

        ArduPilot modes include:
            STABILIZE
            GUIDED
            AUTO
            LOITER
            etc.
        """

        m.set_mode(mode_name)

        # Short delay to allow the autopilot to process the request
        time.sleep(0.5)

    def _guided_takeoff(self, m, takeoff_alt_m: float, timeout: float = 60.0) -> None:
        self._set_status(MissionState.TAKING_OFF, f"takeoff to {takeoff_alt_m:.1f} m")
        m.mav.command_long_send(
            m.target_system,
            m.target_component,
            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
            0,
            0, 0, 0, 0,
            0, 0,
            takeoff_alt_m,
        )

        _ = m.recv_match(type="COMMAND_ACK", blocking=True, timeout=5.0)

        t0 = time.time()
        while time.time() - t0 < timeout:
            self._check_stop()
            msg = m.recv_match(type="GLOBAL_POSITION_INT", blocking=True, timeout=1.0)
            if not msg:
                continue
            rel_alt = msg.relative_alt / 1000.0
            if rel_alt >= 0.8 * takeoff_alt_m:
                return

        raise TimeoutError("Target altitude not reached")

    def _set_mode_auto(self, m) -> bool:
        """
        Switch vehicle to AUTO mode and wait for COMMAND_ACK.
        """
        mode_id = m.mode_mapping().get("AUTO")
        if mode_id is None:
            print("AUTO mode is not available in mode mapping")
            return False

        m.mav.command_long_send(
            m.target_system,
            m.target_component,
            mavutil.mavlink.MAV_CMD_DO_SET_MODE,
            0,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            mode_id,
            0, 0, 0, 0, 0,
        )

        return self._wait_command_ack(m, mavutil.mavlink.MAV_CMD_DO_SET_MODE, timeout=5.0)

    def _set_param(self, m, name: str, value: int, timeout: float = 5.0) -> bool:
        """
        Set an ArduPilot parameter as integer/real via PARAM_SET,
        then verify using PARAM_VALUE.
        """
        m.mav.param_set_send(
            m.target_system,
            m.target_component,
            name.encode("utf-8"),
            float(value),
            mavutil.mavlink.MAV_PARAM_TYPE_REAL32,
        )

        t0 = time.time()
        while time.time() - t0 < timeout:
            msg = m.recv_match(type="PARAM_VALUE", blocking=True, timeout=0.5)
            if msg is None:
                continue

            def normalize_param_id(param_id) -> str:
                """
                Normalize PARAM_VALUE.param_id to a plain Python string.
                Handles both bytes and str cases.
                """
                if isinstance(param_id, bytes):
                    return param_id.decode("utf-8", errors="ignore").rstrip("\x00")
                return str(param_id).rstrip("\x00")

            param_id = normalize_param_id(msg.param_id)
            if param_id != name:
                continue

            actual = msg.param_value
            print(f"PARAM {name} = {actual}")
            return abs(actual - float(value)) < 1e-3

        print(f"PARAM_SET timeout: {name}")
        return False

    def _get_param(self, m, name: str, timeout: float = 5.0):
        """
        Request a parameter and return its value, or None on timeout.
        """
        m.mav.param_request_read_send(
            m.target_system,
            m.target_component,
            name.encode("utf-8"),
            -1,
        )

        t0 = time.time()
        while time.time() - t0 < timeout:
            msg = m.recv_match(type="PARAM_VALUE", blocking=True, timeout=0.5)
            if msg is None:
                continue

            def normalize_param_id(param_id) -> str:
                """
                Normalize PARAM_VALUE.param_id to a plain Python string.
                Handles both bytes and str cases.
                """
                if isinstance(param_id, bytes):
                    return param_id.decode("utf-8", errors="ignore").rstrip("\x00")
                return str(param_id).rstrip("\x00")

            param_id = normalize_param_id(msg.param_id)
            if param_id != name:
                continue

            return msg.param_value

        return None


    def _request_message_interval(self, m, message_id: int, hz: float) -> None:
        interval_us = int(1e6 / hz)
        print(f"[DEBUG] request message_id={message_id}, hz={hz}, interval_us={interval_us}")

        m.mav.command_long_send(
            m.target_system,
            m.target_component,
            mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
            0,
            message_id,      # param1: message id
            interval_us,     # param2: interval in usec
            0, 0, 0, 0, 0
        )

    def _wait_ekf_position_stable(
        self,
        m,
        timeout: float = 30.0,
        variance_threshold: float = 0.5,
        stable_s: float = 3.0,
    ) -> bool:
        """
        Wait until EKF reports usable horizontal position and stable variance.

        This intentionally does not depend on STATUSTEXT messages such as
        "EKF3 IMUx is using GPS" because those messages are one-shot and may
        already have been consumed by an earlier readiness wait.
        """
        self._set_status(MissionState.WAITING_EKF, "waiting for stable EKF position")

        print(
            "[EKF_WAIT] GPS fusion already seen: "
            f"{self._format_ekf_gps_fusion_imus()}"
        )

        self._request_message_interval(
            m,
            mavutil.mavlink.MAVLINK_MSG_ID_EKF_STATUS_REPORT,
            5.0,
        )

        t0 = time.time()
        stable_since = None
        last_report_time = 0.0
        last_flags = None
        last_variance = None
        last_text = None

        while time.time() - t0 < timeout:
            self._check_stop()

            msg = m.recv_match(
                type=["EKF_STATUS_REPORT", "STATUSTEXT"],
                blocking=True,
                timeout=1.0,
            )

            now = time.time()

            if msg is None:
                continue

            if msg.get_type() == "STATUSTEXT":
                text = self._decode_statustext(msg)
                if text:
                    last_text = text
                    print(f"[EKF_WAIT] AP: {text}")
                    self._record_ekf_gps_fusion_text(text)
                continue

            flags = msg.flags
            last_flags = flags
            last_variance = msg.pos_horiz_variance

            attitude_ok = bool(flags & mavutil.mavlink.EKF_ATTITUDE)
            velocity_ok = bool(flags & mavutil.mavlink.EKF_VELOCITY_HORIZ)
            position_ok = bool(flags & mavutil.mavlink.EKF_POS_HORIZ_ABS)
            variance_ok = last_variance < variance_threshold
            ekf_ready = attitude_ok and velocity_ok and position_ok and variance_ok

            if now - last_report_time >= 2.0:
                print(
                    "[EKF_WAIT] "
                    f"att={attitude_ok} vel={velocity_ok} pos={position_ok} "
                    f"variance={last_variance:.3f} "
                    f"gps_fusion_seen={self._format_ekf_gps_fusion_imus()} "
                    f"ready={ekf_ready}"
                )
                last_report_time = now

            if ekf_ready:
                if stable_since is None:
                    stable_since = now
                    print(
                        f"[EKF_WAIT] stable candidate "
                        f"(variance={last_variance:.3f}), waiting {stable_s:.0f}s"
                    )
                elif now - stable_since >= stable_s:
                    print(
                        f"[EKF_WAIT] position stable "
                        f"(variance={last_variance:.3f}), proceeding to arm"
                    )
                    return True
            else:
                stable_since = None

        print(
            "[EKF_WAIT] timeout: "
            f"last_flags={last_flags!r}, "
            f"last_variance={last_variance!r}, "
            f"gps_fusion_seen={self._format_ekf_gps_fusion_imus()}, "
            f"last_text={last_text!r}"
        )
        return False


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
            text = self._decode_statustext(message)
            self._record_ekf_gps_fusion_text(text)
            print(f'[ARDUPILOT] {text}')

    def session_prepare(self, connection, phase):
        self._set_mode(connection, 'GUIDED')
        for name, value in [('AUTO_OPTIONS', 3), ('MIS_RESTART', 0 if phase.get('replay_policy') == 'recorded_time' else 1)]:
            if not self._set_param(connection, name, value):
                raise RuntimeError(f'Failed to set {name}')
        connection.mav.mission_set_current_send(connection.target_system, connection.target_component, 1)
        self._arm(connection)

    def session_start(self, connection, phase):
        if phase['mission_items'][0]['command'] != 22:
            self._guided_takeoff(connection, phase.get('takeoff_alt_m', 10.), timeout=30)
        self._set_mode(connection, 'AUTO')

    def session_mode_name(self, heartbeat):
        return {3:'MISSION',5:'HOLD',6:'RTL',9:'LAND'}.get(heartbeat.custom_mode, f'AP:{heartbeat.custom_mode}')

    def session_set_mode(self, connection, mode):
        number = {'MISSION':3,'HOLD':5,'RTL':6,'LAND':9}[mode]
        connection.mav.set_mode_send(connection.target_system,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, number)

    def session_mode_ok(self, heartbeat, final_land):
        return heartbeat.custom_mode == self._m.mode_mapping()['AUTO']

def main():
    """
    Standalone execution entry point for testing.
    """

    import argparse

    parser = argparse.ArgumentParser(description="ArduPilot Mission Runner")
    parser.add_argument(
        "--scenario",
        type=Path,
        help="Path to scenario.yaml configuration file",
    )
    args = parser.parse_args()

    runner = ArduPilotMissionRunner(args.scenario)
    runner._run()

    status = runner.get_status()
    print(f"Mission finished: {status.state.name}, success={status.success}")

if __name__ == "__main__":
    main()
