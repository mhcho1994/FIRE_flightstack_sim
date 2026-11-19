#!/usr/bin/env python3
from __future__ import annotations

import math
import time
import threading
from enum import Enum, auto
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple, Optional

import yaml
from pymavlink import mavutil

EARTH_RADIUS_M = 6378137.0


# class MissionState(Enum):
#     IDLE = auto()
#     CONNECTING = auto()
#     HEARTBEAT_OK = auto()
#     CLEARING_MISSION = auto()
#     UPLOADING_MISSION = auto()
#     MISSION_UPLOADED = auto()
#     SETTING_GUIDED = auto()
#     ARMING = auto()
#     ARMED = auto()
#     TAKING_OFF = auto()
#     AUTO = auto()
#     RUNNING = auto()
#     COMPLETED = auto()
#     FAILED = auto()


# @dataclass
# class MissionStatus:
#     state: MissionState
#     message: str = ""
#     done: bool = False
#     success: bool = False
#     error: Optional[str] = None


# def _get(d: Dict[str, Any], path: str, default=None):
#     cur: Any = d
#     for k in path.split("."):
#         if not isinstance(cur, dict) or k not in cur:
#             return default
#         cur = cur[k]
#     return cur


# def parse_connect_url(scn: Dict[str, Any]) -> str:
#     url = _get(scn, "autopilots.ardupilot.mavsdk.connect_url")
#     if url:
#         if url.startswith("udp://"):
#             hostport = url[len("udp://"):]
#             return f"udpin:{hostport}"
#         return url
#     port = _get(scn, "autopilots.ardupilot.sim.out_udp_port", 14550)
#     return f"udpin:127.0.0.1:{port}"


# def build_items_from_scenario(scn: Dict[str, Any]) -> Tuple[list, float, bool]:
#     home = _get(scn, "common.scenario.home_lla")
#     home_lat = float(home[0])
#     home_lon = float(home[1])

#     takeoff_alt = float(_get(scn, "common.scenario.takeoff_alt_m", 10.0))
#     commands = _get(scn, "common.scenario.command", [])
#     waypoints = _get(scn, "common.scenario.waypoints_lla", [])
#     speeds = _get(scn, "common.scenario.speed_m_s", [])

#     items = []
#     has_takeoff = False

#     for i, cmd in enumerate(commands):
#         if cmd is None:
#             continue

#         cmd = int(cmd)

#         p1 = p2 = p3 = 0.0
#         p4 = float("nan")
#         lat = lon = alt = 0.0
#         frame = mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT

#         wp = waypoints[i] if i < len(waypoints) else None

#         if cmd == mavutil.mavlink.MAV_CMD_NAV_TAKEOFF:
#             has_takeoff = True
#             lat, lon = home_lat, home_lon
#             alt = takeoff_alt

#         elif cmd == mavutil.mavlink.MAV_CMD_NAV_WAYPOINT:
#             if wp is None:
#                 raise ValueError(f"Waypoint missing for command index {i}")
#             lat, lon, alt = wp
#             p2 = 1.0

#         elif cmd == mavutil.mavlink.MAV_CMD_DO_CHANGE_SPEED:
#             frame = mavutil.mavlink.MAV_FRAME_MISSION
#             p1 = 1.0
#             p2 = float(speeds[i])
#             p3 = -1.0

#         elif cmd == mavutil.mavlink.MAV_CMD_NAV_LAND:
#             lat = lon = alt = 0.0

#         items.append(
#             dict(
#                 frame=frame,
#                 command=cmd,
#                 autocontinue=1,
#                 p1=p1, p2=p2, p3=p3, p4=p4,
#                 lat=lat, lon=lon, alt=alt,
#             )
#         )

#     return items, takeoff_alt, has_takeoff


class ArduPilotMissionRunner:
    """
    Thread-backed mission runner that can be imported and polled by launcher.
    """

    # def __init__(self, scenario_path: Path):
    #     self.scenario_path = Path(scenario_path)
    #     self.status = MissionStatus(MissionState.IDLE, "initialized")
    #     self._lock = threading.Lock()
    #     self._thread: Optional[threading.Thread] = None
    #     self._stop_requested = False
    #     self._m = None

    # def _set_status(
    #     self,
    #     state: MissionState,
    #     message: str = "",
    #     done: bool = False,
    #     success: bool = False,
    #     error: Optional[str] = None,
    # ) -> None:
    #     with self._lock:
    #         self.status = MissionStatus(
    #             state=state,
    #             message=message,
    #             done=done,
    #             success=success,
    #             error=error,
    #         )

    # def get_status(self) -> MissionStatus:
    #     with self._lock:
    #         return MissionStatus(
    #             state=self.status.state,
    #             message=self.status.message,
    #             done=self.status.done,
    #             success=self.status.success,
    #             error=self.status.error,
    #         )

    # def request_stop(self) -> None:
    #     self._stop_requested = True

    # def is_done(self) -> bool:
    #     return self.get_status().done

    # def start(self) -> None:
    #     if self._thread is not None:
    #         raise RuntimeError("Mission runner already started")
    #     self._thread = threading.Thread(target=self._run, daemon=True)
    #     self._thread.start()

    # def join(self, timeout: Optional[float] = None) -> None:
    #     if self._thread is not None:
    #         self._thread.join(timeout)

    # def _check_stop(self) -> None:
    #     if self._stop_requested:
    #         raise RuntimeError("Mission runner stop requested")

    # def _wait_heartbeat(self, m) -> None:
    #     m.wait_heartbeat(timeout=20)
    #     self._set_status(MissionState.HEARTBEAT_OK, "heartbeat received")

    def _clear_mission(self, m) -> None:
        self._set_status(MissionState.CLEARING_MISSION, "clearing mission")
        m.mav.mission_clear_all_send(
            m.target_system,
            m.target_component,
            mavutil.mavlink.MAV_MISSION_TYPE_MISSION,
        )
        ack = m.recv_match(type="MISSION_ACK", blocking=True, timeout=5.0)
        if not ack:
            raise RuntimeError("MISSION_CLEAR_ALL: no ACK received")

    def _upload_mission_int(self, m, items) -> None:
        self._set_status(MissionState.UPLOADING_MISSION, f"uploading {len(items)} items")

        m.mav.mission_count_send(
            m.target_system,
            m.target_component,
            len(items),
            mavutil.mavlink.MAV_MISSION_TYPE_MISSION,
        )

        for _ in range(len(items)):
            self._check_stop()

            req = m.recv_match(
                type=["MISSION_REQUEST_INT", "MISSION_REQUEST"],
                blocking=True,
                timeout=10.0,
            )
            if not req:
                raise RuntimeError("MISSION upload timeout")

            seq = req.seq
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
                int(it["lat"] * 1e7),
                int(it["lon"] * 1e7),
                float(it["alt"]),
                mavutil.mavlink.MAV_MISSION_TYPE_MISSION,
            )

        ack = m.recv_match(type="MISSION_ACK", blocking=True, timeout=10.0)
        if not ack:
            raise RuntimeError("MISSION upload failed")

        self._set_status(MissionState.MISSION_UPLOADED, "mission uploaded")

    def _set_mode(self, m, mode_name: str) -> None:
        m.set_mode(mode_name)
        time.sleep(0.5)

    def _arm(self, m) -> None:
        self._set_status(MissionState.ARMING, "arming vehicle")
        m.arducopter_arm()
        m.motors_armed_wait()
        self._set_status(MissionState.ARMED, "vehicle armed")

    def _guided_takeoff(self, m, takeoff_alt_m: float) -> None:
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

        _ = m.recv_match(type="COMMAND_ACK", blocking=True, timeout=3)

        t0 = time.time()
        while time.time() - t0 < 30:
            self._check_stop()
            msg = m.recv_match(type="GLOBAL_POSITION_INT", blocking=True, timeout=1.0)
            if not msg:
                continue
            rel_alt = msg.relative_alt / 1000.0
            if rel_alt >= 0.8 * takeoff_alt_m:
                return

        raise TimeoutError("Target altitude not reached")

    def _wait_disarmed(self, m) -> None:
        self._set_status(MissionState.RUNNING, "mission running")
        t0 = time.time()
        while time.time() - t0 < 180:
            self._check_stop()
            hb = m.recv_match(type="HEARTBEAT", blocking=True, timeout=1.0)
            if not hb:
                continue

            armed = (
                hb.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
            ) != 0

            if not armed:
                self._set_status(MissionState.COMPLETED, "mission completed", done=True, success=True)
                return

        raise TimeoutError("Vehicle did not disarm")

    def _run(self) -> None:
        try:
            scn = yaml.safe_load(self.scenario_path.read_text())
            connect = parse_connect_url(scn)
            items, takeoff_alt, has_takeoff = build_items_from_scenario(scn)

            self._set_status(MissionState.CONNECTING, f"connecting to {connect}")

            m = mavutil.mavlink_connection(
                connect,
                source_system=246,
                source_component=190,
            )
            self._m = m

            self._wait_heartbeat(m)

            m.target_system = 1
            m.target_component = 1

            self._clear_mission(m)
            self._upload_mission_int(m, items)

            self._set_status(MissionState.SETTING_GUIDED, "switching to GUIDED")
            self._set_mode(m, "GUIDED")

            self._arm(m)

            if not has_takeoff:
                self._guided_takeoff(m, takeoff_alt)

            self._set_status(MissionState.AUTO, "switching to AUTO")
            self._set_mode(m, "AUTO")

            self._wait_disarmed(m)

        except Exception as e:
            self._set_status(
                MissionState.FAILED,
                message="mission failed",
                done=True,
                success=False,
                error=str(e),
            )