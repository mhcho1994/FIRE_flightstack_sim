#!/usr/bin/env python3
"""
PX4 Mission Runner (YAML-driven, MAVSDK-based)

Purpose
-------
Mission runner used for PX4 SITL + Gazebo experiments.

This script:
  1) Parses scenario.yaml
  2) Connects to PX4 via MAVSDK
  3) Waits for connection + health readiness
  4) Builds and uploads a waypoint mission
  5) Arms the vehicle
  6) Starts the mission
  7) Waits until mission finishes and vehicle disarms

Notes
-----
- This version uses MAVSDK Mission API rather than raw MAVLink.
- It is intended for PX4 SITL, especially with Gazebo GZ.
- QGC can be run separately for monitoring.
"""

from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, Optional, List

import yaml
from mavsdk import System
from mavsdk.action import ActionError
from mavsdk.mission import MissionItem, MissionPlan, MissionError
from mavsdk.telemetry import LandedState


class MissionState(Enum):
    IDLE = auto()
    CONNECTING = auto()
    CONNECTED = auto()
    HEALTH_OK = auto()
    MISSION_UPLOADING = auto()
    MISSION_UPLOADED = auto()
    ARMING = auto()
    ARMED = auto()
    STARTING_MISSION = auto()
    RUNNING = auto()
    LANDING = auto()
    COMPLETED = auto()
    FAILED = auto()


@dataclass
class MissionStatus:
    state: MissionState
    message: str = ""
    done: bool = False
    success: bool = False
    error: Optional[str] = None


def _get(d: Dict[str, Any], path: str, default=None):
    cur: Any = d
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _normalize_connect_url(url: str) -> str:
    """
    MAVSDK usually accepts:
      udp://:14540
      udp://127.0.0.1:14540
    We keep the YAML value as-is unless missing.
    """
    if not url:
        return "udp://:14540"
    return url


def build_mission_items_from_scenario(scn: Dict[str, Any]) -> List[MissionItem]:
    """
    Convert the common.scenario YAML into MAVSDK MissionItem list.

    Supported command mapping
    -------------------------
    22 : MAV_CMD_NAV_TAKEOFF   -> handled by PX4 mission execution
    16 : MAV_CMD_NAV_WAYPOINT  -> converted to MissionItem
    178: MAV_CMD_DO_CHANGE_SPEED -> applied to subsequent waypoint speed
    21 : MAV_CMD_NAV_LAND      -> represented via final waypoint + mission completion;
                                  explicit land command is not inserted as MAVSDK
                                  high-level MissionItem doesn't expose raw LAND item.

    Important simplification
    ------------------------
    MAVSDK Mission API is higher-level than raw MAVLink mission items.
    So this code converts waypoint-type navigation robustly, while keeping
    speed updates when provided.

    If later you need exact raw command parity with ArduPilot YAML
    (including explicit LAND command as a raw item), then MissionRaw
    would be the next step.
    """
    commands = _get(scn, "common.scenario.command", [])
    waypoints_lla = _get(scn, "common.scenario.waypoints_lla", [])
    takeoff_alt_m = float(_get(scn, "common.scenario.takeoff_alt_m", 10.0))
    speeds = _get(scn, "common.scenario.speed_m_s", [])
    default_speed_m_s = 5.0

    mission_items: List[MissionItem] = []
    current_speed = default_speed_m_s

    for i, cmd in enumerate(commands):
        if cmd is None:
            continue

        cmd = int(cmd)

        if cmd == 178:
            # MAV_CMD_DO_CHANGE_SPEED
            if i < len(speeds) and speeds[i] is not None:
                current_speed = float(speeds[i])
            continue

        if cmd == 22:
            # MAV_CMD_NAV_TAKEOFF
            # MAVSDK mission items don't need a separate explicit takeoff item
            # in many PX4 flows, but we keep the takeoff altitude available by
            # ensuring subsequent waypoint altitudes are relative and valid.
            continue

        if cmd == 16:
            # MAV_CMD_NAV_WAYPOINT
            wp = waypoints_lla[i] if i < len(waypoints_lla) else None
            if wp is None:
                raise ValueError(f"Waypoint missing for command index {i}")

            lat, lon, alt = float(wp[0]), float(wp[1]), float(wp[2])

            mission_items.append(
                MissionItem(
                    latitude_deg=lat,
                    longitude_deg=lon,
                    relative_altitude_m=alt if alt is not None else takeoff_alt_m,
                    speed_m_s=current_speed,
                    is_fly_through=True,
                    gimbal_pitch_deg=float("nan"),
                    gimbal_yaw_deg=float("nan"),
                    camera_action=MissionItem.CameraAction.NONE,
                    loiter_time_s=float("nan"),
                    camera_photo_interval_s=float("nan"),
                    acceptance_radius_m=float("nan"),
                    yaw_deg=float("nan"),
                    camera_photo_distance_m=float("nan"),
                    vehicle_action=MissionItem.VehicleAction.NONE,
                )
            )
            continue

        if cmd == 21:
            # MAV_CMD_NAV_LAND
            # High-level MAVSDK Mission API does not expose raw LAND item in the
            # same direct way as MAVLink mission protocol. We therefore treat
            # landing as post-mission action or mission-end behavior.
            continue

        raise ValueError(f"Unsupported command in PX4 high-level mission API: {cmd}")

    if not mission_items:
        raise ValueError("No waypoint mission items were generated from scenario")

    return mission_items


class PX4MissionRunner:
    def __init__(self, scenario_path: Path):
        self.scenario_path = Path(scenario_path)

        self.status = MissionStatus(MissionState.IDLE, "initialized")
        self._lock = threading.Lock()

        self._thread: Optional[threading.Thread] = None
        self._stop_requested = False
        self._has_started_run = False

    def _set_status(
        self,
        state: MissionState,
        message: str = "",
        done: bool = False,
        success: bool = False,
        error: Optional[str] = None,
    ) -> None:
        with self._lock:
            self.status = MissionStatus(
                state=state,
                message=message,
                done=done,
                success=success,
                error=error,
            )

    def get_status(self) -> MissionStatus:
        with self._lock:
            return MissionStatus(
                state=self.status.state,
                message=self.status.message,
                done=self.status.done,
                success=self.status.success,
                error=self.status.error,
            )

    def request_stop(self) -> None:
        self._stop_requested = True

    def is_done(self) -> bool:
        return self.get_status().done

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("Mission runner already started")

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def join(self, timeout: Optional[float] = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def _check_stop(self) -> None:
        if self._stop_requested:
            raise RuntimeError("Mission runner stop requested")

    async def _wait_connected(self, drone: System, timeout_s: float = 30.0) -> None:
        self._set_status(MissionState.CONNECTING, "waiting for MAVSDK connection")
        t0 = time.time()

        async for state in drone.core.connection_state():
            self._check_stop()

            if state.is_connected:
                self._set_status(MissionState.CONNECTED, "vehicle connected")
                return

            if time.time() - t0 > timeout_s:
                raise TimeoutError(f"PX4 connection timeout ({timeout_s:.1f}s)")

    async def _wait_health_ok(self, drone: System, timeout_s: float = 60.0) -> None:
        """
        Wait until PX4 reports enough readiness for mission flight.

        We require:
          - global position ok
          - home position ok
        """
        t0 = time.time()

        async for health in drone.telemetry.health():
            self._check_stop()

            if health.is_global_position_ok and health.is_home_position_ok:
                self._set_status(MissionState.HEALTH_OK, "health/global/home ready")
                return

            if time.time() - t0 > timeout_s:
                raise TimeoutError(f"PX4 health timeout ({timeout_s:.1f}s)")

    async def _wait_in_air(self, drone: System, timeout_s: float = 30.0) -> None:
        t0 = time.time()
        async for in_air in drone.telemetry.in_air():
            self._check_stop()
            if in_air:
                return
            if time.time() - t0 > timeout_s:
                raise TimeoutError("Vehicle did not become airborne")

    async def _wait_disarmed(self, drone: System, timeout_s: float = 180.0) -> None:
        t0 = time.time()

        async for armed in drone.telemetry.armed():
            self._check_stop()
            if not armed:
                self._set_status(MissionState.COMPLETED, "vehicle disarmed", done=True, success=True)
                return
            if time.time() - t0 > timeout_s:
                raise TimeoutError("Vehicle did not disarm before timeout")

    async def _monitor_mission_progress(self, drone: System, timeout_s: float = 300.0) -> None:
        self._set_status(MissionState.RUNNING, "mission running")
        t0 = time.time()
        last_pair = None

        async for progress in drone.mission.mission_progress():
            self._check_stop()

            pair = (progress.current, progress.total)
            if pair != last_pair:
                print(f"[MISSION] current={progress.current} total={progress.total}")
                last_pair = pair

            if progress.total > 0 and progress.current >= progress.total:
                return

            if time.time() - t0 > timeout_s:
                raise TimeoutError("Mission progress timeout")

    async def _run_async(self) -> None:
        scn = yaml.safe_load(self.scenario_path.read_text(encoding="utf-8")) or {}

        connect_url = _normalize_connect_url(
            _get(scn, "autopilots.px4.mavsdk.connect_url", "udp://:14540")
        )
        takeoff_alt_m = float(_get(scn, "common.scenario.takeoff_alt_m", 10.0))
        do_land = bool(_get(scn, "common.scenario.land", True))
        mission_items = build_mission_items_from_scenario(scn)

        drone = System()
        await drone.connect(system_address=connect_url)

        await self._wait_connected(drone)
        self._check_stop()

        await self._wait_health_ok(drone)
        self._check_stop()

        self._set_status(MissionState.MISSION_UPLOADING, f"uploading {len(mission_items)} mission items")
        mission_plan = MissionPlan(mission_items)

        # Keep behavior explicit.
        await drone.action.set_takeoff_altitude(takeoff_alt_m)
        await drone.mission.set_return_to_launch_after_mission(False)
        await drone.mission.upload_mission(mission_plan)

        self._set_status(MissionState.MISSION_UPLOADED, "mission uploaded")
        self._check_stop()

        self._set_status(MissionState.ARMING, "arming vehicle")
        await drone.action.arm()
        self._set_status(MissionState.ARMED, "vehicle armed")

        self._set_status(MissionState.STARTING_MISSION, "starting mission")
        await drone.mission.start_mission()

        # The mission API on PX4 generally arms/flies mission after start_mission
        # when the vehicle is ready. We watch progress and airborne state.
        try:
            await self._wait_in_air(drone, timeout_s=40.0)
        except TimeoutError:
            # Not fatal immediately; mission progress may still advance.
            print("[WARN] In-air state was not observed before timeout")

        await self._monitor_mission_progress(drone, timeout_s=300.0)
        self._check_stop()

        if do_land:
            self._set_status(MissionState.LANDING, "waiting for landing/disarm")
            await self._wait_disarmed(drone, timeout_s=180.0)
        else:
            self._set_status(MissionState.COMPLETED, "mission completed", done=True, success=True)

    def _run(self) -> None:
        if self._has_started_run:
            raise RuntimeError("Mission runner _run already entered")
        self._has_started_run = True

        print(f"[ENTER] PX4 runner self_id={id(self)} thread={threading.get_ident()}")

        try:
            asyncio.run(self._run_async())
        except Exception as e:
            self._set_status(
                MissionState.FAILED,
                message="execution failed, check errors",
                done=True,
                success=False,
                error=str(e),
            )


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="PX4 MAVSDK Mission Runner")
    ap.add_argument("--scenario", type=Path, required=True, help="Path to scenario.yaml")
    args = ap.parse_args()

    runner = PX4MissionRunner(args.scenario)
    runner.start()
    runner.join()

    status = runner.get_status()
    print(f"Mission finished: {status.state.name}, success={status.success}, error={status.error}")
    return 0 if status.success else 1


if __name__ == "__main__":
    raise SystemExit(main())