#!/usr/bin/env python3
"""
ArduPilot Mission Runner (YAML-driven)

This script reads a scenario.yaml file and executes the mission using
pymavlink with ArduPilot SITL.

Main capabilities
-----------------
1. Parse scenario configuration from YAML
2. Construct MAVLink mission items
3. Upload mission to ArduPilot
4. Arm vehicle and optionally perform guided takeoff
5. Switch to AUTO mode and execute mission
6. Wait until vehicle disarms after landing

The YAML format is assumed to follow the structure used by the
scenario generator:

common.scenario
    home_lla
    takeoff_alt_m
    command
    waypoints_lla
    speed_m_s
    land

autopilots.ardupilot
    sim.out_udp_port
    mavsdk.connect_url
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import yaml
from pymavlink import mavutil

EARTH_RADIUS_M = 6378137.0


# -----------------------------------------------------------------------------
# Coordinate Utility
# -----------------------------------------------------------------------------

def add_north_east_m_to_gps(lat_deg, lon_deg, north_m, east_m):
    """
    Convert a local NED displacement (north/east) to latitude/longitude.

    Parameters
    ----------
    lat_deg : float
        Reference latitude (deg)
    lon_deg : float
        Reference longitude (deg)
    north_m : float
        North displacement (meters)
    east_m : float
        East displacement (meters)

    Returns
    -------
    tuple
        (lat, lon) in degrees
    """
    lat_rad = math.radians(lat_deg)

    dlat = north_m / EARTH_RADIUS_M
    dlon = east_m / (EARTH_RADIUS_M * math.cos(lat_rad))

    return (
        lat_deg + math.degrees(dlat),
        lon_deg + math.degrees(dlon)
    )


# -----------------------------------------------------------------------------
# MAVLink Utility Functions
# -----------------------------------------------------------------------------

# def wait_heartbeat(m):
#     """
#     Block until the first MAVLink heartbeat is received.
#     """
#     m.wait_heartbeat()
#     print(f"Heartbeat OK (sys={m.target_system}, comp={m.target_component})")


def clear_mission(m):
    """
    Remove all mission items currently stored on the autopilot.
    """
    m.mav.mission_clear_all_send(m.target_system, m.target_component)

    ack = m.recv_match(type="MISSION_ACK", blocking=True, timeout=5.0)

    if not ack:
        raise RuntimeError("MISSION_CLEAR_ALL: no ACK received")


def upload_mission_int(m, items):
    """
    Upload a list of mission items using the MISSION_ITEM_INT protocol.

    Parameters
    ----------
    m : mavutil.mavlink_connection
        Active MAVLink connection
    items : list
        List of mission item dictionaries
    """

    # Inform autopilot how many mission items will be uploaded
    m.mav.mission_count_send(
        m.target_system,
        m.target_component,
        len(items)
    )

    # Autopilot will sequentially request each item
    for _ in range(len(items)):

        req = m.recv_match(
            type=["MISSION_REQUEST_INT", "MISSION_REQUEST"],
            blocking=True,
            timeout=10.0
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
            0,  # current flag
            it["autocontinue"],
            it["p1"], it["p2"], it["p3"], it["p4"],
            int(it["lat"] * 1e7),
            int(it["lon"] * 1e7),
            float(it["alt"]),
            0
        )

    ack = m.recv_match(type="MISSION_ACK", blocking=True, timeout=10.0)

    if not ack:
        raise RuntimeError("MISSION upload failed")

    print("Mission upload OK:", ack)


def set_mode(m, mode_name: str):
    """
    Change flight mode.

    Parameters
    ----------
    mode_name : str
        ArduPilot flight mode (e.g., GUIDED, AUTO)
    """
    m.set_mode(mode_name)
    time.sleep(0.5)


def wait_altitude(m, target_alt_m: float, timeout=30):
    """
    Wait until the vehicle reaches a relative altitude threshold.

    Parameters
    ----------
    target_alt_m : float
        Target altitude above home (meters)
    """

    t0 = time.time()

    while time.time() - t0 < timeout:

        msg = m.recv_match(
            type="GLOBAL_POSITION_INT",
            blocking=True,
            timeout=1.0
        )

        if not msg:
            continue

        rel_alt = msg.relative_alt / 1000.0

        if rel_alt >= 0.8 * target_alt_m:
            print(f"Reached altitude ~ {rel_alt:.2f} m")
            return

    raise TimeoutError("Target altitude not reached")


def wait_disarmed(m, timeout=180):
    """
    Wait until the autopilot reports that the vehicle is disarmed.
    """

    t0 = time.time()

    while time.time() - t0 < timeout:

        hb = m.recv_match(type="HEARTBEAT", blocking=True, timeout=1.0)

        if not hb:
            continue

        armed = (
            hb.base_mode &
            mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
        ) != 0

        if not armed:
            print("Vehicle disarmed")
            return

    raise TimeoutError("Vehicle did not disarm")


def guided_takeoff(m, takeoff_alt_m):
    """
    Send a NAV_TAKEOFF command while in GUIDED mode.
    """

    m.mav.command_long_send(
        m.target_system,
        m.target_component,
        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
        0,
        0, 0, 0, 0,
        0, 0,
        takeoff_alt_m
    )

    ack = m.recv_match(type="COMMAND_ACK", blocking=True, timeout=3)

    if ack:
        print("TAKEOFF ACK:", ack)


# -----------------------------------------------------------------------------
# YAML Parsing
# -----------------------------------------------------------------------------

def _get(d: Dict[str, Any], path: str, default=None):
    """
    Utility for safely retrieving nested dictionary values.

    Example
    -------
    _get(config, "common.scenario.home_lla")
    """

    cur: Any = d

    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]

    return cur


def parse_connect_url(scn: Dict[str, Any]) -> str:
    """
    Determine MAVLink connection URL from scenario configuration.
    """

    url = _get(scn, "autopilots.ardupilot.mavsdk.connect_url")

    if url:

        if url.startswith("udp://"):
            hostport = url[len("udp://"):]
            return f"udp:{hostport}"

        return url

    port = _get(scn, "autopilots.ardupilot.sim.out_udp_port", 14550)

    return f"udp:127.0.0.1:{port}"


# -----------------------------------------------------------------------------
# Mission Construction
# -----------------------------------------------------------------------------

def build_items_from_scenario(scn: Dict[str, Any]) -> Tuple[list, float, bool]:
    """
    Convert scenario.yaml into MAVLink mission items.

    Returns
    -------
    items : list
        Mission item dictionaries
    takeoff_alt : float
        Takeoff altitude
    has_takeoff : bool
        Whether TAKEOFF command exists in mission
    """

    home = _get(scn, "common.scenario.home_lla")

    home_lat = float(home[0])
    home_lon = float(home[1])

    takeoff_alt = float(
        _get(scn, "common.scenario.takeoff_alt_m", 10.0)
    )

    commands = _get(scn, "common.scenario.command")
    waypoints = _get(scn, "common.scenario.waypoints_lla", [])
    speeds = _get(scn, "common.scenario.speed_m_s", [])

    items = []
    has_takeoff = False

    for i, cmd in enumerate(commands):

        if cmd is None:
            continue

        cmd = int(cmd)

        p1 = p2 = p3 = 0.0
        p4 = float("nan")
        lat = lon = alt = 0.0
        frame = mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT

        wp = waypoints[i] if i < len(waypoints) else None

        if cmd == mavutil.mavlink.MAV_CMD_NAV_TAKEOFF:

            has_takeoff = True
            lat, lon = home_lat, home_lon
            alt = takeoff_alt

        elif cmd == mavutil.mavlink.MAV_CMD_NAV_WAYPOINT:

            lat, lon, alt = wp

            p2 = 1.0  # acceptance radius

        elif cmd == mavutil.mavlink.MAV_CMD_DO_CHANGE_SPEED:

            frame = mavutil.mavlink.MAV_FRAME_MISSION

            p1 = 1.0
            p2 = speeds[i]
            p3 = -1.0

        elif cmd == mavutil.mavlink.MAV_CMD_NAV_LAND:

            lat = lon = alt = 0.0

        items.append(
            dict(
                frame=frame,
                command=cmd,
                autocontinue=1,
                p1=p1, p2=p2, p3=p3, p4=p4,
                lat=lat, lon=lon, alt=alt,
            )
        )

    return items, takeoff_alt, has_takeoff


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    """
    Entry point of the mission runner.
    """

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--scenario",
        type=Path,
        default=Path("./data/run_000/scenario.yaml")
    )

    args = ap.parse_args()

    scn = yaml.safe_load(args.scenario.read_text())

    connect = parse_connect_url(scn)

    print("Connect:", connect)

    items, takeoff_alt, has_takeoff = build_items_from_scenario(scn)

    m = mavutil.mavlink_connection(connect)

    wait_heartbeat(m)

    clear_mission(m)

    upload_mission_int(m, items)

    set_mode(m, "GUIDED")

    m.arducopter_arm()
    m.motors_armed_wait()

    print("Armed")

    if not has_takeoff:
        guided_takeoff(m, takeoff_alt)
        wait_altitude(m, takeoff_alt)

    set_mode(m, "AUTO")

    print("AUTO mission running")

    wait_disarmed(m)

    print("Mission completed")


if __name__ == "__main__":
    main()